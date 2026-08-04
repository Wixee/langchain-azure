from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass
from typing import Any, ClassVar, TypedDict
from unittest.mock import patch

from azure.ai.agentserver.core.storage import FoundryStateStore
from langgraph.checkpoint.base import Checkpoint
from langgraph.graph import START, StateGraph

from foundry_checkpointer import FoundryCheckpointSaver


class _GraphState(TypedDict):
    value: int


@dataclass
class _StoredItem:
    id: str
    value: dict[str, Any]
    tags: dict[str, str]


@dataclass
class _ItemKey:
    id: str
    key: str


@dataclass
class _ItemPage:
    keys: list[_ItemKey]
    has_more: bool
    last_id: str | None


@dataclass
class _ItemValue:
    value: dict[str, Any]


class _FakeStateStore:
    stores: ClassVar[dict[str, dict[str, _StoredItem]]] = {}
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    next_id = 1

    def __init__(self, name: str) -> None:
        self.name = name
        self.stores.setdefault(name, {})

    @classmethod
    def reset(cls) -> None:
        cls.stores = {}
        cls.calls = []
        cls.next_id = 1

    @classmethod
    async def get_or_create(cls, name: str, **kwargs: Any) -> _FakeStateStore:
        cls.calls.append((name, kwargs))
        return cls(name)

    async def set_item(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: dict[str, str],
    ) -> None:
        existing = self.stores[self.name].get(key)
        item_id = existing.id if existing is not None else self._new_id()
        self.stores[self.name][key] = _StoredItem(
            id=item_id,
            value=copy.deepcopy(value),
            tags=dict(tags),
        )

    async def create_item(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: dict[str, str],
    ) -> None:
        if key in self.stores[self.name]:
            raise AssertionError(f"duplicate test item: {key}")
        await self.set_item(key, value, tags=tags)

    async def get_item(self, key: str) -> _ItemValue | None:
        item = self.stores[self.name].get(key)
        return _ItemValue(copy.deepcopy(item.value)) if item is not None else None

    async def list_keys(
        self,
        *,
        tags: dict[str, str],
        limit: int,
        order: str,
        after: str | None = None,
    ) -> _ItemPage:
        items = [
            (key, item)
            for key, item in self.stores[self.name].items()
            if all(item.tags.get(name) == value for name, value in tags.items())
        ]
        items.sort(key=lambda pair: pair[1].id, reverse=order == "desc")
        if after is not None:
            items = items[
                next(
                    (
                        index + 1
                        for index, (_, item) in enumerate(items)
                        if item.id == after
                    ),
                    len(items),
                ) :
            ]
        selected = items[:limit]
        return _ItemPage(
            keys=[_ItemKey(item.id, key) for key, item in selected],
            has_more=len(items) > len(selected),
            last_id=selected[-1][1].id if selected else None,
        )

    async def delete(self) -> None:
        self.stores.pop(self.name, None)

    async def aclose(self) -> None:
        return None

    @classmethod
    def _new_id(cls) -> str:
        item_id = f"{cls.next_id:08d}"
        cls.next_id += 1
        return item_id


def _checkpoint(checkpoint_id: str, message: str) -> Checkpoint:
    return Checkpoint(
        v=1,
        id=checkpoint_id,
        ts=f"2026-08-04T00:00:0{checkpoint_id[-1]}Z",
        channel_values={"messages": [message]},
        channel_versions={"messages": checkpoint_id},
        versions_seen={},
        updated_channels=["messages"],
    )


class FoundryCheckpointSaverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeStateStore.reset()
        self.store_patch = patch.object(
            FoundryStateStore,
            "get_or_create",
            side_effect=_FakeStateStore.get_or_create,
        )
        self.store_patch.start()

    def tearDown(self) -> None:
        self.store_patch.stop()

    async def test_round_trip_history_and_pending_writes(self) -> None:
        saver = FoundryCheckpointSaver(item_ttl_seconds=3600)
        config = {
            "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
        }
        first = await saver.aput(
            config,
            _checkpoint("0001", "first"),
            {"source": "input", "step": -1},
            {},
        )
        await saver.aput_writes(
            first,
            [("messages", {"content": "pending"})],
            "task-1",
        )
        second = await saver.aput(
            first,
            _checkpoint("0002", "second"),
            {"source": "loop", "step": 0},
            {},
        )

        latest = await saver.aget_tuple(config)
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.checkpoint["id"], "0002")
        self.assertEqual(latest.parent_config, first)

        exact = await saver.aget_tuple(first)
        self.assertIsNotNone(exact)
        assert exact is not None
        self.assertEqual(
            exact.pending_writes,
            [("task-1", "messages", {"content": "pending"})],
        )

        history = [item async for item in saver.alist(config)]
        self.assertEqual([item.checkpoint["id"] for item in history], ["0002", "0001"])
        before = [item async for item in saver.alist(config, before=second)]
        self.assertEqual([item.checkpoint["id"] for item in before], ["0001"])
        filtered = [
            item
            async for item in saver.alist(config, filter={"source": "input"})
        ]
        self.assertEqual([item.checkpoint["id"] for item in filtered], ["0001"])

        store_name, options = _FakeStateStore.calls[0]
        self.assertEqual(store_name, "langGraphCheckpoints/thread-1")
        self.assertTrue(options["user_isolation"])
        self.assertEqual(options["item_ttl_seconds"], 3600)

    async def test_delete_thread_removes_its_store(self) -> None:
        saver = FoundryCheckpointSaver()
        config = {
            "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
        }
        await saver.aput(
            config,
            _checkpoint("0001", "first"),
            {"source": "input", "step": -1},
            {},
        )

        await saver.adelete_thread("thread-1")

        self.assertNotIn("langGraphCheckpoints/thread-1", _FakeStateStore.stores)

    async def test_runs_as_a_langgraph_checkpointer(self) -> None:
        async def increment(state: _GraphState) -> dict[str, int]:
            return {"value": state["value"] + 1}

        builder = StateGraph(_GraphState)
        builder.add_node("increment", increment)
        builder.add_edge(START, "increment")
        saver = FoundryCheckpointSaver()
        graph = builder.compile(checkpointer=saver)
        config = {
            "configurable": {"thread_id": "graph-thread", "checkpoint_ns": ""}
        }

        result = await graph.ainvoke({"value": 1}, config)
        snapshot = await graph.aget_state(config)

        self.assertEqual(result, {"value": 2})
        self.assertEqual(snapshot.values, {"value": 2})


if __name__ == "__main__":
    unittest.main()