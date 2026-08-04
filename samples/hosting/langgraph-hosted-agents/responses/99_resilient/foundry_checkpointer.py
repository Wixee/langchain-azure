"""LangGraph checkpointer backed by Foundry's durable state store."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from azure.ai.agentserver.core.storage import (
    FoundryStateStore,
    FoundryStorageConflictError,
    StateStoreItem,
)
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

_MAX_STORE_NAME_LENGTH = 128
_PAGE_SIZE = 100


class FoundryCheckpointSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph checkpoints in per-thread Foundry state stores."""

    def __init__(
        self,
        *,
        item_ttl_seconds: int = 30 * 24 * 60 * 60,
        store_name_prefix: str = "langGraphCheckpoints",
        credential: AsyncTokenCredential | None = None,
        endpoint: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._item_ttl_seconds = item_ttl_seconds
        self._store_name_prefix = store_name_prefix.strip("/")
        self._credential = credential
        self._endpoint = endpoint
        if not self._store_name_prefix:
            raise ValueError("store_name_prefix must not be empty")

    def _store_name(self, thread_id: str) -> str:
        name = f"{self._store_name_prefix}/{thread_id}"
        if len(name) > _MAX_STORE_NAME_LENGTH:
            raise ValueError(
                "Foundry checkpoint store name exceeds 128 characters: "
                f"{name!r}"
            )
        return name

    @asynccontextmanager
    async def _store(self, thread_id: str) -> AsyncIterator[FoundryStateStore]:
        store = await FoundryStateStore.get_or_create(
            self._store_name(thread_id),
            credential=self._credential,
            endpoint=self._endpoint,
            user_isolation=True,
            item_ttl_seconds=self._item_ttl_seconds,
            description="LangGraph checkpoint store",
            tags={"framework": "langgraph"},
        )
        try:
            yield store
        finally:
            await store.aclose()

    def _serialize(self, value: Any) -> dict[str, str]:
        type_name, data = self.serde.dumps_typed(value)
        return {
            "type": type_name,
            "data": base64.b64encode(data).decode("ascii"),
        }

    def _deserialize(self, value: Mapping[str, Any]) -> Any:
        return self.serde.loads_typed(
            (
                cast(str, value["type"]),
                base64.b64decode(cast(str, value["data"])),
            )
        )

    @staticmethod
    def _checkpoint_key(checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{checkpoint_ns}/{checkpoint_id}"

    @staticmethod
    def _write_key(
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        index: int,
    ) -> str:
        return f"{checkpoint_ns}/writes/{checkpoint_id}/{task_id}/{index}"

    async def _iter_keys(
        self,
        store: FoundryStateStore,
        *,
        tags: Mapping[str, str],
    ) -> AsyncIterator[str]:
        after: str | None = None
        while True:
            page = await store.list_keys(
                tags=tags,
                limit=_PAGE_SIZE,
                order="desc",
                after=after,
            )
            for key in page.keys:
                yield key.key
            if not page.has_more or page.last_id is None:
                return
            after = page.last_id

    async def _load_pending_writes(
        self,
        store: FoundryStateStore,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        writes: list[tuple[str, int, str, Any]] = []
        tags = {
            "kind": "write",
            "ns": checkpoint_ns,
            "ckpt": checkpoint_id,
        }
        async for key in self._iter_keys(store, tags=tags):
            item = await store.get_item(key)
            if item is None:
                continue
            value = cast(Mapping[str, Any], item.value)
            writes.append(
                (
                    cast(str, value["task_id"]),
                    cast(int, value["index"]),
                    cast(str, value["channel"]),
                    self._deserialize(cast(Mapping[str, Any], value["value"])),
                )
            )
        writes.sort(key=lambda write: (write[0], write[1]))
        return [
            (task_id, channel, value)
            for task_id, _, channel, value in writes
        ]

    async def _to_checkpoint_tuple(
        self,
        store: FoundryStateStore,
        item: StateStoreItem,
        thread_id: str,
        checkpoint_ns: str,
    ) -> CheckpointTuple:
        value = cast(Mapping[str, Any], item.value)
        checkpoint_id = cast(str, value["checkpoint_id"])
        parent_checkpoint_id = cast(str | None, value.get("parent_checkpoint_id"))
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config: RunnableConfig | None = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint=cast(
                Checkpoint,
                self._deserialize(
                    cast(Mapping[str, Any], value["checkpoint"])
                ),
            ),
            metadata=cast(
                CheckpointMetadata,
                self._deserialize(cast(Mapping[str, Any], value["metadata"])),
            ),
            parent_config=parent_config,
            pending_writes=await self._load_pending_writes(
                store,
                checkpoint_ns,
                checkpoint_id,
            ),
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)
        async with self._store(thread_id) as store:
            if checkpoint_id is not None:
                item = await store.get_item(
                    self._checkpoint_key(checkpoint_ns, checkpoint_id)
                )
            else:
                page = await store.list_keys(
                    tags={"kind": "checkpoint", "ns": checkpoint_ns},
                    limit=1,
                    order="desc",
                )
                item = (
                    await store.get_item(page.keys[0].key)
                    if page.keys
                    else None
                )
            if item is None:
                return None
            return await self._to_checkpoint_tuple(
                store,
                item,
                thread_id,
                checkpoint_ns,
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            return
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        before_id = get_checkpoint_id(before) if before is not None else None
        tags = {"kind": "checkpoint", "ns": checkpoint_ns}
        if filter is not None:
            for name in ("source", "step"):
                if name in filter:
                    tags[name] = str(filter[name])

        count = 0
        async with self._store(thread_id) as store:
            async for key in self._iter_keys(store, tags=tags):
                item = await store.get_item(key)
                if item is None:
                    continue
                checkpoint_tuple = await self._to_checkpoint_tuple(
                    store,
                    item,
                    thread_id,
                    checkpoint_ns,
                )
                checkpoint_id = str(
                    checkpoint_tuple.config["configurable"]["checkpoint_id"]
                )
                if before_id is not None and checkpoint_id >= before_id:
                    continue
                if filter is not None and not all(
                    checkpoint_tuple.metadata.get(name) == expected
                    for name, expected in filter.items()
                ):
                    continue
                yield checkpoint_tuple
                count += 1
                if limit is not None and count >= limit:
                    return

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = checkpoint["id"]
        full_metadata = get_checkpoint_metadata(config, metadata)
        tags = {"kind": "checkpoint", "ns": checkpoint_ns}
        for name in ("source", "step"):
            if name in full_metadata:
                tags[name] = str(full_metadata[name])

        value = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": get_checkpoint_id(config),
            "checkpoint": self._serialize(checkpoint),
            "metadata": self._serialize(full_metadata),
        }
        async with self._store(thread_id) as store:
            await store.set_item(
                self._checkpoint_key(checkpoint_ns, checkpoint_id),
                value,
                tags=tags,
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            raise ValueError("checkpoint_id is required when storing writes")

        replace_existing = all(channel in WRITES_IDX_MAP for channel, _ in writes)
        async with self._store(thread_id) as store:
            for index, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, index)
                key = self._write_key(
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_index,
                )
                item = {
                    "task_id": task_id,
                    "task_path": task_path,
                    "index": write_index,
                    "channel": channel,
                    "value": self._serialize(value),
                }
                tags = {
                    "kind": "write",
                    "ns": checkpoint_ns,
                    "ckpt": checkpoint_id,
                }
                if replace_existing:
                    await store.set_item(key, item, tags=tags)
                    continue
                try:
                    await store.create_item(key, item, tags=tags)
                except FoundryStorageConflictError:
                    pass

    async def adelete_thread(self, thread_id: str) -> None:
        async with self._store(str(thread_id)) as store:
            await store.delete()