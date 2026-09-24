"""Opt-in response checkpoint branching and strict restoration tests."""

from __future__ import annotations

import asyncio
import json
import operator
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses.store._memory import InMemoryResponseProvider
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from starlette.testclient import TestClient
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._responses import (
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    HostingRunnableConfig,
)
from langchain_azure_ai.agents.hosting._responses.branching import (
    BRANCH_MODE,
    BRANCH_MODE_HEADER,
    BRANCH_MODE_METADATA,
    BRANCH_ORIGIN_KEY,
    ResponseBranchStore,
)

from .test_responses_host import _context, _parse_sse, _request, _response_object
from .hitl.graphs import build_simple_interrupt_graph


@pytest.fixture(autouse=True)
def _isolate_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "azure.ai.agentserver.core._tracing._configure_tracing", lambda *_, **__: None
    )


def _post(
    client: TestClient,
    text: Any,
    *,
    previous_response_id: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    request = {"model": "test", "input": text, "stream": stream, "store": True}
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    response = client.post("/responses", json={**request, **kwargs})
    assert response.status_code == 200, response.text
    if stream:
        return next(
            payload["response"]
            for kind, payload in reversed(_parse_sse(response.text))
            if kind in {"response.completed", "response.failed"}
        )
    return response.json()


def _text(response: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in response["output"]
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


class _BranchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ledger: Annotated[list[str], operator.add]


def _branch_graph() -> tuple[CompiledStateGraph, list[str]]:
    executions: list[str] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        text = next(
            str(message.content)
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        executions.append(text)
        ledger = [*state.get("ledger", []), text]
        return {"ledger": [text], "messages": [AIMessage(content=",".join(ledger))]}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=InMemorySaver()), executions


@pytest.mark.parametrize("operation", ["state", "execute"])
async def test_strict_saver_rejects_checkpoint_deleted_after_preflight(
    operation: str,
) -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import (
        BranchingError,
        StrictCheckpointSaver,
    )

    graph, executions = _branch_graph()
    config = {"configurable": {"thread_id": "parent"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent = (await graph.aget_state(config)).config
    saver = graph.checkpointer
    guarded = graph.copy({"checkpointer": StrictCheckpointSaver(saver)})
    assert (await guarded.aget_state(parent)).values["ledger"] == ["A"]
    await saver.adelete_thread("parent")

    with pytest.raises(BranchingError, match="checkpoint") as failure:
        if operation == "state":
            await guarded.aget_state(parent)
        else:
            await guarded.ainvoke({"messages": [HumanMessage(content="C")]}, parent)

    assert failure.value.code == "checkpoint_unavailable"
    assert executions == ["A"]
    assert graph.checkpointer is saver


async def test_strict_saver_retains_parent_and_isolates_graph_copy() -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import (
        StrictCheckpointSaver,
    )

    graph, _ = _branch_graph()
    saver = graph.checkpointer
    guarded = graph.copy({"checkpointer": StrictCheckpointSaver(saver)})
    config = {"configurable": {"thread_id": "parent"}}
    await guarded.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent = (await guarded.aget_state(config)).config
    await guarded.ainvoke({"messages": [HumanMessage(content="B")]}, parent)
    fork = await guarded.ainvoke({"messages": [HumanMessage(content="C")]}, parent)

    assert fork["ledger"] == ["A", "C"]
    assert (await guarded.aget_state(parent)).values["ledger"] == ["A"]
    assert graph.checkpointer is saver


@pytest.mark.parametrize("stream", [False, True])
def test_response_branches_preserve_non_message_state(stream: bool) -> None:
    graph, _ = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "A", stream=stream)
        original = _post(client, "B", previous_response_id=root["id"], stream=stream)
        fork = _post(client, "C", previous_response_id=root["id"], stream=stream)
        original_next = _post(
            client, "D", previous_response_id=original["id"], stream=stream
        )
        fork_next = _post(client, "E", previous_response_id=fork["id"], stream=stream)
        regenerated = _post(
            client, "B", previous_response_id=root["id"], stream=stream
        )

    assert _text(root) == "A"
    assert _text(original) == "A,B"
    assert _text(fork) == "A,C"
    assert _text(original_next) == "A,B,D"
    assert _text(fork_next) == "A,C,E"
    assert _text(regenerated) == "A,B"
    assert regenerated["id"] != original["id"]
    for response in (root, original, fork, original_next, fork_next, regenerated):
        assert response["status"] == "completed"
        assert "_internal_metadata" not in response.get("metadata", {})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("progress", ["none", "saved", "missing", "malformed"])
async def test_recovery_uses_confirmed_origin_and_recorded_progress(
    enabled: bool, progress: str
) -> None:
    graph, executions = _branch_graph()
    config = {"configurable": {"thread_id": "root"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent_config = (await graph.aget_state(config)).config
    parent_ref = HostingRunnableConfig(parent_config).checkpoint_ref
    assert parent_ref is not None
    metadata: dict[str, Any] = {BRANCH_MODE_METADATA: BRANCH_MODE}
    if progress == "saved":
        await graph.ainvoke({"messages": [HumanMessage(content="B")]}, parent_config)
        saved_ref = HostingRunnableConfig((await graph.aget_state(config)).config).checkpoint_ref
        assert saved_ref is not None
        metadata[METADATA_LANGGRAPH_THREAD_ID] = saved_ref.thread_id
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = saved_ref.checkpoint_id
    elif progress == "missing":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = "root"
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = "deleted-checkpoint"
    elif progress == "malformed":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = "root"
    await graph.ainvoke({"messages": [HumanMessage(content="C")]}, parent_config)
    executions.clear()

    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    await server._conversation_chain_store.set(
        "child",
        BRANCH_ORIGIN_KEY,
        {
            **ResponseBranchStore._record(parent_ref, paused=False),
            "mode": BRANCH_MODE,
            "parent_response_id": "parent",
        },
    )
    context = _context(response_id="child", conversation_id=None, current_text="B")
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    context.is_recovery = True
    context.persisted_response = _response_object(
        "child", previous_response_id="parent", internal_metadata=metadata
    )
    provider = AsyncMock()
    context._provider = provider
    events = [
        event
        async for event in server.handle_create(
            _request(previous_response_id="parent"), context, asyncio.Event()
        )
    ]

    terminal = events[-1]["response"]
    if progress in {"missing", "malformed"}:
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] in {
            "checkpoint_unavailable", "invalid_branch_state"
        }
        assert executions == []
    else:
        assert terminal["status"] == "completed", terminal
        assert executions == (["B"] if progress == "none" else [])
    provider.get_response.assert_not_awaited()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_interrupted_root_is_not_replayed_or_deferred(shutdown: bool) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=False
    )
    context = _context(response_id="root", conversation_id=None)
    context.is_recovery = True
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    if shutdown:
        context.shutdown.set()
    context.persisted_response = _response_object(
        "root",
        internal_metadata={
            BRANCH_MODE_METADATA: BRANCH_MODE,
            METADATA_LANGGRAPH_THREAD_ID: "root",
            METADATA_LANGGRAPH_CHECKPOINT_ID: "partial-root",
        },
    )

    events = [
        event
        async for event in server.handle_create(_request(), context, asyncio.Event())
    ]

    assert events[-1]["response"]["error"]["code"] == "root_recovery_unsupported"
    assert executions == []
    context.exit_for_recovery.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
def test_normal_approval_and_waiting_preserved_but_second_answer_rejected(
    stream: bool,
) -> None:
    server = ResponsesHostServer(
        build_simple_interrupt_graph(),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        paused = _post(client, "What is my name?", stream=stream)
        waiting = _post(
            client, "still waiting", previous_response_id=paused["id"], stream=stream
        )
        pending = next(item for item in paused["output"] if item["type"] == "function_call")
        answer = {
            "type": "function_call_output",
            "call_id": pending["call_id"],
            "output": json.dumps({"resume": "Alice"}),
        }
        approved = _post(
            client, [answer], previous_response_id=waiting["id"], stream=stream
        )
        second_answer = {**answer, "output": json.dumps({"resume": "Bob"})}
        historical = _post(
            client, [second_answer], previous_response_id=paused["id"], stream=stream
        )

    assert paused["status"] == waiting["status"] == approved["status"] == "completed"
    assert _text(approved) == "ok:Alice"
    assert historical["status"] == "failed"
    assert historical["error"]["code"] == "unsupported_approval_branch"


def test_duplicate_response_identity_never_runs_graph_twice() -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "A")
        duplicate = client.post(
            "/responses",
            json={"input": "B", "model": "test", "store": True},
            headers={"x-agent-response-id": root["id"]},
        )

    assert duplicate.status_code in {200, 400, 409}
    assert executions == ["A"]
    if duplicate.status_code == 200:
        assert duplicate.json()["id"] == root["id"]
        assert _text(duplicate.json()) == "A"