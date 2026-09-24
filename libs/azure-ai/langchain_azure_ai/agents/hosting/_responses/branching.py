"""Internal state and strict checkpoint access for response-ID branches."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from azure.ai.agentserver.responses import ResponseContext, ResponseEventStream
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from starlette.types import ASGIApp, Receive, Scope, Send

from .checkpoint_ref import CheckpointRef
from .conversation_chain_store import ConversationChainStoreProtocol
from .task_storage_manager import TaskStorageManager

BRANCH_MODE_HEADER = "x-langchain-response-branching"
BRANCH_MODE = "checkpoint-v1"
BRANCH_ORIGIN_KEY = "langgraph_branch_origin_v1"
BRANCH_BOUNDARY_KEY = "langgraph_response_boundary_v1"
BRANCH_MODE_METADATA = "langgraph_response_branching"


class BranchingAdmissionMiddleware:
    """Stamp trusted mode selection into SDK-persisted request headers.

    Args:
        app: The next ASGI application.
        enabled: Whether fresh requests may use response branching. Incoming
            client values are removed even when the feature is disabled.
    """

    def __init__(self, app: ASGIApp, *, enabled: bool) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Preserve request bodies while replacing the server-owned marker."""
        if scope["type"] == "http":
            header = BRANCH_MODE_HEADER.encode("ascii")
            headers = [
                (name, value)
                for name, value in scope.get("headers", [])
                if name.lower() != header
            ]
            if self.enabled:
                headers.append((header, BRANCH_MODE.encode("ascii")))
            scope = {**scope, "headers": headers}
        await self.app(scope, receive, send)


class BranchingError(ValueError):
    """A branch failure with a stable, client-safe error code and message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ResponseBranchStore:
    """Keep confirmed origins separate from completed response boundaries.

    Args:
        store: The existing conversation-chain store. SDK admission remains
            responsible for a single execution owner for each response ID.
    """

    def __init__(self, store: ConversationChainStoreProtocol) -> None:
        self._store = store

    @staticmethod
    def _record(ref: CheckpointRef, *, paused: bool) -> dict[str, str]:
        return {
            "version": "1",
            "checkpoint_ns": "",
            "paused": str(paused).lower(),
            **ref.to_dict(),
        }

    @staticmethod
    def _reference(record: Any) -> CheckpointRef:
        if (
            not isinstance(record, dict)
            or record.get("version") != "1"
            or record.get("checkpoint_ns") != ""
            or record.get("paused") not in {"true", "false"}
        ):
            raise BranchingError(
                "invalid_branch_state", "The response checkpoint record is invalid."
            )
        ref = CheckpointRef.from_dict(record)
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable", "The response checkpoint is unavailable."
            )
        return ref

    async def prepare(
        self,
        *,
        response_key: str,
        parent_key: str,
        parent_id: str,
        context: ResponseContext,
    ) -> CheckpointRef:
        """Confirm the parent origin before graph execution or restore it."""
        existing = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        if context.is_recovery:
            ref = self._reference(existing)
            if (
                existing is None
                or existing.get("mode") != BRANCH_MODE
                or existing.get("parent_response_id") != parent_id
            ):
                raise BranchingError(
                    "invalid_branch_state", "The confirmed branch origin is invalid."
                )
            return ref

        provider = getattr(context, "_provider", None)
        if provider is None:
            raise BranchingError(
                "invalid_branch_state", "The Responses provider is unavailable."
            )
        parent = await provider.get_response(
            parent_id, context=context.platform_context
        )
        if parent is None or parent.get("status") != "completed":
            raise BranchingError(
                "invalid_branch_state", "The parent response must be stored and completed."
            )
        metadata = (parent.get("metadata") or {}).get("_internal_metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise BranchingError(
                    "invalid_branch_state", "The parent checkpoint metadata is invalid."
                ) from exc
        if not isinstance(metadata, Mapping):
            raise BranchingError(
                "invalid_branch_state", "The parent checkpoint metadata is invalid."
            )
        boundary = metadata.get(BRANCH_BOUNDARY_KEY)
        if boundary is not None:
            ref = self._reference(boundary)
            indexed = await self._store.get(parent_key, BRANCH_BOUNDARY_KEY)
            if indexed != boundary:
                raise BranchingError(
                    "invalid_branch_state", "The parent checkpoint records do not agree."
                )
        else:
            if BRANCH_MODE_METADATA in metadata:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent has no completed checkpoint."
                )
            ref = TaskStorageManager(dict(metadata)).checkpoint_ref
            if ref is None:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent has no completed checkpoint."
                )
            boundary = self._record(ref, paused=False)

        origin = {
            **boundary,
            "mode": BRANCH_MODE,
            "parent_response_id": parent_id,
        }
        if existing is not None and existing != origin:
            raise BranchingError(
                "invalid_branch_state", "The confirmed branch origin cannot be changed."
            )
        if existing is None:
            await self._store.set(response_key, BRANCH_ORIGIN_KEY, origin)
        return ref

    async def publish(
        self,
        response_key: str,
        stream: ResponseEventStream,
        ref: CheckpointRef | None,
        *,
        paused: bool,
    ) -> None:
        """Index this run's checkpoint before the SDK commits its terminal event."""
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable", "The response produced no checkpoint boundary."
            )
        record = self._record(ref, paused=paused)
        existing = await self._store.get(response_key, BRANCH_BOUNDARY_KEY)
        if existing is not None and existing != record:
            raise BranchingError(
                "invalid_branch_state", "The response boundary cannot be replaced."
            )
        stream.internal_metadata[BRANCH_BOUNDARY_KEY] = record
        if existing is None:
            await self._store.set(response_key, BRANCH_BOUNDARY_KEY, record)


class StrictCheckpointSaver(BaseCheckpointSaver[Any]):
    """Delegate to a saver while rejecting unavailable explicit checkpoints.

    Args:
        saver: The graph-owned saver. Ownership and lifecycle stay with its
            caller; this request-scoped adapter does not open or close it.
    """

    def __init__(self, saver: BaseCheckpointSaver[Any]) -> None:
        super().__init__(serde=saver.serde)
        self._saver = saver

    @property
    def config_specs(self) -> Any:
        """Preserve the wrapped saver's configurable fields."""
        return self._saver.config_specs

    @staticmethod
    def _validate(
        config: RunnableConfig, saved: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        requested = config.get("configurable") or {}
        checkpoint_id = requested.get("checkpoint_id")
        if not checkpoint_id:
            return saved
        if saved is None:
            raise BranchingError(
                "checkpoint_unavailable", "The required checkpoint is unavailable."
            )
        actual = saved.config.get("configurable") or {}
        if (
            actual.get("thread_id") != requested.get("thread_id")
            or actual.get("checkpoint_ns", "") != requested.get("checkpoint_ns", "")
            or actual.get("checkpoint_id") != checkpoint_id
            or saved.checkpoint.get("id") != checkpoint_id
        ):
            raise BranchingError(
                "checkpoint_unavailable",
                "The saver did not return the required checkpoint.",
            )
        return saved

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read the requested checkpoint without an empty-state fallback."""
        return self._validate(config, self._saver.get_tuple(config))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read the requested checkpoint without an async empty-state fallback."""
        return self._validate(config, await self._saver.aget_tuple(config))

    def list(self, *args: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        """Delegate checkpoint history without changing retention."""
        return self._saver.list(*args, **kwargs)

    async def alist(self, *args: Any, **kwargs: Any) -> AsyncIterator[CheckpointTuple]:
        """Delegate asynchronous checkpoint history without changing retention."""
        async for saved in self._saver.alist(*args, **kwargs):
            yield saved

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write a checkpoint using the original saver."""
        return self._saver.put(config, checkpoint, metadata, new_versions)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write an asynchronous checkpoint using the original saver."""
        return await self._saver.aput(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve the saver's pending-write semantics."""
        self._saver.put_writes(config, writes, task_id, task_path)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve the saver's asynchronous pending-write semantics."""
        await self._saver.aput_writes(config, writes, task_id, task_path)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        """Allocate versions with the original saver's version scheme."""
        return self._saver.get_next_version(current, channel)