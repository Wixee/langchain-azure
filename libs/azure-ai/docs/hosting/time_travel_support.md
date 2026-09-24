# LangGraph Time Travel in Hosting

Reviewed 2026-09-24 against the current working tree using primary sources,
18 focused host tests, and deterministic local HTTP and graph probes.

## Conclusion

A compatible checkpointed LangGraph can time travel behind either host, but
neither default HTTP adapter exposes checkpoint history, arbitrary checkpoint
selection, state editing, or an explicit replay/fork operation. Hosting an
arbitrary `Runnable` with Invocations does not itself add LangGraph capabilities.
Both hosts expose the original graph: [Responses](../../langchain_azure_ai/agents/hosting/_responses_host.py#L400) and [Invocations](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L531).

The narrower proposal to select a parent through `previous_response_id` is
feasible without a new HTTP field or an upstream SDK change in non-steerable
mode. See [Responses Parent Selection Feasibility](#responses-parent-selection-feasibility)
for the contract, local proof, and remaining production requirements.
The [proposed branching design](#proposed-responses-branching-design) records the
subsequently discussed interface and compatibility decisions; it is not yet
implemented.

## LangGraph Contract

The official [time-travel guide](https://docs.langchain.com/oss/python/langgraph/use-time-travel), [persistence overview](https://docs.langchain.com/oss/python/langgraph/persistence), and [checkpointer contract](https://docs.langchain.com/oss/python/langgraph/checkpointers) establish:

- Compile with a history-preserving checkpointer; identify the correct `thread_id` and `checkpoint_ns`.
- `get_state_history(config)` / `aget_state_history(config)` enumerate snapshots newest-first. A snapshot's config identifies its exact `checkpoint_id`.
- Replay with `invoke(None, snapshot.config)` / `ainvoke(None, snapshot.config)`. Nodes after that checkpoint execute again; a final checkpoint with no next nodes is a no-op.
- Fork with `update_state(snapshot.config, values, as_node=...)` / `aupdate_state(...)`, then invoke with `None` and the returned config. This creates a new checkpoint, preserves the original history, and applies normal reducers; `as_node` controls successor selection when specified.
- LLM calls, tools, and other external calls after the checkpoint run again and may differ. Re-executed interrupts pause again and need a new `Command(resume=...)`.

## ResponsesHostServer

- [Config construction](../../langchain_azure_ai/agents/hosting/_responses_host.py#L613) reads `(user-scoped context.conversation_chain_id, "langgraph_checkpoint")`; it does not read a client-supplied `checkpoint_id` or the selected previous response's checkpoint metadata.
- [Turn completion](../../langchain_azure_ai/agents/hosting/_responses_host.py#L971) replaces that pointer. The [store contract](../../langchain_azure_ai/agents/hosting/_responses/conversation_chain_store.py#L52) explicitly provides replacement, not per-response versioning.
- If a pointer exists, its exact checkpoint is used. If absent, config contains only the resolved thread and response context; [thread resolution](../../langchain_azure_ai/agents/hosting/_responses_host.py#L689) follows response ancestry to a conversation ID or root response ID. LangGraph then selects that thread's latest checkpoint.
- Consequently, once a shared chain pointer advances, an accepted request with an earlier `previous_response_id` does not select that response's historic graph snapshot. When the SDK assigns a fresh chain key, the normal resolved-thread fallback likewise selects latest state, not that parent's snapshot. SDK admission rules can reject a request before either path.
- [Input conversion](../../langchain_azure_ai/agents/hosting/_responses_host.py#L469) sends only current input when a graph checkpointer exists. Without one, it prepends Responses history instead. Branching chat history through an old response ID is not replay of graph state, node progress, or interrupts.

### SDK Chain Identity

The [2.1.0b2 release implementation][sdk-chain-release] and [upstream implementation][sdk-chain-main] agree on these cases:

| Request / option | `conversation_chain_id` basis |
| --- | --- |
| Explicit conversation | Conversation partition plus agent/session scope |
| No conversation, steering enabled | Embedded partition of `previous_response_id` or initial `response_id`, plus agent/session scope |
| No conversation, steering disabled | Current `response_id` verbatim |

Steering defaults to `False` in the checked [release options][sdk-options].
The SDK chain key is not a parent-response checkpoint ID, nor is it universally
the literal root response ID. Shared embedded partitions identify steerable
chains; non-steerable requests have separate keys. The host's ancestry-derived
LangGraph thread ID is a separate concept. Thus the "one mutable pointer for
the whole chain" explanation applies to shared-key modes, not every SDK mode.

## Responses Parent Selection Feasibility

This section assesses a proposed change, not a feature already implemented in
the default host. Invocations is outside this proposal's scope.

### External Contract

The OpenAI [migration guide][openai-migration] explicitly says that
`previous_response_id` can create "response chains or forks". The
[create reference][openai-create] makes it mutually exclusive with
`conversation` and states that previous top-level `instructions` are not carried
forward. The [conversation-state guide][openai-state] describes stored response
context and its retention limitations.

For a completed response A and a response B created with parent A:

| User operation | New request | Intended state |
| --- | --- | --- |
| Continue B | Parent B, new input D | A, B, D |
| Fork from A | Parent A, new input C | A, C, independent of B |
| Regenerate B | Parent A, resend B's input | A, new B attempt with a new response ID |
| Regenerate the first turn | No parent, resend its input | New independent root response |

Referencing B means continuing after B, not rerunning B. Regeneration creates
another response; it is not a promise of idempotent HTTP retry, identical model
output, or exactly-once tool effects. Crash recovery of B and SSE event replay
remain separate operations. A response ID selects a response boundary, not an
arbitrary node or super-step inside that response.

### Local Implementation Route

The required selection is:

```text
authorized previous_response_id
-> saved parent (thread_id, checkpoint_id)
-> graph execution with only the new request input
-> new checkpoints and a new response ID
```

No graph copy or additional public `checkpoint_id`, `retry`, or `fork` field was
needed for the completed, non-paused turns proved below. LangGraph can retain
multiple descendants in one thread when every invocation pins its parent
checkpoint. This result does not prove isolation of HITL resume state or strict
failure when the selected checkpoint disappears during restore.

An important existing capability makes this practical: without an explicit
conversation and with steering disabled, the SDK uses the **current response
ID** as its chain key [source][sdk-chain-release]. The host already
[stores each run's checkpoint under that key](../../langchain_azure_ai/agents/hosting/_responses_host.py#L971).
However, [config construction](../../langchain_azure_ai/agents/hosting/_responses_host.py#L613)
reads the incoming response's chain key instead of the selected parent's key.
Its fallback then loads latest state from the common LangGraph thread.

For this mode, looking up the user-scoped **parent response ID** in the existing
[chain store](../../langchain_azure_ai/agents/hosting/_responses/conversation_chain_store.py)
and passing its reference to
[`HostingRunnableConfig.create_from_checkpoint`](../../langchain_azure_ai/agents/hosting/_responses/hosting_runnable_config.py)
is sufficient for the successful-turn scenarios proved below. Formalizing an
immutable per-response mapping is preferable to assuming every SDK mode will
always use response IDs as chain keys. Keep any mutable conversation-head pointer
separate from that mapping.

An alternative source is the per-response metadata already written by
[`TaskStorageManager`](../../langchain_azure_ai/agents/hosting/_responses/task_storage_manager.py).
The SDK [provider protocol][sdk-provider] has tenant-context-aware `get_response`,
but [ResponseContext][sdk-context] does not expose a public parent-response
getter; the existing ancestry resolver reaches its private `_provider` field.
The design below requires the stored response as completion authority, not just
a host-owned mapping. Access to the effective provider through a supported
integration must therefore be verified before implementation.

### Local Proof

A disposable subclass outside the repository overrode only
`build_runnable_config`: on a fresh parent-linked request it read the parent's
existing store entry and returned an exact checkpoint config. The production
host, SDK, and dependencies were not modified. The graph stored a non-message
`ledger` channel, so these checks demonstrate graph-state branching, not just
filtered chat history. Concurrent requests synchronized inside the graph before
writing, ensuring overlapping execution.

| Probe | Observed result |
| --- | --- |
| Unchanged host: A, B after A, C after A | `A, B, C` |
| Subclass: same requests | `A, C` for C |
| Continue B after C exists | `A, B, D` |
| Continue C after D exists | `A, C, E` |
| Regenerate B from A | `A, B`, with a distinct response ID |
| Concurrent siblings from A | `A, parallel-left` and `A, parallel-right` |
| Parent mapping after all descendants | A's checkpoint reference unchanged |

All subclass checks passed through real local HTTP handling in three modes:
foreground JSON, foreground SSE, and background SSE with
`resilient_background=True`. All used `steerable_conversations=False`, stored
responses, and no explicit conversation. Foreground cases used the SDK's
in-memory response provider; the background case used its file-backed provider
in an isolated temporary state directory. All graph checkpoints and the custom
chain store were in memory. Runtime versions match the verification section
below. This proves execution feasibility, not cross-process recovery or cloud
storage correctness. The disposable script and state were removed afterward.

Follow-up design-review probes used isolated in-memory graphs on LangGraph
`1.2.11`, without changing production code:

| Probe | Observed result and design consequence |
| --- | --- |
| Check existence, delete the checkpoint, then invoke with its original config | LangGraph executed new input from empty state. A preflight lookup alone is insufficient. |
| Replay root input on the same thread after the saver advanced but without a response-side checkpoint | Additive state was duplicated. The agreed new-mode policy below terminates failed roots instead of replaying them. |
| Resume the same paused checkpoint with `approve`, then with `reject` | Both results contained `approve`. Ordinary HITL continuation and independent historical approval branches must not be treated as equivalent. |

The checked SDK [checkpoint persistence implementation][sdk-execution] logs
provider failures without raising them back into the handler. Yielding
`stream.checkpoint()` is consequently not an acknowledged write for newly
required origin metadata. These findings motivate the design; they are not
verification of its proposed fixes.

### SDK Constraint

With a task manager active, the SDK [primitive selector][sdk-orchestrator]
uses a one-shot task for non-steerable response chains without `conversation`.
It uses a multi-turn task for explicit conversations or steering, and passes
`previous_response_id` as `if_last_input_id` only on that multi-turn path. The
[execution orchestrator][sdk-execution] propagates failed head preconditions;
the endpoint translates them to `conversation_fork_not_supported`.

This task routing also applies to stored foreground requests, not only resilient
background requests. If the task subsystem is disabled, the SDK can fall back
to in-process execution, so disabling crash recovery alone is not a reliable
fork policy. The earlier steering probe observed HTTP 409 for an old parent.
Supporting historical forks **while retaining steering** needs a separate
upstream-compatible admission/task-identity design; a graph config override
cannot bypass rejection before the handler runs. Do not silently disable an
existing steering configuration.

### Production Requirements

The proof establishes successful-turn feasibility, not a production feature.
The following design addresses the discussed compatibility, persistence, and
failure requirements. Its acceptance checks remain implementation work, not
additional results from the local probe.

## Proposed Responses Branching Design

Status: design recorded from the discussion on 2026-09-24; not implemented.
The hard compatibility constraint is that existing behavior must not change
unless the application explicitly enables the new feature.
Agreed behavior and proposed mechanisms are separated below. Integration points
that still need a prototype are listed as implementation gates, not as solved
capabilities.

### Host Interface

Add a keyword-only `enable_response_branching: bool = False` parameter to
`ResponsesHostServer`. It controls graph-state branching, not automatic saver
creation. Proposed usage, after implementation:

```python
graph = builder.compile(checkpointer=saver)
server = ResponsesHostServer(graph, enable_response_branching=True)
```

The saver belongs to the compiled graph. Reuse it; do not require a second saver
argument on the host. Client requests keep using `previous_response_id` with
the existing Responses schema. No new endpoint or public `checkpoint_id`,
`retry`, or `fork` request field is needed.

| Configuration | Required behavior |
| --- | --- |
| Flag omitted or `False` | New requests retain existing execution, storage access, validation, errors, and fallback behavior. Already admitted tasks retain their recorded recovery mode. |
| Flag `True`, graph has no usable saver | Raise `ValueError` in `__init__`, before creating the SDK host or registering its handler. |
| Flag `True`, graph has a history-preserving saver | Enable exact parent selection for eligible response-ID chains. |
| Flag `True` with steering enabled | Reject the unsupported configuration explicitly; never silently disable steering. |
| Explicit `conversation` request | Keep the existing conversation path, outside the new branching semantics. |
| Root request on the new path fails or crashes | Do not automatically resume or replay it, even when background resilience is enabled. Successful roots still publish a boundary for later requests. |

Initialization validates local configuration only, without storage network
requests. It must also validate the graph when attaching to an existing `app`.
A disabled or inherited checkpointer marker is not itself a saver configured
for this root host. Keep new validation separate from legacy detection so
disabled-feature behavior does not change.

When attaching to an existing `app`, validate its effective steering settings,
not a host `options` argument that the attached app ignores. If those settings
cannot be established through a supported interface, fail the new opt-in
configuration explicitly rather than assuming steering is disabled. Existing
attachment behavior with branching disabled remains unchanged.

Suggested missing-saver error:

```text
enable_response_branching=True requires a graph compiled with a checkpoint saver.
Configure checkpointer when creating the graph.
```

A history-preserving `InMemorySaver` is sufficient for local, single-process
use. Production durability and cross-process recovery require persistent
storage, such as `FoundryCheckpointSaver`. A shallow/latest-only saver cannot
satisfy historical selection. Configuration checks do not guarantee that any
particular checkpoint still exists; availability is checked at restore time.
The presence of a saver object does not certify a custom implementation's
history support. State historical reads as a required saver contract, validate
the usable local interface, and enforce exact reads at runtime; do not probe
storage or infer capability solely from a class name during initialization.
Do not automatically supply an in-memory saver or downgrade to message-history
branching when this feature is requested.

### Compatibility Scope

- Keep the flag off by default. Requests admitted on the legacy path do not gain new snapshot reads/writes or new validation failures. Recovery follows the mode recorded at admission, not a later flag change.
- Preserve existing explicit-conversation behavior, including currently accepted mixed `conversation` / `previous_response_id` requests. Do not introduce global OpenAI-style mutual-exclusion validation in this feature.
- Do not change `instructions`, cancellation, SSE replay, model configuration, or other unrelated Responses behavior. This is parent-selection compatibility, not a claim of complete OpenAI field compatibility.
- Do not change `resilient_background` or steering settings automatically. Background execution remains independently configured.
- Keep Invocations, public override-hook signatures, and the existing chain-store interface unchanged. Custom hooks that replace the default pipeline must honor the new contract when opting in.
- Failed-root termination applies only to requests admitted on the new path. Do not change legacy root recovery or disable the SDK's resilience configuration globally.
- Ordinary HITL continuation remains supported. Do not reject all paused checkpoints simply because branching is enabled; independent historical approval branches are a separate, out-of-scope capability.

### State Ownership

Keep these state roles distinct:

| Reference | Purpose | Update rule |
| --- | --- | --- |
| Existing conversation checkpoint pointer | Preserve legacy next-turn behavior | Keep the existing identity and semantics. |
| Confirmed origin record | Fix the starting point of a parent-linked request before graph execution | Write once for that response; identical retries are idempotent. |
| Per-response boundary checkpoint | Select a stable parent for a new response | The persisted completed response is authoritative; an independent index is only a lookup aid. |
| Current response execution checkpoint | Recover the same interrupted parent-linked task | Advances through the existing durable response-checkpoint mechanism; it does not authorize recovery of a failed new-mode root. |

Reuse `ConversationChainStoreProtocol.get/set` for origin records and boundary
indexes, with distinct versioned keys and trusted user/deployment-scoped
response identities.
Do not replace the existing `langgraph_checkpoint` record or assume every SDK
chain key is a response ID. The response store owns response status and recovery
metadata; the LangGraph saver owns actual graph state. Neither a response object
nor a boundary-reference dictionary substitutes for that state.

The minimal origin record contains a schema version, admitted mode, parent
response ID, and exact parent reference (`thread_id`, `checkpoint_ns`,
`checkpoint_id`; the root graph namespace is `""`). The current response is
identified by the scoped record key. Keep origin and execution progress separate;
an origin reference is not evidence that any node has run. Persist the admitted
route in the SDK-owned response/admission state as well, so recovery does not
infer it from a later flag value. Exact key names and version handling must be
fixed and tested before release; malformed or unknown new-mode records fail
explicitly rather than falling back to legacy execution.

### Boundary Publication

Put the exact final reference captured from the specific run into its response's
internal metadata, including successful root responses. Never obtain it afterward
through a latest-thread lookup. The reference and the `completed` status must be
part of the same persisted response envelope; an independent index cannot
declare completion. Internal metadata must remain absent from client-facing
output.

Proposed order:

1. Capture the run-specific final checkpoint reference and attach it to the response's internal metadata.
2. Await the boundary-index write. A write failure stops publication and surfaces an error; it must not publish a usable parent.
3. Emit the terminal response through the SDK, which owns response persistence and the final wire event. Do not write a competing terminal envelope directly from the handler.
4. A future parent lookup checks the authorized persisted response, its terminal status and boundary metadata, any required index consistency, and actual checkpoint availability.

| Publication state | Parent eligibility |
| --- | --- |
| Index exists, but the persisted response is absent, running, failed, or cancelled | Not eligible. The index alone is insufficient. |
| Persisted response is completed, references agree, and the checkpoint is readable | Eligible, subject to the HITL continuation rules below. |
| Index and completed response disagree | Fail explicitly; do not choose either value heuristically. |
| Completed response or a required reference/checkpoint becomes unavailable | Fail explicitly; do not reconstruct state from a mutable conversation head. |

The SDK `2.1.0b2` [terminal persistence path][sdk-execution] attempts to save the
response before returning the terminal event and maps storage failures to an
error. Unlike its intermediate `stream.checkpoint()` handling, this provides a
completion decision the new feature can consume. Preservation of the internal
boundary reference must be tested through foreground JSON, SSE, and background
paths, including subsequent retrieval and restart.

No distributed transaction between the index and response store is proposed.
An index written before a crash is harmless only while parent admission still
requires the authoritative completed response. Retry identical publication
idempotently; never replace an already committed response boundary with a
different checkpoint. Concurrent branches have different response identities.
Publication for one response must have a single owner across recovery; verify
that SDK admission provides this guarantee. An unguarded `get` followed by
`set` does not supply compare-and-set or multi-writer immutability.

Boundary publication happens after graph execution, so its failure does not
undo tool effects already performed. Do not rerun completed graph work merely
to repair an index. Keep transient index records under the existing retention
policy instead of adding a cleanup transaction.

### Checkpoint Retention

Creating, retrying, or completing a branch must not delete its parent checkpoint
or prune the parent's checkpoint history. Each descendant records its own
checkpoints, and multiple branches may reference the same retained parent.
Checkpoint deletion is not a prerequisite or a step in branching.

All stores needed for recovery must survive restarts. Checkpoint expiration or
deletion belongs to the configured storage TTL/retention policy or explicit
external cleanup, not to the branching flow. The feature must not introduce
automatic cleanup or change those policies. Retention across stores should be
coordinated, but referencing a parent does not guarantee it will be retained
forever. If it becomes unavailable, restore must fail explicitly.

External tools, shared long-term stores, files, and other resources outside the
checkpoint are not rolled back or automatically branch-isolated. Applications
remain responsible for idempotent or replay-safe side effects.

### Parent Selection and Recovery

For a fresh request on the new path:

1. With neither a parent nor an explicit conversation, start an independent root. Record its route in the response lifecycle, but do not create a special recoverable empty origin. Absence of a parent checkpoint is normal here.
2. With `previous_response_id`, preserve provider existence and authorization checks, require an eligible stored parent, and resolve the exact reference recorded in its completed response. An opaque response ID or a standalone index is not authorization or proof of completion.
3. Save the parent-linked origin record through `ConversationChainStoreProtocol.set` and await its successful return before graph execution. If the write fails, do not run graph nodes. Duplicate admission must not change the recorded origin.
4. Restore through the strict-read path below and pin graph execution, state lookups, and interrupt handling to the selected checkpoint. Use the current request input, or the existing HITL resume command for a matching pending interrupt.

Use a storage call whose errors reach the caller to confirm the origin write.
Yielding `stream.checkpoint()` alone is insufficient: the checked SDK logs
intermediate persistence errors without raising them into the handler. Keep
the normal response-stream snapshots for execution recovery, but do not claim
that every emitted checkpoint event was successfully persisted.

For a failed or crashed root admitted on the new path:

- Do not automatically resume or replay it, even if its saver already contains intermediate checkpoints. A user retry creates a new root response.
- Do not publish a branchable boundary from partial work. If the SDK later re-enters the handler for recovery, terminate that response as failed without invoking the graph; do not defer it into a recovery loop.
- Do not add a special retention or cleanup mechanism. Existing response lifecycle records and saver-managed intermediate records may remain under their configured TTL; this is not a promise that the attempt leaves no stored data.
- A successfully completed root still publishes its final boundary. A root response that intentionally completes with an HITL approval request is not a crashed root.

For recovery of an already admitted parent-linked response:

1. Prefer its own durably recorded execution checkpoint and continue with input `None`. Do not replace it with a parent checkpoint or the latest thread state.
2. If no execution checkpoint has yet been durably recorded, replay the original input or approval from its confirmed parent origin. This is different from a recorded execution checkpoint becoming unavailable, which must fail.
3. Once the origin is confirmed, recovery must not depend on rereading the parent response. The referenced graph state must still exist and remain readable. A missing, corrupt, or unsupported origin record is an error, not permission to choose latest state.
4. Preserve the admitted mode across restart or configuration changes. Existing legacy tasks keep their legacy recovery path; a new-mode task must not silently become legacy because required metadata is missing. Proving this distinction at the first durable admission is an implementation gate.

Only confirmed response-side progress authorizes mid-run continuation. If the
graph saver advanced beyond that snapshot, unconfirmed work may run again after
recovery. This design does not provide exactly-once tool effects. Preserving
the existing snapshot protocol avoids an unrelated recovery rewrite, but its
failure and retry behavior must remain explicit to applications.

The first scope uses completed response boundaries, not mutable checkpoints of
in-progress, failed, or cancelled responses. A running response already has an
ID; that fact alone does not make it an eligible parent.

A request using `store=false` may consume an otherwise eligible stored parent,
but its new response is not thereby guaranteed to be a durable future parent.
Parent response availability, authorization, boundary eligibility, and actual
checkpoint availability must all hold independently.

### HITL Compatibility

HITL is an existing workflow to preserve, not a requirement to add arbitrary
approval-history branching. In this host, the response and the graph have
different lifetimes:

1. The response ID exists before graph execution and is carried by the initial response events.
2. At an HITL pause, the host emits approval/tool-call items and then `response.completed`. The response is finished, while the graph remains paused in its checkpoint.
3. The client submits a new request with the approval result. It may use top-level `previous_response_id` to reference the completed paused response, or the existing explicit-conversation path. `approval_request_id` / `call_id` identifies the interrupt, not the parent response.

This is visible in the [host's completion path](../../langchain_azure_ai/agents/hosting/_responses_host.py#L996)
and the [background approval example](../../../../samples/hosting/langgraph-hosted-agents/responses/10_resilient/README.md#L197).
Filtering only on response status `completed` therefore does not exclude HITL.

| Request | First-scope behavior |
| --- | --- |
| Normal approval of an active pending interrupt | Preserve the existing matching, validation, rejection, and resume protocol; do not blanket-reject the paused parent. |
| Ordinary input with no matching approval while the selected graph is paused | Preserve existing pending-interrupt handling; do not bypass the pause to execute fresh work. |
| Independent branches from a completed, non-paused graph boundary | Support the parent-selection behavior proved above. |
| Return to an already answered approval checkpoint with a different answer, or create independent sibling approvals from the same pause | No new support in the first scope. Do not advertise isolated HITL forks based only on checkpoint pinning. |

The follow-up probe showed that repeated resume against one paused checkpoint
can reuse the first answer. The implementation must distinguish ordinary active
continuation from unsupported historical or competing approval branches and
define explicit handling for the latter before release. It must not silently
substitute an earlier decision. The admission/isolation mechanism still needs
validation; preserving normal HITL is not evidence that this problem is solved.
Existing duplicate/unmatched approval behavior on the legacy path stays intact.

### Restore Failure Contract

For the new branching path, restore the selected checkpoint or fail explicitly.
Never silently use the latest checkpoint, an empty state, a different parent,
or a full-history rerun as a substitute for a failed restore.

| Failure | Required handling |
| --- | --- |
| Missing, expired, deleted, or invalid parent checkpoint reference | Fail with a stable machine-readable reason; `checkpoint_unavailable` is a proposed code. |
| Reference exists but the actual checkpoint is absent | Fail before graph execution. A surviving index entry does not prove recoverability. |
| Storage timeout or other transient backend failure | Preserve the failure category. Any retry must follow the configured policy and target the same checkpoint. |
| Authorization or deserialization failure | Preserve the appropriate error category without leaking inaccessible state; do not reinterpret it as an empty graph. |

Do not report TTL expiration as a confirmed cause merely because a record is
absent. The saver may return `None` rather than raise on a missing checkpoint;
the host must handle this explicitly. Proposed mechanism: a request-scoped
checkpointer adapter or a supported LangGraph invocation hook must guard the
actual reads of required checkpoint IDs. A `None` result, a mismatched checkpoint
identity, or a failed read must raise before graph nodes execute; it must not
reach LangGraph's empty-state fallback. Preserve backend error categories and
only retry the same requested checkpoint under the configured retry policy.

Do not globally replace or mutate the user's saver on a shared graph, and do
not change legacy reads or ordinary writes. The adapter/hook must work for
state and interrupt inspection as well as execution, preserve saver lifecycle,
and distinguish required restores from legitimate creation of new checkpoints.
Its supported integration point and concurrency behavior require a focused
prototype. A preflight lookup followed by an unguarded `astream` is not an
acceptable substitute, because expiration or external cleanup can occur between
them. A failed restore must not execute graph nodes or produce new tool effects.

The review's deliberate deletion was fault injection in a temporary in-memory
saver, simulating a checkpoint becoming unavailable between validation and
restore. It did not access production storage and is not an implementation step
for branching. The intended behavior is to retain and reference the parent,
then fail explicitly if that checkpoint is no longer available.

Use existing Responses error mechanisms. Before a stream or background request
is accepted, errors can use the available request-error path. After acceptance
or SSE headers have been sent, surface a terminal `response.failed` with the
appropriate code rather than assuming the HTTP status can still be changed.
Freeze stable codes and their retry semantics before implementation is released:
checkpoint/reference unavailability, invalid or conflicting branch state,
storage failure, and unsupported historical approval branching must be
distinguishable where the client can act on them. Reuse SDK error categories
where appropriate; `checkpoint_unavailable` remains a proposed new code pending
schema validation. Do not leak private checkpoint state or credentials in
client-visible errors. Verify the same logical failure across JSON, SSE,
background retrieval, and recovery instead of treating error mapping as an
unobservable implementation detail.

These strict rules apply to the new path, including its recovery operations;
they do not change legacy fallback behavior. Fresh roots do not require a parent
checkpoint. Failed new-mode roots terminate rather than recover; parent-linked
tasks without confirmed execution progress may replay only from their confirmed,
still-readable origin.

### Long-Running Behavior

Assume A is completed and B is a long-running response started from A:

| Operation | Intended behavior after implementation |
| --- | --- |
| Continue or fork after B completes | Use B's fixed boundary checkpoint. |
| Create C from A while B is still running | Run an independent sibling; do not automatically cancel B. |
| Retry B from A after B fails | Submit B's input again with parent A, creating a new response ID. |
| Recover parent-linked B after a process crash | Use its confirmed execution checkpoint, or its confirmed parent origin when no progress was durably recorded. Required state missing means failure. |
| The initial root request fails before publishing a completed response | Do not resume or replay it automatically. A client retry starts a new root. |
| B completes its response with an HITL approval request | Continue through a new ordinary approval request; do not keep B's response open while waiting. |
| Fork from an arbitrary internal step while B is running | Out of scope; B's response ID does not identify a particular intermediate checkpoint. |

Long runtime does not change selection semantics or the failed-root policy.
Recovery of eligible parent-linked tasks still requires the existing resilience
configuration and persistent task, response, origin/boundary-reference, and
graph stores. The local proof covered
background execution and concurrent branches, not hours-long runs, process
restart, or a deployed Foundry backend.

### Existing Data

Do not rewrite existing records or require a destructive migration. A trusted
legacy per-response snapshot may be reused only when its identity, eligibility,
and exact checkpoint are verifiable. A shared mutable conversation pointer is
never evidence of a historical response boundary. If no trustworthy reference
exists, reject its use on the new branching path rather than guessing; callers
can retain legacy mode or start a new root with the feature enabled.

### Implementation Scope and Acceptance

Keep the implementation local to Responses hosting, but treat the earlier file
counts as a preliminary estimate, not a limit or verified guarantee:

| Area | Expected changes |
| --- | --- |
| Initially estimated 2-3 hosting source modules | [Host constructor](../../langchain_azure_ai/agents/hosting/_responses_host.py#L318), selection, publication, and execution handling; origin/boundary records and recovery metadata in existing helpers. Strict-read and HITL admission integration may require additional focused work. |
| Approximately 1-2 test modules | Extend existing Responses host tests and reuse current fixtures. |
| Approximately 1-2 documentation/sample locations | Explain opt-in setup, saver requirements, branching, and background behavior. |
| No planned changes | Invocations, dependencies, endpoints, chain-store protocol, or persistent backend implementations. Ordinary completed-turn forks were proved without an SDK change; the full design still depends on the integration gates below. |

Implementation gates:

- Prove a supported, request-scoped strict-read integration that cannot affect concurrent legacy runs or silently restore empty state.
- Verify effective provider/options access for both host-created and injected `app` instances, including tenant-context-aware parent reads and stored internal metadata.
- Prove the first durable admission identifies new-mode roots/children versus legacy tasks. Define missing/unknown metadata behavior and ensure root recovery is terminated rather than silently reclassified or retried.
- Verify per-response single-writer ownership, idempotent boundary publication, and the SDK terminal-persistence failure paths. Do not assume atomic replacement is conditional creation.
- Preserve ordinary HITL while detecting unsupported historical or competing approvals; specify and test their behavior without claiming independent paused-checkpoint branches.

If a gate cannot be satisfied through current supported interfaces, revise the
implementation scope explicitly instead of weakening the contract or adding
private SDK patches implicitly. The earlier local proof does not settle these
gates.

Required checks before shipping:

- Requests admitted with the flag omitted or explicitly `False` preserve existing results, event flow, fallback behavior, and storage calls; existing regression tests continue to pass.
- Missing/invalid saver fails during construction before SDK setup, including `app` attachment; initialization performs no storage I/O and does not auto-create a saver. Effective steering configuration is validated rather than guessed.
- Root creation, linear continuation, sibling forks, regeneration, and continuation of each branch preserve non-message graph state and immutable parent references.
- Branch creation, regeneration, and completion do not delete parent checkpoints, prune parent history, or change configured TTL/retention policies.
- Concurrent branches cannot capture one another's checkpoint; verify on the intended persistent saver, not only `InMemorySaver`.
- Origin writes are acknowledged before graph execution; failed writes invoke no nodes. Missing/corrupt origin data on recovery fails explicitly instead of selecting latest state.
- Boundary-index writes alone do not admit parents. Cover terminal-persistence failure, missing/mismatched references, conflicting publication, and recovery between index and terminal writes.
- Missing references, expired/deleted checkpoints, restore-time disappearance, authorization failures, and backend errors fail explicitly without graph/tool execution or latest-state fallback.
- Foreground JSON, foreground SSE, and resilient background execution preserve their response/error contracts.
- New-mode roots that fail before completion are not replayed, including SDK recovery re-entry after saver advancement. Successful roots publish usable boundaries; legacy root recovery is unchanged.
- Parent-linked recovery before the first durable execution checkpoint, mid-run recovery, and interruption around publication use the recorded origin/progress and admitted mode. Missing required progress must not fall back to the parent.
- Ordinary HITL approvals, rejections, parallel pending interrupts, and response-ID/conversation linkage preserve existing behavior. Historical/competing approval requests must not silently inherit another branch's answer; independent historical approval forks remain unsupported.
- Cancellation, legacy conversations, flag changes, and unsupported steering combinations retain their defined behavior. Failed roots must not enter an automatic recovery loop.
- User/deployment isolation, old records, unstored responses, retention mismatch, and the supported Python/dependency versions are covered.

The main verification effort is compatibility and recovery correctness, not
the parent lookup itself. Automatic recovery of failed new-mode roots, arbitrary
node-level time travel, independent historical HITL forks, steering-compatible
historical forks, and unrelated OpenAI field changes remain outside this scope.

## InvocationsHostServer

- The [default parser](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L616) requires non-empty `message` text or structured HITL items. It reads `stream`; [execution options](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L664) read `background` and `previous_invocation_id`. No default field selects a graph checkpoint or supplies state edits.
- [Foreground config](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L698) and [task config](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L722) select a user-scoped session thread, without a client-selected checkpoint.
- [Task admission](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1123) maps `previous_invocation_id` to `if_last_input_id`. The [SDK precondition][sdk-task] compares it with the last accepted input ID, not graph history. A mismatch on an existing head produces [HTTP 409](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L944); with no stored head the SDK accepts and seeds it. Without task-backed mode, the host [rejects the option](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L863) with HTTP 400.

## Backend and Related Features

- `FoundryCheckpointSaver` is [async-only](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L111). It implements [exact-ID/latest lookup](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L171), [newest-first history](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L206), [append-only checkpoint writes with parent IDs](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L296), and [restored parent configs and pending writes](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L475). These are the backend primitives for async graph time travel, not evidence of an HTTP feature or runtime conformance.
- Its history listing requires a known thread; namespaces, `before`, metadata filters, and limits are supported. Use `aget_state_history`, `aupdate_state`, and `ainvoke` with this saver. Retained checkpoints and the correct user partition are required; [default TTL](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L43) is 30 days, and expiration/deletion removes available history.
- Crash recovery: [Responses](../../langchain_azure_ai/agents/hosting/_responses_host.py#L891) and [task-backed Invocations](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1296) restore internally recorded checkpoint IDs and pass graph input `None`. This continues interrupted work, not an arbitrary user-chosen past run. Cross-process recovery requires surviving stores; Responses also requires the [resilience flags and providers][sdk-resilience].
- SSE replay: the [Responses reconnect contract][sdk-resilience] returns stored events after `starting_after`, then live-tails. [Invocations event subscription](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1860) also consumes events. Replaying transport events is not re-executing graph nodes.
- HITL resume: [Responses](../../langchain_azure_ai/agents/hosting/_responses_host.py#L558) and [Invocations](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L760) map matching approval/tool-output items to a resume command for currently pending interrupts. This does not expose historical checkpoint selection.

## Extension Hooks

- Direct Python access through `host.graph` retains native LangGraph APIs when the hosted object and saver support them. Preserve the host's user-scoped thread identity and storage context.
- Responses provides `build_runnable_config`, `build_input`, and [custom `handle_create`](../../langchain_azure_ai/agents/hosting/_responses_host.py#L797); a custom SDK response handler is another option. A true time-travel interface must define authorized checkpoint selection, replay input `None`, forks, and branch bookkeeping. Replacing the chain store alone does not supply those controls.
- Invocations provides `parse_request`, `parse_execution_options`, `build_input`, and separate foreground/task config hooks. Overriding only the foreground config misses task-backed execution. Simply returning `None` from `build_input` is also insufficient: default handlers [short-circuit foreground](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L881) and [fresh task-backed](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1329) `None` inputs instead of invoking a replay. A custom execution route/handler is needed for that operation.

## Verification Boundaries

Local verification used Python 3.14, LangGraph `1.2.11`, Agent Server Core and
Responses `2.1.0b2`, and Invocations `1.1.0b1` in a temporary uv-managed overlay.
The project environment, dependency files, and host implementations were not
changed. The 18 selected tests in the existing Responses and Invocations host
modules passed; SDK telemetry setup was disabled in the verification process.

Additional probes used real local HTTP adapters through Starlette `TestClient`,
a deterministic two-node graph, `InMemorySaver`, and the existing fake Foundry
state-store fixtures. Each row below used an isolated graph unless noted.

| Probe | Observed result |
| --- | --- |
| Checkpointed Responses: A, then B after A, then C referencing A | Graph saw `A, B, C`, not `A, C`. |
| Same graph: next request also supplies A's top-level `checkpoint_id` | The field did not select A; graph saw `A, B, C, E`. |
| Responses without a graph checkpointer: A, B after A, C after A | History branch saw `A, C`; this is transcript branching only. |
| Steerable Responses: A, B after A, C after A | HTTP 409, `conversation_fork_not_supported`. |
| Foreground Invocations: A, B, C with A's `checkpoint_id` and `config` | Graph saw `A, B, C`; supplied checkpoint configuration was ignored. |
| Foreground Invocations with `previous_invocation_id` | HTTP 400 because task-backed mode was not enabled. |
| Task-backed Invocations: A, B after A, C with A's invocation ID | HTTP 409: `previous_invocation_id does not match the latest turn.` |
| Native graph replay from A's pre-report checkpoint | `ainvoke(None, snapshot.config)` produced only A. |
| Native graph fork from that checkpoint, adding D | `aupdate_state(..., as_node="remember")`, then `ainvoke(None, ...)`, produced `A, D`. |

SDK source review additionally covered release tag
`azure-ai-agentserver-responses_2.1.0b2` and commit
`4993311d43a35b34cb47220eff0df969d73d7832` ([Responses version `2.2.0b2`][sdk-version]).
The [package requirements](../../pyproject.toml#L1) allow more versions than were
executed. No deployed Foundry service, persistent Foundry saver, or model was
called; backend capabilities above are source-verified, not cloud-tested.

[sdk-chain-release]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_chain_id.py
[sdk-chain-main]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_chain_id.py
[sdk-options]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/_options.py
[sdk-task]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-core/azure/ai/agentserver/core/tasks/_decorator.py
[sdk-resilience]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md
[sdk-version]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/_version.py
[openai-migration]: https://developers.openai.com/api/docs/guides/migrate-to-responses
[openai-create]: https://developers.openai.com/api/reference/resources/responses/methods/create
[openai-state]: https://developers.openai.com/api/docs/guides/conversation-state
[sdk-provider]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/store/_base.py
[sdk-context]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/_response_context.py
[sdk-orchestrator]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_resilient_orchestrator.py
[sdk-execution]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_orchestrator.py
