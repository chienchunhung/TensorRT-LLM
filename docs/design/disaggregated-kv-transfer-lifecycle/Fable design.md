# Disaggregated Inference Request Lifecycle — Fable Design

| | |
|---|---|
| **Status** | Design proposal |
| **Created** | 2026-07-25 |
| **Scope** | TensorRT-LLM disaggregated prefill/decode serving |
| **Transceivers** | Python and C++ |
| **Related incident** | NVBUG 6480621 |
| **Related implementation** | [PR #16396](https://github.com/NVIDIA/TensorRT-LLM/pull/16396), [PR #16347](https://github.com/NVIDIA/TensorRT-LLM/pull/16347) |
| **Companion doc** | [GPT design](./GPT%20design.md) |

## Executive Summary

Disaggregated serving does not need a coordinated request state machine between
CTX and GEN. It needs coordinated *cross-side obligations and terminal
outcomes*: admission, KV ownership, transfer completion, cancellation, and
attempt termination. Scheduler states stay local and are free to diverge.

This document derives that position from a survey of production disaggregated
systems (vLLM, SGLang, Dynamo, DistServe, Mooncake, Splitwise, TetriInfer,
MemServe, P/D-Serve, DéjàVu, llm-d), from TRT-LLM observations (NVBUG 6480621's
asymmetric timers; the current control plane's gap inventory), and from first
principles. No surveyed system runs a distributed request state machine; every
system that approached the need found a way to eliminate it — single ownership
(Dynamo), leases (vLLM), one admission handshake (SGLang), or predictive early
rejection (Mooncake). The residual failure modes in each system map precisely
to the windows their chosen mechanism does not cover.

The design: four state domains with a single writer each; one cross-side
handoff-attempt protocol that both context-first and generation-first flows
instantiate as schedules over the same event set; four independent timers;
renewable obligation leases as ground truth for cross-node resource pins,
composing with PR #16396's evidence-gated ownership as the local safety layer;
attempt fencing with two-level terminality; a coordinator that owns routing and
client-facing outcomes but only rebuildable shadow resource state; and a
connector capability contract that lets the C++ and Python transceivers satisfy
the same lifecycle contract through different mechanisms.

---

## 1. Design principles (each validated by a production incident)

**P1 — Hard state vs. soft state.** KV block ownership is *hard* state:
mishandling it corrupts inference or leaks GPU memory (Mooncake's
Conductor-restart corruption, ToS'25 §5.3). The request lifecycle is *soft*
state: prefill is recomputable, so lifecycle divergence costs at worst wasted
work or a client error. Protocol effort goes where the state is hard.
Lifecycle correctness is achieved by fencing and laziness, never by
distributed transactions on the hot path.

**P2 — Single writer per state domain.** Every piece of state has exactly one
authoritative writer. Symmetric peer state machines are the anti-pattern;
SGLang's residual failure modes (hangs, abort storms — #9266, #10111) all live
in windows where two writers disagree.

**P3 — Leases are ground truth for *obligations*; messages are accelerators.**
Every resource pinned across the CTX/GEN boundary carries a renewable lease.
Abort/completion notifications exist to reclaim *fast*; lease expiry reclaims
*always*. No component's liveness — including the coordinator's — is a
correctness dependency for reclamation (vLLM lease design). **Scope limit:**
lease expiry ends the cross-side obligation, never physical safety — reuse of
possibly-exposed memory additionally requires quiescence evidence or
remote-access revocation (§8a; PR #16347 contract: "wall-clock expiry never
makes an in-doubt allocation safe to reuse").

**P4 — No implicit waiting.** Every wait state has an owner, an explicit
signal, and a bound. Backpressure discovered via a peer's timeout is a bug
class, not a mechanism (NVBUG 6480621; SGLang #10111 — prefill never receives
KV indices while decode silently queues — is the same archetype).

**P5 — Admission is prediction, not protocol.** Even perfectly synchronized
current-load admission oscillates, because the accept→observe feedback loop is
delayed by exactly the prefill duration (Mooncake §7.3, anti-phase pool
oscillation in production). The admission signal must be *predicted GEN load
at prefill-completion time* (Mooncake §7.4: system-level simulation, not
per-request output-length prediction — they tried that and rejected it).

---

## 2. Invariants

- **I1 — Single terminal outcome.** Each attempt reaches exactly one terminal
  state; the logical request's outcome is monotone (never SUCCEEDED→retried).
- **I2 — Safe ownership.** CTX never frees source KV while GEN or the
  transport may access it; GEN never frees destination memory during an
  in-flight write (the transceiver already enforces the latter).
- **I3 — Bounded cleanup under any single loss.** If CTX, GEN, *or the
  coordinator* disappears, all survivors reclaim resources within a bounded
  window without operator action.
- **I4 — Queueing is not transport failure.** A healthy request waiting for
  GEN admission never expires under an active-transfer timer.
- **I5 — No resurrection.** Late messages from a superseded attempt or a
  previous coordinator epoch are inert (fenced), and cannot re-create
  receivers, leases, or scheduler entries.

---

## 3. Four state domains, one writer each

| Domain | Writer | Content | Durability |
|---|---|---|---|
| **Logical request record** | Coordinator | `ACTIVE → SUCCEEDED \| FAILED \| CANCELLED \| DEADLINE_EXCEEDED`; client deadline; attempt counter | **Soft / rebuildable.** Reconstructed at restart by querying workers; epoch-fenced (§6) |
| **Handoff attempt** | Coordinator creates & terminates; CTX/GEN advance interior states | `CREATED → GEN_ADMITTED → RECEIVER_READY → TRANSFERRING → COMMITTED \| ABORTED(reason)` | Ephemeral; tombstoned per §6 |
| **Scheduler state** | Each engine, locally | Queues, batching, chunking, preemption — fully local, free to diverge | Local only |
| **Connector resource state** | Transfer layer on each side | `pinned → leased → transferring → quiescing → reclaimable` | Lease-governed (P3); evidence-gated per §8a |

The **handoff attempt** is the only cross-side protocol object. Scheduler
enums are never exchanged. The coordinator's domain is explicitly soft (P1/P3,
Mooncake lesson), and attempt interior states are advanced by whichever side
owns the corresponding obligation, avoiding a chatty coordinator round-trip
per transition.

---

## 4. Handoff attempt protocol

Identity: `(logical_request_id, attempt_id, coordinator_epoch)`. All messages
idempotent, attempt-scoped, safe to duplicate or drop (leases backstop drops).

```
GEN_ACCEPT   (attempt, reservation?)          GEN → coord, CTX
GEN_QUEUED   (attempt, est_wait)              GEN → coord          # explicit, replaces silent FIFO
GEN_REJECT   (attempt, reason, retry_after?)  GEN → coord
KV_READY     (attempt, lease_id, ttl, block_desc)  CTX → coord, GEN
LEASE_RENEW  (attempt, lease_id)              GEN → CTX            # from scheduler insertion, ~ttl/6 cadence
RECEIVER_READY(attempt, credit)               GEN → CTX            # starts active-transfer clock
TRANSFER_COMMITTED(attempt)                   transport → both
ABORT        (attempt, reason)                any → coord → fan-out to both legs
QUIESCED     (attempt)                        each side → coord    # local cancellation complete; resources at safe point
```

**Ordering constraint (the only one):** `TRANSFERRING` requires `KV_READY`
(per chunk, for chunked prefill) ∧ `RECEIVER_READY`. Everything else is a free
partial order.

**Flow-direction neutrality.** Context-first and gen-first are *schedules over
the same event set*, not different protocols:

- *Context-first* (current TRT-LLM default): CTX prefills ahead → `KV_READY`
  may precede `GEN_ACCEPT`; CTX KV pinned under lease while GEN admits.
- *Gen-first* (Dynamo-style, TRT-LLM gen-first path): `GEN_ACCEPT` +
  destination reservation precede prefill scheduling; `RECEIVER_READY` may
  precede `KV_READY`. This ordering makes orphaned receivers,
  late-admission-after-CTX-timeout, and reject-before-receiver structurally
  unrepresentable — roughly half of the context-first regression matrix
  vanishes. Recommendation: implement the protocol once, then cost gen-first
  as the default policy for long-prefill traffic rather than inheriting
  context-first as a constraint.

**Cancellation.** Any leg or the client can emit `ABORT`. The coordinator fans
out and awaits `QUIESCED` from both sides before recording the terminal
outcome — but fan-out is an accelerator (P3): if the coordinator is down, each
side's lease expiry and local timeout produce the same end state. Direct peer
abort is permitted as a fast path once peers know each other; the vLLM lesson
stands: *a lease must cover every window in which no peer-to-peer relationship
exists yet* (the proxy-gap disconnect case).

---

## 5. Timer model (four independent clocks)

| Timer | Owner | Starts | Expiry action |
|---|---|---|---|
| `T_e2e` | Coordinator | Client arrival | `ABORT(deadline_exceeded)` both legs |
| `T_handoff` | Coordinator | Attempt `CREATED` | Bound on `CREATED → RECEIVER_READY`. Expiry → abort attempt → **reroute** (§7) or fail |
| `T_lease` | CTX transfer layer | `KV_READY` | Reclaim pinned KV (via §8a). Renewed by GEN from **scheduler insertion** (not transfer admission) — vLLM precedent: 30 s TTL, ~5 s renew, ~20 s reclaim on GEN death |
| `T_transfer` | Both transfer layers | `RECEIVER_READY` / first progress | Transport-stall detection only. **Never covers queueing** (I4) |

This directly fixes NVBUG 6480621: today CTX's single timer spans
`GEN admission delay + handshake + transfer` while GEN's spans
`handshake + transfer`. Under this model the admission delay lives under
`T_handoff` (coordinator-owned, reroute-capable) and CTX's `T_transfer` cannot
start before `RECEIVER_READY`. The 600 s workaround becomes unnecessary rather
than tolerated.

---

## 6. Fencing and coordinator restart

- **Attempt fencing:** every retry increments `attempt_id`. Workers and
  transports reject messages, receiver registrations, and lease renewals
  bearing a stale attempt. Tombstones for terminal attempts are retained for
  `max(T_lease, transport_max_delay)`.
- **Tombstone semantics distinguish** `ATTEMPT_ABORTED (retryable)` from
  `REQUEST_TERMINAL`. Required for GEN-side preemption *after*
  `TRANSFER_COMMITTED` (vLLM RFC #24256 lockup precedent; SGLang #6857 is the
  production instance — post-transfer decode-OOM retraction re-entered the
  prealloc queue in an unpoppable state): preemption bounces the request to
  the coordinator as a **new attempt** — never re-prefilled locally on GEN —
  while a client cancel is terminal.
- **Coordinator restart (Mooncake lesson):** the coordinator holds no state
  that workers cannot re-supply. On restart it bumps `coordinator_epoch`,
  rebuilds its shadow table by querying workers for live attempts, and lazily
  aborts attempts from prior epochs it cannot account for. Old-epoch messages
  are inert (I5). Worker-side leases guarantee I3 throughout the outage. This
  is Mooncake's shadow-copy + node-ownership remedy, generalized.
- **Connector-level fencing:** where the transport supports it, embed
  `(attempt_id, epoch)` in transfer notifications/rkeys so even
  transport-layer stragglers are rejected below the protocol.

---

## 7. Admission: three layers

1. **Router predictive paired admission (P5).** Accept only if predicted CTX
   *and* GEN load — GEN load evaluated at the request's predicted
   prefill-completion time — are under SLO-derived thresholds; otherwise
   reject early (HTTP 429/503). System-level load simulation à la Mooncake
   §7.4. Separate workstream with its own evaluation; the protocol below is
   correct with or without it, just wasteful without.
2. **GEN hard admission.** On attempt arrival GEN responds explicitly:
   `GEN_ACCEPT` (reservation of request slot + destination budget + receiver
   credit pipeline), `GEN_QUEUED` (with estimate — coordinator may hold,
   subject to `T_handoff`, or preemptively reroute), or `GEN_REJECT`. **Silent
   queueing is prohibited** (P4). This is P/D-Serve's reject-don't-queue
   insight grafted onto SGLang's prealloc-before-transfer, minus its
   implicit-backpressure flaw. The grant may reserve capacity *accounting*
   rather than physical pages, materializing blocks near `KV_READY`.
3. **Continuous admission.** `LEASE_RENEW` from scheduler insertion keeps CTX
   pinning honest; GEN may *revoke* an admission (preemption, memory
   pressure) → attempt aborted → coordinator reroutes. A live-but-overloaded
   GEN can therefore not pin CTX memory indefinitely: `T_handoff` and reroute
   cap it, closing the known gap in a pure-lease design.

**Reroute policy:** on `GEN_REJECT`/revoke/`T_handoff` expiry with CTX KV
still leased, the coordinator selects another GEN and issues attempt N+1
referencing the same `KV_READY` lease (KV need not be recomputed). Bounded
retries within `T_e2e`; then `DEADLINE_EXCEEDED`.

---

## 8. Connector capability contract

Backends differ materially (SGLang: Mooncake sends `ABORT_ACK` but doesn't use
it as deferred-release proof; NIXL has an open quiescence TODO; llm-d
documents a cancellation window where prefill KV persists until restart). **In
TRT-LLM this contract is load-bearing, not optional:** the C++ transceiver
(`BindKvCacheTransceiver`, current default) and the Python transceiver
(`KvCacheTransceiverV2`, per-model opt-in via `transceiver_runtime="auto"`)
will coexist indefinitely — Python-only features (gen-first disagg params,
Mamba hybrid cache manager) force migration while C++-only features
(UCX/Mooncake backends, the qualified in-flight-cancel configuration) keep C++
alive. The lifecycle protocol (domains 1–3) therefore sits strictly above the
`KvCacheTransceiver` interface, and each runtime declares its row:

| Capability | C++ transceiver | Python transceiver (post-#16396) |
|---|---|---|
| Completion semantics | Executor polling of sender futures/responders | Registry-owned evidence; `LifecycleAction` callbacks |
| Quiescence ACK | No (local destruction treated as sufficient) | Yes: `PhysicalState.DRAINED` gates all frees |
| In-flight cancel | Only qualified config (NIXL+UCX plugin, finite timeout, `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL`), via buffer **poisoning** | Logical cancel latched immediately; memory retained until drain |
| Late-message fencing | None | Publication gating, conflict latching, sender tombstones (full ABA/epoch fencing deferred) |
| Timeout return semantics | Returns on `kv_transfer_timeout_ms` | Fail-closed: cannot return while a published target is in doubt (until allocator-lease follow-up) |
| Remote-access revocation | No | No (candidate follow-up; see §8a) |

Generic capabilities and compensations when absent:

| Capability | Compensation when absent |
|---|---|
| Abort before peer creation | Coordinator tombstone rejects late receiver registration |
| Abort during active transfer | Quarantine buffers until `T_transfer` expiry, then reclaim |
| Remote abort notification | Lease expiry is sole reclamation path (sizes `T_lease`) |
| Quiescence ACK | Time-based quarantine before destination buffer reuse |
| Lease/TTL reclamation | Must be emulated at transfer layer — hard requirement, no waiver |
| Late-message fencing | Protocol-level fencing only; lengthen tombstone retention |

The orchestrator selects the compensation path per connector instead of
assuming NIXL/Mooncake/store-backed behave identically. Store-backed
connectors (LMCache/MooncakeStore-style) shift ownership into the store:
domain 4 becomes a store-lifetime problem, domains 1–3 unchanged — evidence
the split is at the right joint.

## 8a. Liveness/safety layering: leases × evidence-gated ownership (PR #16396/#16347)

TRT-LLM's Python-transceiver ownership hardening (PR
[#16396](https://github.com/NVIDIA/TensorRT-LLM/pull/16396), contract
[#16347](https://github.com/NVIDIA/TensorRT-LLM/pull/16347)) is **not a lease
design and must not become one** — it is the *safety* layer this proposal
assumes exists. Its mechanisms (registry as root owner outliving sessions;
`UNEXPOSED → POSSIBLY_EXPOSED → PUBLISHED` exposure tracking with atomically
gated publication; exact writer-cohort ledgers; fail-closed conflict latching;
**`LogicalState` vs `PhysicalState` separation with reuse gated on `DRAINED`**)
answer a question no TTL can: whether a possibly-exposed one-sided RDMA
operation might still land. Its explicit anti-expiry stance is correct at that
layer.

The two designs compose as layers with one bridge:

- **Safety (domain 4, local):** evidence-gated release per #16347. Memory
  reuse requires terminal evidence (commit, positive non-delivery proof, or
  backend quiescence fence) — never elapsed time. The
  `LogicalState`/`PhysicalState` split *is* this document's domain-2/domain-4
  separation implemented locally; #16347's planned split of `cancel_request()`
  into logical outcome + physical disposition (`RETIRED/DRAINING/IN_DOUBT`) is
  the `QUIESCED` boundary of §4.
- **Liveness (cross-node obligation):** the renewable lease of §5 answers the
  question #16396 deliberately punts on (in-doubt allocations retained
  indefinitely; teardown veto; the accepted fail-closed synchronous-timeout
  regression): *when does CTX's obligation to hold source KV for a
  possibly-dead GEN end?* Lease expiry terminates the obligation and triggers
  local cancellation — it does not by itself permit reuse.
- **The bridge — revocation turns expiry into evidence:** on lease expiry,
  revoke remote access at the transport (NIXL memory deregistration / rkey
  invalidation) so any straggler one-sided op *fails* rather than lands.
  Completed revocation is exactly the "positive non-delivery proof" #16347
  requires, letting expiry-driven reclamation satisfy the evidence rule. Where
  the backend cannot revoke (see capability rows — currently neither
  transceiver can), the compensation is #16396's behavior: quarantine in-doubt
  allocations until backend quiescence, and surface the capacity pressure as a
  first-class metric. The deferred allocator-issued `KVTransferLease` (#16347
  O7, prior art [#15803](https://github.com/NVIDIA/TensorRT-LLM/pull/15803))
  is the natural home for the liveness half.

Net: adopt #16396 as the domain-4 foundation for the Python transceiver; add
the obligation-lease and revocation follow-up on top; require the C++
transceiver to meet the same *contract* via its compensation row rather than
the same mechanism.

---

## 9. Regression matrix

| Case | Mechanism |
|---|---|
| GEN admission delayed >60 s, healthy | `GEN_QUEUED` + `T_handoff`; CTX `T_transfer` not started (I4) |
| GEN rejects before creating receiver | `GEN_REJECT` → reroute; lease keeps CTX KV valid |
| CTX times out before late GEN admission | Attempt tombstone rejects late receiver registration (I5) |
| Client disconnect while GEN queued | Coordinator `ABORT` fan-out; lease backstops if fan-out lost |
| Client disconnect in coordinator gap (GEN never knew) | Lease expiry (vLLM proxy-gap case; the window no notification can cover) |
| Cancellation during active DMA | `ABORT` → transfer layer quiesces → `QUIESCED` gates frees (I2; #16396 exposure/drain machinery) |
| GEN crash after CTX pins KV | Renewals stop → `T_lease` reclaim (~20 s at vLLM-like parameters) |
| CTX crash mid-transfer | GEN transfer error → `ABORT(transport)` → coordinator may re-attempt full prefill (soft state, P1) |
| Retry races delayed messages from old attempt | Attempt fencing (I5) |
| GEN preemption after `TRANSFER_COMMITTED` | Bounce as new attempt via coordinator; never local re-prefill on GEN (vLLM RFC #24256; SGLang #6857) |
| Coordinator crash/restart | Epoch bump + rebuild from workers; leases carry I3 through outage (Mooncake lesson) |
| Admission oscillation under load | Predictive router admission (P5); protocol layers unaffected |

---

## 10. Incremental path for TRT-LLM

1. **Instrument.** Per-attempt timestamped event log: CTX prefill complete,
   GEN scheduler arrival, GEN admission, receiver ready, first transfer
   progress, committed, abort propagation, resource release. Confirms
   NVBUG 6480621's causal chain and baselines every later step. Counters:
   lease expiries, orphaned attempts, reroutes.
2. **Split the timers** (§5). Smallest change that kills the 600 s workaround:
   CTX active-transfer clock starts at receiver-ready.
3. **Attempt IDs + fencing** (§6). Prerequisite for any retry/reroute
   semantics.
4. **Lease renewal** replacing fixed pin timeouts (vLLM-proven design; start
   renewals at GEN scheduler insertion). Implement as the *obligation* layer
   over #16396's evidence-gated ownership (§8a): expiry triggers cancellation
   + (future) remote-access revocation; physical reuse remains drain-gated.
   Land on the Python transceiver first; C++ transceiver satisfies the
   contract via its §8 compensation row (poisoning + polling), with parity
   regression tests run against **both** runtimes since both remain in
   production.
5. **Explicit GEN admission** (`ACCEPT/QUEUED/REJECT`) + coordinator abort
   fan-out with `QUIESCED` tracking. The current coordinator reservation
   expiry is router *accounting* only — it does not terminate worker work;
   this step makes termination authoritative-but-optional per P3.
6. **Reroute policy + predictive router admission** (§7). Independent; ship
   last.
7. **Evaluate gen-first as default policy** on the unified protocol (§4),
   using step-1 telemetry to compare wasted-prefill and TTFT overlap against
   context-first.

---

## 11. Non-goals (the honest boundary)

Exactly-once semantics with zero recomputation, and mid-request migration
without token replay, require KV state to outlive its worker (DéjàVu-style
replication). This design deliberately keeps the lifecycle soft (P1): failure
recovery is recompute-based (Dynamo's migration precedent — replay prompt +
emitted tokens). Every production system surveyed has declined the
richer-shared-state trade; if SLOs on very long contexts ever make prefill
recompute unaffordable, that is the assumption to revisit — as a KV durability
feature in domain 4, not as request-lifecycle coordination. Note also that
transparent post-output retry additionally requires committed output cursor,
duplicate-token suppression, and sampling/RNG/speculative state; without
those, migration is only transparent before the first visible token.

---

## 12. What the surveyed systems do (evidence base)

| System | Coordination model | Timeout/abort crossing sides |
|---|---|---|
| **vLLM** | Stateless P and D; proxy runs two sequential HTTP legs carrying `kv_transfer_params`; deliberate design choice ([PR #15960](https://github.com/vllm-project/vllm/pull/15960)) | Best-effort NIXL abort notify D→P; **lease renewal** (30 s TTL, ~5 s heartbeat from D scheduler insertion) replaced the 480 s fixed timeout ([lease design](https://docs.vllm.ai/en/stable/design/nixl_kv_cache_lease/)) |
| **SGLang** | Handshake-then-independent: bootstrap-room handshake (decode preallocates, pushes KV indices), then two independent local FSMs converged by one-way ZMQ status pushes | Explicit ABORT/ABORT_ACK D→P; P→D implicit (status push, 5 s heartbeats); 300 s local timeouts. Silent prealloc queueing = implicit backpressure (#10111) |
| **Dynamo** | Single ownership: decode worker owns request state + KV memory; prefill is a stateless sub-request pulled from a NATS queue, RDMA-writing into decode-owned blocks | Hierarchical cancellation through linked contexts; recompute-based migration ([cancellation](https://docs.nvidia.com/dynamo/latest/architecture/request_cancellation.html)) |
| **DistServe** | Central FCFS dispatch; decode-pull with prefill GPU memory as queue buffer; "without complex coordination" by design | None — fault tolerance explicitly out of scope ([OSDI'24](https://arxiv.org/abs/2401.09670)) |
| **Mooncake** | Conductor-anchored: P/D pair bound up front; the only 2PC anywhere (`TxAllocate`/`TxCommit`) protects **block metadata**, not lifecycle | Predictive early rejection (HTTP 429) before prefill; abort + cleanup with P↔D failure-cause sync; Conductor made stateless after a restart-corruption incident ([ToS'25](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/ToS2025-Qin.pdf)) |
| **P/D-Serve** | No global state: busy prefill rejects, gateway retries within a deadline, then terminates at admission | Rejection/retry/timeout replaces scheduler state ([arXiv](https://arxiv.org/abs/2408.08147)) |
| **Splitwise** | Cluster scheduler pairs P+D at arrival for layer-wise transfer overlap; local schedulers otherwise | Handoff commit synchronized only ([ISCA'24](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)) |

Three archetypes: decode-anchored ownership (Dynamo, llm-d, DistServe),
scheduler-anchored (Mooncake, Splitwise, MemServe, TetriInfer), and
rejection/retry-based (P/D-Serve). No system synchronizes scheduler states;
this design composes the strongest element of each: Dynamo's ownership chain,
vLLM's leases, SGLang/P/D-Serve's explicit admission, Mooncake's predictive
rejection and soft coordinator.

---

## 13. Implementation effort assessment (TRT-LLM `main` @ `cf44a1c`, 2026-07-24)

Grounded against current code: `serve/openai_disagg_service.py`,
`serve/disagg_coordinator.py`, `serve/openai_client.py`, `serve/router.py`,
`serve/openai_server.py`, `_torch/pyexecutor/py_executor.py`,
`serve/perf_metrics.py`. Estimates assume a senior engineer familiar with the
codebase and **exclude landing #16396 itself** (the domain-4 safety layer,
already in flight as its own effort).

### Current-state summary (gap inventory)

- **No per-request state anywhere in the control plane**: the disagg service
  holds state in coroutine locals; the coordinator holds only expiring
  load-accounting reservation tasks
  (`TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT`, 180 s) whose expiry
  releases router accounting but never touches the worker. Coordinator
  restart = total amnesia, no reconciliation; it cannot enumerate or cancel
  in-flight requests.
- **No abort RPC to workers.** `PyExecutor.cancel_request` exists but is
  reachable only via each worker's own client-disconnect watcher. The disagg
  orchestrator has no disconnect watcher and no abort fan-out; after client
  disconnect, CTX-leg cleanup relies entirely on the 60 s
  `kv_transfer_timeout_ms`. Gen-first non-streaming `asyncio.gather` does not
  cancel the surviving leg when one fails.
- **Retry identity is the anti-pattern**: `OpenAIHttpClient._post_with_retry`
  mints a fresh snowflake `disagg_request_id` per retry ("to avoid ID
  collision on workers"), orphaning the previous attempt with no cancellation
  and no correlation; worker 4xx/5xx (`ClientResponseError` ⊂ `ClientError`)
  are blindly retried like network errors.
- **GEN admission deferral is silent**: `DisaggTransferAdmissionController`
  (FCFS over a transfer-block budget) defers requests in
  `DISAGG_GENERATION_INIT` with debug-log visibility plus pull-based iteration
  stats only; no coordinator-visible QUEUED state. This is the direct
  mechanism behind NVBUG 6480621's asymmetric aging.
- **Instrumentation is post-hoc, not lifecycle**: `RequestPerfMetrics` timing
  (incl. `kv_cache_transfer_start/end`) and `DisaggPerfMetricsCollector`
  (ctx/gen join with clock-offset sync) exist, but there are no events for
  retries, admission defer/admit, receiver-ready, abort, quiescence, or
  terminal state; mid-flight deaths linger unjoined.

### Workstream estimates

| WS | Design element | Exists today | Delta | Size | Risk |
|---|---|---|---|---|---|
| 1 | Attempt event log (§10.1) | Perf metrics + collector + orchestrator hooks | Lifecycle events (defer/admit, receiver-ready, first progress, abort, quiesce, terminal, attempt#); push/ring not join-on-completion | **2–4 wk** | Low |
| 2 | Timer split (§5) | Single 60 s `kv_transfer_timeout_ms` starting at CTX send / GEN receive-start | CTX clock from first-progress/receiver-ready (signal needed in **both** transceivers); orchestrator-owned handoff-queue timeout | **3–6 wk** | Med (dual runtime, TP-consensus timeout paths) |
| 3 | Attempt identity + fencing (§6) | Per-retry ID minting (anti-pattern); no tombstones outside #16396 | `(logical_id, attempt_id)` through serve + worker intake + transceiver headers; tombstones; retry = abort-both-legs-then-new-attempt; stop retrying 4xx/5xx | **4–8 wk** | Med-high (protocol versioning, rolling upgrade) |
| 4 | Abort RPC + fan-out (§4) | Internal `cancel_request` only | Worker `/abort` by (id, attempt); orchestrator disconnect watcher + fan-out; disposition reporting per #16347 `RETIRED/DRAINING/IN_DOUBT` | **3–5 wk** | Med |
| 5 | Obligation leases (§5, §8a) | Nothing in serve layer; #16396 (safety, draft), #15803 (C++ prior art, draft) | CTX source-lease records; GEN renewal from scheduler insertion; expiry → cancel; C++ compensation row; both runtimes | **6–10 wk** | High (hot-path messaging, TP/PP fan-out, dual runtime) |
| 6 | Explicit GEN admission (§7.2) | Silent FCFS deferral; conditional-disagg GEN reservation as primitive grant | `ACCEPT/QUEUED/REJECT` to orchestrator; reroute under `T_handoff`; receiver credit | **4–8 wk** | Med |
| 7 | Coordinator soft-state/epoch (§6) | In-memory, restart amnesia | Epochs, worker live-attempt query API, rebuild-on-restart, tombstone retention | **3–5 wk** | Med (after WS3/4) |
| 8 | Predictive admission (§7.1) | RR / load-balancing / KV-aware `load_cap` routers; no backpressure/429 | Load model + system-level simulation + hysteresis + eval | **6–12+ wk** | Research-flavored; decoupled |
| 9 | Gen-first default eval (§10.7) | Gen-first path exists (Python transceiver only) | Policy + benchmarking off WS1 telemetry | **1–2 wk** | Low |

**Core protocol total (WS1–7): ~25–45 engineer-weeks** — 2–3 quarters for one
engineer, ~1 quarter for a 2–3 person effort. Sequencing: WS1→2 (kills the
600 s workaround early) → WS3→4 (correctness) → WS5/6 in parallel → WS7. WS8
runs as a separate project; WS9 rides on WS1.

**Cost multipliers**: (a) any new per-request state in `py_executor` must be
rank-consistent — the existing timeout path already does allreduce/allgather
consensus, and lease/admission state inherits that pattern; (b)
dual-transceiver parity testing roughly doubles validation for WS2/3/5.

**Pull-forward fixes (days each, independent of the larger effort)**: stop
retrying worker 4xx/5xx as network errors in `_post_with_retry`; cancel the
surviving leg in gen-first non-streaming `gather`; add a disconnect watcher to
the disagg orchestrator.

---

## References

- [vLLM disaggregated prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM NIXL KV cache lease design](https://docs.vllm.ai/en/stable/design/nixl_kv_cache_lease/)
- [vLLM NIXL push connector design](https://docs.vllm.ai/en/latest/design/nixl_kv_push_connector/)
- [vLLM KV Connector API V1 (PR #15960)](https://github.com/vllm-project/vllm/pull/15960)
- [vLLM disagg abort handling (PR #19223)](https://github.com/vllm-project/vllm/pull/19223)
- [vLLM cache-hit-threshold RFC (#24256)](https://github.com/vllm-project/vllm/issues/24256)
- [SGLang PD disaggregation docs](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)
- [SGLang large-scale EP blog](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- [SGLang decode preallocation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/decode.py)
- SGLang issues [#6857](https://github.com/sgl-project/sglang/issues/6857), [#9266](https://github.com/sgl-project/sglang/issues/9266), [#10111](https://github.com/sgl-project/sglang/issues/10111)
- [Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/latest/architecture/disagg_serving.html)
- [Dynamo request cancellation](https://docs.nvidia.com/dynamo/latest/architecture/request_cancellation.html)
- [Dynamo request migration](https://docs.nvidia.com/dynamo/latest/architecture/request_migration.html)
- [DistServe (OSDI'24)](https://arxiv.org/abs/2401.09670)
- [Mooncake (FAST'25 / ToS'25)](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/ToS2025-Qin.pdf)
- [Splitwise (ISCA'24)](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)
- [TetriInfer](https://arxiv.org/abs/2401.11181)
- [MemServe](https://arxiv.org/abs/2406.17565)
- [P/D-Serve](https://arxiv.org/abs/2408.08147)
- [DéjàVu (ICML'24)](https://arxiv.org/abs/2403.01876)
- [llm-d disaggregation architecture](https://llm-d.ai/docs/dev/architecture/advanced/disaggregation)
- [llm-d vLLM operations](https://llm-d.ai/docs/architecture/advanced/disaggregation/operations-vllm)
- [TRT-LLM GEN transfer admission controller](https://github.com/NVIDIA/TensorRT-LLM/blob/4b7d7199752f41960eedbf2846755e174940f164/tensorrt_llm/_torch/pyexecutor/py_executor.py#L263-L349)
- [TRT-LLM transceiver mid-write protection](https://github.com/NVIDIA/TensorRT-LLM/blob/4b7d7199752f41960eedbf2846755e174940f164/tensorrt_llm/_torch/disaggregation/transceiver.py#L800-L833)
- [TRT-LLM coordinator reservation expiry](https://github.com/NVIDIA/TensorRT-LLM/blob/4b7d7199752f41960eedbf2846755e174940f164/tensorrt_llm/serve/disagg_coordinator.py#L243-L261)
- [TRT-LLM PR #16396 — harden Python native KV transfer ownership](https://github.com/NVIDIA/TensorRT-LLM/pull/16396)
- [TRT-LLM PR #16347 — Python native KV transfer ownership contract](https://github.com/NVIDIA/TensorRT-LLM/pull/16347)
- [TRT-LLM PR #15803 — C++ explicit KV transfer leases (prior art)](https://github.com/NVIDIA/TensorRT-LLM/pull/15803)
- [TRT-LLM kv_cache_transceiver runtime selection](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py)
