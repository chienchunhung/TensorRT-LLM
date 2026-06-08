# Phase 1 - In-Flight Request Cancellation Architecture

| | |
|---|---|
| **Phase** | 1, load-bearing architecture before default-ON cancellation |
| **JIRA** | [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Owner** | Chien-Chun Hung |
| **Status as of 2026-06-08** | Design draft. Assumes the `dataTransceiver` `shared_ptr<LlmRequest>` lifetime fix in <https://github.com/NVIDIA/TensorRT-LLM/pull/14979> will merge soon. |
| **Main dependency** | The in-flight cancel prototype at <https://github.com/NVIDIA/TensorRT-LLM/pull/13713>, which remains the only known implementation that passes the Dynamo field reproducer. |
| **Related prior art** | Timeout-flag rank consensus in <https://github.com/NVIDIA/TensorRT-LLM/pull/14746>; V1 packed consensus proposal in [`phase1-consensus-collective-design.md`](phase1-consensus-collective-design.md). |

## Motivation

Disaggregated serving makes high-throughput deployments possible by
separating prefill and decode workers, but it also creates a long-lived
KV-transfer critical section. Under high workload, a single request can
hold several scarce resources at once:

- decode-side KV blocks reserved for the incoming prompt KV;
- context-side computed KV that cannot be released until send-side
  ownership is resolved;
- transfer-buffer slots advertised to the transport;
- request state inside `CacheTransceiver` futures and PyExecutor's
  active request lists.

If a client disconnects, a deadline expires, or a peer pauses while the
transfer is in flight, continuing to wait wastes capacity and can make
the deployment non-responsive. The failure mode is multiplicative under
burst load: each stuck transfer pins KV blocks and buffer slots, which
reduces admission headroom, which increases transfer latency for the
remaining requests, which makes more deadlines fire. This is the
customer-visible NVBug 6104831 wedge.

The Dynamo exp4 reproducer stresses exactly this shape: NIXL transport,
small decode-side KV headroom (`free_gpu_memory_fraction=0.2`), burst
concurrency 16, and input length around 8k. The observed fingerprint is
`exceeded total timeout` followed by `cancelled before send`, then
requests that never complete and KV blocks that never return. The report
infers a KV-block / credit-pressure deadlock: decode has no free blocks
to receive into while prefill is holding computed KV it cannot push.
That trigger still needs direct block-accounting logs, but it explains
why timeout cancellation without safe cleanup cascades under high load.

In-flight cancellation is needed to break that spiral. The important
constraint is that cancellation is not allowed to mean "free everything
now". A request can be logically cancelled immediately, but its KV
blocks and transfer buffers can only be reused after the transport is
known to be terminal, or after the system has failed closed and stopped
serving from the affected memory.

## Current Status

The current state is best understood through the F1 / F2 / F3
decomposition in
[`19-exp4-f1-f2-f3-decomposition.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/19-exp4-f1-f2-f3-decomposition.md):

| Layer | Status after PR #14979 | What it means |
|---|---|---|
| **F1: request lifetime UAF / `Broken promise`** | Closed by PR #14768 plus PR #14979. `cacheTransceiver` and `dataTransceiver` workers hold `shared_ptr<LlmRequest>` while async work can still dereference the request. | Necessary foundation. It prevents a real crash class but does not recover the NIXL wedge by itself. |
| **F2: engine-loop freeze on unbounded `future.get()`** | Not closed by PR #14979. PR #13713 capped polling with bounded `wait_for` slices. | Required so the PyExecutor loop can keep scheduling, checking deadlines, serving health, and processing cancellation while a transfer is stuck. |
| **F3: eager-free poisons transport / permanent wedge** | Not closed by PR #14979. PR #13713 deferred Python cleanup until transfer state was terminal, but paired that with aggressive pool-wide poison for memory safety. | The load-bearing recovery problem. Detecting timeout and calling cancel is insufficient; cleanup must be quiescence-gated and rank-consistent. |

The raw exp4 report adds three important qualifications:

- **F2 is an engine-liveness fix, not transfer recovery.** On NIXL,
  UCX progress runs on its own background thread. Replacing `future.get()`
  with bounded `wait_for` keeps the PyExecutor loop alive, but it does
  not make a stuck transfer complete. Trial 1 removed `Hang detected`
  and still wedged `5/5`.
- **Detection and active drain are not the missing piece.** Trial 2's
  deadline drain engaged (`Cannot cancel` dropped from 243 to 1 and
  `exceeded total timeout; cancelling` fired), but it still wedged
  because the drain freed eagerly, exactly like rc17.
- **There is no small `#14979 + cacheTransceiver` recovery subset.**
  PR #14979 closes F1; bounded polling closes F2; the deployment only
  recovers when F3 is also fixed through the Python termination /
  transfer-manager redesign that gates freeing on transport quiescence.

PR #14746 is relevant but not sufficient. It closes **L1** consensus for
the timeout flag: all ranks agree on which request IDs timed out. It
does not close **L2** consensus for the state transition: all ranks must
also apply the same CANCELLED / FAILED / COMPLETED outcome and the same
deferred-cleanup decision.

The in-flight cancel surface from PR #13713 remains default-OFF because
its safety posture is correct but operationally aggressive:

- mid-flight NIXL cancel may leave remote memory quiescence unknown;
- unknown quiescence poisons the whole transfer pool;
- with the default pool size of one slot, one poisoned slot means
  PyExecutor shutdown and pod restart;
- V1 + C++ transceiver state transitions can still diverge across ranks
  unless L2 consensus is added.

## What Remains From PR #13713 After PR #14979

Assuming PR #14979 lands, the remaining merge work from PR #13713 is not
"just call cancel". The remaining pieces are:

| Piece | Why it remains required | Merge shape |
|---|---|---|
| **Bounded polling** | Prevents `checkContextTransferStatus` / `checkGenTransferStatus` from parking the engine loop on an unready future. Without this, timeout and cancellation code may never run. | Make status checks non-blocking or bounded, with a small cap such as 50 ms per future. Keep shutdown-drain paths as the only intentionally blocking paths. |
| **Deadline enforcement** | A transfer whose future never becomes ready must still age out. The check must scan every pending transfer, not only transfers that were selected by readiness consensus. | Represent deadline expiry as a local intent first. Use PR #14746-style L1 union so every rank sees the same timed-out IDs before action. |
| **L2 cancellation outcome consensus** | A consistent timeout flag is not enough if ranks independently decide cancel success, request state, or cleanup timing. | Add a V1 consensus outcome equivalent to V2's `_consensus_outcome`. The preferred encoding is the packed `(rid, state)` allgather in [`phase1-consensus-collective-design.md`](phase1-consensus-collective-design.md). |
| **Transport-safe cancel** | NIXL `releaseXferReq` / cancel flags can request unwind, but they are not a synchronous proof that the peer has stopped touching advertised memory. | Preserve the PR #13713 fail-closed checks, but route unknown-quiescence slots into quarantine instead of immediately treating the whole process as unrecoverable. |
| **Quiescence-gated cleanup** | The Dynamo exp4 Trial 2 failure shows that active drain plus eager free still wedges. The raw report also notes rc17 already reaches `_check_kv_transfer_timeout` and terminates timed-out transfers; safe freeing, not detection, is the gap. | Keep Python KV resources, request transfer metadata, and buffer ownership pinned until C++ reports terminal status or the quarantine deadline expires. This requires the `py_executor.py` / `AsyncTransferManager` redesign, not a cacheTransceiver-only patch. |
| **Bounded quarantine / deferred un-poison** | Pool-wide poison at first cancel is safe but too disruptive. | Add a `PendingQuiescenceTracker` that polls NIXL status for quarantined slots and returns them to the pool when terminal. Only fail closed when the quarantine deadline expires or the pool is exhausted. |
| **Block-accounting diagnostics** | The report's F3 origin is inferred from config pressure and log signatures, not directly traced. | Add sender/receiver block-accounting logs around timeout, cancel, ready/error status, and resource release so the trigger can be proven and future regressions can be localized. |
| **Stress coverage** | The first prototype exposed rank-divergence bugs only under multi-rank / high-pressure tests. | Gate the feature with Phase 0 marathons plus TP/PP/EP consensus-focused configs before flipping default-ON. |

## Timed-Out Request Workflow

The target workflow separates four decisions that were previously too
easy to collapse: detection, consensus, cancellation, and cleanup.

```text
1. Register transfer
   - Request enters context-side send or generation-side receive.
   - C++ records request id, start time, deadline, future, transfer handle,
     and transfer-buffer slot.
   - Python marks the request as in disagg transfer and pins KV resources
     through AsyncTransferManager.

2. Poll status without wedging the engine
   - Every executor iteration calls check*TransferStatus().
   - The status check performs wait_for(0) / bounded wait_for slices only.
   - Ready futures are consumed; unready futures stay tracked.

3. Detect deadline expiry as local intent
   - Each rank compares monotonic elapsed time against kv_transfer_timeout_ms.
   - The local result is "rid X wants CANCELLED because deadline expired".
   - This step does not free resources and does not unilaterally change the
     request's final state.

4. Reach rank consensus
   - L1: union timed-out request IDs across ranks.
   - L2: gather packed `(rid, state)` intents and reduce to one global
     outcome per request:
       CANCELLED or FAILED on any rank => global terminal outcome;
       COMPLETED only if every rank reports completed;
       otherwise IN_PROGRESS.

5. Apply the consensus outcome
   - IN_PROGRESS: keep polling.
   - COMPLETED: finish transfer and release resources normally.
   - CANCELLED / FAILED: stop scheduling the request and report an error
     response, but keep KV blocks and transfer buffers pinned until cleanup is
     safe.

6. Request transport unwind once
   - For NIXL, call the cancel / release path once per request and mark the
     affected slot as quarantined if transport quiescence is unknown.
   - For direct UCX / MPI / Mooncake, use the best available upper-layer
     cancellation semantics and document that true mid-flight transport cancel
     is unavailable unless the backend grows it.

7. Cleanup only after quiescence
   - If C++ transfer status becomes terminal, Python runs
     _end_transfer_and_maybe_terminate exactly once and frees KV resources.
   - If a quarantined slot's NIXL status becomes SUCCESS / FAILURE, release the
     handle and return the slot to the pool.
   - If the quarantine deadline expires, leave the slot poisoned and fail
     closed only if no clean capacity remains.
```

The central invariant is: **logical cancellation can be immediate;
physical reuse is delayed until quiescence or fail-closed isolation.**

## Implementation Approach

### Recommended Path: V1 Consensus + Quarantine, Then Default-ON

The shortest safe path is to keep V1 + C++ supported for current
deployments while adding the missing consensus and cleanup semantics:

1. Land the bounded status polling from PR #13713, scoped to
   `checkContextTransferStatus` and `checkGenTransferStatus`.
2. Use PR #14746's L1 timeout-flag union where available.
3. Add V1 L2 outcome consensus with the packed-state allgather.
4. Re-introduce NIXL cancel / fail-closed memory-safety checks behind
   the existing env var.
5. Replace immediate pool-wide poison escalation with per-slot
   quarantine plus `PendingQuiescenceTracker`.
6. Add block-accounting diagnostics that prove whether F3's
   originating trigger is KV-block / credit pressure on both ends.
7. Keep default-OFF until the Phase 0 marathon and consensus configs
   pass; then flip the env var default.

This path intentionally does not require migrating production V1 users
to V2 before they can get cancellation recovery.

### Alternative: Route Deployments To V2

V2's Python transceiver already has the right consensus shape, so the
clean architectural answer is to move disaggregated serving to V2 where
possible. This is not enough as the immediate answer because:

- V2 is currently tied to the Python transceiver and NIXL path;
- V1 + C++ remains the customer deployment shape in the reproducer;
- UCX / MPI / Mooncake policy still has to be documented or implemented;
- V2 still needs Phase 0 coverage for multi-node, ADP, and EP
  cancellation pressure before it can be treated as universally covered.

### Alternative: Shared Cancellation Orchestrator

The long-term clean design is a shared orchestrator used by both V1
`BindKvCacheTransceiver` and V2 `KvCacheTransceiverV2`. The orchestrator
would own:

- deadline intent collection;
- consensus outcome reduction;
- request terminal-state application;
- quiescence / quarantine lifecycle;
- metrics and debug reporting.

This avoids maintaining two cancellation models, but it is a larger
refactor. It should follow the V1 follow-up unless we decide to pause
V1 work entirely.

## Action Items

| Priority | Item | Acceptance signal |
|---|---|---|
| P0 | Finish Phase 0 stress harness: `load_thread`, marathon YAMLs, canary references, pytest registration, full-duration runs. | Weekly stress suite can run two 2 h marathons and produce canary / log / KV-util evidence. |
| P0 | Add TP/PP/EP consensus-focused stress configs. | Rank-batch divergence, collective deadlock, and unreclaimed-KV regressions fail loudly. |
| P1 | Land bounded C++ transfer-status polling. | A stuck future cannot block the engine loop longer than the configured slice; shutdown drain remains explicit. |
| P1 | Land V1 L2 cancellation outcome consensus. | The same request receives the same CANCELLED / FAILED / COMPLETED / IN_PROGRESS outcome on every rank. |
| P1 | Route deadline expiry through consensus before state changes. | Timeout detection may be per-rank, but state mutation is never per-rank. |
| P1 | Reintroduce transport-safe NIXL cancel with request lifetime already protected by PR #14979. | No request UAF, no broken promise, no eager free after cancel. |
| P1 | Add transceiver-level block-accounting logs around cancel / timeout. | F3's inferred KV-block / credit-pressure trigger becomes directly provable or falsifiable in logs. |
| P2 | Implement per-slot quarantine and `PendingQuiescenceTracker`. | A mid-flight cancel reserves only the affected slot and returns it after terminal NIXL status; process restart is last resort. |
| P2 | Retarget `has_poisoned_transfer_buffer()` semantics. | Python fail-closed fires on unrecoverable pool exhaustion, not on the first quarantined slot. |
| P2 | Audit V2 coverage against C1-C5 and the same stress matrix. | V2 has explicit pass/fail evidence for TP/PP/EP/ADP cells claimed as supported. |
| P3 | Decide backend policy for direct UCX, MPI, and Mooncake. | Docs and code agree on whether those backends support true mid-flight cancel, observe-only cancel, or fail-closed isolation. |
| P3 | Flip default after stress evidence. | `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL` can become default-ON without replaying RC-1 / RC-2 / RC-3. |

## Open Questions

1. **Where should V1 L2 consensus live?** C++ is closest to the state
   transition, but Python is where PR #14746-style timeout flags live.
   The implementation should avoid a Python consensus that can still be
   invalidated by a later C++ local state mutation.
2. **What is the quarantine deadline?** It must be long enough for
   transient SIGSTOP / backpressure recovery, short enough to avoid
   silently losing all slots. Phase 0 injections should size it.
3. **Can NIXL status polling prove remote quiescence for every failure
   mode we care about?** If not, the fallback remains fail-closed
   isolation.
4. **Do we keep pool size 1 as the default?** Pool size 1 plus
   quarantine avoids UAF but pauses transfer serving while the only slot
   is quarantined. Larger defaults improve availability but cost VRAM.
5. **Should progress-based deadlines replace elapsed-time deadlines?**
   Elapsed-time deadlines can cancel healthy-but-slow transfers. A
   future NIXL progress signal would let us cancel only no-progress
   transfers.

## Cross-References

- [`README.md`](README.md) - overall roadmap and evidence summary.
- [`phase1-consensus-collective-design.md`](phase1-consensus-collective-design.md) - packed V1 consensus collective proposal.
- [`phase0-stress-test-suite.md`](phase0-stress-test-suite.md) - regression gate for this design.
- [`../../investigations/nvbug-6104831-disagg-permanent-wedge/19-exp4-f1-f2-f3-decomposition.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/19-exp4-f1-f2-f3-decomposition.md) - external Dynamo exp4 F1/F2/F3 analysis.
- [`../../investigations/nvbug-6104831-disagg-permanent-wedge/18-pr-14746-prior-art-and-v1-two-layer-gap.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/18-pr-14746-prior-art-and-v1-two-layer-gap.md) - PR #14746 prior art and V1 L1/L2 gap.
