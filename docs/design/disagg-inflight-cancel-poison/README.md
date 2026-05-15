# Improving In-Flight Cancellation + Poison for Disaggregated KV Transfer

**Outline — to be filled in when work begins.**

| | |
|---|---|
| **JIRA** | [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Author** | Chien-Chun Hung |
| **Status** | Planned follow-up after PR `#13713` lands |
| **Depends on** | [PR `#13713`](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) — introduces the in-flight cancel + poison feature, ships disabled by default |

## Context

PR `#13713` introduced in-flight cancellation of disaggregated KV
transfers as part of the NVBug 6104831 fix
(see [investigation](../../investigations/nvbug-6104831-disagg-permanent-wedge/)).
The cancellation surface requires a memory-safety mechanism because
NIXL's `releaseXferReq` does not guarantee synchronous remote
quiescence — the local handle is released, but the remote peer may
still be reading from / writing to the advertised memory range. The
shipping shape in `#13713` chose the strictest fail-closed answer:

- On a cancel-mid-flight catch, set a **pool-wide** `mPoisoned` flag.
- All subsequent buffer acquires throw.
- Python's `_check_cache_transfer_errors` observes
  `has_poisoned_transfer_buffer() == true` → escalates to
  `_fail_closed_for_unquiesced_disagg_transfer` → PyExecutor shutdown
  → orchestrator restarts the pod.

This is *correct for safety* (no UAF; orchestrator gets a loud signal)
but **operationally aggressive**: one cancel = pod restart. The
feature is disabled by default in `#13713` for that reason.

The 2026-05-13 Qwen3-Coder-480B production incident showed this firing
at production-default 60 s timeout under natural transient backpressure
(decode worker briefly slow → both pools poisoned → 3 container
restarts in 10 min → NVCF instance recycle, ~25 min outage).

## Goals

1. **No UAF.** Memory safety is preserved on in-flight cancellation.
   The remote peer can finish its work without writing into a slot
   that's already been handed to another request.
2. **Minimize blast radius.** Freeze only the specific slot(s) that
   were holding cancelled transfers; un-freeze when the network
   backend reports terminal status, or after an application-level
   deadline if the backend never responds.
3. **Minimize user-visible disruption.** PyExecutor shutdown becomes
   the *last resort* — fires only when the pool is genuinely
   exhausted (every slot poisoned), not on individual cancel events.
4. **Convert the feature from off-by-default to on-by-default.** Once
   the blast radius is sized correctly, the feature provides real
   operability benefit (bounded recovery from terminal peer failures)
   that's worth enabling everywhere.

## Non-goals (for this iteration)

- **NIXL API changes.** The immediate implementation works entirely
  within TRT-LLM. A NIXL completion-callback API would simplify
  things (replace polling with callbacks), and a NIXL progress-signal
  API would let us distinguish stuck-vs-slow transfers — both worth
  pursuing with the NIXL team, but tracked separately.
- **Multi-backend abstraction.** This work targets NIXL specifically.
  Direct UCX, MPI, and other transports are not in scope (they already
  have synchronous cancellation primitives where they need them).

## Design directions (to be detailed in phase docs)

### Phase 1 — Convert pool-wide poison to per-slot poison

Smallest viable change. Drops the pool-wide kill switch; relies on the
existing `mBufferIndexFlag[i] == 2` marker (already in the data
structure but not gating selection).

- `BaseTransBufferManager::poisonBufferIndex`: drop
  `resource.mPoisoned.store(true)`; keep per-slot marker.
- `BaseTransBufferManager::assignBufferIndex`: drop the pool-wide
  `TLLM_CHECK_WITH_INFO(!mPoisoned)` gate. The existing search loop
  already skips slots where `flag != 0`.
- New: `BaseTransBufferManager::isPoolExhausted()` — returns true
  when every slot has `flag == 2`.
- Re-target Layer 5's `has_poisoned_transfer_buffer()` →
  `is_transfer_pool_exhausted()`.

**Effect:** one cancel-mid-flight reserves one slot permanently
(until process restart) but the pool keeps serving from the
remaining slots. Drain rate is bounded by *terminal failure rate*,
not cancel rate.

**Trade-off:** under sustained terminal failures, the pool slowly
drains. Acceptable for typical configs (terminal failures are rare).
Phase 2 addresses this.

### Phase 2 — Deferred un-poison via NIXL status polling

Adds a background tracker that holds reserved slots until NIXL
confirms terminal status. Most cancellations resolve via NIXL's own
eventual transition; only truly terminal failures expire the
application deadline.

- New class: `PendingQuiescenceTracker` (one per pool).
  - Holds `(NixlTransferStatus, slot_id, shared_ptr<LlmRequest>,
    deadline)` tuples.
  - Background thread polls each entry's `getXferStatus` at ~1 Hz.
  - On `SUCCESS` / `FAILURE`: `releaseXferReq`, reset
    `mBufferIndexFlag[slot] = 0` (slot returns to pool).
  - On deadline expiry: leave `mBufferIndexFlag[slot] = 2`
    permanently (effectively Phase 1's behavior).
- Modify `cacheFormatter.cpp` / `mlaCacheFormatter.cpp` /
  `dataTransceiver.cpp::requestSync` catch blocks: hand the
  `(status_handle, slot_id, llmRequest)` to the tracker instead of
  calling `sendHolder.poison()` directly.
- Lifetime management: tracker drains on process shutdown.

**Effect:** under transient failures (peer pause, transient
backpressure), slots return to the pool naturally once NIXL drains.
Pool exhaustion becomes very rare in practice.

**Trade-off:** background thread complexity, dependence on NIXL's
`getXferStatus` eventually transitioning. Acceptable in practice; the
deadline fallback covers pathological cases.

### Phase 3 (long-term) — NIXL completion callback

Requires NIXL API change. Replace the polling thread with
`registerOnComplete(handle, cb)`. UCX (NIXL's backend) already has
completion callbacks, so the architectural distance is small.

Track separately with the NIXL team; not blocking this work.

### Phase 4 (long-term) — Progress-based cancellation

Requires NIXL API change. Replace the current `kvTransferTimeoutMs`
total-elapsed-time criterion with a "no NIXL progress for X seconds"
criterion. Distinguishes healthy-but-slow from truly-stuck transfers,
avoiding cancellation of transfers that are about to complete.

Track separately with the NIXL team; not blocking this work.

## Dependency

```
PR #13713 lands
  │
  └─> Phase 1 — per-slot poison + Layer 5 re-target
       │
       └─> Phase 2 — deferred un-poison via NIXL polling
            │
            └─> Feature on-by-default
                 │
                 ├─> Phase 3 — NIXL callback API (parallel, requires NIXL)
                 └─> Phase 4 — progress-based cancel (parallel, requires NIXL)
```

## Open questions to resolve before implementation

1. **Empirical characterization of NIXL `getXferStatus` behavior**
   under transient (SIGSTOP/SIGCONT) and terminal (SIGKILL) peer
   failures. Does NIXL always eventually transition out of `IN_PROG`?
   If not, what's the empirical distribution of transition times?
   This sizes the application-level deadline in Phase 2.

2. **Lifetime ordering of NIXL handles and `shared_ptr<LlmRequest>`**
   across the tracker. Process shutdown needs to release handles
   before destroying the LlmRequest, or vice versa? Worth verifying
   with NIXL.

3. **Per-pool tracker vs single global tracker.** The current
   `BaseTransBufferManager` has multiple instances (one per buffer
   kind: kKV, kKV_INDEXER, kRNN). One tracker per manager keeps
   lifetime clean; one global tracker is fewer threads. Pick before
   implementation.

4. **Test fixture for deterministic NIXL `IN_PROG` → `SUCCESS`
   transition** in unit tests. The current §10 reproducers use real
   workloads to trigger the path; we want a way to exercise the
   tracker without running 480B inference.

## Empirical evidence motivating this work

- **§10 of the NVBug 6104831 investigation** —
  [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md).
  Six controlled A/B experiments comparing PR `#13713` head against
  an ablation branch with the cancel surface removed. Shows the
  pool-wide-poison cascade firing at 1 s timeout / under SIGSTOP.

- **2026-05-13 Qwen3-Coder-480B production incident.** Real-world
  cascade: 60 s default timeout fired on healthy-but-slow transfer
  (decode worker handling 119 inflight requests, transfer took 62.9
  s), both pools poisoned, 3 container restarts, NVCF instance
  recycle. The empirical proof that the current shape is too
  aggressive at production defaults.

## Detailed design documents

To be authored when work begins:

- `phase1-per-slot-poison.md` — phase 1 design + diff sketch + tests.
- `phase2-deferred-un-poison.md` — phase 2 design, tracker class
  design, lifetime / threading model, deadline policy.
- (Optional) `phase3-nixl-callback.md` — interface ask for the NIXL
  team if/when Phase 3 is prioritized.
- (Optional) `phase4-progress-based-cancel.md` — interface ask for
  the NIXL team if/when Phase 4 is prioritized.
