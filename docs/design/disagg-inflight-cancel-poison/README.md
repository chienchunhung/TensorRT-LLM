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

### A complication from the default config

The transfer-buffer pool is **size 1 by default** on both sender and
receiver sides (see
`cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp:37-38` —
`mRecvBufferCount = 1` and `mSendBufferCount = 1` unless
`TRTLLM_REQUEST_KV_CACHE_CONCURRENT`,
`TRTLLM_KVCACHE_RECV_BUFFER_COUNT`, or
`TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM` are set).

This means "per-slot poison" and "pool-wide poison" are operationally
equivalent in the default config: one cancel = one slot = entire pool
dead. The 2026-05-13 incident logs are consistent with this — the
poison messages reference `index=0`, the only slot.

The design ordering below reflects this: **temporal recovery (deferred
un-poison via polling) is the load-bearing change** — it works
regardless of pool size. Per-slot poison + multi-slot configuration is
a finer-grained improvement on top, useful only when the deployment
opts into a multi-slot pool.

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

## Roadmap

### Phase 0 — Stress-testing infrastructure for the cancel + poison surface

**Status:** Prerequisite. Should land *before* any of the behavioural
phases so we can A/B each one and not regress earlier ones.

The §10 ablation harness already gives us most of what we need — six
controlled A/B experiments comparing PR `#13713` head against an
ablation branch — but it lives as ad-hoc scripts in
`local/pr13713-rc13-clean/.repro/`. We should productionize a subset
of it as a permanent stress-test suite:

- **Ablation: 1 s timeout, CONC=64, NIXL native.** Exercises the
  saturation regime where cancels race NIXL mid-flight. Pre-#13713 /
  ablation-branch behaviour: wedge with 89 `Broken promise` events.
  Head: 0 broken promises, 2/5 PASS.
- **Ablation: 5 s timeout + SIGSTOP-gen-8004 for 20 s.** Transient
  peer pause as a controlled stand-in for the 2026-05-13 incident.
  Pre-#13713: NO RECOVERY in 60 s. Head: HTTP 200 at +1.71 s, then
  Layer 5 fail-closed by design.
- **Baseline: 60 s timeout, CONC=64 / 256, NIXL+UCX, no injected
  failures.** Smoke test that defenses are dormant under
  production-default load.
- **Stress: 60 s timeout, sustained CONC=128, slow receiver simulated
  via artificial inflight-count pressure on decode.** Closest in-CI
  approximation of the Qwen 480B production incident.

Each test should report: `Broken promise` count, `Cannot cancel
request` count, `Poisoned ... cache transfer buffer` count, recovery
time after injected failure, final PyExecutor health.

Code organisation:

- Add a stress-harness module under
  `tests/integration/disagg/` (or wherever the existing disagg
  integration tests live) that wraps the §10 harness scripts.
- Mark them as nightly / opt-in (they take minutes per case and
  require multi-process / NIXL setup).
- Wire each behavioural phase below to a *specific* set of tests so
  regression is locally visible: e.g., Phase 1 must keep the SIGSTOP
  test recovering in <5 s, Phase 2 must add a "pool exhausts gracefully
  after N cancels" test, etc.

The §10 doc itself is the spec for what these tests should measure;
the work is mostly converting that doc into a maintained suite.

### Phase 1 — Deferred un-poison via NIXL status polling

**This is the load-bearing change.** Works regardless of pool size,
including the default size-1 configuration.

Adds a background tracker that holds reserved slots until NIXL
confirms terminal status. Most cancellations resolve via NIXL's own
eventual transition; only truly terminal failures expire the
application deadline.

- New class: `PendingQuiescenceTracker` (one per pool).
  - Holds `(NixlTransferStatus, slot_id, shared_ptr<LlmRequest>,
    deadline)` tuples.
  - Background thread polls each entry's `getXferStatus` at ~1 Hz.
  - On `SUCCESS` / `FAILURE`: `releaseXferReq`, reset
    `mBufferIndexFlag[slot] = 0` (slot returns to pool), pool-wide
    `mPoisoned` clears (or is recomputed from "any slot still
    reserved").
  - On deadline expiry: leave `mBufferIndexFlag[slot] = 2`
    permanently and keep `mPoisoned` set — back to Phase 1's old
    fail-closed behaviour, but only for slots whose remote peer
    genuinely never quiesced.
- Modify `cacheFormatter.cpp` / `mlaCacheFormatter.cpp` /
  `dataTransceiver.cpp::requestSync` catch blocks: hand the
  `(status_handle, slot_id, llmRequest)` to the tracker instead of
  calling `sendHolder.poison()` directly.
- Lifetime management: tracker drains on process shutdown.

**Effect at pool size 1:** the single slot gets reserved on
cancel-mid-flight; the pool stops serving briefly while we wait for
NIXL. When NIXL drains (typically seconds for transient failures),
the slot returns to the pool and serving resumes — *without* a pod
restart. Only truly terminal peer failures expire the deadline and
preserve the current fail-closed behaviour.

**Effect at pool size > 1:** same as above, but the pool keeps
serving from other clean slots during the wait, so there's no serving
pause at all.

**Trade-off:** background thread complexity; dependence on NIXL's
`getXferStatus` eventually transitioning out of `IN_PROG`.
Acceptable in practice — the deadline fallback covers pathological
cases. Worth a short empirical study on NIXL's status transition
behaviour under SIGSTOP/SIGKILL before committing the deadline value.

### Phase 2 — Configurable multi-slot pools + finer-grained per-slot poison

Builds on Phase 1. Provides additional fault tolerance by letting
deployments opt into pool sizes > 1, so individual cancellations
become invisible at the API level (other slots keep serving).

- Surface pool sizes through `CacheTransceiverConfig` instead of
  (or in addition to) env vars. Cleaner ergonomics than the current
  `TRTLLM_REQUEST_KV_CACHE_CONCURRENT` /
  `TRTLLM_KVCACHE_RECV_BUFFER_COUNT` /
  `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM` triad.
- Decide on defaults: stay at 1/1, or raise to e.g. 2/2 or 4/4? The
  VRAM cost per slot is `mTransferBufferSize` (often GB-scale for
  large models). Needs measurement + a deployment-shape recommendation.
- Make per-slot poison a strict improvement (today it's inert at
  pool size 1):
  - Re-target `has_poisoned_transfer_buffer()` →
    `is_transfer_pool_exhausted()` (only fires when *every* slot is
    poisoned with no recovery in sight).
  - Layer 5 fail-closed triggers only on full exhaustion, not on any
    individual slot poisoning.
- Document the operational characteristic: with pool size N and an
  application deadline of D seconds, the deployment can tolerate up
  to N cancel-mid-flight events within D seconds before
  serving-degradation starts.

**Note:** Phase 1's tracker already handles per-slot lifecycle
correctly. Phase 2 is mostly about (a) making the pool size easy to
configure, (b) re-targeting Layer 5's trigger, and (c) measuring +
deciding the default.

### Phase 3 — NIXL completion callback (requires NIXL API change)

Replace Phase 1's polling thread with `registerOnComplete(handle, cb)`.
UCX (NIXL's typical backend) already has completion callbacks, so the
architectural distance is small. Cleaner than polling: no background
thread, lower latency from "NIXL done" to "slot returned to pool".

Track separately with the NIXL team. Not blocking; Phase 1's polling
implementation is sufficient as an interim.

### Phase 4 — Progress-based cancellation (requires NIXL API change)

Replace the current `kvTransferTimeoutMs` total-elapsed-time criterion
with a "no NIXL progress for X seconds" criterion. Distinguishes
healthy-but-slow from truly-stuck transfers:

- Healthy-but-slow (the 2026-05-13 incident pattern): NIXL is steadily
  moving bytes; `last_progress_time` keeps advancing; no cancel fires
  even past the elapsed-time threshold. Transfer completes normally.
- Truly stuck (peer dead, NIXL hung): byte count is frozen;
  `last_progress_time` doesn't advance; cancel fires once stall
  exceeds the threshold.

**Requires NIXL to expose a progress signal** on `XferStatus` (e.g.,
`getBytesTransferred()` or an incremental callback). UCX tracks this
internally; NIXL would need to surface it.

Track separately with the NIXL team. Highest long-term leverage —
this is what would let us avoid cancelling healthy transfers entirely,
which is the cleanest answer to the 2026-05-13 incident's root cause.

## Dependency

```
PR #13713 lands
  │
  └─> Phase 0 — productionize stress-test suite
       │
       └─> Phase 1 — deferred un-poison via NIXL polling
            │       (the load-bearing change; works at pool size 1)
            │
            └─> Phase 2 — multi-slot config + per-slot poison
                 │       (finer-grained when pool size > 1)
                 │       Feature on-by-default after this
                 │
                 ├─> Phase 3 — NIXL callback API (parallel, requires NIXL)
                 └─> Phase 4 — progress-based cancel (parallel, requires NIXL)
```

## Open questions to resolve before implementation

1. **Empirical characterization of NIXL `getXferStatus` behavior**
   under transient (SIGSTOP/SIGCONT) and terminal (SIGKILL) peer
   failures. Does NIXL always eventually transition out of `IN_PROG`?
   If not, what's the empirical distribution of transition times?
   This sizes the application-level deadline in Phase 1.

2. **Lifetime ordering of NIXL handles and `shared_ptr<LlmRequest>`**
   across the tracker. Process shutdown needs to release handles
   before destroying the LlmRequest, or vice versa? Worth verifying
   with NIXL.

3. **Per-pool tracker vs single global tracker.** The current
   `BaseTransBufferManager` has multiple instances (one per buffer
   kind: kKV, kKV_INDEXER, kRNN). One tracker per manager keeps
   lifetime clean; one global tracker is fewer threads. Pick before
   implementation.

4. **VRAM cost vs. fault tolerance trade-off for Phase 2 defaults.**
   Stay at 1/1 (current), raise to 2/2, or larger? Each slot
   pre-allocates `mTransferBufferSize` of VRAM. Needs measurement on
   representative models (DeepSeek, Qwen 480B, Llama-class) before
   recommending a default.

5. **Test fixture for deterministic NIXL `IN_PROG` → `SUCCESS`
   transition** in unit tests. Phase 0's productionized harness uses
   real workloads; we also want a way to exercise the tracker
   without running 480B inference.

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

- [`phase0-stress-test-suite.md`](phase0-stress-test-suite.md) —
  **READY** — disaggregated cancellation stress-test suite (two
  2-hour marathon tests covering the V1+C++ and V2+Python mainstream
  configs at 3P3D, parametrized YAML configs, threaded harness,
  pass-criteria definition, file layout, implementation-handover
  spec). Targets [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648).
- `phase1-deferred-un-poison.md` — to be authored — tracker class
  design, lifetime / threading model, deadline policy, NIXL status
  polling cadence.
- `phase2-multi-slot-config.md` — to be authored — `CacheTransceiverConfig`
  surface, default recommendation with VRAM measurements, Layer 5
  re-target.
- (Optional) `phase3-nixl-callback.md` — interface ask for the NIXL
  team if/when Phase 3 is prioritized.
- (Optional) `phase4-progress-based-cancel.md` — interface ask for
  the NIXL team if/when Phase 4 is prioritized.
