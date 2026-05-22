# Architecturally Correct Request Cancellation for Disaggregated KV Transfer

| | |
|---|---|
| **JIRA** | [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Author** | Chien-Chun Hung |
| **Status** | Architectural design under active rework after PR `#13713` lands as default-OFF |
| **Depends on** | [PR `#13713`](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) — introduces the in-flight cancel + poison + deferred-cleanup surface; ships gated under `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL`, default OFF |

> **Reframe (2026-05-22).** This document was originally scoped as
> "make the in-flight cancel + poison surface less operationally
> aggressive". That scope is too narrow. The CI failures uncovered
> while merging PR `#13713` with `upstream/main` (RC-1 MTP scheduler
> interaction, RC-2 TP allgather rank-batch divergence, RC-3 PP
> termination retry) showed that the cancellation + deferred-cleanup
> machinery has the same defect class across all of them: **per-rank
> decisions made without a consensus story across the parallelism
> strategies the executor supports (TP, PP, EP)**. The right
> architectural answer is not another round of patch-by-patch fixes;
> it is to design the cancellation semantics from first principles
> *with* the variation matrix and the consensus invariants in scope.
>
> This document now drives that re-design. The original
> "less-aggressive poison" phases are preserved as later items in the
> roadmap, but the load-bearing addition is a new Phase 1: an
> architectural design effort that pins down the cancellation
> contract across V1/V2 KV cache managers, C++/Python transceivers,
> NIXL/UCX network backends, and TP/PP/EP parallelism modes.

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

### Three CI failures that expanded the scope

While merging PR `#13713` with `upstream/main`, three real
regressions surfaced — all caused by the **deferred-cleanup
machinery** that PR `#13713` added alongside the cancel + poison
surface, not the cancel surface itself:

| RC | Test | Per-rank decision that diverges |
|---|---|---|
| RC-1 | `TestQwen3_5_35B_A3B.test_bf16_mtp[mtp_on]` | `_can_terminate_request_now` defers in disagg-transmission state; MTP speculative state is partially torn down; next `_prepare_inputs` returns `None`. |
| RC-2 | `TestDeepSeekV3Lite.test_auto_dtype_with_helix` (`pp1dp2cp2`, `pp2tp1cp2`) | Per-rank C++ deadline check fires at slightly different wall-clock; ranks produce different batches; one rank enters `tp_cp_allgather`, the other doesn't; MPI collective deadlocks. |
| RC-3 | `TestDisagg.test_asymmetric_executor[llama-4proc-mpi_kvcache-90]` | `DisaggPPTerminationHandler` switched to `_do_terminate_request_if_safe` which defers, but no retry path is wired. KV blocks pinned indefinitely; pool exhaustion; CUDA illegal memory access. |

These are not three independent bugs. They are three customer-visible
faces of one architectural gap: **the V1 + C++ transceiver path has
no consensus mechanism, and PR `#13713` extended it with logic that
quietly assumed one.** Per-rank "is this request in transmission?"
queries diverge under TP/PP/EP; per-rank "defer or terminate?"
decisions create split-brain on the next iteration's batch and on
KV-block reclamation.

The investigation report's
[§10 "Why we ship default-OFF"](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md#why-we-ship-default-off)
documents the empirical case in detail.

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
un-poison via polling) is the load-bearing change** for the poison-
operability dimension. It works regardless of pool size. Per-slot
poison + multi-slot configuration is a finer-grained improvement on
top, useful only when the deployment opts into a multi-slot pool.

## Scenario matrix

Disaggregated KV transfer has four orthogonal axes of variation. Any
cancellation design has to either work on the full Cartesian product
or call out the cells it intentionally excludes. This is the matrix:

| Axis | Variants | Currently observed cancellation surface |
|---|---|---|
| **KV cache manager** | V1 (`KVCacheManager`), V2 (`KVCacheManagerV2`) | V1: independent per-rank state, no consensus. V2: per-rank state, but the V2 *transceiver* (below) enforces consensus on top. |
| **Cache transceiver runtime** | C++ (`BindKvCacheTransceiver`), Python (`KvCacheTransceiverV2`) | C++: `cancelRequest` per rank, no consensus. Python (V2): `cancel_request` returns `False` if mid-write + `_consensus_outcome` across TP/PP. |
| **Network backend** | NIXL, direct UCX, MPI (deprecated), Mooncake | NIXL: PR `#13713` adds `release()` / per-request cancel flag. UCX/MPI: no cancellation primitives at the connection layer (cancellation lives in the upper transceiver). |
| **Parallelism mode** | TP, PP, CP (context-parallel), EP (expert-parallel), ADP (attention-DP), combinations | TP/PP exercised in PR `#13713` testing; CP exercised in RC-2; EP not yet exercised under cancellation pressure. All require per-iteration consensus on which requests are in the batch. |

**Compatibility restrictions (already in code):**

- `(V2 cache manager, C++ transceiver)` is rejected at startup — see
  `StressConfig.validate()` in the disagg cancellation stress
  harness, and the same constraint is enforced by
  `CacheTransceiverConfig`.
- `(Python transceiver, non-NIXL backend)` is rejected — see
  `kv_cache_transceiver.create_kv_cache_transceiver` line 113-120.
- Multi-node and ADP combinations need explicit verification per
  cell (none are excluded by code, but several are not on the test
  matrix yet).

**The two mainstream cells:**

1. `(V1, C++ transceiver, NIXL or direct UCX, TP/PP/EP/ADP)` — heritage
   path, the customer deployment shape, what PR `#13713` extends.
2. `(V2, Python transceiver, NIXL, TP/PP/EP/ADP)` — newer path, has
   built-in consensus, intended forward direction.

The architectural question this design has to answer is: **how do we
land a cancellation contract that is consistent across both
mainstream cells (and the network-backend variants under each),
without re-introducing the per-rank-divergence failure mode that
PR `#13713`'s deferred-cleanup did?**

## The consensus invariants

These are the per-iteration invariants any cancellation design must
preserve. The V2 transceiver's `_consensus_outcome` already
satisfies them for the V2 path; the V1 + C++ path currently does not.

> **C1 (Cancellation propagates globally).** If any rank reports a
> request as CANCELLED (whether user-initiated or
> timeout/peer-failure-initiated), all ranks must treat the request
> as CANCELLED on the same iteration.

> **C2 (Completion requires unanimity).** A request is COMPLETED
> only when every rank agrees. A subset-COMPLETED outcome on one
> rank holds the request open on all ranks until the next
> iteration's consensus.

> **C3 (Failure propagates globally, like cancellation).** If any
> rank reports FAILED, all ranks treat as FAILED on the same
> iteration.

> **C4 (Deferred cleanup is a globally consistent decision).** If
> the cleanup of a request's KV / pinned resources is deferred to a
> later iteration, every rank must defer for the same iterations.
> Per-rank "defer here / terminate there" is forbidden — that's the
> RC-1 / RC-2 / RC-3 failure mode.

> **C5 (Per-iteration batch composition is the consensus output).**
> The set of requests entering attention / collective ops on
> iteration N is the consensus over per-rank batches as of the
> latest cancellation/completion decision. No rank may unilaterally
> add or drop a request from its own batch.

V2 satisfies all five via `_consensus_outcome` followed by uniform
session-state updates. V1 + C++ satisfies none of them by
construction — it makes per-rank decisions and relies on the upper
layers to converge eventually.

## Goals

1. **Architectural correctness across the scenario matrix.** A
   cancellation contract that satisfies C1–C5 in both mainstream
   cells, with intentional and documented behaviour for every
   currently-supported variant (V1/V2 × C++/Python × NIXL/UCX/Mooncake
   × TP/PP/CP/EP/ADP).
2. **No UAF.** Memory safety is preserved on in-flight cancellation
   regardless of which cell of the matrix is active. The remote peer
   can finish its work without writing into a slot that's already
   been handed to another request.
3. **Minimize blast radius.** Freeze only the specific slot(s) that
   were holding cancelled transfers; un-freeze when the network
   backend reports terminal status, or after an application-level
   deadline if the backend never responds.
4. **Minimize user-visible disruption.** PyExecutor shutdown becomes
   the *last resort* — fires only when the pool is genuinely
   exhausted (every slot poisoned), not on individual cancel events.
5. **Convert the feature from off-by-default to on-by-default.** Once
   architectural correctness is established and the blast radius is
   sized correctly, the feature provides real operability benefit
   (bounded recovery from terminal peer failures) that's worth
   enabling everywhere. **Stretch goal**: same default behaviour for
   both mainstream cells, so operators don't need to think about
   which transceiver runtime they're using.

## Anti-goals (lessons from PR `#13713`)

These are the failure modes the architectural design is explicitly
avoiding:

- **Patch-by-patch fixes that solve each customer-visible symptom
  individually.** This is what produced the RC-1 / RC-2 / RC-3 set —
  each one looks like its own bug, all share the same per-rank
  decision root. Future fixes that touch cancellation must consult
  C1–C5 before landing.
- **"Make the existing thing less aggressive" framing.** The poison
  + fail-closed surface was the right *safety* answer for the
  no-quiescence-proof scenario; the wrong move would be to dilute it
  for operability and re-open the UAF window. Operability gains
  should come from making the decision globally consistent, not from
  weakening the safety stance.
- **Two divergent cancellation models for V1+C++ and V2+Python.**
  V2's consensus model is the architectural template; the V1 + C++
  path should adopt the same contract rather than inventing a
  parallel mechanism. (How exactly — pin to V2, extend V1, or write
  a thin adapter — is the open design question Phase 1 has to close.)

## Non-goals (for this iteration)

- **NIXL API changes.** The immediate implementation works entirely
  within TRT-LLM. A NIXL completion-callback API would simplify
  things (replace polling with callbacks), and a NIXL progress-signal
  API would let us distinguish stuck-vs-slow transfers — both worth
  pursuing with the NIXL team, but tracked separately as later
  phases.
- **Backend-specific cancellation primitives beyond what already
  ships.** Direct UCX and MPI do not have per-request mid-flight
  cancellation at the connection layer; the application-level
  cancellation contract (C1–C5) is the integration point, not a new
  UCX/MPI capability. Mooncake's cancellation semantics need a brief
  study before the design lands, but no Mooncake-internal changes
  are in scope.
- **Replacing the V2 transceiver.** V2 is the architectural template
  for this work; we're aligning around it, not replacing it.

## Roadmap

The phases below are ordered by architectural dependency, not by
chronological convenience. **Phase 0 (stress test) and Phase 1
(architectural design) must complete before any behavioural code
changes land** — otherwise we replay the patch-by-patch failure mode
that produced RC-1 / RC-2 / RC-3.

### Phase 0 — Stress-testing infrastructure for the cancel + poison surface

**Status:** Prerequisite. Should land *before* any of the behavioural
phases so we can A/B each one and not regress earlier ones.

The §10 ablation harness already gives us most of what we need — six
controlled A/B experiments comparing PR `#13713` head against an
ablation branch — but it lives as ad-hoc scripts in
`local/pr13713-rc13-clean/.repro/`. We should productionize a subset
of it as a permanent stress-test suite that **covers both mainstream
cells of the scenario matrix**:

- `(V1, C++ transceiver, NIXL + UCX plugin, TP/PP/EP/ADP)` — the
  customer deployment shape, what PR `#13713` extends.
- `(V2, Python transceiver, NIXL, TP/PP/EP/ADP)` — the architectural
  template.

Per-cell tests:

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
- **Consensus stress: TP=2 / PP=2 / EP variants with mid-flight
  cancellation under load.** Specifically targets the C1–C5
  invariants. Pass criterion: no rank-batch divergence; no MPI
  collective deadlock; KV blocks reclaimed within bounded iterations
  on cancel.

Each test should report: `Broken promise` count, `Cannot cancel
request` count, `Poisoned ... cache transfer buffer` count, recovery
time after injected failure, per-rank batch divergence count
(consensus violation), final PyExecutor health.

Code organisation:

- The stress-harness skeleton has landed in
  `tests/integration/defs/stress_test/disagg_cancel/` (PR `#14375`).
  Marathon configs cover both mainstream cells.
- Mark them as nightly / opt-in (they take minutes per case and
  require multi-process / NIXL setup).
- Wire each behavioural phase below to a *specific* set of tests so
  regression is locally visible: e.g., Phase 2 must keep the SIGSTOP
  test recovering in <5 s; Phase 3 must add a "pool exhausts
  gracefully after N cancels" test; consensus tests gate any change
  to the V1 + C++ deferred-cleanup machinery.

The §10 doc itself is the spec for what these tests should measure;
the work is mostly converting that doc into a maintained suite plus
extending it for the consensus dimension.

### Phase 1 — Architectural design for request cancellation across the scenario matrix

**This is the load-bearing change after the merge with `upstream/main`.**
It precedes every operability/poison-shape phase below. Without it,
landing changes to either mainstream cell risks reopening the same
defect class that produced RC-1 / RC-2 / RC-3.

**Deliverable:** `phase1-architectural-design.md`, covering the
following sections at minimum.

1. **Cancellation contract.** A formal statement of C1–C5 above
   (cancellation/failure propagate globally, completion requires
   unanimity, deferred cleanup is a global decision, per-iteration
   batch composition is the consensus output). Naming the contract
   makes it reviewable; every later phase's design notes cite the
   contract.

2. **V2 transceiver as the template.** Document `_consensus_outcome`,
   `_ctx_consensus_outcome`, `_gen_consensus_outcome` from
   `tensorrt_llm/_torch/disaggregation/transceiver.py` as the
   reference implementation. Identify gaps if any (multi-node?
   ADP-specific paths? EP not yet exercised).

3. **V1 + C++ path alignment strategy.** Three candidates to choose
   from, with explicit trade-offs:

   - **(a)** Add an equivalent consensus layer to the V1 + C++ path
     in Python (one rank's `cancel_request` outcome broadcast,
     mirroring V2). Pros: smallest change to C++; keeps V1 as a
     supported runtime for at least one more rc cycle. Cons:
     duplication of consensus logic that already lives in V2;
     ongoing maintenance of two near-identical paths.
   - **(b)** Migrate disagg workloads to the V2 runtime where the
     compat matrix allows, and put V1 + C++ in maintenance mode for
     cancellation. Pros: single source of truth; V2 is already the
     intended forward direction. Cons: V2 is NIXL-only; UCX
     deployments are stuck on V1 unless V2 grows UCX support; the
     V1 → V2 migration has its own correctness story to write.
   - **(c)** Extract a shared cancellation orchestrator that both
     `BindKvCacheTransceiver` and `KvCacheTransceiverV2` delegate
     to. The orchestrator owns the consensus state; the two
     runtimes just propagate "session is mid-write" / "session has
     completed" signals. Pros: cleanest long-term shape. Cons:
     largest design effort; touches both code paths; needs a
     migration plan to land safely.

   Picking (a) / (b) / (c) is the central decision Phase 1 has to
   close. Out of the box, (a) looks like the shortest path for
   `rc14`/`rc15`; (c) looks like the architectural endgame. The
   recommendation should justify the choice against operational
   risk, code-churn, and the deprecation arc for V1.

4. **Deferred cleanup as a consensus output.** Define when "defer
   the freeing of Python KV resources" is a globally consistent
   decision (rather than a per-rank one). The naive version is
   "defer iff *any* rank says the session is still TRANSFERRING";
   that's correct but may be too aggressive at scale. The design
   should empirically size the cost.

5. **Network-backend asymmetry handling.** NIXL has a per-request
   cancel primitive (`releaseXferReq` + the cancel-flag branch
   PR `#13713` added). Direct UCX and MPI do not; cancellation for
   those backends is purely Python-level (drop the session, let the
   transport run to completion). The design has to either (i)
   accept that UCX/MPI deployments cannot truly cancel a mid-flight
   transfer and document the deployment-level consequence (longer
   recovery on peer failures), or (ii) propose what the backend
   needs to grow. Be explicit.

6. **TP / PP / CP / EP / ADP coverage.** Each parallelism axis
   imposes its own consistency invariant; the design should call
   out which ones are covered by C1–C5 and which (if any) need
   axis-specific reasoning. EP is the one we haven't exercised
   under cancellation pressure yet — the design should commit to
   exercising it in Phase 0 before the behavioural phases land.

7. **Migration plan.** How to turn the env-var-gated default-OFF
   surface into the consensus-driven default-ON surface without
   re-running RC-1 / RC-2 / RC-3. Suggested arc:
   `rc14`: design lands as a doc + targeted tests in Phase 0;
   `rc15`: V1 + C++ consensus implementation lands behind the same
   env var, flips to default-ON when the consensus tests are green;
   `rc16`: env var removed; cancellation is unconditional and
   correct on both mainstream cells.

**Why this is "load-bearing".** Every later phase (deferred
un-poison, multi-slot, NIXL callback, progress-based cancel) makes
sense only after the consensus contract is in place. Otherwise
they're more per-rank surface area that the next deployment shape
will re-divide-and-conquer.

### Phase 2 — Deferred un-poison via NIXL status polling

(Was Phase 1 in the original plan. Renumbered after the architectural
phase was inserted.)

**Status:** Behavioural change; predicated on Phase 1's consensus
contract. The deferred-un-poison mechanism is a per-iteration
decision on whether each pool slot can be returned to circulation;
Phase 1 makes that decision globally consistent. Once consensus is
in place, the mechanism itself follows the original plan:

- New class: `PendingQuiescenceTracker` (one per pool).
  - Holds `(NixlTransferStatus, slot_id, shared_ptr<LlmRequest>,
    deadline)` tuples.
  - Background thread polls each entry's `getXferStatus` at ~1 Hz.
  - On `SUCCESS` / `FAILURE`: `releaseXferReq`, reset
    `mBufferIndexFlag[slot] = 0` (slot returns to pool), pool-wide
    `mPoisoned` clears (or is recomputed from "any slot still
    reserved").
  - On deadline expiry: leave `mBufferIndexFlag[slot] = 2`
    permanently and keep `mPoisoned` set — back to current
    fail-closed behaviour, but only for slots whose remote peer
    genuinely never quiesced.
- Modify `cacheFormatter.cpp` / `mlaCacheFormatter.cpp` /
  `dataTransceiver.cpp::requestSync` catch blocks: hand the
  `(status_handle, slot_id, llmRequest)` to the tracker instead of
  calling `sendHolder.poison()` directly.
- Lifetime management: tracker drains on process shutdown.
- **Per-rank slot reservation must agree across TP/PP/EP** (this is
  the consensus requirement Phase 1 establishes; Phase 2 honours
  it).

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

### Phase 3 — Configurable multi-slot pools + finer-grained per-slot poison

(Was Phase 2. Predicated on Phase 2's tracker.)

Provides additional fault tolerance by letting deployments opt into
pool sizes > 1, so individual cancellations become invisible at the
API level (other slots keep serving).

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

**Note:** Phase 2's tracker already handles per-slot lifecycle
correctly. Phase 3 is mostly about (a) making the pool size easy to
configure, (b) re-targeting Layer 5's trigger, and (c) measuring +
deciding the default.

### Phase 4 — NIXL completion callback (requires NIXL API change)

Replace Phase 2's polling thread with `registerOnComplete(handle, cb)`.
UCX (NIXL's typical backend) already has completion callbacks, so the
architectural distance is small. Cleaner than polling: no background
thread, lower latency from "NIXL done" to "slot returned to pool".

Track separately with the NIXL team. Not blocking; Phase 2's polling
implementation is sufficient as an interim.

### Phase 5 — Progress-based cancellation (requires NIXL API change)

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
PR #13713 lands (default-OFF, gated under TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL)
  │
  ├─> Phase 0 — productionize stress-test suite (covers both mainstream cells)
  │
  └─> Phase 1 — architectural design (consensus contract; V1/V2 alignment strategy)
       │       Output: phase1-architectural-design.md + targeted Phase 0 tests
       │
       └─> Phase 2 — deferred un-poison via NIXL polling
            │       (load-bearing for operability; works at pool size 1;
            │        honours Phase 1's consensus contract)
            │
            └─> Phase 3 — multi-slot config + per-slot poison
                 │       (finer-grained when pool size > 1)
                 │       Feature on-by-default after this on both mainstream cells
                 │
                 ├─> Phase 4 — NIXL callback API (parallel, requires NIXL)
                 └─> Phase 5 — progress-based cancel (parallel, requires NIXL)
```

## Open questions to resolve before implementation

### Architectural (gating Phase 1)

1. **V1 alignment strategy: (a) duplicate consensus in Python on the
   V1 + C++ path, (b) migrate disagg workloads to V2 where the
   compat matrix allows, or (c) build a shared cancellation
   orchestrator that both runtimes delegate to.** This is the
   biggest open question; the recommendation should be supported by
   code-churn and risk analysis on at least one representative
   model.

2. **What is V2's true coverage of the scenario matrix?** Multi-node?
   ADP-specific paths? EP under cancellation pressure? The current
   `_consensus_outcome` implementation needs an audit against
   C1–C5; the design should call out any cells where V2 itself does
   not yet meet the contract.

3. **Network-backend asymmetry policy.** Direct UCX and MPI have no
   per-request cancellation primitive at the connection layer.
   Acceptable to document a "longer recovery on peer failure for
   UCX/MPI deployments" deployment-time consequence, or do we need
   to push a cancellation primitive into UCX/MPI?

4. **EP coverage in Phase 0.** Expert-parallel hasn't been exercised
   under cancellation pressure yet. Need a representative MoE model
   (DeepSeek-V3 or Qwen3-MoE) in the stress harness before Phase 1
   can claim EP coverage.

### Operational (gating Phase 2 and beyond)

5. **Empirical characterization of NIXL `getXferStatus` behavior**
   under transient (SIGSTOP/SIGCONT) and terminal (SIGKILL) peer
   failures. Does NIXL always eventually transition out of `IN_PROG`?
   If not, what's the empirical distribution of transition times?
   This sizes the application-level deadline in Phase 2.

6. **Lifetime ordering of NIXL handles and `shared_ptr<LlmRequest>`**
   across the tracker. Process shutdown needs to release handles
   before destroying the LlmRequest, or vice versa? Worth verifying
   with NIXL.

7. **Per-pool tracker vs single global tracker.** The current
   `BaseTransBufferManager` has multiple instances (one per buffer
   kind: kKV, kKV_INDEXER, kRNN). One tracker per manager keeps
   lifetime clean; one global tracker is fewer threads. Pick before
   implementation.

8. **VRAM cost vs. fault tolerance trade-off for Phase 3 defaults.**
   Stay at 1/1 (current), raise to 2/2, or larger? Each slot
   pre-allocates `mTransferBufferSize` of VRAM. Needs measurement on
   representative models (DeepSeek, Qwen 480B, Llama-class) before
   recommending a default.

9. **Test fixture for deterministic NIXL `IN_PROG` → `SUCCESS`
   transition** in unit tests. Phase 0's productionized harness uses
   real workloads; we also want a way to exercise the tracker
   without running 480B inference.

## Empirical evidence motivating this work

- **§10 of the NVBug 6104831 investigation** —
  [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md).
  Six controlled A/B experiments comparing PR `#13713` head against
  an ablation branch with the cancel surface removed. Shows the
  pool-wide-poison cascade firing at 1 s timeout / under SIGSTOP.
  The "Why we ship default-OFF" section documents the RC-1 / RC-2 /
  RC-3 CI failures that motivated this design's architectural
  reframe.

- **2026-05-13 Qwen3-Coder-480B production incident.** Real-world
  cascade: 60 s default timeout fired on healthy-but-slow transfer
  (decode worker handling 119 inflight requests, transfer took 62.9
  s), both pools poisoned, 3 container restarts, NVCF instance
  recycle. The empirical proof that the current shape is too
  aggressive at production defaults.

- **V2 transceiver's `_consensus_outcome` implementation** —
  `tensorrt_llm/_torch/disaggregation/transceiver.py`. The reference
  point for the cancellation contract C1–C5; Phase 1's design
  starts from this code.

## Detailed design documents

- [`phase0-stress-test-suite.md`](phase0-stress-test-suite.md) —
  **READY** — disaggregated cancellation stress-test suite (two
  2-hour marathon tests covering the V1+C++ and V2+Python mainstream
  configs at 3P3D, parametrized YAML configs, threaded harness,
  pass-criteria definition, file layout, implementation-handover
  spec). Targets [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648).
  **Action item:** extend with the consensus dimension (TP/PP/EP
  rank-batch divergence checks) and add an EP-bearing model
  (DeepSeek-V3 or Qwen3-MoE) before Phase 1 can claim EP coverage.
- `phase1-architectural-design.md` — **to be authored — load-bearing.**
  Cancellation contract (C1–C5), V2 template walkthrough, V1
  alignment strategy decision (a / b / c), deferred-cleanup as a
  consensus output, network-backend asymmetry policy, parallelism-
  axis coverage, migration plan. Every later phase cites this doc.
- `phase2-deferred-un-poison.md` — to be authored — tracker class
  design, lifetime / threading model, deadline policy, NIXL status
  polling cadence. Honours the consensus contract from Phase 1.
- `phase3-multi-slot-config.md` — to be authored — `CacheTransceiverConfig`
  surface, default recommendation with VRAM measurements, Layer 5
  re-target.
- (Optional) `phase4-nixl-callback.md` — interface ask for the NIXL
  team if/when Phase 4 is prioritized.
- (Optional) `phase5-progress-based-cancel.md` — interface ask for
  the NIXL team if/when Phase 5 is prioritized.
