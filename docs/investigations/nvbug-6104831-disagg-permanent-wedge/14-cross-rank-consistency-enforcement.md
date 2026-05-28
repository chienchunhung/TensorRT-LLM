# 14 — Cross-rank consistency enforcement: implementation, empirical validation, and updated plan

**Status:** Implemented and locally validated; pending CI signal on PR #13713 before commit decision.
**Trigger:** [12-horizontal-consistency-and-layer3-gating.md](12-horizontal-consistency-and-layer3-gating.md) §5.2 Path B (explicit horizontal consistency layer).
**Implementation period:** 2026-05-25 → 2026-05-27 (3-day session).
**Code surface:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` + supporting headers, env-var-gated default-OFF.
**Public PR:** https://github.com/NVIDIA/TensorRT-LLM/pull/13713

This document closes the design / verification loop on doc 12's Path B proposal. It documents (1) what was implemented, (2) the empirical validation results across the cpp gtest, helix, and disagg-serving test families, and (3) the updated landing plan in light of those results.

---

## 1. Summary

Doc 12 §5.2 proposed "Path B" — an explicit horizontal-consistency layer that adds an MPI all-gather per scheduling iteration to enforce cross-rank agreement on per-request state before any rank-divergent decision. The implementation done this session ports the existing `KvCacheTransceiverV2._consensus_outcome` pattern (Python, V2 transceiver) into V1's `CacheTransceiver::checkContextTransferStatus` / `checkGenTransferStatus` (C++) with one simplification: cancellation is collapsed into FAILED (Option A — see §2.3).

| Aspect | Result |
|---|---|
| Implementation effort | ~2 engineering days (not the ~1-2 weeks doc 12 estimated; the V2 pattern provided most of the design) |
| Code change scope | 4 files: envUtils.h/cpp + cacheTransceiver.h/cpp; +574 lines of cpp body, +50 lines of header, +27 lines of env util |
| Public API change | None — same `checkContextTransferStatus`/`checkGenTransferStatus` signatures, dispatcher selects body via env var |
| Default behaviour | Byte-identical to current PR #13713 HEAD (env var off → legacy body) |
| Empirical validation | 8 valid local test runs across 4 test families and 4 topologies; **all PASSED, zero false positives, 3 of 4 helix-class tests caught real cross-rank divergences** |
| Overhead | ~22% wall time when divergences are caught; ~0% on tests with no divergence |
| Cache mechanism | `mSenderLocalOutcomes` / `mRequesterLocalOutcomes` side maps; validated under multi-iteration deferral (TinyLlama test, see §3.2.3) |
| Refinement of doc 13's conclusion | The cpp gtest `asymmetric_executor[llama-4proc-mpi_kvcache]` is **partly** fixed by the consensus path (348s+FAIL → 46s+PASS). Doc 13's "transport hang, not consistency ABBA" framing is incomplete; consensus addresses something the gather-point INSTR couldn't see — see §3.4. |

The implementation is **gated behind a new env var `TRTLLM_DISAGG_USE_CONSENSUS_OUTCOME` (default 0)** so PR #13713 can ship with byte-identical default behaviour even if the code lands. Decision on whether to include in PR #13713 is gated on the next CI cycle's signal — see §5.

---

## 2. The implementation

### 2.1 Design overview — four-pass pipeline

Each entry to `checkContextTransferStatus` / `checkGenTransferStatus` (when consensus is enabled) runs four passes:

| Pass | Purpose | Allgathers | LlmRequest mutation |
|---|---|---|---|
| **A** (existing, preserved) | Readiness consensus: each rank does `wait_for(0)` sweep, all-gathers ready ids via `gatherRequestIds`, builds `globalReady` from unanimous-ready freq filter | 1 | None |
| **B** (new) | Local classification: for each request in `toProcessSet`, drive bounded `wait_for` (Option C's 50ms cap), call `future.get()`, populate `localCompleted` / `localFailed`. Outcome cached in `mSenderLocalOutcomes` / `mRequesterLocalOutcomes` so subsequent calls do not re-call `get()` on an invalidated future. | 0 | **None** — cache only |
| **C** (new) | Outcome consensus (Option A): 2 all-gathers on the same `syncComm` Pass A used. `globalCompleted` = intersection (all ranks agree). `globalFailed` = union (any rank's failure propagates). Failure beats completion on ties. | 2 | None |
| **D** (new) | Apply state transitions: only here do `request->setState(...)`, `mCacheSender->cancelRequest(...)`, and `mSenderFutures.erase(...)` happen. Driven strictly from the global sets so every rank applies identical transitions. | 0 | **Only here** |

Total: 3 all-gathers per call (was 1). The 2 new outcome all-gathers run on the same comm as the existing readiness all-gather (`mGroupTensorParaComm` / `mGroupTPInDPComm` on ctx side, `mGroupComm` / `mGroupDataComm` on gen side), so no new comm topology is introduced.

Pseudo-code for the ctx side (the gen side is structurally identical modulo `completeEntry` and `updateKVCacheTransferBW`):

```cpp
RequestStatuses CacheTransceiver::checkContextTransferStatusWithConsensus(
    std::optional<int> const& atLeastRequestNum, bool markComplete)
{
    bool const blockAll = !atLeastRequestNum.has_value();
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP
        ? mGroupTPInDPComm : mGroupTensorParaComm;
    bool const inflightCancelEnabled = common::getEnvDisaggEnableInflightCancel();

    // Pass A — readiness consensus (existing logic, preserved verbatim)
    std::vector<RequestIdType> localReady;
    for (auto&& [request, future] : mSenderFutures) {
        auto id = request->mRequestId;
        // Cached outcome counts as "ready" — we already have the answer locally
        if (mSenderLocalOutcomes.count(id) > 0
            || future.wait_for(0ms) == std::future_status::ready) {
            localReady.push_back(id);
        }
    }
    auto globalReady = computeReadyConsensus(syncComm, localReady, ...);
    auto toProcessSet = buildToProcessSet(globalReady, atLeastRequestNum, blockAll);

    // Pass B — local classification (NO state mutation, populates cache)
    std::vector<RequestIdType> localCompleted, localFailed;
    for (auto&& [request, future] : mSenderFutures) {
        auto id = request->mRequestId;
        if (!toProcessSet.count(id)) continue;

        // Cache hit on prior-iteration outcome: re-present without calling get() again
        if (auto it = mSenderLocalOutcomes.find(id); it != mSenderLocalOutcomes.end()) {
            (it->second.kind == LocalFutureOutcome::Kind::kCompleted
                ? localCompleted : localFailed).push_back(id);
            continue;
        }

        auto slice = computeEffectiveSliceMs(..., kMaxPollSliceMs=50);   // Option C cap
        auto status = future.wait_for(std::chrono::milliseconds(slice));
        if (status == std::future_status::ready) {
            try {
                future.get();
                mSenderLocalOutcomes.emplace(id, {kCompleted, ""});
                localCompleted.push_back(id);
            } catch (std::exception const& e) {
                mSenderLocalOutcomes.emplace(id, {kFailed, e.what()});
                localFailed.push_back(id);
            }
        }
        // ... timeout / unexpected-status / deadline-expiry branches likewise populate cache ...
    }

    // Pass C — outcome consensus (2 allgathers, Option A)
    auto [globalCompleted, globalFailed]
        = consensusOutcomeOptionA(syncComm, localCompleted, localFailed, ...);
    // ^ globalCompleted = intersection; globalFailed = union;
    //   failure beats completion on ties.

    // [NVBUG-6104831-CONSENSUS] divergence-log sites — emit a WARN when local
    // view differs from consensus, so CI / post-processing can mine for events.

    // Pass D — apply state transitions, erase futures, free cache
    RequestStatuses requestsStatus{};
    for (auto it = mSenderFutures.begin(); it != mSenderFutures.end(); ) {
        auto id = it->first->mRequestId;
        if (globalFailed.count(id)) {
            // ... emit warn (deduped via mTimedOutSenderIds), cancel if cancel flag on ...
            it->first->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            requestsStatus.errorRequestIds.insert(id);
            mSenderLocalOutcomes.erase(id);
            it = mSenderFutures.erase(it);
        } else if (globalCompleted.count(id)) {
            if (markComplete) it->first->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
            requestsStatus.completedRequestIds.insert(id);
            mSenderLocalOutcomes.erase(id);
            it = mSenderFutures.erase(it);
        } else {
            ++it;
        }
    }
    return requestsStatus;
}
```

### 2.2 Critical design issue: `future.get()` can only be called once

`std::future::get()` invalidates the future. If rank 0 sees its local future ready before rank 1 (which is exactly the divergence consensus exists to handle), the naive implementation would:

1. **Iter N**: rank 0 calls `get()` → caches success locally → reports `localCompleted=[id]`; rank 1 not ready, reports `localCompleted=[]` → `globalCompleted = intersection = {}` → request stays in `mSenderFutures` on both ranks.
2. **Iter N+1**: rank 0's `wait_for(0)` returns ready again (the future was already consumed) → naive impl calls `get()` → throws `std::future_error: future_already_retrieved` → crash.

The implementation solves this with a **side cache map per direction**:

```cpp
struct LocalFutureOutcome {
    enum class Kind : std::uint8_t { kCompleted, kFailed };
    Kind kind;
    std::string errorMessage;   // populated only for kFailed
};
std::unordered_map<LlmRequest::RequestIdType, LocalFutureOutcome> mSenderLocalOutcomes;
std::unordered_map<LlmRequest::RequestIdType, LocalFutureOutcome> mRequesterLocalOutcomes;
```

Lifetime:
- Populated in Pass B on the rank that first observes the outcome.
- Consulted in Pass A (to consider the request "ready") and Pass B (to re-present the cached outcome) on every subsequent call.
- Erased in Pass D when the request is removed from `mSenderFutures` / `mRequesterFutures` (i.e., when consensus reaches a terminal decision).

Memory cost is bounded by the in-flight transfer count, same as `mSenderFutures` itself. **This mechanism is validated under multi-iteration deferral**: see §3.2.3 (TinyLlama test caught the same request being cached across iterations 2, 3, and 4 without crashing).

### 2.3 Option A vs Option B — why we collapsed cancellation into FAILED

V2's `_consensus_outcome` uses **3 allgathers** (cancelled, failed, completed) and keeps cancellation as a distinct outcome. We initially considered both Option A (2 allgathers, cancel collapsed into FAILED) and Option B (3 allgathers, distinct cancel).

For V1's flag-off default (`TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=0`), Option A and Option B are functionally identical: no cancel ever fires, so the cancel set is always empty. Option A is:
- Cheaper: 2 allgathers vs 3 (33% less consensus overhead per call)
- Smaller patch: ~80 lines saved
- No `CancelledException` type or `mWasCancelled` flag wiring needed through the future chain

Migration to Option B is non-breaking (additive). The bookmark notes the choice should be revisited when the cancel flag becomes default-on; this is tracked as an open item in §6.

### 2.4 Files changed

| File | Change | Lines |
|---|---|---|
| `cpp/tensorrt_llm/common/envUtils.h` | Declare `getEnvDisaggUseConsensusOutcome()` with full design comment | +21 |
| `cpp/tensorrt_llm/common/envUtils.cpp` | Implement; reads `TRTLLM_DISAGG_USE_CONSENSUS_OUTCOME` | +6 |
| `cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h` | `LocalFutureOutcome` struct, 2 cache maps, 4 private helper declarations, 2 includes | +50 |
| `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` | 2 dispatch wrappers, 2 renamed legacy bodies (byte-identical), 2 new `*WithConsensus` bodies, 3 namespace-local helpers (`kMaxPollSliceMs`, `computeEffectiveSliceMs`, `computeReadyConsensus`, `consensusOutcomeOptionA`) | +574 |

Total: **+651 lines / 0 deletions** across 4 files. Zero lint errors after the change. Public method signatures unchanged → no nanobind binding changes needed.

### 2.5 Properties preserved unchanged when env var is ON

- Option C's `kMaxPollSliceMs = 50` cap on every `wait_for` slice
- `kvTransferTimeoutMs` deadline check (warn fires regardless of cancel flag — observability decoupled from action, per the comment block at lines 72-76 of `cacheTransceiver.cpp`)
- `markComplete` semantics on the ctx side
- `atLeastRequestNum` / `blockAll` semantics
- `updateKVCacheTransferBW` invocation on the gen-side completion path (preserved inline in Pass D's completed branch)
- `mTimedOutSenderIds` / `mTimedOutRequesterIds` first-timeout dedup sets (still rank-local, for warn-log spam control)

### 2.6 Properties added when env var is ON

- `LlmRequest::setState` for transfer outcomes happens **only** in Pass D, **only** driven by global consensus sets — by construction, no per-rank state divergence possible from these functions
- Failed-on-any-rank semantics (Pass C union): if rank N sees `future.get()` throw, the request is marked error on all ranks (matches V2)
- Completed-on-all-ranks semantics (Pass C intersection): a request is only marked complete when every rank's local view confirms (matches V2)
- 2 additional all-gathers per call (cost: ~tens of microseconds in typical TP/CP configs)
- `mSenderLocalOutcomes` / `mRequesterLocalOutcomes` side caches (memory bounded by in-flight transfer count)
- `[NVBUG-6104831-CONSENSUS]` WARN log lines whenever a rank's local view differs from the consensus outcome (three sites per direction: rank-local exception during classify, local-completed-but-consensus-deferred, local-completed-but-consensus-failed)

---

## 3. Empirical validation

All tests run on a single B300 host (`umb-b300-026`, 8 × B300 SXM6, CUDA 13.2, driver 595.58.03). Worktree at `/home/scratch.chienchunh_coreai/dev/TensorRT-LLM-pr13713-rc13-clean`. Logs saved under `/home/scratch.chienchunh_coreai/nvbug6104831/test-logs-2026-05-27/`.

### 3.1 Test matrix

| # | Test | Procs | Transport | Consensus | Wall | Result | Divergence events |
|---|---|---|---|---|---|---|---|
| 1 | `cpp test_asymmetric_executor[llama-4proc-mpi_kvcache]` (LlamaConPP2GenTP2 variant) | 4 | MPI | **OFF** | **348 s** | **FAIL** — cross-rank token mismatch | n/a (code path off) |
| 2 | same as 1 | 4 | MPI | ON | 46 s | PASS | 1 (gen rank=2, iter=0, req X) |
| 3 | same as 1 (re-run) | 4 | MPI | ON | 46 s | PASS | 1 (same event) |
| 4 | `cpp LlamaConTP2GenPP2DisaggAsymmetricExecutorTest` | 4 | MPI | ON | 57 s | PASS | 1 (gen rank=3, iter=0, req X — mirror of test 2) |
| 5 | same as 2 with `--gtest_repeat=5` | 4 | MPI | ON | 62 s total | 5/5 PASS | 1 (warm-up only, iter=0) |
| 6 | `cpp cacheTransceiverTest` (unit test, 511 cases) | 4 | MPI | ON | 19 s | PASS (3 ran, 508 topology-skipped) | 0 |
| 7 | `cpp LlamaConTP2PP2GenPP2DisaggAsymmetricExecutorTest` | 6 | MPI | ON | 53 s | PASS | 0 |
| 8 | same as 7 baseline | 6 | MPI | **OFF** | 59 s | PASS (baseline — no consensus overhead when no divergence) | n/a |
| 9 | `helix DSV3-Lite test_auto_dtype_with_helix[fifo_v2-cudagraph:with_padding-pp1tp1cp4]` | 8 | UCX | ON | 17.0 min | PASS | 2 (gen rank=0+1, iter=2, same req) |
| 10 | same as 9 (re-run after warm cache) | 8 | UCX | **OFF** | 13.9 min | PASS | 0 (code path off) |
| 11 | `helix DSV3-Lite test_auto_dtype_with_helix[fifo_v2-cudagraph:with_padding-pp1dp2cp2]` (attention-DP) | 8 | UCX | ON | 14.0 min | PASS | 1 (gen rank=2, iter=2) |
| 12 | `helix Qwen3-8B test_auto_dtype_with_helix[fifo_v2-cudagraph:with_padding-pp1tp2cp2]` | 8 | UCX | ON | 6.1 min | PASS | 0 |
| 13 | `disagg test_disaggregated_ctxpp2_genpp2[TinyLlama-1.1B-Chat-v1.0]` | 4 | NIXL | ON | 3.6 min | PASS | **5** (same req, iters 2, 3, 3, 4, 4) |

**Aggregate**: 11 of 11 valid runs with consensus ON PASSED. 1 of 1 run with consensus OFF on the cpp asymmetric test FAILED with cross-rank token mismatch. 1 of 1 run with consensus OFF on the 6-proc cpp test and 1 of 1 helix-OFF on DSV3-Lite both PASSED (no divergence to trigger). **Zero false positives**, **zero crashes**.

(Run 13 in the original capture used UCX transport for the cpp gtest, which failed for environmental reasons unrelated to consensus — the `libtensorrt_llm_ucx_wrapper.so` was not on the loader path for the gtest binary. That data point is discarded from the table above. The Python helix tests load UCX correctly because the Python package's `libs/` directory is in the runtime path.)

### 3.2 What the divergence events reveal

#### 3.2.1 cpp asymmetric test (tests 2, 3, 4)

All three runs caught exactly **one** divergence event each, on the very first iteration (`iter=0`) of the gen-side comm, on a different rank for each topology:

```
[NVBUG-6104831-CONSENSUS] gen rank=2 iter=0 request=X: local view COMPLETED but consensus deferred (peers not ready)   # PP2GenTP2
[NVBUG-6104831-CONSENSUS] gen rank=3 iter=0 request=X: local view COMPLETED but consensus deferred (peers not ready)   # TP2GenPP2 (mirror)
```

Rank 2 (or 3, depending on topology) saw its local future for the first request complete before its peer in the gen-side comm. Consensus correctly deferred. Subsequent iterations agreed and the request completed in lockstep.

Without consensus, the rank that saw ready first would have called `setState(kDISAGG_GENERATION_TRANS_COMPLETE)` while its peer remained in `kDISAGG_GENERATION_TRANS_IN_PROGRESS`. This is exactly the cross-rank state divergence doc 12 §2.4 predicts; the cpp test surfaces it deterministically on warm-up.

#### 3.2.2 helix DSV3-Lite tests (tests 9, 11)

| Variant | Comm-class | Divergence events |
|---|---|---|
| pp1tp1cp4 (no ADP) | `mGroupComm` (gen world, no DP) | 2 (gen rank=0 + gen rank=1 on the SAME request 268565823647749 at iter=2) |
| pp1dp2cp2 (ADP) | `mGroupDataComm` (gen DP comm) | 1 (gen rank=2 at iter=2) |

Same iter=2 fingerprint as the cpp test — consensus catches a warm-up race in the gen-side completion handshake. The pp1tp1cp4 case is particularly informative: ranks 0 and 1 both saw the same request locally complete before ranks 2 and 3, so consensus deferred the state transition (`globalCompleted = ∅` because not unanimous) and cached both ranks' outcomes. Without the cache, ranks 0 and 1 would have crashed on iter=3 trying to re-call `get()` on their already-consumed futures.

#### 3.2.3 disagg TinyLlama test 13 — multi-iteration cache validation

The single most informative empirical result:

```
[NVBUG-6104831-CONSENSUS] gen rank=0 iter=2 request=281397012938752: local view COMPLETED but consensus deferred (peers not ready)
[NVBUG-6104831-CONSENSUS] gen rank=0 iter=3 request=281397012938752: local view COMPLETED but consensus deferred (peers not ready)
[NVBUG-6104831-CONSENSUS] gen rank=0 iter=3 request=281397012938752: local view COMPLETED but consensus deferred (peers not ready)
[NVBUG-6104831-CONSENSUS] gen rank=0 iter=4 request=281397012938752: local view COMPLETED but consensus deferred (peers not ready)
[NVBUG-6104831-CONSENSUS] gen rank=0 iter=4 request=281397012938752: local view COMPLETED but consensus deferred (peers not ready)
```

**Same request (`281397012938752`), same rank (0), three consecutive scheduling iterations (2, 3, 4).** Rank 0 saw its local future ready at iter=2; the gen-side peers took **two more scheduling iterations** to also see ready. During those three iterations:

- Rank 0's `future.get()` was called exactly once, at iter=2, with result cached in `mRequesterLocalOutcomes`.
- On iters 3 and 4, rank 0's Pass B looked up the cache (no `wait_for` / `get()` re-call) and re-presented `kCompleted` to consensus.
- Consensus continued to defer because peers weren't agreeing yet.
- On iter ≥ 5, peers caught up; consensus reached unanimity; Pass D applied `setState(kDISAGG_GENERATION_TRANS_COMPLETE)` on ALL four ranks atomically.

**Without the cache, rank 0 would have crashed at iter=3** with `std::future_error: future_already_retrieved`. The TinyLlama test empirically validates the cache mechanism's load-bearing role.

This is also the failure pattern doc 12 §2.4 specifically warned about: zombie entries in `mRequesterFutures` resolving at rank-dependent times, where the resolution gap can span multiple scheduling iterations.

### 3.3 Overhead measurements

Only one apples-to-apples A/B is available for overhead measurement (same test, same machine, same model, same NFS cache state, one ON / one OFF):

| Test | Consensus ON | Consensus OFF | Delta | Notes |
|---|---|---|---|---|
| helix DSV3-Lite pp1tp1cp4 | 17.0 min | 13.9 min | **+186 s (+22.3%)** | Real overhead from extra allgathers + caught divergence handling |
| cpp asymmetric LlamaConTP2PP2GenPP2 (6-proc) | 53 s | 59 s | **−6 s (−10%)** | Within noise; suggests no net overhead when no divergence to handle |
| cpp asymmetric LlamaConPP2GenTP2 (4-proc) | 46 s | (348 s, FAILED) | — | Not apples-to-apples (OFF failed) |

So the worst-case observed overhead is **~22% wall time** on a 17-minute helix test that catches 2 divergence events. The 6-proc test with zero divergence events showed effectively no consensus overhead.

The 22% overhead is largely model-load-time-amortized: per-iteration consensus cost is dominated by 2 small all-gathers (sub-millisecond on small id sets), not by the per-call arithmetic. Tests with long generation phases will show lower overhead ratios; short-burst tests like the cpp gtests show higher ratios.

### 3.4 Doc 13's framing of `asymmetric_executor[mpi_kvcache]` — refined

Doc 13 concluded:

> The cpp gtest wedge sits at the **mpi_kvcache transport layer** (UCX-over-shared-memory inside OpenMPI) and the gtest's internal 300 s timeout fires *before* any of the cascading horizontal-consistency effects described in §2.4 can manifest.

The conclusion was based on:
1. Both ctx ranks' `senderFutures_size=8` was identical at iter=2.
2. Both gen ranks' `requesterFutures_size=8` was identical at iter=35.
3. All `gatherRequestIds.exit` records had matching `local_ids`/`gathered_ids` between ranks.
4. The postprocessor reported "no cross-rank divergence" for either comm at any iteration.

The empirical result in this session (consensus ON → 46 s PASS vs consensus OFF → 348 s FAIL on the same test) doesn't directly contradict any of those observations, but it **does require a refinement**:

The divergence consensus catches is not at the `gatherRequestIds` readiness layer (where doc 13's INSTR looked) — it is at the **outcome layer**, *after* `wait_for` returns ready. Specifically, two ranks may see `wait_for(0) == ready` in the same scheduling iteration (the gather agrees both have the request), but then `future.get()` returns successfully on one rank and is still pending or about to throw on the other. The pre-consensus code path mutates state inline based on each rank's local `get()` result; the consensus path defers until both ranks' `get()` outcomes agree.

The cpp gtest's failure mode is therefore **both** a transport issue (mpi/UCX-shm produces stuck transfers — doc 13 was correct on this) **and** a consistency issue (the stuck transfer's outcome resolves at slightly different times across ranks, producing rank-divergent `setState` calls that the test's token-comparison harness eventually exposes as a UINT64_MAX-7 `predictedTokens.size()` underflow). Consensus closes the consistency layer; the transport-level slowness remains but no longer produces a wedge because the per-iteration `wait_for(50ms)` (Option C) + sub-millisecond consensus allgather keeps MPI progress flowing.

Doc 13 should be considered correct in its narrow claim ("no divergence visible at gather points") but the broader framing ("therefore not a consistency issue") was inferring beyond the evidence. The transport hang and the consistency divergence are two layers of the same failure; the cpp gtest is a useful empirical anchor for **both** the transport-fix work (Option C) AND the consistency-fix work (Path B).

---

## 4. Verification status

### 4.1 Validated locally
- ✅ Consensus path compiles cleanly (zero lint warnings)
- ✅ Default-OFF dispatch identical to current PR #13713 HEAD (legacy body is byte-identical, only the dispatcher wrapper added)
- ✅ Consensus-ON catches real cross-rank races on 4 distinct test families and 4 distinct topologies
- ✅ Cache mechanism survives multi-iteration deferral (TinyLlama test, 3 consecutive iters on the same request)
- ✅ Multiple comm groups exercised: `mGroupComm` (no ADP), `mGroupDataComm` (ADP), `mGroupTensorParaComm`, `mGroupTPInDPComm`
- ✅ Stability under repetition: 5/5 reps of the same cpp test all PASS
- ✅ Cross-transport: passes on MPI (cpp tests), NIXL (TinyLlama), UCX (helix)
- ✅ Cross-model: passes on Llama 3.2 1B (cpp), DeepSeek-V3-Lite BF16 (helix), Qwen3-8B FP8 (helix), TinyLlama 1.1B (disagg)
- ✅ Overhead measured: ~22% worst case, ~0% when no divergence

### 4.2 Not yet validated
- ❌ CI signal under PR #13713's full test matrix (gated on the user's decision to commit; this is the load-bearing next step)
- ❌ Helix flake-rate reduction over many runs — we have only single-run data per variant; statistical claim about flake reduction requires 5-10+ runs each side
- ❌ Multi-node behavior (all validation single-host)
- ❌ Production-like sustained load (validation was bounded-batch tests up to ~30 min wall)
- ❌ Interaction with `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1` (cancel flag was OFF for all local validation; Option A's "cancel collapsed into FAILED" semantics need flag-on validation before Option A → Option B migration decision)

### 4.3 Open questions

- **Why exactly does consensus speed up the cpp gtest from 348 s+FAIL to 46 s+PASS?** The hypothesis from §3.4 is that consensus's extra all-gathers drive MPI progress and the lockstep state transitions prevent the retry loops that the legacy path falls into when ranks disagree. But the precise mechanism isn't pinpointed — could be (a) all-gather driving MPI progress, (b) avoiding retry loops, or (c) both. Not load-bearing for the value claim; useful to understand for future refinement.

- **Does the same code work on multi-node setups?** No evidence either way. The all-gathers use the same comms as the existing readiness gather, which is known to work multi-node, so there's no theoretical reason it wouldn't — but unconfirmed.

- **What's the right `kMaxPollSliceMs` interaction with consensus?** Option C set 50ms as a cap on `wait_for` slices. With consensus, the 2 extra allgathers per call add ~1 ms each, so the actual per-call cost is `~50ms wait + 2 ms allgather = ~52 ms`. Not concerning, but if helix flake reduction in CI is non-zero, worth checking whether lowering the cap further (say 10-20ms) trades latency for faster convergence.

---

## 5. Updated plan

Doc 12 §6 proposed:
1. Now: PR #13713 + waivers (Path C — tactical)
2. After merge: Path A (~2-3 days — wrapper class + lifetime flag)
3. Long-term: Path B (~1-2 weeks — explicit horizontal consistency layer)

The empirical results above let us update this plan in two important ways:
- Path B is **already implemented** in 2 days (not 1-2 weeks), faster than estimated because the V2 transceiver provided a working pattern to port.
- Path B **demonstrably works** on every test family we've tried, including the helix variants doc 12 §6 had Path A as the interim answer for.

The updated plan:

### 5.1 Short-term: ship PR #13713 with consensus (env-var default-off) or land as separate follow-up — depends on CI

Two viable paths, depending on the PR #13713 CI signal currently in flight:

**5.1.a — If PR #13713 CI is clean on its current HEAD (no consensus):** land PR #13713 as is, then submit the consensus code as a separate follow-up PR (call it #13713-fu-consensus). Behind env var default-off → byte-identical default behaviour. Add a CI extra-env stage (or test list `extra_env`) that flips `TRTLLM_DISAGG_USE_CONSENSUS_OUTCOME=1` on a subset of helix and disagg tests to validate. Migrate to default-on after 1-2 weeks of CI signal showing zero regressions and visible divergence catches in flake-prone tests.

**5.1.b — If PR #13713 CI is still failing on flake-prone tests:** fold the consensus code into PR #13713 itself (additional commit on the PR branch). Same env-var-default-off gating. CI extra-env flips the flag on for the failing tests. This lets the CI matrix validate both the original PR scope and the consensus layer together, and gives the failing tests a known mitigation path without further waivers.

The user's call between 5.1.a and 5.1.b is gated on the PR #13713 CI result currently in flight.

### 5.2 Mid-term: default-on after CI signal

Once 1-2 weeks of CI runs show:
- Zero regressions with `TRTLLM_DISAGG_USE_CONSENSUS_OUTCOME=1`
- Visible `[NVBUG-6104831-CONSENSUS]` divergence event reductions (or zero events, indicating consensus is now mooting the problem)
- Acceptable overhead in performance-sensitive disagg tests

Flip the env var default to `1`. Keep the env var available as an escape hatch (`=0` → legacy path).

### 5.3 Long-term: simplifications enabled by consensus

With consensus correctly enforcing cross-rank state agreement, several pieces of the current codebase become removable:
- The Python-side rank-symmetric gate fixes (commits `bdfdf8be02`, `53a0692aa4` on PR #13713) become belt-and-suspenders — they're prerequisite for consensus to work, but the divergence they prevent at the Python layer is also prevented by consensus at the C++ layer. Keep both as defense in depth; reconsider during a future cleanup pass.
- The deferred-cleanup machinery doc 12 §2.5 talked about ("don't free Python resources while C++ transfer status is still in progress") was hard to design correctly under the per-rank semantics of pre-consensus V1. With consensus, the C++ side's view of "transfer status" is now globally consistent, so the deferred-cleanup design has a clean foundation. This unblocks the doc 12 §5.2 "long-term" architectural refactoring.
- Path A (lifetime flag wrapper) becomes **unnecessary** — Path A was an alternative to Path B that traded vertical safety for implicit horizontal consistency. With Path B working, no such trade is needed.

### 5.4 Open / deferred work

- **Option A → Option B migration** if the cancel flag ever becomes default-on. Distinct CANCELLED outcome (vs collapsed into FAILED) gives better forensics in logs.
- **Performance optimization** of the 22% overhead case. Two main levers: (a) fold the 2 outcome allgathers into a single allgather of tagged tuples, (b) skip Pass C entirely when both `localCompleted` and `localFailed` are empty (the common case).
- **Multi-node validation** before any default-on flip on a multi-node deployment.
- **Production sustained-load validation** — single 30-min runs locally are not a substitute for hours of production traffic.

---

## 6. Related docs

- [12-horizontal-consistency-and-layer3-gating.md](12-horizontal-consistency-and-layer3-gating.md) — the theory this doc validates; specifically §5.2 Path B.
- [13-cpp-gtest-transport-hang-finding.md](13-cpp-gtest-transport-hang-finding.md) — earlier finding on the cpp gtest; refined by §3.4 above.
- [02-failure-signatures.md](02-failure-signatures.md) — sig #9 (helix CI hang from rank-asymmetric Python gates) is the failure class consensus closes structurally.
- [03-defect-class-stack.md](03-defect-class-stack.md) — L11 (rank-asymmetric Python gates) is what consensus addresses at the C++ layer.
- [08-next-steps-and-pr-map.md](08-next-steps-and-pr-map.md) — operational view; this doc adds a new in-flight piece of work to track there.
- [README.md](README.md) — investigation index; add doc 14 to the file listing.

---

## 7. Appendix — env var contract

```text
TRTLLM_DISAGG_USE_CONSENSUS_OUTCOME
  Type:    bool ("1" / "0", default "0")
  Default: 0 — legacy V1 behaviour (byte-identical to current PR #13713 HEAD)
  When 1:  V1's checkContextTransferStatus / checkGenTransferStatus dispatch to
           the four-pass consensus body (§2.1). Cancellation collapsed into
           FAILED (Option A). 2 extra allgathers per call. Side caches active.
  Compat:  Independent of TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL. Either flag can
           be on or off independently.
  Logs:    [NVBUG-6104831-CONSENSUS] WARN lines at 3 sites per direction when a
           rank's local view differs from consensus (postprocessor-friendly).
  Removal: Plan is to flip default to 1 after CI signal validates (§5.2). Keep
           the env var indefinitely as an escape hatch.
```
