# 18 — PR #14746 prior art and the two-layer cross-rank-consistency gap on V1

**Status:** Discussion summary (2026-06-01). Captures the gap analysis surfaced when reviewing an upstream parallel effort.
**Trigger:** Review of <https://github.com/NVIDIA/TensorRT-LLM/pull/14746> (Shixiaowei02, *"Make disagg timeout cancellation rank-consistent"*, targeting `feat/deepseek_v4`).
**Closes / refines:** [17 §6](17-tier1-ablation-result.md#6-implication-for-the-follow-up-cancellation-pr) — sharpens the design constraints on the cancellation follow-up.

## 1. What PR #14746 does (one-line scope)

Adds `PyExecutor._sync_kv_transfer_timed_out_flags()` — a per-iter `dist.allgather(local_ids)` + **union** that promotes the per-rank wall-clock `py_kv_transfer_timed_out` detection into a rank-consistent flag set, then routes the (renamed) `warn_if_kv_transfer_timed_out` inner function down to *diagnostic-only*. Called at lockstep sites in the executor loop (`handle_executed_batches` and the main loop) every multi-rank iter.

| Aspect | Pre-PR-#14746 | Post-PR-#14746 |
|---|---|---|
| Wall-clock detection | per-rank in `_check_kv_transfer_timeout` | unchanged (per-rank) |
| Flag-setting | per-rank in `flag_if_kv_transfer_timed_out` → divergent across ranks | cross-rank `dist.allgather` → union → identical on all ranks |
| Cancellation sites (line 4006 ctx, line 4488 gen in PR branch) | read divergent flag → divergent cancel decisions | read identical flag → identical cancel decisions *(but see §3 below)* |
| ADP-only `tp_allgather(bool)` mechanism | already present (narrow: "did any rank time out?", not "which?") | retained, now redundant with the new ID-level allgather; cleanup opportunity |
| Non-ADP multi-rank | no consensus at all | gets the new allgather |

So PR #14746 is **not the first attempt at timeout consensus** — ADP had a narrow bool-only mechanism. It's the first attempt at ID-level union, extended to all multi-rank configs.

## 2. The two-layer cross-rank-consistency model

The PR #14746 review surfaced a useful decomposition: cross-rank-consistent cancellation needs **two** layers of coordination, not one.

| Layer | Question | PR #14746 status | V2 status (today) | V1 status (today) |
|---|---|---|---|---|
| **L1 — flag/intent consensus** | Do all ranks agree on *which* requests are timed-out / to-be-cancelled? | **Fixed** for the timeout flag | Already consistent (this PR closes the last gap) | Inherits the fix (`_sync_kv_transfer_timed_out_flags` runs at PyExecutor level, above the transceiver) |
| **L2 — state-transition consensus** | Do all ranks make identical state transitions in response to a consistent flag? | **Not addressed** | `KvCacheTransceiverV2._ctx/_gen_consensus_outcome` already handles this | **No analog exists** — each rank reads its own `is_cancelled = cancel_request(req)` boolean and transitions state locally |

The asymmetry between V2 and V1 at L2 is the key gap. On V2, even before PR #14746, every per-rid state transition flowed through `_consensus_outcome`; the timeout flag was the one dangling input. PR #14746 plugs that input, and the V2 stack becomes end-to-end consistent. On V1, plugging the L1 input still leaves L2 open: `cancel_request` is a rank-local call whose return value depends on per-rank race state (was the request currently being sent on this rank? was it already in `mReadyResponses`?). Different ranks see different outcomes → different `request.state` assignments, different `_end_transfer_and_maybe_terminate` calls, different `timed_out_requests` lists → divergent state.

Doc 14 ported the V2 `_consensus_outcome` pattern into V1 as an opt-in env-gated path (`TRTLLM_DISAGG_USE_CONSENSUS_OUTCOME`). That work covers L2 for the normal completion / failure path. **It does not cover the cancellation L2** — cancellation is collapsed into FAILED in doc 14 §2.3 (Option A), which works when the only cancellation trigger is internal failure, but not when an external timeout decision needs to be applied identically across ranks.

## 3. Concrete L2 evidence — V1 vs V2, with line citations

The L1/L2 distinction is grounded in code, not just framing. Below is the side-by-side that shows where each path applies (or doesn't) a consensus collective between timeout-detection and state-transition.

### V2 — consensus on the cancellation *outcome* before state changes

`tensorrt_llm/_torch/disaggregation/transceiver.py:280-298`:

```python
def _consensus_outcome(self, to_process, cancelled, failed, completed, allgather, need_sync):
    # CANCELLED/FAILED on any rank → global; COMPLETED only when ALL ranks agree.
    all_c    = self._allgather_or_passthrough(cancelled, allgather, need_sync)   # ← allgather of locally-CANCELLED rids
    all_f    = self._allgather_or_passthrough(failed,    allgather, need_sync)
    all_done = self._allgather_or_passthrough(completed, allgather, need_sync)
    n = len(all_c)
    global_cancelled = self._union(all_c)                                        # ← UNION across ranks
    global_failed    = self._union(all_f)
    global_completed = self._intersection(all_done, n)                           # ← INTERSECTION across ranks
    new_cancelled = [rid for rid in to_process if rid in global_cancelled]
    ...
```

Then `transceiver.py:478` (`_ctx_consensus_outcome`) and `:530` (`_gen_consensus_outcome`) apply state transitions *strictly from these global sets*. Even when individual ranks see different `cancel_request` results, V2 reconciles into a rank-consistent state.

### V1 — no consensus between detection and state change

`cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp:662-700` (gated by `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL`):

```cpp
if (kvTransferTimeoutMs.has_value()) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        LlmRequest::getSteadyClockNow() - request->getKvCacheTransferStart());
    auto elapsedMs = static_cast<long>(elapsed.count());
    if (elapsedMs > kvTransferTimeoutMs.value()) {                               // ← per-rank steady-clock check
        // ... WARN ...
        if (inflightCancelEnabled) {
            mCacheSender->cancelRequest(*request);                               // ← per-rank cancel
            request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);             // ← per-rank state transition
            requestsStatus.errorRequestIds.insert(request->mRequestId);
            mTimedOutSenderIds.erase(request->mRequestId);
            it = mSenderFutures.erase(it);
            continue;
        }
    }
}
```

No allgather between line 667 (detection) and line 690 (`setState(kDISAGG_TRANS_ERROR)`). The same function does perform a readiness allgather earlier (`gatherRequestIds(syncComm, contextCompleteRequestIds)` at line ~610), but the timeout-cancellation path at 662-700 is intentionally outside that consensus.

The Python layer above this has matching per-rank-divergent sites at `tensorrt_llm/_torch/pyexecutor/py_executor.py:4332` (ctx) and `:4971` (gen) using per-rank dedup sets `_disagg_timed_out_ctx_cancelled_ids` / `_disagg_timed_out_gen_cancelled_ids`. The Python sites defer the actual state change to C++ (per the "deferred cleanup contract" comment in `py_executor.py:4977`), so the *load-bearing* state-divergence site is the C++ `setState(kDISAGG_TRANS_ERROR)` above.

### Side-by-side

| Layer | Detection | Pre-action consensus? | Action |
|---|---|---|---|
| **V2** Python transceiver | per-rank wall-clock + state inspection | **`_consensus_outcome`** allgather UNION of cancelled rids (`transceiver.py:280-298`) | State transition driven from `global_cancelled` — rank-consistent |
| **V1** C++ CacheTransceiver | per-rank `steady_clock` against `kvTransferTimeoutMs` (`cacheTransceiver.cpp:667`) | **None** between detection and action | `cancelRequest` + `setState(kDISAGG_TRANS_ERROR)` per rank (`cacheTransceiver.cpp:689-690`) |

This is the concrete code answer to "is V1 state transition synced across ranks": **no**. V2 has an explicit consensus primitive named `_consensus_outcome` doing exactly that work. V1 has no equivalent at the timeout-cancellation site.

### PR #14746 against this picture

PR #14746 adds an L1 consensus (`_sync_kv_transfer_timed_out_flags` allgather-unions the *flag*) at the Python layer. On the V1 path that flag is consumed by `py_executor.py:4332` / `:4971` cancellation sites, which then call into the C++ `CacheSender::cancelRequest`. The C++ state-transition at `cacheTransceiver.cpp:690` is unchanged by this PR.

So PR #14746 closes the divergence at *one* of the three V1 sites:

| V1 divergence site | Pre-PR-#14746 | Post-PR-#14746 |
|---|---|---|
| Python `py_kv_transfer_timed_out` flag | per-rank (clock-skew detection) | **rank-consistent** (union via allgather) |
| Python `is_cancelled` boolean from `cancel_request` | per-rank | per-rank (unchanged) |
| C++ `setState(kDISAGG_TRANS_ERROR)` at `cacheTransceiver.cpp:690` | per-rank | per-rank (unchanged) |

### Is PR #14746 net-negative on V1?  No — strictly improves divergence

Walking through the state machine for "rank A detects timeout, rank B does not (clock skew)" on rid 5:

| Scenario | Before PR #14746 | After PR #14746 | Net |
|---|---|---|---|
| Only A's request is at a cancellable point | A: cancels + setState; B: doesn't flag, doesn't try cancel; state unchanged on B | A: cancels + setState; B: union-flags, calls `cancel_request`, returns false (request not cancellable), state unchanged on B | ✅ **same outcome** |
| Only B's request is at a cancellable point | A: flags + tries cancel, succeeds via local rank's view, setState on A; B: doesn't flag, doesn't cancel; state unchanged on B | A: same as before; B: union-flags, calls `cancel_request`, succeeds, setState on B | ⚠️ **B now also cancels** — but cancellation propagating to more ranks is desirable (the timeout has been declared instance-wide) |
| Both ranks' requests cancellable | A: cancels + setState; B: doesn't flag, doesn't cancel | A and B: both cancel + setState | ⚠️ **B now also cancels** — same direction, desirable |
| Neither cancellable | A: tries, no-op; B: doesn't try | A and B: try, no-op | ✅ **same outcome** |

The ⚠️ rows are where PR #14746 changes behavior on V1. In every such case, the change is **more uniform cancellation propagation**, which is the desirable direction for a wall-clock timeout: when the configured timeout has fired anywhere on the instance, we want all ranks to act in concert, not for some to keep the transfer alive.

So:

- **V1 cancellation outcome and state-transition divergence pre-existed PR #14746 and remain.** The PR doesn't introduce them, it just doesn't close them.
- **PR #14746 strictly reduces V1's divergence surface** by removing the flag-level divergence (clock-skew-driven flag flips happening on different iterations on different ranks).
- **PR #14746 never introduces new divergence.** No state transition fires "later" or in a new direction because of this change; cancel attempts that wouldn't have succeeded before still don't succeed; cancel attempts that would have succeeded propagate now to more ranks — but in the direction of "more uniformly cancelled," not "more divergent."

The remaining L2 gap on V1 (`cacheTransceiver.cpp:689-690` per-rank state transition) is a *separate* item, owned by our cancellation follow-up. PR #14746 is not the right place to close it because:

1. The PR explicitly targets the V2 path where `_consensus_outcome` already handles L2; the author's scope is appropriate.
2. Closing V1's L2 requires either a sibling allgather of `is_cancelled` outcomes or landing doc 14's V1 `_consensus_outcome` port — both bigger changes than this PR's scope.
3. The follow-up cancellation PR can build on the L1 pattern PR #14746 establishes.

## 4. Empirical evidence and the V1 divergence question

This section answers a question the PR #14746 discussion provoked: *given what we've observed in the doc 17 ablation, does V1 today have potential cross-rank divergence, only non-deterministic?*

**Short answer:** Yes — V1 has *structural* potential for cross-rank divergence that is non-deterministic when triggered, **but the divergence does not fire in V1's current baseline flow**. It surfaces when per-rank Python state is layered on top.

Evidence chain:

1. **Doc 17 ablation result**: the always-on baseline branch ([https://github.com/NVIDIA/TensorRT-LLM/pull/14768](https://github.com/NVIDIA/TensorRT-LLM/pull/14768), `A1+A2+A4+A7+A8`, no `A3/A5/A6/A9/A10`, no `G1–G8`) **passed** `test_asymmetric_executor[llama-6proc-ucx_kvcache-90]` on the standard L0 CI matrix. PR #13713 with `A3/A5/A6/A9/A10` present **failed** the same test consistently for ~4 weeks across ~8 rebuilds.

2. **Doc 16 diagnostic instrumentation** identified `A3`'s per-rank Python dedup sets (`_disagg_gen_init_prepared_ids` / `_disagg_gen_kv_recv_started_ids`) as the most-likely smoking gun via cross-rank trace divergence at Layer A-E DIAG tags.

3. **PR #14746's framing** independently shows the same antipattern is present elsewhere in disagg (the timeout flag) and applies a generic consensus primitive to fix it.

What this combination establishes:

- **V1 baseline flow is empirically rank-consistent** on the tested workload — the scheduler's broadcast keeps all ranks scheduling the same batch, and absent per-rank Python state, no rank-local decision feeds back into the collective.
- **V1 has no structural protection** against rank-divergence. Unlike V2, it has no `_consensus_outcome` mechanism (doc 14 introduces one but only under an opt-in env flag, default off; the consensus covers completion / failure, not cancellation).
- **The divergence is non-deterministic when triggered.** The A3 mechanism — per-rank dedup sets diverging on whether `request_and_receive_async` should fire — depends on the order of: scheduler broadcast arrival, prior-iteration state of each rank's local set, and timing of UCX / agent send-recv calls. The PR #14746 timeout mechanism likewise depends on clock skew across ranks. Both can fail to trigger on quiet workloads, fail to trigger on subsequent retries, or trigger on a different rank pair from one run to the next.
- **Future per-rank state additions to V1 will re-expose the gap.** Any work that lands per-rank Python attributes that gate collective behavior (cancellation surface, per-rank metrics-driven decisions, per-rank quota enforcement, etc.) inherits the L1 + L2 consensus requirement.

So the precise framing is: V1 is not *inherently* divergent today, but it is *latently* vulnerable to divergence whenever per-rank state is added without an explicit consensus primitive. The ablation in doc 17 didn't disprove the V1 divergence potential — it removed the trigger (A3 and friends), letting baseline V1 pass. The PR #14746 discussion shows the trigger family is broader than just A3, and the fix family needs to cover both L1 (flag) and L2 (state-transition) — not just L1 as PR #14746 does.

## 5. Implications for the in-flight cancellation follow-up PR

Updates [17 §6](17-tier1-ablation-result.md#6-implication-for-the-follow-up-cancellation-pr) with the L1 / L2 distinction:

- **L1 consensus for the dedup state** (the A3 problem): apply PR #14746's pattern — `dist.allgather` of the local "started" / "prepared" set per iteration, union, set every rank's flag from the consensus set. Called at a rank-symmetric site inside `_recv_disagg_gen_cache`.

- **L2 consensus for the cancellation outcome** (the new gap surfaced by §3 above): apply the same pattern to `is_cancelled` — gather locally-cancelled ids per iter, union, drive state transitions strictly from the consensus set. Called at the cancellation sites in `_handle_responses` and `_check_disagg_ctx_cache_transfer_status`.

The previously-proposed alternative designs (move dedup upstream; different idempotency primitive) remain on the table for L1 but don't address L2. The L2 gap is inherent to the V1 transceiver's lack of a `_consensus_outcome`. The minimal-blast-radius fix is the allgather pattern; the deeper fix is to land doc 14's `_consensus_outcome` port and extend its outcome set to include CANCELLED (vs. doc 14 §2.3's current "cancellation collapses into FAILED" simplification).

### 5.1 Collective shape

The detailed design — packed `(rid, state)` encoding, single-allgather collective shape, priority-encoded reduce, V2 propagation strategy — lives with the rest of the cancellation re-design work in [`docs/design/disagg-inflight-cancel-poison/phase1-consensus-collective-design.md`](../../design/disagg-inflight-cancel-poison/phase1-consensus-collective-design.md). Headline: one allgather of `vector<uint64>` (rid in low 60 bits, state in high 4 bits) reusing V1's existing `gatherRequestIds` helper, instead of porting V2's three-list shape — saves 3→1 collectives per call and avoids introducing new collective primitives. V2 propagation is deferred behind V1 measurements.

## 6. Cross-references

- [doc 12 §5.2 Path B](12-horizontal-consistency-and-layer3-gating.md): the original horizontal-consistency proposal that motivated doc 14.
- [doc 14](14-cross-rank-consistency-enforcement.md): the env-gated port of `_consensus_outcome` into V1; covers L2 for completion/failure, not cancellation.
- [doc 16 §5.1](16-diag-instrumentation-and-wedge-mobility.md#51-earlier-run-layer-ad-only): cross-rank divergence evidence from Layer A-E DIAG tags.
- [doc 17](17-tier1-ablation-result.md): ablation result implicating A3 / A5 / A6 / A9 / A10 cluster.
- Upstream PR #14746 (parallel effort on `feat/deepseek_v4`): concrete L1 consensus implementation worth referencing as prior art when designing our L1+L2 follow-up.
