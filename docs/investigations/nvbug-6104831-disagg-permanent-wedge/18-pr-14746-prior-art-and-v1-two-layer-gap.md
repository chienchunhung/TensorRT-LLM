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

## 3. Concrete L2 holes that remain on V1 after PR #14746

Sites where V1 + PR #14746 still has rank-divergent code on the cancellation path (from the PR's own diff, reading from `pr-14746-head:py_executor.py`):

1. **Site 1, ctx-side cancel (line 4006):**
   ```python
   if request.py_kv_transfer_timed_out and request_id not in completed_req_ids:
       is_cancelled = self.kv_cache_transceiver.cancel_request(request)   # ← per-rank bool, no allgather
       if is_cancelled:
           request.py_kv_transfer_start_time = None
           request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE         # ← rank-local state set
           self._end_transfer_and_maybe_terminate(request)
   ```
   Even with `py_kv_transfer_timed_out` rank-consistent, `is_cancelled` can disagree (the C++ `CacheSender::cancelRequest` returns true only if the request is currently in `mReadyResponses` and not the active one, or — under the in-flight cancel flag — if the per-request cancel flag was successfully flipped). Different ranks → different state transitions.

2. **Site 2, gen-side cancel (line 4488):**
   ```python
   if request.py_kv_transfer_timed_out:
       is_cancelled = self.kv_cache_transceiver.cancel_request(request)   # ← per-rank bool, no allgather
       if is_cancelled:
           timed_out_requests.append(request)
       continue
   ```
   Same problem: `timed_out_requests` composition diverges across ranks.

3. **Downstream of Site 2, gen-side cleanup (line 4581-4584):**
   ```python
   if self.enable_attention_dp and self.dist.world_size != 1:
       self._pending_timed_out_requests.extend(timed_out_requests)        # ADP: deferred to synced drain
   else:
       for req in timed_out_requests:
           self._handle_errors(error_msg=..., requests=[req])              # ← non-ADP: rank-local cleanup from divergent list
   ```
   The ADP branch buffers and drains under the existing bool-allgather, masking some of the divergence. The non-ADP branch executes `_handle_errors` directly per rank on a possibly-divergent list.

The fix shape, mirroring PR #14746's pattern: a second `dist.allgather(cancelled_ids)` after `cancel_request` calls, take the union (or intersection — design choice), then apply state transitions strictly from the consensus set. Equivalent in cost (one more small allgather/iter), structurally identical to what PR #14746 added for L1.

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

- **L1 consensus for the dedup state** (the A3 problem): apply PR #14746's pattern — `dist.allgather` of the local "started" / "prepared" set per iteration, union, set every rank's flag from the consensus set. Called at a rank-symmetric site inside `_recv_disagg_gen_cache`. Cost: one small allgather per iter under cancel-enabled disagg.

- **L2 consensus for the cancellation outcome** (the new gap surfaced by §3 above): apply the same pattern to `is_cancelled` — `dist.allgather` of the locally-cancelled ids per iter, union, drive state transitions strictly from the consensus set. Called at the cancellation sites in `_handle_responses` and `_check_disagg_ctx_cache_transfer_status`. Cost: one more small allgather per iter under cancel-enabled disagg.

Both allgathers are lockstep / possibly-empty / same shape as PR #14746's. Total added per-iter coordination under cancel-enabled disagg V1: two small `dist.allgather` calls (plus PR #14746's one for timeouts, if running on a base that includes it). Worth measuring on a 6-rank disagg test before committing.

The previously-proposed alternative designs (move dedup upstream; different idempotency primitive) remain on the table for L1 but don't address L2. The L2 gap is inherent to the V1 transceiver's lack of a `_consensus_outcome`. The minimal-blast-radius fix is the allgather pattern; the deeper fix is to land doc 14's `_consensus_outcome` port and extend its outcome set to include CANCELLED (vs. doc 14 §2.3's current "cancellation collapses into FAILED" simplification).

## 6. Cross-references

- [doc 12 §5.2 Path B](12-horizontal-consistency-and-layer3-gating.md): the original horizontal-consistency proposal that motivated doc 14.
- [doc 14](14-cross-rank-consistency-enforcement.md): the env-gated port of `_consensus_outcome` into V1; covers L2 for completion/failure, not cancellation.
- [doc 16 §5.1](16-diag-instrumentation-and-wedge-mobility.md#51-earlier-run-layer-ad-only): cross-rank divergence evidence from Layer A-E DIAG tags.
- [doc 17](17-tier1-ablation-result.md): ablation result implicating A3 / A5 / A6 / A9 / A10 cluster.
- Upstream PR #14746 (parallel effort on `feat/deepseek_v4`): concrete L1 consensus implementation worth referencing as prior art when designing our L1+L2 follow-up.
