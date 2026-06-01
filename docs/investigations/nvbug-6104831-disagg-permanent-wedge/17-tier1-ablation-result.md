# 17 — Ablation result: the wedge race lives in the always-on auxiliaries, not the cancel surface

**Status:** Empirical finding (2026-05-31). Closes the open hypothesis in [16 §8](16-diag-instrumentation-and-wedge-mobility.md).
**Trigger:** CI on the "always-on baseline" PR (`A1 + A2 + A4 + A7 + A8`, with `A3 / A5 / A6 / A9 / A10` removed and the entire `G1–G8` cancel surface excluded). The PR head is at <https://github.com/NVIDIA/TensorRT-LLM/pull/14768>.

## 1. The hypothesis we set up

Doc 16 §8 ended with:

> If the always-on-only branch wedges on the same test, the race is in code that's *not* in the cancellation surface — and the focus shifts to the rank-symmetric collective entry, the per-rank dedup, or the post-ready-signal data-transfer code. If the always-on-only branch does *not* wedge on the same test, then by elimination the cancellation surface is implicated and Phase 4 instrumentation will pin the specific site.

Three parallel branches were proposed in §8 to ablate the components. The "merge-intended baseline" branch is the one this doc reports on.

## 2. What landed in the baseline branch

Strict subset of the parent PR's always-on changes plus an empty cancel surface:

| Component | In baseline branch | Notes |
|---|---|---|
| `A1` — `shared_ptr<LlmRequest>` in `mSenderFutures` / `mRequesterFutures` | ✓ | always-on |
| `A2` — `BufferIndexHolder` RAII + integration at 4 formatter sites | ✓ | always-on |
| `A4` — `mTimedOutSenderIds` / `mTimedOutRequesterIds` dedup | ✓ | always-on |
| `A7` — observe-only timeout WARN | ✓ | always-on |
| `A8` — NIXL `nb::keep_alive<0, 1>()` | ✓ | always-on |
| `A3` — Python recv-side dedup sets | ✗ | always-on in parent; removed here |
| `A5` — state reorder in `requestAndReceiveAsync` | ✗ | always-on in parent; removed here |
| `A6` — eager `setKvCacheTransferStart` at function entry | ✗ | always-on in parent; removed here |
| `A9` — gen-side rank-symmetric collective entry | ✗ | always-on in parent; reverted here |
| `A10` — ctx-side rank-symmetric collective entry | ✗ | always-on in parent; reverted here |
| `G1–G8` — in-flight cancellation surface | ✗ | env-gated, default OFF in parent; absent here |

## 3. The empirical result

The baseline branch **passed** `cpp/test_multi_gpu.py::TestDisagg::test_asymmetric_executor[llama-6proc-ucx_kvcache-90]` on the standard L0 merge-request CI matrix.

The parent PR consistently failed the same test for ~four weeks across `~8` rebuilds at the cluster of commits that included `A3` / `A5` / `A6` / `A9` / `A10`.

## 4. What that resolves and what it doesn't

**Resolves:** the cancellation surface (`G1–G8`) is **not** implicated in the wedge — those code paths are dormant by default in the parent PR and entirely absent in the baseline. Removing them did not on its own fix the test; the parent PR with `G1–G8` dormant (default config) still failed CI.

**Implicates:** the wedge race lives in the always-on auxiliaries that the baseline removed (`A3` / `A5` / `A6` / `A9` / `A10`). The diagnostic instrumentation (doc 16) identified `A3`'s per-rank Python dedup sets (`_disagg_gen_init_prepared_ids`, `_disagg_gen_kv_recv_started_ids`) as the most-likely smoking gun: the sets are per-rank state with no cross-rank consensus, so two gen ranks can diverge on whether `request_and_receive_async` should fire for the same scheduler-broadcast request, leaving ctx waiting for a request-info that never arrives.

**Doesn't formally isolate:** which of the five removed components is *individually* sufficient to fix the test. The baseline removes all five together. Per-component isolation would require five additional CI cycles each ablating one component back in. Given the diagnostic-instrumentation evidence in doc 16 points cleanly at `A3`, the cost/value of those cycles is low.

## 5. Why `A3` (and `A5` / `A6`) are unnecessary without the cancel surface

`A3` exists to suppress duplicate `request_and_receive_async` calls under the *cancel-throw retry pattern*: if `receiveAsync` throws mid-call (which only happens with the per-request cancel flag triggering inside the UCX / agent send-recv loops), state stays in `DISAGG_GENERATION_INIT` and the scheduler re-presents the request. The dedup set prevents that re-presentation from double-emplacing into `mRequesterFutures`. The coupled `A5` reordering (`setState` moved after `receiveAsync`) ensures the throw leaves the request "atomically not in flight" so the retry is consistent; `A6` is the timestamp scaffolding that makes `A5`'s timing observable to `checkGenTransferStatus`. The three are a cluster; they only make sense as a group when the cancel-throw path can fire.

Without the cancel surface, `receiveAsync` doesn't throw under normal flow. State transitions to `DISAGG_GENERATION_TRANS_IN_PROGRESS` reliably. The scheduler's state-based filter (already in upstream, predates this work) naturally excludes IN_PROGRESS requests from `fitting_disagg_gen_init_requests` on the next iteration. No dedup needed.

`A9` / `A10` are rank-symmetric collective-entry fixes for ABBA hazards exposed by the parent PR's new C++ `gatherRequestIds` Allgather inside `check{Gen,Context}TransferStatus`. The baseline doesn't add that Allgather, so the ABBA hazards `A9` / `A10` exist to fix are not present, and `A9` / `A10` themselves are not needed.

## 6. Implication for the follow-up cancellation PR

When in-flight cancellation lands, the cancel-throw retry pattern returns and `A3` / `A5` / `A6` come back with it. But the baseline finding — that `A3`'s per-rank dedup is the divergence mechanism — means the design needs more care than a straight port from the parent PR:

- Either the dedup state needs cross-rank consensus (an Allgather of the active "started" set per iteration), or
- The dedup point needs to move upstream of the rank-broadcast (so the scheduler sees a consistent input across ranks), or
- The cancel-throw retry needs a different idempotency primitive that doesn't rely on per-rank Python state.

This is the open design question the follow-up PR must answer. Not a port; a redesign.

## 7. Doc relationships

- Closes the hypothesis open at [16 §8](16-diag-instrumentation-and-wedge-mobility.md#8-implications-for-the-decomposition-plan-and-the-parallel-verification-branches).
- Refines [15 — PR decomposition plan](15-pr-decomposition-plan/README.md): the baseline tier landed; the cancellation tier requires the redesign in §6 above before it can move.
- Builds on the cross-rank divergence finding in [doc 16 §5.1](16-diag-instrumentation-and-wedge-mobility.md#51-earlier-run-layer-ad-only).
