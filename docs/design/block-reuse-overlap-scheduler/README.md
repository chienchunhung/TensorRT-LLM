# Revisiting Block Reuse in TRT-LLM

**Unifying Prefix Caching for Overlap Scheduling and Disaggregated Serving**

| | |
|---|---|
| **JIRA** | [TRTLLM-10938](https://jirasw.nvidia.com/browse/TRTLLM-10938), [TRTLLM-10939](https://jirasw.nvidia.com/browse/TRTLLM-10939) |
| **PRs** | [#12816](https://github.com/NVIDIA/TensorRT-LLM/pull/12816) (merged — minimal fix to unblock block reuse + overlap), [#12416](https://github.com/NVIDIA/TensorRT-LLM/pull/12416) (closed — superseded by #12816) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-17 |
| **Last Updated** | 2026-04-13 |
| **Status** | Block reuse + overlap scheduler unblocked (PR #12816 merged). Phase 2 and Phase 3 deprioritized — design docs retained for future reference. |

## Context

KV cache block reuse (prefix caching) and the overlap scheduler are two of the most impactful performance features in TensorRT-LLM's PyTorch backend. Block reuse avoids redundant prefill computation for requests sharing common token prefixes. The overlap scheduler hides CPU overhead behind GPU execution by pipelining consecutive batch processing. Both are enabled by default.

However, these two features were **mutually exclusive** in disaggregated serving — an explicit guard rejected context-only requests when both were active. This meant disaggregated context servers, the workload where prefix caching provides the most benefit, could not use it.

Investigation revealed three layers of issues:

1. **The guard was precautionary, not required for correctness.** Block reuse + overlap is functionally safe — the only effect is a one-iteration delay in block availability.
2. **Removing the guard exposed a latent double-termination bug.** The disagg partial reuse path terminated context-only requests during KV transfer, causing `end_transfer` to terminate the same request again under the overlap scheduler.
3. **The disagg partial reuse mechanism itself is unnecessarily complex.** It uses a pin/unpin lifecycle that, with the double-termination fix, is now redundant — reference counting already provides equivalent block protection.

## What Landed

[PR #12816](https://github.com/NVIDIA/TensorRT-LLM/pull/12816) (merged Apr 13, 2026) delivered a **minimal fix** to unblock block reuse with the overlap scheduler:

- Removed the `ValueError` guard in `base_worker.py`.
- Fixed the double-termination by guarding the redundant `_terminate_request` call in `_end_transfer_and_maybe_terminate` with `if not should_store_blocks` — when `should_store_blocks` is True, `_handle_responses` already terminated the request via the early-termination path.
- Fixed `end_transfer` to return `False` (instead of bare `return`) on `KeyError`.
- Added tests for overlap + block reuse consistency and cache-hit verification.
- Enabled `enable_block_reuse: true` in disaggregated overlap test configs.

This approach preserves the existing early-termination + pin/unpin mechanism and avoids the larger refactoring proposed in Phases 2 and 3.

## Action Items

### Phase 1: Re-Enable Block Reuse with Overlap Scheduler

**Status:** ✅ Complete ([PR #12816](https://github.com/NVIDIA/TensorRT-LLM/pull/12816) merged)

Minimal fix: guard removal + `should_store_blocks` conditional in `_end_transfer_and_maybe_terminate`. Block reuse now works with the overlap scheduler in both aggregated and disaggregated serving.

The original PR #12416 explored a broader refactoring approach (extracting `_maybe_update_speculation_gate`, `_should_emit_response`, collapsing termination branches in `_handle_responses`). That approach was superseded by the minimal fix in #12816 based on reviewer feedback to keep the change focused.

### Phase 2: Unify Block Reuse and Disaggregated Partial Reuse

**Status:** Design complete — deprioritized
**Priority:** P1 (when prioritized) — Code simplification and PP enablement

Remove the pin/unpin lifecycle from `AsyncTransferManager`. Rely on reference counting for block protection during KV transfer. Drop the `pp_size == 1` restriction. The design is documented below for future reference when this work is picked up.

### Phase 3: Partial Overlap for Immediate Block Reuse

**Status:** Design complete — deprioritized
**Priority:** P2 (when prioritized) — Performance optimization

Add a conditional early-phase resource release to the overlap loop to eliminate the one-iteration delay in block availability. The design is documented below for future reference.

## Dependency

```
Phase 1: PR #12816 (guard removal + should_store_blocks conditional) ✅ MERGED
  |
  +---> Phase 2 (unify reuse mechanisms, remove pinning) — deprioritized
  |
  +---> Phase 3 (partial overlap for immediate reuse) — deprioritized
```

Phase 2 and Phase 3 are independent of each other but both depend on Phase 1.

## Feature Interaction Summary

| Feature | Interaction |
|---------|------------|
| Overlap scheduler | Direct: guard removed, combination now works |
| Disaggregated serving | Direct: context-only + overlap + block_reuse functional |
| Pipeline parallelism | Phase 1: orthogonal (overlap loop only for PP=1). Phase 2: lifts PP restriction on disagg partial reuse |
| Speculative decoding | Orthogonal: draft/target resources managed independently |
| In-flight batching | Orthogonal: block reuse only affects context requests |
| Tensor/Expert parallelism | Orthogonal: all operations are rank-local |

## Detailed Design Documents

- [Phase 1: Re-Enable Block Reuse with Overlap Scheduler](phase1-enable-block-reuse.md)
- [Phase 2: Unify Block Reuse and Disaggregated Partial Reuse](phase2-unify-reuse-mechanisms.md)
- [Phase 3: Partial Overlap for Immediate Block Reuse](phase3-partial-overlap.md)
