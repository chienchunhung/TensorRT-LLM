# Revisiting Block Reuse in TRT-LLM

**Unifying Prefix Caching for Overlap Scheduling and Disaggregated Serving**

| | |
|---|---|
| **JIRA** | [TRTLLM-10938](https://jirasw.nvidia.com/browse/TRTLLM-10938), [TRTLLM-10939](https://jirasw.nvidia.com/browse/TRTLLM-10939) |
| **PRs** | [#12416](https://github.com/NVIDIA/TensorRT-LLM/pull/12416) (Phase 1: enable block reuse + overlap) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-17 |
| **Last Updated** | 2026-04-03 |
| **Status** | Phase 1 in review; Phase 2 and Phase 3 design only |

## Context

KV cache block reuse (prefix caching) and the overlap scheduler are two of the most impactful performance features in TensorRT-LLM's PyTorch backend. Block reuse avoids redundant prefill computation for requests sharing common token prefixes. The overlap scheduler hides CPU overhead behind GPU execution by pipelining consecutive batch processing. Both are enabled by default.

However, these two features were **mutually exclusive** in disaggregated serving — an explicit guard rejected context-only requests when both were active. This meant disaggregated context servers, the workload where prefix caching provides the most benefit, could not use it.

Investigation revealed three layers of issues:

1. **The guard was precautionary, not required for correctness.** Block reuse + overlap is functionally safe — the only effect is a one-iteration delay in block availability.
2. **Removing the guard exposed a latent double-termination bug.** The disagg partial reuse path terminated context-only requests during KV transfer, causing `end_transfer` to terminate the same request again under the overlap scheduler.
3. **The disagg partial reuse mechanism itself is unnecessarily complex.** It uses a pin/unpin lifecycle that, with the double-termination fix, is now redundant — reference counting already provides equivalent block protection.

## Action Items

### Phase 1: Re-Enable Block Reuse with Overlap Scheduler

**Status:** In review ([PR #12416](https://github.com/NVIDIA/TensorRT-LLM/pull/12416))
**Priority:** P0 — Blocking feature enablement

Remove the guard, fix the double-termination bug, refactor `_handle_responses`. This is the prerequisite for all subsequent work.

### Phase 2: Unify Block Reuse and Disaggregated Partial Reuse

**Status:** Design complete
**Priority:** P1 — Code simplification and PP enablement

Remove the pin/unpin lifecycle from `AsyncTransferManager`. Rely on reference counting for block protection during KV transfer. Drop the `pp_size == 1` restriction.

### Phase 3: Partial Overlap for Immediate Block Reuse

**Status:** Design complete
**Priority:** P2 — Performance optimization

Add a conditional early-phase resource release to the overlap loop to eliminate the one-iteration delay in block availability.

## Dependency

```
Phase 1: PR #12416 (guard removal + double-termination fix)
  |
  +---> Phase 2 (unify reuse mechanisms, remove pinning)
  |
  +---> Phase 3 (partial overlap for immediate reuse)
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
