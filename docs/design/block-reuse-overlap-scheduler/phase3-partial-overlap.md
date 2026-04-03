# Phase 3: Partial Overlap for Immediate Block Reuse

| | |
|---|---|
| **Depends on** | Phase 1 ([PR #12416](https://github.com/NVIDIA/TensorRT-LLM/pull/12416)) |
| **Status** | Design only |

## Problem

With the overlap scheduler enabled and block reuse active (Phase 1), there is a one-iteration delay in block availability. Blocks from batch N-1 are not in the radix tree when batch N's `prepare_resources` runs, because the overlap loop defers `_process_previous_batch` (which frees blocks and stores them for reuse) to after the current batch's resource preparation and GPU forward pass.

For disaggregated context servers processing many sequential requests with shared system prompts, the delay means blocks from the immediately preceding request are never available for the next.

## Proposed Design

Split previous-batch processing into two phases. Move lightweight resource management (free blocks, store context blocks) to before scheduling. Keep heavier work (response building, enqueuing, stats) in the overlap window. Gate the early phase on `enable_kv_cache_reuse` so workloads without block reuse pay zero overhead.

### Iteration flow (block reuse enabled)

```
 1. _update_requests(N-1)                     ← CPU (early phase)
 2. _free_completed_request_resources(N-1)    ← CPU: free blocks, store ctx blocks
 3. schedule(N)                                ← CPU: radix-tree lookup finds N-1's blocks
 4. prepare_resources(N)                       ← CPU
 5. forward(N)                                 ← GPU launch
 6. _send_kv_async(N-1)                        ← CPU  ┐
 7. drafter.cleanup(N-1)                       ← CPU  │ overlap with GPU
 8. _pause_requests(N)                         ← CPU  │
 9. _sample_async(N)                           ← GPU  │
10. _process_previous_batch(N-1,               ← CPU  │ responses + stats only
        skip_resource_update=True)                     ┘
11. save previous_batch = batch_N
```

### Iteration flow (block reuse disabled — identical to original)

```
 1. schedule(N)                          ← CPU
 2. prepare_resources(N)                 ← CPU
 3. forward(N)                           ← GPU launch
 4. _update_requests(N-1)                ← CPU  ┐
 5. _send_kv_async(N-1)                  ← CPU  │ overlap with GPU
 6. drafter.cleanup(N-1)                 ← CPU  │
 7. _pause_requests(N)                   ← CPU  ┘
 8. _sample_async(N)                     ← GPU
 9. _process_previous_batch(N-1)         ← CPU: full processing
10. save previous_batch = batch_N
```

### New method: `_free_completed_request_resources`

```python
def _free_completed_request_resources(self):
    self._early_freed_request_ids.clear()
    for request in self.active_requests:
        if request.is_finished or request.is_attention_dp_dummy:
            self.resource_manager.free_resources(request)
            self._early_freed_request_ids.add(request.py_request_id)

    scheduled_requests = self.previous_batch.scheduled_requests
    attn_metadata = getattr(self.model_engine, 'attn_metadata', None)
    kv_cache_dtype_byte_size = getattr(
        self.model_engine, 'kv_cache_dtype_byte_size', None)
    self.resource_manager.update_resources(
        scheduled_requests, attn_metadata, kv_cache_dtype_byte_size)
```

### Double-operation prevention

- `_early_freed_request_ids` tracking: `_do_terminate_request` checks this set and skips `free_resources` for already-freed requests (V1's `remove_sequence` is not idempotent).
- `skip_resource_update` parameter: `_process_previous_batch(skip_resource_update=True)` skips the duplicate `update_resources` call.

### Conditional gating

```python
if self.previous_batch is not None and self.enable_kv_cache_reuse:
    self._update_requests(self.previous_batch.sample_state)
    self._free_completed_request_resources()
```

When `enable_kv_cache_reuse` is False, the early phase is skipped entirely — zero overhead, identical to the original overlap loop.

## Performance Analysis

### CPU cost estimates

| Operation | Estimated cost | Notes |
|-----------|---------------|-------|
| `_update_requests(N-1)` | 0.1–0.5 ms | GPU sync (no-op) + O(batch_size) Python iteration |
| `_free_completed_request_resources` | 0.1–1.0 ms | O(finished) `free_resources` + `update_resources` |
| **Total early phase** | **0.2–1.5 ms** | Only when block reuse is enabled |

### Impact by model size

| Model size | GPU forward | Early phase | % overhead | Cache hit savings |
|------------|------------|-------------|------------|-------------------|
| Large (70B+) | 50–100 ms | ~0.5–1.5 ms | ~0.5–1.5% | 10–100 ms |
| Medium (7–13B) | 5–20 ms | ~0.3–1.0 ms | ~2–5% | 5–20 ms |
| Small (1–3B) | 1–5 ms | ~0.2–0.5 ms | ~5–15% | 1–5 ms |

For all model sizes, a single cache hit saves more than the early phase costs.

### Comparison

| Approach | Added to critical path | Overlap preserved | Block reuse immediate |
|----------|----------------------|-------------------|----------------------|
| **Original** (reuse disabled) | 0 ms | Full | N/A |
| **Phase 1** (guard removed) | 0 ms | Full | No (one-iteration delay) |
| **Full move** (rejected) | 0.7–4.6 ms | None | Yes |
| **Partial overlap** (this proposal, reuse on) | 0.2–1.5 ms | Most | Yes |
| **Partial overlap** (reuse off) | 0 ms | Full | N/A |

## Concerns

| Concern | Impact | Mitigation |
|---------|--------|------------|
| GPU idle time +0.2–1.5 ms | Negligible for medium/large models; ~5–15% for small models | Gated on `enable_kv_cache_reuse`; users can disable |
| `_update_requests` GPU sync moves earlier | No-op sync (previous sampling already completed) | < 0.01 ms measured |
| Double-operation prevention complexity | `_early_freed_request_ids` set + `skip_resource_update` flag | Localized; set cleared each iteration |
| Spec decode interaction | None — `_update_requests` touches `.host` tensors; spec decode reads `.device` | Confirmed orthogonal |
| Attention DP interaction | Safe — inter-rank ops in `_can_queue` occur after early phase | `early_phase_ran` flag gates deferred `_update_requests` |

## Pipeline Parallelism

The overlap loop only runs when `pp_size == 1`. The PP loop (`_executor_loop_pp`) has a different structure (microbatch pipelining with ring-broadcast) where the cross-microbatch same-iteration delay is less impactful and the implementation complexity of early release would be significantly higher (multi-threaded queue coordination, inter-rank batch-count synchronization). The partial overlap approach is not recommended for the PP loop.
