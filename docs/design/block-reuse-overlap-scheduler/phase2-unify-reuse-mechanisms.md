# Phase 2: Unify Block Reuse and Disaggregated Partial Reuse

| | |
|---|---|
| **Depends on** | Phase 1 ([PR #12416](https://github.com/NVIDIA/TensorRT-LLM/pull/12416)) |
| **Status** | Design only |

## Problem

TensorRT-LLM has two overlapping mechanisms for making KV cache blocks reusable:

- **Block reuse** (general-purpose prefix caching): stores blocks in the radix tree when a request is terminated via `free_resources` → `releaseBlocks` → `storeBlocks`.
- **Disagg partial reuse**: stores blocks in the radix tree AND pins them during `start_transfer`, before the request is terminated, so blocks are discoverable during KV transfer.

The disagg partial reuse mechanism adds a pin/unpin lifecycle that introduces code complexity, a pipeline parallelism restriction (`pp_size == 1`), and a risk of permanent pin leaks on cancellation/timeout.

## Background

### Block reuse lifecycle

```
Allocated (ref > 0)  ──releaseBlocks──►  Cached (ref = 0)  ──getFreeBlock──►  Recycled
  not evictable                            in eviction queue
                                           in radix tree
                                                 │
                                           loadOrAllocate
                                           (cache hit)
                                                 │
                                           claimBlock
                                           incRefCount
                                                 │
                                           back to Allocated
```

Blocks enter the radix tree when the request is terminated. They remain in GPU memory and the radix tree until memory pressure forces eviction via `getFreeBlock`. On a cache hit, `claimBlock` pulls the block out of the eviction queue and increments its ref count.

### Disagg partial reuse lifecycle

```
start_transfer:
  → store_blocks_for_reuse(pin=True)    ← blocks in radix tree + pinned
  → start async KV transfer

_handle_responses:
  → (Phase 1 fix) skip termination during transfer

end_transfer:
  → unpin_blocks_by_id                  ← unpin
  → _terminate_request → free_resources ← ref count → 0
```

The pin protects blocks from eviction even with ref count 0. But with Phase 1's fix (sequence stays alive during transfer), ref count > 0 already provides equivalent protection — pinning is redundant.

## Proposed Design

Replace `store_blocks_for_reuse(request, True)` with `store_blocks_for_reuse(request, False)` — keep early radix-tree visibility, remove pinning.

### `start_transfer`

```python
if self.should_store_blocks:
    self.kv_cache_manager.store_blocks_for_reuse(request, False)  # no pin
```

### `end_transfer`

```python
if transfer_metadata.end_transfer():
    self._requests_in_transfer.pop(request.py_request_id)
    self._request_transfer_metadata.pop(request.py_request_id)
    # No unpin needed
    if request.state != LlmRequestState.DISAGG_TRANS_ERROR:
        request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
    return True
```

### `should_store_blocks`

Remove `pp_size == 1` restriction:

```python
should_store_blocks=self.enable_partial_reuse_for_disagg
    and not self.kv_cache_manager.is_vswa  # no pp_size check
```

### `RequestTransferMetadata`

Remove `block_id` field — only a transfer counter is needed.

## Why Reference Counting Is Sufficient

1. **Sequence alive → ref count > 0**: `start_transfer` does not call `free_resources`. The sequence remains in `mAllocatedBlocksPerSeq`.
2. **Ref count > 0 → not evictable**: the eviction policy only operates on blocks with ref count 0 (via `releaseBlock`). Blocks with ref count > 0 are never in the eviction queue.
3. **Termination only after transfer**: Phase 1's fix ensures `_handle_responses` skips termination during transfer. `_end_transfer_and_maybe_terminate` handles it after completion.
4. **V2 suspend safe**: in-transfer requests are dropped from `active_requests`, so the V2 scheduler cannot suspend them.

## Benefits

| Benefit | Detail |
|---------|--------|
| Early reusable blocks | Preserved — `store_blocks_for_reuse(pin=False)` still stores in radix tree |
| No pin/unpin lifecycle | No `pin=True`, no `unpin_blocks_by_id`, no `block_id` tracking |
| PP support | `pp_size == 1` restriction removed — no cross-rank pin/unpin coordination needed |
| No pin leak risk | Cancelled/timed-out requests can't leave permanently pinned blocks |
| Simpler code | `RequestTransferMetadata` reduced to a counter; `end_transfer` simplified |

## Concerns

| Concern | Impact | Mitigation |
|---------|--------|------------|
| Sequence stays alive longer | Negligible — `start_transfer` already frees SEQ_SLOT and SPEC resources; only KV cache tracking remains for ~1–5 ms | N/A |
| Scheduler sees fewer free blocks | Slight — scheduler counts in-transfer blocks as "allocated" vs. "free" (pinned) | Conservative scheduling; affects at most one scheduling decision per transfer window |
| Behavioral change | Pin-count introspection tools would need updating | No known tools in the codebase inspect pin state |

## Comparison

| Aspect | Current (pin-based) | Proposed (ref-count-based) |
|--------|--------------------|-----------------------------|
| Blocks available during transfer | Yes | Yes (no change) |
| PP support | No (`pp_size == 1`) | Yes |
| Pin leak risk | Yes (on cancel/timeout) | None |
| `RequestTransferMetadata` | Stores `block_id` | Counter only |
| `end_transfer` complexity | Unpin + cleanup | Pop from maps |
| `should_store_blocks` condition | 3 conditions | 2 conditions |
| Protection mechanism | Pin flag | Sequence ref count |
| Protection strength | Equivalent | Equivalent |
