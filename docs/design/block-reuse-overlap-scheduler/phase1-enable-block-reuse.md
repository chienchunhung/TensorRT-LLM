# Phase 1: Re-Enable Block Reuse with Overlap Scheduler

| | |
|---|---|
| **PR** | [#12416](https://github.com/NVIDIA/TensorRT-LLM/pull/12416) |
| **Status** | In review |

## Problem

Block reuse (`enable_block_reuse`, default `True`) and the overlap scheduler (`disable_overlap_scheduler=False`, default) are both enabled by default, but their combination was explicitly blocked for disaggregated context-only requests via a `ValueError` guard in `base_worker.py`.

## Analysis: Was the Guard Necessary?

No. The combination is functionally correct with the original overlap loop ordering. The overlap loop defers previous-batch processing (`_process_previous_batch`) to after the current batch's `prepare_resources`. This means blocks from batch N-1 are not yet in the radix tree when batch N's resource preparation runs — a one-iteration delay in block availability. This is a minor performance miss, not a correctness issue.

Safety properties verified:

- **No data corruption**: KV cache blocks are write-once. Shared blocks are read-only.
- **No use-after-free**: blocks in transfer are protected by reference counting (sequence alive → ref count > 0).
- **No double-free**: both V1 (`storeContextBlocks` checks `mSequences.find`) and V2 (`kv_cache_map` membership check) handle already-removed sequences gracefully.
- **No scheduling error**: the C++ scheduler's `startScheduling` snapshots free block counts correctly.

## Root Cause of the CI Failure

Removing the guard exposed a latent bug in `_handle_responses`. When `enable_partial_reuse_for_disagg` was True, context-only requests were terminated unconditionally — even while their KV cache was still being transferred:

```python
# Before fix — unconditional termination:
if self.enable_partial_reuse_for_disagg and not self.kv_cache_manager.is_vswa and self.dist.pp_size == 1:
    requests_to_terminate.append(request)     # ignores transmission state
else:
    if not request.is_disagg_context_transmission_state:
        requests_to_terminate.append(request)  # checks transmission state
```

The `else` branch correctly skipped requests in transmission, but the `if` branch did not. Under the overlap scheduler, this caused double-termination:

1. `_send_kv_async(N-1)` → `start_transfer` → stores blocks, pins them, starts transfer
2. `_handle_responses` → terminates the request → `free_resources` → ref count drops to 0
3. Transfer completes → `_end_transfer_and_maybe_terminate` → `_terminate_request` called again

The underlying issue is that the early-termination path assumed pinned blocks were sufficient protection during KV transfer. While pinning protects the blocks from eviction, the request lifecycle was not accounted for: `end_transfer` later attempts to terminate the same request again. Under the non-overlap scheduler this was benign (the transfer typically completed before `_handle_responses` ran), but under the overlap scheduler the deferred processing created a window where both code paths could race on the same request.

## Changes

### Guard removal (`base_worker.py`)

Remove the `ValueError` guard for context-only + overlap + block_reuse + KV cache transceiver.

### Double-termination fix (`py_executor.py`)

Skip termination for requests in `DISAGG_CONTEXT_TRANS_IN_PROGRESS` state. After the fix, both branches have identical logic and are collapsed:

```python
# After fix — single, unified condition:
if not request.is_disagg_context_transmission_state:
    requests_to_terminate.append(request)
```

Requests in transmission are terminated by `_end_transfer_and_maybe_terminate` when the transfer completes.

### `end_transfer` return value fix

`AsyncTransferManager.end_transfer` returned `None` (bare `return`) on `KeyError` instead of `False`. Changed to `return False` to prevent unintended termination by the caller.

### `_handle_responses` refactoring

- Extract `_maybe_update_speculation_gate()` — flattens 5-level nesting into guard-clause helper.
- Extract `_should_emit_response()` — names the streaming/final response condition.
- Collapse duplicate termination branches.
- Use early `continue` to flatten the done/not-done flow.

Result: ~115 lines with 6 levels of nesting → ~80 lines with max 4 levels.

### Tests

- `test_overlap_scheduler_consistency`: add `enable_block_reuse` as parametrized axis (`[False, True]`).
- `test_overlap_scheduler_block_reuse_cache_hit` (new): sends same prompts twice, asserts `cached_tokens > 0` on second pass.
- Remove `pytest.skip` for overlap + block_reuse in disaggregated serving test.
- Enable `enable_block_reuse: true` in `disagg_config_overlap.yaml`.
- Enable `enable_block_reuse` unconditionally in `_test_chunked_prefill_helper` (was gated on `ctx_pp == 1`; regular block reuse has no PP restriction).

## Performance Impact

None. The changes are to request-termination logic, not the hot path. The fix adds one boolean check (`is_disagg_context_transmission_state`) per finished request in `_handle_responses`.

The one-iteration delay in block availability is accepted in Phase 1. Phase 3 proposes an optimization to eliminate it.
