# ADP Dummy Request Handling

## Why Dummies Exist

With Attention Data Parallelism (ADP), each TP rank processes a different subset of requests (unlike standard TP where all ranks process the same batch). The model forward pass still contains NCCL collective operations (allgather, broadcast) that require all ranks to participate. If a rank has zero requests, it can't participate in collectives, which would either deadlock or cause `_can_queue` to block the entire forward pass.

The dummy request is a single-token placeholder that lets idle ranks participate in collectives. Its output is discarded. Dummies are cheap (one KV cache slot per idle rank) and localized (only in the executor, invisible to the model code).

## The Permanence Problem

Dummies follow an add-forward-terminate lifecycle: added before the forward pass, used during collectives, terminated after the forward pass in `_update_request_states_tp`. However, during the benchmark disagg fill phase, the `can_forward` gate prevents forward passes from running. A dummy added during the fill phase is never terminated — it permanently occupies a KV cache slot for the rest of the fill phase.

## Two Phases: Fill vs Taper-Down

The benchmark disagg mode has two distinct phases with different dummy requirements:

### Fill Phase (`_benchmark_fill_phase_active = True`)

The period between starting the GEN executor and the first forward pass. KV transfers are in progress, the `can_forward` gate is closed, and no forward-pass collectives run.

- **Dummies should be skipped.** The gate prevents collectives, so empty ranks are safe. Adding a dummy would permanently waste a KV cache slot (no forward pass to terminate it).

### Taper-Down Phase (`_benchmark_fill_phase_active = False`)

The period after the gate opens and the benchmark is running. Requests generate tokens and finish at different rates (e.g., due to varied speculative decoding acceptance rates). Some ranks may temporarily become empty while others still have work.

- **Dummies should be allowed.** Forward passes are running normally, so dummies follow the normal add-forward-terminate lifecycle. Ranks that temporarily empty out need dummies to participate in collectives.

## The `_benchmark_fill_phase_active` Flag

A runtime flag that starts `True` (when `is_benchmark_disagg` is True) and is cleared to `False` when the `can_forward` gate opens in `_check_benchmark_disagg_gate`. This cleanly separates the two phases:

- `is_benchmark_disagg`: configuration fact — "we are in benchmark disagg mode" (never changes)
- `_benchmark_fill_phase_active`: runtime state — "we are in the fill phase" (transitions once: True → False)

## Refactored Helper Methods

### `_count_schedulable_active_requests() -> int`

Counts active requests that have completed KV transfer. In non-disagg mode, all active requests count. In disagg mode, requests in INIT or transmission-in-progress state are excluded.

```python
def _is_awaiting_kv_transfer(req) -> bool:
    return (req.is_disagg_generation_init_state
            or req.is_disagg_generation_transmission_in_progress)

return sum(1 for req in self.active_requests
           if not _is_awaiting_kv_transfer(req))
```

### `_should_skip_dummy_for_benchmark_disagg(num_schedulable_requests) -> bool`

Simple check gated by the fill phase flag:

```python
if not self._benchmark_fill_phase_active or self.is_warmup:
    return False    # not in fill phase — use normal dummy lifecycle

return True         # fill phase active — skip dummies
```

## Why Not Add Dummies Early?

Adding dummies on temporarily-empty ranks during fill is harmful:
- The dummy permanently occupies a KV cache slot (never cleaned up during fill)
- A real request arriving later at that rank must compete with the stuck dummy for KV cache
- The dummy doesn't serve any purpose — the gate blocks all collectives anyway

## Duplicate Prevention

Once a dummy is added, `num_active_request` becomes > 0, so the condition `num_active_request == 0 and expected_num_active_requests > 0` is False. No second dummy is ever added. After the gate opens, the normal add-forward-terminate lifecycle handles cleanup, and the dummy is removed after each forward pass.

## Distribution Equivalence

This PR does not change how requests are distributed across ADP ranks. The `ADPRouter` code (`route_requests`, `_balance_requests_across_ranks`) is completely untouched. The router's min-heap algorithm distributes requests evenly regardless of how many arrive per fetch cycle.
