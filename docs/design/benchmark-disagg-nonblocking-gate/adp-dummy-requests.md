# ADP Dummy Request Handling

## Why Dummies Exist

With Attention Data Parallelism (ADP), each TP rank processes a different subset of requests (unlike standard TP where all ranks process the same batch). The model forward pass still contains NCCL collective operations (allgather, broadcast) that require all ranks to participate. If a rank has zero requests, it can't participate in collectives, which would either deadlock or cause `_can_queue` to block the entire forward pass.

The dummy request is a single-token placeholder that lets idle ranks participate in collectives. Its output is discarded. Dummies are cheap (one KV cache slot per idle rank) and localized (only in the executor, invisible to the model code).

## The Permanence Problem

Dummies follow an add-forward-terminate lifecycle: added before the forward pass, used during collectives, terminated after the forward pass in `_update_request_states_tp`. However, during the benchmark disagg fill phase, the `can_forward` gate prevents forward passes from running. A dummy added during the fill phase is never terminated — it permanently occupies a KV cache slot for the rest of the fill phase.

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

Encapsulates the skip decision with clear early returns:

```python
if not self.is_benchmark_disagg or self.is_warmup:
    return False           # not in benchmark disagg mode
if num_schedulable_requests > 0:
    return False           # some requests are ready — don't skip

fill_phase_complete = (self.num_fetch_requests
                       >= self.benchmark_req_queues_size)
some_ranks_permanently_empty = (self.enable_attention_dp
                                and self.benchmark_req_queues_size
                                < self.dist.tp_size)

if fill_phase_complete and some_ranks_permanently_empty:
    return False           # allow dummy — rank will never get a real request

return True                # skip dummy in all other benchmark disagg cases
```

## The Skip Logic Explained

Dummies are skipped throughout the benchmark disagg fill phase because:

1. **The `can_forward` gate prevents forward-pass collectives during fill.** Temporarily-empty ranks are safe because no collectives run.
2. **Dummies added during fill are never cleaned up.** The termination logic only runs after a forward pass. A stuck dummy permanently wastes a KV cache slot.
3. **More requests will arrive.** During fill, temporarily-empty ranks will eventually receive real requests from the ADP router.

The **one exception** is the terminal case: all benchmark requests have been fetched (`fill_phase_complete`) AND there are fewer total requests than TP ranks (`some_ranks_permanently_empty`). In that case, some ranks will **never** receive a real request and need a permanent dummy for forward-pass collectives once the gate opens.

## Why Not Add Dummies Early?

Adding dummies on temporarily-empty ranks during fill is harmful:
- The dummy permanently occupies a KV cache slot (never cleaned up during fill)
- A real request arriving later at that rank must compete with the stuck dummy for KV cache
- The dummy doesn't serve any purpose — the gate blocks all collectives anyway

## Duplicate Prevention

Once a dummy is added, `num_active_request` becomes > 0, so the condition `num_active_request == 0 and expected_num_active_requests > 0` is False. No second dummy is ever added. After the gate opens, the normal add-forward-terminate lifecycle handles cleanup, and the dummy is removed after each forward pass.

## Distribution Equivalence

This PR does not change how requests are distributed across ADP ranks. The `ADPRouter` code (`route_requests`, `_balance_requests_across_ranks`) is completely untouched. The router's min-heap algorithm distributes requests evenly regardless of how many arrive per fetch cycle.
