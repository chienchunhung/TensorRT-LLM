# Design and Implementation

## Core Principle

Move the fill-complete check **out of** `_prepare_and_schedule_batch` and into the outer executor loop as a **non-blocking gate**. Each main-loop iteration performs its normal work (fetch, schedule, service transfers), then checks whether the gate should open. If not, it sleeps briefly and retries.

## New Components

### `is_benchmark_disagg` Attribute

A derived boolean computed once during `PyExecutor.__init__`:

```python
self.is_benchmark_disagg = (self.benchmark_req_queues_size > 0
                            and self.kv_cache_transceiver is not None)
```

Replaces the repeated compound condition that appeared in four locations with inconsistent formatting.

### `_is_benchmark_disagg_fill_complete(scheduled_batch) -> bool`

Checks whether the total number of *real* (non-dummy) generation requests has reached `benchmark_req_queues_size`:

- **Non-ADP path:** Counts local generation requests, excluding `is_attention_dp_dummy`.
- **ADP path:** Same local count, then aggregates across TP ranks via `dist.tp_allgather`.

Raises `RuntimeError` if called outside benchmark disagg mode (defensive precondition).

### `_check_benchmark_disagg_gate(scheduled_batch, can_forward) -> tuple[bool, bool]`

Consolidates the gate logic shared by both `_executor_loop` and `_executor_loop_overlap`. Returns `(can_forward, should_retry)`:

- When `is_warmup` is True: bypasses the gate (warmup must proceed normally).
- When `can_forward` is already True: no-op (gate is latching).
- When fill is incomplete: sleeps 1 second, returns `should_retry=True`.

The `tuple[bool, bool]` return exists because `can_forward` and `should_retry` are independent during warmup: `can_forward` stays False (so the gate activates after warmup ends) while `should_retry` is False (so the warmup forward pass proceeds). Collapsing to a single return would require either moving the warmup check to both callers (duplicating logic) or setting `can_forward=True` during warmup (permanently disabling the gate).

## Executor Loop Integration

Both `_executor_loop` and `_executor_loop_overlap` follow the same pattern:

```python
can_forward = not self.is_benchmark_disagg   # initially gated

while True:
    scheduled_batch, _ = self._prepare_and_schedule_batch()
    # ↑ non-blocking: fetches once, services transfers, returns
    # ↑ PR #12206's stuck-request detection also runs here

    can_forward, should_retry = self._check_benchmark_disagg_gate(
        scheduled_batch, can_forward)
    if should_retry:
        continue   # retry — transfers can progress between iterations

    # ... proceed with forward pass ...
```

## Interaction with PR #12206

The two fixes handle orthogonal failure modes and coexist naturally:

| Scenario | Which fix handles it | Mechanism |
|----------|---------------------|-----------|
| CTX KV cache too small to send all at once | **This PR (#12208)** | Non-blocking gate allows incremental progress |
| GEN KV cache too small to hold all requests | **PR #12206** | Detects stuck INIT requests, returns error |
| Both CTX and GEN have enough capacity | Neither fires | Gate opens normally after all transfers complete |

The execution order within each loop iteration is:
1. `_prepare_and_schedule_batch` — fetches requests, services transfers, checks PR #12206's stuck-request condition
2. `_check_benchmark_disagg_gate` — this PR's gate, checks fill-complete count

If PR #12206's check fires (GEN KV insufficient), `_prepare_and_schedule_batch` returns `None`, the loop breaks, and the gate is never reached. If the gate fires (CTX still sending), the loop retries, giving `_prepare_and_schedule_batch` another chance to fetch and service transfers.

## Request Lifecycle in Benchmark Disagg Mode

```
Iteration 1:  fetch 2 requests  →  KV transfer starts  →  gate: 2/8, retry
Iteration 2:  fetch 2 more      →  transfers progress   →  gate: 4/8, retry
Iteration 3:  fetch 2 more      →  iter-1 transfers done →  gate: 6/8, retry
Iteration 4:  fetch 2 more      →  iter-2 transfers done →  gate: 8/8, OPEN
Iteration 5:  forward pass executes with full batch
```
