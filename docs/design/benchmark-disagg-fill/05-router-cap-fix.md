# v2.1 — ADP Router Per-Rank Cap Fix

[< Back to index](README.md)

**Bug reference:** nvbug 6071070
**PR:** [#13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347)
**Prerequisite reading:** [`02-regression-investigation.md`](02-regression-investigation.md)

---

## 1. Problem

In benchmark disaggregated serving with ADP (Attention Data Parallelism),
the `DefaultADPRouter` and `KVCacheAwareADPRouter` compute a target
per-rank request count (`expected_num_active_requests`) using ceiling
division:

```python
expected_num_active_requests = max(
    (total_num_active_requests + num_new_requests + tp_size - 1) // tp_size,
    max(all_ranks_num_active_requests),
)
```

When `benchmark_req_queues_size = tp_size × max_batch_size` (the standard
benchmark configuration), a single bulk fetch can push
`expected_num_active_requests` above `max_batch_size`. The heap-based
balancer then assigns more requests to a rank than the scheduler can
physically process. These excess requests remain permanently in
`DISAGG_GENERATION_INIT` state because the scheduler never allocates
KV cache for them — triggering the fail-fast error:
`RequestError: Insufficient KV cache for gen-only benchmark mode`.

### Concrete example (from failing CI test)

- Model: Kimi-K2-Thinking-NVFP4 on GB200/GB300
- Config: `tp_size=32`, `max_batch_size=256`, `concurrency=8192`
- `TLLM_BENCHMARK_REQ_QUEUES_SIZE = min(256 × 32, 8192) = 8192`
- After first bulk fetch: `expected = ceil(8192/32) = 256` ✓
- After second partial fetch (e.g. 32 more arrive):
  `expected = max(ceil(8224/32), 256) = max(257, 256) = 257` ✗
- Rank assigned 257 requests, but scheduler caps at 256 → 1 request
  permanently stuck

---

## 2. Root cause analysis

The root cause is **not** in `PyExecutor`'s gate logic — it is in the
router's admission logic. The router computes a target that exceeds the
physical constraint (`max_num_active_requests`) it receives as a parameter
but does not enforce.

This makes the fix architectural: the router is the correct layer to
enforce per-rank capacity limits, as it is the admission point where
requests are distributed.

---

## 3. Fix

A one-line `min()` guard wrapping the existing computation:

```python
expected_num_active_requests = min(
    max(
        (total_num_active_requests + num_new_requests + tp_size - 1) // tp_size,
        max(all_ranks_num_active_requests),
    ),
    max_num_active_requests,  # <-- new: hard cap
)
```

Applied to both `DefaultADPRouter.route_requests` and
`KVCacheAwareADPRouter.route_requests` in
`tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py`.

### Why `min()` at this level is correct

1. **Enforces an existing invariant.** `max_num_active_requests` is passed
   to `route_requests` specifically to limit how many requests a rank
   handles. The function already uses it to guard pre-scheduled requests
   (line 227: `all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests`).
   The cap makes the balancer's target consistent with this guard.

2. **No behavioral change in normal operation.** Under typical serving
   loads, `expected` is well below `max_num_active_requests`. The `min()`
   is a no-op — it only activates during bulk arrivals in benchmark mode.

3. **Deterministic and stateless.** The fix is pure arithmetic — no new
   state, no timing dependency, no interaction with the executor loop.

---

## 4. Relationship to the state-based gate rewrite

The state-based gate rewrite ([`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md))
and the router cap fix solve different parts of the same failure:

| Aspect | Router cap | State-based gate rewrite |
|--------|--------------------|--------------------|
| **What it fixes** | Requests assigned to overflowing ranks | Gate not opening due to count imbalance |
| **Where** | `adp_router.py` | `py_executor.py` |
| **Scope** | 3 lines + comment | ~100 lines of predicate + fail-fast rework |
| **Risk** | Minimal — pure arithmetic guard | Moderate — changes state machine |

The router cap is necessary but not sufficient by itself. It prevents
new overflow INIT requests, but the original count-based gate can still
depend on an exact per-rank request count. The current PR therefore keeps
both fixes: state-based readiness for gate correctness and router capping
for admission correctness.

The follow-up structural cleanup is not another gate rewrite; it is the
separation of admission control from routing discussed in §6.

---

## 5. Test coverage

### Unit tests (added in this PR)

File: `tests/unittest/_torch/executor/test_benchmark_disagg.py`,
class `TestADPRouterPerRankCap`.

| Test | Scenario | Both routers? |
|------|----------|---------------|
| `test_expected_capped_at_max` | Rank 0 has 6 active, 3 new requests, max=4. Without cap: expected=6. With cap: expected=4. Rank 0 gets 0 new assignments. | ✓ (parameterized) |
| `test_no_rank_exceeds_max` | Near-capacity: ranks at [255, 254, 250, 253], 10 new requests, max=256. No rank exceeds 256 after routing. | ✓ (parameterized) |
| `test_cap_not_applied_when_below_max` | Empty ranks, 8 new requests, max=256. Expected=2 (no cap effect). All 8 assigned. | ✓ (parameterized) |

6 parameterized tests total (3 scenarios × 2 router implementations).

This complements the state-based gate tests and the fill-phase fail-fast
tests in [`06-fill-phase-fail-fast.md`](06-fill-phase-fail-fast.md).

### Integration validation

The failing test (`perf/test_perf_sanity.py::test_e2e[disagg-gen_only-wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_con8192_ccb-NIXL]`) runs in the QA/release sanity pipeline, not in pre-merge CI. It requires 36+ GPUs across 9 nodes (GB200/GB300). CI validation will occur via the QA pipeline after merge.

---

## 6. Future consideration: separating admission control from routing

The current design conflates two responsibilities in `ADPRouter.route_requests`:

1. **Admission control** — deciding *how many* requests each rank should
   accept (enforcing `max_num_active_requests`).
2. **Routing** — deciding *which* requests go to *which* rank (load
   balancing, prefix affinity, KV cache awareness).

The v2.1 fix patches the symptom (the target exceeding max) but doesn't
address the structural issue: the router computes `expected_num_active_requests`
as both a routing target and an admission limit, using a formula that can
violate the physical constraint.

A cleaner architecture would separate these:

- **Admission layer**: Computes per-rank available capacity as
  `max_num_active_requests - current_active[rank]`. This is a hard
  constraint, not a target. Requests exceeding total available capacity
  are deferred (left unrouted for the next iteration).
- **Routing layer**: Given the admitted request set and per-rank capacity
  budgets, distributes requests optimally (load balance, prefix affinity,
  etc.) without ever needing to reason about capacity limits.

Benefits of separation:

| Aspect | Current (combined) | Separated |
|--------|-------------------|-----------|
| **Capacity invariant** | Enforced by `min()` patch; easy to regress if formula changes | Structural — admission layer owns it unconditionally |
| **Excess requests** | Silently assigned then stuck in INIT | Explicitly deferred; caller can log or apply backpressure |
| **Testability** | Must test capacity + routing together | Each layer testable in isolation |
| **Router complexity** | Mixes capacity math with balancing heuristics | Router only does balancing within pre-approved budgets |

This refactor is out of scope for the current bug fix but would prevent
this class of bug from recurring. It could be combined with v4 (external
orchestrator) or done independently as a router-layer cleanup.

---

## 7. Files changed

| File | Change |
|------|--------|
| `tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py` | `min()` cap in `DefaultADPRouter.route_requests` and `KVCacheAwareADPRouter.route_requests` |
| `tests/unittest/_torch/executor/test_benchmark_disagg.py` | `TestADPRouterPerRankCap` class with 6 parameterized tests |
