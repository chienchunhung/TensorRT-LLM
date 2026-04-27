# v2.1 — Fill-Phase Fail-Fast Suppression

[< Back to index](README.md)

**Bug references:** nvbug 6071070 / nvbug 6093911  
**PR:** [#13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347)  
**Prerequisite reading:** [`02-regression-investigation.md`](02-regression-investigation.md), [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md), [`05-router-cap-fix.md`](05-router-cap-fix.md)

---

## 1. Problem

After the state-based gate rewrite and ADP router cap, the wide-EP
Kimi-K2-Thinking gen-only test still failed with:

```text
Insufficient KV cache for gen-only benchmark mode:
61 request(s) are waiting for KV cache allocation but the scheduler could
not fit any of them. Increase free_gpu_memory_fraction or reduce
TLLM_BENCHMARK_REQ_QUEUES_SIZE (currently 8192).
```

The log confirms the error came from the PR #12206 fail-fast path in
`_prepare_and_schedule_batch()`.

---

## 2. Root cause

The executor loop calls `_prepare_and_schedule_batch()` before
`_check_benchmark_disagg_gate()`:

```python
scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
...
can_forward, should_retry = self._check_benchmark_disagg_gate(
    scheduled_batch, can_forward)
```

That means the PR #12206 fail-fast can terminate requests before the
state-based gate has a chance to observe that the fill phase is still in
progress.

During benchmark fill, INIT requests are expected:

1. Some requests have completed KV transfer and occupy KV cache.
2. Other requests are still in `DISAGG_GENERATION_INIT`, waiting for
   their KV transfer to arrive from CTX.
3. The scheduler may not be able to fit those INIT requests in the same
   iteration because KV cache is currently occupied by already-transferred
   requests.

This is a transient fill-phase state, not genuine KV insufficiency. The
fail-fast should only diagnose genuine insufficiency once the fill gate
has opened.

---

## 3. Fix

Suppress the fail-fast while `_benchmark_fill_phase_active` is true:

```python
if (self.benchmark_req_queues_size > 0 and not self.is_warmup
        and not self._benchmark_fill_phase_active
        and not fitting_disagg_gen_init_requests):
    ...
```

`_benchmark_fill_phase_active` is cleared only when
`_check_benchmark_disagg_gate()` opens the state-based gate. After that
point, a stuck INIT request means the request is no longer waiting for
normal benchmark fill progress; it is a real KV-capacity failure and the
fail-fast remains valid.

---

## 4. Why this is safe

| Case | Before | After |
|------|--------|-------|
| Fill phase, transfers still in progress | Fail-fast may kill all requests prematurely | Fail-fast suppressed; gate keeps polling |
| Fill phase complete, all requests ready | Gate opens and generation starts | Same |
| Fill phase complete, INIT requests remain | Fail-fast reports insufficient KV cache | Same |
| Warmup | Fail-fast bypassed | Same |

This preserves the intent of PR #12206 — avoiding silent hangs when GEN
KV cache is genuinely insufficient — while preventing false positives
during the benchmark fill barrier.

---

## 5. Test coverage

Added to `tests/unittest/_torch/executor/test_benchmark_disagg.py`:

| Test class | Tests | Coverage |
|------------|-------|----------|
| `TestFailFastSuppressedDuringFill` | 5 | Fill-phase suppression, post-fill fail-fast, warmup behavior |
| `TestFillPhaseEndToEnd` | 1 | Small reproducer for the complete failure sequence: all fetched, INIT requests remain, scheduler cannot fit them, no kill during fill, gate opens after transfers complete, post-fill stuck INIT kills |

The full benchmark-disagg unit test file now has 43 passing tests.

```text
43 passed, 3 warnings
```

---

## 6. Relationship to the other fixes

The final PR #13347 fix is three-part:

1. **State-based gate**: removes dependency on exact ADP request counts.
2. **ADP router cap**: prevents assigning more requests to a rank than
   it can schedule.
3. **Fill-phase fail-fast suppression**: prevents PR #12206 from firing
   before the state-based gate can observe fill progress.

All three are needed for the wide-EP Kimi-K2-Thinking gen-only failure.
