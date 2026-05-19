# Issue #13318 — Fix Plan

This plan separates the **substantive** fix (close the admission-budget
overshoot) from the **hardening** fixes (no single batch error should brick
the server). They are independent: either alone is shippable, both together
close the gap.

---

## Track A — Substantive: Close the V1 Admission-Budget Overshoot

### A1. Port PR #12806 semantics to the C++ V1 MicroBatchScheduler

The customer's failing path is V1 = C++ MicroBatchScheduler with V1
KVCacheManager. PR #12806 patched the Python side on `feat/bench_y` only,
and even that surface has since been refactored on `main`. The C++ side
has its own copy of `reuse_adjusted_compute(chunk_size, reusable,
context_remaining)` and the same defect: it accounts for first-chunk reuse
adjustment but does not pre-subtract committed costs from non-first-chunk
context requests already in the batch.

**Change.** Before evaluating per-first-chunk reuse-adjusted budgets in the
C++ scheduler's admission loop, walk the current `scheduled_batch.context_requests`
and pre-subtract `context_chunk_size` for every request where
`!is_first_context_chunk`. Mirror the Python diff:

```python
# PR #12806 (Python, feat/bench_y branch)
for req in scheduled_batch.context_requests:
    if not req.is_first_context_chunk:
        remaining_budget -= req.context_chunk_size
```

C++ equivalent goes wherever the V1 scheduler computes
`remaining_budget = max_num_tokens - gen_tokens` (likely in
`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp` — verify file/line
during implementation).

**Why this is correct.** Non-first-chunk context requests have no reuse to
re-validate; their compute cost is fixed at `context_chunk_size` for the
chunks they advance through. The first-chunk reuse check must see the budget
with those costs already removed; otherwise it computes against a budget
that is larger than the budget actually available once the batch is
committed, and a last-chunk request with `reusable > 0` lands the batch over
`max_num_tokens` by exactly the amount that PR #12806 was pre-subtracting.

### A2. Audit the Python `_reuse_adjusted_compute` callers introduced by the refactor

`upstream/main:tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` calls
`_reuse_adjusted_compute` at L465/505/577/706/750/812, and
`scheduler_v2.py:453+` has its own `remaining_budget` path. PR #12806's
surface (the `remaining_budget` accumulator in `resource_manager.py`) is no
longer there.

**Change.** For each call site in `scheduler.py` and `scheduler_v2.py`, add
a focused unit test that exercises:

- A batch with at least one non-first-chunk context request and at least one
  first-chunk context request with `estimated_reusable_tokens > 0`.
- Verify that the post-admission sum of per-request append-side costs
  (`context_chunk_size` for each, regardless of `is_first_context_chunk`)
  does not exceed `max_num_tokens` minus generation tokens.

If any call site fails the test, apply the same pre-subtract semantic.
`scheduler_v2.py` may already get this right via its different shape
(`up_fits_budget = ... up_chunk_size <= remaining_budget` at L641) — confirm
or fix.

### A3. Cross-language invariant test

A test that runs the same scenario through V1 C++ scheduler and through
Python scheduler back-to-back and asserts both admit the same batch shape.
This catches the "fix landed in Python, customer runs C++" hazard that made
PR #12806 ineffective even when merged.

---

## Track B — Hardening: A Single Batch Error Must Not Brick the Server

### B1. Stop the second-stage assertion from killing `_event_loop_wrapper`

`tensorrt_llm/_torch/pyexecutor/py_executor.py:1600` and `:2599`:

```python
assert sample_state is not None, "Sampling failed"
```

This bare `assert` lives *outside* the forward-step try/except that already
captured the underlying error. When `_prepare_tp_inputs` raises, the inner
handler logs it, sets `sample_state = None`, and returns. The next line
then raises this assertion, which escapes the loop body and kills the thread.

**Change.** Replace the bare assertion with a per-batch recoverable path:

```python
if sample_state is None:
    # Forward step failed; the inner handler already logged and (if fatal)
    # set _fatal_error / classified via ErrorBudget. Fail the offending
    # batch's requests with the captured exception and continue the loop.
    self._fail_active_batch_with(captured_exception, scheduled_requests)
    continue
```

Requires the forward-step try/except to *expose* the captured exception
(today it logs and returns `None`); thread one channel through (e.g., an
instance attribute `self._last_forward_error` cleared each iteration, or
return `(sample_state, error)` from the call).

### B2. Top-level guard on `_event_loop_wrapper`

`py_executor.py:669` logs `"Error in event loop: {e}"` but does not catch
`BaseException` and walk the in-flight queues. Today the exception simply
escapes the thread.

**Change.** Wrap the loop body in:

```python
try:
    while not self._is_shutdown():
        ...
except BaseException as e:  # noqa: BLE001
    self._set_fatal_error(e)
    self._fail_all_inflight(e)
    raise
```

Where `_fail_all_inflight(e)`:

- Iterates `self.active_requests`, `self.waiting_queue`, and
  `self.executor_request_queue`.
- For each, locates the owning `GenerationResult` and completes it with the
  captured exception (or an `ErrorResponse` constructed from it).
- Drains the executor request queue so newly-arriving submissions also fail
  immediately with the same error rather than queuing indefinitely.

This is the missing complement to PR #12718: that PR set `_fatal_error` and
made `/health` self-terminate, but it relied on the loop body still running
long enough to drain the error queue. If the loop body itself dies, the
queue is never drained — the only artifact is a dead daemon thread and an
unbounded set of awaiting `GenerationResult` futures.

### B3. Promote prep-time accounting assertions to recoverable errors

`model_engine.py:2672` should distinguish two cases:

- **Assertion violated on a single request's contribution** — fail just that
  request with a clear error (`RequestError("admission accounting overshoot:
  request would contribute N tokens to a budget of M")`), drop it from
  `scheduled_requests`, and let the rest of the batch proceed.
- **Assertion violated globally for the batch shape** — fail the batch's
  requests with the same `RequestError` and continue the loop. Charge the
  `ErrorBudget` as `severe` (this is a scheduler bug, not a transient one;
  if it recurs immediately, the engine should crash and the orchestrator
  should restart the pod via PR #12718's mechanism).

Both paths route through the same per-request error channel
[PR #13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) added, so the
client sees a real `5xx` body instead of a permanent hang.

### B4. Make the liveness probe surface honest

`/v1/models` currently returns 200 regardless of executor state. K8s and
Dynamo operators are accustomed to using `/v1/models` as a cheap "alive"
probe; the OP did exactly that and concluded "naive liveness probes do not
detect the outage."

**Choose one (the first is preferred):**

- **Option 1 (preferred):** plumb `check_health()` into `/v1/models` and
  any other public "alive" surface. Returns 503 once `_fatal_error` is set.
- **Option 2:** add prominent documentation that `/health` is the only
  liveness probe; `/v1/models` is a model-metadata endpoint, not a
  liveness check.

---

## Sequencing

```
A1 (C++ scheduler pre-subtract)           B1 (no assertion thread-kill)
A2 (Python scheduler audit)                B2 (event-loop top-level guard)
A3 (cross-language invariant test)         B3 (recoverable prep assertions)
                                           B4 (honest liveness surface)
```

- **Ship A1+A3 together as a single PR.** A2 is its own audit-and-test PR
  that may or may not need a code change.
- **Ship B1+B2 together as a single PR.** They are the same architectural
  change (thread death → in-flight error propagation), split across two
  call sites. B3 is a small follow-up. B4 is its own (trivial) PR.
- A and B are independent. B alone makes this bug visible to clients
  (HTTP 5xx instead of permanent hang) without fixing the underlying
  overshoot. A alone fixes the overshoot without protecting against the
  next thread-killing assertion in the same family.
- Both should target `main` and be considered for backport to the active
  release branch the customer's image was based on.

## Test Plan

| Area | Test |
|---|---|
| A1 | C++ unit test: V1 MicroBatchScheduler with mixed first-chunk reusable + non-first-chunk context + near-full gen budget → total committed tokens ≤ `max_num_tokens`. |
| A2 | Python unit tests at each `_reuse_adjusted_compute` call site (same scenario). |
| A3 | Integration test that runs the same scenario through C++ V1 and Python schedulers and compares admitted batch shapes. |
| B1 | Unit test: inject `total_num_tokens > max_num_tokens` from `_prepare_tp_inputs`; assert the event-loop thread survives, the offending batch's requests fail with a real error, and the next iteration runs. |
| B2 | Unit test: inject an arbitrary `RuntimeError` inside the loop body; assert `_fatal_error` is set, all in-flight `GenerationResult` futures complete with the captured exception, and the executor request queue is drained. |
| B3 | OpenAI-server integration test: send a request that triggers the overshoot; assert the client sees a real HTTP 5xx with the assertion message in the body (not a timeout). |
| B4 | Integration test: after `_fatal_error` is set, `GET /v1/models` returns 503 (Option 1) **or** the docs explicitly state otherwise (Option 2). |
| Regression | Reproduce the OP's mnt=4096, partial=on, qps=7.0 scenario; verify the server keeps responding past round 2 with the A-track fix applied; verify a synthetic injected overshoot returns HTTP 5xx and the server continues serving with the B-track fix applied. |
