# Issue #13318 — Fix Plan

This plan separates the **substantive** fix (close the admission-budget
overshoot) from the **hardening** fixes (no single batch error should brick
the server). They are independent: either alone is shippable, both together
close the gap.

---

## Verified State on `main` (2026-05-19)

This section pins down the three claims the rest of the plan depends on,
against the live tree at `tensorrt_llm/_torch/pyexecutor/` and
`cpp/tensorrt_llm/batch_manager/` on current `main`. Line numbers are from
the same snapshot used to author the rest of this document.

### V1. PR #12718 has merged

PR #12718's primitives are present on `main` under the following names:

| Primitive | Location |
| --- | --- |
| `_error_queue` (background error sink) | `tensorrt_llm/executor/executor.py:99`; consumed by `_handle_background_error` at `executor.py:259-296` |
| `ManagedThread` (routes task exceptions to `_error_queue`) | `tensorrt_llm/llmapi/utils.py:311-358` (`run()` catches `Exception` in the loop and `self.error_queue.put(e)`) |
| `await_response_thread` (drains `PyExecutor.responses` to per-client queues) | Started as `ManagedThread(... error_queue=self._error_queue)` in `tensorrt_llm/executor/worker.py:72-75` |
| HTTP self-terminate on fatal | `signal.raise_signal(signal.SIGINT)` from OpenAI handlers on `CppExecutorError` at `tensorrt_llm/serve/openai_server.py:1160, 1257, 1456, 1695` |
| `LLM._check_health()` | `tensorrt_llm/llmapi/llm.py:990` — returns `not self._executor.is_shutdown()` |
| `Executor.is_shutdown()` | `tensorrt_llm/executor/executor.py:298` — returns `self.doing_shutdown` |

**Why this machinery does not catch this bug.** Three independent reasons:

1. The `PyExecutor` event loop is started as a **plain** `threading.Thread`,
   not a `ManagedThread` (`py_executor.py:698-700`):
   ```python
   self.worker_thread = threading.Thread(
       target=self._event_loop_wrapper, daemon=True)
   self.worker_thread.start()
   ```
   When `_event_loop_wrapper` re-raises (`py_executor.py:661-664`), the
   exception is lost to the thread's default handler. Nothing pushes it
   into `_error_queue`, so `_handle_background_error` never sees it.
2. `LLM._check_health()` reads `Executor.is_shutdown()` which reads
   `self.doing_shutdown` — a flag flipped **only** by an explicit call to
   `Executor.shutdown()`. `PyExecutor.is_shutdown` (a separate boolean set
   in `_executor_loop_cleanup` when the event loop dies) is **never read by
   `_check_health`**. So `/health` continues to return 200 after a thread
   death.
3. The escaping exception is `AssertionError`, not `CppExecutorError`. The
   HTTP handlers' `signal.raise_signal(SIGINT)` path is gated on
   `except CppExecutorError`, so the self-terminate path never fires for
   this failure mode.

PR #12718 narrows the **engine-death** failure shape. This bug's failure
shape is **executor-thread-death-between-iterations**, which neither the
classification path (`_handle_background_error`) nor the self-terminate
path (`CppExecutorError → SIGINT`) is wired to observe.

### V2. What actually happens to in-flight responses on thread death

Read in order; line numbers in current `main`.

1. `_prepare_tp_inputs` raises `AssertionError: total_num_tokens > max_num_tokens`
   (`model_engine.py:2667-2670`).
2. The forward-step `try / except Exception` catches it, logs
   `"Encountered an error in forward function: …"`, and calls
   `self._handle_errors(error_msg)` with `requests=None`
   (`py_executor.py:3411-3417`).
3. `_handle_errors(error_msg, requests=None)` (`py_executor.py:3546-3569`):
   - Builds an `LlmResponse(error_msg=…)` for **every** request currently in
     `self.active_requests` (not just the offending batch — `requests=None`
     means "fail all active").
   - Calls `_enqueue_responses(...)`, which writes the responses into
     `self.responses[req_id]` and notifies `self.response_cv`
     (`py_executor.py:3660-3673`).
   - Clears `self.active_requests`; terminates each failed request.
4. `_forward_step` returns `None`; control returns to the loop body where
   `sample_state = None`.
5. The bare assertion fires: `assert sample_state is not None, "Sampling failed"`
   (`py_executor.py:1491` in `_executor_loop_pp` and `:2444` in
   `_executor_loop_overlap`). This is **outside** the forward-step
   `try / except` from step 2.
6. The new `AssertionError` propagates to `_event_loop_wrapper`'s
   `except Exception` (`py_executor.py:661-664`), which logs and re-raises.
7. `_event_loop_wrapper`'s `finally:` clause runs
   `_executor_loop_cleanup` (`py_executor.py:1236-1247`), which sets
   `self.is_shutdown = True` and notifies `response_cv`.
8. The worker thread terminates. `LLM._check_health()` still returns
   `True` because it reads `Executor.doing_shutdown`, not
   `PyExecutor.is_shutdown` (see V1).

`await_response_thread` is a separate `ManagedThread` and keeps polling
`engine.await_responses(timeout=0.1s)` → `PyExecutor.await_responses` →
`_await_any_response`. Its predicate is
`len(self.responses) > 0 or self.is_shutdown` (`py_executor.py:3831-3836`),
so it wakes up promptly — first to deliver the error responses queued in
step 3, then on every subsequent poll to return `[]` (because
`self.is_shutdown` is `True` and `self.responses` is empty).

**Therefore the response-delivery story bifurcates:**

| Request class | What happens to its `await promise.aresult()` |
| --- | --- |
| In-flight at the moment of assertion | Receives the `LlmResponse(error_msg=…)` enqueued in step 3. `handle_for_worker` (`base_worker.py:792-820`) routes it to the per-client `_SyncQueue`/`Queue`; `result.py:855-857` unblocks. **HTTP client gets a 5xx with the assertion message in the body.** |
| Submitted *after* the thread dies | `executor_request_queue.enqueue_requests` (`executor_request_queue.py:110-114`) accepts it blindly — there is **no** liveness check at submit time. The dead event loop never picks it up. `_await_single_response` (`py_executor.py:3843-3855`) waits on `key_has_response = id in self.responses.keys()` with **no** `is_shutdown` short-circuit. **HTTP client hangs forever.** |
| Liveness probe (`/health`) | Returns 200, because `LLM._check_health` reads `Executor.doing_shutdown` (still `False`), not `PyExecutor.is_shutdown` (`True` from step 7). |
| Liveness probe (`/v1/models`) | Returns 200 unconditionally (no health check). |

Two corollaries for the rest of this plan:

- The per-request error channel **does work** for the batch that triggers
  the assertion. Item B3 below therefore does not need to build that
  channel; it only needs to use it correctly (fail the offending batch's
  requests with a typed error, not the entire `active_requests` list).
- The wedge is "everything submitted after the thread dies." Items B1+B2
  remove the thread death; items B4+V1-derived fixes (`_check_health`
  reading the right flag, `/v1/models` consulting `_check_health`) make
  the wedge visible to operators when items B1+B2 are not enough.

### V3. C++ admission-budget call sites for Track A1

`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp` is the V1 surface
the customer's failing path actually runs:

```cpp
// L33-44 — semantically identical to the Python _reuse_adjusted_compute
static SizeType32 reuse_adjusted_compute(
    SizeType32 chunkSize, SizeType32 reusable, SizeType32 contextRemaining)
{
    if (reusable <= 0)               return chunkSize;
    if (reusable + chunkSize < contextRemaining) return chunkSize;
    return std::max<SizeType32>(0, contextRemaining - reusable);
}
```

Three accumulator sites that exhibit the PR #12806 defect:

| Site | What it does |
| --- | --- |
| `fitDraftTokens` — L66-68 | Walks `contextsToBeChunked`; sums `reuse_adjusted_compute(...)` into `numCtxTokens` to decide draft-token discard. `reusable` is set to `0` for non-first chunks (correct guard for the helper) but the sum uses the reuse-adjusted value, not `chunkSize` — same defect family. |
| `operator()` non-chunked branch — L358-378 | For each `ContextInitState` request, `contextCompute = reuse_adjusted_compute(...)` is added to `batchNumTokens`. |
| `operator()` chunked branch — L385-386, L459-461 | Same accumulation in both the first pass and the post-chunk-adjust loop. Last-chunk first-context-chunk requests with `reusable > 0` pay `contextRemaining - reusable` here, but the Python append side spends `context_chunk_size`. |

There is **no C++ analog of the prep-time `total_num_tokens > max_num_tokens`
assertion**. The C++ scheduler has `TLLM_CHECK_WITH_INFO` macros at L343
and L369 that check per-request `computeTokens <= mMaxContextLength`, but
nothing checks the per-batch budget across all admitted requests. The
Python `_prepare_tp_inputs` assertion (`model_engine.py:2667-2670`) is the
**first place** the overshoot is observed.

That has two consequences for the plan:

- **A1's exact patch site** is the same three accumulator sites above. The
  semantic mirror of PR #12806 is: for every non-first-chunk request that
  will end up in this batch, pre-subtract `context_chunk_size` from
  `batchNumTokens` before evaluating the first-chunk reuse-adjusted budget
  for any admission decision against `maxNumTokensRuntime`.
- **There is no native-side error class for B3 to catch.** When B3
  introduces `BatchAdmissionError`, the Python side is the sole producer.
  No nanobind binding work is required.

---

## Track A — Substantive: Close the V1 Admission-Budget Overshoot

### A1. Port PR #12806 semantics to the C++ V1 MicroBatchScheduler

The customer's failing path is V1 = C++ MicroBatchScheduler with V1
KVCacheManager. PR #12806 patched the Python side on `feat/bench_y` only,
and even that surface has since been refactored on `main`. The C++ side
has its own copy of `reuse_adjusted_compute(chunkSize, reusable,
contextRemaining)` and the same defect: it accounts for first-chunk reuse
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

**Concrete C++ patch sites** (verified — see V3 above):
`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp`

- Line 33-44 — `reuse_adjusted_compute`: helper signature/behavior is
  identical to Python; no change needed.
- Line 358-378 / Line 385-386 — `operator()`: the per-request admission
  decision against `maxNumTokensRuntime` (`if (maxNumTokensRuntime &&
  batchNumTokens + computeTokens > maxNumTokensRuntime.value()) break;`).
  Before this check fires for the **first** chunk of a request with
  `getEstimatedReusableTokens() > 0`, `batchNumTokens` must already reflect
  the actual per-iteration cost of every non-first-chunk request previously
  admitted into this batch — which means adding `context_chunk_size` (not
  `reuse_adjusted_compute(...)`) for those requests.
- Line 459-461 — post-chunk-adjust loop: same accumulation defect; needs
  the same correction.
- Line 66-68 — `fitDraftTokens`: independent code path that decides draft
  discards; audit and apply the same correction. Probably small
  customer-visible impact in V1 (V1 has limited draft support) but should
  be consistent.

**Why this is correct.** Non-first-chunk context requests have no reuse to
re-validate; their compute cost is fixed at `context_chunk_size` for the
chunks they advance through. The first-chunk reuse check must see the budget
with those costs already removed; otherwise it computes against a budget
that is larger than the budget actually available once the batch is
committed, and a last-chunk request with `reusable > 0` lands the batch over
`max_num_tokens` by exactly the amount that PR #12806 was pre-subtracting.

**Assumption / risk note.** Pre-subtracting `context_chunk_size` for a
non-first-chunk request assumes it actually consumes that many tokens at
forward time. This holds for V1 (no early-exit between admission and
append in current code paths). If a future change introduces such an
early-exit, the pre-subtract becomes a slight over-budget reservation, not
a correctness bug.

### A2. Audit the Python `_reuse_adjusted_compute` callers introduced by the refactor

`upstream/main:tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` defines
`_reuse_adjusted_compute` at L321 and calls it at L441, L472, L545, L676,
L677, L720, L734, L746, L779 (9 call sites — the refactor since the
original investigation has added more), and
`scheduler_v2.py:451-467` has its own `remaining_budget` path. PR #12806's
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
(`min(remaining_budget, context_remaining) if remaining_budget is not None`
at L466-467) — confirm or fix.

### A3. Cross-language invariant test

A test that runs the same scenario through V1 C++ scheduler and through
Python scheduler back-to-back and asserts both admit the same batch shape.
This catches the "fix landed in Python, customer runs C++" hazard that made
PR #12806 ineffective even when merged.

---

## Track B — Hardening: A Single Batch Error Must Not Brick the Server

### Architectural note (per V2)

The forward-step `except Exception` (`py_executor.py:3411-3417`) already
calls `_handle_errors(error_msg)` with `requests=None`, which fails the
**entire** `active_requests` list with the captured message. Those
responses are written into `PyExecutor.responses` and reach the HTTP
client via the independent `await_response_thread`. So the in-flight batch
does receive a 5xx today.

The wedge is everything that follows: the bare assertion at
`py_executor.py:1491` / `:2444` kills the event-loop thread, no new
requests get scheduled, `_await_single_response` waits forever, and
`/health` keeps reporting 200. Track B closes that gap.

### B1. Stop the second-stage assertion from killing `_event_loop_wrapper`

`tensorrt_llm/_torch/pyexecutor/py_executor.py:1491` (inside
`_executor_loop_pp`) and `:2444` (inside `_executor_loop_overlap`):

```python
assert sample_state is not None, "Sampling failed"
```

This bare `assert` lives **outside** the forward-step `try / except` that
already captured the underlying error. When `_prepare_tp_inputs` raises,
the inner handler logs it, calls `_handle_errors` (which enqueues per-
request errors for all `active_requests`), and returns `None`. The next
line then raises this assertion, which escapes the loop body and kills the
thread.

**Change.** Replace each bare assertion with a per-batch recoverable path:

```python
if sample_state is None:
    # Forward step failed and the inner handler has already failed every
    # in-flight request via _handle_errors. Skip the rest of this iteration
    # and let the loop pick up new work.
    continue
```

This single edit is sufficient to prevent the customer's exact wedge — the
proximate cause of the thread death is the assertion, not anything else.
Ship it independently if helpful (smallest possible patch that closes the
reproduction).

### B2. Top-level guard on `_event_loop_wrapper`

`py_executor.py:652-666` catches `Exception`, logs `"Error in event loop:
{e}"`, and re-raises. The `finally:` runs `_executor_loop_cleanup` which
sets `self.is_shutdown = True` and notifies `response_cv`. But nothing
fails the requests left over in `self.active_requests` /
`self.waiting_queue` / `self.executor_request_queue.request_queue`, and
nothing surfaces the death to `LLM._check_health` (see V1).

**Change.** Convert the wrapper into an architectural fail-stop:

```python
def _event_loop_wrapper(self):
    try:
        enable_profiler = bool(os.environ.get(
            "TLLM_LINE_PROFILER_PATH")) and not self.is_warmup
        with host_profiler_context(enable=enable_profiler), \
             customized_gc_thresholds(self.garbage_collection_gen0_threshold):
            self.event_loop()
    except BaseException as e:               # was: except Exception
        logger.error(f"Error in event loop: {e}")
        logger.error(traceback.format_exc())
        self._fatal_error = e                # new — see B4
        self._fail_all_inflight(e)           # new — drains all queues
        raise
    finally:
        self._executor_loop_cleanup()
```

Where `_fail_all_inflight(e)`:

- Calls `self._handle_errors(str(e), requests=self.active_requests)` to
  fail any remaining active requests via the existing per-request error
  channel.
- Drains `self.executor_request_queue.request_queue` non-blockingly. For
  each `RequestQueueItem` pulled out, synthesize an `LlmResponse(
  request_id=..., error_msg=str(e), client_id=...)` and call
  `_enqueue_responses([(req_id, resp)])`. After draining, flip
  `self.executor_request_queue.active = False` so subsequent
  `enqueue_requests` calls fail fast instead of queuing indefinitely.
- Walks `self.waiting_queue` (and the PP `executed_batch_response_queue`
  when `dist.pp_size > 1`) and applies the same synthesis.

**Sync safety.** The dying thread may hold `self.response_cv` /
`self.enqueue_lock` re-entrantly. `threading.Condition` and `threading.Lock`
in CPython are re-entrant-safe only via `RLock`; `_enqueue_responses` uses
`with self.response_cv:` which acquires the underlying lock. Verify before
implementation that the lock guarding `response_cv` is an `RLock`, or
restructure `_fail_all_inflight` to snapshot the queue contents outside the
lock first.

This is the missing complement to PR #12718. That PR set up the
`_error_queue` / `_handle_background_error` / `ManagedThread` machinery so
that **ManagedThread** task failures self-classify and self-shut-down. But
the `PyExecutor` event loop is a plain `threading.Thread` (V1, reason 1),
so its exceptions never reach that channel. B2 either (a) bridges the
plain-thread death to the existing channel by routing through
`_handle_errors` / `_check_health`, or (b) — preferred — upgrades
`_event_loop_wrapper` itself to put its exception into `_error_queue`:

```python
except BaseException as e:
    self._handle_background_error_queue.put(e)  # via BaseWorker
    ...
```

Option (b) is the minimal touch and folds this failure mode into PR
#12718's classification path. Preferred path for the patch.

### B3. Promote prep-time accounting assertions to typed errors

`model_engine.py:2667-2670` should distinguish two cases:

- **Assertion violated on a single request's contribution** — fail just that
  request with a clear error (`BatchAdmissionError("admission accounting
  overshoot: request would contribute N tokens to a budget of M")`), drop
  it from `scheduled_requests`, and let the rest of the batch proceed.
- **Assertion violated globally for the batch shape** — fail the batch's
  requests with the same `BatchAdmissionError` and continue the loop.
  Charge `_handle_background_error` as severe (this is a scheduler bug,
  not a transient one; if it recurs immediately, the engine should crash
  and the orchestrator should restart the pod via the existing
  `_handle_background_error → shutdown` path at `executor.py:286`).

**Patch shape:**

```python
# model_engine.py
class BatchAdmissionError(RuntimeError):
    """Per-batch admission accounting violation; recoverable per-batch."""

# replace the bare assert at L2667-2670
if total_num_tokens > self.max_num_tokens:
    raise BatchAdmissionError(
        f"total_num_tokens ({total_num_tokens}) > max_num_tokens "
        f"({self.max_num_tokens}); admission accounting overshoot")
```

```python
# py_executor.py:3411-3417 — forward-step except
except BatchAdmissionError as e:
    logger.error(f"Admission accounting violation: {e}")
    self._handle_errors(
        str(e),
        requests=list(scheduled_requests.context_requests) +
                 list(scheduled_requests.generation_requests),
    )
    return None
except Exception as e:
    ... # existing path
```

Today's `_handle_errors(error_msg)` (called with `requests=None`) fails the
**entire** `active_requests` list, not just the offending batch. Passing
`requests=` scopes the failure to the batch that actually triggered it.

C++ side: V3 confirms no native producer exists for this error class, so
no nanobind binding work is required.

The client sees a real `5xx` body with the typed message instead of either
a permanent hang (today, for subsequent requests) or a 5xx for unrelated
in-flight requests (today, for the over-broad `requests=None` fan-out).

### B4. Make the liveness probe surface honest

Two distinct holes — both must be closed:

1. **`LLM._check_health()` reads the wrong flag.**
   `tensorrt_llm/llmapi/llm.py:990` reads `self._executor.is_shutdown()`,
   which returns `self.doing_shutdown` (`executor.py:298`). That flag is
   only flipped by an explicit `Executor.shutdown()` call. The
   `PyExecutor.is_shutdown` boolean set in `_executor_loop_cleanup` is
   **not** observed.

   **Change.** Add a dedicated `_fatal_error` field set in B2's
   `except BaseException`; extend `_check_health()`:

   ```python
   def _check_health(self) -> bool:
       if not hasattr(self, "_executor") or self._executor is None:
           return False
       if self._executor.is_shutdown():
           return False
       # PyExecutor-specific: event loop has died but Executor.shutdown()
       # was not called → still unhealthy.
       if getattr(self._executor, "_fatal_error", None) is not None:
           return False
       return True
   ```

2. **`/v1/models` does not consult `_check_health()`.** Operators commonly
   probe `/v1/models` as a cheap "alive" check; the OP did exactly that
   and concluded "naive liveness probes do not detect the outage." Today
   `openai_server.py:767-769`:
   ```python
   async def get_model(self) -> JSONResponse:
       model_list = ModelList(data=[ModelCard(id=self.model)])
       return JSONResponse(content=model_list.model_dump())
   ```

   **Choose one (the first is preferred):**

   - **Option 1 (preferred):** wrap `get_model` with a `_check_health()`
     guard mirroring `health()` at L704. Returns 503 when the executor is
     unhealthy. Repeat for the two other `/v1/models` registrations in
     `register_mm_encoder_routes` (L646) and `register_visual_gen_routes`
     (L670).
   - **Option 2:** add prominent documentation that `/health` is the only
     liveness probe; `/v1/models` is a model-metadata endpoint, not a
     liveness check.

   Option 1 is the same trivial guard for three routes and removes the
   operator footgun. Recommended.

---

## Sequencing

```
A1 (C++ scheduler pre-subtract)           B1 (no assertion thread-kill)
A2 (Python scheduler audit)                B2 (event-loop top-level guard)
A3 (cross-language invariant test)         B3 (typed admission error)
                                           B4 (honest liveness surface)
```

- **Ship A1+A3 together as a single PR.** A2 is its own audit-and-test PR
  that may or may not need a code change.
- **B1 alone** prevents the customer's exact wedge (it is the proximate
  cause). It is a one-edit change at two call sites and is independently
  testable; ship first if the architectural changes in B2 need more
  review.
- **Ship B1+B2+B4 together as a single PR.** Together they turn any
  escaping exception into "loop dies cleanly, all in-flight requests get
  5xx, `/health` and `/v1/models` flip to 503." B3 is a follow-up that
  hardens the family of prep-time accounting bugs.
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
| B1 | Unit test: inject `total_num_tokens > max_num_tokens` from `_prepare_tp_inputs`; assert (a) the event-loop thread survives, (b) the offending batch's requests fail with a real error, (c) the next iteration runs. |
| B2 | Unit test: inject an arbitrary `RuntimeError` inside the loop body; assert (a) `_fatal_error` is set, (b) all in-flight `GenerationResult` futures complete with the captured exception, (c) the `executor_request_queue` is drained, (d) subsequent `enqueue_requests` returns immediately with the error rather than blocking. |
| B3 | OpenAI-server integration test: send a request that triggers the overshoot; assert (a) the offending request's client sees a real HTTP 5xx with `BatchAdmissionError` in the body (not a timeout), (b) unrelated in-flight requests are **not** failed (verify the fix to over-broad `requests=None` fan-out). |
| B4 | Integration test: after `_fatal_error` is set, both `GET /health` and `GET /v1/models` return 503. Repeat for the `register_mm_encoder_routes` and `register_visual_gen_routes` `/v1/models` registrations. |
| Regression | Reproduce the OP's mnt=4096, partial=on, qps=7.0 scenario; verify the server keeps responding past round 2 with the A-track fix applied; verify a synthetic injected overshoot returns HTTP 5xx and the server continues serving with the B-track fix applied. |
