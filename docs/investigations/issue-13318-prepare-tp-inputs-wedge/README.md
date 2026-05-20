# Issue #13318: Scheduler Deadlock on `total_num_tokens > max_num_tokens` Assertion

- **Severity:** P1 (silent server wedge; only `docker kill` recovers)
- **Reported by:** External user (GitHub issue)
- **Tracking:** [GitHub #13318](https://github.com/NVIDIA/TensorRT-LLM/issues/13318)
- **Affected configuration:** PyTorch backend, V1 C++ MicroBatchScheduler + V1 KVCacheManager
  (`use_python_scheduler=False`, `use_kv_cache_manager_v2=False`),
  `enable_chunked_prefill`, `enable_block_reuse`, `enable_partial_reuse`,
  KV host offload, FP8 KV cache
- **Affected images:** TRT-LLM `main` @ `fdfaeb27` (2026-04-20) and `main` @
  `0d2bea7c` (2026-04-22). Also reproduced on 1.3.0rc12.
- **Date:** 2026-05-19
- **Status:** Open — fix plan drafted (see [`fix-plan.md`](fix-plan.md))

---

## Executive Summary

Under `enable_block_reuse + enable_partial_reuse + enable_chunked_prefill` with
KV host offload, an admission-budget vs. token-accounting mismatch lets the V1
C++ MicroBatchScheduler admit a batch whose actual `position_ids` length exceeds
`max_num_tokens`. The Python forward path catches the resulting
`AssertionError` in `_prepare_tp_inputs`, but the **downstream**
`assert sample_state is not None, "Sampling failed"` in the executor loop body
is not inside the same `try/except`. It escapes `_event_loop_wrapper`, kills
the executor thread, and leaves the OpenAI server queue parked on a
`GenerationResult` that nobody will ever complete. The HTTP layer keeps
returning 200 on `/v1/models`, so naive liveness probes never notice.

The two prior zombie-worker / disagg-wedge fixes ([PR #12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718),
[PR #13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119)) **do not close
this case**. Both are now on `main` (the customer's image predates them, but
re-running on current `main` still wedges — see `fix-plan.md` §V1 for the
verified primitives and the three reasons they miss this bug). The bug sits
in the seam between "fatal engine error" (covered by #12718) and
"per-request error" (covered by #13119): the executor thread dies *between
iterations*, which neither layer owns.

The substantive root cause is the same one [PR #12806](https://github.com/NVIDIA/TensorRT-LLM/pull/12806)
attempted to fix on the `feat/bench_y` branch (pre-subtract non-first-chunk
context costs in the reuse-aware budget check) — but that PR never reached
`main`, and the surface it patched has since been refactored. The fix needs to
be ported to the **C++** V1 MicroBatchScheduler (the path the customer actually
runs), and the Python `_reuse_adjusted_compute` callers introduced by the
refactor need to be audited for the same invariant.

---

## Reproduction (from the issue)

| Run | Config | QPS | Stable P50 e2e | Failure |
|---|---|---|---|---|
| 1 | mnt=4096, partial=on | 5.2 | 5–6 s | clean across 12 rounds |
| 2 | mnt=4096, partial=on | 7.0 | — | ~R2: `total_num_tokens=6986 > 4096` → hang |
| 3 | mnt=8192, partial=on | 6.0 | ~5 s | clean across 12 rounds |
| 4 | mnt=8192, partial=on | 40 (stress) | — | clean across tens of thousands of requests |
| 5 | mnt=4096, partial=off | 5.2 | 69–81 s (15×) | R4: `total_num_tokens=4118 > 4096` → hang |
| 6 | mnt=8192, partial=off | 6.0 | 81–83 s | R4: `total_num_tokens=8196 > 8192` → hang |

Overshoots observed: **+1, +4, +22, +2890** tokens.

Takeaways from the matrix:

- Raising `max_num_tokens` from 4096 to 8192 hides the bug in practice but is
  not a root-cause fix — the budget math is still wrong, the customer's load
  just doesn't push close enough to the new ceiling to expose it.
- `enable_partial_reuse=false` makes it **worse**, not better: the assertion
  fires at lower QPS, in earlier rounds, and P50 regresses 15–20×. So "disable
  partial reuse" is not a workaround.

---

## Root Cause Analysis

### The Admission-Side Budget (C++ V1 MicroBatchScheduler)

V1 admits context requests using a reuse-adjusted compute budget:

```text
reuse_adjusted_compute(chunk_size, reusable, context_remaining)
  non-last chunk: returns chunk_size
  last chunk:     returns max(0, context_remaining - reusable)
```

For a **last-chunk** context request with `reusable > 0`, the budget the
scheduler "pays" is `context_remaining - reusable` — i.e., the post-reuse
*new* tokens to compute.

### The Append-Side Cost (Python `_prepare_tp_inputs`)

`tensorrt_llm/_torch/pyexecutor/model_engine.py:2667-2670` (current `main`):

```python
total_num_tokens = len(position_ids)
assert total_num_tokens <= self.max_num_tokens, (
    f"total_num_tokens ({total_num_tokens}) should be less than or equal "
    f"to max_num_tokens ({self.max_num_tokens})"
)
```

`position_ids` is built by appending, per context request,
`range(begin_compute, begin_compute + context_chunk_size)`. The actual
length appended for that same last-chunk request equals the **full chunk
size**, not the reuse-adjusted size that the admission budget paid.

### The Mismatch

When a last-chunk context request with `reusable > 0` is admitted alongside
near-full generation traffic, the admission side reserves
`context_remaining - reusable` tokens but the Python append side spends
`context_chunk_size`. The delta — typically a few tokens around block
boundaries, occasionally large in pathological alignments — is the observed
overshoot (`+1`, `+4`, `+22`, `+2890`).

`enable_partial_reuse=false` worsens this because the actual append spans more
of the chunk, so the absolute overshoot grows.

### Why the Mismatch Bricks the Server (Not Just the Batch)

The actual failure flow (line numbers verified against current `main`; see
`fix-plan.md` §V2 for the full 8-step trace):

```
1. model_engine.py:2668     AssertionError: total_num_tokens > max_num_tokens
                            (raised inside the forward step)
2. forward-step try/except  catches it; logs "Encountered an error in forward
   (py_executor.py:3411)    function"; calls _handle_errors(error_msg,
                            requests=None) which fails EVERY active request
                            with the error message and enqueues responses;
                            returns sample_state = None
3. py_executor.py:1491      assert sample_state is not None, "Sampling failed"
   or :2444                 (NOT inside the forward-step try/except)
                            => AssertionError escapes the loop body
4. _event_loop_wrapper      catches Exception, logs, re-raises; finally:
   (py_executor.py:652)     calls _executor_loop_cleanup which sets
                            PyExecutor.is_shutdown=True. Thread dies.
5. await_response_thread    (a separate ManagedThread) drains the queued
                            error responses; in-flight HTTP requests
                            receive a 5xx with the assertion message.
6. /v1/chat/completions     submitted AFTER step 4: enqueue_requests accepts
                            blindly (no liveness check); the dead event loop
                            never picks them up; _await_single_response has
                            no is_shutdown short-circuit => permanent hang.
7. /health                  still returns 200, because LLM._check_health
                            reads Executor.doing_shutdown (False) not
                            PyExecutor.is_shutdown (True).
   /v1/models               still returns 200 (no health check at all).
```

The crucial gap is steps 3–4: the bare `assert sample_state is not None`
sits outside the forward-step try/except, so a recoverable per-batch error
becomes a thread death. The in-flight batch is delivered correctly via the
existing per-request error channel (step 5) — what makes this a customer-
visible wedge is steps 6–7: every subsequent request hangs, and both
public liveness probes continue to report healthy.

---

## Why Prior Hardening Did Not Catch It

### Timing

| PR | What it adds | Merged | Image base | Reporter's latest tested `main` |
|---|---|---|---|---|
| [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) | Propagate real errors to disagg server | 2026-04-24 | 2026-04-20 (`fdfaeb27`) | 2026-04-22 (`0d2bea7c`) |
| [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | Fatal error detection + zombie-worker prevention | 2026-04-27 | " | " |

Both protections merged **after** the binary the customer reproduced on.
Both are now on current `main`; the design-gap reasoning below applies
unchanged to a fresh rebuild — see `fix-plan.md` §V1 for the verified
primitive names and call sites.

### Design gaps (even if applied)

The root architectural fact is **(2) below — the executor-loop thread is
not wrapped by #12718's machinery, so its death is invisible to every
downstream observer.** (1), (3), and (4) are downstream consequences of
that gap. (1) is additionally an independent observability hole.

1. **Wrong probe surface.** OpenAI handlers wire `signal.raise_signal(SIGINT)`
   into the `CppExecutorError` path (`openai_server.py:1160, 1257, 1456,
   1695`). `/v1/models` is a static handler that never reaches executor
   health. `/health` *does* consult `LLM._check_health()` — but that method
   reads `Executor.doing_shutdown` (only flipped by an explicit
   `Executor.shutdown()` call), not `PyExecutor.is_shutdown` (which *is*
   flipped on thread death by `_executor_loop_cleanup`). So both `/health`
   and `/v1/models` continue to report 200 after the wedge.
2. **Thread death short-circuits the fatal-error path.** #12718's primitives
   on `main` are `_error_queue` + `_handle_background_error` +
   `ManagedThread` (`executor/executor.py:259`, `llmapi/utils.py:311-358`).
   `ManagedThread.run()` catches task exceptions and puts them on
   `_error_queue`. But the `PyExecutor` event loop is started as a **plain**
   `threading.Thread` (`py_executor.py:698-700`), not a `ManagedThread`.
   When `_event_loop_wrapper` re-raises, the exception is lost to the
   thread's default handler; nothing reaches `_error_queue`, so
   `_handle_background_error` never classifies it and the SIGINT
   self-terminate path never fires.
3. **Classification would not trip.** Even if the exception did reach
   `_handle_background_error`, the escaping type is `AssertionError`, not
   `CppExecutorError`. The HTTP handlers' `signal.raise_signal(SIGINT)`
   self-terminate is gated on `except CppExecutorError` — an
   `AssertionError` walks past it.
4. **#13119 is disagg-scoped and also assumes a live loop.** Its primitives
   (`GenerationResultBase.error`, `OpenAIHttpClient` body preservation,
   disagg-id regeneration) all need the executor to call back into the
   request's `GenerationResult` to set the error. The in-flight batch *does*
   get a callback today via `_handle_errors` (see the failure-flow box
   above), but every subsequent request hangs because there is no live loop
   to enqueue the next batch.

In one sentence: **the prior work fixed the engine-death and request-error
layers; this bug lives in the gap "the executor thread itself dies between
iterations, and #12718's classification machinery does not observe it
because the thread is not a `ManagedThread`."**

---

## Why #12806 Is Not on `main`

The reporter pointed at [PR #12806](https://github.com/NVIDIA/TensorRT-LLM/pull/12806)
("[fix] Pre-subtract non-first-chunk context costs in reuse budget check") as
addressing the same failure mode. Verified:

- `gh pr view 12806`: `state=MERGED`, `mergedAt=2026-04-07`,
  `baseRefName=feat/bench_y`, `mergeCommit=01c49479`.
- The umbrella PR that would have promoted `feat/bench_y → main`,
  [#12865](https://github.com/NVIDIA/TensorRT-LLM/pull/12865), was **closed
  unmerged on 2026-04-21**. So `01c49479` has no path to `main`.

A naive cherry-pick will not apply. PR #12806's diff lives in
`tensorrt_llm/_torch/pyexecutor/resource_manager.py:prepare_resources`, in a
`remaining_budget` accumulator that walks `scheduled_batch.context_requests`
and pre-subtracts non-first-chunk costs:

```python
# (from PR #12806)
for req in scheduled_batch.context_requests:
    if not req.is_first_context_chunk:
        remaining_budget -= req.context_chunk_size
```

On current `upstream/main`, the corresponding region of `prepare_resources`
(around `resource_manager.py:680-760`) has been refactored to the
`add_sequence_batch` two-phase claim path introduced by
[PR #13029](https://github.com/NVIDIA/TensorRT-LLM/pull/13029), and there is
no `remaining_budget` accumulator at that call site. The semantically
equivalent budget math now lives in
`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` (`_reuse_adjusted_compute`
defined at L321, used at **9** call sites: L441, L472, L545, L676, L677,
L720, L734, L746, L779 — the refactor since the original investigation
added more) and in `scheduler_v2.py` (L451-467).

Additionally, the reporter runs **V1 = C++ MicroBatchScheduler**, not the
Python scheduler. PR #12806's fix is Python-only. Even if it had landed on
`main`, it would not have closed the customer's failure path on the C++ side
(`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp`; see `fix-plan.md`
§V3 for the three accumulator sites that need the same correction).

---

## Affected Components (Current `main`)

| File | Surface | Issue |
|---|---|---|
| `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp` (C++ V1) | `reuse_adjusted_compute(chunkSize, reusable, contextRemaining)` at L33-44; accumulator sites at L66-68, L358-378/L385-386, L459-461 | Admission accounting does not pre-subtract non-first-chunk context costs (see `fix-plan.md` §V3) |
| `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` | `_reuse_adjusted_compute` callers (9 sites: L441, L472, L545, L676, L677, L720, L734, L746, L779) | Same invariant must be audited after #13029 / #13095 refactor |
| `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py:451-467` | V2 budget path (`remaining_budget`) | Independent code path; needs the same audit |
| `tensorrt_llm/_torch/pyexecutor/model_engine.py:2667-2670` | `_prepare_tp_inputs` defensive assertion | Crashes the forward step on accounting mismatch instead of failing the offending request |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py:1491, 2444` | `assert sample_state is not None, "Sampling failed"` | Bare assertion outside the forward-step try/except — kills the event-loop thread |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py:652` | `_event_loop_wrapper` top-level | Catches `Exception` not `BaseException`; does not walk in-flight queues; does not route exception into `_error_queue` so PR #12718's classification path never observes the death |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py:698-700` | Event loop thread construction | Plain `threading.Thread`, not `ManagedThread` — root cause of (2) above |
| `tensorrt_llm/llmapi/llm.py:990` | `LLM._check_health()` | Reads `Executor.doing_shutdown` (only set by explicit shutdown), not `PyExecutor.is_shutdown` (set on thread death) |
| `tensorrt_llm/serve/openai_server.py:767` (`/v1/models` handler) | Liveness surface | Returns 200 without consulting executor health |

---

## Related PRs

Already in the failing image, do not close the bug:

- [#12976](https://github.com/NVIDIA/TensorRT-LLM/pull/12976) — Fix compute
  token accounting for KV cache reuse with context chunking (merged 2026-04-18)
- [#13029](https://github.com/NVIDIA/TensorRT-LLM/pull/13029) — Batch
  `addSequence` with two-phase claim (merged 2026-04-18)

Merged after image build, verified not to fix this:

- [#13095](https://github.com/NVIDIA/TensorRT-LLM/pull/13095) — Radix-tree
  walk consolidation; pure perf refactor with no semantic change on the
  budget path
- [#13104](https://github.com/NVIDIA/TensorRT-LLM/pull/13104) — V2 scheduler /
  V2 KV cache manager only; the customer runs V1

Target this exact failure mode but are not on `main`:

- [#12806](https://github.com/NVIDIA/TensorRT-LLM/pull/12806) — Pre-subtract
  non-first-chunk context costs. **Merged to `feat/bench_y` (2026-04-07),
  not `main`.** Umbrella PR #12865 closed unmerged on 2026-04-21.
- [#12665](https://github.com/NVIDIA/TensorRT-LLM/pull/12665) — Same
  failure-mode description; closed unmerged 2026-04-01.
- [#12658](https://github.com/NVIDIA/TensorRT-LLM/pull/12658) — Open draft
  with Python-side pre-validation, stale since 2026-04-01.

Hardening adjacent to this gap (both now merged on `main`):

- [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — Fatal error
  detection / zombie worker prevention. Primitives landed on `main` are
  `_error_queue` + `_handle_background_error` + `ManagedThread` +
  `signal.raise_signal(SIGINT)` on `CppExecutorError` (see `fix-plan.md`
  §V1). Owns the **engine-death** failure shape but does not catch this
  bug because the `PyExecutor` event loop is a plain `threading.Thread`,
  not a `ManagedThread`. See
  [`../nvbug-6043291-zombie-worker-pods/`](../nvbug-6043291-zombie-worker-pods/).
- [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) — Real error
  propagation to disagg server. Owns the **per-request** failure shape;
  also requires a live executor loop to set the per-request error.

---

## Fix Plan

Detailed in [`fix-plan.md`](fix-plan.md), with verified-state findings in
§V1 (PR #12718 primitives on `main`), §V2 (response-delivery semantics on
thread death), and §V3 (exact C++ patch sites). High-level:

1. **Substantive (admission accounting):** port PR #12806's "pre-subtract
   non-first-chunk context costs" semantic to the C++ V1 MicroBatchScheduler
   (Track A1, three patch sites in `microBatchScheduler.cpp` named in §V3)
   and audit the Python `_reuse_adjusted_compute` callers introduced by the
   #13029 / #13095 refactor (Track A2).
2. **Hardening (thread death → real client error):**
   - **B1:** remove the two bare `assert sample_state is not None`
     statements — the proximate cause of the wedge. This single edit closes
     the customer's exact reproduction.
   - **B2:** wrap `_event_loop_wrapper` so its exception is routed into
     `_error_queue` (folding this failure mode into PR #12718's existing
     classification path) and the in-flight queues are drained with the
     captured exception.
   - **B3:** promote prep-time assertions like
     `total_num_tokens > max_num_tokens` to a typed `BatchAdmissionError`
     so a single offending batch fails just the responsible requests
     (today's `_handle_errors(requests=None)` fails *all* active requests,
     not just the offending batch).
3. **Liveness surface (B4):** fix `LLM._check_health()` to observe
   `PyExecutor._fatal_error` (added in B2), and make `/v1/models` consult
   `_check_health()` mirroring `/health`.

---

## Cross-References

- [`../nvbug-6043291-zombie-worker-pods/`](../nvbug-6043291-zombie-worker-pods/)
  — Prior investigation; closest sibling. Defines the `_error_queue` /
  `_handle_background_error` / `ManagedThread` / `CppExecutorError → SIGINT`
  machinery (PR #12718) this bug bypasses (see `fix-plan.md` §V1 for why).
- [`../nvbug-6104831-disagg-permanent-wedge/`](../nvbug-6104831-disagg-permanent-wedge/)
  — Symptom-class cousin. Different root cause (disagg KV-transceiver
  cancellation vs. agg admission accounting) but identical failure shape:
  one in-flight error → permanent server wedge.
- [`../../design/block-reuse-overlap-scheduler/`](../../design/block-reuse-overlap-scheduler/)
  — The block-reuse correctness design this bug belongs to; updated with
  this issue as additional motivation.
- [`../../design/chunked-kv-transfer/`](../../design/chunked-kv-transfer/)
  — Chunked KV transfer design; updated with this issue as a precondition
  risk on V1 chunked-prefill accounting.
