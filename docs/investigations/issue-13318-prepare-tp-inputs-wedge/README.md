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
this case**, both because of merge timing and because of design scope. The bug
sits in the seam between "fatal engine error" (covered by #12718) and
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

`tensorrt_llm/_torch/pyexecutor/model_engine.py:2671-2673` (current `main`):

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

The actual failure flow:

```
1. model_engine.py:2672  AssertionError: total_num_tokens > max_num_tokens
                         (raised inside the forward step)
2. forward-step try/except catches it
                         => log "Encountered an error in forward function"
                         => sample_state = None
3. py_executor.py:1600   assert sample_state is not None, "Sampling failed"
                         (NOT inside the forward-step try/except)
                         => AssertionError escapes _event_loop_wrapper
4. Thread-3 dies.        No code revives it; no _fatal_error is set;
                         no in-flight GenerationResult is completed.
5. /v1/chat/completions  awaits a future that nobody owns -> permanent hang
   /v1/models            still returns 200 (static handler, no executor call)
```

Step 3 is the crucial gap. The forward-step exception handler set
`sample_state = None` but did not break the loop or fail the in-flight
batch. The `assert sample_state is not None` was added as a defensive
guard, but it is itself a fatal assertion outside the try/except — turning
a recoverable per-batch error into a thread death.

---

## Why Prior Hardening Did Not Catch It

### Timing

| PR | What it adds | Merged | Image base | Reporter's latest tested `main` |
|---|---|---|---|---|
| [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) | Propagate real errors to disagg server | 2026-04-24 | 2026-04-20 (`fdfaeb27`) | 2026-04-22 (`0d2bea7c`) |
| [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | Fatal error detection + zombie-worker prevention | 2026-04-27 | " | " |

Both protections merged **after** the binary the customer reproduced on.

### Design gaps (even if applied)

1. **Wrong probe surface.** #12718 wires `SIGINT` into `/health`. The reporter
   probed `/v1/models`, which is a static handler that never reaches executor
   health. Even with #12718 active, `/v1/models` would still report 200.
2. **Thread death short-circuits the fatal-error path.** #12718's
   `_fatal_error` + `ErrorBudget` mechanism assumes the executor loop survives
   long enough for the next `/health` poll to drain the error queue. Here the
   loop thread dies between iterations, so no later code sets `_fatal_error`
   and no health probe ever sees an error.
3. **Classification would not trip.** An `AssertionError` from
   `_prepare_tp_inputs` is not CUDA-corruption, not OOM, not NCCL. Per
   #12718's three-tier model, it lands in "transient: ~10 occurrences before
   crash." First-occurrence prep-time assertions are not currently classified
   as immediate-fatal, even though they are deterministic and won't self-heal.
4. **#13119 is disagg-scoped and also assumes a live loop.** Its primitives
   (`GenerationResultBase.error`, `OpenAIHttpClient` body preservation,
   disagg-id regeneration) all need the executor to call back into the
   request's `GenerationResult` to set the error. With the executor thread
   dead, no callback ever runs.

In one sentence: **the prior work fixed the engine-death and request-error
layers; this bug lives in the gap "the executor thread itself dies between
iterations."**

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
`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` (`_reuse_adjusted_compute`,
used at L465/505/577/706/750/812) and in `scheduler_v2.py` (L453+).

Additionally, the reporter runs **V1 = C++ MicroBatchScheduler**, not the
Python scheduler. PR #12806's fix is Python-only. Even if it had landed on
`main`, it would not have closed the customer's failure path on the C++ side.

---

## Affected Components (Current `main`)

| File | Surface | Issue |
|---|---|---|
| `cpp/.../MicroBatchScheduler` (C++ V1) | `reuse_adjusted_compute(chunk_size, reusable, context_remaining)` | Admission accounting does not pre-subtract non-first-chunk context costs |
| `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py` | `_reuse_adjusted_compute` callers (L465/505/577/706/750/812) | Same invariant must be audited after #13029 / #13095 refactor |
| `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py:453+` | V2 budget path | Independent code path; needs the same audit |
| `tensorrt_llm/_torch/pyexecutor/model_engine.py:2671-2673` | `_prepare_tp_inputs` defensive assertion | Crashes the forward step on accounting mismatch instead of failing the offending request |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py:1600,2599` | `assert sample_state is not None, "Sampling failed"` | Bare assertion outside the forward-step try/except — kills the event-loop thread |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py:669` | `_event_loop_wrapper` top-level | No `except BaseException` walking `active_requests` to complete each `GenerationResult` with the captured exception |
| `tensorrt_llm/serve/openai_server.py` (`/v1/models` handler) | Liveness surface | Returns 200 without consulting executor health |

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

Hardening adjacent to this gap:

- [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — Fatal error
  detection / zombie worker prevention. Owns the **engine-death** failure
  shape; does not handle the **executor-thread-dies-between-iterations**
  shape this bug exhibits. See
  [`../nvbug-6043291-zombie-worker-pods/`](../nvbug-6043291-zombie-worker-pods/).
- [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) — Real error
  propagation to disagg server. Owns the **per-request** failure shape;
  also requires a live executor loop to set the per-request error.

---

## Fix Plan

Detailed in [`fix-plan.md`](fix-plan.md). High-level:

1. **Substantive (admission accounting):** port PR #12806's "pre-subtract
   non-first-chunk context costs" semantic to the C++ V1 MicroBatchScheduler
   and audit the Python `_reuse_adjusted_compute` callers introduced by the
   #13029 / #13095 refactor.
2. **Hardening (thread death → real client error):** wrap
   `_event_loop_wrapper` with a top-level handler that sets `_fatal_error`
   **and** walks `active_requests` / `waiting_queue` / `executor_request_queue`
   completing each `GenerationResult` with the captured exception. Promote
   prep-time assertions like `total_num_tokens > max_num_tokens` to a
   per-batch recoverable error so a single offending batch fails the
   responsible requests instead of bricking the loop.
3. **Liveness surface:** make `/v1/models` (or any other public "alive"
   surface) consult `check_health()`, or document `/health` as the only
   valid liveness probe.

---

## Cross-References

- [`../nvbug-6043291-zombie-worker-pods/`](../nvbug-6043291-zombie-worker-pods/)
  — Prior investigation; closest sibling. Defines the
  `_fatal_error` / `check_health()` / `ErrorBudget` machinery this bug bypasses.
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
