# Bench Shutdown Hang — Investigation Diary

- **Date:** 2026‑04‑26
- **Branch:** `fix-zombie-worker-health-check`
- **PR:** [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718)
- **Symptom:** `unittest/tools/test_layer_wise_benchmarks.py::test_performance_alignment[1]` hung at exactly 2400 s (test timeout) on every CI run for this PR, while the same test passed in 245–300 s on `main` and on every other open PR in the same CI window.
- **Outcome:** One‑line bug introduced earlier in this PR; ~1 day to localize, < 1 hour to fix once reproduced. Six new regression tests added. Verified locally and by code review.

---

## TL;DR

`GenerationExecutorProxy.pre_shutdown()` was changed earlier in this PR from
`all(not f.done() for f in self.mpi_futures)` to bare
`any(not f.done() for f in self.mpi_futures)` — intentionally, to support the
partial‑crash scenario where some workers have already exited.

But for the empty‑list case (`mpi_futures == []`), the two predicates differ:

| expression | empty list | non‑empty |
|---|---|---|
| `all(not f.done() for f in [])` | **`True`** (vacuous) | True iff all alive |
| `any(not f.done() for f in [])` | **`False`** | True iff any alive |

`RemoteMpiCommSessionClient.submit()` — the path used by
`trtllm-llmapi-launch` and `mgmn_leader_node` — returns `[]` because workers
run in a separate process and the proxy has no local future handles. Switching
to `any(...)` therefore silently dropped the `None` quit sentinel for that
path. The worker loop blocked on `request_queue.get()` forever, never called
`notify_proxy_threads_to_quit()`, never put `None` on `result_queue`, and the
proxy's `dispatch_result_thread.join()` hung. The 2400 s test timeout was just
the outer pytest finally noticing.

**Fix:** `if not self.mpi_futures or any(not f.done() ...)`.

---

## How we got here

I'll describe the path I actually walked, including the mistakes, because the
audit document we wrote earlier (`docs/design/wide-ep-fault-tolerance/audit-1a-findings.md`)
shaped several of the choices and I want that to be reproducible for the next
person.

### Stage 0 — assuming flakiness (~1 day wasted)

Three CI runs in a row showed `test_performance_alignment[1]` failing at 2425 s
(timeout). Other failures in those runs were all "Test terminated unexpectedly"
on different tests, so my first hypothesis was that the layer‑wise bench was
just another flaky GPU test on B200. Two more CI cycles confirmed that the
test consistently took ≈245 s on every other PR in the same window.

**Lesson:** when the same test fails for *only one PR* with high reliability,
that's not flakiness. The fact that the failure was 2425 s every time
(matching the pytest timeout to the second) was the giveaway I missed.

### Stage 1 — disabling the monitor thread (didn't help, but was right to try)

The most obvious suspect from our PR was the new `_error_monitor_thread` —
that's a thread that runs every 5 s and could interact with shutdown. I
gated it behind `DIAG_ENABLE_MONITOR=1` and pushed. Build #35665 showed the
test still hanging at 2425 s with the monitor disabled. Verdict: monitor is
not the cause.

This was still the right move — it eliminated the largest single hypothesis
in one CI cycle. If it had passed, we'd have been done.

### Stage 2 — instrumented timing (the breakthrough)

Same diag commit added `[DIAG-PR12718]` markers around every step of
`proxy.shutdown()`:
- `Proxy.__init__ done`
- `shutdown ENTER`
- `pre_shutdown ENTER` / `EXIT`
- `f.result() loop elapsed=…`
- `monitor join elapsed=…`
- (then the next step's elapsed marker)

What CI showed was hard to read because pytest captured the inner subprocess's
stdout and the 2400 s kill destroyed the captures. Locally, on `umb-b300-004`
with `Llama-3.2-1B`, the markers came through cleanly:

```
[DIAG-PR12718] shutdown ENTER         t=21403.088 lifetime=0.000s
[DIAG-PR12718] pre_shutdown ENTER     elapsed_so_far=0.000s
[DIAG-PR12718] pre_shutdown EXIT      elapsed=0.000s
[DIAG-PR12718] shutdown f.result()    elapsed=0.000s
[DIAG-PR12718] shutdown monitor join  elapsed=0.000s
                                      ← *** HANGS FOR 9+ MINUTES HERE ***
[no further DIAG markers]
[no shutdown EXIT]
```

The hang lives between "monitor join" and the next instrumented call site.
Looking at proxy.py, the next thing in `shutdown()` is
`dispatch_result_thread.stop() / join()`. **`dispatch_result_task` blocks on
`self.result_queue.get()`**, and `ManagedThread.stop()` only sets a flag that
gets checked at the *top* of the loop — it can't interrupt a blocking IPC
read.

So the question became: who normally puts `None` on `result_queue` to unblock
`dispatch_result_thread.get()` at shutdown? Answer: the worker, in
`worker_main`'s `notify_proxy_threads_to_quit()`. That only runs after the
worker's request loop exits — and the request loop only exits when it sees
`None` on `request_queue`. And the only thing that puts `None` on
`request_queue` is `proxy.pre_shutdown()`.

That's a 4‑link chain:

```
proxy.pre_shutdown()
  → request_queue.put(None)
    → worker reads None, loop exits
      → worker calls notify_proxy_threads_to_quit()
        → result_queue.put(None)
          → dispatch_result_thread.get() unblocks
            → dispatch_result_thread exits
              → proxy.shutdown() unblocks
```

If the first link is silently broken, the whole chain hangs and no one logs
anything. Which is exactly what was happening. The DIAG markers narrowed the
search to "the chain starts at `pre_shutdown()`", and the diff against
`upstream/main` was tiny in that function (one line). The bug was visible
within a minute of opening the file.

### Stage 3 — verifying the fix

Pre‑fix:
```
| upstream/main                       | rc=0 (success)  |  72 s  |
| PR before this fix (any())          | rc=124 timeout  | 600 s+ |
```

Post‑fix (`if not self.mpi_futures or any(...)`):
```
| PR with this fix                    | rc=0 (success)  |  70 s  |
```

To make sure the new regression test actually catches the bug,
`test_empty_mpi_futures_sends_sentinel` was run against a temporarily‑reverted
mock that uses bare `any(...)`. It correctly fails. Six related cases were
added: empty‑list, all‑alive, all‑done, partial‑crash, idempotency,
not‑yet‑started.

---

## Why the audit findings (`audit-1a-findings.md`) helped

This bug was a six‑character change buried in 600+ lines of PR diff. Three
separate things I'd written down in the Audit 1a notes earlier in the week
shaped the path to it:

### 1. "Hangs are real, not just slowness" (Day 2 F2 / F4)

The audit's Day 2 finding was: when an MPI rank dies abnormally, `mpirun`
propagates termination, and **survivors block inside `MPI_Allreduce` because
it has no timeout — it spins until completion or process death**. The same
shape sits one layer up in our case: workers were stuck inside
`request_queue.get()` because (a) the proxy never sent the `None` sentinel
and (b) `request_queue.get()` has no timeout — it spins until completion or
process death.

When my first local repro showed a "hang during shutdown," I almost
dismissed it as an environment artifact (slow NFS, missing C++ extensions).
The audit reading kept me on the trail — it had already walked this exact
failure category and shown that *real* hangs in the MPI / IPC plumbing look
exactly like this. So I instrumented instead of giving up.

### 2. "PyTorch / NCCL / MPI watchdogs don't surface to user code" (Day 1)

Audit Day 1 showed that the canonical detection mechanisms
(`TORCH_NCCL_ASYNC_ERROR_HANDLING`, `BLOCKING_WAIT`, `dist.shrink_group`) are
unusable in PyTorch 2.11 — they either crash the process via
`std::terminate()` or hang past the test budget. The audit recommended a
**main‑thread polling loop with a timeout and explicit per‑step timing
markers** as the right tool.

The `[DIAG-PR12718]` markers in `pre_shutdown` / `f.result()` /
`monitor join` / `dispatch_result_thread join` / `mpi_session.shutdown`
follow that recipe directly. Without that prior justification I'd have
spent a CI cycle trying logger‑level debug or attaching `gdb` to the wrong
process.

### 3. "The worker / proxy split has independent IPC plumbing — assume any one link can be broken" (background reasoning, not in the audit but motivated by it)

The audit isolated the NCCL / MPI / driver layers separately because each
runs on different state. That same instinct drove me to draw the four‑link
chain (`pre_shutdown → request_queue → worker loop → result_queue →
dispatch_thread`) explicitly on paper before re‑reading the diff. Once the
chain was on paper, the question became "which link is broken?" — and the
DIAG markers had already pointed at the first link.

The audit didn't say "look at line 414 of `proxy.py`." It said "this
category of bug looks like X, the right tool is Y, and don't trust
process‑exit‑time error reporting." That framing turned the diagnosis from
"I have no idea" into "the sentinel‑send gate is the only thing that fits."

---

## Reproducer

Single B300, < 90 s wall clock per attempt, no Docker:

```bash
# 0) Pre‑copy the model to local fast disk (NFS makes the bench start dominate
#    the budget).
mkdir -p /tmp/models && \
  cp -r /scratch.trt_llm_data/llm-models/llama-3.2-models/Llama-3.2-1B /tmp/models/

# 1) Tiny dataset
python3 /home/chienchunh/dev/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
    --tokenizer /tmp/models/Llama-3.2-1B --stdout --random-seed 42 \
    token-norm-dist --num-requests 32 --input-mean 2048 \
    --input-stdev 0 --output-mean 256 --output-stdev 0 \
    > /tmp/dataset.jsonl

# 2) Empty config (the bench works without layer_wise overrides)
echo "print_iter_log: true" > /tmp/cfg.yaml

# 3) Reproducer — exactly mirrors Step 1 of sample_performance_alignment.sh
mpirun --allow-run-as-root --np 1 -x TLLM_AUTOTUNER_CACHE_PATH \
    /home/chienchunh/dev/TensorRT-LLM/tensorrt_llm/llmapi/trtllm-llmapi-launch \
    python3 -m tensorrt_llm.commands.bench \
        --model meta-llama/Llama-3.2-1B \
        --model_path /tmp/models/Llama-3.2-1B \
        throughput --tp 1 --ep 1 --warmup 0 \
        --dataset /tmp/dataset.jsonl --max_batch_size 32 --max_num_tokens 3072 \
        --disable_chunked_context --num_requests 32 --concurrency 32 \
        --config /tmp/cfg.yaml
```

On the buggy revision: `mpirun` reports `rc=124` after the wall clock kicks
in. On the fix: bench finishes in ≈ 70 s and `mpirun` returns `rc=0`.

---

## Lessons

1. **Empty collections are sneaky.** `all([]) == True` ≠ `any([]) == False`.
   When changing the gate predicate of a collection check, always think
   about the empty case explicitly. This applies to every `any` / `all`
   refactor, not just MPI plumbing.

2. **Pure flakiness vs. PR‑specific hangs look the same in the failure
   list, but the duration distribution distinguishes them.** Three CI runs
   that all timed out at exactly the same wall clock are *not* flakiness —
   they're a deterministic hang.

3. **Instrument the lifecycle, not the request path.** The bench *worked
   correctly* — all 32 requests completed in 1 s. The bug was entirely in
   the shutdown plumbing. Make sure the instrumentation covers
   `__init__` → `__exit__`, not just the hot path.

4. **Pytest's stdout capture eats logs when the test is killed.** If you
   need DIAG output to survive a `--timeout=` SIGKILL, write to a side
   file (or set `PYTHONUNBUFFERED=1` and route the inner subprocess's
   output to a known path). I didn't, and burned half a CI cycle on
   captures that never materialized.

5. **A local reproducer is cheaper than a CI cycle.** The first local
   repro took 22 minutes (model copy, env verification, bench command
   shape) and saved at least 3 × 30‑min CI runs. If a bug is consistently
   reproducible on a single CI configuration, getting that configuration
   onto a dev box is almost always a net win.

---

## Cross‑references

- [Original investigation notes (Layer 1–4 design)](README.md)
- [Audit 1a findings — intra‑node FT prototyping](../../design/wide-ep-fault-tolerance/audit-1a-findings.md)
- [PR #12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718)
- Fix commit: `[https://nvbugs/6043291][fix] proxy: send shutdown sentinel when mpi_futures empty`
