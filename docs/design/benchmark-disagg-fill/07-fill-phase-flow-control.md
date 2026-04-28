# v2.2 — Fill-Phase Admission Flow Control

[< Back to index](README.md)

**Bug references:** nvbug 6094XXX (TBD)
**PR:** [#13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347) (4th commit, on top of the v2.1 three-part fix)
**Prerequisite reading:** [`02-regression-investigation.md`](02-regression-investigation.md), [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md), [`05-router-cap-fix.md`](05-router-cap-fix.md), [`06-fill-phase-fail-fast.md`](06-fill-phase-fail-fast.md)

---

## 1. Problem

After the v2.1 three-part fix (state-based gate + ADP router cap + fill-phase fail-fast suppression) was applied, the wide-EP Kimi-K2-Thinking gen-only test on a `dep32 × bs256 == con8192` configuration **OOM-killed the GEN server** during the fill phase:

```text
*** STEP 1596431.10 ON lyris0073 CANCELLED AT 2026-04-27T04:29:13 DUE to SIGNAL Killed ***
```

`STEP ... DUE to SIGNAL Killed` is SIGKILL from the OS OOM-killer, not a SLURM walltime cancel (which would be `DUE TO TIME LIMIT`). At crash time the GEN log contained:

```text
[I] Skipped adding dummy requests: num_fetch_requests=8192, num_schedulable_requests=203
```

That string is emitted only by `_should_skip_dummy_for_benchmark_disagg`, and only while `_benchmark_fill_phase_active=True`. So the gate was still closed — the v2.1 state-based predicate was working as designed; the failure was elsewhere.

---

## 2. Root cause

PR #12091 (v1a) had introduced a per-iteration `min(tp_size, remaining)` admission throttle inside the blocking fill loop. PR #12208 (v2) removed the entire blocking fill loop along with that throttle, replacing it with a non-blocking gate. This was correct for the gate's purpose but accidentally also removed the only *implicit* memory-pressure regulator the system had: the per-iteration cap on how fast new INIT requests could be created.

After PR #12208, on the first iteration of the benchmark fill phase:

1. The waiting queue is preloaded with `benchmark_req_queues_size = 8192` requests.
2. `_pop_from_waiting_queue` computes `max_new_requests = total_max - total_active = tp_size * max_batch_size = 8192`.
3. All 8192 requests are admitted in a single iteration.
4. `_prepare_disagg_gen_init` reserves KV-cache blocks and posts NIXL/UCX recv buffers for all 8192 simultaneously.
5. Per-rank peak memory at this instant = `model_weights + activations + 256 × max_seq_len × kv_dtype_size + 256 × cache_transceiver_buffer_size + cuda_graph + autotuner_workspace + ...`
6. On the boundary case `dep32 × bs256 == con8192` with `free_gpu_memory_fraction=0.6`, this peak exceeds the cgroup memory budget → OS OOM-killer fires → SIGKILL.

The state-based gate (v2.1) cannot help here because the OOM occurs before any forward step has a chance to free transfer-buffer slots, and v2.1's fail-fast suppression intentionally disables the only graceful-error path that might have surfaced this as `RequestError: Insufficient KV cache` instead of a hard kill.

In short: v2.1 fixed the gate predicate, but it uncovered a latent fill-phase memory-pressure problem that the v0/v1a throttle had been masking.

---

## 3. Fix

Reintroduce a deterministic per-iteration admission cap **inside `_pop_from_waiting_queue`**, scoped to the benchmark fill phase only:

```python
# tensorrt_llm/_torch/pyexecutor/py_executor.py
max_new_requests = total_max - total_num_active_requests

# Benchmark disagg fill-phase admission control:
# Without this cap, the executor can admit up to `total_max` requests
# in a single iteration as soon as the benchmark queue is preloaded,
# which spikes peak KV-cache reservations + recv-buffer reservations
# and can OOM-kill the GEN server before any KV transfer drains. By
# capping admission to `tp_size` per iteration during the fill phase,
# the system bleeds requests in slowly so transfers and reservations
# interleave at steady state.
if (self.is_benchmark_disagg and self._benchmark_fill_phase_active
        and not self.is_warmup):
    max_new_requests = min(max_new_requests, self.dist.tp_size)
```

The cap is `tp_size` so that, on average, **one request is admitted per rank per iteration** — independent of cluster size. With a 100 ms outer loop, the entire benchmark queue is admitted in `total_max / tp_size × 100 ms ≈ 25 s` regardless of whether the deployment is 4 ranks or 32 ranks.

The cap is only active while three conditions all hold:

| Condition | Why |
|---|---|
| `self.is_benchmark_disagg` | Normal serving never has a queue this large; the cap would be a no-op anyway |
| `self._benchmark_fill_phase_active` | Once the gate fires, the fill is over and normal admission resumes (e.g., during taper-down) |
| `not self.is_warmup` | Warmup paths admit synthetically and must not be slowed |

---

## 4. Interaction with v2.1 fixes

The four PR #13347 + this PR fixes form a coherent set:

| Layer | Fix | Purpose |
|---|---|---|
| **State** | v2.1 state-based gate | Correctly decides *when* the fill is complete |
| **State** | v2.1 ADP router cap | Prevents assigning more requests to a rank than it can schedule |
| **Error reporting** | v2.1 fail-fast suppression | Prevents PR #12206 from firing prematurely during fill |
| **Resource budget** | **v2.2 admission cap** | Prevents the GEN server from OOM-killing itself during the burst admission |

v2.1 made the gate correct; v2.2 makes the runtime path leading up to the gate *survivable*.

The fail-fast (PR #12206) remains a post-fill safety net: if the system genuinely cannot make progress after the fill gate has opened, the fail-fast still fires with a clear error rather than hanging.

---

## 5. Why this is safe

| Case | Before v2.2 | After v2.2 |
|---|---|---|
| Fill phase, large queue (`tp_size × max_batch_size` reqs) | All admitted in 1 iter → memory spike → OOM on tight budgets | Spread across `~256` iters → steady ramp; transfer and reservation peaks interleave |
| Fill phase complete, gate fires | Normal admission resumes | Same — cap predicate is False (`_benchmark_fill_phase_active=False`) |
| Normal (non-benchmark) serving | No effect (`is_benchmark_disagg=False`) | Same |
| Warmup | No effect (`is_warmup=True`) | Same |

The cap only constrains the *peak admission rate*; it does not change steady-state throughput or the eventual filled state. For deployments that previously fit comfortably (no OOM risk), the only observable difference is a one-time ~25 s smoother ramp at benchmark startup.

---

## 6. Test coverage

Added to `tests/unittest/_torch/executor/test_benchmark_disagg.py`:

| Test class | Tests | Coverage |
|---|---|---|
| `TestBenchmarkFillAdmissionFlowControl` | 2 | (a) verifies admission is capped to `tp_size` while `_benchmark_fill_phase_active=True`; (b) verifies the cap is lifted once the gate fires |

Combined with the v2.1 tests, the benchmark-disagg unit test file has **45 passing tests**.

```text
45 passed, 3 warnings
```

### Local integration verification

A new YAML config and bash launcher were added so the failure mode can be reproduced on an 8-GPU node without Slurm:

- `tests/scripts/perf-sanity/disaggregated/gb200_deepseek-v32-fp4_8k1k_con1024_ctx1_dep4_gen1_dep4_bs256_fill-pressure_ccb-UCX.yaml`
- `tests/integration/defs/perf/run_local_disagg_perf_sanity.sh`

The launcher injects the `TLLM_BENCHMARK_REQ_QUEUES_SIZE` and `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP` environment variables that the SLURM `submit.py` wrapper normally sets, so the benchmark fill gate is genuinely engaged in local runs.

In before-fix runs the GEN log shows the gate engaged (`Skipped adding dummy requests` accumulating) and either OOMs (on tight memory budgets) or hangs (on B200 with more headroom — see `08-fill-phase-stuck-state-finding.md`). In after-fix runs the same admission gate engages, but the burst is spread out and the run survives.

Local 4-GPU/B200 runs cannot reproduce the *exact* OOM that Kimi hits on GB200 — peak memory with 1024 admissions on 4 ranks is far below 32 ranks × 256 admissions on Kimi's setup — so authoritative validation must happen in CI with the original Kimi config.

---

## 7. Open question: cap value

`tp_size` is the most conservative safe value (1 admission per rank per iteration). Higher values trade peak memory for faster fill:

| Cap | Per-rank rate | Time to fill 8192-deep queue (100 ms loop) |
|---|---|---|
| `tp_size` | 1/iter | 25.6 s |
| `tp_size * 2` | 2/iter | 12.8 s |
| `tp_size * 4` | 4/iter | 6.4 s |

If 25 s of additional benchmark startup proves to be a measurement nuisance, the cap can be raised. The v3 follow-up (`04-step2-external-orchestrator-plan.md`) eliminates the question entirely by moving fill orchestration to the client.
