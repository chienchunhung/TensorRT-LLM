# Phase 0 — Disaggregated Cancellation Stress-Test Suite

| | |
|---|---|
| **Phase** | 0 (prerequisite to Phases 1–4) |
| **JIRA** | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) (weekly stress CI) tied to [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) (the cancellation/poison improvement initiative) |
| **Owner** | Chien-Chun Hung |
| **Status** | Skeleton + `log_scanner_thread` + `metrics_thread` landed in `upstream/main` at `tests/integration/defs/stress_test/disagg_cancel/`. Three threads remain stubs: `injector_thread`, `canary_thread`, `load_thread`. Also pending: marathon YAML configs + canary references + pytest registration. See [Implementation PR chain](#implementation-pr-chain) below for the per-step status. |

## Goal

The suite tests one contract:

> **A disaggregated TRT-LLM deployment runs for hours under cancellation-heavy load without permanent failure.**

"Permanent failure" means any of:

- Process crash (SIGSEGV, abort, unexpected exit)
- Permanent server wedge (after a transient failure, server fails to recover)
- Silent memory corruption (canary responses don't match expected outputs — UAF symptom)
- Monotonic resource leak (KV cache utilization grows without bound)

Failures **during** stress are explicitly OK — request errors, cancellations, retries during traffic bursts, error spikes during injected peer pauses. What we test is **graceful recovery**: after the transient event ends (SIGCONT, peer respawn, burst subsides), the server returns to a healthy baseline within a bounded time.

This is the regression gate for the bug class fixed by the
disaggregated cancellation / poison work at
<https://github.com/NVIDIA/TensorRT-LLM/pull/13713> and the
follow-up work in Phases 1–4 of this design (deferred un-poison,
multi-slot configs, NIXL callback, progress-based cancel). It is
also the empirical safety net required by
[TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) for the
build.nvidia.com issues.

## Implementation PR chain

Phase 0 lands incrementally — one PR per thread body — to keep each PR
small and reviewable, and to start exercising the harness in CI as soon
as each component is in. The chain below tracks per-step status. Each
URL is the upstream merged commit (or pending placeholder); intentionally
written as the full URL (not the GitHub `#NNNNN` shorthand) so this
design doc doesn't auto-post cross-reference comments on the implementation
PRs.

| Step | Component | Status | Landed in |
|---|---|---|---|
| 1 | Harness skeleton + initial YAML config + README + `log_scanner_thread` body | **Merged 2026-05-28** | <https://github.com/NVIDIA/TensorRT-LLM/pull/14375> |
| 2 | `metrics_thread` body (per-worker `trtllm_kv_cache_utilization` scraper, time-series for leak detection) | **Merged 2026-06-02** | <https://github.com/NVIDIA/TensorRT-LLM/pull/14807> |
| 3 | `injector_thread` body (SIGSTOP/SIGCONT/SIGKILL schedule + optional worker respawn) | Pending | — |
| 4 | `canary_thread` body (canary client + deterministic prompts + token-equivalence check against precomputed references) | Pending | — |
| 5 | `load_thread` body (steady-state + burst load wrapper around `run_cancel_stress_test`) | Pending | — |
| 6 | Marathon YAML configs (`marathon_a_v1_cpp_deepseek.yaml`, `marathon_b_v2_py_qwen.yaml`) + `stress_canary_prompts.json` + reference-generation tool | Pending | — |
| 7 | Pytest entry points + L0 test list registration in `tests/integration/test_lists/qa/llm_function_stress.txt` | Pending | — |

The chain order is deliberate: `log_scanner_thread` first because it's
the fail-fast guard the rest of the harness depends on; `metrics_thread`
second because it's read-only and stand-alone; `injector_thread` /
`canary_thread` / `load_thread` after because they form the
read-write/synchronised core; YAML configs and pytest registration last
because they tie everything together and need all five threads
operational.

Each PR after the skeleton is expected to be small (~200–500 lines + a
unit test), and the existing infrastructure tests in
`tests/unittest/disaggregated/stress_test/` are extended with the new
thread's coverage as it lands.

## Background — what this suite is testing against

The regression class to catch:

| Class | Example signatures | Trigger |
|---|---|---|
| Cleanup-path bugs | sigs `#1`, `#4`, `#5`, `#6`, `#7` from the investigation | High concurrency + cancellations + race-window timing |
| Lifetime UAFs | sig `#7` variants C/D | Cancel-during-transfer load + raw-pointer dereference of LlmRequest in async workers |
| Cascade outages | 2026-05-13 production incident (Qwen3-Coder-480B) | Cancel races mid-NIXL transfer → pool-wide poison → PyExecutor shutdown |
| Block-reuse interactions | sig `#8` (rc13 regression) | Disagg + block reuse + in-flight cancel |

Full investigation:
[`docs/investigations/nvbug-6104831-disagg-permanent-wedge/`](../../investigations/nvbug-6104831-disagg-permanent-wedge/),
particularly:

- [§02 failure signatures](../../investigations/nvbug-6104831-disagg-permanent-wedge/02-failure-signatures.md)
- [§10 ablation experiments](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md)
  — six controlled A/B experiments that this suite generalises into a CI gate.

The §10 ablation harness lives in
`local/pr13713-rc13-clean/.repro/` on the developer's machine — that's
the conceptual basis for this suite, but the production CI version
must be self-contained inside the TRT-LLM repository.

## Existing infrastructure (reuse, do not duplicate)

The implementation must build on what already exists in TRT-LLM rather
than create parallel infrastructure.

### Reuse as-is

| File / symbol | What it does |
|---|---|
| `tests/integration/defs/disaggregated/test_disaggregated.py::setup_disagg_cluster(...)` | Starts ctx workers, gen workers, and the disagg server from a YAML config. Returns worker handles + server URL. |
| `tests/integration/defs/disaggregated/test_disaggregated.py::wait_for_server(...)` | Probes `/health` until ready. |
| `tests/integration/defs/disaggregated/test_disaggregated.py::cleanup_output_files(...)` and `terminate(...)` | Standard teardown. |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_cancel_stress_test*.yaml` | YAML config schema for ctx/gen worker configuration (model, TP, kv_cache_config, cache_transceiver_config, cuda_graph_config). **Extend** this schema with new `stress_config:` keys (see "Config schema" section below). |

### Reuse and extend

| File / symbol | What's there | What to add |
|---|---|---|
| `tests/integration/defs/disaggregated/test_disaggregated.py::run_cancel_stress_test(server_url, num_bursts, requests_per_burst, prompt_len_range, cancel_after_range)` | Async coroutine sending N bursts of K requests with client-side disconnect-during-prefill cancellation. Runs via `asyncio.run`. | Must be drivable from an external thread (so it can run in parallel with the canary client + log scanner + injector). Two options: refactor to expose the inner coroutine (cleanest), or wrap the existing function in a worker thread (lower-touch). **Recommendation: wrap in a worker thread to minimize disruption to existing usage.** |
| `tests/integration/defs/disaggregated/test_disaggregated.py::run_disaggregated_cancel_test(...)` | Wraps `run_cancel_stress_test` with full disagg cluster setup + final health-check probe via `disagg_client.py`. | The marathon harness is a *generalisation* of this — multi-component (cancel-stress + canary + injector + log scanner + metrics scraper), longer duration, parametrized via YAML. Don't try to extend this function in place — write the new harness as a separate module that can be invoked independently. |

### Orthogonal — do not modify

- `tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_stress_test` — combined stress + accuracy + aiperf. Different goal.
- `tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestKimiK2::test_nvfp4_longseq_trtllm_moe_async_cancel` — aggregated (non-disagg) cancellation test, KimiK2 long-seq. Different scope.

## What's missing from the existing test (the deltas to implement)

The existing `test_disaggregated_cancel_large_context_requests` is a
short-burst test (~5 bursts × 32 requests) that ends with one normal
request to verify the server is alive. The marathon suite needs:

1. **Long duration** — 2 h per test, not minutes.
2. **Failure injection** — SIGSTOP / SIGCONT / SIGKILL on individual workers, on a schedule.
3. **Canary client** running in parallel with the load client; deterministic prompts; token-equivalence check vs precomputed reference outputs.
4. **Log-pattern scanner** — tails worker logs continuously, fails the test if "hard zero" patterns appear (`Broken promise`, `NO RECOVERY`, `Segfault`, `SIGSEGV`, `0xffffffffffffffff`, `Poisoned ... cache transfer buffer`).
5. **KV cache utilization monitor** — scrapes the `trtllm_kv_cache_utilization` metric from each worker periodically; verifies it doesn't grow monotonically (leak detection).
6. **Recovery-time measurement** — after each SIGCONT / respawn, measures how long the canary error rate takes to return to baseline.
7. **Config-knob parametrization** — V1/V2 KV cache, C++/Python transceiver, block reuse on/off, overlap scheduler on/off, transport NIXL/UCX, KV transfer timeout, etc., all controlled by the YAML config.

## Test suite specification

### Suite composition (4 h total budget)

| # | Test ID | Mode | Config | Model | Duration |
|---|---|---|---|---|---|
| 1 | **Marathon A** | 3P3D local | V1 + C++ transceiver + NIXL, block reuse on, overlap on, 60s timeout | DeepSeek-class | 2 h |
| 2 | **Marathon B** | 3P3D local | V2 + Python transceiver + NIXL, block reuse on, overlap on, 60s timeout | Qwen-class | 2 h |

Both marathons use 3P3D (3 ctx workers + 3 gen workers) for:

1. **Redundancy** — SIGKILL of one worker doesn't kill the deployment; the remaining workers should absorb load.
2. **Multi-pair coordination** — exercises the L10 dual-cleanup-path scenario that surfaced sig `#8`.
3. **Mainstream-deployment matching** — typical production deployments have multiple ctx/gen pairs.

The two marathons differ on the (KV-cache, transceiver) axis:

| Combination | Valid? | Why |
|---|---|---|
| V1 + C++ | ✓ | Mainstream established path — Marathon A |
| V1 + Python | ✓ | Secondary — deferred to follow-up YAML |
| **V2 + C++** | **✗** | **Invalid — does not make sense; do not test** |
| V2 + Python | ✓ | Mainstream modern path — Marathon B |

### Hardware budget

Single 8-GPU node (B200 or H100). With 3P3D and TP=1 per worker, we use
6 GPUs (3 ctx + 3 gen), leaving 2 spare for the disagg server +
load/canary clients. **No TP > 1 per worker for the initial cut** —
TP coverage is exercised by other tests and not the regression class
this suite is gating against.

### Deployment shape — multi-node decision

**Multi-node is deferred** from this initial suite. Rationale:

- The cancellation regression class fires single-host (the §10
  ablation experiments are single-host and catch all of sigs `#1`–`#8`).
- Multi-node only adds unique coverage if ctx pod and gen pod are on
  *different* nodes (real cross-node RDMA over network). Co-located
  multi-node = N copies of single-host, which 3P3D already exercises.
- Cross-node multi-node testing is a deployment-infra correctness
  concern (orchestrator, Slurm provisioning, network), not a
  cancellation correctness concern. Track separately.

### Workload schedule (used by both marathons)

Each marathon runs a single workload pattern. The cycle below repeats
for 2 h.

**Steady-state (continuous, between bursts):**

- CONC = 64 concurrent in-flight requests
- Prompt lengths: uniform 4 k – 12 k input tokens; 512 output tokens
- Client-side cancellation rate: 10% (random subset disconnect during prefill)

**Burst events (every ~8 min for 2 h = 15 bursts):**

- CONC ramps to 256 for 60 s
- Prompts: uniform 12 k – 16 k input tokens (heavier KV transfer load)
- Returns to steady-state after

**Injection schedule (interleaved with bursts):**

```
T+ 15 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+ 30 min : SIGSTOP random gen worker for 30 s, then SIGCONT
T+ 45 min : SIGSTOP random ctx worker for 20 s, then SIGCONT
T+ 60 min : SIGKILL gen_worker_0 + respawn within 60 s
T+ 75 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+ 90 min : SIGSTOP random ctx worker for 30 s, then SIGCONT
T+105 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+120 min : end
```

The schedule mixes:
- **Transient peer pauses** (SIGSTOP/SIGCONT) — exercise the cancel-mid-flight race + recovery
- **Terminal peer failure** (SIGKILL + respawn) — exercise the worker-loss path that 3P3D redundancy is supposed to absorb
- **Targeting both gen and ctx workers** — exercises sender-side and receiver-side cleanup symmetrically

### Canary client

Runs in parallel with the load client for the full 2 h.

- 5 requests per minute (300 over the 2 h)
- Prompts: small fixed set (e.g., 10 prompts) loaded from
  `stress_canary_prompts.json`, with **precomputed reference outputs**
  (also stored in the JSON file)
- **Greedy decoding** + fixed seed for determinism
- Token-equivalence check: canary's response tokens must exactly match
  the reference

Reference outputs must be generated once (in a separate one-shot
script using the same model + same engine config) and committed
alongside the YAML config. Regenerate when the model or engine config
materially changes.

### Pass criteria (gates that fail the test)

| Gate | Threshold | Rationale |
|---|---|---|
| Hard-zero log patterns | 0 occurrences in any worker log | `Broken promise`, `NO RECOVERY`, `Segfault`, `SIGSEGV`, `0xffffffffffffffff`, `Poisoned ... cache transfer buffer` |
| All workers alive at end | `is_alive() == True` for every worker process | Crash detection |
| Final health probe | 5 sequential canary requests must all succeed within 30 s of test end | Permanent-wedge detection |
| Canary correctness | 100% of returned canaries token-equivalent to reference | UAF detection |
| Canary error rate (overall) | < 1% over the full 2 h | Baseline service quality |
| Canary error rate (per-burst / per-injection window) | < 10% during any 1-min window containing burst or injection | Degraded but not failed |
| Recovery time after injection | < 30 s from SIGCONT (or worker respawn) until canary error rate returns to < 1% baseline | Graceful recovery |
| KV cache utilization growth | End-of-test utilization ≤ baseline + 10 percentage points | Leak detection |

`Cannot cancel request` log lines have **no upper bound** — they're
expected behaviour at the L3 invariant boundary when cancel hits a
mid-flight request.

## YAML config schema

The harness reads a single YAML per test. Extends the existing
`disagg_config_cancel_stress_test*.yaml` schema by adding a top-level
`stress_config:` block. The ctx/gen worker config sections (`hostname`,
`model`, `backend`, `context_servers`, `generation_servers`) stay
identical to the existing schema and are passed through to
`setup_disagg_cluster`.

```yaml
# === Existing schema (passed through to setup_disagg_cluster) ===
hostname: localhost
model: <model path under llm_models_root, e.g. DeepSeek-R1-Distill-Llama-8B>
backend: pytorch

context_servers:
  num_instances: 3                  # 3P3D
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  disable_overlap_scheduler: false  # overlap ON
  max_num_tokens: 16384
  max_seq_len: 16384
  kv_cache_config:
    enable_block_reuse: true        # block reuse ON
    enable_partial_reuse: true
    free_gpu_memory_fraction: 0.3
    # kv_cache_manager_class: v1    # or v2 — see "Backend knobs" below
  cache_transceiver_config:
    backend: NIXL                   # NIXL transport
    max_tokens_in_buffer: 16384
    kv_transfer_timeout_ms: 60000   # production default

generation_servers:
  num_instances: 3
  # ... same shape as context_servers

# === New: stress harness configuration ===
stress_config:
  duration_min: 120                  # 2 h marathon

  # Backend knob selections (control the V1/V2 × C++/Python axis)
  # These get translated into the appropriate worker env vars / config
  # by the harness before launching trtllm-serve.
  kv_cache_manager: v1               # v1 | v2  (V2 + C++ is invalid)
  transceiver: cpp                   # cpp | python

  # Load shape
  base_concurrency: 64
  client_cancel_rate: 0.10
  input_length:
    distribution: uniform
    min_tokens: 4096
    max_tokens: 12288
  output_length: 512

  # Burst schedule (repeats every burst_interval_min for duration_min)
  bursts:
    interval_min: 8
    concurrency: 256
    duration_s: 60
    input_length:
      distribution: uniform
      min_tokens: 12288
      max_tokens: 16384

  # Injection schedule (absolute timestamps in minutes)
  injections:
    - at_min: 15
      type: sigstop
      target: gen_worker_random
      duration_s: 20
    - at_min: 30
      type: sigstop
      target: gen_worker_random
      duration_s: 30
    - at_min: 45
      type: sigstop
      target: ctx_worker_random
      duration_s: 20
    - at_min: 60
      type: sigkill
      target: gen_worker_0
      respawn_within_s: 60
    - at_min: 75
      type: sigstop
      target: gen_worker_random
      duration_s: 20
    - at_min: 90
      type: sigstop
      target: ctx_worker_random
      duration_s: 30
    - at_min: 105
      type: sigstop
      target: gen_worker_random
      duration_s: 20

  # Canary client
  canary:
    prompts_file: stress_canary_prompts.json
    rate_per_min: 5
    greedy_decoding: true
    seed: 42
    max_tokens: 128
    check_token_equivalent: true
    error_rate_overall_max: 0.01
    error_rate_injection_window_max: 0.10
    recovery_time_max_s: 30

  # Log scanning
  log_scan:
    hard_zero_patterns:
      - "Broken promise"
      - "NO RECOVERY"
      - "Segfault"
      - "SIGSEGV"
      - "0xffffffffffffffff"
      - "Poisoned .* cache transfer buffer"

  # KV cache leak detection
  kv_cache_growth_max: 0.10   # final utilization ≤ baseline + 10 percentage points
```

### Backend-knob translation (V1/V2 + C++/Python)

The `kv_cache_manager` and `transceiver` knobs in `stress_config:`
control config that doesn't have a clean field in the existing schema.
Implementer must verify the *current* mechanism for each knob:

- **V1/V2 KV cache manager** — there's a `kv_cache_manager_class` (or
  similarly named) option somewhere in the TRT-LLM config surface;
  verify the exact field name and supported values by reading
  `tensorrt_llm/_torch/pyexecutor/resource_manager.py` and the
  surrounding config classes. The existing test_llm_api_pytorch tests
  use `v1_kv_cache` as a parametrize ID — find the corresponding
  Python-side config field.
- **C++ vs Python transceiver** — controls whether the NIXL agent
  goes through C++ bindings (`tensorrt_llm/_torch/disaggregation/nixl/_agent_cpp.py`)
  or pure Python (`tensorrt_llm/_torch/disaggregation/nixl/_agent_py.py`).
  Verify the exact selection mechanism (env var? config field?).

If neither knob is currently surfaceable through YAML, the harness
can set the appropriate env vars at worker-launch time before invoking
`setup_disagg_cluster`. Document the mapping clearly in the harness
module's docstring.

## File layout for the new code

Per the user's preference for a dedicated directory (to attach a README):

```
tests/integration/defs/stress_test/disagg_cancel/
├── README.md                       # User-facing: how to run, what to expect, troubleshooting
├── __init__.py
├── harness.py                      # The marathon harness module (see "Harness architecture" below)
├── test_disagg_cancel_stress.py    # Pytest test definitions (1 per marathon scenario)
└── configs/
    ├── stress_canary_prompts.json  # Deterministic canary prompts + reference outputs
    ├── marathon_a_v1_cpp_deepseek.yaml
    ├── marathon_b_v2_py_qwen.yaml
    └── README.md                   # Per-config notes (which knobs are exercised)
```

**Note on placement:** `tests/integration/defs/stress_test/` already
contains `stress_test.py` (the existing aggregated stress test). Adding
a `disagg_cancel/` subdirectory keeps the new suite isolated while
co-locating with the existing stress-test conventions.

The pytest test IDs to register in
`tests/integration/test_lists/qa/llm_function_stress.txt`:

```
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_a_v1_cpp_deepseek]
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_b_v2_py_qwen]
```

## Harness architecture

Thread-based composition. The harness is a single class that owns five
worker threads + the disagg cluster:

```
DisaggCancellationStressHarness
├── disagg_cluster        : ctx_workers + gen_workers + disagg_server (from setup_disagg_cluster)
├── load_thread           : runs run_cancel_stress_test in a loop until duration elapses
├── canary_thread         : sends 5 canaries/min, records per-request success + token equivalence
├── injector_thread       : reads injection schedule, fires SIGSTOP/SIGCONT/SIGKILL on schedule
├── log_scanner_thread    : tails all worker logs, fails fast on hard-zero patterns
└── metrics_thread        : scrapes trtllm_kv_cache_utilization every 30s, records timeseries
```

Coordination via `threading.Event` (e.g., a `stop_event` to signal all
threads to wind down at end of test, a `failed_event` for fail-fast
signalling if log scanner spots a hard-zero pattern).

### Why threads, not async

The canary client and load client are HTTP — async would be cleaner.
But the injector is fundamentally subprocess-based (`os.kill`,
re-launching workers), the log scanner is file-tailing, and the
metrics scraper is HTTP-or-subprocess. Forcing all of these into one
asyncio event loop creates more coupling than it saves. Threads keep
each component failure-isolated and easier to debug independently.

The existing `run_cancel_stress_test` is async internally (`asyncio.run(...)`
inside the function). Run it inside the `load_thread` as-is; each
thread can have its own asyncio event loop.

### Fail-fast behaviour

Three failure modes:

1. **Hard-zero pattern** (log scanner spots it) — set `failed_event` immediately, terminate other threads, fail the pytest assertion with the pattern and the offending log line.
2. **Worker process exits unexpectedly** — `disagg_cluster.assert_all_alive()` called periodically; same as above.
3. **Pass-criteria gate violation at end** (canary error rate, recovery time, KV growth) — collected at end of test, asserted in the pytest test function.

### Worker respawn on SIGKILL

The injector needs to be able to re-launch a worker after SIGKILL.
This requires:

- Recording the worker's launch command, env, and log file when the
  cluster is initially set up. `setup_disagg_cluster` returns worker
  handles — verify whether the handle exposes enough to relaunch, or
  add a `relaunch()` method to the worker handle class.
- Updating the cluster's view of "which worker is which port" if the
  respawn comes back on a different port.
- Waiting for the respawned worker's `/health` to return 200 before
  considering the respawn complete (within `respawn_within_s`).

If this turns out to be too invasive, the alternative is to limit the
SIGKILL injection to the kill-only step (skip the respawn) and verify
that the *remaining* workers absorb the load. That's a softer test but
still meaningful — 3P3D should survive losing one of six workers.
Document the choice taken.

## Canary references — how to generate

A one-shot script (separate from the pytest test) that:

1. Launches a single inference engine with the same model + engine
   config (TP, dtype, etc.) as the marathon would use.
2. Runs each of the canary prompts with greedy decoding + fixed seed.
3. Records the output token IDs (not the detokenized text — token-level
   equivalence is what we check).
4. Writes the prompt + reference token IDs to
   `stress_canary_prompts.json`.

Suggested location:
`tests/integration/defs/stress_test/disagg_cancel/tools/generate_canary_references.py`

The README.md should document when to regenerate (model checkpoint
change, dtype change, max_seq_len change, anything that could shift
greedy-decode outputs).

## What's deferred (follow-up YAMLs / test IDs)

The harness is parametric, so all of the below land as additional YAML
configs + test IDs in `llm_function_stress.txt`, **no Python changes
needed**:

| Deferred scenario | Notes |
|---|---|
| 1P1D local | Catches single-pair edge cases; small additional time budget |
| 4P2D local | Asymmetric P/D ratio; tests sender-bias scenarios |
| V1 + Python combination | The third valid (KV-cache × transceiver) combination |
| Direct UCX transport | Exercises the non-NIXL code path (no poison on UCX cancellations) |
| Block reuse off | Verify the cancellation paths work without rc13-style block reuse |
| Overlap scheduler off | Verify behaviour without overlap |
| Aggressive timeout (1 s) | The §10 deterministic-race-regression variant; ~20 min run |
| Multi-node (cross-node 1P1D) | Deployment-infra concern; separate effort |

Track these as a follow-up issue, file/add YAML configs as needed.

## Open questions for the implementing agent to resolve

1. **`run_cancel_stress_test` integration.** The function exits when
   its inner `asyncio.run(...)` completes. For a marathon, it must
   loop until `stop_event` is set. Two options:
   - Wrap the existing function in a `while not stop_event:` loop in
     the harness's load_thread (multiple short runs back-to-back).
   - Refactor `run_cancel_stress_test` to take a duration parameter
     and run continuously.
   The first is lower-touch and recommended.

2. **Burst-mode vs steady-state mixing.** The current
   `run_cancel_stress_test` is burst-only. Either extend it to also
   send steady-state traffic between bursts, or add a separate
   `run_steady_state_load` companion. Pick based on what's simpler.

3. **Exact field name for V1/V2 KV cache manager** — verify against
   current code. The investigation references this as
   `kv_cache_manager_class` informally; find the canonical name.

4. **Exact mechanism for selecting C++ vs Python transceiver** —
   verify against current code. May be an env var
   (`TRTLLM_USE_PYTHON_NIXL_AGENT`?), a Python config option, or
   model-specific selection logic.

5. **`setup_disagg_cluster` worker-handle API** — verify that the
   handle supports SIGKILL + relaunch, or that it can be extended to.

6. **Model selection for both marathons** — pick specific HF model
   paths that:
   - Are already in the CI model cache (`$LLM_MODELS_ROOT`)
   - Are big enough to exercise non-trivial KV transfer times (the
     existing `DeepSeek-V3-Lite-bf16` may be too small)
   - Match the "DeepSeek-class MLA" and "Qwen-class non-MLA"
     archetypes
   - Fit in single 8-GPU node with TP=1 per worker (3P3D = 6 GPUs)

7. **Greedy-decode determinism.** Confirm that the TRT-LLM PyTorch
   backend produces deterministic outputs under greedy decoding +
   fixed seed across runs. If not, the canary token-equivalence check
   needs an alternative (text-equivalent after detokenize, or BLEU
   threshold).

## Acceptance criteria for the implementation PR

The implementation PR should land:

- [ ] `tests/integration/defs/stress_test/disagg_cancel/` directory created with the structure above.
- [ ] `harness.py` module implementing `DisaggCancellationStressHarness` with the 5-thread architecture.
- [ ] `test_disagg_cancel_stress.py` with one pytest test parametrized over the two YAML configs.
- [ ] `configs/marathon_a_v1_cpp_deepseek.yaml` and `configs/marathon_b_v2_py_qwen.yaml`.
- [ ] `configs/stress_canary_prompts.json` with deterministic prompts + reference token IDs for the chosen models.
- [ ] `tools/generate_canary_references.py` for regenerating references.
- [ ] `README.md` in the new directory — usage, goals, expected results, troubleshooting.
- [ ] Two test IDs registered in `tests/integration/test_lists/qa/llm_function_stress.txt`.
- [ ] At least one full-duration (2 h) successful run of each marathon on the developer's machine before submitting for review.
- [ ] Clear log output during the test (progress markers every minute, injection events logged loudly).
- [ ] Documented "how to debug a failure" section in the README — which logs to check, which Prometheus metrics to inspect.

## Cross-references

- [`README.md`](README.md) — overall TRTLLM-12721 design doc this is part of.
- [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/`](../../investigations/nvbug-6104831-disagg-permanent-wedge/) — the investigation that motivates this suite.
- [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md) — the §10 ablation experiments this suite generalises into a CI gate.
- [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) — the weekly stress CI JIRA ticket this satisfies.
- [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) — the cancellation/poison improvement initiative this is Phase 0 of.
- <https://github.com/NVIDIA/TensorRT-LLM/pull/13713> — the disaggregated cancellation / poison bug fix this suite is gating regressions against.
