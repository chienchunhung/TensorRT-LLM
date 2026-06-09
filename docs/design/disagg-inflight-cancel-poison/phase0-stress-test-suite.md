# Phase 0 - Disaggregated Cancellation Stress-Test Suite

| | |
|---|---|
| **Phase** | 0, prerequisite to behavioural cancellation / poison changes |
| **JIRA** | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) for weekly stress CI; part of [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Owner** | Chien-Chun Hung |
| **Status as of 2026-06-08** | Skeleton, `log_scanner_thread`, `metrics_thread`, `injector_thread`, and `canary_thread` have landed in `upstream/main` under `tests/integration/defs/stress_test/disagg_cancel/`. The remaining implementation work is `load_thread`, marathon configs / canary references, pytest registration, and at least one full-duration run of each marathon. |
| **Next step** | Implement `load_thread` as a duration-bounded wrapper around the existing cancellation load generator, then wire the two marathon YAMLs and canary references. |

## Scope And Motivation

Phase 0 is the permanent regression gate for the NVBug 6104831
failure class: disaggregated KV-transfer cancellation races that can
leave the system wedged, crash worker processes, or force a fail-closed
pool-wide poison cascade.

The immediate motivation is the cancellation / poison work introduced
at <https://github.com/NVIDIA/TensorRT-LLM/pull/13713>. That change is
memory-safe and deliberately fail-closed: if a request is cancelled
while the remote peer may still be reading or writing an advertised
transfer buffer, TRT-LLM poisons the transfer pool and the Python
executor shuts down. That avoids UAF, but it is operationally
aggressive, so the feature shipped gated and default-OFF.

The customer-visible pressure came from the 2026-05-13 Qwen3-Coder-480B
incident: the default 60 s transfer timeout fired during transient
backpressure, both transfer pools became poisoned, containers restarted
repeatedly, and the serving instance recycled. The investigation also
found related cleanup-path, lifetime, and block-reuse failures. Phase 0
turns those ad-hoc reproductions into an in-repository stress suite
that can run continuously before later phases change behaviour.

The suite tests one contract:

> A disaggregated TRT-LLM deployment can run for hours under
> cancellation-heavy load, transient peer pauses, and worker loss
> without permanent failure.

"Permanent failure" means any of:

- Process crash, including SIGSEGV, abort, or unexpected worker exit.
- Permanent server wedge after a transient event ends.
- Silent memory corruption, detected by deterministic canary responses
  that no longer match references.
- Monotonic resource leak, detected by KV cache utilization growth.
- Cross-rank divergence once consensus-focused configs are added.

Transient failures during stress are allowed. Request errors,
cancellations, retries during bursts, and short canary error spikes
during injected peer pauses are expected. The assertion is graceful
recovery: after SIGCONT, worker-loss absorption, or burst completion,
the deployment must return to a healthy baseline within a bounded time.

### Coverage Goals

The initial weekly suite covers the two mainstream disaggregated
deployment cells:

| Cell | Why it matters | Initial coverage |
|---|---|---|
| V1 KV cache manager + C++ transceiver + NIXL | Established production path and the cell extended by the fail-closed cancellation work | Marathon A |
| V2 KV cache manager + Python transceiver + NIXL | Newer path and the architectural template for consensus semantics | Marathon B |

The harness must also be parametric enough to add follow-up YAMLs for
the third valid cell (V1 + Python), direct UCX, block-reuse off, overlap
off, aggressive timeout, asymmetric P/D ratios, and cross-node
multi-node coverage without rewriting Python harness code.

The bug patterns to catch are:

| Pattern | Example signal | Test pressure |
|---|---|---|
| Cleanup-path race | Broken promise, stuck futures, no recovery after peer pause | Long-running cancellation load plus SIGSTOP/SIGCONT |
| Lifetime UAF | Crashes or corrupt canary output after request cancellation | Cancel-during-transfer load and deterministic canaries |
| Poison cascade | Pool-wide poison followed by executor shutdown / restart | 60 s timeout under high concurrency and injected receiver slowness |
| Block-reuse interaction | KV blocks pinned or reclaimed incorrectly | Block reuse enabled in both initial marathons |
| Worker-loss absorption | Deployment does not survive one worker loss in a 3P3D shape | SIGKILL of one worker with kill-only fallback |
| Consensus divergence | Rank-batch mismatch, collective deadlock, unreclaimed KV blocks | Follow-up TP/PP/EP consensus-focused configs before Phase 1 claims full axis coverage |

### Initial Non-Scope

- Cross-node multi-node is deferred. The observed cancellation class
  reproduces single-host, and true cross-node coverage mostly adds
  deployment-infrastructure variables.
- TP greater than 1 per worker is deferred from the first two marathon
  configs to keep the weekly suite within a single 8-GPU node. The
  consensus dimension is tracked as follow-up coverage, not forgotten.
- Backend-internal cancellation primitives for UCX/MPI/Mooncake are not
  part of Phase 0. Phase 0 validates the TRT-LLM harness and regression
  signatures; later phases decide backend behaviour.

## Continuous Test Requirements

### Budget And Frequency

| Requirement | Value |
|---|---|
| Weekly CI budget | 4 h total |
| Test composition | Two serial 2 h marathons |
| Local developer mode | Optional smoke mode around 10 min, with one burst and one injection |
| CI frequency | Weekly stress CI for TRTLLM-12648; opt-in for changes touching disagg cancellation, KV transfer cleanup, or the stress harness |
| Required environment | `LLM_MODELS_ROOT` set to the chosen model root; NIXL-capable local disagg setup |

### Hardware Budget

The initial suite targets one 8-GPU B200 or H100 node. With 3P3D and
TP=1 per worker, each marathon uses six GPUs for workers, leaving two
for the disagg server and clients. The two marathons run serially.

### Marathon Configurations

| Test ID | Shape | Config | Model class | Duration |
|---|---|---|---|---|
| `marathon_a_v1_cpp_deepseek` | 3P3D local | V1 + C++ transceiver + NIXL, block reuse on, overlap on, 60 s transfer timeout | DeepSeek-class | 2 h |
| `marathon_b_v2_py_qwen` | 3P3D local | V2 + Python transceiver + NIXL, block reuse on, overlap on, 60 s transfer timeout | Qwen-class | 2 h |

3P3D is deliberate:

- It exercises multi-pair cleanup paths instead of only a single
  context/generation pair.
- It gives redundancy for the kill-only SIGKILL injection.
- It is close to common production disaggregated shapes while still
  fitting on one node.

The valid KV-cache / transceiver combinations are:

| Combination | Initial plan |
|---|---|
| V1 + C++ | Marathon A |
| V1 + Python | Follow-up YAML |
| V2 + C++ | Invalid; reject in config validation |
| V2 + Python | Marathon B |

### Workload Pattern

Each marathon repeats the following pattern for 2 h.

**Steady state**

- 64 concurrent in-flight requests.
- Prompt length uniformly distributed from 4k to 12k input tokens.
- 512 output tokens.
- 10 percent client-side cancellation rate, disconnecting during
  prefill.

**Bursts**

- Every 8 min, ramp to 256 concurrent requests for 60 s.
- Prompt length uniformly distributed from 12k to 16k input tokens.
- Return to steady state after the burst.

**Injections**

```text
T+ 15 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+ 30 min : SIGSTOP random gen worker for 30 s, then SIGCONT
T+ 45 min : SIGSTOP random ctx worker for 20 s, then SIGCONT
T+ 60 min : SIGKILL gen_worker_0, with kill-only absorption in the initial suite
T+ 75 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+ 90 min : SIGSTOP random ctx worker for 30 s, then SIGCONT
T+105 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+120 min : end
```

SIGSTOP/SIGCONT creates transient peer pauses, which exercise
cancel-mid-flight plus recovery. SIGKILL creates terminal worker loss;
the initial shipped shape validates that the remaining five workers
absorb load. Full respawn support is a follow-up once worker handles
expose a stable relaunch API.

### Canary Requirements

The canary client runs in parallel for the full marathon:

- 5 requests per minute.
- Fixed prompt set loaded from `stress_canary_prompts.json`.
- Greedy decoding with fixed seed.
- Reference token IDs generated once with the same model and engine
  config, committed beside the YAMLs.
- Exact token-equivalence check when deterministic; fallback order is
  exact detokenized text, BLEU / ROUGE threshold, then length-only
  sanity. Each fallback is weaker and must be documented in the test
  README.

Reference outputs are generated by:

```text
tests/integration/defs/stress_test/disagg_cancel/tools/generate_canary_references.py
```

Regenerate references when the model checkpoint, dtype, tokenizer,
max sequence length, or engine config changes in a way that can affect
greedy output.

### Pass Criteria

| Gate | Threshold |
|---|---|
| Hard-zero log patterns | 0 occurrences in any worker log |
| Worker liveness | Every non-intentionally-killed worker alive at end |
| Final health probe | 5 sequential canaries succeed within 30 s of test end |
| Canary correctness | 100 percent of returned canaries token-equivalent to reference, unless a documented fallback is active |
| Canary error rate, overall | Less than 1 percent over the full marathon |
| Canary error rate, burst / injection window | Less than 10 percent during any 1 min window containing burst or injection |
| Recovery time after injection | Less than 30 s from SIGCONT or accepted worker-loss event until canary error rate returns below 1 percent |
| KV cache utilization growth | End-of-test utilization no more than baseline plus 10 percentage points |

Hard-zero log patterns include:

- `Broken promise`
- `NO RECOVERY`
- `Segfault`
- `SIGSEGV`
- `0xffffffffffffffff`
- `Poisoned .* cache transfer buffer`

`Cannot cancel request` is not capped. It is expected at the boundary
where cancellation races an in-flight transfer.

### YAML Configuration Contract

The harness reads one YAML file per marathon. The normal disaggregated
worker config is passed through to `setup_disagg_cluster`. Phase 0 adds
a top-level `stress_config:` block.

```yaml
hostname: localhost
model: <model path under LLM_MODELS_ROOT>
backend: pytorch

context_servers:
  num_instances: 3
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  disable_overlap_scheduler: false
  max_num_tokens: 16384
  max_seq_len: 16384
  kv_cache_config:
    enable_block_reuse: true
    enable_partial_reuse: true
    free_gpu_memory_fraction: 0.3
  cache_transceiver_config:
    backend: NIXL
    max_tokens_in_buffer: 16384
    kv_transfer_timeout_ms: 60000

generation_servers:
  num_instances: 3
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  # Same shape as context_servers.

stress_config:
  duration_min: 120
  kv_cache_manager: v1        # v1 | v2
  transceiver: cpp            # cpp | python; v2 + cpp is invalid

  base_concurrency: 64
  client_cancel_rate: 0.10
  input_length:
    distribution: uniform
    min_tokens: 4096
    max_tokens: 12288
  output_length: 512

  bursts:
    interval_min: 8
    concurrency: 256
    duration_s: 60
    input_length:
      distribution: uniform
      min_tokens: 12288
      max_tokens: 16384

  injections:
    - at_min: 15
      type: sigstop
      target: gen_worker_random
      duration_s: 20
    - at_min: 60
      type: sigkill
      target: gen_worker_0

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

  log_scan:
    hard_zero_patterns:
      - "Broken promise"
      - "NO RECOVERY"
      - "Segfault"
      - "SIGSEGV"
      - "0xffffffffffffffff"
      - "Poisoned .* cache transfer buffer"

  kv_cache_growth_max: 0.10
```

The harness owns translation of `kv_cache_manager` and `transceiver`
into the current worker config / environment variables. Implementers
must verify the exact names against the current config classes before
landing the YAML step, and document the mapping in the harness module
docstring.

## Harness Architecture

The implementation lives under:

```text
tests/integration/defs/stress_test/disagg_cancel/
```

It reuses the existing disaggregated test infrastructure instead of
creating a parallel launcher:

| Existing file / symbol | Use |
|---|---|
| `setup_disagg_cluster(...)` | Launch context workers, generation workers, and disagg server from YAML |
| `wait_for_server(...)` | Probe readiness |
| `cleanup_output_files(...)` and `terminate(...)` | Standard teardown |
| `run_cancel_stress_test(...)` | Existing cancellation-heavy load coroutine; `load_thread` wraps it |
| `disagg_config_cancel_stress_test*.yaml` | Starting point for worker config schema |

The harness owns the cluster and five worker threads:

```text
DisaggCancellationStressHarness
├── disagg_cluster        : ctx_workers + gen_workers + disagg_server
├── load_thread           : repeats cancellation-heavy load until duration elapses
├── canary_thread         : sends canaries, records success and token equivalence
├── injector_thread       : fires SIGSTOP/SIGCONT/SIGKILL on schedule
├── log_scanner_thread    : tails worker logs and fails fast on hard-zero patterns
└── metrics_thread        : scrapes trtllm_kv_cache_utilization every 30 s
```

Threads coordinate through `threading.Event`:

- `stop_event` asks all threads to wind down.
- `failed_event` triggers fail-fast shutdown.
- Per-thread result objects carry metrics and assertion context back
  to the pytest entry point.

Threads are preferred over a single asyncio event loop because the
components are heterogeneous: HTTP load, deterministic canaries,
subprocess signals, worker relaunch decisions, file tailing, and
Prometheus scraping. Keeping each component in a thread makes failures
isolated and easier to debug.

Fail-fast behaviour:

1. A hard-zero log pattern sets `failed_event` immediately and records
   the offending log line.
2. Unexpected worker exit sets `failed_event`, except for the
   intentionally killed worker in the kill-only scenario.
3. End-of-test gates assert canary correctness, recovery time, final
   health, and KV utilization growth.

### File Layout

```text
tests/integration/defs/stress_test/disagg_cancel/
├── README.md
├── __init__.py
├── harness.py
├── test_disagg_cancel_stress.py
└── configs/
    ├── stress_canary_prompts.json
    ├── marathon_a_v1_cpp_deepseek.yaml
    ├── marathon_b_v2_py_qwen.yaml
    └── README.md
```

Planned stress test-list entries:

```text
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_a_v1_cpp_deepseek]
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_b_v2_py_qwen]
```

## Implementation Roadmap

Phase 0 lands incrementally, one component at a time, so the harness can
be reviewed and exercised before the full 4 h suite is enabled.

| Step | Boundary | Status | Upstream link |
|---|---|---|---|
| 1 | Harness skeleton, initial config, README, `log_scanner_thread` | Merged 2026-05-28 | <https://github.com/NVIDIA/TensorRT-LLM/pull/14375> |
| 2 | `metrics_thread` and KV-utilization time series | Merged 2026-06-02 | <https://github.com/NVIDIA/TensorRT-LLM/pull/14807> |
| 3 | `injector_thread`, SIGSTOP/SIGCONT/SIGKILL schedule, kill-only worker-loss handling | Merged 2026-06-04 | <https://github.com/NVIDIA/TensorRT-LLM/pull/14920> |
| 4 | `canary_thread`, deterministic prompts, token-equivalence checks | Merged 2026-06-08 | <https://github.com/NVIDIA/TensorRT-LLM/pull/15015> |
| 5 | `load_thread`, duration-bounded wrapper around `run_cancel_stress_test` | Next | - |
| 6 | Marathon YAMLs, canary JSON, reference-generation tool | Pending | - |
| 7 | Pytest marathon entry point and stress test-list registration | Pending | - |

The current status is:

- Done: harness structure, log scanner, metrics scraper, injector, and
  canary client.
- In progress next: `load_thread`.
- Pending after that: two full marathon configs, canary references,
  reference-generation tool, pytest parametrization, test-list
  registration, and full-duration validation.

### Step 5 - Load Thread

`load_thread` should drive the existing
`run_cancel_stress_test(server_url, num_bursts, requests_per_burst,
prompt_len_range, cancel_after_range)` implementation repeatedly until
the configured duration elapses. It must:

- Maintain steady-state load and burst windows from `stress_config`.
- Respect `stop_event` and `failed_event`.
- Record request counts, cancellation counts, load errors, and burst
  timestamps for correlation with canary and metrics output.
- Avoid refactoring the existing disaggregated cancel test unless the
  current coroutine shape makes wrapping impossible.

### Step 6 - Marathon Configs And References

Add:

- `configs/marathon_a_v1_cpp_deepseek.yaml`
- `configs/marathon_b_v2_py_qwen.yaml`
- `configs/stress_canary_prompts.json`
- `tools/generate_canary_references.py`
- `configs/README.md` describing model assumptions and config knobs.

This step should also pin the exact config-field / env-var translation
for V1/V2 and C++/Python selection.

### Step 7 - Pytest And CI Registration

Add a parametrized pytest entry point over the two marathon YAMLs and
register the two test IDs in:

```text
tests/integration/test_lists/qa/llm_function_stress.txt
```

This step should also update the test README with:

- How to run full marathons.
- How to run smoke mode.
- How to inspect logs and metrics after failure.
- Which failures are expected transient events and which fail the test.

### Acceptance Checklist

- [x] `tests/integration/defs/stress_test/disagg_cancel/` directory exists.
- [x] `harness.py` contains the five-thread architecture.
- [x] `README.md` exists in the stress-test directory.
- [x] Log scanner fails fast on hard-zero patterns.
- [x] Metrics scraper records `trtllm_kv_cache_utilization`.
- [x] Injector supports scheduled SIGSTOP/SIGCONT/SIGKILL.
- [x] Canary thread records deterministic canary results.
- [ ] `load_thread` runs cancellation-heavy load for the configured duration.
- [ ] Marathon A and Marathon B YAMLs are committed.
- [ ] Canary prompts and reference token IDs are committed.
- [ ] Reference-generation tool is committed.
- [ ] Parametrized pytest marathon entry point is committed.
- [ ] Two weekly-stress test IDs are registered.
- [ ] Each marathon has at least one successful full-duration run before weekly CI enablement.
- [ ] The stress-test README documents failure debugging.

## Follow-Up Coverage

The harness should support these as extra YAMLs / test IDs without
Python rewrites:

| Scenario | Why |
|---|---|
| 1P1D local | Single-pair edge cases |
| 4P2D local | Asymmetric sender/receiver pressure |
| V1 + Python | Third valid KV-cache / transceiver combination |
| Direct UCX | Non-NIXL transport behaviour |
| Block reuse off | Cancellation without the block-reuse interaction |
| Overlap scheduler off | Cancellation without overlap scheduling |
| Aggressive 1 s timeout | Deterministic race-regression variant from the investigation |
| TP/PP/EP consensus configs | Rank-batch divergence and collective-deadlock coverage |
| Cross-node 1P1D | True multi-node RDMA and deployment-infra coverage |

## Cross-References

- [`README.md`](README.md) - overall TRTLLM-12721 design roadmap.
- [`phase1-architectural-design.md`](phase1-architectural-design.md) - in-flight cancellation workflow and implementation plan gated by this suite.
- [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/`](../../investigations/nvbug-6104831-disagg-permanent-wedge/) - investigation motivating the suite.
- [`10-ablation-no-midflight-cancel.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md) - controlled A/B experiments generalized by this suite.
- [`appendix-v1-consensus-collective.md`](appendix-v1-consensus-collective.md) - V1 consensus collective appendix that will be measured against Phase 0 configs.
