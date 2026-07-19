# Phase 0 - Disaggregated Cancellation Stress-Test Suite

| | |
|---|---|
| **Phase** | 0, prerequisite to behavioural cancellation / poison changes |
| **JIRA** | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) for weekly stress CI; part of [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Owner** | Chien-Chun Hung |
| **Status as of 2026-07-19** | The harness, five harness-thread bodies, two YAMLs, pytest entry point, and C++/V1 QA registration have landed through PR #15174. The registered marathon remains a 10 min `log_only` guard; `full_cancel_poison` and the Python/V2 YAML are not enabled in CI. PR #16402 is separately validating the finite Qwen3-32B deadline/cancellation acceptance test. |
| **Next step** | Complete PR #16402's focused and full CI gates, then qualify a bounded `full_cancel_poison` run before enabling it. Keep admission tuning and the original 10,000-request soak as separately measured follow-up work. |

## Scope And Motivation

Phase 0 is intended to become the permanent regression gate for the
NVBug 6104831 failure class: disaggregated KV-transfer cancellation
races that can leave the system wedged, crash worker processes, or
force a fail-closed pool-wide poison cascade.

The initial cancellation/poison prototype was reviewed in
<https://github.com/NVIDIA/TensorRT-LLM/pull/13713> but did not merge.
The gated implementation later merged in
<https://github.com/NVIDIA/TensorRT-LLM/pull/15238>. Its active-transfer
cancellation path is deliberately default-off while the project
qualifies lifecycle, peer-consensus, and poison-buffer behavior.

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
recovery: after SIGCONT, worker respawn, or burst completion,
the deployment must return to a healthy baseline within a bounded time.

## Current Test Inventory And Roles

Two disaggregated stress-test families now cover related failure modes.
They share cluster setup and exercise the C++/V1 transfer path, but they
have different operating contracts and should not be treated as
interchangeable evidence.

| Dimension | Qwen3 finite QA acceptance test | Phase 0 cancellation marathon |
|---|---|---|
| Exact test | `disaggregated/test_disaggregated.py::test_disaggregated_stress_test[input8k-output1k-conc512-qwen3_32b_fp8_stress]` | `stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_cpp_v1_deepseek.yaml]` |
| Origin | PR #14278; PR #16008 later added 10 percent cancellation | PRs #14375, #14807, #14920, #15015, #15124, and #15174 |
| Model and shape | Qwen3-32B FP8 plus Eagle3; 4 context TP1 workers and 1 generation TP4 worker | DeepSeek-V3-Lite; 3 context and 3 generation TP1 workers |
| Load | Upstream profile: 10,000 requests at concurrency 512, 8k input and 1k output. PR #16402 uses a bounded 512-request acceptance profile at the same concurrency. | Registered profile: one normal probe every 30 s for 10 min. The full-mode YAML intends batches of 64, periodic 256-request batches, 4k-16k inputs, and 512 output tokens for 2 h. Today the driver treats the concurrency values as batch sizes and hard-codes 10 output tokens. |
| Cancellation | AIPerf disconnects 10 percent of requests after 0.5 s. PR #16402 keeps active C++ in-flight cancellation default-off, so active transfers retain ownership and drain. | `log_only` sends no cancellations. The full-mode YAML specifies 10 percent, but that knob is not wired: the current custom load disconnects every request in each batch. Full mode is opt-in and unregistered. |
| Fault injection | None | `full_cancel_poison` schedules SIGSTOP/SIGCONT and SIGKILL/respawn events. |
| Primary assertions | Direct end-to-end gates are a successful AIPerf process, no configured fatal log signatures, a successful follow-on GSM8K run, and score at or above 0.42. The lifecycle contract exercised by PR #16402 is that every client operation terminates within a bound: deliberately disconnected clients cancel, other requests complete or receive an explicit terminal error, and server-side ownership and router reservations are released exactly once. | No fatal log signatures and at least one successful normal probe in `log_only`. Full mode is designed to add canary correctness, bounded recovery, worker liveness, and KV-utilization growth, but those aggregate gates are not implemented yet. |
| Intended cadence | Focused QA gate for changes to proxy deadline, cancellation terminalization, and cleanup; then ordinary full PR CI. | Short registered guard today; full fault-injection soak should run weekly/on demand after qualification, not on every PR. |
| Verified boundary | PR #14278 validated only a smoke and a temporary 16-request reduction. PR #16402 experimental heads passed the 512-request profile with admission multipliers 1 and 2. Validation must be repeated on its final head before merge. The original 10,000-request profile remains unverified. | Unit/component coverage is established and the real `log_only` cluster path is registered. Neither full-mode YAML has a successful full-duration CI qualification, and the current pytest collector does not enforce the proposed aggregate full-mode gates. |

PR #16008 also added
`test_disaggregated_mixed_stress_test[req10k-conc512-qwen3_32b_fp8_mixed_stress]`.
That variant mixes normal, streaming, structured-output, and cancelled
requests, but its 10,000-request profile was explicitly not validated
end to end when merged. It is adjacent to the finite acceptance test,
not a substitute for the fault-injection marathon.

### Consolidation Decision

Keep the finite QA acceptance test and the Phase 0 marathon separate:

- The finite Qwen test is an end-to-end request-lifecycle gate. It
  should stay bounded enough to provide a result during PR validation
  and should make terminal success, cancellation, or timeout explicit.
- The marathon is a system-integrity soak. Its unique value is elapsed
  time, deliberate worker faults, canaries, log scanning, and resource
  leak detection. Folding those concerns into the finite test would
  make failures slower and harder to attribute.
- Share helpers rather than workloads: cluster launch/teardown,
  terminal-outcome accounting, cancellation telemetry, cleanup timing,
  log preservation, and post-load recovery probes should have one
  implementation where practical.
- Do not claim `full_cancel_poison` coverage from the registered
  marathon until the YAML mode is actually enabled and qualified.
- Before comparing full-mode throughput or cancellation rates with the
  finite Qwen test, wire `client_cancel_rate`, `output_length`, and a
  true maintained-concurrency contract into the marathon load driver.

There is a narrower future consolidation question between the two
10,000-request Qwen variants. Once the mixed-stress profile has a clean
end-to-end qualification, compare its unique assertions with the
legacy uniform AIPerf soak and retain only one long Qwen soak if their
signals are redundant. That decision does not affect the Phase 0
fault-injection marathon.

### Validation Policy

For a PR that changes disaggregated request lifetime or cleanup:

1. Run unit tests for timeout propagation, queued cancellation,
   exact-once router cleanup, and terminal accounting.
2. Run the exact finite Qwen QA test using:

   ```text
   /bot run --only-qa-verify test disaggregated/test_disaggregated.py::test_disaggregated_stress_test[input8k-output1k-conc512-qwen3_32b_fp8_stress]
   ```

3. After the focused test passes on the current commit, run the full PR
   pipeline with `/bot run --disable-fail-fast`.
4. Run the full marathon separately for soak/fault qualification. It is
   not a replacement for the focused gate and need not block every PR
   while in-flight cancellation remains default-off.

### Coverage Goals

The full-mode design targets two mainstream disaggregated deployment
cells. Current CI coverage is narrower and must be reported separately
from the target:

| Cell | Why it matters | Current state |
|---|---|---|
| V1 KV cache manager + C++ transceiver + NIXL | Established production path and the cell extended by the fail-closed cancellation work | `marathon_cpp_v1_deepseek.yaml` is registered, but only in `log_only` mode. |
| V2 KV cache manager + Python transceiver + NIXL | Newer path and the architectural template for consensus semantics | `marathon_python_v2_qwen.yaml` is a parse-validated template; it is not parametrized or registered. |

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
| Worker-loss recovery | Deployment does not recover after one worker is killed in a 3P3D shape | SIGKILL of one worker followed by bounded respawn |
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
| Current registered budget | One 10 min `log_only` run with a 45 min test-list timeout including setup and teardown |
| Target weekly CI budget | 4 h total |
| Target full-mode composition | Two serial 2 h marathons after both are qualified |
| Manual shortened full-mode profile | Proposed 10 min profile with one burst and one injection; this is a duration/configuration override, not a separate harness mode |
| CI frequency | Focused Qwen QA acceptance on relevant PRs; Phase 0 soak weekly/on demand after qualification |
| Required environment | `LLM_MODELS_ROOT` set to the chosen model root; NIXL-capable local disagg setup |

### Hardware Budget

The Phase 0 marathon targets one 8-GPU B200 or H100 node. With 3P3D
and TP=1 per worker, it uses six GPUs for workers. The disagg server
and clients do not consume the remaining GPUs. If both full-mode
marathons are enabled, they run serially.

### Marathon Configurations

| YAML / pytest identity | Shape | Config | Enabled behavior | Status |
|---|---|---|---|---|
| `marathon_cpp_v1_deepseek.yaml` | 3P3D local | V1 + C++ transceiver + NIXL, block reuse and overlap on, 60 s transfer timeout | 10 min `log_only`; 2 h `full_cancel_poison` values are present but inactive | Parametrized and QA-registered |
| `marathon_python_v2_qwen.yaml` | 3P3D local | V2 + Python transceiver + NIXL | Intended 2 h `full_cancel_poison` profile | Template only; not parametrized or registered |

3P3D is deliberate:

- It exercises multi-pair cleanup paths instead of only a single
  context/generation pair.
- It gives redundancy while a SIGKILL target is restarted.
- It is close to common production disaggregated shapes while still
  fitting on one node.

The valid KV-cache / transceiver combinations are:

| Combination | Coverage plan |
|---|---|
| V1 + C++ | Registered DeepSeek marathon; full mode still needs qualification |
| V1 + Python | Follow-up YAML |
| V2 + C++ | Invalid; reject in config validation |
| V2 + Python | Qwen template; parametrization and qualification pending |

### Workload Pattern

The following is the intended `full_cancel_poison` pattern. It is not
run by the currently registered `log_only` mode, and its load knobs are
not fully wired today: `base_concurrency` and burst `concurrency` select
the number of requests in a sequential batch, `client_cancel_rate` is
unused because every request is disconnected, and `output_length` is
unused because the shared load generator hard-codes 10 output tokens.

**Target steady state**

- 64 concurrent in-flight requests.
- Prompt length uniformly distributed from 4k to 12k input tokens.
- 512 output tokens.
- 10 percent client-side cancellation rate, disconnecting during
  prefill.

**Target bursts**

- Every 8 min, ramp to 256 concurrent requests for 60 s.
- Prompt length uniformly distributed from 12k to 16k input tokens.
- Return to steady state after the burst.

**Injections**

```text
T+ 15 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+ 30 min : SIGSTOP random gen worker for 30 s, then SIGCONT
T+ 45 min : SIGSTOP random ctx worker for 20 s, then SIGCONT
T+ 60 min : SIGKILL gen_worker_0, respawn within 60 s
T+ 75 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+ 90 min : SIGSTOP random ctx worker for 30 s, then SIGCONT
T+105 min : SIGSTOP random gen worker for 20 s, then SIGCONT
T+120 min : end
```

SIGSTOP/SIGCONT creates transient peer pauses, which exercise
cancel-mid-flight plus recovery. SIGKILL creates terminal worker loss;
the full-mode contract requires the selected worker to be restarted
within the configured bound and the cluster to become healthy again.

### Canary Requirements

The intended canary client runs in parallel for the full marathon:

- 5 requests per minute.
- Fixed prompt set loaded from `stress_canary_prompts.json`.
- Greedy decoding with fixed seed.
- Reference token IDs generated once with the same model and engine
  config, committed beside the YAMLs.
- The current implementation compares token IDs only when
  `reference_token_ids` are present. It parses but does not use
  `reference_text`.
- A proposed fallback order is exact detokenized text, BLEU / ROUGE
  threshold, then length-only sanity. These fallbacks are not
  implemented; each is weaker and must be documented before it can
  become an accepted test outcome.

The planned reference generator location is:

```text
tests/integration/defs/stress_test/disagg_cancel/tools/generate_canary_references.py
```

Neither that tool nor `stress_canary_prompts.json` is currently checked
in. Add both before enabling `full_cancel_poison`. Regenerate references
when the model checkpoint, dtype, tokenizer, max sequence length, or
engine config changes in a way that can affect greedy output.

### Pass Criteria

The following table is the target `full_cancel_poison` contract. The
current collector returns raw observations plus `failure_reason`, and
the pytest entry point checks only result shape, fail-fast state, and a
successful probe for `log_only`. It does not yet compute or enforce the
aggregate thresholds below.

| Gate | Threshold |
|---|---|
| Hard-zero log patterns | 0 occurrences in any worker log |
| Worker liveness | The SIGKILL target is respawned within its configured bound and every worker is alive and healthy at end |
| Final health probe | 5 sequential canaries succeed within 30 s of test end |
| Canary correctness | 100 percent of returned canaries token-equivalent to reference, unless a documented fallback is active |
| Canary error rate, overall | Less than 1 percent over the full marathon |
| Canary error rate, burst / injection window | Less than 10 percent during any 1 min window containing burst or injection |
| Recovery time after injection | Less than 30 s from SIGCONT or worker respawn until canary error rate returns below 1 percent |
| KV cache utilization growth | End-of-test utilization no more than baseline plus 10 percentage points |

The registered C++/V1 YAML currently scans for:

- `Broken promise`
- `Segfault`
- `Segmentation fault`
- `SIGSEGV`
- `0xffffffffffffffff`
- `use-after-free`
- `heap-use-after-free`
- `AddressSanitizer:.*use-after-free`
- `double[- ]free`

The planned Python/V2 template additionally lists `NO RECOVERY` and
`Poisoned .* cache transfer buffer`. Before full-mode enablement, align
the two YAMLs and the runtime contract on one required hard-zero set.

The final policy for `Cannot cancel request` is intentionally unresolved
while `full_cancel_poison` is disabled. The current runtime can emit it
at the boundary where cancellation races an active transfer, whereas
the future full-mode README treats it as a candidate hard-zero signal.
Before enabling full mode, align the runtime contract, YAML log list,
and README so the test has one explicit expectation.

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
  mode: full_cancel_poison
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
      respawn_within_s: 60

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

The nested worker configuration is the runtime source of truth:
`kv_cache_config.use_kv_cache_manager_v2` selects V1/V2 and
`cache_transceiver_config.transceiver_runtime` selects C++/Python. The
harness strips `stress_config` before calling `setup_disagg_cluster`;
the top-level `kv_cache_manager` and `transceiver` fields are validation
metadata only. Current validation rejects invalid selector pairs but
does not cross-check them against the nested worker configuration. Add
that cross-check before enabling either full-mode YAML so the displayed
coverage cannot silently differ from the exercised path.

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

Current fail-fast behaviour:

1. A hard-zero log pattern sets `failed_event` immediately and records
   the offending log line.
2. A failed `log_only` probe, a load-runner failure, or expiry of a
   configured respawn deadline sets `failed_event`.
3. `wait_until_done()` observes harness events and its overall timeout;
   it does not generically monitor worker processes for unexpected
   exits. Add that monitoring before enabling full mode.
4. The target end-of-test gates assert canary correctness, recovery
   time, final health, and KV utilization growth. The current result
   collector records the inputs for those gates but does not aggregate
   or enforce them.

### File Layout

```text
tests/integration/defs/stress_test/disagg_cancel/
├── README.md
├── __init__.py
├── _testing.py
├── harness.py
├── test_disagg_cancel_stress.py
├── test_canary.py
├── test_injector.py
├── test_load_thread.py
├── test_log_scanner.py
├── test_metrics_thread.py
└── configs/
    ├── marathon_cpp_v1_deepseek.yaml
    ├── marathon_python_v2_qwen.yaml
    └── README.md
```

The current stress test-list entry is:

```text
stress_test/disagg_cancel/test_disagg_cancel_stress.py::test_disagg_cancellation_marathon[marathon_cpp_v1_deepseek.yaml] TIMEOUT (45)
```

`marathon_python_v2_qwen.yaml` is covered by parse/validation tests but
is intentionally absent from `_MARATHON_CONFIGS` and the QA list.
Canary reference JSON and its generator are also not yet checked in.

## Implementation Roadmap

Phase 0 landed incrementally, one component at a time. The harness is
implemented; enabling and qualifying its full behavior remains open.

| Step | Boundary | Status | Upstream link |
|---|---|---|---|
| 1 | Harness skeleton, initial config, README, `log_scanner_thread` | Merged 2026-05-28 | <https://github.com/NVIDIA/TensorRT-LLM/pull/14375> |
| 2 | `metrics_thread` and KV-utilization time series | Merged 2026-06-02 | <https://github.com/NVIDIA/TensorRT-LLM/pull/14807> |
| 3 | `injector_thread`, SIGSTOP/SIGCONT/SIGKILL schedule, optional respawn | Merged 2026-06-04 | <https://github.com/NVIDIA/TensorRT-LLM/pull/14920> |
| 4 | `canary_thread`, deterministic prompts, token-equivalence checks | Merged 2026-06-08 | <https://github.com/NVIDIA/TensorRT-LLM/pull/15015> |
| 5 | `load_thread`, duration-bounded wrapper around `run_cancel_stress_test` | Merged 2026-06-09 | <https://github.com/NVIDIA/TensorRT-LLM/pull/15124> |
| 6 | C++/V1 and Python/V2 YAMLs, mode switch, config validation | Merged 2026-06-10 | <https://github.com/NVIDIA/TensorRT-LLM/pull/15174> |
| 7 | Pytest entry point and C++/V1 `log_only` QA registration | Merged 2026-06-10 | <https://github.com/NVIDIA/TensorRT-LLM/pull/15174> |

The current status is:

- Done: harness structure, all five thread bodies, YAML parsing and
  validation, pytest entry point, real-cluster setup, `log_only`, and
  one registered C++/V1 test.
- Pending: canary references and generator, a qualified
  `full_cancel_poison` run, aggregate full-mode assertions,
  required-input validation, generic worker-exit monitoring,
  selector-to-worker-config cross-validation, load-rate/output wiring,
  aligned hard-zero patterns, Python/V2 parametrization, and an
  explicit weekly/on-demand schedule for full mode.

### Step 5 - Load Thread

`load_thread` currently drives the existing
`run_cancel_stress_test(server_url, num_bursts, requests_per_burst,
prompt_len_range, cancel_after_range)` implementation repeatedly until
the configured duration elapses. Its target contract is to:

- Maintain steady-state load and burst windows from `stress_config`.
- Respect `stop_event` and `failed_event`.
- Record request counts, cancellation counts, load errors, and burst
  timestamps for correlation with canary and metrics output.
- Reuse the existing cancellation load generator without creating a
  second request implementation.

Meeting that contract still requires the load-rate, output-length, and
maintained-concurrency wiring listed in the acceptance checklist.

### Step 6 - Marathon Configs And References

The two YAMLs and their config README are present. The C++/V1 YAML is
the only parametrized configuration. Before full mode is enabled, add
the referenced `stress_canary_prompts.json` and a reproducible
reference-generation tool, then record the model/config identity used
to produce those references.

### Step 7 - Pytest And CI Registration

The parametrized pytest entry point exists, but `_MARATHON_CONFIGS`
contains only `marathon_cpp_v1_deepseek.yaml`. That ID is registered in:

```text
tests/integration/test_lists/qa/llm_function_stress.txt
```

The README documents modes and invocation. Before adding Python/V2 or
turning on full mode, it must continue to explain:

- How to run full marathons.
- How to run a manually shortened full-mode profile.
- How to inspect logs and metrics after failure.
- Which failures are expected transient events and which fail the test.

### Acceptance Checklist

- [x] `tests/integration/defs/stress_test/disagg_cancel/` directory exists.
- [x] `harness.py` contains the five-thread architecture.
- [x] `README.md` exists in the stress-test directory.
- [x] Log scanner fails fast on hard-zero patterns.
- [x] Metrics scraper records `trtllm_kv_cache_utilization`.
- [x] Injector supports scheduled SIGSTOP/SIGCONT/SIGKILL.
- [x] Canary thread records request outcomes and optional token equivalence.
- [x] `load_thread` runs cancellation-heavy load for the configured duration.
- [x] C++/V1 and Python/V2 YAMLs are committed and parse-validated.
- [ ] Canary prompts and reference token IDs are committed.
- [ ] Reference-generation tool is committed.
- [x] Parametrized pytest marathon entry point is committed.
- [x] The C++/V1 `log_only` test ID is registered.
- [ ] `collect_results()` and pytest enforce canary, recovery, final-health, KV-growth, and worker-liveness gates in full mode.
- [ ] Full mode fails setup when required canary prompts or reference data are absent.
- [ ] `wait_until_done()` detects an unexpected worker exit outside the configured respawn flow.
- [ ] Top-level KV-manager/transceiver selectors are cross-checked against both nested worker configurations.
- [ ] `client_cancel_rate`, `output_length`, and maintained concurrency are wired to the full-mode load generator.
- [ ] Hard-zero patterns and the `Cannot cancel request` policy are aligned across runtime, YAMLs, and README.
- [ ] Python/V2 is parametrized and registered when its runtime contract is ready.
- [ ] Each enabled full-mode marathon has a successful full-duration qualification.
- [x] The stress-test README documents modes, invocation, and failure debugging.

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
