<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVBug 6396413: DeepSeek V3.2 Helix Admission-Control Investigation

- **Status:** Admission deprioritized for the hang (warm-up A≈B); pivot to MoE/DSA model-forward follow-ups; root cause not confirmed
- **Created:** 2026-07-06
- **Updated:** 2026-07-08 with Arm-A/Arm-B B300 warm-up comparison and early pivot off admission
- **Component:** PyTorch backend, disaggregated serving, Helix context parallelism
- **Execution hardware:** One exclusive eight-GPU B300 node; original incident hardware was B200
- **Related change:** [PR #15356](https://github.com/NVIDIA/TensorRT-LLM/pull/15356) (admission hypothesis; now deprioritized for hang)
- **Incident job:** [LLM/main/L0_MergeRequest_PR #45198](http://tensorrt-llm.tensorrt-llm-ci-report.sc2-paas.nvidia.com/?job=LLM%2Fmain%2FL0_MergeRequest_PR&build=45198)

> **Agent execution entry point:** Follow [`B300_AGENT_RUNBOOK.md`](B300_AGENT_RUNBOOK.md) from top to bottom. It
> contains the pinned-image build, exact test node, required diagnostic changes, B300 preflight, run order, artifact
> contract, and comparison command. The remainder of this README is the investigation rationale and B200 incident/CI
> background; do not substitute its older snippets for the B300 runbook.

## Executive summary

This plan determines whether the disaggregated-generation transfer admission controller merged by PR #15356:

1. materially slows the exact DeepSeek V3.2 Helix test;
2. contributes to its completed-but-inaccurate results or model-forward hangs; or
3. protects the workload from transfer-buffer over-admission.

The primary experiment is a controlled two-arm ablation on one diagnostic commit:

| Arm | Admission behavior | UCX transfer buffer |
| --- | --- | ---: |
| A | Current/default admission behavior | 8192 tokens |
| B | Bypass only `_apply_disagg_transfer_admission` | 8192 tokens |

Both arms must use the same source SHA, container image digest, physical node, GPU allocation, model files, caches, and
test configuration. Only a diagnostic environment switch may differ.

Start with five measured runs per arm in an interleaved order. Five runs are screening evidence, not closure evidence.
Extend to at least ten valid runs per arm when outcomes differ, either arm fails, or completed-run medians differ by 5%
or more. Prefer 20 runs per arm when estimating an intermittent failure rate.

Do **not** use `max_tokens_in_buffer=0` as arm B. That changes C++ transfer-buffer behavior in addition to disabling the
Python admission gate, so it does not isolate the hypothesis.

## Source-of-truth branches

This document lives on the `docs-and-plans` branch, but that branch intentionally diverged from `main` before PR
#15356. Its checked-out runtime source must not be used for this experiment.

Create the diagnostic experiment branch from a freshly fetched `upstream/main`:

```bash
git fetch upstream main
git switch --detach upstream/main
git switch -c nvbug-6396413-admission-ab
```

At the time this plan was written, the verified upstream snapshot was:

```text
7c8dde830bac813e23605d47a1d27c92d5437a92
```

Always record the actual base SHA used by the experiment. Pinned links in this document describe the verified snapshot;
they are not instructions to silently use an old revision.

## Exact test under investigation

```text
tests/integration/defs/accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]
```

The parameter ID says `fifo`; the test internally selects Helix FIFO protocol version 2. FIFO version 2 is a Helix
communication protocol and is unrelated to KV cache manager V2.

The test currently resolves to:

| Setting | Value |
| --- | --- |
| Runtime backend | PyTorch |
| Model | `DeepSeek-V3.2-Exp-FP4-v2` |
| Evaluation | GSM8K |
| KV cache manager | V1, through `DSACacheManager` / `KVCacheManager` |
| KV cache dtype | FP8 |
| KV transceiver | C++ `BindKvCacheTransceiver` |
| Transfer backend | UCX |
| KV transfer | Asynchronous |
| Disaggregated schedule | Context-first |
| Transfer-buffer capacity | 8192 tokens |
| Tokens per KV block | 32 |
| Admission budget | 256 blocks |
| Client concurrency | 128 workers |
| Context topology | PP1, TP4, CP1 |
| Generation topology | PP1, TP1, CP4, EP4 |
| Helix communication | FIFO, internally `fifo_version=2` |
| Generation CUDA graph | Padding enabled |
| CUDA graph batch sizes | 1, 2, 4, 8, 16, 32, 64 |
| Chunked prefill | Disabled |
| Overlap scheduler | Disabled |
| Block and partial reuse | Disabled |

The context and generation servers consume four GPUs each. The source configuration is in
[`test_disaggregated_serving.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/7c8dde830bac813e23605d47a1d27c92d5437a92/tests/integration/defs/accuracy/test_disaggregated_serving.py#L1675-L1839).

Before each experiment, verify that current `main` still resolves to these settings. Record configuration drift rather
than silently treating a changed test as equivalent to the incident test.

## Current waiver and CI assignment

The exact test is currently waived by NVBug 6396413 in
[`tests/integration/test_lists/waives.txt`](https://github.com/NVIDIA/TensorRT-LLM/blob/7c8dde830bac813e23605d47a1d27c92d5437a92/tests/integration/test_lists/waives.txt#L2).

- A direct pytest invocation does not apply `waives.txt` unless `--waives-file` is supplied.
- Jenkins supplies the waiver file, so a diagnostic CI branch must remove only the exact NVBug 6396413 line.
- Keep the sibling NVBug 6396415 waiver unless that sibling test is intentionally added as a topology control.

The test is in the eight-GPU B200, pre-merge, PyTorch/MPI pool. Recent CI assigned it to
`DGX_B200-8_GPUs-PyTorch-3`, but the concrete `-{1,2,3,4}` shard is load-split and is not a stable property of the test.

There is currently no eight-GPU B300 functional stage, and this exact node is absent from the four-GPU B300 test list.
The B300 experiment must therefore use direct pytest as documented in [`B300_AGENT_RUNBOOK.md`](B300_AGENT_RUNBOOK.md).
It is an out-of-matrix mechanism test and does not replace final B200 CI validation.

Run the repository mapping helper on the pinned experiment SHA:

```bash
python scripts/test_to_stage_mapping.py --tests \
  "accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]"
```

As of the verified snapshot, this helper prints no result because its Groovy parser does not accept a trailing Boolean
in the current B200 stage definition. If it returns no stage, confirm the current shard from the generated CI test list,
the Test Details Dashboard, or the latest CI report. Do not permanently hardcode `PyTorch-3`.

## What is already known

### Failure signatures

The recent exact-test census, after excluding `/root/.triton/cache` `FileNotFoundError` records, contained three distinct
failure classes:

| Failure class | Count | Behavior |
| --- | ---: | --- |
| Completed accuracy failure | 20 | Requests completed, but GSM8K accuracy was below threshold |
| PyExecutor model-forward hang | 27 | Hang detector found ranks stalled after transfer/startup progress |
| Service-registration timeout | 1 | Generation registered; context did not publish its service |

The model-forward hangs and completed accuracy failures were distributed across unrelated PRs and hosts. The results do
not support a bad-node explanation.

Build 45198 progressed past transfer/startup and entered model forward. Its sampled stacks were in DSA projection. That
does not match the transfer-status stall observed in an intermediate PR #15356 revision.

### Historical duration evidence

Successful runs from the same logical B200 stage showed no sustained end-to-end slowdown after the final admission
controller and rank-consistency fix:

| Cohort | Runs | Mean | Median | Range |
| --- | ---: | ---: | ---: | ---: |
| Before admission controller | 8 | 538.602 s | 526.667 s | 512.159-595.904 s |
| After controller and rank fix | 3 | 529.648 s | 526.296 s | 525.622-537.026 s |

The post-admission median differed by -0.371 seconds (-0.07%). These observations used different commits and hosts, so
they rule out an obvious sustained regression but do not replace a controlled same-SHA ablation.

An intermediate admission revision, `ddcf9a7`, stalled the exact test in build 44759. The subsequent
[`d636503`](https://github.com/NVIDIA/TensorRT-LLM/commit/d636503484ad87d9d17270ed110ef0e6dcf136b8)
rank-consistency fix was followed by passes in builds 44792, 44871, and final-head build 45002. Treat the intermediate
defect as fixed history, not as evidence that the merged controller necessarily causes the current failure.

## Relationship to PR #15356

PR #15356 originally targeted bounded polling and burst admission exposed by Qwen Helix cases. It did not add or modify
the DeepSeek V3.2 test under NVBug 6396413.

DeepSeek nevertheless exercises the final generic controller because the gate is shared PyExecutor logic. It applies
when all of the following are present:

1. a KV cache transceiver;
2. a positive `max_tokens_in_buffer`; and
3. generation-init candidates.

This test supplies an 8192-token buffer and 32-token blocks, so the controller has a 256-block FCFS budget under a
128-client burst. The admission code has no model, KV-manager-version, or transceiver-runtime exclusion.

Only `_revert_deferred_disagg_gen_init_alloc` is V2-specific; it returns immediately for this V1 test. The V2 allocation
rollback therefore cannot explain this test's accuracy failures or hangs.

Treat PR #15356 as an A/B hypothesis, not as an established culprit.

## Other recent PRs in the regression window

The reported good commit `c25c23f71786` predates all four PRs below, while the reported first bad commit
`798989ab7913` contains all four. Membership in that 14-commit window establishes correlation only; it does not identify
the culprit.

| PR | Merge commit | Direct overlap with this test | Initial relevance |
| --- | --- | --- | --- |
| [#15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409) | `aaffa2f9f` | Large changes to shared DSA sparse attention and `modules/attention.py`; observed hangs sampled DSA projection | High alternative hypothesis for forward hangs or accuracy |
| [#15356](https://github.com/NVIDIA/TensorRT-LLM/pull/15356) | `b6eacd1f7` | Shared PyExecutor disaggregated admission and progress path | Direct admission hypothesis, but not yet causal |
| [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) | `6f7c57c6c` | Further shared DSA backend and `pyexecutor/model_engine.py` changes, despite DSv4-focused intent | Medium-to-high alternative hypothesis |
| [#15626](https://github.com/NVIDIA/TensorRT-LLM/pull/15626) | `b02b6b464` | Shared autotuner and custom-op wrappers; no Helix or admission edit | Indirect timing/tactic hypothesis |

Prioritize #15409 and #15414 if the admission A/B is null because their shared DSA changes align more closely with the
model-forward and completed-accuracy signatures. Keep #15626 in the bisect set because autotuner choices can affect
kernel execution or CUDA-graph warm-up, but do not call it a transfer-path change.

## Hypotheses

### H0: Admission is unrelated

Both arms have similar completion time, accuracy, and failure incidence. The failure is elsewhere in Helix, CUDA graph
replay, distributed state, the runtime image, or model kernels.

### H1: Admission causes measurable backpressure overhead

Arm A is consistently slower, and measured request deferral or bounded-poll wait time accounts for a plausible portion
of the difference.

### H2: Admission contributes to a correctness or liveness failure

Arm A has more accuracy failures or model-forward hangs than arm B. Telemetry connects those outcomes to excessive
deferral, blocked progress polling, or rank-inconsistent admission state.

### H3: Admission is protective

Arm B has more hangs, accuracy failures, or transfer pressure. In this outcome, do not remove admission; tune or harden
it if necessary.

### H4: Both arms expose the same independent problem

Both arms fail similarly. Proceed to the secondary Helix/CUDA-graph/runtime matrix after completing the primary
ablation.

## Experimental controls

The following are mandatory:

- one pinned `upstream/main` base SHA;
- one diagnostic commit containing both the toggle and telemetry;
- one immutable container image digest;
- one physical B300 node and one uninterrupted eight-GPU allocation per experiment block;
- one model snapshot and `LLM_MODELS_ROOT`;
- one CUDA-visible GPU order;
- one test node ID and timeout;
- an 8192-token UCX transfer buffer in both arms;
- identical Triton, autotuner, model, and compilation caches;
- fresh pytest and serving processes for every repetition;
- an interleaved, predeclared arm order;
- identical logging volume in both arms when comparing performance.

Separate experiment blocks from different nodes or allocations in the analysis. Do not silently pool them.

The primary A/B removes two coupled behaviors in arm B:

1. FCFS candidate deferral; and
2. the `is_blocked_by_active_transfers` signal that triggers bounded generation-transfer progress polling.

If the arms differ, use telemetry and a follow-up ablation to separate selection/backpressure from progress polling. Do
not attribute a difference solely to selector CPU overhead.

## Phase 1: Prepare the diagnostic revision

### 1. Create a clean branch from current main

Run from a clean TensorRT-LLM checkout, not from `docs-and-plans`:

```bash
git status --short
git fetch upstream main
git switch --detach upstream/main
git switch -c nvbug-6396413-admission-ab

export BASE_SHA="$(git rev-parse HEAD)"
printf 'BASE_SHA=%s\n' "${BASE_SHA}"
```

Read `CODING_GUIDELINES.md` before modifying Python. Update the NVIDIA copyright year on every modified source file.

### 2. Verify the exact test

```bash
export TEST_ID='accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]'
export TEST_NODE="tests/integration/defs/${TEST_ID}"

python -m pytest --collect-only -q "${TEST_NODE}"

rg -n \
  'TestDeepSeekV32Exp|test_auto_dtype_with_helix|fifo_version|max_tokens_in_buffer|max_workers' \
  tests/integration/defs/accuracy/test_disaggregated_serving.py

rg -n '6396413' tests/integration/test_lists/waives.txt
```

Confirm that exactly one parameterized node is collected and that direct pytest does not mark it skipped. Stop if the
test configuration no longer matches the configuration table above.

### 3. Locate the current admission boundary

```bash
rg -n \
  '_apply_disagg_transfer_admission|DisaggTransferAdmissionController|limited_by_budget|blocked_by_active' \
  tensorrt_llm/_torch/pyexecutor/py_executor.py \
  tests/unittest/_torch/executor/test_py_executor.py
```

At the verified snapshot, the narrow boundary is `_apply_disagg_transfer_admission`. Run `controller.select(...)` in
both arms to obtain a counterfactual result. Put the diagnostic arm branch after that call and before V2 rollback and
the method return.

### 4. Add one test-only arm switch

Use a parent-harness arm selector:

```text
NVBUG6396413_ARM
```

Accept only `A` or `B`, defaulting to `A`. In the exact test, translate that selector into generation-only environment
variables supplied through the existing `gen_extra_env` argument to `launch_disaggregated_llm`:

```text
TRTLLM_DIAG_BYPASS_DISAGG_TRANSFER_ADMISSION
TRTLLM_DIAG_DISAGG_TRANSFER_TELEMETRY
```

For example, construct the extra generation environment immediately before the existing launcher call:

```python
diagnostic_arm = os.getenv("NVBUG6396413_ARM", "A")
if diagnostic_arm not in ("A", "B"):
    raise ValueError(f"Invalid NVBUG6396413_ARM: {diagnostic_arm}")

diagnostic_gen_env = {
    "TRTLLM_DIAG_BYPASS_DISAGG_TRANSFER_ADMISSION":
        "1" if diagnostic_arm == "B" else "0",
    "TRTLLM_DIAG_DISAGG_TRANSFER_TELEMETRY": "1",
}

launch_disaggregated_llm(
    # Preserve every existing argument.
    gen_extra_env=diagnostic_gen_env,
)
```

Merge with any existing `gen_extra_env` instead of replacing it. The parent selector is harmless if inherited by
context/router processes; the product diagnostic variables must be present only in generation subprocesses. Do not
add an `arm` pytest parameter or rename the existing parameterized node: that would change CI collection and the
waiver identity.

Required semantics:

| Value | Behavior |
| --- | --- |
| Unset or `0` | Unmodified current admission path |
| `1` | Return all otherwise eligible candidates and `False` for blocked progress |

Conceptually:

```python
admission_result = controller.select(self.active_requests, candidates)
if os.getenv("TRTLLM_DIAG_BYPASS_DISAGG_TRANSFER_ADMISSION") == "1":
    effective_admitted = candidates
    wait_for_progress = False
else:
    effective_admitted = admission_result.admitted_requests
    wait_for_progress = admission_result.is_blocked_by_active_transfers()

self._revert_deferred_disagg_gen_init_alloc(candidates, effective_admitted)
return effective_admitted, wait_for_progress
```

The real diagnostic patch must also calculate the default controller result in arm B for telemetry, without enforcing
that result. This gives arm B `would_admit` and `would_defer` values while preserving its bypass behavior.

Do not bypass:

- construction of the C++ UCX transceiver;
- the 8192-token transfer buffer;
- request scheduling or KV allocation;
- asynchronous send or receive;
- ordinary non-admission transfer-status polling;
- transfer completion, error, or timeout handling.

Log exactly one startup configuration record per participating generation rank, using a single-line form that the
harness can parse:

```text
NVBUG6396413_CONFIG rank=<rank> arm=<A-or-B> bypass=<0-or-1> max_tokens_in_buffer=8192 tokens_per_block=32 budget_blocks=256
```

Abort a run if any rank reports the wrong arm, buffer size, tokens per block, or 256-block budget. This TP1/CP4
generation topology should produce four distinct startup-rank records.

### 5. Add structured counters and wait timing

Current code already emits a DEBUG line when requests are deferred. It reports deferred requests, active blocks,
admitted blocks, and budget. It does not report total candidates, cumulative deferral, or bounded-wait duration.

Emit parseable single-line JSON after the marker `NVBUG6396413_JSON ` and record:

| Field | Meaning |
| --- | --- |
| `monotonic_ns` | Monotonic event timestamp |
| `rank` | Generation CP rank; all four ranks independently execute admission for this PP1 topology |
| `iteration` | Executor iteration |
| `arm` | `A` or `B` |
| `candidate_count` / `candidate_blocks` | Full generation-init candidate set |
| `active_transfer_count` / `active_transfer_blocks` | Already in-flight transfers |
| `budget_blocks` | Expected to be 256 |
| `would_admit_count` / `would_defer_count` | Default-controller result in both arms |
| `effective_admit_count` | Result actually enforced by the arm |
| `limited_by_budget` | Whether the default controller reached its budget |
| `blocked_by_active` | Whether active transfers exhausted the budget |
| `candidate_ids_sha256` | SHA-256 of the ordered candidate request-ID sequence |
| `would_admit_ids_sha256` | SHA-256 of the default controller's ordered result |
| `effective_admit_ids_sha256` | SHA-256 of the ordered result enforced by the selected arm |
| `decision_us` | Admission-decision CPU time |

Measure blocked-poll time around `_check_disagg_gen_cache_transfer_status(1)` using `time.monotonic()`. At DEBUG, emit
a `WAIT_BEGIN` record before the call and `WAIT_END` after it so a verbose run in which the call never returns remains
diagnosable.

This exact generation topology is PP1/TP1/CP4, so all four CP ranks independently run the normal executor loop and
admission selection. The PP schedule-broadcast path does not apply. Emit candidate, shadow-admitted, and
effective-admitted SHA-256 values on every generation rank, then compare them offline by iteration. Do not add runtime
communication to calculate a mismatch counter. Never use Python's process-randomized `hash()` for cross-process
comparison.

Accumulate and periodically/finally emit:

- admission decisions and budget-limited decisions;
- total and unique deferred requests;
- maximum active-transfer blocks;
- blocked-poll count and total time;
- admission-decision CPU time;
- maximum effective request deferral time; and
- per-rank decision fingerprints for offline consistency analysis.

Do not add synchronization or collectives for telemetry. Observe existing state only. Keep per-decision and per-wait
records at DEBUG. Avoid INFO logging for every iteration; high-volume logging can change the timing under investigation.
Use the same bounded aggregate cadence and logging configuration in both arms and emit a final aggregate at INFO.
Treatment-dependent event counts need not be identical.

`total_request_deferral_ms` is a sum of per-request wait intervals. Requests can overlap, so this value is request-ms,
not elapsed wall time, and must never be subtracted from end-to-end duration. Report maximum/critical-path request
deferral and measured blocked-poll wall time separately when explaining a slowdown.

Deferral durations describe the effective arm only. Arm B immediately admits requests that the shadow controller would
have deferred, so it can report `would_defer_count` at each decision but cannot reconstruct a hypothetical longitudinal
"would-have-waited" duration. Report arm B effective deferral and admission-specific blocked-poll time as zero or N/A.

### 6. Add focused unit tests

Run the existing admission tests first:

```bash
python -m pytest -q tests/unittest/_torch/executor/test_py_executor.py \
  -k 'DisaggTransferAdmission or DisaggTransferIdleProgress or DisaggTransferAdmissionPP'
```

Add cases proving:

- switch unset and switch `0` preserve current behavior;
- switch `1` returns all candidates and `wait_for_disagg_gen_transfer_progress=False`;
- arm B still calculates the default `would_admit` / `would_defer` telemetry;
- arm B leaves the configured buffer and 256-block budget unchanged;
- V1 does not execute V2 rollback;
- candidate ordering is unchanged;
- wait timing emits BEGIN and END records and accumulates duration;
- all generation child processes inherit the selected arm while context/router processes do not receive the product
  diagnostic variables; and
- decision fingerprints match on all CP ranks in a synthetic schedule.

Commit the diagnostic revision with DCO sign-off:

```bash
git add \
  tensorrt_llm/_torch/pyexecutor/py_executor.py \
  tests/integration/defs/accuracy/test_disaggregated_serving.py \
  tests/unittest/_torch/executor/test_py_executor.py
git commit -s -m '[NVBUG 6396413][test] instrument disagg admission experiment'

export EXPERIMENT_SHA="$(git rev-parse HEAD)"
printf 'BASE_SHA=%s\nEXPERIMENT_SHA=%s\n' "${BASE_SHA}" "${EXPERIMENT_SHA}"
git status --short
```

Both arms must use `EXPERIMENT_SHA`. Do not edit source between arms.

## Phase 2: Prepare one B300 environment

This section is retained as background for the original plan. The executable B300 build and preflight are maintained in
[`B300_AGENT_RUNBOOK.md`](B300_AGENT_RUNBOOK.md); an agent should use that runbook instead of the snippets below.

### 1. Reserve one persistent node

Use one exclusive B300 allocation with all eight GPUs for the entire experiment block. Separate bot/Jenkins reruns
cannot guarantee the same physical host and may reuse successful test results, so they are unsuitable for the primary
same-node A/B.

Use the team's approved B300 allocation and container launcher. A site-specific Slurm shape is:

```bash
salloc \
  --nodes=1 \
  --gres=gpu:8 \
  --exclusive \
  --time='REPLACE_WITH_LONG_ENOUGH_DURATION' \
  --partition='REPLACE_WITH_B300_PARTITION' \
  --account='REPLACE_WITH_ACCOUNT'

srun \
  --pty \
  --container-image='REPLACE_WITH_PINNED_CI_IMAGE_AND_DIGEST' \
  --container-mounts='REPLACE_WITH_REPOSITORY_MODELS_AND_RESULTS' \
  bash
```

Replace placeholders with the approved internal launcher. Do not copy unverified partition, account, or image names
from an older CI log.

### 2. Enter the pinned checkout

Inside the allocation/container:

```bash
cd /path/to/TensorRT-LLM

export BASE_SHA='REPLACE_WITH_RECORDED_40_CHARACTER_BASE_SHA'
export EXPERIMENT_SHA='REPLACE_WITH_RECORDED_40_CHARACTER_DIAGNOSTIC_SHA'
export CI_IMAGE_TAG='REPLACE_WITH_PINNED_IMAGE_TAG'
export CI_IMAGE_DIGEST='REPLACE_WITH_PINNED_SHA256_IMAGE_DIGEST'
export TEST_ID='accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]'
export TEST_NODE="tests/integration/defs/${TEST_ID}"

: "${BASE_SHA:?BASE_SHA is not set}"
: "${EXPERIMENT_SHA:?EXPERIMENT_SHA is not set}"
: "${CI_IMAGE_TAG:?CI_IMAGE_TAG is not set}"
: "${CI_IMAGE_DIGEST:?CI_IMAGE_DIGEST is not set}"
[[ "${BASE_SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Replace BASE_SHA'; exit 1; }
[[ "${EXPERIMENT_SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Replace EXPERIMENT_SHA'; exit 1; }
[[ "${CI_IMAGE_TAG}" != REPLACE_WITH_* ]] || { echo 'Replace CI_IMAGE_TAG'; exit 1; }
[[ "${CI_IMAGE_DIGEST}" == *sha256:* ]] || { echo 'Replace CI_IMAGE_DIGEST'; exit 1; }

test "$(git rev-parse HEAD)" = "${EXPERIMENT_SHA}" || {
  echo 'Checkout does not match EXPERIMENT_SHA'; exit 1;
}
test -z "$(git status --porcelain)" || {
  echo 'Checkout is dirty'; exit 1;
}

export LLM_MODELS_ROOT=/path/to/model/root
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXPECTED_HOST="$(hostname)"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export RESULTS_PARENT=/path/to/persistent/results
export RESULT_ROOT="${RESULTS_PARENT}/nvbug-6396413/$(date -u +%Y%m%dT%H%M%SZ)-${EXPERIMENT_SHA:0:12}"

mkdir -p "${RESULT_ROOT}"
```

`RESULTS_PARENT` must survive container or allocation teardown.

Verify that Python imports the diagnostic checkout rather than a stale installed wheel:

```bash
command -v trtllm-serve | tee "${RESULT_ROOT}/trtllm-serve.path.txt"

python - <<'PY'
import tensorrt_llm

print(tensorrt_llm.__file__)
PY
```

Stop if the printed path is not the intended diagnostic source or installation. `PYTHONPATH` also makes an externally
launched `trtllm-serve` entrypoint resolve the checkout. Retain the per-rank `NVBUG6396413_CONFIG` startup records as the
final proof that serving workers imported the instrumented revision.

### 3. Capture immutable metadata

```bash
{
  printf 'BASE_SHA=%s\n' "${BASE_SHA}"
  printf 'EXPERIMENT_SHA=%s\n' "${EXPERIMENT_SHA}"
  printf 'HOST=%s\n' "${EXPECTED_HOST}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'LLM_MODELS_ROOT=%s\n' "${LLM_MODELS_ROOT}"
  printf 'CI_IMAGE_TAG=%s\n' "${CI_IMAGE_TAG:-UNKNOWN}"
  printf 'CI_IMAGE_DIGEST=%s\n' "${CI_IMAGE_DIGEST:-UNKNOWN}"
  printf 'UTC_START=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  python -VV
  git show --no-patch --format=fuller HEAD
} > "${RESULT_ROOT}/environment.txt"

nvidia-smi -L > "${RESULT_ROOT}/nvidia-smi-L.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi-q.before.txt"
python -m pip freeze > "${RESULT_ROOT}/pip-freeze.txt"
git diff "${BASE_SHA}..${EXPERIMENT_SHA}" > "${RESULT_ROOT}/diagnostic.patch"
```

Do not dump the full environment into an artifact that may be attached to the NVBug. Record an allowlist instead:

```bash
for name in \
  BASE_SHA EXPERIMENT_SHA CI_IMAGE_TAG CI_IMAGE_DIGEST EXPECTED_HOST \
  CUDA_VISIBLE_DEVICES LLM_MODELS_ROOT PYTHONPATH TEST_ID TEST_NODE; do
  printf '%s=%s\n' "${name}" "${!name:-UNSET}"
done > "${RESULT_ROOT}/environment-variables.allowlist.txt"
```

Populate `CI_IMAGE_TAG` and `CI_IMAGE_DIGEST` before running. A mutable tag without its digest is insufficient.

Verify:

```bash
test "$(hostname)" = "${EXPECTED_HOST}" || { echo 'Host changed'; exit 1; }
test "$(nvidia-smi -L | wc -l)" -eq 8 || { echo 'Expected eight GPUs'; exit 1; }
test "$(git rev-parse HEAD)" = "${EXPERIMENT_SHA}" || { echo 'SHA changed'; exit 1; }
test -d "${LLM_MODELS_ROOT}/DeepSeek-V3.2-Exp-FP4-v2" || {
  echo 'Model snapshot is missing'; exit 1;
}
```

### 4. Stabilize caches without masking cache failures

The `/root/.triton/cache` `FileNotFoundError` is outside the product hypothesis. Ensure the expected directory exists
before warm-up and record its metadata:

```bash
mkdir -p /root/.triton/cache
stat /root/.triton/cache > "${RESULT_ROOT}/triton-cache.before.txt"
```

Use the same Triton, autotuner, model, and compilation caches for both arms. Do not clear caches between measured runs.
If a cache setup failure still occurs, classify it as excluded infrastructure/setup and rerun the same arm.

### 5. Confirm exact collection

```bash
python -m pytest --collect-only -q "${TEST_NODE}" \
  | tee "${RESULT_ROOT}/collect-only.log"
```

Stop if the exact node is not collected or is unexpectedly skipped.

## Phase 3: Run the controlled A/B

### 1. Configure logging

```bash
export TLLM_LOG_LEVEL=info
unset TLLM_LOG_LEVEL_BY_MODULE
export PYTHONUNBUFFERED=1
```

Primary timing runs use only the added startup, fixed-cadence aggregate, and final aggregate records. They must have
the same logging configuration and bounded aggregate cadence in both arms. Do not enable the existing per-decision
`_torch` DEBUG stream for the primary comparison; its I/O can perturb scheduler timing. `WAIT_BEGIN` and `WAIT_END`
remain DEBUG-only and are collected in the separate verbose passes below.

Do not add `CUDA_LAUNCH_BLOCKING`, profilers, forced synchronization, or synchronous KV transfer to the primary A/B.
Those are separate follow-up experiments.

### 2. Warm both arms

Run one uncounted warm-up for each arm in an interleaved order. Preserve the logs, but exclude warm-ups from outcome and
duration statistics. A warm-up must still use a fresh pytest process and fresh serving subprocesses.

```bash
set -uo pipefail

for arm in A B; do
  export NVBUG6396413_ARM="${arm}"
  if [[ "${arm}" == "A" ]]; then expected_bypass=0; else expected_bypass=1; fi
  warmup_dir="${RESULT_ROOT}/warmup-arm-${arm}"
  mkdir -p "${warmup_dir}"

  if pgrep -a -f '[t]rtllm-serve|[t]est_disaggregated_serving|[p]ytest' \
      > "${warmup_dir}/processes.before.txt"; then
    echo "Stale test process before warm-up arm ${arm}; stop."
    break
  fi

  timeout --signal=TERM --kill-after=120s 75m \
    python -m pytest -sv -ra --tb=long \
      --junitxml="${warmup_dir}/junit.xml" \
      "${TEST_NODE}" \
    2>&1 | tee "${warmup_dir}/pytest.log"
  printf 'exit_code=%s\n' "${PIPESTATUS[0]}" > "${warmup_dir}/status.txt"

  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
    --format=csv,noheader > "${warmup_dir}/gpu-processes.after.txt"
  pgrep -a -f '[t]rtllm-serve|[t]est_disaggregated_serving|[p]ytest' \
    > "${warmup_dir}/processes.after.txt" || true
  if [[ -s "${warmup_dir}/gpu-processes.after.txt" || \
        -s "${warmup_dir}/processes.after.txt" ]]; then
    echo "Residual test process after warm-up arm ${arm}; stop before measuring."
    break
  fi

  config_count="$(rg -c \
    "NVBUG6396413_CONFIG.*arm=${arm}.*bypass=${expected_bypass}.*max_tokens_in_buffer=8192.*tokens_per_block=32.*budget_blocks=256" \
    "${warmup_dir}/pytest.log" || true)"
  [[ "${config_count}" -eq 4 ]] || {
    echo "Expected four valid arm ${arm} startup markers; found ${config_count}."; break;
  }
done
```

Run this under Bash with `set -uo pipefail`, as for the measured harness. Do not begin measured runs unless both warm-ups
used the expected arm and completed cleanup.

### 3. Run five measured repetitions per arm

Run under Bash because the harness below uses `PIPESTATUS`:

```bash
set -uo pipefail

ORDER=(A B B A B A A B A B)

for index in "${!ORDER[@]}"; do
  arm="${ORDER[$index]}"
  ordinal=$((index + 1))
  run_id="$(printf '%02d_arm-%s' "${ordinal}" "${arm}")"
  run_dir="${RESULT_ROOT}/${run_id}"

  mkdir -p "${run_dir}"

  export NVBUG6396413_ARM="${arm}"
  if [[ "${arm}" == "A" ]]; then expected_bypass=0; else expected_bypass=1; fi

  test "$(hostname)" = "${EXPECTED_HOST}" || {
    echo 'Host changed; aborting experiment.'; exit 1;
  }
  test "$(git rev-parse HEAD)" = "${EXPERIMENT_SHA}" || {
    echo 'Source SHA changed; aborting experiment.'; exit 1;
  }

  if pgrep -a -f '[t]rtllm-serve|[t]est_disaggregated_serving|[p]ytest' \
      > "${run_dir}/processes.before.txt"; then
    echo "Stale test process before ${run_id}; aborting experiment."
    exit 1
  fi

  start_epoch="$(date +%s)"

  {
    printf 'run_id=%s\n' "${run_id}"
    printf 'arm=%s\n' "${arm}"
    printf 'harness_arm=%s\n' "${NVBUG6396413_ARM}"
    printf 'host=%s\n' "$(hostname)"
    printf 'sha=%s\n' "$(git rev-parse HEAD)"
    printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${run_dir}/metadata.before.txt"

  timeout \
    --signal=TERM \
    --kill-after=120s \
    75m \
    python -m pytest \
      -sv \
      -ra \
      --tb=long \
      --junitxml="${run_dir}/junit.xml" \
      "${TEST_NODE}" \
    2>&1 | tee "${run_dir}/pytest.log"

  test_exit="${PIPESTATUS[0]}"
  end_epoch="$(date +%s)"

  {
    printf 'exit_code=%s\n' "${test_exit}"
    printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'duration_s=%s\n' "$((end_epoch - start_epoch))"
  } > "${run_dir}/metadata.after.txt"

  rg -n \
    'NVBUG6396413_|GSM8K|accuracy|Hang detected|Test terminated unexpectedly|FileNotFoundError|FAILED|PASSED|timeout' \
    "${run_dir}/pytest.log" \
    > "${run_dir}/key-events.txt" || true

  nvidia-smi \
    --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
    --format=csv,noheader \
    > "${run_dir}/gpu-processes.after.txt"

  pgrep -a -f '[t]rtllm-serve|[t]est_disaggregated_serving|[p]ytest' \
    > "${run_dir}/processes.after.txt" || true

  if [[ -s "${run_dir}/gpu-processes.after.txt" || \
        -s "${run_dir}/processes.after.txt" ]]; then
    echo "Residual test process after ${run_id}; stop before continuing."
    break
  fi

  config_count="$(rg -c \
    "NVBUG6396413_CONFIG.*arm=${arm}.*bypass=${expected_bypass}.*max_tokens_in_buffer=8192.*tokens_per_block=32.*budget_blocks=256" \
    "${run_dir}/pytest.log" || true)"
  if [[ "${config_count}" -ne 4 ]]; then
    echo "Expected four valid startup markers in ${run_id}; found ${config_count}."
    break
  fi
done
```

The test fixture launches fresh router/context/generation subprocesses for every invocation and attempts graceful
termination before killing leaked children. Do not automatically kill arbitrary system processes. If cleanup is
incomplete, preserve the diagnostics and stop the block rather than contaminating the next arm.

The 75-minute wrapper allows the test's normal 60-minute timeout to fire first. If frequent hangs make the screening
block impractical, preserve one complete standard-timeout example per arm, then predeclare and apply the same shorter
diagnostic timeout to both arms. Do not compare mixed timeout policies.

### 4. Verify the arm actually propagated

Each run must contain configuration records from every generation rank showing the expected translated state:

```text
arm=<expected A or B>
bypass=<expected 0 for A or 1 for B>
max_tokens_in_buffer=8192
tokens_per_block=32
budget_blocks=256
```

Invalidate and rerun a sample if any generation rank is missing or disagrees.

### 5. Run one verbose diagnostic pass per arm

After the measured block, run one additional, uncounted repetition per arm with the existing `_torch` DEBUG messages
enabled:

```bash
export TLLM_LOG_LEVEL=warning
export TLLM_LOG_LEVEL_BY_MODULE='debug:_torch'
```

Use the same arm order, timeout, node, image, SHA, and capture procedure. Label these runs `verbose_diagnostic` and do
not include their durations in the primary performance comparison. They satisfy the need to inspect individual
admitted/deferred decisions without confounding the timing samples. Restore the primary logging configuration before
any additional measured run.

## Phase 4: Classify and analyze results

### Valid-run requirements

A sample is valid only if:

- host, SHA, image, GPU UUIDs, and model snapshot match the experiment manifest;
- every generation rank reports the expected arm;
- the exact test started;
- buffer size and budget are 8192 tokens and 256 blocks;
- no preceding run left serving or GPU processes;
- there was no Triton-cache, allocation, node, Slurm, storage, or unrelated setup failure.

Excluded setup/infra failures do not count toward an arm's denominator. Rerun the same arm.

### Failure classification

Classify every valid sample as exactly one of:

| Outcome | Definition |
| --- | --- |
| `pass` | Test completed and met its accuracy requirement |
| `accuracy_failure` | Requests completed, but the accuracy assertion failed |
| `model_forward_hang` | Hang detector or worker stacks identify a forward stall |
| `admission_or_transfer_stall` | Unmatched admission `WAIT_BEGIN` plus no later forward progress and corroborating transfer-side stack/log evidence |
| `registration_timeout` | Context or generation worker failed to register |
| `crash_or_oom` | Product process crashed or ran out of memory |
| `outer_timeout_unknown` | Outer timeout lacks enough evidence for a narrower class |
| `cleanup_leak` | Test finished, but owned serving/GPU processes remained |

Do not classify a timeout from pytest's final subprocess-cleanup stack alone. Use last server progress, admission wait
markers, and hang-detector stacks. Do not treat hangs as very slow completed runs.

### Results schema

Create `results.csv` with:

```csv
run_id,arm,host,base_sha,experiment_sha,image_digest,exit_code,outcome,duration_s,accuracy_score,first_context_response_s,first_generation_forward_s,admission_decisions,limited_decisions,unique_deferred_requests,total_request_deferral_ms,max_request_deferral_ms,blocked_poll_count,blocked_poll_total_ms,decision_total_ms,decision_rank_consistent,notes
```

Report separately for each arm:

- pass/failure counts by class;
- median, mean, minimum, and maximum completed-test duration;
- GSM8K score distribution;
- time to first context response and first generation forward;
- admission decisions and budget-limited decisions;
- total request-ms, maximum/critical-path deferral, and blocked-poll wall time;
- maximum active-transfer blocks;
- rank-consistency violations;
- hang stack family, when applicable.

Total pytest wall time includes setup, model load, evaluation, and cleanup. Also measure evaluation/progress intervals so a
startup fluctuation is not attributed to admission.

### Initial and confirmatory thresholds

Five valid runs per arm are exploratory. Extend to at least ten per arm if:

- either arm has any product failure;
- outcomes differ between arms;
- completed-run medians differ by 5% or more;
- admission wait could plausibly explain a duration difference; or
- telemetry shows any cross-rank inconsistency.

A material admission slowdown requires all of:

1. arm A's completed-run median is at least 5% slower;
2. the direction persists across time blocks;
3. at least ten valid runs exist per arm; and
4. measured deferral or bounded-poll time plausibly explains the difference.

### Decision matrix

| Arm A: admission on | Arm B: bypass | Interpretation |
| --- | --- | --- |
| Healthy | Healthy, similar time | Admission is unlikely to cause the current failure or material slowdown |
| Stable but slower | Faster and equally stable | Measure whether backpressure overhead warrants budget/poll tuning |
| Fails or hangs | Repeatedly healthy | Admission or its rank/timing interaction is implicated |
| Healthy | Fails or hangs | Admission is protective; bypass causes over-admission/pressure |
| Both fail similarly | Both fail similarly | Root cause likely lies outside admission |
| Accuracy differs only | Liveness similar | Investigate timing-sensitive KV/slot state, not performance alone |
| Rank divergence only in A | Rank-consistent in B | Strong evidence of admission rank-consensus failure |
| High run-to-run variance, no arm split | High variance | Increase repetitions or isolate another factor |

A faster run is not a fix if accuracy or hang incidence worsens.

## Phase 5: Follow-up ablations

> **2026-07-08 decision:** B300 warm-up A≈B both reproduced the same FP4 MoE model-forward hang with a verified
> admission bypass on Arm B. Measured interleaved admission A/B is **skipped** for hang attribution. Proceed to the
> MoE/DSA / regression-window follow-ups below. Revisit measured admission A/B only if a later failure mode looks
> admission-specific (for example unmatched `WAIT_BEGIN` / transfer stall) or if accuracy-only failures need a separate
> admission screen.

Only start these after the admission A/B is complete **or** after an explicit early-exit decision as above. Change one
factor at a time:

1. TP1/CP4, FIFO, padded CUDA graph: primary control.
2. TP1/CP4, FIFO, no CUDA graph.
3. TP1/CP4, NCCL, padded CUDA graph.
4. TP2/CP2 sibling topology.
5. Client concurrency 1, 16, 64, and 128.
6. Asynchronous transfer versus `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1`.
7. Current runtime image versus the incident-era image on the same source, if available.

If arm A and B differ, first split the controller into two follow-ups:

- apply FCFS selection but suppress the admission-specific blocking progress wait;
- preserve the progress signal while varying only the selection/budget behavior.

Current-main A/B answers whether the current controller affects the current symptom. It cannot by itself prove or
disprove historical causality for an older source/image combination. If historical attribution is required, repeat the
smallest possible toggle on the first classifiable failing descendant with its original image.

### Regression-window PR isolation

If both admission arms behave alike, test the PR boundaries that align with the observed signatures. Use the same
physical experiment node (B300 for screening, or B200 for historical attribution), compatible image digest, model
snapshot, test node, timeout, and result schema. Do not
compare CI runs from arbitrary hosts as if they were a controlled bisect.

| Candidate | Parent/control | Merge/treatment |
| --- | --- | --- |
| PR #15409, shared DSA attention | `434dc3345d03` | `aaffa2f9fef3` |
| PR #15356, disaggregated admission | `85665f5fd331` | `b6eacd1f725d` |
| PR #15414, DSA/model-engine follow-up | `5b6c3ed91510` | `6f7c57c6c297` |
| PR #15626, autotuner follow-up | `70c5e430c854` | `b02b6b464218` |

Prepare immutable detached worktrees from a fully fetched upstream repository:

```bash
git fetch upstream main

for sha in \
  434dc3345d033944076b98eae570ea4bf8bc3337 \
  aaffa2f9fef3025e0f698d978385a73460344e0b \
  85665f5fd331d0154a78172954846d843085e83f \
  b6eacd1f725d0fbeb888e09ff2585e5ab23b0856 \
  5b6c3ed9151020f8bea50f7358177b44d3be82e7 \
  6f7c57c6c297808a193f74d28f8a3fbf06927efb \
  70c5e430c8544638f95d0c8dd20cb3cf862a0331 \
  b02b6b464218b6801209575b677cc96f5e576b64; do
  git worktree add --detach "/path/to/nvbug-6396413-worktrees/${sha}" "${sha}"
done
```

First reproduce the reported endpoints with at least five valid runs each at `c25c23f71786` and `798989ab7913`. If the
good/bad separation does not reproduce in the pinned environment, stop: a source-only PR attribution is not supported.
If it does reproduce, run at least five valid, interleaved repetitions per SHA for one parent/merge pair at a time.
Prioritize #15409, then #15414, then #15626 after the dedicated #15356 A/B. Warm each SHA once; exclude warm-ups. Keep
per-SHA build artifacts separate, and do not silently change the image or dependencies to make one boundary build.

A candidate is implicated only when its parent/merge difference repeats, the failure signature matches, and at least
one focused revert or minimized test confirms the mechanism. A single pass/fail transition in a flaky five-run screen
is insufficient; extend a separating pair to at least ten valid runs per SHA.

## Phase 6: CI verification and waiver removal

The controlled same-node experiment should run manually in one allocation. Use CI afterward to validate the proposed
fix/default behavior in the supported pipeline.

### Diagnostic CI repetitions

If CI repetitions are needed:

1. remove only the NVBug 6396413 waiver on the diagnostic branch;
2. determine the current concrete shard from the generated test list or latest report;
3. request the mapped pre-merge stage;
4. disable successful-test reuse on every repetition; and
5. verify that each new build collected and executed the exact test.

Example, replacing `<CURRENT_STAGE>` with the current generated shard:

```text
/bot run --disable-reuse-test --detailed-log --stage-list "<CURRENT_STAGE>"
```

If the mapping helper and generated test list cannot establish a concrete shard reliably, do not guess from a stale
report. Run the full pre-merge pipeline instead:

```text
/bot run --disable-reuse-test --detailed-log
```

Without `--disable-reuse-test`, a prior pass on the same commit may be appended to the stage waiver list and skipped,
invalidating a repetition. `--debug` is an interactive SSH mode; it does not enable TRT-LLM module logging.

Separate CI repetitions cannot satisfy the same-node requirement. Record the host of every run and analyze each host as
a separate block.

### Final fix/waiver-removal validation

Before changing NVBug 6396413 to `Dev Open Verify-To-Close`:

1. land the product or test fix, if required;
2. remove temporary diagnostic behavior unless intentionally retained and reviewed;
3. complete at least ten clean repetitions of the final default configuration;
4. remove the exact NVBug 6396413 waiver;
5. do not skip the CI pipeline for the waiver-removal change;
6. run the current mapped pre-merge stage without test reuse;
7. verify that Jenkins launched and that the exact test passed rather than being skipped;
8. run any corresponding post-merge stage with `/bot run --extra-stage "<STAGE>"` when applicable;
9. attach the run table, raw logs, telemetry summaries, SHA, image digest, host/allocation ID, and CI report to the NVBug;
10. search `waives.txt` and relevant branches to confirm no waiver linked to NVBug 6396413 remains.

## Closure criteria

### Admission not causal in current main

All of the following should hold:

- at least ten valid runs per arm;
- no excess accuracy failures or hangs in arm A;
- no cross-rank admission inconsistency;
- no material arm-A duration regression;
- maximum/critical-path admission deferral and blocked-poll wall time are bounded and compatible with observed runtime;
- bypass does not uniquely remove the historical symptom.

This conclusion is scoped to the pinned current-main source and image.

### Admission implicated

Require:

- a repeatable arm-dependent outcome;
- identical SHA, image, host, GPUs, model, and test configuration;
- at least ten valid samples per arm;
- telemetry connecting the outcome to admission state;
- no cache or infrastructure explanation; and
- a minimized follow-up reproduction or focused regression test.

### NVBug closure

Require:

- the final default path is stable and accurate;
- the exact test is unwaived;
- the mapped CI stage executes and passes without result reuse;
- related post-merge coverage passes when applicable; and
- all evidence is attached to NVBug 6396413.

## B300 experiment progress (2026-07-08)

Warm-up A/B completed on the same host/SHA/image wiring. Measured interleaved admission samples were **not** started;
the owner decision is to treat warm-up A≈B as sufficient to **deprioritize admission for hang attribution** and pivot to
MoE/DSA model-forward follow-ups. Scope of that decision: the reproduced **model-forward hang** class on this pinned
experiment, not every historical accuracy-failure mode in the NVBug census.

### Experiment identity

| Item | Value |
| --- | --- |
| Result root | `/home/scratch.chienchunh_coreai/dev/nvbug-6396413-results/20260707T025616Z-861f44e21660` |
| Base SHA (container-matching) | `5ec0c84ad1684c7d08e17a49cf0d53a061fd85cd` |
| Diagnostic / experiment SHA | `9f82ceda7dee872e8f6f3f39ac758cd6a6c12de2` |
| Image tag | `urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.02-py3-x86_64-ubuntu24.04-trt10.15.1.29-skip-tritondevel-202606051544-14972` |
| Physical host (Arm-A / Arm-B) | `umb-b300-dp-217` (runbook expected host was `umb-b300-023`; allocation resumed on a different B300 after eviction) |
| Model | `DeepSeek-V3.2-Exp-FP4-v2` under `LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models` |
| Exact test | `TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]` |
| Preflight | Exact-node `--collect-only` collected 1 test; focused admission unit tests 25/25 after diagnostic amend |

### Setup notes that unblocked the warm-up

Earlier Arm-A attempts failed before product execution for infrastructure reasons, not admission:

1. Home quota (`/home/chienchunh`, 5 GB) was full, so DeepGEMM / CUDA JIT caches under `~/.deep_gemm` and `~/.nv` produced `CUDA_ERROR_INVALID_IMAGE`. Redirected `DG_JIT_CACHE_DIR` and `CUDA_CACHE_PATH` to scratch.
2. NFS model load of the ~386 GB checkpoint exceeded default server/test timeouts. Raised outer/pytest/server/test timeouts for warm-up (`13000s` / `12000s` / `9000s` / `12000s`).
3. Wheel install used a scratch site-dir + build venv Python because home-quota `pip --user` was impossible.

These are excluded setup/infra issues under the runbook contract. Arm-A attempt-4 is the first end-to-end product run.

### Arm-A warm-up (`warmup-A-attempt-4`, admission on)

| Field | Observation |
| --- | --- |
| Arm / bypass | `arm=A`, `bypass=false` on all four generation CP ranks |
| Config markers | 4/4 `NVBUG6396413_JSON` config events: `max_tokens_in_buffer=8192`, `tokens_per_block=32`, `budget_blocks=256`, `experiment_sha=9f82ceda7dee...` |
| Pipeline progress | Context + generation + disagg router launched; GSM8K client reached **810/1319** completions (~61%) before hang |
| Outcome class | **`model_forward_hang`** (not transfer-status stall, not registration timeout, not completed accuracy failure) |
| Hang detector | All four generation ranks: `Hang detected after 300 seconds` in `PyExecutor` at `2026-07-08 15:14:41` |
| Hang stack family | `_executor_loop` → `_forward_step` → `model_engine.forward` → DeepSeek V3 `forward_MoE` → `fused_moe` → `run_fp4_block_scale_moe` → `fp4_block_scale_moe_runner` |
| Last progress before hang | Many `POST /v1/completions` 200 OK responses; eval still advancing at ~6.9 it/s |

Last per-rank aggregate telemetry before the hang (cadence summary at `admission_decisions=600`; ranks agree):

| Metric | Rank 0–3 (same within rounding) |
| --- | ---: |
| `would_defer_count` | 1039 |
| `effective_deferred_requests` | 119 |
| `effective_deferral_request_ms` (request-ms, overlapping) | ~267192 |
| `max_effective_deferral_ms` | ~5998 |
| `blocked_poll_count` | 2 |
| `blocked_poll_ms` | ~5962 |
| `decision_ms` (CPU) | ~7.3–7.5 |
| `max_active_transfer_blocks` | 254 |
| Cross-rank `decision_digest` | Matched across ranks at each summary cadence |

### Arm-B warm-up (`warmup-B-attempt-2`, admission bypassed)

| Field | Observation |
| --- | --- |
| Arm / bypass | `arm=B`, `bypass=true` on all four generation CP ranks; gen env confirmed `TRTLLM_DIAG_BYPASS_DISAGG_TRANSFER_ADMISSION=1` |
| Config markers | 4/4 valid config events (8192 / 32 / 256 / experiment SHA) |
| Pipeline progress | GSM8K client reached **821/1319** completions (~62%) before hang |
| Outcome class | **`model_forward_hang`** — same class as Arm A |
| Hang detector | All four generation ranks: `Hang detected after 300 seconds` at `2026-07-08 18:11:11` (~22.8 min after test start) |
| Hang stack family | Same as Arm A: DeepSeek V3 `forward_MoE` → `fp4_block_scale_moe_runner` (not DSA projection; not admission wait) |
| Cleanup note | Pytest/serve still tearing down after the hang (etcd expire noise / `threads can only be started once`); product hang outcome is already decisive |

Last per-rank aggregate telemetry before the hang (cadence summary at `admission_decisions=600`; ranks agree):

| Metric | Rank 0–3 (same within rounding) |
| --- | ---: |
| `would_defer_count` (shadow only) | 120 |
| `effective_deferred_requests` | **0** |
| `effective_deferral_request_ms` | **0** |
| `max_effective_deferral_ms` | **0** |
| `blocked_poll_count` | **0** |
| `blocked_poll_ms` | **0** |
| `decision_ms` (CPU) | ~8.0–8.4 |
| `max_active_transfer_blocks` | **3450** |
| Cross-rank agreement | Summaries matched across ranks |

Bypass validation: Arm B still computed the shadow controller (`would_defer_count=120`) but did **not** enforce deferral
or admission-specific blocked-progress polling. `max_active_transfer_blocks=3450` (≫ 256 budget) is direct evidence of
over-admission relative to Arm A’s capped ~254 active blocks.

### Warm-up A vs B comparison

| Dimension | Arm A | Arm B | Implication |
| --- | --- | --- | --- |
| Admission enforced? | Yes | No (verified) | Toggle worked |
| GSM8K progress at hang | ~61% (810/1319) | ~62% (821/1319) | Same failure timing window |
| Outcome | model-forward hang | model-forward hang | No arm split |
| Hang locus | FP4 MoE runner | FP4 MoE runner | Outside admission wait path |
| Effective deferral / blocked-poll | 119 / ~6 s | 0 / 0 | Bypass removed admission backpressure |
| Active transfer blocks | 254 | 3450 | Bypass increased transfer pressure; hang still occurred |

Decision matrix mapping: **both arms fail similarly** → H0/H4 for this hang class (admission unrelated / same independent
problem). Not H2 (admission causes hang) and not H3 (admission protective against this hang).

### Decision: deprioritize admission; pivot to MoE/DSA

Owner decision on 2026-07-08:

1. Treat warm-up A≈B as enough to **deprioritize PR #15356 admission** as the root cause of the reproduced hang.
2. **Skip** the measured 5×2 / 10×2 interleaved admission block for hang attribution.
3. **Pivot** to model-forward follow-ups aligned with the observed stacks and the regression-window alternatives:
   - Primary fingerprint from these warm-ups: **FP4 block-scale MoE** (`fp4_block_scale_moe_runner`).
   - Keep **DSA** ([PR #15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409), [PR #15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414)) in the next screen because historical CI hang samples were in DSA projection and those PRs touch shared attention / model-engine code in the same window — but do **not** claim the B300 warm-up stacks were DSA.
   - Prefer same-node parent/merge isolation from Phase 5 (`#15409` then `#15414`, then `#15626` if needed), plus focused MoE ablations (CUDA graph on/off, concurrency) once a separating boundary appears.
4. Re-open measured admission A/B only if a later signature looks admission-specific, or if completed-accuracy failures need a separate admission screen.

## Results template

### Environment

| Item | Value |
| --- | --- |
| Base `main` SHA | `5ec0c84ad1684c7d08e17a49cf0d53a061fd85cd` |
| Diagnostic SHA | `9f82ceda7dee872e8f6f3f39ac758cd6a6c12de2` |
| Container image tag | `pytorch-26.02-py3-...-202606051544-14972` (see `IMAGE_REF` in result `active.env`) |
| Container image digest | Not pinned as an immutable `sha256:` in the warm-up manifest; pin before any future measured block |
| DGX host | `umb-b300-dp-217` (warm-up block; expected runbook host was `umb-b300-023`) |
| Allocation/job ID | Initial `3017028`; later resumed under a new Slurm allocation after eviction |
| Driver | `595.58.03` |
| CUDA | From pinned 26.02 image |
| PyTorch | From pinned 26.02 image / build venv |
| Model snapshot | `/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V3.2-Exp-FP4-v2` |
| Test start/end UTC | Arm A ~2026-07-08 14:50:51 → 15:14:50; Arm B ~17:48:25 → hang 18:11:11 (local log timestamps) |

### Outcome summary

| Arm | Valid | Pass | Accuracy failure | Forward hang | Transfer stall | Other | Median completed duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: admission on | 0 measured (1 warm-up) | 0 | 0 | 1 warm-up | 0 | 0 | N/A (hang) |
| B: admission bypassed | 0 measured (1 warm-up) | 0 | 0 | 1 warm-up | 0 | 0 | N/A (hang) |

### Admission summary

| Arm | Limited decisions | Unique deferred | Total deferral | Max wait | Blocked-poll time | Rank mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A (warm-up last summary) | budget pressure (`max_active_transfer_blocks=254`) | 119 effective deferred | ~267192 request-ms | ~5998 ms | ~5962 ms (2 polls) | none observed |
| B effective; shadow counts only | shadow `would_defer=120`; effective unlimited (`max_active_transfer_blocks=3450`) | **0** effective | **0** | **0** | **0** | none observed |

### Conclusion

Warm-up A/B on B300 shows the waived test’s **PyExecutor model-forward hang** in **FP4 block-scale MoE** with and
without admission enforcement. Admission bypass was validated (zero effective deferral / blocked-poll; much higher
active transfer blocks) and **did not remove the hang**. Admission is therefore **deprioritized** as the hang root
cause on this pinned experiment (H0/H4 for the hang). Next work is MoE/DSA / regression-window follow-up, not measured
admission repetitions. This does not by itself close the NVBug or attribute a specific MoE/DSA PR.

## References

- [Eight-GPU B300 agent runbook](B300_AGENT_RUNBOOK.md)
- [B300 run-artifact extractor](extract_results.py)
- [B300 A/B result comparator](compare_results.py)
- [NVBug 6396413](https://nvbugs/6396413)
- [PR #15356](https://github.com/NVIDIA/TensorRT-LLM/pull/15356)
- [PR #15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409)
- [PR #15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414)
- [PR #15626](https://github.com/NVIDIA/TensorRT-LLM/pull/15626)
- [PR #15356 merged commit](https://github.com/NVIDIA/TensorRT-LLM/commit/b6eacd1f725d0fbeb888e09ff2585e5ab23b0856)
- [Rank-consistency fix](https://github.com/NVIDIA/TensorRT-LLM/commit/d636503484ad87d9d17270ed110ef0e6dcf136b8)
- [Incident CI report](http://tensorrt-llm.tensorrt-llm-ci-report.sc2-paas.nvidia.com/?job=LLM%2Fmain%2FL0_MergeRequest_PR&build=45198)
- [Exact test at verified current-main snapshot](https://github.com/NVIDIA/TensorRT-LLM/blob/7c8dde830bac813e23605d47a1d27c92d5437a92/tests/integration/defs/accuracy/test_disaggregated_serving.py#L1675-L1839)
- [Admission controller at verified snapshot](https://github.com/NVIDIA/TensorRT-LLM/blob/7c8dde830bac813e23605d47a1d27c92d5437a92/tensorrt_llm/_torch/pyexecutor/py_executor.py#L163-L262)
- [Admission call and progress path at verified snapshot](https://github.com/NVIDIA/TensorRT-LLM/blob/7c8dde830bac813e23605d47a1d27c92d5437a92/tensorrt_llm/_torch/pyexecutor/py_executor.py#L2823-L2919)
- [Current waiver at verified snapshot](https://github.com/NVIDIA/TensorRT-LLM/blob/7c8dde830bac813e23605d47a1d27c92d5437a92/tests/integration/test_lists/waives.txt#L2)
- [TRT-LLM Test Case Detail dashboard](https://gpuwa.nvidia.com/os-dashboards/app/dashboards?security_tenant=TRT-LLM-Infra#/view/15f841f0-7f4f-11ef-979e-b7e76100ed73)
