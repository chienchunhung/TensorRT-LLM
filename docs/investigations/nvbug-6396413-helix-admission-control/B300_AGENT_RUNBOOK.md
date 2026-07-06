<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVBug 6396413: 8-GPU B300 Agent Runbook

## 0. Agent execution contract

Execute this runbook in order. Do not start measured runs until every checkpoint through **BUILD-OK** and **TEST-OK**
passes. Stop rather than guessing when a required SHA, image digest, model, dataset, GPU, or telemetry field is missing.
All shell snippets require Bash.

Use one persistent host Bash for sections 3-5. The launcher creates a detached, named container: either open one
persistent Bash in it for sections 5-13, or execute each numbered section in a fresh `docker exec` Bash. In the latter
case, keep all fences in a numbered section in the same shell and source the generated
`/results/nvbug-6396413-active.env` at the start of every later section, as shown.

The objective is to determine whether the current disaggregated-transfer admission controller changes progress,
accuracy, or liveness for the exact DeepSeek V3.2 Helix test on one eight-GPU B300 host.

This is a same-host B300 mechanism experiment. The incident and mapped CI stage use B200. The exact test is not in the
current B300 CI matrix, which has only four-GPU functional stages. Therefore:

- run this experiment directly with pytest, not a guessed `DGX_B300-8_GPUs-*` stage;
- do not compare B300 durations directly with historical B200 durations;
- a null B300 A/B does not exclude a B200-only kernel or timing interaction; and
- confirm a positive result on B200 before using it to close NVBug 6396413.

The source paths and configuration below were last verified against upstream `main` commit
`7c8dde830bac813e23605d47a1d27c92d5437a92` on 2026-07-06. The runner must still fetch and record its actual
`BASE_SHA`; stop and update this runbook if the named test or admission boundary has drifted.

Minimum deliverable:

- one uncounted warm-up per arm;
- five valid measured runs per arm in order `A B B A B A A B A B`;
- one uncounted verbose diagnostic run per arm;
- a complete manifest, per-run artifacts, `results.csv`, and generated comparison report; and
- a written conclusion that stays within the B300 scope above.

Budget one uninterrupted allocation of at least 24 hours. Two warm-ups, ten measured runs, and two verbose runs can
consume up to 17.5 hours at the 75-minute outer limit, before checkout and build time. Reserve roughly 35 hours up front
if a same-allocation ten-per-arm confirmation is likely.

## 1. Immutable test and arm definitions

Run from `tests/integration/defs` with this exact node ID:

```bash
export TEST_ID='accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]'
```

Do not add an arm parameter to the pytest test or rename the node. Do not pass `--waives-file`; direct pytest does not
apply the NVBug waiver unless that option is supplied.

The test configuration must remain:

| Setting | Required value |
| --- | --- |
| Backend | PyTorch |
| Model files | `${LLM_MODELS_ROOT}/DeepSeek-V3.2-Exp-FP4-v2` |
| Evaluation | Full 1,319-example GSM8K dataset |
| KV cache manager | V1 |
| KV transceiver | C++ `BindKvCacheTransceiver` |
| Transfer backend | UCX |
| Transfer mode | Asynchronous |
| Context workers | PP1/TP4/CP1 on GPUs 0-3 |
| Generation workers | PP1/TP1/CP4/EP4 on GPUs 4-7 |
| Helix | FIFO protocol version 2 |
| CUDA graph | Padding enabled; batch sizes 1, 2, 4, 8, 16, 32, 64 |
| KV cache | FP8, 32 tokens per block, reuse disabled |
| Transfer buffer | 8,192 tokens, hence a 256-block admission budget |
| Scheduling | Overlap scheduler and chunked prefill disabled |
| Client workers | 128 |

Only the diagnostic arm and telemetry may differ:

| Arm | Effective behavior | Buffer |
| --- | --- | ---: |
| A | Current controller result and current blocked-progress signal | 8,192 tokens |
| B | Admit every otherwise eligible candidate and suppress only the admission-specific blocked-progress signal | 8,192 tokens |

Never use `max_tokens_in_buffer=0` as arm B. Never force synchronous transfer, change UCX settings, disable CUDA graphs,
reduce GSM8K, or change client concurrency in the primary A/B.

## 2. Required inputs

The runner must resolve and record these values before changing source:

| Variable | Requirement |
| --- | --- |
| `SOURCE_REPO` | Clean TensorRT-LLM clone with `upstream` pointing to NVIDIA/TensorRT-LLM |
| `BASE_SHA` | Freshly fetched `upstream/main` commit used for the diagnostic change |
| `EXPERIMENT_CHECKOUT` | New standalone clone dedicated to this experiment |
| `EXPERIMENT_SHA` | Signed diagnostic commit containing both arms and telemetry |
| `RUNBOOK_DIR` | Host path to this investigation directory, including `compare_results.py` |
| `LLM_MODELS_ROOT_HOST` | Host path containing the model and GSM8K dataset |
| `RESULTS_HOST` | Persistent, writable host directory for all artifacts |
| `CACHE_HOST` | Persistent, writable host directory shared by both arms |
| `IMAGE` | x86 `LLM_DOCKER_IMAGE` read from `BASE_SHA` |
| `IMAGE_REF` | Immutable pulled image digest, not only a mutable tag |
| `EXPECTED_HOST` | Exclusive eight-GPU B300 hostname |
| `ALLOCATION_ID` | Scheduler/allocation identifier |

Do not put credentials, tokens, or a complete environment dump in the result bundle.

## 3. Create an isolated source checkout

Run on the host:

```bash
set -Eeuo pipefail
test -n "${BASH_VERSION:-}" || { echo 'Run this runbook with Bash.' >&2; exit 2; }

: "${SOURCE_REPO:?set SOURCE_REPO}"
: "${RUNBOOK_DIR:?set RUNBOOK_DIR}"
: "${LLM_MODELS_ROOT_HOST:?set LLM_MODELS_ROOT_HOST}"
: "${RESULTS_HOST:?set RESULTS_HOST}"
: "${CACHE_HOST:?set CACHE_HOST}"
: "${ALLOCATION_ID:?set ALLOCATION_ID}"

export TEST_ID='accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype_with_helix[fifo-cudagraph:with_padding-pp1tp1cp4]'

command -v git >/dev/null
git lfs version
command -v pre-commit >/dev/null

SOURCE_REPO="$(realpath "${SOURCE_REPO}")"
RUNBOOK_DIR="$(realpath "${RUNBOOK_DIR}")"
LLM_MODELS_ROOT_HOST="$(realpath "${LLM_MODELS_ROOT_HOST}")"
mkdir -p "${RESULTS_HOST}" "${CACHE_HOST}"
RESULTS_HOST="$(realpath "${RESULTS_HOST}")"
CACHE_HOST="$(realpath "${CACHE_HOST}")"

cd "${SOURCE_REPO}"
test -z "$(git status --porcelain)" || {
  echo 'SOURCE_REPO is dirty; use a clean clone or preserve the existing work first.' >&2
  exit 2
}

git fetch upstream main
export BASE_SHA="$(git rev-parse upstream/main)"
export UPSTREAM_URL="$(git remote get-url upstream)"
export EXPERIMENT_CHECKOUT="${EXPERIMENT_CHECKOUT:-${SOURCE_REPO%/*}/TensorRT-LLM-nvbug-6396413-${BASE_SHA:0:12}}"
EXPERIMENT_CHECKOUT="$(realpath -m "${EXPERIMENT_CHECKOUT}")"
export EXPECTED_HOST="$(hostname)"

test -f "${RUNBOOK_DIR}/compare_results.py" && test -f "${RUNBOOK_DIR}/extract_results.py" || {
  echo "RUNBOOK_DIR does not contain both result tools: ${RUNBOOK_DIR}" >&2
  exit 2
}
test ! -e "${EXPERIMENT_CHECKOUT}" || {
  echo "EXPERIMENT_CHECKOUT already exists: ${EXPERIMENT_CHECKOUT}" >&2
  exit 2
}

git clone --no-checkout "${UPSTREAM_URL}" "${EXPERIMENT_CHECKOUT}"
cd "${EXPERIMENT_CHECKOUT}"
git checkout --detach "${BASE_SHA}"
git switch -c nvbug-6396413-b300-admission-ab
git submodule update --init --recursive
git lfs install --local
git lfs pull
git lfs fsck

export IMAGE="$(sed -n 's/^LLM_DOCKER_IMAGE=//p' jenkins/current_image_tags.properties)"
test -n "${IMAGE}" || { echo 'Unable to resolve LLM_DOCKER_IMAGE.' >&2; exit 2; }

printf 'BASE_SHA=%s\nIMAGE=%s\n' "${BASE_SHA}" "${IMAGE}"
```

Do not run the experiment from the `docs-and-plans` branch. It is documentation history, not the source-under-test.

## 4. Prepare one diagnostic commit

If a reviewed diagnostic commit already exists on top of `BASE_SHA`, fetch and check it out, then audit it against this
section. Otherwise implement exactly the contract below. Do not begin the expensive B300 build from an uncommitted or
partially instrumented tree.

### 4.1 Required source changes

| File | Required change |
| --- | --- |
| `tests/integration/defs/accuracy/test_disaggregated_serving.py` | Read parent selector `NVBUG6396413_ARM`, accept only `A` or `B`, verify `EXPERIMENT_SHA`, and translate both through the existing `gen_extra_env` launcher argument into generation-only product variables. Preserve the existing pytest parameterization and every test configuration value. |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py` | In `_apply_disagg_transfer_admission`, call the current controller in both arms to obtain the shadow result. Arm A enforces it. Arm B enforces the original full candidate list and returns `False` for the admission-specific blocked-progress signal. Pass the effective list to V2 rollback. Do not bypass ordinary transfer polling or any C++ buffer behavior. |
| `tests/unittest/_torch/executor/test_py_executor.py` | Cover default behavior, bypass behavior, shadow selection, effective ordering, wait-signal behavior, and telemetry. |

Use these generation-only product variables:

```text
TRTLLM_DIAG_BYPASS_DISAGG_TRANSFER_ADMISSION=0|1
TRTLLM_DIAG_DISAGG_TRANSFER_TELEMETRY=1
TRTLLM_DIAG_EXPERIMENT_SHA=<40-character diagnostic commit SHA>
```

The parent pytest process may carry `NVBUG6396413_ARM=A|B`. Context and router processes must not receive the product
diagnostic variables.

Conceptual admission flow:

```python
shadow = controller.select(self.active_requests, candidates)
if bypass:
    effective_admitted = candidates
    wait_for_admission_progress = False
else:
    effective_admitted = shadow.admitted_requests
    wait_for_admission_progress = shadow.is_blocked_by_active_transfers()

self._revert_deferred_disagg_gen_init_alloc(candidates, effective_admitted)
return effective_admitted, wait_for_admission_progress
```

### 4.2 Required telemetry

Emit single-line JSON after the marker `NVBUG6396413_JSON `. Use monotonic timestamps. Keep per-decision and per-wait
events at DEBUG; emit startup, bounded-cadence aggregate, and final summaries at INFO.

Required event types:

| Event | Required fields |
| --- | --- |
| `config` | `rank`, `arm`, `bypass` as a JSON boolean, `max_tokens_in_buffer`, `tokens_per_block`, `budget_blocks`, `experiment_sha` |
| `decision` | `rank`, `iteration`, candidate/active/shadow/effective counts and blocks, `limited_by_budget`, `blocked_by_active`, ordered candidate/shadow/effective ID SHA-256 values, `decision_us` |
| `wait_begin` / `wait_end` | `rank`, `iteration`, monotonic timestamp, and elapsed microseconds on `wait_end` |
| `summary` | `rank`, `final` boolean, decisions, limited decisions, effective deferred requests, request-ms, maximum effective deferral, blocked-poll count/time, decision time, maximum active blocks, and a rolling ordered-decision digest |

This exact generation topology is PP1/TP1/CP4. All four CP generation ranks independently schedule and execute
admission control in the normal executor loop. There is no PP schedule-broadcast boundary for this node. Emit decision
fingerprints on every generation rank and compare them offline by iteration. Do not add telemetry collectives or new
synchronization.

Emit exactly one `final=true` summary per generation rank. Final summaries must use the exact keys consumed by
`extract_results.py`:
`rank`, `admission_decisions`, `would_defer_count`, `effective_deferred_requests`,
`effective_deferral_request_ms`, `max_effective_deferral_ms`, `blocked_poll_count`, `blocked_poll_ms`, `decision_ms`,
and `decision_digest`.

Arm B can report the controller's per-decision `would_admit` and `would_defer` values, but it cannot report a
hypothetical multi-iteration deferral duration because those requests are admitted immediately. Its effective admission
deferral and admission-specific blocked-poll summary values must be numeric zero; the comparator enforces this.

### 4.3 Validate and commit the diagnostic revision

```bash
set -Eeuo pipefail
cd "${EXPERIMENT_CHECKOUT}"

pre-commit run --files \
  tensorrt_llm/_torch/pyexecutor/py_executor.py \
  tests/integration/defs/accuracy/test_disaggregated_serving.py \
  tests/unittest/_torch/executor/test_py_executor.py

git diff --check
git add \
  tensorrt_llm/_torch/pyexecutor/py_executor.py \
  tests/integration/defs/accuracy/test_disaggregated_serving.py \
  tests/unittest/_torch/executor/test_py_executor.py
git commit -s -m '[NVBUG 6396413][test] instrument B300 admission experiment'

export EXPERIMENT_SHA="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "${BASE_SHA}" || {
  echo 'Diagnostic commit is not exactly one commit above BASE_SHA.' >&2
  exit 2
}
test -z "$(git status --porcelain --untracked-files=no)" || {
  echo 'Tracked source changed after the diagnostic commit.' >&2
  exit 2
}

git diff "${BASE_SHA}..${EXPERIMENT_SHA}" -- \
  tensorrt_llm/_torch/pyexecutor/py_executor.py \
  tests/integration/defs/accuracy/test_disaggregated_serving.py \
  tests/unittest/_torch/executor/test_py_executor.py
```

Record **PATCH-OK** only after the diff contains no unrelated runtime or test-config changes.

## 5. Start or verify the pinned container

If the agent is already inside the image resolved from `BASE_SHA`, record the platform-provided immutable digest and
ensure the standalone checkout is available at `/code/tensorrt_llm` and this investigation directory is read-only at
`/runbook`, then continue. Otherwise run on the host:

```bash
set -Eeuo pipefail

docker pull "${IMAGE}"
export IMAGE_REF="$(docker image inspect "${IMAGE}" --format '{{index .RepoDigests 0}}')"
[[ "${IMAGE_REF}" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "Image does not have a valid immutable RepoDigest: ${IMAGE_REF}" >&2
  exit 2
}

mkdir -p "${RESULTS_HOST}" "${CACHE_HOST}/huggingface" "${CACHE_HOST}/triton"

export CONTAINER_NAME="nvbug-6396413-${EXPERIMENT_SHA:0:12}"
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container already exists; inspect it rather than replacing it: ${CONTAINER_NAME}" >&2
  exit 2
fi

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --ipc=host \
  --uts=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus=all \
  --tmpfs /tmp:exec \
  --env LLM_MODELS_ROOT=/models \
  --env MODEL_CACHE_DIR=/models \
  --env HF_HOME=/cache/huggingface \
  --env TRITON_CACHE_DIR=/cache/triton \
  --env CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache \
  --env CCACHE_BASEDIR=/code/tensorrt_llm \
  --env CONAN_HOME=/code/tensorrt_llm/cpp/.conan \
  --env BASE_SHA="${BASE_SHA}" \
  --env EXPERIMENT_SHA="${EXPERIMENT_SHA}" \
  --env IMAGE_REF="${IMAGE_REF}" \
  --env EXPECTED_HOST="${EXPECTED_HOST}" \
  --env ALLOCATION_ID="${ALLOCATION_ID}" \
  --env TEST_ID="${TEST_ID}" \
  --volume "${EXPERIMENT_CHECKOUT}:/code/tensorrt_llm:rw" \
  --volume "${RUNBOOK_DIR}:/runbook:ro" \
  --volume "${LLM_MODELS_ROOT_HOST}:/models:ro" \
  --volume "${RESULTS_HOST}:/results:rw" \
  --volume "${CACHE_HOST}:/cache:rw" \
  --workdir /code/tensorrt_llm \
  "${IMAGE_REF}" \
  sleep infinity

printf 'container_name=%s\n' "${CONTAINER_NAME}"
```

The remaining commands run inside the container. For an attached persistent shell, use
`docker exec -it "${CONTAINER_NAME}" bash`. A non-interactive agent can use `docker exec -i "${CONTAINER_NAME}" bash`
and provide one numbered section on standard input; it must not request a TTY. Set `IMAGE_REF`, `BASE_SHA`,
`EXPERIMENT_SHA`, `EXPECTED_HOST`, and `ALLOCATION_ID` again if a site-specific launcher did not propagate them. Keep
the named container until the evidence bundle has been copied and verified.

Before the first Git command in a container running as a different UID:

```bash
git config --global --add safe.directory /code/tensorrt_llm
git -C /code/tensorrt_llm rev-parse --is-inside-work-tree
git -C /code/tensorrt_llm rev-parse HEAD
test -f /runbook/extract_results.py
test -f /runbook/compare_results.py
```

### 5.1 Cheap pre-build gates

Reject the wrong node, missing data, inactive NVLink, or insufficient disk before compiling:

```bash
set -Eeuo pipefail

export TRTLLM_ROOT=/code/tensorrt_llm
export RESULT_ROOT="/results/nvbug-6396413-$(date -u +%Y%m%dT%H%M%SZ)-${EXPERIMENT_SHA:0:12}"
mkdir -p "${RESULT_ROOT}"

test -d /models/DeepSeek-V3.2-Exp-FP4-v2 || { echo 'Model directory is missing.' >&2; exit 2; }
test -d /models/datasets/openai/gsm8k || { echo 'GSM8K dataset is missing.' >&2; exit 2; }

python3 - <<'PY' | tee "${RESULT_ROOT}/hardware-prebuild.txt"
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    print(index, name, capability)
    assert capability == (10, 3), (index, name, capability)
    assert any(token in name.upper() for token in ("B300", "GB110")), (index, name)
PY

repo_free_kb="$(df -Pk "${TRTLLM_ROOT}" | awk 'NR == 2 {print $4}')"
results_free_kb="$(df -Pk "${RESULT_ROOT}" | awk 'NR == 2 {print $4}')"
(( repo_free_kb >= 80 * 1024 * 1024 )) || { echo 'Require at least 80 GiB for build.' >&2; exit 2; }
(( results_free_kb >= 5 * 1024 * 1024 )) || { echo 'Require at least 5 GiB for results.' >&2; exit 2; }

nvidia-smi nvlink -s | tee "${RESULT_ROOT}/nvidia-smi-nvlink.prebuild.txt"
grep -Eq 'Link [0-9]+:[[:space:]]+[0-9.]+[[:space:]]+GB/s' \
  "${RESULT_ROOT}/nvidia-smi-nvlink.prebuild.txt" || {
    echo 'The test launcher would not detect active NVLink.' >&2
    exit 2
  }

for name in \
  TRTLLM_ROOT RESULT_ROOT BASE_SHA EXPERIMENT_SHA IMAGE_REF EXPECTED_HOST ALLOCATION_ID TEST_ID; do
  printf 'export %s=%q\n' "${name}" "${!name}"
done > /results/nvbug-6396413-active.env
```

## 6. Build once for B300 SM103

Use a full wheel build to freeze Python and C++ code from `EXPERIMENT_SHA`. Do not use `TRTLLM_USE_PRECOMPILED`,
`--fast_build`, or `--cuda_architectures=native`. Build explicitly for B300 `103-real`, install that wheel once, and do
not rebuild between arms.

```bash
set -Eeuo pipefail
source /results/nvbug-6396413-active.env

export BUILD_JOBS="${BUILD_JOBS:-16}"

cd "${TRTLLM_ROOT}"
test "$(git rev-parse HEAD)" = "${EXPERIMENT_SHA}" || {
  echo 'Container checkout does not match EXPERIMENT_SHA.' >&2
  exit 2
}
mkdir -p "${RESULT_ROOT}"
test -d /opt/nvidia/nvda_nixl || {
  echo 'Pinned image does not contain /opt/nvidia/nvda_nixl.' >&2
  exit 2
}

python3 -m pip install -r requirements-dev.txt \
  2>&1 | tee "${RESULT_ROOT}/requirements-install.log"

{
  printf 'build_jobs=%s\n' "${BUILD_JOBS}"
  uname -a
  python3 --version
  nvcc --version
  cmake --version
  ninja --version
  c++ --version
} > "${RESULT_ROOT}/build-toolchain.txt"

python3 scripts/build_wheel.py \
  --clean \
  --use_ccache \
  -G Ninja \
  -j "${BUILD_JOBS}" \
  -D 'WARNING_IS_ERROR=ON' \
  --nixl_root /opt/nvidia/nvda_nixl \
  --cuda_architectures '103-real' \
  2>&1 | tee "${RESULT_ROOT}/build.log"

mapfile -t wheels < <(find build -maxdepth 1 -type f -name 'tensorrt_llm-*.whl' -print)
test "${#wheels[@]}" -eq 1 || { echo 'Expected exactly one built wheel.' >&2; exit 2; }
export WHEEL="${wheels[0]}"

sha256sum "${WHEEL}" | tee "${RESULT_ROOT}/wheel.sha256"
export WHEEL_SHA256="$(cut -d' ' -f1 "${RESULT_ROOT}/wheel.sha256")"
python3 -m pip install --force-reinstall --no-deps "${WHEEL}" \
  2>&1 | tee "${RESULT_ROOT}/wheel-install.log"
python3 -m pip show tensorrt-llm | tee "${RESULT_ROOT}/wheel-info.txt"
command -v trtllm-serve | tee "${RESULT_ROOT}/trtllm-serve.path.txt"

printf 'export WHEEL=%q\nexport WHEEL_SHA256=%q\n' "${WHEEL}" "${WHEEL_SHA256}" \
  >> /results/nvbug-6396413-active.env
```

NIXL is not exercised by this UCX test. The NIXL root is retained only to match the repository's full CI wheel build.
If `/opt/nvidia/nvda_nixl` is absent, stop and resolve the matching CI image; do not silently alter the build contract or
substitute a different runtime image.

## 7. B300 and data preflight

Set the test environment. Full accuracy is mandatory:

```bash
set -Eeuo pipefail
source /results/nvbug-6396413-active.env

export LLM_ROOT=/code/tensorrt_llm
export LLM_BACKEND_ROOT=/code/tensorrt_llm/triton_backend
export LLM_MODELS_ROOT=/models
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export COLUMNS=300
export PYTHONUNBUFFERED=1

unset INTEGRATION_TEST
unset TRTLLM_ACCURACY_NO_REFERENCE
unset TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP
unset TLLM_LOG_LEVEL_BY_MODULE

test -d "${LLM_MODELS_ROOT}/DeepSeek-V3.2-Exp-FP4-v2" || {
  echo 'DeepSeek V3.2 model directory is missing.' >&2
  exit 2
}
test -d "${LLM_MODELS_ROOT}/datasets/openai/gsm8k" || {
  echo 'GSM8K dataset directory is missing.' >&2
  exit 2
}

pushd /tmp >/dev/null
python3 - <<'PY' | tee "${RESULT_ROOT}/b300-preflight.txt"
import pathlib

import tensorrt_llm
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    print(index, name, capability)
    assert capability == (10, 3), (index, name, capability)
    assert any(token in name.upper() for token in ("B300", "GB110")), (index, name)

package_path = pathlib.Path(tensorrt_llm.__file__).resolve()
print("tensorrt_llm", package_path)
assert not package_path.is_relative_to(pathlib.Path("/code/tensorrt_llm/tensorrt_llm")), package_path
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
popd >/dev/null

nvidia-smi -L | tee "${RESULT_ROOT}/nvidia-smi-L.txt"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total \
  --format=csv,noheader | LC_ALL=C sort | tee "${RESULT_ROOT}/gpu-inventory.csv"
sha256sum "${RESULT_ROOT}/gpu-inventory.csv" > "${RESULT_ROOT}/gpu-fingerprint.sha256"
nvidia-smi topo -m | tee "${RESULT_ROOT}/nvidia-smi-topo.txt"
nvidia-smi nvlink -s | tee "${RESULT_ROOT}/nvidia-smi-nvlink.txt"
nvidia-smi -q | tee "${RESULT_ROOT}/nvidia-smi.before.txt"
```

Stop if NVLink is not active or if the launcher later reports a different UCX transport. Do not externally override
`UCX_TLS`; the test launcher owns it and must configure both arms identically.

Capture non-secret identities:

```bash
set -Eeuo pipefail
source /results/nvbug-6396413-active.env

cd "${TRTLLM_ROOT}"

{
  printf 'base_sha=%s\n' "${BASE_SHA}"
  printf 'experiment_sha=%s\n' "${EXPERIMENT_SHA}"
  printf 'image_ref=%s\n' "${IMAGE_REF}"
  printf 'host=%s\n' "${EXPECTED_HOST}"
  printf 'allocation_id=%s\n' "${ALLOCATION_ID}"
  printf 'test_id=%s\n' "${TEST_ID}"
  printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RESULT_ROOT}/manifest.txt"

git show --no-patch --format=fuller HEAD > "${RESULT_ROOT}/experiment-commit.txt"
git diff "${BASE_SHA}..${EXPERIMENT_SHA}" > "${RESULT_ROOT}/diagnostic.diff"
python3 -m pip freeze > "${RESULT_ROOT}/pip-freeze.txt"
sha256sum /runbook/extract_results.py /runbook/compare_results.py \
  > "${RESULT_ROOT}/result-tools.sha256"

find "${LLM_MODELS_ROOT}/DeepSeek-V3.2-Exp-FP4-v2" -maxdepth 1 -type f \
  -printf '%f %s %T@\n' | LC_ALL=C sort | sha256sum \
  > "${RESULT_ROOT}/model-manifest.sha256"
find "${LLM_MODELS_ROOT}/datasets/openai/gsm8k" -type f \
  -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum \
  > "${RESULT_ROOT}/gsm8k-manifest.sha256"

export GPU_FINGERPRINT="$(cut -d' ' -f1 "${RESULT_ROOT}/gpu-fingerprint.sha256")"
export MODEL_FINGERPRINT="$(cut -d' ' -f1 "${RESULT_ROOT}/model-manifest.sha256")"
export DATASET_FINGERPRINT="$(cut -d' ' -f1 "${RESULT_ROOT}/gsm8k-manifest.sha256")"
printf 'wheel_sha256=%s\ngpu_fingerprint=%s\nmodel_fingerprint=%s\ndataset_fingerprint=%s\n' \
  "${WHEEL_SHA256}" "${GPU_FINGERPRINT}" "${MODEL_FINGERPRINT}" "${DATASET_FINGERPRINT}" \
  >> "${RESULT_ROOT}/manifest.txt"

for name in \
  TRTLLM_ROOT RESULT_ROOT LLM_ROOT LLM_BACKEND_ROOT LLM_MODELS_ROOT \
  BASE_SHA EXPERIMENT_SHA IMAGE_REF EXPECTED_HOST ALLOCATION_ID TEST_ID \
  WHEEL_SHA256 GPU_FINGERPRINT MODEL_FINGERPRINT DATASET_FINGERPRINT \
  CUDA_VISIBLE_DEVICES NCCL_DEBUG COLUMNS PYTHONUNBUFFERED; do
  printf 'export %s=%q\n' "${name}" "${!name}"
done > /results/nvbug-6396413-active.env
```

The launcher allocates router and worker ports dynamically. Do not reject the node based on fixed ports 8000-8002.

Record **BUILD-OK** only when the wheel install, all eight SM103 checks, model/data checks, and NVLink checks pass.

## 8. Unit and collection checkpoint

```bash
set -Eeuo pipefail
source /results/nvbug-6396413-active.env

cd "${TRTLLM_ROOT}"
python3 -m pytest -q tests/unittest/_torch/executor/test_py_executor.py \
  -k 'DisaggTransferAdmission or DisaggTransferIdleProgress or DisaggTransferAdmissionPP' \
  --junitxml="${RESULT_ROOT}/unit-admission.xml" \
  2>&1 | tee "${RESULT_ROOT}/unit-admission.log"

cd "${TRTLLM_ROOT}/tests/integration/defs"
python3 -m pytest --collect-only -q "${TEST_ID}" \
  2>&1 | tee "${RESULT_ROOT}/collect-only.log"

test "$(grep -F -c "${TEST_ID}" "${RESULT_ROOT}/collect-only.log")" -eq 1 || {
  echo 'The exact test did not collect exactly once.' >&2
  exit 2
}
```

The class is `skip_pre_blackwell` and the method requires eight GPUs; B300 SM103 satisfies both. It has no
post-Blackwell-Ultra skip. Record **TEST-OK** only after the focused unit tests pass and exactly one test node collects.

## 9. Run warm-ups and measured samples

### 9.1 Runtime and validity rules

Use the same wheel, SHA, host, eight GPU UUIDs, image digest, model/data manifests, CUDA-visible order, and caches for
every run. Start a fresh pytest process for every sample. Do not clear caches after warm-up.

Before the first run, capture the allocation baseline:

```bash
source /results/nvbug-6396413-active.env
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
  --format=csv,noheader | LC_ALL=C sort > "${RESULT_ROOT}/compute-baseline.csv"
pgrep -a -f '[t]rtllm-serve|[t]est_disaggregated_serving|[p]ytest' \
  > "${RESULT_ROOT}/process-baseline.txt" || true

test ! -s "${RESULT_ROOT}/compute-baseline.csv" || {
  echo 'Exclusive B300 allocation already has GPU compute processes.' >&2
  exit 2
}
test ! -s "${RESULT_ROOT}/process-baseline.txt" || {
  echo 'Exclusive B300 allocation already has matching test/serve processes.' >&2
  exit 2
}
```

If either baseline is non-empty, stop and identify every process. Never kill an unknown process merely to make the
baseline empty.

After every run, the compute-process and matching-process snapshots must equal these baselines. A leak is not merely an
outcome label: preserve diagnostics and stop before contaminating the next sample.

The normal limits are:

- post-hoc GSM8K evaluation-duration threshold: 1,500 seconds after evaluation returns;
- service-readiness limit: 2,100 seconds;
- pytest timeout: 3,600 seconds; and
- outer process limit: 4,500 seconds, with 120 seconds between TERM and KILL.

### 9.2 One-run command

For each run, replace `RUN_ID` and `ARM`. `ARM` must be exactly `A` or `B`.

```bash
set -Eeuo pipefail
source /results/nvbug-6396413-active.env

: "${RUN_ID:?set RUN_ID}"
: "${ARM:?set ARM}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${BASE_SHA:?set BASE_SHA}"
: "${EXPERIMENT_SHA:?set EXPERIMENT_SHA}"
: "${IMAGE_REF:?set IMAGE_REF}"
: "${EXPECTED_HOST:?set EXPECTED_HOST}"
: "${ALLOCATION_ID:?set ALLOCATION_ID}"
: "${WHEEL_SHA256:?set WHEEL_SHA256}"
: "${GPU_FINGERPRINT:?set GPU_FINGERPRINT}"
: "${MODEL_FINGERPRINT:?set MODEL_FINGERPRINT}"
: "${DATASET_FINGERPRINT:?set DATASET_FINGERPRINT}"
: "${TEST_ID:?set TEST_ID}"
[[ "${ARM}" == A || "${ARM}" == B ]] || { echo 'ARM must be A or B.' >&2; exit 2; }

export NVBUG6396413_ARM="${ARM}"
export RUN_MODE="${RUN_MODE:-primary}"
case "${RUN_MODE}" in
  primary)
    export TLLM_LOG_LEVEL=info
    unset TLLM_LOG_LEVEL_BY_MODULE
    ;;
  verbose)
    export TLLM_LOG_LEVEL=warning
    export TLLM_LOG_LEVEL_BY_MODULE='debug:_torch'
    ;;
  *)
    echo "RUN_MODE must be primary or verbose, got ${RUN_MODE}." >&2
    exit 2
    ;;
esac

RUN_DIR="${RESULT_ROOT}/${RUN_ID}"
test ! -e "${RUN_DIR}" || { echo "Refusing to overwrite ${RUN_DIR}." >&2; exit 2; }
mkdir -p "${RUN_DIR}"

test "$(hostname)" = "${EXPECTED_HOST}" || { echo 'Host changed.' >&2; exit 2; }
test "$(git -C "${TRTLLM_ROOT}" rev-parse HEAD)" = "${EXPERIMENT_SHA}" || {
  echo 'Experiment SHA changed.' >&2
  exit 2
}
git -C "${TRTLLM_ROOT}" diff HEAD --exit-code || {
  echo 'Tracked source changed after the wheel build.' >&2
  exit 2
}

cd "${TRTLLM_ROOT}/tests/integration/defs"

start_ns="$(date +%s%N)"
set +e
timeout --signal=TERM --kill-after=120s 4500s \
  python3 -m pytest \
    -sv \
    -ra \
    --tb=long \
    --timeout=3600 \
    --timeout-method=thread \
    --junitxml="${RUN_DIR}/junit.xml" \
    "${TEST_ID}" \
  2>&1 | tee "${RUN_DIR}/pytest.log"
pipeline_status=("${PIPESTATUS[@]}")
pytest_rc="${pipeline_status[0]}"
tee_rc="${pipeline_status[1]}"
set -e
test "${tee_rc}" -eq 0 || { echo 'Failed to persist pytest.log.' >&2; exit 2; }
end_ns="$(date +%s%N)"

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'arm=%s\n' "${ARM}"
  printf 'run_mode=%s\n' "${RUN_MODE}"
  printf 'host=%s\n' "${EXPECTED_HOST}"
  printf 'base_sha=%s\n' "${BASE_SHA}"
  printf 'experiment_sha=%s\n' "${EXPERIMENT_SHA}"
  printf 'image_digest=%s\n' "${IMAGE_REF}"
  printf 'allocation_id=%s\n' "${ALLOCATION_ID}"
  printf 'wheel_sha256=%s\n' "${WHEEL_SHA256}"
  printf 'gpu_fingerprint=%s\n' "${GPU_FINGERPRINT}"
  printf 'model_fingerprint=%s\n' "${MODEL_FINGERPRINT}"
  printf 'dataset_fingerprint=%s\n' "${DATASET_FINGERPRINT}"
  printf 'test_id=%s\n' "${TEST_ID}"
  printf 'pytest_rc=%s\n' "${pytest_rc}"
  printf 'start_ns=%s\n' "${start_ns}"
  printf 'end_ns=%s\n' "${end_ns}"
  printf 'duration_s=%s\n' "$(( (end_ns - start_ns) / 1000000000 ))"
} > "${RUN_DIR}/status.txt"

grep -nE \
  'NVBUG6396413_JSON|Evaluated accuracy|took too long|Hang detected|register|FileNotFoundError|OutOfMemory|FAILED|PASSED|Timeout' \
  "${RUN_DIR}/pytest.log" > "${RUN_DIR}/key-events.txt" || true

nvidia-smi -q > "${RUN_DIR}/nvidia-smi.after.txt"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
  --format=csv,noheader | LC_ALL=C sort > "${RUN_DIR}/compute-after.csv"
pgrep -a -f '[t]rtllm-serve|[t]est_disaggregated_serving|[p]ytest' \
  > "${RUN_DIR}/process-after.txt" || true

diff -u "${RESULT_ROOT}/compute-baseline.csv" "${RUN_DIR}/compute-after.csv" \
  > "${RUN_DIR}/compute-baseline.diff" || {
    echo 'GPU process baseline changed; stop before another run.' >&2
    exit 2
  }
diff -u "${RESULT_ROOT}/process-baseline.txt" "${RUN_DIR}/process-after.txt" \
  > "${RUN_DIR}/process-baseline.diff" || {
    echo 'Test process baseline changed; stop before another run.' >&2
    exit 2
  }
touch "${RUN_DIR}/cleanup.ok"

python3 - "${RUN_DIR}/pytest.log" "${ARM}" "${EXPERIMENT_SHA}" <<'PY'
import json
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
expected_arm = sys.argv[2]
expected_sha = sys.argv[3]
marker = "NVBUG6396413_JSON "
configs = []
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if marker not in line:
        continue
    payload = json.loads(line.split(marker, 1)[1])
    if payload.get("event") == "config":
        configs.append(payload)

assert len(configs) == 4, f"expected four config events, got {len(configs)}"
assert len({item["rank"] for item in configs}) == 4, configs
for item in configs:
    assert item["arm"] == expected_arm, item
    assert item["bypass"] == (expected_arm == "B"), item
    assert item["max_tokens_in_buffer"] == 8192, item
    assert item["tokens_per_block"] == 32, item
    assert item["budget_blocks"] == 256, item
    assert item["experiment_sha"] == expected_sha, item
PY
touch "${RUN_DIR}/protocol.ok"
```

The final validator requires exactly four `config` events—one per generation CP rank—with the expected arm, buffer,
block size, budget, and `EXPERIMENT_SHA`. Any missing or conflicting marker makes the sample `protocol_invalid`; stop
the block and fix instrumentation rather than counting it.

Product failures are valid experimental data when configuration and cleanup remain valid. Do not stop merely because
pytest returned nonzero. Infrastructure/setup failures are excluded: retry the same scheduled arm in a new attempt
directory, with at most three attempts total, and never advance the fixed order until that slot has a valid product
outcome. Use deterministic names: `NN-A-attempt-1`, then `NN-A-attempt-2` or `NN-A-attempt-3` for retries.

### 9.3 Required sequence

Run one uncounted warm-up per arm:

```bash
RUN_ID=warmup-A-attempt-1 ARM=A RUN_MODE=primary
# Execute the one-run command.

RUN_ID=warmup-B-attempt-1 ARM=B RUN_MODE=primary
# Execute the one-run command.
```

A warm-up is ready only if all four config markers are correct, the test reaches evaluation, the final aggregate exists,
and cleanup returns to baseline. Its accuracy result does not enter the measured comparison.

Then run these measured slots with `RUN_MODE=primary`, retrying an excluded slot without advancing:

| Slot | Arm | Canonical run ID |
| ---: | --- | --- |
| 1 | A | `01-A-attempt-1` |
| 2 | B | `02-B-attempt-1` |
| 3 | B | `03-B-attempt-1` |
| 4 | A | `04-A-attempt-1` |
| 5 | B | `05-B-attempt-1` |
| 6 | A | `06-A-attempt-1` |
| 7 | A | `07-A-attempt-1` |
| 8 | B | `08-B-attempt-1` |
| 9 | A | `09-A-attempt-1` |
| 10 | B | `10-B-attempt-1` |

Do not reorder successful slots after seeing results.

## 10. Classify results

Validity and product outcome are separate fields.

| Field | Allowed values |
| --- | --- |
| `validity` | `valid`, `infra_excluded`, `protocol_invalid` |
| `cleanup_ok` | `true`, `false` |
| `product_outcome` | `pass`, `accuracy_below_threshold`, `accuracy_evaluation_duration_failure`, `model_forward_hang`, `transfer_stall`, `registration_timeout`, `crash_oom`, `outer_timeout_unknown`, `unexpected_product_failure`; use `not_run` only for an excluded/invalid attempt |
| `rank_consistent` | `true`, `false`, or `unknown` when a failure prevents comparable decision digests |

Use `Evaluated accuracy:` from the log. The current NVFP4+FP8 GSM8K reference is 95.2%; with 1,319 examples, the
computed pass threshold is approximately 91.997%. Never set `INTEGRATION_TEST=1` or
`TRTLLM_ACCURACY_NO_REFERENCE`; either would invalidate the accuracy experiment.

Classify in this order:

1. Cache-path `FileNotFoundError`, image/mount failure, node Xid/ECC failure, allocation loss, or artifact-write failure:
   `infra_excluded` and `not_run`; retry the same slot.
2. Pytest success with accuracy at or above threshold: `valid` and `pass`.
3. Completed `Evaluated accuracy:` below threshold: `valid` and `accuracy_below_threshold`.
4. Evaluation returns but exceeds its 1,500-second post-hoc duration check: `valid` and
   `accuracy_evaluation_duration_failure`. A hung evaluator is interrupted only by pytest or the outer timeout.
5. Context/generation service readiness or registration expiry: `valid` and `registration_timeout`.
6. Hang detector or stacks with model-forward evidence: `valid` and `model_forward_hang`.
7. Corroborated unresolved KV transfer before forward: `valid` and `transfer_stall`.
8. CUDA OOM or product-process crash: `valid` and `crash_oom`.
9. Outer timeout without a narrower signature: `valid` and `outer_timeout_unknown`.
10. Any other product assertion or exception: `valid` and `unexpected_product_failure`.

Do not classify a test as `transfer_stall` from one unmatched `wait_begin` alone. Require no later forward progress and
corroborating transfer-side logs or stacks. A final cleanup stack is not proof of the original hang location.

Create `${RESULT_ROOT}/results.csv` with one row per measured attempt and this header. Keep warm-up and verbose runs out
of this CSV:

```csv
slot,attempt,run_id,arm,validity,cleanup_ok,product_outcome,host,base_sha,experiment_sha,image_digest,allocation_id,wheel_sha256,gpu_fingerprint,model_fingerprint,dataset_fingerprint,test_id,pytest_rc,duration_s,accuracy_pct,admission_decisions,would_defer_count,effective_deferred_requests,effective_deferral_request_ms,max_effective_deferral_ms,blocked_poll_count,blocked_poll_ms,decision_ms,rank_consistent,notes
```

For completed tests, prefer the JUnit testcase duration over the outer wrapper duration. Do not include hang timeout
durations in completed-test performance statistics.

## 11. Compare measured results

Generate a draft CSV from measured run directories:

```bash
source /results/nvbug-6396413-active.env
python3 /runbook/extract_results.py \
  "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/results.csv"
```

Review every row against `key-events.txt`, the full pytest log, JUnit, final telemetry, and any stacks. The extractor marks
each row `REVIEW_REQUIRED`; replace that note with the evidence supporting validity, outcome, accuracy, and rank
consistency. Preserve CSV quoting when notes contain commas. Do not relabel an ambiguous hang as a transfer stall
without corroboration.

Then run the checked-in comparator:

```bash
source /results/nvbug-6396413-active.env
python3 \
  /runbook/compare_results.py \
  "${RESULT_ROOT}/results.csv" \
  --expected-order A,B,B,A,B,A,A,B,A,B \
  --min-valid-per-arm 5 \
  --output "${RESULT_ROOT}/summary.md" \
  --json-output "${RESULT_ROOT}/summary.json"

cat "${RESULT_ROOT}/summary.md"
```

The comparator must fail on missing/duplicate measured slots, order drift, fewer than five valid samples per arm, mixed
SHA/image/host/GPU/model fingerprints, cleanup failure, or an observed rank inconsistency. An `unknown` rank result
does not fail validation, but it forces confirmatory runs and must be explained.

Interpret five runs per arm as screening evidence only. Extend to at least ten valid runs per arm, preserving an
interleaved predeclared order, if any of these is true:

- either arm has a product failure;
- outcome counts differ by arm;
- the completed-run median differs by at least 5%;
- admission deferral or blocked-poll time could explain a duration difference; or
- rank fingerprints disagree.

Admission is implicated only by a repeatable arm-dependent outcome with matching telemetry. Both arms failing similarly
points away from admission. Arm B failing more often means admission may be protective. A faster arm is not better if
accuracy or liveness regresses.

For a same-allocation confirmation, predeclare slots 11-20 as `B A A B A B B A B A`, producing ten valid samples per
arm across 20 slots. Request more allocation time before starting slot 11. Then compare with:

```bash
source /results/nvbug-6396413-active.env
python3 /runbook/extract_results.py \
  "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/results-confirmatory.csv"
```

Review all 20 measured rows in `results-confirmatory.csv` and replace every `REVIEW_REQUIRED` note, using the same
evidence rules as the initial screen. Preserve the already reviewed 10-run `results.csv`; do not overwrite it. Then run:

```bash
source /results/nvbug-6396413-active.env
python3 /runbook/compare_results.py \
  "${RESULT_ROOT}/results-confirmatory.csv" \
  --expected-order A,B,B,A,B,A,A,B,A,B,B,A,A,B,A,B,B,A,B,A \
  --min-valid-per-arm 10 \
  --output "${RESULT_ROOT}/summary-confirmatory.md" \
  --json-output "${RESULT_ROOT}/summary-confirmatory.json"
```

If the allocation changes, create a new result root and analyze it as a separate block. Do not bypass the comparator's
allocation-ID invariant or silently pool blocks.

## 12. Run verbose diagnostics after measured timing

Run one additional uncounted sample per arm with:

```bash
source /results/nvbug-6396413-active.env
RUN_ID=verbose-A-attempt-1 ARM=A RUN_MODE=verbose
# Execute the one-run command.

RUN_ID=verbose-B-attempt-1 ARM=B RUN_MODE=verbose
# Execute the one-run command.
```

Use the exact run IDs `verbose-A-attempt-1` and `verbose-B-attempt-1`, the same one-run command, and the same invariants.
Do not include these durations in the primary timing comparison. They exist to capture per-decision, per-rank, and
wait-begin/end details without perturbing the measured runs.

## 13. Evidence bundle and handoff

The result root must contain:

- source, image, wheel, host, allocation, GPU, model, and dataset fingerprints;
- `diagnostic.diff`, build/install logs, unit-test JUnit, and exact collection log;
- per-run pytest/JUnit/key-event/GPU/process artifacts;
- raw `NVBUG6396413_JSON` records;
- `results.csv`, `summary.md`, and `summary.json`; and
- a conclusion stating whether B300 supports, weakens, or leaves the admission hypothesis unresolved.

Do not remove the NVBug waiver or move the bug to Verify-To-Close from B300 evidence alone. Closure still requires the
exact unwaived B200 stage to execute without result reuse, plus any required post-merge validation. See
[`README.md`](README.md) for incident history, related PRs, and closure requirements.
