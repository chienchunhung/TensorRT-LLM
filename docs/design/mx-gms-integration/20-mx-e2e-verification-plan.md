<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 20. ModelExpress End-to-End Verification Plan

[< Back to README](README.md)

**Status:** Ready to execute against a combined integration head

**Last Updated:** 2026-07-09

**Implementations under test:**

- [NVIDIA/TensorRT-LLM#15641](https://github.com/NVIDIA/TensorRT-LLM/pull/15641): optional MX packaging, local
  Docker/Redis lifecycle, and ModelExpress 0.4.1 integration.
- [NVIDIA/TensorRT-LLM#16159](https://github.com/NVIDIA/TensorRT-LLM/pull/16159): ArtifactIdentity and SourceIdentity
  format v2.

**Intended executor:** An AI agent with shell access to a Linux cluster, one allocated GPU node, Docker, model weights,
and permission to build TensorRT-LLM.

---

## 1. Objective

Prove that the TensorRT-LLM ModelExpress (MX) integration works end to end on a single cluster node:

1. A donor loads a supported Llama checkpoint from disk and publishes its post-transform GPU weights through MX.
2. A receiver with the same model and parallel layout receives those weights directly into GPU parameter buffers.
3. The receiver uses the staged post-load path instead of transforming already-transformed weights again.
4. Baseline, donor, and receiver produce identical deterministic token IDs.
5. A receiver that cannot read checkpoint weight shards still succeeds, proving that success did not come from a
   silent Hugging Face disk fallback.
6. TRT-LLM's optional local Docker launcher creates and reuses the MX server and Redis containers correctly.
7. Source and receiver match on a content-bound ArtifactIdentity as well as runtime layout identity.
8. Unsupported, artifact-mismatched, or runtime-incompatible configurations reject P2P transfer before RDMA and fall
   back safely.

This is a **functional qualification** plan. Collect startup and transfer performance data, but do not fail the core
experiment solely because a first-run latency target is missed.

## 2. Scope and Known Boundaries

### In scope

- One Linux cluster node.
- Two-GPU TP=1 smoke test or four-GPU TP=2 qualification test.
- PyTorch backend.
- `LlamaForCausalLM`, transform protocol version 1.
- ModelExpress client and server version `0.4.1`.
- ArtifactIdentity format version 1 nested in SourceIdentity format version 2.
- Explicitly managed MX server first, then TRT-LLM automatic local-server launch.
- Deterministic inference, transfer evidence, canonical-snapshot disk-isolation proof, artifact/runtime identity
  mismatch, and server-failure fallback.

### Out of scope

- Cross-node fabric qualification. A single-node pass does not prove cross-node IB/RoCE behavior.
- Non-Llama post-transform reception.
- A separately loaded draft model or target-plus-draft transfer.
- MX+GMS composition.
- Production performance sign-off.
- Optimization of local-checkpoint ArtifactIdentity hashing. PR #16159 intentionally reads local checkpoint files in
  full; this plan records the cost but does not set its production SLO.
- Component-scoped ArtifactIdentity for target, draft, language, vision, or adapters.

## 3. Required Result

The executor must finish with one of these explicit outcomes:

- **PASS:** All core gates G0-G5 pass.
- **FAIL:** A core gate produces reproducible evidence of an implementation defect.
- **BLOCKED:** The environment cannot satisfy a prerequisite such as Docker, NIXL, model access, or a compatible GPU
  allocation. Environment blockers must not be reported as product failures.

Do not report PASS from logs alone. PASS requires exact token-ID equality, matching ArtifactIdentity evidence, and the
canonical-snapshot no-weight-shards receiver proof.

## 4. Recommended Topology

### Preferred qualification topology

Use one node with four H100, H200, B100, B200, or B300 GPUs:

```text
GPU 0,1: donor, TP=2       MX server + Redis       GPU 2,3: receiver, TP=2
          ranks 0,1  -------- NIXL/UCX -------->             ranks 0,1
```

This validates rank-to-rank matching and tensor-parallel shard identity while keeping the test on one node.

### Minimum smoke topology

Use two GPUs:

```text
GPU 0: donor, TP=1         MX server + Redis         GPU 1: receiver, TP=1
```

Run the two-GPU smoke first when environment setup is uncertain. Promote to the four-GPU TP=2 test before declaring
the integration qualified.

## 5. Execution Rules for the AI Agent

1. Work in one isolated allocation and one timestamped run directory.
2. Use an isolated local integration worktree containing the exact two PR heads. Do not modify or push either PR
   branch or the temporary merge.
3. Record every resolved input and exact command before running the experiment.
4. Keep the donor process alive until every receiver test finishes. MX source tensors and NIXL registrations are
   owned by the live donor process.
5. Stop at the first failed core gate. Preserve logs and diagnose before continuing.
6. Do not change source code to make a test pass unless the user separately authorizes a fix.
7. Treat donor disk fallback as expected when no source exists. Treat receiver disk fallback as a failure in the
   positive P2P tests.
8. Run positive transfer tests with an immutable checkpoint and record the ArtifactIdentity format, scheme, and digest.
   Use a canonical Hugging Face snapshot path for G4 so identity can be verified without opening weight shards.
9. Keep baseline, donor, and receiver model settings identical except where a negative test intentionally changes one
   field.
10. Archive evidence before cleanup.

## 6. Inputs to Resolve

The executor must resolve and record these values in `$RUN/manifest.txt`:

| Variable | Required value |
|:--|:--|
| `MX_PR_NUMBER` | `15641`, unless the user provides a successor PR |
| `ARTIFACT_PR_NUMBER` | `16159`, unless the user provides a successor PR |
| `MX_PR_HEAD` | Exact fetched PR #15641 commit SHA |
| `ARTIFACT_PR_HEAD` | Exact fetched PR #16159 commit SHA |
| `TEST_HEAD` | Exact temporary integration commit containing both PRs |
| `MODEL` | Canonical immutable Hugging Face snapshot path for a `LlamaForCausalLM` checkpoint; required for G4 |
| `MODEL_REVISION` | Hub commit, LFS object IDs, or a SHA-256 shard manifest |
| `ARTIFACT_IDENTITY` | Runtime-reported format version, scheme, and digest |
| `TP` | `1` for two GPUs or `2` for four GPUs |
| `DONOR_GPUS` | `0` or `0,1` |
| `RECEIVER_GPUS` | `1` or `2,3` |
| `ARCH` | `80-real`, `90-real`, or `100-real`, matching the node |
| `MX_PORT` | Unused host port, recommended `18001` |
| `LOCAL_MX_PORT` | Different unused host port, recommended `18002` |

Recommended first model: TinyLlama or another small unquantized Llama checkpoint for setup. After the smoke test,
repeat G1-G5 with the representative Llama model and quantization configuration intended for support. Resolve the
model to its cache path containing `models--<org>--<repo>/snapshots/<immutable-revision>`; a mutable Hub model name or
an arbitrary copied directory is not sufficient for the G4 no-shards gate.

## 7. Gate G0: Allocate and Qualify the Environment

**Goal:** Establish that failures after this gate are meaningful integration failures rather than missing
infrastructure.

### Step 7.1: Allocate one node

Inspect cluster availability and allocate one node with the required GPU count. On clusters with the TRT-LLM helper,
prefer:

```bash
auto-dev-safe --no-update -p <NODE_OR_PARTITION> --gpus <2_OR_4> -n <JOBS>
```

Verify the local helper's argument semantics with `auto-dev-safe --help`. Keep the allocation and all donor/receiver
processes on the same physical node.

### Step 7.2: Create the run directory

```bash
export RUN="${SCRATCH:-$HOME}/mx-e2e-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"/{artifacts,docker,logs,outputs,worktrees}
```

### Step 7.3: Capture hardware and runtime state

```bash
hostname | tee "$RUN/hostname.txt"
nvidia-smi -L | tee "$RUN/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$RUN/nvidia-topology.txt"
python3 --version | tee "$RUN/python-version.txt"
docker version | tee "$RUN/docker-version.txt"
```

Also record, when available:

```bash
ibv_devinfo > "$RUN/ibv-devinfo.txt" 2>&1 || true
env | sort > "$RUN/environment.txt"
```

Do not proceed unless:

- The requested GPUs are visible and not shared with an unrelated workload.
- Docker is usable from the environment that will run TRT-LLM.
- The model path is readable and resolves under a canonical immutable Hugging Face snapshot directory for the core
  G0-G5 run.
- The node has enough HBM for donor and receiver concurrently.
- The same execution environment can reach `127.0.0.1:<MX_PORT>`.

If Docker is unavailable, the core transfer could be tested against an externally provisioned MX server, but the
local-launch gate G5 will remain BLOCKED.

**G0 pass criterion:** GPU, Docker, model, storage, and local-network prerequisites are recorded and usable.

## 8. Prepare the Exact PR Source and Wheel

**Goal:** Run the experiment against one reproducible integration commit containing both PRs and verify the optional MX
dependency and SourceIdentity v2 contracts.

### Step 8.1: Create a temporary combined integration worktree

```bash
export REPO="$RUN/repos/TensorRT-LLM"
mkdir -p "$RUN/repos"
git clone https://github.com/NVIDIA/TensorRT-LLM.git "$REPO"
git -C "$REPO" fetch origin \
  "pull/15641/head:refs/remotes/origin/pr/15641" \
  "pull/16159/head:refs/remotes/origin/pr/16159"

git -C "$REPO" rev-parse refs/remotes/origin/pr/15641 | tee "$RUN/mx-pr-head.txt"
git -C "$REPO" rev-parse refs/remotes/origin/pr/16159 | tee "$RUN/artifact-pr-head.txt"

export SRC="$RUN/worktrees/mx-artifact-integration"
export INTEGRATION_BRANCH="mx-artifact-integration-$(basename "$RUN")"
git -C "$REPO" worktree add -b "$INTEGRATION_BRANCH" \
  "$SRC" refs/remotes/origin/pr/16159
git -C "$SRC" \
  -c user.name="MX E2E Integration" \
  -c user.email="mx-e2e-integration@example.invalid" \
  merge --no-edit refs/remotes/origin/pr/15641

git -C "$SRC" rev-parse HEAD | tee "$RUN/test-head.txt"
git -C "$SRC" log --oneline --decorate -8 | tee "$RUN/integration-history.txt"
```

This merge is a local test artifact only. Do not push it or modify either PR branch. If the merge conflicts, stop and
report **BLOCKED** with the conflict paths; do not invent an unreviewed integration resolution. If one PR is rebased on
the other before execution, use that descendant head directly and record the ancestry proof instead of creating a
redundant merge.

If the persistent cluster clone already exists, reuse it and create only the isolated integration worktree.

### Step 8.2: Build for the allocated GPU

Run from the normal TRT-LLM development container:

```bash
cd "$SRC"
./scripts/build_wheel.py \
  --trt_root /usr/local/tensorrt \
  --benchmarks \
  --use_ccache \
  -a "$ARCH" \
  -f \
  --nvtx 2>&1 | tee "$RUN/logs/build-wheel.log"
```

Use `90-real` for H100/H200 and `100-real` for B100/B200/B300. Use `80-real` for A100/A800.

### Step 8.3: Install the wheel with the MX extra

Resolve the actual wheel path instead of assuming a filename:

```bash
export WHEEL="$(find "$SRC" -path '*/tensorrt_llm-*.whl' -print -quit)"
test -n "$WHEEL"
python3 - "$WHEEL" <<'PY' | tee "$RUN/wheel-mx-metadata.txt"
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata_path = next(
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    )
    metadata = wheel.read(metadata_path).decode("utf-8")

requirements = [
    line for line in metadata.splitlines()
    if line.lower().startswith("requires-dist: modelexpress")
]
print("\n".join(requirements))
assert len(requirements) == 1
assert "modelexpress==0.4.1" in requirements[0]
assert "extra == \"mx\"" in requirements[0]
PY

python3 -m pip install --force-reinstall "${WHEEL}[mx]" \
  2>&1 | tee "$RUN/logs/install-wheel-mx.log"
python3 -m pip check | tee "$RUN/pip-check.txt"
```

### Step 8.4: Verify exact package and private API compatibility

```bash
python3 - <<'PY' | tee "$RUN/package-versions.txt"
import importlib.metadata as metadata

print("tensorrt-llm", metadata.version("tensorrt-llm"))
print("modelexpress", metadata.version("modelexpress"))
assert metadata.version("modelexpress") == "0.4.1"

from modelexpress import trtllm_live_transfer as transfer
from modelexpress.nixl_transfer import is_nixl_available
from tensorrt_llm._torch.weight_sharing.artifact_identity import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
)
from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
)

for name in ("MxClient", "MxLiveWeightLoader", "publish_model_params", "_build_trtllm_identity"):
    assert hasattr(transfer, name), name
assert ARTIFACT_IDENTITY_FORMAT_VERSION == 1
assert SOURCE_IDENTITY_FORMAT_VERSION == 2
print("artifact-identity-format", ARTIFACT_IDENTITY_FORMAT_VERSION)
print("source-identity-format", SOURCE_IDENTITY_FORMAT_VERSION)
assert is_nixl_available(), "NIXL Python bindings are unavailable"
PY
```

### Step 8.5: Run the combined focused unit suite

```bash
cd "$SRC"
pytest -q \
  tests/unittest/_torch/executor/test_model_loader_gms.py \
  tests/unittest/_torch/executor/test_model_loader_mx.py \
  tests/unittest/_torch/models/checkpoints/mx/test_mx_checkpoint_loader.py \
  tests/unittest/_torch/models/checkpoints/mx/test_mx_local_server.py \
  tests/unittest/_torch/weight_sharing/test_artifact_identity.py \
  tests/unittest/_torch/weight_sharing/test_source_identity.py \
  tests/unittest/_torch/weight_sharing/test_mx_source_identity_gate.py \
  tests/unittest/_torch/weight_sharing/test_gms_source_identity_gate.py \
  tests/unittest/llmapi/test_mx_args.py \
  2>&1 | tee "$RUN/logs/focused-mx-artifact-tests.log"
```

**Preparation pass criterion:** The exact combined `TEST_HEAD` builds, `tensorrt_llm[mx]` installs, `pip check` passes,
ModelExpress is exactly `0.4.1`, SourceIdentity format v2 resolves, required transfer symbols resolve, and NIXL is
available. The combined focused suite passes without excluding tests changed by either PR.

## 9. Record the Immutable Model Artifact

**Goal:** Exercise the actual PR #16159 implementation and prove the core run uses one immutable HF snapshot identity.

```bash
test -f "$MODEL/config.json"
python3 - "$MODEL" <<'PY' | tee "$RUN/artifact-identity.json"
import json
import sys
import time

from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity

started = time.perf_counter()
identity = ArtifactIdentity.from_checkpoint(sys.argv[1])
payload = identity.to_dict()
payload["construction_seconds"] = time.perf_counter() - started
print(json.dumps(payload, indent=2, sort_keys=True))

assert payload["format_version"] == 1
assert payload["scheme"] == "hf_snapshot_revision", (
    "Core G4 requires a canonical Hugging Face snapshot path; arbitrary local "
    "checkpoints use full-content hashing and cannot support the no-shards view."
)
PY
```

Record the same identity from donor and receiver startup evidence. The format, scheme, and digest must match exactly.

For a separate local-checkpoint characterization, run the same command on the local path and record
`scheme=checkpoint_manifest_sha256` plus `construction_seconds`. PR #16159 reads every retained file in full for that
scheme. Do not count those reads as Hugging Face weight loading, but do include them in startup latency and storage-I/O
analysis. A local checkpoint run cannot satisfy G4 unless a future trusted precomputed-identity input is added.

## 10. Install the Deterministic Worker Harness

**Goal:** Use one reproducible driver for baseline, donor, receiver, explicit-server, and local-launch runs.

Create `$RUN/mx_e2e_worker.py` with this content:

```python
#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from tensorrt_llm import LLM, SamplingParams


PROMPTS = [
    "The capital of France is",
    "Write three prime numbers greater than ten:",
    "In one sentence, explain why the sky appears blue.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("baseline", "donor", "receiver"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ready-file")
    parser.add_argument("--stop-file")
    parser.add_argument("--mx-url")
    parser.add_argument("--mx-port", type=int, default=18002)
    return parser.parse_args()


def main():
    args = parse_args()
    kwargs = {
        "model": args.model,
        "backend": "pytorch",
        "checkpoint_format": "HF" if args.role == "baseline" else "MX",
        "tensor_parallel_size": args.tp,
    }
    if args.role != "baseline":
        if args.mx_url:
            mx_config = {
                "server_url": args.mx_url,
                "server_query_timeout_s": 120,
                "local_server": {"enabled": False},
            }
        else:
            mx_config = {
                "server_query_timeout_s": 120,
                "local_server": {"enabled": True, "port": args.mx_port},
            }
        kwargs["mx_config"] = mx_config

    started = time.perf_counter()
    with LLM(**kwargs) as llm:
        load_seconds = time.perf_counter() - started
        sampling = SamplingParams(max_tokens=64, temperature=0.0, seed=42)
        results = list(llm.generate(PROMPTS, sampling))
        payload = {
            "role": args.role,
            "model": args.model,
            "tp": args.tp,
            "load_seconds": load_seconds,
            "outputs": [
                {
                    "prompt": result.prompt,
                    "token_ids": list(result.outputs[0].token_ids),
                    "text": result.outputs[0].text,
                }
                for result in results
            ],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
        if args.ready_file:
            Path(args.ready_file).write_text("ready\n")

        if args.role == "donor":
            if not args.stop_file:
                raise ValueError("donor requires --stop-file")
            stop_file = Path(args.stop_file)
            while not stop_file.exists():
                time.sleep(1)


if __name__ == "__main__":
    main()
```

The donor wait loop is intentional. Do not replace it with a process that exits after generation.

## 11. Start an Explicit MX Server

**Goal:** Validate transfer functionality without coupling the first result to TRT-LLM's Docker lifecycle helper.

Use run-specific Docker object names:

```bash
export MX_PORT=18001
export MX_NETWORK="mx-e2e-${MX_PORT}"
export MX_REDIS="${MX_NETWORK}-redis"
export MX_SERVER="${MX_NETWORK}-server"
export MX_URL="http://127.0.0.1:${MX_PORT}"

docker network create "$MX_NETWORK"
docker run -d --name "$MX_REDIS" --network "$MX_NETWORK" redis:8-alpine
docker run -d --name "$MX_SERVER" \
  --network "$MX_NETWORK" \
  -p "127.0.0.1:${MX_PORT}:8001" \
  -e MODEL_EXPRESS_SERVER_PORT=8001 \
  -e MODEL_EXPRESS_LOG_LEVEL=info \
  -e MX_METADATA_BACKEND=redis \
  -e "REDIS_URL=redis://${MX_REDIS}:6379" \
  nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.4.1
```

Wait for Redis and the MX port:

```bash
docker exec "$MX_REDIS" redis-cli ping | tee "$RUN/docker/redis-ping.txt"
python3 - "$MX_PORT" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"MX port {port} did not become ready")
PY
```

Capture initial logs:

```bash
docker logs "$MX_REDIS" > "$RUN/docker/redis-startup.log" 2>&1
docker logs "$MX_SERVER" > "$RUN/docker/server-startup.log" 2>&1
```

If NGC authentication or image access fails, classify the run as BLOCKED at G0.

## 12. Gate G1: Hugging Face Baseline

**Goal:** Produce deterministic reference token IDs and confirm the model works without MX.

Set topology variables. Four-GPU example:

```bash
export TP=2
export DONOR_GPUS=0,1
export RECEIVER_GPUS=2,3
```

Two-GPU smoke alternative:

```bash
export TP=1
export DONOR_GPUS=0
export RECEIVER_GPUS=1
```

Run the baseline before starting the donor:

```bash
CUDA_VISIBLE_DEVICES="$DONOR_GPUS" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 "$RUN/mx_e2e_worker.py" \
  --role baseline \
  --model "$MODEL" \
  --tp "$TP" \
  --output "$RUN/outputs/baseline.json" \
  > "$RUN/logs/baseline.log" 2>&1
```

Verify that `baseline.json` contains three non-empty token-ID arrays.

**G1 pass criterion:** Baseline construction and deterministic generation succeed.

## 13. Gate G2: Donor Disk Load and Publication

**Goal:** Load from disk, publish post-transform weights for every rank, and keep the source alive.

```bash
rm -f "$RUN/donor.ready" "$RUN/donor.stop"
mkdir -p "$RUN/logs/donor-mx"

CUDA_VISIBLE_DEVICES="$DONOR_GPUS" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MX_TRANSFER_LOG_DIR="$RUN/logs/donor-mx" \
python3 "$RUN/mx_e2e_worker.py" \
  --role donor \
  --model "$MODEL" \
  --tp "$TP" \
  --mx-url "$MX_URL" \
  --output "$RUN/outputs/donor.json" \
  --ready-file "$RUN/donor.ready" \
  --stop-file "$RUN/donor.stop" \
  > "$RUN/logs/donor.log" 2>&1 &
export DONOR_PID=$!
echo "$DONOR_PID" > "$RUN/donor.pid"
```

Wait up to 20 minutes for `$RUN/donor.ready`. While waiting, confirm `kill -0 "$DONOR_PID"` continues to succeed.
If the process exits, stop and inspect `donor.log`.

Required donor evidence:

```text
ModelExpress worker rank <rank> ... published
Published post-transform weights to MX server
```

For TP=2, both ranks 0 and 1 must publish. The donor may report that no source exists and that it is loading from disk;
that is expected for the first source.

**G2 pass criterion:** Every expected rank publishes nonzero weight bytes, donor output matches the baseline, and the
donor remains alive.

## 14. Gate G3: Full-Checkpoint Receiver P2P Transfer

**Goal:** Verify the normal receiver path before adding disk isolation.

```bash
mkdir -p "$RUN/logs/receiver-full-mx"

CUDA_VISIBLE_DEVICES="$RECEIVER_GPUS" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MX_TRANSFER_LOG_DIR="$RUN/logs/receiver-full-mx" \
python3 "$RUN/mx_e2e_worker.py" \
  --role receiver \
  --model "$MODEL" \
  --tp "$TP" \
  --mx-url "$MX_URL" \
  --output "$RUN/outputs/receiver-full.json" \
  > "$RUN/logs/receiver-full.log" 2>&1
```

Required receiver evidence:

```text
MX P2P weight transfer succeeded
MX receiver using staged post-load path for LlamaForCausalLM
Matched <N>/<N> params for direct RDMA transfer
Rank <rank>: transferred <N> params (<GB> GB) ... DIRECT into model params
```

For every expected rank, require:

- A per-rank log under `$RUN/logs/receiver-full-mx/`.
- Transferred bytes greater than zero.
- Matched count equals source descriptor count.
- No size mismatch, PVC fallback, or missing tensors.

Reject the run if receiver logs contain any of these:

```text
falling back to disk
partial fallback
Size mismatch
Still missing after PVC fallback
MX P2P transfer failed
source SourceIdentity incompatible
SourceIdentity mismatch on fields ['artifact_identity']
Unsupported SourceIdentity format version
invalid SourceIdentity
```

Compare token IDs:

```bash
python3 - "$RUN/outputs/baseline.json" "$RUN/outputs/donor.json" "$RUN/outputs/receiver-full.json" <<'PY'
import json
import sys

def tokens(path):
    payload = json.load(open(path))
    return [item["token_ids"] for item in payload["outputs"]]

baseline, donor, receiver = map(tokens, sys.argv[1:])
assert donor == baseline, "donor token IDs differ from HF baseline"
assert receiver == baseline, "receiver token IDs differ from HF baseline"
print("PASS: baseline, donor, and full receiver token IDs are identical")
PY
```

**G3 pass criterion:** Every rank performs full P2P transfer, the staged Llama path is selected, no fallback occurs,
and all token IDs are exact matches.

## 15. Gate G4: No-Weight-Shards Receiver Proof

**Goal:** Prove that a positive receiver result cannot be explained by silent disk fallback while still allowing
SourceIdentity v2 to verify the immutable snapshot without reading weight shards.

PR #16159 recognizes Hugging Face snapshots by the canonical
`models--<org>--<repo>/snapshots/<immutable-revision>/<optional-subpath>` structure. Create a metadata-only receiver
view with the same repository cache name, revision, and subpath. Do not copy the weight shards.

```bash
export RECEIVER_MODEL="$(python3 - "$MODEL" "$RUN" <<'PY'
from pathlib import Path
import sys

model = Path(sys.argv[1]).resolve()
run = Path(sys.argv[2]).resolve()
parts = model.parts

for index, part in enumerate(parts[:-1]):
    if part != "snapshots" or index == 0:
        continue
    repository_cache_name = parts[index - 1]
    revision = parts[index + 1].lower()
    if not repository_cache_name.startswith("models--"):
        continue
    if len(revision) not in (40, 64) or any(char not in "0123456789abcdef" for char in revision):
        continue
    subpath = parts[index + 2 :]
    destination = run / "receiver-hf-cache" / repository_cache_name / "snapshots" / revision
    if subpath:
        destination = destination.joinpath(*subpath)
    print(destination)
    break
else:
    raise SystemExit(
        "MODEL is not a canonical immutable Hugging Face snapshot path; G4 cannot proceed"
    )
PY
)"

mkdir -p "$RECEIVER_MODEL"
rsync -aL \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.gguf' \
  "$MODEL/" "$RECEIVER_MODEL/"

test -f "$RECEIVER_MODEL/config.json"
test -z "$(find "$RECEIVER_MODEL" -type f \( \
  -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o \
  -name '*.pth' -o -name '*.ckpt' -o -name '*.gguf' \
  \) -print -quit)"

python3 - "$MODEL" "$RECEIVER_MODEL" <<'PY' | tee "$RUN/artifact-identity-g4.txt"
import sys

from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity

donor = ArtifactIdentity.from_checkpoint(sys.argv[1])
receiver = ArtifactIdentity.from_checkpoint(sys.argv[2])
print("donor", donor.to_dict())
print("receiver", receiver.to_dict())
assert donor.scheme == "hf_snapshot_revision"
assert receiver == donor, "metadata-only receiver ArtifactIdentity differs from donor"
PY
```

Keep the donor alive and run a fresh receiver process:

```bash
mkdir -p "$RUN/logs/receiver-no-shards-mx"

CUDA_VISIBLE_DEVICES="$RECEIVER_GPUS" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MX_TRANSFER_LOG_DIR="$RUN/logs/receiver-no-shards-mx" \
python3 "$RUN/mx_e2e_worker.py" \
  --role receiver \
  --model "$RECEIVER_MODEL" \
  --tp "$TP" \
  --mx-url "$MX_URL" \
  --output "$RUN/outputs/receiver-no-shards.json" \
  > "$RUN/logs/receiver-no-shards.log" 2>&1
```

Repeat all G3 log checks and compare `receiver-no-shards.json` with `baseline.json`.

If P2P silently falls back, this run should fail because no weight shards are available. A successful run with matching
ArtifactIdentity and exact tokens is therefore the strongest functional evidence in this plan.

**G4 pass criterion:** Donor and receiver report the same `hf_snapshot_revision` ArtifactIdentity, the receiver has no
checkpoint weight files, every rank transfers nonzero bytes, no fallback is logged, and token IDs exactly match the HF
baseline.

## 16. Gate G5: Automatic Local-Server Launch and Reuse

**Goal:** Qualify TRT-LLM's Docker-backed MX and Redis lifecycle independently from the already-proven transfer path.

### Step 16.1: Stop the explicit-server phase

```bash
touch "$RUN/donor.stop"
wait "$DONOR_PID"
docker logs "$MX_SERVER" > "$RUN/docker/server-explicit-final.log" 2>&1
docker logs "$MX_REDIS" > "$RUN/docker/redis-explicit-final.log" 2>&1
docker rm -f "$MX_SERVER" "$MX_REDIS"
docker network rm "$MX_NETWORK"
```

### Step 16.2: Start a donor with no explicit URL

```bash
unset MODEL_EXPRESS_URL
export LOCAL_MX_PORT=18002
rm -f "$RUN/local-donor.ready" "$RUN/local-donor.stop"

CUDA_VISIBLE_DEVICES="$DONOR_GPUS" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MX_TRANSFER_LOG_DIR="$RUN/logs/local-donor-mx" \
python3 "$RUN/mx_e2e_worker.py" \
  --role donor \
  --model "$MODEL" \
  --tp "$TP" \
  --mx-port "$LOCAL_MX_PORT" \
  --output "$RUN/outputs/local-donor.json" \
  --ready-file "$RUN/local-donor.ready" \
  --stop-file "$RUN/local-donor.stop" \
  > "$RUN/logs/local-donor.log" 2>&1 &
export LOCAL_DONOR_PID=$!
```

After readiness, verify these objects exist and are running:

```text
trtllm-mx-18002
trtllm-mx-18002-redis
trtllm-mx-18002-server
```

Record the server and Redis container IDs.

### Step 16.3: Start the local-launch receiver

Run the receiver without `--mx-url`, using the same `--mx-port`. Require the same G3 transfer and token evidence.
After it exits, record container IDs again.

The IDs must be unchanged: the receiver must reuse the donor-created containers rather than creating replacements.
The receiver log should identify `http://127.0.0.1:18002` as the MX server.

### Step 16.4: Verify stopped-container recovery

1. Stop `trtllm-mx-18002-server` without removing it.
2. Start another receiver with the full checkpoint available.
3. Verify TRT-LLM restarts and reuses the same compatible container.
4. Require exact token equality and P2P transfer success.

If the cluster's dev container cannot see the host Docker socket or host loopback, report G5 as BLOCKED by the
environment. Do not reinterpret that as an MX transfer defect when G3 and G4 already passed with an explicit server.

**G5 pass criterion:** Automatic launch creates the expected objects, a second TRT-LLM instance reuses them, a stopped
compatible server recovers, and receiver inference remains exact.

## 17. Gate G6: Negative Controls

**Goal:** Verify fail-closed compatibility checks and safe fallback behavior. G6 is required for merge confidence but
is not part of the minimum G0-G5 functional PASS.

Use the full checkpoint for negative controls so an expected disk fallback can complete.

### N1: Identity mismatch before RDMA

Keep a TP=2 donor and launch a TP=1 receiver, or change another layout-affecting setting. Expected result:

- No direct-transfer log.
- Source identity is rejected or no exact source is selected.
- Receiver falls back to Hugging Face loading.
- Deterministic output remains correct for the mismatched receiver's own HF baseline.

### N2: Unreachable explicit server

Stop the external MX server and launch a receiver configured with its URL and local launch disabled. Expected result:

- A bounded query or connection failure.
- No hang.
- Hugging Face fallback succeeds.
- Output remains correct.

### N3: Occupied automatic-launch port

Bind the configured local port with an unrelated process, then launch MX with automatic local-server startup.
Expected result:

- TRT-LLM refuses to treat the unrelated listener as its MX server.
- It logs a clear port/configuration warning.
- It falls back to Hugging Face loading without terminating the application.

### N4: Unsupported model family, optional

Use a small non-Llama model only when time and storage permit. Expected result: post-transform reception is not
accepted and the receiver safely uses the standard checkpoint path.

### N5: Artifact mismatch before RDMA

Use two valid small checkpoint artifacts with the same architecture and tensor shapes but different immutable revisions
or contents. Keep the receiver artifact fully readable so expected disk fallback can complete. Expected result:

- Source discovery reports an ArtifactIdentity or SourceIdentity mismatch.
- No direct-transfer log or transferred bytes appear.
- Rejection happens before P2P registration/transfer.
- The receiver loads its own artifact from disk and matches that artifact's HF baseline.

For a local-checkpoint variant, use a valid second checkpoint or re-save a changed tensor with the checkpoint library;
do not corrupt a shard byte and then mistake the expected disk-load failure for identity-gate evidence.

### N6: SourceIdentity v1 compatibility fallback

Use the focused test fixture or a controlled v1 publisher to present metadata without the required v2 ArtifactIdentity.
Expected result: the v2 receiver rejects it with an explicit unsupported/missing identity reason and falls back before
P2P. Record the result as upgrade-order evidence; do not relax the v2 requirement.

## 18. Optional Performance Characterization

Run this section only after G0-G5 pass.

1. Keep one donor alive.
2. Launch three fresh receiver processes serially on the receiver GPU set.
3. Record `load_seconds` from each output JSON.
4. Parse each per-rank MX log for transferred parameters, bytes, elapsed time, and Gbps.
5. Compare the median receiver load time with the HF baseline.
6. Record filesystem read bytes when tooling such as `pidstat -d`, `iostat`, or cgroup I/O accounting is available.
7. Run a short steady-state generation or throughput check to ensure the load path does not change runtime behavior.

Report performance as measured evidence, not as a core correctness gate. Do not compare cold disk, warm page cache,
and MX numbers without labeling cache state.

### 18.1 July 2026 characterization results

The following results were collected on July 17-19, 2026 against PR #15641 at
`752c05c9af87813ced3622836585c20c7c6f8e20`. They characterize the explicit-server MX data path across three Llama
model sizes. They do **not** constitute the overall PASS defined in Section 3: the run did not include PR #16159,
Docker was unavailable for G5, and G6 was not run.

#### Method

- PyTorch backend on B300 nodes, with donor GPUs 0-3 and receiver GPUs 4-7 for TP=4.
- Native Redis and MX server at `http://127.0.0.1:8001`; automatic Docker lifecycle was not exercised.
- Checkpoints resided on NFS. Before every measured stage, the harness requested eviction of the local client page
  cache for weight files with `POSIX_FADV_DONTNEED`, but it did not gate execution on measured residency.
  Subsequent `mincore` validation on the same mount showed that the advisory could leave 100% of pages resident.
  These measurements must therefore be treated as client-cache warm/uncontrolled, not as first-read NFS results;
  server-side NFS cache was also uncontrolled.
- Each complete run executed an HF baseline, an MX donor, a full-checkpoint MX receiver, and an MX receiver whose
  local view omitted weight shards. Scenario ordering rotated between cycles.
- The 70B run used two-hour donor/heartbeat timeouts. The 405B-FP8 run used four-hour timeouts,
  `max_seq_len=2048`, and `kv_cache_config.free_gpu_memory_fraction=0.30`.
- `LLM init` is wall time around `LLM(...)`. `Model loader` and `checkpoint source` are the maximum observed rank
  durations. `MX transfer` is the mean across cycles of each cycle's maximum rank duration.
- Reductions are relative to the HF baseline at the same model, cache mode, and TP:
  `(HF - MX) / HF * 100`.

#### Run matrix and correctness

| Model | Quantization | TP | Complete runs | Result |
|:--|:--|--:|--:|:--|
| TinyLlama-1.1B-Chat-v1.0 | BF16 | 1 | 2 of 3 | Two complete runs passed; cycle 2 hit a harness quoting error during G4 and is excluded |
| TinyLlama-1.1B-Chat-v1.0 | BF16 | 2 | 3 of 3 | All four scenarios passed |
| TinyLlama-1.1B-Chat-v1.0 | BF16 | 4 | 3 of 3 | All four scenarios passed |
| Llama-3.3-70B-Instruct | BF16 | 4 | 5 of 5 | All four scenarios passed |
| Llama-3.1-405B-Instruct-FP8 | FP8 | 4 | 5 of 5 | All four scenarios passed |

For every complete run, baseline, donor, full receiver, and no-shards receiver produced the same deterministic token
hash. Every receiver rank matched all published tensors and logged direct transfer into model parameters. The observed
TP=4 per-rank payloads were 135 tensors/0.55 GB for TinyLlama, 483 tensors/35.28 GB for 70B, and 3,279
tensors/102.52 GB for 405B-FP8. The corresponding aggregate logical payloads across four ranks were 2.20 GB,
141.12 GB, and 410.08 GB.

The no-shards result proves the PR #15641 receiver did not silently load local weight shards. It does not satisfy the
updated G4 ArtifactIdentity requirement by itself: these runs predated the combined #15641 + #16159 head and did not
record a canonical-snapshot ArtifactIdentity v1 digest inside SourceIdentity v2.

#### End-to-end LLM initialization

Times are arithmetic means over the complete runs above.

| Model | TP | N | HF (s) | MX full (s) | Reduction | MX no-shards (s) | Reduction |
|:--|--:|--:|--:|--:|--:|--:|--:|
| TinyLlama-1.1B BF16 | 1 | 2 | 268.08 | 263.26 | 1.8% | 261.91 | 2.3% |
| TinyLlama-1.1B BF16 | 2 | 3 | 226.05 | 208.37 | 7.8% | 217.72 | 3.7% |
| TinyLlama-1.1B BF16 | 4 | 3 | 152.47 | 106.47 | 30.2% | 114.36 | 25.0% |
| Llama-3.3-70B BF16 | 4 | 5 | 345.64 | 117.99 | 65.9% | 119.34 | 65.5% |
| Llama-3.1-405B FP8 | 4 | 5 | 592.61 | 190.39 | 67.9% | 190.29 | 67.9% |

The total-init benefit grows with model size because fixed process and distributed-startup costs dominate TinyLlama,
whereas NFS checkpoint reads dominate the larger baselines.

#### Model-loader and checkpoint-source durations

| Model | TP | HF loader (s) | MX full loader (s) | Reduction | MX no-shards loader (s) | Reduction |
|:--|--:|--:|--:|--:|--:|--:|
| TinyLlama-1.1B BF16 | 1 | 9.64 | 1.36 | 85.9% | 1.35 | 86.0% |
| TinyLlama-1.1B BF16 | 2 | 7.21 | 1.57 | 78.2% | 1.46 | 79.8% |
| TinyLlama-1.1B BF16 | 4 | 7.64 | 2.12 | 72.3% | 2.16 | 71.7% |
| Llama-3.3-70B BF16 | 4 | 212.61 | 2.37 | 98.9% | 2.24 | 98.9% |
| Llama-3.1-405B FP8 | 4 | 380.35 | 5.60 | 98.5% | 5.62 | 98.5% |

| Model | TP | HF checkpoint source (s) | MX full source (s) | Reduction | MX no-shards source (s) | Reduction |
|:--|--:|--:|--:|--:|--:|--:|
| TinyLlama-1.1B BF16 | 1 | 8.83 | 1.21 | 86.3% | 1.20 | 86.4% |
| TinyLlama-1.1B BF16 | 2 | 6.55 | 1.34 | 79.6% | 1.31 | 80.0% |
| TinyLlama-1.1B BF16 | 4 | 6.45 | 1.62 | 74.8% | 1.68 | 74.0% |
| Llama-3.3-70B BF16 | 4 | 206.13 | 1.30 | 99.4% | 1.26 | 99.4% |
| Llama-3.1-405B FP8 | 4 | 361.18 | 4.36 | 98.8% | 4.35 | 98.8% |

#### Direct-transfer duration

| Model | TP | Timing samples | MX full max-rank mean (s) | MX no-shards max-rank mean (s) |
|:--|--:|--:|--:|--:|
| TinyLlama-1.1B BF16 | 1 | 1 | 0.29 | 0.28 |
| TinyLlama-1.1B BF16 | 2 | 1 | 0.51 | 0.52 |
| TinyLlama-1.1B BF16 | 4 | 1 | 0.99 | 1.35 |
| Llama-3.3-70B BF16 | 4 | 5 | 0.37 | 0.35 |
| Llama-3.1-405B FP8 | 4 | 5 | 1.05 | 1.08 |

These are MX-reported operation durations, not an independent fabric-throughput measurement. In particular, the
logical tensor byte counts and derived Gbps can exceed a physical-link interpretation when the same-node path uses
GPU-memory registration and asynchronous operations. Use the end-to-end initialization and loader timings for the
startup comparison. The TinyLlama parser emitted transfer durations for only one cycle at each TP, so those values
should not be treated as multi-cycle means.

#### Remaining qualification work

- Repeat the scaling characterization with a privileged node page-cache drop and a post-reset `mincore` residency
  requirement below 1%; do not use the existing results as true-cold NFS measurements.
- Repeat the representative positive path on the exact combined #15641 + #16159 integration head.
- Record ArtifactIdentity v1 and SourceIdentity v2 from a canonical immutable Hugging Face snapshot, including the G4
  metadata-only receiver proof.
- Run G5 in an environment with Docker access.
- Run the G6 negative controls. Until these steps complete, retain the Section 3 outcome as **BLOCKED/INCOMPLETE**,
  not PASS.

## 19. Acceptance Matrix

| Gate | Required evidence | Pass condition |
|:--|:--|:--|
| G0 Environment | GPU, Docker, NIXL, model, network records | All prerequisites usable |
| Build/install | Both PR heads, integration head, build log, wheel path, `pip check`, versions | Combined wheel, MX 0.4.1, ArtifactIdentity v1, and SourceIdentity v2 installed; NIXL available |
| G1 Baseline | `baseline.json`, baseline log | Three deterministic outputs produced |
| G2 Donor | Donor log, published rank metadata, `donor.json` | Every rank publishes; tokens equal baseline; donor stays alive |
| G3 Full receiver | Receiver and per-rank MX logs, output JSON | Full P2P, staged Llama path, no fallback, exact tokens |
| G4 No-shards receiver | Matching ArtifactIdentity, metadata-only canonical snapshot listing, logs, output JSON | P2P succeeds with no weight files; exact tokens |
| G5 Local launch | Docker inspect/logs before and after receiver | Create, reuse, restart, and exact receiver success |
| G6 Negative controls | Logs and outputs for N1-N3 and N5-N6; N4 when available | Runtime/artifact/version reject and fallback paths are bounded and correct |
| Performance | Three run JSON files and transfer metrics | Reported without overstating cache-state comparisons |

## 20. Failure Classification

Use these categories in the final report:

| Category | Examples | Required action |
|:--|:--|:--|
| Environment | No Docker socket, image pull denied, no NIXL, insufficient HBM | Mark BLOCKED; preserve preflight evidence |
| Build/package | Combined wheel does not build, `[mx]` cannot resolve, wrong MX/identity API | Mark FAIL; include exact resolver/build error |
| Discovery/identity | Donor publishes but receiver cannot select exact source or matching artifact | Mark FAIL; preserve ArtifactIdentity/SourceIdentity metadata, model names, and rank logs |
| Transfer | NIXL init/register/receive fails or bytes are zero | Mark FAIL; preserve UCX/NIXL topology and per-rank logs |
| Layout | Partial fallback, size mismatch, unmatched tensors | Mark FAIL; list all affected names and layouts |
| Staged hooks | Receiver transfers but reruns transforms or tokens differ | Mark FAIL; preserve staged-path and output evidence |
| Docker lifecycle | Explicit transfer passes but local creation/reuse fails | G3/G4 may PASS; mark G5 FAIL or BLOCKED with reason |
| Expected fallback | Negative control rejects P2P and loads from disk correctly | PASS for that negative control |

## 21. Evidence Layout

The completed run directory should contain at least:

```text
$RUN/
├── manifest.txt
├── mx-pr-head.txt
├── artifact-pr-head.txt
├── test-head.txt
├── integration-history.txt
├── artifact-identity.json
├── artifact-identity-g4.txt
├── environment.txt
├── nvidia-topology.txt
├── package-versions.txt
├── wheel-mx-metadata.txt
├── pip-check.txt
├── mx_e2e_worker.py
├── artifacts/
├── docker/
│   ├── server-explicit-final.log
│   ├── redis-explicit-final.log
│   └── local-container-inspect.json
├── logs/
│   ├── build-wheel.log
│   ├── focused-mx-artifact-tests.log
│   ├── baseline.log
│   ├── donor.log
│   ├── receiver-full.log
│   ├── receiver-no-shards.log
│   └── <run>-mx/rank<N>.log
├── outputs/
│   ├── baseline.json
│   ├── donor.json
│   ├── receiver-full.json
│   └── receiver-no-shards.json
└── report.md
```

Do not delete failed-run artifacts before the owner has reviewed them.

## 22. Final Report Template

The executor must write `$RUN/report.md` and return its path to the user.

```markdown
# MX E2E Verification Report

## Result

PASS | FAIL | BLOCKED

## Reproduction Identity

- MX PR and head SHA:
- ArtifactIdentity PR and head SHA:
- Combined test-head SHA:
- TRT-LLM wheel version:
- ModelExpress client/server version:
- ArtifactIdentity format/scheme/digest and construction time:
- Model path and immutable revision:
- Cluster, node, container:
- GPU type/count and topology:
- TP and GPU assignment:

## Gate Results

| Gate | Result | Key evidence |
|:--|:--|:--|
| G0 Environment | | |
| Build/install | | |
| G1 Baseline | | |
| G2 Donor | | |
| G3 Full receiver | | |
| G4 No-shards receiver | | |
| G5 Local launch | | |
| G6 Negative controls | | |

## Correctness

- Baseline token IDs:
- Donor equality:
- Full receiver equality:
- No-shards receiver equality:

## Transfer Evidence

- Published ranks and bytes:
- Receiver ranks and bytes:
- Matched tensor counts:
- Transfer duration and bandwidth:
- Fallback or mismatch messages:
- Artifact/version mismatch controls:

## Performance

- HF load time:
- Receiver load times:
- Median receiver load time:
- Cache state and disk-read evidence:
- Local-manifest identity construction time, if characterized:

## Failures or Deviations

- Exact failing command:
- First causal error:
- Classification:
- Reproduction steps:
- Suggested next action:

## Artifact Paths

- Run directory:
- Logs:
- Outputs:
- Docker evidence:
```

## 23. Cleanup

Capture logs first, then stop processes and remove only this run's Docker objects:

```bash
test -n "${LOCAL_DONOR_PID:-}" && touch "$RUN/local-donor.stop"
test -n "${LOCAL_DONOR_PID:-}" && wait "$LOCAL_DONOR_PID" || true

docker logs "trtllm-mx-${LOCAL_MX_PORT}-server" \
  > "$RUN/docker/local-server-final.log" 2>&1 || true
docker logs "trtllm-mx-${LOCAL_MX_PORT}-redis" \
  > "$RUN/docker/local-redis-final.log" 2>&1 || true

docker rm -f \
  "trtllm-mx-${LOCAL_MX_PORT}-server" \
  "trtllm-mx-${LOCAL_MX_PORT}-redis" || true
docker network rm "trtllm-mx-${LOCAL_MX_PORT}" || true
```

Do not remove shared model checkpoints, persistent repository clones, or artifacts from unrelated users. Release the
cluster allocation only after `report.md` and all required evidence are durable.

## 24. Completion Checklist

- [ ] Exact #15641, #16159, and combined test heads recorded.
- [ ] Immutable model revision and ArtifactIdentity format/scheme/digest recorded.
- [ ] Environment and NIXL preflight passed.
- [ ] Combined wheel built and installed with `[mx]`; base install remains free of the MX dependency.
- [ ] HF baseline generated deterministic token IDs.
- [ ] Every donor rank published nonzero bytes.
- [ ] Full receiver transferred all tensors and used the staged Llama path.
- [ ] Full receiver token IDs exactly matched baseline.
- [ ] Canonical-snapshot receiver without weight shards reported matching ArtifactIdentity, succeeded, and exactly
  matched baseline.
- [ ] Automatic local server created, reused, and recovered compatible containers.
- [ ] Runtime identity, artifact identity, v1 metadata, and unavailable-server controls fell back safely.
- [ ] Logs, outputs, Docker evidence, topology, and package versions archived.
- [ ] Final report classified the run as PASS, FAIL, or BLOCKED with evidence.
