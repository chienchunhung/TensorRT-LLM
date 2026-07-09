<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 20. ModelExpress End-to-End Verification Plan

[< Back to README](README.md)

**Status:** Ready to execute  
**Last Updated:** 2026-07-08  
**Primary implementation under test:**
[NVIDIA/TensorRT-LLM#15641](https://github.com/NVIDIA/TensorRT-LLM/pull/15641)  
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
7. Unsupported or incompatible configurations reject P2P transfer before RDMA and fall back safely.

This is a **functional qualification** plan. Collect startup and transfer performance data, but do not fail the core
experiment solely because a first-run latency target is missed.

## 2. Scope and Known Boundaries

### In scope

- One Linux cluster node.
- Two-GPU TP=1 smoke test or four-GPU TP=2 qualification test.
- PyTorch backend.
- `LlamaForCausalLM`, transform protocol version 1.
- ModelExpress client and server version `0.4.1`.
- Explicitly managed MX server first, then TRT-LLM automatic local-server launch.
- Deterministic inference, transfer evidence, disk-isolation proof, identity mismatch, and server-failure fallback.

### Out of scope

- Cross-node fabric qualification. A single-node pass does not prove cross-node IB/RoCE behavior.
- Non-Llama post-transform reception.
- A separately loaded draft model or target-plus-draft transfer.
- MX+GMS composition.
- Production performance sign-off.
- Exact checkpoint-content isolation. `SourceIdentity` does not yet include `ArtifactIdentity`; use one immutable,
  recorded checkpoint artifact throughout this experiment.

## 3. Required Result

The executor must finish with one of these explicit outcomes:

- **PASS:** All core gates G0-G5 pass.
- **FAIL:** A core gate produces reproducible evidence of an implementation defect.
- **BLOCKED:** The environment cannot satisfy a prerequisite such as Docker, NIXL, model access, or a compatible GPU
  allocation. Environment blockers must not be reported as product failures.

Do not report PASS from logs alone. PASS requires exact token-ID equality and the no-weight-shards receiver proof.

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
2. Use a detached worktree at the current PR head. Do not modify or force-push the PR branch while testing.
3. Record every resolved input and exact command before running the experiment.
4. Keep the donor process alive until every receiver test finishes. MX source tensors and NIXL registrations are
   owned by the live donor process.
5. Stop at the first failed core gate. Preserve logs and diagnose before continuing.
6. Do not change source code to make a test pass unless the user separately authorizes a fix.
7. Treat donor disk fallback as expected when no source exists. Treat receiver disk fallback as a failure in the
   positive P2P tests.
8. Run positive transfer tests with an immutable checkpoint. Record its revision or shard hashes because current
   `SourceIdentity` does not bind identity to checkpoint contents.
9. Keep baseline, donor, and receiver model settings identical except where a negative test intentionally changes one
   field.
10. Archive evidence before cleanup.

## 6. Inputs to Resolve

The executor must resolve and record these values in `$RUN/manifest.txt`:

| Variable | Required value |
|:--|:--|
| `PR_NUMBER` | `15641`, unless the user provides a successor PR |
| `PR_HEAD` | Exact fetched commit SHA; never assume a stale SHA from this document |
| `MODEL` | Immutable local path to a `LlamaForCausalLM` checkpoint |
| `MODEL_REVISION` | Hub commit, LFS object IDs, or a SHA-256 shard manifest |
| `TP` | `1` for two GPUs or `2` for four GPUs |
| `DONOR_GPUS` | `0` or `0,1` |
| `RECEIVER_GPUS` | `1` or `2,3` |
| `ARCH` | `80-real`, `90-real`, or `100-real`, matching the node |
| `MX_PORT` | Unused host port, recommended `18001` |
| `LOCAL_MX_PORT` | Different unused host port, recommended `18002` |

Recommended first model: TinyLlama or another small unquantized Llama checkpoint for setup. After the smoke test,
repeat G1-G5 with the representative Llama model and quantization configuration intended for support.

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
- The model path is readable.
- The node has enough HBM for donor and receiver concurrently.
- The same execution environment can reach `127.0.0.1:<MX_PORT>`.

If Docker is unavailable, the core transfer could be tested against an externally provisioned MX server, but the
local-launch gate G5 will remain BLOCKED.

**G0 pass criterion:** GPU, Docker, model, storage, and local-network prerequisites are recorded and usable.

## 8. Prepare the Exact PR Source and Wheel

**Goal:** Run the experiment against a reproducible PR commit and verify the optional MX dependency contract.

### Step 8.1: Fetch the current PR head into an isolated worktree

```bash
export REPO="$RUN/repos/TensorRT-LLM"
mkdir -p "$RUN/repos"
git clone https://github.com/NVIDIA/TensorRT-LLM.git "$REPO"
git -C "$REPO" fetch origin "pull/15641/head:refs/remotes/origin/pr/15641"
git -C "$REPO" worktree add --detach "$RUN/worktrees/pr15641" refs/remotes/origin/pr/15641
export SRC="$RUN/worktrees/pr15641"
git -C "$SRC" rev-parse HEAD | tee "$RUN/pr-head.txt"
```

If the persistent cluster clone already exists, reuse it and create only the detached worktree.

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

for name in ("MxClient", "MxLiveWeightLoader", "publish_model_params", "_build_trtllm_identity"):
    assert hasattr(transfer, name), name
assert is_nixl_available(), "NIXL Python bindings are unavailable"
PY
```

**Preparation pass criterion:** The exact PR head builds, `tensorrt_llm[mx]` installs, `pip check` passes,
ModelExpress is exactly `0.4.1`, required transfer symbols resolve, and NIXL is available.

## 9. Record the Immutable Model Artifact

**Goal:** Ensure donor and receiver intentionally use one known checkpoint artifact while ArtifactIdentity is pending.

```bash
test -f "$MODEL/config.json"
python3 - "$MODEL" <<'PY' > "$RUN/model-manifest.sha256"
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
patterns = ("config.json", "*.safetensors.index.json", "*.safetensors", "*.bin")
files = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
for path in files:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    print(digest.hexdigest(), path.relative_to(root))
PY
```

For very large checkpoints, accepted alternatives are a checked-in manifest of LFS object IDs or a trusted immutable
Hub revision. Record the exact mechanism in the final report.

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

**Goal:** Prove that a positive receiver result cannot be explained by silent disk fallback.

Create a metadata-only receiver view. Its directory basename must match the donor model basename because MX discovery
normalizes local model paths to that basename.

```bash
export RECEIVER_MODEL="$RUN/receiver-model/$(basename "$MODEL")"
mkdir -p "$RECEIVER_MODEL"
rsync -a \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  "$MODEL/" "$RECEIVER_MODEL/"

test -f "$RECEIVER_MODEL/config.json"
test -z "$(find "$RECEIVER_MODEL" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -print -quit)"
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

If P2P silently falls back, this run should fail because no weight shards are available. A successful run with exact
tokens is therefore the strongest functional evidence in this plan.

**G4 pass criterion:** The receiver has no checkpoint weight files, every rank transfers nonzero bytes, no fallback is
logged, and token IDs exactly match the HF baseline.

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

Do not use same-config/different-checkpoint bytes as a safety control until ArtifactIdentity lands. Current identity
does not make that case safe.

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

## 19. Acceptance Matrix

| Gate | Required evidence | Pass condition |
|:--|:--|:--|
| G0 Environment | GPU, Docker, NIXL, model, network records | All prerequisites usable |
| Build/install | Build log, wheel path, `pip check`, versions | PR wheel and MX 0.4.1 installed; NIXL available |
| G1 Baseline | `baseline.json`, baseline log | Three deterministic outputs produced |
| G2 Donor | Donor log, published rank metadata, `donor.json` | Every rank publishes; tokens equal baseline; donor stays alive |
| G3 Full receiver | Receiver and per-rank MX logs, output JSON | Full P2P, staged Llama path, no fallback, exact tokens |
| G4 No-shards receiver | Metadata-only model listing, logs, output JSON | P2P succeeds with no weight files; exact tokens |
| G5 Local launch | Docker inspect/logs before and after receiver | Create, reuse, restart, and exact receiver success |
| G6 Negative controls | Logs and outputs for N1-N3 | Reject/fallback paths are bounded and correct |
| Performance | Three run JSON files and transfer metrics | Reported without overstating cache-state comparisons |

## 20. Failure Classification

Use these categories in the final report:

| Category | Examples | Required action |
|:--|:--|:--|
| Environment | No Docker socket, image pull denied, no NIXL, insufficient HBM | Mark BLOCKED; preserve preflight evidence |
| Build/package | PR wheel does not build, `[mx]` cannot resolve, wrong MX API | Mark FAIL; include exact resolver/build error |
| Discovery/identity | Donor publishes but receiver cannot select exact source | Mark FAIL; preserve identities, model basenames, rank logs |
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
├── pr-head.txt
├── model-manifest.sha256
├── environment.txt
├── nvidia-topology.txt
├── package-versions.txt
├── pip-check.txt
├── mx_e2e_worker.py
├── artifacts/
├── docker/
│   ├── server-explicit-final.log
│   ├── redis-explicit-final.log
│   └── local-container-inspect.json
├── logs/
│   ├── build-wheel.log
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

- PR and head SHA:
- TRT-LLM wheel version:
- ModelExpress client/server version:
- Model path and immutable revision/manifest:
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

## Performance

- HF load time:
- Receiver load times:
- Median receiver load time:
- Cache state and disk-read evidence:

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

- [ ] Exact PR head and immutable model artifact recorded.
- [ ] Environment and NIXL preflight passed.
- [ ] PR wheel built and installed with `[mx]`.
- [ ] HF baseline generated deterministic token IDs.
- [ ] Every donor rank published nonzero bytes.
- [ ] Full receiver transferred all tensors and used the staged Llama path.
- [ ] Full receiver token IDs exactly matched baseline.
- [ ] Receiver without weight shards succeeded and exactly matched baseline.
- [ ] Automatic local server created, reused, and recovered compatible containers.
- [ ] Identity mismatch and unavailable-server controls fell back safely.
- [ ] Logs, outputs, Docker evidence, topology, and package versions archived.
- [ ] Final report classified the run as PASS, FAIL, or BLOCKED with evidence.
