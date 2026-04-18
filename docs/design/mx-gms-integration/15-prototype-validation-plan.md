# 15. Prototype Validation Plan

[< Back to README](README.md)

**Status:** Phase A complete; Phase B blocked on GMS API alignment  •  **Last Updated:** 2026-04-18
**Scope:** Validation strategy for the [PR #13045 prototype](https://github.com/NVIDIA/TensorRT-LLM/pull/13045) (MX + GMS integration) using the §10/§11 benchmark infrastructure.

> This file is a working plan. Once validation is executed, results will be folded into [§11 Results & Analysis](11-results-analysis.md) and this file can be retired or moved to a "completed plans" archive.

> **Skip to current state:** [Execution Status](#execution-status) (what's done, what's blocked, what's next).

---

## Goals

Quantitatively verify that PR #13045 delivers the projected wins from [§11 Impact Projection](11-results-analysis.md#mxgms-impact-projection):

| Scenario | Baseline (measured) | Target (projected) | Verification test |
|----------|--------------------:|-------------------:|-------------------|
| Cold start, MX 1st on new node | 306s (Qwen 72B S2) | ~75–80s | **B4** |
| Failover activation (GMS shadow) | ~75s (cold restart) | <5s | **B6** |
| Memory overhead of shadow worker | N/A (no shadow today) | ~0 GB additional weights | **B2** |
| Throughput regression | N/A | <2% | **B5** |

And establish go/no-go gates:

- **Bit-exactness** (B1): MX-loaded weights must produce identical outputs to HF-loaded weights — pure correctness gate before any perf measurement.
- **Profile diagnostic** (Phase C): verify projected MX/worker-init overlap actually holds, or quantify the shortfall.

---

## Branch Strategy

PR #13045 lives on `chienchunhung:dynamo-integration-prototype`. The benchmark + profiler infrastructure lives on `dynamo/startup-profiling` (see [§10 Methodology](10-methodology.md)). They must be combined.

**Approach:** rebase the prototype onto current `upstream/main` first (so the integration branch starts from the same base as the §11 v3 baseline), then create a fresh integration branch on top of the rebased prototype and cherry-pick the bench commits. This keeps the two original branches (`dynamo-integration-prototype` and `dynamo/startup-profiling`) untouched as canonical references.

### Branches in fork (`github.com/chienchunhung/TensorRT-LLM`)

| Branch | Purpose | Base | Tip SHA |
|--------|---------|------|---------|
| `dynamo-integration-prototype` | **Original prototype** (PR #13045 source). Untouched. | upstream/main as of 2026-04-14 | `84dfb2aa7` |
| `dynamo/startup-profiling` | **Bench + profiler infrastructure** (§10/§11 source). Untouched. | upstream/main as of 2026-04-17 | `f9771e571` |
| `docs-and-plans` | This design doc. | — | (current) |
| `dynamo/proto-rebased` | **NEW.** Prototype's 2 commits replayed on current `upstream/main`. Zero rebase conflicts. | upstream/main `4a848ccce` | `7bb11db6a` |
| `dynamo/proto-bench-integration-v2` | **NEW.** Working integration branch — `dynamo/proto-rebased` + the 7 bench commits cherry-picked on top. | `dynamo/proto-rebased` | `5e9ee91c8` |

```text
upstream/main (4a848ccce)
└── dynamo/proto-rebased (7bb11db6a)              ← prototype's 2 commits replayed
    │   • [feat] Add MX and GMS integration prototype       (0dcb9f920)
    │   • [feat] Align GMS backend with merged GMS API #7575 (7bb11db6a)
    └── dynamo/proto-bench-integration-v2 (5e9ee91c8)   ← +7 bench cherry-picks
        ├── [feat] Add hierarchical startup profiling and benchmark instrumentation
        ├── [feat] Split HF cache probe and remote download timers
        ├── [feat] Add startup benchmark automation scripts
        ├── [feat] Fix S2 NFS cold benchmark with per-run fresh copy
        ├── [fix]  Use offline mode and local tokenizer for S2/S3 benchmark tiers
        ├── [fix]  Use representative-run approach for benchmark aggregation
        └── [feat] Add failover floor benchmark script (Test 4a)
```

These two new branches are throwaway — discard after validation completes.

### Setup commands (reproducible)

```bash
# 1. Rebase the prototype onto current upstream/main (clean — 0 conflicts)
git fetch upstream main
git fetch fork dynamo-integration-prototype dynamo/startup-profiling
git checkout -b dynamo/proto-rebased fork/dynamo-integration-prototype
git branch --unset-upstream            # Safety: avoid pushing to original prototype branch
git rebase upstream/main

# 2. Build the integration branch by cherry-picking the bench commits
git checkout -b dynamo/proto-bench-integration-v2
git cherry-pick 3c9aebfdd 800ba9751 642dcd05a 667a89be0 4bd024b5e bba6bb505 f9771e571
# (5 conflicts on the foundational bench commit, all in
#  py_executor_creator.py + model_loader.py — see "Conflict Resolutions" below)

# 3. Push to fork for safekeeping (no force, new refs only)
git push --no-verify fork dynamo/proto-rebased dynamo/proto-bench-integration-v2

# 4. No C++ rebuild needed: both new branches share the same upstream/main HEAD
#    as the existing in-container build artifacts (option 1 from the validation plan).
```

### Conflict Resolutions (one-time, captured for the record)

All 5 conflicts landed on the foundational bench commit `3c9aebfdd` (hierarchical startup profiling). Resolution strategy: **keep prototype semantics, add bench timers without changing control flow.**

| File | Region | Resolution |
|------|--------|------------|
| `py_executor_creator.py` | `_construct_checkpoint_loader` call | Kept prototype's new `mx_server_url=llm_args.mx_server_url` arg; wrapped `load_config_and_apply_defaults` in `executor.load_config_and_apply_defaults` timer. |
| `model_loader.py` | Materialize tensors path | `executor.materialize_model_tensors` timer now wraps prototype's `virtual_memory_scope` block; `elif is_meta_init:` retains the `and load_format != LoadFormat.GMS` guard so GMS skips meta→CUDA init. |
| `model_loader.py` | `model.to("cuda")` | Combined: `if load_format != LoadFormat.GMS: with startup_timer("executor.move_model_to_cuda"): model.to("cuda")`. GMS RO path stays a no-op. |
| `model_loader.py` | `LoadFormat.AUTO` weight load | Kept MX-aware `load_weights_kwargs` with `model=` injection and the `mx_p2p_succeeded` short-circuit; weight-mapper init and `_call_load_weights` are now wrapped in `executor.weight_mapper_init.main_weights` and `executor.apply_model_weights.main_weights`, but only inside `if not mx_p2p_succeeded:` (so MX P2P still skips the mapping pipeline). |
| `model_loader.py` | `post_load_weights` block | Added `executor.mx_publish_as_source` timer around prototype's new MX publish call; gated `executor.post_load_weights` on `not gms_ro_done` to preserve prototype's "skip post_load_weights for GMS RO" behavior. |

Net effect: the integration branch produces an exact superset of the prototype's behavior (no semantic changes), with full §10 timer coverage on every code path.

---

## Execution Status

Snapshot of progress as of **2026-04-18**. This section is updated as we work through the plan.

### ✅ Phase A — Branch Integration (DONE)

| Step | Outcome |
|------|---------|
| Rebase `dynamo-integration-prototype` onto current `upstream/main` | **Clean** — 0 conflicts |
| Cherry-pick the 7 bench commits onto rebased prototype | 5 conflicts on foundational profiler commit, all resolved (see [Conflict Resolutions](#conflict-resolutions-one-time-captured-for-the-record)) |
| Push both new branches to fork | Done — `dynamo/proto-rebased`, `dynamo/proto-bench-integration-v2` |
| C++ rebuild | **Not needed** — both branches sit on the same `upstream/main @ 4a848ccce` HEAD as the existing in-container build (option 1 path; see "Smoke verification" below) |

### ✅ Smoke Verification on Integrated Branch (DONE)

Confirmed the integration is healthy and the M1 (baseline AUTO/HF) path is unaffected:

| Check | Result |
|-------|--------|
| `import tensorrt_llm` | ✅ |
| Prototype symbols (`tensorrt_llm._torch.memory.GMSBackend`) | ✅ importable |
| Prototype Pydantic fields (`mx_server_url`, `gms_socket_path`, `gms_mode`, `gms_tag`) | ✅ present, render correct defaults |
| Bench symbols (`tensorrt_llm.llmapi.startup_profiler.startup_timer`, `get_startup_profiler`) | ✅ |
| `trtllm-serve --help` | ✅ |
| `trtllm-serve` boots Qwen2.5-7B-Instruct TP=1, `LoadFormat.AUTO` (no MX, no GMS) | ✅ ready in ~45s on warm NFS |
| Inference correctness (`The capital of France is` → ` Paris…`) | ✅ |
| Bench profiler captures full hierarchy on integrated branch | ✅ 67.6s server / 36.3s worker, all expected timers populated (`executor.load_model_weights`, `executor.warmup.*`, `executor.recreate_py_executor_instance`, etc.) |

The MX-aware `if not mx_p2p_succeeded:` guard in `model_loader.py` correctly takes the standard HF path when no MX server is configured, and `executor.apply_model_weights.main_weights: 2.288s` confirms `_call_load_weights` runs as expected.

### ⚠️ Phase B — Blocked on GMS API Alignment

The Phase B verification tests require either MX (`nvidia-modelexpress`) or GMS (`gpu-memory-service`) to be installed and runnable.

#### GMS install: package OK, prototype API mismatch found

| Step | Outcome |
|------|---------|
| Install GMS (`pip install -e dynamo/lib/gpu_memory_service`) | ✅ `gpu-memory-service 0.9.0` installed, daemon binary works, package importable |
| Prototype's `GMSBackend.connect()` against installed GMS | ❌ **API mismatch — blocker** |

Specifically, the prototype's `tensorrt_llm/_torch/memory/gpu_memory_backend.py` (commit `7bb11db6a`, message: *"Align GMS backend with merged GMS library API (PR #7575)"*) calls a high-level convenience API:

```python
# Prototype expects:
from gpu_memory_service import client as gms_client
self._client = gms_client.connect(self._socket_path, mode="rw")
self._is_rw = not self._client.has_committed_weights(self._tag)
gms_client.get_mem_pool(self._client)                                # RW path
gms_client.materialize_module_from_gms(self._client, model)          # RO path
```

But the actual merged API in `ai-dynamo/dynamo` post-PR #7575 exposes a **class-based low-level API** plus a **monkey-patch integration**:

```python
# What's actually merged (verified Apr 18):
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.allocator import gms_use_mem_pool   # context manager
from gpu_memory_service.client.torch.module import materialize_module_from_gms

mgr = GMSClientMemoryManager(socket_path, device=N)
mgr.connect(lock_type=RequestedLockType.RW)  # explicit lock_type, no auto-mode wrapper
# Allocate via context manager:
with gms_use_mem_pool(tag, device):
    ...
materialize_module_from_gms(mgr, model, device_index=N)  # requires keyword arg
```

There is also an **official TRT-LLM integration shim** at `gpu_memory_service.integrations.trtllm.setup_gms()` (added by PR #7575 itself) that monkey-patches TRT-LLM's `ModelLoader` from outside — this is the supported way to enable GMS in TRT-LLM today. The prototype's `GMSBackend` is a parallel re-implementation of that integration that pre-dates the merge.

Diagnostic facts:

- PR #7575 ("feat: add TRT-LLM sleep/wake integration with GMS") merged into `ai-dynamo/dynamo` main on **2026-04-15** (commit `d96a2cf1`).
- Zero commits to `lib/gpu_memory_service/client/` between #7575 and our checkout (verified 2026-04-18). The installed API is exactly the post-#7575 API.
- Prototype's "Align GMS backend…" commit is dated **2026-04-16** (one day after #7575 merged) but was clearly written against an earlier, un-merged iteration of the API.

Impact: any attempt to launch with `--load-format gms` will throw `AttributeError: module 'gpu_memory_service.client' has no attribute 'connect'`. M2 (GMS-only), M4 (MX+GMS), B2 (shadow memory), and B6 (failover E2E) are all blocked until this is resolved.

Available paths forward (decision pending):

1. **Patch the prototype's `GMSBackend`** to use the current `GMSClientMemoryManager` + `gms_use_mem_pool` context-manager API. Estimated: a few hours of focused refactor on the `dynamo/proto-bench-integration-v2` branch (or directly on the prototype). We own the code and it's an explicit prototype.
2. **Adopt the official `setup_gms()` integration** from PR #7575 and remove the prototype's custom `GMSBackend`. Cleaner long-term, but the prototype's "two-axis MX+GMS" composition (e.g., GMS RW path that also publishes for MX P2P readers) would need to be reworked on top of the monkey-patch model.
3. **File the API gap upstream** against PR #13045 / the prototype owner; pause GMS validation until the prototype catches up.

#### MX install: not feasible on this node

| Requirement | Status on current node |
|-------------|------------------------|
| Rust 1.90+ (cargo) | ❌ not installed |
| `protoc` | ✅ 3.21.12 |
| Docker (for Redis metadata backend) | ❌ not installed |
| `redis-server` | ❌ not installed |

`nvidia-modelexpress` lives at [github.com/ai-dynamo/modelexpress](https://github.com/ai-dynamo/modelexpress) — Rust server + Python client; build from source via `cargo build` then run with `MX_METADATA_BACKEND=redis cargo run --bin modelexpress-server`. Setting this up would require either provisioning Rust + Docker on the current node or migrating to a node that already has them. Estimated: 30–60 min of one-time setup before any MX test can start.

Also relevant: per [modelexpress's known issues](https://github.com/ai-dynamo/modelexpress#known-issues), MLA-architecture models (DeepSeek-V2/V3, Kimi K2) are blocked from MX P2P transfer and silently fall back to disk. Our Qwen 72B (no MLA) and DeepSeek-R1-Distill-Llama-70B (Llama architecture, not MLA) are both safe.

### 🔭 Recommended Next Step

Resolve the GMS API mismatch first (path 1 or 2 above) before investing in MX node prep. GMS-only validation (B2 + B6) gives an independent, high-value go/no-go signal on the prototype's most novel claim (zero-copy shadow + sub-5s failover) and exercises ~half the prototype's new code with the lighter dependency footprint. MX validation (B1 + B3 + B4) can follow once a node with Rust + Docker is available.

In parallel, M1 baseline regression on `dynamo/proto-bench-integration-v2` (i.e., re-run a single §11 v3 config — Qwen 7B TP=1 S2 — and confirm startup-time delta is within run-to-run noise) is a cheap (~15 min) sanity check that the prototype's edits to the AUTO/HF path don't subtly regress the baseline. The smoke test above is consistent with no regression but it ran on warm NFS, not the §11 cold protocol.

---

## Reuse §11 Baselines (Don't Re-Measure)

The PR rebased onto `upstream/main @ 4a848ccce`, which is exactly the same codebase used for the v3 dataset in §11. **There is no need to re-run baseline measurements** — the §11 v3 numbers are the "before" for our comparison.

Reusable baselines (from [§11 Part 1 v3 results](11-results-analysis.md#part-1--model-size-scaling-s2-nfs-cold--production-baseline)):

| Config | Qwen 7B TP=1 | Qwen 72B TP=8 | DS 7B TP=1 | DS 70B TP=8 |
|--------|------------:|--------------:|------------:|--------------:|
| **S2 (NFS cold) total** | 77.4s | **306.3s** | 94.4s | **389.8s** |
| &nbsp;&nbsp;checkpoint prefetch | 16.9 | 233.2 | 35.7 | 318.4 |
| &nbsp;&nbsp;warmup (1st + 2nd) | 39.1 | 42.7 | 37.9 | 40.9 |
| **S3 (warm) total** | — | **74.6s** | — | **77.7s** |
| &nbsp;&nbsp;checkpoint prefetch | — | 3.5 | — | 6.0 |

Steady-state inference floor (from [§11 Part 5 / Test 4a](11-results-analysis.md#part-5--failover-latency-floor-test-4a)):

| Config | TTFT median | E2E median |
|--------|------------:|-----------:|
| Qwen 72B TP=8 | 63 ms | 108 ms |
| DS 70B TP=8 | 56 ms | 98 ms |

These are the comparison anchors. Verification tests below produce numbers compared directly against these.

---

## Test Matrix

The PR's two-axis design gives 4 configurable modes:

| Mode | `checkpoint_format` | `LoadFormat` | Use case |
|------|--------------------:|-------------:|----------|
| **M1: Baseline** | HF | AUTO | Current behavior — baseline (= §11 v3) |
| **M2: GMS-only** | HF | GMS | Within-node weight sharing + crash resilience |
| **M3: MX-only** | MX | AUTO | Cross-node P2P weight transfer |
| **M4: MX + GMS** | MX | GMS | Full vision — cross-node P2P + within-node sharing |

---

## Verification Tests (Priority-Ordered)

Tests are ordered so failures stop us early before wasting time on later tests.

### B1 — Bit-Exactness (Correctness Gate)

**Why first:** if MX-loaded weights diverge from HF-loaded weights, all subsequent perf measurements are meaningless.

**Protocol:**
1. Start two servers: M1 (`--checkpoint-format HF`) and M3 (`--checkpoint-format MX`)
2. Send identical prompts with greedy decoding (`temperature=0`)
3. Compare output token IDs

**Pass criterion:** Identical token IDs across all prompts.

**Failure mode:** MX is not applying post-load transforms (quant, weight-mapper, etc.) identically to HF. This is a correctness bug in the MX path, not a perf issue.

**Cost:** ~20 min (one Qwen 72B TP=8 server pair).

---

### B2 — GMS Shadow Memory Overhead

**Why second:** zero-copy shadow import is the foundational claim of GMS. If shadow adds full weight-bytes of memory, everything else GMS-related is suspect.

**Protocol:**
1. Start GMS daemon: `gpu-memory-service --socket /tmp/gms-0.sock &`
2. Start primary (`--load-format GMS --gms-mode rw`), wait until ready
3. Record GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
4. Start shadow (`--load-format GMS --gms-mode ro`) on same node
5. Re-record GPU memory after shadow ready
6. Time the GMS RO import phase from the profiler (`executor.load_model_weights` should be near-instant)

**Pass criteria:**
- Shadow adds **~0 GB** of weight memory (some bookkeeping in 10s of MB is OK)
- GMS RO import latency: **<500ms** per [§04](04-implementation-plan.md) success criteria; [§11 Impact Projection](11-results-analysis.md#mxgms-impact-projection) says ~100ms

**Failure mode:** if shadow adds anything close to "weights / TP" bytes (e.g., ~14 GB per rank for Qwen 72B TP=8), the RO zero-copy import is broken.

**Cost:** ~30 min (Qwen 72B TP=8, three runs, plus daemon setup).

---

### B3 — P2P Transfer Throughput

**Why:** anchors whether any shortfall in B4 (cold-start) comes from MX itself or from surrounding TRT-LLM glue. Sub-measurement of B4.

**Protocol:**
- Same node (NVLink): donor and receiver on different ranks of same node
- Cross-node (whatever fabric available — IB HDR, RoCE 100G)
- Measure: weight bytes transferred ÷ MX `_try_p2p_transfer()` duration
- If MX SDK emits transfer telemetry, use that as ground truth

**Pass criteria** (from [§10 Test 2](10-methodology.md#test-2-p2p-transfer-throughput--not-yet-executed)):
- Same node (NVLink): **>50 GB/s**
- Cross-node IB HDR: **>20 GB/s**
- Cross-node RoCE 100G: **>10 GB/s**

**Cost:** ~10 min (sub-test of B4; reuses same servers).

---

### B4 — Cold-Start Headline (the demo)

**Why:** this is the headline number that justifies the whole MX integration. Compares M3 / M4 against §11 baseline.

**Protocol:**
1. Set up MX donor: start an M1 instance (`--checkpoint-format HF`) and let it fully load — this IS the §11 baseline, no separate measurement needed
2. Start MX server: `modelexpress-server --port 8001 &`
3. Profile M3 receiver:
   ```bash
   TRTLLM_PROFILE_STARTUP=1 \
   TRTLLM_STARTUP_PROFILE_OUTPUT=/tmp/mx_b2_run1.json \
   trtllm-serve Qwen/Qwen2.5-72B-Instruct \
       --backend pytorch --tensor_parallel_size 8 \
       --max_batch_size 4 --max_num_tokens 1024 --max_seq_len 4096 \
       --checkpoint-format MX --mx-server-url http://localhost:8001 \
       --port 8002
   # Then drive with benchmark_serving.py --save-startup-metrics (per §10)
   ```
4. 3 runs per config, median-representative protocol (matches §11)
5. Repeat for M4 (`--checkpoint-format MX --load-format GMS`)

**Configs:** B2 (Qwen 72B TP=8) and B4 (DS 70B TP=8). Skip small models — prefetch is already cheap there.

**Pass criteria:**

| Mode | Qwen 72B S2 baseline | M3 target | M4 target |
|------|---------------------:|----------:|----------:|
| Cold start total | 306.3s | **~75–80s** | **~64s** |
| `executor.checkpoint_prefetch` | 233.2s | **~10–15s** | **~0.1s** (zero-copy) |
| CPU memory peak | ~9× model size | ~1× model size | ~1× model size |

**Cost:** ~1 hr (4 configs × 3 runs each, including server setup overhead).

---

### B5 — Throughput Regression

**Why:** ensures the loaded model is bit-identical regardless of how it got into GPU memory. Cheap to run, high-value for sign-off.

**Protocol:**
1. Run steady-state throughput benchmark (sharegpt or random dataset) on M1
2. Run same benchmark on M3 (and M4 if relevant)
3. Compare tokens/sec

**Pass criterion:** **<2% throughput delta** vs M1.

**Failure mode:** weights from MX path haven't gone through the same post-load transforms — this is a correctness bug surfacing as a perf gap. Shouldn't happen if B1 passes, but worth verifying.

**Cost:** ~20 min.

---

### B6 — Failover Latency E2E

**Why:** full validation of the failover story. Only meaningful if Phase 2 / shadow is in scope for the prototype.

**Protocol** (from [§10 Test 4](10-methodology.md#test-4-shadow-failover-latency--partially-executed)):
1. Start primary + GMS shadow on same node
2. Send warmup requests to primary (populates compile cache if implemented)
3. `kill -9` primary
4. Orchestrator routes to shadow; measure time to first response

**Pass criterion:** **<5s** from primary kill → shadow responds. Note: this assumes warm compile cache. Without compile cache, warmup adds ~43s ([§11 Insight #6](11-results-analysis.md#6-warmup-overhead-regression-from-pr-12407-new-in-v3)) and the budget is blown.

**Cost:** ~30 min.

---

### Skipped Tests (Don't Need Re-Measurement)

| Test | Why skipped |
|------|-------------|
| M1 baseline cold-start re-run | Reuse [§11 v3 numbers](11-results-analysis.md#part-1--model-size-scaling-s2-nfs-cold--production-baseline) — same codebase |
| Test 4b (cold-restart failover) | Same as M1 baseline — already in §11 |
| S1 (remote cold) | Already deprioritized in §11 |
| Small-model S2/S3 with MX | Prefetch is already cheap; less compelling demo |
| vLLM comparison ([§10 Test 6](10-methodology.md#test-6-vllm-comparison--not-yet-executed)) | Only meaningful once B4 is stable; follow-up work |

---

## Critical Diagnostic: MX/Worker-Init Overlap

The §11 projection assumes MX P2P transfer **overlaps with the ~21s worker init** (Python imports, CUDA ctx, NCCL setup) — similar to how S1 hides worker init behind the HF download (see [§11 Insight #1](11-results-analysis.md#1-cold-nfs-io-dominates-production-cold-start) and the [Worker Init Investigation](11-results-analysis.md#worker-init-investigation-results)).

If this overlap doesn't actually happen — e.g., MX waits for worker init to complete before starting transfer — we lose ~20s of the projected win and B4 will come in closer to **~95s instead of ~75s**.

**This is not a prototype bug per se**, but it is the most likely explanation for any shortfall vs projection.

### Verification

After each B4 run, check the hierarchical profile:

```python
import json
p = json.load(open('mx_b2_run1.json'))
records = p['attached_profiles']['executor_workers']['ranks'][0]['records']
for rec in records:
    name = rec['name']
    if 'checkpoint_prefetch' in name or 'worker.initialize' in name or 'load_model_weights' in name:
        print(f"{name}: starts {rec['start_offset_s']:.1f}s, dur {rec['duration_s']:.1f}s")
```

**Expected (overlap working):**
```
executor_worker.initialize: starts 0.0s, dur 21.5s   ← worker init happens
executor.load_model_weights: starts 5.0s, dur 15.0s  ← MX transfer overlaps
executor.checkpoint_prefetch: starts 5.5s, dur 10.0s ← P2P starts during init
```

**Shortfall (no overlap):**
```
executor_worker.initialize: starts 0.0s, dur 21.5s   ← worker init first
executor.load_model_weights: starts 21.5s, dur 15.0s ← MX transfer waits
executor.checkpoint_prefetch: starts 22.0s, dur 10.0s ← lost 20s of overlap
```

If shortfall is observed, document it explicitly in the §11 update and treat it as a follow-up optimization (separate from the prototype itself).

---

## Service Setup Reference

### GMS daemon (M2, M4)

```bash
gpu-memory-service --socket /tmp/gms-0.sock &
```

### MX server + donor (M3, M4)

```bash
# 1. MX server
modelexpress-server --port 8001 &

# 2. Donor instance — this IS the §11 v3 baseline
trtllm-serve Qwen/Qwen2.5-72B-Instruct \
    --backend pytorch --tensor_parallel_size 8 \
    --max_batch_size 4 --max_num_tokens 1024 --max_seq_len 4096 \
    --checkpoint-format HF \
    --port 8001 &
# Wait for donor ready before launching M3/M4 receivers
```

### M3 / M4 receiver (the measured instance)

```bash
TRTLLM_PROFILE_STARTUP=1 \
TRTLLM_STARTUP_PROFILE_OUTPUT=/tmp/mx_b2_run<N>.json \
trtllm-serve Qwen/Qwen2.5-72B-Instruct \
    --backend pytorch --tensor_parallel_size 8 \
    --max_batch_size 4 --max_num_tokens 1024 --max_seq_len 4096 \
    --checkpoint-format MX --mx-server-url http://localhost:8001 \
    [--load-format GMS --gms-socket-path /tmp/gms-0.sock --gms-mode auto] \
    --port 8002 &
```

---

## Benchmark Script Changes

`run_startup_bench.sh` needs ~5–10 lines added to forward the new flags:

```bash
# In argument parsing:
        --checkpoint-format) CHECKPOINT_FORMAT="$2"; shift 2 ;;
        --load-format)       LOAD_FORMAT="$2";       shift 2 ;;
        --mx-server-url)     MX_SERVER_URL="$2";     shift 2 ;;
        --gms-socket-path)   GMS_SOCKET_PATH="$2";   shift 2 ;;
        --gms-mode)          GMS_MODE="$2";          shift 2 ;;

# In the trtllm-serve invocation:
        ${CHECKPOINT_FORMAT:+--checkpoint-format $CHECKPOINT_FORMAT} \
        ${LOAD_FORMAT:+--load-format $LOAD_FORMAT} \
        ${MX_SERVER_URL:+--mx-server-url $MX_SERVER_URL} \
        ${GMS_SOCKET_PATH:+--gms-socket-path $GMS_SOCKET_PATH} \
        ${GMS_MODE:+--gms-mode $GMS_MODE} \
```

`run_failover_floor_bench.sh` works as-is — it measures steady-state response time, which is independent of how weights were loaded.

---

## Execution Plan & Time Budget

| Phase | Duration | Output |
|-------|---------:|--------|
| **Phase A**: Branch integration + rebuild | ~30 min | Working `dynamo/proto-bench-integration` branch |
| **B1**: Bit-exactness | ~20 min | Pass/fail correctness gate |
| **B2**: GMS shadow memory overhead | ~30 min | Pass/fail GMS zero-copy gate |
| **B3 + B4**: Cold-start headline + P2P throughput | ~1 hr | The demo numbers + diagnostic profile data |
| **B5**: Throughput regression | ~20 min | <2% sign-off measurement |
| **B6**: Failover E2E | ~30 min | <5s validation (if Phase 2 is in scope) |
| **Total** | **~3.5 hr** | Full validation dataset |

---

## Documentation Outputs (Post-Validation)

After execution, fold results back into the design doc:

1. **Add to [§11 Results & Analysis](11-results-analysis.md):**
   - **Part 6: Prototype Validation** — measured M2/M3/M4 numbers in standard format
   - **Part 7: Bit-exactness verification** — confirms M3 = M1 outputs

2. **Update [§11 Impact Projection](11-results-analysis.md#mxgms-impact-projection):**
   - Replace projected scenarios 3–7 with measured numbers from M2/M3/M4
   - Color-code which lines are now measured vs still projected (e.g., bold for measured)
   - Note any shortfall vs projection with link to the prefetch-overlap diagnostic

3. **Update [§10 Methodology](10-methodology.md):**
   - Mark Test 1 MX/GMS rows as Completed with link to Part 6
   - Mark Test 2 (P2P throughput) as Completed
   - Mark Test 3 (memory efficiency) as Completed
   - Mark Test 4 as Completed (or partially, if only B6 runs)
   - Mark Test 5 (throughput regression) as Completed

4. **This file:** retire or move to a `completed-plans/` archive once Section 11 is updated.

---

## Recommendation

**Start with Phase A + B1.** It's the smallest commitment that gives a meaningful go/no-go signal:

- Phase A (~30 min) confirms branches integrate cleanly + build is good
- B1 (~20 min) is a pure correctness test
- If B1 fails → stop, fix prototype, no perf work wasted
- If B1 passes → B4 (the headline) becomes a 1-hour demo with high impact

After B1 passes, B4 + B3 are the highest-impact next steps; B2 and B5 are validation/sign-off; B6 depends on Phase 2 scope.
