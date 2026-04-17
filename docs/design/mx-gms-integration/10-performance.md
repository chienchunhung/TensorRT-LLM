# 10. Performance Expectations and Benchmark Plan

[< Back to Overview](README.md)

**Last Updated:** 2026-04-17

## Target Metrics

| Scenario | Baseline | Target | Improvement |
|:---------|:---------|:-------|:-----------|
| Cold-start (DeepSeek-V3, 681GB) | 5-10 min | 15-30s | **10-20x** |
| Replica scale-up (Llama-70B) | 2-3 min | 5-10s | **12-36x** |
| Shadow failover time | Cold-start (50-114s) | < 5s | **10-23x** |
| Shadow GPU memory overhead | N/A (no shadow today) | Weights only (no KV cache) | Zero-copy import |
| Multi-node scale-out (N replicas) | N x load time | ~constant | **Near-constant** |
| P2P transfer throughput | N/A | > 20 GB/s (NVLink) | — |
| GMS import latency | N/A | < 500ms | — |
| Throughput regression | — | < 2% | Negligible |

## Benchmark Scenarios

### Test 1: Cold-Start Latency (Baseline Profiling) — Completed

**What:** Time from process start to first successful inference under the standard HF weight-loading path (no MX, no GMS). Establishes the baseline startup breakdown across model sizes, storage tiers, autotuner settings, and serving configurations.

**Status:** Completed (2026-04-17). 14 configurations × 3 runs = **42 profiles** on the current rebased codebase (v3). See [Benchmark Results](#benchmark-results) below. Earlier v2 dataset (pre-PR #12407, different node) is preserved as reference.

**Configurations tested:**
| Config | `checkpoint_format` | `load_format` | Status |
|:-------|:-------------------|:-------------|:-------|
| Baseline (HF/AUTO) | `HF` (default) | `AUTO` (default) | **Completed** — see v2 results |
| MX (2nd replica) | `MX` | `AUTO` | Not yet tested |
| GMS (2nd worker) | `HF` | `GMS` | Not yet tested |
| MX+GMS (2nd replica+worker) | `MX` | `GMS` | Not yet tested |

### Test 2: P2P Transfer Throughput — Not Yet Executed

**What:** GB/s during MX weight transfer. Requires MX integration to be implemented.

**Configurations:**
- Same node (NVLink): expect > 50 GB/s
- Cross-node (InfiniBand HDR): expect > 20 GB/s
- Cross-node (RoCE 100G): expect > 10 GB/s

### Test 3: Shadow Failover Memory Overhead — Not Yet Executed

**What:** Verify that a GMS shadow worker (RO import, no KV cache) adds minimal GPU memory overhead alongside an active primary. Requires GMS integration to be implemented.

**Validation:**
```
Primary alone: memory_primary = weights (1/TP) + kv_cache + overhead
Primary + shadow: memory_total ≈ weights (1/TP, shared via GMS) + kv_cache + overhead
                                  # Shadow imports same physical memory — near-zero additional cost
```

> **Note:** For large models (70B+ with TP=8), a single active instance already consumes most GPU HBM. The shadow worker holds only weight references (zero-copy RO import) without KV cache, so the additional memory is negligible. Testing with N=4 active workers sharing weights is unrealistic for large models — the realistic scenario is 1 active + 1 shadow. Multi-instance sharing (N>2) applies only to small models or multi-LoRA deployments where multiple instances with independent KV caches fit on a single GPU.

### Test 4: Shadow Failover Latency — Not Yet Executed

**What:** Time from primary crash to shadow serving first request. Requires GMS + executor failover integration.

**Steps:**
1. Start primary + shadow with GMS
2. Send warmup requests to primary (this also populates compile cache)
3. Kill primary (`kill -9`)
4. Measure time until shadow returns first response
5. **Target:** < 5s

**Critical dependency:** The <5s target assumes compile cache (disk or GMS-backed) is warm. Without compile cache, warmup adds ~16s (autotuner ~12s + CUDA graphs ~4s), making failover ~17-19s. Since primary and shadow are co-located on the same node and share the filesystem, disk-based compile cache is sufficient for Phase 2. See [Compile Cache: Closing the Warmup Gap](06-executor-failover.md#compile-cache-closing-the-warmup-gap) for the tiered cache design.

### Test 5: Throughput Regression — Not Yet Executed

**What:** Steady-state throughput with MX/GMS-loaded weights vs. standard.

**Validation:** < 2% regression. MX/GMS affect startup only; the loaded model should be identical.

### Test 6: vLLM Comparison — Not Yet Executed

**What:** Compare TRT-LLM `--checkpoint-format mx` against vLLM `--load-format mx`.

**Metrics:**
- Cold-start latency (2nd replica)
- P2P transfer throughput
- **Target:** Within 20% of vLLM

---

## Test Matrix (Test 1: Baseline Profiling)

See [startup-benchmark-plan-v2.md](startup-benchmark-plan-v2.md) for the full test plan document, including S2 methodology details.

### Model Matrix

| ID | Series | Variant | HF Repo | Approx Size | TP |
|----|--------|---------|---------|-------------|-----|
| B1 | Qwen | Small | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | 1 |
| B2 | Qwen | Large | `Qwen/Qwen2.5-72B-Instruct` | ~145 GB | 8 |
| B3 | DeepSeek | Small | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14 GB | 1 |
| B4 | DeepSeek | Large | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | ~131 GB | 8 |

Default serving config: `max_batch_size=4, max_num_tokens=1024, max_seq_len=4096`.

### Storage Tier Matrix

| Tier | ID | What it measures | Production analog | Setup |
|------|----|-----------------|-------------------|-------|
| Remote cold | S1 | Full HF download over network + model load | Dev/experimentation, or first-time use without pre-staged model | `HF_HOME=/tmp` (tmpfs); every run downloads from scratch |
| **NFS cold** | **S2** | **NFS file read (no page cache) + model load** | **First cold start on a node with model pre-staged on NFS** | Model files pre-copied to a **fresh NFS directory per run** (new inodes) |
| Local warm | S3 | Page-cache-warm file read + model load | Second instance on same node, rapid restart, or scale-up | Model files on NFS, page cache hot from a prior run |
| Local NVMe cold | S4 | Local SSD read (no page cache, no network) + model load | Model pre-staged to node-local NVMe (e.g., by DaemonSet or init container) | Fresh NVMe directory per run. *Planned — not yet executed* |

**Default tier: S2 (NFS cold)** — the most realistic production cold-start scenario. Models are typically pre-staged on shared storage, not downloaded from HF Hub at serve time.

### Scenario Coverage and Gaps

All three tiers exercise valid code paths and represent real operational contexts:

| Tier | When it happens | Frequency in production |
|------|----------------|------------------------|
| S1 | Model not pre-staged; `trtllm-serve` downloads from HF Hub | Rare (dev/experimentation) |
| S2 | Model on NFS, first access on node (cold page cache) | Common (fresh node, reboot, idle eviction) |
| S3 | Model on NFS, page cache warm from prior load | Common (2nd instance, restart, scale-up) |

**Not yet covered:**

| Scenario | Expected behavior | Priority |
|----------|------------------|----------|
| **Model on local NVMe SSD (cold)** | Prefetch at NVMe sequential read speed (~3–7 GB/s); ~20–40s for 145GB. Falls between S2 (65–99s cold NFS) and S3 (3–5s warm cache). Common in production where models are pre-copied to node-local storage. | Medium — would provide a useful data point between S2 and S3 |
| **Multi-node TP** (workers on different nodes) | Each node's workers prefetch independently from NFS; no shared page cache across nodes. Per-node I/O = S2 or S3 depending on local cache state. | Low — current benchmarks are single-node TP=8 |

### Statistical Protocol

- Each configuration runs **3 times**.
- Uses the **representative-run** approach: the run whose total startup is the median is selected, and all per-component metrics are reported from that single run so components sum consistently to the total. Min/max across all runs are reported for range context.
- Automate via `run_startup_bench.sh` (single config) and `run_startup_bench_all.sh` (full matrix).
- Post-process with `aggregate_startup_results.py` for representative-run extraction.

### Part 1: Model Size Scaling (S1 remote cold, default config) — Completed

| Test ID | Model | Tier | TP | Runs | Status |
|---------|-------|------|----|------|--------|
| B1-S1 | Qwen 7B | S1 | 1 | 3 | **Done** |
| B2-S1 | Qwen 72B | S1 | 8 | 3 | **Done** |
| B3-S1 | DeepSeek 7B | S1 | 1 | 3 | **Done** |
| B4-S1 | DeepSeek 70B | S1 | 8 | 3 | **Done** |

### Part 2: Storage Tier Comparison (large models only) — Completed

| Test ID | Model | Tier | TP | Runs | Status |
|---------|-------|------|----|------|--------|
| B2-S2 | Qwen 72B | S2 | 8 | 3 | **Done** |
| B2-S3 | Qwen 72B | S3 | 8 | 3 | **Done** |
| B4-S2 | DeepSeek 70B | S2 | 8 | 3 | **Done** |
| B4-S3 | DeepSeek 70B | S3 | 8 | 3 | **Done** |

### Part 3: Autotuner Impact (large models, S2/S3) — Completed

Ran on S2 and S3 tiers (instead of S1 as originally planned) to isolate warmup behavior from download variability.

| Test ID | Model | Tier | TP | Autotuner | Runs | Status |
|---------|-------|------|----|-----------|------|--------|
| B2-S2-C | Qwen 72B | S2 | 8 | OFF | 3 | **Done** |
| B2-S3-C | Qwen 72B | S3 | 8 | OFF | 3 | **Done** |
| B4-S2-C | DeepSeek 70B | S2 | 8 | OFF | 3 | **Done** |
| B4-S3-C | DeepSeek 70B | S3 | 8 | OFF | 3 | **Done** |

Compare against B2-S2/S3 and B4-S2/S3 (autotuner ON by default from Part 2).

### Part 4: Serving Config Sensitivity (large models, S3) — Completed

Ran on S3 (warm cache) to isolate serving config impact from I/O variability.

| Test ID | Model | Tier | TP | Config | Runs | Status |
|---------|-------|------|----|--------|------|--------|
| B2-S3-D1 | Qwen 72B | S3 | 8 | bs=64, nt=8192 | 3 | **Done** |
| B4-S3-D1 | DeepSeek 70B | S3 | 8 | bs=64, nt=8192 | 3 | **Done** |
| B2-S3-D2 | Qwen 72B | S3 | 8 | max_seq_len=16384 | 3 | **Done** |
| B4-S3-D2 | DeepSeek 70B | S3 | 8 | max_seq_len=16384 | 3 | **Done** |

Compare against B2-S3 and B4-S3 (default bs=4, nt=1024, max_seq_len=4096). `nt` = `max_num_tokens`.

### Summary

**Completed:** 21 configurations × 3 runs = **62 benchmark profiles** (Parts 1–4).

**Not yet run (requires MX/GMS implementation):**
- Tests 2–6: P2P throughput, memory efficiency, shadow failover, throughput regression, vLLM comparison

---

## MX+GMS Impact Projection

Scenario-based projection using **v3 measured baselines** for Qwen 72B (TP=8) on the current rebased codebase. The "first pays upfront, rest benefit" property of MX and GMS is reflected explicitly.

| Scenario | Worker Init | Weight Load | Warmup | Total Startup | Notes |
|:---------|:------------|:------------|:-------|:--------------|:------|
| **1. Baseline S2 (NFS cold)** | 21s (measured) | 240s (measured)* | 43s (measured) | **306s** | Full cold start; S2 prefetch is environment-dependent |
| **2. Baseline S3 (warm cache)** | 21s (measured) | 7s (measured) | 41s (measured) | **75s** | Page cache hot |
| **3. MX (1st on new node)** | 21s | ~10–15s (GPU P2P) | 43s | **~75–80s** | MX streams weights from donor; worker init overlaps with MX P2P transfer |
| **4. GMS shadow (same node)** | 21s | ~0.1s (zero-copy) | 43s | **~64s** | Shadow pre-imports weights via GMS RO for fast failover activation |
| **5. MX+GMS shadow (new node)** | 21s | ~0.1s (zero-copy) | 43s | **~64s** | 1st fetches via MX; shadow pre-imports via GMS RO |
| **6. MX+GMS+compile cache** | 21s | ~0.1s (zero-copy) | ~2s (cached) | **~24s** | Warmup artifacts pre-cached (disk or GMS compile_cache tag); recovers most of PR #12407 regression too |
| **7. MX+GMS+compile+reduced worker init** | ~2s (optimized) | ~0.1s | ~2s | **~5s** | Requires MPI process pool optimization (see [worker init investigation](#worker-init-investigation-results)) |

\* The S2 prefetch (~240s on the current node) is the cold-NFS network read time and varies significantly by node-to-NFS network path. The v2 dataset on a different node showed ~70s for the same workload. Both are valid measurements of "cold NFS read" — the underlying point (cold NFS dominates, MX eliminates it) is unchanged regardless of the absolute number.

**Key takeaways:**
- **MX** eliminates the 70–320s NFS cold-read penalty by streaming only the relevant TP shard directly to each rank's GPU (~10–15s), also eliminating the ~9× CPU memory spike of the current load-all-then-shard pattern. For S1, MX P2P transfer provides enough concurrent server work to hide the 21s worker init naturally (similar to how HF download hides it today).
- **GMS** eliminates the weight loading cost for shadow/standby workers via zero-copy GPU memory import (~100ms). The primary use case is pre-staging shadow workers for <5s failover activation, not running multiple active serving instances.
- **Worker init (~21s) is on the critical path** for all scenarios without substantial concurrent server work. See [investigation results](#worker-init-investigation-results) for why early warm-up alone cannot hide this cost.
- **Neither MX nor GMS can reduce the ~43s warmup floor** — only compilation/autotuner caching can address this. The warmup floor grew from ~16s (v2) to ~43s (v3) due to PR #12407; see [Insight #6](#6-warmup-overhead-regression-from-pr-12407-new-in-v3).
- **The very first cluster-wide instance always pays full cost** (306–390s with NFS cold on this node). MX requires a donor node; GMS requires a prior instance on the same node.

---

## Benchmark Results

**Primary dataset (v3):** Current rebased codebase at upstream `4a848ccce` (2026-04-16), measured on node `umb-b300-dp-186`. Reflects today's TRT-LLM startup behavior.

**Reference dataset (v2):** Earlier codebase predating [PR #12407](https://github.com/NVIDIA/TensorRT-LLM/pull/12407) (warmup refactor), measured on node `umb-b300-dp-199`. Preserved in collapsible sections under each Part for the v2→v3 comparison and Insight #6.

**Environment:** 8x NVIDIA B300 SXM6 AC (275 GB each), NFS-backed storage, CUDA 13.1.
**Contract:** `first_request_ready` — profile finalized after first successful end-to-end request.
**Statistical protocol:** 3 runs per configuration; **representative-run** approach (the run whose total startup is the median is selected, and all per-component metrics are reported from that single run so components sum consistently to the total). Min/max across runs reported in per-config aggregate JSON for range context.
**Total profiles:** v3 = 14 configs × 3 runs = 42. v2 = 21 configs × 3 runs = 62.
**Note:** S1 (remote cold) downloads in v2 used an internal HF CDN/mirror achieving ~2 GB/s. Public cloud download times will be significantly longer. v3 did not run S1 (HF rate limits + storage tier story already validated).

### Reading the Results Tables

The startup path is **strictly serial** across two processes:

1. **Server process** — parses CLI args, creates MPI pool (for TP>1), runs `CachedModelLoader` (downloads/resolves the model), then calls `create_executor` which dispatches `worker_main` to the MPI pool.
2. **Executor worker process** — receives `worker_main` dispatch (only AFTER the server's model loader completes), then runs the full initialization: config loading, model construction (meta tensors), tensor materialization, checkpoint reading, weight application, warmup. Signals ready when done.

**Key: download and worker initialization are sequential.** The `create_executor` call happens AFTER `CachedModelLoader` completes — there is no concurrent worker initialization during download. (Verified via code: `llm.py:1284` runs `_build_model()` → download, then `llm.py:1309` calls `create_executor` → dispatches `worker_main`.)

The tables below show a hierarchical timer tree. Indented rows (prefixed with `└─` or `├─`) are **children** of the row above — their time is already included in the parent. Only top-level (non-indented) rows are additive. For example, "HF remote download" is *inside* "Cached model loader", not separate from it.

**How total startup adds up** (v3 measurements, Qwen 72B TP=8 across tiers):

```
S2 (NFS cold) — production baseline
Total = 306.3s
├─ [Worker]  MPI worker cold start       ~21.5s  ← Python imports, CUDA ctx, NCCL (see note)
├─ [Worker]  Checkpoint prefetch         233.2s  ← cold NFS reads (node-dependent)
├─ [Worker]  Apply weights                 3.7s
├─ [Worker]  Warmup (1st pass)            38.0s  ← torch_compile (25s) + autotuner (1.5s) + cuda graphs (11.4s)
├─ [Worker]  Warmup (2nd pass)             4.7s
├─ [Server]  Cached model loader          0.003s ← no download needed
└─ [Both]    Other overhead               ~5.4s  ← model construction, sampler, KV cache, IPC
                                          -----
             Sum                         ~306.5s

S3 (NFS warm) — best case (page cache hot)
Total = 74.6s
├─ [Worker]  MPI worker cold start       ~21.0s  ← Python imports, CUDA ctx, NCCL (see note)
├─ [Worker]  Checkpoint prefetch           3.5s  ← page cache hit
├─ [Worker]  Apply weights                 3.6s
├─ [Worker]  Warmup (1st pass)            36.3s  ← torch_compile (23.6s) + autotuner (1.5s) + cuda graphs (11.1s)
├─ [Worker]  Warmup (2nd pass)             4.6s
├─ [Server]  Cached model loader          0.002s
└─ [Both]    Other overhead               ~5.6s
                                          -----
             Sum                          ~74.6s
```

**Note on MPI worker cold start:** The ~21s "executor overhead" gap visible in S2/S3 is the **MPI worker first-dispatch latency** — Python imports, CUDA context creation, NCCL init triggered when the first task is `submit()`-ed to the pool. In S1 (remote cold), `cached_model_loader` dispatches the HF download to workers early via `_submit_to_all_workers`, so workers cold-start during the 63s download and are already warm by the time `worker_main` runs. In S2/S3, the model is local (`is_hub_model = False`), so no dispatch occurs until `create_executor` calls `worker_main` — the ~21s cold start is fully visible. See [Analysis §1](#executor-overhead-gap-s1-5s-vs-s2s3-25s--root-cause) for the detailed investigation.

<details>
<summary>v2 walkthrough (reference, pre-PR #12407)</summary>

```
S1 (remote cold):
Total = 93.4s
├─ [Server]  Cached model loader         63.5s  ← includes HF download (43.9s) + cache mgmt
├─ [Worker]  Weight loading total          8.7s  ← prefetch from warm page cache (3.5s) + apply (3.8s)
├─ [Worker]  Warmup (1st pass)            12.1s  ← autotuner (11.3s) + CUDA graphs (0.6s)
├─ [Worker]  Warmup (2nd pass)             4.1s
└─ [Both]    Executor overhead            ~5.0s  ← model construction, sampler, KV cache, IPC
                                          -----
             Sum                          ~93.4s

S2 (NFS cold):
Total = 114.4s
├─ [Worker]  MPI worker cold start       ~21.5s
├─ [Worker]  Checkpoint prefetch          65.0s  ← cold NFS reads (different node, ~3.5× faster than v3)
├─ [Worker]  Apply weights                 4.3s
├─ [Worker]  Warmup (1st pass)            12.4s  ← autotuner (11.5s) + CUDA graphs (0.7s); no torch_compile pass
├─ [Worker]  Warmup (2nd pass)             4.2s
├─ [Server]  Cached model loader          0.003s
└─ [Both]    Other overhead               ~6.9s
                                          -----
             Sum                         ~114.3s

S3 (NFS warm):
Total = 50.2s
├─ [Worker]  MPI worker cold start       ~21.0s
├─ [Worker]  Checkpoint prefetch           3.4s
├─ [Worker]  Apply weights                 3.6s
├─ [Worker]  Warmup (1st pass)            11.9s
├─ [Worker]  Warmup (2nd pass)             4.1s
├─ [Server]  Cached model loader          0.002s
└─ [Both]    Other overhead               ~6.2s
                                          -----
             Sum                          ~50.2s
```

</details>

### Weight Loading Data Flow (TP=8)

Understanding the weight loading pattern is important for interpreting results and for MX/GMS design:

```
Server process:
  CachedModelLoader — resolves model path only (0.003s for local models)
                      does NOT read weight files for PyTorch backend
  create_executor   — dispatches worker_main to 8 MPI pool workers

Each worker rank (independently, in parallel):
  1. Cooperative prefetch: each rank reads 1/8 of .safetensors files
     via raw f.read() (bytes discarded) → warms OS page cache
  2. MPI barrier — ensures all files are in page cache
  3. Full load: each rank loads ALL safetensors files into CPU memory
     (safetensors.torch.load_file, backed by mmap hitting warm cache)
  4. Shard: each rank slices its TP=1/8 partition from every tensor
  5. Copy shard to GPU, free CPU memory for consumed tensors
```

**Why every rank loads all files:** HF safetensors files are sharded by layer groups (file 1 = layers 0-15, file 2 = layers 16-31, etc.), but TP shards by *tensor slicing* — each rank needs a column/row slice of every layer's weights. Under pure TP, no file can be skipped.

**Why cooperative prefetch then full load:** Without prefetch, 8 ranks would simultaneously issue cold NFS reads for all files — an I/O contention storm. The prefetch ensures each file is read from NFS exactly once (total I/O = 1x model size), then all subsequent `load_file()` calls hit warm OS page cache.

**Memory behavior:** While `safetensors.torch.load_file()` uses mmap internally (shared physical pages across ranks via `MAP_PRIVATE` + `PROT_READ`), its `get_tensor()` implementation **copies** each tensor's bytes from the mmap region into a new CPU-allocated `torch.Tensor`. So each rank creates a private CPU copy of every tensor before slicing its 1/8 shard. Peak CPU memory: ~1x model (shared page cache) + 8x model (private tensor copies) ≈ **~9x model size** (~1.3TB for Qwen 72B). The copies are freed incrementally via `mark_consumed()`, but the transient spike is real.

**Existing optimization path (side observation):** The `safetensors` library supports selective tensor loading via `safe_open().get_slice()`, which reads only the requested byte range without materializing the full tensor. TRT-LLM's `load_weight_shard` (`_torch/modules/linear.py:129-136`) already has a code path for `PySafeSlice` objects — but it is never reached because `load_file()` materializes everything first. Switching from `load_file()` to `safe_open()` + `get_slice()` would let each rank read only its TP shard, reducing peak CPU memory from ~9x to ~2x model size. This is a potential quick-win optimization for the existing HF loading path, but becomes moot once MX integration is complete (MX streams only the relevant shard directly to each rank's GPU, bypassing storage entirely).

### Part 1 — Model Size Scaling (S2, NFS Cold — Production Baseline)

S2 (NFS cold) is the primary baseline because it represents the realistic production cold-start scenario: model files pre-staged on shared storage, no prior page cache warming, no HF download. This is the cost that MX integration aims to eliminate.

All times in seconds (representative run). Percentages are of total startup.

| Metric | B1: Qwen 7B (TP=1) | B3: DS 7B (TP=1) | B2: Qwen 72B (TP=8) | B4: DS 70B (TP=8) |
|:-------|----:|----:|----:|----:|
| **Total startup** | **77.4 (100%)** | **94.4 (100%)** | **306.3 (100%)** | **389.8 (100%)** |
| MPI worker cold start (included above) | hidden (TP=1) | hidden (TP=1) | ~21 (7%) | ~21 (5%) |
| Checkpoint prefetch (worker) | 16.9 (22%) | 35.7 (38%) | 233.2 (76%) | 318.4 (82%) |
| Apply weights (worker) | 3.4 (4%) | 2.8 (3%) | 3.7 (1%) | 3.6 (1%) |
| Warmup — 1st pass (worker) | 38.3 (49%) | 37.2 (39%) | 38.0 (12%) | 36.4 (9%) |
| ├─ torch_compile (general warmup) | 17.5 (23%) | 17.1 (18%) | 25.0 (8%) | 23.3 (6%) |
| ├─ Autotuner forward | 0.0 (0%) | 0.0 (0%) | 1.5 (<1%) | 2.0 (<1%) |
| └─ CUDA graphs | 20.6 (27%) | 19.9 (21%) | 11.4 (4%) | 11.0 (3%) |
| Warmup — 2nd pass (worker) | 0.8 (1%) | 0.7 (<1%) | 4.7 (2%) | 4.5 (1%) |

**Key observations:**
- **Cold NFS checkpoint prefetch dominates** for large models (76–82% of total). This is the primary target for MX (GPU P2P streaming eliminates NFS reads entirely).
- **For small models (7B, TP=1)**, warmup dominates (~49% of total) because model size is small and cold NFS reads are relatively quick (~17s for 14GB).
- **Warmup floor is ~36–38s** across all sizes due to torch_compile general warmup (~25s for 72B, ~17s for 7B) + CUDA graph capture (~11–20s) + autotuner (~0–2s).
- **MPI worker cold start (~21s)** is included in the totals but only visible for TP>1 (single-rank pools have no MPI dispatch latency).

<details>
<summary>v2 reference (pre-PR #12407, different node)</summary>

Only large models measured in v2; small-model S2 was not part of the original v2 matrix.

| Metric | B2: Qwen 72B (TP=8) | B4: DS 70B (TP=8) |
|:-------|----:|----:|
| **Total startup** | **114.4 (100%)** | **146.1 (100%)** |
| MPI worker cold start | ~21 (18%) | ~21 (14%) |
| Checkpoint prefetch (worker) | 65.0 (57%) | 99.2 (68%) |
| Apply weights (worker) | 4.3 (4%) | 3.6 (2%) |
| Warmup — 1st pass (worker) | 12.4 (11%) | 13.0 (9%) |
| ├─ Autotuner forward | 11.5 (10%) | 12.3 (8%) |
| └─ CUDA graphs | 0.7 (<1%) | 0.5 (<1%) |
| Warmup — 2nd pass (worker) | 4.2 (4%) | 4.0 (3%) |

**v2 → v3 deltas (Qwen 72B):**

| Component | v2 | v3 | Δ | Cause |
|:----------|---:|---:|---:|:------|
| Checkpoint prefetch | 65.0s | 233.2s | +168s | Different NFS path on new node (environment) |
| Total warmup | ~16.6s | ~42.7s | +26.1s | PR #12407 added general warmup pass (code) |
| Total startup | 114.4s | 306.3s | +192s | Sum of above |

The v2→v3 storage delta is fully attributable to the new node's slower cold NFS path (verified: v3 S3 prefetch is identical to v2 S3 prefetch — only S2 differs). The warmup delta is fully attributable to PR #12407 (see [Insight #6](#6-warmup-overhead-regression-from-pr-12407-new-in-v3)).

</details>

### Part 1b — Model Size Scaling (S1, Remote Cold Download — Reference Only)

S1 was not re-measured in v3 (HF rate-limiting concerns and the storage tier story is already validated by the S3 measurements that match v2). The v2 S1 data is preserved for reference and to demonstrate the "S1 hides MPI worker cold start" effect.

<details>
<summary>v2 S1 results (HF download path)</summary>

S1 measures the full HF download path. It is complementary to S2 because (a) S1's download inherently warms the page cache, making worker prefetch fast (~3s), and (b) S1 hides the ~21s MPI worker cold start behind the 63s download. These two effects make S1 appear faster than S2 despite involving a network download.

| Metric | B1: Qwen 7B (TP=1) | B3: DS 7B (TP=1) | B2: Qwen 72B (TP=8) | B4: DS 70B (TP=8) |
|:-------|----:|----:|----:|----:|
| **Total startup** | **36.1 (100%)** | **38.2 (100%)** | **93.4 (100%)** | **95.5 (100%)** |
| Cached model loader (server) | 6.4 (18%) | 6.3 (16%) | 63.5 (68%) | 62.5 (65%) |
| └─ HF remote download | 6.4 (18%) | 6.3 (16%) | 43.9 (47%) | 43.0 (45%) |
| Weight loading total (worker) | 5.5 (15%) | 7.5 (20%) | 8.7 (9%) | 10.5 (11%) |
| ├─ Checkpoint prefetch | 2.2 (6%) | 4.2 (11%) | 3.5 (4%) | 5.0 (5%) |
| └─ Apply weights | 2.8 (8%) | 2.8 (7%) | 3.8 (4%) | 3.7 (4%) |
| Warmup — 1st pass (worker) | 4.9 (14%) | 4.9 (13%) | 12.1 (13%) | 13.0 (14%) |
| ├─ Autotuner forward | 4.6 (13%) | 4.6 (12%) | 11.3 (12%) | 12.3 (13%) |
| └─ CUDA graphs | 0.1 (<1%) | 0.1 (<1%) | 0.6 (<1%) | 0.5 (<1%) |
| Warmup — 2nd pass (worker) | 1.7 (5%) | 1.7 (4%) | 4.1 (4%) | 4.0 (4%) |

**Why S1 appears faster than S2:** Two effects combine: (1) the HF download warms the page cache, so workers prefetch in ~3.5s instead of ~65s; (2) the ~21s MPI worker cold start is hidden behind the 63s download (workers are dispatched early for the download, and are already warm by the time `create_executor` runs). Note: the internal HF CDN achieves ~2 GB/s; with a slower public CDN, S1 would be significantly slower.

</details>

### Part 2 — Storage Tier Comparison (72B/70B Models, TP=8)

| Metric | Qwen 72B S2 | Qwen 72B S3 | DS 70B S2 | DS 70B S3 |
|:-------|----:|----:|----:|----:|
| **Total startup** | **306.3** | **74.6** | **389.8** | **77.7** |
| MPI worker cold start | ~21s visible | ~21s visible | ~21s visible | ~21s visible |
| Checkpoint prefetch (worker) | 233.2 | 3.5 | 318.4 | 6.0 |
| Apply weights (worker) | 3.7 | 3.6 | 3.6 | 3.7 |
| Warmup — 1st pass (worker) | 38.0 | 36.3 | 36.4 | 36.4 |
| ├─ torch_compile / general warmup | 25.0 | 23.6 | 23.3 | 23.4 |
| ├─ Autotuner forward | 1.5 | 1.5 | 2.0 | 2.0 |
| └─ CUDA graphs | 11.4 | 11.1 | 11.0 | 10.9 |
| Warmup — 2nd pass (worker) | 4.7 | 4.6 | 4.5 | 4.5 |

**Storage tier ordering (v3, unchanged from v2):**

| Tier | Page cache state | Worker prefetch (Qwen 72B) | MPI worker cold start | Why |
|------|-----------------|----------------------------|----------------------|-----|
| S2 | Cold (fresh inodes) | 233.2s | Visible (~21s) | No prior dispatch to workers |
| S3 | Warm (prior run's reads) | 3.5s | Visible (~21s) | No prior dispatch to workers |
| (S1 v2 ref) | Warm (download populates cache) | 3.5s | Hidden (behind download) | Download dispatches to workers early |

**Key takeaway:** S2 cold NFS reads dominate (76–82% of total for large models); S1/S3 enjoy page-cache-speed reads (~3–6s). MX is the optimization target for the cold-NFS path.

<details>
<summary>v2 reference (pre-PR #12407, different node)</summary>

| Metric | Qwen 72B S1 | Qwen 72B S2 | Qwen 72B S3 | DS 70B S1 | DS 70B S2 | DS 70B S3 |
|:-------|----:|----:|----:|----:|----:|----:|
| **Total startup** | **93.4** | **114.4** | **50.2** | **95.5** | **146.1** | **52.9** |
| MPI worker cold start | hidden in download | ~21s visible | ~21s visible | hidden in download | ~21s visible | ~21s visible |
| Cached model loader (server) | 63.5 | 0.003 | 0.002 | 62.5 | 0.003 | 0.001 |
| Checkpoint prefetch (worker) | 3.5 | 65.0 | 3.4 | 5.0 | 99.2 | 6.0 |
| Apply weights (worker) | 3.8 | 4.3 | 3.6 | 3.7 | 3.6 | 3.6 |
| Warmup — 1st pass (worker) | 12.1 | 12.4 | 11.9 | 13.0 | 13.0 | 12.5 |
| Warmup — 2nd pass (worker) | 4.1 | 4.2 | 4.1 | 4.0 | 4.0 | 4.0 |

**v3 S3 prefetch is essentially identical to v2 S3 prefetch** (3.5s vs 3.4s for Qwen 72B; 6.0s vs 6.0s for DS 70B), confirming the v2→v3 S2 delta is purely a node-specific NFS path difference, not codebase change.

</details>

**Planned: S4 (Local NVMe SSD, cold):** A fourth tier measuring cold reads from node-local NVMe (no network hop) is planned. This represents the model-pre-staged-to-local-disk scenario and provides the storage-speed ceiling against which MX P2P throughput should be compared. The node has 8x 3.5TB NVMe SSDs available. S4 uses the same fresh-inode-copy methodology as S2 to ensure cold page cache. Results pending.

### Part 3 — Autotuner Impact

Comparing autotuner ON vs OFF on S2 and S3 tiers. In v3, autotuner forward only takes ~1.5–2s (down from 11s in v2 due to PR #12407 refactor), so disabling it has a much smaller direct impact than it appeared to in v2.

| Metric | Qwen 72B S2 ON | Qwen 72B S2 OFF | Qwen 72B S3 ON | Qwen 72B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **306.3 (100%)** | **293.3 (100%)** | **74.6 (100%)** | **73.6 (100%)** |
| Warmup (1st pass) | 38.0 (12%) | 35.3 (12%) | 36.3 (49%) | 35.7 (48%) |
| — torch_compile | 25.0 (8%) | 23.6 (8%) | 23.6 (32%) | 23.9 (32%) |
| — Autotuner | 1.5 (<1%) | 0.0 (0%) | 1.5 (2%) | 0.0 (0%) |
| — CUDA graphs | 11.4 (4%) | 11.5 (4%) | 11.1 (15%) | 11.6 (16%) |
| — Memory pool | 0.1 (<1%) | 0.1 (<1%) | 0.1 (<1%) | 0.1 (<1%) |

| Metric | DS 70B S2 ON | DS 70B S2 OFF | DS 70B S3 ON | DS 70B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **389.8 (100%)** | **410.7 (100%)** | **77.7 (100%)** | **76.7 (100%)** |
| Warmup (1st pass) | 36.4 (9%) | 35.3 (9%) | 36.4 (47%) | 35.1 (46%) |
| — torch_compile | 23.3 (6%) | 23.8 (6%) | 23.4 (30%) | 23.6 (31%) |
| — Autotuner | 2.0 (<1%) | 0.0 (0%) | 2.0 (3%) | 0.0 (0%) |
| — CUDA graphs | 11.0 (3%) | 11.4 (3%) | 10.9 (14%) | 11.4 (15%) |
| — Memory pool | 0.1 (<1%) | 0.1 (<1%) | 0.1 (<1%) | 0.1 (<1%) |

Note: The DS 70B S2 OFF total is higher than ON (410.7s vs 389.8s) entirely due to NFS variability in the cold-prefetch phase (340s vs 318s) — not from autotuner being disabled. Warmup itself is unchanged.

<details>
<summary>v2 reference (pre-PR #12407)</summary>

In v2, autotuner ran for ~11s and CUDA graphs took only ~0.6s; disabling autotuner caused CUDA graphs to slow down by almost the same amount (~11s). Net warmup change was <1s.

| Metric | Qwen 72B S2 ON | Qwen 72B S2 OFF | Qwen 72B S3 ON | Qwen 72B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **114.4** | **104.9** | **50.2** | **49.8** |
| Warmup (1st pass) | 12.4 | 11.9 | 11.9 | 11.8 |
| — Autotuner | 11.5 | 0.0 | 11.0 | 0.0 |
| — CUDA graphs | 0.7 | 11.1 | 0.6 | 11.0 |

| Metric | DS 70B S2 ON | DS 70B S2 OFF | DS 70B S3 ON | DS 70B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **146.1** | **155.2** | **52.9** | **52.1** |
| Warmup (1st pass) | 13.0 | 12.1 | 12.5 | 11.9 |
| — Autotuner | 12.3 | 0.0 | 11.8 | 0.0 |
| — CUDA graphs | 0.5 | 11.4 | 0.5 | 11.1 |

</details>

### Part 4 — Serving Config Sensitivity

**D1: Large config** (bs=64, nt=8192 vs default bs=4, nt=1024):

| Metric | Qwen 72B S3 default | Qwen 72B S3 large | DS 70B S3 default | DS 70B S3 large |
|:-------|----:|----:|----:|----:|
| **Total startup** | **74.6 (100%)** | **99.7 (100%)** | **77.7 (100%)** | **100.8 (100%)** |
| Warmup (1st pass) | 36.3 (49%) | 54.0 (54%) | 36.4 (47%) | 52.6 (52%) |
| — torch_compile | 23.6 (32%) | 23.7 (24%) | 23.4 (30%) | 23.4 (23%) |
| — CUDA graphs | 11.1 (15%) | 28.9 (29%) | 10.9 (14%) | 27.3 (27%) |
| Warmup (2nd pass) | 4.6 (6%) | 11.7 (12%) | 4.5 (6%) | 11.3 (11%) |

D1 effect: +25s on both models, mostly in CUDA graphs (more variants captured) and 2nd-pass warmup.

**D2: Long sequence** (seq_len=16384 vs default 4096):

| Metric | Qwen 72B S3 default | Qwen 72B S3 seq16k | DS 70B S3 default | DS 70B S3 seq16k |
|:-------|----:|----:|----:|----:|
| **Total startup** | **74.6 (100%)** | **80.6 (100%)** | **77.7 (100%)** | **77.8 (100%)** |
| Warmup (1st pass) | 36.3 (49%) | 42.6 (53%) | 36.4 (47%) | 36.5 (47%) |
| — torch_compile | 23.6 (32%) | 29.7 (37%) | 23.4 (30%) | 23.4 (30%) |

D2 effect is model-dependent: Qwen 72B sees +6s (the new general warmup specializes over more KV-cache window-size variants), DS 70B is unchanged.

<details>
<summary>v2 reference (pre-PR #12407)</summary>

D1 (large config) in v2 added only ~+8s; D2 (seq16384) added ~+0.6s. Both effects are amplified in v3 because (a) per-CUDA-graph-variant capture time grew, and (b) the new general warmup pass adds a shape-iteration loop.

| Metric | Qwen 72B S3 default | Qwen 72B S3 large | DS 70B S3 default | DS 70B S3 large |
|:-------|----:|----:|----:|----:|
| **Total startup (D1)** | **50.2** | **58.8** | **52.9** | **61.6** |
| Warmup (1st pass) | 11.9 | 16.7 | 12.5 | 17.3 |
| — CUDA graphs | 0.6 | 4.9 | 0.5 | 4.1 |
| Warmup (2nd pass) | 4.1 | 8.2 | 4.0 | 7.6 |

| Metric | Qwen 72B S3 default | Qwen 72B S3 seq16k | DS 70B S3 default | DS 70B S3 seq16k |
|:-------|----:|----:|----:|----:|
| **Total startup (D2)** | **50.2** | **50.8** | **52.9** | **52.9** |
| Warmup (1st pass) | 11.9 | 12.4 | 12.5 | 12.6 |

</details>

### Analysis and Key Insights

All insights below are based on **v3 (current codebase)** measurements, with v2 numbers cited for the regression observation in Insight #6.

#### 1. Cold NFS I/O Dominates Production Cold Start

For large models (70–72B), the dominant bottleneck is cold NFS reads in S2:

- **S2 (NFS cold, production baseline):** **233–318s checkpoint prefetch** = 76–82% of total startup. No prior operation has warmed the page cache. This is the realistic first-cold-start cost that MX aims to eliminate.
- **S3 (page cache warm):** Worker prefetch is **3.5–6.0s** (the 7B/70B file size difference). With I/O removed, the bottleneck shifts to MPI worker cold start (~21s) + warmup (~41s).
- **S1 (remote cold, v2 only):** The internal HF mirror download was actually faster than this node's cold NFS, and the download itself warmed the page cache so worker prefetch was also ~3.5s. With a slower CDN (public HF Hub), S1 would be slower than S2.

The weight *application* phase (`apply_weights`) is constant at 3.6–3.7s regardless of storage tier, confirming it's GPU-bound, not I/O-bound.

**Note on v3 absolute numbers:** The S2 cold-prefetch time is highly node-dependent. v2 measured 65–99s on a different node; v3 measures 233–318s on the current node. Same NFS server, different network path. The qualitative story (S2 ≫ S3, MX target = cold-NFS) is unchanged either way.

#### Executor Overhead Gap: S1 (~5s) vs S2/S3 (~25s) — Root Cause

Investigation of the full hierarchical JSON profiles identified the source of the ~20s discrepancy. It is the **MPI worker first-dispatch latency** — the time for worker processes to receive their first task and begin executing.

The MPI worker pool is created in `llm.init_mpi_session` (`MpiPoolSession.__init__` → `MPIPoolExecutor`, `mpi_session.py:138`), but workers remain idle until they receive a dispatched function call via `mpi_session.submit()`. The first dispatch triggers worker-side cold start: Python module imports, CUDA context creation, and NCCL communicator setup (~21s on this hardware).

**The branching point** is `CachedModelLoader._download_hf_model_if_needed` (`llm_utils.py:719`):

```python
def _download_hf_model_if_needed(self, model_obj, revision=None):
    if model_obj.is_hub_model:          # ← S1: True (model is HF ID string)
        model_dirs = self._submit_to_all_workers(   # ← FIRST MPI DISPATCH
            CachedModelLoader._node_download_hf_model, ...)
        ...
    return model_obj.model_dir          # ← S2/S3: returns immediately (model is local Path)
```

Where `is_hub_model` (`llm_args.py:2467`) simply checks if the model is a string (HF ID) vs a `Path` (local directory):

```python
@property
def is_hub_model(self) -> bool:
    return not self.is_local_model      # True for "Qwen/Qwen2.5-72B-Instruct"

@property
def is_local_model(self) -> bool:
    return isinstance(self.model, Path) # True for Path("/home/.../Qwen2.5-72B-Instruct")
```

**Time-sequence comparison** (Qwen 72B, measured from profiler data):

```
S1 (remote cold) — worker cold start HIDDEN in download
═══════════════════════════════════════════════════════════════════════════════
Server clock:  0s        20s              64s                    93s
               │          │                │                      │
Server:        ├─ init_mpi_session (0s)    │                      │
               │  └─ MPIPoolExecutor       │                      │
               │     spawns 8 workers      │                      │
               │                           │                      │
               ├─ cached_model_loader ─────┤ (63.5s)              │
               │  └─ _submit_to_all_workers(_node_download_hf_model)
               │     ← FIRST MPI DISPATCH (server time 0.001s)   │
               │                           │                      │
               │                           ├─ create_executor ────┤ (28.7s)
               │                           │  └─ submit(worker_main)
               │                           │     ← workers already warm, gap=0.6s
               │                           │                      │
Workers:       │..cold start (~21s)........│                      │
               ├──────────────────────────>│                      │
               │  Python imports, CUDA ctx,│                      │
               │  NCCL init                │                      │
               │          │                │                      │
               │          ├─ _node_download_hf_model (43.9s) ────>│
               │          │                │                      │
               │          │                ├─ executor_worker.initialize (28.1s)──>│
               │          │                │                      │
═══════════════════════════════════════════════════════════════════════════════
                          ▲                ▲
                          │                └─ Workers warm; create_executor starts
                          └─ Workers finally ready after cold start


S2/S3 (local model) — worker cold start VISIBLE in create_executor
═══════════════════════════════════════════════════════════════════════════════
Server clock:  0s        21s                                    50/114s
               │          │                                      │
Server:        ├─ init_mpi_session (0s)                          │
               │  └─ MPIPoolExecutor spawns 8 workers            │
               │                                                 │
               ├─ cached_model_loader (0.003s)                   │
               │  └─ is_hub_model = False → return immediately   │
               │     ← NO MPI DISPATCH                          │
               │                                                 │
               ├─ create_executor ───────────────────────────────┤ (49/113s)
               │  └─ submit(worker_main)                         │
               │     ← FIRST MPI DISPATCH (server time 0.2s)    │
               │                                                 │
Workers:       │..cold start (~21s)..│                           │
               ├────────────────────>│                           │
               │  Python imports,    │                           │
               │  CUDA ctx, NCCL     │                           │
               │                     │                           │
               │                     ├─ executor_worker.initialize ──>│
               │                     │   (28s for S3, 92s for S2)│
               │                     │                           │
═══════════════════════════════════════════════════════════════════════════════
                                     ▲
                                     └─ Workers finally ready; all 21s is INSIDE
                                        create_executor, making it visible
```

**Evidence from server/worker clock correlation** (Qwen 72B):

| | S1 | S2 | S3 |
|---|---|---|---|
| First MPI dispatch (server clock) | 0.001s (`cached_model_loader`) | 0.213s (`create_executor`) | 0.229s (`create_executor`) |
| Worker profiler starts (server clock) | ~20s | ~22s | ~21s |
| Worker cold-start latency | **~20s** (hidden in download) | **~22s** (visible) | **~21s** (visible) |
| `create_executor` − `executor_worker.initialize` gap | 0.6s | 21.5s | 21.0s |

The ~21s is constant across all tiers. It is hidden in S1 because workers warm up during the download phase, and visible in S2/S3 because there is no server-side work to overlap with.

#### Worker Init Investigation Results

Experimental verification (branch `dynamo/worker-warmup-investigation`) tested three approaches to hiding the ~21s worker cold start:

| Approach | What happens | Total S3 startup | Savings |
|----------|-------------|-----------------|---------|
| **Baseline** (no warm-up) | Workers cold-start during `worker_main` | **~47s** | — |
| **Blocking warm-up** (noop + wait before worker_main) | Cold start in `ensure_workers_ready()`, then fast `worker_main` dispatch (0.05s entry vs 0.37s) | **~47s** | 0s |
| **Non-blocking warm-up** (noop queued before worker_main) | Both tasks queue on each worker; cold start during noop, then `worker_main` | **~47s** | 0s |

**Why early warm-up doesn't help for S2/S3:** There is only ~0.2s of server work between `init_mpi_session` and `create_executor` (for local models). The warm-up noop triggers worker cold start, but `worker_main` is dispatched just 0.2s later — workers are still cold-starting. Whether cold start happens during the noop or during `worker_main`, the total is the same because there's no concurrent server work to overlap with.

**Why it works for S1:** The first dispatch (`_node_download_hf_model`) provides 63.7s of server-side work. Workers complete their cold start during this time. The second dispatch (`worker_main`) finds workers warm and enters in 0.003s.

**Reducing the 21s itself** requires addressing the `mpi4py.futures.MPIPoolExecutor` lazy-spawn behavior: processes are spawned on first `submit()`, and the combined process creation + Python `sys.path` setup + `import_main` takes ~20s. Potential approaches:
- Eager process spawning at pool creation time (mpi4py configuration or custom pool)
- Pre-warmed persistent worker processes across server restarts
- Reducing Python import overhead in worker processes (lazy imports, import caching)

This is orthogonal to MX/GMS integration. MX P2P transfers (~10–15s) would naturally provide enough concurrent server work to hide most of the worker cold start, similar to how S1's HF download does today.

#### 2. Warmup Is the Irreducible Floor

With warm cache (S3), I/O is negligible. The remaining startup time breaks down as:

| Component | Qwen 72B S3 | % of Total | Reducible by |
|:----------|------------:|-----------:|:-------------|
| MPI worker cold start | ~21s | ~28% | MPI pool optimization or concurrent server work (see [investigation](#worker-init-investigation-results)) |
| Weight loading (prefetch + apply) | 7.1s | 10% | GMS zero-copy (~0.1s) |
| Warmup — 1st pass | 36.3s | 49% | Compilation caching (~5s) |
| — torch_compile / general warmup | 23.6s | 32% | Compilation caching of shape-specialization pass |
| — Autotuner forward | 1.5s | 2% | Autotuner caching |
| — CUDA graphs | 11.1s | 15% | Persistent graph cache |
| Warmup — 2nd pass | 4.6s | 6% | Compilation caching |
| Other overhead | ~5.6s | 8% | Minor |
| **Total** | **74.6s** | 100% | |

The two largest components are the MPI worker cold start (~21s) and the new general warmup pass (~24s). Experimental verification showed that simply dispatching a warm-up noop does NOT reduce total startup because there's only ~0.2s of server work to overlap with for local models. The cold start is on the critical path regardless of when it's triggered. Reducing it requires either (a) providing concurrent server work (as MX P2P transfer would), or (b) optimizing the MPI process pool itself.

MX and GMS address weight loading but cannot reduce warmup:
- **Warmup (~41s):** general warmup (~24s) + autotuner (~1.5s) + CUDA graph capture (~11s) + 2nd-pass warmup (~5s). Only compilation/autotuner caching can reduce this. PR #12407 raised this floor by ~27s vs the pre-#12407 baseline (see [Insight #6](#6-warmup-overhead-regression-from-pr-12407-new-in-v3)).
- **MPI worker cold start (~21s):** Lazy process spawning + Python imports in `mpi4py.futures.MPIPoolExecutor`. Not addressable by simple warm-up dispatch; requires MPI pool or import optimization.

#### 3. Disabling Autotuner Has No Net Benefit

In v3, autotuner forward only takes ~1.5–2s (down from ~11s in v2 due to PR #12407 refactor), while CUDA graph capture stays roughly constant (~11s) regardless of autotuner state. Disabling autotuner saves only ~1s of total startup.

| Comparison | Δ when autotuner OFF (v3) |
|:-----------|--------------:|
| Qwen 72B S3: warmup 1st pass | -0.6s |
| Qwen 72B S3: total startup | -1.0s |
| DS 70B S3: warmup 1st pass | -1.3s |
| DS 70B S3: total startup | -1.0s |

**Conclusion:** Disabling autotuner is not a useful optimization lever. (In v2 the conclusion was the same but for a different reason — autotuner saved ~11s but CUDA graphs slowed by ~11s in compensation. In v3, autotuner is small to begin with.)

#### 4. Serving Config Affects CUDA Graph Capture

| Config | v3 Δ (S3) | Cause |
|:-------|----------:|:------|
| D1: Qwen 72B large bs/nt (`bs=64, nt=8192`) | **+25.1s** | More CUDA graph variants captured (default has ~16 variants for bs∈{1..16}; large config has ~34 variants for bs∈{1..32, 64, 128}) |
| D1: DS 70B large bs/nt | **+23.1s** | Same as above |
| D2: Qwen 72B seq16384 (`max_seq_len=16384`) | **+6.0s** | The new general warmup pass specializes over more KV-cache window-size variants |
| D2: DS 70B seq16384 | +0.1s | DS does not have multiple window sizes; only Qwen does |

**Takeaways:**
- **Larger serving configs cost more startup time**, primarily through CUDA graph capture (proportional to graph variants) and 2nd-pass warmup. This is a real tradeoff users make when configuring high-throughput servers.
- **D2 is model-dependent**: only models with sliding/window attention pay the seq_len cost. Qwen 72B uses sliding window, DS 70B (Llama-based) does not.

(In v2, D1 added only ~+8s and D2 added ~+0.6s. The v3 amplification is due to (a) PR #12407's new general warmup pass, and (b) per-variant CUDA graph capture being slower in the rebased code.)

#### 5. Model Architecture Has Minor Impact

Qwen 72B and DeepSeek 70B show nearly identical startup patterns at the same tier and TP configuration. Qwen 72B S2 is 306s vs DS 70B S2 at 390s — the difference (~85s) is almost entirely from checkpoint prefetch (DS is ~15% more file data to read, and that delta scales with cold-NFS read time). On warm cache (S3): Qwen 72B is 75s, DS 70B is 78s — only a 3s difference.

Warmup times are within a few seconds of each other (Qwen 38s, DS 36s). The model-architecture-specific impact shows up only in:
- Sliding-window attention (D2 seq16384): Qwen pays +6s for window-size specialization, DS pays nothing
- File size for NFS prefetch (S2): DS files are ~15% larger

#### 6. Warmup Overhead Regression from PR #12407 (new in v3)

PR #12407 ("Refactor warmup orchestration in MTP", merged 2026-04-13) **increased total warmup time by ~27s** on B300 (TP=8) across all models we measured. This is an unintended regression effect of the refactor, not an environment/build artifact — same node, same GPU, same model checkpoint would show similar numbers if the code were checked out at the pre-#12407 commit.

**What the PR added:** A new general warmup pass in `PyTorchModelEngine.warmup()` that runs a shape-specialization forward pass over `_get_full_general_warmup_requests(resource_manager)` inside a `no_cuda_graph()` context, followed by MoE workspace cleanup and GC. The timer label `executor.warmup.torch_compile` is misleading — the pass runs regardless of whether `torch.compile` is enabled (`torch_compile_config` default remains `None`). See `tensorrt_llm/_torch/pyexecutor/model_engine.py:729-775`.

**Measured warmup deltas (v2 → v3, Qwen 72B TP=8):**

| Phase | v2 (pre-#12407) | v3 (post-#12407) | Δ |
|:------|----------------:|-----------------:|---:|
| torch_compile / general warmup | 0.0s | +25.0s | +25s |
| Autotuner forward | 11.5s | 1.5s | -10s |
| CUDA graphs | 0.7s | 11.4s | +10.7s |
| Memory pool | 0.1s | 0.1s | 0 |
| Warmup 2nd pass | 4.2s | 4.7s | +0.5s |
| **Total warmup** | **~16.6s** | **~42.7s** | **+26.1s** |

**Why it matters:**
- Every cold-start path on rebased code — S1, S2, S3, with or without MX/GMS — now pays this ~27s penalty. It's additive to all scenarios in the impact projection.
- The new general warmup pass is a candidate for the same compilation-caching optimization that applies to autotuner and CUDA graphs. If those artifacts can be persisted and restored across starts (the "MX+GMS+compile cache" scenario in our projection), this 27s becomes recoverable.
- **Recommendation**: raise this finding with the TRT-LLM team. Even if the refactor has a correctness benefit we're unaware of, it would be worth checking whether the new general warmup is strictly required for all serving configs, or whether it can be gated/skipped (e.g., only when torch.compile is actually enabled, or only for MTP models).

<details>
<summary>Previous Results (2026-04-10, initial profiling)</summary>

### Model Size Scaling

| Phase | DeepSeek 1.5B (3GB) | Llama 8B (16GB) | DeepSeek-V3-Lite (53GB) |
|:------|:----------------------|:-------------------|:---------------------------|
| **Total executor startup** | **49.1s** | **47.1s** | **114.2s** |
| Weight loading total | 19.6s (40%) | 38.3s (81%) | 79.3s (69%) |
| Warmup total (1st pass) | 25.0s (51%) | 6.1s (13%) | 31.1s (27%) |

### Remote-Cold Download

| Phase | 1.5B Remote-Cold | 72B Remote-Cold |
|:------|:---------------------|:--------------------|
| **Total startup** | **38.4s** | **96.4s** |
| `llm.hf.remote_download` | 3.4s | 44.0s |
| Weight loading (worker) | 2.9s | 8.7s |
| Warmup (1st pass) | 7.4s | 14.7s |

</details>

---

## Performance Regression Detection

```python
# tests/benchmarks/test_startup_performance.py

@pytest.mark.parametrize("scenario", ["baseline", "mx_p2p", "gms_ro"])
def test_cold_start_time(scenario, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Verify cold-start time within budget."""
    baselines = load_baselines()
    expected = baselines[model_name][scenario]

    start = time.perf_counter()
    llm = create_llm(model_name, scenario)
    elapsed = time.perf_counter() - start

    assert elapsed < expected * 1.2, (
        f"{scenario} cold-start {elapsed:.1f}s exceeds budget {expected}s"
    )
```
