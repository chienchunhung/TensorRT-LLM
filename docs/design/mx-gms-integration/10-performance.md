# 10. Performance Expectations and Benchmark Plan

[< Back to Overview](README.md)

**Last Updated:** 2026-04-16

## Target Metrics

| Scenario | Baseline | Target | Improvement |
|:---------|:---------|:-------|:-----------|
| Cold-start (DeepSeek-V3, 681GB) | 5-10 min | 15-30s | **10-20x** |
| Replica scale-up (Llama-70B) | 2-3 min | 5-10s | **12-36x** |
| Memory per worker (same GPU) | N x weights | 1 x weights | **Nx reduction** |
| Failover time (shadow takeover) | Cold-start | < 5s | **60-120x** |
| Multi-node scale-out (N replicas) | N x load time | ~constant | **Near-constant** |
| P2P transfer throughput | N/A | > 20 GB/s (NVLink) | — |
| GMS import latency | N/A | < 500ms | — |
| Throughput regression | — | < 2% | Negligible |

## Benchmark Scenarios

### Test 1: Cold-Start Latency (Baseline Profiling) — Completed

**What:** Time from process start to first successful inference under the standard HF weight-loading path (no MX, no GMS). Establishes the baseline startup breakdown across model sizes, storage tiers, autotuner settings, and serving configurations.

**Status:** Completed (2026-04-16). 62 profiles across 21 configurations. See [Benchmark Results (v2)](#benchmark-results-v2) below.

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

### Test 3: Memory Efficiency — Not Yet Executed

**What:** Peak GPU memory with N workers sharing via GMS. Requires GMS integration to be implemented.

**Validation:**
```
N=1: memory_1 = model_size + kv_cache + overhead
N=2: memory_2 ≈ model_size + 2*(kv_cache + overhead)  # NOT 2*model_size
N=4: memory_4 ≈ model_size + 4*(kv_cache + overhead)
```

### Test 4: Shadow Failover Latency — Not Yet Executed

**What:** Time from primary crash to shadow serving first request. Requires GMS + executor failover integration.

**Steps:**
1. Start primary + shadow with GMS
2. Send warmup requests to primary
3. Kill primary (`kill -9`)
4. Measure time until shadow returns first response
5. **Target:** < 5s

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

| Tier | ID | What it measures | Setup |
|------|----|-----------------|-------|
| Remote cold | S1 | Full HF download over network + model load | Isolated empty HF cache on `/tmp` (tmpfs); every run downloads from scratch |
| NFS cold | S2 | NFS file read (no page cache) + model load | Model files pre-copied to a **fresh NFS directory per run** (new inodes), guaranteeing cold page cache without needing `drop_caches` privileges |
| Local warm | S3 | Page-cache-warm file read + model load | Model files on NFS, page cache hot from a prior run; simulates 2nd replica on the same node |

**Default tier: S1 (remote cold).**

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

Scenario-based projection using measured S2/S3 baselines for Qwen 72B (TP=8). The "first pays upfront, rest benefit" property of MX and GMS is reflected explicitly.

| Scenario | Weight Load Cost | Warmup Cost | Total Startup | Notes |
|:---------|:-----------------|:------------|:--------------|:------|
| **1. Baseline S2 (NFS cold)** | 71.8s (measured) | 16.6s (measured) | **114.4s** | Every instance pays full NFS cold read |
| **2. Baseline S3 (warm cache)** | 8.9s (measured) | 16.0s (measured) | **50.2s** | 2nd instance on same node, page cache hot |
| **3. MX (1st on new node)** | ~10–15s (GPU P2P) | 16.6s | **~50–55s** | MX streams weights from donor node, converts S2→S3 equivalent |
| **4. GMS (2nd+ on same node)** | ~0.1s (zero-copy) | 16.6s | **~25–30s** | 1st instance loads normally; 2nd+ near-free via shared GPU memory |
| **5. MX+GMS (2nd+ on new node)** | ~0.1s (zero-copy) | 16.6s | **~25–30s** | 1st fetches via MX; 2nd+ near-free via GMS |
| **6. MX+GMS+compile cache** | ~0.1s (zero-copy) | ~2s (cached) | **~12–15s** | Best case: all warmup artifacts pre-cached |

Key takeaways for MX/GMS integration:
- **MX** eliminates the 65–99s NFS cold-read penalty by streaming weights GPU-to-GPU (~10–15s).
- **GMS** eliminates the 8.9s weight loading cost entirely for 2nd+ instances via zero-copy sharing.
- **Neither MX nor GMS can reduce the ~16s warmup floor** — only compilation/autotuner caching can address this.
- **The very first cluster-wide instance always pays full cost** (114–146s with NFS cold). MX requires a donor node; GMS requires a prior instance on the same node.

---

## Benchmark Results (v2)

**Environment:** 8x NVIDIA B300 SXM6 AC (275 GB each), NFS-backed storage, CUDA 13.1, TRT-LLM 1.3.0rc11
**Contract:** `first_request_ready` — profile finalized after first successful end-to-end request
**Statistical protocol:** 3 runs per configuration; **representative-run** approach (the run whose total startup is the median is selected, and all per-component metrics are reported from that single run so components sum consistently to the total)
**Total profiles collected:** 62 across 21 configurations
**Note:** S1 (remote cold) downloads used an internal HF CDN/mirror achieving ~2 GB/s. Public cloud download times will be significantly longer.

### Reading the Results Tables

The startup path is **strictly serial** across two processes:

1. **Server process** — parses CLI args, creates MPI pool (for TP>1), runs `CachedModelLoader` (downloads/resolves the model), then calls `create_executor` which dispatches `worker_main` to the MPI pool.
2. **Executor worker process** — receives `worker_main` dispatch (only AFTER the server's model loader completes), then runs the full initialization: config loading, model construction (meta tensors), tensor materialization, checkpoint reading, weight application, warmup. Signals ready when done.

**Key: download and worker initialization are sequential.** The `create_executor` call happens AFTER `CachedModelLoader` completes — there is no concurrent worker initialization during download. (Verified via code: `llm.py:1284` runs `_build_model()` → download, then `llm.py:1309` calls `create_executor` → dispatches `worker_main`.)

The tables below show a hierarchical timer tree. Indented rows (prefixed with `└─` or `├─`) are **children** of the row above — their time is already included in the parent. Only top-level (non-indented) rows are additive. For example, "HF remote download" is *inside* "Cached model loader", not separate from it.

**How total startup adds up** (using Qwen 72B as example across tiers):

```
S1 (remote cold):
Total = 93.4s
├─ [Server]  Cached model loader         63.5s  ← includes HF download (43.9s) + cache mgmt
├─ [Worker]  Weight loading total          8.7s  ← prefetch from tmpfs (3.5s) + apply (3.8s)
├─ [Worker]  Warmup (1st pass)            12.1s  ← autotuner (11.3s) + CUDA graphs (0.6s)
├─ [Worker]  Warmup (2nd pass)             4.1s
└─ [Both]    Executor overhead            ~5.0s  ← model construction, sampler, KV cache, IPC
                                          -----
             Sum                          ~93.4s

S2 (NFS cold):
Total = 114.4s
├─ [Server]  Cached model loader          0.003s ← no download needed
├─ [Worker]  Checkpoint prefetch          65.0s  ← cold NFS reads
├─ [Worker]  Apply weights                 4.3s
├─ [Worker]  Warmup (1st pass)            12.4s
├─ [Worker]  Warmup (2nd pass)             4.2s
└─ [Both]    Executor overhead           ~28.5s  ← see note below
                                          -----
             Sum                         ~114.4s
```

**Note on executor overhead:** The summary tables highlight the largest phases. The remaining time ("executor overhead") covers model construction on meta tensors, tensor materialization, CUDA context setup, NCCL communicator initialization, sampler creation, KV cache allocation/configuration, and IPC coordination. This overhead is ~5s for S1 but ~25–28s for S2/S3 — the difference is not yet fully characterized and warrants further investigation using the full hierarchical JSON profiles.

### Part 1 — Model Size Scaling (S1, Remote Cold Download)

Fresh HF download to `/tmp` (tmpfs) each run. All times in seconds (representative run). Percentages are of total startup.

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

For S1, the four highlighted phases (cached model loader + weight loading + warmup 1st + warmup 2nd) sum to ~88s for Qwen 72B; the remaining ~5s is executor overhead (model construction, sampler, KV cache, IPC).

### Part 2 — Storage Tier Comparison (72B/70B Models, TP=8)

S1 = remote cold download, S2 = NFS cold (fresh inode copy per run), S3 = NFS warm (page cache hot).

| Metric | Qwen 72B S1 | Qwen 72B S2 | Qwen 72B S3 | DS 70B S1 | DS 70B S2 | DS 70B S3 |
|:-------|----:|----:|----:|----:|----:|----:|
| **Total startup** | **93.4 (100%)** | **114.4 (100%)** | **50.2 (100%)** | **95.5 (100%)** | **146.1 (100%)** | **52.9 (100%)** |
| Cached model loader (server) | 63.5 (68%) | 0.003 (<1%) | 0.002 (<1%) | 62.5 (65%) | 0.003 (<1%) | 0.001 (<1%) |
| Checkpoint prefetch (worker) | 3.5 (4%) | 65.0 (57%) | 3.4 (7%) | 5.0 (5%) | 99.2 (68%) | 6.0 (11%) |
| Apply weights (worker) | 3.8 (4%) | 4.3 (4%) | 3.6 (7%) | 3.7 (4%) | 3.6 (2%) | 3.6 (7%) |
| Warmup — 1st pass (worker) | 12.1 (13%) | 12.4 (11%) | 11.9 (24%) | 13.0 (14%) | 13.0 (9%) | 12.5 (24%) |
| Warmup — 2nd pass (worker) | 4.1 (4%) | 4.2 (4%) | 4.1 (8%) | 4.0 (4%) | 4.0 (3%) | 4.0 (8%) |

**Why S2 (NFS cold) is slower than S1 (remote download):**

1. **I/O speed:** Cold NFS reads (65–99s for checkpoint prefetch) are genuinely slower than the CDN download path (~44s download to tmpfs + ~4s worker prefetch = ~48s total I/O in S1 vs 65–99s in S2).

2. **Executor overhead gap:** S2/S3 show ~25–28s of executor initialization overhead (model construction, tensor materialization, CUDA context, NCCL, sampler, KV cache setup) not broken out in the summary table, compared to only ~5s in S1. The source of this ~20s discrepancy is not yet fully characterized — the full hierarchical JSON profiles contain additional phase detail that may explain it.

S3 (warm cache) eliminates the I/O penalty entirely (page cache serves reads in 3–6s) and shows the clearest picture of the executor overhead floor (~27s), making it the most useful baseline for projecting MX/GMS improvements.

### Part 3 — Autotuner Impact

Comparing autotuner ON vs OFF on S2 and S3 tiers. Warmup component shift when autotuner is disabled.

| Metric | Qwen 72B S2 ON | Qwen 72B S2 OFF | Qwen 72B S3 ON | Qwen 72B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **114.4 (100%)** | **104.9 (100%)** | **50.2 (100%)** | **49.8 (100%)** |
| Warmup (1st pass) | 12.4 (11%) | 11.9 (11%) | 11.9 (24%) | 11.8 (24%) |
| — Autotuner | 11.5 (10%) | 0.0 (0%) | 11.0 (22%) | 0.0 (0%) |
| — CUDA graphs | 0.7 (<1%) | 11.1 (11%) | 0.6 (1%) | 11.0 (22%) |
| — Memory pool | 0.1 (<1%) | 0.6 (<1%) | 0.1 (<1%) | 0.6 (1%) |

| Metric | DS 70B S2 ON | DS 70B S2 OFF | DS 70B S3 ON | DS 70B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **146.1 (100%)** | **155.2 (100%)** | **52.9 (100%)** | **52.1 (100%)** |
| Warmup (1st pass) | 13.0 (9%) | 12.1 (8%) | 12.5 (24%) | 11.9 (23%) |
| — Autotuner | 12.3 (8%) | 0.0 (0%) | 11.8 (22%) | 0.0 (0%) |
| — CUDA graphs | 0.5 (<1%) | 11.4 (7%) | 0.5 (1%) | 11.1 (21%) |
| — Memory pool | 0.1 (<1%) | 0.6 (<1%) | 0.1 (<1%) | 0.6 (1%) |

### Part 4 — Serving Config Sensitivity

**D1: Large config** (bs=64, nt=8192 vs default bs=4, nt=1024):

| Metric | Qwen 72B S3 default | Qwen 72B S3 large | DS 70B S3 default | DS 70B S3 large |
|:-------|----:|----:|----:|----:|
| **Total startup** | **50.2 (100%)** | **58.8 (100%)** | **52.9 (100%)** | **61.6 (100%)** |
| Warmup (1st pass) | 11.9 (24%) | 16.7 (28%) | 12.5 (24%) | 17.3 (28%) |
| — CUDA graphs | 0.6 (1%) | 4.9 (8%) | 0.5 (1%) | 4.1 (7%) |
| Warmup (2nd pass) | 4.1 (8%) | 8.2 (14%) | 4.0 (8%) | 7.6 (12%) |

**D2: Long sequence** (seq_len=16384 vs default 4096):

| Metric | Qwen 72B S3 default | Qwen 72B S3 seq16k | DS 70B S3 default | DS 70B S3 seq16k |
|:-------|----:|----:|----:|----:|
| **Total startup** | **50.2 (100%)** | **50.8 (100%)** | **52.9 (100%)** | **52.9 (100%)** |
| Warmup (1st pass) | 11.9 (24%) | 12.4 (24%) | 12.5 (24%) | 12.6 (24%) |

### Analysis and Key Insights

#### 1. Storage I/O Dominates Cold Start

For large models (70–72B), I/O is the dominant bottleneck:

- **S1 (remote cold):** CDN download (43–44s) to tmpfs dominates, with the full model loader phase taking 62–64s (includes HF cache management overhead beyond raw download). Worker prefetch from tmpfs is fast (3–5s).
- **S2 (NFS cold):** No download needed, but worker prefetch from cold NFS is 65–99s — genuinely slower than the CDN path. This is the primary reason S2 total exceeds S1.
- **S3 (warm cache):** No download, page-cache-warm prefetch (3–6s). Provides the clearest view of the non-I/O startup floor.

The weight *application* phase (`apply_weights`) is constant at 3.6–4.3s regardless of storage tier, confirming it's GPU-bound, not I/O-bound.

**Open question:** S2/S3 show ~25–28s of executor overhead not broken out in the summary tables, compared to ~5s in S1. This gap warrants further investigation with the full hierarchical profiles to determine whether it reflects unmeasured initialization phases, profiler coverage differences, or other factors.

#### 2. Warmup Is the Irreducible Floor

With warm cache (S3), I/O is negligible. The remaining startup time breaks down as:

| Component | Qwen 72B S3 | % of Total |
|:----------|------------:|-----------:|
| Weight loading (prefetch + apply) | 7.0s | 14% |
| Warmup — 1st pass | 11.9s | 24% |
| Warmup — 2nd pass | 4.1s | 8% |
| Executor overhead (model construction, CUDA ctx, NCCL, sampler, KV cache, IPC) | ~27s | ~54% |
| **Total** | **50.2s** | 100% |

MX and GMS can reduce the 7.0s weight loading cost (GMS: ~0.1s zero-copy) but cannot address:
- **Warmup (~16s):** Autotuner forward (11.0s) + CUDA graph capture (0.6s) + 2nd-pass warmup (4.1s). Only compilation/autotuner caching can reduce this.
- **Executor overhead (~27s):** Model construction, tensor materialization, CUDA context, NCCL communicators, sampler/KV cache creation. Only process-level optimizations (persistent workers, pre-warmed containers) can reduce this. The exact breakdown of this overhead requires further profiling.

#### 3. Disabling Autotuner Has No Net Benefit

When autotuner is disabled, CUDA graph capture time increases by almost exactly the same amount (0.6s → 11s). This is because the autotuner's kernel selections make subsequent CUDA graph capture faster. **Net warmup change: <1s (<1% of total).**

#### 4. Serving Config Affects CUDA Graph Capture

Increasing `max_batch_size` (4→64) and `max_num_tokens` (1024→8192) adds 4–5s to CUDA graph capture and doubles 2nd-pass warmup (4→8s), adding ~8–9s total. This is proportional to the number of graph variants captured.

Increasing `max_seq_len` (4096→16384) has minimal impact on startup (~0.6s difference with S3), affecting only KV cache block allocation.

#### 5. Model Architecture Has Minor Impact

Qwen 72B and DeepSeek 70B show nearly identical startup patterns at the same tier and TP configuration. The small differences are in checkpoint prefetch time (DeepSeek files are slightly larger) and autotuner duration (model-specific kernel search).

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
