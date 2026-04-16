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

The startup profiler spans **two processes** that run sequentially:

1. **Server process** — downloads/resolves the model, then spawns the executor worker.
2. **Executor worker process** — loads weights from disk, runs warmup, signals ready.

The tables below show a hierarchical timer tree. Indented rows (prefixed with `└─` or `├─`) are **children** of the row above — their time is already included in the parent. Only top-level (non-indented) rows are additive. For example, "HF remote download" is *inside* "Cached model loader", not separate from it.

**How total startup adds up** (using Qwen 72B S1 as example):

```
Total startup = 93.4s
│
├─ [Server]  Cached model loader         63.5s  ← includes HF download (43.9s)
├─ [Worker]  Weight loading total          8.7s  ← includes prefetch (3.5s) + apply (3.8s) + other
├─ [Worker]  Warmup (1st pass)            12.1s  ← includes autotuner (11.3s) + CUDA graphs (0.6s)
├─ [Worker]  Warmup (2nd pass)             4.1s
└─ [Both]    Other overhead               ~5.0s  ← config, sampler, KV cache setup, IPC, etc.
```

### Part 1 — Model Size Scaling (S1, Remote Cold Download)

Fresh HF download to `/tmp` (tmpfs) each run. All times in seconds (representative run).

| Metric | B1: Qwen 7B (TP=1) | B3: DS 7B (TP=1) | B2: Qwen 72B (TP=8) | B4: DS 70B (TP=8) |
|:-------|----:|----:|----:|----:|
| **Total startup** | **36.1** | **38.2** | **93.4** | **95.5** |
| Cached model loader (server) | 6.4 | 6.3 | 63.5 | 62.5 |
| └─ HF remote download | 6.4 | 6.3 | 43.9 | 43.0 |
| Weight loading total (worker) | 5.5 | 7.5 | 8.7 | 10.5 |
| ├─ Checkpoint prefetch | 2.2 | 4.2 | 3.5 | 5.0 |
| └─ Apply weights | 2.8 | 2.8 | 3.8 | 3.7 |
| Warmup — 1st pass (worker) | 4.9 | 4.9 | 12.1 | 13.0 |
| ├─ Autotuner forward | 4.6 | 4.6 | 11.3 | 12.3 |
| └─ CUDA graphs | 0.1 | 0.1 | 0.6 | 0.5 |
| Warmup — 2nd pass (worker) | 1.7 | 1.7 | 4.1 | 4.0 |

The four top-level phases (cached model loader + weight loading + warmup 1st + warmup 2nd) sum to ~88s for Qwen 72B; the remaining ~5s is executor overhead (config loading, sampler creation, KV cache setup, IPC signaling).

### Part 2 — Storage Tier Comparison (72B/70B Models, TP=8)

S1 = remote cold download, S2 = NFS cold (fresh inode copy per run), S3 = NFS warm (page cache hot).

| Metric | Qwen 72B S1 | Qwen 72B S2 | Qwen 72B S3 | DS 70B S1 | DS 70B S2 | DS 70B S3 |
|:-------|----:|----:|----:|----:|----:|----:|
| **Total startup** | **93.4** | **114.4** | **50.2** | **95.5** | **146.1** | **52.9** |
| Cached model loader (server) | 63.5 | 0.003 | 0.002 | 62.5 | 0.003 | 0.001 |
| Checkpoint prefetch (worker) | 3.5 | 65.0 | 3.4 | 5.0 | 99.2 | 6.0 |
| Apply weights (worker) | 3.8 | 4.3 | 3.6 | 3.7 | 3.6 | 3.6 |
| Warmup — 1st pass (worker) | 12.1 | 12.4 | 11.9 | 13.0 | 13.0 | 12.5 |
| Warmup — 2nd pass (worker) | 4.1 | 4.2 | 4.1 | 4.0 | 4.0 | 4.0 |

**Why S2 (NFS cold) is slower than S1 (remote download):** The I/O cost shifts between processes. In S1, the server downloads from an internal CDN at ~2 GB/s and writes to tmpfs — the worker's prefetch then reads from fast local tmpfs (3.5s). In S2, there is no download (0.003s), but the worker's checkpoint prefetch reads from cold NFS with no page cache (65–99s), which is slower than the CDN. The net result: S1 pays ~63s in the server + ~4s in the worker = ~67s of I/O, while S2 pays ~0s in the server + ~65–99s in the worker = ~65–99s of I/O. S3 (warm cache) eliminates this entirely since page cache serves the reads (~3–6s).

### Part 3 — Autotuner Impact

Comparing autotuner ON vs OFF on S2 and S3 tiers. Warmup component shift when autotuner is disabled.

| Metric | Qwen 72B S2 ON | Qwen 72B S2 OFF | Qwen 72B S3 ON | Qwen 72B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **114.4** | **104.9** | **50.2** | **49.8** |
| Warmup (1st pass) | 12.4 | 11.9 | 11.9 | 11.8 |
| — Autotuner | 11.5 | 0.0 | 11.0 | 0.0 |
| — CUDA graphs | 0.7 | 11.1 | 0.6 | 11.0 |
| — Memory pool | 0.1 | 0.6 | 0.1 | 0.6 |

| Metric | DS 70B S2 ON | DS 70B S2 OFF | DS 70B S3 ON | DS 70B S3 OFF |
|:-------|----:|----:|----:|----:|
| **Total startup** | **146.1** | **155.2** | **52.9** | **52.1** |
| Warmup (1st pass) | 13.0 | 12.1 | 12.5 | 11.9 |
| — Autotuner | 12.3 | 0.0 | 11.8 | 0.0 |
| — CUDA graphs | 0.5 | 11.4 | 0.5 | 11.1 |
| — Memory pool | 0.1 | 0.6 | 0.1 | 0.6 |

### Part 4 — Serving Config Sensitivity

**D1: Large config** (bs=64, nt=8192 vs default bs=4, nt=1024):

| Metric | Qwen 72B S3 default | Qwen 72B S3 large | DS 70B S3 default | DS 70B S3 large |
|:-------|----:|----:|----:|----:|
| **Total startup** | **50.2** | **58.8** | **52.9** | **61.6** |
| Warmup (1st pass) | 11.9 | 16.7 | 12.5 | 17.3 |
| — CUDA graphs | 0.6 | 4.9 | 0.5 | 4.1 |
| Warmup (2nd pass) | 4.1 | 8.2 | 4.0 | 7.6 |

**D2: Long sequence** (seq_len=16384 vs default 4096):

| Metric | Qwen 72B S3 default | Qwen 72B S3 seq16k | DS 70B S3 default | DS 70B S3 seq16k |
|:-------|----:|----:|----:|----:|
| **Total startup** | **50.2** | **50.8** | **52.9** | **52.9** |
| Warmup (1st pass) | 11.9 | 12.4 | 12.5 | 12.6 |

### Analysis and Key Insights

#### 1. Storage I/O Dominates Cold Start

For large models (70–72B), total I/O cost = server-side download/resolution + worker-side checkpoint prefetch. The dominant bottleneck shifts by tier:

- **S1 (remote cold):** Server downloads from CDN (43–44s) to tmpfs; worker prefetches from fast tmpfs (3–5s). Total I/O: ~47–49s. Note: the internal CDN achieves ~2 GB/s; public cloud would be significantly slower.
- **S2 (NFS cold):** No download needed (0.003s), but worker prefetches from cold NFS (65–99s) — cold inode reads are slower than the CDN. Total I/O: ~65–99s. **S2 is slower than S1** because cold NFS throughput < CDN throughput.
- **S3 (warm cache):** No download (0.002s), worker prefetches from page cache (3–6s). Total I/O: ~3–6s — **2–3x faster** than S1.

The weight *application* phase (`apply_weights`) is constant at 3.6–4.3s regardless of storage tier, confirming it's GPU-bound, not I/O-bound.

#### 2. Warmup Is the Irreducible Floor

With warm cache (S3), I/O is negligible and warmup dominates the remaining startup time:

| Component | Qwen 72B S3 | % of Total |
|:----------|------------:|-----------:|
| Cached model loader (server) | ~0s | ~0% |
| Weight loading (worker) | 8.9s | 17.7% |
| Warmup — 1st pass (worker) | 11.9s | 23.7% |
| Warmup — 2nd pass (worker) | 4.1s | 8.2% |
| Executor overhead | ~25s | ~50% |
| **Total** | **50.2s** | 100% |

The autotuner forward pass alone is 11.0s (21.9% of total). This is the irreducible floor that MX and GMS cannot improve — only compilation caching can address it.

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
