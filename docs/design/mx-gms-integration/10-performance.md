# 10. Performance Expectations and Benchmark Plan

[< Back to Overview](README.md)

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

### Test 1: Cold-Start Latency

**What:** Time from process start to first successful inference.

**Configurations:**
| Config | `checkpoint_format` | `load_format` | Expected |
|:-------|:-------------------|:-------------|:---------|
| Baseline | `HF` (default) | `AUTO` (default) | Minutes |
| MX (2nd replica) | `MX` | `AUTO` | 15-30s |
| GMS (2nd worker) | `HF` | `GMS` | < 5s |
| MX+GMS (2nd replica+worker) | `MX` | `GMS` | < 30s |

### Test 2: P2P Transfer Throughput

**What:** GB/s during MX weight transfer.

**Configurations:**
- Same node (NVLink): expect > 50 GB/s
- Cross-node (InfiniBand HDR): expect > 20 GB/s
- Cross-node (RoCE 100G): expect > 10 GB/s

### Test 3: Memory Efficiency

**What:** Peak GPU memory with N workers sharing via GMS.

**Validation:**
```
N=1: memory_1 = model_size + kv_cache + overhead
N=2: memory_2 ≈ model_size + 2*(kv_cache + overhead)  # NOT 2*model_size
N=4: memory_4 ≈ model_size + 4*(kv_cache + overhead)
```

### Test 4: Shadow Failover Latency

**What:** Time from primary crash to shadow serving first request.

**Steps:**
1. Start primary + shadow with GMS
2. Send warmup requests to primary
3. Kill primary (`kill -9`)
4. Measure time until shadow returns first response
5. **Target:** < 5s

### Test 5: Throughput Regression

**What:** Steady-state throughput with MX/GMS-loaded weights vs. standard.

**Validation:** < 2% regression. MX/GMS affect startup only; the loaded model should be identical.

### Test 6: vLLM Comparison

**What:** Compare TRT-LLM `--checkpoint-format mx` against vLLM `--load-format mx`.

**Metrics:**
- Cold-start latency (2nd replica)
- P2P transfer throughput
- **Target:** Within 20% of vLLM

## Detailed Test Matrix

### Model Matrix

| ID | Series | Variant | HF Repo | Approx Size | TP |
|----|--------|---------|---------|-------------|-----|
| B1 | Qwen | Small | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | 1 |
| B2 | Qwen | Large | `Qwen/Qwen2.5-72B-Instruct` | ~145 GB | 8 |
| B3 | DeepSeek | Small | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14 GB | 1 |
| B4 | DeepSeek | Large | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | ~131 GB | 8 |

Default serving config: `max_batch_size=4, max_num_tokens=1024, max_seq_len=4096`.

### Storage Tier Matrix

Three tiers measuring different weight-loading scenarios:

| Tier | ID | What it measures | Setup |
|------|----|-----------------|-------|
| Remote cold | S1 | Full HF download over network + model load | Isolated empty HF cache on `/tmp` (tmpfs); every run downloads from scratch |
| NFS cold | S2 | NFS file read (no page cache) + model load | Model files pre-copied to a **fresh NFS directory per run** (new inodes), guaranteeing cold page cache without needing `drop_caches` privileges |
| Local warm | S3 | Page-cache-warm file read + model load | Model files on NFS, page cache hot from a prior run; simulates 2nd replica on the same node |

**Default tier: S1 (remote cold).**

#### S2 Methodology: Ensuring True NFS-Cold Reads

The Linux page cache is keyed on `(device, inode)`. Once a file is read, subsequent reads of the same inode are served from RAM regardless of the file path. This means:
- Simply pointing at the same NFS directory for multiple runs would make runs 2+ effectively warm (S3).
- Dropping page cache (`echo 3 > /proc/sys/vm/drop_caches`) requires root/sysctl privileges that may not be available on shared nodes.

To guarantee cold NFS reads without special privileges, each S2 run:
1. Copies the model to a **new directory** (`_s2_nfs_cold/<model>_runN/`) using `cp -rL`, creating fresh inodes.
2. Serves from the new directory.
3. Cleans up the previous run's copy to keep disk usage at ~1x model size.

The copy time is **not** included in the benchmark measurement.

### Statistical Protocol

- Each configuration runs **3 times**.
- Report: **median**, **min**, **max** for each profiled phase.
- Automate via `run_startup_bench.sh` (single config) and `run_startup_bench_all.sh` (full matrix).
- Post-process with `aggregate_startup_results.py` for median/min/max extraction.

### Part 1: Model Size Scaling (S1 remote cold, default config)

| Test ID | Model | Tier | TP | Runs | Purpose |
|---------|-------|------|----|------|---------|
| B1-S1 | Qwen 7B | S1 | 1 | 3 | Small baseline (Qwen) |
| B2-S1 | Qwen 72B | S1 | 8 | 3 | Large baseline (Qwen) |
| B3-S1 | DeepSeek 7B | S1 | 1 | 3 | Small baseline (DeepSeek) |
| B4-S1 | DeepSeek 70B | S1 | 8 | 3 | Large baseline (DeepSeek) |

### Part 2: Storage Tier Comparison (large models only)

| Test ID | Model | Tier | TP | Runs | Purpose |
|---------|-------|------|----|------|---------|
| B2-S2 | Qwen 72B | S2 | 8 | 3 | NFS cold (fresh inode copy per run) |
| B2-S3 | Qwen 72B | S3 | 8 | 3 | Warm page cache (2nd replica) |
| B4-S2 | DeepSeek 70B | S2 | 8 | 3 | NFS cold (fresh inode copy per run) |
| B4-S3 | DeepSeek 70B | S3 | 8 | 3 | Warm page cache (2nd replica) |

### Part 3: Autotuner Impact (large models, S1)

| Test ID | Model | Tier | TP | Autotuner | Runs | Purpose |
|---------|-------|------|----|-----------|------|---------|
| B2-S1-C | Qwen 72B | S1 | 8 | OFF | 3 | Isolate autotuner cost (Qwen) |
| B4-S1-C | DeepSeek 70B | S1 | 8 | OFF | 3 | Isolate autotuner cost (DeepSeek) |

Compare against B2-S1 and B4-S1 (autotuner ON by default).

### Part 4: Serving Config Sensitivity (large models, S1)

| Test ID | Model | Tier | TP | Config | Runs | Purpose |
|---------|-------|------|----|--------|------|---------|
| B2-S1-D1 | Qwen 72B | S1 | 8 | bs=64, nt=8192 | 3 | Large batch + token budget |
| B4-S1-D1 | DeepSeek 70B | S1 | 8 | bs=64, nt=8192 | 3 | Large batch + token budget |
| B2-S1-D2 | Qwen 72B | S1 | 8 | max_seq_len=16384 | 3 | Long-sequence KV cache impact |
| B4-S1-D2 | DeepSeek 70B | S1 | 8 | max_seq_len=16384 | 3 | Long-sequence KV cache impact |

Compare against B2-S1 and B4-S1 (default bs=4, nt=1024, max_seq_len=4096). `nt` = `max_num_tokens`.

### Summary

Total: **14 configurations x 3 runs = 42 benchmark runs**.

## Impact Projection Matrix

Scenario-based projection using measured S2/S3 baselines for Qwen 72B (TP=8). Shows both 1st and 2nd+ instance costs to reflect the "first pays upfront, rest benefit" property of MX and GMS.

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

## Benchmark Results (v2)

**Environment:** 8x NVIDIA B300 SXM6 AC (275 GB each), NFS-backed storage, CUDA 13.1, TRT-LLM 1.3.0rc11
**Contract:** `first_request_ready` — profile finalized after first successful end-to-end request
**Statistical protocol:** 3 runs per configuration; **representative-run** approach (the run whose total startup is the median is selected, and all per-component metrics are reported from that single run so components sum consistently to the total)
**Total profiles collected:** 62 across 21 configurations
**Note:** S1 (remote cold) downloads used an internal HF CDN/mirror achieving ~2 GB/s. Public cloud download times will be significantly longer.

### Part 1 — Model Size Scaling (S1, Remote Cold Download)

Fresh HF download to `/tmp` (tmpfs) each run. All times in seconds (representative run).

| Metric | B1: Qwen 7B (TP=1) | B3: DS 7B (TP=1) | B2: Qwen 72B (TP=8) | B4: DS 70B (TP=8) |
|:-------|----:|----:|----:|----:|
| **Total startup** | **36.1** | **38.2** | **93.4** | **95.5** |
| HF remote download | 6.4 | 6.3 | 43.9 | 43.0 |
| Cached model loader | 6.4 | 6.3 | 63.5 | 62.5 |
| Weight loading total | 5.5 | 7.5 | 8.7 | 10.5 |
| — Checkpoint prefetch | 2.2 | 4.2 | 3.5 | 5.0 |
| — Apply weights | 2.8 | 2.8 | 3.8 | 3.7 |
| Warmup (1st pass) | 4.9 | 4.9 | 12.1 | 13.0 |
| — Autotuner forward | 4.6 | 4.6 | 11.3 | 12.3 |
| — CUDA graphs | 0.1 | 0.1 | 0.6 | 0.5 |
| Warmup (2nd pass) | 1.7 | 1.7 | 4.1 | 4.0 |

### Part 2 — Storage Tier Comparison (72B/70B Models, TP=8)

S1 = remote cold download, S2 = NFS cold (fresh inode copy per run), S3 = NFS warm (page cache hot).

| Metric | Qwen 72B S1 | Qwen 72B S2 | Qwen 72B S3 | DS 70B S1 | DS 70B S2 | DS 70B S3 |
|:-------|----:|----:|----:|----:|----:|----:|
| **Total startup** | **93.4** | **114.4** | **50.2** | **95.5** | **146.1** | **52.9** |
| Model loader (server) | 63.5 | 0.003 | 0.002 | 62.5 | 0.003 | 0.001 |
| Checkpoint prefetch | 3.5 | 65.0 | 3.4 | 5.0 | 99.2 | 6.0 |
| Apply weights | 3.8 | 4.3 | 3.6 | 3.7 | 3.6 | 3.6 |
| Warmup (1st pass) | 12.1 | 12.4 | 11.9 | 13.0 | 13.0 | 12.5 |
| Warmup (2nd pass) | 4.1 | 4.2 | 4.1 | 4.0 | 4.0 | 4.0 |

### Part 3 — Autotuner Impact (Group C)

Comparing autotuner ON vs OFF. Warmup component shift when autotuner is disabled.

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

### Part 4 — Serving Config Sensitivity (Group D)

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

For large models (70–72B), the dominant bottleneck shifts based on storage tier:

- **S1 (remote cold):** HF download is 43–44s (46% of total). The internal CDN makes this faster than expected; public cloud would be significantly worse.
- **S2 (NFS cold):** Checkpoint prefetch from cold NFS is the slowest path at 65–99s (57–68% of total). Cold NFS reads without page cache are slower than the CDN download.
- **S3 (warm cache):** Page cache eliminates I/O bottleneck entirely. Prefetch drops to 3–6s, yielding **50–53s total** — a 2–3x improvement over S2.

The weight *application* phase (`apply_weights`) is constant at 3.6–4.3s regardless of storage tier, confirming it's GPU-bound, not I/O-bound.

#### 2. Warmup Is the Irreducible Floor

With warm cache (S3), warmup dominates the remaining startup time:

| Component | Qwen 72B S3 | % of Total |
|:----------|------------:|-----------:|
| Weight loading | 8.9s | 17.7% |
| Warmup (1st pass) | 11.9s | 23.7% |
| Warmup (2nd pass) | 4.1s | 8.2% |
| Executor overhead | ~21s | ~42% |
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
