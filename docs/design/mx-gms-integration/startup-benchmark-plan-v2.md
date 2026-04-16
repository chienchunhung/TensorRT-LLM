# Revised Startup Benchmark Test Plan (v2)

**Status:** Completed (2026-04-16). 62 profiles across 21 configurations. Results in [10-performance.md](10-performance.md#benchmark-results-v2).

## Model Matrix (Group B)

| ID | Series | Variant | HF Repo | Approx Size | TP |
|----|--------|---------|---------|-------------|-----|
| B1 | Qwen | Small | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | 1 |
| B2 | Qwen | Large | `Qwen/Qwen2.5-72B-Instruct` | ~145 GB | 8 |
| B3 | DeepSeek | Small | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14 GB | 1 |
| B4 | DeepSeek | Large | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | ~131 GB | 8 |

Default serving config: `max_batch_size=4, max_num_tokens=1024, max_seq_len=4096`.

## Storage Tier Matrix (Group S)

Three tiers measuring different weight-loading scenarios:

| Tier | ID | What it measures | Setup |
|------|----|-----------------|-------|
| Remote cold | S1 | Full HF download over network + model load | Isolated empty HF cache on `/tmp` (tmpfs); every run downloads from scratch |
| NFS cold | S2 | NFS file read (no page cache) + model load | Model files pre-copied to a **fresh NFS directory per run** (new inodes), guaranteeing cold page cache without needing `drop_caches` privileges |
| Local warm | S3 | Page-cache-warm file read + model load | Model files on NFS, page cache hot from a prior run; simulates 2nd replica on the same node |

**Default tier: S1 (remote cold).**

### S2 Methodology: Ensuring True NFS-Cold Reads

The Linux page cache is keyed on `(device, inode)`. Once a file is read, subsequent reads of the same inode are served from RAM regardless of the file path. This means:
- Simply pointing at the same NFS directory for multiple runs would make runs 2+ effectively warm (S3).
- Dropping page cache (`echo 3 > /proc/sys/vm/drop_caches`) requires root/sysctl privileges that may not be available on shared nodes.

To guarantee cold NFS reads without special privileges, each S2 run:
1. Copies the model to a **new directory** (`_s2_nfs_cold/<model>_runN/`) using `cp -rL`, creating fresh inodes.
2. Serves from the new directory.
3. Cleans up the previous run's copy to keep disk usage at ~1x model size.

The copy time is **not** included in the benchmark measurement.

## Statistical Protocol

- Each configuration runs **3 times**.
- Report: **median**, **min**, **max** for each profiled phase.
- Automate via `run_startup_bench.sh` (single config) and `run_startup_bench_all.sh` (full matrix).
- Post-process with `aggregate_startup_results.py` for median/min/max extraction.

## Test Matrix

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

### Part 3: Autotuner Impact (Group C, large models, S1)

| Test ID | Model | Tier | TP | Autotuner | Runs | Purpose |
|---------|-------|------|----|-----------|------|---------|
| B2-S1-C | Qwen 72B | S1 | 8 | OFF | 3 | Isolate autotuner cost (Qwen) |
| B4-S1-C | DeepSeek 70B | S1 | 8 | OFF | 3 | Isolate autotuner cost (DeepSeek) |

Compare against B2-S1 and B4-S1 (autotuner ON by default).

### Part 4: Serving Config Sensitivity (Group D, large models, S1)

| Test ID | Model | Tier | TP | Config | Runs | Purpose |
|---------|-------|------|----|--------|------|---------|
| B2-S1-D1 | Qwen 72B | S1 | 8 | bs=64, nt=8192 | 3 | Large batch + token budget |
| B4-S1-D1 | DeepSeek 70B | S1 | 8 | bs=64, nt=8192 | 3 | Large batch + token budget |
| B2-S1-D2 | Qwen 72B | S1 | 8 | max_seq_len=16384 | 3 | Long-sequence KV cache impact |
| B4-S1-D2 | DeepSeek 70B | S1 | 8 | max_seq_len=16384 | 3 | Long-sequence KV cache impact |

Compare against B2-S1 and B4-S1 (default bs=4, nt=1024, max_seq_len=4096).

`nt` = `max_num_tokens` (max tokens per executor iteration).

### Summary

Total: **14 configurations x 3 runs = 42 benchmark runs**.

## Impact Projection Matrix

Scenario-based projection using **B2-S1 median** as measured baseline. Shows both 1st and 2nd+ instance costs to reflect the "first pays upfront, rest benefit" property of MX and GMS.

| Scenario | 1st Instance Weight Load | 2nd+ Instance Weight Load | Warmup (each) | Notes |
|----------|--------------------------|---------------------------|---------------|-------|
| 1. Baseline (no MX, no GMS) | Full storage I/O (measured) | Full storage I/O (measured) | Full (measured) | Every instance pays full cost |
| 2. MX only (no GMS) | ~15s (P2P from donor node) | ~15s (P2P again) | Full (measured) | Each instance fetches via MX independently |
| 3. GMS only (no MX) | Full storage I/O (measured) | ~0.1s (zero-copy) | Full (measured) | 1st pays storage; 2nd+ near-free on same node |
| 4. MX + GMS | ~15s (P2P from donor node) | ~0.1s (zero-copy) | Full (measured) | 1st cheaper via MX; 2nd+ near-free via GMS |
| 5. MX + GMS + compile cache | ~15s (P2P) | ~0.1s (zero-copy) | ~2s (cached) | Best case for all replicas |
