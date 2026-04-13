# Revised Startup Benchmark Test Plan (v2)

## Model Matrix (Group B)

| ID | Series | Variant | HF Repo | Approx Size | TP |
|----|--------|---------|---------|-------------|-----|
| B1 | Qwen | Small | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | 1 |
| B2 | Qwen | Large | `Qwen/Qwen2.5-72B-Instruct` | ~145 GB | 8 |
| B3 | DeepSeek | Small | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14 GB | 1 |
| B4 | DeepSeek | Large | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | ~131 GB | 8 |

Default serving config: `max_batch_size=4, max_num_tokens=1024, max_seq_len=4096`.

## Storage Tier Matrix (Group S)

| Tier | ID | Description | Setup |
|------|----|-------------|-------|
| Remote cold | S1 | Fresh download from HF, no local cache | Isolated empty `HF_HOME` + `HUGGINGFACE_HUB_CACHE` |
| NFS cache | S2 | Model files on NFS, cold page cache | Existing NFS path; drop page cache before run |
| Local node cache | S3 | Model files hot in OS page cache | Run after a prior load (2nd replica scenario) |

**Default tier: S1 (remote cold).**

## Statistical Protocol

- Each configuration runs **5 times**.
- Report: **median**, **min**, **max** for each profiled phase.

## Test Matrix

### Part 1: Model Size Scaling (S1 remote cold, default config)

| Test ID | Model | Tier | TP | Runs | Purpose |
|---------|-------|------|----|------|---------|
| B1-S1 | Qwen 7B | S1 | 1 | 5 | Small baseline (Qwen) |
| B2-S1 | Qwen 72B | S1 | 8 | 5 | Large baseline (Qwen) |
| B3-S1 | DeepSeek 7B | S1 | 1 | 5 | Small baseline (DeepSeek) |
| B4-S1 | DeepSeek 70B | S1 | 8 | 5 | Large baseline (DeepSeek) |

### Part 2: Storage Tier Comparison (large models only)

| Test ID | Model | Tier | TP | Runs | Purpose |
|---------|-------|------|----|------|---------|
| B2-S2 | Qwen 72B | S2 | 8 | 5 | NFS cold page cache |
| B2-S3 | Qwen 72B | S3 | 8 | 5 | Warm page cache (2nd replica) |
| B4-S2 | DeepSeek 70B | S2 | 8 | 5 | NFS cold page cache |
| B4-S3 | DeepSeek 70B | S3 | 8 | 5 | Warm page cache (2nd replica) |

### Part 3: Autotuner Impact (Group C, large models, S1)

| Test ID | Model | Tier | TP | Autotuner | Runs | Purpose |
|---------|-------|------|----|-----------|------|---------|
| B2-S1-C | Qwen 72B | S1 | 8 | OFF | 5 | Isolate autotuner cost (Qwen) |
| B4-S1-C | DeepSeek 70B | S1 | 8 | OFF | 5 | Isolate autotuner cost (DeepSeek) |

Compare against B2-S1 and B4-S1 (autotuner ON by default).

### Part 4: Serving Config Sensitivity (Group D, large models, S1)

| Test ID | Model | Tier | TP | Config | Runs | Purpose |
|---------|-------|------|----|--------|------|---------|
| B2-S1-D1 | Qwen 72B | S1 | 8 | bs=64, nt=8192 | 5 | Large batch + token budget |
| B4-S1-D1 | DeepSeek 70B | S1 | 8 | bs=64, nt=8192 | 5 | Large batch + token budget |
| B2-S1-D2 | Qwen 72B | S1 | 8 | max_seq_len=16384 | 5 | Long-sequence KV cache impact |
| B4-S1-D2 | DeepSeek 70B | S1 | 8 | max_seq_len=16384 | 5 | Long-sequence KV cache impact |

Compare against B2-S1 and B4-S1 (default bs=4, nt=1024, max_seq_len=4096).

`nt` = `max_num_tokens` (max tokens per executor iteration).

### Summary

Total: **14 configurations x 5 runs = 70 benchmark runs**.

## Impact Projection Matrix

Scenario-based projection using **B2-S1 median** as measured baseline. Shows both 1st and 2nd+ instance costs.

| Scenario | 1st Instance Weight Load | 2nd+ Instance Weight Load | Warmup (each) | Notes |
|----------|--------------------------|---------------------------|---------------|-------|
| 1. Baseline (no MX, no GMS) | Full storage I/O (measured) | Full storage I/O (measured) | Full (measured) | Every instance pays full cost |
| 2. MX only (no GMS) | ~15s (P2P from donor node) | ~15s (P2P again) | Full (measured) | Each instance fetches via MX independently |
| 3. GMS only (no MX) | Full storage I/O (measured) | ~0.1s (zero-copy) | Full (measured) | 1st pays storage; 2nd+ near-free on same node |
| 4. MX + GMS | ~15s (P2P from donor node) | ~0.1s (zero-copy) | Full (measured) | 1st cheaper via MX; 2nd+ near-free via GMS |
| 5. MX + GMS + compile cache | ~15s (P2P) | ~0.1s (zero-copy) | ~2s (cached) | Best case for all replicas |
