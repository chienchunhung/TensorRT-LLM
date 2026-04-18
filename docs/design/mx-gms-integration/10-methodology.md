# 10. Methodology & Test Plan

[< Back to Overview](README.md)

**Status:** Profiling framework implemented on branch `dynamo/startup-profiling`
**Last Updated:** 2026-04-17

This section covers the measurement framework, test scenarios, and test matrix used to generate the baseline startup data. Measured results and their analysis live in [§11 Results & Analysis](11-results-analysis.md).

---

## Part A — Profiling Framework

### Overview

A hierarchical startup profiling framework is implemented in TensorRT-LLM. It measures cold-start timing across the entire `trtllm-serve` bring-up path, from CLI argument parsing to the first successful inference request.

Startup completion is defined by the **first-request-ready** contract: the profile is finalized only after the first successful end-to-end request completes. This ensures the reported total includes all warmup, compilation, and CUDA graph capture work that happens lazily during the first forward pass.

### Profiler Core

The profiler lives at `tensorrt_llm/llmapi/startup_profiler.py`. It provides:

- `StartupProfiler`: singleton, thread-safe, hierarchical timer tree
- `startup_timer(name, **metadata)`: context-manager shorthand for adding a phase
- `get_startup_profiler()`: process-global singleton access
- Zero overhead when disabled (all timers are no-ops)

Enabled via environment variable:

```bash
export TRTLLM_PROFILE_STARTUP=1
```

Optional JSON file output:

```bash
export TRTLLM_STARTUP_PROFILE_OUTPUT=/path/to/startup_profile.json
```

### Instrumented Components

The profiler instruments the full server startup path across two processes (main server + executor worker):

```
trtllm-serve CLI
├── serve.get_llm_args                      # CLI arg parsing
├── serve.parse_metadata_server_config      # Config file loading
├── serve.create_llm                        # LLM object construction
│   ├── llm.parse_args                      # Pydantic args validation
│   ├── llm.init_mpi_session                # MPI session setup
│   ├── llm.build_model                     # Model loading orchestration
│   │   ├── llm.cached_model_loader         # HF download / cache resolution
│   │   ├── llm.load_tokenizer              # Tokenizer initialization
│   │   ├── llm.load_hf_model_config        # HF config.json loading
│   │   ├── llm.load_generation_config      # generation_config.json
│   │   ├── llm.create_input_processor      # Multimodal input processor
│   │   └── llm.create_executor             # Executor creation (spawns worker)
│   └── llm.init_tracing                    # OTLP tracer init
├── serve.create_openai_server              # OpenAI server construction
└── server.lifespan_startup                 # ASGI lifespan (metadata, energy, stats)
```

Inside the executor worker process (reported via `attached_profiles.executor.ranks[]`):

```
executor_worker.initialize
├── executor.load_config_and_apply_defaults
├── executor.create_model_engine.main
│   └── executor.load_model_weights
│       ├── executor.load_model_config          # Config loading + validation
│       ├── executor.model_init.meta            # Model object construction (meta tensors)
│       ├── executor.materialize_model_tensors  # Meta → CUDA tensor allocation
│       ├── executor.move_model_to_cuda         # model.to("cuda") finalization
│       ├── executor.checkpoint_read.main_weights
│       │   ├── executor.checkpoint_discovery   # glob for .safetensors files
│       │   ├── executor.checkpoint_prefetch    # Disk prefetch to page cache
│       │   └── executor.checkpoint_parallel_load  # Parallel safetensor deserialization
│       ├── executor.weight_mapper_init.main_weights
│       ├── executor.apply_model_weights.main_weights  # model.load_weights()
│       ├── executor.post_load_weights          # Per-module post-load hooks
│       ├── executor.moe_finalize_model         # MoE load balancer finalization
│       └── executor.weight_load_cuda_sync      # Final CUDA synchronization
├── executor.create_sampler
├── executor.create_kv_cache
├── executor.create_py_executor_instance
│   └── executor.warmup.main_model
│       ├── executor.warmup.torch_compile       # torch.compile specialization
│       ├── executor.warmup.autotuner           # Autotuner kernel selection
│       │   ├── executor.warmup.autotuner.setup_state
│       │   ├── executor.warmup.autotuner.create_request
│       │   ├── executor.warmup.autotuner.acquire_batch
│       │   ├── executor.warmup.autotuner.forward    # Synthetic forward pass
│       │   ├── executor.warmup.autotuner.cache_exchange
│       │   └── executor.warmup.autotuner.cuda_sync
│       ├── executor.warmup.cuda_graphs         # CUDA graph capture
│       └── executor.warmup.memory_pool         # Memory pool warmup
├── executor.configure_kv_cache_capacity
├── executor.rebuild_kv_cache
├── executor.recreate_py_executor_instance      # Second pass with final KV cache
│   └── executor.warmup.main_model (repeat)
└── executor.start_worker
```

### Warmup and Compilation Semantics (What Each Component Means)

The following clarifies what each warmup-related timer actually measures and why it costs time.

- `executor.warmup.torch_compile`
  - **What it does:** runs shape-specialization warmup for the model path when torch compile is enabled.
  - **What it is not:** this is not "compile to a standalone executable binary". It is runtime graph/kernel specialization and caching for the current workload shapes.
  - **Primary cost drivers:** first-time graph specialization, codegen/JIT overhead, and initial kernel materialization.
  - **Code reference:** `tensorrt_llm/_torch/pyexecutor/model_engine.py` (`PyTorchModelEngine._run_torch_compile_warmup`, `PyTorchModelEngine._general_warmup`).

- `executor.warmup.autotuner` (especially `executor.warmup.autotuner.forward`)
  - **What it does:** executes a synthetic forward pass under autotune mode so candidate kernels/configs can be profiled and selected.
  - **Why it costs time:** the forward pass runs real GPU compute and may include distributed communication (for multi-GPU), plus tuner bookkeeping and cache population.
  - **Code reference:** `tensorrt_llm/_torch/pyexecutor/model_engine.py` (`PyTorchModelEngine._run_autotuner_warmup`).

- `executor.warmup.cuda_graphs`
  - **What it does:** captures generation execution into CUDA Graphs for selected batch sizes (and draft-length variants when applicable), so later iterations replay a pre-captured graph with lower launch overhead.
  - **How it differs from torch compile/autotune:** this is graph-capture/replay optimization of launch behavior, not kernel-choice search (`autotuner`) and not high-level graph specialization (`torch_compile`).
  - **Why it runs after compile/autotune:** capture should record the already-specialized/selected execution path to avoid capturing suboptimal pre-warmup behavior.
  - **Code reference:** `tensorrt_llm/_torch/pyexecutor/model_engine.py` (`PyTorchModelEngine._run_cuda_graph_warmup`, `PyTorchModelEngine._capture_generation_cuda_graphs`).

- `executor.warmup.memory_pool`
  - **What it does:** runs additional general warmup requests to pre-touch allocator paths and reduce runtime memory fragmentation.
  - **Clarification:** this is not the same as creating KV cache capacity; KV cache manager creation/rebuild happens in `executor.create_kv_cache`, `executor.configure_kv_cache_capacity`, and `executor.rebuild_kv_cache`.
  - **Code reference:** `tensorrt_llm/_torch/pyexecutor/model_engine.py` (`PyTorchModelEngine.warmup`, `PyTorchModelEngine._general_warmup`), plus KV cache setup in `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` (`KvCacheCreator.build_managers`, `configure_kv_cache_capacity` flow).

- `executor.recreate_py_executor_instance` (second-pass warmup)
  - **What it does:** when KV cache capacity is first estimated using a temporary/minimal setup, the executor is rebuilt with final KV cache sizing, then warmup runs again on the final runtime state.
  - **Why it is needed:** first pass establishes sizing/profiling context; second pass ensures compile/capture/warmup artifacts match the final KV cache and executor resource layout used in serving.
  - **Code reference:** `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` (`estimating_kv_cache`, first `create_py_executor_instance`, `configure_kv_cache_capacity`, manager teardown/rebuild, second `create_py_executor_instance`).

### Data Flow

```
┌─────────────┐    startup_timer()     ┌──────────────────┐
│ Server Proc │ ──────────────────────>│ StartupProfiler  │
│ (rank 0)    │                        │ (main process)   │
└─────────────┘                        └────────┬─────────┘
                                                │
┌─────────────┐    startup_timer()     ┌────────┴─────────┐
│ Worker Proc │ ──────────────────────>│ StartupProfiler  │
│ (MPI rank)  │                        │ (worker process) │
└──────┬──────┘                        └────────┬─────────┘
       │                                        │
       │  MPI gather + IPC ready_msg            │
       └────────────────────────────────────────┘
                        │
                ┌───────┴────────┐
                │ Merged Profile │──> /startup_metrics endpoint
                │  (JSON dict)   │──> TRTLLM_STARTUP_PROFILE_OUTPUT file
                └────────────────┘──> Server log summary
```

### How to Run the Startup Benchmark

#### Step 1: Start the Server with Profiling Enabled

```bash
TRTLLM_PROFILE_STARTUP=1 \
TRTLLM_STARTUP_PROFILE_OUTPUT=/tmp/startup_profile.json \
trtllm-serve <model_path_or_hf_id> \
    --backend pytorch \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor_parallel_size 1 \
    --max_batch_size 4 \
    --max_num_tokens 1024 \
    --max_seq_len 4096
```

#### Step 2: Run the Benchmark Client

```bash
python tensorrt_llm/serve/scripts/benchmark_serving.py \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --model <model_name> \
    --tokenizer <model_path_or_hf_id> \
    --dataset-name random --random-ids \
    --num-prompts 1 \
    --random-input-len 16 --random-output-len 8 \
    --request-rate inf \
    --save-result \
    --save-startup-metrics \
    --result-dir /tmp
```

Key flags:
- `--save-startup-metrics`: enables the startup benchmark flow (wait for reachability, probe, fetch `/startup_metrics`, write artifacts)
- `--startup-timeout 600`: max seconds to wait for server to become reachable (default 600)

#### Configuration Summary

| Setting | Environment Variable | Default |
|:--------|:--------------------|:--------|
| Enable profiling | `TRTLLM_PROFILE_STARTUP=1` | `0` (disabled) |
| JSON file output | `TRTLLM_STARTUP_PROFILE_OUTPUT=<path>` | None (no file) |
| Benchmark flag | `--save-startup-metrics` | Not set |
| Benchmark timeout | `--startup-timeout <seconds>` | `600` |

#### Output Artifacts

The benchmark produces three artifacts:

| Artifact | Filename Pattern | Content |
|:---------|:----------------|:--------|
| Main benchmark JSON | `openai-infqps-<model>-<timestamp>.json` | Throughput results + embedded `startup_metrics` + `startup_summary` |
| Startup JSON | `*-startup_metrics.json` | Full hierarchical profiler tree with per-rank executor data |
| Startup Markdown | `*-startup_metrics-summary.md` | SSH-friendly indented tree with durations and percentages |

Additionally, if `TRTLLM_STARTUP_PROFILE_OUTPUT` is set, the server itself writes the raw profile to that path upon first-request finalization.

#### Reading the Results

The **Markdown summary** is the quickest way to inspect results over SSH:

```
cat /tmp/openai-infqps-bf16-20260407-094644-startup_metrics-summary.md
```

The **JSON artifact** contains:
- `schema_version`: integer for forward compatibility
- `completed`: boolean, true only after first successful request
- `total_duration_s`: wall-clock time from profiler creation to completion
- `records[]`: hierarchical tree of `{name, start_offset_s, duration_s, metadata, children[]}`
- `metadata`: server-level info (model, host, port, startup_contract, first_request_id)
- `attached_profiles.executor.ranks[]`: per-rank executor worker profiles (same tree schema)

### Schema Reference

The startup profile JSON follows schema version 1:

```json
{
  "schema_version": 1,
  "enabled": true,
  "completed": true,
  "total_duration_s": 125.623,
  "records": [
    {
      "name": "serve.create_llm",
      "start_offset_s": 0.000,
      "duration_s": 125.122,
      "metadata": {"backend": "pytorch"},
      "children": [...]
    }
  ],
  "metadata": {
    "server_type": "openai",
    "model": "bf16",
    "startup_contract": "first_request_ready",
    "startup_status": "completed",
    "first_request_id": 2
  },
  "attached_profiles": {
    "executor": {
      "ranks": [
        {
          "schema_version": 1,
          "total_duration_s": 109.443,
          "records": [...],
          "metadata": {"backend": "pytorch", "mpi_rank": 0}
        }
      ]
    }
  },
  "pid": 12345
}
```

The benchmark result JSON includes additional fields:

```json
{
  "startup_contract": "first_request_ready",
  "startup_probe": {
    "success": true,
    "attempts": 1,
    "request_latency_s": 0.097,
    "completed_since_start_s": 112.439
  },
  "startup_reachability": {
    "path": "/health",
    "status_code": 200,
    "reachable_since_start_s": 112.341
  },
  "startup_metrics": { ... },
  "startup_summary": {
    "serve.create_llm": 125.122,
    "serve.create_openai_server": 0.270
  }
}
```

### How to Read the Results Tables

The startup path is **strictly serial** across two processes:

1. **Server process** — parses CLI args, creates MPI pool (for TP>1), runs `CachedModelLoader` (downloads/resolves the model), then calls `create_executor` which dispatches `worker_main` to the MPI pool.
2. **Executor worker process** — receives `worker_main` dispatch (only AFTER the server's model loader completes), then runs the full initialization: config loading, model construction (meta tensors), tensor materialization, checkpoint reading, weight application, warmup. Signals ready when done.

**Key: download and worker initialization are sequential.** The `create_executor` call happens AFTER `CachedModelLoader` completes — there is no concurrent worker initialization during download. (Verified via code: `llm.py:1284` runs `_build_model()` → download, then `llm.py:1309` calls `create_executor` → dispatches `worker_main`.)

The result tables in [§11](11-results-analysis.md) show a hierarchical timer tree. Indented rows (prefixed with `└─` or `├─`) are **children** of the row above — their time is already included in the parent. Only top-level (non-indented) rows are additive. For example, "HF remote download" is *inside* "Cached model loader", not separate from it.

---

## Part B — Benchmark Scenarios

### Test 1: Cold-Start Latency (Baseline Profiling) — Completed

**What:** Time from process start to first successful inference under the standard HF weight-loading path (no MX, no GMS). Establishes the baseline startup breakdown across model sizes, storage tiers, autotuner settings, and serving configurations.

**Status:** Completed on v3 (current rebased codebase `upstream/main @ 4a848ccce`, node `umb-b300-dp-186`). 14 configurations × 3 runs = **42 profiles**. Earlier v2 dataset (pre-PR #12407, different node) is preserved as reference. See [§11 Results & Analysis](11-results-analysis.md).

**Configurations:**

| Config | `checkpoint_format` | `load_format` | Status |
|:-------|:-------------------|:-------------|:-------|
| Baseline (HF/AUTO) | `HF` (default) | `AUTO` (default) | **Completed** |
| MX (2nd replica) | `MX` | `AUTO` | Not yet tested (requires MX integration) |
| GMS (2nd worker) | `HF` | `GMS` | Not yet tested (requires GMS integration) |
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

### Test 4: Shadow Failover Latency — Partially Executed

**What:** Time from primary crash to shadow serving first request. Full end-to-end test requires GMS + executor failover integration. Measurable today: floor (Test 4a) and ceiling (cold-restart cost = Test 1 S2/S3).

**End-to-end protocol (requires GMS, not yet runnable):**
1. Start primary + shadow with GMS
2. Send warmup requests to primary (this also populates compile cache)
3. Kill primary (`kill -9`)
4. Measure time until shadow returns first response
5. **Target:** < 5s

**Critical dependency:** The <5s target assumes compile cache (disk or GMS-backed) is warm. Without compile cache, warmup adds ~16s (v2) or ~43s (v3, post-PR #12407) to activation, making failover exceed the budget. Since primary and shadow are co-located on the same node and share the filesystem, disk-based compile cache is sufficient for Phase 2. See [§07 Tiered Compile Cache](07-compile-cache.md).

#### Test 4a: Hot-Server First-Request Latency Floor — Completed

**What:** Establishes the lower bound on activation latency by measuring how fast a fully warm server responds to a single request. This is the absolute floor that no failover scheme can beat — useful as the "target ceiling" for GMS+compile_cache.

**Protocol:**
1. Start `trtllm-serve`, wait until `Application startup complete`
2. Send 1 warmup request (results discarded, populates lazy caches)
3. Send 10 measurement requests (input=16, output=8 tokens)
4. Capture per-request **TTFT** (Time-To-First-Token, via streaming API) and **E2E latency**
5. Repeat 3× per config; take median-of-medians (matches Test 1 representative-run protocol)

**Configs run:** Same 4 models as Test 1 Part 1 (Qwen 7B/72B, DS 7B/70B), all on S3 (storage tier doesn't affect steady-state response time).

**Status:** Completed (2026-04-17). Results in [§11 Results & Analysis](11-results-analysis.md#test-4a-failover-latency-floor).

#### Test 4b: Cold-Restart Failover Cost — Reuses Test 1 Data

**What:** Today's failover cost = full cold restart of a new process. No special test needed — this is exactly what Test 1 measures (Part 1 S2 = "cold restart on a node with model on NFS"; Part 2 S3 = "cold restart with page cache warm from prior load"). The S3 number is the realistic upper bound on failover today (a fresh process on the same node, page cache warm from the dead primary).

**Status:** Implicit in Test 1 results. Qwen 72B TP=8: ~75s S3 / ~306s S2.

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

## Part C — Baseline Profiling Test Matrix (Test 1)

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

### Part 1: Model Size Scaling (S1 remote cold, default config) — v2 only

S1 was measured on v2 only (HF rate-limiting concerns + storage story already validated by v3 S3).

| Test ID | Model | Tier | TP | Runs | Status |
|---------|-------|------|----|------|--------|
| B1-S1 | Qwen 7B | S1 | 1 | 3 | **Done (v2)** |
| B2-S1 | Qwen 72B | S1 | 8 | 3 | **Done (v2)** |
| B3-S1 | DeepSeek 7B | S1 | 1 | 3 | **Done (v2)** |
| B4-S1 | DeepSeek 70B | S1 | 8 | 3 | **Done (v2)** |

### Part 2: Storage Tier Comparison (large models) — v3 primary

| Test ID | Model | Tier | TP | Runs | v2 | v3 |
|---------|-------|------|----|------|----|----|
| B1-S2 | Qwen 7B | S2 | 1 | 3 | — | **Done** |
| B2-S2 | Qwen 72B | S2 | 8 | 3 | **Done** | **Done** |
| B2-S3 | Qwen 72B | S3 | 8 | 3 | **Done** | **Done** |
| B3-S2 | DeepSeek 7B | S2 | 1 | 3 | — | **Done** |
| B4-S2 | DeepSeek 70B | S2 | 8 | 3 | **Done** | **Done** |
| B4-S3 | DeepSeek 70B | S3 | 8 | 3 | **Done** | **Done** |

### Part 3: Autotuner Impact (large models, S2/S3) — v3 primary

Ran on S2 and S3 tiers (instead of S1 as originally planned) to isolate warmup behavior from download variability.

| Test ID | Model | Tier | TP | Autotuner | Runs | v2 | v3 |
|---------|-------|------|----|-----------|------|----|----|
| B2-S2-C | Qwen 72B | S2 | 8 | OFF | 3 | **Done** | **Done** |
| B2-S3-C | Qwen 72B | S3 | 8 | OFF | 3 | **Done** | **Done** |
| B4-S2-C | DeepSeek 70B | S2 | 8 | OFF | 3 | **Done** | **Done** |
| B4-S3-C | DeepSeek 70B | S3 | 8 | OFF | 3 | **Done** | **Done** |

Compare against B2-S2/S3 and B4-S2/S3 (autotuner ON by default).

### Part 4: Serving Config Sensitivity (large models, S3) — v3 primary

Ran on S3 (warm cache) to isolate serving config impact from I/O variability.

| Test ID | Model | Tier | TP | Config | Runs | v2 | v3 |
|---------|-------|------|----|--------|------|----|----|
| B2-S3-D1 | Qwen 72B | S3 | 8 | bs=64, nt=8192 | 3 | **Done** | **Done** |
| B4-S3-D1 | DeepSeek 70B | S3 | 8 | bs=64, nt=8192 | 3 | **Done** | **Done** |
| B2-S3-D2 | Qwen 72B | S3 | 8 | max_seq_len=16384 | 3 | **Done** | **Done** |
| B4-S3-D2 | DeepSeek 70B | S3 | 8 | max_seq_len=16384 | 3 | **Done** | **Done** |

Compare against B2-S3 and B4-S3 (default bs=4, nt=1024, max_seq_len=4096). `nt` = `max_num_tokens`.

### Summary

**Completed (v3, primary dataset):** 14 configurations × 3 runs = **42 benchmark profiles** (Parts 2–4 plus B1-S2 / B3-S2).
**Completed (v2, reference dataset):** 21 configurations × 3 runs = **62 benchmark profiles** (Parts 1–4).

**Not yet run (requires MX/GMS implementation):** Tests 2–6 (P2P throughput, memory overhead, shadow failover, throughput regression, vLLM comparison).

---

## Performance Regression Detection

Template for a CI-integrated regression test, once MX and GMS scenarios are implemented:

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
