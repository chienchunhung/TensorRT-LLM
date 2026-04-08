# 10. Startup Performance Profiling

[< Back to Overview](README.md)

**Status:** Implemented on branch `dynamo/startup-profiling`
**Last Updated:** 2026-04-08

---

## Overview

A hierarchical startup profiling framework is now implemented in TensorRT-LLM. It measures cold-start timing across the entire `trtllm-serve` bring-up path, from CLI argument parsing to the first successful inference request.

Startup completion is defined by the **first-request-ready** contract: the profile is finalized only after the first successful end-to-end request completes. This ensures the reported total includes all warmup, compilation, and CUDA graph capture work that happens lazily during the first forward pass.

---

## Architecture

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

---

## How to Run the Startup Benchmark

### Step 1: Start the Server with Profiling Enabled

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

### Step 2: Run the Benchmark Client

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

### Configuration Summary

| Setting | Environment Variable | Default |
|:--------|:--------------------|:--------|
| Enable profiling | `TRTLLM_PROFILE_STARTUP=1` | `0` (disabled) |
| JSON file output | `TRTLLM_STARTUP_PROFILE_OUTPUT=<path>` | None (no file) |
| Benchmark flag | `--save-startup-metrics` | Not set |
| Benchmark timeout | `--startup-timeout <seconds>` | `600` |

### Output Artifacts

The benchmark produces three artifacts:

| Artifact | Filename Pattern | Content |
|:---------|:----------------|:--------|
| Main benchmark JSON | `openai-infqps-<model>-<timestamp>.json` | Throughput results + embedded `startup_metrics` + `startup_summary` |
| Startup JSON | `*-startup_metrics.json` | Full hierarchical profiler tree with per-rank executor data |
| Startup Markdown | `*-startup_metrics-summary.md` | SSH-friendly indented tree with durations and percentages |

Additionally, if `TRTLLM_STARTUP_PROFILE_OUTPUT` is set, the server itself writes the raw profile to that path upon first-request finalization.

### Reading the Results

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

---

## Example Results

### DeepSeek-V3-Lite BF16 (local checkpoint, single B300 GPU)

**Environment:**
- GPU: NVIDIA B300 SXM6 AC (275 GB)
- Model: `DeepSeek-V3-Lite/bf16` (53 GB checkpoint, local NFS)
- Backend: PyTorch, TP=1, max_batch_size=4, max_num_tokens=1024, max_seq_len=4096
- Contract: `first_request_ready`

**Server Process Tree:**

```
- serve.create_llm: 125.122s (99.6%)
  - llm.build_model: 125.121s (99.6%)
    - llm.cached_model_loader: 0.005s (0.0%)
    - llm.load_tokenizer: 0.313s (0.2%)
    - llm.create_executor: 124.765s (99.3%)
  - llm.init_tracing: 0.000s (0.0%)
- serve.create_openai_server: 0.270s (0.2%)
```

**Executor Worker Tree:**

```
- executor_worker.initialize: 109.443s (100.0%)
  - executor.create_model_engine.main: 69.731s (63.7%)
    - executor.load_model_weights: 69.714s (63.7%)
      - executor.load_model_config: 0.001s (0.0%)
      - executor.model_init.meta: 0.193s (0.2%)
      - executor.materialize_model_tensors: 0.015s (0.0%)
      - executor.checkpoint_read.main_weights: 1.688s (1.5%)
        - executor.checkpoint_prefetch: 1.682s (1.5%)
        - executor.checkpoint_parallel_load: 0.005s (0.0%)
      - executor.apply_model_weights.main_weights: 0.700s (0.6%)
      - executor.post_load_weights: 0.001s (0.0%)
  - executor.create_py_executor_instance: 35.856s (32.8%)
    - executor.warmup.main_model: 35.590s (32.5%)
      - executor.warmup.autotuner: 31.250s (28.6%)
        - executor.warmup.autotuner.forward: 31.249s (28.6%)
      - executor.warmup.cuda_graphs: 3.284s (3.0%)
      - executor.warmup.memory_pool: 0.339s (0.3%)
  - executor.configure_kv_cache_capacity: 0.408s (0.4%)
  - executor.recreate_py_executor_instance: 2.382s (2.2%)
  - executor.start_worker: 0.001s (0.0%)
```

**Key Findings:**
- Weight loading dominated by checkpoint prefetch + weight application (~70s total)
- Autotuner warmup is the second-largest bucket (~31s), almost entirely in the forward pass
- CUDA graph capture is a smaller but significant ~3.3s
- Server/config overhead is negligible (<1s)

### DeepSeek-R1-Distill-Qwen-1.5B (HuggingFace download, single B300 GPU)

**Environment:**
- GPU: NVIDIA B300 SXM6 AC (275 GB)
- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (3.3 GB, downloaded from HF)
- Backend: PyTorch, TP=1
- Contract: `first_request_ready`

**Executor Worker Tree:**

```
- executor_worker.initialize: 11.353s (100.0%)
  - executor.create_model_engine.main: 2.602s (22.9%)
    - executor.load_model_weights: 2.600s (22.9%)
      - executor.model_init.meta: 0.193s (1.7%)
      - executor.checkpoint_read.main_weights: 1.688s (14.9%)
        - executor.checkpoint_prefetch: 1.682s (14.8%)
        - executor.checkpoint_parallel_load: 0.005s (0.0%)
      - executor.apply_model_weights.main_weights: 0.700s (6.2%)
  - executor.create_py_executor_instance: 7.319s (64.5%)
    - executor.warmup.main_model: 7.041s (62.0%)
      - executor.warmup.autotuner: 6.718s (59.2%)
        - executor.warmup.autotuner.forward: 6.717s (59.2%)
      - executor.warmup.cuda_graphs: 0.170s (1.5%)
      - executor.warmup.memory_pool: 0.024s (0.2%)
```

**Key Findings:**
- For small models, warmup/autotuner dominates startup (59% vs 23% for weight loading)
- HF download itself (~3s) was captured under `llm.cached_model_loader` in the server process tree
- Checkpoint prefetch is the bottleneck within weight loading even for a 3.3 GB model

---

## Bottleneck Analysis

| Phase | DeepSeek-V3-Lite BF16 | DeepSeek-R1-Distill-1.5B | Bottleneck Type |
|:------|:---------------------|:------------------------|:----------------|
| **Weight Loading** | 69.7s (63.7%) | 2.6s (22.9%) | I/O-bound |
| **Warmup / Autotuner** | 35.6s (32.5%) | 7.0s (62.0%) | Compute-bound |
| **CUDA Graphs** | 3.3s (3.0%) | 0.2s (1.5%) | Compute-bound |
| **KV Cache** | 0.5s (0.4%) | 0.1s (0.4%) | Memory-bound |
| **Config / Server** | 0.6s (0.5%) | 0.3s (2.6%) | Negligible |

### Observations

1. **Weight loading scales with model size** — it is I/O-bound (disk prefetch + deserialization). MX/GMS targets this directly.
2. **Autotuner warmup scales with model complexity** — it runs a full forward pass for kernel tuning. For large MoE models this is very expensive.
3. **CUDA graph capture** is proportional to the number of batch sizes configured.
4. **Two-pass warmup**: the executor does warmup twice (once for KV cache estimation, once for the real KV cache). The second pass is much faster because autotuner results are cached.

### MX/GMS Impact on These Results

| Phase | Current | With MX (Phase 1) | With MX + GMS (Phase 3) |
|:------|:--------|:-------------------|:------------------------|
| Weight Loading (V3-Lite BF16) | 69.7s | ~15s (P2P transfer) | ~0.1s (zero-copy import) |
| Weight Loading (R1-Distill-1.5B) | 2.6s | ~1s (P2P transfer) | ~0.1s (zero-copy import) |
| Warmup / Autotuner | 35.6s / 7.0s | Same | Same (future: cache sharing) |
| CUDA Graphs | 3.3s / 0.2s | Same | Same (future: cache sharing) |

---

## Schema Reference

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
