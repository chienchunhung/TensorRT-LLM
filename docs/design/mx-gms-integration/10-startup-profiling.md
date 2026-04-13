# 10. Startup Performance Profiling

[< Back to Overview](README.md)

**Status:** Implemented on branch `dynamo/startup-profiling`
**Last Updated:** 2026-04-10

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

## Benchmark Plan (v2)

See [startup-benchmark-plan-v2.md](startup-benchmark-plan-v2.md) for the full revised test matrix, including model selection, storage tiers, statistical protocol, and impact projection.

**Previous results** from the initial profiling runs (2026-04-10) are preserved below for reference. They will be superseded by v2 results once the 70-run matrix completes.

<details>
<summary>Previous Results (2026-04-10, initial profiling)</summary>

### Model Size Scaling (old Group B)

| Phase | DeepSeek 1.5B (3GB) | Llama 8B (16GB) | DeepSeek-V3-Lite (53GB) |
|:------|:----------------------|:-------------------|:---------------------------|
| **Total executor startup** | **49.1s** | **47.1s** | **114.2s** |
| Weight loading total | 19.6s (40%) | 38.3s (81%) | 79.3s (69%) |
| Warmup total (1st pass) | 25.0s (51%) | 6.1s (13%) | 31.1s (27%) |

### Remote-Cold Download (old Group G)

| Phase | G1: 1.5B Remote-Cold | G3: 72B Remote-Cold |
|:------|:---------------------|:--------------------|
| **Total startup** | **38.4s** | **96.4s** |
| `llm.hf.remote_download` | 3.4s | 44.0s |
| Weight loading (worker) | 2.9s | 8.7s |
| Warmup (1st pass) | 7.4s | 14.7s |

### Old Summary Table

| ID | Model | Config | Executor Total | Weight Load | Warmup (1st) | Dominant Bottleneck |
|:---|:------|:-------|:--------------|:-----------|:------------|:-------------------|
| B1 | DeepSeek 1.5B | TP=1,bs=4,nt=1024 | 49.1s | 19.6s (40%) | 25.0s (51%) | Warmup/autotuner |
| B2 | Llama 8B | TP=1,bs=4,nt=1024 | 47.1s | 38.3s (81%) | 6.1s (13%) | Weight loading |
| B3 | DeepSeek-V3-Lite 53GB | TP=1,bs=4,nt=1024 | 114.2s | 79.3s (69%) | 31.1s (27%) | Weight loading |
| G1 | DeepSeek 1.5B (HF remote-cold) | TP=1,bs=4,nt=1024 | 14.1s | 2.9s (21%) | 7.4s (53%) | Warmup/autotuner |
| G3 | Qwen2.5-72B (HF remote-cold) | TP=8,bs=4,nt=1024 | 75.7s | 8.7s (11%) | 14.7s (19%) | HF remote download |

</details>

---

## Benchmark Results (v2)

**Environment:** NVIDIA B300 SXM6 AC (275 GB), 8x GPUs available, PyTorch backend
**Contract:** `first_request_ready` — profile finalized after first successful end-to-end request
**Statistical protocol:** 5 runs per configuration; report median (min-max)

*Results pending — will be populated after executing the 70-run benchmark matrix.*

---

## MX+GMS Impact Projection

Scenario-based projection showing both first-instance and second-instance costs. The "first pays upfront, rest benefit" property of MX and GMS is reflected explicitly.

Baseline uses **B2-S1 (Qwen 72B remote cold) median** once measured.

| Scenario | 1st Instance Weight Load | 2nd+ Instance Weight Load | Warmup (each) | Notes |
|----------|--------------------------|---------------------------|---------------|-------|
| 1. Baseline (no MX, no GMS) | Full storage I/O (measured) | Full storage I/O (measured) | Full (measured) | Every instance pays full cost |
| 2. MX only (no GMS) | ~15s (P2P from donor node) | ~15s (P2P again) | Full (measured) | Each instance fetches independently via MX |
| 3. GMS only (no MX) | Full storage I/O (measured) | ~0.1s (zero-copy) | Full (measured) | 1st pays storage cost; 2nd+ near-free on same node |
| 4. MX + GMS | ~15s (P2P from donor node) | ~0.1s (zero-copy) | Full (measured) | 1st cheaper via MX; 2nd+ near-free via GMS |
| 5. MX + GMS + compile cache | ~15s (P2P) | ~0.1s (zero-copy) | ~2s (cached) | Best case for all replicas |

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
