# 1. High-Level Architecture

[< Back to Overview](README.md)

TensorRT-LLM is NVIDIA's open-source library for optimized LLM inference on NVIDIA GPUs. It provides a Python + C++ stack bridging user-facing APIs with high-performance GPU execution, supporting three backends that share a common C++ core.

## 1.1 Backend Overview

| Backend | Status | Entry Point | Description |
|:--------|:-------|:------------|:------------|
| **PyTorch** | Default & active development | `TorchLlmArgs` -> `PyExecutor` | Native PyTorch with custom CUDA kernels via CuTE DSL |
| **AutoDeploy** | Beta (maturing rapidly) | `_torch/auto_deploy/` shim | `torch.export` + graph transforms + MLIR elementwise fusion |
| **TensorRT** | Legacy (maintenance mode) | `TrtLlmArgs` -> `trtllm.Executor` | TensorRT engine compilation |

**What's changed (v1.2-v1.3):**
- PyTorch backend is now stable and default since v1.0; actively developed with CuTE DSL-based custom kernels.
- AutoDeploy is maturing rapidly — now supports DeepSeek-R1 and Qwen3.5, with MLIR-based auto-generated elementwise fusion (e.g., `SiLU+Mul` transform) and custom attention mask support.
- C++ sampler (`TLLM Sampler`) is now default (breaking change in v1.1), replacing TorchSampler for most paths. TorchSampler still required for beam search.

## 1.2 Architecture Diagram

```mermaid
graph TB
    subgraph "User Layer"
        CLI["trtllm-serve / trtllm-bench"]
        LLMAPI["LLM API — Python"]
        CLI --> LLMAPI
    end

    subgraph "Model Source"
        HF["HuggingFace Checkpoints"]
        HF --> LLMAPI
    end

    subgraph "PyTorch Backend — Default"
        PE["PyExecutor"]
        MEngine["PyTorchModelEngine"]
        CustomOps["CuTE DSL + Custom CUDA Ops"]
        PyOps["PyTorch Ops"]
        KernelLib["Kernel Libraries<br/>TRTLLM-Gen, FlashInfer"]
        LLMAPI --> PE
        PE --> MEngine
        MEngine --> CustomOps
        MEngine --> PyOps
        MEngine --> KernelLib
    end

    subgraph "AutoDeploy Backend — Beta"
        ADE["ADExecutor"]
        ADEng["ADEngine"]
        GraphTx["Graph Transforms + MLIR Fusion"]
        TorchExp["torch.export"]
        LLMAPI --> ADE
        ADE --> ADEng
        ADEng --> GraphTx
        GraphTx --> TorchExp
    end

    subgraph "Shared C++ Core — Nanobind"
        Sched["Scheduler<br/>C++ or Python"]
        BM["BatchManager — IFB"]
        KVC["KV Cache Manager<br/>V1 C++ / V2 Python"]
        Dec["Decoder"]
        Samp["TLLM C++ Sampler<br/>(default)"]
        Sched --> BM
        BM --> KVC
        Dec --> Samp
    end

    PE --> Sched
    PE --> Dec
    ADE --> Sched
    ADE --> Dec

    subgraph "Outputs"
        Tokens["Generated Tokens"]
        Stats["Performance + Energy Metrics"]
    end

    PE --> Outputs
    ADE --> Outputs
```

## 1.3 Request Flow

```mermaid
sequenceDiagram
    participant U as User / Client
    participant S as OpenAI Server — FastAPI
    participant L as LLM API
    participant E as PyExecutor — Background Loop
    participant Sch as Scheduler
    participant KV as KV Cache Manager
    participant M as ModelEngine — GPU
    participant Sa as TLLM Sampler

    U->>S: POST /v1/chat/completions
    S->>L: generate_async(prompt, params)
    L->>E: enqueue(LlmRequest)

    loop Every Iteration
        E->>E: _fetch_and_activate_new_requests()
        E->>Sch: schedule(active_requests)
        Sch-->>E: ScheduledRequests — context + generation
        E->>KV: prepare_resources(scheduled_batch)
        KV-->>E: KV blocks allocated
        E->>M: _forward_step(batch)
        M-->>E: logits
        E->>Sa: _sample_async(logits)
        Sa-->>E: new tokens
        E->>E: _update_requests() / _handle_responses()
    end

    E-->>S: streaming tokens / final response
    S-->>U: SSE stream / JSON response
```

## 1.4 Key Files Reference

| File | Role |
|:-----|:-----|
| `tensorrt_llm/llmapi/llm.py` | Main API entry point (`LLM` class) |
| `tensorrt_llm/llmapi/llm_args.py` | Configuration schema (Pydantic): `BaseLlmArgs`, `TorchLlmArgs` |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py` | Core execution loop (~3750 lines, the heart of the system) |
| `tensorrt_llm/_torch/pyexecutor/model_engine.py` | Model loading and forward pass |
| `tensorrt_llm/_torch/pyexecutor/resource_manager.py` | KV cache and resource management |
| `tensorrt_llm/_torch/pyexecutor/scheduler/` | Scheduler implementations (V1 bound C++, unified Python, V2) |
| `tensorrt_llm/executor/executor.py` | Execution abstraction (`GenerationExecutor`) |
| `tensorrt_llm/mapping.py` | Parallelism topology (`Mapping`, process groups) |
| `tensorrt_llm/serve/openai_server.py` | OpenAI-compatible HTTP server |
| `tensorrt_llm/_torch/speculative/` | All speculative decoding algorithms |
| `tensorrt_llm/_torch/auto_deploy/` | AutoDeploy backend with MLIR fusion |
| `docs/source/features/feature-combination-matrix.md` | Feature compatibility matrix |
