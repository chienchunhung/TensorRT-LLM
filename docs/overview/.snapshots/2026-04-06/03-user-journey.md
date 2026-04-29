# 3. End-to-End User Journey (PyTorch Backend)

[< Back to Overview](README.md)

## 3.1 Launch & Initialization

```mermaid
sequenceDiagram
    participant U as User
    participant LLM as LLM constructor
    participant CML as CachedModelLoader
    participant GE as GenerationExecutor
    participant W as BaseWorker
    participant PEC as py_executor_creator
    participant ML as ModelLoader
    participant PE as PyExecutor

    U->>LLM: LLM(model="meta-llama/...")
    LLM->>LLM: Build TorchLlmArgs — Pydantic validation
    LLM->>LLM: Init MPI/Ray process groups if multi-GPU
    LLM->>CML: _build_model()
    CML->>CML: Download/resolve HF model directory
    CML-->>LLM: (None, hf_model_dir)
    LLM->>LLM: Load tokenizer, create input processor
    LLM->>GE: GenerationExecutor.create(hf_model_dir, llm_args)
    GE->>W: Spawn worker process per rank
    W->>PEC: create_py_executor(llm_args, checkpoint_dir)
    PEC->>ML: load_config_and_apply_defaults()
    ML->>ML: Read HF config, resolve model class
    ML->>ML: model_class.get_model_defaults(), merge into llm_args
    PEC->>ML: load(checkpoint_dir)
    ML->>ML: AutoModelForCausalLM.from_config() — meta init
    ML->>ML: checkpoint_loader.load_weights(), model.load_weights()
    ML-->>PEC: nn.Module on CUDA
    PEC->>PEC: Build scheduler, KV cache manager, sampler
    PEC->>PE: PyExecutor(model_engine, scheduler, kv_cache, ...)
    PE->>PE: Start background event loop thread
    PE-->>LLM: Executor ready
    LLM-->>U: LLM instance ready
```

**Key files:**

- `llm.py`: `_TorchLLM.__init__`, `_build_model`
- `llm_utils.py`: `CachedModelLoader.__call__`
- `executor.py`: `GenerationExecutor.create`
- `py_executor_creator.py`: `create_py_executor`
- `model_loader.py`: `ModelLoader.load`

For **`trtllm-serve`**, the CLI (`tensorrt_llm/commands/serve.py`) builds `llm_args` from CLI + YAML, constructs the same `LLM` instance, then wraps it in `OpenAIServer` (FastAPI).

## 3.2 Model Loading & Weight Loading

The PyTorch backend has a **distinct** weight loading path from the TRT engine build pipeline:

```mermaid
flowchart TD
    A["HuggingFace Model Directory<br/>safetensors / bin files"] --> B["ModelLoader.load_config_and_apply_defaults()"]
    B --> C["Read HF config.json"]
    C --> D["AutoModelForCausalLM._resolve_class()"]
    D --> E["model_class.get_model_defaults()<br/>attn backend, quant, etc."]
    E --> F["Merge defaults into TorchLlmArgs<br/>user-set fields always win"]
    F --> G["ModelLoader.load()"]
    G --> H["from_config() with meta-device init<br/>avoids CPU memory spike"]
    H --> I["Move to CUDA"]
    I --> J["checkpoint_loader.load_weights()<br/>model.load_weights()"]
    J --> K["nn.Module on GPU — ready"]
```

The `get_model_defaults()` pattern is important: each model class (e.g., `modeling_llama.py`, `modeling_qwen3_next.py`) defines its own preferred defaults for attention kernel, quantization, speculative decoding, etc. These are merged into `TorchLlmArgs` but **never override user-explicit settings**.

## 3.3 Request Handling & Response

```mermaid
flowchart TD
    A["HTTP POST /v1/chat/completions"] --> B["OpenAIServer parses request"]
    B --> C["Build SamplingParams + PostprocParams"]
    C --> D["LLM.generate_async(prompt, params)"]
    D --> E["Tokenize input"]
    E --> F["Create LlmRequest"]
    F --> G["Enqueue to ExecutorRequestQueue"]

    subgraph "PyExecutor Background Loop"
        H["_fetch_and_activate_new_requests()"]
        I["_schedule()<br/>CapacityScheduler + MicroBatchScheduler"]
        J["resource_manager.prepare_resources()<br/>allocate KV blocks"]
        K["_forward_step()<br/>GPU forward pass"]
        L["_sample_async()<br/>TLLM C++ Sampler"]
        M["_update_requests()<br/>append tokens, check stop criteria"]
        N{"Request complete?"}
        H --> I --> J --> K --> L --> M --> N
        N -->|No| H
        N -->|Yes| O["_handle_responses()<br/>Finalize result"]
    end

    G --> H
    O --> P["Return via streaming SSE<br/>or JSON response"]
    P --> Q["Client receives response"]
```

## 3.4 Failover & Fault Tolerance

**Current state: Limited.** The system relies on external orchestration for production resilience.

| Mechanism | What It Does | Limitation |
|:----------|:------------|:-----------|
| `LLM._check_health()` | Verifies executor not shutdown | No automatic recovery |
| Disagg router health check | Pings `/health`, prunes dead backends | Reactive, not preventive |
| `DisaggClusterWorker` heartbeat | Registration + heartbeat for membership | Re-registration only, no request retry |
| `CppExecutorError` handler | Logs error, raises `SIGINT` | Clean shutdown, not failover |
| Service discovery (new) | Dynamic node join/leave for disagg | Membership only, no fault recovery |

**What's missing:** Automatic request retry, hot-standby replicas, checkpoint/restore for in-flight requests, graceful degradation under partial failures, circuit breakers, elastic expert redistribution on GPU failure.

**Contrast:** SGLang now has **elastic EP for partial failure tolerance** — when a GPU fails, experts are redistributed to surviving GPUs without restart. TRT-LLM has no equivalent.

## 3.5 Auto-Scaling

**Current state: Elastic membership + Dynamo integration.**

The `DisaggClusterManager` (`serve/disagg_auto_scaling.py`) supports:

- **Dynamic worker join/leave**: Workers can register and deregister
- **Role switching**: Workers can transition between context and generation roles
- **Readiness tracking**: Manager exposes `is_ready` based on minimum instance counts
- **Dynamo integration**: External orchestration for production scaling

**What's missing:** HPA-style automatic replica scaling based on QPS/latency/GPU metrics, automatic GPU provisioning, queue-depth-based scaling policies. Production scaling is delegated to **Dynamo** or **Kubernetes operators**.
