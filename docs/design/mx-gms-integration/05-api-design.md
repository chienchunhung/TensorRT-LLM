# 5. API Design: TRT-LLM Changes

[< Back to Overview](README.md)

> **Scope clarification:** This section covers only the code that needs to be written or modified in TRT-LLM. The MX and GMS client libraries provide substantial functionality that TRT-LLM calls but does not reimplement. See [Section 5.5](#5-what-mx-and-gms-client-libraries-already-provide) for a full inventory of what each library already provides.

## 1. MX Checkpoint Loader (New TRT-LLM Code)

A new `MXCheckpointLoader` subclassing TRT-LLM's existing `BaseCheckpointLoader`. This follows the same pattern as `HfCheckpointLoader` and `MistralCheckpointLoader` — the only difference is the weight source.

```python
# tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py

@register_checkpoint_loader("MX")
class MXCheckpointLoader(BaseCheckpointLoader):
    """Loads model weights via MX P2P transfer from existing replicas."""

    def __init__(self, mx_server_url: str, fallback_loader=None):
        self._mx_client = modelexpress.client.connect(mx_server_url)
        self._fallback_loader = fallback_loader or HfCheckpointLoader()

    def load_weights(self, checkpoint_dir, mapping, **kwargs):
        identity = self._build_identity(mapping, checkpoint_dir)
        sources = self._mx_client.list_sources(identity)
        compatible = [s for s in sources
                      if s.worker_rank == mapping.tp_rank
                      and s.extra_params["pp_rank"] == str(mapping.pp_rank)]

        if compatible:
            # Fast path: P2P receive from existing source
            return self._mx_client.receive(compatible[0])
        else:
            # Seed path: load from disk, then register as source
            weights = self._fallback_loader.load_weights(
                checkpoint_dir, mapping, **kwargs)
            self._register_as_source(weights, mapping)
            return weights

    def _build_identity(self, mapping, checkpoint_dir):
        """Map TRT-LLM's Mapping to MX's SourceIdentity protobuf."""
        return mx_proto.SourceIdentity(
            model_name=checkpoint_dir,
            dtype=str(mapping.dtype),
            extra_params={
                "tp_size": str(mapping.tp_size),
                "pp_size": str(mapping.pp_size),
                "ep_size": str(mapping.moe_ep_size),
                "worker_rank": str(mapping.tp_rank),
                "pp_rank": str(mapping.pp_rank),
            },
        )
```

**What TRT-LLM implements:** The loader class (~200 lines), identity mapping from `Mapping` to MX protobuf, fallback logic.

**What the MX client SDK provides:** gRPC connection, source discovery (`list_sources`), NIXL-based P2P transfer (`receive`), source registration, heartbeat.

## 2. GMS Weight Loading Mode (New TRT-LLM Code)

A new `LoadFormat.GMS` branch in the existing `ModelLoader.load()` method. This orchestrates calls to the GMS client library at the right TRT-LLM lifecycle points.

```python
# Additions to tensorrt_llm/_torch/pyexecutor/model_loader.py

# In ModelLoader.load():
if load_format == LoadFormat.GMS:
    gms_client = get_gms_client(self.llm_args)

    if gms_client.has_committed_weights(tag="model_weights"):
        # RO path: zero-copy import from existing GMS pool
        model = AutoModelForCausalLM.from_config(config)   # meta init
        model.post_load_weights()      # Set up module aliases first
        materialize_module_from_gms(gms_client, model)     # GMS library call
    else:
        # RW path: load normally under GMS memory pool, then commit
        gms_pool = gms_client.get_mem_pool()               # GMS library call
        with torch.cuda.use_mem_pool(gms_pool, device=device):
            model = self._load_standard(checkpoint_dir, checkpoint_loader)
        gms_client.finalize_write(model)                   # GMS library call
```

**What TRT-LLM implements:** The `LoadFormat.GMS` branch in `ModelLoader.load()` (~50 lines), the `get_gms_client()` helper that reads config and creates the client connection.

**What the GMS client library provides:**
- `CUDAPluggableAllocator` + `MemPool` (intercepts `torch` allocations via CUDA VMM)
- `materialize_module_from_gms()` (creates zero-copy tensors from shared GPU memory using `torch._C._construct_storage_from_data_pointer`)
- `register_module_tensors()` (walks model params/buffers, records metadata in GMS)
- `finalize_write()` / `commit()` (publishes memory for RO readers)
- RW/RO lock management via Unix domain socket connection

## 3. Configuration Schema (New TRT-LLM Code)

New fields on `TorchLlmArgs` and corresponding CLI options:

```python
# Additions to tensorrt_llm/llmapi/llm_args.py

class LoadFormat(Enum):
    AUTO = 0
    DUMMY = 1
    VISION_ONLY = 2
    MX = 3          # New
    GMS = 4         # New
    MX_GMS = 5      # New

class TorchLlmArgs(BaseLlmArgs):
    # ... existing fields ...

    # Weight loading format (extend existing field)
    load_format: Union[str, LoadFormat] = LoadFormat.AUTO

    # MX-specific configuration
    mx_server_url: Optional[str] = None
    mx_metadata_backend: Optional[Literal["redis", "kubernetes"]] = None
    mx_heartbeat_interval_secs: int = 30

    # GMS-specific configuration
    gms_socket_path: Optional[str] = None  # Default: /tmp/gms-{device_id}.sock
    gms_mode: Literal["auto", "rw", "ro"] = "auto"
    gms_tag: str = "model_weights"
```

**CLI usage:**

```bash
# MX only (P2P across nodes)
trtllm-serve meta-llama/Llama-3.1-70B --load-format mx \
    --mx-server-url http://mx:8001

# GMS only (sharing within node)
trtllm-serve meta-llama/Llama-3.1-70B --load-format gms \
    --gms-socket-path /tmp/gms-0.sock

# Combined
trtllm-serve meta-llama/Llama-3.1-70B --load-format mx-gms \
    --mx-server-url http://mx:8001 --gms-socket-path /tmp/gms-0.sock
```

## 4. GMS API Stability Abstraction (New TRT-LLM Code)

A thin protocol to insulate TRT-LLM from potential GMS API changes. This is the only abstraction layer TRT-LLM should own — everything below it is the GMS library's responsibility.

```python
# tensorrt_llm/_torch/memory/gpu_memory_backend.py

class GPUMemoryBackend(Protocol):
    """Thin abstraction over GMS client for API stability."""
    def has_committed_weights(self, tag: str) -> bool: ...
    def get_mem_pool(self) -> torch.cuda.MemPool: ...
    def materialize_module(self, model: torch.nn.Module) -> None: ...
    def finalize_write(self, model: torch.nn.Module) -> None: ...
    def commit(self, tag: str) -> None: ...
    def release(self, tag: str) -> None: ...
    def upgrade_lock(self) -> None: ...
```

**Why this exists:** The GMS API ([PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053)) is functional but not formally stabilized. If the API changes, only this adapter needs updating. If GMS proves unstable, a `CudaIpcBackend` could implement the same protocol as a fallback (less featured — no crash resilience or zero-copy, but uses stable CUDA IPC APIs).

## 5. What MX and GMS Client Libraries Already Provide

The following functionality is provided by the MX and GMS client libraries and should **not** be reimplemented in TRT-LLM:

### MX Client Library (`modelexpress.client`)

| Capability | MX API | TRT-LLM Calls It From |
|:-----------|:-------|:----------------------|
| gRPC connection to MX server | `modelexpress.client.connect(url)` | `MXCheckpointLoader.__init__` |
| Source discovery by identity | `client.list_sources(identity)` | `MXCheckpointLoader.load_weights` |
| NIXL-based P2P GPU-to-GPU transfer | `client.receive(source)` | `MXCheckpointLoader.load_weights` |
| Source registration (make weights available) | `client.register_source(tensors, identity)` | `MXCheckpointLoader._register_as_source` |
| Heartbeat and lifecycle | `client.heartbeat()` | Background thread |
| Three-tier fallback (RDMA -> GDS -> Disk) | Built into `client.receive()` | Transparent to TRT-LLM |

### GMS Client Library (`gpu_memory_service.client`)

| Capability | GMS API | TRT-LLM Calls It From |
|:-----------|:--------|:----------------------|
| CUDA VMM allocation (cuMemCreate + FD export) | `memory_manager.create_mapping()` | Via `MemPool` during RW loading |
| CUDA VMM import (cuMemImportFromShareableHandle + cuMemMap) | `memory_manager.import_mapping()` | Via `materialize_module_from_gms` |
| CUDAPluggableAllocator + MemPool | `allocator.get_mem_pool()` | `ModelLoader.load()` RW path |
| Zero-copy tensor creation from GPU pointer | `tensor._tensor_from_pointer()` using `torch._C._construct_storage_from_data_pointer` | Via `materialize_module_from_gms` |
| Module tensor registration (walks params/buffers) | `module.register_module_tensors()` | Via `finalize_write()` |
| Module materialization (meta -> GPU tensors) | `module.materialize_module_from_gms()` | `ModelLoader.load()` RO path |
| RW/RO socket-based locking | `client.connect(mode=RW\|RO)` | `get_gms_client()` |
| Lock upgrade (RO -> RW) | `client.upgrade_lock()` | Shadow activation |
| Tagged memory commit/release | `client.commit(tag)` / `client.release(tag)` | Post-load / sleep-wake |
| Sleep/wake (unmap/remap VMM VAs) | `manager.unmap_all_vas()` / `manager.remap_all_vas()` | Executor sleep/wake |

> **Key takeaway:** TRT-LLM's role is **orchestration** — calling these library APIs at the correct points in TRT-LLM's model loading and executor lifecycle. The heavy lifting (CUDA VMM, RDMA transfer, tensor construction from pointers, FD passing) is entirely in the external libraries.
