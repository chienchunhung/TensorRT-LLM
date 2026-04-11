# 5. API Design: TRT-LLM Changes

[< Back to Overview](README.md)

> **Scope clarification:** This section covers only the code that needs to be written or modified in TRT-LLM. The MX and GMS client libraries provide substantial functionality that TRT-LLM calls but does not reimplement. See [Section 5.6](#6-what-mx-and-gms-client-libraries-already-provide) for a full inventory of what each library already provides.

## 1. Design Principle: Two Orthogonal Axes

TRT-LLM's `ModelLoader.load()` already separates two independent concerns:

| Axis | Controlled by | Current values | What it decides |
|:-----|:-------------|:---------------|:----------------|
| **Weight source** | `checkpoint_format` (string) → `@register_checkpoint_loader` | `"HF"`, `"mistral"`, `"mistral_large_3"` | *Where* weights come from (file format / transfer mechanism) |
| **Loading mode** | `LoadFormat` (enum) → `if/elif` branches in `ModelLoader.load()` | `AUTO`, `DUMMY`, `VISION_ONLY` | *How* the loading pipeline behaves (memory management, orchestration) |

These compose independently. The checkpoint loader is selected by `checkpoint_format`; the loading pipeline branch is selected by `LoadFormat`. This gives us a clean mapping for all four integration modes:

| Mode | `checkpoint_format` | `LoadFormat` | Behavior |
|:-----|:-------------------|:-------------|:---------|
| **Pure TRT-LLM** | `"HF"` (default) | `AUTO` (default) | Current behavior, unchanged |
| **MX only** | `"MX"` | `AUTO` | MX P2P source, standard CUDA allocator |
| **GMS only** | `"HF"` (default) | `GMS` | Disk source, GMS memory management |
| **MX + GMS** | `"MX"` | `GMS` | MX P2P source, GMS memory management |

**Why this matters:** MX is a weight *source* (it replaces where weights come from — P2P instead of disk). GMS is a memory *management mode* (it replaces how GPU memory is allocated and shared). Conflating them into a single enum (e.g., a combined `LoadFormat.MX_GMS`) creates combinatorial explosion and doesn't compose. The two-axis approach avoids this.

### Relationship to PR #12898

[PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) from the MX team adds `LoadFormat.PRESHARDED = 3` as a prototype MX integration. That approach conflates both axes into a single `LoadFormat` value — "weights are pre-sharded" (a property of the weight source) AND "use this loading pipeline." This works for MX alone but prevents composition with GMS, because `LoadFormat` is a single enum — you can't express "MX source + GMS memory" without adding a new `PRESHARDED_GMS` variant.

Key insights from PR #12898 that we adopt:
- **Pre-`post_load_weights()` publish timing**: Publishing weights before `post_load_weights()` so targets run their own transforms independently. This is correct and more robust than publishing after.
- **`_weights_presharded` flag on Linear modules**: Setting `tp_size = 1` to skip TP slicing for pre-sharded weights. This concept is valid — but should be a **context-derived flag** (set when weights arrive pre-sharded from MX P2P or GMS RO import), not tied to a specific `LoadFormat`.

## 2. MX Checkpoint Loader (New TRT-LLM Code)

A new `MXCheckpointLoader` registered via `@register_checkpoint_loader("MX")`, following the same pattern as `HfCheckpointLoader` and `MistralCheckpointLoader`. MX replaces the weight *source* — P2P transfer from another replica instead of reading from disk.

```python
# tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py

@register_checkpoint_loader("MX")
class MXCheckpointLoader(BaseCheckpointLoader):
    """Weight source: P2P via MX, with disk fallback."""

    def __init__(self, mx_server_url: str, fallback_loader=None):
        self._mx_client = modelexpress.client.connect(mx_server_url)
        self._fallback_loader = fallback_loader or HfCheckpointLoader()
        self._weights_presharded = False  # Set True when P2P receive succeeds

    def load_weights(self, checkpoint_dir, mapping, **kwargs):
        identity = self._build_identity(mapping, checkpoint_dir)
        sources = self._mx_client.list_sources(identity)
        compatible = [s for s in sources
                      if s.worker_rank == mapping.tp_rank
                      and s.extra_params["pp_rank"] == str(mapping.pp_rank)]

        if compatible:
            # Fast path: P2P receive — weights arrive pre-sharded
            self._weights_presharded = True
            return self._mx_client.receive(compatible[0])
        else:
            # Seed path: load from disk, register as MX source
            self._weights_presharded = False
            weights = self._fallback_loader.load_weights(
                checkpoint_dir, mapping, **kwargs)
            return weights

    def publish_as_source(self, model):
        """Publish model weights as MX source for other replicas.
        Called BEFORE post_load_weights() so targets receive raw state
        and run their own transforms."""
        self._mx_client.register_source(model, self._identity)

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

**What TRT-LLM implements:** The loader class (~200 lines), identity mapping from `Mapping` to MX protobuf, fallback logic, source publish hook.

**What the MX client SDK provides:** gRPC connection, source discovery (`list_sources`), NIXL-based P2P transfer (`receive`), source registration, heartbeat.

## 3. GMS Loading Mode (New TRT-LLM Code)

A new `LoadFormat.GMS` branch in the existing `ModelLoader.load()` method. GMS changes the *memory management* — not where weights come from. The GMS branch composes with **any** checkpoint loader (HF, Mistral, or MX).

```python
# Additions to tensorrt_llm/_torch/pyexecutor/model_loader.py

# In ModelLoader.load():
if load_format == LoadFormat.GMS:
    gms_client = get_gms_client(self.llm_args)

    if gms_client.has_committed_weights(tag="model_weights"):
        # GMS RO path: zero-copy import from existing GMS pool
        # No checkpoint_loader needed — weights come directly from GMS
        model = AutoModelForCausalLM.from_config(config)   # meta init
        model.post_load_weights()      # Set up module aliases first
        materialize_module_from_gms(gms_client, model)     # GMS library call
        # Weights are already per-rank-sharded (written by RW worker)
    else:
        # GMS RW path: load via checkpoint_loader under GMS memory pool
        # checkpoint_loader could be HF (disk) or MX (P2P) — doesn't matter
        gms_pool = gms_client.get_mem_pool()               # GMS library call
        with torch.cuda.use_mem_pool(gms_pool, device=device):
            # Standard loading pipeline, but allocations go to GMS
            weights = checkpoint_loader.load_weights(checkpoint_dir, mapping)
            # ... weight mapper setup, model.load_weights() ...
        gms_client.finalize_write(model)                   # GMS library call
```

**What TRT-LLM implements:** The `LoadFormat.GMS` branch in `ModelLoader.load()` (~50 lines), the `get_gms_client()` helper.

**What the GMS client library provides:**
- `CUDAPluggableAllocator` + `MemPool` (intercepts `torch` allocations via CUDA VMM)
- `materialize_module_from_gms()` (creates zero-copy tensors from shared GPU memory)
- `register_module_tensors()` (walks model params/buffers, records metadata in GMS)
- `finalize_write()` / `commit()` (publishes memory for RO readers)
- RW/RO lock management via Unix domain socket connection

## 4. Pre-Sharded Weight Handling (New TRT-LLM Code)

The `_weights_presharded` flag is a **context-derived property**, not tied to any specific `LoadFormat` or `checkpoint_format`. Multiple loading paths produce pre-sharded weights:

- **MX P2P receive** → weights arrive already sliced for this TP rank
- **GMS RO import** → weights were already sliced when the RW worker loaded them

```python
# In ModelLoader.load(), after weight loading but before post_load_weights():

# Determine if weights are pre-sharded (skip TP slicing)
weights_presharded = (
    getattr(checkpoint_loader, '_weights_presharded', False)  # MX P2P
    or (load_format == LoadFormat.GMS
        and gms_client.has_committed_weights(tag="model_weights"))  # GMS RO
)

if weights_presharded:
    from tensorrt_llm._torch.modules.linear import Linear
    for m in model.modules():
        if isinstance(m, Linear):
            m._weights_presharded = True

# MX source publish hook (before post_load_weights)
if hasattr(checkpoint_loader, 'publish_as_source'):
    checkpoint_loader.publish_as_source(model)

# post_load_weights() runs unconditionally (existing behavior)
for module in model.modules():
    if hasattr(module, 'post_load_weights') and not getattr(
            module, '_weights_removed', False):
        module.post_load_weights()
```

The Linear module changes from [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) — using `tp_size = 1` when `_weights_presharded` — are adopted as-is. The difference is who sets the flag and when.

## 5. Configuration Schema (New TRT-LLM Code)

The two-axis model maps cleanly to TRT-LLM's existing configuration:

```python
# Additions to tensorrt_llm/llmapi/llm_args.py

class LoadFormat(Enum):
    AUTO = 0
    DUMMY = 1
    VISION_ONLY = 2
    GMS = 3           # New: GMS memory management mode

class TorchLlmArgs(BaseLlmArgs):
    # Existing fields (checkpoint_format already exists):
    checkpoint_format: str = "HF"      # Add "MX" as a valid value
    load_format: LoadFormat = LoadFormat.AUTO

    # MX-specific (only when checkpoint_format="MX")
    mx_server_url: Optional[str] = None
    mx_metadata_backend: Optional[Literal["redis", "kubernetes"]] = None
    mx_heartbeat_interval_secs: int = 30

    # GMS-specific (only when load_format=GMS)
    gms_socket_path: Optional[str] = None  # Default: /tmp/gms-{device_id}.sock
    gms_mode: Literal["auto", "rw", "ro"] = "auto"
    gms_tag: str = "model_weights"
```

**CLI usage:**

```bash
# Mode 1: MX only (P2P across nodes)
trtllm-serve meta-llama/Llama-3.1-70B \
    --checkpoint-format mx --mx-server-url http://mx:8001

# Mode 2: GMS only (sharing within node)
trtllm-serve meta-llama/Llama-3.1-70B \
    --load-format gms --gms-socket-path /tmp/gms-0.sock

# Mode 3: MX + GMS (combined)
trtllm-serve meta-llama/Llama-3.1-70B \
    --checkpoint-format mx --mx-server-url http://mx:8001 \
    --load-format gms --gms-socket-path /tmp/gms-0.sock

# Mode 4: Pure TRT-LLM (unchanged, default)
trtllm-serve meta-llama/Llama-3.1-70B
```

Note: `--checkpoint-format` and `--load-format` are independent flags. Each is only needed when using that specific integration. The defaults (`"HF"` and `AUTO`) preserve current behavior.

## 6. GMS API Stability Abstraction (New TRT-LLM Code)

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

## 7. What MX and GMS Client Libraries Already Provide

The following functionality is provided by the MX and GMS client libraries and should **not** be reimplemented in TRT-LLM:

### MX Client Library (`modelexpress.client`)

| Capability | MX API | TRT-LLM Calls It From |
|:-----------|:-------|:----------------------|
| gRPC connection to MX server | `modelexpress.client.connect(url)` | `MXCheckpointLoader.__init__` |
| Source discovery by identity | `client.list_sources(identity)` | `MXCheckpointLoader.load_weights` |
| NIXL-based P2P GPU-to-GPU transfer | `client.receive(source)` | `MXCheckpointLoader.load_weights` |
| Source registration (make weights available) | `client.register_source(tensors, identity)` | `MXCheckpointLoader.publish_as_source` |
| Heartbeat and lifecycle | `client.heartbeat()` | Background thread |
| Three-tier fallback (RDMA -> GDS -> Disk) | Built into `client.receive()` | Transparent to TRT-LLM |

### GMS Client Library (`gpu_memory_service.client`)

| Capability | GMS API | TRT-LLM Calls It From |
|:-----------|:--------|:----------------------|
| CUDA VMM allocation (cuMemCreate + FD export) | `memory_manager.create_mapping()` | Via `MemPool` during RW loading |
| CUDA VMM import (cuMemImportFromShareableHandle + cuMemMap) | `memory_manager.import_mapping()` | Via `materialize_module_from_gms` |
| CUDAPluggableAllocator + MemPool | `allocator.get_mem_pool()` | `ModelLoader.load()` GMS RW path |
| Zero-copy tensor creation from GPU pointer | `tensor._tensor_from_pointer()` using `torch._C._construct_storage_from_data_pointer` | Via `materialize_module_from_gms` |
| Module tensor registration (walks params/buffers) | `module.register_module_tensors()` | Via `finalize_write()` |
| Module materialization (meta -> GPU tensors) | `module.materialize_module_from_gms()` | `ModelLoader.load()` GMS RO path |
| RW/RO socket-based locking | `client.connect(mode=RW\|RO)` | `get_gms_client()` |
| Lock upgrade (RO -> RW) | `client.upgrade_lock()` | Shadow activation |
| Tagged memory commit/release | `client.commit(tag)` / `client.release(tag)` | Post-load / sleep-wake |
| Sleep/wake (unmap/remap VMM VAs) | `manager.unmap_all_vas()` / `manager.remap_all_vas()` | Executor sleep/wake |

> **Key takeaway:** TRT-LLM's role is **orchestration** — calling these library APIs at the correct points in TRT-LLM's model loading and executor lifecycle. The heavy lifting (CUDA VMM, RDMA transfer, tensor construction from pointers, FD passing) is entirely in the external libraries.
