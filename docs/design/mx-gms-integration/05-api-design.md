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
| **MX + GMS** | `"MX"` | `GMS` | MX P2P source, GMS memory management **(see note below)** |

**Why this matters:** MX is a weight *source* (it replaces where weights come from — P2P instead of disk). GMS is a memory *management mode* (it replaces how GPU memory is allocated and shared). Conflating them into a single enum (e.g., a combined `LoadFormat.MX_GMS`) creates combinatorial explosion and doesn't compose. The two-axis approach avoids this.

> **Important: current prototype limitation.** In the current prototype, the "MX + GMS" combined mode behaves **identically** to "GMS only." The GMS RW path (first writer on a node) always loads from disk — it cannot leverage MX P2P because the MX SDK allocates received buffers outside the GMS memory pool (`torch.cuda.use_mem_pool` context). Weights received via MX P2P would land in regular CUDA memory that GMS cannot manage or share. See [Section 3 — architecture](03-architecture.md) for the full explanation and future optimization path. The two-axis model is architecturally correct and future-proof, but the composed "MX + GMS" benefit requires a future optimization (pre-allocating GMS pool buffers as MX P2P receive targets).

### Relationship to PR #12898

[PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) from the MX team adds `LoadFormat.PRESHARDED = 3` as a prototype MX integration. That approach conflates both axes into a single `LoadFormat` value — "weights are pre-sharded" (a property of the weight source) AND "use this loading pipeline." This works for MX alone but prevents composition with GMS, because `LoadFormat` is a single enum — you can't express "MX source + GMS memory" without adding a new `PRESHARDED_GMS` variant.

Key insights from PR #12898 that we adopt:
- **Pre-`post_load_weights()` publish timing**: Publishing weights before `post_load_weights()` so targets run their own transforms independently. This is correct and more robust than publishing after.
- **`_weights_presharded` flag on Linear modules**: Setting `tp_size = 1` to skip TP slicing for pre-sharded weights. This concept is valid — but should be a **context-derived flag** (set when weights arrive pre-sharded from MX P2P or GMS RO import), not tied to a specific `LoadFormat`.

## 2. MX Checkpoint Loader (New TRT-LLM Code)

A new `MXCheckpointLoader` registered via `@register_checkpoint_loader("MX")`. It **subclasses `HfCheckpointLoader`** so that the HF disk-loading path is inherited as the built-in fallback — no separate fallback loader is needed. MX replaces the weight *source* — P2P transfer from another replica instead of reading from disk.

> **Prototype:** [`dynamo-integration-prototype`](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype) branch, file `tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py` (~230 lines).

Key design decisions in the prototype:

- **Inherits `HfCheckpointLoader`** rather than `BaseCheckpointLoader`. This reuses the HF weight loader, config loader, and weight mapper registries (`@register_checkpoint_weight_loader("MX")`, `@register_config_loader("MX")`, `@register_mapper("MX")`) so that MX checkpoints on disk use the same loading pipeline as HF. Fallback is simply `super().load_weights()`.
- **Lazy MX connection.** The constructor stores `mx_server_url` but does not connect eagerly. Connection and source discovery happen inside `_try_p2p_transfer()`, called from `load_weights()`. This avoids blocking init when the MX server is unavailable and keeps the constructor compatible with TRT-LLM's `BaseCheckpointLoader.get()` factory.
- **Property-based `checkpoint_format` override.** The parent sets `self._checkpoint_format = "HF"` during `__init__`. `MXCheckpointLoader` overrides the `checkpoint_format` property to return `"MX"` and also sets the backing attribute for code that reads it directly.
- **`p2p_succeeded` property** (not `_weights_presharded`). The loader exposes a `p2p_succeeded` boolean. `ModelLoader.load()` reads this after `load_weights()` and sets `_weights_presharded` on the model's `Linear` modules. This keeps the presharded flag on the model (where it's consumed) rather than on the loader.

```python
# tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py

@register_checkpoint_loader("MX")
class MXCheckpointLoader(HfCheckpointLoader):
    """Weight source: P2P via MX, with HF disk fallback."""

    def __init__(self, *, weight_loader=None, weight_mapper=None,
                 config_loader=None, mx_server_url=None):
        super().__init__(weight_loader=weight_loader,
                         weight_mapper=weight_mapper,
                         config_loader=config_loader)
        self._checkpoint_format = "MX"  # align backing attr with property
        self._mx_server_url = mx_server_url
        self._p2p_succeeded = False
        self._identity = None

    @property
    def checkpoint_format(self) -> str:
        return "MX"

    @property
    def p2p_succeeded(self) -> bool:
        return self._p2p_succeeded

    def load_weights(self, checkpoint_dir, mapping, **kwargs):
        model = kwargs.pop("model", None)
        self._p2p_succeeded = False

        if self._mx_server_url is not None and model is not None:
            if self._try_p2p_transfer(model, mapping, checkpoint_dir):
                self._p2p_succeeded = True
                return {}  # Weights already in model params

        # Fallback: load from disk via inherited HF pipeline
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    def _try_p2p_transfer(self, model, mapping, checkpoint_dir) -> bool:
        """Lazy-connect to MX, discover compatible sources, receive via P2P."""
        try:
            from modelexpress import client as mx_client

            identity = self._build_identity(mapping, checkpoint_dir)
            if identity is None:
                return False
            self._identity = identity

            connection = mx_client.connect(self._mx_server_url)
            sources = connection.list_sources(identity)
            compatible = [
                s for s in sources
                if s.worker_rank == mapping.tp_rank
                and s.extra_params.get("pp_rank") == str(mapping.pp_rank)
            ]

            if compatible:
                connection.receive(compatible[0])
                return True
            return False
        except ImportError:
            logger.warning("modelexpress library not installed")
            return False
        except Exception as e:
            logger.warning("MX P2P transfer failed: %s", e)
            return False

    def publish_as_source(self, model, mapping=None, checkpoint_dir=None):
        """Publish model weights as MX source for other replicas.
        Called BEFORE post_load_weights() so targets receive raw state."""
        if self._mx_server_url is None:
            return
        try:
            from modelexpress import client as mx_client

            identity = self._identity
            if identity is None and mapping is not None:
                identity = self._build_identity(mapping, checkpoint_dir)
            if identity is None:
                return

            connection = mx_client.connect(self._mx_server_url)
            connection.register_source(model, identity)
        except (ImportError, Exception) as e:
            logger.warning("Failed to publish MX source: %s", e)

    def _build_identity(self, mapping, checkpoint_dir):
        """Map TRT-LLM's Mapping to MX's SourceIdentity protobuf."""
        try:
            from modelexpress import proto as mx_proto
            return mx_proto.SourceIdentity(
                model_name=checkpoint_dir,
                dtype=str(mapping.dtype) if hasattr(mapping, 'dtype') else "",
                extra_params={
                    "tp_size": str(mapping.tp_size),
                    "pp_size": str(mapping.pp_size),
                    "ep_size": str(mapping.moe_ep_size),
                    "worker_rank": str(mapping.tp_rank),
                    "pp_rank": str(mapping.pp_rank),
                },
            )
        except ImportError:
            return None
```

**What TRT-LLM implements:** The loader class (~230 lines), identity mapping from `Mapping` to MX protobuf, lazy connection/fallback logic, source publish hook.

**What the MX client SDK provides:** gRPC connection, source discovery (`list_sources`), NIXL-based P2P transfer (`receive`), source registration, heartbeat.

## 3. GMS Loading Mode (New TRT-LLM Code)

A new `LoadFormat.GMS` branch in the existing `ModelLoader.load()` method. GMS changes the *memory management* — not where weights come from. The GMS branch composes with **any** checkpoint loader (HF, Mistral, or MX for disk fallback).

> **Prototype:** [`dynamo-integration-prototype`](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype) branch. The `GMSBackend` class lives at `tensorrt_llm/_torch/memory/gpu_memory_backend.py`; the `LoadFormat.GMS` branch is in `tensorrt_llm/_torch/pyexecutor/model_loader.py`.

Key design decisions in the prototype:

- **`GMSBackend` class instead of bare `get_gms_client()`.** The prototype wraps the GMS client SDK in a `GMSBackend` class that encapsulates connection, mode resolution, and lifecycle. This is the concrete implementation of the `GPUMemoryBackend` protocol (see [Section 6](#6-gms-api-stability-abstraction-new-trt-llm-code)).
- **Mode resolved at connect time.** `GMSBackend.connect()` resolves `gms_mode="auto"` to RW or RO by checking `has_committed_weights(tag)`. The `is_rw` property is then used to branch in `ModelLoader.load()`.
- **Meta-init preserved for GMS.** The prototype skips both the `meta→CUDA` tensor init and `model.to("cuda")` for `LoadFormat.GMS`. The RW path allocates under the GMS mem pool during weight loading; the RO path replaces meta tensors via `materialize_module()`.
- **MX P2P not used in GMS RW mode — MX+GMS combined = GMS-only in current prototype.** When `checkpoint_format="MX"` + `load_format=GMS`, the GMS RW path does NOT attempt MX P2P. The root cause is CUDA memory pool isolation: GMS requires all weight memory to be allocated under `torch.cuda.use_mem_pool(gms_pool)` so RO readers can zero-copy import it. MX P2P receives allocate buffers inside the MX/NIXL SDK, outside the GMS pool context — those weights cannot be managed or shared by GMS. Consequently, the GMS RW path loads from disk under the GMS pool, making `checkpoint_format="MX"` + `load_format=GMS` functionally identical to `checkpoint_format="HF"` + `load_format=GMS`. The future optimization (pre-allocate GMS pool buffers, then MX P2P into them) would make the combined mode genuinely faster than either alone.
- **`post_load_weights()` ordering differs by path.** For GMS RO, `post_load_weights()` runs *before* `materialize_module()` so module aliases are set up correctly. For GMS RW and all other modes, `post_load_weights()` runs after weight loading as normal. A guard prevents double-execution.

```python
# In ModelLoader.load() — tensorrt_llm/_torch/pyexecutor/model_loader.py

elif load_format == LoadFormat.GMS:
    from tensorrt_llm._torch.memory import GMSBackend

    gms_backend = GMSBackend(
        socket_path=self.llm_args.gms_socket_path,
        mapping=self.mapping,
        mode=self.llm_args.gms_mode or "auto",
        tag=self.llm_args.gms_tag or "model_weights",
    )

    if not gms_backend.connect():
        raise RuntimeError("Failed to connect to GMS")

    if gms_backend.is_rw:
        # GMS RW path: load via checkpoint_loader under GMS memory pool
        gms_pool = gms_backend.get_mem_pool()
        device = torch.device('cuda')

        with torch.cuda.use_mem_pool(gms_pool, device=device):
            weights = checkpoint_loader.load_weights(
                checkpoint_dir, mapping=self.mapping)
            if weights:
                self.weight_mapper = (
                    checkpoint_loader
                    .get_initialized_weight_mapper(model, config))
                self._call_load_weights(
                    model.load_weights, weights, self.weight_mapper)

        gms_backend.finalize_write(model)
    else:
        # GMS RO path: zero-copy import from existing GMS pool
        # post_load_weights() BEFORE materialize (sets up module aliases)
        for module in model.modules():
            if hasattr(module, 'post_load_weights') and not getattr(
                    module, '_weights_removed', False):
                module.post_load_weights()

        gms_backend.materialize_module(model)

    self._gms_backend = gms_backend

# Later: guard to skip post_load_weights() for GMS RO (already ran above)
gms_ro_done = (load_format == LoadFormat.GMS
               and self._gms_backend is not None
               and not self._gms_backend.is_rw)
if not gms_ro_done:
    for module in model.modules():
        if hasattr(module, 'post_load_weights') ...
```

**What TRT-LLM implements:** The `LoadFormat.GMS` branch in `ModelLoader.load()` (~60 lines), the `GMSBackend` class (~240 lines), meta-init skip, `post_load_weights()` ordering guard.

**What the GMS client library provides:**
- `CUDAPluggableAllocator` + `MemPool` (intercepts `torch` allocations via CUDA VMM)
- `materialize_module_from_gms()` (creates zero-copy tensors from shared GPU memory)
- `register_module_tensors()` (walks model params/buffers, records metadata in GMS)
- `commit()` (publishes memory for RO readers)
- RW/RO lock management via Unix domain socket connection

## 4. Pre-Sharded Weight Handling (New TRT-LLM Code)

The `_weights_presharded` flag is a **context-derived property**, not tied to any specific `LoadFormat` or `checkpoint_format`. Multiple loading paths produce pre-sharded weights:

- **MX P2P receive** → weights arrive already sliced for this TP rank
- **GMS RO import** → weights were already sliced when the RW worker loaded them

In the prototype, each loading path sets the flag independently rather than in a single combined expression. This is cleaner because each path has different timing requirements:

```python
# In LoadFormat.AUTO branch (MX P2P case):
mx_p2p_succeeded = (hasattr(checkpoint_loader, 'p2p_succeeded')
                     and checkpoint_loader.p2p_succeeded)
if mx_p2p_succeeded:
    from tensorrt_llm._torch.modules.linear import Linear
    # Exclude draft model modules — loaded separately from disk
    draft_modules = set()
    if hasattr(model, 'draft_model') and model.draft_model is not None:
        draft_modules = set(id(m) for m in model.draft_model.modules())
    for module in model.modules():
        if isinstance(module, Linear) and id(module) not in draft_modules:
            module._weights_presharded = True

# In LoadFormat.GMS RO branch (inside GMSBackend.materialize_module):
# _weights_presharded is set as part of materialization — see Section 3

# MX source publish hook (before post_load_weights)
# Fires for AUTO and GMS-RW (MX+GMS combo), not for GMS-RO or DUMMY
should_publish = (
    load_format == LoadFormat.AUTO
    or (load_format == LoadFormat.GMS
        and self._gms_backend is not None
        and self._gms_backend.is_rw))
if (should_publish
        and hasattr(checkpoint_loader, 'publish_as_source')):
    checkpoint_loader.publish_as_source(
        model, mapping=self.mapping, checkpoint_dir=checkpoint_dir)

# post_load_weights() — skipped for GMS RO (already ran before materialize)
gms_ro_done = (load_format == LoadFormat.GMS
               and self._gms_backend is not None
               and not self._gms_backend.is_rw)
if not gms_ro_done:
    for module in model.modules():
        if hasattr(module, 'post_load_weights') and not getattr(
                module, '_weights_removed', False):
            module.post_load_weights()
```

The `_weights_presharded` attribute is declared on `Linear.__init__` (defaulting to `False`) so that `getattr` with a default is not required in the load helpers. The Linear module changes from [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) — using `tp_size = 1` when `_weights_presharded` — are adopted as-is. The difference is who sets the flag and when.

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
    mx_server_url: Optional[str] = Field(default=None, status="prototype")

    # GMS-specific (only when load_format=GMS)
    gms_socket_path: Optional[str] = Field(
        default=None, status="prototype")  # Default: /tmp/gms-{device_id}.sock
    gms_mode: Optional[str] = Field(
        default="auto", status="prototype")  # "auto", "rw", or "ro"
    gms_tag: str = Field(
        default="model_weights", status="prototype")
```

All four fields have `status="prototype"` and are registered in the API stability YAML (`tests/unittest/api_stability/references/llm.yaml`).

**Validators:**
- `validate_mx_config()`: warns if `mx_server_url` is set but `checkpoint_format != "MX"`
- `validate_gms_config()`: validates `gms_mode` is one of `"auto"`, `"rw"`, `"ro"`; warns if `gms_socket_path` is set but `load_format != GMS`

> **Note:** The design doc previously included `mx_metadata_backend` and `mx_heartbeat_interval_secs` fields. These are **not in the prototype** — MX metadata backend selection and heartbeat are responsibilities of the `modelexpress` client library, not TRT-LLM configuration. They may be added later if the MX SDK requires explicit configuration from the caller.

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

A thin `Protocol` to insulate TRT-LLM from potential GMS API changes, plus a concrete `GMSBackend` implementation. This is the only abstraction layer TRT-LLM should own — everything below it is the GMS library's responsibility.

```python
# tensorrt_llm/_torch/memory/gpu_memory_backend.py

@runtime_checkable
class GPUMemoryBackend(Protocol):
    """Thin abstraction over GMS client for API stability."""
    def has_committed_weights(self, tag: str) -> bool: ...
    def get_mem_pool(self) -> torch.cuda.MemPool: ...
    def materialize_module(self, model: torch.nn.Module) -> None: ...
    def finalize_write(self, model: torch.nn.Module, tag: str) -> None: ...
    def release(self, tag: str) -> None: ...
    def cleanup(self) -> None: ...


class GMSBackend:
    """Concrete GPUMemoryBackend using gpu_memory_service.client."""

    def __init__(self, socket_path, mapping, mode="auto", tag="model_weights"):
        ...  # See prototype for full implementation (~240 lines)

    def connect(self) -> bool:
        """Connect to GMS. In auto mode, resolves RW vs RO."""
        ...

    @property
    def is_rw(self) -> Optional[bool]:
        """Whether this backend resolved to RW mode."""
        ...

    # Protocol methods: has_committed_weights, get_mem_pool,
    # materialize_module (also sets _weights_presharded on Linear modules),
    # finalize_write (calls register_module_tensors + commit),
    # release, cleanup
```

**Key differences from initial design:**
- `finalize_write()` takes both `model` and `tag` (tag defaults to the instance's configured tag).
- `upgrade_lock()` is **not in the base protocol** — it's specific to the shadow failover path (see [Section 6: Executor Integration](06-executor-failover.md)) and will be added when that phase is implemented.
- `cleanup()` replaces the previous `disconnect()` — it handles both unmapping and disconnection.
- `GMSBackend.connect()` resolves `mode="auto"` at connect time by querying `has_committed_weights()`, exposing the result via the `is_rw` property.

**Why this exists:** The GMS API ([PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053)) is functional but not formally stabilized. If the API changes, only `GMSBackend` needs updating. If GMS proves unstable, a `CudaIpcBackend` could implement the same protocol as a fallback (less featured — no crash resilience or zero-copy, but uses stable CUDA IPC APIs).

## 7. What MX and GMS Client Libraries Already Provide

The following functionality is provided by the MX and GMS client libraries and should **not** be reimplemented in TRT-LLM:

### MX Client Library (`modelexpress.client`)

| Capability | MX API | TRT-LLM Calls It From |
|:-----------|:-------|:----------------------|
| gRPC connection to MX server | `modelexpress.client.connect(url)` | `MXCheckpointLoader._try_p2p_transfer` (lazy) |
| Source discovery by identity | `connection.list_sources(identity)` | `MXCheckpointLoader._try_p2p_transfer` |
| NIXL-based P2P GPU-to-GPU transfer | `connection.receive(source)` | `MXCheckpointLoader._try_p2p_transfer` |
| Source registration (make weights available) | `connection.register_source(model, identity)` | `MXCheckpointLoader.publish_as_source` |
| Identity protobuf | `modelexpress.proto.SourceIdentity` | `MXCheckpointLoader._build_identity` |
| Heartbeat and lifecycle | `client.heartbeat()` | Background thread (managed by MX SDK) |
| Three-tier fallback (RDMA -> GDS -> Disk) | Built into `connection.receive()` | Transparent to TRT-LLM |

### GMS Client Library (`gpu_memory_service.client`)

| Capability | GMS API | TRT-LLM Calls It From |
|:-----------|:--------|:----------------------|
| CUDA VMM allocation (cuMemCreate + FD export) | `memory_manager.create_mapping()` | Via `MemPool` during RW loading |
| CUDA VMM import (cuMemImportFromShareableHandle + cuMemMap) | `memory_manager.import_mapping()` | Via `materialize_module_from_gms` |
| CUDAPluggableAllocator + MemPool | `gms_client.get_mem_pool(client)` | `GMSBackend.get_mem_pool()` |
| Zero-copy tensor creation from GPU pointer | `tensor._tensor_from_pointer()` using `torch._C._construct_storage_from_data_pointer` | Via `materialize_module_from_gms` |
| Module tensor registration (walks params/buffers) | `gms_client.register_module_tensors(client, model)` | `GMSBackend.finalize_write()` |
| Module materialization (meta -> GPU tensors) | `gms_client.materialize_module_from_gms(client, model)` | `GMSBackend.materialize_module()` |
| RW/RO socket-based locking | `gms_client.connect(socket_path, mode=...)` | `GMSBackend.connect()` |
| Check for committed weights | `client.has_committed_weights(tag)` | `GMSBackend.connect()` (for auto mode resolution) |
| Tagged memory commit | `gms_client.commit(client, tag)` | `GMSBackend.finalize_write()` |
| Tagged memory release | `gms_client.release(client, tag)` | `GMSBackend.release()` |
| Lock upgrade (RO -> RW) | `client.upgrade_lock()` | Shadow activation (Phase 2) |
| Sleep/wake (unmap/remap VMM VAs) | `manager.unmap_all_vas()` / `manager.remap_all_vas()` | Executor sleep/wake (Phase 2) |

> **Key takeaway:** TRT-LLM's role is **orchestration** — calling these library APIs at the correct points in TRT-LLM's model loading and executor lifecycle. The heavy lifting (CUDA VMM, RDMA transfer, tensor construction from pointers, FD passing) is entirely in the external libraries.
