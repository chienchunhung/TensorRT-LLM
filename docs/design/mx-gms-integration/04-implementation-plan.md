# 4. Implementation Plan

[< Back to Overview](README.md)

**Last Updated:** 2026-04-08

---

## Integration Approach

### Relationship to Existing Prototypes

This plan treats MX and GMS as **library dependencies**, not things to reimplement. The existing prototypes demonstrate that the core functionality already works:

- **MX (ModelExpress)**: The [`modelexpress`](https://github.com/ai-dynamo/modelexpress) library provides a gRPC server, Python client SDK, and NIXL-based GPU-to-GPU transfer. vLLM's `--load-format mx` integration is a thin loader (~500 lines) that calls the MX client API. [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) demonstrates a working MX prototype with TRT-LLM using `LoadFormat.PRESHARDED`.
- **GMS (GPU Memory Service)**: The [`gpu_memory_service`](https://github.com/ai-dynamo/dynamo/pull/7053) library provides the CUDA VMM allocator, RW/RO client, and socket-based locking. PR #7053 shows a working TRT-LLM integration (~300 lines of TRT-LLM-specific model loading code) that calls the GMS client.

### Two-Axis Integration Model

MX and GMS map onto two **independent** axes in TRT-LLM's existing loading pipeline (see [API Design](05-api-design.md) Section 5.1 for full rationale):

- **MX** is a weight *source* → integrates as a new `checkpoint_format` via `@register_checkpoint_loader("MX")`
- **GMS** is a memory *management mode* → integrates as a new `LoadFormat.GMS` branch in `ModelLoader.load()`

This separation means all four modes compose naturally without combinatorial config explosion:

| Mode | `checkpoint_format` | `LoadFormat` |
|:-----|:-------------------|:-------------|
| Pure TRT-LLM | `"HF"` (default) | `AUTO` (default) |
| MX only | `"MX"` | `AUTO` |
| GMS only | `"HF"` (default) | `GMS` |
| MX + GMS | `"MX"` | `GMS` |

**What TRT-LLM needs to implement:**

| Area | TRT-LLM-side work | Integration axis | Dynamo-side work (external) |
|:-----|:-------------------|:----------------|:---------------------------|
| MX weight loading | `@register_checkpoint_loader("MX")` that calls MX client APIs | `checkpoint_format` (weight source) | Maintain `modelexpress` library |
| GMS weight sharing | `LoadFormat.GMS` branch in `ModelLoader.load()` calling GMS client | `LoadFormat` (memory management) | Maintain `gpu_memory_service` library |
| Pre-sharded TP skip | Set `_weights_presharded` flag when MX P2P or GMS RO produces per-rank weights | Cross-cutting (derived from context) | None |
| Configuration | `checkpoint_format`, `LoadFormat`, MX/GMS fields on `TorchLlmArgs` | Both axes | None |
| Sleep/wake | Map `release_with_tag("kv_cache")` to GMS tag operations (already demonstrated in PR #7053) | `LoadFormat.GMS` | None |
| Shadow failover | `PyExecutor` shadow mode with GMS-backed activation | `LoadFormat.GMS` | None |
| Testing | TRT-LLM CI with MX/GMS enabled | Both axes | GMS integration tests in Dynamo repo |

**What TRT-LLM does NOT need to implement** (provided by MX/GMS libraries):
- NIXL wrapper or RDMA transfer logic (use `modelexpress` client directly)
- CUDA VMM FD import/export (use `gpu_memory_service.client` directly — `cuda_utils.py`)
- GPU memory allocator (use GMS's `CUDAPluggableAllocator` + `MemPool`)
- Tensor enumeration for GMS commit (GMS's `register_module_tensors()` walks the model)
- Zero-copy tensor construction from GPU pointers (GMS's `materialize_module_from_gms()`)
- gRPC server/client for MX (use `modelexpress` SDK)
- Socket-based RW/RO locking (GMS client handles this transparently)

See [API Design](05-api-design.md) Section 5.7 for a full inventory of what each library provides.

### Target Backend Scope

All implementation targets the **PyTorch backend** with:
- **KV Cache Manager V1** (the default `KVCacheManager`, not V2)
- **C++ transceiver** (the default NIXL/UCX-based `CacheTransceiver`)
- **`trtllm-serve`** as the primary serving surface

The TensorRT (legacy) backend is out of scope. AutoDeploy inherits PyTorch backend behavior and should work without additional changes.

### Glossary

| Term | Meaning |
|:-----|:--------|
| **MX** | ModelExpress — GPU-to-GPU model weight streaming service |
| **GMS** | GPU Memory Service — out-of-process GPU memory management for zero-copy sharing and crash resilience |
| **GDS** | GPUDirect Storage — NVIDIA technology for direct DMA between NVMe storage and GPU memory, bypassing CPU |
| **NIXL** | NVIDIA Inference eXchange Library — unified transfer API used by both MX and TRT-LLM's disaggregated serving |
| **KVBM** | KV Block Manager — tiered KV cache management in Dynamo |

---

## Phased Approach

```mermaid
gantt
    title MX + GMS Integration Timeline
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Phase 1: MX (P1)
    MX checkpoint loader + config      :p1a, 2026-04-14, 2w
    Testing + vLLM comparison           :p1b, after p1a, 2w

    section Phase 2: GMS (P2)
    GMS weight loader + sleep/wake     :p2a, after p1b, 2w
    Shadow failover + testing          :p2b, after p2a, 2w

    section Phase 3: Combined (P2)
    Combined loader + disagg + E2E     :p3a, after p2b, 2w
```

---

## Phase 1: MX + TRT-LLM (Cross-Node P2P) — 3-4 Weeks

**Priority:** P1 (Tier 1) — competitive catch-up with vLLM
**Objective:** Enable P2P weight transfer across nodes via `--checkpoint-format mx`

### Deliverables

| # | Deliverable | Description | TRT-LLM Files |
|:--|:-----------|:-----------|:------|
| 1.1 | MX checkpoint loader | `@register_checkpoint_loader("MX")` that calls MX client SDK | `_torch/models/checkpoints/mx/` |
| 1.2 | Configuration schema | `checkpoint_format: "MX"`, `mx_server_url` in `TorchLlmArgs` | `llmapi/llm_args.py` |
| 1.3 | Pre-sharded TP skip | `_weights_presharded` flag on Linear modules when P2P receive succeeds | `_torch/modules/linear.py` (adopt from [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898)) |
| 1.4 | Fallback logic | MX P2P -> GPUDirect Storage -> disk, with graceful degradation | Inside MX loader |
| 1.5 | Tests and documentation | Unit tests, 2-node integration test, user guide | `tests/`, `docs/` |

### Implementation Details

**Weeks 1-2: MX Checkpoint Loader + Configuration**

The MX loader follows the same `BaseCheckpointLoader` pattern as the existing HF loader. The key difference is the weight source. MX integrates as a `checkpoint_format` (weight source axis), not a `LoadFormat` (memory management axis) — see [API Design](05-api-design.md) Section 5.1 for rationale.

```python
@register_checkpoint_loader("MX")
class MXCheckpointLoader(BaseCheckpointLoader):
    def load_weights(self, checkpoint_dir, mapping, **kwargs):
        # 1. Query MX server for existing sources with matching identity
        sources = self._mx_client.list_sources(self._build_identity(mapping))
        compatible = [s for s in sources if self._rank_matches(s, mapping)]

        if compatible:
            # 2a. P2P receive from existing source (fast path)
            self._weights_presharded = True  # Signal to skip TP slicing
            return self._mx_client.receive(compatible[0])
        else:
            # 2b. Load from disk normally, then register as MX source
            self._weights_presharded = False
            weights = self._fallback_loader.load_weights(checkpoint_dir, mapping)
            return weights
```

The MX client SDK (`modelexpress.client`) handles:
- gRPC connection to MX server
- Source registration and discovery
- NIXL-based P2P transfer
- Heartbeat and lifecycle

TRT-LLM only needs to:
- Call the client at the right time (inside `load_weights`)
- Build the `SourceIdentity` with TRT-LLM's parallelism config (TP/PP/EP ranks)
- Handle the config plumbing (`TorchLlmArgs.checkpoint_format`, `TorchLlmArgs.mx_server_url`)
- Publish as MX source BEFORE `post_load_weights()` (learned from [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) — targets run their own transforms)

**Weeks 3-4: Testing + Comparison**

- Unit tests: identity matching, fallback logic
- Integration test: 2-node cluster, cold-start with MX vs baseline
- Benchmark comparison against vLLM's `--load-format mx` (vLLM uses `--load-format`; TRT-LLM uses `--checkpoint-format`)
- **Gate:** Cold-start within 20% of vLLM MX for same model

### Phase 1 Success Criteria

| Metric | Target |
|:-------|:-------|
| Cold-start (Llama-70B, 2nd replica) | < 30s (vs. 2-3 min baseline) |
| P2P transfer throughput | > 20 GB/s (NVLink), > 10 GB/s (InfiniBand) |
| Graceful fallback to disk | Yes, with warning log |
| vLLM parity | Within 20% of vLLM MX cold-start time |

---

## Phase 2: GMS + TRT-LLM (Within-Node Sharing) — 3-4 Weeks

**Priority:** P2 (Tier 2) — differentiation; enables fault tolerance
**Objective:** Enable zero-copy weight sharing and crash-resilient failover

### Deliverables

| # | Deliverable | Description | TRT-LLM Files |
|:--|:-----------|:-----------|:------|
| 2.1 | GMS loading mode | `LoadFormat.GMS` branch in `ModelLoader.load()` with RW/RO paths | `_torch/pyexecutor/model_loader.py` |
| 2.2 | Sleep/wake GMS tag mapping | Map `release_with_tag("kv_cache")` to GMS-safe operations | `_torch/pyexecutor/py_executor_creator.py` |
| 2.3 | Shadow failover integration | PyExecutor shadow mode with GMS-backed activation | `_torch/pyexecutor/py_executor.py` |
| 2.4 | Configuration schema | `LoadFormat.GMS`, `gms_socket_path`, `gms_mode` in `TorchLlmArgs` | `llmapi/llm_args.py` |
| 2.5 | Tests and documentation | Failover tests, memory sharing verification | `tests/`, `docs/` |

### Implementation Details

**Weeks 1-2: GMS Weight Loader + Sleep/Wake**

GMS integrates as a `LoadFormat` (memory management axis), not a `checkpoint_format` — it changes *how GPU memory is managed*, not *where weights come from*. The `LoadFormat.GMS` branch composes with **any** checkpoint loader (HF for disk, MX for P2P). Pattern validated in [PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053):

```python
# Inside ModelLoader.load() — new LoadFormat.GMS branch
if load_format == LoadFormat.GMS:
    gms_client = get_gms_client(self.llm_args)

    if gms_client.has_committed_weights(tag="model_weights"):
        # RO mode: zero-copy import from existing GMS pool (~100ms)
        model = AutoModelForCausalLM.from_config(config)  # meta init
        model.post_load_weights()  # Fix module aliases (PR #7053 fix)
        materialize_module_from_gms(gms_client, model)     # GMS library call
    else:
        # RW mode: load via checkpoint_loader (HF or MX) under GMS pool
        gms_pool = gms_client.get_mem_pool()               # GMS library call
        with torch.cuda.use_mem_pool(gms_pool, device=device):
            # checkpoint_loader could be HF or MX — GMS doesn't care
            weights = checkpoint_loader.load_weights(checkpoint_dir, mapping)
            model.load_weights(weights, weight_mapper)
        gms_client.finalize_write(model)                   # GMS library call
```

The GMS client library (`gpu_memory_service.client`) handles:
- CUDA VMM allocation and FD-based sharing
- RW/RO locking semantics
- `materialize_module_from_gms()` for zero-copy tensor import

TRT-LLM only needs to:
- Call `materialize_module_from_gms` at the right lifecycle point (after `post_load_weights()`)
- Map sleep/wake tags to GMS operations: `release_with_tag("kv_cache")` frees KV cache via virtual-memory tagged operations while keeping GMS-managed weights untouched
- Handle configuration (`TorchLlmArgs.load_format`, `TorchLlmArgs.gms_socket_path`, `TorchLlmArgs.gms_mode`)

**Weeks 3-4: Shadow Failover + Testing**

- Implement shadow mode in PyExecutor:
  - Shadow worker starts with GMS RO import (weights only, no KV cache)
  - On primary failure: upgrade GMS lock (RO -> RW), allocate KV cache, start serving
- Failover tests: primary crash -> shadow takeover -> continued serving
- Memory tests: N workers, 1x memory footprint
- **Gate:** Shadow takeover < 5s; memory sharing verified

### Phase 2 Success Criteria

| Metric | Target |
|:-------|:-------|
| Memory per worker (same GPU) | 1x weights (vs. Nx baseline) |
| Failover time (shadow takeover) | < 5s |
| GMS import latency | < 500ms |
| Inference correctness after import | Bit-exact with standard loading |

---

## Phase 3: Combined MX+GMS + Extensions — 2-3 Weeks

**Priority:** P2 (Tier 2) — full solution
**Objective:** Unified cross-node P2P + within-node sharing + extension paths

### Deliverables

| # | Deliverable | Description | TRT-LLM Files |
|:--|:-----------|:-----------|:------|
| 3.1 | Combined mode validation | `checkpoint_format="MX"` + `LoadFormat.GMS` with priority cascade | `_torch/pyexecutor/model_loader.py` |
| 3.2 | Disagg interaction | MX/GMS behavior for context vs. generation workers | Integration code |
| 3.3 | E2E tests | Multi-node, multi-worker, failover, disagg scenarios | `tests/` |
| 3.4 | Startup benchmarks | Cold-start benchmarks using the [startup profiling framework](10-startup-profiling.md) | `tests/benchmarks/` |

### Implementation Details

**Weeks 1-2: Combined Loader + Disagg**

Priority cascade for `checkpoint_format="MX"` + `LoadFormat.GMS`:
1. Local GMS (if committed weights exist) — fastest (~100ms, GMS RO import)
2. Remote MX source (P2P to local GPU under GMS pool, then commit to GMS) — fast (~15-30s)
3. Disk/HuggingFace (seed load under GMS pool, commit to GMS, register as MX source) — slow (minutes)

The two-axis model means this cascade requires no special combined code — `LoadFormat.GMS` naturally checks for committed weights first, then falls through to the checkpoint loader (which happens to be MX).

**Weeks 2-3: E2E Testing + Benchmarks**

| Scenario | Config | Validation |
|:---------|:-------|:-----------|
| 3-node scale-out | MX P2P | Cold-start < 30s |
| 4-worker same-GPU | GMS sharing | Memory 1x |
| Primary crash | GMS shadow | Failover < 5s |
| Disagg P/D | MX+GMS | Context + generation workers |
| TP=8, PP=2 | MX rank-matched | Correct inference |

### Phase 3 Success Criteria

| Metric | Target |
|:-------|:-------|
| Cold-start (DeepSeek-V3, 681GB) | < 30s |
| Replica scale-up (Llama-70B) | < 10s |
| Memory per worker (same GPU) | 1x weights |
| Failover time | < 5s |
| Throughput regression | < 2% vs. standard loading |

---

## Total Timeline

| Phase | Duration | Cumulative |
|:------|:---------|:-----------|
| Phase 1: MX | 3-4 weeks | 3-4 weeks |
| Phase 2: GMS | 3-4 weeks | 6-8 weeks |
| Phase 3: Combined | 2-3 weeks | 8-11 weeks |

This is compressed from the original 18-22 week estimate because TRT-LLM is integrating with existing MX/GMS libraries, not building them from scratch.
