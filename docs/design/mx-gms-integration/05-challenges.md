# 5. Challenges and Mitigations

[< Back to Overview](README.md)

> **Status note:** [§18](18-dynamo-pr11000-gaps.md) is authoritative for current GMS API, packaging, and
> readiness. In particular, the pinned `finalize_gms_write()` returns a stats object, native RO attach is blocked on
> SourceIdentity, and no supported `tensorrt_llm[gms]` extra should be claimed yet.

## 1. FP8/Quantization Compatibility

**Challenge:** Source and target must produce identical tensor layouts after post-processing. FP8 quantization, AWQ, and other quant schemes transform weight tensors during `post_load_weights()`. If source and target have different quantization configs, P2P-transferred weights will produce wrong results silently.

**Mitigation:**
- Include quantization config in `MXSourceIdentity` hash
- Both sides must run identical `post_load_weights()` before registration
- Validate tensor shapes/dtypes before transfer acceptance
- FP8 conversion on receive (if source has FP16, target wants FP8 — MX supports this)

```python
identity = MXSourceIdentity(
    model_name="meta-llama/Llama-3.1-70B",
    quantization="fp8",
    quant_config_hash=sha256(serialize(quant_config)),  # Ensures exact match
    ...
)
```

## 2. Non-Contiguous Tensors

**Challenge:** RDMA requires contiguous memory. Some TRT-LLM operations create non-contiguous views (e.g., weight slicing for TP, transposed buffers).

**Mitigation:**
- Detect non-contiguous tensors during enumeration
- Register underlying storage (contiguous) with `__storage` suffix
- Include view reconstruction metadata (shape, stride, offset) in transfer
- Reconstruct views on target after transfer

```python
def _handle_non_contiguous(tensor, name):
    if tensor.is_contiguous():
        return [TensorDescriptor(name=name, ...)]
    else:
        storage = tensor.untyped_storage()
        return [TensorDescriptor(
            name=name,
            data_ptr=storage.data_ptr(),
            size_bytes=storage.nbytes(),
            is_contiguous=False,
            view_shape=tensor.shape,
            view_stride=tensor.stride(),
        )]
```

## 3. Tensor Parallelism Rank Matching

**Challenge:** Each TP rank has different weight slices. Transferring rank 0's weights to rank 3 would produce wrong results.

**Mitigation:**
- Include parallelism sizes (`tensor_parallel_size`, `pipeline_parallel_size`, `expert_parallel_size`, `dtype`) in [`p2p_pb2.SourceIdentity`](https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/p2p_pb2.py)
- Filter `WorkerMetadata` by `worker_rank` (== MPI rank) during discovery
- Validate parallelism config before transfer

```python
from modelexpress.client import MxClient
from modelexpress.trtllm_live_transfer import _build_trtllm_identity  # private today; see MX-2 in §15

client = MxClient(server_url=mx_server_url)
identity = _build_trtllm_identity(model_name, tp_size=mapping.tp_size, ep_size=mapping.moe_ep_size)
list_resp = client.list_sources(identity=identity)

# Per-rank metadata is fetched separately (avoids one giant response)
my_rank = mapping.tp_rank  # for the no-MPI case; MPI deployments use MPI.COMM_WORLD.Get_rank()
for inst in list_resp.instances:
    meta = client.get_metadata(mx_source_id=inst.mx_source_id, worker_id=inst.worker_id)
    if meta.found and meta.worker.worker_rank == my_rank:
        # candidate found
        ...
```

> **Open upstream alignment:** the current `WorkerMetadata` schema only carries `worker_rank` (== MPI rank), not explicit `tp_rank`/`pp_rank`/`ep_rank` fields. For MPI deployments this works because MPI rank == TP rank in our setups, but Ray/K8s deployments without MPI need explicit per-rank addressing. Tracked as MX-3 in [§15 Upstream Alignment Requests](15-prototype-validation-plan.md#upstream-alignment-requests).

## 4. Pipeline Parallelism Layers

**Challenge:** Different PP ranks have different layer subsets. Rank 0 might have layers 0-15, rank 1 has layers 16-31.

**Mitigation:**
- Include `pp_rank` in `MXSourceIdentity`
- Each PP rank only transfers its layer subset
- Tensor enumeration only includes locally-held layers
- Validate layer ranges match before transfer

## 5. MoE Expert Distribution

**Challenge:** Expert parallelism distributes experts differently across ranks. Load balancer state (EPLB slot assignments) may vary.

**Mitigation:**
- Include `ep_rank` in `MXSourceIdentity`
- Re-run `load_balancer.finalize()` after P2P transfer
- Transfer load balancer state separately if static (offline EPLB)
- For online EPLB: new replica starts with default assignment, adapts dynamically

## 6. CUDA VMM Integration (GMS-Specific)

**Challenge:** GMS uses CUDA Virtual Memory Management (VMM) with file descriptor passing. TRT-LLM uses PyTorch's default CUDA allocator. Mixing these requires careful lifecycle management.

**Mitigation:**
- Use `torch.cuda.memory.CUDAPluggableAllocator` API (provided by the GMS client library)
- The GMS library already implements the allocator (`CUDAPluggableAllocator` + `MemPool`), the CUDA VMM FD import/export, and zero-copy tensor construction (`materialize_module_from_gms`). **TRT-LLM does not reimplement any of this** — it only wraps model loading inside the `gms_use_mem_pool(tag, device)` context manager for the RW path, or calls `materialize_module_from_gms(mgr, model, device_index=N)` for the RO path. See [Implementation & API Design](04-implementation-plan.md#library-inventory) for a full inventory.
- Handle allocation/deallocation lifecycle at TRT-LLM orchestration level:
  - RW mode: wrap model loading in `gms_backend.mem_pool_scope(device)` context manager (which delegates to `gms_use_mem_pool`); move stray params into pool via `move_untracked_params()`; finalize via `finalize_write()` which delegates to upstream `finalize_gms_write()` (commits, disconnects RW, reconnects as RO, and remaps virtual addresses in one call)
  - RO mode: call `materialize_module_from_gms(mgr, model, device_index=...)` after `post_load_weights()`
  - VMM safety: `connect()` applies `patch_empty_cache()` from `gpu_memory_service.integrations.common.patches` to prevent segfaults from `torch.cuda.empty_cache()` on VMM-backed allocations
  - Shutdown: `cleanup()` calls `mgr.close()` and `evict_gms_client_memory_manager()` to release the per-tag registry slot
- **Risk mitigation:** Start with simple contiguous allocations; handle edge cases (views, in-place ops) incrementally

**Known complexity from PR #7053:**
- Module path resolution issues with aliased layers — `post_load_weights()` creates aliases that the GMS import path must resolve
- Multi-rank support — each GPU has its own GMS process (one per tag), so multi-GPU setups require connecting each rank to its corresponding per-GPU GMS socket (auto-resolved via GPU UUID)

## 7. Module Path Resolution (GMS-Specific)

**Challenge:** TRT-LLM's `post_load_weights()` creates layer aliases (e.g., `LlamaForCausalLM` assigns `layer.next_attn = self.model.layers[idx + 1].self_attn`). Because `self_attn` is an `nn.Module`, PyTorch's `__setattr__` registers it in `layer._modules['next_attn']`. This causes GMS to store duplicate keys for the same physical tensor (e.g., both `model.layers.0.next_attn.o_proj.weight` and `model.layers.1.self_attn.o_proj.weight` point to the same tensor). On the read path, if `post_load_weights()` has not been called before `materialize_module_from_gms()`, the `next_attn` attribute is still `None` and resolution fails with `AttributeError: Cannot resolve 'o_proj' in 'model.layers.0.next_attn.o_proj.weight'`.

**This bug was discovered and fixed in the [GMS prototype PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053#discussion_r2105412837).**

**Mitigation (proven in PR #7053):**
- Call `model.post_load_weights()` (top-level only) **before** `materialize_module_from_gms()` to set up structural cross-references
- This is safe because `post_load_weights()` only performs Python pointer assignments at meta-init time — no tensor operations
- The duplicate GMS keys are harmless after the fix: both resolve to the same `nn.Module` object, and the second assignment is a no-op
- Test with all models that use `post_load_weights()` aliases, especially `LlamaForCausalLM`, `DeepSeek`, and any model with shared embedding/LM head

**Limitation of the per-PR mitigation:** running the full `post_load_weights()` against meta tensors so that aliases get wired before `materialize_module_from_gms()` works for category A (alias wiring) but means categories B (data transforms) and D (derived Python-side state) execute on meta tensors — a soft RO/RW divergence. The same conflated-method problem surfaces from the MX side in PR #14151 when receivers re-run `post_load_weights()` on already-transformed bytes. The holistic fix is to decompose `post_load_weights()` into staged hooks (`setup_aliases` / `transform_weights` / `cache_derived_state`) so each consumer can pick the exact subset it needs. See [§16 Staged Post-Load Hooks](16-staged-post-load-hooks.md) for the full design.

## 8. GMS API Stability

**Challenge:** The GPU Memory Service (GMS) API has stabilized as of [PR #7575](https://github.com/ai-dynamo/dynamo/pull/7575) (the official TRT-LLM sleep/wake integration). The prior `LoadFormat.GMS` integration in TRT-LLM PR #13045 was originally written against an unmerged convenience-function API that did not survive PR #7575; the API used in production today is the class-based [`GMSClientMemoryManager`](https://github.com/ai-dynamo/dynamo/blob/main/lib/gpu_memory_service/client/memory_manager.py) plus the helpers under `gpu_memory_service.client.torch.*` and `gpu_memory_service.integrations.common.utils.finalize_gms_write`. PR #13045 has been refactored to call this stable API (commit `62ac40f6b`).

**Mitigation:**
- Keep a thin `GPUMemoryBackend` protocol in TRT-LLM (see [Implementation & API Design](04-implementation-plan.md#gms-api-stability-abstraction)) so all upstream API calls go through one adapter class. When the upstream Layer 2 API drifts, only this adapter needs to change — call sites in `model_loader.py` are stable:
  ```python
  class GPUMemoryBackend(Protocol):
      def connect(self) -> bool: ...
      @property
      def is_rw(self) -> Optional[bool]: ...
      def has_committed_weights(self) -> bool: ...
      def mem_pool_scope(
          self,
          device: Optional[torch.device] = None,
      ) -> "Iterator[None]": ...
      def materialize_module(self, model: torch.nn.Module) -> None: ...
      def finalize_write(self, model: torch.nn.Module) -> int: ...
      def move_untracked_params(self, model: torch.nn.Module) -> None: ...
      def cleanup(self) -> None: ...
  ```
- `mem_pool_scope(device)` is a context manager that scopes CUDA allocations to the GMS pool — replaces the older `get_mem_pool() -> torch.cuda.MemPool` style. Internally delegates to upstream `gms_use_mem_pool(tag, device)`.
- `finalize_write()` delegates to upstream `gpu_memory_service.integrations.common.utils.finalize_gms_write()`, which performs `register_module_tensors → cuda.synchronize → commit → connect(RO) → remap_all_vas` in one call. Returns the total bytes committed.
- `move_untracked_params()` mirrors the upstream private `gpu_memory_service.integrations.trtllm.model_loader._move_untracked_params` — iterates via `_iter_module_tensors`, dedups by storage pointer, allocates fresh GMS mappings via `create_mapping()`, and rebinds via `_tensor_from_pointer`. We intentionally re-implement instead of importing the private symbol; promoting it to public is tracked as GMS-2 in [§15 Upstream Alignment Requests](15-prototype-validation-plan.md#upstream-alignment-requests).
- `connect()` applies `patch_empty_cache()` from `gpu_memory_service.integrations.common.patches` to prevent `torch.cuda.empty_cache()` from segfaulting on VMM-backed GMS allocations — this is critical for MoE models whose load balancer calls `empty_cache()` during `make_tensor_host_accessible()`
- We deliberately do **NOT** use the upstream `setup_gms()` integration entry point. `setup_gms()` works by `_trt_loader.ModelLoader.load = patched_load` — runtime monkey-patching of TRT-LLM internals from outside, which is opaque at code-review time and conflicts with TRT-LLM's two-axis design. TRT-LLM owns the integration policy; the `GPUMemoryBackend` adapter is the explicit, reviewable boundary.
- Pin the upstream dep to a tested major: `gpu-memory-service>=0.9.0,<0.10.0` (declared as the `gms` extra in `setup.py`).
- Have a fallback plan: if GMS API drifts further, a `CudaIpcBackend` could implement the same protocol (less featured — no crash resilience or zero-copy, but uses stable CUDA IPC APIs)

## 9. Transfer Backend Selection

**Open question:** Should TRT-LLM use NIXL (like vLLM) or Mooncake TransferEngine?

| Backend | Pros | Cons |
|:--------|:-----|:-----|
| **NIXL** | vLLM-proven; MX default; broader fabric support | — |
| **Mooncake TransferEngine** | MX proto suggests this for TRT-LLM; TRT-LLM already uses Mooncake for disagg | Less mature for weight transfer |

**Recommendation:** Start with NIXL for Phase 1 (proven, matches vLLM). Evaluate Mooncake as an alternative backend in Phase 3. Note that the transfer backend selection is largely **transparent to TRT-LLM** — it is handled inside the MX client library. TRT-LLM calls `MxLiveWeightLoader(mx_server=url).load_weights(checkpoint_dir, mapping=, model=)` for the receive side and `publish_model_params(torch_model)` for the publish side; the MX library handles the underlying transfer mechanism (NIXL today; Mooncake or other backends in the future).

## 10. Coordination with MX Prototype (PR #12898)

**Challenge:** [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) from the MX team adds `LoadFormat.PRESHARDED = 3` as a prototype MX integration. This conflates weight source and memory management into a single `LoadFormat` value, which prevents clean composition with GMS. Specific conflicts:
- `LoadFormat.PRESHARDED = 3` occupies the enum slot we need for `LoadFormat.GMS`
- The `MODEL_EXPRESS_URL` / `MODEL_EXPRESS_TARGET` env vars bypass `TorchLlmArgs` configuration
- Source publish logic is split between `model_loader.py` and `worker.py` (duplicated `publish_from_worker` calls) with env-var-gated control flow

**Mitigation:**
- Coordinate with MX team to refactor toward the two-axis model: MX as `checkpoint_format="MX"` (weight source), not a `LoadFormat` (see [Implementation & API Design](04-implementation-plan.md#design-principle-two-orthogonal-axes))
- Adopt PR #12898's validated insights: pre-`post_load_weights()` publish timing, `_weights_presharded` TP-skip on Linear modules
- If PR #12898 lands first: build on top of it incrementally — extract the `PRESHARDED` logic into an `MXCheckpointLoader`, migrate env var to `TorchLlmArgs` field, and add `LoadFormat.GMS` at the next available enum slot

## Complexity Summary

| Area | Integration Axis | Complexity | Phase | Owner | Risk |
|:-----|:----------------|:----------|:------|:------|:-----|
| MX checkpoint loader | `checkpoint_format` | Medium | 1 | TRT-LLM | Low |
| MX identity/rank matching | `checkpoint_format` | Medium | 1 | TRT-LLM | Low |
| Pre-sharded TP skip | Cross-cutting | Low | 1 | TRT-LLM (adopt from [PR #12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898)) | Low |
| MX fallback logic | `checkpoint_format` | Low | 1 | TRT-LLM (orchestration); MX library (transfer) | Low |
| Non-contiguous tensors | `checkpoint_format` | Medium | 1 | MX library handles NIXL registration | Medium |
| NIXL/RDMA transfer | `checkpoint_format` | Medium | 1 | MX library (transparent to TRT-LLM) | Medium |
| PR #12898 coordination | Both | Low | 1 | TRT-LLM + MX team | Medium |
| GMS loading mode (`LoadFormat.GMS`) | `LoadFormat` | Medium | 2 | TRT-LLM (orchestration); GMS library (allocator, VMM) | Medium |
| CUDA VMM / GMS allocator | `LoadFormat` | **High** | 2 | GMS library (TRT-LLM wraps with context manager) | **High** |
| Module path resolution | `LoadFormat` | Medium | 2 | TRT-LLM (call `post_load_weights()` before GMS import) | Medium |
| Sleep/wake GMS tag mapping | `LoadFormat` | Medium | 2 | TRT-LLM | Low |
| Shadow failover executor | `LoadFormat` | **High** | 2 | TRT-LLM (new code; GMS provides only `upgrade_lock()`) | **High** |
| GMS API stability protocol | `LoadFormat` | Low | 2 | TRT-LLM | Low |
| Combined mode validation | Both (composition) | Low | 3 | TRT-LLM | Low |
| Disagg interaction | Both | Medium | 3 | TRT-LLM | Medium |
