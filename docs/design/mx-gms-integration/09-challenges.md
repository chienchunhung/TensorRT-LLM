# 9. Challenges and Mitigations

[< Back to Overview](README.md)

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
- Include `worker_rank`, `tp_size`, `pp_rank`, `ep_rank` in `MXSourceIdentity`
- Filter sources by exact rank match during discovery
- Validate parallelism config before transfer

```python
sources = mx_client.list_sources(identity, status=READY)
my_rank = mapping.tp_rank
candidates = [s for s in sources if s.worker_rank == my_rank]
```

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
- Use `torch.cuda.memory.CUDAPluggableAllocator` API
- Implement `GMSAllocator` that routes alloc/free to GMS
- Handle allocation/deallocation lifecycle:
  - RW mode: allocator creates mappings in GMS pool
  - RO mode: allocator imports existing mappings via FD
  - Shutdown: release all GMS handles before process exit
- **Risk mitigation:** Start with simple contiguous allocations; handle edge cases (views, in-place ops) incrementally

**Known complexity from PR #7053:**
- Module path resolution issues with aliased layers — `post_load_weights()` creates aliases that the GMS import path must resolve
- Limited multi-rank support — multi-GPU GMS sharing requires per-device socket management

## 7. Module Path Resolution (GMS-Specific)

**Challenge:** TRT-LLM's `post_load_weights()` creates layer aliases (e.g., shared embedding/LM head). When importing from GMS, the module path used to find tensors may not match the storage path.

**Mitigation:**
- Build a module-path-to-storage-path mapping during RW commit
- Store this mapping as GMS metadata alongside the tensors
- During RO import, use the mapping to reconstruct aliases
- Test with all models that use `post_load_weights()` aliases

## 8. GMS API Stability

**Challenge:** "GPU Memory Service" does not appear as a formally named component in public Dynamo documentation. The prototype in PR #7053 may be using an internal/pre-release API that could change.

**Mitigation:**
- Define a thin abstraction layer between TRT-LLM and GMS:
  ```python
  class GPUMemoryBackend(Protocol):
      def create_mapping(self, size: int, tag: str) -> MemoryHandle: ...
      def import_mapping(self, tag: str) -> MemoryHandle: ...
      def commit(self, tag: str) -> None: ...
      def release(self, tag: str) -> None: ...
  ```
- GMS client implements this protocol; if the API changes, only the adapter changes
- Verify API stability with Dynamo team before Phase 2 starts
- Have a fallback plan: if GMS API is unstable, Phase 2 can use CUDA IPC directly (less featured but stable)

## 9. Transfer Backend Selection

**Open question:** Should TRT-LLM use NIXL (like vLLM) or Mooncake TransferEngine?

| Backend | Pros | Cons |
|:--------|:-----|:-----|
| **NIXL** | vLLM-proven; MX default; broader fabric support | — |
| **Mooncake TransferEngine** | MX proto suggests this for TRT-LLM; TRT-LLM already uses Mooncake for disagg | Less mature for weight transfer |

**Recommendation:** Start with NIXL for Phase 1 (proven, matches vLLM). Evaluate Mooncake as an alternative backend in Phase 3. The `WeightLoaderProtocol` abstraction allows swapping transfer backends without changing the loader logic.

## Complexity Summary

| Area | Complexity | Phase | Risk |
|:-----|:----------|:------|:-----|
| Tensor enumeration | Medium | 1 | Low |
| NIXL wrapper | Medium | 1 | Medium |
| TP/PP/EP rank matching | Medium | 1 | Low |
| MX gRPC integration | Low | 1 | Low |
| Non-contiguous tensors | Medium | 1 | Medium |
| CUDA VMM / GMS allocator | **High** | 2 | **High** |
| Module path resolution | Medium | 2 | Medium |
| Sleep/wake mapping | Medium | 2 | Low |
| Shadow failover executor | **High** | 2 | **High** |
| Combined loader | Low | 3 | Low |
| Disagg interaction | Medium | 3 | Medium |
