# 12. Risk Assessment

[< Back to Overview](README.md)

## Technical Risks

| Risk | Impact | Probability | Mitigation |
|:-----|:-------|:-----------|:-----------|
| CUDA VMM complexity | High | Medium | Start simple; incremental rollout; GMS team support |
| Quantization incompatibility across P2P | High | Low | Strict identity matching with quant config hash; validation before transfer |
| GMS API instability | High | Medium | Thin abstraction layer; verify with Dynamo team; CUDA IPC fallback |
| Performance regression from GMS allocator | Medium | Low | Benchmark gates; GMS allocator only for weight tensors, not compute |
| Multi-rank race conditions | Medium | Medium | Careful synchronization; barrier after P2P; extensive multi-GPU tests |
| Module path resolution (aliased layers) | Medium | Medium | Build explicit path mapping during RW commit; test all model architectures |
| FP8 fusion pattern mismatch after transfer | Medium | Low | Both sides run identical `post_load_weights()`; binary validation |

## Strategic Risks

### 1. Dependency on External Projects

**Risk:** MX and GMS are developed by the Dynamo team. API changes could break the integration.

**Mitigation:**
- Define `GPUMemoryBackend` protocol as abstraction layer
- Version compatibility matrix
- Automated integration tests in CI
- Regular sync with MX/GMS teams

### 2. GMS Naming and API Stability

**Risk:** "GPU Memory Service" does not appear in public Dynamo documentation. The component may be renamed, restructured, or have its API changed before GA.

**Specific concerns:**
- PR #7053 is a prototype — the API may not be finalized
- KVBM (KV Block Manager) in public docs may subsume GMS functionality
- The socket-based locking protocol may change

**Mitigation:**
- Verify component name and API stability with Dynamo team before Phase 2
- The `GPUMemoryBackend` protocol allows swapping implementations
- If GMS is unstable, fall back to:
  - **CUDA IPC** (stable CUDA API) for same-node memory sharing — less featured but proven
  - **torch.multiprocessing.shared_memory** for simpler cases
- Phase 2 start should be gated on GMS API stability confirmation

### 3. vLLM Has MX Already

**Risk:** vLLM shipped `--load-format mx` first. If TRT-LLM delays, the Dynamo ecosystem may optimize primarily for vLLM.

**Mitigation:**
- Phase 1 (MX) is P1 priority — start immediately
- Study vLLM's implementation (`vllm/model_executor/model_loader/mx_loader.py`) to learn from their approach
- Target within-20% performance parity as Phase 1 exit criterion

### 4. Ecosystem Lock-In

**Risk:** Deep integration with Dynamo ecosystem may reduce TRT-LLM's independence.

**Mitigation:**
- MX and GMS are optional — `--load-format hf` remains the default
- Abstraction layers (`WeightLoaderProtocol`, `GPUMemoryBackend`) allow alternative backends
- Both MX and GMS are open-source

### 5. Timeline Risk

**Risk:** 16-22 week estimate may stretch due to CUDA VMM complexity and edge cases.

**Analysis of original 12-18 week estimate:**
- The CUDA VMM integration alone is notoriously tricky
- PR #7053 prototype already shows "module path resolution issues" and "limited multi-rank support"
- These edge cases typically consume 2-4 extra weeks

**Revised estimate:** 16-22 weeks total (vs. original 12-18 weeks)
- Phase 1: 6-8 weeks (was 4-6) — added vLLM comparison testing
- Phase 2: 6-8 weeks (was 3-5) — added CUDA VMM complexity buffer
- Phase 3: 4-6 weeks (was 2-3) — added disagg interaction and KV extension design

## Alternative Approaches Considered

| Alternative | Pros | Cons | Verdict |
|:-----------|:-----|:-----|:--------|
| **External wrapper only** (like PR #7053) | No core changes; fast start | Fragile; breaks with updates; poor UX | Not for production |
| **GMS-only (no MX)** | Simpler; within-node only | Doesn't solve cross-node cold-start | Insufficient |
| **MX-only (no GMS)** | Solves cross-node cold-start | No memory sharing; no crash resilience | Partial solution |
| **Custom implementation** | Full control | 6-12 months; duplicates existing systems | Not recommended |
| **Phased MX + GMS** | Leverages existing systems; incremental value | Dynamo dependency | **Recommended** |
