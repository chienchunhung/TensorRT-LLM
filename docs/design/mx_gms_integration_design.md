# TensorRT-LLM Integration with ModelExpress and GPU Memory Service

**Status:** Draft
**Authors:** TensorRT-LLM Team
**Created:** 2026-04-01
**Last Updated:** 2026-04-01

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background](#background)
3. [Problem Statement](#problem-statement)
4. [Goals and Non-Goals](#goals-and-non-goals)
5. [Current State Analysis](#current-state-analysis)
6. [Proposed Architecture](#proposed-architecture)
7. [Implementation Strategy](#implementation-strategy)
8. [API Design](#api-design)
9. [Challenges and Mitigations](#challenges-and-mitigations)
10. [Performance Expectations](#performance-expectations)
11. [Complexity Assessment](#complexity-assessment)
12. [Alternative Approaches](#alternative-approaches)
13. [Risks and Concerns](#risks-and-concerns)
14. [Timeline and Milestones](#timeline-and-milestones)
15. [Open Questions](#open-questions)
16. [Startup Performance Profiling](#startup-performance-profiling)
17. [References](#references)

---

## Executive Summary

This document proposes integrating TensorRT-LLM with two complementary systems from the Dynamo ecosystem:

- **ModelExpress (MX)**: GPU-to-GPU model weight streaming via NIXL/RDMA for fast cold-start across nodes
- **GPU Memory Service (GMS)**: Out-of-process GPU memory management for zero-copy sharing and crash-resilient failover within nodes

The integration aims to dramatically reduce model loading time (from minutes to seconds), enable efficient multi-worker memory sharing, and support fault-tolerant inference deployments.

**Recommended Approach:** Phased integration starting with independent MX and GMS support, then combining them for the complete solution.

---

## Background

### ModelExpress (MX)

ModelExpress is a Rust-based service that coordinates GPU-to-GPU model weight transfers across a cluster:

- **Single-source download**: Only one pod downloads from HuggingFace; others receive via P2P
- **RDMA transfers**: Uses NIXL/UCX for high-speed GPU-to-GPU transfer (~15s for 681GB DeepSeek-V3)
- **Content-addressed coordination**: SHA256-based source identity ensures compatible transfers
- **Three-tier fallback**: RDMA → GPUDirect Storage → Disk
- **Metadata backends**: Redis or Kubernetes CRD

### GPU Memory Service (GMS)

GMS is an out-of-process GPU memory manager that decouples memory ownership from processes:

- **Zero-copy sharing**: Multiple workers share the same GPU memory for model weights
- **Crash resilience**: Memory persists when worker crashes; new worker imports existing memory
- **VA-stable failover**: Shadow engines can release/reclaim memory while keeping tensor pointers valid
- **Socket-based locking**: Connection IS the lock; automatic release on crash

### TensorRT-LLM Current Architecture

TensorRT-LLM's PyTorch backend provides:

- **Virtual memory tagging**: `scope(tag)`, `release_with_tag()`, `materialize_with_tag()`
- **Checkpoint loader registry**: `@register_checkpoint_loader()` for custom formats
- **Weight mapper abstraction**: Pluggable weight transformation and assignment
- **Sleep/wake infrastructure**: Memory release and restoration for failover scenarios

---

## Problem Statement

### Current Pain Points

| Problem | Impact | Current State |
|---------|--------|---------------|
| **Slow cold-start** | Minutes to serve first request | Each replica loads from disk/network independently |
| **Memory waste** | Limits workers per GPU | Multiple workers duplicate model weights |
| **Slow failover** | Service degradation during recovery | Failed worker requires full reload |
| **Storage bottleneck** | Scaling limited by I/O bandwidth | All replicas compete for storage bandwidth |
| **No crash resilience** | Lost work on process crash | GPU memory released when process dies |

### Target Use Cases

1. **Autoscaling**: Spin up new replicas in seconds, not minutes
2. **Multi-tenant serving**: Multiple workers share weights on same GPU
3. **Shadow failover**: Instant switchover when primary fails
4. **Rolling updates**: Zero-downtime model version updates
5. **Disaggregated serving**: Efficient prefill/decode separation

---

## Goals and Non-Goals

### Goals

1. **Native MX support**: `--load-format mx` for P2P weight loading
2. **Native GMS support**: `--load-format gms` for shared memory loading
3. **Combined MX+GMS**: Cross-node P2P with within-node sharing
4. **Backward compatibility**: Existing workflows unchanged
5. **Extension points**: Clean APIs for future backends

### Non-Goals

1. Modifying MX or GMS core implementations
2. Supporting legacy TensorRT engine backend (PyTorch backend only)
3. KV cache sharing via GMS (handled separately by KVBM)
4. Automatic MX server deployment (separate concern)

---

## Current State Analysis

### Existing Prototype: GMS + TRT-LLM (PR #7053)

**Location**: https://github.com/ai-dynamo/dynamo/pull/7053

**Approach**:
- External patches to TRT-LLM model loading
- Two-phase initialization: meta tensors → GMS materialization
- Dual mode: RW (writer) and RO (reader)
- Integrates with TRT-LLM's virtual memory tagging for KV cache

**Key Implementation Details**:
```python
# Phase 1: Meta initialization (establish cross-references)
with MetaInitMode():
    model = AutoModelForCausalLM.from_config(config)
    model.post_load_weights()  # Establish layer aliases

# Phase 2: GMS materialization
if rw_mode:
    load_weights_normally(model)
    move_to_gms_pool(model)
    gms_client.commit()
else:  # RO mode
    materialize_module_from_gms(model, gms_client)
```

**Limitations**:
- Requires external patching (not native TRT-LLM)
- Module path resolution issues with aliased layers
- Limited multi-rank support

### Existing Prototype: MX + TRT-LLM

**Location**: https://github.com/ai-dynamo/modelexpress (branch: kavink/trtllm)

**Status**: Framework declared but not implemented

**Proto Definition**:
```protobuf
enum BackendFramework {
    BACKEND_FRAMEWORK_VLLM = 1;
    BACKEND_FRAMEWORK_SGLANG = 2;
    BACKEND_FRAMEWORK_TRT_LLM = 3;  // Declared, not implemented
}

message WorkerMetadata {
    oneof backend_metadata {
        bytes nixl_metadata = 2;           // vLLM path
        string transfer_engine_session_id = 10;  // TRT-LLM path (Mooncake)
    }
}
```

**Gaps**:
- No `trtllm_loader.py` implementation
- No TransferEngine/NIXL wrapper for TRT-LLM
- No integration tests

### TRT-LLM Extension Points

**Existing**:
- `@register_checkpoint_loader(format)` - Custom checkpoint formats
- `virtual_memory_scope(tag)` - Tagged memory allocation
- `release_with_tag()` / `materialize_with_tag()` - Sleep/wake
- `ModelLoader` with configurable weight loading

**Missing**:
- Pluggable GPU memory allocator
- Tensor enumeration API for P2P registration
- Post-load callback for external registration
- External memory import (CUDA VMM FD import)

---

## Proposed Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Dynamo Orchestration                            │
│                    (Router, Planner, Frontend, KVBM)                         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────────┐
│                          MX Metadata Server                                  │
│                     (Redis/K8s CRD coordination)                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ gRPC
        ┌──────────────────────────┼──────────────────────────────────────────┐
        │                          │                                          │
   ┌────┴────┐                ┌────┴────┐                               ┌────┴────┐
   │ Node A  │                │ Node B  │                               │ Node C  │
   │ (Seed)  │                │(Replica)│                               │(Replica)│
   └────┬────┘                └────┬────┘                               └────┬────┘
        │                          │                                          │
   ┌────┴────┐                ┌────┴────┐                               ┌────┴────┐
   │   GMS   │ ──────────────▶│   GMS   │                               │   GMS   │
   │ (Local) │   P2P via MX   │ (Local) │                               │ (Local) │
   └────┬────┘                └────┬────┘                               └────┬────┘
        │                          │                                          │
   ┌────┴────┐                ┌────┴────┐                               ┌────┴────┐
   │ TRT-LLM │                │ TRT-LLM │                               │ TRT-LLM │
   │Worker 1 │                │Worker 2 │                               │Worker N │
   └─────────┘                └─────────┘                               └─────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **MX Server** | Coordinate P2P transfers across nodes; track source availability |
| **GMS (per node)** | Manage GPU memory; enable zero-copy sharing within node |
| **TRT-LLM** | Load models; integrate with MX/GMS via clean APIs |
| **NIXL/UCX** | Execute actual GPU-to-GPU RDMA transfers |

### Data Flow

**Scenario: New Replica Startup**

```
1. TRT-LLM Worker starts with --load-format mx-gms

2. Check local GMS:
   └─ If weights exist in GMS (RO mode available):
      └─ Import from GMS → Done (fastest path)

3. If no local GMS weights, query MX server:
   └─ ListSources(identity, status=READY)
   └─ If sources exist:
      └─ GetMetadata(source_id)
      └─ P2P receive via NIXL
      └─ Store in local GMS
      └─ Done (P2P path)

4. If no MX sources:
   └─ Load from disk/HuggingFace
   └─ Store in local GMS
   └─ Publish to MX server
   └─ Done (seed path)
```

---

## Implementation Strategy

### Phased Approach

```
Phase 1: MX + TRT-LLM          Phase 2: GMS + TRT-LLM         Phase 3: MX + GMS + TRT-LLM
─────────────────────          ──────────────────────         ────────────────────────────

┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│  MX Integration │            │ GMS Integration │            │ Combined System │
│                 │            │                 │            │                 │
│ • P2P transfer  │            │ • Zero-copy     │            │ • Cross-node P2P│
│ • Cross-node    │     +      │ • Within-node   │     =      │ • Local sharing │
│ • Cold-start    │            │ • Failover      │            │ • Full solution │
└─────────────────┘            └─────────────────┘            └─────────────────┘
```

### Phase 1: MX + TRT-LLM (Cross-Node P2P)

**Objective**: Enable P2P weight transfer across nodes

**Deliverables**:
1. `@register_weight_loader("mx")` implementation
2. Tensor enumeration API
3. NIXL/TransferEngine wrapper for TRT-LLM
4. Integration with MX gRPC client
5. Three-tier fallback (P2P → GDS → Disk)

**TRT-LLM Changes**:
```python
# New file: tensorrt_llm/_torch/weight_loaders/mx_loader.py
class ModelExpressWeightLoader(BaseWeightLoader):
    def load_weights(self, model, mapping, config):
        # 1. Query MX for sources
        # 2. If source exists: P2P receive
        # 3. Else: load from disk, publish to MX
        pass

    def register_as_source(self, model):
        # Enumerate tensors, register with NIXL, publish to MX
        pass
```

### Phase 2: GMS + TRT-LLM (Within-Node Sharing)

**Objective**: Enable zero-copy weight sharing within a node

**Deliverables**:
1. `@register_weight_loader("gms")` implementation
2. Pluggable GPU memory allocator hook
3. CUDA VMM FD import support
4. Sleep/wake integration
5. RW/RO mode handling

**TRT-LLM Changes**:
```python
# New file: tensorrt_llm/_torch/weight_loaders/gms_loader.py
class GMSWeightLoader(BaseWeightLoader):
    def __init__(self, gms_client, mode="auto"):
        self.gms_client = gms_client
        self.mode = mode  # "rw", "ro", "auto"

    def load_weights(self, model, mapping, config):
        if self._should_use_ro_mode():
            self._import_from_gms(model)
        else:
            self._load_and_commit_to_gms(model)
```

### Phase 3: MX + GMS + TRT-LLM (Combined)

**Objective**: Full solution with cross-node P2P and within-node sharing

**Deliverables**:
1. `@register_weight_loader("mx-gms")` combined loader
2. Unified configuration
3. Optimized data paths
4. End-to-end testing

**TRT-LLM Changes**:
```python
# New file: tensorrt_llm/_torch/weight_loaders/mx_gms_loader.py
class MXGMSWeightLoader(BaseWeightLoader):
    def load_weights(self, model, mapping, config):
        # Priority order:
        # 1. Local GMS (if committed weights exist)
        # 2. Remote MX source (P2P to local GMS)
        # 3. Disk/HuggingFace (seed, commit to GMS, publish to MX)
        pass
```

---

## API Design

### Public APIs to Add

#### 1. Weight Loader Protocol

```python
# tensorrt_llm/_torch/weight_loaders/base.py

from typing import Protocol, Dict, Optional, Tuple
from dataclasses import dataclass
import torch

@dataclass
class TensorDescriptor:
    """Describes a GPU tensor for P2P registration."""
    name: str
    data_ptr: int
    size_bytes: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    is_contiguous: bool
    storage_offset: int = 0

class WeightLoaderProtocol(Protocol):
    """Interface for custom weight loading backends."""

    def load_weights(
        self,
        model: torch.nn.Module,
        mapping: "Mapping",
        config: "PretrainedConfig",
    ) -> None:
        """Load weights into model from custom source."""
        ...

    def get_tensor_descriptors(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, TensorDescriptor]:
        """Return tensor metadata for P2P registration."""
        ...

    def supports_source_mode(self) -> bool:
        """Whether this loader can act as P2P source."""
        ...

    def cleanup(self) -> None:
        """Release any resources held by the loader."""
        ...
```

#### 2. Tensor Enumeration API

```python
# tensorrt_llm/_torch/utils/tensor_utils.py

def enumerate_model_tensors(
    model: torch.nn.Module,
    include_buffers: bool = True,
    include_quantization_scales: bool = True,
    deduplicate_by_storage: bool = True,
) -> Dict[str, TensorDescriptor]:
    """
    Enumerate all GPU tensors in a model for P2P registration.

    Args:
        model: The model to enumerate tensors from
        include_buffers: Include registered buffers (not just parameters)
        include_quantization_scales: Include FP8/INT8 scale tensors
        deduplicate_by_storage: Deduplicate tied weights by data_ptr

    Returns:
        Dictionary mapping tensor names to descriptors

    Handles:
        - Parameters and buffers
        - Tied weights (deduplicated)
        - Non-contiguous views (reports underlying storage)
        - Quantization scales (weight_scale_inv, etc.)
    """
    ...
```

#### 3. Memory Allocator Hook

```python
# tensorrt_llm/_torch/pyexecutor/model_loader.py

from typing import Callable, Optional

AllocatorFn = Callable[[int, torch.dtype, str], torch.Tensor]

class ModelLoader:
    def __init__(
        self,
        llm_args: "TorchLlmArgs",
        mapping: "Mapping",
        *,
        custom_allocator: Optional[AllocatorFn] = None,
        post_load_callback: Optional[Callable[[torch.nn.Module], None]] = None,
        **kwargs,
    ):
        """
        Args:
            custom_allocator: Function(size_bytes, dtype, tag) -> torch.Tensor
                             Routes GPU allocation through custom backend (e.g., GMS)
            post_load_callback: Called after model loaded and post_load_weights()
                               completed. Use for P2P tensor registration.
        """
        self._custom_allocator = custom_allocator
        self._post_load_callback = post_load_callback
```

#### 4. External Memory Import

```python
# tensorrt_llm/_torch/memory/external_memory.py

def import_cuda_memory(
    fd: int,
    size: int,
    device: int,
) -> torch.Tensor:
    """
    Import external CUDA memory via file descriptor.

    Used by GMS to import memory allocated by the GMS server.

    Args:
        fd: File descriptor from GMS (via cuMemExportToShareableHandle)
        size: Size in bytes
        device: CUDA device index

    Returns:
        torch.Tensor backed by imported memory
    """
    ...

def export_cuda_memory(
    tensor: torch.Tensor,
) -> Tuple[int, int]:
    """
    Export CUDA memory as file descriptor.

    Used by GMS to share memory with other processes.

    Args:
        tensor: GPU tensor to export

    Returns:
        (fd, size) tuple
    """
    ...
```

### Configuration Schema

```python
# tensorrt_llm/llmapi/llm_args.py

class TorchLlmArgs(BaseLlmArgs):
    # Existing fields...

    # New fields for MX/GMS integration
    load_format: Literal["auto", "hf", "dummy", "mx", "gms", "mx-gms"] = "auto"

    # MX-specific configuration
    mx_server_url: Optional[str] = None
    mx_metadata_backend: Optional[Literal["redis", "kubernetes"]] = None
    mx_heartbeat_interval_secs: int = 30

    # GMS-specific configuration
    gms_socket_path: Optional[str] = None
    gms_mode: Literal["auto", "rw", "ro"] = "auto"
    gms_tag: str = "model_weights"

    # Combined configuration
    enable_weight_sharing: bool = False  # Shorthand for MX+GMS
```

---

## Challenges and Mitigations

### 1. FP8/Quantization Compatibility

**Challenge**: Source and target must produce identical tensor layouts after post-processing.

**Mitigation**:
- Include quantization config in SourceIdentity hash
- Both sides run identical `post_load_weights()` before registration
- Validate tensor shapes/dtypes before transfer

```python
# Source identity includes:
identity = SourceIdentity(
    model_name="meta-llama/Llama-3.1-70B",
    dtype="float16",
    quantization="fp8",  # Ensures compatible quantization
    tp_size=8,
    pp_size=1,
    extra_params={"quant_config": serialize(quant_config)},
)
```

### 2. Non-Contiguous Tensors

**Challenge**: RDMA requires contiguous memory; some TRT-LLM operations create views.

**Mitigation**:
- Detect non-contiguous tensors during enumeration
- Register underlying storage with `__storage` suffix
- Reconstruct views on target after transfer

```python
def _handle_non_contiguous(tensor: torch.Tensor, name: str) -> List[TensorDescriptor]:
    if tensor.is_contiguous():
        return [TensorDescriptor(name=name, ...)]
    else:
        # Register storage, include view metadata
        storage = tensor.untyped_storage()
        return [
            TensorDescriptor(
                name=f"{name}__storage",
                data_ptr=storage.data_ptr(),
                size_bytes=storage.nbytes(),
                # Include view reconstruction info in metadata
            )
        ]
```

### 3. Tensor Parallelism Rank Matching

**Challenge**: Each TP rank has different weight slices; must transfer to matching rank.

**Mitigation**:
- Include `worker_rank` in source metadata
- Filter sources by matching rank during discovery
- Validate TP configuration matches before transfer

```python
# Target discovery
sources = mx_client.list_sources(identity, status=READY)
my_rank = torch.distributed.get_rank()
candidates = [s for s in sources if s.worker_rank == my_rank]
```

### 4. Pipeline Parallelism Layers

**Challenge**: Different PP ranks have different layer subsets.

**Mitigation**:
- Include `pp_rank` in SourceIdentity
- Each PP rank only transfers its layer subset
- Validate layer ranges match

### 5. MoE Expert Distribution

**Challenge**: Expert parallelism distributes experts differently; load balancer state varies.

**Mitigation**:
- Include `ep_rank` in SourceIdentity
- Re-run `load_balancer.finalize()` after P2P transfer
- Transfer load balancer state separately if needed

### 6. CUDA VMM Integration

**Challenge**: GMS uses CUDA VMM; TRT-LLM uses PyTorch allocator.

**Mitigation**:
- Implement `CUDAPluggableAllocator` that routes to GMS
- Use `torch.cuda.memory.CUDAPluggableAllocator` API
- Handle allocation/deallocation lifecycle correctly

```python
class GMSAllocator:
    def __init__(self, gms_client):
        self.gms_client = gms_client
        self._allocations = {}

    def malloc(self, size: int, device: int, stream) -> int:
        ptr = self.gms_client.create_mapping(size=size)
        self._allocations[ptr] = size
        return ptr

    def free(self, ptr: int, size: int, device: int, stream):
        self.gms_client.destroy_mapping(ptr)
        del self._allocations[ptr]
```

---

## Performance Expectations

### Target Metrics

| Scenario | Baseline | Target | Improvement |
|----------|----------|--------|-------------|
| Cold-start (DeepSeek-V3, 681GB) | 5-10 min | 15-30s | **10-20x** |
| Replica scale-up (Llama-70B) | 2-3 min | 5-10s | **12-36x** |
| Memory per worker (same GPU) | N × weights | 1 × weights | **N× reduction** |
| Failover time | Cold-start | < 5s | **60-120x** |
| Multi-node scale-out | Linear I/O | P2P tree | **Near-constant** |

### Benchmark Plan

1. **Cold-start latency**: Time from process start to first inference
2. **P2P transfer throughput**: GB/s across network configurations
3. **Memory efficiency**: Peak GPU memory with N workers
4. **Failover latency**: Time from primary failure to shadow serving
5. **Scale-out efficiency**: Time to add N replicas

---

## Complexity Assessment

### Overall Complexity: **Medium-High**

### Phase Breakdown

| Phase | Complexity | Effort Estimate | Risk Level |
|-------|------------|-----------------|------------|
| Phase 1: MX + TRT-LLM | Medium | 4-6 weeks | Medium |
| Phase 2: GMS + TRT-LLM | Medium | 3-5 weeks | Low-Medium |
| Phase 3: Combined | Low-Medium | 2-3 weeks | Low |
| Testing & Hardening | Medium | 3-4 weeks | Medium |
| **Total** | **Medium-High** | **12-18 weeks** | **Medium** |

### Complexity Factors

**High Complexity Areas**:
1. **CUDA VMM integration**: Low-level memory management with FD passing
2. **Distributed coordination**: Multi-rank, multi-node synchronization
3. **Quantization compatibility**: Ensuring identical layouts across transfers
4. **Non-contiguous tensor handling**: View reconstruction on target

**Medium Complexity Areas**:
1. **Weight loader abstraction**: Clean API design and registration
2. **NIXL/TransferEngine wrapper**: Adapting existing patterns from vLLM
3. **Configuration schema**: Extending TorchLlmArgs with new options
4. **Testing infrastructure**: Multi-node test environments

**Lower Complexity Areas**:
1. **MX gRPC client integration**: Well-defined proto interface
2. **GMS client integration**: Existing Python client available
3. **Phase 3 combination**: Building on Phase 1 & 2 foundations

### Dependencies

| Dependency | Owner | Risk |
|------------|-------|------|
| MX Python client | MX team | Low (stable) |
| GMS Python client | Dynamo team | Low (stable) |
| NIXL bindings | NVIDIA | Low (stable) |
| TRT-LLM PyTorch backend | TRT-LLM team | None (internal) |
| CUDA VMM APIs | NVIDIA | Low (stable) |

---

## Alternative Approaches

### Alternative 1: External Wrapper Only

**Approach**: Keep all integration logic external (like current prototypes)

**Pros**:
- No TRT-LLM core changes required
- Faster initial implementation

**Cons**:
- Fragile; breaks with TRT-LLM updates
- Requires patching internal APIs
- Poor user experience (complex setup)
- Limited optimization opportunities

**Assessment**: Not recommended for production

### Alternative 2: GMS-Only (No MX)

**Approach**: Only integrate GMS; rely on shared storage for cross-node

**Pros**:
- Simpler architecture
- No cross-node coordination needed

**Cons**:
- Doesn't solve cold-start for new nodes
- Requires high-performance shared storage (expensive)
- No P2P scaling benefits

**Assessment**: Insufficient for target use cases

### Alternative 3: MX-Only (No GMS)

**Approach**: Only integrate MX; no within-node sharing

**Pros**:
- Solves cross-node cold-start
- Simpler than combined approach

**Cons**:
- Memory waste with multiple workers per GPU
- No crash resilience
- No shadow failover support

**Assessment**: Partial solution; combine with GMS for full benefits

### Alternative 4: Custom Implementation (No MX/GMS)

**Approach**: Build equivalent functionality from scratch in TRT-LLM

**Pros**:
- Full control over implementation
- No external dependencies

**Cons**:
- Massive engineering effort (6-12 months)
- Duplicates existing, proven systems
- Diverges from Dynamo ecosystem
- Maintenance burden

**Assessment**: Not recommended; leverage existing systems

### Recommended: Phased MX + GMS Integration

**Rationale**:
- Leverages battle-tested components (MX, GMS)
- Aligns with Dynamo ecosystem roadmap
- Enables incremental value delivery
- Clean separation of concerns
- Supported by MX/GMS teams

---

## Risks and Concerns

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| CUDA VMM complexity | High | Medium | Start with simple cases; incremental rollout |
| Quantization incompatibility | High | Low | Strict identity matching; validation before transfer |
| Performance regression | Medium | Low | Benchmark gates; fallback to disk loading |
| Multi-rank race conditions | Medium | Medium | Careful synchronization; extensive testing |

### Strategic Concerns

#### 1. Dependency on External Projects

**Concern**: MX and GMS are developed by different teams; API changes could break integration.

**Mitigation**:
- Define clear interface contracts
- Version compatibility matrix
- Automated integration tests
- Regular sync with MX/GMS teams

#### 2. Long-term Maintenance

**Concern**: TRT-LLM team must maintain integration code indefinitely.

**Mitigation**:
- Clean API boundaries minimize coupling
- Shared ownership with Dynamo team for integration layer
- Documentation and knowledge transfer

#### 3. Ecosystem Lock-in

**Concern**: Deep integration with Dynamo ecosystem.

**Mitigation**:
- MX and GMS are optional; default behavior unchanged
- Abstraction layer allows alternative backends
- Open-source implementations available

### Unnecessary Work Concerns

#### 1. Duplicate Effort with Prototypes

**Concern**: Prototypes already exist; are we duplicating work?

**Assessment**: Prototypes use external patching which is fragile. Native integration provides:
- Stability across TRT-LLM versions
- Better performance (no indirection)
- Cleaner user experience
- This is necessary work, not duplication

#### 2. Over-Engineering

**Concern**: Building too much abstraction upfront.

**Mitigation**:
- Start with minimal APIs
- Expand based on actual needs
- Avoid speculative features

---

## Timeline and Milestones

### Proposed Timeline

```
Month 1-2: Phase 1 (MX + TRT-LLM)
├── Week 1-2: API design and tensor enumeration
├── Week 3-4: NIXL wrapper implementation
├── Week 5-6: MX loader integration
└── Week 7-8: Testing and documentation

Month 2-3: Phase 2 (GMS + TRT-LLM)
├── Week 1-2: Pluggable allocator hook
├── Week 3-4: GMS loader implementation
├── Week 5-6: Sleep/wake integration
└── Week 7-8: Testing and documentation

Month 4: Phase 3 (Combined)
├── Week 1-2: Combined loader implementation
├── Week 3-4: End-to-end testing
└── Week 5-6: Performance optimization

Month 4-5: Hardening
├── Multi-node testing
├── Edge case handling
├── Documentation
└── Release preparation
```

### Milestones

| Milestone | Target Date | Success Criteria |
|-----------|-------------|------------------|
| M1: API Design Complete | Week 2 | API spec reviewed and approved |
| M2: MX Integration Alpha | Week 8 | P2P transfer working on 2 nodes |
| M3: GMS Integration Alpha | Week 14 | Zero-copy sharing on single node |
| M4: Combined Beta | Week 18 | Full MX+GMS flow working |
| M5: Production Ready | Week 22 | All tests passing; docs complete |

---

## Open Questions

1. **Transfer backend**: Should TRT-LLM use NIXL (like vLLM) or Mooncake TransferEngine?
   - NIXL: More mature, vLLM proven
   - TransferEngine: MX proto suggests this for TRT-LLM

2. **GMS socket location**: Standard path or configurable?
   - Suggest: `/tmp/gms-{device_id}.sock` with override via env var

3. **Multi-model support**: How to handle multiple models in same GMS?
   - Suggest: Tag-based separation (`{model_name}:{rank}`)

4. **Partial transfer**: Support transferring subset of layers (for PP)?
   - Suggest: Yes, include in Phase 1

5. **Fallback behavior**: What happens if MX server unreachable?
   - Suggest: Graceful fallback to disk with warning

6. **Metrics exposure**: How to expose P2P metrics to users?
   - Suggest: Prometheus metrics + logging

---

## Startup Performance Profiling

To effectively optimize TRT-LLM startup time, we need a comprehensive profiling framework that breaks down the end-to-end launch process. This section defines the benchmark methodology, identifies critical bottlenecks, and proposes instrumentation.

### E2E Startup Timeline

The TRT-LLM startup process consists of multiple sequential and parallel phases:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           TRT-LLM E2E STARTUP TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  Process Start                                                                           │
│       │                                                                                  │
│       ▼                                                                                  │
│  ┌─────────────────┐                                                                     │
│  │ 1. Python Init  │ ~1-3s                                                               │
│  │    & Imports    │ (import torch, tensorrt_llm, etc.)                                  │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐                                                                     │
│  │ 2. Config Load  │ ~0.5-2s                                                             │
│  │    & Validate   │ (HF config, tokenizer, args validation)                             │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐  ◄─── CRITICAL BOTTLENECK (Network-bound)                           │
│  │ 3. Model        │ ~30s - 30min (varies by model size & network)                       │
│  │    Download     │ (HuggingFace Hub, NGC, S3, etc.)                                    │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐  ◄─── CRITICAL BOTTLENECK (I/O-bound)                               │
│  │ 4. Weight       │ ~10s - 5min (varies by model size & storage)                        │
│  │    Loading      │ (disk → CPU → GPU, safetensors/pickle)                              │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐                                                                     │
│  │ 5. Weight       │ ~5-30s                                                              │
│  │    Processing   │ (dtype conversion, quantization, TP sharding)                       │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐  ◄─── SIGNIFICANT DELAY (Compute-bound)                             │
│  │ 6. Model        │ ~10s - 2min (for torch.compile, CUDA graphs)                        │
│  │    Compilation  │ (torch.compile, Triton kernels, CUDA graphs capture)               │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐                                                                     │
│  │ 7. KV Cache     │ ~1-10s                                                              │
│  │    Allocation   │ (paged cache setup, memory pool init)                               │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐                                                                     │
│  │ 8. Executor     │ ~1-5s                                                               │
│  │    Init         │ (scheduler, sampler, resource manager)                              │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  ┌─────────────────┐                                                                     │
│  │ 9. Server       │ ~0.5-2s                                                             │
│  │    Startup      │ (gRPC/HTTP server, health checks)                                   │
│  └────────┬────────┘                                                                     │
│           ▼                                                                              │
│  Ready to Serve                                                                          │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Critical Bottlenecks Analysis

| Phase | Typical Duration | Bottleneck Type | Impact | MX/GMS Solution |
|-------|------------------|-----------------|--------|-----------------|
| **Model Download** | 30s - 30min | Network I/O | **CRITICAL** | MX: P2P from existing replica |
| **Weight Loading** | 10s - 5min | Disk I/O | **CRITICAL** | GMS: Zero-copy import; MX: RDMA |
| **Model Compilation** | 10s - 2min | Compute | **HIGH** | Future: Compile cache sharing |
| **Weight Processing** | 5-30s | CPU/GPU compute | **MEDIUM** | Both sides run identical processing |
| **KV Cache Allocation** | 1-10s | GPU memory | **LOW** | Separate concern (KVBM) |
| **Python Imports** | 1-3s | CPU/Disk | **LOW** | Pre-warming, lazy imports |

### Detailed Phase Breakdown

#### Phase 1: Python Initialization (~1-3s)

**Components**:
- Python interpreter startup
- `import torch` (loads CUDA runtime)
- `import tensorrt_llm` (loads C++ bindings)
- Third-party imports (transformers, safetensors, etc.)

**Instrumentation Points**:
```python
# tensorrt_llm/__init__.py
import time
_import_start = time.perf_counter()

import torch  # Heavy import
_torch_import_time = time.perf_counter() - _import_start

# ... other imports ...

_total_import_time = time.perf_counter() - _import_start
logger.debug(f"Import times: torch={_torch_import_time:.2f}s, total={_total_import_time:.2f}s")
```

#### Phase 2: Configuration Loading (~0.5-2s)

**Components**:
- HuggingFace config fetch (may hit network)
- Tokenizer loading
- Argument parsing and validation
- Model-specific default resolution

**Instrumentation Points**:
```python
# tensorrt_llm/llmapi/llm.py
with StartupTimer("config_load") as t:
    config = AutoConfig.from_pretrained(model_path)

with StartupTimer("tokenizer_load") as t:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

with StartupTimer("args_validation") as t:
    llm_args = TorchLlmArgs.model_validate(user_args)
```

#### Phase 3: Model Download (~30s - 30min) ⚠️ CRITICAL

**Components**:
- HuggingFace Hub authentication
- Manifest fetch (model index)
- Weight file downloads (parallel)
- Checksum verification
- Cache management

**Bottleneck Analysis**:
- **DeepSeek-V3 (681GB)**: 30+ minutes on 1Gbps, ~5 min on 10Gbps
- **Llama-70B (140GB)**: 10+ minutes on 1Gbps, ~2 min on 10Gbps
- **Network variability**: CDN congestion, regional latency

**MX Solution Impact**:
```
Before MX: Each replica downloads independently
           N replicas × D download_time = N×D total wait

After MX:  First replica downloads, others receive via P2P
           1×D + P2P_transfer_time ≈ D + 15-30s

Improvement: (N-1) × D saved for cluster
```

**Instrumentation Points**:
```python
# tensorrt_llm/llmapi/llm_utils.py
with StartupTimer("model_download") as t:
    with StartupTimer("hf_auth"):
        token = get_hf_token()
    with StartupTimer("manifest_fetch"):
        files = list_repo_files(model_name)
    with StartupTimer("weight_download"):
        for file in weight_files:
            with StartupTimer(f"download_{file}"):
                hf_hub_download(model_name, file)
```

#### Phase 4: Weight Loading (~10s - 5min) ⚠️ CRITICAL

**Components**:
- File I/O (safetensors/pickle deserialization)
- CPU memory allocation
- CPU → GPU transfer
- Memory mapping (if applicable)

**Bottleneck Analysis**:
- **Disk speed**: NVMe (~3GB/s) vs HDD (~150MB/s) = 20x difference
- **PCIe bandwidth**: ~25GB/s theoretical, ~15GB/s practical
- **CPU memory**: May spill to swap if insufficient RAM

**GMS Solution Impact**:
```
Before GMS: Each worker loads from disk
            Worker 1: Disk → CPU → GPU (full I/O)
            Worker 2: Disk → CPU → GPU (full I/O, duplicate)
            Worker N: Disk → CPU → GPU (full I/O, duplicate)

After GMS:  First worker loads, others import from GMS
            Worker 1: Disk → CPU → GPU → GMS commit
            Worker 2: GMS import (zero-copy, ~100ms)
            Worker N: GMS import (zero-copy, ~100ms)

Improvement: (N-1) workers save full loading time
```

**Instrumentation Points**:
```python
# tensorrt_llm/_torch/pyexecutor/model_loader.py
with StartupTimer("weight_loading") as t:
    with StartupTimer("checkpoint_discovery"):
        files = find_checkpoint_files(checkpoint_dir)

    for file in files:
        with StartupTimer(f"load_{file}"):
            with StartupTimer(f"read_{file}"):
                data = read_file(file)  # Disk I/O
            with StartupTimer(f"deserialize_{file}"):
                tensors = deserialize(data)  # CPU compute
            with StartupTimer(f"to_gpu_{file}"):
                gpu_tensors = {k: v.cuda() for k, v in tensors.items()}
```

#### Phase 5: Weight Processing (~5-30s)

**Components**:
- Data type conversion (FP32 → FP16/BF16/FP8)
- Quantization application (INT4 AWQ, INT8 SQ)
- Tensor parallel sharding
- Weight fusion/transformation
- `post_load_weights()` hooks

**Instrumentation Points**:
```python
# tensorrt_llm/_torch/pyexecutor/model_loader.py
with StartupTimer("weight_processing") as t:
    with StartupTimer("dtype_conversion"):
        model = model.to(dtype)

    with StartupTimer("tp_sharding"):
        shard_weights(model, mapping)

    with StartupTimer("post_load_weights"):
        model.post_load_weights()

    with StartupTimer("moe_finalize"):
        if moe_load_balancer:
            moe_load_balancer.finalize()
```

#### Phase 6: Model Compilation (~10s - 2min) ⚠️ HIGH IMPACT

**Components**:
- `torch.compile()` tracing and optimization
- Triton kernel compilation
- CUDA graph capture
- DeepGEMM kernel compilation (for MoE)

**Bottleneck Analysis**:
- **First compilation**: Full trace + codegen (expensive)
- **Cache hit**: Load compiled artifacts (fast)
- **Graph capture**: Multiple warmup iterations required

**Future Optimization** (not in current MX/GMS scope):
- Compile cache sharing via MX
- Pre-compiled model artifacts

**Instrumentation Points**:
```python
# tensorrt_llm/_torch/pyexecutor/model_engine.py
with StartupTimer("compilation") as t:
    with StartupTimer("torch_compile"):
        if use_torch_compile:
            model = torch.compile(model, **compile_config)

    with StartupTimer("cuda_graphs_capture"):
        if use_cuda_graphs:
            for batch_size in batch_sizes:
                with StartupTimer(f"graph_capture_bs{batch_size}"):
                    capture_cuda_graph(model, batch_size)

    with StartupTimer("triton_warmup"):
        warmup_triton_kernels(model)
```

#### Phase 7-9: Executor/Server Initialization (~2-17s)

**Components**:
- KV cache pool allocation
- Scheduler initialization
- Sampler setup
- gRPC/HTTP server binding
- Health check registration

**Instrumentation Points**:
```python
# tensorrt_llm/_torch/pyexecutor/py_executor_creator.py
with StartupTimer("executor_init") as t:
    with StartupTimer("kv_cache_allocation"):
        kv_cache_manager = create_kv_cache_manager(config)

    with StartupTimer("scheduler_init"):
        scheduler = create_scheduler(config)

    with StartupTimer("sampler_init"):
        sampler = create_sampler(config)

# tensorrt_llm/serve/openai_server.py
with StartupTimer("server_startup") as t:
    with StartupTimer("grpc_bind"):
        server.add_insecure_port(address)

    with StartupTimer("health_check"):
        await server.wait_for_ready()
```

### Proposed Profiling Framework

#### StartupTimer Context Manager

```python
# tensorrt_llm/_torch/utils/startup_profiler.py

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os

@dataclass
class TimingRecord:
    name: str
    start_time: float
    end_time: float
    duration: float
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class StartupProfiler:
    """Hierarchical profiler for TRT-LLM startup phases."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.records: Dict[str, TimingRecord] = {}
        self.stack: List[str] = []
        self.enabled = os.environ.get("TRTLLM_PROFILE_STARTUP", "0") == "1"
        self._start_time = time.perf_counter()

    @classmethod
    def get_instance(cls) -> "StartupProfiler":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @contextmanager
    def timer(self, name: str, **metadata):
        if not self.enabled:
            yield
            return

        full_name = f"{self.stack[-1]}.{name}" if self.stack else name
        parent = self.stack[-1] if self.stack else None

        self.stack.append(full_name)
        start = time.perf_counter()

        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start

            record = TimingRecord(
                name=full_name,
                start_time=start - self._start_time,
                end_time=end - self._start_time,
                duration=duration,
                parent=parent,
                metadata=metadata,
            )
            self.records[full_name] = record

            if parent and parent in self.records:
                self.records[parent].children.append(full_name)

            self.stack.pop()

    def summary(self) -> str:
        """Generate human-readable timing summary."""
        if not self.records:
            return "No timing records (enable with TRTLLM_PROFILE_STARTUP=1)"

        lines = ["=" * 80, "TRT-LLM STARTUP TIMING BREAKDOWN", "=" * 80]

        # Sort by start time
        sorted_records = sorted(self.records.values(), key=lambda r: r.start_time)

        for record in sorted_records:
            depth = record.name.count(".")
            indent = "  " * depth
            pct = (record.duration / sorted_records[-1].end_time) * 100

            # Highlight critical phases
            marker = ""
            if record.duration > 10:
                marker = " ⚠️ CRITICAL"
            elif record.duration > 5:
                marker = " ⚡ SLOW"

            lines.append(f"{indent}{record.name}: {record.duration:.2f}s ({pct:.1f}%){marker}")

        lines.append("=" * 80)
        lines.append(f"Total startup time: {sorted_records[-1].end_time:.2f}s")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export timing data as JSON for analysis."""
        return json.dumps({
            name: {
                "duration": r.duration,
                "start": r.start_time,
                "end": r.end_time,
                "parent": r.parent,
                "children": r.children,
                "metadata": r.metadata,
            }
            for name, r in self.records.items()
        }, indent=2)

    def to_chrome_trace(self) -> str:
        """Export as Chrome trace format for visualization."""
        events = []
        for name, r in self.records.items():
            events.append({
                "name": name.split(".")[-1],
                "cat": "startup",
                "ph": "X",  # Complete event
                "ts": r.start_time * 1_000_000,  # Microseconds
                "dur": r.duration * 1_000_000,
                "pid": 1,
                "tid": 1,
                "args": r.metadata,
            })
        return json.dumps({"traceEvents": events})


# Convenience function
def StartupTimer(name: str, **metadata):
    return StartupProfiler.get_instance().timer(name, **metadata)
```

#### Integration with TRT-LLM

```python
# tensorrt_llm/llmapi/llm.py

from tensorrt_llm._torch.utils.startup_profiler import StartupTimer, StartupProfiler

class LLM:
    def __init__(self, model: str, **kwargs):
        profiler = StartupProfiler.get_instance()

        with StartupTimer("llm_init"):
            with StartupTimer("args_parse"):
                self._args = self._parse_args(model, **kwargs)

            with StartupTimer("model_resolve"):
                self._model_path = self._resolve_model(model)

            with StartupTimer("executor_create"):
                self._executor = self._create_executor()

        # Print summary if profiling enabled
        if profiler.enabled:
            print(profiler.summary())

            # Save detailed trace
            trace_path = f"/tmp/trtllm_startup_trace_{os.getpid()}.json"
            with open(trace_path, "w") as f:
                f.write(profiler.to_chrome_trace())
            print(f"Chrome trace saved to: {trace_path}")
```

#### CLI Integration

```bash
# Enable startup profiling
TRTLLM_PROFILE_STARTUP=1 trtllm-serve meta-llama/Llama-3.1-70B --port 8000

# Output:
# ================================================================================
# TRT-LLM STARTUP TIMING BREAKDOWN
# ================================================================================
# llm_init: 245.32s (100.0%)
#   args_parse: 0.12s (0.0%)
#   model_resolve: 0.05s (0.0%)
#   executor_create: 245.15s (99.9%)
#     config_load: 1.23s (0.5%)
#     model_download: 180.45s (73.6%) ⚠️ CRITICAL
#       hf_auth: 0.34s (0.1%)
#       manifest_fetch: 0.89s (0.4%)
#       weight_download: 179.22s (73.1%) ⚠️ CRITICAL
#     weight_loading: 35.67s (14.5%) ⚠️ CRITICAL
#       checkpoint_discovery: 0.02s (0.0%)
#       load_model.safetensors: 35.65s (14.5%) ⚠️ CRITICAL
#         read: 28.34s (11.6%) ⚠️ CRITICAL
#         deserialize: 2.12s (0.9%)
#         to_gpu: 5.19s (2.1%)
#     weight_processing: 8.45s (3.4%)
#       dtype_conversion: 2.34s (1.0%)
#       tp_sharding: 4.56s (1.9%)
#       post_load_weights: 1.55s (0.6%)
#     compilation: 15.23s (6.2%) ⚡ SLOW
#       torch_compile: 8.45s (3.4%)
#       cuda_graphs_capture: 6.78s (2.8%)
#     kv_cache_allocation: 2.34s (1.0%)
#     server_startup: 1.78s (0.7%)
# ================================================================================
# Total startup time: 245.32s
# ================================================================================
# Chrome trace saved to: /tmp/trtllm_startup_trace_12345.json
```

### Benchmark Suite

```python
# tests/benchmarks/test_startup_latency.py

import pytest
import time
import os
from tensorrt_llm import LLM
from tensorrt_llm._torch.utils.startup_profiler import StartupProfiler

class TestStartupLatency:
    """Benchmark suite for startup latency measurement."""

    MODELS = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tiny"),
        ("meta-llama/Llama-3.1-8B", "small"),
        ("meta-llama/Llama-3.1-70B", "large"),
    ]

    SCENARIOS = [
        ("cold_start", {}),
        ("warm_cache", {"skip_download": True}),  # Model already downloaded
        ("gms_ro", {"load_format": "gms", "gms_mode": "ro"}),
        ("mx_p2p", {"load_format": "mx"}),
    ]

    @pytest.fixture(autouse=True)
    def setup_profiler(self):
        os.environ["TRTLLM_PROFILE_STARTUP"] = "1"
        yield
        os.environ.pop("TRTLLM_PROFILE_STARTUP", None)

    @pytest.mark.parametrize("model,size", MODELS)
    @pytest.mark.parametrize("scenario,kwargs", SCENARIOS)
    def test_startup_latency(self, model, size, scenario, kwargs, benchmark):
        """Measure startup latency for various configurations."""

        def create_llm():
            profiler = StartupProfiler()
            start = time.perf_counter()

            llm = LLM(model=model, **kwargs)

            end = time.perf_counter()
            return {
                "total_time": end - start,
                "profiler": profiler,
            }

        result = benchmark(create_llm)

        # Assert phase budgets
        records = result["profiler"].records

        if scenario == "gms_ro":
            # GMS RO mode should be fast
            assert records.get("weight_loading", {}).get("duration", 999) < 1.0

        if scenario == "mx_p2p":
            # MX P2P should skip download
            assert records.get("model_download", {}).get("duration", 0) < 5.0

    def test_phase_breakdown_report(self):
        """Generate detailed phase breakdown report."""
        os.environ["TRTLLM_PROFILE_STARTUP"] = "1"

        llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        profiler = StartupProfiler.get_instance()

        # Verify all expected phases are recorded
        expected_phases = [
            "config_load",
            "weight_loading",
            "weight_processing",
            "executor_init",
        ]

        for phase in expected_phases:
            matching = [k for k in profiler.records.keys() if phase in k]
            assert len(matching) > 0, f"Phase {phase} not recorded"

        # Export report
        print(profiler.summary())
```

### Performance Regression Detection

```python
# tests/benchmarks/startup_baselines.yaml

# Baseline startup times for regression detection
# Update when performance improves

baselines:
  TinyLlama-1.1B:
    cold_start: 15.0  # seconds
    warm_cache: 8.0
    gms_ro: 2.0
    mx_p2p: 5.0

  Llama-3.1-8B:
    cold_start: 45.0
    warm_cache: 25.0
    gms_ro: 3.0
    mx_p2p: 10.0

  Llama-3.1-70B:
    cold_start: 300.0
    warm_cache: 120.0
    gms_ro: 5.0
    mx_p2p: 30.0

# Phase budgets (percentage of total)
phase_budgets:
  model_download: 50%  # Should be 0% with MX
  weight_loading: 30%  # Should be <1% with GMS
  weight_processing: 10%
  compilation: 10%
  other: 5%
```

### Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **Time to First Token (TTFT)** | Process start → first generated token | < 30s (with MX+GMS) |
| **Time to Ready (TTR)** | Process start → server accepting requests | < 20s (with MX+GMS) |
| **Download Time** | Model download duration | 0s (with MX P2P) |
| **Weight Load Time** | Disk → GPU time | < 1s (with GMS RO) |
| **Compilation Time** | torch.compile + CUDA graphs | Track for future optimization |
| **Memory High Water Mark** | Peak GPU memory during startup | Track for OOM prevention |
| **P2P Transfer Rate** | GB/s during MX transfer | > 20 GB/s (NVLink) |
| **GMS Import Rate** | GB/s during GMS import | > 100 GB/s (local) |

### Visualization

The Chrome trace format export enables visualization in:
- **Chrome DevTools**: `chrome://tracing`
- **Perfetto**: https://ui.perfetto.dev
- **Custom dashboards**: Parse JSON for Grafana/Prometheus

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Chrome Trace Visualization (Perfetto)                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Time (s) 0       50      100     150     200     250                            │
│           │───────│───────│───────│───────│───────│                             │
│                                                                                  │
│  config   ██                                                                     │
│  download ████████████████████████████████████████████████████████               │
│  loading                                                          ████████       │
│  process                                                                  ███    │
│  compile                                                                     ██  │
│  server                                                                       █  │
│                                                                                  │
│  Legend: ██ = Active phase                                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. **ModelExpress Repository**: https://github.com/ai-dynamo/modelexpress
2. **Dynamo Repository**: https://github.com/ai-dynamo/dynamo
3. **GMS + TRT-LLM PR**: https://github.com/ai-dynamo/dynamo/pull/7053
4. **TensorRT-LLM Documentation**: https://nvidia.github.io/TensorRT-LLM/
5. **NIXL Documentation**: (internal NVIDIA)
6. **CUDA VMM Guide**: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **MX** | ModelExpress - GPU-to-GPU weight streaming service |
| **GMS** | GPU Memory Service - out-of-process GPU memory manager |
| **NIXL** | NVIDIA Interconnect Library - high-speed GPU-to-GPU transfer |
| **RDMA** | Remote Direct Memory Access |
| **VMM** | Virtual Memory Management |
| **TP** | Tensor Parallelism |
| **PP** | Pipeline Parallelism |
| **EP** | Expert Parallelism (for MoE models) |
| **KVBM** | KV Block Manager - KV cache management in Dynamo |

---

## Appendix B: Code Examples

### Example: MX Weight Loader Usage

```python
from tensorrt_llm import LLM

# With MX P2P loading
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    load_format="mx",
    mx_server_url="http://mx-server:8001",
    tensor_parallel_size=8,
)

# First replica: loads from HuggingFace, publishes to MX
# Subsequent replicas: receives via P2P from existing replica
```

### Example: GMS Weight Loader Usage

```python
from tensorrt_llm import LLM

# With GMS shared memory
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    load_format="gms",
    gms_socket_path="/tmp/gms-0.sock",
    gms_mode="auto",  # auto-detect RW vs RO
)

# First worker: loads from disk, commits to GMS (RW mode)
# Subsequent workers: imports from GMS (RO mode)
```

### Example: Combined MX + GMS Usage

```python
from tensorrt_llm import LLM

# With combined MX + GMS
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    load_format="mx-gms",
    mx_server_url="http://mx-server:8001",
    gms_socket_path="/tmp/gms-0.sock",
)

# Loading priority:
# 1. Local GMS (fastest)
# 2. Remote MX P2P (fast)
# 3. Disk/HuggingFace (slowest, seeds the cluster)
```
