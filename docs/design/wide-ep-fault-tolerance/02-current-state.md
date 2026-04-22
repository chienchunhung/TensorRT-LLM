# 2. Current State Analysis

[< Back to Overview](README.md)

## WideEP Architecture in TRT-LLM

### Communication Backends

WideEP uses a pluggable communication architecture with five AlltoAll strategies, auto-selected based on hardware capabilities:

```mermaid
graph TD
    subgraph "Communication Factory"
        CF["CommunicationFactory.create_strategy()"]
    end

    CF -->|"Priority 1<br/>MNNVL + throughput"| NV1["NVLinkOneSided<br/>nvlink_one_sided.py"]
    CF -->|"Priority 2<br/>MNNVL + latency"| NV2["NVLinkTwoSided<br/>nvlink_two_sided.py"]
    CF -->|"Priority 3<br/>NVLink + RDMA"| DEP["DeepEP<br/>deep_ep.py"]
    CF -->|"Priority 4<br/>RDMA low-latency"| DEPL["DeepEPLowLatency<br/>deep_ep_low_latency.py"]
    CF -->|"Priority 5<br/>Always available"| AGRS["AllGatherReduceScatter<br/>allgather_reducescatter.py"]

    NV1 --- NV1D["Symmetric memory P2P writes<br/>GB200 NVL72 primary path<br/>kMaxRanks = 64"]
    NV2 --- NV2D["Symmetric memory FIFO queues<br/>Lower latency variant<br/>Requires MNNVL"]
    DEP --- DEPD["NVSHMEM transport<br/>Intra: {2,4,8} ranks<br/>Inter: {16,32,...,128} ranks"]
    DEPL --- DEPLD["IBGDA-based RDMA<br/>Small batch optimized<br/>Limited hidden_size support"]
    AGRS --- AGRSD["Standard NCCL collectives<br/>Fallback, always works<br/>Least performant"]

    style NV1 fill:#4CAF50,color:#fff
    style NV2 fill:#4CAF50,color:#fff
    style DEP fill:#2196F3,color:#fff
    style DEPL fill:#2196F3,color:#fff
    style AGRS fill:#FF9800,color:#fff
```

**Key files:**
- Communication factory: `tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py`
- NVLink one-sided: `tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_one_sided.py`
- NVLink two-sided: `tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_two_sided.py`
- DeepEP: `tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep.py`
- C++ AlltoAll kernels: `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h`

### EPLB Architecture

EPLB decouples logical experts from physical GPU slots, enabling hot expert replication and live weight migration:

```mermaid
graph LR
    subgraph "Logical Experts (256)"
        E0["Expert 0"]
        E1["Expert 1"]
        E255["Expert 255"]
    end

    subgraph "Physical Slots (288 = 36 EP × 8 slots/rank)"
        subgraph "Rank 0"
            S0["Slot 0: Expert 0"]
            S1["Slot 1: Expert 5"]
            S7["Slot 7: Expert 42"]
        end
        subgraph "Rank 1"
            S8["Slot 8: Expert 1"]
            S9["Slot 9: Expert 5 ★"]
            S15["Slot 15: Expert 100"]
        end
        subgraph "Rank 35"
            S280["Slot 280: Expert 200"]
            S287["Slot 287: Expert 255"]
        end
    end

    E0 --> S0
    E1 --> S8
    E5_note["Expert 5 (hot) ★<br/>Replicated to 2 slots"] --> S1
    E5_note --> S9

    style E5_note fill:#FF9800,color:#fff
    style S1 fill:#FF9800,color:#fff
    style S9 fill:#FF9800,color:#fff
```

**Key components:**

| Component | File | Role |
|:----------|:-----|:-----|
| `MoeLoadBalancer` | `_torch/modules/fused_moe/moe_load_balancer.py:842` | Global load balancer, wraps C++ `_tbr.MoeLoadBalancer` |
| `SingleLayerMoeLoadBalancer` | `moe_load_balancer.py:374` | Per-layer routing and weight management |
| `HostMoeTensorSharer` | `moe_load_balancer.py:127` | POSIX shared memory for all expert weights on host |
| `MoeLoadBalancerConfig` | `llm_args.py:432` | Configuration: `num_slots`, `initial_global_assignments`, `layer_updates_per_iter` |
| `doReplication()` | C++ `moeLoadBalancer.cpp:57` | Greedy priority-queue algorithm for expert replication |
| `doPlacement()` | C++ `moeLoadBalancer.cpp:124` | Assigns replicated experts to physical slots across ranks |

**Online EPLB weight migration flow (per iteration):**

```mermaid
sequenceDiagram
    participant GPU as GPU (Forward Pass)
    participant LB as Load Balancer (CPU)
    participant Host as Host Shared Memory

    GPU->>LB: Signal: forward complete for layer L
    LB->>LB: Read expert load statistics from GPU
    LB->>LB: doReplication() — decide which experts to replicate
    LB->>LB: doPlacement() — assign experts to slots
    LB->>Host: Read new expert weights from shared memory
    LB->>GPU: cudaMemcpy2D: copy weights to new GPU slots
    LB->>GPU: Update MoePlacementInfo (routing table)
    Note over GPU: Next forward uses new routing
```

**Critical property for fault tolerance:** `HostMoeTensorSharer` loads ALL expert weights into POSIX shared memory at startup. Every rank on the same node can access any expert's weights. This means when a rank dies, its experts' weights are already available on host — surviving ranks can load them in ~0.1-0.3ms per expert via gdrcopy.

### Failure Modes by Backend

| Backend | Failure Behavior | Timeout? | Recovery Path |
|:--------|:----------------|:---------|:-------------|
| NVLinkOneSided | Combine kernel spins on `completion_flags[dead_rank]` | **None** — infinite spin | Requires CUDA kernel modification to add rank masking to symmetric memory completion flag protocol |
| NVLinkTwoSided | FIFO queue polling hangs waiting for dead rank | **None** — infinite spin | Same — these are custom CUDA kernels using symmetric memory P2P writes; modifying their synchronization behavior requires understanding multi-GPU memory ordering and completion flag protocols |
| DeepEP | NVSHMEM operations hang indefinitely | **None** | `mask_buffer_ptr` planned but not public |
| DeepEPLowLatency | Same as DeepEP | **None** | Same; also only supports specific rank counts |
| AllGatherReduceScatter | NCCL timeout (~30min default) | **30 min** (unusable) | Requires process group reconstruction |

### Current Fault Tolerance Infrastructure

**What exists (from [PR #12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718), currently in review).** The table below summarizes the primitives; the per-EP-rank extensions built on top of them are specified in [§07](07-failure-detection.md). Note that PR #12718's commits are not yet on the `docs-and-plans` branch HEAD — see [§07 status callout](07-failure-detection.md#overview) for the sequencing implication.

| Mechanism | What It Does | Limitation for WideEP |
|:----------|:-------------|:---------------------|
| Three-tier error classification | `immediate_fatal` / `severe` / `transient` | No EP-specific error patterns |
| Token-bucket `ErrorBudget` | Rate-limits error impact; prevents single transient from killing server | All-or-nothing: entire executor is fatal or healthy |
| `charge_budget=False` for request-scoped errors | KV transfer timeouts don't poison server health | Could extend to EP routing failures |
| `_check_mpi_futures()` | Detects individual MPI worker death | Per-worker, not per-EP-rank granularity |
| `_error_monitor_loop()` | 5s background polling for worker crashes | Detects worker death, but triggers full shutdown |
| Fatal shutdown drain | Drains `active_requests`, `waiting_queue`, `executor_request_queue` | Drains everything, not just requests affected by dead rank |

**What does not exist (each representing a distinct design challenge):**

- **Per-EP-rank health tracking** — requires extending the executor's binary health model (healthy/fatal) to a per-rank vector, with independent error budgets per rank
- **Partial failure concept** — today, fatal = shut everything down; WideEP FT needs "fatal for rank 37, but ranks 0-36 and 38-71 are fine" — a fundamentally new failure semantics
- **AlltoAll timeout or abort mechanism** — the GPU kernels spin forever; adding timeout requires either kernel modification (hard, interacts with memory ordering) or host-side watchdog (new thread monitoring GPU-side completion flags)
- **Expert redistribution on topology change** — EPLB's C++ core assumes immutable topology; enabling dynamic reconfiguration requires pausing concurrent threads, reallocating arrays, and migrating weights across 58 MoE layers without corrupting routing state
- **Process group reconstruction** — NCCL/NVSHMEM/MPI all assume collective participation from all original ranks; rebuilding with a dead rank risks deadlocks at every layer of the communication stack
- **Failure broadcast consensus** — surviving ranks must agree on the dead set without the dead rank's participation, a variant of the failure detection problem in asynchronous distributed systems
