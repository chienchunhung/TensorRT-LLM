# 8. Integration with MX-GMS

[< Back to Overview](README.md)

## The Three Workstreams

This chapter maps how three concurrent workstreams — [PR #12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) (error detection), WideEP FT (this design), and [MX+GMS+TRT-LLM integration](https://docs.google.com/document/d/14SZmmFcoakgIx2OC4dt8pWcHU14PDTN9KlAKqLoZ15s/edit?usp=sharing) (fast recovery) — form a layered reliability stack that is greater than the sum of its parts.

## Layered Architecture

```mermaid
graph TB
    subgraph "Layer 3: Fast Recovery (MX-GMS)"
        MX["MX: P2P Weight Streaming<br/>Cross-node RDMA, ~15-30s"]
        GMS["GMS: Crash-Resilient Memory<br/>Zero-copy import, ~100ms"]
        Shadow["Shadow EP Ranks<br/>Pre-loaded weights, <1s activation"]
    end

    subgraph "Layer 2: Partial Failure Handling (WideEP FT — this design)"
        Mask["Rank Masking<br/>AlltoAll skips dead ranks"]
        EPLB_R["EPLB Reconfigure<br/>Expert redistribution"]
        PG["Process Group Reconstruction<br/>(Phase 2 only)"]
    end

    subgraph "Layer 1: Failure Detection (PR #12718)"
        EC["Error Classification<br/>immediate_fatal / severe / transient"]
        EB["Error Budget<br/>Token-bucket rate limiting"]
        FE["Fatal Error Propagation<br/>_fatal_error + check_health()"]
        EM["Error Monitor Loop<br/>5s background polling"]
    end

    EC --> Mask
    EB --> EPLB_R
    FE --> Shadow
    EM --> Mask

    EPLB_R --> GMS
    PG --> MX
    Shadow --> PG

    style MX fill:#4CAF50,color:#fff
    style GMS fill:#4CAF50,color:#fff
    style Shadow fill:#4CAF50,color:#fff
    style Mask fill:#2196F3,color:#fff
    style EPLB_R fill:#2196F3,color:#fff
    style PG fill:#2196F3,color:#fff
    style EC fill:#FF9800,color:#fff
    style EB fill:#FF9800,color:#fff
    style FE fill:#FF9800,color:#fff
    style EM fill:#FF9800,color:#fff
```

## Dependency and Parallelization Map

The three workstreams have **limited hard dependencies** and can largely be developed in parallel:

```mermaid
gantt
    title Workstream Parallelization
    dateFormat YYYY-MM
    axisFormat %b %Y

    section PR #12718 (Error Detection)
    Error classification + budget     :done, pr1, 2026-03, 2026-04
    Fatal error propagation           :done, pr2, 2026-03, 2026-04
    MPI worker crash detection        :done, pr3, 2026-03, 2026-04
    Review + merge                    :active, pr4, 2026-04, 2026-05

    section WideEP FT (This Design)
    Phase 1a: AlltoAll timeout + rank masking  :ft1, 2026-05, 2026-07
    Phase 1b: EPLB reconfigure()               :ft2, 2026-05, 2026-07
    Phase 1c: Failure broadcast + integration  :ft3, 2026-06, 2026-08
    Phase 1: End-to-end validation             :ft4, 2026-08, 2026-09
    Phase 2: Process group reconstruction      :ft5, 2026-08, 2026-10
    Phase 3: Proactive latency monitoring      :ft6, 2026-10, 2026-12

    section MX-GMS Integration
    MX Phase 1: P2P weight streaming   :mx1, 2026-04, 2026-06
    GMS Phase 2: Zero-copy + shadow    :mx2, 2026-06, 2026-09
    Combined Phase 3: MX+GMS unified   :mx3, 2026-09, 2026-11
    Shadow EP rank for WideEP FT       :mx4, after ft5, 2026-12
```

### Hard Dependencies

| Step | Depends On | Reason |
|:-----|:-----------|:-------|
| WideEP FT Phase 1 | PR #12718 merged | Needs error classification + budget infrastructure |
| WideEP FT Phase 2 | WideEP FT Phase 1 | Can't restore if can't survive |
| Shadow EP ranks | GMS Phase 2 + WideEP FT Phase 2 | Needs both GMS zero-copy AND process group reconstruction |
| GMS Phase 2 | MX Phase 1 | MX-GMS design positions MX first |

### Soft Dependencies (Beneficial but Not Blocking)

| Step | Benefits From | How |
|:-----|:-------------|:----|
| WideEP FT Phase 1 | MX Phase 1 | Not needed, but MX identity matching includes `ep_rank` — validates EP rank concepts |
| WideEP FT Phase 2 | GMS Phase 2 | GMS makes recovery faster (<1s vs minutes), but Phase 2 works without GMS (disk loading) |
| MX-GMS Phase 3 | WideEP FT Phase 1 | MX-GMS can validate failover against WideEP FT's survival capability |

## How PR #12718 Enables WideEP FT

PR #12718 introduces several building blocks that WideEP FT directly extends:

### 1. Error Classification → EP-Specific Patterns

PR #12718's `classify_error()` function categorizes errors by string pattern matching. WideEP FT adds EP-specific patterns:

```python
# PR #12718 existing patterns:
IMMEDIATE_FATAL = ["cudaerrorillegaladdress", "device-side assert", ...]
SEVERE = ["cuda out of memory", "nccl error", ...]
TRANSIENT = [...]  # everything else

# WideEP FT additions:
EP_IMMEDIATE_FATAL = ["nccl communicator abort", "nvshmem peer unreachable", ...]
EP_SEVERE = ["alltoall timeout", "deep_ep buffer barrier hang", ...]
```

**Key design point:** PR #12718's classification operates at the executor level (entire system). WideEP FT adds a **per-rank** dimension — the same error type can be fatal for one rank but not for the system.

### 2. Error Budget → Per-Rank Budgets

PR #12718's `ErrorBudget` (token-bucket with 1.0 capacity, 0.1/s recovery) determines when accumulated errors become fatal. WideEP FT creates **per-rank budgets**:

- Each EP rank has its own `ErrorBudget` instance
- A rank-specific severe error (e.g., AlltoAll timeout from rank 37) charges rank 37's budget, not the system budget
- When rank 37's budget is exhausted, rank 37 is marked failed — but the system continues serving

### 3. `charge_budget=False` → EP Routing Failures

PR #12718 already uses `charge_budget=False` for KV transfer timeouts (request-scoped, not system-scoped). WideEP FT extends this pattern:

- Tokens that **would have been routed** to a just-failed rank are treated as request-scoped failures
- These tokens are re-routed to redundant expert copies on other ranks
- The request continues (with slight latency increase), rather than failing entirely

### 4. `_error_monitor_loop()` → EP Rank Health Loop

PR #12718's background monitor thread (5s polling) checks MPI futures and the error queue. WideEP FT extends this to include EP-specific health checks:

- AlltoAll completion flag monitoring (Layer 1 detection)
- Per-rank latency tracking (Layer 3 detection)
- EP group health status aggregation

### 5. Fatal Shutdown Drain → Partial Drain

PR #12718's fatal shutdown drains all queues (`active_requests`, `waiting_queue`, `executor_request_queue`). WideEP FT modifies this for partial failure:

- **Rank failure:** Only drain the current batch (which was using the failed rank). Don't drain the waiting queue — those requests haven't been affected.
- **System failure:** Full drain (same as PR #12718).

## How MX-GMS Accelerates WideEP FT Phase 2

Without MX-GMS, Phase 2 recovery requires loading expert weights from disk — typically 1-3 minutes for a DeepSeek-V3 expert shard. MX-GMS provides two acceleration paths:

### GMS Zero-Copy Import (Fastest: <1s)

When GMS is available, the failed rank's expert weights may still be in GPU memory (GMS's out-of-process crash-resilient memory). The replacement rank can import them via GMS zero-copy:

```mermaid
sequenceDiagram
    participant Dead as Dead Rank (GPU 37)
    participant GMS as GMS (Out-of-Process)
    participant New as Replacement Rank

    Note over Dead: Process crashes
    Note over GMS: GPU memory persists!<br/>(crash-resilient)
    Dead->>GMS: Socket disconnects → lock auto-released

    New->>GMS: Request RW lock for rank 37's weights
    GMS-->>New: Lock granted (previous lock auto-released)
    New->>GMS: materialize_module_from_gms()
    GMS-->>New: Zero-copy weight import (~100ms)
    Note over New: Expert weights ready!
```

**Key insight:** GMS's crash resilience means the dead process's GPU memory **persists**. The replacement rank doesn't need to reload from disk — it imports the existing GPU memory in ~100ms.

**Limitation:** This only works if the replacement rank is on the **same node** as the failed rank (GMS sharing is intra-node via CUDA VMM FD handles). For cross-node replacement, use MX P2P.

### MX P2P Streaming (Fast: ~15-30s for full model, less for expert shard)

When the replacement rank is on a different node, MX provides cross-node P2P via NIXL/RDMA:

- The replacement rank requests its expert shard from a peer (any surviving rank with the same EP topology)
- MX identity matching includes `ep_size` and `ep_rank`, ensuring the correct shard is transferred
- Only the expert shard is transferred (not the full model), so the transfer is proportionally smaller

For DeepSeek-V3 with EP=72: each rank holds ~9.5 GB of expert weights (681GB / 72). MX P2P at 20+ GB/s transfers this in <0.5s. Total Phase 2 recovery with MX: ~1-2s.

### Shadow EP Ranks (Sub-Second Activation)

The MX-GMS design (Section 6: Executor Integration and Failover) describes shadow workers that pre-load weights via GMS RO import. This concept extends naturally to WideEP:

| MX-GMS Shadow Worker (Original) | Shadow EP Rank (Extended for WideEP) |
|:-|:-|
| Shadows one entire executor | Shadows one EP rank's expert shard |
| Pre-loads full model weights | Pre-loads only the expert shard for one rank |
| Activates on executor death | Activates on EP rank death |
| KV cache allocation = 1-3s (bottleneck) | **No KV cache needed** (EP ranks don't own per-request KV) |
| Total activation: <5s | **Total activation: <1s** |

**Key architectural insight — Why shadow EP ranks are fundamentally faster than general shadow workers:**

The MX-GMS design identifies KV cache allocation (1-3s) as the activation bottleneck for shadow workers. But in WideEP with `enable_attention_dp=True`, individual EP ranks run data-parallel attention independently — each GPU processes its own requests' attention computation with its own KV cache. The EP ranks exchange *activations* (tokens routed to experts) via AlltoAll during MoE layers, not KV cache state. This means a shadow EP rank needs only expert weights and process group membership — not per-request KV cache state. The KV cache bottleneck simply doesn't apply.

This is not just an optimization — it's a **structural property of WideEP's architecture** that makes sub-second shadow activation architecturally possible in a way that is impossible for general-purpose shadow failover (where KV cache allocation is an irreducible cost). No competitor has exploited this insight because no competitor has both shadow workers (MX-GMS) and WideEP fault tolerance in the same system.

## Cross-Workstream Benefits

### WideEP FT Benefits MX-GMS

1. **Completes the failover story:** MX-GMS's shadow failover design handles whole-executor death but doesn't address partial EP failure. WideEP FT fills this gap, making the MX-GMS failover story complete for the most important production use case.

2. **Validates GMS crash resilience:** WideEP FT Phase 2 is a concrete, high-value use case for GMS's crash-resilient memory. It provides a clear justification for GMS Phase 2 investment.

3. **Defines EP-aware MX identity:** WideEP FT clarifies exactly how `ep_rank` and `ep_size` should be used in MX identity matching for expert shard transfers.

### MX-GMS Benefits WideEP FT

1. **Reduces Phase 2 from minutes to sub-second:** Without GMS, Phase 2 = disk loading (minutes). With GMS, Phase 2 = zero-copy import (<1s). This is the difference between "acceptable degraded mode" and "imperceptible recovery."

2. **Enables shadow EP ranks:** A capability no competitor has. SGLang's Elastic EP permanently runs degraded. This design with MX-GMS restores full capacity in sub-second.

3. **Startup profiling data:** The MX-GMS design includes startup profiling (already implemented) that provides real measurements for weight loading times, directly informing Phase 2 recovery time estimates.

### PR #12718 Benefits Both

1. **Error infrastructure:** Both WideEP FT and MX-GMS shadow failover need to detect failures. PR #12718 provides the classification, budgeting, and propagation infrastructure.

2. **Health check chain:** PR #12718 fixes the zombie worker bug, making health checks actually work. Both WideEP FT (degraded status reporting) and MX-GMS (shadow activation trigger) depend on accurate health reporting.

3. **Fatal shutdown mechanics:** PR #12718's queue drain on fatal shutdown is used by WideEP FT (partial drain for rank failure) and could be used by MX-GMS (full drain before shadow activation).

## Combined Architecture Vision

When all three workstreams are complete, the system handles the full failure lifecycle:

```mermaid
sequenceDiagram
    participant GPU37 as GPU 37
    participant Detect as Layer 1: Detection<br/>(PR #12718)
    participant Survive as Layer 2: Survival<br/>(WideEP FT Phase 1)
    participant Restore as Layer 3: Recovery<br/>(MX-GMS + WideEP FT Phase 2)

    Note over GPU37: ☠️ GPU failure

    GPU37->>Detect: AlltoAll timeout (1-5s)
    Detect->>Detect: classify_error() → EP severe
    Detect->>Detect: rank_budget[37].consume() → exhausted
    Detect->>Survive: mark_failed(rank=37)

    Survive->>Survive: Update active_rank_mask (< 1ms)
    Survive->>Survive: EPLB emergency reconfigure (~10-35ms)
    Survive->>Survive: Migrate weights from host shared memory (~1-5ms/layer)
    Survive->>Survive: Resume serving at N-1 capacity

    Note over Survive: Serving continues (degraded)
    Note over Survive: Total Phase 1: ~5-10s

    par Background: Phase 2
        Restore->>Restore: Orchestrator provisions replacement GPU
        alt GMS available
            Restore->>Restore: GMS zero-copy import (~100ms)
        else MX available
            Restore->>Restore: MX P2P RDMA (~1-2s for expert shard)
        else Disk only
            Restore->>Restore: Load from checkpoint (~1-3 min)
        end
        Restore->>Restore: Reconstruct process group (~100ms)
        Restore->>Restore: EPLB full rebalance (~10ms)
        Restore->>Restore: Update active_rank_mask: all active
    end

    Note over Restore: Full capacity restored
    Note over Restore: Total Phase 2: <1s (GMS) / ~2s (MX) / ~3 min (disk)
```

## What Each Workstream Must NOT Do

Clear boundaries prevent duplicate work:

| Workstream | Responsible For | NOT Responsible For |
|:-----------|:---------------|:-------------------|
| **PR #12718** | Error classification, budget, fatal propagation, health check fix | Per-EP-rank tracking, rank masking, expert redistribution |
| **WideEP FT** | Rank masking, EPLB reconfigure, AlltoAll timeout, failure broadcast, Phase 1+2 orchestration | Weight loading acceleration, crash-resilient memory, GMS/MX APIs |
| **MX-GMS** | Weight streaming (MX), zero-copy import (GMS), shadow workers, crash resilience | Failure detection, AlltoAll modification, expert redistribution logic |
