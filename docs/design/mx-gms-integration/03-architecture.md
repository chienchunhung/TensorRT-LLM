# 3. Proposed Architecture

[< Back to Overview](README.md)

## High-Level Architecture

```mermaid
graph TB
    subgraph "Dynamo Orchestration"
        MXServer["MX Metadata Server<br/>Redis / K8s CRD"]
    end

    subgraph "Node A (Seed)"
        GMS_A["GMS Launcher<br/>(spawns per-GPU processes)"]
        GMS_A0["GMS GPU0<br/>(weights + kv_cache)"]
        GMS_A1["GMS GPU1<br/>(weights + kv_cache)"]
        GMS_AN["GMS GPU2..7<br/>(weights + kv_cache)"]
        GMS_A --> GMS_A0
        GMS_A --> GMS_A1
        GMS_A --> GMS_AN
        W_A1["TRT-LLM Worker 1<br/>(RW mode)"]
        W_A2["TRT-LLM Worker 2<br/>(RO mode, shadow)"]
        W_A1 --> GMS_A0
        W_A2 --> GMS_A0
    end

    subgraph "Node B (Replica)"
        GMS_B["GMS Launcher"]
        GMS_B0["GMS GPU0..7"]
        GMS_B --> GMS_B0
        W_B1["TRT-LLM Worker 1"]
        W_B2["TRT-LLM Worker 2<br/>(shadow)"]
        W_B1 --> GMS_B0
        W_B2 --> GMS_B0
    end

    subgraph "Node C (Replica)"
        GMS_C["GMS Launcher"]
        GMS_C0["GMS GPU0..7"]
        GMS_C --> GMS_C0
        W_C1["TRT-LLM Worker 1"]
        W_C1 --> GMS_C0
    end

    MXServer <-->|gRPC| W_A1
    MXServer <-->|gRPC| W_B1
    MXServer <-->|gRPC| W_C1
    W_A1 -->|"P2P via MX<br/>NIXL/RDMA"| W_B1
    W_A1 -->|"P2P via MX<br/>NIXL/RDMA"| W_C1
```

## Component Responsibilities

| Component | Responsibility | Owner |
|:----------|:--------------|:------|
| **MX Server** | Coordinate P2P transfers across nodes; track source availability | Dynamo/MX team |
| **GMS (per GPU, per tag)** | Manage GPU memory; enable zero-copy sharing within node. Runs as one independent process per GPU per tag (e.g., 16 processes on an 8-GPU node: 8 for `weights` + 8 for `kv_cache`). A per-node launcher spawns all processes. Socket paths use GPU UUID for stability: `{GMS_SOCKET_DIR}/gms_{GPU_UUID}_{tag}.sock`. Sharing is strictly per-GPU (CUDA VMM constraint). | Dynamo team |
| **TRT-LLM Weight Loaders** | Integrate with MX/GMS via clean APIs | TRT-LLM team (this proposal) |
| **NIXL/UCX** | Execute actual GPU-to-GPU RDMA transfers | NVIDIA |
| **TRT-LLM Executor** | Shadow failover, sleep/wake lifecycle | TRT-LLM team (this proposal) |

## Data Flow: New Replica Startup

```mermaid
flowchart TD
    Start["TRT-LLM Worker starts<br/>--checkpoint-format mx --load-format gms"] --> CheckGMS{"Local GMS has<br/>committed weights?<br/>(GMSBackend.connect resolves mode)"}

    CheckGMS -->|"Yes (RO mode)"| PostLoadRO["Run post_load_weights()<br/>(set up module aliases FIRST)"]
    PostLoadRO --> ImportGMS["materialize_module()<br/>(GMS RO, zero-copy import)"]
    ImportGMS --> Ready["Ready to serve<br/>(~100ms path)"]

    CheckGMS -->|"No (RW mode)"| LoadUnderPool["Load from disk inside<br/>gms_backend.mem_pool_scope(device)<br/>(delegates to gms_use_mem_pool)"]
    LoadUnderPool --> CommitGMS["move_untracked_params() +<br/>finalize_write() →<br/>commit to local GMS"]
    CommitGMS --> PublishMX["Publish as MX source<br/>(BEFORE post_load_weights)"]
    PublishMX --> PostLoad2["Run post_load_weights()"]
    PostLoad2 --> Ready2["Ready to serve<br/>(minutes path, first writer)"]
```

> **Critical design constraint — MX+GMS combined mode:**
>
> In the current prototype, `checkpoint_format="MX"` + `load_format=GMS` behaves **identically** to `checkpoint_format="HF"` + `load_format=GMS` (GMS-only). The MX checkpoint format provides no additional benefit when GMS is active. Here is why:
>
> **The root cause: CUDA memory pool isolation.** In GMS RW mode, all weight memory must live in the GMS-managed memory pool so that RO readers can later zero-copy import it. The loading path wraps weight loading in `gms_backend.mem_pool_scope(device)` (a context manager that delegates to upstream `gms_use_mem_pool(tag, device)`), ensuring all CUDA allocations land in GMS memory. However, when MX receives weights via P2P RDMA, the MX/NIXL layer allocates CUDA buffers **inside the MX SDK** — outside the pool-scope context. Those received weights land in regular CUDA memory, not the GMS pool, so GMS cannot track, manage, or share them with RO readers — defeating the entire purpose of GMS RW mode. Aligning MX-NIXL to write into pre-allocated GMS-pool buffers is tracked as MX-5 in [§15 Upstream Alignment Requests](15-prototype-validation-plan.md#-api-alignment--prototype--current-gms--mx-done); until that lands, MX P2P is intentionally bypassed inside GMS RW mode and the GMS RW path falls back to disk loading.
>
> **What this means in practice:**
>
> | Mode | Node B, Worker 1 (first on node) | Node B, Worker 2+ |
> |:-----|:---------------------------------|:-------------------|
> | **MX only** (`LoadFormat.AUTO`) | P2P from Node A (~15-30s), regular CUDA memory | Must load independently (no sharing) |
> | **GMS only** (`LoadFormat.GMS`) | Load from disk (minutes), commits to GMS | Shadow: zero-copy RO import (~100ms) |
> | **MX + GMS** (`checkpoint_format="MX"`, `LoadFormat.GMS`) | Load from disk (minutes), commits to GMS — **same as GMS-only** | Shadow: zero-copy RO import (~100ms) |
>
> MX and GMS currently operate as **separate modes**, not a truly composed solution. MX provides fast cross-node startup (in `LoadFormat.AUTO`); GMS provides within-node crash resilience + shadow failover (in `LoadFormat.GMS`). They do not compose because the GMS RW path cannot leverage MX P2P.
>
> **Future optimization path:** Pre-allocate empty CUDA buffers under the GMS pool, then pass those buffer pointers to the MX SDK as P2P receive targets. This would allow MX to write directly into GMS-managed memory, giving the best of both: P2P speed (~15-30s) + GMS crash resilience (~100ms shadow import). This requires MX SDK support for receiving into pre-allocated buffers rather than SDK-managed allocations.

For **MX-only mode** (`--checkpoint-format mx`, no GMS):

```mermaid
flowchart TD
    Start2["TRT-LLM Worker starts<br/>--checkpoint-format mx"] --> QueryMX{"MX server has<br/>compatible sources?"}

    QueryMX -->|Yes| FilterRank["Filter by matching<br/>TP/PP/EP rank"]
    FilterRank --> P2PReceive["P2P receive via NIXL<br/>(GPU-to-GPU RDMA)"]
    P2PReceive --> MarkPresharded["Mark Linear modules<br/>_weights_presharded = True"]
    MarkPresharded --> PublishMX3["Publish as MX source<br/>(BEFORE post_load_weights)"]
    PublishMX3 --> PostLoad4["Run post_load_weights()"]
    PostLoad4 --> Ready3["Ready to serve<br/>(~15-30s path)"]

    QueryMX -->|No| LoadDisk2["Load from disk/HuggingFace<br/>(inherited HF fallback)"]
    LoadDisk2 --> PublishMX4["Publish as MX source"]
    PublishMX4 --> PostLoad5["Run post_load_weights()"]
    PostLoad5 --> Ready4["Ready to serve<br/>(minutes path)"]
```

## Data Flow: Shadow Failover

```mermaid
sequenceDiagram
    participant Primary as Primary Worker
    participant GMS as GMS (Local)
    participant Shadow as Shadow Worker
    participant Router as Dynamo Router
    participant Exec as PyExecutor

    Note over Primary,Exec: Normal Operation
    Primary->>GMS: Holds RW lock (socket connection)
    Shadow->>GMS: Holds RO import (weights shared)
    Shadow->>Exec: PyExecutor in sleep mode (KV cache released)
    Router->>Primary: Routes all requests

    Note over Primary,Exec: Primary Crashes
    Primary--xGMS: Socket disconnects → RW lock auto-released
    GMS->>GMS: Memory persists (out-of-process)

    Note over Primary,Exec: Shadow Takeover (<5s)
    Shadow->>GMS: Upgrade RO → RW lock
    Shadow->>Exec: Wake: materialize_with_tag("kv_cache")
    Shadow->>Exec: Rebuild KV cache manager state
    Shadow->>Router: Register as new primary
    Router->>Shadow: Routes new requests

    Note over Primary,Exec: In-Flight Request Handling
    Router->>Router: Detect primary failure (health check)
    Router->>Router: Re-queue pending requests
    Router->>Shadow: Replay re-queued requests
```

## Integration with TRT-LLM Model Loading Pipeline

```mermaid
flowchart LR
    subgraph "Existing Pipeline"
        A["HF Config"] --> B["AutoModelForCausalLM<br/>._resolve_class()"]
        B --> C["from_config()<br/>meta-device init"]
        C --> D["checkpoint_loader<br/>.load_weights()"]
        D --> E["model.load_weights()<br/>post_load_weights()"]
        E --> F["nn.Module on GPU"]
    end

    subgraph "Weight Source Axis (checkpoint_format)"
        D1["MX: HfCheckpointLoader subclass<br/>lazy connect, P2P via modelexpress SDK<br/>p2p_succeeded → _weights_presharded"]
    end

    subgraph "Memory Mgmt Axis (LoadFormat)"
        D3["GMS RW: load inside<br/>gms_backend.mem_pool_scope(device)<br/>then move_untracked_params() + finalize_write()"]
        D4["GMS RO: post_load_weights() first<br/>then materialize_module()<br/>(zero-copy from GMS pool)"]
    end

    subgraph "Post-Load Hooks (TRT-LLM orchestration)"
        D2["MX: publish_as_source(model)<br/>delegates to publish_model_params()<br/>BEFORE post_load_weights<br/>Fires for AUTO and GMS-RW modes"]
    end

    D -.->|"checkpoint_format=MX"| D1
    C -.->|"LoadFormat.GMS (RW)"| D3
    C -.->|"LoadFormat.GMS (RO)"| D4
    E -.->|"Add hooks"| D2
```

> **Prototype reference:** See the [`dynamo-integration-prototype`](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype) branch for the full working implementation.
