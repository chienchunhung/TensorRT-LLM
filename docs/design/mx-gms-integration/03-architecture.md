# 3. Proposed Architecture

[< Back to Overview](README.md)

## High-Level Architecture

```mermaid
graph TB
    subgraph "Dynamo Orchestration"
        MXServer["MX Metadata Server<br/>Redis / K8s CRD"]
    end

    subgraph "Node A (Seed)"
        GMS_A["GMS<br/>(Local)"]
        W_A1["TRT-LLM Worker 1<br/>(RW mode)"]
        W_A2["TRT-LLM Worker 2<br/>(RO mode, shadow)"]
        W_A1 --> GMS_A
        W_A2 --> GMS_A
    end

    subgraph "Node B (Replica)"
        GMS_B["GMS<br/>(Local)"]
        W_B1["TRT-LLM Worker 1"]
        W_B2["TRT-LLM Worker 2<br/>(shadow)"]
        W_B1 --> GMS_B
        W_B2 --> GMS_B
    end

    subgraph "Node C (Replica)"
        GMS_C["GMS<br/>(Local)"]
        W_C1["TRT-LLM Worker 1"]
        W_C1 --> GMS_C
    end

    MXServer <-->|gRPC| W_A1
    MXServer <-->|gRPC| W_B1
    MXServer <-->|gRPC| W_C1
    GMS_A -->|"P2P via MX<br/>NIXL/RDMA"| GMS_B
    GMS_A -->|"P2P via MX<br/>NIXL/RDMA"| GMS_C
```

## Component Responsibilities

| Component | Responsibility | Owner |
|:----------|:--------------|:------|
| **MX Server** | Coordinate P2P transfers across nodes; track source availability | Dynamo/MX team |
| **GMS (per node)** | Manage GPU memory; enable zero-copy sharing within node | Dynamo team |
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

    CheckGMS -->|"No (RW mode)"| LoadUnderPool["Load from disk under<br/>torch.cuda.use_mem_pool(gms_pool)"]
    LoadUnderPool --> CommitGMS["finalize_write() →<br/>commit to local GMS"]
    CommitGMS --> PublishMX["Publish as MX source<br/>(BEFORE post_load_weights)"]
    PublishMX --> PostLoad2["Run post_load_weights()"]
    PostLoad2 --> Ready2["Ready to serve<br/>(minutes path, first writer)"]
```

> **Prototype note (MX+GMS combined):** In the current prototype, the GMS RW path always loads from disk — MX P2P is **not** used in GMS RW mode because model parameters are meta tensors at that point (no CUDA buffers for P2P to write into). The priority cascade works through GMS RO: the first worker loads from disk into GMS (RW), subsequent workers import from GMS (RO, ~100ms). Cross-node replication happens via MX in `LoadFormat.AUTO` mode (without GMS). A future optimization could allocate CUDA buffers under the GMS pool first, then attempt MX P2P into those buffers.

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
        D3["GMS RW: load under<br/>torch.cuda.use_mem_pool(gms_pool)<br/>then finalize_write() + commit"]
        D4["GMS RO: post_load_weights() first<br/>then materialize_module()<br/>(zero-copy from GMS pool)"]
    end

    subgraph "Post-Load Hooks (TRT-LLM orchestration)"
        D2["MX: publish_as_source(model, mapping, checkpoint_dir)<br/>BEFORE post_load_weights<br/>Fires for AUTO and GMS-RW modes"]
    end

    D -.->|"checkpoint_format=MX"| D1
    C -.->|"LoadFormat.GMS (RW)"| D3
    C -.->|"LoadFormat.GMS (RO)"| D4
    E -.->|"Add hooks"| D2
```

> **Prototype reference:** See the [`dynamo-integration-prototype`](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype) branch for the full working implementation.
