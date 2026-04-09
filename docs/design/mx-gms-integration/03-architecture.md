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
    Start["TRT-LLM Worker starts<br/>--load-format mx-gms"] --> CheckGMS{"Local GMS has<br/>committed weights?"}

    CheckGMS -->|Yes| ImportGMS["Import from GMS<br/>(RO mode, zero-copy)"]
    ImportGMS --> PostLoad["Run post_load_weights()<br/>Validate tensor shapes"]
    PostLoad --> Ready["Ready to serve<br/>(~100ms path)"]

    CheckGMS -->|No| QueryMX{"MX server has<br/>READY sources?"}

    QueryMX -->|Yes| FilterRank["Filter by matching<br/>TP/PP/EP rank"]
    FilterRank --> P2PReceive["P2P receive via NIXL<br/>(GPU-to-GPU RDMA)"]
    P2PReceive --> CommitGMS["Commit to local GMS<br/>(for future sharing)"]
    CommitGMS --> PostLoad2["Run post_load_weights()"]
    PostLoad2 --> PublishMX["Publish as MX source"]
    PublishMX --> Ready2["Ready to serve<br/>(~15-30s path)"]

    QueryMX -->|No| LoadDisk["Load from disk/HuggingFace<br/>(standard path)"]
    LoadDisk --> CommitGMS2["Commit to local GMS"]
    CommitGMS2 --> PostLoad3["Run post_load_weights()"]
    PostLoad3 --> PublishMX2["Publish as MX source<br/>(seed the cluster)"]
    PublishMX2 --> Ready3["Ready to serve<br/>(minutes path)"]
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

    subgraph "MX/GMS Integration Points"
        D1["MX: New MXCheckpointLoader<br/>(TRT-LLM code, calls MX SDK)<br/>GMS RO: Call materialize_module_from_gms<br/>(GMS library function)"]
        D2["Post-load (TRT-LLM orchestration):<br/>MX: mx_client.register_as_source()<br/>GMS: gms_client.commit()"]
        D3["GMS RW: wrap with<br/>torch.cuda.use_mem_pool(gms_pool)<br/>(GMS library provides allocator)"]
    end

    D -.->|"Replace with"| D1
    E -.->|"Add"| D2
    C -.->|"Inject"| D3
```
