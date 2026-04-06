# 2.7 Parallelism Strategies

[< Back to Overview](README.md)

## Overview and Decision Tree

```mermaid
flowchart TD
    Start["Model too large for one GPU?"]
    Start -->|No| None["No parallelism"]
    Start -->|Yes| Dense{"Dense or MoE?"}
    Dense -->|Dense| TP["Tensor Parallel<br/>split weights"]
    TP --> TPFit{"Fits with TP?"}
    TPFit -->|No| TPPP["TP + Pipeline Parallel<br/>split layers"]
    TPFit -->|Yes| BatchQ{"High batch?"}
    BatchQ -->|Yes| ADP["+ Attention Data Parallel"]
    BatchQ -->|No| TPDone["TP sufficient"]

    Dense -->|MoE| EP["Expert Parallel<br/>split experts"]
    EP --> Scale{"Large scale<br/>DeepSeek/Llama4?"}
    Scale -->|Yes| WEP["Wide-EP + EPLB<br/>load-balanced slots"]
    Scale -->|No| HybridQ{"Expert too large?"}
    HybridQ -->|Yes| ETP["Hybrid EP x TP"]
    HybridQ -->|No| EPDone["EP sufficient"]

    ADP --> LongCtx{"Long context?"}
    WEP --> LongCtx
    LongCtx -->|Yes| CP["+ Context Parallel<br/>Ulysses or Helix"]

    WEP --> NVL{"NVL72?"}
    NVL -->|Yes| DWDP["DWDP<br/>Distributed Weight DP"]
```

## Strategy Details

| Strategy | Abbrev | What It Splits | Communication | Best For |
|:---------|:-------|:--------------|:--------------|:---------|
| **Tensor Parallel** | TP | Weight matrices across GPUs | AllReduce / AllGather | Small batch; memory-constrained |
| **Pipeline Parallel** | PP | Layers across GPUs | P2P send/recv of activations | Very large models; limited bandwidth |
| **Data Parallel** | DP / ADP | Requests across replicas | None (independent); KV cache partitioned | Large batch; high throughput |
| **Expert Parallel** | EP | MoE experts across GPUs | All-to-all token dispatch/combine | MoE with many experts |
| **Context Parallel** | CP | Long sequences across GPUs | All-to-all (Ulysses) or AllGather/ReduceScatter (Helix) | 100K+ token contexts |
| **Wide-EP** | Wide-EP | Experts with load-balanced replication | Custom NVLink all-to-all; one-sided AlltoAll | Large MoE (DeepSeek-V3/R1, Llama4) |
| **Distributed Weight DP** | DWDP | Weights + data across NVL72 | NVLink all-reduce + custom scheduling | NVL72 rack-scale deployments |

## Tensor Parallelism (TP)

Splits weight matrices across GPUs within the same layer. Implemented in `_torch/modules/linear.py`:

- **Column-parallel:** Output features split; optional AllGather when `gather_output=True`
- **Row-parallel:** Input features split; AllReduce to combine partial GEMMs

For attention, `num_heads` is divided across TP ranks. Communication uses `torch.ops.trtllm.allreduce*` with automatic strategy selection (NCCL, Unified Buffer, MNNVL).

## Pipeline Parallelism (PP)

Distributes consecutive layers across GPUs. Implemented in `_torch/models/modeling_utils.py`:

- `DecoderModel.__pp_init__` assigns layer ranges via `mapping.pp_layers(num_layers)`
- First local layer wrapped with `forward_after_recv` (P2P receive)
- Last local layer wrapped with `forward_before_send` (P2P send)

The executor uses `_executor_loop_pp` with micro-batching and async P2P via `PPCommTorch`/`PPCommNCCL`.

## Wide Expert Parallelism (Wide-EP)

```mermaid
graph LR
    subgraph "Traditional EP"
        G1_T["GPU 0: Expert 0, 1"]
        G2_T["GPU 1: Expert 2, 3"]
        G3_T["GPU 2: Expert 4, 5"]
        G4_T["GPU 3: Expert 6, 7"]
    end

    subgraph "Wide-EP with EPLB"
        G1_W["GPU 0: Slot 0=Expert 0<br/>Slot 1=Expert 2 replica"]
        G2_W["GPU 1: Slot 0=Expert 2<br/>Slot 1=Expert 3"]
        G3_W["GPU 2: Slot 0=Expert 4<br/>Slot 1=Expert 2 replica"]
        G4_W["GPU 3: Slot 0=Expert 6<br/>Slot 1=Expert 7"]
    end
```

Decouples **logical experts** from **physical slots**, enabling:

- Multiple replicas of "hot" experts across GPUs
- **Offline EPLB**: Pre-computed placement from historical workload statistics
- **Online EPLB**: Dynamic placement adapting to real-time traffic
- Custom all-to-all kernels optimized for NVLink/MNNVL

**Key file:** `_torch/modules/fused_moe/fused_moe_wide_ep.py` (`WideEPMoE`)

## Distributed Weight Data Parallelism (DWDP)

**New in v1.3.** Designed for NVL72 rack-scale deployments. Distributes both model weights and data across all 72 GPUs for maximum throughput. Documented in blog19 (April 2026).

## What's New (v1.2-v1.3)

- **DWDP** for NVL72 rack-scale deployments.
- **One-sided AlltoAll over NVLink** for MoE expert dispatch — eliminates synchronization overhead in EP communication (blog18).
- **KV cache-aware ADP router** with prefix-affinity request routing.
- **Helix CP for DeepSeek v3.2 with GQA**.
- **EPLB for TRTLLM-Gen** — expert load balancing integrated with the TRTLLM-Gen attention backend.
- **CUDA graph support for DeepEP**.
- **Dynamic SMEM block routing in MoE**.
- **LM Head Sharding** — distributes the language model head across GPUs.

## Framework Comparison

| Framework | Parallelism Support |
|:----------|:-------------------|
| **TensorRT-LLM** | TP, PP, EP, ADP, CP (Ulysses/Helix), Wide-EP + EPLB, DWDP — most comprehensive |
| **vLLM** | TP, PP, EP; elastic EP with NIXL for dynamic scaling |
| **SGLang** | TP, PP, EP, DP; elastic EP for partial failure tolerance |

TRT-LLM's **DWDP**, **Wide-EP with EPLB**, **Helix CP**, and **one-sided AlltoAll** are distinctive capabilities not matched by other frameworks.
