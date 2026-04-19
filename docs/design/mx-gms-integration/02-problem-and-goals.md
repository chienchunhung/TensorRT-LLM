# 2. Problem Statement and Goals

[< Back to Overview](README.md)

## Current Pain Points

| Problem | Impact | Current State | MX/GMS Solution |
|:--------|:-------|:-------------|:----------------|
| **Slow cold-start** | Minutes to serve first request | Each replica loads from disk/network independently | MX: P2P from existing replica (10-20x faster) |
| **Slow failover** | Service degradation during recovery | Failed worker requires full reload (~75–390s on v3 code, node/tier-dependent — see [§11 Results](11-results-analysis.md)) | GMS + compile cache: Crash-resilient memory + shadow failover (<5s recovery) |
| **No crash resilience** | Lost work on process crash | GPU memory released when process dies | GMS: Out-of-process memory survives crashes |
| **Storage bottleneck** | Scaling limited by I/O bandwidth | All replicas compete for storage bandwidth | MX: P2P tree distribution |
| **No zero-downtime updates** | Service interruption during model updates | Stop → reload → restart | GMS + MX: Shadow loads new version while old serves |

## Target Use Cases

### UC1: Autoscaling
Spin up new replicas in seconds (not minutes) when load increases. MX provides P2P weight streaming from existing replicas, eliminating the download/load bottleneck.

### UC2: Shadow Failover (Primary GMS Use Case)
Instant switchover when primary worker fails. A shadow worker pre-imports weights via GMS RO zero-copy (~100ms) and holds them in GPU memory without allocating KV cache or serving traffic. When the primary crashes, the shadow activates in <5s: lock upgrade (~10ms) → KV cache allocation (~1-3s) → **cache-warm warmup** (~0.5-2s via [§07 Tiered Compile Cache](07-compile-cache.md)) → executor start (~100ms). Without GMS, recovery requires full weight reload (~75–390s for Qwen 72B depending on storage tier and code vintage — see [§11 Results](11-results-analysis.md)). Without a warm compile cache, warmup alone adds ~43s on v3 code (post-PR #12407), breaking the <5s budget — compile cache is therefore a hard dependency, not optional.

> **Why "shadow on the same GPU" is realistic:** The shadow holds only weights (1/TP per GPU, ~18GB for Qwen 72B) with no KV cache. Since the primary's KV cache is the main memory consumer, the shadow's weight-only footprint fits alongside the active instance. This is not about running two active serving instances (which wouldn't fit for large models), but about pre-staging for fast failover.

### UC3: Rolling Updates
Zero-downtime model version updates. New version loads via MX while old version continues serving. GMS enables atomic switchover.

### UC4: Multi-Model / LoRA Sharing (Niche)
For smaller models or multi-LoRA deployments, multiple instances can share base model weights on the same GPU via GMS zero-copy, enabling denser packing. This is realistic for models small enough to fit multiple instances with independent KV caches on a single GPU, but is not the primary GMS use case for large models.

### UC5: Disaggregated Serving
Efficient prefill/decode separation where context and generation workers may need different startup patterns and memory sharing strategies (see [Disaggregated Serving Interaction](08-disagg-interaction.md)).

## Where MX and GMS Help — and Where They Don't

MX and GMS cover **orthogonal axes**: MX handles **inter-node** weight distribution; GMS handles **intra-GPU** (same physical device) weight lifetime. Neither is a general "faster loading" knob — both have explicit scope boundaries that matter for deployment planning.

**Scope constraints (verified against the Dynamo repo):**

- **GMS is strictly per-GPU, not per-node.** From the [GMS README](https://github.com/ai-dynamo/dynamo/tree/main/lib/gpu_memory_service): *"Each GMS server is responsible for managing memory of only 1 GPU, and does not interact with GMS servers corresponding to other GPUs."* CUDA VMM handles cannot cross GPUs, so two workers on the same node using **disjoint** GPUs (e.g., 2× TP=4 on an 8-GPU box) **do not share weights via GMS** even though they're on the same node.
- **MX requires a warm GPU source somewhere in the cluster** (or falls back to disk). The [ModelExpress README](https://github.com/ai-dynamo/modelexpress) describes coordinating downloads so *"no other node duplicates this download, reducing external ingress"* — the first node still pulls from HuggingFace (or equivalent), and subsequent nodes get P2P via RDMA/InfiniBand from that node's GPU memory. There is no dedicated "MX cache server" holding weights without serving.
- **MX is focused on inter-node RDMA.** The README discusses InfiniBand/RDMA transport; intra-node NVLink P2P is not an explicit use case. Within a node, GMS zero-copy (same GPU) is the right tool.

**Scenario matrix:**

| Scenario | MX helps? | GMS helps? | Why |
|:---------|:----------|:-----------|:----|
| First-ever replica on a fresh cluster (nothing warm anywhere) | No — falls back to disk/HF download | No — no prior GPU state | Neither has a source |
| Nth replica on a **different node**, seed already serving | **Yes** — RDMA pull from seed's GPU | No — cross-node, can't share VMM | This is MX's core win |
| Many replicas launching simultaneously across nodes | **Yes, biggest win** — avoids storage contention | No | MX P2P distribution vs. each replica hammering storage |
| Same-node second worker on **disjoint GPUs** (2× TP=4 on 8-GPU box) | Not the MX focus (NVLink P2P not explicit) | **No — different GPUs, VMM can't cross** | Each worker loads independently from disk |
| Same-GPU shadow standby (active + shadow co-located) | Not applicable (same GPU) | **Yes** — RO import (~100ms) for fast failover | GMS's core win |
| Process crash & restart on the same GPU | No | **Yes** — weights survive in GMS pool, ~100ms reattach | Out-of-process memory persists |
| Rolling binary upgrade on the same GPU | No | **Yes** — new process re-imports from GMS pool | No reload cost |
| Multi-LoRA or small-model co-location on the same GPU | No | **Yes** — deduplicated base weights (UC4) | Per-GPU zero-copy sharing |

**One-line framing:**

- **MX = inter-node weight distribution.** *"I have weights on node A; how do I get them to nodes B, C, D without each of them hammering shared storage?"* Benefit surfaces when a warm source exists somewhere in the cluster and we're distributing to a **different** node.
- **GMS = intra-GPU weight lifetime.** *"Weights are already on this physical GPU; how do I avoid reloading them when the process restarts, gets upgraded, or when a shadow needs to pre-stage for failover?"* Benefit is strictly **per-GPU, same physical device** — never crosses GPUs, even on the same node.

They compose naturally in MX+GMS: each node's first worker pulls weights via MX (fast cross-node), commits them into that GPU's GMS pool, and subsequent same-GPU processes (shadows, upgrades) import zero-copy from GMS.

## Goals

1. **Native MX support**: `--checkpoint-format mx` for P2P weight loading (parity with vLLM)
2. **Native GMS support**: `--load-format gms` for shared memory loading
3. **Combined MX+GMS**: `--checkpoint-format mx --load-format gms` for cross-node P2P with within-node crash resilience and shadow failover (the two axes compose independently — see [Implementation & API Design](04-implementation-plan.md#design-principle-two-orthogonal-axes))
4. **Executor-level failover**: Shadow failover integrated with PyExecutor sleep/wake
5. **KV cache extension path**: Ensure Phases 1–3 don't block a future KVBM-based KV cache integration via the KV Cache Connector API (KV cache is out of GMS's scope — see [§09](09-kv-cache-extension.md))
6. **Backward compatibility**: Existing workflows unchanged; MX/GMS are opt-in
7. **Clean extension points**: APIs usable by future backends beyond MX/GMS

## Non-Goals

1. Modifying MX or GMS core implementations
2. Supporting legacy TensorRT engine backend (PyTorch backend only)
3. Automatic MX server deployment (separate concern)
4. KV cache persistence or sharing via GMS — KV cache is out of GMS's scope. Tiered KV cache is addressed by Dynamo's KVBM via the KV Cache Connector API, on a separate track (see [KV Cache Extension Path](09-kv-cache-extension.md))
5. Compile cache sharing via MX (future work, noted in [§14 Open Questions](14-open-questions.md))
