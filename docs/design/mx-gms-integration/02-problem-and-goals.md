# 2. Problem Statement and Goals

[< Back to Overview](README.md)

## Current Pain Points

| Problem | Impact | Current State | MX/GMS Solution |
|:--------|:-------|:-------------|:----------------|
| **Slow cold-start** | Minutes to serve first request | Each replica loads from disk/network independently | MX: P2P from existing replica (10-20x faster) |
| **Slow failover** | Service degradation during recovery | Failed worker requires full reload (50-114s) | GMS: Crash-resilient memory + shadow failover (<5s recovery) |
| **No crash resilience** | Lost work on process crash | GPU memory released when process dies | GMS: Out-of-process memory survives crashes |
| **Storage bottleneck** | Scaling limited by I/O bandwidth | All replicas compete for storage bandwidth | MX: P2P tree distribution |
| **No zero-downtime updates** | Service interruption during model updates | Stop → reload → restart | GMS + MX: Shadow loads new version while old serves |

## Target Use Cases

### UC1: Autoscaling
Spin up new replicas in seconds (not minutes) when load increases. MX provides P2P weight streaming from existing replicas, eliminating the download/load bottleneck.

### UC2: Shadow Failover (Primary GMS Use Case)
Instant switchover when primary worker fails. A shadow worker pre-imports weights via GMS RO zero-copy (~100ms) and holds them in GPU memory without allocating KV cache or serving traffic. When the primary crashes, the shadow activates in <5s: lock upgrade (~10ms) → KV cache allocation (~1-3s) → executor start (~100ms). Without GMS, recovery requires full weight reload (50-114s for Qwen 72B).

> **Why "shadow on the same GPU" is realistic:** The shadow holds only weights (1/TP per GPU, ~18GB for Qwen 72B) with no KV cache. Since the primary's KV cache is the main memory consumer, the shadow's weight-only footprint fits alongside the active instance. This is not about running two active serving instances (which wouldn't fit for large models), but about pre-staging for fast failover.

### UC3: Rolling Updates
Zero-downtime model version updates. New version loads via MX while old version continues serving. GMS enables atomic switchover.

### UC4: Multi-Model / LoRA Sharing (Niche)
For smaller models or multi-LoRA deployments, multiple instances can share base model weights on the same GPU via GMS zero-copy, enabling denser packing. This is realistic for models small enough to fit multiple instances with independent KV caches on a single GPU, but is not the primary GMS use case for large models.

### UC5: Disaggregated Serving
Efficient prefill/decode separation where context and generation workers may need different startup patterns and memory sharing strategies (see [Disaggregated Serving Interaction](07-disagg-interaction.md)).

## Goals

1. **Native MX support**: `--checkpoint-format mx` for P2P weight loading (parity with vLLM)
2. **Native GMS support**: `--load-format gms` for shared memory loading
3. **Combined MX+GMS**: `--checkpoint-format mx --load-format gms` for cross-node P2P with within-node crash resilience and shadow failover (the two axes compose independently — see [Implementation & API Design](04-implementation-plan.md#design-principle-two-orthogonal-axes))
4. **Executor-level failover**: Shadow failover integrated with PyExecutor sleep/wake
5. **KV cache extension path**: Design that enables future KV cache persistence via GMS/KVBM
6. **Backward compatibility**: Existing workflows unchanged; MX/GMS are opt-in
7. **Clean extension points**: APIs usable by future backends beyond MX/GMS

## Non-Goals

1. Modifying MX or GMS core implementations
2. Supporting legacy TensorRT engine backend (PyTorch backend only)
3. Automatic MX server deployment (separate concern)
4. Full KV cache sharing via GMS in this proposal (designed for, not implemented — see [KV Cache Extension](08-kv-cache-extension.md))
5. Compile cache sharing via MX (future work, noted in [Startup Profiling](09-startup-profiling.md))
