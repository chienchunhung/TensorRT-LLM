# 2. Problem Statement and Goals

[< Back to Overview](README.md)

## Current Pain Points

| Problem | Impact | Current State | MX/GMS Solution |
|:--------|:-------|:-------------|:----------------|
| **Slow cold-start** | Minutes to serve first request | Each replica loads from disk/network independently | MX: P2P from existing replica (10-20x faster) |
| **Memory waste** | Limits workers per GPU | Multiple workers duplicate model weights | GMS: Zero-copy sharing (Nx to 1x) |
| **Slow failover** | Service degradation during recovery | Failed worker requires full reload | GMS: Crash-resilient memory (<5s recovery) |
| **Storage bottleneck** | Scaling limited by I/O bandwidth | All replicas compete for storage bandwidth | MX: P2P tree distribution |
| **No crash resilience** | Lost work on process crash | GPU memory released when process dies | GMS: Out-of-process memory survives crashes |

## Target Use Cases

### UC1: Autoscaling
Spin up new replicas in seconds (not minutes) when load increases. MX provides P2P weight streaming from existing replicas, eliminating the download/load bottleneck.

### UC2: Multi-Tenant Serving
Multiple workers share model weights on the same GPU via GMS zero-copy. This enables denser packing — N workers with 1x memory instead of Nx memory.

### UC3: Shadow Failover
Instant switchover when primary worker fails. A shadow worker maintains VA-stable memory references via GMS. When the primary crashes, the shadow imports the same memory and resumes serving in <5s.

### UC4: Rolling Updates
Zero-downtime model version updates. New version loads via MX while old version continues serving. GMS enables atomic switchover.

### UC5: Disaggregated Serving
Efficient prefill/decode separation where context and generation workers may need different startup patterns and memory sharing strategies (see [Disaggregated Serving Interaction](08-disagg-interaction.md)).

## Goals

1. **Native MX support**: `--load-format mx` for P2P weight loading (parity with vLLM)
2. **Native GMS support**: `--load-format gms` for shared memory loading
3. **Combined MX+GMS**: `--load-format mx-gms` for cross-node P2P with within-node sharing
4. **Executor-level failover**: Shadow failover integrated with PyExecutor sleep/wake
5. **KV cache extension path**: Design that enables future KV cache persistence via GMS/KVBM
6. **Backward compatibility**: Existing workflows unchanged; MX/GMS are opt-in
7. **Clean extension points**: APIs usable by future backends beyond MX/GMS

## Non-Goals

1. Modifying MX or GMS core implementations
2. Supporting legacy TensorRT engine backend (PyTorch backend only)
3. Automatic MX server deployment (separate concern)
4. Full KV cache sharing via GMS in this proposal (designed for, not implemented — see [KV Cache Extension](07-kv-cache-extension.md))
5. Compile cache sharing via MX (future work, noted in [Startup Profiling](10-startup-profiling.md))
