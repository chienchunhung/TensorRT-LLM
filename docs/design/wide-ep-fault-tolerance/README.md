# WideEP Fault Tolerance for TensorRT-LLM

**Status:** Draft
**Created:** 2026-04-10
**Last Updated:** 2026-04-10

---

## Executive Summary

WideEP (Wide Expert Parallelism) distributes MoE experts across 32-72+ GPUs for models like DeepSeek-V3/R1 (256 experts, 681GB). At this scale, GPU failures are a daily reality — and today, a single GPU failure in a WideEP group takes down the entire serving instance (infinite hang in AlltoAll communication, no detection, no recovery).

This design proposes a **two-phase fault tolerance architecture** for WideEP:

- **Phase 1 — Immediate Survival:** Mask the failed rank in AlltoAll communication, redistribute experts to surviving ranks via EPLB, and continue serving at reduced capacity within seconds.
- **Phase 2 — Full Restoration:** Optionally bring up a replacement rank (accelerated by MX/GMS if available), reconstruct the process group, and restore full capacity.

The design builds on two concurrent workstreams:
- [PR #12718: Fatal Error Detection](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — Error classification and fatal error detection infrastructure
- [MX+GMS+TRT-LLM Integration](https://docs.google.com/document/d/14SZmmFcoakgIx2OC4dt8pWcHU14PDTN9KlAKqLoZ15s/edit?usp=sharing) — Fast weight loading and crash-resilient memory for accelerated recovery

Together, the three workstreams form a **layered reliability stack**: detect failures (PR #12718) → survive partial failures (this design) → recover quickly (MX-GMS).

**Competitive context:** SGLang shipped Elastic EP in March 2026 (~6.5s recovery, tolerates up to 16/32 rank failures). vLLM has an active RFC (#27774) for fault-tolerant EP. TRT-LLM currently has no EP-level fault tolerance — this is a critical gap for production WideEP deployments.

## Scope and Non-Goals

### In Scope (Primary Track — Phase 1 Priority)

- **Aggregated WideEP serving** — a single `trtllm-serve` instance runs prefill and decode on one EP group spanning 32-72+ GPUs (the DeepSeek-V3/R1 NVL72 configuration)
- **PyTorch backend** (the default backend that WideEP is built on)
- **NVLink communication backends** as the primary target: `NVLinkOneSided` (primary), `NVLinkTwoSided`, and `AllGatherReduceScatter` (NCCL fallback)
- **Intra-node and multi-node EP** where the collective paths are the ones listed above
- **Single GPU failure** in MVP; **multi-failure consensus** in full Phase 1

### In Scope (Deferred Track — After Phase 1 MVP)

- **Disaggregated serving fault tolerance.** Production TRT-LLM disagg separates prefill and decode into independent worker pools, each with its own EP group, connected by KV cache transfer (NIXL/UCX/MPI). A failure in one pool has different semantics from an aggregated failure:
  - Request state is split across pools — prefill-side context and decode-side KV cache are separate
  - KV cache transfers in flight at the time of failure have their own failure mode
  - The orchestration layer (`trtllm-serve` proxy) has to coordinate recovery across pools, not just within one

  This is scoped as a separate **Phase 1-DS** track that starts after Phase 1 MVP lands and can proceed in parallel with Phase 1 v1. The per-pool collective-level FT from the primary track applies unchanged within each pool; Phase 1-DS adds the cross-pool coordination layer. See [§09 Phase 1-DS](09-implementation-plan.md#phase-1-ds-disaggregated-serving-ft-p1) and [§10 Q7](10-risks.md#q7-how-does-wideep-ft-interact-with-disaggregated-serving).

### Out of Scope (Explicit Non-Goals)

- **DeepEP / DeepEPLowLatency backends** — masking requires a public NVSHMEM `mask_buffer_ptr` API that does not exist yet. In scope as a v1 target *if* the external API becomes available; otherwise deferred indefinitely. See [§10 Risk 2](10-risks.md#risk-2-deepep-backend-limitations).
- **TensorRT engine backend** — legacy backend per `AGENTS.md`; FT work targets PyTorch only.
- **Standard EP (≤8 GPUs)** — not the WideEP bottleneck. Today's infinite-hang failure mode is an issue specific to multi-node NVLink AlltoAll. Intra-node EP failures are handled adequately by process restart + PR #12718's error classification.
- **Inference request durability** — if a request is mid-flight when a rank fails, that specific request is lost. Recovering individual requests across failures is an orchestration-layer concern, not a collective-layer one.
- **Preemptive / predictive failure mitigation** — deferred to Phase 3.

## Why This Is Technically Hard

This is not routine integration work. The design tackles several problems that are individually challenging and collectively require cross-layer systems expertise:

1. **GPU kernel-level synchronization modification.** The NVLink AlltoAll kernels implement multi-GPU coordination via completion flags in symmetric memory. GPU threads spin-wait on flag values written by peer GPUs — there is no timeout, no abort, no fallback. Adding rank masking requires modifying CUDA kernel code that interacts with multi-GPU memory ordering and synchronization primitives. This is fundamentally different from the API-level masking used by SGLang (Mooncake `activeRanks`) or proposed by vLLM (DeepEP `mask_buffer_ptr`), where masking is provided by an external library. We modify the kernel itself.

2. **Distributed consensus without the dead participant.** When a rank dies, the surviving N-1 ranks must agree on which rank is dead — but the dead rank cannot participate in the agreement protocol, and the communication infrastructure is itself degraded. This is a variant of the classic failure detection problem in asynchronous distributed systems, where a slow process is indistinguishable from a dead one.

3. **Runtime topology change in a static-topology system.** EPLB was designed with immutable `epSize` and `epRank` — the entire data structure (CPU placement arrays, GPU routing tables, shared memory layout) assumes a fixed number of ranks. Extending this for dynamic topology changes while the system is actively serving (concurrent worker threads, in-flight weight migrations, per-layer state machines) is a qualitatively different design problem than what EPLB was built for.

4. **Cross-layer design spanning 5 abstraction levels.** Changes must be consistent across GPU kernels → communication backends → load balancer → executor → serving layer. An error detected in a CUDA kernel must propagate through Python communication wrappers, trigger EPLB reconfiguration in C++, update the executor's health status, and surface in HTTP health check responses — all within seconds, all correctly ordered.

---

## Table of Contents

1. [Background and Motivation](01-background.md) — Why WideEP fault tolerance is critical now
2. [Current State Analysis](02-current-state.md) — WideEP architecture, communication backends, and failure modes
3. [Competitive Landscape](03-competitive-landscape.md) — SGLang Elastic EP, vLLM RFC, Ray 2.55, DeepSeek production
4. [Design: Two-Phase Recovery](04-two-phase-recovery.md) — Phase 1 (immediate survival) and Phase 2 (full restoration)
5. [Design: Rank Masking in Communication](05-rank-masking.md) — AlltoAll modifications for NVLink, DeepEP, and fallback backends
6. [Design: EPLB Topology Adaptation](06-eplb-adaptation.md) — Expert redistribution, routing table updates, weight migration
7. [Design: Failure Detection and Classification](07-failure-detection.md) — Extending PR #12718 for per-EP-rank health tracking
8. [Integration with MX-GMS](08-mx-gms-integration.md) — How the three workstreams align, parallelize, and benefit each other
9. [Implementation Plan](09-implementation-plan.md) — Phased rollout, dependencies, and milestones
10. [Risks and Open Questions](10-risks.md) — Technical risks, feasibility concerns, and design trade-offs

---

## Phase Priorities

| Phase | Priority | Timeline | Rationale |
|:------|:---------|:---------|:----------|
| **Phase 1 MVP (v0)** | **P0** | **6-8 weeks** | **Single-failure survival on NVLinkOneSided (primary NVL72 backend); eliminates 7-8 min downtime for the dominant failure mode. Scope cuts: no NVLinkTwoSided, no full EPLB reconfigure, no multi-failure. See [§09 Phase 1 MVP](09-implementation-plan.md#phase-1-mvp-v0-vs-full-scope).** |
| Phase 1 full (v1) | P0 | ~8-12 weeks after MVP | Broadens to all NVLink backends, adds full EPLB reconfigure with weight migration, multi-failure consensus |
| **Phase 1-DS: Disagg FT** | **P1** | **~4-6 weeks, parallelizable with Phase 1 v1** | **Extends FT to disaggregated serving. Starts after Phase 1 MVP; does not block v1.** |
| Phase 2: Full Restoration | P1 | 3-6 months | Process group reconstruction; restores full N-rank capacity; benefits from MX-GMS for fast recovery |
| Phase 3: Proactive Resilience | P2 | 6-12 months | Predictive failure detection, preemptive expert migration |

## Related Work

| Workstream | Status | Relationship |
|:-----------|:-------|:-------------|
| [PR #12718: Fatal Error Detection](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | Open (in review) | **Foundation** — Provides error classification, error budget, and fatal error propagation that this design extends |
| [MX+GMS+TRT-LLM Integration](https://docs.google.com/document/d/14SZmmFcoakgIx2OC4dt8pWcHU14PDTN9KlAKqLoZ15s/edit?usp=sharing) | Design complete | **Acceleration** — GMS crash-resilient memory and shadow workers reduce Phase 2 recovery from minutes to sub-second |
| Wide-EP + EPLB Hardening | Tier 1 priority | **Prerequisite** — EPLB correctness and reliability are assumed by this design |
