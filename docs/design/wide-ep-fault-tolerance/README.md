# WideEP Fault Tolerance for TensorRT-LLM

**Status:** Draft
**Created:** 2026-04-10
**Last Updated:** 2026-04-22

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

## Terminology

This doc uses specific terms consistently. The same concept sometimes appears under different labels in prior drafts or in adjacent work (MX-GMS design, SGLang papers); the definitions below are what this doc means.

| Term | Definition |
|:---|:---|
| **WideEP** | Wide Expert Parallelism: MoE expert-parallel distribution across ≥32 GPUs (vs. standard EP within a single 8-GPU node). |
| **EP group** | The set of ranks participating in a single AlltoAll collective. For DeepSeek-V3 on NVL72, `ep=72`. |
| **EPLB** | **Expert-Parallel Load Balancer** — an *existing* TRT-LLM component (`cpp/tensorrt_llm/runtime/moeLoadBalancer/`, `tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py`). Owns the (expert → rank, slot) placement table (`MoePlacementInfo`), updates it at iteration boundaries, and supports redundant expert replicas. **In scope:** extend EPLB with an out-of-band "emergency reconfigure" trigger and a mask-only fast path. **Out of scope:** rewriting EPLB's routing algorithm, changing its balancing heuristic, or replacing it. |
| **Rank masking** | *Kernel-level* behavior: the AlltoAll kernel reads a bitmask and skips dead peers in its send/poll loops. Implemented in §05. |
| **Slot remap** | *EPLB-level* operation: rewrite `MoePlacementInfo` so the dead rank's slots point to surviving replicas. No H2D weight copy. Implemented in §06. |
| **Emergency reconfigure** | The combined MVP recovery action = rank masking + slot remap. Triggered by failure detection; completes at the next iteration boundary. Target <10ms end-to-end. |
| **Weight migration** | *v1-only* operation: H2D copy of expert weights into a new slot. Required when an expert loses all replicas (replication=1 case). The MVP avoids this by requiring replication ≥ 2. |
| **MVP (v0)** | The first shipping milestone. Single failure, NVLinkOneSided backend only, slot-remap recovery (no weight move). 6-7 weeks with AI coding-agent assistance (8-10 weeks baseline). See [§09](09-implementation-plan.md) for the assumption. |
| **v1** | Full Phase 1 scope: all NVLink backends, weight migration, multi-failure consensus. Ships after MVP. |
| **Phase 2** | Full restoration: bring up a replacement rank and rebuild the process group. Optional — Phase 1 is sufficient for continued serving. |
| **Work tracks 1a/1b/1c/1d** | Parallel engineering tracks inside Phase 1 (see §09). Not time-slices. |
| **Emergency mode / mask-only mode** | Older names for "emergency reconfigure." The API surface in code is `reconfigure_mask_only()`; the prose name is "emergency reconfigure." |

## Why This Is Technically Hard

This is not routine integration work. The design tackles several problems that are individually challenging and collectively require cross-layer systems expertise:

1. **GPU kernel-level synchronization modification.** TRT-LLM's NVLinkOneSided AlltoAll kernel spins on PTX-level completion-flag polling with no host-side abort hook. Adding rank masking requires modifying the CUDA kernel itself — a different class of work from SGLang's and vLLM's API-level integration against third-party libraries (Mooncake, DeepEP). See [§03 "Why kernel-level, and not API-level"](03-competitive-landscape.md#why-kernel-level-and-not-api-level-like-sglang--vllm) for the full argument.

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
| **Phase 1 MVP (v0)** | **P0** | **6-7 weeks** (AI-assisted) | **Single-failure survival on NVLinkOneSided (primary NVL72 backend); eliminates 7-8 min downtime for the dominant failure mode. Scope cuts: no NVLinkTwoSided, no full EPLB reconfigure, no multi-failure. Estimate history: initial 6-8 weeks → post-April-review baseline 8-10 weeks (added scope: kMaxRanks bump, NCCL FT wiring, MPI FT subcomm, fault-injection harness) → AI coding-agent assistance absorbs the added scope back to 6-7 weeks. See [§09 Phase 1 MVP](09-implementation-plan.md#phase-1-mvp-v0-vs-full-scope).** |
| Phase 1 full (v1) | P0 | ~6-9 weeks after MVP (AI-assisted) | Broadens to all NVLink backends, adds full EPLB reconfigure with weight migration, multi-failure consensus |
| **Phase 1-DS: Disagg FT** | **P1** | **~3-4 weeks, parallelizable with Phase 1 v1** | **Extends FT to disaggregated serving. Starts after Phase 1 MVP; does not block v1.** |
| Phase 2: Full Restoration | P1 | 2.5-3.5 months (AI-assisted) | Process group reconstruction; restores full N-rank capacity; benefits from MX-GMS for fast recovery |
| Phase 3: Proactive Resilience | P2 | 4-6 months (AI-assisted) | Predictive failure detection, preemptive expert migration |

> **All timelines above assume engineers work with AI coding-agent assistance that reduces coding and review time by ~30-40% on S/M-size PRs and ~20-25% on design-heavy L-size PRs. Without that assistance, apply ~1.3× to every figure. See [§09 "How to Read This Plan"](09-implementation-plan.md#how-to-read-this-plan) for the rationale.

## Related Work

| Workstream | Status | Relationship |
|:-----------|:-------|:-------------|
| [PR #12718: Fatal Error Detection](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | Open (in review) | **Foundation** — Provides error classification, error budget, and fatal error propagation that this design extends |
| [MX+GMS+TRT-LLM Integration](https://docs.google.com/document/d/14SZmmFcoakgIx2OC4dt8pWcHU14PDTN9KlAKqLoZ15s/edit?usp=sharing) | Design complete | **Acceleration** — GMS crash-resilient memory and shadow workers reduce Phase 2 recovery from minutes to sub-second |
| Wide-EP + EPLB Hardening | Tier 1 priority | **Prerequisite** — EPLB correctness and reliability are assumed by this design |
