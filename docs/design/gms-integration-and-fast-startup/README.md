<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM GMS Integration and Fast Startup

**Status:** Draft — current implementation source of truth is §18; the
[combined MX/GMS prototype](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype) is
historical
**Created:** 2026-04-01
**Last Updated:** 2026-07-02

---

## Executive Summary

This proposal centers the TensorRT-LLM integration with GPU Memory Service and covers the supporting fast-start
mechanisms needed to make it operational:

- **GPU Memory Service (GMS)**: Out-of-process GPU memory management for zero-copy sharing and crash-resilient failover within nodes
- **ModelExpress (MX)**: An optional GPU-to-GPU weight source via NIXL/RDMA for fast cold-start across nodes
- **Compile and graph reuse**: Process-local and serialized startup state needed to keep promotion off the warmup path

The integration targets three critical production pain points simultaneously: slow cold-start through optional MX P2P,
crash recovery through parked GMS shadow workers that preserve pre-captured graphs and caches, and zero-downtime
operations through promotion and shadow replenishment. Serialized compile caches support cold and replacement-shadow
startup; they are not work to perform on the promotion path.

> [§18 GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-pr-plan.md) is the current source of
> truth for the GMS lifecycle, ownership boundary, and implementation order. It supersedes older RO-to-RW promotion
> and executor-state assumptions retained in earlier sections for historical context.

**Measured baselines (v3, current code `upstream/main @ 4a848ccce`):** Qwen 72B TP=8 takes **306s** (S2 NFS cold) / **75s** (S3 warm cache); DeepSeek 70B TP=8 takes **390s** / **78s**. Warmup is a **~43s floor** on v3 code (up from ~16s in v2 due to [PR #12407](https://github.com/NVIDIA/TensorRT-LLM/pull/12407)'s new general warmup pass). A warm promotion must therefore preserve the shadow's pre-captured graphs and caches; [§07 Tiered Compile Cache](07-compile-cache.md) explores serialization for cold initialization and replacement-shadow startup. See [§11 Results & Analysis](11-results-analysis.md) for the full dataset.

---

## Table of Contents

### Part I: Overview & Motivation

1. [Background and Motivation](01-background.md) — What MX and GMS are, why they matter, current state analysis
2. [Problem Statement and Goals](02-problem-and-goals.md) — Pain points, target use cases, goals and non-goals

### Part II: Core Design & Implementation

3. [Architecture](03-architecture.md) — High-level design, data flows, component responsibilities
4. [Implementation & API Design](04-implementation-plan.md) — Two-axis integration model, weight loading pipeline (TP/PP/EP), MX/GMS/Combined implementation, configuration
5. [Challenges and Mitigations](05-challenges.md) — FP8 compatibility, non-contiguous tensors, TP/PP/EP rank matching, CUDA VMM, module path resolution

### Part III: Extensions

6. [Executor Integration and Failover](06-executor-failover.md) — Earlier lifecycle exploration; §18 supersedes its executor-state and RO-to-RW promotion assumptions
7. [Tiered Compile Cache](07-compile-cache.md) — GMS + disk compile/autotuner cache options for cold and replacement-shadow startup
8. [Disaggregated Serving Interaction](08-disagg-interaction.md) — How MX/GMS interact with P/D separation
9. [KV Cache Extension Path](09-kv-cache-extension.md) — KV persistence remains a KVBM concern; §18 separately defines scratch/stable-VA KV backing for warm shadows

### Part IV: Performance & Benchmarks

10. [Methodology & Test Plan](10-methodology.md) — **Profiler implemented.** Framework, scenarios, test matrix, statistical protocol
11. [Performance Results & Analysis](11-results-analysis.md) — Target metrics, **v3 benchmark results (42 profiles)**, v2 reference (62 profiles), analysis, MX+GMS impact projections

### Part V: Strategy & Risk

12. [Risk Assessment](12-risks.md) — Technical risks, strategic concerns, GMS API stability, vLLM comparison
13. [Strategic Alignment](13-strategic-alignment.md) — How this fits into the TRT-LLM opportunity roadmap

### Part VI: Open Questions & Working Plans

14. [Open Questions & Discussion](14-open-questions.md) — Performance follow-ups, compile cache design, API stability, operational questions, deferred items
15. [Prototype Validation Plan](15-prototype-validation-plan.md) — Historical validation plan for the closed, unmerged PR #13045 prototype
16. [Staged Post-Load Hooks](16-staged-post-load-hooks.md) — Holistic fix for the conflated `post_load_weights()` semantics surfaced by PR #13926 (GMS RO ordering) and PR #14151 (MX publish-pre vs publish-post-transform). Decomposes into `setup_aliases` / `transform_weights` / `cache_derived_state` stages; tiny prep PR scope + family-PR migration sequence.
17. [Snapshot Integration Assessment](17-snapshot-assessment.md) — Assesses how Dynamo Snapshot, MX, and GMS fit together for TRT-LLM fast startup, including standalone `trtllm-serve` ownership versus Dynamo orchestration.
18. [GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-pr-plan.md) — Clarifies the existing SourceIdentity versus the missing GMS transport, explains PR #15432's staged-layout implications, and records the sleep/wake contract, ownership boundary, dependency-ordered PR stack, and acceptance gates.

---

## Current Delivery Priorities

| Gate | Priority | Outcome |
|:--|:--|:--|
| M0 — Native RW/RO weight sharing | **P0** | Fix the real-package API, persist/retrieve the existing identity and committed-layout metadata, and qualify an explicit model/protocol pair before failover work is advertised. |
| M1 — Functional warm promotion | **P0/P1** | Extend existing sleep/wake, multi-rank control, admission safety, graph readiness, and Dynamo election. |
| M2 — Replenishable redundancy | **P1** | Add scratch-backed stable-VA KV and provision a replacement shadow beside the live primary. |
| M3 — Supported SLO/product path | **P1** | Optimize GMS remap, meet the p95 target, and publish supported packages, diagnostics, and recipes. |

See [§18](18-gms-integration-gaps-and-pr-plan.md#delivery-gates) for the exact cumulative gates and dependency-ordered
PR stack. MX remains an optional weight-source track rather than a prerequisite for the GMS lifecycle.
