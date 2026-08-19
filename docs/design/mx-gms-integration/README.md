<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM Integration with ModelExpress and GPU Memory Service

**Status:** Draft — §22 is the current GMS/Snapshot architecture and cross-repository delivery source of truth, §18
remains the detailed GMS V0/standalone implementation plan, and §21 is the current MX readiness and model-family
qualification source of truth. The
[combined MX/GMS prototype](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype) is
historical.
**Created:** 2026-04-01
**Last Updated:** 2026-08-19

---

## Executive Summary

This proposal covers the complementary TensorRT-LLM integrations with ModelExpress and GPU Memory Service, plus the
supporting fast-start mechanisms needed to make them operational:

- **GPU Memory Service (GMS)**: Out-of-process GPU memory management for zero-copy sharing and crash-resilient
  failover within nodes
- **ModelExpress (MX)**: An optional GPU-to-GPU weight source via NIXL/RDMA for fast cold-start across nodes
- **Run:ai Model Streamer**: An optional high-throughput source for first-replica SafeTensors loading from local or
  object storage
- **Compile and graph reuse**: Process-local and serialized startup state needed to keep promotion off the warmup path

The integration targets three critical production pain points: slow cold-start through optional MX P2P, standalone
weight reuse and optional live-shadow recovery through GMS V0, and initialized-engine restore through Snapshot plus
GMS V1. Serialized compile caches support cold and replacement startup; they are not work to perform on a preserved
engine's restore path.

> [§22 GMS and Snapshot Integration: Four-Lane Delivery Plan](22-gms-snapshot-four-lane-integration-plan.md) is the
> current source of truth for the GMS V0/V1 boundary, Snapshot readiness, restored-owner ownership, and overall
> delivery priority. [§18](18-gms-integration-gaps-and-concrete-pr-plan.md) remains the detailed V0/standalone loading
> and live-shadow plan.

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
9. [KV Cache Extension Path](09-kv-cache-extension.md) — KV persistence remains a KVBM concern; §18 defines the V0/live-shadow scratch design, while §22 records the unresolved bridge to the GMS V1 ephemeral-KV domain

### Part IV: Performance & Benchmarks

10. [Methodology & Test Plan](10-methodology.md) — **Profiler implemented.** Framework, scenarios, test matrix, statistical protocol
11. [Performance Results & Analysis](11-results-analysis.md) — Target metrics, **v3 benchmark results (42 profiles)**, v2 reference (62 profiles), analysis, MX+GMS impact projections

### Part V: Strategy & Risk

12. [Risk Assessment](12-risks.md) — Technical risks, strategic concerns, GMS API stability, vLLM comparison
13. [Strategic Alignment](13-strategic-alignment.md) — How this fits into the TRT-LLM opportunity roadmap

### Part VI: Assessments, Open Questions & Working Plans

14. [Open Questions & Discussion](14-open-questions.md) — Performance follow-ups, compile cache design, API stability, operational questions, deferred items
15. [Prototype Validation Plan](15-prototype-validation-plan.md) — Historical validation plan for the closed, unmerged PR #13045 prototype
16. [Staged Post-Load Hooks](16-staged-post-load-hooks.md) — Holistic fix for the conflated `post_load_weights()` semantics surfaced by PR #13926 (GMS RO ordering) and PR #14151 (MX publish-pre vs publish-post-transform). Decomposes into `setup_aliases` / `transform_weights` / `cache_derived_state` stages; tiny prep PR scope + family-PR migration sequence.
17. [Snapshot Integration Assessment](17-snapshot-assessment.md) — Historical component layering for Dynamo Snapshot,
    MX, GMS, and standalone `trtllm-serve`; §22 supersedes its GMS V1 and restored-owner architecture.
18. [GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-concrete-pr-plan.md) — Detailed
    GMS V0/standalone loading and live-shadow plan; its shared sleep/wake invariants feed the §22 Snapshot/V1 lanes.
19. [ModelStreamer and Weight-Loading Integration Assessment](19-model-streamer-weight-loading-assessment.md) —
    Assesses Run:ai Model Streamer against MX, GMS, GMS storage snapshots, Dynamo process Snapshot, and the TRT-LLM
    weight-loading proposals; defines the recommended ownership boundaries, shared contracts, and phased delivery plan.
20. [ModelExpress End-to-End Verification Plan](20-mx-e2e-verification-plan.md) — Agent-executable single-node runbook
    for validating a combined PR #15641/#16159 head, content-bound donor publication, GPU-to-GPU transfer, staged
    Llama reception, deterministic output equivalence, canonical-snapshot no-disk proof, local-server lifecycle,
    negative controls, and evidence collection.
21. [ModelExpress Readiness Gaps and Model-Family Expansion Plan](21-mx-readiness-gaps-and-model-family-plan.md) —
    Accounts for the in-flight standalone MX and ArtifactIdentity PRs, defines the claims that can be made after Llama
    qualification, and gives a dependency-ordered procedure for qualifying Qwen, DeepSeek, GLM, Kimi, and other
    families.
22. [GMS and Snapshot Integration: Four-Lane Delivery Plan](22-gms-snapshot-four-lane-integration-plan.md) — Separates
    V0/standalone GMS, shared Snapshot readiness, the TRT-LLM GMS V1 adapter, and restored-owner infrastructure;
    defines the Dynamo and native TRT-LLM user paths, ownership boundaries, gaps, and investment gates.

---

## Current Delivery Priorities

| Lane / gate | Priority | Outcome |
|:--|:--|:--|
| Lane 2 — Shared Snapshot readiness | **P0** | Finish engine-owned admission, all-rank quiescence, stable-VA communication-resource lifecycle, and public pause/resume readiness. |
| Lane 3 — TRT-LLM GMS V1 adapter | **P0 spike** | Prove dense TP1 weight/KV domains, exact-VA wake, and repeated Snapshot-plus-GMS restore without model reconstruction. |
| Lane 1 — GMS V0 / standalone | **P1 focused** | Finish exact API/identity and one qualified RW-to-RO path; do not expand fresh-process reconstruction without a declared product need. |
| Lane 4 — Restored-owner infrastructure | **External P0 dependency** | GMS/Snapshot/Dynamo restore the owner before the engine with generation binding, efficient artifact movement, supervision, and failure semantics. |
| Distributed/product expansion | **Gated** | Qualify same-node TP/EP/MoE, then cross-node and broader features only after TP1 V1 and restored-owner gates pass. |

See [§22](22-gms-snapshot-four-lane-integration-plan.md#prioritized-gap-and-delivery-matrix) for the cross-lane gaps and
investment gates. See [§18](18-gms-integration-gaps-and-concrete-pr-plan.md#delivery-gates) for the detailed V0/live-shadow
stack. MX remains an optional initial weight source rather than a GMS V1 restore prerequisite. ModelStreamer remains an
optional cold-storage fallback and first-writer seed.
