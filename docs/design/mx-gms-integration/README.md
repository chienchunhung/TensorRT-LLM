# TensorRT-LLM Integration with ModelExpress and GPU Memory Service

**Status:** Draft (Revised) — [Prototype available](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype)
**Created:** 2026-04-01
**Last Updated:** 2026-06-26

---

## Executive Summary

This proposal integrates TensorRT-LLM with two complementary systems from the Dynamo ecosystem:

- **ModelExpress (MX)**: GPU-to-GPU model weight streaming via NIXL/RDMA for fast cold-start across nodes
- **GPU Memory Service (GMS)**: Out-of-process GPU memory management for zero-copy sharing and crash-resilient failover within nodes

The integration targets three critical production pain points simultaneously: slow cold-start (minutes to seconds via MX P2P), crash recovery (full reload to <5s failover via GMS shadow workers + tiered compile cache), and zero-downtime operations (rolling updates, elastic scaling). vLLM already ships `--load-format mx` — this is a competitive catch-up for the MX integration and a differentiation opportunity for GMS and combined modes.

**Measured baselines (v3, current code `upstream/main @ 4a848ccce`):** Qwen 72B TP=8 takes **306s** (S2 NFS cold) / **75s** (S3 warm cache); DeepSeek 70B TP=8 takes **390s** / **78s**. Warmup is a **~43s floor** on v3 code (up from ~16s in v2 due to [PR #12407](https://github.com/NVIDIA/TensorRT-LLM/pull/12407)'s new general warmup pass) — hitting the <5s shadow failover target therefore requires a warm compile cache (see [§07 Tiered Compile Cache](07-compile-cache.md)). See [§11 Results & Analysis](11-results-analysis.md) for the full dataset.

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

6. [Executor Integration and Failover](06-executor-failover.md) — Shadow failover mechanics, sleep/wake mapping, in-flight request handling, **restart-after-death failover + self-managed deployment recipe (no Dynamo required)**
7. [Tiered Compile Cache](07-compile-cache.md) — GMS + disk tiered compile/autotuner cache to close the shadow activation warmup gap
8. [Disaggregated Serving Interaction](08-disagg-interaction.md) — How MX/GMS interact with P/D separation
9. [KV Cache Extension Path](09-kv-cache-extension.md) — Why KV cache is out of GMS's scope; deferred to KVBM via the KV Cache Connector API

### Part IV: Performance & Benchmarks

10. [Methodology & Test Plan](10-methodology.md) — **Profiler implemented.** Framework, scenarios, test matrix, statistical protocol
11. [Performance Results & Analysis](11-results-analysis.md) — Target metrics, **v3 benchmark results (42 profiles)**, v2 reference (62 profiles), analysis, MX+GMS impact projections

### Part V: Strategy & Risk

12. [Risk Assessment](12-risks.md) — Technical risks, strategic concerns, GMS API stability, vLLM comparison
13. [Strategic Alignment](13-strategic-alignment.md) — How this fits into the TRT-LLM opportunity roadmap

### Part VI: Open Questions & Working Plans

14. [Open Questions & Discussion](14-open-questions.md) — Performance follow-ups, compile cache design, API stability, operational questions, deferred items
15. [Prototype Validation Plan](15-prototype-validation-plan.md) — Strategy for validating PR #13045 against the §11 baseline (working plan; results will fold back into §11)
16. [Staged Post-Load Hooks](16-staged-post-load-hooks.md) — Holistic fix for the conflated `post_load_weights()` semantics surfaced by PR #13926 (GMS RO ordering) and PR #14151 (MX publish-pre vs publish-post-transform). Decomposes into `setup_aliases` / `transform_weights` / `cache_derived_state` stages; tiny prep PR scope + family-PR migration sequence.
17. [Snapshot Integration Assessment](17-snapshot-assessment.md) — Assesses how Dynamo Snapshot, MX, and GMS fit together for TRT-LLM fast startup, including standalone `trtllm-serve` ownership versus Dynamo orchestration.
18. [Dynamo GMS Standalone Failover Gap Analysis](18-dynamo-pr11000-gaps.md) — Reviews ai-dynamo/dynamo PR #11000 against the TRT-LLM MX/GMS design and records packaging, launch, executor, KV/cache, and validation gaps.

---

## Priority Classification

| Phase | Priority | Timeline | Rationale |
|:------|:---------|:---------|:----------|
| MX Integration (Part II) | **P1 (Tier 1)** | 0-2 months | vLLM already has `--load-format mx`; competitive catch-up |
| GMS Integration (Part II) | **P2 (Tier 2)** | 2-4 months | Enables crash resilience, <5s shadow failover, and zero-downtime operations; differentiation |
| Extensions (Part III) | **P2 (Tier 2)** | 4-6 months | Shadow failover, compile cache, disagg interaction, KV cache extension path |
| Snapshot Compatibility (Part VI) | **P2 (Tier 2)** | parallel investigation | Defines Dynamo-agnostic TRT-LLM lifecycle hooks while keeping standalone MX/GMS fast-start first-class |
