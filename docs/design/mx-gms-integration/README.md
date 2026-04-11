# TensorRT-LLM Integration with ModelExpress and GPU Memory Service

**Status:** Draft (Revised)
**Created:** 2026-04-01
**Last Updated:** 2026-04-08

---

## Executive Summary

This proposal integrates TensorRT-LLM with two complementary systems from the Dynamo ecosystem:

- **ModelExpress (MX)**: GPU-to-GPU model weight streaming via NIXL/RDMA for fast cold-start across nodes
- **GPU Memory Service (GMS)**: Out-of-process GPU memory management for zero-copy sharing and crash-resilient failover within nodes

The integration targets three critical production pain points simultaneously: slow cold-start (minutes to seconds), memory waste from weight duplication (Nx to 1x), and crash recovery (full reload to <5s failover). vLLM already ships `--load-format mx` — this is a competitive catch-up for Phase 1 and a differentiation opportunity for Phases 2-3.

**Recommended Approach:** Three-phase integration — MX first (competitive parity, 6-8 weeks), GMS second (differentiation, 6-8 weeks), combined + KV cache extension (full solution, 4-6 weeks).

---

## Table of Contents

1. [Background and Motivation](01-background.md) — What MX and GMS are, why they matter, current state analysis
2. [Problem Statement and Goals](02-problem-and-goals.md) — Pain points, target use cases, goals and non-goals
3. [Proposed Architecture](03-architecture.md) — High-level design, data flows, component responsibilities
4. [Implementation Plan](04-implementation-plan.md) — Three-phase approach with detailed deliverables and timelines
5. [API Design](05-api-design.md) — Two-axis integration model, MX checkpoint loader, GMS loading mode, configuration, library inventory
6. [Executor Integration and Failover](06-executor-failover.md) — Shadow failover mechanics, sleep/wake mapping, in-flight request handling
7. [KV Cache Extension Path](07-kv-cache-extension.md) — Future extension to KV cache persistence via GMS/KVBM
8. [Disaggregated Serving Interaction](08-disagg-interaction.md) — How MX/GMS interact with P/D separation
9. [Challenges and Mitigations](09-challenges.md) — FP8 compatibility, non-contiguous tensors, TP/PP/EP rank matching
10. [Startup Performance Profiling](10-startup-profiling.md) — **Implemented.** Hierarchical profiler, benchmark workflow, real DeepSeek results
11. [Performance Expectations and Benchmarks](11-performance.md) — Target metrics, benchmark plan, regression detection
12. [Risk Assessment](12-risks.md) — Technical risks, strategic concerns, GMS API stability, vLLM comparison
13. [Strategic Alignment](13-strategic-alignment.md) — How this fits into the TRT-LLM opportunity roadmap

---

## Priority Classification

Per the [TRT-LLM opportunity strategy](../../overview/06-strategic-prioritization.md):

| Phase | Priority | Timeline | Rationale |
|:------|:---------|:---------|:----------|
| Phase 1: MX | **P1 (Tier 1)** | 0-2 months | vLLM already has `--load-format mx`; competitive catch-up |
| Phase 2: GMS | **P2 (Tier 2)** | 2-4 months | Enables crash resilience and memory sharing; differentiation |
| Phase 3: Combined + KV | **P2 (Tier 2)** | 4-6 months | Full solution; enables elastic fault tolerance |
