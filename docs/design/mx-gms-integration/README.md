# TensorRT-LLM Integration with ModelExpress and GPU Memory Service

**Status:** Draft (Revised) — [Prototype available](https://github.com/chienchunhung/TensorRT-LLM/tree/dynamo-integration-prototype)
**Created:** 2026-04-01
**Last Updated:** 2026-04-14

---

## Executive Summary

This proposal integrates TensorRT-LLM with two complementary systems from the Dynamo ecosystem:

- **ModelExpress (MX)**: GPU-to-GPU model weight streaming via NIXL/RDMA for fast cold-start across nodes
- **GPU Memory Service (GMS)**: Out-of-process GPU memory management for zero-copy sharing and crash-resilient failover within nodes

The integration targets three critical production pain points simultaneously: slow cold-start (minutes to seconds), memory waste from weight duplication (Nx to 1x), and crash recovery (full reload to <5s failover). vLLM already ships `--load-format mx` — this is a competitive catch-up for the MX integration and a differentiation opportunity for GMS and combined modes.

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

6. [Executor Integration and Failover](06-executor-failover.md) — Shadow failover mechanics, sleep/wake mapping, in-flight request handling
7. [Disaggregated Serving Interaction](07-disagg-interaction.md) — How MX/GMS interact with P/D separation
8. [KV Cache Extension Path](08-kv-cache-extension.md) — Future extension to KV cache persistence via GMS/KVBM

### Part IV: Performance & Benchmarks

9. [Startup Profiling Framework](09-startup-profiling.md) — **Implemented.** Hierarchical profiler, instrumentation, how to run, schema reference
10. [Performance Expectations and Benchmark Plan](10-performance.md) — Target metrics, test matrix, **v2 benchmark results (62 profiles)**, analysis, MX+GMS impact projections

### Part V: Strategy & Risk

11. [Risk Assessment](11-risks.md) — Technical risks, strategic concerns, GMS API stability, vLLM comparison
12. [Strategic Alignment](12-strategic-alignment.md) — How this fits into the TRT-LLM opportunity roadmap

---

## Priority Classification

| Phase | Priority | Timeline | Rationale |
|:------|:---------|:---------|:----------|
| MX Integration (Part II) | **P1 (Tier 1)** | 0-2 months | vLLM already has `--load-format mx`; competitive catch-up |
| GMS Integration (Part II) | **P2 (Tier 2)** | 2-4 months | Enables crash resilience and memory sharing; differentiation |
| Extensions (Part III) | **P2 (Tier 2)** | 4-6 months | Shadow failover, disagg interaction, KV cache extension path |
