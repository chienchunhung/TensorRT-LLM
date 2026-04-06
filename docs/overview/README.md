# TensorRT-LLM Architecture & Codebase Learning Overview

**Scope:** Deep-dive learning guide covering TensorRT-LLM's architecture, key features, end-to-end user journey, competitive landscape, and future development opportunities. Includes code references, design rationale, framework comparisons, and Mermaid diagrams.

**Last updated:** April 2026 — reflects TensorRT-LLM v1.3.0 (main branch), vLLM v0.19.0, SGLang v0.5.10, LMCache v0.4.2.

---

## Table of Contents

### 1. [High-Level Architecture](01-high-level-architecture.md)
Backend overview (PyTorch, AutoDeploy, TensorRT), architecture diagram, request flow, and key files reference.

### 2. Key Features Deep-Dive
- [2.1 In-Flight Batching (IFB)](02-01-in-flight-batching.md) — Continuous batching with two-phase scheduling
- [2.2 Overlap Scheduler](02-02-overlap-scheduler.md) — CPU/GPU pipelined execution with early-exit optimization
- [2.3 KV Cache Manager V1 & V2](02-03-kv-cache-manager.md) — Block-based cache with radix tree, prioritized LRU, multi-tier storage
- [2.4 Block Reuse (Prefix Caching)](02-04-block-reuse.md) — Cross-request KV cache sharing via radix tree matching
- [2.5 Disaggregated Serving](02-05-disaggregated-serving.md) — Prefill/decode separation with NIXL/UCX/Mooncake KV transfer
- [2.6 Speculative Decoding](02-06-speculative-decoding.md) — 7 algorithms (EAGLE3, MTP, NGram, PARD, SA, Draft/Target) + SA hybrids
- [2.7 Parallelism Strategies](02-07-parallelism-strategies.md) — TP, PP, EP, ADP, CP, Wide-EP, DWDP
- [2.8 Other Notable Features](02-08-other-features.md) — CUDA graphs, chunked prefill, guided decoding, LoRA, multimodal, visual generation, quantization

### 3. [End-to-End User Journey (PyTorch Backend)](03-user-journey.md)
Launch & initialization, model loading, request handling, failover & fault tolerance, auto-scaling.

### 4. [Framework Comparison](04-framework-comparison.md)
Architecture comparison, feature matrix (TRT-LLM vs. vLLM vs. SGLang vs. LMCache), and performance positioning.

### 5. Future Development Opportunities
- [5.1 Critical Feature Gaps vs. Mainstream Frameworks](05-01-feature-gaps.md) — Multi-vendor GPU, model catalog, elastic fault tolerance, TTFT, LoRA, structured generation, multi-model serving, API compatibility
- [5.2 Critical Bugs and Architectural Issues](05-02-bugs-and-issues.md) — Disagg reliability, executor complexity, KV cache V1/V2 divergence, feature combination gaps, CUDA crashes, OOM, FP8 fragility
- [5.3 Innovative and Futuristic Features](05-03-innovative-features.md) — Multi-modal platform, agentic workflows, GPU+LPU hybrid, CXL memory pooling, KVaaS, sparse attention, inference-time compute, privacy-preserving inference

### 6. [Strategic Prioritization](06-strategic-prioritization.md)
Investment priority matrix, prioritized roadmap (Tier 1-4), and where TRT-LLM should win.
