# 5.1 Critical Feature Gaps vs. Mainstream Frameworks

[< Back to Overview](README.md) | [Next: Bugs and Issues >](05-02-bugs-and-issues.md)

These are features where competitors (vLLM, SGLang, LMCache, NVIDIA Dynamo) have working implementations that TRT-LLM lacks, creating real risk of user attrition.

*[Updated 2026-04-29: re-classified gaps using closed/widened/new tags below; bumped competitor versions in citations.]*

## Status Summary (2026-04-29)

| Gap | Status since 2026-04-06 |
|:----|:------------------------|
| **Block reuse + overlap scheduler** (was a known internal gap) | ✓ **Closed** — combined via #12816 (`[TRTLLM-10939]`) |
| **First-class LMCache integration** | ✓ **Closed** — `lmcache` + `kvbm` shorthand connectors (#12626) |
| **Production Prometheus observability** | ✓ **Closed** — iteration stats + token counters + phase histograms (#12545) |
| **Mamba/hybrid prefix caching** | ✓ **Closed (partial)** — Qwen3.5, Nemotron Super V3 (#12185) |
| **LoRA + spec-dec combination** | ✓ **Closed** — generic spec-dec (#12661), EAGLE3 specifically (#13005) |
| **Tool-parser breadth** | ▲ **Improved** — GLM-4.7/5 (#13150) + qwen3, qwen3_coder, deepseekv31, deepseekv32, minimax_m2 |
| **Multi-vendor GPU** | ▼ **Widened** — TPU v7 GA, MI355X MLPerf 6.0 ≥1M tok/s |
| **Model catalog velocity** | ▼ **Widened** — vLLM v0.20 added DeepSeek V4, Hunyuan v3, Granite 4.1 Vision |
| **Low-bit KV cache** *[New]* | New gap — vLLM TurboQuant 2-bit KV (4× capacity) |
| **End-to-end IR** *[New]* | New gap — vLLM IR foundation + Model Runner V2 |
| **MLA prefill kernel** *[New]* | New gap — vLLM v0.20 default = FA4 for MLA prefill (head-dim 512, paged-KV, SM90+) |
| **Elastic fault tolerance (in-place failover)** | ▬ **Stable** — Dynamo Snapshot offsets restart speed but not in-place failover |
| **TTFT competitiveness (single GPU)** | ▬ **Stable** — needs re-benchmark vs. vLLM v0.20 + Model Runner V2 |
| **Multi-model serving** | ▬ **Stable** — partially offset by Dynamo orchestration |

---

## 1.1 Multi-Vendor GPU Support

**Gap:** TRT-LLM only runs on NVIDIA GPUs. vLLM and SGLang both support CUDA, ROCm (AMD), and TPU (Google). SGLang additionally supports Apple Silicon via MLX. *[Updated 2026-04-29: gap widened — Google TPU v7 (Ironwood) GA on March 31, 2026; AMD MI355X surpassed 1M tokens/sec in MLPerf Inference 6.0 (April 2026), with strong vLLM integration on DeepSeek-R1, GPT-OSS-120B, Qwen3-235B, Llama-3.3-70B.]*

**Impact:** Enterprise customers with multi-cloud strategies (Azure with AMD MI355X, GCP with TPU v7) cannot standardize on TRT-LLM. Cloud providers building managed inference services prefer vendor-neutral engines deployable across their fleet.

**What competitors offer:**
- **vLLM v0.20:** CUDA 13.0.2 default, ROCm (MI355X), TPU v6e/v7; B300/GB300 SM10.3 tuned allreduce; PyTorch 2.11
- **SGLang:** CUDA, ROCm, TPU, native MLX backend for Apple Silicon
- **Mitigation:** AutoDeploy's `torch.export` path could theoretically target non-NVIDIA backends; standalone-package work (#13155) makes this more credible than 6 months ago. The pragmatic strategy is positioning TRT-LLM as the "NVIDIA-optimized backend" behind vendor-neutral frontends (NVIDIA Dynamo v1.0 already does this for SGLang/vLLM/TRT-LLM, Triton Inference Server).

---

## 1.2 Model Catalog Breadth and Onboarding Velocity

**Gap:** vLLM supports ~2x more model architectures (~100+ vs ~50+), and new models typically get vLLM support first. *[Updated 2026-04-29: vLLM v0.20 added DeepSeek V4 initial support, Hunyuan v3 (Hy3) preview, Granite 4.1 Vision, EXAONE-4.5, BharatGen Param2MoE, Phi-4-reasoning-vision-15B in a single release.]*

**Impact:** Every unsupported model is a potential user lost. The gap is self-reinforcing: more models attract more users, which attract more community contributors, which add more models.

**What competitors offer:**
- **vLLM v0.20:** ~100+ architectures; HuggingFace Transformers v5 compatibility; new in v0.20 — DeepSeek V4, Hunyuan v3, Granite 4.1 Vision, EXAONE-4.5, BharatGen Param2MoE, Phi-4-reasoning-vision-15B
- **SGLang v0.5.10:** Stronger Transformers modeling backend with TP, PP, MoE, VLM, torch.compile; rapid community adoption

**TRT-LLM progress in this window:** AutoDeploy onboarded DeepSeek-R1 (#12601), Gemma-4 (#12710), Gemma 4-31B incl. NVFP4 (#12866), MiniMax-M2.7 custom model (#12963). Main path also added new tool parsers for several agentic-flavored models.

**Opportunity:** AutoDeploy as the primary onboarding path — *[Updated 2026-04]* now standalone-package-ready (#13155, #13418) — automatically convert HuggingFace models without manual model class implementation. Streamlined contribution workflow with automated testing.

---

## 1.3 Elastic Fault Tolerance

**Gap:** TRT-LLM has no mechanism for graceful GPU failure recovery. When a GPU fails, the entire serving instance crashes.

**Impact:** Production deployments at scale (hundreds of GPUs) experience regular hardware failures. Without elastic recovery, every GPU failure causes full-instance downtime and loss of all in-flight requests.

**What competitors offer:**
- **SGLang:** Elastic EP for partial failure tolerance — when a GPU fails, experts are redistributed to surviving GPUs without restart. This is a production-critical capability for large MoE deployments.
- **vLLM:** Elastic EP with NIXL for dynamic GPU scaling (add/remove GPUs without restart).

**Opportunity:** Implement elastic expert redistribution for EP workloads. Extend to TP/PP with hot-standby replicas.

---

## 1.4 TTFT Competitiveness

**Gap:** vLLM achieves ~35% lower TTFT on single-GPU benchmarks (~123ms vs. ~194ms).

**Impact:** For interactive/chat workloads — the highest-value inference use case — TTFT is the most user-perceptible metric. A 35% deficit drives framework selection decisions regardless of throughput advantages.

**Root causes:**
- Two-phase C++ scheduler overhead vs. vLLM V1's simpler unified scheduler
- Overlap scheduler introduces one extra decoding step
- CUDA graph lookup overhead for initial prefill
- KV cache block allocation path with nanobind crossing overhead

**Opportunity areas:**
- Prefill-specific CUDA graphs
- Smarter first-request scheduling (bypass two-phase overhead)
- FlashAttention 4 integration (vLLM already integrated)
- Async tokenization (vLLM V1's approach)
- Systematic end-to-end TTFT profiling

---

## 1.5 Quantized LoRA and LoRA Feature Completeness

**Gap:** vLLM supports quantized LoRA (QLoRA direct loading). SGLang supports LoRA for MoE layers with JIT alignment kernels. TRT-LLM's LoRA is untested with EP, Helix, ADP, disaggregated serving. *[Updated 2026-04-29 — partially closed]* **LoRA + speculative decoding** now works (generic path #12661, EAGLE3 specifically #13005); Qwen3 LoRA fixed (#12785); Nemotron NAS LoRA over-allocation partially fixed (#12817).

**Impact:** LoRA is the primary mechanism for multi-tenant model customization in production. Incomplete LoRA support blocks enterprise deployments that need per-customer model variants with advanced infrastructure features.

**Remaining opportunity:** Systematic LoRA compatibility testing across all feature combinations. Prioritize LoRA + EP (MoE with per-user adaptations) and LoRA + disaggregated serving (multi-tenant disaggregated deployments). *[Updated 2026-04-29]* The spec-dec combo is now off the list — focus shifts to LoRA + EP/Wide-EP + disagg.

---

## 1.6 Structured Generation Performance

**Gap:** SGLang achieves up to 5x throughput for structured generation via its DSL and RadixAttention.

**Impact:** Agentic workflows (tool use, function calling, JSON output) are growing rapidly. Structured output is becoming a table-stakes feature for production deployments.

**Opportunity:**
- Prefix-aware scheduling for structured output prefixes
- Grammar-aware KV cache reuse
- Batched constraint checking to amortize grammar engine overhead
- Constrained draft generation in speculative decoding

---

## 1.7 Multi-Model Serving

**Gap:** vLLM V1 has native multi-model serving. TRT-LLM requires separate instances per model.

**Impact:** Production AI platforms serve hundreds of model variants from the same GPU fleet. Single-model instances waste GPU resources on underutilized models.

**Opportunity:** Model multiplexing within a single executor, with shared GPU memory management and LoRA hot-swapping.

---

## 1.8 API Compatibility Breadth

**Gap:** vLLM now supports both OpenAI and Anthropic API compatibility (thinking blocks, count_tokens, Responses API with streaming tool calls). TRT-LLM only supports OpenAI-compatible API.

**Impact:** Applications built against the Anthropic API cannot seamlessly switch to TRT-LLM for self-hosted inference.

**Opportunity:** Add Anthropic API compatibility endpoint alongside existing OpenAI endpoint.

---

## 1.9 Low-Bit KV Cache Compression *[New 2026-04-29]*

**Gap:** vLLM v0.20 ships **TurboQuant 2-bit KV cache compression** with 4× capacity. TRT-LLM has FP8 / INT4 quantization paths and FP4 residual quant (#13117), but no equivalent 2-bit KV-cache option.

**Impact:** Long-context and high-concurrency workloads — exactly the workloads where TRT-LLM's prefix caching, ADP routing, and Wide-EP shine — are KV-cache-capacity-bound. A 4× capacity multiplier is a category-defining feature.

**Opportunity:** Add 2-bit KV-cache compression that interoperates with V1 + V2 managers, the radix-tree, and disagg KV transfer. Especially impactful for multi-turn agentic workloads where the KV cache dominates.

---

## 1.10 End-to-End Inference IR + Runner V2 *[New 2026-04-29]*

**Gap:** vLLM v0.20 introduces a **vLLM IR foundation** (with a first `rms_norm` op + testing infrastructure) and a **Model Runner V2** that includes Eagle prefill full-CUDA-graph and multiple prompt-logprobs.

**Impact:** vLLM is investing in a structural reorganization that promises faster optimization velocity — every per-op kernel improvement gets to compose with full-graph capture and the new runner. TRT-LLM's competing assets (CuTE DSL kernels, custom scheduling, AutoDeploy graph transforms) are not unified under a single IR.

**Opportunity:** Either (a) align AutoDeploy more aggressively as the unifying IR (with CuTE DSL kernels as the lowering target), or (b) embrace torch.compile / Inductor as the substrate for the PyTorch backend. Either way the question of "what is TRT-LLM's IR strategy" needs a public answer.

---

## 1.11 MLA Prefill Kernel Defaults *[New 2026-04-29]*

**Gap:** vLLM v0.20 re-enables **FlashAttention 4 as default MLA prefill backend** with head-dim 512 + paged-KV on SM90+. TRT-LLM has trtllm-gen, FlashInfer, and FA backends but has not advertised an equivalent MLA prefill default.

**Impact:** DeepSeek-V3/R1, Qwen-Next, and other MLA-class models are the marquee MoE workloads. Default kernel choice is heavily TTFT- and throughput-relevant.

**Opportunity:** Audit current MLA prefill backend selection on Hopper/Blackwell; document the recommended path; benchmark vs. vLLM v0.20 + FA4.
