# 5.1 Critical Feature Gaps vs. Mainstream Frameworks

[< Back to Overview](README.md) | [Next: Bugs and Issues >](05-02-bugs-and-issues.md)

These are features where competitors (vLLM, SGLang, LMCache) have working implementations that TRT-LLM lacks, creating real risk of user attrition.

---

## 1.1 Multi-Vendor GPU Support

**Gap:** TRT-LLM only runs on NVIDIA GPUs. vLLM and SGLang both support CUDA, ROCm (AMD), and TPU (Google). SGLang additionally supports Apple Silicon via MLX.

**Impact:** Enterprise customers with multi-cloud strategies (Azure with AMD MI300X, GCP with TPUs) cannot standardize on TRT-LLM. Cloud providers building managed inference services prefer vendor-neutral engines deployable across their fleet.

**What competitors offer:**
- **vLLM:** CUDA, ROCm, TPU; B300/GB300 SM10.3 tuned allreduce
- **SGLang:** CUDA, ROCm, TPU; native MLX backend for Apple Silicon
- **Mitigation:** AutoDeploy's `torch.export` path could theoretically target non-NVIDIA backends. The pragmatic strategy is positioning TRT-LLM as the "NVIDIA-optimized backend" behind vendor-neutral frontends (Dynamo, Triton Inference Server).

---

## 1.2 Model Catalog Breadth and Onboarding Velocity

**Gap:** vLLM supports ~2x more model architectures (~100+ vs ~50+), and new models typically get vLLM support first.

**Impact:** Every unsupported model is a potential user lost. The gap is self-reinforcing: more models attract more users, which attract more community contributors, which add more models.

**What competitors offer:**
- **vLLM:** ~100+ architectures; Transformers v5 compatibility; Gemma 4 (full MoE/multimodal/reasoning); ASR models (Cohere ASR, Granite Speech); GPU-less render serving for multimodal preprocessing
- **SGLang:** Stronger Transformers modeling backend with TP, PP, MoE, VLM, torch.compile; rapid community adoption

**Opportunity:** AutoDeploy as the primary onboarding path — automatically convert HuggingFace models without manual model class implementation. Streamlined contribution workflow with automated testing.

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

**Gap:** vLLM supports quantized LoRA (QLoRA direct loading). SGLang supports LoRA for MoE layers with JIT alignment kernels. TRT-LLM's LoRA is untested with EP, Helix, ADP, disaggregated serving, and speculative decoding.

**Impact:** LoRA is the primary mechanism for multi-tenant model customization in production. Incomplete LoRA support blocks enterprise deployments that need per-customer model variants with advanced infrastructure features.

**Opportunity:** Systematic LoRA compatibility testing across all feature combinations. Prioritize LoRA + EP (MoE with per-user adaptations) and LoRA + disaggregated serving (multi-tenant disaggregated deployments).

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
