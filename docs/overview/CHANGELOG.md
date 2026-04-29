# Overview Refresh Changelog

This file is appended to by `UPDATE-PROMPT.md` on each periodic refresh of the
`docs/overview/` learning guide. Newest entries on top.

Each entry follows the schema documented in `UPDATE-PROMPT.md` §2 Phase F.
Tags of the form `docs-overview/YYYY-MM-DD` mark the commit at which each
entry was created, so a reader can `git diff docs-overview/<old>..docs-overview/<new> -- docs/overview/`
to see the literal text changes between two refreshes.

---

## 2026-04-29 — overview refresh

**Anchors.** Previous: `03c0ac33f7` (`2026-04-06`, oldest commit touching `docs/overview/`; no prior `docs-overview/*` tag).
Code anchor on `upstream/main`: `2b80f8d15f` (2026-04-06) → today's `3b7af1c21f` (2026-04-29). TRT-LLM `1.3.0rc?` → `1.3.0rc14`.
Snapshot of pre-refresh content saved at `docs/overview/.snapshots/2026-04-06/`.

### What changed in TRT-LLM since last update

154 commits in the window touching `tensorrt_llm/_torch`, `llmapi`, `executor`, `serve`, `cpp/tensorrt_llm`. Highlights:

- **Block reuse + overlap scheduler now combined** (#12816, `[TRTLLM-10939]`) — closes a long-standing internal gap; companion design lives at `docs/design/block-reuse-overlap-scheduler/` on this branch.
- **First-class KV connectors for `lmcache` and `kvbm`** (#12626) — see `tensorrt_llm/_torch/pyexecutor/connectors/registry.py:23,33`. Companion guard: attention-DP disabled when KV connector is in use (#13448).
- **Speculative decoding: 7 → 8 algorithms** — DFlash worker added (#12794, `_torch/speculative/dflash.py`); EAGLE3 dynamic-tree re-enabled (#13081); LoRA + spec-dec generic (#12661) and EAGLE3-specific (#13005); Mamba2 MTP custom-op invocation path (#12787).
- **AutoDeploy major progression:** standalone-package-ready (#13155, #13418), DeepSeek-R1 (#12601), Gemma-4 + Gemma 4-31B NVFP4 (#12710, #12866), MiniMax-M2.7 (#12963); legacy EdgeLLM ONNX export removed (#13418); TP deadlock in multi-stream MoE fixed (#13220); QKV+RoPE fusion with TRT-LLM attention (#12357).
- **Production observability stack:** Prometheus iter stats + token counters + phase histograms (#12545); modular logger with auto module detection + per-module filtering (#13202); NvTelemetry/GXT-compliant usage telemetry (#12384); per-iteration request-aggregate counters in `InflightBatchingStats` (#13199).
- **Tool-parser breadth:** GLM-4.7/GLM-5 (#13150) joining qwen3, qwen3_coder, glm4, deepseekv31, deepseekv32, minimax_m2 under `tensorrt_llm/serve/tool_parser/`.
- **Disagg reliability wave:** real errors propagated to disagg server (#13119, `[TRTLLM-11123]`), agg PP4 hang fixed (#12888), aiohttp session consolidated in disagg router (#13408), conversation-affinity router (#12526), zombie-worker-pod fatal-error detection (#12718, NVBug 6043291), Python cache transceiver extended to Qwen-Next (#12772), CP cache transmission contiguous → round-robin (#13180), gen-only `can_forward` 10s-sleep hang (#12640), prebuild ctx response (#12466), `disaggregated_params` propagation (#12513), multimodal KV cache block reuse for disagg (#12472).
- **KV cache V2 progress (still default OFF):** scheduler V2 fixes (#13104), V2 bug fixes (#12306), SWA capacity fix (#12968), gen-only sync transfer V2 + manager V2 (#12882). Companion V1 cleanup: legacy `addSequence` removed (#13280), reuse/non-reuse code path unified (#10437), batched two-phase claim with VSWA + non-reuse (#13029), `analyzePrefixReuse` consolidated to single radix-tree walk (#13095), KV-reuse + chunked-prefill compute-token accounting fixed (#12976).
- **Mamba/hybrid prefix caching for Qwen3.5 + Nemotron Super V3** (#12185).
- **ADP router: hit-rate gate + fair-share cap** (#13198).
- **Sharding + parallelism:** new sharding infrastructure (#12419, `[TRTLLM-12291]`); GEMM → AR fusion with output in registered buffers (#11589, `[TRTLLM-10004]`); centralized perfect-router integration + validation (#13250); DwdpConfig refactor (#12974); CuteDSL MoE backend onboarded for Qwen3.5 (#12799); customMoeRouting kernel extended for Qwen3.5 (#13433); sparse MQA/GQA attention (#12470); SageAttention refresh (#12937).
- **CUDA graphs:** +64 batch sizes for padding-enabled CUDA graphs (#12895); stale CUDA graphs dropped on beam-width change (#13255, NVBug 6052050).
- **Visual generation:** Cache-DiT + unified cache accelerator (#12548); LTX-2 CUDA graph (#12653); LTX-2 cached constant text computations across denoise steps (#12677); FLUX scheduler off-by-one fix (#13091); fast PNG compression (#13074); double PNG-encoding eliminated (#12903); multi-node diffusion workers via torchrun/SLURM (#13140); audio extraction from video for Nemotron Nano VL (#12921); video temporal compression (#12649); ViT attention kernel optimized on Nemotron (#12911); image-as-tensor unification (#12994); multimodal data cleared upon prefill completion (#13259).
- **Quantization:** tunable NVFP4 quantize via FlashInfer backend (#12126, `[TRTLLM-11091]`); FP4 residual quantization kernel without channel reorder (#13117); NVFP4 fused norm dim guard (#12901); RMS norm + FP4 quant kernel supports more dims (#13033).
- **Crash-class fixes:** DSA illegal memory access with CUDA graph + host KV cache offload (#13124, NVBug 6018172); DS V3.2 IMA WAR + trtllm-gen cubin/lib/src refresh (#13379, NVBug 6098442); WindowBlockManager destructor stats (#12448); VLM guided decoding startup crash from missing `vocab_size_padded` (#12284); `perf_metrics_manager` CUDA event guard (#12868); MTP+PP hang on last PP rank (#12555); Mamba cache correctness with MTP + CUDA-graph padding (#13151).
- **Encoder-only fast path:** `llm.encode()` (#12801).
- **Single-GPU host-overhead win:** request broadcast skipped when `world_size == 1` (#13412).
- **HMAC enforcement:** HMAC key requirement enforced in codebase (#9850); HMAC enabled in VisualGen ZMQ IPC (#12680).
- **Async-RL hooks:** abort + resume support for verl async RL (#12272, `[TRTLLM-10703]`).
- **Attention developer guide added** (#12693, `tensorrt_llm/_torch/modules/ATTENTION_DEVELOPER_GUIDE.md`).

### What changed in competitors

- **vLLM v0.19.0 → v0.20.0** (released 2026-04-27, 752 commits / 320 contributors / 123 new). Highlights:
  - CUDA 13.0 default (CUDA 13.0.2 to match PyTorch 2.11), Python 3.14, HuggingFace Transformers v5.
  - Models: DeepSeek V4 initial support, Hunyuan v3 preview, Granite 4.1 Vision built-in multimodal, EXAONE-4.5, BharatGen Param2MoE, Phi-4-reasoning-vision-15B.
  - **FlashAttention 4 re-enabled as default MLA prefill backend** (head-dim 512, paged-KV, SM90+).
  - **TurboQuant 2-bit KV cache compression** (~4× capacity).
  - New end-to-end online quantization frontend.
  - **vLLM IR foundation** (rms_norm op + testing infrastructure).
  - **Model Runner V2:** Eagle prefill full-CUDA-graph + multiple prompt-logprobs.
  - Extensive MoE refactor consolidating components.
  - ([release page](https://github.com/vllm-project/vllm/releases/tag/v0.20.0))
- **SGLang v0.5.10 → v0.5.10.post1** (2026-04-09): minor patch — flashinfer bumped v0.6.7.post2 → post3 to fix JIT cubin downloader.
- **LMCache v0.4.2 → v0.4.4** (2026-04-22). Highlights:
  - Different KV cache shapes / dtypes across layers.
  - Multi-path local-disk backend for multi-device I/O.
  - L0 Subscriber feature.
  - MP mode improvements (lazy heartbeat thread startup).
  - ValkeyConnector with cluster mode and TLS support.
  - ([release page](https://github.com/LMCache/LMCache/releases/tag/v0.4.4))
- **NVIDIA Dynamo v1.0.0** (March 2026) — production disaggregated serving wrapping SGLang/TRT-LLM/vLLM. Highlights:
  - Multimodal support across all three engines (text/image/video, encoder disagg, content-addressed hashing).
  - Agent hints + priority scheduling + KV retention for long agent sessions.
  - K8s `v1beta1 DynamoGraphDeploymentRequest` API + rolling updates + GPU auto-discovery.
  - Dynamo Snapshot fast worker recovery; GlobalPlanner + load-based scaling.
  - **AIConfigurator** open-source companion for offline P/D split + HW selection without GPU-hour search.
  - ([release page](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0))

### What changed in hardware / academic

**Hardware:**
- **NVIDIA Rubin (R200/VR200)** — full production at CES 2026; volume H2 2026 / sampling Q4 2026. 336B transistors, 288 GB HBM4, 22 TB/s, 50 PF FP4, NVLink 6 @ 3.6 TB/s, paired with Vera CPU (88 ARM cores). Vera Rubin NVL144 CPX = 8 EF AI / 100 TB / 1.7 PB/s per rack. ([NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/))
- **AMD Instinct MI355X** — surpassed 1M tokens/sec in MLPerf Inference 6.0 (April 2026), 3.1× over MI325X on Llama 2 70B Server. 288 GB HBM3E, 10 PF FP4. Strong vLLM integration on DeepSeek-R1, GPT-OSS-120B, Qwen3-235B, Llama-3.3-70B. ([AMD blog](https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html))
- **Google TPU v7 (Ironwood)** — GA 2026-03-31 (preview 2025-11-24). Supports LLM/MoE/diffusion training & inference. ([Cloud TPU release notes](https://cloud.google.com/tpu/docs/release-notes))
- **Groq LPU** — still LPU1 in production; no LPU2/LPU3 announced.
- **Etched Sohu** — still pre-shipping as of March 2026; only early-access; no independent benchmarks.

**Academic / research (April 2026):**
- **GOOSE — Anisotropic Speculation Trees for Training-Free Speculative Decoding** ([arXiv 2604.02047](https://arxiv.org/abs/2604.02047v1)): training-free 1.9–4.3× speedup combining n-gram and statistical sources; ~6× median acceptance gap → adaptive spine trees.
- **StreamServe — Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving** ([arXiv 2604.09562](https://arxiv.org/abs/2604.09562)): combines disagg P/D with online speculation-depth tuning; 11–18× latency reduction vs. tensor-parallel vLLM baseline.
- **Dual-Pool Token-Budget Routing** ([arXiv 2604.08075](https://arxiv.org/abs/2604.08075)): partitioning vLLM fleets into short-context vs. long-context pools cuts GPU-hours 31–42% and P99 TTFT 6%.
- **Prefill-as-a-Service (PrfaaS)** ([arXiv 2604.15039](https://arxiv.org/abs/2604.15039v1)): cross-datacenter KV transfer feasible for hybrid-attention models; 54% throughput gain vs. homogeneous baselines.
- **FlowKV** ([arXiv 2504.03775](https://arxiv.org/pdf/2504.03775)): block-wise KV transfer + load-aware scheduler; 0.944s → 0.053s (96% reduction) avg transfer time.
- **Anthropic automatic prompt caching** (Claude 3.7 Sonnet) — sets a new user-visible UX bar for prompt caching as a default.
- **Snowflake SuffixDecoding in ArcticInference** — production deployment of suffix-decoding-class spec dec in a competitor's stack.

### Per-file diff highlights

- `README.md`
  - "Last updated" → 2026-04-29; version pins → TRT-LLM 1.3.0rc14, vLLM 0.20.0, SGLang 0.5.10.post1, LMCache 0.4.4, **+ Dynamo v1.0.0**.
  - Spec-dec count corrected from 7 → 8; DFlash + EAGLE3 dynamic tree noted.
  - Maintenance section already linked `UPDATE-PROMPT.md` + `CHANGELOG.md` from the 2026-04-29 prep commit.
- `01-high-level-architecture.md`
  - "What's changed (v1.2-v1.3)" rewritten as "v1.2 → v1.3.0rc14"; AutoDeploy standalone-ready, EdgeLLM ONNX removal, DeepSeek-R1/Gemma-4/MiniMax onboardings called out; observability stack and Dynamo orchestration framing added.
  - Key files table extended with `connectors/registry.py`, `serve/tool_parser/`, and the new `ATTENTION_DEVELOPER_GUIDE.md`.
- `02-01-in-flight-batching.md`
  - Batched `addSequence` (#13029), per-iteration aggregate counters (#13199), `llm.encode()` fast path (#12801), single-GPU broadcast skip (#13412) added; framework-comparison row for Dynamo added.
- `02-02-overlap-scheduler.md`
  - **Headline change:** block reuse + overlap scheduler now coexist (#12816); DWDP + overlap exclusivity surfaced.
- `02-03-kv-cache-manager.md`
  - V2 progress + still-default-OFF status surfaced; comparison table refreshed for vLLM TurboQuant 2-bit, LMCache v0.4.4 features.
- `02-04-block-reuse.md`
  - ADP router hit-rate gate + fair-share cap (#13198); single-walk `analyzePrefixReuse` (#13095); compute-token accounting fix (#12976); chunked-prefill interaction; Dynamo row added to comparison.
- `02-05-disaggregated-serving.md`
  - "What's new" rewritten with the conversation-affinity router, round-robin CP transfer, fail-fast wave, lmcache/kvbm shorthand. New "Academic Frontier" section adds PrfaaS, FlowKV, StreamServe with arXiv links. Comparison table adds Dynamo v1.0.
- `02-06-speculative-decoding.md`
  - **Algorithm count corrected 7 → 8.** DFlash node + row added to mermaid + table. EAGLE3 dynamic tree re-enabled. New "Academic Frontier" section adds GOOSE + StreamServe.
- `02-07-parallelism-strategies.md`
  - DwdpConfig, GEMM→AR fusion, new sharding infra, sparse MQA/GQA, SageAttention refresh, CuteDSL MoE for Qwen3.5 added. New **Hardware Roadmap Implications** section adds Rubin, MI355X, TPU v7.
- `02-08-other-features.md`
  - Three new rows: **Observability**, **External KV connectors**, **Async RL hooks**. CUDA-graph row gets +64 batch padding + stale-graph fix. Multimodal row gets Cache-DiT, LTX-2 CUDA graph, audio extraction, multi-node diffusion. LoRA row gets spec-dec combo. Tool-parser row enumerates the new parsers.
- `03-user-journey.md`
  - Disagg reliability wins itemized (#13119, #12718, #13408, #13199); Dynamo v1.0 reframed as the recommended orchestration layer with autoscaling, K8s API, and AIConfigurator.
- `04-framework-comparison.md`
  - Architecture diagram bumped to v1.3.0rc14 / v0.20.0 / v0.5.10.post1 / v0.4.4 with a new Dynamo subgraph. Feature matrix gained Dynamo column; rows refreshed (TurboQuant 2-bit, FA4 MLA prefill, LMCache v0.4.4 features, PyTorch 2.11 / CUDA 13). Gap analysis adds three new rows (low-bit KV, end-to-end IR, MLA prefill kernel). Leadership table caveat-flagged that throughput numbers predate v0.20.
- `05-01-feature-gaps.md`
  - New **Status Summary** table classifying gaps as closed / improved / widened / new / stable.
  - Three new gap subsections: §1.9 Low-Bit KV Cache, §1.10 End-to-End IR + Runner V2, §1.11 MLA Prefill Kernel Defaults.
- `05-02-bugs-and-issues.md`
  - Disagg reliability section restructured into "recently-fixed" vs. "still open"; chaos-test next-step.
  - V2 default-on milestone framing added.
  - Feature combination matrix gains 5 new rows including 3 new closed combos (block-reuse + overlap, LoRA + spec-dec generic, LoRA + EAGLE3) and 2 explicitly-asserted-off (attention DP + KV connector, DWDP + overlap).
  - New crash-class fixes itemized.
- `05-03-innovative-features.md`
  - Promoted §3.4 KVaaS and §3.6 inference-time compute from "speculative" to "academically validated" with arXiv citations.
  - Hardware co-design section refreshed for shipping Rubin / TPU v7 / MI355X; Etched and Groq calibrated.
  - Self-optimizing engine §3.8 reframed around the new Prometheus + AIConfigurator substrate.
  - **New §3.9 Production Reliability + Multi-Engine Orchestration** anchored on Dynamo Snapshot, KV-block-utilization autoscaling, AIConfigurator, cross-engine KV reuse.
- `06-strategic-prioritization.md`
  - **Quadrant chart fully re-ranked** for 2026-04-29 with new entries (TTFT re-benchmark, low-bit KV, MLA prefill default, Dynamo Snapshot integration, IR strategy, adaptive spec-dec depth, GOOSE-style hybrid trees, Rubin co-design).
  - New "Items closed in this window" preamble.
  - Tier 1–4 lists rewritten; Dynamo orchestration reframing added at the end.

### Priority shifts (vs. previous `06-strategic-prioritization.md`)

| Item | Old tier | New tier (2026-04-29) | Reason |
|:-----|:---------|:----------------------|:-------|
| Block reuse + overlap scheduler | (implicit feature gap) | **Closed** | Shipped #12816 |
| First-class LMCache integration | (implicit feature gap) | **Closed** | Shipped #12626 |
| Production Prometheus | (Tier 2 implicit) | **Closed** | Shipped #12545 |
| LoRA + spec-dec | Tier 2 P2 | **Closed** | Shipped #12661 + #13005 |
| Mamba/hybrid prefix caching | (innovation) | **Closed** (partial) | Shipped #12185 |
| TTFT re-benchmark + targeted opt | Tier 1 P0 (legacy "TTFT optimization") | **Tier 1 P0 (sharpened)** | Old gap framing was stale; need fresh numbers vs. vLLM v0.20 + FA4 MLA |
| Low-bit KV (TurboQuant-class) | — | **Tier 1 P0 (NEW)** | vLLM v0.20 ships 4× capacity; long-context workloads are the user |
| MLA prefill kernel default audit | — | **Tier 1 P1 (NEW)** | vLLM v0.20 default = FA4 for MLA |
| Disagg chaos-test harness | — | **Tier 1 P0 (NEW)** | Recent fail-fast wave (#13119/#13408/#12718) demands fault-injection coverage |
| Dynamo Snapshot integration | — | **Tier 2 P2 (NEW)** | Closes part of elastic-FT gap without waiting for in-place failover |
| KV V2 default-on milestone | Tier 2 P2 | **Tier 2 P2 (with explicit gating)** | Cadence is slowing; needs a timebox |
| TRT-LLM IR strategy answer | — | **Tier 2 P2 (NEW)** | vLLM IR foundation forces a strategic decision |
| Adaptive spec-dec depth + GOOSE-style trees | — | **Tier 2 P2 (NEW)** | Academically validated 1.9–18× gains on `speculation_gate.py`-shaped problem |
| KVaaS / cross-DC KV (PrfaaS-class) | Tier 3 P3 | **Tier 3 P3 (sharpened)** | Now backed by arXiv 2604.15039 + connector-shorthand (#12626) |
| Vera Rubin HW-SW co-design (CPX) | Tier 4 P4 | **Tier 4 P4 (urgent flag)** | Volume H2 2026 — co-design window opens this year |

### Sources

Web sources (accessed 2026-04-29):
- [vLLM v0.20.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
- [SGLang v0.5.10.post1 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.10.post1)
- [LMCache v0.4.4 release](https://github.com/LMCache/LMCache/releases/tag/v0.4.4)
- [NVIDIA Dynamo v1.0.0 release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0)
- [NVIDIA Dynamo overall architecture](https://docs.nvidia.com/dynamo/latest/design-docs/overall-architecture)
- [Inside the NVIDIA Vera Rubin Platform — NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [NVIDIA Rubin enters full production — Introl Blog](https://introl.com/blog/nvidia-rubin-full-production-ces-2026-ai-infrastructure)
- [AMD MI355X distributed inference](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)
- [AMD MLPerf Inference 6.0 results](https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html)
- [Cloud TPU release notes (TPU v7 GA 2026-03-31)](https://cloud.google.com/tpu/docs/release-notes)

arXiv (accessed 2026-04-29):
- GOOSE — [arXiv 2604.02047](https://arxiv.org/abs/2604.02047v1)
- StreamServe — [arXiv 2604.09562](https://arxiv.org/abs/2604.09562)
- Dual-Pool Token-Budget Routing — [arXiv 2604.08075](https://arxiv.org/abs/2604.08075)
- PrfaaS — [arXiv 2604.15039](https://arxiv.org/abs/2604.15039v1)
- FlowKV — [arXiv 2504.03775](https://arxiv.org/pdf/2504.03775)

Code anchors (verified 2026-04-29 against `upstream/main` `3b7af1c21f`):
- `tensorrt_llm/_torch/pyexecutor/connectors/registry.py:23,33` — `lmcache`/`kvbm` shorthand entries.
- `tensorrt_llm/_torch/speculative/__init__.py` — exports DFlash + 7 other workers.
- `tensorrt_llm/_torch/pyexecutor/sampler.py:1152,4430` — TorchSampler / TRTLLMSampler.
- `tensorrt_llm/_torch/pyexecutor/_util.py:68` — V1/V2 selection switch.
- `tensorrt_llm/_torch/pyexecutor/py_executor.py:578` — DWDP requires `disable_overlap_scheduler=True`.

### Blocked / Skipped

- **TTFT and throughput numbers in §4.3 / §4.5 not re-measured.** The legacy "vLLM ~35% lower TTFT" claim and "TRT-LLM ~40% higher throughput on H100" claim both predate vLLM v0.20's Model Runner V2 + FA4 MLA prefill default. The doc now flags this caveat in §4.5 and promotes "TTFT re-benchmark" to Tier 1 P0; producing fresh numbers requires GPU access + benchmark runs that this autonomous refresh cannot do.
- **vLLM detailed PR-by-PR delta** not enumerated for v0.20 (752 commits, 320 contributors). Captured at the feature-headline level; deeper reading is left to follow-up.
- **NVIDIA internal release notes for v1.3.0** not consulted (no public file beyond v1.2 in `docs/source/release-notes.md` at the time of this run); v1.3 deltas reconstructed from commit log.

---

## Baseline — 2026-04 (pre-changelog)

The first dated `Last updated:` value in `docs/overview/README.md` was
**April 2026**, reflecting TensorRT-LLM v1.3.0, vLLM v0.19.0, SGLang v0.5.10,
and LMCache v0.4.2. No `docs-overview/*` tag exists for this baseline; the
first periodic refresh that runs `UPDATE-PROMPT.md` is responsible for
creating the first tag and the first dated changelog entry above this line.

If you are running the refresh prompt for the first time:
1. Treat the baseline as the previous-update anchor (`PREV_DATE = 2026-04-30`,
   `PREV_SHA = HEAD at the time of the first run`).
2. Snapshot the current `docs/overview/*.md` into
   `docs/overview/.snapshots/2026-04-30/` so future diffs have a real anchor
   even if no tag is created.
3. Append your first dated entry above this baseline note.
