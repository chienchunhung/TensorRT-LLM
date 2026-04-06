# 2.8 Other Notable Features

[< Back to Overview](README.md)

| Feature | Description | Impact | What's New (v1.2-v1.3) |
|:--------|:-----------|:-------|:------------------------|
| **CUDA Graph** | Captures kernel sequences as replayable graphs; padding to nearest captured size | Up to 22% throughput improvement | PDL (Programmatic Dependent Launch) now default |
| **Chunked Prefill** | Splits long prompts across iterations, interleaving with decode | Reduces TPOT variance | Chunked Pipeline Parallelism for million-token context (SGLang) |
| **Guided Decoding** | Grammar/schema-constrained generation (JSON mode) | Structural output guarantees | Now works with all spec decode methods and disagg serving |
| **LoRA** | Runtime adapter loading without restart; per-request adapter selection | Multi-task serving efficiency | Still untested with EP, Helix, ADP, Disagg |
| **Multimodal** | Vision-language models + audio + visual generation (LTX-2, WAN, FLUX) | Multi-modal inference | FA4 attention for diffusion; audio support; dynamic resolution |
| **KV Cache Salting** | Security isolation for multi-tenant prefix caching | Prevents prompt theft attacks | — |
| **FlexKV** | Flexible KV cache backend | Configurable cache strategies | New in v1.3 |
| **Quantization** | FP8, NVFP4, MXFP8, 2FP4/Arcquant | Memory/compute efficiency | Mixed quant for shared/routed MoE experts |
| **Visual Generation** | Diffusion model support (LTX-2, WAN, FLUX) | Image/video generation | Fused DiT QK Norm + RoPE kernel; two-stage pipeline |
| **Agentic Support** | Tool parsers (GLM-4), interleaved thinking, Harmony parser | Agentic workflows | Auto option for tool/reasoning parsers |
| **Energy Metrics** | Power consumption monitoring via `trtllm-serve` | Cost tracking | New in v1.2 |
