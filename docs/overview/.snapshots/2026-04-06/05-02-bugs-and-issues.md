# 5.2 Critical Bugs and Architectural Issues

[< Back to Overview](README.md) | [Prev: Feature Gaps](05-01-feature-gaps.md) | [Next: Innovative Features >](05-03-innovative-features.md)

These are bugs, design debt, and inefficiencies in the current codebase that cause reliability issues, resource waste, or developer friction.

---

## 2.1 Disaggregated Serving Reliability

**Known bugs (recent fixes indicate systemic issues):**
- **Gen-only hang** where 10s sleep blocks KV transfers and overflows CTX memory (#12640) — fixed but indicates fragile timing assumptions in the disagg pipeline.
- **Disagg hang on DGX B200 8-GPU** PyTorch path (#12656) — hardware-specific reliability issue.
- **Context pipeline parallelism + generation tensor parallelism hang** — documented known issue in release notes, not yet resolved.
- **CacheTransceiver memory leak** in disaggregated serving — fixed in v1.1 but indicates the transfer path needs memory lifecycle hardening.
- **Multimodal KV cache block reuse** broken for disaggregated serving (#12472) — fixed but shows multi-feature interaction bugs.

**Systemic concern:** Disaggregated serving is a critical differentiator, but the combination of KV transfer, multiple communication backends, heterogeneous parallelism, and overlap optimization creates a large surface area for timing-dependent bugs. Each fix often reveals new edge cases.

**Recommended action:** Comprehensive stress testing framework for disaggregated serving with failure injection (network delays, partial transfers, backend switching). Formal verification of the KV transfer state machine.

---

## 2.2 Codebase Complexity and Monolithic Executor

**Problem:** `py_executor.py` is ~3,750 lines with three execution loops (`_executor_loop`, `_executor_loop_overlap`, `_executor_loop_pp`), extensive conditional branches for disaggregated serving, speculative decoding, attention DP, pipeline parallel, and overlap scheduling.

**Specific pain points:**
- **Three backends** (PyTorch, TRT, AutoDeploy) create confusion and triplicate maintenance
- **Two KV cache managers** (V1 C++, V2 Python) with different feature sets and different bugs
- **Feature combination matrix** has 19+ features with multiple "No", "Untested", and "Known Issues" entries
- Adding a new feature requires understanding interactions with speculative decoding, disaggregated serving, overlap scheduling, pipeline parallelism, and CUDA graphs — all interleaved in the same executor loop

**Impact on velocity:** vLLM has a significantly larger community contributor base, partly because lower complexity reduces the barrier to contribution. Model support velocity (100+ vs ~50+ architectures) is a downstream effect.

**Recommended action:**
- Refactor `py_executor.py` into composable executor stages (scheduling, resource allocation, forward, sampling, response handling)
- Converge on AutoDeploy as the primary backend to reduce backend maintenance
- Complete V2 KV cache feature parity to eliminate dual-manager confusion

---

## 2.3 KV Cache V1/V2 Feature Divergence

**Problem:** Two KV cache managers with different feature sets create reliability risks.

**V2 gaps to close before becoming default:**
- Beam search support
- KV cache events for monitoring
- KV connector for disaggregated serving (currently limited in V2)
- Star attention / star CP support
- Performance validation vs. C++ V1 (especially block allocation hot path)

**V2 advantages over V1:**
- Constraint-based memory partitioning
- SSM cache reuse for hybrid models
- Heterogeneous `tokens_per_block`
- Scheduler-driven suspend/resume
- Python-first = faster experimentation and community contribution

**Recommended action:** Close V2 gaps systematically, then make V2 default, then deprecate V1.

---

## 2.4 Feature Combination Matrix Gaps

**Problem:** The feature combination matrix reveals several unsupported or untested combinations that block real-world deployments.

| Combination | Status | Why It Matters |
|:------------|:-------|:---------------|
| Spec decoding (MTP, EAGLE3) + PP | **No** | Cannot use spec decoding for models requiring PP |
| Helix + ADP | **Known issues** | Limits advanced parallelism for long-context MoE |
| LoRA + EP/Helix/ADP/Disagg | **Untested** | Blocks production MoE deployments with LoRA |
| Logits Post Processor + Disagg | **No** | Cannot do custom logits processing in disaggregated mode |
| C++ Sampler + any spec decoding | **No** | Forces Python sampler (higher overhead) for spec decoding |
| Helix + Overlap Scheduler | **Untested** | Uncertainty for long-context performance |

**Deeper issue:** These gaps often reflect fundamental architectural assumptions (e.g., spec decoding + PP fails because the draft model and target model must be synchronized across PP stages). Fixing requires non-trivial executor changes.

**Recommended action:** Systematic testing campaign to resolve "Untested" entries (many may work already), then targeted engineering for "No" entries prioritized by user impact.

---

## 2.5 CUDA Event and Metrics Crashes

**Known bug:** CUDA event crash with performance metrics (#12639) — performance instrumentation causing crashes indicates fragile resource lifecycle management in the metrics path.

**Recommended action:** Audit all CUDA event creation/destruction patterns in the metrics and profiling code. Ensure proper event lifecycle management even under error conditions.

---

## 2.6 Weights Loading OOM

**Known bug:** H20 weights loading OOM for large models (#11321) — memory spike during weight loading on memory-constrained GPUs.

**Root cause:** The meta-device init path avoids CPU memory spikes but the CUDA allocation pattern during weight loading can still exceed GPU memory for very large models on GPUs with less VRAM.

**Recommended action:** Streaming weight loading with per-layer allocation/deallocation. Profile peak GPU memory during weight loading for all supported model sizes.

---

## 2.7 FP8 Quantization Fragility

**Known issue:** FP8 quant fusion matching breaks after PyTorch updates (#12750) — the fusion pattern matching for FP8 quantization is tightly coupled to PyTorch internal representations.

**Recommended action:** Abstract the fusion pattern matching to be robust against PyTorch internal changes. Add PyTorch version compatibility tests for all quantization paths.
