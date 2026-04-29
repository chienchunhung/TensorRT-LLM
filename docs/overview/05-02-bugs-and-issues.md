# 5.2 Critical Bugs and Architectural Issues

[< Back to Overview](README.md) | [Prev: Feature Gaps](05-01-feature-gaps.md) | [Next: Innovative Features >](05-03-innovative-features.md)

These are bugs, design debt, and inefficiencies in the current codebase that cause reliability issues, resource waste, or developer friction.

---

## 2.1 Disaggregated Serving Reliability

**Recently-fixed bugs (2026-04 window):**
- **Agg PP4 hang** (#12888, NVBug 6050489) — fixed.
- **Real errors propagated to disagg server** (#13119, `[TRTLLM-11123]`) — replaces silent stalls.
- **`aiohttp` session management consolidated** in disagg router (#13408) — drops a class of "connection died" failures.
- **Conversation-affinity disagg router** (#12526) — sticks multi-turn requests to the same gen rank.
- **Zombie worker pods detected via fatal-error tracking** (#12718, NVBug 6043291) — closes a long-standing operational gap; companion design under `docs/design/wide-ep-fault-tolerance/` on this branch.
- **Gen-only hang** where 10s sleep blocks KV transfers and overflows CTX memory (#12640) — fixed.
- **Prebuild ctx response to avoid `ctx_request_id` race** (#12466) — fixed.
- **`disaggregated_params` propagation through `PostprocWorker`** (#12513) — fixed.

**Still open (carried forward):**
- **Disagg hang on DGX B200 8-GPU** PyTorch path (#12656) — hardware-specific reliability issue (no clear close in this window).
- **Context PP + generation TP hang** — still listed in release notes for v1.2 known issues.
- **CacheTransceiver memory leak** — fixed in v1.1; lifecycle hardening still warranted given continued bug discovery.

**Systemic concern (still relevant):** Disaggregated serving is a critical differentiator, but the combination of KV transfer, multiple communication backends, heterogeneous parallelism, and overlap optimization creates a large surface area for timing-dependent bugs. Each fix often reveals new edge cases.

**Recommended action:** Comprehensive stress testing framework for disaggregated serving with failure injection (network delays, partial transfers, backend switching). Formal verification of the KV transfer state machine. *[Updated 2026-04-29]* The recent run of fail-fast / error-propagation fixes (#13119, #13408, #12718) is a turning point — the right next step is a chaos-test harness that injects the exact failure modes those PRs handle.

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

**Problem:** Two KV cache managers with different feature sets create reliability risks. *[Updated 2026-04-29: V2 progress this window — multiple V2 fixes (#13104, #12306, #12968 SWA, #461f3b97fc V2 SWA capacity, #12882 gen-only sync transfer V2). V2 is **still default OFF** (`_torch/pyexecutor/_util.py:68`).]*

**V2 gaps to close before becoming default (status updated 2026-04):**
- Beam search support — *still open*
- KV cache events for monitoring — *still open*
- KV connector for disaggregated serving — *partial: gen-only sync transfer V2 added (#12882)*
- Star attention / star CP support — *still open*
- Performance validation vs. C++ V1 (especially block allocation hot path) — *partial: cleanup commits #13280 (legacy `addSequence`), #10437 (unified reuse/non-reuse path), #13029 (batched two-phase claim) suggest V2-shaped allocator concepts are being absorbed into V1 too*

**V2 advantages over V1 (unchanged):**
- Constraint-based memory partitioning
- SSM cache reuse for hybrid models
- Heterogeneous `tokens_per_block`
- Scheduler-driven suspend/resume
- Python-first = faster experimentation and community contribution

**Recommended action:** Close V2 gaps systematically, then make V2 default, then deprecate V1. *[Updated 2026-04-29: trajectory looks correct but cadence is slow. A timeboxed "V2 default-on milestone with explicit gating criteria" would help avoid indefinite drift.]*

---

## 2.4 Feature Combination Matrix Gaps

**Problem:** The feature combination matrix reveals several unsupported or untested combinations that block real-world deployments. *[Updated 2026-04-29: 2 entries closed in this window (block-reuse + overlap, LoRA + spec-dec). Several others changed status.]*

| Combination | Status | Why It Matters |
|:------------|:-------|:---------------|
| **Block reuse + Overlap Scheduler** | ✓ **Now Yes** *[2026-04 #12816]* | Long-standing exclusivity removed |
| **LoRA + Speculative decoding (generic)** | ✓ **Now Yes** *[2026-04 #12661]* | Per-customer adapters with spec dec |
| **LoRA + EAGLE3 specifically** | ✓ **Now Yes** *[2026-04 #13005]* | EAGLE3 + LoRA path |
| **Attention DP + KV connector** | **Asserted off** *[2026-04 #13448]* | Now explicitly guarded; documents previously-implicit incompatibility |
| Spec decoding (MTP, EAGLE3) + PP | **Partially fixed** | MTP+PP hang fixed (#12555), but the broader spec-dec + PP combination is still constrained |
| Helix + ADP | **Known issues** | Limits advanced parallelism for long-context MoE |
| LoRA + EP/Helix/ADP/Disagg | **Untested** | Blocks production MoE deployments with LoRA (spec-dec sub-row now resolved) |
| Logits Post Processor + Disagg | **No** | Cannot do custom logits processing in disaggregated mode |
| C++ Sampler + any spec decoding | **No** | Forces Python sampler (higher overhead) for spec decoding |
| Helix + Overlap Scheduler | **Untested** | Uncertainty for long-context performance |
| DWDP + Overlap Scheduler | **Asserted off** *[2026-04]* | DWDP requires `disable_overlap_scheduler=True` (`py_executor.py:578`) |

**Deeper issue:** These gaps often reflect fundamental architectural assumptions (e.g., spec decoding + PP fails because the draft model and target model must be synchronized across PP stages). Fixing requires non-trivial executor changes.

**Recommended action:** Systematic testing campaign to resolve "Untested" entries (many may work already), then targeted engineering for "No" entries prioritized by user impact.

---

## 2.5 CUDA Event and Metrics Crashes

**Known bug:** CUDA event crash with performance metrics (#12639) — performance instrumentation causing crashes indicates fragile resource lifecycle management in the metrics path. *[Updated 2026-04-29: partial mitigation — `perf_metrics_manager` now guards `cuda.event.elapsed_time` to prevent executor crash (#12868). New Prometheus stack (#12545) is the right place to consolidate this hot-path safety.]*

**Recommended action:** Audit all CUDA event creation/destruction patterns in the metrics and profiling code. Ensure proper event lifecycle management even under error conditions.

**New crash-class fixes in this window:**
- DSA illegal memory access with CUDA graph + host KV cache offload (#13124, NVBug 6018172) — fixed.
- Stale CUDA graphs dropped on beam-width change (#13255, NVBug 6052050) — fixed.
- VLM guided decoding startup crash from missing `vocab_size_padded` (#12284) — fixed.
- WindowBlockManager destructor stats race (#12448) — fixed.
- DS V3.2 IMA WAR + trtllm-gen cubin/lib/src refresh (#13379, NVBug 6098442) — fixed.

---

## 2.6 Weights Loading OOM

**Known bug:** H20 weights loading OOM for large models (#11321) — memory spike during weight loading on memory-constrained GPUs.

**Root cause:** The meta-device init path avoids CPU memory spikes but the CUDA allocation pattern during weight loading can still exceed GPU memory for very large models on GPUs with less VRAM.

**Recommended action:** Streaming weight loading with per-layer allocation/deallocation. Profile peak GPU memory during weight loading for all supported model sizes.

---

## 2.7 FP8 Quantization Fragility

**Known issue:** FP8 quant fusion matching breaks after PyTorch updates (#12750) — the fusion pattern matching for FP8 quantization is tightly coupled to PyTorch internal representations.

**Recommended action:** Abstract the fusion pattern matching to be robust against PyTorch internal changes. Add PyTorch version compatibility tests for all quantization paths.
