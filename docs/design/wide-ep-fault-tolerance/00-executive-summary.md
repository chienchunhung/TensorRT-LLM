# 0. Executive Summary

[< Back to Overview](README.md)

**Status:** Draft v2 (substantial rewrite of v1) | **Last updated:** 2026-04-23

## The problem

WideEP — distributing MoE experts across 32–72+ GPUs for DeepSeek-V3/R1 class models — has no fault tolerance in TRT-LLM today. A single GPU failure in a 72-rank EP group on NVL72 causes **8–20+ minutes of full downtime** because (a) the AlltoAll kernel has no abort hook and spins for 300s on the dead peer's `completion_flags` slot, or (b) the MPI signal handler at `mpiUtils.cpp:195–215` calls `MPI_Abort(MPI_COMM_WORLD)` which kills every other rank. The 5-minute hang detection then forces a full restart, which costs another 3–15+ minutes depending on whether the 681 GB checkpoint is on local NVMe, on cluster shared storage, or has to be re-downloaded. With 72 GPUs at 2–5 % AFR, MTBF is 3–7 days — daily downtime is not far off.

Competitors have shipped: SGLang's Elastic EP (March 2026, ~6.5s recovery), vLLM's RFC #27774 (active). TRT-LLM has nothing.

## The approach

Three-phase architecture:

- **Phase 1 (Survive, P0).** Mask the dead rank in the AlltoAll kernel; rewrite EPLB placement so dead-rank slots are unreachable; tokens route to surviving replicas. **No process-group reconstruction.** Target: < 10 s end-to-end (detection-dominated), < 10 ms for the EPLB step. Requires replication ≥ 2, which is the DeepSeek production default.
- **Phase 2 (Restore, P1).** A replacement process joins the EP group; communicators are torn down + rebuilt; EPLB rebalances for full N. Survivor processes stay alive throughout — only the dead rank's process is replaced. Sub-second target with MX-GMS-accelerated shadow ranks; minutes-class baseline with disk reload.
- **Phase 3 (Prevent / Scale, P2).** Latency anomaly detection, preemptive expert migration, elastic scale up/down, predictive failure detection. Reuses Phase 1 + 2 primitives.

## Why this design fits TRT-LLM specifically

Four structural properties differentiate TRT-LLM's stack:

1. **Kernel ownership.** TRT-LLM's primary AlltoAll backend (`NVLinkOneSided`) is a custom kernel over MNNVL fabric memory — no Mooncake, no DeepEP between us and the hardware. Masking *must* live inside the kernel because there's no library API to call. SGLang and vLLM solve this by integrating with Mooncake's `activeRanks` API; we own the equivalent work.
2. **EPLB maturity.** Online weight migration via `cudaMemcpy2D`, host-side POSIX shm with all experts mapped node-locally, mature C++ implementation. MVP recovery is a placement-pointer rewrite — no H2D copy at recovery time, because every surviving rank already has every expert's weights mapped.
3. **MX-GMS roadmap.** Crash-resilient GPU memory + cross-node weight streaming enables shadow EP ranks with sub-second activation. No competitor has the equivalent.
4. **NVL72-native design.** The fabric, the rank count (72, not 64 or 128), the node-local shm scope are all designed for the rack-scale fabric.

## Two failure modes, both must be addressed

The reviewer feedback that shaped this rewrite identified that today's stack has *two* distinct failure modes, and FT must close both:

- **Mode A** — signal-handler `MPI_Abort` propagation (Layer 1, MPI-specific). Verified at `mpiUtils.cpp:195–215`. Closes via signal handler replacement (PR 1d.0, in flight as PR #14160) under the FT feature flag.
- **Mode B** — AlltoAll kernel hangs on a dead peer's completion flag (Layer 3, kernel-level). Verified in `moeAlltoAllKernels.cu`. Closes via in-kernel rank masking (PR 1a.2, in flight as PR #13404).

Pivoting to Ray would close Mode A structurally but doesn't help Mode B. Mode B is orthogonal to orchestrator choice.

## The orchestrator decision

**MVP stays on MPI.** Reviewer's argument for Ray (decoupled process and communicator lifecycle) is structurally correct, but pivoting today carries three concrete costs that block MVP:

1. `HostMoeTensorSharer` hard-bakes `MPI.COMM_TYPE_SHARED` with no `TLLM_DISABLE_MPI` guard — real refactor work before WideEP runs on Ray.
2. Ray-path WideEP perf is uncharacterized; largest CI test is TP=4 (Llama-3.1 8B). Shipping production FT on a code path we haven't benchmarked at EP=32+ is unacceptable.
3. Ray + disagg + NIXL is unsupported (waive at `test_disaggregated.py:597`); blocks Phase 1-DS on Ray.

Ray remains an open future-migration question, gated on a named perf-characterization audit (§9 Audit 2) and the three preconditions in §9 Q8.

## Headline numbers

| Phase | Target | Status |
|:---|:---|:---|
| Phase 1 MVP | ~7 weeks (AI coding-agent assisted) — 14 PRs | 1a.1 (PR #13302), 1a.2 (PR #13404), and 1d.0 (PR #14160) in flight |
| Phase 1 v1 | +6–9 weeks after MVP — 11 PRs | Includes NVLinkTwoSided, full EPLB reconfigure with weight migration, multi-failure consensus, kernel-side `check_timeout` tightening |
| Phase 1-DS (disagg) | 3–4 weeks, parallelizable with v1 — 6 PRs | After MVP lands |
| Phase 2 (Restoration) | 10–14 weeks — 16 PRs (sizes provisional pending Audit 1) | After Phase 1 v1 + MNNVL/NVSHMEM audit |
| Phase 3 (Beyond failover) | ~3 months — work-track sized | After Phase 2 |
| **Full program** | **7–10 months** with AI assistance, 10–14 months without | |

## Two named audits

- **Audit 1 (Phase 2 prerequisite), split by hardware dependency:**
  - **1a — Intra-node, ~1 week, can start immediately on any ≥ 4-GPU NVLink node.** Validates NCCL rebuild, MPI signal-handler replacement, `cuMemUnmap` under owner death, DeepEP destructor mitigation, intra-node MNNVL teardown + rebuild, NVSHMEM teardown. Brings Phase 2 sizing within ±20%.
  - **1b — Rack-fabric validation, ~2–3 days NVL72 time.** Confirms intra-node results extrapolate to rack scale; resolves provisional-sizing caveat. Sequenced after 1a so rack time is targeted validation, not from-scratch prototyping.
- **Audit 2 (future-migration prerequisite):** Ray-path WideEP perf characterization. Gated on Ray-path CI existing at EP≥32 first. Empirical input for any future Ray pivot.

## What this design changes vs v1

The v1 doc was reviewed and several gaps surfaced. v2 addresses them:

- **User journey upfront** ([§1](01-user-journey-and-stack.md)): grounds the reader in *the system* before discussing failures.
- **Layered stack model (L1/L2/L3)** ([§1.2](01-user-journey-and-stack.md#12-the-stack-at-each-layer)): disentangles MPI/Ray (L1), control plane (L2), MNNVL/NVSHMEM/NCCL (L3) so readers don't conflate them.
- **TRT-LLM uniqueness argument** ([§2](02-stack-comparison-and-positioning.md)): names what the design depends on that competitors don't have, so the FT approach is defensible.
- **Two failure modes named explicitly** ([§3.1](03-failure-modes-and-gaps.md#31-two-failure-modes-that-todays-stack-does-not-survive)).
- **Ray pivot question decided in writing** ([§3.3](03-failure-modes-and-gaps.md#33-why-not-just-pivot-to-ray)): not deferred, not implied — explicitly answered with cost analysis.
- **What restarts vs what stays alive** ([§6.1](06-phase-2-full-restoration.md#61-what-restarts-and-what-stays-alive)): clarifies that Phase 2 is per-rank replacement, not whole-group rebuild.
- **Per-backend PG reconstruction semantics** ([§6.2](06-phase-2-full-restoration.md#62-pg-reconstruction-per-backend)): NCCL works, MNNVL needs audit, NVSHMEM deferred, MPI without ULFM is structurally limited.
- **Phase 3 promoted to its own section** ([§7](07-phase-3-beyond-failover.md)): not just a footnote in implementation plan.
- **Two audits as named risks** ([§9.1](09-risks-and-open-questions.md#91-named-audits-gating-risks)): not buried in implementation plan; each is a 1–2 week scoped prototype.
- **Section count: 10 → 10** but with cleaner phase boundaries (one section per phase).

Drafting and source verification are anchored against the [research pass report](redesign-research-pass-report.md).
