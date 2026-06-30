# 0. Executive Summary

[< Back to Overview](README.md)

**Status:** Draft v2 (corrected MVP contract) | **Last updated:** 2026-06-30

## The problem

WideEP — distributing MoE experts across 32–72+ GPUs for DeepSeek-V3/R1 class models — has no fault tolerance in TRT-LLM today. A single GPU failure in a 72-rank EP group on NVL72 causes **8–20+ minutes of full downtime** because (a) the AlltoAll kernel has no abort hook and spins for 300s on the dead peer's `completion_flags` slot, or (b) the MPI signal handler at `mpiUtils.cpp:195–215` calls `MPI_Abort(MPI_COMM_WORLD)` which kills every other rank. The 5-minute hang detection then forces a full restart, which costs another 3–15+ minutes depending on whether the 681 GB checkpoint is on local NVMe, on cluster shared storage, or has to be re-downloaded. With 72 GPUs at 2–5 % AFR, MTBF is 3–7 days — daily downtime is not far off.

Competitors have shipped: SGLang's Elastic EP (March 2026, ~6.5s recovery), vLLM's RFC #27774 (active). TRT-LLM has nothing.

## The approach

Three-phase architecture:

- **Phase 1 (Survive, P0).** Detect the failure; abort the failed execution epoch; reconcile evidence; prove every expert still has a copy on an admitted failure domain; quiesce; prepare EPLB placement; rebuild survivor-only MPI/attention-DP and NCCL membership; apply CUDA-graph policy; atomically publish one mask + immutable `ActiveRankMap` + generation; dispose failed-epoch requests; then resume at N-1. The EPLB copy remains one sub-step, not the whole recovery transaction.
- **Phase 2 (Restore, P1).** A replacement process joins the EP group; communicators are torn down + rebuilt; EPLB rebalances for full N. Survivor processes stay alive throughout — only the dead rank's process is replaced. Sub-second target with MX-GMS-accelerated shadow ranks; minutes-class baseline with disk reload.
- **Phase 3 (Prevent / Scale, P2).** Latency anomaly detection, preemptive expert migration, elastic scale up/down, predictive failure detection. Reuses Phase 1 + 2 primitives.

## Why this design fits TRT-LLM specifically

Four structural properties differentiate TRT-LLM's stack:

1. **Kernel ownership.** TRT-LLM's primary AlltoAll backend (`NVLinkOneSided`) is a custom kernel over MNNVL shared CUDA memory — no third-party backend between us and the hardware. Masking *must* live inside the kernel because there is no library API to call. SGLang's Mooncake path can use `activeRanks`; vLLM's in-flight FT work instead uses backend-specific DeepEP timeout handling or NIXL-EP topology mutation. We own the equivalent MNNVL kernel work.
2. **EPLB maturity.** Online weight migration via `cudaMemcpy2D`, host-side POSIX shm, and a mature C++ implementation make a no-copy emergency remap possible *when* item 1b.2a proves that every layer/expert has a surviving resident copy on a distinct admitted failure domain. Slot count or average replication is not proof of that invariant.
3. **MX-GMS roadmap.** Crash-resilient GPU memory + cross-node weight streaming enables shadow EP ranks with sub-second activation. No competitor has the equivalent.
4. **NVL72-native design.** The fabric, the rank count (72, not 64 or 128), the node-local shm scope are all designed for the rack-scale fabric.

## Four failure quadrants, all explicitly classified

The canonical model uses two independent axes: whether survivors receive prompt host/process evidence and whether the peer's shared CUDA memory remains readable. Their 2×2 produces four quadrants:

- **Q1 — prompt evidence, memory readable.** Catchable signals used to call `MPI_Abort`; SIGKILL/OOM/other exits are instead observed by survivors or the launcher. Merged 1d.0 removes handler `MPI_Abort`, but 1d.1 must admit a launcher/runtime mode proven to preserve survivors; 1c.3a/1c.4a rebuild membership and 1d.0a owns poisoned lifecycle.
- **Q2 — no prompt evidence, memory readable.** On the supported MNNVL route, a live/silent peer can leave AlltoAll spinning on its completion flag. The launch-time rank-mask foundation landed as 1a.2 / #13404; 1a.4 supplies detection-only evidence and promoted 1a.8 supplies running-kernel escape.
- **Q3 — prompt evidence, memory unreadable.** The same control recovery may be usable only if rack-level 1d.4a testing proves that peer-memory loss does not poison survivor CUDA contexts; otherwise the path fails closed.
- **Q4 — no prompt evidence, memory unreadable.** In-process detection is not dependable, so external heartbeat and restart remain the baseline.

Pivoting to Ray removes the MPI-specific propagation and poisoned-lifecycle risks within Q1/Q3. It does not eliminate process failures, the Q2 live/silent MNNVL hang, or the Q3/Q4 peer-memory containment problem.

## The orchestrator decision

**MVP stays on MPI.** Reviewer's argument for Ray (decoupled process and communicator lifecycle) is structurally correct, but pivoting today carries three concrete costs that block MVP:

1. `HostMoeTensorSharer` hard-bakes `MPI.COMM_TYPE_SHARED` with no `TLLM_DISABLE_MPI` guard — real refactor work before WideEP runs on Ray.
2. Ray-path WideEP perf is uncharacterized; largest CI test is TP=4 (Llama-3.1 8B). Shipping production FT on a code path we haven't benchmarked at EP=32+ is unacceptable.
3. Ray + disagg + NIXL is unsupported (waive at `test_disaggregated.py:597`); blocks Phase 1-DS on Ray.

Ray remains an open future-migration question, gated on a named perf-characterization audit (§9 Audit 2) and the three preconditions in §9 Q8.

## Headline numbers

| Phase | Target | Status |
|:---|:---|:---|
| Phase 1 MVP | Correctness-first execution graph | 1a.1 (#13302), 1a.2 (#13404), 1b.1+1b.2 (#15525), and 1d.0 (#14160) are merged; the corrected graph includes promoted 1a.8/1a.11 and newly discovered integration work |
| Phase 1 v1 | Re-estimate after corrected MVP | Includes NVLinkTwoSided, full EPLB reconfigure with weight migration, and multi-failure consensus; 1a.8 and 1a.11 are no longer deferred |
| Phase 1-DS (disagg) | 3–4 weeks, parallelizable with v1 — 6 PRs | After MVP lands |
| Phase 2 (Restoration) | Rebaseline after teardown audits — 17 plan IDs, several conditional | After Phase 1 v1 + MNNVL audit; DeepEP/NVSHMEM audit is conditional on that backend |
| Phase 3 (Beyond failover) | Work-track sized; no fixed calendar yet | After Phase 2 |
| **Full program** | **No credible fixed calendar until corrected MVP and Phase 2 audits are sized** | Sequence is authoritative; the retired month estimate is not |

## Three named audit tracks

- **Audit 1 (Phase 2 prerequisite), split by hardware dependency:**
  - **1a — Intra-node, ~1 week, can start immediately on any ≥ 4-GPU NVLink node.** Validates NCCL rebuild, MPI signal-handler replacement, `cuMemUnmap` under owner death, DeepEP destructor mitigation, intra-node MNNVL teardown + rebuild, NVSHMEM teardown. Brings Phase 2 sizing within ±20%.
  - **1b — Rack-fabric validation, ~2–3 days NVL72 time.** Confirms intra-node results extrapolate to rack scale; resolves provisional-sizing caveat. Sequenced after 1a so rack time is targeted validation, not from-scratch prototyping.
- **Audit 2 (future-migration prerequisite):** Ray-path WideEP perf characterization. Gated on Ray-path CI existing at EP≥32 first. Empirical input for any future Ray pivot.
- **Audit 3 (cross-IB prerequisite):** NIXL-EP evaluation for deployments where DeepEP-family transport, rather than MNNVL, is selected.

## What this design changes vs v1

The v1 doc was reviewed and several gaps surfaced. v2 addresses them:

- **User journey upfront** ([§1](01-user-journey-and-stack.md)): grounds the reader in *the system* before discussing failures.
- **Layered stack model (L1/L2/L3)** ([§1.2](01-user-journey-and-stack.md#12-the-stack-at-each-layer)): disentangles MPI/Ray (L1), control plane (L2), MNNVL/NVSHMEM/NCCL (L3) so readers don't conflate them.
- **TRT-LLM uniqueness argument** ([§2](02-stack-comparison-and-positioning.md)): names what the design depends on that competitors don't have, so the FT approach is defensible.
- **Two-axis, four-quadrant failure model made explicit** ([§3.1](03-failure-modes-and-gaps.md#31-failure-modes-the-2x2)).
- **Ray pivot question decided in writing** ([§3.3](03-failure-modes-and-gaps.md#33-why-not-just-pivot-to-ray)): not deferred, not implied — explicitly answered with cost analysis.
- **What restarts vs what stays alive** ([§6.1](06-phase-2-full-restoration.md#61-what-restarts-and-what-stays-alive)): clarifies that Phase 2 is per-rank replacement, not whole-group rebuild.
- **Per-backend PG reconstruction semantics** ([§6.2](06-phase-2-full-restoration.md#62-pg-reconstruction)): NCCL works, MNNVL needs audit, direct NVSHMEM rebuild is deferred, and MPI without ULFM is structurally limited.
- **Phase 3 promoted to its own section** ([§7](07-phase-3-beyond-failover.md)): not just a footnote in implementation plan.
- **Three audit tracks as named risks** ([§9.1](09-risks-and-open-questions.md#91-named-audits-gating-risks)): not buried in the implementation plan and not confused with MVP correctness gates.
- **Section count: 10 → 10** but with cleaner phase boundaries (one section per phase).

Drafting and source verification are anchored against the [research pass report](redesign-research-pass-report.md).
