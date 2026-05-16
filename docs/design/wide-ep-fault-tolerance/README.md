# WideEP Fault Tolerance for TensorRT-LLM

**Status:** Draft v2 (full rewrite) | **Last updated:** 2026-04-23

## Quick read

- **Problem:** A single GPU failure in a 72-rank EP group causes 7–8 min downtime (AlltoAll kernel hangs + MPI signal handlers abort the world). At 72-GPU scale, MTBF is 3–7 days.
- **Approach:** Three-phase architecture. Phase 1 survives via in-kernel rank masking + EPLB slot remap (<10s). Phase 2 restores via per-rank replacement + PG rebuild (<1s with MX-GMS). Phase 3 prevents + scales.
- **Key decision:** MPI stays as orchestrator for MVP. Ray is a future migration question, gated on perf characterization and two preconditions.
- **Two audits name-gated in §9:** MNNVL/NVSHMEM teardown (Phase 2 prereq), Ray-path WideEP perf (future-migration prereq).

See [§0 Executive Summary](00-executive-summary.md) for the full headline picture.

## Sections

| § | Title | What it covers |
|:---|:---|:---|
| **0** | [Executive Summary](00-executive-summary.md) | Problem, approach, decisions, headline numbers. Start here. |
| **1** | [User Journey, Stack & Motivation](01-user-journey-and-stack.md) | Canonical NVL72 launch, three-layer stack (L1/L2/L3), why FT now. |
| **2** | [Stack Comparison & TRT-LLM's Unique Position](02-stack-comparison-and-positioning.md) | Layer-level vs vLLM/SGLang; kernel ownership, EPLB maturity, MX-GMS, NVL72-native. |
| **3** | [Failure Modes & FT Gaps](03-failure-modes-and-gaps.md) | Two failure modes (signal handler + kernel hang); gap analysis per layer; Ray-pivot decision. |
| **4** | [Three-Phase Architecture](04-architecture-overview.md) | Phase 1/2/3 overview, layered reliability stack (with PR #12718 + MX-GMS). |
| **5** | [Phase 1: Immediate Survival](05-phase-1-immediate-survival.md) | Rank masking + EPLB + detection + MPI-path FT-enabling + end-to-end flow. |
| **6** | [Phase 2: Full Restoration](06-phase-2-full-restoration.md) | What restarts vs stays alive; per-backend PG reconstruction; shadow + GMS; second-failure handling. |
| **7** | [Phase 3: Beyond Failover](07-phase-3-beyond-failover.md) | Latency anomaly detection, preemptive migration, elastic scaling, predictive detection. |
| **8** | [Implementation Plan](08-implementation-plan.md) | Per-PR breakdown for Phase 1 + Phase 2, Phase 3 rough plan, timelines. |
| **9** | [Risks and Open Questions](09-risks-and-open-questions.md) | Named audits, 14 technical risks with residual ratings, 8 open design questions. |

## Related workstreams

| Workstream | Status | Relationship to this design |
|:---|:---|:---|
| [PR #12718: Fatal Error Detection](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | In review (squashed; bench-shutdown regression fixed) | Foundation — provides `classify_error()` + `ErrorBudget` + `pre_shutdown()` non-blocking pattern that §5.3 extends per-rank. See [§5.3 *Lessons from PR #12718 implementation*](05-phase-1-immediate-survival.md#lessons-from-pr-12718-implementation) for design takeaways. |
| MX + GMS + TRT-LLM Integration | Design complete | Acceleration — GMS zero-copy import cuts Phase 2 recovery from minutes to ~100ms; enables shadow EP ranks (§6.3) |

## In-flight PRs against this design

| PR | Title | Status | Section |
|:---|:---|:---|:---|
| [#13302](https://github.com/NVIDIA/TensorRT-LLM/pull/13302) | WideEP FT: add EPGroupHealth thread-safe rank mask | In review | §5.3, PR 1a.1 |
| [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) | WideEP FT: NVLinkOneSided kernel mask | Open | §5.1, PR 1a.2 |
| [#14160](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) | WideEP FT: add MPI signal handler replacement (1d.0) | Open | §5.4, PR 1d.0 |
| [#14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) | WideEP FT: scaffold MVP end-to-end prototype | Draft (DO NOT SUBMIT — preview only) | [mvp-prototype-plan.md](mvp-prototype-plan.md) |

## MVP de-risking — end-to-end prototype

Before all 14 MVP PRs land, a **3–5 day end-to-end prototype** on a 4 or 8-GPU node validates the integration seams between tracks (kernel mask ↔ EPLB ↔ watchdog ↔ broadcast ↔ engine hook) ahead of the production PRs. See [MVP prototype plan](mvp-prototype-plan.md) for the full plan, including hardware options (HGX/DGX B200/B300/H100 vs. GB200/GB300 NVL72 tray), IMEX setup steps for GB200/GB300, the kill-and-survive test recipe, and exit criteria. The prototype reuses PR #13302 (`EPGroupHealth`) and PR #14160 (1d.0 signal-handler replacement) as-is; everything else is stubbed. Scaffolding is shipped as preview-only draft **[PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198)** (`prototypes/wide_ep_ft_mvp/`); the directory is discarded once the production PRs land.

## Forward-looking research exploration

- [Straggler speculation research](straggler-speculation-research/README.md) — sub-directory capturing the research arm of straggler mitigation (Option B in §7.5: speculative redundant compute in synchronous AlltoAll). Three docs: problem framing, literature survey + search plan, publication venue analysis. Not committed engineering work; the production track (A + D in §7.5) is independent.

## Related FT work in vLLM and SGLang

External fault-tolerance work in adjacent inference frameworks. Surveyed May 2026; see [§1.3](01-user-journey-and-stack.md#13-why-fault-tolerance-now) and [§3.3](03-failure-modes-and-gaps.md#33-why-not-just-pivot-to-ray) for how this informs our framing.

| Reference | What it is | Status |
|:---|:---|:---|
| [vLLM PR #34833](https://github.com/vllm-project/vllm/pull/34833) | Fault-reporting framework — ZMQ-based sentinels, HTTP `GET /fault_tolerance/status`, ZMQ PUB on `vllm_fault` topic | In flight; targets Ray + internal LB only |
| [vLLM PR #38534](https://github.com/vllm-project/vllm/pull/38534) | Pause-on-error workflow — DeepEP / NIXL-EP "FT-enabled backends" with active-rank-mask and 100s static kernel timeout; HTTP `POST /fault_tolerance/apply` | In flight; builds on #34833 |
| [vLLM PR #40468](https://github.com/vllm-project/vllm/pull/40468) | Cleanup + retry — non-blocking NCCL `commAbort`, DP cpu_group rebuild, in-flight requests preempted to waiting queue, prefix-cache-driven retry without replacement rank (operates at N-1 indefinitely) | In flight; reviewers flagged bugs |
| [SGLang FT RFC (gaidandawang-afk fork)](https://github.com/gaidandawang-afk/sglang/issues/1) | Three-plane framework proposal: data plane (Mooncake-EP / NIXL-EP), control plane (SGLang FT Framework with ZMQ sentinels), decision plane (serving framework). Same `/fault_tolerance/status` + `/fault_tolerance/apply` API as vLLM | RFC on a personal fork; not yet on the official sgl-project/sglang |

Three observations from the survey shape framing in this design:
- **Convergent architecture** — vLLM and SGLang are converging on the same three-phase rollout (report → pause → cleanup/retry) and the same HTTP+ZMQ control surface. Worth aligning our `check_health()` (PR 1d.2) and replacement-rank API (PR 2c.1) so deployments using vLLM/SGLang FT tooling can extend to TRT-LLM.
- **Both target Ray, not MPI** — strengthens the long-term Ray-pivot argument; doesn't change our MPI-for-MVP decision.
- **vLLM's kernel-side mask is 100s auto-mask, not 300s `trap;`** — validates PR 1a.8's direction (replace `trap;` with host-visible flag).

## Workflow artifacts

Planning materials that informed this rewrite (preserved for record-keeping):

- [Redesign outline](redesign-outline.md) — agreed 10-section structure before drafting
- [Research pass items](redesign-research-pass.md) — source-verification checklist
- [Research pass report](redesign-research-pass-report.md) — findings that anchor the rewrite
- [Audit 1a findings](audit-1a-findings.md) — Days 1–3 empirical results (NCCL rebuild, MPI signal handlers, `cuMemUnmap` on dead-peer regions)
- [Research pass prototypes](research-pass-prototypes/README.md) — runnable scripts that produced the audit findings

## Scope & non-goals

**In scope (primary):** aggregated WideEP serving, PyTorch backend, NVLinkOneSided primary + NVLinkTwoSided + AllGatherReduceScatter, single-GPU failure (MVP) then multi-failure (v1).

**In scope (deferred track):** Phase 1-DS for disaggregated serving, after MVP.

**Out of scope:** DeepEP / DeepEPLowLatency (blocked on NVSHMEM `mask_buffer_ptr`); TensorRT engine backend (legacy); standard EP ≤ 8 GPUs; individual request durability across failures.

## Terminology

| Term | Definition |
|:---|:---|
| **WideEP** | Expert parallelism across ≥ 32 GPUs (vs. standard EP within a single 8-GPU node). |
| **EP group** | The set of ranks participating in a single AlltoAll collective. For DeepSeek-V3 on NVL72, `ep=72`. |
| **Rank / Process / Slot** | One rank = one process = one GPU. One rank has *multiple* slots (`slotCountPerRank`, typically 4–8). One slot holds one expert's weights. One expert can be replicated to slots on multiple ranks. |
| **Mode A** | Failure via MPI signal handler `MPI_Abort(MPI_COMM_WORLD)`. Kills all ranks. Layer 1 problem. |
| **Mode B** | Failure via silent dead peer; AlltoAll kernel spins on `completion_flags[*][dead_rank]`. Layer 3 problem. |
| **EPLB** | Expert-parallel load balancer. `MoeLoadBalancer` C++ + Python. |
| **Rank masking** | Kernel-level: AlltoAll reads a bitmask and skips dead peers in send/poll loops. |
| **Slot remap** | EPLB-level: rewrite `MoePlacementInfo` so dead-rank slots are unreachable. |
| **Emergency reconfigure** | MVP recovery = rank masking + slot remap at next iteration boundary. < 10 ms. |
| **Weight migration** | v1 operation: H2D copy of expert weights to a new slot. Needed when an expert has zero surviving replicas. MVP avoids by requiring replication ≥ 2. |
| **MVP (v0) / v1** | First shipping milestone (single failure, NVLinkOneSided, slot-remap only) vs full Phase 1 (all backends, weight migration, multi-failure). |

---

## v1 archive

The v1 version of this design doc (10 split files + README + COMBINED) has been replaced. v1 was reviewed, and substantive reviewer feedback — particularly around MPI failure modes, the Ray-pivot question, and Phase 2 reconstruction mechanics — motivated this v2 rewrite.

v1 files are removed; the research pass report ([redesign-research-pass-report.md](redesign-research-pass-report.md)) anchors every factual claim against current source.
