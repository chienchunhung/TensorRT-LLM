# WideEP Fault Tolerance for TensorRT-LLM

**Status:** Draft v2 (full rewrite) | **Last updated:** 2026-06-30

## Quick read

- **Problem:** A single GPU failure in a 72-rank EP group causes roughly 8–20+ minutes of downtime, depending on checkpoint locality and restart conditions (AlltoAll kernel hangs + MPI signal handlers abort the world). At 72-GPU scale, MTBF is 3–7 days.
- **Approach:** Three-phase architecture. Phase 1 detects, aborts the failed epoch, reconciles evidence, validates admission, quiesces, prepares EPLB, rebuilds survivor control/NCCL, applies graph policy, commits mask + `ActiveRankMap` + generation, applies request disposition, and resumes at N-1. Phase 2 adds a replacement rank and restores full N. Phase 3 prevents + scales.
- **Key decision:** MPI stays as orchestrator for MVP. Ray is a future migration question, gated on perf characterization and two preconditions.
- **Safety invariant:** detection is not commitment. Watchdogs and failure-broadcast threads report evidence; only the recovery coordinator may publish a committed mask + immutable `ActiveRankMap` + generation, after placement, survivor communicators, and graph policy are ready. Request disposition precedes resume.
- **Three audit tracks are named in §9:** baseline MNNVL teardown (with DeepEP/NVSHMEM conditional on backend selection), Ray-path WideEP performance, and NIXL-EP evaluation for the cross-IB path.

See [§0 Executive Summary](00-executive-summary.md) for the full headline picture.

## Sections

| § | Title | What it covers |
|:---|:---|:---|
| **0** | [Executive Summary](00-executive-summary.md) | Problem, approach, decisions, headline numbers. Start here. |
| **1** | [User Journey, Stack & Motivation](01-user-journey-and-stack.md) | Canonical NVL72 launch, three-layer stack (L1/L2/L3), why FT now. |
| **2** | [Stack Comparison & TRT-LLM's Unique Position](02-stack-comparison-and-positioning.md) | Layer-level vs vLLM/SGLang; kernel ownership, EPLB maturity, MX-GMS, NVL72-native. |
| **3** | [Failure Modes & FT Gaps](03-failure-modes-and-gaps.md) | Two-axis, four-quadrant Q1–Q4 failure model; gap analysis per layer; Ray-pivot decision. |
| **4** | [Three-Phase Architecture](04-architecture-overview.md) | Phase 1/2/3 overview, layered reliability stack (with PR #12718 + MX-GMS). |
| **5** | [Phase 1: Immediate Survival](05-phase-1-immediate-survival.md) | Rank masking + EPLB + detection + MPI-path FT-enabling + end-to-end flow. |
| **6** | [Phase 2: Full Restoration](06-phase-2-full-restoration.md) | What restarts vs stays alive; per-backend PG reconstruction; shadow + GMS; second-failure handling. |
| **7** | [Phase 3: Beyond Failover](07-phase-3-beyond-failover.md) | Latency anomaly detection, preemptive migration, elastic scaling, predictive detection. |
| **8** | [Implementation Plan](pr-execution/08-implementation-plan.md) | Per-PR breakdown for Phase 1 + Phase 2, Phase 3 rough plan, timelines. |
| **9** | [Risks and Open Questions](09-risks-and-open-questions.md) | Named audits, technical risks, admission/containment limits, and open design questions. |

## PR execution workspace

All implementation planning and dependency tracking is consolidated in the [PR execution workspace](pr-execution/README.md).

| Artifact | Document | Scope |
|:---|:---|:---|
| **MVP (v0)** | [MVP PR dependency graph](pr-execution/mvp-dependency-graph.md) | Live upstream PR status, corrected Phase 1 scope, blocked/unblocked edges, and physical acceptance gates. |
| **V1** | [V1 PR dependency graph](pr-execution/v1-dependency-graph.md) | Phase 1 v1 plus the parallel Phase 1-DS and conditional Phase 1-IB tracks. |
| **V2** | [V2 PR dependency graph](pr-execution/v2-dependency-graph.md) | Maps V2 to Phase 2 Restoration because the roadmap has no separately named V2 product milestone. |

The [JIRA work-item ledger](pr-execution/jira-work-item-ledger.md) preserves all 22 supplied ticket mappings and records newly discovered work as `JIRA: TBD` until tickets are assigned. The [source-of-truth correction checklist](pr-execution/source-of-truth-correction-checklist.md) tracks the 2026-06-30 correction across docs and PRs.

## Related workstreams

| Workstream | Status | Relationship to this design |
|:---|:---|:---|
| [PR #12718: Fatal Error Detection](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | Merged 2026-04-27 | Foundation — provides `classify_error()` + `ErrorBudget` + `pre_shutdown()` non-blocking pattern that §5.3 extends per-rank. See [§5.3 *Lessons from PR #12718 implementation*](05-phase-1-immediate-survival.md#lessons-from-pr-12718-implementation) for design takeaways. |
| MX + GMS + TRT-LLM Integration | Design complete | Acceleration — GMS zero-copy import cuts Phase 2 recovery from minutes to ~100ms; enables shadow EP ranks (§6.3) |

## Tracked MVP PRs

Status snapshot: 2026-07-02 14:32 PDT. The [MVP dependency graph](pr-execution/mvp-dependency-graph.md) is authoritative for live PR status, blocked/unblocked edges, the dependency-ready action frontier, and corrected scope. This table is a delivery map, not a substitute for the graph.

| Plan ID | PR | JIRA work item(s) | Title | PR status | Section |
|:---|:---|:---|:---|:---|:---|
| 1a.1 | [#13302](https://github.com/NVIDIA/TensorRT-LLM/pull/13302) | [TRTLLM-12199](https://jirasw.nvidia.com/browse/TRTLLM-12199) | WideEP FT: add EPGroupHealth thread-safe rank mask | Merged 2026-06-17 PDT | §5.3 |
| 1d.0 | [#14160](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) | [TRTLLM-13550](https://jirasw.nvidia.com/browse/TRTLLM-13550) | WideEP FT: add MPI signal handler replacement | Merged 2026-06-22 PDT | §5.4 |
| 1a.2 | [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) | [TRTLLM-12200](https://jirasw.nvidia.com/browse/TRTLLM-12200) | WideEP FT: NVLinkOneSided kernel mask | Merged 2026-06-30 PDT | §5.1 |
| 1a.3 + 1a.4 | [#15524](https://github.com/NVIDIA/TensorRT-LLM/pull/15524) | [TRTLLM-12556](https://jirasw.nvidia.com/browse/TRTLLM-12556), [TRTLLM-12557](https://jirasw.nvidia.com/browse/TRTLLM-12557) | WideEP FT: add Python rank-mask wiring and AlltoAll watchdog | Draft; corrected head `d19aadea`; DCO/pre-commit green; `blossom-ci` pending | §5.1 |
| 1b.1 + 1b.2 | [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525) | [TRTLLM-13543](https://jirasw.nvidia.com/browse/TRTLLM-13543), [TRTLLM-13544](https://jirasw.nvidia.com/browse/TRTLLM-13544) | WideEP FT: add EPLB mask-only reconfigure | Merged 2026-06-29 PDT | §5.2 |
| 1c.1 | [#15677](https://github.com/NVIDIA/TensorRT-LLM/pull/15677) | [TRTLLM-13546](https://jirasw.nvidia.com/browse/TRTLLM-13546) | Add WideEP FT error-classification patterns | Merged 2026-07-02 14:32 PDT; all reported checks green | §5.3 |
| 1a.7 | [#15789](https://github.com/NVIDIA/TensorRT-LLM/pull/15789) | [TRTLLM-12560](https://jirasw.nvidia.com/browse/TRTLLM-12560) | Add NCCL fault-tolerance wrapper for WideEP | Draft; `blossom-ci` pending | §5.1 |
| 1c.3 | [#15785](https://github.com/NVIDIA/TensorRT-LLM/pull/15785) | [TRTLLM-13548](https://jirasw.nvidia.com/browse/TRTLLM-13548) | Add MPI FT subcommunicator and broadcast thread | Draft; corrected head `ee9aa0a4`; DCO/pre-commit green; `blossom-ci` pending | §5.3 |
| 1d.3 | [#15788](https://github.com/NVIDIA/TensorRT-LLM/pull/15788) | [TRTLLM-13553](https://jirasw.nvidia.com/browse/TRTLLM-13553) | Add WideEP rank-health telemetry | Draft; corrected head `94274a3f`; DCO/pre-commit green; `blossom-ci` pending | §5.5 |
| MVP integration prototype | [#15801](https://github.com/NVIDIA/TensorRT-LLM/pull/15801) | [TRTLLM-12728](https://jirasw.nvidia.com/browse/TRTLLM-12728) | Production-component E2E MVP integration vehicle | Draft; head `5a76856e`; mergeable; all initial checks green (base freshness skipped for draft) | [Prototype plan](mvp-prototype-plan.md) |
| Historical scaffold | [#14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) | [TRTLLM-12728](https://jirasw.nvidia.com/browse/TRTLLM-12728), historical predecessor; current vehicle #15801 | Stub-heavy seam-finding prototype; not MVP proof | Draft, paused, `DO NOT SUBMIT` | [Prototype findings](mvp-prototype-findings.md) |

## MVP de-risking — end-to-end prototype

Draft [PR #15801](https://github.com/NVIDIA/TensorRT-LLM/pull/15801) is the new **production-component vertical-slice** integration vehicle, stacked from `main` plus the actual in-flight PR heads. It must run a real MoE model/workload on physical GPUs, kill a real non-rank-0 worker, discard every output from the failed epoch, and prove that later requests complete correctly through the same recovery transaction production will use. Unit mocks remain useful, but cannot satisfy the MVP exit gate. See [MVP prototype plan](mvp-prototype-plan.md).

[PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) is retained only as historical seam-finding evidence. Its stub-heavy runs found useful issues—especially poisoned `MPI_Finalize`—but did not prove an end-to-end working recovery path. Intra-node physical validation is item 1d.4; 1d.4a adds production FABRIC/IMEX process-death and approved inaccessible-peer-memory/device-loss acceptance at rack scale.

## Forward-looking research exploration

- [Straggler speculation research](straggler-speculation-research/README.md) — sub-directory capturing the research arm of straggler mitigation (Option B in §7.5: speculative redundant compute in synchronous AlltoAll). Three docs: problem framing, literature survey + search plan, publication venue analysis. Not committed engineering work; the production track (A + D in §7.5) is independent.

## Related FT work in vLLM and SGLang

External fault-tolerance work in adjacent inference frameworks. Surveyed May 2026; see [§1.3](01-user-journey-and-stack.md#13-why-fault-tolerance-now) and [§3.3](03-failure-modes-and-gaps.md#33-why-not-just-pivot-to-ray) for how this informs our framing.

| Reference | What it is | Status |
|:---|:---|:---|
| [vLLM PR #34833](https://github.com/vllm-project/vllm/pull/34833) | Fault-reporting framework — ZMQ-based sentinels, HTTP `GET /fault_tolerance/status`, ZMQ PUB on `vllm_fault` topic | In flight; targets Ray + internal LB only |
| [vLLM PR #38534](https://github.com/vllm-project/vllm/pull/38534) | Pause-on-error workflow for DeepEP / NIXL-EP "FT-enabled backends" with backend-specific timeout/failure handling; HTTP `POST /fault_tolerance/apply`. The verified NIXL-EP API itself uses `connect_ranks` / `disconnect_ranks`, not `activeRanks`. | In flight; builds on #34833 |
| [vLLM PR #40468](https://github.com/vllm-project/vllm/pull/40468) | Cleanup + retry — non-blocking NCCL `commAbort`, DP cpu_group rebuild, in-flight requests preempted to waiting queue, prefix-cache-driven retry without replacement rank (operates at N-1 indefinitely) | In flight; reviewers flagged bugs |
| [vLLM Elastic EP blog (2026-05-14)](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism) | Production framework for runtime EP-group resizing via `POST /scale_elastic_ep`. **Elastic EP** is the framework; **NIXL EP** is one of two backends (other: `allgather_reducescatter`). FT path = "scale-down then scale-up" — same mechanism as elastic scaling. | Shipped |
| [vLLM PR #35627](https://github.com/vllm-project/vllm/pull/35627) | NIXL-EP integration into vLLM Elastic EP. Verified API surface: `buffer.connect_ranks()` / `buffer.disconnect_ranks()` for incremental topology mutation; `torch.distributed.TCPStore` dependency. No `activeRanks` masking primitive — FT is topology mutation, not masking. | Merged 2026-03-13 |
| [vLLM RFC #30112](https://github.com/vllm-project/vllm/issues/30112) | Fault-Tolerant Expert Parallelism — keeps survivor DP ranks alive when NCCL watchdog kills one; rebuilds NCCL via MoE scale-down. Parent-process-level fix vs our signal-handler-level 1d.0. Claims 3s recovery. | Open, Dec 2025 |
| [ai-dynamo/nixl#1415](https://github.com/ai-dynamo/nixl/pull/1415) | NIXL EP VMM allocator — adds `cuMemCreate(... FABRIC ...)` with cudaMalloc fallback. Enables MNNVL substrate for NIXL-EP. | Merged 2026-04-05; production NVL72 maturity still in flight (see ai-dynamo/nixl#1655, #1499, #1530 — open) |
| [SGLang FT RFC (gaidandawang-afk fork)](https://github.com/gaidandawang-afk/sglang/issues/1) | Three-plane framework proposal: data plane (Mooncake-EP / NIXL-EP), control plane (SGLang FT Framework with ZMQ sentinels), decision plane (serving framework). Same `/fault_tolerance/status` + `/fault_tolerance/apply` API as vLLM | RFC on a personal fork; not yet on the official sgl-project/sglang |

Five observations from the survey shape framing in this design:
- **Convergent control surface** — vLLM (`/fault_tolerance/status`, `/fault_tolerance/apply`, `/scale_elastic_ep`) and SGLang are converging on HTTP+ZMQ. Worth aligning our `check_health()` (PR 1d.2) and replacement-rank API (PR 2c.1).
- **Unified FT + scaling architecture validated in production.** vLLM's Elastic EP treats FT as "scale-down then scale-up" — same mechanism. This validates our §7.5 unified-variability framing (which we converged on independently); the kernel-level first-wins combine (research-arm question 1) and joint placement formulation (research-arm question 4) remain the genuinely-novel contributions.
- **NIXL-EP is for cross-IB transport, not a general MNNVL replacement.** Verified API is `connect_ranks` / `disconnect_ranks` (topology mutation, not masking); needs `torch.distributed.TCPStore`; MNNVL substrate landed but production NVL72 maturity still in flight. TRT-LLM's NVL72 path stays on kernel-mask architecture; cross-IB deployments are where NIXL-EP fits. See [§9.1 Audit 3](09-risks-and-open-questions.md#audit-3--nixl-ep-evaluation-as-cross-ib-data-plane-backend).
- **Both vLLM and SGLang target Ray, not MPI** — strengthens the long-term Ray-pivot argument *specifically for cross-IB deployments* (NIXL-EP requires TCPStore); doesn't change our MPI-for-NVL72-MVP decision.
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

**Out of corrected NVL72 MVP scope:** direct DeepEP / DeepEPLowLatency survivor recovery; cross-IB deployments instead use the conditional Phase 1-IB track (NIXL-EP topology mutation if Audit 3 is positive, or a limited DeepEP timeout interim). Also out of scope: TensorRT engine backend (legacy), standard EP ≤ 8 GPUs, and transparent replay of already-emitted tokens. The failed execution epoch must still produce no partial or zero-filled logits.

## Terminology

| Term | Definition |
|:---|:---|
| **WideEP** | Expert parallelism across ≥ 32 GPUs (vs. standard EP within a single 8-GPU node). |
| **EP group** | The set of ranks participating in a single AlltoAll collective. For DeepSeek-V3 on NVL72, `ep=72`. |
| **Rank / Process / Slot** | One rank = one process = one GPU. One rank has *multiple* slots (`slotCountPerRank`, typically 4–8). One slot holds one expert's weights. One expert can be replicated to slots on multiple ranks. |
| **Q1** | Prompt survivor-visible host/process evidence; peer shared memory remains readable. The MPI handler/launcher lifecycle is one Q1 mechanism, not the quadrant definition. |
| **Q2** | No prompt host/process evidence; peer shared memory remains readable. On MNNVL, a live/silent peer can leave AlltoAll spinning on `completion_flags[*][dead_rank]`. |
| **Q3** | Prompt survivor-visible host/process evidence; peer shared memory is not readable. In-place recovery is admitted only if 1d.4a proves survivor-context containment. |
| **Q4** | No prompt host/process evidence; peer shared memory is not readable. External heartbeat and restart remain the fail-closed baseline. |
| **EPLB** | Expert-parallel load balancer. `MoeLoadBalancer` C++ + Python. |
| **Rank masking** | Kernel-level: AlltoAll reads a bitmask and skips dead peers in send/poll loops. |
| **Slot remap** | EPLB-level: rewrite `MoePlacementInfo` so dead-rank slots are unreachable. |
| **Detected rank state** | Failure evidence from a watchdog, MPI worker death, or FT notification. Detection never changes the data-plane mask by itself. |
| **Committed recovery generation** | A survivor-agreed active-rank map published only after the failed epoch is aborted, placement passes admission, survivor control/data communicators are ready, and graph policy has been applied. |
| **Emergency reconfigure** | The atomic recovery transaction: failed-epoch abort, reconciliation, admission, quiesce, EPLB preparation, survivor communicator rebuild, graph policy, mask + `ActiveRankMap` + generation commit, request disposition, and resume. The EPLB copy may be <10 ms; the complete transaction is larger. |
| **Weight migration** | H2D copy of an expert into a new slot. MVP avoids this only where item 1b.2a proves, per layer and expert, a surviving copy on a distinct admitted failure domain. |
| **MVP (v0) / v1** | First shipping milestone: admitted single non-frontend failure, failed-epoch suppression, survivor membership, and atomic N-1 recovery on supported backends. V1 adds broader backends, online weight migration, and multi-failure policy. |

---

## v1 archive

The v1 version of this design doc (10 split files + README + COMBINED) has been replaced. v1 was reviewed, and substantive reviewer feedback — particularly around MPI failure modes, the Ray-pivot question, and Phase 2 reconstruction mechanics — motivated this v2 rewrite.

v1 files are removed; the research pass report ([redesign-research-pass-report.md](redesign-research-pass-report.md)) anchors every factual claim against current source.
