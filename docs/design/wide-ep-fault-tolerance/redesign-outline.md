# WideEP FT Design — Rewrite Outline

**Created:** 2026-04-23
**Purpose:** Agreed structure for the v2 rewrite of `docs/design/wide-ep-fault-tolerance/`. Captures the section list, what each section covers, and which v1 material it replaces or merges. Ratified before drafting begins so the diff plan can be derived against it.

**Strategy:** Full rewrite in place on the `docs-and-plans` branch, replacing the v1 files. Reviewer comment anchors against the v1 files will break — accepted trade-off for clarity.

---

## Outline

### §0. Executive Summary
Short. What changed in this version + the headline decisions (MPI for MVP, Ray as future migration, two named audits as risks).

### §1. WideEP Today: User Journey, Stack, & Motivation
*Merges v1 §1 (Background) + §3 (motivation framing) + new stack-walkthrough material.*

- **§1.1 How users run WideEP today** — anchored on the **aggregated NVL72 single-rack** scenario as the canonical case. Real launch command, real env vars, real component instantiation order. Other deployment models (multi-node MPI, disagg, K8s+Ray) summarized in a table.
- **§1.2 The stack at each layer** — three-layer model (L1 process orchestration / L2 control plane / L3 data plane) with concrete components named at each layer for the canonical scenario.
- **§1.3 Why fault tolerance matters now** — the 7–8 minute downtime story, MTBF math, competitive pressure (carries forward from v1 §1).

### §2. Stack Comparison & TRT-LLM's Unique Position
*Merges v1 §2 (Current State) + §3 (Competitive Landscape) into a single comparison-and-positioning section.*

- **§2.1 Layer-by-layer comparison: TRT-LLM vs vLLM vs SGLang** — at L1, L2, L3, what does each engine use? Not a feature checklist but a structural comparison.
- **§2.2 What makes TRT-LLM's position unique** — kernel ownership of MNNVL/NVLinkOneSided (SGLang/vLLM depend on Mooncake/DeepEP); EPLB maturity (online migration, host-side shm, replication); MX-GMS roadmap (no competitor has this); NVL72-native design. The point: design choices in this doc only make sense given these advantages — they cannot be ported wholesale to vLLM or SGLang.
- **§2.3 Implications for FT design strategy** — kernel-level masking is the natural path because we own the kernel; full-restoration via shadow EP ranks is a unique capability TRT-LLM can offer.

### §3. Failure Modes & FT Gaps in TRT-LLM's Stack
*Replaces v1 §2's partial-failure infrastructure section + folds in v1 §7 (orchestrator critique).*

- **§3.1 Two failure modes**
    - **Mode A — Signal-handler `MPI_Abort` propagation.** A rank catches a signal, MPI's installed handler calls `MPI_Abort(MPI_COMM_WORLD)`, every other worker is killed before any FT logic can run. Source: `mpiUtils.cpp:199-210` (per reviewer).
    - **Mode B — AlltoAll kernel hang on dead-peer flag.** A rank dies silently (no signal); surviving ranks' AlltoAll kernel spins on `completion_flags[dead_rank]` indefinitely with no abort hook.
- **§3.2 Gap analysis by layer** — for each of L1 / L2 / L3 / EPLB / Detection, what's missing today, mapped to which failure mode it enables/blocks.
- **§3.3 Why not just pivot to Ray?** *(merges v1 §7.1–7.3.)* The orchestrator question, addressed here because it's the natural counterfactual to the Mode A gap. Reviewer's argument stated honestly; what Ray would buy and cost; soft claim on Ray-path perf characterization gap; **decision to stay on MPI for MVP**, with link forward to §5.4 for the required MPI-path work.

### §4. Three-Phase Recovery & Resilience Architecture
*Carries v1 §4 forward, expanded to three phases.*

- **§4.1 Phase 1 — Survive.** Mask failed rank, EPLB slot remap, continue serving at N-1.
- **§4.2 Phase 2 — Restore.** PG reconstruction, replacement rank joins, full N capacity.
- **§4.3 Phase 3 — Prevent / Scale.** Proactive degradation detection, preemptive migration, elastic scaling.
- **§4.4 Phase comparison table.**
- **§4.5 Layered reliability stack diagram** — where this design fits relative to PR #12718 (detection) and MX-GMS (recovery acceleration).

### §5. Phase 1: Immediate Survival
*Merges v1 §5 (Rank Masking) + §6 (EPLB) + §7 (Detection) + the §8 PR #12718 integration + the §7.4 MPI-path FT-enabling work.*

- **§5.1 Rank masking in communication kernels** — NVLinkOneSided primary path, NVLinkTwoSided + AllGatherReduceScatter follow-ons; what changes in the kernel and why.
- **§5.2 EPLB topology adaptation** — `reconfigure_mask_only` for MVP slot-remap; full reconfigure with weight migration for v1.
- **§5.3 Failure detection & PR #12718 integration** *(absorbs v1 §8 PR #12718 part.)* Three-layer detection stack; how we extend PR #12718's classifier without changing the three string-literal classes; per-rank `ErrorBudget`; failure broadcast protocol.
- **§5.4 MPI-path FT-enabling work** *(absorbs v1 §7.4.)* Signal handler replacement for `mpiUtils.cpp:199-210`; `MPIPoolExecutor` audit + routing-around for FT signaling; FT subcomm with `MPI_ERRORS_RETURN` + non-blocking Isend/Irecv.
- **§5.5 End-to-end flow & timing** — the <10s overall budget vs <10ms reconfigure-step distinction; what happens at each iteration boundary.

### §6. Phase 2: Full Restoration
*Carries v1 §4 Phase-2 portion forward, deepened. Absorbs v1 §8 MX-GMS part.*

- **§6.1 What restarts vs what stays alive** — only the dead rank's process is replaced; surviving processes keep CUDA contexts, weights, KV cache. The collective rebuild is participation, not failover.
- **§6.2 PG reconstruction per backend** — NCCL (`ncclCommAbort` + `ncclCommInitRank`); MNNVL (unmap + reallocate + re-handshake); NVSHMEM/DeepEP (no clean rebuild on current versions — flagged as audit risk); MPI (ULFM-or-blocked).
- **§6.3 Shadow rank + GMS roles** *(absorbs v1 §8 MX-GMS part.)* Per-rank shadow (not whole-group); GMS speeds weight load (~100ms) but does not change rebuild mechanics; MX P2P fallback for cross-node.
- **§6.4 Second-failure-during-rebuild handling** — the rebuild is collective and can't survive another death mid-operation; mitigation is to abandon the rebuild, mask the newly dead rank, and retry.

### §7. Phase 3: Beyond Failover
*Promoted from v1 implementation-plan section to a discussion-level section in its own right.*

- **§7.1 Latency anomaly detection** — per-rank AlltoAll latency via CUDA events; 3×-median anomaly detector.
- **§7.2 Preemptive expert migration** — degradation-signal-triggered migration off ranks showing thermal/ECC issues, before they fully fail. Reuses Phase 1 v1 weight-migration path.
- **§7.3 Elastic scaling (up/down)** — *new for v2.* Adding capacity to a healthy WideEP group (scale-up) and gracefully reducing capacity (scale-down) using the same primitives developed for Phase 2 (rank join, PG rebuild) and Phase 1 (rank mask, EPLB reconfigure).
- **§7.4 Predictive failure detection** — *new for v2.* Beyond reactive 3×-median detection: model-based predictions using error-rate trends, thermal patterns, ECC corrections.

### §8. Implementation Plan
*Carries v1 §9 forward, with Phase 3 added as rough plan.*

- **§8.1 Phase 1 PR breakdown** — detailed (carries forward from v1 §9 Phase 1).
- **§8.2 Phase 2 PR breakdown** — detailed (carries forward from v1 §9 Phase 2). MNNVL/NVSHMEM audit gates the size estimates.
- **§8.3 Phase 3 rough plan** — *new.* Sized at the work-track level (not per-PR) since the audits and Phase 2 work need to land first.
- **§8.4 Timeline summary** — phase totals, dependencies, critical path.

### §9. Risks and Open Questions
*Carries v1 §10 forward; promotes both audits to named risks per Q5.*

- **MNNVL/NVSHMEM teardown capability audit** as a named Phase-2-prerequisite risk.
- **Ray-path WideEP perf characterization** as a named risk gating any future Ray pivot.
- All v1 risks carry forward, restated against the new doc structure.

---

## What's removed from v1

- Standalone "Integration with MX-GMS" section (was v1 §8) — its content is split between §5.3 (PR #12718 part) and §6.3 (MX-GMS part).
- Standalone "Orchestrator Choice" section (was a new §7 in the interim outline) — its content is split between §3.3 (the why-not-Ray decision) and §5.4 (the MPI-path FT work).
- Multi-section Phase 1 split across "Rank Masking", "EPLB Adaptation", "Failure Detection" — now unified in §5.

## What's new vs v1

- §1.1 — the user-journey walkthrough anchored on aggregated NVL72.
- §1.2 — the L1/L2/L3 stack model.
- §2.1 — layer-level engine comparison (vs feature-level in v1 §3).
- §2.2 — TRT-LLM's unique position (kernel ownership argument).
- §3.1 — explicit two-failure-mode framing (signal handler + kernel hang).
- §3.3 — Ray pivot question, decided in writing.
- §6.1 — what-restarts-vs-stays-alive treatment.
- §6.2 — per-backend PG reconstruction semantics.
- §7.3, §7.4 — elastic scaling and predictive failure detection.

## Section count

10 numbered sections (§0 through §9). Down from 14 in v1.

---

## Next steps

1. ✅ Outline ratified (this file).
2. ✅ Research items recorded (`redesign-research-pass.md`).
3. ⏳ Run the research pass (~half day).
4. ⏳ Produce per-section diff plan from research findings.
5. ⏳ Sign-off on diff plan.
6. ⏳ Draft, section by section.
7. ⏳ Replace v1 files; refresh condensed doc as 2-pager per Q2.
8. ⏳ Commit + push.
