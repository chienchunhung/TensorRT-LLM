# WideEP Fault Tolerance Source-of-Truth Correction Checklist

[< Back to PR execution](README.md) · [MVP dependency graph](mvp-dependency-graph.md) · [Implementation plan](08-implementation-plan.md)

**Correction started:** 2026-06-30 PDT

**Progress snapshot:** 2026-06-30 14:23 PDT. Independent final audits of the in-repo contract, Google Doc, and MVP/V1/V2 dependency proof pass; correction commit `36bb8d58` is published on `docs-and-plans`. Corrected heads `d19aadea` (#15524), `ee9aa0a4` (#15785), and `94274a3f` (#15788) are published with green DCO/pre-commit, and `blossom-ci` is pending. Unchecked canonical MVP items below are genuine implementation, review/CI, or physical-hardware gates, not hidden documentation debt.

**Scope:** in-repo design and execution documents, the canonical Google Doc, all affected in-flight PRs, and the no-mock end-to-end MVP prototype branch.

This checklist is the control record for the 2026-06-30 design correction. A checked item means the change has been applied and verified in its destination, not merely discussed. Existing JIRA-backed IDs are preserved. New work uses a suffix ID and remains `JIRA: TBD` until a ticket is assigned.

## Completion rules

- [x] **C00 — One contract:** the in-repo design, PR implementation plan, dependency graphs, JIRA ledger, Google Doc, PR descriptions, and code use the same item names, state machine, dependencies, and MVP exit criteria.
- [x] **C01 — Safe publication:** no detection path can publish a new communication mask before placement, control-plane membership, and data-plane communicators are ready for the same generation.
- [ ] **C02 — Failed-epoch safety:** a timeout or peer failure aborts the current execution epoch; no partial or zero-filled logits from that epoch may be returned.
- [ ] **C03 — Real survivor invariant:** MVP admission proves that every expert in every layer has a surviving copy on a distinct failure domain. Aggregate spare-slot count is not accepted as proof.
- [ ] **C04 — Real control plane:** degraded execution removes the failed rank from both the MPI/control plane and attention-DP/PyExecutor collectives; the FT broadcast thread alone is not treated as a replacement for those collectives.
- [ ] **C05 — Physical E2E proof:** MVP completion requires a real model and workload, real processes, real CUDA/MPI/NCCL/MNNVL components, and physical GPU fault injection. Mocks and stubs may remain unit-test aids but cannot satisfy the MVP exit gate.

## Canonical MVP item corrections

### Promoted from V1 to MVP

- [ ] **C10 / 1a.8 — In-flight kernel abort and generation primitive:** replace the 300-second `trap;` escape with a stable device/host-visible abort or generation mechanism that a running kernel can observe and return through without poisoning the CUDA context. PR #13404 remains the pre-launch/next-launch rank-mask foundation; it does not by itself satisfy 1a.8.
- [ ] **C11 / 1a.11 — CUDA graph recovery policy:** ship an eager-mode recovery path plus graph invalidation/recapture after communicator or membership generation changes. The prototype may force eager mode; production MVP must define and test graph recovery.

### New MVP items

- [ ] **C12 / 1b.2a — FT placement invariant and admission:** verify, per layer and per expert, at least one survivor after any admitted single-rank failure and require copies to occupy distinct failure domains. Fail closed before serving if the invariant is not met.
- [ ] **C13 / 1c.3a — Survivor control communicator and `ActiveRankMap`:** create a survivor-only MPI/control communicator and a logical-to-physical rank map for all post-failure management collectives.
- [ ] **C14 / 1c.4a — Degraded attention-DP/PyExecutor membership:** rebuild or bypass the blocking rank-state, request, batch-size, token-count, and model-input gathers so they operate only over survivors.
- [ ] **C15 / 1c.4b — Atomic recovery coordinator:** own `detect → abort failed epoch → reconcile evidence → validate admission → quiesce → prepare EPLB → rebuild survivor control/NCCL → apply graph policy → commit mask + ActiveRankMap + generation`; 1c.4c applies request disposition before resume.
- [ ] **C16 / 1c.4c — Failed epoch and request disposition:** define the boundary between the aborted epoch, retryable queued work, in-flight requests, and externally visible request errors; integrate the contracts from PRs #12718 and #13119.
- [ ] **C17 / 1d.0a — Poisoned-MPI lifecycle and shutdown:** when the world communicator is poisoned, prohibit world collectives and `MPI_Finalize` paths that can hang; provide a deterministic survivor and failed-rank shutdown policy.
- [ ] **C18 / 1d.4a — NVL72 FABRIC/IMEX containment acceptance:** validate rack-scale membership and real process death, then use an approved IMEX-grant revocation, GPU reset/isolation, or equivalent injection that makes peer memory inaccessible. Prove survivor-context containment and recovery, or keep Q3 fail-closed/restart. A healthy-GPU process death cannot prove Q3; intra-node NVSwitch testing remains the earlier 1d.4 gate.

### Expanded existing MVP items

- [ ] **C19 / 1c.4 — Model-engine recovery hook:** connect classification, rank-health tracking, the watchdog, and the recovery coordinator without directly publishing membership or duplicating coordinator state.
- [ ] **C20 / 1d.1 — Feature and admission gate:** unify the feature flag; require MPI thread support, supported launcher/backend, placement admission, HBM/fabric prerequisites, rank-0 policy, and CUDA-graph policy; reject unsupported MegaMoE/backend routes explicitly.
- [ ] **C21 / 1d.4 — Production-component E2E harness:** replace the prior stub-heavy prototype contract with real process death, a realistic model/workload, request-level correctness, no-output-from-failed-epoch checks, and recovery timing assertions.
- [x] **C22 / 1a.4 — Detection-only watchdog contract:** the watchdog reports a suspected failure to the coordinator. It must not directly mutate the committed data-plane mask.
- [x] **C23 / 1a.7 — Manual NCCL recovery primitive contract:** abort/rebuild is coordinator-driven and generation-scoped. Static sharding and unsupported communicators remain fail-closed until their membership integration lands.
- [x] **C24 / 1c.3 — Failure-notification contract:** the FT subcommunicator reports failure evidence and consensus state; it does not replace normal MPI/attention-DP collectives or commit the active mask itself.

## In-repo document synchronization

- [x] **C30 — Overview:** update `README.md` and `00-executive-summary.md` with the corrected MVP boundary, safety invariants, and source-of-truth links.
- [x] **C31 — User journey and stack:** correct DeepSeek-V3 slot arithmetic, remove the false “replication ≥ 2” assumption, and distinguish NVSwitch topology from FABRIC/IMEX handle behavior.
- [x] **C32 — Failure modes:** add stale-running-kernel, premature-mask-publication, static MPI/ADP membership, failed-epoch output, CUDA-graph, poisoned-MPI, and unsupported-backend failure modes.
- [x] **C33 — Architecture:** document detected versus committed rank state, the recovery generation/state machine, survivor control/data planes, admission gates, and explicit rank-0/front-end scope.
- [x] **C34 — Phase 1:** update immediate-survival sequence, invariants, acceptance criteria, and non-goals.
- [x] **C35 — Implementation plan:** update all item definitions, scope tags, dependencies, PR merge-unit notes, current PR status, critical path, estimates, and the no-mock prototype strategy.
- [x] **C36 — Risks and open questions:** record unresolved hardware/fabric, frontend/rank-0, physical-failure containment, placement, and backend-selection risks with owners or gates.
- [x] **C37 — Prototype docs:** mark PR #14198 as historical seam-finding evidence, and define the new stacked-on-main no-mock prototype and its hardware acceptance matrix.
- [x] **C38 — JIRA ledger:** promote 1a.8, add all suffix items as `JIRA: TBD`, preserve supplied workflow state, and separate work-item status from delivery-PR status.
- [x] **C39 — Folder index and redirect:** make `pr-execution/` the canonical execution workspace and point the legacy `08-implementation-plan.md` to it.

## Dependency-graph proof

- [x] **C40 — MVP graph completeness:** every corrected, promoted, expanded, and new MVP item appears once with its work-item ID, JIRA/PR mapping, status, and hard parents.
- [x] **C41 — Edge truth:** green hard edges originate only from merged prerequisites; red hard edges originate from non-merged prerequisites; gray dashed edges are explicitly non-blocking.
- [x] **C42 — Action frontier:** the gold `★` outline appears only on non-merged production items whose hard parents are all merged. Root production items qualify; paused or historical prototypes do not.
- [x] **C43 — V1/V2 reconciliation:** remove promoted work from the V1 delivery set, preserve cross-milestone dependencies, and ensure V2 depends on the corrected MVP/V1 contract rather than stale assumptions.
- [x] **C44 — Mermaid validation:** verify node classes, link indices, labels, hyperlinks, and rendering-safe syntax after every graph edit.

## Google Doc synchronization

- [x] **C50 — Target/readback guard:** confirm the exact Google Doc ID, title, tab, and revision before each write batch.
- [x] **C51 — Canonical correction section:** add a dated source-of-truth section containing the corrected invariants, item changes, recovery state machine, and MVP exit gates.
- [x] **C52 — Existing-section repair:** update or supersede every repeated stale claim, including replication, process-group reconstruction, 1a.8/1a.11 scope, prototype use of mocks, and old status/dependency summaries.
- [x] **C53 — Links and structure:** preserve heading/list style and link the in-repo plan, graphs, PRs, and supporting sources with readable labels.
- [x] **C54 — Connector verification:** re-read the edited document and verify target identity, section order, text, headings, lists, and hyperlinks; record any rendered-layout limitation.

Google Docs body synchronization is complete and independently audited at revision `ALtnJHzhKIb008QuJYeQ3YEcq3OGbHgR51Yman1OMFjEZ-RqTAj9E1w06x_XpVI1_DdOPA4YJ4b13AhKrJfUHTjE0lCH-LuKh_v_8huA6SY`. The separate Drive Comments API rejected attempts to annotate stale comment threads with `appNotAuthorizedToFile`; no comment thread was changed, and this does not affect the verified document body.

## In-flight PR alignment

- [x] **C60 / #13404:** keep the pre-launch kernel-mask scope; correct claims that imply a running kernel observes later mask updates or that `trap;` recovery is solved; link the promoted 1a.8 follow-up.
- [x] **C61 / #15524:** prevent watchdog detection from directly publishing the committed `EPGroupHealth` mask; add/adjust tests for detection-only behavior; align the feature flag and describe the coordinator contract.
- [x] **C62 / #15677:** verify the PR remains a pattern-only classifier slice and does not claim ownership of recovery, membership, or mask publication.
- [x] **C63 / #15785:** clarify failure-notification versus survivor-collective scope; add the integration contract and identify destructive peer-death coverage as 1c.3a/1d.4 work unless safely included here.
- [x] **C64 / #15789:** document coordinator-driven generation-scoped recovery, corrected prerequisites, static-sharding fail-closed behavior, and the CUDA-graph dependency.
- [x] **C65 / #15788:** repair DCO/sign-off, keep telemetry passive, and ensure it reports coordinator-committed membership/generation without conflating raw detected state or driving recovery.
- [x] **C66 — PR description parity:** every affected PR description points to the same item definition, dependencies, limitations, and follow-up work as the source-of-truth plan.
- [x] **C67 — Stacked-branch integrity:** preserve parent/child history, create backup refs before rewrites, use `--force-with-lease` where required, and verify each PR diff against its intended base.

The final composition check merged the published #15524 and #15788 heads with `git merge-tree --write-tree` without conflict. Their shared `EPGroupHealth` header now states the same committed-membership contract.

## No-mock MVP prototype and validation

- [x] **C70 — Integration worktree:** create `WideEP-FT/e2e-mvp-prototype` from current upstream `main` and stack the published PR heads without changing the source PR branches.
- [x] **C71 — Stack construction:** integrate #13404, #15524, #15677, #15785, #15789, and #15788; resolve the shared communicator import/header conflict while retaining both changes.
- [x] **C72a — Correction restack:** rebuild the branch on current `upstream/main` (`81290eb6` at validation), then bring published #15524 head `d19aadea`, #15785 head `ee9aa0a4`, and #15788 head `94274a3f` semantics into the prototype while preserving the integrated PR stack and conflict resolution. The local branch is 14 commits ahead of that base and clean.
- [ ] **C72b — Missing vertical slices:** implement the coordinator, survivor membership, admission, kernel-abort, request-disposition, and poisoned-lifecycle slices needed for a real vertical path.
- [x] **C73a — Local validation:** run formatting/lint, compile checks, focused behavioral tests or smoke tests, DCO checks, graph/stale-claim checks, and diff checks in every changed branch. The #15785 lightweight detected-state suite passed 57 focused tests (the broader earlier harness passed 71); repository-native pytest for #15524/#15788 remains unavailable locally because the environment lacks PyTorch. The restacked integration branch passes syntax compilation and `git diff --check`.
- [ ] **C73b — CI acceptance:** complete `blossom-ci` on the corrected PR heads and triage any head-specific failure without attributing unrelated infrastructure failures to the changes.
- [ ] **C74 — Intra-node physical test:** run realistic serving on at least four NVLink-connected GPUs, kill a non-rank-0 worker during communication, prove failed-epoch suppression, and verify survivor correctness and continued requests.
- [ ] **C75 — Rack-fabric test:** run 1d.4a on NVL72/equivalent with IMEX/FABRIC for both real process death and an approved inaccessible-peer-memory/device-loss case; capture timestamped traces, survivor-context outcome, recovery/steady-state metrics, and the explicit fail-closed boundary.

## Publication and closure

- [x] **C80 — In-repo docs published:** commit with DCO and push the corrected `docs-and-plans` branch.
- [x] **C81 — Google Doc published:** connector readback proves the corrected content is present in the intended document.
- [x] **C82 — PR branches published:** push every necessary code/history correction and update the upstream PR descriptions.
- [x] **C83 — CI requested:** trigger the appropriate TensorRT-LLM CI after branch changes and record blockers without mislabeling PR review state.
- [x] **C84 — Final consistency audit:** no stale critical claim remains; all links resolve; the dependency graph action frontier agrees with the tables; unchecked items are genuine hardware/review/CI follow-ups rather than documentation debt.
