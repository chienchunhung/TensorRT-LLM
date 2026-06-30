# Historical WideEP FT MVP Prototype — Findings and Corrections

[< Back to Overview](README.md) • [Corrected prototype plan](mvp-prototype-plan.md) • [Audit 1a findings](audit-1a-findings.md)

**Status:** Archived evidence from historical draft [PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198); conclusions corrected against the production design • **Owner:** WideEP FT track • **Last updated:** 2026-06-30

## Evidence boundary

PR #14198 used POSIX shared-memory flags, a Python watchdog, pseudo-AlltoAll work, direct local health mutation, a stub MPI notification path, and simplified reconfiguration. It was useful for finding interface and lifecycle questions, but it was not an end-to-end working prototype and cannot prove the production recovery path.

The corrected [prototype plan](mvp-prototype-plan.md) requires real worker processes, physical GPUs, a real model/workload, and the production CUDA/MNNVL, NCCL, MPI, EPLB, PyExecutor, and request-lifecycle paths. No code or timing result below satisfies 1d.4 or 1d.4a acceptance.

| Historical finding | Current disposition | Owning corrected item |
|:---|:---|:---|
| F1: watchdog reads must not require the failed peer | **Retained, with limited evidence.** The semantic requirement is correct; the POSIX mock did not validate MNNVL behavior. | 1a.4, 1d.4, 1d.4a |
| F2: survivors can hang in `MPI_Finalize` after peer death | **Retained and promoted to explicit MVP work.** | 1d.0a; 1c.3 supplies failure evidence, not shutdown ownership |
| F3: independent detection makes broadcast non-critical | **Invalid as a production conclusion.** Detection is evidence only; reconciliation, common survivor membership, and atomic commit remain on the critical path. | 1c.3, 1c.3a, 1c.4b |
| F4: watchdog timeout is the only meaningful latency knob | **Invalid.** Running-kernel escape, reconciliation, admission, control/data communicator rebuild, graph policy, and request disposition can all dominate. | 1a.8, 1b.2a, 1c.3a, 1c.4a–1c.4c, 1a.11 |
| F5: recovery is scale-independent | **Invalid beyond the mocked 4/8-process loop.** | 1d.4a and scale/performance evidence |
| OQ2: iteration-hook placement is near-noise | **Invalid.** The boundary is a correctness gate: launches must quiesce before a common generation is committed. | Existing 1c.4 hook, 1c.4b, 1a.11 |

## F1. Watchdog observation must not require peer participation

The scaffolding initially considered `MPI.COMM_WORLD.allgather` for completion-flag observation and correctly rejected it: a detector cannot depend on a collective that includes the suspected dead rank. The historical POSIX shared-memory substitute preserved that one property.

The production contract is stronger:

- the observation path is zero-collective and does not require peer progress;
- a watchdog publishes failure evidence rather than updating the committed data-plane mask;
- item 1a.8 provides an independent bounded escape for an already-running polling kernel; and
- 1d.4 exercises the real intra-node path under destructive process death; 1d.4a adds rack FABRIC/IMEX process death plus an approved inaccessible-peer-memory/device-loss case for Q3 containment.

This finding informs 1a.4 review, but the mock did not establish that the real MNNVL mapping remains readable, correctly ordered, or sufficient for recovery after a peer dies.

## F2. Poisoned-world finalization can hang

In the historical eight-process B300 smoke run, the victim received `SIGKILL`; surviving Python interpreters later entered mpi4py's `MPI_Finalize` path and did not complete. This corroborated the earlier isolated MPI audit: removing the signal-handler `MPI_Abort` propagation is necessary but does not make an MPI world healthy after peer death.

The corrected ownership is:

- merged 1d.0 / [#14160](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) replaces the old signal-time abort behavior;
- 1c.3 records/reconciles failure evidence and may expose a poisoned-world signal;
- **1d.0a** owns the lifecycle policy: prohibit unsafe world collectives and finalization after peer death, select the survivor control path, and provide deterministic survivor/failed-rank shutdown; and
- 1d.4 validates the behavior in a real inference process, including implicit teardown collectives and normal/abnormal shutdown variants.

The historical unconditional `os._exit(0)` was a diagnostic workaround, not a production fix. Production behavior must be conditional, observable, and coordinated with resource cleanup; it must not silently turn all shutdowns into abrupt exits.

## F3. Local detector agreement did not prove recovery consensus

In one mocked four-process run, three survivors' local timers reported the same dead peer within roughly 2 ms, before the stub notification callback recorded a separate receive event. That observation proves only that identical local timers can fire at similar times in a quiet test.

The previous conclusions—“broadcast is off the critical path,” “every survivor may call `mark_failed` directly,” and “the iteration hook can pick up any local mark”—are unsafe and withdrawn. Real detectors may disagree, arrive at different times, or observe different evidence. A direct detector-to-mask path can launch different generations, reconfigure placement before communicator readiness, and corrupt the failed epoch.

The corrected flow is:

1. watchdog/MPI/NCCL paths publish evidence and 1a.8 aborts the failed epoch;
2. 1c.3 reconciles the failed-rank set;
3. 1c.4b validates 1b.2a admission, quiesces launches, and prepares EPLB placement;
4. under the same transaction, 1c.3a/1c.4a prepare survivor control membership and an immutable `ActiveRankMap`, 1a.7 rebuilds supported NCCL communicators, and 1a.11 applies eager fallback plus graph invalidation;
5. only 1c.4b atomically commits the common mask, `ActiveRankMap`, and generation; and
6. 1c.4c disposes failed-epoch requests before serving resumes.

Notification can be concurrent; common committed state is still a correctness-critical synchronization boundary.

## F4. Stub timing did not characterize recovery latency

The historical timeout sweep measured approximately `configured Python timer + 100 ms` in a pseudo-workload:

| Stub watchdog timeout | Historical mocked-loop result |
|:---|:---|
| 1 s | 1.10 s |
| 2 s | 2.10 s |
| 5 s | 5.11 s |
| 10 s | 10.13 s |

This is useful only as a check that the timer stub behaved as configured. It does not select a production default or establish recovery headroom. The mock omitted the real running-kernel escape, failed-epoch drain, per-expert admission, survivor MPI/ADP/NCCL rebuild, CUDA graph transition, request disposition, and realistic first-response latency.

Production timeout remains configurable and must be chosen from physical-hardware false-positive and recovery measurements. The event trace in the corrected prototype plan reports every recovery phase separately; no single timeout is assumed to dominate.

## F5. Four-versus-eight stub parity is not scale evidence

The mocked loop produced nearly identical timestamps at four and eight processes because each local timer ran the same code and the omitted recovery work did not scale with rank count. It did not exercise a 72×72 completion table, real MNNVL/NVSwitch traffic, rack FABRIC/IMEX, survivor communicator construction, 58-layer EPLB state, attention-DP gathers, or realistic request scheduling.

Therefore the prior statement that recovery time is scale-independent, and its extrapolation to 72 ranks, are withdrawn. Only 1d.4a and the associated steady-state/recovery measurements can support a rack-scale claim.

## OQ2. Iteration-boundary placement remains a correctness question

The historical run observed its stub generation callback within one synthetic loop interval. That does not make hook placement a performance-only choice. A launch after failure evidence but before quiescence/atomic commit may use stale placement, mask, communicator, or CUDA graph state.

Item 1c.4 remains the existing model-engine health-check hook. Item 1c.4b turns it into the coordinated transition `detect → abort failed epoch → reconcile evidence → validate admission → quiesce → prepare EPLB → rebuild survivor control/NCCL → apply graph policy → commit mask + ActiveRankMap + generation`; item 1c.4c then applies request disposition before resume. Item 1a.11 invalidates stale captures and selects eager execution before the commit; generation-bound graph recapture starts only after the new generation is committed.

## Reopened questions for the production-component prototype

- [ ] Can 1a.8 release a real dispatch/combine kernel in bounded time without `trap;` or CUDA-context poisoning?
- [ ] Do 1a.7 and 1c.3a rebuild NCCL and control communicators over exactly the same survivor map?
- [ ] Does 1c.4a remove the dead rank from every attention-DP/PyExecutor management collective?
- [ ] Does 1b.2a prove every layer/expert remains served for the injected failure, rather than relying on aggregate slot count?
- [ ] Does 1c.4c suppress every failed-epoch output and give every queued/in-flight request an explicit disposition?
- [ ] Does 1d.0a avoid poisoned-world collectives and complete deterministic shutdown?
- [ ] Do eager fallback and 1a.11 prevent any stale CUDA graph from crossing the committed generation?
- [ ] What are the measured phase-by-phase latency and false-positive rates on intra-node NVSwitch and on rack FABRIC/IMEX?

## Status

Historical PR #14198 remains archived as seam-finding evidence and should not be resumed as the integration vehicle. Build the new integration worktree from current upstream `main`, stack the exact production PR heads, implement missing production-shaped slices under their final item IDs, and record new results separately. The corrected prototype plan and the source-of-truth implementation/dependency documents supersede all scheduling and completion claims that appeared here before 2026-06-30.
