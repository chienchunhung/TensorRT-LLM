# MVP End-to-End Production-Component Prototype Plan

[< Back to Overview](README.md)

**Status:** Active corrected plan; [TRTLLM-12728](https://jirasw.nvidia.com/browse/TRTLLM-12728) integration tracked in draft [PR #15801](https://github.com/NVIDIA/TensorRT-LLM/pull/15801) • **Owner:** WideEP FT track • **Last updated:** 2026-06-30

## 1. Goal and evidence boundary

Build a working vertical slice of the corrected MVP on physical GPUs, using real worker processes, a real MoE model and representative serving workload, and the production CUDA/MNNVL, NCCL, MPI, EPLB, PyExecutor, and request-lifecycle paths. The prototype exists to expose integration defects early, guide the owning PRs, and make realistic hardware testing possible while those PRs are still being reviewed.

Draft [PR #15801](https://github.com/NVIDIA/TensorRT-LLM/pull/15801), branch `WideEP-FT/e2e-mvp-prototype`, is the aggregate integration and hardware-test vehicle. It is intentionally not a substitute merge unit for the component PRs and does not claim E2E success while the missing vertical slices below remain open.

[PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) is historical seam-finding scaffolding. Its mocks and stubs helped expose issues such as poisoned `MPI_Finalize`, but it never demonstrated a working end-to-end recovery path. No seam-correctness, output-correctness, scale-independence, or recovery-latency claim from that branch is an MVP proof.

The new prototype is not throwaway mock code. Missing pieces are implemented as production-shaped integration slices, kept aligned with their owning work items, and either moved into or replaced by the reviewed production PRs. Unit-level fault injectors remain useful; a mock communication, membership, placement, or request-recovery component cannot satisfy this prototype.

## 2. Non-negotiable recovery contracts

1. **Detected health is not committed membership.** Watchdogs and 1c.3 publish failure evidence only. They must not mutate the data-plane rank mask directly.
2. **One coordinator owns the commit.** Item 1c.4b orders `detect → abort failed epoch → reconcile evidence → 1b.2a admission → quiesce → prepare EPLB → rebuild survivor control/NCCL → apply graph policy → commit mask + ActiveRankMap + generation → 1c.4c request disposition → resume`.
3. **The running epoch has a bounded escape.** Merged 1a.2 / [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) supplies the launch-time/next-launch mask. Item 1a.8 must give an already-running polling kernel a host/device-visible abort or generation signal and a recoverable return path; the 300-second `trap;` path is not recovery.
4. **All post-failure collectives exclude the failed rank.** Item 1c.3a creates the survivor control communicator and immutable `ActiveRankMap`; 1c.4a applies that membership to attention-DP/PyExecutor management collectives; 1a.7 rebuilds the raw NCCL data communicator over survivors.
5. **Placement is admitted, not assumed.** Item 1b.2a proves, for every layer and expert, that at least one copy survives every admitted single-rank failure and that FT copies occupy distinct declared failure domains. FT mode fails closed when the proof is absent.
6. **A failed epoch produces no client-visible partial result.** Item 1c.4c owns queued, in-flight, retried, rerouted, and failed-request disposition using the contracts from [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) and [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119).
7. **Poisoned MPI is never treated as healthy MPI.** Item 1d.0a prohibits unsafe world collectives and finalization after peer death and supplies deterministic survivor and failed-rank shutdown behavior.
8. **Captured graphs cannot cross a membership generation.** The first prototype may force eager execution. Item 1a.11 owns eager fallback plus generation-scoped CUDA graph invalidation and recapture before production acceptance.

## 3. Prototype stack and ownership

The integration worktree starts from current upstream `main` and stacks the exact source heads of the relevant PRs. Temporary integration commits are allowed only when they are production-shaped and clearly mapped to the owning item below.

| Area | Production foundation | Required prototype completion |
|:---|:---|:---|
| Rank health and launch mask | 1a.1 / #13302; 1a.2 / #13404 | 1a.8 running-kernel escape; 1a.3 binding; 1a.4 watchdog; no direct watchdog-to-mask mutation |
| Survivor data communicator | 1a.7 / #15789 | Real NCCL abort and survivor-only communicator construction; no environment-variable-only substitute |
| EPLB placement | 1b.1 + 1b.2 / #15525; 1b.3 | 1b.2a per-layer/per-expert admission and distinct-failure-domain validation |
| Failure evidence | 1c.1 / #15677; 1c.3 / #15785 | Notification/reconciliation only; destructive peer-death coverage |
| Survivor control membership | Existing 1c.4 model-engine hook | 1c.3a `ActiveRankMap`, 1c.4a degraded ADP/PyExecutor membership, and 1c.4b atomic coordinator |
| Request semantics | #12718 and #13119 | 1c.4c failed-epoch suppression and explicit request disposition |
| Launcher + lifecycle | 1d.0 / #14160 | 1d.1 admits a tested survivor-preserving launcher/runtime mode; 1d.0a owns poisoned-MPI lifecycle and deterministic shutdown |
| Graph policy | Existing eager/graph execution paths | 1a.11 eager fallback, invalidation, and recapture |
| Observability | 1d.2 and 1d.3 / #15788 | Common generation, degraded health, per-event timeline, and recovery metrics |
| Acceptance | Fault-injection driver | 1d.4 intra-node E2E; then 1d.4a rack FABRIC/IMEX process death plus approved inaccessible-peer-memory containment |

The source-of-truth scope, live PR status, and dependencies remain in [the implementation plan](pr-execution/08-implementation-plan.md), [MVP dependency graph](pr-execution/mvp-dependency-graph.md), and [JIRA ledger](pr-execution/jira-work-item-ledger.md). This document defines the integration proof, not a parallel scope list.

## 4. Hardware paths

| Gate | Representative platform | CUDA shareable-handle path | What it proves |
|:---|:---|:---|:---|
| **1d.4 intra-node** | x86_64 DGX/HGX B200 or B300 with NVSwitch/NVLink; supported H100 configuration may also be useful | TRT-LLM's current `MnnvlMemory` path selects `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` on x86_64 | Real kernels, watchdog, process death, survivor MPI/NCCL/ADP membership, EPLB placement, request semantics, and continued service on one node |
| **1d.4a rack acceptance** | Grace/aarch64 GB200 or GB300 NVL72, or equivalent rack fabric | `CU_MEM_HANDLE_TYPE_FABRIC`, working IMEX, and a lab-approved peer-memory-invalidation/device-loss method | Production rack membership and real process death, plus an inaccessible-peer-memory case proving survivor-context containment or the Q3 fail-closed boundary |

DGX/HGX B200 and B300 do contain NVSwitch. Their current x86_64 TRT-LLM memory-sharing path can nevertheless use POSIX file descriptors; NVSwitch presence alone does not imply that IMEX/FABRIC handles are exercised. Conversely, a successful x86 intra-node run does not prove the Grace/aarch64 FABRIC/IMEX path. Both gates are required for the stated production target.

Before a 1d.4a run, verify the driver-supported IMEX configuration, container device exposure, and a real `CU_MEM_HANDLE_TYPE_FABRIC` export/import round trip. Record driver, CUDA, container, firmware, topology, and IMEX versions with the test evidence.

## 5. Model, placement, and fault scope

Use a real supported MoE model that exercises the selected `NVLinkOneSided` backend and representative request scheduling. The final acceptance configuration should be close enough to the production model and workload to expose CUDA graph, attention-DP, load-balancing, and request-lifecycle interactions; a tiny model is useful only as an earlier bring-up stage.

Do not configure or document FT through a blanket “replication ≥ 2” assumption. For the canonical DeepSeek-V3 shape, 256 experts with EP=72 and four slots per rank provide 288 slots—only 32 copies beyond the first copy, so at least 224 experts can be singletons. Before serving, 1b.2a must emit an auditable per-layer/per-expert coverage result for the declared failure domain. The run is rejected if any admitted failure leaves an expert without a survivor.

The initial destructive test kills a non-rank-0 worker unless an external front end provides rank-0 failover. The admitted MVP scope is one worker/rank failure at a time; a second failure during recovery is a separate state-machine test and must not be implied by the first acceptance.

## 6. Required execution sequence

1. **Baseline.** Start the real server and workload; confirm admitted placement, common generation, healthy survivor/control communicators, correct outputs, and stable throughput.
2. **Inject.** Kill a real worker process at a controlled point during dispatch, combine, a non-MoE NCCL collective, and an iteration boundary in separate runs.
3. **Detect and abort.** Record independent host/watchdog/MPI/NCCL evidence. Signal 1a.8 so any running polling kernel returns without `trap;`. Mark the epoch failed; do not publish a new committed mask yet.
4. **Reconcile and admit.** Reconcile a common failed-rank set on the surviving control path. Run 1b.2a against that set and fail closed if coverage is not proven.
5. **Quiesce and prepare.** Stop new launches, prepare EPLB placement, create 1c.3a/1c.4a survivor control membership, abort/rebuild the 1a.7 NCCL communicator over the same survivors, and apply 1a.11 eager fallback plus stale-graph invalidation.
6. **Commit.** Item 1c.4b atomically publishes one mask, immutable `ActiveRankMap`, and membership generation only after placement, control, data-plane, and graph-policy participants report readiness.
7. **Dispose and resume.** Item 1c.4c suppresses all failed-epoch output, preserves queued work where safe, and applies explicit retry/reroute/error policy to in-flight requests. Resume only on the committed generation.
8. **Verify.** Confirm correct post-recovery tokens, continued HTTP-visible service, no collective includes the dead rank, no poisoned-MPI finalization hang, common survivor state, and expected degraded throughput.

## 7. Evidence and acceptance

Every run writes a machine-readable event trace containing at least:

- process IDs, physical/logical ranks, topology, model/config digest, and admitted placement result;
- fault issue time and injection point;
- per-detector evidence and reconciliation time;
- running-kernel abort request and recoverable-return time;
- epoch-aborted, quiesced, placement-ready, control-ready, NCCL-ready, and graph-invalidated events;
- atomic mask + `ActiveRankMap` + generation commit, request disposition, and first resumed launch;
- disposition of every queued/in-flight request from the failed epoch;
- first correct post-recovery response, steady-state recovery, and shutdown completion.

The production-component prototype succeeds only when all of the following hold:

1. No communication, placement, membership, recovery, or request-disposition component is mocked or stubbed.
2. A real worker dies while a representative workload is active; surviving workers do not exit or hang.
3. A running data-plane kernel escapes in bounded time without poisoning the CUDA context.
4. No partial or zero-filled failed-epoch result reaches a client, and every request has an explicit recorded disposition.
5. All survivors use one `ActiveRankMap`, mask, and generation; no post-failure collective addresses the dead rank.
6. Correct requests complete after recovery and continued service is visible through the normal serving interface.
7. Placement admission, latency, correctness, throughput, and shutdown evidence are retained for review.
8. The intra-node run satisfies 1d.4. The separate NVL72/equivalent FABRIC/IMEX run satisfies 1d.4a only after both real process death and an approved inaccessible-peer-memory/device-loss case are recorded; if the latter cannot recover without poisoning survivors, Q3 remains fail-closed. A prototype run informs those gates but does not replace reviewed production code.

No sub-10-second production guarantee is claimed until the physical acceptance data supports it. Detection timeout is only one term; kernel escape, reconciliation, admission, communicator rebuild, graph policy, and request recovery all contribute to the measured interval.

## 8. Relationship between draft PR #15801 and historical PR #14198

Draft #15801 supersedes #14198 as the integration vehicle. The historical branch may contribute test-driver mechanics, event-log ideas, and the observation that poisoned MPI shutdown needs explicit ownership. The following conclusions are explicitly invalid as proof because they were derived from mocked or isolated paths:

- direct watchdog → `mark_failed` is a safe commit path;
- failure broadcast is off the recovery critical path;
- an iteration-boundary generation check is merely a low-cost hook;
- detection timing or integration behavior is scale-independent;
- timeout selection is the only meaningful latency control;
- completed-request output correctness was established; or
- mocked seam execution proves production-component interoperability.

Those questions are reopened and answered only by the execution and evidence requirements above.
