# 9. Risks and Open Questions

[< Back to Overview](README.md)

## 9.1 Named audits (gating risks)

Three audits are called out as named risks because they gate downstream work and their outcomes will meaningfully shift the design. Each is bounded prototyping or characterization, not open-ended research.

### Audit 1 — Baseline MNNVL teardown and rack containment capability

**Severity × Probability:** High × Medium | **Phase:** 2 | **Residual risk:** Medium (novel work; outcome gates Phase 2 sizing)

**Why it's named.** Phase 2's accelerated target assumes MNNVL teardown + reallocation + handle exchange is fast enough, while MVP 1d.4a needs evidence for FABRIC/IMEX peer-memory containment. The audit measures both. NVSHMEM is only a secondary, conditional target when DeepEP is selected.

**Structured in two phases by hardware dependency.** Most of the audit work does not need NVL72 rack access and can start immediately on any ≥ 4-GPU node. A smaller set of items is specifically about rack-fabric behavior and needs NVL72 (or equivalent) time. Splitting this way lets Phase 1a surface most findings on commodity hardware, so Phase 1b rack time is efficient validation rather than from-scratch prototyping.

#### Audit 1a — Intra-node (can start immediately on ≥ 4-GPU node)

**Scope:** ~1 week, one engineer. Any supported DGX/HGX-class node with ≥ 4 NVLink-connected GPUs (for example H100 or B200/B300). DGX/HGX B200/B300 uses NVSwitch for intra-node NVLink connectivity, but the current x86_64 TRT-LLM `MnnvlMemory` path selects POSIX-FD shareable handles. It does **not** require NVL72 access and does **not** validate the Grace/aarch64 FABRIC/IMEX path.

**Historical empirical findings (isolated Days 1–3 complete; production-component Days 4–5 remain open):** see [audit-1a-findings.md](audit-1a-findings.md) for the corrected evidence boundary, or the condensed historical record in [`redesign-research-pass-report.md`](redesign-research-pass-report.md). Runnable micro-prototypes and sample results live in [`research-pass-prototypes/`](research-pass-prototypes/); they are not MVP E2E evidence.

| Day | Work | Output |
|:---|:---|:---|
| 1 | NCCL abort + reinit micro-prototype with SIGKILL fault injection. | The high-level PyTorch `destroy_process_group` / re-init attempt did **not** demonstrate recovery after peer death. It motivated 1a.7's lower-level raw-NCCL abort and survivor-only rebuild path. |
| 2 | MPI signal-handler replacement micro-prototype. Test the `_exit(2)` variant from [§5.4](05-phase-1-immediate-survival.md#54-mpi-path-ft-enabling-work). | Signal-time propagation de-risked for merged 1d.0 / #14160; poisoned world collectives and finalization remain 1d.0a. |
| 3 | `cuMemUnmap` semantics on dead-peer regions. Isolation test: `cuMemCreate` with POSIX-FD handle type, map cross-process, SIGKILL owner, test unmap on survivors. | Documents only the local POSIX-FD CUDA-driver case; it does not establish FABRIC/IMEX behavior. |
| 3 | DeepEP destructor behavior. Construct `Buffer`, kill one rank, observe `__del__` → `intranode::barrier` deadlock on gc. Test explicit `destroy()` ordering (proposed in PR 2a.4). | Verified mitigation for known deadlock; sizes PR 2a.4 realistically. |
| 4–5 | **No-mock intra-node production-component recovery prototype.** 4+ GPUs, real `MnnvlMemory`, real AlltoAll/NCCL/MPI/EPLB/PyExecutor paths, real model/workload, and SIGKILL during active serving. Measure running-kernel escape, survivor communicator/membership rebuild, request correctness, teardown, and continued service. On x86_64 this normally exercises POSIX-FD sharing. | 1d.4 intra-node evidence. It does not generalize to rack FABRIC/IMEX; 1d.4a is separate. |
| 5 | NVSHMEM teardown / `nvshmem_finalize` behavior on shipping version. | Version-specific notes for any future DeepEP / NVSHMEM work. |
| 5 | Written report: what's validated, what's pending NVL72 access. | Provisional Phase 2 inputs; confidence is re-estimated only after the no-mock intra-node and rack runs. |

**What the completed isolated work answers:**

- The attempted high-level PyTorch NCCL shrink path does not recover from peer death; it does not measure the 1a.7 survivor rebuild.
- The signal-handler replacement removes explicit handler `MPI_Abort`; the tested default launcher still propagated abnormal exit, leaving 1d.1 launcher admission and 1d.0a lifecycle work.
- POSIX-FD `cuMemUnmap` under owner death succeeds in the tested micro-case only.
- The inspected DeepEP destructor contains a peer-dependent cleanup risk; production mitigation remains unproven.

The isolated work does **not** answer real MNNVL recovery, NVSHMEM teardown under the target workload, request correctness, scale independence, or FABRIC/IMEX behavior. Those claims require the no-mock 1d.4 and 1d.4a paths.

**Output informs Phase 2 sizing.** The production-component 2a.0a intra-node run supplies provisional Phase 2 sizing within ±20%; Audit 1b/2a.0b validates rack-fabric extrapolation and supplies the final 2a.2 ship decision.

#### Audit 1b — Rack-fabric validation (pending NVL72 access)

**Scope:** ~2–3 days of NVL72 time, executed *after* Audit 1a so rack time is validation rather than from-scratch prototyping.

**Why it's separate.** What distinguishes NVL72 fabric memory from intra-node NVLink is the rack-scale fabric — direct P2P between GPUs on *different* nodes via NVSwitch's fabric manager. Some failure behaviors plausibly differ:

- NVSwitch fabric manager's cleanup path when a rack member disappears (may or may not differ from intra-node NVLink cleanup).
- Page table / handle invalidation across fabric boundaries vs within one node.
- Scale-specific issues at 72 ranks (e.g., `kMaxRanks=128` layout, 72×72 completion-flag table interaction with fabric-page caching).

**What Audit 1b must confirm:**
1. The production-component intra-node recovery/teardown results, once completed, still hold when peers are across the fabric rather than only inside one node.
2. Rebuild latency at 72-rank scale matches the intra-node 4-rank extrapolation (or doesn't — flag scaling artifacts).
3. Under ordinary process death, FABRIC/IMEX membership and recovery behave correctly.
4. Under a lab-approved IMEX-grant revocation, GPU reset/isolation, or equivalent inaccessible-peer-memory injection, survivor CUDA contexts remain contained and recover—or Q3 is explicitly recorded as fail-closed/restart.

**Deliverables:** (a) median/tail MNNVL rebuild latency for Phase 2 sizing, and (b) a 1d.4a containment verdict for both process death and inaccessible-peer-memory/device loss. A process-death timing number alone cannot close Q3.

#### Combined deliverable

After both 1a and 1b land: empirical answer to "MNNVL rebuild on the NVL72 fabric is a 100 ms operation / 1 s operation / not feasible on this version." Sizes [§8.2 PR 2a.2](pr-execution/08-implementation-plan.md#2a--process-group-reconstruction) definitively.

**Mitigation if worse than expected.** If MNNVL rebuild is > 1 s in the best case, Phase 2's sub-second claim softens to "multi-second." Shadow+GMS still provides most of the win (weight load time dominated, ~100 ms), but the overall Phase 2 target moves.

**Sequencing benefit.** Running 1a before 1b means rack time is ~2–3 days of targeted validation rather than 1–2 weeks of prototyping. Rack access is scarce and expensive; arrive with a working intra-node prototype.

### Audit 2 — Ray-path WideEP perf characterization

**Severity × Probability:** Medium × High | **Phase:** Future-migration decision | **Residual risk:** Medium (gates Ray pivot, doesn't affect MVP)

**Why it's named.** [§3.3](03-failure-modes-and-gaps.md#33-why-not-just-pivot-to-ray) decides to stay on MPI for MVP partly because Ray-path WideEP is not characterized at scale. If we ever revisit the pivot, the audit is the empirical input.

**Scope.** Once Ray-path tests exist at EP ≥ 32, run a benchmark comparison:

1. DeepSeek-V3 serving on `mpirun -np 72 trtllm-serve …` baseline (MPI path).
2. Same config on `orchestrator_type=ray` (Ray path).
3. Metrics: throughput (tok/s), latency (p50/p99), per-iteration AlltoAll latency, steady-state overhead.
4. Target: Ray-path within `Z%` of MPI-path (pick a threshold — 5 % is a reasonable starting point).

**Output:** empirical basis for a future pivot decision. If Ray is within threshold, pivot becomes viable and the compensating MPI-path FT work ([§5.4](05-phase-1-immediate-survival.md#54-mpi-path-ft-enabling-work)) becomes redundant for future features. If Ray is not within threshold, pivot is blocked on closing the perf gap first.

**Pre-requisites that make the audit possible:** Ray-path CI needs EP ≥ 32 tests first. Today largest is TP = 4 (research pass report). So the audit itself is 1–2 weeks *after* Ray-path test coverage is built out.

### Audit 3 — NIXL-EP evaluation as cross-IB data-plane backend

**Severity × Probability:** Medium × Medium | **Phase:** Phase 1-IB (cross-IB transport only) | **Residual risk:** Medium (outcome shapes Phase 1-IB feasibility, not NVL72 MVP)

**Why it's named.** The NIXL team has built **NIXL-EP** (NIXL's expert-parallel example at `ai-dynamo/nixl/examples/device/ep`), and vLLM has integrated it as an Elastic-EP backend (vLLM PR [#35627](https://github.com/vllm-project/vllm/pull/35627), merged 2026-03-13). For TRT-LLM, NIXL-EP is interesting **specifically for the cross-IB transport role** — replacing DeepEP/DeepEPLowLatency in deployments where MNNVL is not in the path (multi-node B200+IB and similar). It is *not* a candidate to replace `NVLinkOneSided` on NVL72 today; that primary backend stays on the kernel-mask architecture.

**Why it's not MVP.** Multi-node B200+IB is not yet a primary production deployment; the kernel-mask MVP closes the gap for the NVLink-substrate footprint (single-node NVL boxes + GB200/GB300 NVL72 rack) and PR 1a.7 closes the NCCL fallback. NIXL-EP serves the third transport regime (DeepEP-family / cross-IB), which is Phase 1-IB scope.

**Verified API surface (from vLLM PR #35627 + NIXL EP README):**

```python
buffer = nixl_ep.Buffer(rank, tcp_store_group=...)
buffer.update_memory_buffers(num_ranks, num_experts_per_rank, num_rdma_bytes, num_nvl_bytes=0)
buffer.connect_ranks(remote_ranks, activate=True)   # incremental add; activate=False = LL-mode masked
buffer.disconnect_ranks(remote_ranks)               # incremental remove
buffer.dispatch(...)
buffer.combine(...)
```

Key properties:
- **Topology mutation, not rank masking.** FT model is `disconnect_ranks([dead])` → topology becomes N-1, not "keep dead rank in topology with bit masked off." This collapses Phase 1 + Phase 2 into one "scale-down then scale-up" path for this transport.
- **`torch.distributed.TCPStore` dependency.** Means NIXL-EP cannot ride a pure-MPI orchestrator path; needs either Ray, or TCPStore wired alongside MPI.
- **MNNVL substrate landed** (ai-dynamo/nixl#1415, merged 2026-04-05). **But not yet production-validated at 72-rank scale** — ai-dynamo/nixl#1655 (open 2026-05-19) is still adding 4-GPU single-node topology test coverage. So even if we wanted to use NIXL-EP on NVL72 someday, it isn't ready.
- **RDMA + NVLink transports** (from NIXL EP README). Currently no de-duplication optimizations per itayalroy's comment on PR #35627.

**Scope.** ~2 weeks, one engineer, runnable in parallel with MVP (no critical-path dependency):

| Step | Work | Output |
|:---|:---|:---|
| E1 | Technical-fit assessment — does NIXL-EP fit `CommunicationFactory` as a 6th backend (sibling to DeepEP)? What's the TCPStore-alongside-MPI co-existence story, or does the deployment need to switch to Ray orchestrator? | Integration sketch + orchestrator decision |
| E2 | FT primitive validation — measure `disconnect_ranks([dead]) + connect_ranks([replacement])` latency at 4-GPU + 8-GPU + 16-GPU + 32-GPU scales (matching B200+IB deployment ranges). Confirm topology mutation works mid-iteration with a quiesce point. | Recovery-latency curve vs rank count |
| E3 | Perf comparison vs DeepEPLowLatency (the current cross-IB production backend) — bandwidth, latency, kernel launch overhead, MoE-layer round-trip. Calibrate against Peiheng's 94 µs NVFP4 number. | Quantitative comparison |
| E4 | Maturity assessment — version stability, MNNVL roadmap (when is ai-dynamo/nixl#1655 + #1499 + #1530 expected to complete?), NVIDIA-internal NIXL-team support story for cross-IB at production scale. | Risk register entry |
| E5 | Write-up + go/no-go recommendation — integrate as Phase 1-IB primary path (PRs 1a.9/1a.10), defer (stay on DeepEP 100s-timeout interim IB.1), or decline. | Decision document |
| E6 | **(new)** Phase 2 impact assessment — confirm that topology mutation collapses Phase 1 + Phase 2 for this transport. What does "Phase 2 = scale-up via `connect_ranks(replacement)`" mean for our §6 design? | Phase-2 design delta |

**Integration scope if E5 is positive.** Two new PRs land Phase 1-IB's primary path: **1a.9** (NIXL-EP `CommunicationFactory` strategy + factory registration) and **1a.10** (NIXL-EP topology-mutation + FT coordinator integration). Sizes M + M; both depend on PR 1a.1 (committed membership), the NIXL-EP version selected in E2, and the TCPStore-alongside-MPI or Ray-pivot decision from E1.

**Strategic value if E5 is positive.** Phase 1-IB ships on NIXL-EP with `disconnect_ranks` + EPLB redistribute as the recovery mechanism. The DeepEP FT gap stops being "blocking for multi-node B200+IB" because NIXL-EP replaces DeepEP for that transport. Backend priority order for cross-IB deployments becomes: 1) NIXL-EP (preferred for new FT-aware cross-IB deployments), 2) DeepEPLowLatency (legacy / NCCL-domain), 3) AllGatherReduceScatter (safety net).

**Strategic value if E5 is negative.** Phase 1-IB ships on the DeepEP 100s-timeout interim (IB.1). Bounded cost (~2 engineer-weeks for the audit), no MVP impact, no impact on NVL72 deployment path. Plus E6 still gives us a calibrated answer on whether to even consider topology mutation as a Phase 2 architecture for NVL72 down the road.

## 9.2 Technical risks

### Risk — NVLink kernel modification complexity

**Severity × Probability:** High × High | **Phase:** 1a (MVP) | **Residual:** **High** until draft 1a.8 / #15895 is validated and merged — merged 1a.2 supplies a launch-time mask, not a running-kernel recovery path

The kernel mask change touches performance-critical CUDA synchronization. Merged 1a.2 / #13404 copies the rank mask into launch parameters, so a kernel already polling a failed peer does not observe a later host-side mask update; its remaining escape is the roughly 300-second `trap;` path, which can poison the CUDA context. Potential issues include thread divergence, memory ordering interactions with MNNVL symmetric memory, races on abort/generation reads, and partial output from an aborted epoch. Mitigations are:

- Keep the merged launch-time bit test as the next-launch foundation.
- Validate and merge draft #15895: its stable host/device-visible execution-epoch control must be observed inside every running polling loop, expose sticky completion status to the host, return recoverably within a bounded interval, and avoid `trap;`.
- Suppress the entire failed epoch through 1c.4c; a recoverable kernel return is not valid model output.
- Correctness and destructive process-death tests before performance tests.
- < 0.1 % steady-state overhead gate with all ranks active.

### Risk — Detection state can race committed data-plane membership

**Severity × Probability:** Critical × High | **Phase:** 1c (MVP) | **Residual:** **High** until 1c.4b

If watchdog, MPI, or NCCL callbacks mutate `EPGroupHealth` independently, survivors can launch with different masks/generations or update placement before their control/data communicators are ready. The historical mock's direct watchdog → `mark_failed` path is explicitly invalid. Detection/suspicion state remains separate from committed communication state.

**Mitigation:** 1c.4b is the only committed-membership writer and owns `detect → abort failed epoch → reconcile evidence → validate admission → quiesce → prepare EPLB → rebuild survivor control/NCCL → apply graph policy → commit mask + ActiveRankMap + generation`; 1c.4c applies request disposition before resume. Readiness and common-generation assertions are required in 1d.4/1d.4a.

### Risk — Aggregate expert slots do not guarantee a survivable placement

**Severity × Probability:** Critical × High | **Phase:** 1b (MVP) | **Residual:** **High** until 1b.2a

For the canonical DeepSeek-V3 shape, 256 experts with EP=72 and four slots per rank provide 288 slots—only 32 extra copies—so at least 224 experts can be singletons. Even multiple copies do not provide FT if they share the same admitted failure domain. Mask-only reconfiguration cannot serve an expert that has no surviving resident copy.

**Mitigation:** 1b.2a validates every layer/expert against every admitted single-rank failure, enforces distinct failure-domain placement for FT copies, and fails closed at startup and recovery when the invariant is absent. Memory/capacity estimates are evaluated only after admission succeeds.

### Risk — Ordinary control and attention-DP collectives still include the dead rank

**Severity × Probability:** Critical × High | **Phase:** 1c (MVP) | **Residual:** **High** until 1c.3a and 1c.4a

A dedicated failure-notification thread does not repair the normal rank-state, request, batch-size, token-count, or model-input gathers. Any such collective over the original membership can hang after the next iteration begins, even when the MoE data-plane mask is correct.

**Mitigation:** 1c.3a creates a survivor-only control communicator and immutable logical-to-physical `ActiveRankMap`; 1c.4a applies it to attention-DP/PyExecutor management collectives. 1d.4 asserts that no post-failure collective addresses the dead rank.

### Risk — Failed-epoch output can leak to clients

**Severity × Probability:** Critical × Medium | **Phase:** 1c (MVP) | **Residual:** **High** until 1c.4c

A timeout, early-returning kernel, or zero-filled rank contribution may let a partially computed batch enter postprocessing. Restarting on the next iteration does not define what happens to queued and in-flight requests, and a completed HTTP response can silently carry corrupt logits.

**Mitigation:** 1c.4c marks the epoch aborted before recovery, guarantees no failed-epoch output becomes externally visible, preserves queued work only when safe, and records explicit retry/reroute/error disposition for every in-flight request using the #12718/#13119 contracts.

### Risk — CUDA graphs retain stale pointers and membership

**Severity × Probability:** Critical × Medium | **Phase:** 1a (MVP) | **Residual:** **High** until 1a.11

Captured graphs can retain old communicator, workspace, mask, and rank-map assumptions. Replaying a graph after recovery may use stale pointers even when host-side state shows the new generation.

**Mitigation:** 1a.11 makes recovery enter eager mode, invalidates every graph bound to the old generation, and recaptures only after the new membership and communicators are committed. The first no-mock prototype may force eager mode but production acceptance must exercise the final graph policy.

### Risk — NIXL-EP integration timing

**Severity × Probability:** Medium × Medium | **Phase:** Phase 1-IB | **Residual:** **Medium** — gated on Audit 3 outcome

The NIXL team has built NIXL-EP and vLLM already uses it as an FT-enabled backend. For TRT-LLM, the open question is whether its verified incremental `disconnect_ranks` / `connect_ranks` topology mutation fits cleanly into `CommunicationFactory`, the coordinator, and EPLB for the Phase 1-IB cross-IB path. It is not `activeRanks` masking and it is not an NVL72 replacement. Two risks remain: integration/version/performance complexity, and letting the cross-IB coverage gap persist if the evaluation is deferred.

**Mitigation:** Audit 3 ([§9.1](#audit-3--nixl-ep-evaluation-as-cross-ib-data-plane-backend)) is a bounded two-week parallel evaluation that produces a go/no-go for Phase 1-IB. If positive, the conditional 1a.9/1a.10 work implements that transport track; if negative, the NVL72 corrected-MVP path is unchanged. NIXL-EP does not gate the NVL72 MVP.

### Risk — DeepEP backend limitations (applies to cross-IB transport deployments only)

**Severity × Probability:** Medium × High | **Phase:** Phase 1-IB | **Residual:** **Medium–High** — was "deferred indefinitely accepted"; scope sharpened to "applies when DeepEP-family is the selected L3 transport"

DeepEP only supports specific EP sizes ({2,4,8} intra-node, {16,32,...,128} inter-node); post-failure EP=71 isn't supported. The `mask_buffer_ptr` parameter referenced in vLLM's RFC #27774 is not in DeepEP's public API. `Buffer.__del__` → `intranode::barrier` deadlock is a known issue (acknowledged at `deep_ep.py:86`).

**Not a blocker for the NVL72 MVP only because that release fails closed on unsupported DeepEP-family routes.** The corrected NVLink-substrate path requires the full MVP recovery transaction—not just 1a.2 and 1a.7—including 1a.8/1a.11, admitted EPLB placement, survivor control/ADP membership, atomic commit, request disposition, poisoned-MPI lifecycle, and 1d.4/1d.4a. The DeepEP-family transport applies when MNNVL is unavailable (cross-IB/cross-fabric peers) and remains Phase 1-IB scope. Item 1d.1 must reject or explicitly divert that backend when the NVL72 FT mode is enabled.

**Applies to [Phase 1-IB](pr-execution/08-implementation-plan.md#phase-1-ib--cross-ib-transport-coverage-nixl-ep-track) deployments** (multi-node B200+IB, multi-rack non-NVLink-fabric, anything where `CommunicationFactory` falls through to DeepEP-family). Two mitigation paths:

- **IB.1 (interim).** Host-side static kernel timeout (vLLM PR #38534's 100s "FT-enabled backend" pattern). Softer than `trap;`; doesn't require NVSHMEM-side changes; doesn't bound recovery latency tightly.
- **IB.2 (preferred if Audit 3 positive).** Substitute NIXL-EP for DeepEP. NIXL-EP exposes `connect_ranks` / `disconnect_ranks` for incremental topology mutation; Audit 3 must measure our recovery bound rather than inheriting the external ~3-second claim.

The IB.2 path is preferred when Audit 3 outcome is positive; IB.1 is the fallback when it isn't.

### Risk — Process-group reconstruction deadlocks

**Severity × Probability:** High × Medium | **Phase:** MVP + 2a | **Residual:** **Medium** — survivor-only rebuild is MVP; replacement-inclusive rebuild extends it in Phase 2

The MVP already risks deadlock while 1c.3a/1c.4a build survivor control membership and 1a.7 aborts/reinitializes supported raw NCCL communicators; those operations must be coordinator-ordered and fail closed. Phase 2 extends the risk to replacement-inclusive MNNVL/NCCL/bootstrap work. DeepEP `Buffer.__del__`/NVSHMEM teardown is conditional on selecting that backend, not a baseline MVP dependency. Use `MPI_ERRORS_RETURN`, bounded operations, the last valid control path, and opportunistic ULFM where available. Audit 1 sizes MNNVL reconstruction; 1d.4 validates the survivor path.

### Risk — NVSwitch fabric manager behavior under mid-collective rank death

**Severity × Probability:** High × Medium | **Phase:** MVP 1d.4a + Phase 2a | **Residual:** **Medium** — gated on destructive rack acceptance, cross-team engagement, and Audit 1b

When an MNNVL domain member disappears, the NVSwitch fabric manager's reaction is unspecified from outside the fabric-manager team. Possible behaviors: cleanly invalidate routes (good); retry communication with the dead rank indefinitely (bad — can mark survivors' fabric mappings as suspect); suspend the whole domain temporarily (worst). Whichever it does, it directly affects whether mid-flight `cuMemUnmap` on dead-peer regions completes cleanly and whether survivors can re-allocate fabric memory in a smaller-N topology.

**Mitigation:** named cross-team dependency in [§9.5](#95-cross-team-dependencies-nvidia-internal); engage the fabric-manager/driver owners before Audit 1b. MVP item 1d.4a must pair process death with a lab-approved inaccessible-peer-memory/device-loss injection and either prove survivor-context containment or retain Q3 fail-closed. Audit 1b also characterizes replacement-era teardown/reallocation for Phase 2.

### Risk — IMEX dynamic re-grant support

**Severity × Probability:** High × Medium | **Phase:** 2a, 2b | **Residual:** **Medium** — fundamentally changes Phase 2 sub-second feasibility if the answer is "no"

For MNNVL on NVL72, fabric memory grants are managed by the `nvidia-imex` daemon. Phase 2 needs IMEX to support: (a) invalidating the dead rank's grants when it dies, (b) issuing new grants to the replacement rank when it joins, both *without* restarting the daemon. **We don't control whether IMEX supports dynamic re-grant.** If IMEX requires a daemon restart, Phase 2 on MNNVL gets multi-second-class even with a pre-staged shadow rank, because daemon restart adds orchestration coordination cost.

**Mitigation:** named cross-team dependency in [§9.5](#95-cross-team-dependencies-nvidia-internal); engage IMEX team before PR 2a.2 starts to either confirm dynamic re-grant works or scope the daemon-restart workaround. If "no" answer: fall back to MX P2P RDMA path (~2 s) as the primary recovery mode and accept that shadow + GMS sub-second is gated on IMEX roadmap.

### Risk — MPI rank-add architecture undefined

**Severity × Probability:** Medium × Medium | **Phase:** 2c | **Residual:** **Medium** — architectural decision pending PR 2c.2 design

Default `mpirun` doesn't natively support adding a rank to a running job. `MPI_Comm_spawn` exists but is complex to wire, and the existing 1c.3 signaling / 1c.3a survivor communicator cannot contain a process that was not already a member. The [§6.2](06-phase-2-full-restoration.md#62-pg-reconstruction) design therefore leaves the mechanism explicit: does the replacement become an MPI peer via `MPI_Comm_spawn`, enter through a pre-staged bootstrap group, or bypass MPI entirely and join NCCL + MNNVL through an external control channel?

**Mitigation:** explicit open design question for PR 2c.2 (Join protocol for new rank entering EP group). Both options are viable; the choice has follow-on implications for how the replacement coordinates with surviving ranks (which collectives go through MPI vs which go directly through NCCL/MNNVL). Settle before PR 2c.2 design freeze.

### Risk — Failure broadcast consensus (false positives)

**Severity × Probability:** Critical × Medium | **Phase:** 1c | **Residual:** **High** until reconciliation and atomic commit are implemented

Split-brain scenarios (rank A thinks rank B is dead, rank B is still running) can corrupt routing. Requiring “timeout AND MPI-worker-death” is not a universal rule: an alive-but-hung rank may never produce worker-death evidence, while an immediate-fatal CUDA/NCCL signal may be authoritative. Detector-specific evidence enters 1c.3 reconciliation; no detector independently commits membership. Item 1c.4b publishes a monotonic common mask + immutable `ActiveRankMap` + generation only after placement, survivor communicators, and graph policy are ready.

### Risk — EPLB reconfigure during active serving

**Severity × Probability:** High × Medium | **Phase:** 1b, 1c | **Residual:** **Medium–High** until 1c.4b coordinates the safe point

`reconfigure_mask_only` pauses EPLB worker + compute threads. If the pause lands at the wrong time (mid-weight-migration for a different layer), GPU memory could be inconsistent. A local iteration-boundary callback is not sufficient if other ranks, NCCL, control membership, or CUDA graphs are still on the previous generation. Item 1c.4b owns the common quiesce and placement/communicator readiness sequence; the existing 1c.4 hook is its model-engine integration point.

### Risk — MPI `COMM_WORLD` poisoning after Q1/Q3 prompt evidence

**Severity × Probability:** High × High | **Phase:** 1c, 1d.0/1d.1/1d.0a | **Residual:** **High** until launcher admission and poisoned-world lifecycle are destructively tested

Merged 1d.0 removes the old handler's explicit `MPI_Abort`, but a launcher/runtime may still terminate the job on abnormal exit; 1d.1 must admit a survivor-preserving mode. Item 1c.3 supplies failure evidence, while neither component makes `COMM_WORLD`, implicit world collectives, or `MPI_Finalize` safe. Item 1d.0a owns poisoned-world policy and deterministic shutdown. ULFM is optional; the non-ULFM path is acceptable only after 1d.0a/1d.4 proof.

### Risk — NCCL fault-tolerance not wired in custom ops

**Severity × Probability:** High × High | **Phase:** 1a (MVP) | **Residual:** **High** until 1a.7 and the common survivor-map integration are complete

The historical audit did not show that a peer-death-poisoned PyTorch process group can be destroyed and reinitialized. Item 1a.7 therefore owns lower-level communicator abort and survivor-only raw-NCCL rebuild for the corrected MVP. It must consume exactly the 1c.3a `ActiveRankMap` committed by 1c.4b; a second independent rank list recreates split-brain risk. `torch.distributed` behavior outside the owned wrapper still requires audit at every call site used after recovery.

### Risk — PR #12718 sequencing dependency

**Severity × Probability:** Medium × Low | **Phase:** 1c | **Residual:** **Low–Medium** — #12718 is merged; integration semantics remain

PR #12718 provides the classification foundation in `tensorrt_llm/_torch/pyexecutor/error_classification.py`. The remaining risk is semantic integration: rank/engine failures must enter failure evidence without charging request-scoped errors, and 1c.4c must preserve that boundary when it disposes the failed epoch. No temporary classification shim should become a second source of truth.

### Risk — PR #13119 error-propagation dependency

**Severity × Probability:** Medium × Medium | **Phase:** 1c, Phase 1-DS | **Residual:** **Low–Medium** — merged into `main`, but streaming and hard-postproc-death paths still need audit

PR #13119 makes request-scoped failures observable (`GenerationResultBase.error`, `ErrorResponse` from postprocessing, preserved HTTP response bodies, disagg ID regeneration). WideEP FT relies on this distinction: request failures must be returned to callers, while rank / engine failures mark health and trigger failover. Mitigations: keep PR #12718's `RequestError` / `str` filter when extending `_drain_error_queue()` to per-rank tracking, add disaggregated end-to-end error-body tests before Phase 1-DS, and audit streaming SSE paths so errors become structured `data: ...` events rather than unstructured stream crashes.

### Risk — detection visibility gap in `RemoteMpiCommSessionClient`

**Severity × Probability:** High × Medium | **Phase:** 1c | **Residual:** **Medium** — Layer 1 watchdog is mandatory for this deployment shape

`trtllm-llmapi-launch` / `mgmn_leader_node` uses `RemoteMpiCommSessionClient`, whose `submit()` returns `[]` because workers are managed in a separate process. PR #12718's `_check_mpi_futures()` has no local future handles to inspect in that path. The bench-shutdown regression exposed this empty-list behavior: the sentinel must still be sent even when `mpi_futures` is empty. For WideEP FT, that future-based detector is inert in this path; zero-collective AlltoAll evidence and the dedicated 1c.3 notification/reconciliation thread are mandatory, followed by 1c.3a/1c.4b—not a direct health broadcast-to-mask shortcut.

### Risk — hung-rank detection without process exit

**Severity × Probability:** Critical × High | **Phase:** 1a, 1c | **Residual:** **High** — detection alone cannot release a running kernel

PR #12718 detects completed MPI futures and queued background errors. It does not detect a rank that is alive but stuck in a CUDA/NCCL/MPI collective. A host-side AlltoAll watchdog can detect lack of progress, but merged 1a.2's launch-time mask cannot change an already-running kernel's polling set. Mitigations: 1a.4 publishes evidence, 1a.8 supplies the running-kernel-visible abort/generation and recoverable return, 1c.4c discards the failed epoch, and 1d.4/1d.4a measure the bounded escape on real hardware.

### Risk — Memory pressure in degraded mode

**Severity × Probability:** High × Medium | **Phase:** 1b, 1d | **Residual:** **Medium** after placement admission and workload measurement

Survivors absorb extra tokens and may need additional resident expert copies. An average `(N-1)/N` load estimate does not establish per-rank HBM, hot-expert capacity, or placement survivability. Item 1b.2a first proves per-layer/per-expert coverage and failure-domain separation; 1d.4/1d.4a then measure worst-survivor memory and throughput on the admitted realistic workload. FT must fail closed rather than rely on nominal GB200 capacity.

### Risk — Second failure during Phase 2 rebuild window

**Severity × Probability:** Medium × Medium | **Phase:** 2a.8 | **Residual:** **Medium** — mitigation is to abandon the rebuild and fall back to Phase 1 + retry

Collective PG rebuild can't survive a second death mid-operation. Mitigated by state-machine transitions ([§6.4](06-phase-2-full-restoration.md#64-second-failure-during-rebuild)): abandon rebuild → Phase 1 mask newly dead rank → retry Phase 2 later. Audit validates whether survivors can recover from a half-completed rebuild.

### Risk — HostMoeTensorSharer MPI hard-bake (blocks Ray pivot)

**Severity × Probability:** Medium × High | **Phase:** Future-migration decision | **Residual:** **Medium** — real engineering work to factor out

Verified: `moe_load_balancer.py:896–897` calls `Split_type(MPI.COMM_TYPE_SHARED)` with no `TLLM_DISABLE_MPI` guard anywhere in the file. On the Ray path, this fails. Any future Ray pivot requires factoring out MPI primitives from `HostMoeTensorSharer` — replace node-local peer discovery with a hostname-based or Ray-placement-group mechanism, audit every reader. Not blocking for MVP (MPI path).

### Risk — Ray-path WideEP perf uncharacterized

**Severity × Probability:** Medium × High | **Phase:** Future-migration decision | **Residual:** **Medium–High** — covered by Audit 2 when it runs

Verified: largest Ray-path test config is TP = 4 (Llama-3.1 8B). No EP ≥ 32 tests, no DS-V3 on Ray, no Ray-vs-MPI perf comparison in regression suite. Pivoting to Ray for FT today would run customer-facing WideEP on a code path we haven't benchmarked at scale. Audit 2 resolves this empirically when the pre-requisite CI coverage exists.

### Risk — Ray + disagg + NIXL unsupported (blocks disagg FT on Ray)

**Severity × Probability:** Medium × High | **Phase:** Phase 1-DS + future-migration | **Residual:** **Medium** — hard gap; needs to be closed before disagg FT can ship on Ray

Verified: explicit waive at `tests/integration/defs/disaggregated/test_disaggregated.py:597` — "Ray orchestrator is not supported with NIXL(DEFAULT) cache transceiver backend." Since NIXL is the production default for disagg, Phase 1-DS on Ray is blocked until this gap closes. Not blocking for MVP (Phase 1-DS ships on MPI).

## 9.3 Open design questions

### Q1 — Kernel-side versus host-side escape

Chosen: **both are required, with separate responsibilities.** The 1a.4 host watchdog detects lack of progress and publishes failure evidence. Item 1a.8 supplies a device/host-visible abort or generation that an already-running kernel can observe and return through in bounded time. A fixed 300-second `clock64()` path ending in `trap;` is neither the recovery mechanism nor a v1-only optimization. Detection still does not authorize mask commit; 1c.4b does.

### Q2 — Policy for in-flight requests during Phase 1 recovery

Required invariant: **no output from the failed epoch reaches a client.** Item 1c.4c defines the exact policy rather than assuming every request can simply retry on the next iteration. It must distinguish queued requests that remain safe, in-flight requests that can be retried or rerouted without violating API semantics, and requests that must return an explicit error. Partial-batch completion is out of MVP unless it can prove the same invariant. The policy and every request disposition are tested through the normal serving interface.

### Q3 — Failure timeout tuning

Configurable per deployment through the unified WideEP FT configuration owned by 1d.1; avoid creating a second undocumented environment-variable source of truth. Initial values are hypotheses, not release defaults:

| Deployment | Recommended | Rationale |
|:---|:---|:---|
| NVL72 single rack | Measure in 1d.4a | NVLink latency alone does not bound scheduling, kernel, or fabric-manager stalls |
| Multi-node + RDMA | Measure in Phase 1-IB acceptance | Transport and workload tails differ from NVL72 |
| Dev / CI | Short, test-specific value | Fast deterministic injection is useful but is not a production recommendation |

The selected value must balance false positives against the full recovery SLO. It does not replace a bounded 1a.8 kernel escape.

### Q4 — DeepEP support

Chosen: **NVLinkOneSided plus supported NCCL for MVP.** Direct DeepEP masking/rebuild requires an upstream NVSHMEM/DeepEP primitive that is not available. Cross-IB deployments are not silently dropped: the conditional Phase 1-IB track evaluates NIXL-EP topology mutation as the preferred path and a limited DeepEP timeout interim as fallback evidence. Full direct DeepEP FT remains conditional on upstream support.

### Q5 — Maximum simultaneous failures

MVP admits **one rank failure only when 1b.2a proves it for every layer and expert**. Aggregate redundancy is not a failure-tolerance calculation. For DeepSeek-V3 with 256 experts, EP=72, and four slots/rank, 288 total slots provide only 32 extra copies, leaving at least 224 singleton experts unless the configured placement/model differs. Even duplicated experts are not protected if their copies share the failed rank or failure domain.

The 128-rank bitmask capacity is merely an encoding bound. Actual failure tolerance comes from the per-layer placement, failure-domain anti-affinity, available survivor memory/capacity, communicator support, and the admitted failure set. Multi-failure admission is post-MVP work and requires an explicit proof, not a replica-count estimate.

### Q6 — WideEP + pipeline parallelism interaction

With `tp=32, pp=2, ep=16`, each PP stage has its own EP group. A failure in one stage doesn't cross into the other via collective; but PP's lockstep batch processing creates a cross-stage capacity coupling problem — the degraded stage becomes the bottleneck. Recommendation: treat each PP stage's EP group independently; accept throughput reduction at the lockstep level. Advanced configuration; Phase 2+ item.

### Q7 — WideEP FT × disaggregated serving

In scope as Phase 1-DS ([§8.2](pr-execution/08-implementation-plan.md#phase-1-ds--disaggregated-serving-ft)). Per-pool FT from the primary track applies unchanged within each pool; Phase 1-DS adds cross-pool coordination. Ray + disagg + NIXL is a hard gap (see above); Phase 1-DS on MPI first, Ray follows if the gap closes.

### Q8 — When to revisit the Ray pivot

Framework: revisit when all three of the following hold:

1. Ray-path WideEP perf characterization (Audit 2) completes with acceptable results.
2. `HostMoeTensorSharer` MPI hard-bake has been factored out.
3. Ray + disagg + NIXL support gap has been closed.

Until all three land, MPI path remains the default.

### Q9 — Error propagation vs failover trigger boundary

Chosen: **request-scoped errors stay request-scoped; rank / engine failures trigger failover.**

PR #13119 intentionally improves request-level propagation: context-server errors, postprocessing exceptions, malformed disaggregated responses, and HTTP error bodies should flow back to the caller with the original reason. PR #12718 intentionally filters `RequestError` / `str` and adds `_handle_errors(charge_budget=False)` for request-scoped paths so those same errors do not consume the process-fatal budget. WideEP FT inherits that boundary:

- If the request is bad or the context response is invalid, fail the request and keep the EP group healthy.
- If the worker process dies, CUDA/NCCL reports an immediate-fatal condition, or the AlltoAll watchdog times out a rank, publish failure evidence and enter reconciliation. Only 1c.4b commits the failed rank after admission and survivor readiness.

Item 1c.4c owns failed-epoch disposition and the streaming SSE audit so the same boundary holds for structured error events and stream completion, without turning a request error into process failure or leaking partial failed-epoch output.

## 9.4 Risk summary matrix

| Risk | Severity | Probability | Phase | Mitigation | Residual |
|:---|:---|:---|:---|:---|:---|
| MNNVL/NVSHMEM audit outcome | High | Medium | 2a | Audit 1 | **Medium** — gates Phase 2 sizing |
| Ray-path perf uncharacterized | Medium | High | Future migration | Audit 2 | **Medium–High** — covered when Audit 2 runs |
| **NIXL-EP integration timing (cross-IB transport)** | Medium | Medium | Phase 1-IB | Audit 3 (bounded 2-week parallel evaluation); go/no-go decides Phase 1-IB primary path (IB.2) vs interim (IB.1) | **Medium** — applies only to cross-IB deployments; NVL72 path unaffected |
| Ray + disagg + NIXL unsupported | Medium | High | Phase 1-DS / future | Close gap upstream; ship on MPI first | **Medium** — hard gap, closes with upstream fix |
| **Running-kernel escape after merged launch mask** | Critical | High | 1a MVP | 1a.8 / [#15895](https://github.com/NVIDIA/TensorRT-LLM/pull/15895) recoverable execution-epoch abort/control; 1c.4c epoch suppression; destructive E2E | **High** until #15895 is validated and merged |
| **Detection/commit split-brain** | Critical | High | 1c MVP | 1c.4b sole writer and atomic common generation | **High** until implemented |
| **Expert placement not survivable** | Critical | High | 1b MVP | 1b.2a per-layer/per-expert admission and failure-domain anti-affinity | **High** until implemented |
| **Dead rank retained in control/ADP collectives** | Critical | High | 1c MVP | 1c.3a `ActiveRankMap`; 1c.4a survivor-aware gathers | **High** until implemented |
| **Failed-epoch output reaches client** | Critical | Medium | 1c MVP | 1c.4c explicit request disposition and no-partial-output assertion | **High** until implemented |
| **Stale CUDA graph crosses generation** | Critical | Medium | 1a MVP | 1a.11 eager fallback, invalidate, recapture | **High** until implemented |
| DeepEP limitations (cross-IB transport only) | Medium | High | Phase 1-IB | NIXL-EP via Audit 3 (preferred) or 100s static kernel timeout interim | **Medium–High** — applies only when DeepEP-family is selected transport |
| PG reconstruction deadlocks | High | Medium | 2a | Coordinated teardown; explicit destroy(); ULFM | **Medium** |
| Failure-evidence reconciliation / false positive | Critical | Medium | 1c MVP | 1c.3 reconciliation; 1c.4b commit after admission/readiness | **Medium–High** |
| EPLB reconfigure timing | High | Medium | 1b/1c MVP | 1c.4b common quiesce and readiness; existing 1c.4 hook | **Medium–High** |
| **MPI launcher propagation + `COMM_WORLD` poisoning after Q1/Q3 evidence** | High | High | 1c, 1d.0/1d.1/1d.0a MVP | 1d.0 removes handler abort; 1d.1 admits survivor-preserving runtime; 1c.3 evidence; 1d.0a lifecycle/shutdown | **High** until destructively tested |
| **NCCL survivor communicator not wired** | High | High | 1a MVP | 1a.7 consumes the common 1c.3a survivor map under 1c.4b | **High** until implemented |
| PR #12718 semantic integration | Medium | Low | 1c | Reuse merged classifier; preserve request-vs-rank boundary | **Low–Medium** |
| PR #13119 error propagation | Medium | Medium | 1c / Phase 1-DS | Preserve request-vs-fatal boundary; add disagg e2e tests | **Low–Medium** |
| RemoteMpiCommSessionClient detection visibility | High | Medium | 1c | Zero-collective detector + 1c.3 notification; explicit empty-futures handling | **Medium** |
| Hung rank without process exit | Critical | High | 1a / 1c MVP | 1a.4 evidence + 1a.8 / [#15895](https://github.com/NVIDIA/TensorRT-LLM/pull/15895) release + 1c.4c suppression | **High** until #15895 is validated and merged |
| Memory/capacity pressure in degraded mode | High | Medium | 1b / 1d MVP | 1b.2a admission + realistic 1d.4/1d.4a measurements | **Medium** |
| Second failure during rebuild | Medium | Medium | 2a.8 | Abandon rebuild, re-mask, retry | **Medium** |
| HostMoeTensorSharer MPI hard-bake | Medium | High | Future migration | Refactor before Ray pivot | **Medium** |
| PP + WideEP interaction | Medium | Low | 2+ | Defer to Phase 2 | **Medium (deferred)** |
| **Cross-team coordination (MNNVL stack, NVSHMEM API)** | Medium | Medium | 2a, §7.5 | Engage NVSHMEM / CUDA driver / fabric manager / IMEX teams early; see [§9.5](#95-cross-team-dependencies-nvidia-internal) | **Medium** — depends on external roadmaps |
| **NVSwitch fabric manager behavior under rank death** | High | Medium | 2a | Cross-team engagement (§9.5); Audit 1b empirical characterization | **Medium** — pending audit + external |
| **IMEX dynamic re-grant support** | High | Medium | 2a, 2b | Cross-team engagement (§9.5); MX P2P RDMA fallback if IMEX answer is "no" | **Medium** — fundamentally changes sub-second feasibility |
| **MPI rank-add architecture undefined** | Medium | Medium | 2c | Settle in PR 2c.2 design (Comm_spawn vs bypass MPI) | **Medium** — architectural decision pending |

Bolded corrected-MVP rows are immediate ship risks. Their residual risk is intentionally not shown as low merely because the code is owned in-repo; each remains open until its implementation and physical acceptance evidence exist.

## 9.5 Cross-team dependencies (NVIDIA-internal)

Phase 2 (PG reconstruction over MNNVL) and [§7.5](07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking) (forward-looking straggler / resize work) depend on components owned by NVIDIA teams outside TRT-LLM. This section captures the dependency map for early engagement. Specific team names should be verified through internal channels (NV Slack, Confluence, internal owner lists) since reorg history isn't visible from the doc; the table below is the *external read* — accurate at the component level, but the team-name column should be confirmed before any formal contact.

### Component ownership map (external read; verify internally)

| Component | What it is | Owning org (external read) | Internal verification path |
|:---|:---|:---|:---|
| **NVSHMEM** | GPU OpenSHMEM library; DeepEP rides on it | NVIDIA HPC Software (sibling to NCCL, MPI integration, HPC SDK) | Internal NVSHMEM owner list; `#nvshmem` Slack equivalent |
| **MNNVL — physical fabric** | NVLink + NVSwitch silicon | Hardware / silicon teams | Generally not on our path |
| **MNNVL — fabric manager** | NVSwitch fabric manager daemon | NV Switch / fabric manager team (system software) | Internal NVSwitch / fabric-manager owner list |
| **MNNVL — IMEX daemon** | `nvidia-imex` user-space daemon for cross-node fabric memory grants | System software team (sibling to fabric manager) | Internal driver / sysSW owner list |
| **CUDA driver — fabric handle** | `cuMemCreate(... CU_MEM_HANDLE_TYPE_FABRIC ...)`, `cuMemMap`, `cuMemUnmap` for fabric memory | CUDA driver team (memory management subsystem) | Internal CUDA driver owner list |
| **DeepEP wrappers** | Python + CUDA kernels wrapping NVSHMEM for MoE AlltoAll | **DeepSeek-AI (external; not NVIDIA)** | NVIDIA-DeepSeek collaboration channel |

### Dependencies by phase

**Phase 1 (MVP + v1).** The implementation is primarily owned by TRT-LLM, but the 1d.4a production acceptance gate depends on access to NVL72/equivalent hardware plus a working driver, fabric-manager, and IMEX environment. That operational/resource dependency can block release evidence even when the source changes are complete. The x86_64 intra-node 1d.4 path normally uses POSIX-FD sharing and cannot substitute for it.

**Phase 2 (Restoration).** Three cross-team engagements:

- **CUDA driver team** — fabric handle teardown semantics under peer death. [Audit 1a Day 3](audit-1a-findings.md) (posix-FD variant) showed `cuMemUnmap` of a dead-peer region completes in ~0.25 ms with no fault. Need explicit confirmation that the fabric-handle path matches and that future driver versions preserve the behavior. Engage when [Audit 1b](#audit-1--baseline-mnnvl-teardown-and-rack-containment-capability) is being planned.
- **NVSwitch fabric manager team** — what does the fabric manager do when an MNNVL domain member disappears? Does it interfere with rank-masked AlltoAll, or with our rebuild flow? Audit 1b is partly about answering this empirically; their team's expectations should be cross-checked beforehand. Engage early.
- **IMEX team** — does IMEX support re-exchanging memory grants among surviving members without daemon restart? Engage when planning [PR 2a.2](pr-execution/08-implementation-plan.md#2a--process-group-reconstruction) (MNNVL teardown + reallocate).

**§7.5 (Forward-looking straggler / resize work).** Same MNNVL stack as Phase 2, plus dynamic-resize requirements (adding a rank to a live fabric domain). Higher bar than Phase 2's "rebuild after death."

**Direct DeepEP/NVSHMEM masking (conditional upstream dependency).** Two-sided dependency:

- **NVSHMEM team (NVIDIA)** — `mask_buffer_ptr` public API. Referenced in vLLM's RFC #27774 since 2024 but not yet shipped in public NVSHMEM. NVSHMEM team's roadmap decision.
- **DeepEP team (DeepSeek-AI; external)** — wiring of any new NVSHMEM masking primitive into DeepEP's public API. NVIDIA-DeepSeek collaboration channel.

### Engagement strategy

Three tiered actions:

1. **Now (independent of MVP critical path).** Identify named contacts for NVSHMEM, CUDA driver fabric memory, and IMEX / fabric manager. Open low-key technical conversations about FT requirements. Even informal awareness that TRT-LLM has a Phase 2 / §7.5 roadmap depending on their components helps those teams factor it into their own planning. No formal commitments needed at this stage; the goal is visibility.

2. **Before Audit 1b (NVL72 validation).** Pre-coordinate with the NVSwitch fabric manager + IMEX teams so audit findings can be cross-checked against their expectations. Avoids a "we measured X; you say it should be Y" loop, and gives them advance notice that audit data will surface.

3. **At Phase 2 design freeze.** Convert the informal contacts into named approvers / consultants on the relevant Phase 2 PRs. Particularly PR 2a.2 (MNNVL teardown + reallocate) and PR 2a.0b (NVL72 rack-fabric audit). For DeepEP-related work, no engagement makes sense until upstream NVSHMEM has signaled `mask_buffer_ptr` is on their roadmap.

### Implications for our roadmap

| Class of work | Owners | Our control of timing |
|:---|:---|:---|
| MNNVL kernel rank masking (PR 1a.2) | TRT-LLM only | **High** — we own the kernel |
| MNNVL fabric teardown + rebuild after peer death (PR 2a.2) | CUDA driver + fabric manager + IMEX | **Low–medium** — depends on documenting / extending external behavior |
| MNNVL dynamic resize (live add/remove rank) — §7.5 | Same as above, larger asks | **Low** — likely needs a multi-team roadmap discussion |
| NVSHMEM rank masking via `mask_buffer_ptr` | NVSHMEM team (NVIDIA) + DeepEP (DeepSeek) | **Low** — two-sided external dependency, slow timeline historically |
| NVSHMEM PE recovery / rejoin | NVSHMEM team | **Low** — library roadmap question |
| NCCL FT (`ncclCommAbort` wiring on our side) | TRT-LLM team for wrapper; NCCL team for upstream | **Medium** — wrapper is ours (PR 1a.7); upstream NCCL FT is mature |

Net implication:

- **Anything purely in TRT-LLM source** (most Phase 1 implementation, §5, §6.1, §6.4) — we own the code schedule, but corrected-MVP completion still depends on physical 1d.4/1d.4a evidence and the rack environment for the latter.
- **Anything that touches MNNVL fabric semantics** (Phase 2, §7.5) — needs early external engagement. Worth opening conversations *now*, not when we hit the audit.
- **Anything requiring NVSHMEM API extensions** (DeepEP FT) — soft dependency on someone else's timeline. Treat as out-of-scope until that timeline is established by upstream.
