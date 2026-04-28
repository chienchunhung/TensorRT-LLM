# 9. Risks and Open Questions

[< Back to Overview](README.md)

## 9.1 Named audits (gating risks)

Two audits are called out as named risks because they gate downstream work and their outcomes will meaningfully shift the design. Both are scoped bounded — they're 1–2 week prototyping passes, not open-ended research.

### Audit 1 — MNNVL / NVSHMEM teardown capability

**Severity × Probability:** High × Medium | **Phase:** 2 | **Residual risk:** Medium (novel work; outcome gates Phase 2 sizing)

**Why it's named.** Phase 2's `< 1 s` recovery target assumes MNNVL fabric-memory teardown + re-allocate + handle re-exchange is achievable in the ~100 ms range. The audit confirms or refutes this empirically. NVSHMEM is a secondary audit target tied to the (deferred) DeepEP scope.

**Structured in two phases by hardware dependency.** Most of the audit work does not need NVL72 rack access and can start immediately on any ≥ 4-GPU node. A smaller set of items is specifically about rack-fabric behavior and needs NVL72 (or equivalent) time. Splitting this way lets Phase 1a surface most findings on commodity hardware, so Phase 1b rack time is efficient validation rather than from-scratch prototyping.

#### Audit 1a — Intra-node (can start immediately on ≥ 4-GPU node)

**Scope:** ~1 week, one engineer. Any DGX-class node with ≥ 4 NVLink-connected GPUs (H100 / A100 / B200). Does **not** require NVL72 access.

**Empirical findings (Days 1–3 complete, Days 4–5 + DeepEP/NVSHMEM gated on IMEX / container / nvshmem unblockers):** see [audit-1a-findings.md](audit-1a-findings.md) for the long-form Day-by-Day write-up, or the condensed version in [`redesign-research-pass-report.md` § Empirical follow-up](redesign-research-pass-report.md#-empirical-follow-up--audit-1a-partial-item-7). Runnable prototypes + 2 sample result files in [`research-pass-prototypes/`](research-pass-prototypes/).

| Day | Work | Output |
|:---|:---|:---|
| 1 | NCCL abort + reinit prototype with SIGKILL fault injection. Measure `ncclCommAbort` + new-comm-init latency on N=4. | Empirical NCCL rebuild latency; confirms PyTorch `destroy_process_group` / `init_process_group` pattern works against our NCCL version. |
| 2 | MPI signal handler replacement prototype. Test the `_exit(2)` variant from [§5.4](05-phase-1-immediate-survival.md#54-mpi-path-ft-enabling-work). | Mechanism de-risked for PR 1d.0. |
| 3 | `cuMemUnmap` semantics on dead-peer regions. Isolation test: `cuMemCreate` with posix handle type (not fabric), map cross-process, SIGKILL owner, test unmap on survivors. | Core CUDA driver behavior documented independent of fabric specifics. |
| 3 | DeepEP destructor behavior. Construct `Buffer`, kill one rank, observe `__del__` → `intranode::barrier` deadlock on gc. Test explicit `destroy()` ordering (proposed in PR 2a.4). | Verified mitigation for known deadlock; sizes PR 2a.4 realistically. |
| 4–5 | **Intra-node MNNVL teardown + reallocate prototype.** 4-GPU node, `MnnvlMemory` allocated symmetrically, small AlltoAll workload. SIGKILL one process mid-collective. Measure: (a) whether `cuMemUnmap` of dead peer's fabric region segfaults / hangs / succeeds; (b) full teardown latency; (c) rebuild via new fabric-handle exchange on N-1 survivors; (d) correctness AlltoAll on new workspace. | Partial MNNVL rebuild validation. Result generalizes to rack fabric with caveats (see Audit 1b). |
| 5 | NVSHMEM teardown / `nvshmem_finalize` behavior on shipping version. | Version-specific notes for any future DeepEP / NVSHMEM work. |
| 5 | Written report: what's validated, what's pending NVL72 access. | Sizes Phase 2 PRs within ±20 % uncertainty; inputs Audit 1b. |

**What Audit 1a definitively answers:**
- NCCL rebuild latency and correctness.
- Signal handler replacement mechanism.
- `cuMemUnmap` behavior under owner death.
- DeepEP destructor deadlock mitigation.
- Intra-node MNNVL rebuild latency and mechanism.
- NVSHMEM teardown semantics on shipping version.

**Output gates Phase 2 sizing** within ±20 %. Precise PR 2a.2 estimate still waits for Audit 1b but Phase 2 v0 planning can proceed against Audit 1a results.

#### Audit 1b — Rack-fabric validation (pending NVL72 access)

**Scope:** ~2–3 days of NVL72 time, executed *after* Audit 1a so rack time is validation rather than from-scratch prototyping.

**Why it's separate.** What distinguishes NVL72 fabric memory from intra-node NVLink is the rack-scale fabric — direct P2P between GPUs on *different* nodes via NVSwitch's fabric manager. Some failure behaviors plausibly differ:

- NVSwitch fabric manager's cleanup path when a rack member disappears (may or may not differ from intra-node NVLink cleanup).
- Page table / handle invalidation across fabric boundaries vs within one node.
- Scale-specific issues at 72 ranks (e.g., `kMaxRanks=128` layout, 72×72 completion-flag table interaction with fabric-page caching).

**What Audit 1b must confirm:**
1. Intra-node MNNVL teardown results from Audit 1a still hold when peers are across the fabric, not just across a local NVLink.
2. Rebuild latency at 72-rank scale matches the intra-node 4-rank extrapolation (or doesn't — flag scaling artifacts).
3. No rack-fabric-specific failure mode is surfaced (e.g., fabric manager retrying on a dead member indefinitely).

**Deliverable:** single empirical number for Phase 2 sizing: "MNNVL rebuild on NVL72 with 1 rank failed completes in X ms in the median, Y ms in the tail." This resolves the provisional-sizing caveat on PR 2a.2.

#### Combined deliverable

After both 1a and 1b land: empirical answer to "MNNVL rebuild on the NVL72 fabric is a 100 ms operation / 1 s operation / not feasible on this version." Sizes [§8.2 PR 2a.2](08-implementation-plan.md#2a--process-group-reconstruction) definitively.

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

## 9.2 Technical risks

### Risk — NVLink kernel modification complexity

**Severity × Probability:** High × Medium | **Phase:** 1a (MVP) | **Residual:** **Low** — absorbed by PR 1a.2; kernel is in-repo, fully in our control

The kernel mask change touches performance-critical CUDA synchronization. Potential issues: thread divergence, memory ordering interactions with MNNVL symmetric memory, races on mask read. Mitigated by:

- Minimal kernel change (one bit-test per rank, in an outer loop already iterating over ranks).
- Correctness tests before performance tests.
- < 0.1 % steady-state overhead gate with all ranks active.
- Kernel source already reviewed; `kMaxRanks` bump is single-line; mask plumbing is additive.

### Risk — DeepEP backend limitations

**Severity × Probability:** Medium × High | **Phase:** 1a | **Residual:** **High (accepted)** — deferred indefinitely pending public `mask_buffer_ptr`

DeepEP only supports specific EP sizes ({2,4,8} intra-node, {16,32,...,128} inter-node); post-failure EP=71 isn't supported. The `mask_buffer_ptr` parameter referenced in vLLM's RFC #27774 is not in DeepEP's public API.

Not a blocker for MVP — NVLinkOneSided is the primary target. Feature flag ([PR 1d.1](08-implementation-plan.md)) warns if DeepEP is the selected backend when FT is enabled.

### Risk — Process-group reconstruction deadlocks

**Severity × Probability:** High × Medium | **Phase:** 2a | **Residual:** **Medium** — novel work; execution risk realizes in Phase 2, not MVP

Per-layer cleanup paths can deadlock on dead peers. DeepEP's `Buffer.__del__` calls `intranode::barrier`; NCCL abort cleanup is best-effort; MPI `MPI_Comm_split` is collective. Mitigated by coordinated teardown (all survivors agree before any starts), explicit `destroy()` sequencing on DeepEP, `MPI_ERRORS_RETURN` on MPI, and opportunistic ULFM. The MNNVL audit above (Audit 1) covers the MNNVL-specific variant.

### Risk — Failure broadcast consensus (false positives)

**Severity × Probability:** Medium × Medium | **Phase:** 1c | **Residual:** **Low–Medium** — tuning reduces but does not eliminate; monotonic-failure means a false positive permanently masks a live rank until Phase 2

Split-brain scenarios (rank A thinks rank B is dead, rank B is still running) could corrupt routing. Mitigated by conservative detection (both timeout AND MPI-worker-death confirmation before marking failed), two-phase suspect → confirm for v1 multi-failure, monotonic failure policy.

### Risk — EPLB reconfigure during active serving

**Severity × Probability:** Medium × Low | **Phase:** 1b | **Residual:** **Low** — design constraint enforced by model-engine hook

`reconfigure_mask_only` pauses EPLB worker + compute threads. If the pause lands at the wrong time (mid-weight-migration for a different layer), GPU memory could be inconsistent. Mitigated by iteration-boundary-only invocation + safe-point polling in worker threads.

### Risk — MPI `COMM_WORLD` failure-poisoning (Mode A persistence)

**Severity × Probability:** High × High | **Phase:** 1c, 1d.0 | **Residual:** **Low–Medium** — ULFM availability depends on MPI build; single-failure MVP survives without ULFM

Already mitigated in the design via 1d.0 (signal handler replacement) and 1c.3 (FT subcomm). ULFM is a further mitigation for multi-failure; its availability depends on the MPI build (opt-in in OpenMPI, patchy in MVAPICH). For single-failure MVP, non-ULFM path is sufficient.

### Risk — NCCL fault-tolerance not wired in custom ops

**Severity × Probability:** Medium × High | **Phase:** 1a (v1) | **Residual:** **Low** — fully in our control; v1 scope

Zero non-test uses of `ncclCommAbort` in TRT-LLM. PR 1a.7 wires it before AllGatherReduceScatter becomes a mask-capable fallback path. `torch.distributed`'s PyTorch-inherited abort path is unaffected.

### Risk — PR #12718 sequencing dependency

**Severity × Probability:** Medium × High | **Phase:** 1c | **Residual:** **Medium** — external dependency on #12718 merge cadence

PRs 1c.1–1c.4 import from `tensorrt_llm/_torch/pyexecutor/error_classification.py` which #12718 introduces. Mitigated by: (a) rebasing #12718 into the FT implementation base branch, or (b) a drop-in `ErrorBudget` + `classify_error()` shim that gets reconciled when #12718 lands. Tracked weekly during MVP execution.

### Risk — PR #13119 error-propagation dependency

**Severity × Probability:** Medium × Medium | **Phase:** 1c, Phase 1-DS | **Residual:** **Low–Medium** — merged into `main`, but streaming and hard-postproc-death paths still need audit

PR #13119 makes request-scoped failures observable (`GenerationResultBase.error`, `ErrorResponse` from postprocessing, preserved HTTP response bodies, disagg ID regeneration). WideEP FT relies on this distinction: request failures must be returned to callers, while rank / engine failures mark health and trigger failover. Mitigations: keep PR #12718's `RequestError` / `str` filter when extending `_drain_error_queue()` to per-rank tracking, add disaggregated end-to-end error-body tests before Phase 1-DS, and audit streaming SSE paths so errors become structured `data: ...` events rather than unstructured stream crashes.

### Risk — detection visibility gap in `RemoteMpiCommSessionClient`

**Severity × Probability:** High × Medium | **Phase:** 1c | **Residual:** **Medium** — Layer 1 watchdog is mandatory for this deployment shape

`trtllm-llmapi-launch` / `mgmn_leader_node` uses `RemoteMpiCommSessionClient`, whose `submit()` returns `[]` because workers are managed in a separate process. PR #12718's `_check_mpi_futures()` has no local future handles to inspect in that path. The bench-shutdown regression exposed this empty-list behavior: the sentinel must still be sent even when `mpi_futures` is empty. For WideEP FT, Layer 2 worker-death detection is inert in this path; Layer 1 AlltoAll watchdog and explicit health broadcast are mandatory.

### Risk — hung-rank detection without process exit

**Severity × Probability:** High × High | **Phase:** 1a, 1c | **Residual:** **Medium** — covered by Layer 1 watchdog, not by PR #12718 alone

PR #12718 detects completed MPI futures and queued background errors. It does not detect a rank that is alive but stuck in a CUDA/NCCL/MPI collective. This is the exact Mode B risk: kernels can spin indefinitely waiting for a dead peer's flag. Mitigations: host-side AlltoAll watchdog with bounded polling (§5.3 Layer 1), per-step timing markers around `EPGroupHealth.mark_failed()` / broadcast / `reconfigure_mask_only`, and eventual main-thread polling for NCCL/torch distributed operations (Audit 1a showed watchdog modes either terminate or hang in PyTorch 2.11).

### Risk — Memory pressure in degraded mode

**Severity × Probability:** Low × Low | **Phase:** 1d | **Residual:** **Low** — headroom is ample on GB200

Surviving ranks absorb extra tokens per AlltoAll. For DS-V3 / EP=72 losing 1 rank: ~1.4 % extra compute per rank, small memory impact. On 192 GB HBM GB200, headroom is comfortable.

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

### Q1 — Kernel-side vs host-side timeout

Chosen: **host-side watchdog** for MVP. Simpler, runtime-configurable, easier to debug. Kernel-side `clock64()` timeout is an optimization candidate for v1 if host watchdog latency is unacceptable.

### Q2 — Policy for in-flight requests during Phase 1 recovery

Chosen: **Option A — fail the current batch, retry on next iteration.** The batch is already in inconsistent state; failing and restarting with new mask is simplest. Latency impact is one batch of requests. Option B (partial-batch completion, fail only tokens routed to dead rank) is more complex and has consistency risks; not worth it for MVP.

### Q3 — Failure timeout tuning

Configurable per deployment via `TRTLLM_EP_FT_TIMEOUT_SEC` env or config field:

| Deployment | Recommended | Rationale |
|:---|:---|:---|
| NVL72 single rack | 2–3 s | NVLink latency is microseconds |
| Multi-node + RDMA | 5–10 s | RDMA tail latencies are real |
| Dev / CI | 1 s | Iterate fast |

### Q4 — DeepEP support

Chosen: **NVLink-only for MVP.** DeepEP requires `mask_buffer_ptr` in public NVSHMEM API; not available. If NVIDIA exposes the API, DeepEP masking becomes a v1 or post-v1 item.

### Q5 — Maximum simultaneous failures

Depends on replication factor:
- 0 redundant experts: **0 failures** tolerable.
- 32 redundant experts (DeepSeek production): ~4 failures tolerable, depending on replica distribution.
- 256 redundant experts (SGLang benchmark config): up to 16 failures (50 % cluster loss).

Bitmask supports up to 128 ranks; actual tolerance is determined by `num_redundant_experts` at deployment time. Relationship between replication factor and failure tolerance documented in the feature's user-facing docs.

### Q6 — WideEP + pipeline parallelism interaction

With `tp=32, pp=2, ep=16`, each PP stage has its own EP group. A failure in one stage doesn't cross into the other via collective; but PP's lockstep batch processing creates a cross-stage capacity coupling problem — the degraded stage becomes the bottleneck. Recommendation: treat each PP stage's EP group independently; accept throughput reduction at the lockstep level. Advanced configuration; Phase 2+ item.

### Q7 — WideEP FT × disaggregated serving

In scope as Phase 1-DS ([§8.2](08-implementation-plan.md#phase-1-ds--disaggregated-serving-ft)). Per-pool FT from the primary track applies unchanged within each pool; Phase 1-DS adds cross-pool coordination. Ray + disagg + NIXL is a hard gap (see above); Phase 1-DS on MPI first, Ray follows if the gap closes.

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
- If the worker process dies, CUDA/NCCL reports an immediate-fatal condition, or the AlltoAll watchdog times out a rank, mark the rank failed and enter Phase 1 recovery.

Open item: streaming SSE helpers must be audited so they follow the same boundary (structured error event + `[DONE]`, not a process-fatal path).

## 9.4 Risk summary matrix

| Risk | Severity | Probability | Phase | Mitigation | Residual |
|:---|:---|:---|:---|:---|:---|
| MNNVL/NVSHMEM audit outcome | High | Medium | 2a | Audit 1 | **Medium** — gates Phase 2 sizing |
| Ray-path perf uncharacterized | Medium | High | Future migration | Audit 2 | **Medium–High** — covered when Audit 2 runs |
| Ray + disagg + NIXL unsupported | Medium | High | Phase 1-DS / future | Close gap upstream; ship on MPI first | **Medium** — hard gap, closes with upstream fix |
| NVLink kernel modification | High | Medium | 1a | PR 1a.2 minimal change; correctness-first | **Low** |
| DeepEP limitations | Medium | High | 1a | NVLink primary; DeepEP deferred | **High (accepted)** |
| PG reconstruction deadlocks | High | Medium | 2a | Coordinated teardown; explicit destroy(); ULFM | **Medium** |
| Failure broadcast consensus | Medium | Medium | 1c | Two-phase suspect/confirm; monotonic failure | **Low** |
| EPLB reconfigure timing | Medium | Low | 1b | Iteration-boundary only | **Low** |
| **MPI `COMM_WORLD` poisoning (Mode A)** | High | High | 1c, 1d.0 | Signal handler replacement + FT subcomm | **Low–Medium** |
| NCCL FT not wired | Medium | High | 1a (v1) | PR 1a.7 | **Low** |
| PR #12718 sequencing | Medium | High | 1c | Rebase or shim | **Medium** |
| PR #13119 error propagation | Medium | Medium | 1c / Phase 1-DS | Preserve request-vs-fatal boundary; add disagg e2e tests | **Low–Medium** |
| RemoteMpiCommSessionClient detection visibility | High | Medium | 1c | Layer 1 watchdog mandatory; explicit empty-futures handling | **Medium** |
| Hung rank without process exit | High | High | 1a / 1c | AlltoAll watchdog + bounded polling | **Medium** |
| Memory pressure (degraded) | Low | Low | 1d | Small impact; ample GB200 headroom | **Low** |
| False positive detection | Medium | Medium | 1c | Conservative timeouts + confirmation | **Low–Medium** |
| Second failure during rebuild | Medium | Medium | 2a.8 | Abandon rebuild, re-mask, retry | **Medium** |
| HostMoeTensorSharer MPI hard-bake | Medium | High | Future migration | Refactor before Ray pivot | **Medium** |
| PP + WideEP interaction | Medium | Low | 2+ | Defer to Phase 2 | **Medium (deferred)** |

Bolded rows are the ones warranting active tracking during MVP execution.
