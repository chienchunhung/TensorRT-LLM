# 10. Risks and Open Questions

[< Back to Overview](README.md)

## Technical Risks

### Risk 1: NVLink Kernel Modification Complexity

**Severity:** High | **Probability:** Medium

The NVLink AlltoAll kernels (`moeAlltoAllKernels.h`) are performance-critical CUDA code. Adding rank masking (a conditional branch per rank in the inner loop) could:
- Introduce thread divergence in the dispatch/combine kernels
- Interact unexpectedly with symmetric memory access patterns
- Cause correctness issues if completion flag management has race conditions with masking

**Mitigation:**
- The conditional is a single bit-test (virtually free compared to memory operations)
- Add comprehensive correctness tests before performance testing
- Benchmark rank masking overhead with all ranks active (should be <0.1% overhead)
- Keep the kernel modification minimal — don't restructure the kernel logic

### Risk 2: DeepEP Backend Limitations

**Severity:** Medium | **Probability:** High

DeepEP only supports specific EP sizes ({2,4,8} intranode, {16,32,...,128} internode). After losing a rank, EP sizes like 31 or 71 are not supported. The `mask_buffer_ptr` API is referenced in vLLM's RFC but not in DeepEP's public API.

**Mitigation:**
- Primary target is NVLink backends (GB200/NVL72 primary production path)
- For DeepEP: fall back to NVLink or AllGatherReduceScatter backend on rank failure
- Monitor DeepEP releases for `mask_buffer_ptr` availability
- Engage with DeepSeek team if needed (they have an interest in this capability for their own production)

### Risk 3: Process Group Reconstruction Deadlocks

**Severity:** High | **Probability:** Medium

Destroying and recreating NCCL/NVSHMEM/MPI process groups with a dead rank is inherently risky. This is a classic example of how fault tolerance in layered systems creates recursive complexity: the cleanup path for one layer (DeepEP buffer destruction) requires coordination from the very component that has failed (the dead rank's barrier participation). Specifically:
- `MPI_Comm_split` is collective — requires all ranks in parent comm (including the dead one)
- NCCL abort may not clean up all internal state (NCCL's internal error recovery is best-effort)
- NVSHMEM symmetric memory deallocation requires all peers to release their mappings
- DeepEP `Buffer.__del__` calls `intranode::barrier` which deadlocks if peers are dead — a documented issue that requires explicit `destroy()` calls with careful ordering

**Mitigation:**
- Use MPI error handlers (`MPI_ERRORS_RETURN`) instead of default abort behavior
- Consider ULFM (User-Level Failure Mitigation) MPI extensions for fault-tolerant comm operations
- Implement coordinated teardown: all surviving ranks agree to tear down before any starts
- Phase 2 process group reconstruction only happens **after** Phase 1 has stabilized the system
- Explicit `destroy()` for all DeepEP buffers before process group teardown

### Risk 4: Failure Broadcast Consensus

**Severity:** Medium | **Probability:** Medium

All surviving ranks must agree on which ranks are dead before applying the mask. Split-brain scenarios could cause data corruption (some ranks route to a "dead" rank that's actually just slow).

**Mitigation:**
- Use conservative detection: require both AlltoAll timeout AND MPI worker death confirmation before marking failed
- Implement two-phase failure protocol: (1) suspect → (2) confirmed
- Monotonic failure: once marked dead, cannot be marked active (until Phase 2 with new process group)
- Timeout tuning: prefer longer timeouts over false positives in Phase 1

### Risk 5: EPLB Reconfiguration During Active Serving

**Severity:** Medium | **Probability:** Low

The `reconfigure()` method pauses the EPLB worker and compute threads. If the pause happens at the wrong time (e.g., mid-weight-migration for a different layer), GPU memory could be in an inconsistent state.

**Mitigation:**
- Reconfiguration only happens between forward iterations (model engine iteration boundary)
- EPLB worker thread checks for reconfigure signal at safe points (after completing current layer)
- Emergency reconfigure is designed to be fast — MVP slot remap <10ms, v1 with weight migration <50ms (see §06, §09) — to minimize serving interruption

### Risk 6: MPI `COMM_WORLD` Failure-Poisoning

**Severity:** High | **Probability:** High (default MPI) | **Phase:** 1c

When a rank in `MPI_COMM_WORLD` dies, subsequent collectives on that communicator fail (or hang) on most common MPI implementations. TRT-LLM's `MPIDist` runs over `MPI.COMM_WORLD` (`tensorrt_llm/_torch/distributed/communicator.py:612`) — naive "use MPI for out-of-band failure broadcast" without further engineering hits exactly the failure mode we are trying to escape, just at a different layer. The April 2026 source review confirmed there is **no failure-tolerant communicator infrastructure in TRT-LLM today** (no `MPI_ERRORS_RETURN` handler, no `MPI_Comm_revoke`, no ULFM wiring, no dedicated FT subcomm).

**Mitigation:** Addressed by PR 1c.3 (dedicated MPI FT subcomm with `MPI_ERRORS_RETURN`, non-blocking `Isend`/`Irecv`+`Test` on a dedicated CPU thread, opportunistic ULFM `MPI_Comm_revoke`). See [§07 Option A](07-failure-detection.md#option-a-out-of-band-via-mpi-failure-tolerant-subcomm-preferred) for the protocol and [§09 PR 1c.3](09-implementation-plan.md) for the implementation contract.

### Risk 7: NCCL Fault-Tolerance Not Wired

**Severity:** Medium | **Probability:** High | **Phase:** 1a (v1)

The April 2026 source review found **zero uses** of `ncclCommAbort`, `NCCL_ASYNC_ERROR_HANDLING`, `ncclCommFinalize`, or `ncclGetLastError` outside test files in TRT-LLM. The only NCCL integration is via `torch.classes.trtllm.NcclCommunicatorOp` (P2P send/recv with no error hook). The original assumption that the AllGatherReduceScatter fallback "eventually times out via NCCL" is **false** for TRT-LLM as-shipped.

**Mitigation:** Addressed by PR 1a.7 (wires `NCCL_ASYNC_ERROR_HANDLING=1` + `ncclCommAbort` + watchdog into the NCCL wrapper *before* enabling AllGatherReduceScatter as a mask-capable fallback). Until 1a.7 lands, backend-switch on rank failure routes to a different NVLink backend rather than NCCL. PR 1d.1 adds a feature-flag validator that warns if `enable_wide_ep_fault_tolerance=True` is configured against AllGatherReduceScatter as primary.

### Risk 8: PR #12718 Sequencing Dependency

**Severity:** Medium | **Probability:** High | **Phase:** 1c

PR #12718's commits (`f32efd01e5`, `e3f84ceb02`, `1128c0ff54`, `4aab3c0afc`) introduce `tensorrt_llm/_torch/pyexecutor/error_classification.py` (`ErrorBudget` dataclass, `classify_error()` function returning string literals). These commits are **not on the `docs-and-plans` branch HEAD** as of 2026-04-21. PRs 1c.1–1c.4 import from `error_classification.py` and assume `ErrorBudget` exists.

**Mitigation:** Addressed in [§09 Prerequisites](09-implementation-plan.md) — the MVP implementation base branch must have PR #12718 merged or rebased in; otherwise a drop-in `ErrorBudget` + `classify_error()` shim is built under `_torch/pyexecutor/` and reconciled when #12718 lands. Status is tracked weekly during MVP execution.

### Risk 9: Memory Pressure During Degraded Mode

**Severity:** Low | **Probability:** Low

Surviving ranks absorb extra experts, consuming additional GPU memory. In memory-tight deployments, this could cause OOM.

**Mitigation:**
- Memory impact is small (~140 MB per rank in FP8 for DeepSeek-V3 losing 1/72 ranks)
- For memory-constrained deployments: reduce EPLB replication factor during degraded mode
- Monitor GPU memory utilization and alert if approaching limits
- GB200 (192 GB HBM) has ample headroom

## Open Design Questions

### Q1: Should Phase 1 use kernel-side or host-side timeout?

**Kernel-side timeout:**
- Pros: Self-contained, no additional thread, precise per-rank detection
- Cons: Requires kernel modification, less flexible, harder to debug

**Host-side watchdog:**
- Pros: No kernel change, configurable at runtime, easier to debug
- Cons: Additional thread, polling overhead, detection latency depends on poll interval

**Current recommendation:** Start with host-side watchdog for Phase 1 (simpler, lower risk). Add kernel-side timeout as an optimization in Phase 2/3 if needed.

### Q2: What happens to in-flight requests during Phase 1 recovery?

**Option A: Fail the current batch, retry on next iteration**
- Pros: Simplest, guaranteed consistency
- Cons: All requests in the current batch fail, even those not routed to the dead rank

**Option B: Partial batch completion — only fail requests routed to dead rank**
- Pros: Minimizes impact on unaffected requests
- Cons: Complex to implement (need to track per-token routing), may have consistency issues

**Current recommendation:** Option A for Phase 1. The current batch is already in an inconsistent state (AlltoAll didn't complete). Failing it and starting fresh with the new mask is simpler and safer. The latency impact is one batch worth of requests (~10-50 requests depending on batch size).

### Q3: How should the failure timeout be tuned?

| Deployment | Recommended Timeout | Rationale |
|:-----------|:-------------------|:----------|
| NVL72 (single rack) | 2-3s | NVLink latency is microseconds; any timeout beyond 1s indicates real failure |
| Multi-node NVLink + RDMA | 5-10s | RDMA has occasional transient delays; need to avoid false positives |
| Development/testing | 1s | Fast detection for iteration speed |

The timeout should be configurable via environment variable (`TRTLLM_EP_FT_TIMEOUT_SEC`) and/or `MoeConfig` field.

### Q4: Should we support DeepEP rank masking or only NVLink?

**NVLink-only (Phase 1):**
- Covers GB200/NVL72 (NVIDIA's primary production hardware)
- We own the kernel code — full control over modifications
- Unblocked by external dependencies

**DeepEP when available (Phase 2+):**
- Wait for `mask_buffer_ptr` in public API
- Multi-node deployments beyond NVL72 may use DeepEP
- Engage with DeepSeek team for timeline

### Q5: What is the maximum number of simultaneous rank failures we should support?

This depends on the redundant expert count:
- With 0 redundant experts: **0 failures** (every expert is unique to one rank)
- With 32 redundant experts (DeepSeek production): **up to ~4 failures** (depends on expert distribution)
- With 256 redundant experts (SGLang benchmark): **up to 16 failures** (50% of cluster)

**Recommendation:** Design for arbitrary number of failures (bitmask supports up to 64/128). The actual tolerance is determined by EPLB replication configuration at deployment time. Document the relationship between `num_redundant_experts` and failure tolerance.

### Q6: How does WideEP FT interact with pipeline parallelism?

With WideEP + PP (e.g., `tp=32, pp=2, ep=16`), each PP stage has its own EP group. A rank failure affects one PP stage's EP group but not the other's.

**Challenge:** PP requires lockstep batch processing across stages. If one EP group enters degraded mode (reduced expert computation capacity) but the other doesn't, the batch must still flow through both stages. This creates a **cross-stage capacity coupling problem**: the degraded stage becomes the bottleneck, and the healthy stage must throttle to match — effectively propagating a single EP rank's failure into a system-wide throughput reduction that exceeds the proportional loss. The interaction between PP's lockstep requirement and EP's partial-failure tolerance is a non-trivial distributed systems design challenge.

**Current recommendation:** Treat each PP stage's EP group independently. If one stage loses a rank, that stage enters degraded mode. The batch size may need to be reduced to match the degraded stage's capacity. This is an advanced configuration that can be addressed in Phase 2.

### Q7: How does WideEP FT interact with disaggregated serving?

Production TRT-LLM disaggregated serving separates **prefill** and **decode** into independent worker pools, each with its own EP group, connected via a KV cache transceiver (NIXL / UCX / MPI). The current design implicitly assumes aggregated serving — a single EP group handling both phases — and does not address the disagg case.

**Why disagg needs its own design:**

- **Independent EP groups.** A prefill pool running `ep=32` and a decode pool running `ep=16` are two separate collectives. Rank masking and EPLB adaptation apply *within* each pool, but a failure in one pool does not propagate to the other through any shared collective. Detection and recovery must be pool-local.
- **Request state is split across pools.** At the moment of failure:
  - **Prefill rank dies mid-prompt processing:** the request's prompt context (tokenized input, early KV cache being generated) is on the dead rank. The decode pool has no KV cache for this request yet — nothing to recover; the request is lost and must be resubmitted.
  - **Decode rank dies mid-generation:** the request's in-progress generation state (partial output tokens, KV cache for both prompt and generated tokens) is on the dead rank. The prefill pool has already completed its work and moved on. Recovery requires either dropping the request, restarting from the prompt (if the prompt is still available upstream), or partial-output recovery if the orchestrator streamed tokens out.
- **In-flight KV cache transfers fail separately.** If the transceiver is mid-transfer when a rank dies, the transfer protocol (NIXL/UCX) surfaces its own failure — not an EP-level failure. The orchestration layer must correlate transfer failures with the underlying rank failure.
- **Orchestration layer must coordinate.** `trtllm-serve`'s disagg router is the only component that sees both pools. It is the natural site for cross-pool failure handling — retry policy, request rerouting to a healthy decode pool, KV cache invalidation. This is a separate codepath from the collective-level recovery in this design.

**What this design does cover in a disagg context:**

- Within each pool (prefill *or* decode), the Phase 1 MVP + Phase 1 full scope apply unchanged — the pool's EP group detects the failure, masks the dead rank, and continues serving at reduced capacity. From each pool's point of view, the design works identically to aggregated.

**What this design does not cover:**

- Cross-pool failure propagation
- In-flight request recovery when state is split across a failing pool boundary
- KV cache transfer failure handling correlated with EP-level failure
- Orchestration-layer retry / rerouting policy

**Recommendation (updated):** Disagg FT is **in scope** but on a deferred track. The primary Phase 1 track (MVP → v1) covers aggregated serving first to keep the critical path focused. Disagg FT lands as **Phase 1-DS** (see [§09](09-implementation-plan.md#phase-1-ds-disaggregated-serving-ft-p1)), which:

- Starts **after** Phase 1 MVP delivers per-pool survival on NVLinkOneSided
- Runs **in parallel** with Phase 1 v1 and does not block it
- Reuses the same Phase 1 primitives (EPGroupHealth, rank masking, EPLB emergency-mask) per-pool, unchanged
- Adds the cross-pool coordination layer in the orchestrator (`trtllm-serve` proxy): KV transceiver failure correlation, cross-pool failure notification, retry/reroute policy

This keeps the basic WideEP FT critical path clean while ensuring disagg is not orphaned.

## Risk Summary Matrix

The **Residual** column reports the risk level expected to remain *after* the mitigation lands, accounting for execution risk, external dependencies, and items consciously accepted as out-of-scope. Readers should prioritize rows where Residual is Medium or higher.

| Risk | Severity | Probability | Phase | Mitigation Status | Residual |
|:-----|:---------|:------------|:------|:------------------|:---------|
| NVLink kernel complexity (kMaxRanks=64, kernel 300s `trap;`) | High | Medium | 1a | Bump kMaxRanks to 128; gate `check_timeout`; comprehensive testing | **Low** — absorbed by PR 1a.2; in-repo kernel, fully in our control |
| DeepEP limitations | Medium | High | 1a | NVLink primary; DeepEP secondary | **High (accepted)** — deferred indefinitely pending public `mask_buffer_ptr` |
| PG reconstruction deadlocks | High | Medium | 2a | Coordinated teardown; ULFM MPI | **Medium** — novel work; execution risk realizes in Phase 2, not MVP |
| Failure broadcast consensus | Medium | Medium | 1c | Conservative detection; two-phase protocol | **Low** — monotonic-failure invariant + suspect/confirm protocol |
| EPLB reconfigure timing | Medium | Low | 1b | Iteration boundary only; safe points | **Low** — design constraint enforced by model-engine hook |
| **MPI `COMM_WORLD` failure-poisoning** | High | High | 1c | Dedicated FT subcomm, `MPI_ERRORS_RETURN`, non-blocking Isend/Irecv+Test on dedicated thread; opportunistic ULFM | **Low–Medium** — ULFM availability depends on linked MPI build; single-failure MVP survives without ULFM |
| **NCCL fault-tolerance not wired** | Medium | High | 1a (v1) | PR 1a.7 resized S→M; wire `ncclCommAbort` + `NCCL_ASYNC_ERROR_HANDLING` before AllGatherReduceScatter mask path | **Low** — fully in our control; v1 scope |
| **PR #12718 sequencing dependency** | Medium | High | 1c | Rebase onto #12718 or build drop-in `ErrorBudget` shim; track weekly | **Medium** — external dependency on #12718 merge cadence; shim is a workaround, not a substitute |
| Memory pressure | Low | Low | 1d | Small impact; monitor + alert | **Low** — headroom is ample on GB200 |
| False positive failure detection | Medium | Medium | 1c | Conservative timeouts; confirmation step | **Low–Medium** — tuning reduces but does not eliminate; monotonic-failure means a false positive permanently masks a live rank until Phase 2 |
| PP + WideEP interaction | Medium | Low | 2+ | Defer to Phase 2 | **Medium (deferred)** — cross-stage capacity coupling is a real design problem, unaddressed in Phase 1 |
| Disagg + WideEP FT interaction | Medium | Medium | Separate track (Phase 1-DS) | Per-pool coverage works unchanged; cross-pool coordination added in Phase 1-DS | **Low** once Phase 1-DS lands; **Medium** in the interval between MVP and DS completion |
