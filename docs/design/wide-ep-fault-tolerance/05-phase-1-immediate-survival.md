# 5. Phase 1: Immediate Survival

[< Back to Overview](README.md)

This section unifies all Phase 1 mechanics. Phase 1 keeps an *admitted* EP deployment serving without a replacement GPU by aborting the failed epoch, establishing survivor-only control/data membership, and atomically committing placement + communication state for one recovery generation. Detection never publishes a mask directly. Five subsections cover communication escape (§5.1), placement admission/adaptation (§5.2), detection and state ownership (§5.3), MPI lifecycle/membership (§5.4), and the end-to-end recovery transaction (§5.5).

## 5.1 Rank masking in communication kernels

The Q2 live/silent MNNVL data-plane foundation. When no prompt process/backend evidence arrives and peer memory remains readable, surviving ranks' AlltoAll kernels can spin on `completion_flags[*][dead_rank]`. Rank masking lets a *new* launch skip committed-dead peers. It is necessary but not sufficient: an already-running invocation needs 1a.8, and no mask may commit before placement and survivor communicators match the same generation.

**Scope of applicability — by L3 transport, not deployment name.** The kernel mask applies whenever `NVLinkOneSided` (or `NVLinkTwoSided`) is the selected transport, which `CommunicationFactory` chooses whenever `MnnvlMemory.supports_mnnvl()` returns True (`_mnnvl_utils.py:380-387` — "all NVLink up" check). That covers:

- Single 8-GPU NVL-class node (DGX/HGX B200, B300, H100) — intra-node NVLink up.
- GB200 / GB300 NVL72 single rack — full rack fabric.
- Multi-node SLURM/MPI deployments where the inter-node fabric is also NVLink (rare, but it qualifies).

It does **not** apply when the transport falls through to DeepEP / DeepEPLowLatency (selected for cross-IB / cross-non-NVLink-fabric peers — see [§3.5 Transport determines mechanism](03-failure-modes-and-gaps.md#35-transport-determines-mechanism)). That regime is covered separately in [§8.2 Phase 1-IB](pr-execution/08-implementation-plan.md#phase-1-ib--cross-ib-transport-coverage-nixl-ep-track) with a different mechanism (NIXL-EP `disconnect_ranks` + EPLB redistribute, gated on Audit 3). The MVP scope below addresses the NVLink-substrate footprint; Phase 1-IB addresses the cross-IB footprint.

### Per-backend approach

| Backend | Mechanism | MVP / v1 |
|:---|:---|:---|
| `NVLinkOneSided` (primary) | Add `active_rank_mask_lo, active_rank_mask_hi` (`uint64_t × 2` for up to 128 ranks) to dispatch + combine kernel pointer structs; guard both release-write and polling loops | **MVP** |
| `NVLinkTwoSided` | Same pattern; FIFO-based sync uses `mTail + kFifoDepth <= mHead` polling — same skip-poll-on-masked-peer rule applies | v1 |
| `AllGatherReduceScatter` and other supported NCCL paths | Coordinator-driven `ncclCommAbort` + generation-scoped reinit over the survivor set | **MVP (1a.7)** |
| `DeepEP` / `DeepEPLowLatency` | Direct masking/rebuild awaits an upstream NVSHMEM/DeepEP primitive; cross-IB deployment coverage is the conditional Phase 1-IB NIXL-EP/topology-mutation track or a limited timeout interim | Out of MVP; Phase 1-IB conditional |

### NVLinkOneSided (the MVP-critical change)

Source: `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.{h,cu}`. Merged #13404 changed two kernel structs and the dispatch/combine release and polling loops.

**Kernel pointer structs.** Merged PR 1a.2 / #13404 adds `uint64_t active_rank_mask_lo, active_rank_mask_hi` to both `DispatchKernelPointers` and `CombineKernelPointers`. The mask is copied by the host before launch and is read-only inside that invocation. This is a next-launch primitive: changing host state cannot release a kernel already running with an old all-active value.

**Constexpr bump.** #13404 raised the former `kMaxRanks = 64` limit to **128**, which accommodates NVL72 and aligns with the two-uint64 mask.

**Loop modifications.** Both dispatch (release-write and polling) and combine (matching pair) now apply a one-bit-test guard before per-peer work.

```cpp
// Dispatch release-write loop, masked
for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize) {
    if (!(active_rank_mask & (1ULL << target_rank))) continue;   // skip dead
    uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
    asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
}
// Dispatch polling loop, also masked
for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
    if (!(active_rank_mask & (1ULL << peer_rank))) continue;     // skip dead
    /* existing spin */
}
```

**Both sides must be masked.** A common mistake is to mask the write side only. That doesn't fix the hang: the surviving peer's polling loop on `completion_flags[Y][37]` would still spin until the 300s `trap;`, because nobody wrote that slot. The mask must short-circuit the *poll*, not just the *write*. Combine has the matching loop pair (release-write + poll) at lines 1190–1217; both need the guard.

**Routing safety.** The combine accumulator has a `dst_idx == -1` zero-fill branch, but that is not a valid model-correctness mechanism for fault recovery. If a failure interrupts dispatch/combine, item 1c.4c discards the entire execution epoch so partial or zero-filled logits cannot escape. After recovery commits, 1b.2a guarantees every routed expert has an admitted survivor; a missing destination is a fail-closed invariant violation.

**Performance impact.** One bit-test per rank per launch, in an outer loop that already loops over ranks. For 72 ranks: 72 bit-tests, negligible vs the memory operations. Performance gate at integration time: < 0.1 % steady-state regression with all ranks active.

**Running-kernel escape is an MVP requirement.** The 300s `trap;` poisons the CUDA context and #13404's by-value mask cannot affect an invocation already in flight. Promoted item 1a.8 adds a stable device/host-visible abort or generation primitive that a running kernel checks and returns through recoverably. The existing `trap;` behavior is not an acceptable MVP fallback.

**Industry precedent for the softer-than-`trap;` direction.** vLLM's in-flight FT work ([PR #38534](https://github.com/vllm-project/vllm/pull/38534)) uses a 100-second static kernel-side timeout as a **DeepEP-specific interim** so a failed peer does not spin forever. NIXL-EP is different: it changes topology through `disconnect_ranks` / `connect_ranks`; it does not expose DeepEP's timeout/auto-mask behavior. The relevant precedent for 1a.8 is the recoverable-error direction rather than the exact mechanism: our owned MNNVL kernel must replace `trap;` with a host-visible abort/generation path that returns control without corrupting the CUDA context.

### NCCL FT wiring (PR 1a.7)

Verified: zero non-test uses of `ncclCommAbort`, `NCCL_ASYNC_ERROR_HANDLING`, `ncclCommFinalize`, `ncclGetLastError` in TRT-LLM. The only NCCL integration is `torch.classes.trtllm.NcclCommunicatorOp` (P2P, no error hook).

**Why this matters more than "the EP fallback."** It's tempting to frame PR 1a.7 as a safety net for the rarely-used `AllGatherReduceScatter` EP backend, but NCCL is in the WideEP data path even when **MNNVL is the chosen AlltoAll backend.** Specifically:

- TP allreduces in non-MoE projections (LM head, embedding, output-side reductions) — NCCL via `torch.distributed` or via `NcclCommunicatorOp`.
- PP send/recv when `pp > 1` — NCCL via `NcclCommunicatorOp`.
- DeepSeek-V3 with `enable_attention_dp=True` reduces but does not eliminate this volume; the per-attention-layer TP allreduces go away, but output-side and (with PP) inter-stage collectives remain.
- The `AllGatherReduceScatter` EP backend itself — only chosen when MNNVL+DeepEP are both unavailable, which is rare on production NVL72.

A dead rank hangs the next NCCL collective on any of these paths just as surely as it hangs the MNNVL AlltoAll. Without `ncclCommAbort` wiring, that hang is unrecoverable — independent of whether MNNVL masking succeeds. So **PR 1a.7 is required for production WideEP regardless of which EP backend is selected**, not just as a fallback safety net. Audit 1a Day 1 corroborates this: even on the `torch.distributed` path (where PT does wire NCCL abort), PT 2.11's recovery is broken, which is why PR 2a.1 in Phase 2 drops below `torch.distributed` for the actual rebuild.

PR 1a.7 exposes the NCCL abort/reinit primitive for **MVP**, but does not autonomously decide membership. Item 1c.4b invokes it with the reconciled generation after quiesce; unsupported static sharding or communicators fail closed. Promoted 1a.11 then invalidates captured communicator/mask state and serves eagerly while safe graphs are recaptured.

### CUDA graph recovery (promoted 1a.11)

A CUDA graph can capture the old communicator handles and the old launch-time rank mask. Replaying it after a recovery commit would reintroduce stale membership even if Python state is correct. MVP therefore enters eager mode before membership changes, invalidates every graph tied to the old generation, resumes only with eager launches using the committed generation, and recaptures graphs in the background. The prototype may force eager mode throughout; production must test invalidation and recapture.

## 5.2 EPLB topology adaptation

EPLB was designed as a static-topology system. `MoeLoadBalanceMetaInfo` stores `epRank` and `epSize` as plain `int` (verified in `moeLoadBalanceCommon.h:40–52`); the data structures (`rankExpertIds[epSize][slotCountPerRank]`, `globalSlotIds[epSize * slotCountPerRank]`) are sized at creation. Phase 1 needs to react to topology changes without rebuilding these structures from scratch every iteration.

### MVP: `reconfigure_mask_only`

The MVP precondition is the **actual survivor-placement invariant**, checked by item 1b.2a: for every layer and expert, every admitted single-rank failure leaves at least one resident copy on a distinct admitted failure domain. The canonical DS-V3 example has `72 × 4 = 288` slots for 256 experts—only 32 extra slots—so at least 224 experts are singleton even under an ideal distribution. A configured replication target, spare-slot count, or average is not proof.

**Mechanism.** New C++ entry point on `MoeLoadBalancer`:

```cpp
void reconfigure_mask_only(std::set<int> const& deadRanks);
```

**What it does:**
1. Pause EPLB worker and compute threads at the next safe point (after the in-flight layer's update completes).
2. For each layer, mark every slot belonging to `deadRanks` as unreachable in `MoePlacementInfo`. Concretely: zero the routing entry's count or set it to a sentinel that the routing kernel skips.
3. `cudaMemcpyAsync` the updated `globalSlotIds` for all 58 MoE layers (DeepSeek-V3) on the EPLB stream.
4. Resume worker + compute threads.

**Target: < 10 ms total** for all 58 layers. No H2D weight copies. No slot reallocations. No `doReplication` / `doPlacement` rerun. The placement table just stops pointing at the dead rank's slots; routing falls through to surviving replicas.

### Why this works only after 1b.2a admission

For an admitted failure, every layer/expert has a survivor whose pointer and resident weights remain valid. `reconfigure_mask_only` can therefore remove dead-rank slots without moving weights. If any expert would have zero survivors—or if nominal copies share the same excluded failure domain—admission fails before serving and the runtime does not commit the mask.

**Memory and load impact.** No new weights arrive on the admitted no-copy path, but traffic does not necessarily redistribute uniformly. The frequently quoted `1/71` average is only a capacity heuristic; routing skew and replica placement determine real throughput/latency and must be measured by 1d.4/1d.5.

### v1: full `reconfigure` with weight migration

When a dead rank held the *only* copy of some expert (replication 1, or pathological replica concentration), MVP is insufficient — the expert has zero surviving replicas and routing has nowhere to fall through. v1 adds:

1. `reconfigure(emergency_mode: bool)` C++ entry point.
2. Detect zero-replica experts (those that only existed on `deadRanks`).
3. For each, pick a surviving rank with free slot capacity, assign the expert there.
4. Copy the expert's weights from the node-local POSIX shm segment (which `HostMoeTensorSharer` already populates) to the new GPU slot via `cudaMemcpy2D` + gdrcopy.
5. Update `MoePlacementInfo` to point routing at the new slot.

Per-expert weight copy: ~0.1–0.3 ms. With ≤ 2 zero-replica experts per layer in pathological cases, total v1 budget: **< 50 ms** for all 58 layers.

**Why the host shm makes this cheap.** `HostMoeTensorSharer` (`moe_load_balancer.py:127–340`, with the node-local discovery at `:896–897`) maps every node's experts into POSIX shm at startup. Every rank on the same node can read every expert's weights without a network transfer. The "weight migration" is really a host-shm-to-GPU copy, not a remote fetch.

**Cross-node consideration.** A whole-node loss with replication=1 on that node loses experts that are not in any other node's shm. This is explicitly out of MVP and v1 scope; it's a node-level failure that needs Phase 2's MX P2P streaming (cross-node weight transfer) to recover. [§9](09-risks-and-open-questions.md) tracks this.

### Threading and atomicity

The EPLB worker thread runs continuously, rotating through layers and updating placement statistics. The compute thread runs `doReplication` + `doPlacement` periodically. `reconfigure_mask_only` must coordinate with both:

- **Worker thread:** poll for a "reconfigure pending" flag at layer boundaries; release the layer's signal token before checking.
- **Compute thread:** poll for the same flag at the start of each `doReplication` cycle; if pending, wait until reconfigure completes.

The `MoeLoadBalanceSingleLayerSignal::stepAndOwner` 64-bit step+owner word (`moeLoadBalanceCommon.h:25–37`) is the existing primitive for ownership coordination. We don't add a new primitive — we add a "reconfigure-pending" flag in `MoeLoadBalanceMetaInfo` that the existing threads check at their existing safe points.

This EPLB-local exclusion is necessary but not sufficient. Item 1c.4b owns the cross-component transaction: no new epoch is admitted while placement, survivor control membership, NCCL communicators, graph state, and the kernel mask refer to different generations. The committed `EPGroupHealth` generation advances only after every component is ready.

## 5.3 Failure detection & PR #12718 integration

Detection is the entry point for everything in §5.1 and §5.2. The design extends [PR #12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718)'s error classification from binary executor health (healthy / fatal) to per-EP-rank health (each rank tracks its own health independently).

### Three-layer detection

| Layer | Mechanism | Latency | Covers which mode | Deployment caveat |
|:---|:---|:---|:---|:---|
| **Layer 1 — MNNVL AlltoAll watchdog** (MVP primary) | Host thread polling the NVLinkOneSided host-visible `completion_flags` table; reports ranks that have not signaled within the configured timeout | Configurable; measure in 1d.4/1d.4a | **Q2** on the supported MNNVL route | Available only when the MNNVL/NVLink completion table is the selected backend; DeepEP/NIXL/NCCL require backend-specific signals |
| **Layer 2 — Worker/process-exit notification** | Supported launcher/runtime monitoring plus PR #12718's `_error_monitor_loop` / per-rank `_check_mpi_futures` where futures exist | Configurable and measured; no universal fixed poll bound | **Q1/Q3** prompt process-exit evidence | `_check_mpi_futures` is **inert when `mpi_session = RemoteMpiCommSessionClient`** (workers in separate `mgmn_leader_node` process; `submit()` returns `[]`); 1d.1 must admit another survivor-preserving monitor for that route |
| **Layer 3 — Latency/degradation telemetry** (Phase 3) | Per-rank transport and compute telemetry; reversible degradation policy | Measurement-derived | Soft degradation, not monotonic Q1–Q4 failure evidence | Backend- and platform-specific |

**Implication for `trtllm-llmapi-launch` deployments.** The `mgmn_leader_node`-based launcher path (used by `trtllm-bench`, `trtllm-serve` with `TLLM_SPAWN_PROXY_PROCESS=1`, and the layer-wise benchmark CI test) instantiates `RemoteMpiCommSessionClient`, whose `submit()` returns `[]`. PR #12718's `_check_mpi_futures()` therefore has no futures to inspect and Layer 2 is silent. On the supported MNNVL MVP route, the completion-flag watchdog is the remaining in-process detector and must be admitted/configured explicitly. Other backends require their own error signal (NCCL async error, NIXL topology/error state, or the DeepEP timeout path); no latency ordering is assumed until measured.

### Where errors come from: producers vs consumers

The detection layers above produce signals; PR #12718's `classify_error()` is the consumer of those signals. The matrix below shows, for each backend in the WideEP data path, **whether failures surface naturally today** versus which transport wiring or consumer-side classification work the design adds.

| Backend | Used for in WideEP | Surfaces failure today? | What this design adds |
|:---|:---|:---|:---|
| NCCL via `torch.distributed` | TP/output-side model tensor collectives | **Yes** — PT watchdog raises Python exception | Classified by PR #12718 patterns; Audit 1a found PT 2.11 shrink recovery broken, so MVP 1a.7 owns the supported survivor primitive |
| NCCL via TRT-LLM custom op | PP send/recv (`NcclCommunicatorOp`), `AllGatherReduceScatter` EP fallback | **No** — `ncclCommAbort`/`getAsyncError` not wired (zero non-test uses) | Closed by **PR 1a.7** (NCCL FT wrapper). Note that NCCL is in the WideEP data path even when MNNVL is the chosen EP backend (TP / PP collectives), so 1a.7 matters more than the "EP fallback" framing implies |
| NIXL | Disagg KV cache transfer (production default for §1-DS) | **Yes** — `check_xfer_state == ERROR` raises `RuntimeError("NIXL transfer failed: …")` (`_agent_py.py:125`) | Already classifiable; **PR 1c.1 adds NIXL-specific regex patterns** (`"nixl transfer failed"`, `"nixl transfer entered error state"`) so the classifier reaches `severe` instead of falling through to `transient` |
| MNNVL / NVLinkOneSided | WideEP MoE AlltoAll (NVL72 primary) | **No** — kernel spins on `completion_flags[*][peer]`; eventual `trap;` corrupts CUDA context | PR 1a.4 reports timeout evidence; promoted 1a.8 makes the running kernel return recoverably; neither commits membership |
| NVSHMEM / DeepEP | Cross-node EP fallback | **Limited** — `Buffer.__del__` deadlocks on peer death; no public `mask_buffer_ptr` | Direct DeepEP masking/rebuild is out of MVP pending an upstream primitive; Phase 1-IB separately evaluates NIXL-EP topology mutation or a limited timeout interim |

So **PR 1a.7 and PR 1a.4 add transport-side signal paths**, while **PR 1c.1 remains strictly consumer-side pattern classification** for errors that already surface (including NIXL). These are separate ownership boundaries; classification does not detect or recover a failed rank.

### Layer 1 — AlltoAll watchdog (detection evidence only)

The kernel's existing `completion_flags[kMaxRanks][kMaxRanks]` table sits in host-visible MNNVL fabric memory. The host can read it without entering the kernel. New component:

```python
class AlltoAllWatchdog:
    def __init__(self, completion_flags, committed_health, on_timeout, timeout_sec):
        self.completion_flags = completion_flags
        self.committed_health = committed_health  # read-only expected peers
        self.on_timeout = on_timeout
        self.timeout_sec = timeout_sec

    def watch(self, expected_flag_val):
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            pending = {
                r for r in range(self.committed_health.moe_world_size)
                if self.committed_health.is_active(r)
                and self.completion_flags[my_rank][r] != expected_flag_val
            }
            if not pending:
                return set()
            time.sleep(0.1)
        self.on_timeout(pending)  # evidence only; does not abort or commit
        return pending
```

**Timeout tuning.** No production default or second environment-variable contract is committed here. Item 1d.1 owns the unified WideEP FT configuration, and 1d.4/1d.4a plus Phase 1-IB acceptance must measure workload-specific false-positive and recovery tradeoffs before release defaults are chosen:

| Deployment | Release setting | Rationale |
|:---|:---|:---|
| NVL72 single rack | Measure in 1d.4a | NVLink latency alone does not bound scheduling, kernel, or fabric-manager stalls |
| Multi-node + RDMA | Measure in Phase 1-IB acceptance | Transport and workload tails differ from NVL72 |
| Dev / CI | Short, test-specific value | Fast deterministic injection is useful but is not a production recommendation |

The watchdog deadline is a detection policy, not the running-kernel escape. Promoted 1a.8 supplies the recoverable device/host-visible abort/generation primitive; `trap;` is not an MVP backstop.

### Layer 2 — Per-rank worker death

PR #12718 introduces `GenerationExecutorProxy._check_mpi_futures()` and `_drain_error_queue()` helpers shared between `check_health()` and the daemon `_error_monitor_loop`. `mpi_done_callback` registered on each future enqueues exceptions to a shared `_error_queue` — there's no per-rank attribution. WideEP FT extends this to track which rank's future raised. Per-request errors (`RequestError` / `str`) are filtered out by PR #12718 in both paths so a single bad request can't promote into a per-rank failure; the per-rank tracker inherits that filter for free.

```python
class EPRankHealthTracker:
    def __init__(self, ep_size, report_failure_evidence):
        self.rank_budgets = {r: ErrorBudget() for r in range(ep_size)}
        self.report_failure_evidence = report_failure_evidence

    def on_mpi_worker_death(self, rank, error):
        cls = classify_error(str(error))     # PR #12718 primitive
        if cls == "immediate_fatal":
            self.report_failure_evidence(rank, cls)
        elif cls == "severe":
            if self.rank_budgets[rank].consume(cost=0.5):
                self.report_failure_evidence(rank, cls)
```

### EP-specific error patterns

PR #12718's classifier is regex-driven over lowercased error messages, returning string literals (`"immediate_fatal"`, `"severe"`, `"transient"`). WideEP FT adds patterns to the existing lists in `error_classification.py`:

```python
EP_IMMEDIATE_FATAL_EXTRA = [
    "nccl communicator abort", "nvshmem peer unreachable",
    "mpi rank terminated", "cuda context destroyed",
]
EP_SEVERE_EXTRA = [
    "alltoall timeout", "nccl timeout", "deep_ep buffer barrier hang",
    "symmetric memory access violation", "rdma timeout",
    # NIXL — disagg KV transceiver (production default). Strings drawn from
    # tensorrt_llm/_torch/disaggregation/nixl/_agent_py.py:40,43,125
    "nixl transfer failed",
    "nixl transfer entered error state",
    "nixl transfer wait timed out",
]
EP_TRANSIENT_EXTRA = ["alltoall slow", "nccl retry", "ecc correctable error"]
```

The classifier still returns the same three string literals; we add patterns, not classes. The NIXL additions ensure that KV-transfer failures on the disaggregated path classify as `severe` (consume per-rank budget; potential FT trigger) rather than falling through to `transient` (logged only). Producer-side, NIXL already raises these messages cleanly; the only design work is the regex patterns themselves.

### Detected state versus committed `EPGroupHealth`

PR 1a.1 / #13302 provides the merged thread-safe mask primitive. In the corrected design, `EPGroupHealth` represents the **committed data-plane generation**, not raw detector output. Watchdog/MPI/FT threads write a separate suspect/evidence set owned by 1c.4b. This prevents communication from observing a new mask while EPLB placement or control membership is still old.

API:
```python
class EPGroupHealth:
    def __init__(self, ep_size: int): ...
    def mark_failed(self, rank: int) -> bool: ...    # coordinator-only commit; not a detector callback
    def mark_active(self, rank: int) -> bool: ...    # for Phase 2 restoration
    def is_active(self, rank: int) -> bool: ...
    def get_mask(self) -> int: ...                    # arbitrary-precision Python int
    def get_mask_words(self, n=2) -> tuple[int, ...]: # uint64 words for kernel ABI
    @property
    def generation(self) -> int: ...                 # bumps on effective change
    def get_failed_ranks(self) -> frozenset[int]: ...
```

`generation` is the cheap **commit** primitive: the model engine caches the last committed generation and admits work only when placement, survivor control/data communicators, graph policy, and mask all match it. Detection has its own evidence epoch and never bumps this counter directly.

### Failure broadcast and cross-rank consensus

When any survivor reports rank 37, every survivor must reconcile the same suspect set before the next clean epoch. Otherwise rank 0 could commit a mask while rank 50 still tries to write to `completion_flags[37][50]`—split-brain.

The Q2 live/silent MNNVL failure (kernel spinning on a non-advancing peer flag) is what makes this hard: the forward thread is stuck inside the kernel and can't participate in consensus. The broadcast must run on a host thread that is independent of GPU state.

**Approach (MPI path).** PR 1c.3 provides a dedicated MPI subcommunicator and host thread for failure notification/evidence. It does not replace ordinary management collectives and cannot commit the mask. Item 1c.3a constructs a survivor-only control communicator plus logical-to-physical `ActiveRankMap`; item 1c.4a moves attention-DP/PyExecutor gathers to that membership.

**Why blocking collectives don't work on the FT subcomm.** A blocking `MPI_Allreduce` on a poisoned communicator deadlocks even with `MPI_ERRORS_RETURN` set. Hence non-blocking + polling.

**ULFM if available.** `MPI_Comm_revoke` from ULFM is the cleanest primitive for "this comm is poisoned, give me a working one." But ULFM availability depends on the MPI build — opt-in in OpenMPI, patchy in MVAPICH, missing in Intel MPI. The MVP design uses ULFM if present and falls back to single-failure-only without it (acceptable for MVP).

**Approach (Ray path, future).** A Ray deployment would replace the MPI notification/control pieces, but it still needs a proven survivor communicator path. Audit 1a showed the shipping PyTorch `shrink_group` path can hang, so a future pivot cannot assume upstream recovery is sufficient without revalidation.

### Single-failure scope simplifies policy, not atomicity

MVP admits at most one failed rank, but a report is still evidence rather than a committed topology. Item 1c.4b reconciles survivor evidence and orders one atomic generation. V1's multi-failure consensus adds richer suspect/confirm policy; it does not remove the MVP requirement for a common commit.

### Lessons from PR #12718 implementation

PR #12718 went through several iterations and one production-affecting regression before merging. Four of those experiences directly inform this design:

**1. Empty-collection predicates need explicit branches.** PR #12718's `pre_shutdown()` originally gated the quit-sentinel send on `all(not f.done() for f in self.mpi_futures)`, which is *vacuously True* for an empty list and therefore covered the `RemoteMpiCommSessionClient` case (no local futures) by accident. A later refactor to `any(...)` regressed that case — `any([]) == False` — and silently dropped the sentinel, hanging `dispatch_result_thread.join()` indefinitely. This manifested as a 2400 s `test_performance_alignment[1]` timeout. The fix is `if not self.mpi_futures or any(...)`. **Implication for WideEP FT:** every predicate over `self.ep_group_health.active_ranks()`, `pending` rank sets, etc. must explicitly handle the empty case. The Layer 1 `AlltoAllWatchdog.watch()` body already does this (`if not pending: return set()`); confirm the same for the failure-broadcast and suspect-set logic in §5.3.

**2. PR #12718's `pre_shutdown()` is non-blocking by design — failure notification must be too.** The notification path records evidence and posts sends without waiting for the dead process. It does **not** update committed `EPGroupHealth`; 1c.4b first aborts the failed epoch and reconciles evidence, then validates admission, quiesces, prepares EPLB, rebuilds survivor control/NCCL, applies graph policy, commits one mask + `ActiveRankMap` + generation, applies request disposition, and only then resumes.

**3. Per-request errors must never promote to per-rank failures.** PR #12718's `_drain_error_queue` filters `RequestError` and bare `str` errors before promoting anything to fatal. The same filter must apply when `EPRankHealthTracker.on_mpi_worker_death` examines the error queue — a malformed prompt that surfaces via the worker's `_error_queue` should not mark the rank failed. The current §5.3 `EPRankHealthTracker` sketch only inspects errors that come through `mpi_done_callback` (i.e., future-level exceptions, which can't be `RequestError`), so the filter is implicit — but if a future revision adds queue-based input it must apply the same `isinstance(e, (str, RequestError))` skip.

**4. Detection from the bench-shutdown investigation: instrumentation, not watchdogs.** Four CI cycles of "test hangs at 2400 s" produced zero useful diagnostic output because pytest captured the inner subprocess and the SIGKILL on timeout destroyed the captures. What localized the bug in 5 minutes was per-step `time.monotonic()` markers around the proxy lifecycle. **Implication:** retain markers for evidence receipt, failed-epoch abort, reconcile, admission, quiesce, EPLB preparation, survivor-communicator/NCCL rebuild, graph policy, commit, request disposition, and resume. Otherwise the next regression may look like a silent hang with no useful exception.

See also: [`docs/investigations/nvbug-6043291-zombie-worker-pods/bench-shutdown-hang.md`](../../investigations/nvbug-6043291-zombie-worker-pods/bench-shutdown-hang.md) for the full investigation diary.

### Model-engine integration work split

| Item | Contract |
|:---|:---|
| **1c.4** | Existing model-engine hook: observe recovery state at a safe iteration boundary and stop admitting new epochs. |
| **1c.4a** | Rebind attention-DP/PyExecutor management exchanges to the survivor `ActiveRankMap`. |
| **1c.4b** | Atomic recovery coordinator and sole owner through `detect → abort → reconcile evidence → validate admission → quiesce → prepare EPLB → rebuild survivor control/NCCL → apply graph policy → commit mask + ActiveRankMap + generation`; it invokes 1c.4c disposition before resume. |
| **1c.4c** | Failed-epoch/request disposition: suppress all failed-epoch output, preserve queued work when safe, and assign each in-flight request an explicit retry, reroute, or request-error result before resume. |

## 5.4 MPI-path FT-enabling work

The MPI propagation/lifecycle portion of Q1/Q3. Today's MPI signal handlers can call `MPI_Abort(MPI_COMM_WORLD)`, killing the whole world before user-space FT logic can run; a launcher may independently propagate an abnormal exit. This subsection separates handler replacement (1d.0), survivor-preserving launcher/runtime admission (1d.1), survivor membership (1c.3a/1c.4a), and poisoned-world lifecycle (1d.0a).

### Signal handler replacement

Merged as PR #14160 (PR 1d.0). Source: `cpp/tensorrt_llm/runtime/utils/mpiUtils.cpp:195–215`. The handler removes the explicit `MPI_Abort`/parent-kill path under FT mode; it does not guarantee survivor preservation because a launcher may still terminate the job on abnormal exit. Item 1d.1 owns the canonical flag and admits only a tested survivor-preserving launcher/runtime configuration; 1d.0a owns poisoned-world lifecycle.

```cpp
// New: no-MPI_Abort handler (FT mode; launcher admission is separate)
previousHandler = std::signal(sig, [](int signal) {
    // Do not call MPI_Abort. Do not send SIGKILL upward.
    // Just exit cleanly; surviving ranks will detect the silent peer
    // via the AlltoAll watchdog (§5.3 Layer 1) under the configured policy and via
    // MPI worker-death notification (Layer 2) shortly after.
    _exit(EXIT_FAILURE);
});
```

**Async-signal-safety.** The handler must only call async-signal-safe functions. `_exit(2)` is async-signal-safe (POSIX); `MPI_Abort` is not in MPI's async-signal-safe set, but its use is forgivable because it terminates anyway. `printf` / logging in the handler is unsafe; we omit it. Errors are logged from the survivors after detection.

**Why `_exit` rather than `exit`.** `exit` runs `atexit` handlers (Python finalizers, MPI cleanup) which can deadlock on a poisoned state. `_exit` skips them and just terminates the process.

**Why not just signal peers directly from the handler.** `MPI_Isend` is not async-signal-safe in most MPI implementations. The host watchdog on surviving ranks is the right detection layer; we don't need the handler to actively notify.

### `MPIPoolExecutor` audit

`MpiPoolSession.abort()` at `mpi_session.py:167–168` calls `comm.Abort(1)` which kills the world — even if our handler doesn't. We change this path under the FT flag:

```python
class MpiPoolSession(MpiSession):
    def abort_for_ft(self, dead_ranks):
        # Don't comm.Abort or pretend static futures can continue.
        for rank in dead_ranks:
            self.ft_subcomm.report_detected_failure(rank)
        # 1c.3 records evidence only; the coordinator quiesces until
        # 1c.3a/1c.4a rebuild ordinary survivor collectives.
```

The proxy's `mpi_done_callback` (`proxy.py:229–234`) currently routes any future failure to `_error_queue` as a fatal error. Under FT mode, the callback consults `EPRankHealthTracker.on_mpi_worker_death(rank, error)` and publishes rank-attributed evidence when policy allows. It does not reroute or continue static futures; 1c.4b quiesces until survivor membership is prepared.

### FT subcomm

Detailed in §5.3 above. PR 1c.3 is built for signaling/evidence only. It does not make the existing attention-DP/PyExecutor collectives survivable. Item 1c.3a creates survivor-only control membership and `ActiveRankMap`; 1c.4a wires the ordinary rank-state/request/batch/token/model-input exchanges to that map.

### Poisoned-world lifecycle (1d.0a)

After any peer death, `MPI.COMM_WORLD` may be poisoned even when the survivors continue serving through a new control communicator. The old #14198 scaffold demonstrated a hang in `MPI_Finalize`. Item 1d.0a prohibits further world collectives after poison is recorded and supplies deterministic survivor shutdown that bypasses a collective finalize when required. This is separate from the signal-time behavior in 1d.0.

### Optional: ULFM

If the MPI build supports ULFM, it can improve revoke/shrink semantics. Without ULFM, MVP remains single-failure-only and relies on the pre-created notification channel plus a survivor-only control communicator. Item 1d.1 also requires the MPI thread level needed by the dedicated control thread and the launcher recovery mode that keeps survivors alive.

### Feature and deployment admission (1d.1)

One user-facing feature gate validates the complete contract before serving: supported MPI launcher/recovery mode and thread level; supported communication backend (no silent MegaMoE/DeepEP fallback); 1b.2a placement invariant; rank-0/frontend policy; required HBM/fabric setup; and eager/CUDA-graph policy. A configuration that misses any prerequisite fails closed with a specific diagnostic.

## 5.5 End-to-end flow & timing

Putting §5.1 through §5.4 together: what happens, in what order, when rank 37 dies in a 72-rank EP group.

```mermaid
sequenceDiagram
    participant Dead as Rank 37 (dying)
    participant Kernel as AlltoAll kernel
    participant Detect as Watchdog / FT evidence
    participant Coord as 1c.4b coordinator
    participant Control as Survivor control + ADP
    participant EPLB as EPLB
    participant NCCL as Data communicators

    Dead->>Dead: GPU/process failure
    Detect->>Coord: report suspect 37 (evidence only)
    Coord->>Kernel: set 1a.8 execution-epoch abort token
    Kernel-->>Coord: failed epoch returned; no output committed
    Coord->>Coord: reconcile survivor evidence
    Coord->>Coord: run 1b.2a placement admission
    Coord->>Coord: quiesce launches and admission
    Coord->>EPLB: prepare reconfigure_mask_only({37})
    EPLB->>EPLB: mark dead-rank slots unreachable (58 layers)
    Coord->>Control: prepare 1c.3a ActiveRankMap<br/>and 1c.4a survivor collectives
    Coord->>NCCL: abort/reinit supported comms for survivors
    Coord->>Coord: apply 1a.11 eager/graph policy
    Coord->>Coord: atomically commit mask + ActiveRankMap + generation
    Coord->>Coord: apply 1c.4c failed-epoch request disposition
    Coord->>Kernel: launch first clean epoch with committed mask
    Kernel->>Kernel: skips rank 37 in dispatch + combine loops
    Kernel-->>Coord: valid N-1 result
    Note over Coord: only now may serving resume
```

**Two generations, one owner for membership.** The 1a.8 device-visible value is an execution-epoch abort token (or prepared kernel epoch) used only to release work already in flight; changing it does not publish a data-plane topology. The committed membership generation belongs solely to 1c.4b and advances only with the atomic mask + `ActiveRankMap` commit after every prerequisite is ready.

### Timing budget

The working acceptance objective is **≤ 10 s end-to-end** from failure to serving at N-1, but it is not a design guarantee. The physical 1d.4/1d.4a tests establish the actual budget. Breakdown hypotheses:

| Step | Time | Dominant component |
|:---|:---|:---|
| Detection (watchdog timeout) | To measure; configurable | Expected to dominate; policy is owned by unified 1d.1 configuration ([§5.3](#layer-1--alltoall-watchdog-detection-evidence-only)) |
| Running-epoch abort | To measure | 1a.8 device/host generation path; must return without context poison |
| Evidence reconcile | To measure | 1c.3 evidence plus 1c.4b reconciliation; not commit authorization |
| Placement admission | To measure | 1b.2a checks every layer/expert/failure domain |
| Quiesce | To measure | 1c.4b stops new launches/admission before preparation |
| EPLB `reconfigure_mask_only` preparation | **< 10 ms target** | 58 layers × in-place `cudaMemcpyAsync`; measured as one sub-step |
| Survivor control/ADP + supported NCCL rebuild | To measure | 1c.3a + 1c.4a + coordinator-driven 1a.7 |
| CUDA-graph policy | Pre-commit eager selection + invalidation | 1a.11 invalidates old-generation captures; recapture starts only after membership commit |
| Atomic commit | O(1) | one mask + `ActiveRankMap` + generation across all consumers |
| Request disposition + clean relaunch | To measure + normal iteration | 1c.4c assigns retry/reroute/error before the first resumed launch |

Detection is expected to dominate, but the old mock prototype did not measure the production communicator, ADP, graph, and coordinator steps. The <10 ms `reconfigure_mask_only` target is one internal sub-step; 1d.4 and 1d.4a must establish the complete physical budget.

**Two distinct numbers.** The doc and the codebase will refer to both:
- **< 10 s** — total Phase 1 recovery (failure → serving at N-1).
- **< 10 ms** — just the EPLB reconfigure step.

These are not in tension; they're at different scopes. The detection budget is configurable, the reconfigure budget is an internal performance gate.

### What happens to in-flight requests

Item 1c.4c defines the execution-epoch boundary. Once failure evidence aborts an epoch, **none of its logits or token decisions may be published**, including apparently completed survivor contributions or zero-filled combine output. Queued work is preserved when safe, and every in-flight request receives an explicit retry, reroute, or request-error disposition through the PR #12718/#13119 contract.

Requests **queued but not yet scheduled** into the failing iteration are unaffected — they're picked up in the next iteration with the new mask and new routing. New requests arriving after the reconfigure are served normally at the reduced capacity.

Transparent replay from the last emitted token is an orchestration-layer concern and remains out of scope; suppression and explicit disposition of the failed epoch are mandatory MVP behavior.

### Serving in degraded mode

| Metric | Effect |
|:---|:---|
| **Throughput** | Expected to decrease at least with lost capacity; routing/replica skew can make the loss larger than the `1/72` average. Measure rather than assert `(N-1)/N`. |
| **Latency** | Deployment- and placement-dependent; survivor load may be uneven and eager fallback is temporarily slower. |
| **Correctness** | Preserved only when 1b.2a admission passes, the failed epoch is suppressed, and every consumer observes the same committed generation. |

### MVP de-risking — no-mock end-to-end prototype

The new prototype is built from current `main` plus the actual production PR heads. It uses real MPI processes, CUDA/NCCL/MNNVL/EPLB components, a real MoE model/workload, and physical GPU fault injection. A non-rank-0 victim is required until a frontend failover policy exists. Success means the failed epoch emits no output, all survivors commit the same generation, and later requests complete with reference-correct results.

The earlier [PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) scaffold remains historical evidence only. Its POSIX-shm counters and mocked collectives found useful seam issues, including poisoned `MPI_Finalize`, but cannot satisfy the MVP exit gate. Item 1d.4 is the intra-node production-component test; 1d.4a repeats the acceptance on NVL72/equivalent with the actual FABRIC/IMEX path. See [MVP prototype plan](mvp-prototype-plan.md).

## 5.6 Phase 1 v1 — what's added

For completeness, items deferred from MVP to v1:

- **NVLinkTwoSided masking** ([§5.1](#per-backend-approach)).
- **Full `reconfigure` with weight migration** for zero-replica experts ([§5.2 v1](#v1-full-reconfigure-with-weight-migration)).
- **Multi-failure consensus** with two-phase suspect → confirm protocol.

NCCL survivor recovery (1a.7), running-kernel escape (1a.8), and CUDA-graph recovery (1a.11) are MVP requirements and are intentionally absent from this deferred list.

§8 sizes each as named PRs.
