# 13 — Empirical refutation: `asymmetric_executor[mpi_kvcache]` is a transport hang, not horizontal-consistency ABBA

**Status:** Empirical finding from local repro on `pr-13713-head` / B300, dated 2026-05-26.
**Trigger:** Local reproduction of the deterministic-failing cpp gtest `cpp/test_multi_gpu.py::TestDisagg::test_asymmetric_executor[llama-4proc-mpi_kvcache]` with `[NVBUG-6104831-INSTR]` instrumentation enabled on both ctx and gen sides, plus live kernel-state inspection (`/proc/<pid>/task/<tid>/wchan`, fd table) of the wedged worker processes.
**Public PR:** https://github.com/NVIDIA/TensorRT-LLM/pull/13713

This document is a corrective addendum to [12-horizontal-consistency-and-layer3-gating.md](12-horizontal-consistency-and-layer3-gating.md) §4.3. It does **not** invalidate the horizontal-consistency theory for the helix Python-side flakes — those remain an open evidentiary gap — but it does refute the use of `asymmetric_executor[mpi_kvcache]` as an empirical anchor for that theory.

---

## 1. Summary

Doc 12 §4.3 framed the deterministically-failing `test_asymmetric_executor[llama-4proc-mpi_kvcache]` as the cleanest evidence for the horizontal-consistency theory:

> Condition 1 (asymmetric ctx/gen) is satisfied by all three transports; only the slower transport reliably hits condition 2. This precisely matches theory prediction 3 (§2.6).

Live evidence from the wedged worker processes shows this framing is wrong. The cpp gtest wedge sits at the **mpi_kvcache transport layer** (UCX-over-shared-memory inside OpenMPI) and the gtest's internal 300 s timeout fires *before* any of the cascading horizontal-consistency effects described in §2.4 can manifest.

The test is still a real, deterministic, PR-#13713-specific failure — but it does not validate the specific ABBA mechanism doc 12 §2 hypothesises. It validates only the prerequisite (condition 2: stuck transfer) and tells us nothing direct about whether the cascade into asymmetric collective entry actually fires under the same load.

The helix Python-side flakes remain the only candidate empirical anchor for the horizontal-consistency theory. They have not yet been reproduced locally (5/5 healthy runs on the same hardware), so the theory remains untested by direct evidence.

---

## 2. What the instrumentation actually captured

The instrumentation suite is described in the parent README and in [scripts/disagg/nvbug6104831_diff.py](../../../scripts/disagg/nvbug6104831_diff.py). Briefly, every rank emits one INSTR record per scheduling-iteration decision or cross-rank collective entry, tagged with `iter`, `rank`, `site`, and a `caller=` field for `gatherRequestIds.exit` so the ctx-side and gen-side gathers do not alias in the postprocessor.

The wedge was captured during variant 2 of the test (`LlamaConTP2GenPP2DisaggAsymmetricExecutorTest`), with 4 MPI procs split 2 ctx (world ranks 0, 1) + 2 gen (world ranks 2, 3). The mpi_kvcache backend uses OpenMPI which is configured to use UCX shared-memory transport on this single-node B300 host (no InfiniBand: `UCX_TLS=^ib`).

### 2.1 The orchestration layer is symmetric, not divergent

By iter=2 on the ctx side, both ctx ranks have an identical, complete sender-future set:

```
iter=2 rank=0 site=cacheTransceiver.checkContextTransferStatus.enter
  senderFutures_size=8 senderFutures_ids=[1,2,3,4,5,6,7,8]
iter=2 rank=1 site=cacheTransceiver.checkContextTransferStatus.enter
  senderFutures_size=8 senderFutures_ids=[1,2,3,4,5,6,7,8]
```

By iter=35 on the gen side, both gen ranks have an identical, complete requester-future set:

```
iter=35 rank=2 site=cacheTransceiver.checkGenTransferStatus.enter
  requesterFutures_size=8 requesterFutures_ids=[1,2,3,4,5,6,7,8]
iter=35 rank=3 site=cacheTransceiver.checkGenTransferStatus.enter
  requesterFutures_size=8 requesterFutures_ids=[1,2,3,4,5,6,7,8]
```

All `cacheTransceiver.gatherRequestIds.exit` records on both sides have matching `local_ids`/`gathered_ids` between the two ranks of their respective `comm_size=2` group. No rank is short an entry. No rank entered a different collective.

The postprocessor reports **no cross-rank divergence** for either the ctx comm or the gen comm at any iteration up to and including the wedge.

### 2.2 The wedge is at the per-request data transfer

After `requestSync.enter` records are emitted for all 8 receives on each gen rank, the gen-side log goes silent — no further `cacheTransceiver.checkGenTransferStatus.enter` records (the polling stops emitting because the C++ control flow is parked). On the ctx side, the main thread is reported `R running` in userspace (busy-polling `checkContextTransferStatus`) but `senderFutures_size=8` never decreases — none of the 8 senders ever complete.

Eventually the gtest's internal wait loop hits its 300000 ms cap:

```
cpp/tests/e2e_tests/executor/disaggExecutorTest.cpp:298: Failure
Expected: (iter) < (maxWaitMs), actual: 300000 vs 300000
```

and the test fails 8 batch-level token comparisons because no tokens were ever produced (`predictedTokens.size() = 2^64-N` is just default-initialised vector state read after the timeout; not UAF).

### 2.3 The kernel-state evidence locates the wedge in MPI/UCX, not in C++ application code

For each of the 4 wedged worker pids, `/proc/<pid>/task/<tid>/wchan` shows:

| Side | Critical threads | wchan | Meaning |
|------|------------------|-------|---------|
| ctx ranks 0, 1 | main `disaggExecutorT` | `0` (userspace, R running) | Busy-polling — likely inside `checkContextTransferStatus`'s `future.wait_for(0ms)` loop |
| ctx ranks 0, 1 | per-process network workers (`do_poll`, `ep_poll`) | kernel network poll | Alive, waiting for incoming traffic that never arrives |
| gen ranks 2, 3 | main `disaggExecutorT` | `futex_wait_queue` | Blocked on a pthread cond_var |
| gen ranks 2, 3 | `dataTransResp` | `futex_wait_queue` | Blocked on a pthread cond_var — waiting for receive-completion signal |
| gen ranks 2, 3 | primary `executionLoop` | `0` (userspace, R running) | Busy-polling — likely inside `checkGenTransferStatus`'s `future.wait_for(0ms)` loop |
| gen ranks 2, 3 | secondary `executionLoop` | `futex_wait_queue` | Blocked on a pthread cond_var |
| gen ranks 2, 3 | per-process network workers (`do_poll`, `ep_poll`) | kernel network poll | Alive, waiting for incoming traffic that never arrives |

Fd-table evidence on the same processes shows OpenMPI's shared-memory channel (`/dev/shm/open_mpi.0000`) and UCX shared-memory segments (`/dev/shm/ucx_shm_posix_*`) are the active transport. `TRTLLM_USE_MPI_KVCACHE=1` selects the MPI backend at the kvcache layer; on a single host with `UCX_TLS=^ib`, OpenMPI's underlying transport collapses to UCX shared memory.

The pattern — multiple sleeping cond_var waits on the gen side, paired with busy-polling main threads on both sides, and live network-poll workers that never get woken — is the classic "missed wakeup / handshake never closes" shape of a flow-control or buffer-pool deadlock inside the messaging stack. There is no thread parked in `MPI_Allgather` related kernel calls; the cross-rank collectives have already returned successfully.

We did not capture userspace stack frames (YAMA ptrace_scope on this host requires root, and we did not escalate). A follow-up capture with `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope; gdb -batch -p ...` against a fresh wedge would distinguish among:

1. **MPI flow-control deadlock**: too many in-flight `MPI_Isend`/`MPI_Irecv`, eager-buffer pool exhausted, rendezvous deadlocked because each side is waiting for the other to drain.
2. **UCX shm-segment exhaustion**: shared-memory rings full, writers blocked waiting for readers to drain, readers blocked elsewhere.
3. **TRT-LLM mpi_kvcache logic bug**: state machine deadlock inside `cpp/tensorrt_llm/executor/cache_transmission/mpi_utils/connection.cpp` (or wherever the MPI backend lives) above the MPI layer.

All three are "transport-layer hang" — the high-level diagnosis is robust to which one is correct.

---

## 3. Implication for doc 12 §4.3

Doc 12 §4.3 used `asymmetric_executor[mpi_kvcache]` as the cleanest deterministic exemplar of horizontal-consistency breakage, on the reasoning that mpi_kvcache satisfies both conditions (asymmetric topology + stuck transfer) reliably while ucx/nixl variants only stochastically satisfy condition 2.

The local evidence here forces a narrower claim:

- **What this test validates**: that mpi_kvcache reliably exhibits "stuck transfer" (condition 2) on this load shape. This is consistent with doc 12's §4.2 enumeration of condition-2 mechanisms (the head-of-line / size-1 send/recv pool, transient peer slowness, edge cases in the protocol).
- **What this test does *not* validate**: that the stuck-transfer condition then cascades into the asymmetric collective entry / ABBA at `gatherRequestIds` that doc 12 §2.4 hypothesises. The cascade requires the cross-rank state to actually diverge in a way that gates entry into the next collective. In our captured wedge, the gtest's 300 s wait loop fires *before* the cross-rank state has any chance to diverge — both sides remain perfectly symmetric in their orchestration view (8 senders ↔ 8 receivers, no decision asymmetry observed) for the entire window.

In short: `asymmetric_executor[mpi_kvcache]` is necessary (the wedge is real and PR-#13713-specific) but not sufficient (the wedge fires for transport reasons, not for horizontal-consistency reasons).

The clean test of the horizontal-consistency theory remains the **helix Python-side flakes** (per doc 12 §4.1 and §4.4). Those wedge stochastically and have *not* yet been reproduced locally on this hardware (a 5-run loop produced 0 wedges). Higher-rate reproduction (20+ runs) is the next data point that could actually validate or refute the cascade hypothesis.

---

## 4. Why does the cpp gtest fail on PR-#13713 specifically and not on `main`?

The test is in the regular CI test list on main (`tests/integration/test_lists/test-db/l0_dgx_h100.yml`), so if it failed reliably on main it would be a persistent CI alarm. It is not. It fails reliably on every PR-#13713 build. So something in the PR-#13713 delta is responsible for *exposing* the transport hang.

This document does not claim a single root cause; the evidence to date supports three live hypotheses, each with a different attribution. Listed in the order I currently weight them:

### 4.1 Hypothesis A: shared_ptr lifetime extension keeps transceiver state alive longer, exposing a pre-existing flow-control bug

PR #13713's load-bearing change is replacing `LlmRequest*` with `std::shared_ptr<LlmRequest>` in `mSenderFutures` / `mRequesterFutures`. Pre-PR, each entry held a raw pointer whose target's lifetime was controlled by Python's timeout-driven termination (which fires synchronously on a 60 s default). Post-PR, the C++ side holds an independent strong reference, so the request — and any in-progress send/recv buffers transitively pinned by it — stays alive longer.

For mpi_kvcache, "longer-lived in-flight state" plausibly means more concurrent transfers piled up in MPI's eager-message buffer pool. If the pool is sized for the pre-PR transfer cadence, the post-PR cadence may exceed it and deadlock as rendezvous waits for buffer drain.

**Evidence that fits**: the wedge consistently fires only after the full 8-request batch has been queued; it does not fire on smaller workloads. The transport stays alive (network polling threads are not in any error state). Both sides hold all 8 entries indefinitely.

**Evidence we lack**: a userspace stack frame inside the MPI library showing rendezvous-blocked semantics. Also no baseline measurement of pre-PR vs post-PR in-flight-state lifetime under this load.

### 4.2 Hypothesis B: always-on deadline checks change the polling cadence and starve the MPI progress engine

PR #13713 adds an unconditional total-deadline check inside `checkContextTransferStatus`'s polling loop that accesses `request->getKvCacheTransferStart()` every iteration. Pre-PR this access was conditional on `kvTransferTimeoutMs.has_value()` being true AND additional gates. Post-PR it runs on every iter regardless.

OpenMPI's progress engine relies on `MPI_*` API calls from user threads to drive forward progress (it does not have a dedicated progress thread by default). Subtle changes in how often the main thread calls `future.wait_for(0ms)` vs other work could shift the progress-cycle distribution in a way that starves one side's send completions while the other side's receives are eager.

**Evidence that fits**: both main threads are in `R running` state, busy-polling. Their polling cadence is *exactly* what PR #13713 changed.

**Evidence we lack**: a controlled comparison (instrument the polling rate, measure before/after the change in cacheTransceiver.cpp at the `unconditional deadline check` site).

### 4.3 Hypothesis C: ordering changes around `requestAndReceiveAsync` (`setKvCacheTransferStart(now())` rewind + state-transition reorder)

PR #13713 also adds `setKvCacheTransferStart(LlmRequest::getSteadyClockNow())` at the top of `requestAndReceiveAsync` (with a guard against repeated rewind) and moves `setState(IN_PROGRESS)` from before `receiveAsync` to after. Neither is the load-bearing change, but both alter the per-request initialisation ordering on the gen side.

If the mpi_kvcache transport relies on observing a specific (state, start-time) tuple before initiating a remote handshake — e.g. the matching ctx-side rank polling for a request in `kDISAGG_GENERATION_TRANS_IN_PROGRESS` state before generating a response — the post-PR ordering could miss a one-shot handshake window.

**Evidence that fits**: the wedge specifically appears on the per-request data path, after orchestration succeeds.

**Evidence we lack**: trace-level visibility into the mpi_kvcache transport's own handshake protocol (the `[NVBUG-6104831-INSTR]` records are at the application layer, not inside the transport).

### 4.4 To narrow the hypothesis space

A targeted bisect over PR #13713's 25 commits, starting with the shared_ptr migration commit, would distinguish hypothesis A from B/C. Specifically:

1. Check out PR #13713's first commit (`630fa3b4` per [README.md](README.md)).
2. Build, run `test_asymmetric_executor[llama-4proc-mpi_kvcache-103]`, observe pass/fail.
3. If it passes there, walk forward commit-by-commit to find the first failing commit.
4. Examine the diff at that commit against `main`.

This is roughly the bisect playbook in [05-investigation-timeline.md](05-investigation-timeline.md) Phase 14 (`bisect plan for helix CUDA illegal memory access regression`) but targeted at this cpp test instead of the helix Python test. Effort estimate: 1-2 engineer-days with a built-and-cached repo.

A cheaper alternative: a `git revert` of just the shared_ptr migration commits (and their direct fixups: `#5` sender broken promise, `#7` eval-order in handleAsyncSend) on top of PR HEAD, then re-run the cpp test. If it passes with the revert applied, hypothesis A is strongly supported. If it still fails, the cause is one of the other changes.

---

## 5. Open evidentiary gaps recap

For tracking against §6 of doc 12:

| Gap | Status pre- this doc | Status post- this doc |
|-----|----------------------|------------------------|
| `asymmetric_executor[mpi_kvcache]` deterministic case validates horizontal-consistency theory | Implied yes per doc 12 §4.3 | **No** — validates only condition 2 (stuck transfer); cascade not observed before timeout |
| Helix flake rate measurement | Not done | Started: 5/5 healthy runs on B300 (P(0/5) ≈ 60-77% if per-run rate is 5-10%). 20-run loop kicked off as part of this update. |
| Path A weak-mode UAF window measurement | Pending | Pending — unchanged |
| Effect on production users running helix-CP with default config | Pending | Pending — unchanged |
| Does Path B all-gather catch every divergence source | Pending | Pending — unchanged |
| **NEW**: which PR-#13713 commit introduces the cpp gtest transport hang | Not articulated | Three live hypotheses (§4); bisect or selective-revert recommended |

---

## 6. Related docs

- [03-defect-class-stack.md](03-defect-class-stack.md) — the L1-L10 invariant model. The cpp gtest wedge does not cleanly map to any single L layer; it is closest to L4 (timeout) but the timeout is not the cause, it is the symptom of an unrelated transport hang.
- [11-bisect-helix-uaf.md](11-bisect-helix-uaf.md) — prior bisect plan for the helix UAF. The selective-revert approach in §4.4 here is the same template applied to the cpp gtest.
- [12-horizontal-consistency-and-layer3-gating.md](12-horizontal-consistency-and-layer3-gating.md) — the theory document this addendum corrects. The theory remains live for the helix flakes; only §4.3's specific empirical anchor is refuted.
