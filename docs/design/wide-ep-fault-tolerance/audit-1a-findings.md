# Audit 1a — Historical intra-node findings and corrected evidence boundary

[< Back to Overview](README.md) | [§9.1 Audit 1](09-risks-and-open-questions.md#audit-1--baseline-mnnvl-teardown-and-rack-containment-capability)

**Status:** Historical Audit 1a snapshot. Days 1–3 completed (NCCL experiment + MPI signal-handler experiment + driver-side `cuMemUnmap`). The original Day 4–5 plan incorrectly treated IMEX as an intra-node B300 prerequisite; see the correction below. FABRIC/IMEX validation is now the explicit 1d.4a rack-fabric acceptance item.
**Hardware:** 8× NVIDIA B300 SXM6 (single node, GPUs connected through the platform's NVSwitch/NVLink domain).
**Software:** PyTorch 2.11.0a0+eb65b36914.nv26.02 (NCCL 2.29.2), OpenMPI 4.1.9a1 (FT Checkpoint: NO; no ULFM module), CUDA 13.1, Python 3.12, cuda-python 13.1.1.
**Date:** 2026-04-25.

## TL;DR (running)

Four Day 1–3 findings that affect Phase 2 sizing and PR 1d.0 scope:

1. **`torch.distributed` on PyTorch 2.11 does not provide a recoverable peer-death path** for any of the documented modes (`TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_BLOCKING_WAIT`). The headline-candidate `dist.shrink_group(ranks_to_exclude=…, shrink_flags=SHRINK_ABORT)` API itself hangs indefinitely after peer death in this build. **Implication for PR 2a.1 (NCCL teardown):** cannot be a one-liner around `dist.shrink_group`; needs lower-level work (drop to `ncclCommAbort` + `ncclCommInitRank` directly, or wait for a fixed PT release).
2. **A bounded host polling mechanism can surface a timeout.** `dist.all_reduce(…, async_op=True)` plus `work.is_completed()` polling delivered an exception at the configured deadline. This does **not** validate degraded execution: survivor-only management collectives (1c.3a/1c.4a), the atomic coordinator (1c.4b), request disposition (1c.4c), and runtime kernel escape (1a.8) were absent.
3. **The MPI propagation/lifecycle portion of prompt-evidence Q1/Q3 needs more than signal-handler replacement.** Replacing `MPI_Abort` with `_exit(N)` in `mpiUtils.cpp` (merged 1d.0) is necessary but not sufficient: this OpenMPI launcher also required `--mca orte_enable_recovery 1` for survivors to outlive a peer death. Even then, ordinary world collectives and `MPI_Finalize` remain unsafe; 1d.1 owns launcher/runtime admission and 1d.0a owns the poisoned-MPI lifecycle and shutdown contract.
4. **One POSIX-FD driver micro-case tears down quickly.** With the owner SIGKILLed, the survivor's `cuMemUnmap` returned `CUDA_SUCCESS` in **0.25 ms**, `cuMemRelease` in **1.27 ms**, and `cuMemAddressFree` in **0.008 ms**. This is useful local evidence for the x86_64 intra-node path, but it does not answer full `MnnvlMemory` recovery, fan-out to 71 survivors, workspace/communicator rebuild, or FABRIC/IMEX behavior. It therefore cannot close or precisely size PR 2a.2 by itself.

## Day 1 — NCCL rebuild prototype

**Question (per §9.1 Audit 1a Day 1):** Does PyTorch's `destroy_process_group` / `init_process_group` pattern work as a recovery path against our NCCL version after a peer SIGKILL? What's the rebuild latency?

**Test rig:** 4 ranks via `subprocess.Popen` workers + `TCPStore` rendezvous. Victim (rank 2) self-`SIGKILL`s after iter 5 of an `all_reduce` warm-up loop. Survivors detect the peer death and attempt to rebuild a working communicator with `world_size = 3`. Six configurations exercised over six runs:

| Run | Mode | Outcome | Survivor exit |
|:---|:---|:---|:---|
| 1 | `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `BLOCKING_WAIT=0` (PT default) | NCCL watchdog fires `std::terminate()` at exactly the configured 15 s timeout (`5044` ms in shorter run) | **All 3 survivors crash with SIGABRT (-6)** |
| 2 | `BLOCKING_WAIT=1`, `ASYNC=1` | Survivors hang silently past iter 5; never raise to main thread; killed by external `timeout 60` | hung |
| 3 | `BLOCKING_WAIT=1`, `ASYNC=0` | Same — `BLOCKING_WAIT` does not surface to main thread | hung |
| 4 | `async_op=True` + main-thread `work.is_completed()` polling, default env | **Detection works.** `TimeoutError` raised in main at the polling deadline (`5054` ms). `pg._abort()` then failed: `AttributeError: 'ProcessGroup' object has no attribute '_abort'` (wrong API name) | bad abort API |
| 5 | Polling + `dist.shrink_group([victim], shrink_flags=SHRINK_ABORT)`, `ASYNC=1` | Detection works; watchdog races with main thread, calls `terminate()` mid-`shrink_group` | SIGABRT mid-shrink |
| 6 | Polling + `shrink_group(SHRINK_ABORT)`, `ASYNC=0` | Detection works; `shrink_group` itself hangs > 60 s. NCCL log shows internal "First PG to signal dumping" but no progress on the shrink call's own internal collective | hung |

**Empirical numbers (from runs that produced data):**

| Metric | Value | Notes |
|:---|:---|:---|
| Detection latency (main-thread polling, 5 s budget) | **~5054 ms** | = budget + ≈ 50 ms poll granularity. Tunable freely. |
| Detection latency (PT default watchdog, 15 s budget) | **~15044 ms** | = configured watchdog. Then SIGABRT. |
| `dist.shrink_group(SHRINK_ABORT)` latency under failure | **> 60 s (timeout)** | Never completed in 2 attempts (with and without watchdog). |
| End-to-end recovery (rank dies → first new collective) | **Not measured** | Blocked on `shrink_group` not completing. |

**Conclusions for the design:**

- Default PT behavior is unrecoverable. **A single rank death today kills every survivor** with SIGABRT, in any code that uses `torch.distributed` collectives — TP `all_reduce` in attention, embedding `all_gather`, the `AllGatherReduceScatter` MoE fallback, etc. This is a real pre-existing exposure, separate from the Q2 live/silent MNNVL kernel hang.
- The bounded polling mechanism is useful for detection, but the earlier conclusion that it validated 1c.4 end-to-end is superseded. 1c.4 remains the model-engine hook; 1c.4a owns degraded membership and 1c.4b owns the recovery transaction.
- **PR 2a.1 (NCCL teardown) sizing must reflect that `dist.shrink_group(SHRINK_ABORT)` is not a working path in PT 2.11.** Either the work moves to a lower abstraction (call `ncclCommAbort` + `ncclCommInitRank` from C / via cython) or PR 2a.1 waits for an upstream PyTorch fix. Either way it's larger than a thin Python wrapper.

**Caveats:**

- All runs intra-node. Cross-node behavior may differ.
- We did not test `TORCH_NCCL_TRACE_BUFFER_SIZE` enabled (NCCL flight recorder); it might surface internal stalls.
- We did not try `pg.abort()` (no underscore) followed by a fresh `init_process_group`. Worth one more run to see if it's strictly better than `shrink_group`. (In practice it amounts to the same code path inside PT.)
- Did not test against a newer PyTorch nightly; if `shrink_group` is fixed upstream the verdict changes.

## Day 2 — MPI signal handler / `_exit` mitigation

**Question (per §9.1 Audit 1a Day 2 + §5.4 + PR 1d.0):** When one MPI rank dies abnormally, does it (A) take down all surviving ranks via `MPI_Abort` propagation, or (B) exit alone, letting survivors continue MPI on a smaller communicator? Does `_exit(2)` in `mpiUtils.cpp`'s signal handler change that?

**Test rig:** `mpirun -np 4` worker + per-mode launcher. Victim (rank 2) "dies" at iter 3 of an `Allreduce` loop in one of five ways selected via `AUDIT_DEATH_MODE`:

| Worker death mode | What happens in the worker |
|:---|:---|
| `default` | Uncaught `RuntimeError`. mpi4py's `atexit` hook catches the exception and calls `MPI_Abort` on `COMM_WORLD`. |
| `exit2` | `os._exit(2)` — bypasses Python finalizers and mpi4py's `atexit` hook. |
| `exit0` | `os._exit(0)` — same as above but with success exit code. |
| `abort` | Explicit `MPI.COMM_WORLD.Abort(2)`. |
| `sigkill` | `os.kill(self, SIGKILL)` — realistic external kill (OOM, hardware). |

Run with each launcher mode (default mpirun vs `--mca orte_enable_recovery 1`) where applicable.

**Run summary (7 modes × 4 ranks each):**

| Death mode | Launcher | Outcome | Survivor exit | Survivor signal caught | Elapsed |
|:---|:---|:---|:---|:---|:---|
| `default` | default mpirun | Hang past 60 s until external timeout | killed by timeout | None | > 60 s |
| `exit2` | default mpirun | All 3 survivors killed by mpirun | killed | None | 19.8 s |
| `exit0` | default mpirun | All 3 survivors killed by mpirun | killed | None | 19.3 s |
| `abort` | default mpirun | All 3 survivors get SIGTERM via PMIx, then killed | killed | SIGTERM (3 of 3) | 18.8 s |
| `sigkill` | default mpirun | All 3 survivors killed by mpirun | killed | None | 20.7 s |
| `recover_exit2` | `--mca orte_enable_recovery 1` | **Survivors stay alive but hang** in broken `Allreduce` | timed out at test budget | None | 45 s (= test budget) |
| `recover_sigkill` | `--mca orte_enable_recovery 1` | **Survivors stay alive but hang** | timed out at test budget | None | 45 s |

**Key finding F1.** mpi4py does **not** override Python signal handlers. The worker installs `_signal_handler` for `SIGTERM`/`SIGINT` *before* importing mpi4py, and `signal.getsignal(...)` after `MPI_Init` confirms the handler is still ours. Good news for PR 1d.0: in-process custom handlers in `mpiUtils.cpp` will survive `MPI_Init`.

**Key finding F2.** `mpirun` terminates the world on **any** abnormal exit, not just `MPI_Abort`. Tested 5 distinct death paths — `os._exit(0)`, `os._exit(2)`, `MPI_Abort(2)`, `SIGKILL`, and uncaught Python exception. All produce the same outcome under default mpirun: survivors die within 20 s. Even `os._exit(0)` (clean, success-coded) triggers mpirun's "child departed unexpectedly" path. **Replacing `MPI_Abort` with `_exit(2)` in `mpiUtils.cpp` does not save the survivors on its own.**

**Key finding F3.** `--mca orte_enable_recovery 1` stopped propagation in this OpenMPI 4.1.9a1 setup. With this flag, the same `exit2` and `sigkill` scenarios ran the full 45-second test budget without mpirun stepping in, although survivors still hung in the broken collective. This is deployment-specific evidence, not a portable MPI contract. Merged 1d.0 owns the catchable signal path; 1d.1 must validate/admit each launcher/runtime mode, and 1d.0a owns poisoned shutdown.

**Key finding F4.** Even with `enable_recovery`, survivors remain stuck inside `MPI.COMM_WORLD.Allreduce(...)` because (a) default error handler is `MPI_ERRORS_ARE_FATAL`, and (b) `Allreduce` has no timeout — it spins until completion or process death. To complete the recovery story the design needs to combine:

1. **Launcher/runtime admission (1d.1):** for this tested OpenMPI build, `mpirun --mca orte_enable_recovery 1`; other supported environments require equivalent destructive evidence rather than assuming this flag is portable.
2. **Failure notification:** `MPI_ERRORS_RETURN` on the 1c.3 FT signaling communicator.
3. **Survivor control membership:** 1c.3a `ActiveRankMap` plus 1c.4a conversion of ordinary PyExecutor/attention-DP management collectives to survivor-only membership.
4. **Recovery transaction:** 1c.4b coordinates abort, reconciliation, admission, quiescence, EPLB preparation, survivor communicator rebuild, graph policy, and the atomic mask + `ActiveRankMap` + generation commit; 1c.4c applies request disposition before resume.

(2) and (3) are already in the design. The new addition surfaced by Day 2 is item (1).

**Key finding F5.** Replacing `MPI_Abort` with `_exit(N)` in TRT-LLM's signal handler also makes failure detection **faster**, even before adding `enable_recovery`. The `default` mode (uncaught RuntimeError → mpi4py atexit → `MPI_Abort`) hung past 60 s while all the `_exit(N)` modes terminated in ~20 s. So PR 1d.0's `_exit(N)` change has standalone value (faster TTL on a bad rank) even if the launch flag is missing.

**Conclusions for the design:**

- The §3 prompt-evidence MPI diagnosis was confirmed empirically: `mpirun` is a propagation mechanism, not just `MPI_Abort` itself.
- The launcher requirement is deployment-specific evidence from this OpenMPI build, not a universal MPI flag. Merged 1d.0 removes handler `MPI_Abort`; 1d.1 admits a tested survivor-preserving launcher/runtime mode, while 1d.0a prevents poisoned-world collectives/`MPI_Finalize` during survivor shutdown.
- §5.3 / PR 1c.3 (FT subcomm with `MPI_ERRORS_RETURN`) is necessary, not optional. Audit confirms.

**Caveats:**

- Single-node OpenMPI 4.1.9a1, no ULFM. Multi-node behavior may differ; cross-node validation needed before Phase 2 work locks in the launcher convention.
- 5 worker death modes, 1 launcher recovery flag. Did not test mpirun under a process supervisor (systemd, K8s) which may behave differently.
- Did not yet test the full F4 combination (`enable_recovery` + `MPI_ERRORS_RETURN` + bounded-wait collective). That's the natural follow-up — would prove the survivor-recovers path end-to-end. Estimated 1–2 hours of additional prototyping if needed for PR 1c.3 sizing.

## Day 3 — `cuMemUnmap` on dead-peer regions

**Question (per §9.1 Audit 1a Day 3):** When the process that allocated a `cuMemCreate`-backed region is `SIGKILL`ed, can a peer that has cross-process-imported the same region still cleanly call `cuMemUnmap` / `cuMemRelease` / `cuMemAddressFree` on its mapping? Or does it segfault, hang, or return a graceful error?

**Test rig:** three coordinated processes (launcher, owner, peer). Owner allocates with `cuMemCreate`, exports a shareable handle, sends it to peer over a Unix-domain socket (FD passing for posix-FD mode; 64-byte struct for fabric mode). Peer imports, maps, reads & verifies the owner's pattern, signals "ready". Launcher then `SIGKILL`s the owner. Peer waits 2 s, then attempts a read, then runs the unmap/release/free sequence and logs each call's `CUresult` and elapsed time.

### Posix-FD variant (`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`) — **passes cleanly**

| Step | Result | Elapsed |
|:---|:---|:---|
| Pre-kill cross-process share verified (peer reads owner's `OWN…` pattern) | ✅ correct bytes | — |
| Owner SIGKILLed | exit code -9 | — |
| Peer `cudaMemcpy DtoH` 2 s after kill | ✅ `CUDA_SUCCESS`, **same `OWN…` pattern still present** | 1.6 ms |
| `cuMemUnmap` on dead-peer mapping | ✅ `CUDA_SUCCESS` | **0.25 ms** |
| `cuMemRelease` on imported handle | ✅ `CUDA_SUCCESS` | **1.27 ms** |
| `cuMemAddressFree` on VA reservation | ✅ `CUDA_SUCCESS` | **0.008 ms** |

**Total driver-side teardown: ~1.5 ms.** No segfault, no hang. The mapping itself survives the owner's death (peer can still read the data 2 s after SIGKILL); cleanup is a clean local-process operation.

This isolates one local CUDA-driver operation from any fabric-specific subsystem. It does not establish an upper bound for full PR 2a.2 recovery: fan-out, imported mappings, fabric membership, handle exchange, workspace allocation, communicator construction, and quiescence were absent.

### Fabric-handle variant (`CU_MEM_HANDLE_TYPE_FABRIC`) — separate rack-path evidence

Repeated the same micro-test with FABRIC handles. This is **not** the normal current `MnnvlMemory` handle type on the tested x86_64 B300 host; it is the relevant path for Grace/aarch64 NVL72. Result:

```
cuMemCreate(prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC)
  -> CUDA_ERROR_NOT_PERMITTED: operation not permitted
```

The device attribute reported fabric support (`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 1`) and the kernel-side channel existed, but the host was not configured as an IMEX domain, so fabric-handle creation was denied. That outcome neither blocks nor invalidates the x86 POSIX-FD intra-node path; it simply provides no 1d.4a evidence.

**Implication for the audit split:**

- The POSIX-FD result proves only its tested local operation. Do not infer FABRIC equivalence from shared VA/mapping APIs.
- Re-running the micro-test on NVL72 is useful Audit 1b setup evidence, but 1d.4a must exercise the full production-component process-death flow and a separately approved inaccessible-peer-memory/device-loss case. Neither a five-minute `cuMemUnmap` test nor healthy-GPU SIGKILL proves Q3 containment.

### Caveats

- 2 MiB allocation (rounded up by `cuMemGetAllocationGranularity` from 4 KiB request). Larger allocations might exhibit different behavior (e.g., page-by-page invalidation under memory pressure); intra-node test does not stress this.
- Single owner / single peer. WideEP has 71 survivor peers per dead rank; whether cleanup latency scales linearly or has fan-out cost is an Audit 1b question.
- Did not test the **rebuild** half (allocate a fresh fabric region to replace the dead one). Driver-side `cuMemCreate` / `cuMemMap` on a healthy region is well-characterized as ms-scale; the new question is only whether the fabric handle exchange across N-1 surviving ranks is bounded — same Audit 1b territory as the cleanup.

## Pending work (Days 3–5 status)

| Day | Item | Status | Blocking dependency |
|:---|:---|:---|:---|
| 3 | `cuMemUnmap` semantics on dead-peer regions (posix-FD) | ✅ **Done** | — |
| 3 | `cuMemUnmap` semantics on dead-peer regions (FABRIC handle) | Deferred to Audit 1b / 1d.4a setup evidence | The tested x86_64 B300 is a POSIX-FD production path; FABRIC/IMEX evidence requires Grace/aarch64 NVL72 or equivalent. No equivalence is assumed. |
| 3 | DeepEP destructor (`Buffer.__del__` → `intranode::barrier`) deadlock + explicit `destroy()` ordering | Not started | DeepEP requires `tensorrt_llm` package import, which is broken in this dev tree (pre-existing `fp4_quantize_with_residual` mirror error). Container with matching C++ binary needed. |
| 4–5 | **Intra-node MNNVL teardown + reallocate prototype** | Superseded by no-mock MVP integration prototype | On x86_64 DGX/HGX B200/B300, TRT-LLM's `MnnvlMemory` selects `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`; IMEX is not required merely because the GPUs are connected by NVSwitch. Run the real production component there. Reserve FABRIC/IMEX for 1d.4a on Grace/aarch64 NVL72 or equivalent. |
| 5 | NVSHMEM teardown / `nvshmem_finalize` behavior | Not started | `nvshmem` Python module not installed in this env |
| 5 | Written report (this document is a partial substitute) | Partial | This file covers Days 1–3; remaining gaps are gated on container / IMEX / nvshmem unblockers |
| 6 | **CUDA graph re-capture latency under NCCL rebuild** | Open MVP validation | Measures recapture after `ncclCommAbort` + `ncclCommInitRank` for promoted MVP item 1a.11. Until measured, the prototype runs eager and treats graph invalidation/recapture as a ship gate, not a v1 optimization. |

### What's left that's actually unblocked on this hardware

- **DeepEP destructor test.** Needs a working `tensorrt_llm` import or a from-scratch DeepEP `Buffer` construction. Either rebuild the C++ binary in this dev tree against current Python, or move the test to an NGC container with a matching pair. ~1–2 hours either way.
- **NVSHMEM `nvshmem_finalize` behavior.** Needs `nvshmem` Python module. If we ship NVSHMEM-Py with TRT-LLM, easy install; otherwise build from source. Probably ~30 min if the binary is available, several hours if not.

### What's left that needs different hardware / IMEX setup

- FABRIC-handle `cuMemUnmap` behavior and full rack recovery, with no equivalence inferred from POSIX-FD (Audit 1b / 1d.4a).
- Production `MnnvlMemory` on the x86_64 B300 node uses the POSIX-FD handle path and is suitable for the intra-node no-mock prototype. It does not prove the Grace/aarch64 FABRIC path.
- **All "rack-fabric specific" questions in §9.1 Audit 1b** — explicitly out of scope for 1a.

## Hardware notes

This audit ran on an **8-GPU B300 SXM6 system with NVSwitch**. The meaningful software-path distinction is not "NVSwitch versus no NVSwitch": TRT-LLM's Python `MnnvlMemory` currently selects POSIX-FD sharing on x86_64 B200/B300 hosts and FABRIC handles on Grace/aarch64 GB200/GB300. The former validates production components intra-node; only the latter exercises IMEX and rack-fabric grant semantics. Item 1d.4a therefore remains a separate acceptance run on NVL72 or equivalent FABRIC/IMEX hardware.

## Files and reproduction

The runnable prototypes live alongside this file in
[`research-pass-prototypes/`](research-pass-prototypes/) on the same branch.
Two representative result files are checked in under
[`research-pass-prototypes/sample-results/`](research-pass-prototypes/sample-results/);
re-running a prototype regenerates a full set of per-run JSONL logs into
`/tmp/audit-1a-prototypes/<test_name>/` (override with `--log-dir` /
`--results-dir`).

```
research-pass-prototypes/
  nccl_rebuild.py                          # Day 1 worker + launcher (single file)
  mpi_signal_handler.py                    # Day 2 worker
  mpi_signal_launcher.py                   # Day 2 launcher (per-mode)
  cumem_unmap_dead_peer.py                 # Day 3 posix-FD variant (passes)
  cumem_unmap_dead_peer_fabric.py          # Day 3 fabric-handle variant (gated on IMEX)
  README.md                                # script index + reproduction commands
  sample-results/
    mpi_signal_handler-summary.json        # Day 2 last-run aggregated summary
    cumem_unmap_dead_peer-peer.jsonl       # Day 3 posix-FD per-event log (full success path)
```

Reproducing on another ≥ 4-GPU node is single-command per track. See
[`research-pass-prototypes/README.md`](research-pass-prototypes/README.md)
for the full command list; the headlines are:

```bash
# NCCL Day 1 — single best run (Run 6, polling + shrink_group + ASYNC=0):
python3 research-pass-prototypes/nccl_rebuild.py --launch --world-size 4 \
  --victim-rank 2 --victim-iter 5 --watchdog-sec 5 --master-port 29508

# MPI Day 2 — full mode sweep:
python3 research-pass-prototypes/mpi_signal_launcher.py --per-mode-timeout-sec 45 \
  --modes default exit2 exit0 abort sigkill recover_exit2 recover_sigkill

# Driver Day 3 — posix-FD variant (works on any CUDA 12+ node):
python3 research-pass-prototypes/cumem_unmap_dead_peer.py

# Driver Day 3 — fabric-handle variant (needs IMEX-configured node, e.g. NVL72):
python3 research-pass-prototypes/cumem_unmap_dead_peer_fabric.py
```
