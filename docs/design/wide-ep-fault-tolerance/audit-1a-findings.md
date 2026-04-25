# Audit 1a — Intra-node findings (in progress)

[< Back to Overview](README.md) | [§9.1 Audit 1](09-risks-and-open-questions.md#audit-1--mnnvlnvshmem-teardown-capability)

**Status:** Days 1–3 of 5 complete (NCCL rebuild + MPI signal handler + driver-side `cuMemUnmap`). Day 4–5 work (intra-node MNNVL fabric prototype) **gated on IMEX daemon configuration** which is not active on this node — fabric handle creation returns `CUDA_ERROR_NOT_PERMITTED`. Equivalent driver-mechanism validation completed via the posix-FD variant; fabric-handle equivalence is an Audit 1b validation point.
**Hardware:** 8× NVIDIA B300 SXM6 (single node), NVLink full mesh.
**Software:** PyTorch 2.11.0a0+eb65b36914.nv26.02 (NCCL 2.29.2), OpenMPI 4.1.9a1 (FT Checkpoint: NO; no ULFM module), CUDA 13.1, Python 3.12, cuda-python 13.1.1.
**Date:** 2026-04-25.

## TL;DR (running)

Four Day 1–3 findings that affect Phase 2 sizing and PR 1d.0 scope:

1. **`torch.distributed` on PyTorch 2.11 does not provide a recoverable peer-death path** for any of the documented modes (`TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_BLOCKING_WAIT`). The headline-candidate `dist.shrink_group(ranks_to_exclude=…, shrink_flags=SHRINK_ABORT)` API itself hangs indefinitely after peer death in this build. **Implication for PR 2a.1 (NCCL teardown):** cannot be a one-liner around `dist.shrink_group`; needs lower-level work (drop to `ncclCommAbort` + `ncclCommInitRank` directly, or wait for a fixed PT release).
2. **Detection in pure Python works.** `dist.all_reduce(…, async_op=True)` plus a main-thread `work.is_completed()` polling loop delivers a clean exception at exactly the configured deadline. This validates §5.3 / 1c.4's design assumption that detection can sit above NCCL rather than inside it.
3. **Mode A (§3) needs a two-part fix, not one.** Replacing `MPI_Abort` with `_exit(N)` in `mpiUtils.cpp` (PR 1d.0) is necessary but **not sufficient**: `mpirun` itself propagates termination on any abnormal child exit, regardless of the exit code. Audit empirically confirms the launch flag `--mca orte_enable_recovery 1` is also required for survivors to outlive a peer death. With both fixes, survivors stay alive; without recovery flag, they die in 18–60 s under default mpirun.
4. **Driver-side teardown of dead-peer regions is essentially free.** With the owner SIGKILLed, the survivor's `cuMemUnmap` returns `CUDA_SUCCESS` in **0.25 ms**, `cuMemRelease` in **1.27 ms**, `cuMemAddressFree` in **0.008 ms** — total ~1.5 ms. The mapping itself even survives the kill: 2 s after the SIGKILL the survivor still reads back the owner's pre-kill data correctly. This is the answer PR 2a.2 was waiting for: **driver-side cleanup is not a Phase 2 bottleneck**, ms-scale at most. (Tested with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`; the equivalent `CU_MEM_HANDLE_TYPE_FABRIC` test is gated on IMEX setup — see Day 3 section.)

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

- Default PT behavior is unrecoverable. **A single rank death today kills every survivor** with SIGABRT, in any code that uses `torch.distributed` collectives — TP `all_reduce` in attention, embedding `all_gather`, the `AllGatherReduceScatter` MoE fallback, etc. This is a real pre-existing exposure, separate from the NVLink kernel hang (Mode B).
- §5.3 / 1c.4's main-thread detection design is sound. The `async_op=True` + polling pattern is a clean substitute for the unusable PT watchdog and gives full control over detection latency.
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

**Key finding F3.** `--mca orte_enable_recovery 1` stops the propagation. With this flag, the same `exit2` and `sigkill` death scenarios run the full 45 s test budget without mpirun stepping in. **This is the missing piece for PR 1d.0.** It's a launch-flag change, not a source change, but it must be documented and (ideally) defaulted on by `trtllm-serve` when FT is enabled.

**Key finding F4.** Even with `enable_recovery`, survivors remain stuck inside `MPI.COMM_WORLD.Allreduce(...)` because (a) default error handler is `MPI_ERRORS_ARE_FATAL`, and (b) `Allreduce` has no timeout — it spins until completion or process death. To complete the recovery story the design needs to combine:

1. **Launch flag:** `mpirun --mca orte_enable_recovery 1` (PR 1d.0 documents this; ideally `trtllm-serve` sets it when `--enable-fault-tolerance`).
2. **Per-rank error handler:** `comm.Set_errhandler(MPI.ERRORS_RETURN)` on the FT subcomm (already PR 1c.3).
3. **Main-thread bounded wait on collectives:** the same `async_op=True` + polling pattern proven in the NCCL track (already PR 1c.4-style detection).

(2) and (3) are already in the design. The new addition surfaced by Day 2 is item (1).

**Key finding F5.** Replacing `MPI_Abort` with `_exit(N)` in TRT-LLM's signal handler also makes failure detection **faster**, even before adding `enable_recovery`. The `default` mode (uncaught RuntimeError → mpi4py atexit → `MPI_Abort`) hung past 60 s while all the `_exit(N)` modes terminated in ~20 s. So PR 1d.0's `_exit(N)` change has standalone value (faster TTL on a bad rank) even if the launch flag is missing.

**Conclusions for the design:**

- §3 Mode A diagnosis confirmed empirically: mpirun is the propagation mechanism, not just `MPI_Abort` itself.
- **PR 1d.0 scope expands by one item:** must include a launcher / docs note that `--mca orte_enable_recovery 1` is required when FT is enabled. Trivial to add but mandatory.
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

This isolates the CUDA driver behavior from any fabric-specific subsystem. **PR 2a.2 sizing: driver-side cleanup is not a Phase 2 bottleneck, accounting for at most 1–2 ms of the recovery budget.**

### Fabric-handle variant (`CU_MEM_HANDLE_TYPE_FABRIC`) — gated on IMEX

Repeated the same test with fabric handles (the actual MNNVL allocation type). Result:

```
cuMemCreate(prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC)
  -> CUDA_ERROR_NOT_PERMITTED: operation not permitted
```

The device attribute reports fabric is supported (`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 1`), and the kernel-side IMEX channel exists (`dmesg`: `nv-caps-imex channel0 created`), but the user-space `nvidia-imex` daemon is not running on this single-node B300 host. Without the IMEX domain configured, fabric handle creation is denied at the driver level.

**Implication for the audit split:**

- The driver mechanism itself is proven to behave gracefully (posix-FD variant). The fabric-handle path uses the same underlying VA / mapping machinery; the difference is the inter-process address-space coupling, which on rack-fabric NVLink is mediated by IMEX.
- Confirming fabric-handle equivalence numerically is therefore an **Audit 1b validation point**, not a separate bottleneck. When NVL72 (or any IMEX-configured node) becomes available, re-running the fabric variant is a 5-minute test against the same prototype script.

### Caveats

- 2 MiB allocation (rounded up by `cuMemGetAllocationGranularity` from 4 KiB request). Larger allocations might exhibit different behavior (e.g., page-by-page invalidation under memory pressure); intra-node test does not stress this.
- Single owner / single peer. WideEP has 71 survivor peers per dead rank; whether cleanup latency scales linearly or has fan-out cost is an Audit 1b question.
- Did not test the **rebuild** half (allocate a fresh fabric region to replace the dead one). Driver-side `cuMemCreate` / `cuMemMap` on a healthy region is well-characterized as ms-scale; the new question is only whether the fabric handle exchange across N-1 surviving ranks is bounded — same Audit 1b territory as the cleanup.

## Pending work (Days 3–5 status)

| Day | Item | Status | Blocking dependency |
|:---|:---|:---|:---|
| 3 | `cuMemUnmap` semantics on dead-peer regions (posix-FD) | ✅ **Done** | — |
| 3 | `cuMemUnmap` semantics on dead-peer regions (fabric handle) | ⛔ Blocked → Audit 1b | `nvidia-imex` daemon not active on this single-node B300; fabric handle creation returns `CUDA_ERROR_NOT_PERMITTED`. Equivalence to posix-FD result is the validation point. |
| 3 | DeepEP destructor (`Buffer.__del__` → `intranode::barrier`) deadlock + explicit `destroy()` ordering | Not started | DeepEP requires `tensorrt_llm` package import, which is broken in this dev tree (pre-existing `fp4_quantize_with_residual` mirror error). Container with matching C++ binary needed. |
| 4–5 | **Intra-node MNNVL teardown + reallocate prototype** (the centerpiece of Audit 1a) | ⛔ Blocked → Audit 1b | Same IMEX gate as the fabric `cuMemUnmap` test. The driver-side mechanism question (the harder of the two) is already answered by Day 3 posix-FD; the remaining open question is fabric-specific (IMEX + NVSwitch fabric manager interaction), which lands naturally in 1b. |
| 5 | NVSHMEM teardown / `nvshmem_finalize` behavior | Not started | `nvshmem` Python module not installed in this env |
| 5 | Written report (this document is a partial substitute) | Partial | This file covers Days 1–3; remaining gaps are gated on container / IMEX / nvshmem unblockers |

### What's left that's actually unblocked on this hardware

- **DeepEP destructor test.** Needs a working `tensorrt_llm` import or a from-scratch DeepEP `Buffer` construction. Either rebuild the C++ binary in this dev tree against current Python, or move the test to an NGC container with a matching pair. ~1–2 hours either way.
- **NVSHMEM `nvshmem_finalize` behavior.** Needs `nvshmem` Python module. If we ship NVSHMEM-Py with TRT-LLM, easy install; otherwise build from source. Probably ~30 min if the binary is available, several hours if not.

### What's left that needs different hardware / IMEX setup

- Fabric-handle `cuMemUnmap` equivalence (already framed as Audit 1b validation).
- Intra-node MNNVL prototype using `MnnvlMemory` (TRT-LLM Python wrapper requires fabric handles internally; same IMEX gate).
- **All "rack-fabric specific" questions in §9.1 Audit 1b** — explicitly out of scope for 1a.

## Hardware notes

This audit ran on **B300 SXM6** which is fabric-capable but is not the same as the NVL72 rack fabric. The intra-node fabric API is exercised by Day 4–5 (when those run), but item §9.1 Audit 1b — rack-fabric validation — remains a separate, independent effort. The work here is intentionally scoped to "run anywhere with ≥ 4 NVLinked GPUs" so rack time is a smaller follow-on.

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
