# Pre-Drafting Research Pass — Findings Report

**Created:** 2026-04-23 (research pass) | **Updated:** 2026-04-25 (Audit 1a Days 1–3 empirical follow-up to Item 7)
**Companion to:** `redesign-research-pass.md` (the items list)
**Time spent:** ~half a day pre-drafting + ~5 hr Audit 1a Days 1–3 prototyping
**Status:** Items 1–6 verified; **item 7 partially answered by Audit 1a Days 1–3** (sections below); item 8 light-pass complete.

> **Historical evidence snapshot (superseded for execution status).** The confirmations below apply to the April 2026 source revision. They must not be read as current "zero uses," capacity, or PR-status claims. In particular, PR #13404 raised `kMaxRanks` from 64 to 128 and added the launch-time mask foundation. The June correction pass promoted 1a.8/1a.11, separated detection from committed membership, and added 1b.2a, 1c.3a, 1c.4a, 1c.4b, 1c.4c, 1d.0a, and 1d.4a; 1c.4 remains the model-engine hook.

---

## ✅ Confirmed

**Item 6 — v1 line anchors all hold against current source:**
- At the April snapshot, `kMaxRanks = 64` in `moeAlltoAllKernels.h`; PR #13404 has since raised the current limit to 128 ✓ historical / resolved
- 300s in-kernel timeout `(300ll * 2000ll * 1000ll * 1000ll)` confirmed in `moeAlltoAllKernels.cu` (`#define check_timeout(s) ...`) ✓
- Dispatch release+wait + combine release+wait loops with `st.relaxed.sys.u32` / `ld.relaxed.sys.u32` PTX confirmed; both end with `asm volatile("trap;")` on timeout ✓
- Combine accumulator's `dst_idx < 0` → `acc[k].fill(0.0f)` skip pattern confirmed (the natural template for routing-pass masking) ✓
- `MoeLoadBalanceMetaInfo` fields (`expertCount, topK, epRank, epSize, slotCountPerRank`) confirmed in `moeLoadBalanceCommon.h` ✓
- `MoePlacementCpuInfo.rankExpertIds` shape and `oldRankExpertIds` rollback confirmed in `moeLoadBalancer.h` ✓
- In-place `cudaMemcpyAsync` placement propagation in `moeLoadBalancer.cpp` (no double buffer) confirmed ✓
- **`mpiUtils.cpp` signal handlers DO call `MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)`** — the reviewer's central claim is verified verbatim. Two variants exist: one additionally `kill(getppid(), SIGKILL)` before MPI_Abort ✓
- **Zero non-test uses** of `ncclCommAbort`, `NCCL_ASYNC_ERROR_HANDLING`, `ncclCommFinalize`, `ncclGetLastError`, `MPI_ERRORS_RETURN`, `MPI_Comm_revoke`, ULFM ✓
- DeepEP `Buffer.__del__` → `intranode::barrier` deadlock awareness present in TRT-LLM Python wrappers (`deep_ep.py:86`, `deep_ep_low_latency.py:103`, `configurable_moe.py:422`) ✓

**Item 1 — MPIPoolExecutor failure semantics (reviewer claim verified):**
- `proxy.py:225–280` (`_start_executor_workers`): registers `mpi_done_callback` per future. The callback only enqueues the exception (`_error_queue.put_nowait(future.exception())`) — no recovery, no salvage.
- `mpi_session.py:136–186` (`MpiPoolSession.abort()`): calls `self.get_comm().Abort(1)` → kills entire MPI world.
- Reviewer claim "MPIPoolExecutor becomes permanently broken" matches reality.

**Item 2 — HostMoeTensorSharer Ray-path:**
- Class at `moe_load_balancer.py:127`. `Split_type(MPI.COMM_TYPE_SHARED)` at lines 896–897.
- **Zero `TLLM_DISABLE_MPI` / `is_mpi_session` / `is_mpi_available` guards in this file.** MPI is hard-baked.
- Implication: porting cost for any future Ray pivot is real. Not a no-op.

**Item 3 — RayExecutor parity:**
- File: `tensorrt_llm/executor/ray_executor.py`. Class `RayExecutor(RpcExecutorMixin, GenerationExecutor)`.
- Has: `submit`, `abort_request`, `shutdown`, `create_workers`, `init_workers_sync/async`, `call_all_ray_workers`, `collective_rpc`, `report_device_ids`, `enable_postprocess_parallel`.
- Same `GenerationExecutor` base class as MPI path. API contract is parity for the operations WideEP uses.

**Item 4 — Aggregated NVL72 launch path:**
- `trtllm-serve` = `tensorrt_llm/commands/serve.py` (1463 lines). `launch_server` at line 270.
- Launch detection: `serve.py:1211` checks `OMPI_COMM_WORLD_RANK` (OpenMPI env var). If set, `trtllm-serve` attaches to the existing MPI world via `MPICommExecutor` (`:1244, 1292`). Otherwise, `LLM` constructor path creates `MpiPoolSession` which spawns workers via `MPIPoolExecutor` (mpi_session.py:178).
- **Canonical launch for WideEP:** user runs `mpirun -np N trtllm-serve <model> --tp <N> --ep <N> ...` (or SLURM `srun` equivalent). The `mpirun` path is the production default; `MpiPoolSession` spawn-from-single-process is a fallback for local/dev.
- Orchestrator selection: `llm_args.py:2903` defines `orchestrator_type: Optional[Literal["rpc", "ray"]] = None`; `None` → MPI default.

## 🔄 Refined / Corrected

**Item 5 — Ray-path CI coverage** *(claim needs softening):*
- Ray-path CI exists. Tests in `l0_dgx_b200.yml` / `l0_dgx_h100.yml` / `l0_h100.yml`:
  - `test_llm_multi_gpu_pytorch.py -m "gpu4"` (4 GPUs)
  - `accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_fp8_4gpus[tp4-...]` (Llama 3.1 8B, TP=4)
  - `examples/test_ray.py::test_llm_inference_distributed_ray[tp2pp2]` (TP=2, PP=2)
  - `examples/test_ray.py::test_ray_disaggregated_serving[tp2]` (Ray + disagg)
  - `disaggregated/test_disaggregated.py::test_disaggregated_ctxpp4_gentp4[TinyLlama-1.1B...]`
  - Dedicated dir: `tests/integration/defs/accuracy/test_llm_api_pytorch_ray.py` + `unittest/_torch/ray_orchestrator/multi_gpu/`
- **No EP markers, no `ep_size` keyword, no DeepSeek-V3, nothing at scale ≥ 8 GPUs in these tests.** Largest is TP=4.
- **Refined framing for §3.3 / §11:** "Ray-path has functional CI at TP ≤ 4 but is not exercised at WideEP scale (EP ≥ 32) or with DeepSeek-class models." The claim isn't "Ray is untested" — it's "Ray is not characterized at the scale where this design lives."

## 🆕 New gaps surfaced

1. **Ray + disagg + NIXL is unsupported** (`tests/integration/defs/disaggregated/test_disaggregated.py:597`): "Ray orchestrator is not supported with NIXL(DEFAULT) cache transceiver backend." Material for §9 future-migration risk and Phase 1-DS scoping. Disagg + Ray works only with non-NIXL transceivers today.

2. **`mpiUtils.cpp` `forwardAbortToParent` variant additionally `kill(getppid(), SIGKILL)`** before `MPI_Abort`. The reviewer's bare-claim quote understates this — it's not just propagation through MPI, the launcher process is also signaled. This became part of §3.1's Q1/Q3 prompt-evidence MPI analysis.

3. **Two-mode launch path** (`mpirun + MPICommExecutor` vs `MpiPoolSession + MPIPoolExecutor`) — both exist, with different failure modes. The §1.1 user journey should specify which case it walks through (the production `mpirun` case).

## 🧪 Empirical follow-up — Audit 1a partial (Item 7)

**Hardware:** 8× B300 SXM6 single node, connected through its NVSwitch/NVLink domain. **Software:** PyTorch 2.11 (NCCL 2.29.2), OpenMPI 4.1.9a1 (no ULFM), CUDA 13.1, cuda-python 13.1.1. **Date:** 2026-04-25. **Tracks completed:** NCCL experiment (Day 1, 6 runs / 6 modes), MPI signal handler (Day 2, 7 runs / 7 modes), driver-side POSIX-FD `cuMemUnmap` variant (Day 3). POSIX-FD is the relevant TRT-LLM `MnnvlMemory` handle mode on x86_64 B200/B300; Grace/aarch64 NVL72 uses FABRIC handles and IMEX and remains 1d.4a scope.

### Headline findings

1. **PT 2.11's `torch.distributed` has no recoverable peer-death path** for any documented mode. `dist.shrink_group(ranks_to_exclude=…, shrink_flags=SHRINK_ABORT)` — the API the design hoped for — itself hangs > 60 s after a peer SIGKILL in this build (with or without `ASYNC_ERROR_HANDLING`). Default config (`ASYNC=1, BLOCKING=0`) SIGABRTs all survivors at the watchdog timeout.
2. **The isolated timeout detector works.** `dist.all_reduce(…, async_op=True)` + main-thread `work.is_completed()` polling delivered a clean exception at the configured deadline. It did not validate survivor-only management collectives or recovery commit; those are now 1c.3a, 1c.4a, and 1c.4b.
3. **The Q1/Q3 MPI propagation path needs a lifecycle fix beyond 1d.0.** Replacing `MPI_Abort` with `_exit(N)` is necessary but not sufficient. This OpenMPI build also needed `--mca orte_enable_recovery 1`; 1d.1 therefore owns launcher/runtime admission, while survivors still require 1c.3 notification, 1c.3a/1c.4a survivor membership, and 1d.0a protection from poisoned-world collectives and `MPI_Finalize`.
4. **One POSIX-FD driver micro-case cleans up quickly.** Cross-process share + owner SIGKILL produced `cuMemUnmap` **0.25 ms**, `cuMemRelease` **1.27 ms**, and `cuMemAddressFree` **0.008 ms** in the tested peer. This bounds only those local calls; it does not bound full MNNVL teardown/reallocation, 71-peer fan-out, communicator/workspace rebuild, or FABRIC/IMEX behavior.

### Empirical numbers (intra-node, single survivor / single peer)

| Metric | Value | Notes |
|:---|:---|:---|
| NCCL detection latency (main-thread polling, 5 s budget) | ~5054 ms | = budget + ~50 ms poll granularity |
| NCCL detection latency (PT default watchdog, 15 s budget) | ~15044 ms | then SIGABRT |
| `dist.shrink_group(SHRINK_ABORT)` post-failure latency | > 60 s (timeout) | never completed in 2 attempts |
| MPI survivors lifetime under default mpirun (any death mode) | 18–60 s | mpirun terminates the world |
| MPI survivors lifetime under `--mca orte_enable_recovery 1` | indefinite (test budget) | survivors stay alive but hang in collective |
| `cuMemUnmap` on dead-peer region (posix-FD) | **0.25 ms** | `CUDA_SUCCESS` |
| `cuMemRelease` on imported handle (posix-FD) | **1.27 ms** | `CUDA_SUCCESS` |
| `cuMemAddressFree` on VA reservation (posix-FD) | **0.008 ms** | `CUDA_SUCCESS` |

### What lands in §9.1 Audit 1b (deferred to NVL72-class hardware)

- `CU_MEM_HANDLE_TYPE_FABRIC` teardown and the full production recovery flow on Grace/aarch64 NVL72. The B300 host was not an IMEX domain, so `cuMemCreate(... FABRIC)` returned `CUDA_ERROR_NOT_PERMITTED`; POSIX-FD behavior does not prove FABRIC equivalence.
- The no-mock x86_64 intra-node `MnnvlMemory` recovery prototype is **not** IMEX-gated because current TRT-LLM selects POSIX-FD there. It is 1d.4 work; rack FABRIC/IMEX is the separate 1d.4a gate.
- DeepEP destructor + explicit `destroy()` ordering (Day 3 secondary item): blocked here on a `tensorrt_llm` package import error (pre-existing `fp4_quantize_with_residual` mirror issue in this dev tree). Needs a container with matching C++ binary.
- NVSHMEM `nvshmem_finalize` behavior (Day 5): `nvshmem` Python module not installed in this env.

### Caveats

- Single-node intra-node only. Cross-node propagation (NCCL FT, MPI mpirun behavior, fabric memory teardown) may differ.
- 4-rank scale on the rebuild and signal-handler tests; 1 owner / 1 peer on the cuMemUnmap test. Fan-out cost at 71 surviving peers per dead rank is unknown.
- Did not run a production-component recovery transaction. Combining launcher settings and `MPI_ERRORS_RETURN` in another micro-test would still omit 1a.8, 1b.2a, 1c.3a, 1c.4a–1c.4c, 1d.0a, real requests, and physical acceptance.

## ❓ Deferred (intentionally)

- **Item 7 — NVSHMEM/MNNVL teardown.** Isolated Days 1–3 evidence is retained above. The no-mock intra-node production path (1d.4), NVSHMEM-specific work if that backend is selected, and rack FABRIC/IMEX acceptance (1d.4a/Audit 1b) remain.
- **Item 8 — Disagg Ray (light pass complete)** — see new gap #1 above.

---

## Implications for the rewrite

1. §3.1 Q1/Q3 MPI propagation path — quote `mpiUtils.cpp` and name the additional `kill(getppid(), SIGKILL)` behavior.
2. §3.3 / §11 — Ray-path soft claim is **"not characterized at WideEP scale,"** not "untested." Cite specific largest config (TP=4).
3. §1.1 user journey — anchor on `mpirun -np N trtllm-serve <model> --tp N --ep N` invocation.
4. §3.2 L1 gap — The April source showed MPI worker death flowing to full executor abort. The corrected design adds survivor paths, but that historical source observation is not proof those new paths work until 1d.4.
5. §3.3 — HostMoeTensorSharer's hard-baked MPI dependency (no `TLLM_DISABLE_MPI` guards) is a concrete cost item for any future Ray pivot.
6. §11 — Add new risk: "Disagg + Ray + NIXL unsupported" if §11 covers cross-track interactions.
7. **§5.4 / 1d.0a (Audit 1a Day 2):** merged 1d.0 addresses the signal path. The follow-up lifecycle item must validate launcher-specific recovery settings, prohibit poisoned-world management collectives, and skip `MPI_Finalize` when it cannot complete. `MPI_ERRORS_RETURN` on 1c.3 alone does not make ordinary executor collectives survivor-safe.
8. **§8.2 PR 2a.1 (Audit 1a Day 1):** NCCL teardown for the AllGatherReduceScatter fallback path cannot be a thin wrapper around `dist.shrink_group(SHRINK_ABORT)` in PT 2.11 — that API hangs after peer death in this build. Either drop below `torch.distributed` (call `ncclCommAbort` + `ncclCommInitRank` directly via cython / C) or wait for an upstream fix. Sizing should reflect the lower-level option.
9. **§8.2 PR 2a.2 (Audit 1a Day 3):** The tested POSIX-FD unmap/release calls consumed ~1–2 ms for one peer. Treat that as a lower-level input, not a bound on PR 2a.2; FABRIC/IMEX, fan-out, handle exchange, workspace, and communicator reconstruction remain unmeasured.

Ready to produce the per-section diff plan from these findings.
