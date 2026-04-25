# Pre-Drafting Research Pass — Findings Report

**Created:** 2026-04-23 (research pass) | **Updated:** 2026-04-25 (Audit 1a Days 1–3 empirical follow-up to Item 7)
**Companion to:** `redesign-research-pass.md` (the items list)
**Time spent:** ~half a day pre-drafting + ~5 hr Audit 1a Days 1–3 prototyping
**Status:** Items 1–6 verified; **item 7 partially answered by Audit 1a Days 1–3** (sections below); item 8 light-pass complete.

---

## ✅ Confirmed

**Item 6 — v1 line anchors all hold against current source:**
- `kMaxRanks = 64` at `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h` ("Maximum supported EP size") ✓
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

2. **`mpiUtils.cpp` `forwardAbortToParent` variant additionally `kill(getppid(), SIGKILL)`** before `MPI_Abort`. The reviewer's bare-claim quote understates this — it's not just propagation through MPI, the launcher process is also signaled. Worth naming in §3.1 Mode A.

3. **Two-mode launch path** (`mpirun + MPICommExecutor` vs `MpiPoolSession + MPIPoolExecutor`) — both exist, with different failure modes. The §1.1 user journey should specify which case it walks through (the production `mpirun` case).

## 🧪 Empirical follow-up — Audit 1a partial (Item 7)

**Hardware:** 8× B300 SXM6 single node, NVLink full mesh. **Software:** PyTorch 2.11 (NCCL 2.29.2), OpenMPI 4.1.9a1 (no ULFM), CUDA 13.1, cuda-python 13.1.1. **Date:** 2026-04-25. **Tracks completed:** NCCL rebuild (Day 1, 6 runs / 6 modes), MPI signal handler (Day 2, 7 runs / 7 modes), driver-side `cuMemUnmap` posix-FD variant (Day 3). Deep prototypes + per-run JSONL logs live at `/home/chienchunh/audit-1-mnnvl-teardown/` on the test host; this section captures the headline findings.

### Headline findings

1. **PT 2.11's `torch.distributed` has no recoverable peer-death path** for any documented mode. `dist.shrink_group(ranks_to_exclude=…, shrink_flags=SHRINK_ABORT)` — the API the design hoped for — itself hangs > 60 s after a peer SIGKILL in this build (with or without `ASYNC_ERROR_HANDLING`). Default config (`ASYNC=1, BLOCKING=0`) SIGABRTs all survivors at the watchdog timeout.
2. **Detection in pure Python works.** `dist.all_reduce(…, async_op=True)` + main-thread `work.is_completed()` polling delivers a clean exception at exactly the configured deadline (5054 ms vs a 5000 ms budget; tunable freely). Validates §5.3 / 1c.4's design assumption that detection sits above NCCL.
3. **Mode A needs a TWO-part fix.** Replacing `MPI_Abort` with `_exit(N)` in `mpiUtils.cpp` (PR 1d.0) is necessary but **not sufficient**: under default `mpirun`, a single rank exit (any code, including `_exit(0)` and `SIGKILL`) takes down all survivors in 18–60 s. Adding the launch flag `--mca orte_enable_recovery 1` stops the propagation; survivors then need `MPI_ERRORS_RETURN` (already PR 1c.3) plus bounded-wait collectives (1c.4-style) to escape the broken collective the dead rank was in.
4. **Driver-side teardown of dead-peer regions is essentially free.** Posix-FD cross-process share + SIGKILL of owner: peer's `cuMemUnmap` returns `CUDA_SUCCESS` in **0.25 ms**, `cuMemRelease` in **1.27 ms**, `cuMemAddressFree` in **0.008 ms** — total **~1.5 ms**. The mapping survives the kill: 2 s after SIGKILL the survivor still reads the owner's pre-kill pattern. **Driver-side cleanup is not a Phase 2 bottleneck.**

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

- `cuMemUnmap` fabric-handle equivalence: `CU_MEM_HANDLE_TYPE_FABRIC` allocation requires the `nvidia-imex` daemon, which is not active on this single-node B300 (kernel-side `nv-caps-imex channel0 created` is present but no IMEX domain configured). `cuMemCreate(... FABRIC)` returns `CUDA_ERROR_NOT_PERMITTED`. The driver mechanism is proven (posix-FD); fabric-handle equivalence is a 5-minute re-run on an IMEX-configured node.
- Intra-node MNNVL `MnnvlMemory` rebuild prototype (Days 4–5 in §9.1 plan): same IMEX gate.
- DeepEP destructor + explicit `destroy()` ordering (Day 3 secondary item): blocked here on a `tensorrt_llm` package import error (pre-existing `fp4_quantize_with_residual` mirror issue in this dev tree). Needs a container with matching C++ binary.
- NVSHMEM `nvshmem_finalize` behavior (Day 5): `nvshmem` Python module not installed in this env.

### Caveats

- Single-node intra-node only. Cross-node propagation (NCCL FT, MPI mpirun behavior, fabric memory teardown) may differ.
- 4-rank scale on the rebuild and signal-handler tests; 1 owner / 1 peer on the cuMemUnmap test. Fan-out cost at 71 surviving peers per dead rank is unknown.
- Did not run the full `--mca orte_enable_recovery 1` + `MPI_ERRORS_RETURN` + bounded-wait collective combination end-to-end. Each piece tested in isolation; integrated path is the next step (1–2 hr).

## ❓ Deferred (intentionally)

- **Item 7 — NVSHMEM/MNNVL teardown.** No longer fully deferred: Audit 1a Days 1–3 done (above). Days 4–5 (intra-node MNNVL prototype, NVSHMEM teardown) and the rack-fabric portion (Audit 1b) remain.
- **Item 8 — Disagg Ray (light pass complete)** — see new gap #1 above.

---

## Implications for the rewrite

1. §3.1 Mode A — quote `mpiUtils.cpp` and name the additional `kill(getppid(), SIGKILL)` behavior.
2. §3.3 / §11 — Ray-path soft claim is **"not characterized at WideEP scale,"** not "untested." Cite specific largest config (TP=4).
3. §1.1 user journey — anchor on `mpirun -np N trtllm-serve <model> --tp N --ep N` invocation.
4. §3.2 L1 gap — Item 1 finding lets us state confidently that today MPI worker death = full executor abort, no salvage path; cite `proxy.py:229–234` and `mpi_session.py:167–168`.
5. §3.3 — HostMoeTensorSharer's hard-baked MPI dependency (no `TLLM_DISABLE_MPI` guards) is a concrete cost item for any future Ray pivot.
6. §11 — Add new risk: "Disagg + Ray + NIXL unsupported" if §11 covers cross-track interactions.
7. **§5.4 / PR 1d.0 (Audit 1a Day 2):** PR 1d.0 scope must include the launch-flag piece (`--mca orte_enable_recovery 1`), not only the `_exit(N)` source change. Without the flag, default `mpirun` propagates termination to survivors regardless of the in-process signal handler; with both, survivors stay alive long enough for `MPI_ERRORS_RETURN` (PR 1c.3) + bounded-wait collectives (1c.4) to drive recovery.
8. **§8.2 PR 2a.1 (Audit 1a Day 1):** NCCL teardown for the AllGatherReduceScatter fallback path cannot be a thin wrapper around `dist.shrink_group(SHRINK_ABORT)` in PT 2.11 — that API hangs after peer death in this build. Either drop below `torch.distributed` (call `ncclCommAbort` + `ncclCommInitRank` directly via cython / C) or wait for an upstream fix. Sizing should reflect the lower-level option.
9. **§8.2 PR 2a.2 (Audit 1a Day 3):** Driver-side cleanup of dead-peer fabric memory budgets at ~1–2 ms (posix-FD; fabric-handle equivalence pending Audit 1b validation). PR 2a.2 sizing previously had to assume "novel work, hard to bound" — now bounded.

Ready to produce the per-section diff plan from these findings.
