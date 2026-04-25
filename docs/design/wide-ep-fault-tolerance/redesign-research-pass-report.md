# Pre-Drafting Research Pass — Findings Report

**Created:** 2026-04-23
**Companion to:** `redesign-research-pass.md` (the items list)
**Time spent:** ~half a day
**Status:** Items 1–6 verified; item 7 deferred to the named §9 audit; item 8 light-pass complete.

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

## ❓ Deferred (intentionally)

- **Item 7 — NVSHMEM/MNNVL teardown literature.** Skipped pre-drafting; this is the substance of the named §9 audit risk and shouldn't be pre-decided. Rewrite preserves the audit-as-risk framing.
- **Item 8 — Disagg Ray (light pass complete)** — see new gap #1 above.

---

## Implications for the rewrite

1. §3.1 Mode A — quote `mpiUtils.cpp` and name the additional `kill(getppid(), SIGKILL)` behavior.
2. §3.3 / §11 — Ray-path soft claim is **"not characterized at WideEP scale,"** not "untested." Cite specific largest config (TP=4).
3. §1.1 user journey — anchor on `mpirun -np N trtllm-serve <model> --tp N --ep N` invocation.
4. §3.2 L1 gap — Item 1 finding lets us state confidently that today MPI worker death = full executor abort, no salvage path; cite `proxy.py:229–234` and `mpi_session.py:167–168`.
5. §3.3 — HostMoeTensorSharer's hard-baked MPI dependency (no `TLLM_DISABLE_MPI` guards) is a concrete cost item for any future Ray pivot.
6. §11 — Add new risk: "Disagg + Ray + NIXL unsupported" if §11 covers cross-track interactions.

Ready to produce the per-section diff plan from these findings.
