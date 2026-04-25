# Pre-Drafting Research Pass — WideEP FT Design Rewrite

**Created:** 2026-04-23
**Purpose:** Verify factual claims that will anchor the rewritten design doc, before drafting begins. The v1 doc had reviewer-flagged inaccuracies that traced back to memory-based assertions; this pass front-loads source verification so the rewrite doesn't repeat that pattern.
**Time budget:** ~half a day. Output is a short report (under 500 words) feeding into the per-section diff plan.

---

## Items to verify

### 1. `MPIPoolExecutor` failure semantics

- **Question:** When one MPI worker process dies, what happens to the surviving workers in the pool? Is the entire pool unrecoverable, or just the dead worker's slot?
- **Why it matters:** The reviewer's claim that "the Python-level worker pool becomes permanently broken and cannot route new work to surviving workers" needs to be verified against actual mpi4py / TRT-LLM behavior. Determines whether routing-around-dead-worker is feasible at the Python-pool level, or whether pool-level FT requires executor replacement.
- **Where to look:** `tensorrt_llm/executor/proxy.py`, `tensorrt_llm/executor/`; mpi4py docs on `MPIPoolExecutor` failure semantics; any existing TRT-LLM tests that simulate worker death.
- **Used in:** §3.2 (L1 gap analysis), §5.4 (MPI-path FT-enabling work).

### 2. `HostMoeTensorSharer` Ray-path behavior

- **Question:** Does the EPLB host-side weight sharer have a non-MPI branch today? Specifically, the `Split_type(MPI.COMM_TYPE_SHARED)` call at `moe_load_balancer.py:894-902` — is this gated behind an MPI-availability check, or does it run unconditionally even on the Ray path?
- **Why it matters:** Critical for sizing any future Ray pivot. If MPI primitives are hard-baked, the pivot has hidden cost. If there's a node-local-discovery abstraction, the pivot is cheaper. Also affects §3.3 (the Ray-pivot discussion) — accuracy of the cost claim.
- **Where to look:** `tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py:127-340, 894-902`.
- **Used in:** §3.3 (why-not-Ray), §11 (Ray-path perf risk — adjacent).

### 3. `RayExecutor` vs `GenerationExecutorProxy` feature parity

- **Question:** Does `RayExecutor` cover all the WideEP-relevant operations that `GenerationExecutorProxy` does? Specifically: per-rank state queries, per-rank health checks, executor lifecycle, error propagation.
- **Why it matters:** Verifies "MPI for MVP" is a strategic choice, not a hidden necessity. Sizes the FT-related diffs we'd take on the Ray path if we ever migrate.
- **Where to look:** `tensorrt_llm/executor/ray_executor.py` (or similar); compare against `tensorrt_llm/executor/proxy.py`.
- **Used in:** §3.3 (why-not-Ray), §5.4 (MPI-path work).

### 4. Aggregated NVL72 launch path

- **Question:** Walk through exactly what happens when a user runs the canonical `trtllm-serve` deployment for DeepSeek-V3 on NVL72. Need: launch command (with real flags), env vars set, processes spawned, Python entry points, executor chain (e.g., PyExecutor → ModelEngine → forward → AlltoAll backend), MPI bootstrap order, shared-memory init order.
- **Why it matters:** §1 anchors the entire rewrite on this user journey. Real names of components and the real launch command keep §1 from drifting into vagueness.
- **Where to look:** `tensorrt_llm/commands/serve.py`, `tensorrt_llm/llmapi/llm.py`, `tensorrt_llm/_torch/pyexecutor/py_executor.py`, any reference / deployment guides under `docs/source/deployment-guide/`.
- **Used in:** §1.1 (user journey walkthrough), §1.2 (stack at each layer).

### 5. Ray-path WideEP CI coverage

- **Question:** Are there integration tests that exercise the Ray path at EP ≥ 32? Are there benchmarks comparing Ray-path vs MPI-path at WideEP scale?
- **Why it matters:** Empirical support for the "Ray code path is not perf-tested at WideEP scale" soft claim in §3.3. If no tests exist, the soft claim is well-supported. If tests exist, claim needs softening or removal.
- **Where to look:** `tests/integration/`, grep for `ray` + EP-related markers; `tests/integration/test_lists/test-db/` YAML files; perf bench scripts under `benchmarks/`.
- **Used in:** §3.3 (why-not-Ray), §11 (Ray perf-test risk).

### 6. Re-verify v1 source-anchored claims

The v1 doc made specific line-level source claims. Reviewer trust depends on these being correct against the current tree. Re-check each before reusing:

- `kMaxRanks = 64` in `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h:31`
- 300s in-kernel `check_timeout` → `asm volatile("trap;")` at `moeAlltoAllKernels.cu:156-161`
- Dispatch release+wait loop locations: `moeAlltoAllKernels.cu:537-584`; combine `:1190-1217`
- Combine accumulator's `dst_idx = -1` skip + `acc[k].fill(0.0f)` at `:725-729`, `:727`
- `MoeLoadBalanceMetaInfo` structure at `cpp/tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceCommon.h:40-52`
- `MoePlacementCpuInfo` at `cpp/tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h:56-70`
- Propagation CPU→GPU at `moeLoadBalancer.cpp:523-542` (in-place `cudaMemcpyAsync`, no double-buffer)
- DeepEP `Buffer.__del__` calling `intranode::barrier` (need to find current source location)
- Zero uses of `ncclCommAbort`, `NCCL_ASYNC_ERROR_HANDLING`, `ncclCommFinalize`, `ncclGetLastError` outside test files
- Zero MPI fault-tolerance infrastructure in `MPIDist` (no `MPI_ERRORS_RETURN`, `MPI_Comm_revoke`, ULFM wiring, FT subcomm)
- Signal handlers at `mpiUtils.cpp:199-210` (the reviewer's specific reference) calling `MPI_Abort(MPI_COMM_WORLD)`

### 7. NVSHMEM / MNNVL teardown semantics (preliminary)

- **Question:** Light review only — what does NVSHMEM (the version TRT-LLM ships against) document for teardown semantics under peer death? What does the CUDA driver API say about `cuMemUnmap` on a region whose owning process has died?
- **Why it matters:** Sizes §6.2 (PG reconstruction per backend) and supports the §9 named risk on the Phase 2 audit. Difference between "well-documented teardown path" and "behavior unspecified by upstream" is material to how aggressively §9 should defer Phase 2 work.
- **Where to look:** NVSHMEM release notes (external); CUDA driver API reference (external); any TRT-LLM internal notes.
- **Note:** This is preliminary literature review only, not the full Phase-2-prerequisite audit. Full audit is the named risk in §9.
- **Used in:** §6.2, §9 (audit risk).

### 8. Disaggregated serving — Ray-path support (deferred)

- **Question:** Does the Ray path currently support disaggregated serving, or only aggregated?
- **Why it matters:** Affects future Phase 1-DS scope on Ray vs MPI. Out of MVP scope but relevant for §9 future-migration risk.
- **Where to look:** `trtllm-serve` proxy code; disagg-related tests; `docs/source/features/disagg-serving.md`.
- **Defer to:** §9 drafting only; not required for §1–§8 drafting.

---

## Output format

When research is complete, produce a short report (target under 500 words) with one bullet per item:

- ✅ **Confirmed** — claim was correct; anchor still valid in current source
- 🔄 **Corrected** — claim was inaccurate or stale; corrected version stated
- ❓ **Couldn't verify** — note what was missing; flag for inline caveat in draft
- 🆕 **New gap surfaced** — something the research uncovered that should also be addressed in the rewrite

That report becomes the input to the per-section diff plan, which gets sign-off before drafting starts.

---

## Out of scope for this pass

- Running benchmarks (Ray vs MPI perf comparison) — that's the named §9 audit, not this pre-drafting pass
- Building the MNNVL/NVSHMEM teardown prototype — also the named §9 audit
- Reading module sources end-to-end — only the specific functions/lines that anchor a claim
- Validating PR #12718 commit details — already documented in v1 doc; trust unless flagged
