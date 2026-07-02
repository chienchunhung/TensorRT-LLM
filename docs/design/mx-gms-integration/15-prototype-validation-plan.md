# 15. Prototype Validation Plan

[< Back to README](README.md)

> **Archived plan:** PR #13045 closed without merge. Use
> [§18 GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-concrete-pr-plan.md) for current implementation
> and validation gates. This file is retained as historical prototype methodology.

**Status:** Archived; no further execution planned  •  **Last Updated:** 2026-06-30
**Scope:** Validation strategy for the [PR #13045 prototype](https://github.com/NVIDIA/TensorRT-LLM/pull/13045) (MX + GMS integration) using the §10/§11 benchmark infrastructure.

> This file was the working validation plan for the unmerged prototype. Its results were not used to qualify the
> current native GMS path.

> **Skip to current state:** [Execution Status](#execution-status) (what's done, what's blocked, what's next).

---

## Goals

Quantitatively verify that PR #13045 delivers the projected wins from [§11 Impact Projection](11-results-analysis.md#mxgms-impact-projection):

| Scenario | Baseline (measured) | Target (projected) | Verification test |
|----------|--------------------:|-------------------:|-------------------|
| Cold start, MX 1st on new node | 306s (Qwen 72B S2) | ~75–80s | **B4** |
| Failover activation (GMS shadow) | ~75s (cold restart) | <5s | **B6** |
| Memory overhead of shadow worker | N/A (no shadow today) | ~0 GB additional weights | **B2** |
| Throughput regression | N/A | <2% | **B5** |

And establish go/no-go gates:

- **Bit-exactness** (B1): MX-loaded weights must produce identical outputs to HF-loaded weights — pure correctness gate before any perf measurement.
- **Profile diagnostic** (Phase C): verify projected MX/worker-init overlap actually holds, or quantify the shortfall.

---

## Branch Strategy

PR #13045 lives on `chienchunhung:dynamo-integration-prototype`. The benchmark + profiler infrastructure lives on `dynamo/startup-profiling` (see [§10 Methodology](10-methodology.md)). They must be combined.

**Approach:** rebase the prototype onto current `upstream/main` first (so the integration branch starts from the same base as the §11 v3 baseline), then create a fresh integration branch on top of the rebased prototype and cherry-pick the bench commits. This keeps the two original branches (`dynamo-integration-prototype` and `dynamo/startup-profiling`) untouched as canonical references.

### Branches in fork (`github.com/chienchunhung/TensorRT-LLM`)

| Branch | Purpose | Base | Tip SHA |
|--------|---------|------|---------|
| `dynamo-integration-prototype` | **Original prototype** (PR #13045 source) before the API alignment work. Will be force-pushed with the rebased + API-aligned commits once that work is reviewed. | upstream/main as of 2026-04-14 | `84dfb2aa7` (original) → `62ac40f6b` (planned, see `dynamo-integration-prototype-rebased`) |
| `dynamo/startup-profiling` | **Bench + profiler infrastructure** (§10/§11 source). Untouched. | upstream/main as of 2026-04-17 | `f9771e571` |
| `docs-and-plans` | This design doc. | — | (current) |
| `dynamo/proto-rebased` | **Pre-alignment snapshot** — prototype's 2 original commits replayed on `upstream/main @ 4a848ccce` with zero rebase conflicts. Kept as a historical reference for the conflict-free rebase. | upstream/main `4a848ccce` | `7bb11db6a` |
| `dynamo/proto-bench-integration-v2` | Working integration branch (pre-alignment) — `dynamo/proto-rebased` + the 7 bench commits cherry-picked on top. Used for the original Phase A smoke verification. | `dynamo/proto-rebased` | `5e9ee91c8` |
| `dynamo-integration-prototype-rebased` | **NEW.** Live working branch for PR #13045 — prototype rebased onto current `upstream/main @ 7b8413697` (95 commits ahead of original base) plus a third commit (`62ac40f6b`) that aligns the GMS and MX adapters with what was actually merged upstream. This is what the PR will be force-pushed to. | upstream/main `7b8413697` | `62ac40f6b` |

```text
upstream/main (7b8413697)                          ← current upstream HEAD
└── dynamo-integration-prototype-rebased (62ac40f6b)   ← LIVE: API-aligned PR #13045
    │   • [feat] Add MX and GMS integration prototype          (b8f2f923d)
    │   • [feat] Align GMS backend with merged GMS API #7575   (d5683cde5; original — kept for history)
    │   • [feat] Update MX and GMS adapters to match merged    (62ac40f6b; this is the actual alignment)
    │     upstream APIs                                          ← [NEW commit, see "API Alignment" below]

# Pre-alignment historical snapshots (kept for traceability):
upstream/main (4a848ccce)                          ← upstream HEAD as of 2026-04-17
└── dynamo/proto-rebased (7bb11db6a)               ← prototype's 2 original commits replayed
    └── dynamo/proto-bench-integration-v2 (5e9ee91c8)  ← + 7 bench cherry-picks
        ├── [feat] Add hierarchical startup profiling and benchmark instrumentation
        ├── [feat] Split HF cache probe and remote download timers
        ├── [feat] Add startup benchmark automation scripts
        ├── [feat] Fix S2 NFS cold benchmark with per-run fresh copy
        ├── [fix]  Use offline mode and local tokenizer for S2/S3 benchmark tiers
        ├── [fix]  Use representative-run approach for benchmark aggregation
        └── [feat] Add failover floor benchmark script (Test 4a)
```

The `dynamo-integration-prototype-rebased` branch is the live working branch for PR #13045. Once review-ready, it will be force-pushed to `dynamo-integration-prototype` (the actual PR source). The `dynamo/proto-rebased` and `dynamo/proto-bench-integration-v2` branches are pre-alignment snapshots kept for traceability and can be discarded after validation completes.

### Setup commands (reproducible)

```bash
# 1. Rebase the prototype onto current upstream/main (clean — 0 conflicts)
git fetch upstream main
git fetch fork dynamo-integration-prototype dynamo/startup-profiling
git checkout -b dynamo/proto-rebased fork/dynamo-integration-prototype
git branch --unset-upstream            # Safety: avoid pushing to original prototype branch
git rebase upstream/main

# 2. Build the integration branch by cherry-picking the bench commits
git checkout -b dynamo/proto-bench-integration-v2
git cherry-pick 3c9aebfdd 800ba9751 642dcd05a 667a89be0 4bd024b5e bba6bb505 f9771e571
# (5 conflicts on the foundational bench commit, all in
#  py_executor_creator.py + model_loader.py — see "Conflict Resolutions" below)

# 3. Push to fork for safekeeping (no force, new refs only)
git push --no-verify fork dynamo/proto-rebased dynamo/proto-bench-integration-v2

# 4. No C++ rebuild needed: both new branches share the same upstream/main HEAD
#    as the existing in-container build artifacts (option 1 from the validation plan).
```

### Conflict Resolutions (one-time, captured for the record)

All 5 conflicts landed on the foundational bench commit `3c9aebfdd` (hierarchical startup profiling). Resolution strategy: **keep prototype semantics, add bench timers without changing control flow.**

| File | Region | Resolution |
|------|--------|------------|
| `py_executor_creator.py` | `_construct_checkpoint_loader` call | Kept prototype's new `mx_server_url=llm_args.mx_server_url` arg; wrapped `load_config_and_apply_defaults` in `executor.load_config_and_apply_defaults` timer. |
| `model_loader.py` | Materialize tensors path | `executor.materialize_model_tensors` timer now wraps prototype's `virtual_memory_scope` block; `elif is_meta_init:` retains the `and load_format != LoadFormat.GMS` guard so GMS skips meta→CUDA init. |
| `model_loader.py` | `model.to("cuda")` | Combined: `if load_format != LoadFormat.GMS: with startup_timer("executor.move_model_to_cuda"): model.to("cuda")`. GMS RO path stays a no-op. |
| `model_loader.py` | `LoadFormat.AUTO` weight load | Kept MX-aware `load_weights_kwargs` with `model=` injection and the `mx_p2p_succeeded` short-circuit; weight-mapper init and `_call_load_weights` are now wrapped in `executor.weight_mapper_init.main_weights` and `executor.apply_model_weights.main_weights`, but only inside `if not mx_p2p_succeeded:` (so MX P2P still skips the mapping pipeline). |
| `model_loader.py` | `post_load_weights` block | Added `executor.mx_publish_as_source` timer around prototype's new MX publish call; gated `executor.post_load_weights` on `not gms_ro_done` to preserve prototype's "skip post_load_weights for GMS RO" behavior. |

Net effect: the integration branch produces an exact superset of the prototype's behavior (no semantic changes), with full §10 timer coverage on every code path.

---

## Execution Status

Snapshot of progress as of **2026-04-18**. This section is updated as we work through the plan.

### ✅ Phase A — Branch Integration (DONE)

| Step | Outcome |
|------|---------|
| Rebase `dynamo-integration-prototype` onto current `upstream/main` | **Clean** — 0 conflicts |
| Cherry-pick the 7 bench commits onto rebased prototype | 5 conflicts on foundational profiler commit, all resolved (see [Conflict Resolutions](#conflict-resolutions-one-time-captured-for-the-record)) |
| Push both new branches to fork | Done — `dynamo/proto-rebased`, `dynamo/proto-bench-integration-v2` |
| C++ rebuild | **Not needed** — both branches sit on the same `upstream/main @ 4a848ccce` HEAD as the existing in-container build (option 1 path; see "Smoke verification" below) |

### ✅ Smoke Verification on Integrated Branch (DONE)

Confirmed the integration is healthy and the M1 (baseline AUTO/HF) path is unaffected:

| Check | Result |
|-------|--------|
| `import tensorrt_llm` | ✅ |
| Prototype symbols (`tensorrt_llm._torch.memory.GMSBackend`) | ✅ importable |
| Prototype Pydantic fields (`mx_server_url`, `gms_socket_path`, `gms_mode`, `gms_tag`) | ✅ present, render correct defaults |
| Bench symbols (`tensorrt_llm.llmapi.startup_profiler.startup_timer`, `get_startup_profiler`) | ✅ |
| `trtllm-serve --help` | ✅ |
| `trtllm-serve` boots Qwen2.5-7B-Instruct TP=1, `LoadFormat.AUTO` (no MX, no GMS) | ✅ ready in ~45s on warm NFS |
| Inference correctness (`The capital of France is` → ` Paris…`) | ✅ |
| Bench profiler captures full hierarchy on integrated branch | ✅ 67.6s server / 36.3s worker, all expected timers populated (`executor.load_model_weights`, `executor.warmup.*`, `executor.recreate_py_executor_instance`, etc.) |

The MX-aware `if not mx_p2p_succeeded:` guard in `model_loader.py` correctly takes the standard HF path when no MX server is configured, and `executor.apply_model_weights.main_weights: 2.288s` confirms `_call_load_weights` runs as expected.

### ✅ API Alignment — Prototype ↔ Current GMS / MX (DONE)

Phase A's smoke verification confirmed that the prototype's GMS adapter (`tensorrt_llm/_torch/memory/gpu_memory_backend.py`) was written against an **unmerged** iteration of the GMS Python API that did not survive PR #7575. A parallel mismatch existed for the MX adapter (`tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py`) against current `modelexpress` v0.3.0. Both adapters have now been refactored against the actually-merged upstream APIs, on the new `dynamo-integration-prototype-rebased` branch (commit `62ac40f6b`).

#### What was wrong (briefly)

GMS — prototype called convenience functions that don't exist in merged GMS:

```python
# Prototype expected:
from gpu_memory_service import client as gms_client
self._client = gms_client.connect(self._socket_path, mode="rw")
gms_client.get_mem_pool(self._client)
gms_client.materialize_module_from_gms(self._client, model)
gms_client.register_module_tensors(self._client, model)
gms_client.commit(self._client, tag)
gms_client.disconnect(self._client)

# Actually merged in ai-dynamo/dynamo (verified 2026-04-20):
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager, gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.integrations.common.utils import finalize_gms_write
mgr = get_or_create_gms_client_memory_manager(socket, device, mode=lock_mode, tag="weights")
with gms_use_mem_pool("weights", torch.device("cuda", device)):
    ...
materialize_module_from_gms(mgr, model, device_index=device)   # keyword required
finalize_gms_write(mgr, model)   # register + sync + commit + RO + remap
```

MX — prototype called methods that no longer exist in `modelexpress` v0.3.0:

```python
# Prototype expected:
from modelexpress import client as mx_client, proto as mx_proto
connection = mx_client.connect(self._mx_server_url)
sources = connection.list_sources(identity)        # returned a list directly
connection.receive(source)                         # one-call transfer
connection.register_source(model, identity)        # one-call publish
mx_proto.SourceIdentity(model_name=..., extra_params={...})  # dict-of-strings schema

# Actually merged in ai-dynamo/modelexpress v0.3.0:
from modelexpress.trtllm_live_transfer import MxLiveWeightLoader, publish_model_params
loader = MxLiveWeightLoader(mx_server=url)
fallback = loader.load_weights(checkpoint_dir, mapping=mapping, model=model)  # returns size-mismatch dict
publish_model_params(model)   # sets up NIXL + publishes to gRPC
# SourceIdentity now uses structured fields:
#   mx_version, mx_source_type, model_name, backend_framework,
#   tensor_parallel_size, pipeline_parallel_size, expert_parallel_size, dtype
# (no extra_params dict)
```

#### What the alignment commit does

Single commit on the prototype branch — `62ac40f6b [None][feat] Update MX and GMS adapters to match merged upstream APIs`:

1. **`tensorrt_llm/_torch/memory/gpu_memory_backend.py`** — full rewrite of `GMSBackend` against the merged class-based API. Notable surface-level change: `get_mem_pool() -> torch.cuda.MemPool` is replaced with `mem_pool_scope(device) -> ContextManager` so the call site uses the upstream pattern (`with gms_use_mem_pool(tag, device): ...`) directly. `_move_untracked_params()` mirrors the upstream `gpu_memory_service.integrations.trtllm.model_loader._move_untracked_params` (iterate via `_iter_module_tensors`, dedup by storage pointer, allocate via `create_mapping()`, rebind via `_tensor_from_pointer`).
2. **`tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py`** — rewritten to delegate the actual NIXL transfer to upstream `MxLiveWeightLoader.load_weights(model=...)` (which handles agent setup, source matching, dtype-cast handling, PVC fallback). `publish_as_source()` delegates to upstream `publish_model_params()`. We keep the `HfCheckpointLoader` subclass shell so disk fallback is inherited.
3. **`tensorrt_llm/_torch/pyexecutor/model_loader.py`** — switches the GMS-RW branch to the new `mem_pool_scope` context manager API and adds an `empty_cache()` drain inside the scope (mirrors the upstream RW reference).
4. **`tensorrt_llm/llmapi/llm_args.py`** — adds `mx_preshard_strategy: per_module | global` (default `per_module`). `gms_tag` default changes from `"model_weights"` to `"weights"` to match the GMS library convention. New validators reject bad values for both.
5. **`setup.py`** — adds optional extras: `pip install tensorrt_llm[mx]`, `[gms]`, or `[dynamo]` (both). Pinned upper bounds (`modelexpress>=0.3.0,<0.4.0`, `gpu-memory-service>=0.9.0,<0.10.0`) protect against future drift across the dependency boundary.
6. **`tests/unittest/api_stability/references/llm.yaml`** — reflects the new field and the changed `gms_tag` default.

#### Design decisions made during alignment

- **No monkey-patching.** We deliberately do **not** use the upstream `setup_gms()` entry point (which patches `ModelLoader.load` from outside). TRT-LLM owns the integration policy; this adapter just calls the GMS library's stable per-call primitives. This preserves the prototype's two-axis composition design.
- **Three-layer separation.** TRT-LLM owns Layer 3 (integration policy: where in the loading pipeline we call GMS/MX, and how it composes with TRT-LLM's MoE loader, weight mapper, post-load hooks). We call upstream Layer 2 (stable per-call primitives: `GMSClientMemoryManager`, `gms_use_mem_pool`, `MxLiveWeightLoader`, `publish_model_params`). We never duplicate Layer 1 (wire protocol, NIXL RDMA, CUDA VMM mechanics).
- **`mx_preshard_strategy='global'` parked for now.** It would map onto `LoadFormat.PRESHARDED`, which the MX team's PR #12898 proposed but was closed in favor of the per-module flag. Selecting `global` raises a friendly `NotImplementedError`. When `LoadFormat.PRESHARDED` lands upstream, the wiring is one branch in `model_loader.py`.
- **Mixed-success MX case is conservative.** When `MxLiveWeightLoader` returns size-mismatched fallback weights, we currently fall through to a full disk load to avoid mixing presharded and non-presharded weights in the same model. Per-tensor presharded marking will need `LoadFormat.PRESHARDED` to be plumbed through.
- **MX team alignment.** The `LoadFormat.PRESHARDED` vs per-module flag question is a real divergence point with the MX team's design (their `MxLiveCheckpointLoader` docstring explicitly says `LoadFormat.PRESHARDED`). The `mx_preshard_strategy` knob preserves both options as future paths so we can converge once the upstream design conversation lands.

#### Smoke verification on the API-aligned branch

Verified import-level correctness on `dynamo-integration-prototype-rebased`:

| Check | Result |
|-------|--------|
| All TRT-LLM-side adapter symbols import | ✅ |
| All Pydantic config fields present (`mx_server_url`, `mx_preshard_strategy`, `gms_socket_path`, `gms_mode`, `gms_tag`) | ✅ |
| Defaults align with current GMS conventions (`gms_tag='weights'`, `mx_preshard_strategy='per_module'`) | ✅ |
| `GMSBackend` exposes `connect`, `is_rw`, `has_committed_weights`, `mem_pool_scope`, `materialize_module`, `finalize_write`, `move_untracked_params`, `cleanup`, `DEFAULT_TAG` | ✅ |
| `MXCheckpointLoader` exposes `load_weights`, `publish_as_source`, `p2p_succeeded`, `mx_server_url`, `checkpoint_format == "MX"` | ✅ |
| All upstream GMS Layer 2 symbols resolve in installed `gpu-memory-service==0.9.0` | ✅ |
| `mx_preshard_strategy='bad_value'` is rejected with friendly Pydantic error | ✅ |

End-to-end runtime verification (boot `trtllm-serve` with `--load-format gms` and a real GMS daemon, plus `--checkpoint-format mx` with a real MX server) is the next step and is gated only on environment availability — see "MX install" and "GMS install" notes below.

#### MX install: still needs a node with Rust + Docker

| Requirement | Status on current node |
|-------------|------------------------|
| Rust 1.90+ (cargo) | ❌ not installed |
| `protoc` | ✅ 3.21.12 |
| Docker (for Redis metadata backend) | ❌ not installed |
| `redis-server` | ❌ not installed |

The MX server (Rust binary at `ai-dynamo/modelexpress`) is built via `cargo build` and runs against a Redis (or Kubernetes) metadata backend. The Python client (`pip install modelexpress`) is independent and works without the server, but `MxLiveWeightLoader.load_weights()` will hang in `_query_source` until a server is reachable. Setting up a server requires provisioning Rust + Docker on a test node or migrating to a node that already has them.

Also relevant: per [modelexpress's known issues](https://github.com/ai-dynamo/modelexpress#known-issues), MLA-architecture models (DeepSeek-V2/V3, Kimi K2) are blocked from MX P2P transfer and silently fall back to disk. Our Qwen 72B (no MLA) and DeepSeek-R1-Distill-Llama-70B (Llama architecture, not MLA) are both safe.

## Upstream Alignment Requests

Concrete asks to the MX (`ai-dynamo/modelexpress`) and GMS (`ai-dynamo/dynamo`) teams. These are the workarounds the prototype currently carries; each one would shrink or disappear if the upstream change lands. Cross-references in [§3](03-architecture.md), [§4](04-implementation-plan.md), [§5](05-challenges.md), and [§6](06-executor-failover.md) point here.

### To: MX team (`ai-dynamo/modelexpress`)

#### MX-1. Decide: `LoadFormat.PRESHARDED` vs per-module `_weights_presharded` flag

**Priority:** High — design conversation that blocks several downstream things.
**Ask:** Take an explicit position. Two viable options:

- **(a) Per-module flag** (what PR #13045 currently uses): MX P2P succeeds → mark each `Linear` module's `_weights_presharded = True`.
- **(b) `LoadFormat.PRESHARDED`** (what `MxLiveCheckpointLoader`'s docstring assumes): MX P2P succeeds → use a `LoadFormat.PRESHARDED` enum value that short-circuits the entire weight pipeline.

**Why now:** `MxLiveCheckpointLoader` ([trtllm_live_transfer.py:11–15](https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/trtllm_live_transfer.py#L11-L15)) explicitly uses `LoadFormat.PRESHARDED` but that enum doesn't exist in TRT-LLM `main`. The MX team's TRT-LLM PR [#12898](https://github.com/NVIDIA/TensorRT-LLM/pull/12898) proposed adding it but was closed. The two approaches differ at **mixed-success** time (some weights via P2P, rest via PVC fallback) — per-module can mark exactly the MX-delivered modules, while `LoadFormat.PRESHARDED` is global.

**Our workaround:** `mx_preshard_strategy: per_module | global` config knob (default `per_module`); `global` raises `NotImplementedError` until (b) is taken upstream. See `tensorrt_llm/llmapi/llm_args.py` and `tensorrt_llm/_torch/pyexecutor/model_loader.py`.

#### MX-2. Promote `_build_trtllm_identity` to a public API

**Priority:** Medium — small ergonomic fix that removes a private-symbol dependency AND lets us drop two env-var dances.
**Ask:** Promote [`_build_trtllm_identity`](https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/trtllm_live_transfer.py#L34) to a public `modelexpress.trtllm.build_identity(model_name, *, tp_size, pp_size=1, ep_size=1, dtype="bfloat16") -> p2p_pb2.SourceIdentity`.

**Why:** Callers that need a `SourceIdentity` outside `MxLiveWeightLoader.load_weights()` currently can either re-implement `_build_trtllm_identity` (drift risk) or import a private symbol (stability risk).

**Our workaround:** in `publish_as_source()` (`tensorrt_llm/_torch/models/checkpoints/mx/checkpoint_loader.py`) we set BOTH `MODEL_EXPRESS_URL` and `MODEL_NAME` env vars temporarily and call `publish_model_params(model)`, which then internally calls `_build_trtllm_identity` from those env vars. Two separate env-var dances would collapse into one direct function call if MX-2 lands. The `MODEL_NAME` resolution itself is plumbed cleanly: `llm_args.model → MXCheckpointLoader(model_name=...) → publish-time resolver` (with HF-snapshot path unmangling).

#### ~~MX-3. Per-rank addressing in identity / metadata~~ — **Resolved (non-issue)**

**Status:** Retracted. Upstream `MxLiveWeightLoader.load_weights` and `publish_model_params` both derive `worker_rank` from `MPI.COMM_WORLD.Get_rank()` internally. TRT-LLM workers are MPI processes, so the global MPI rank is already available on both publish and receive sides and matching works correctly (`worker_rank == mpi_rank`). Our `MXCheckpointLoader` delegates to these upstream functions without overriding rank derivation.

The original concern about `MPI rank != TP rank` in multi-node PP configs is moot: upstream matches on global MPI rank (not TP rank), which is correct regardless of the parallelism topology. MX engineer confirmed the global-rank design is intentional for simplicity.

#### MX-4. Adopt `RdmaStrategy`'s immediate-fallback source discovery in `MxLiveWeightLoader`

**Priority:** Medium — needed for fast disk-fallback on cold clusters.

**Ask:** Port the immediate-fallback source-discovery pattern from the vLLM-facing [`RdmaStrategy._find_source_instances`](https://github.com/ai-dynamo/modelexpress/blob/a8a2fd494f861ee654c4a29716b9c4a2989e9060/modelexpress_client/python/modelexpress/load_strategy/rdma_strategy.py#L70-L118) into the TRT-LLM-facing [`MxLiveWeightLoader._query_source`](https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/trtllm_live_transfer.py#L496), or deprecate `MxLiveWeightLoader` in favor of the strategy-chain API for all engines.

**Why:** The two upstream code paths have diverged:

- **`RdmaStrategy._find_source_instances`** (vLLM path): calls `mx_client.list_sources(identity=..., status_filter=SOURCE_STATUS_READY)` **once**, returns immediately if empty. Caller picks the next strategy (disk, GDS, etc.) with zero polling delay.
- **`MxLiveWeightLoader._query_source`** (TRT-LLM path): polls every 5 s for up to `MX_SOURCE_QUERY_TIMEOUT` (default `3600` = **1 hour**) before raising `TimeoutError`. No `status_filter`, no multi-candidate retry, no shuffle.

Both paths use the same underlying `MxClient` SDK and serve the same purpose (find-or-fallback). The TRT-LLM path should adopt the vLLM path's try-once-and-return semantics so TRT-LLM integrators don't need env-var workarounds to avoid a 1-hour hang.

**Our workaround:** `MXCheckpointLoader.__init__` calls `os.environ.setdefault("MX_SOURCE_QUERY_TIMEOUT", "30")` whenever an MX server URL is configured. This caps the polling at 30 s instead of 1 hour — far better than the default, but still 30 s of unnecessary polling vs the correct "try once, fall back immediately" pattern that `RdmaStrategy` already implements. The workaround is removed once upstream updates `_query_source`.

#### MX-5. Clarify NIXL agent ownership across MX + GMS composition

**Priority:** Medium — design clarification that affects the MX+GMS combined story.
**Ask:** Confirm (or add support for) MX-RDMA-into-GMS-pool memory.

**Why:** The prototype's PR description says *"MX P2P is NOT used in GMS RW mode. Model params are meta tensors at that point — no CUDA buffers for P2P to write into."* If MX-NIXL writes to **already-allocated** GPU buffers (the typical RDMA pattern), then in GMS RW mode we *could* allocate buffers under `gms_use_mem_pool("weights", device)` first and then have MX RDMA into those buffers, giving us true MX+GMS composition.

**Our workaround:** PR #13045 supports MX-only, GMS-only, and explicitly does NOT use MX inside GMS-RW. This is a real cold-path optimization left on the table until the MX team confirms the contract.

#### MX-6. MLA-architecture model support (informational)

**Priority:** Low — already on the MX known-issues list.
**Status:** MLA-arch models (DeepSeek-V2/V3, Kimi K2/K2.5) are blocked from MX P2P transfer per upstream; they silently fall back to disk. Tracked here only so our Section 11 impact projection can call this out.

#### MX-7. Onboard `modelexpress` into NVIDIA's OSS package allowlist

**Priority:** Medium — blocks CI and one-line install ergonomics.
**Ask:** File the NVIDIA-internal OSS-allowlist onboarding request for `modelexpress` (Apache-2.0, PyPI, single Beta release `0.3.0`). This is a TRT-LLM-side administrative action (with MX team coordination for provenance docs).

**Why:** NVIDIA's Blossom-CI `Vulnerability scan` job scans all dependencies declared in `setup.py`. `modelexpress` is a brand-new PyPI package (single release, Beta status) and is not yet in the internal OSS allowlist. Until onboarded, declaring `"modelexpress>=0.3.0,<0.4.0"` as an `extras_require` entry causes the scan to fail, blocking the entire L0 pipeline. PR #13045 removed the extras from `setup.py` as a workaround (users install manually); restoring the one-line `pip install tensorrt_llm[mx]` / `tensorrt_llm[dynamo]` ergonomics requires this onboarding.

**Our workaround:** `[mx]` / `[gms]` / `[dynamo]` extras removed from `setup.py` with inline comment documenting the rationale and manual install instructions.

---

### To: GMS team (`ai-dynamo/dynamo` `lib/gpu_memory_service`)

#### GMS-1. Document and support a non-monkey-patch integration path

**Priority:** High — direct feedback from a downstream integrator.
**Ask:** Treat the class-based primitives (`GMSClientMemoryManager` + `gms_use_mem_pool` + `materialize_module_from_gms` + `finalize_gms_write`) as a first-class integration path, equally documented and supported alongside `setup_gms()`.

**Why:** `gpu_memory_service.integrations.trtllm.setup_gms()` works by `_trt_loader.ModelLoader.load = patched_load` — runtime monkey-patching of TRT-LLM internals from outside. PR #13045 deliberately doesn't use it because: (1) opaque at code-review time, (2) future TRT-LLM `ModelLoader.load` refactors silently break it, (3) it conflicts with TRT-LLM's two-axis design where `--checkpoint-format mx` and `--load-format gms` should compose orthogonally.

**Suggested resolution:** Add a "Non-monkey-patch integration path" section to the GMS README that walks through the class-based API with our `tensorrt_llm/_torch/memory/gpu_memory_backend.py` as a worked example. Commit to keeping the class-based API stable across 0.x.

#### GMS-2. Promote `_move_untracked_params` to public API

**Priority:** High — small change that removes a private-symbol copy-paste.
**Ask:** Promote [`_move_untracked_params`](https://github.com/ai-dynamo/dynamo/blob/main/lib/gpu_memory_service/integrations/trtllm/model_loader.py#L237) to a public `gpu_memory_service.client.torch.module.move_untracked_params(model, gms_client, target_device, *, tag="weights")`.

**Why:** Adapters that don't use `setup_gms()` (us, plus future vLLM/SGLang integrations that want to control the loading pipeline) need this exact functionality. Today it's private (leading underscore) and lives in the `trtllm`-specific submodule despite being model-engine-agnostic.

**Our workaround:** `GMSBackend.move_untracked_params()` is a near-byte-for-byte port of the upstream private function. High maintenance burden — when upstream's logic changes (e.g., handling a new tensor type), we have to chase it.

#### GMS-3. Lightweight "is anything committed?" peek RPC

**Priority:** Medium — health checks, readiness gates, dashboards.
**Ask:** Add a lightweight check that does NOT acquire a session lock, e.g. `gpu_memory_service.client.is_committed(socket_path, tag="weights", timeout_ms=100) -> bool`.

**Why:** Today, to determine whether RO weights are ready, we have to call `connect()` and inspect `granted_lock_type`. This (a) acquires a session, which is heavyweight, (b) blocks if a writer is active, and (c) means readiness gates must connect-and-disconnect (polluting client-state metrics).

**Our workaround:** `GMSBackend.has_committed_weights()` only returns a meaningful answer if we're already connected. Pre-connect peek isn't supported.

#### GMS-4. Reconsider default tag name (informational)

**Priority:** Low — discoverability.
**Ask:** Document that `tag="weights"` (model weights) and `tag="kv_cache"` (KV cache) are the canonical names downstream integrators should use; mention this in the GMS README's "API conventions" section.

**Why:** PR #13045 originally defaulted `gms_tag="model_weights"` (more descriptive English) before discovering the convention is just `"weights"` by reading [`GMS_TAGS`](https://github.com/ai-dynamo/dynamo/blob/main/lib/gpu_memory_service/integrations/common/utils.py#L20) in the source. A sentence in the README would have saved the rediscovery.

#### GMS-5. Stable contract: `materialize_module_from_gms` keyword arg requirement (informational)

**Priority:** Low — already documented by signature.
**Status:** Confirms `materialize_module_from_gms(gms_client, model, *, device_index)` requires `device_index` as a keyword arg. PR #13045 handles correctly. No action needed.

#### GMS-6. Publish `gpu-memory-service` to PyPI

**Priority:** Medium — blocks CI and one-line install ergonomics.
**Ask:** Publish `gpu-memory-service` v0.9.0 to PyPI from `ai-dynamo/dynamo/lib/gpu_memory_service/`.

**Why:** The package currently ships only as source inside the `ai-dynamo/dynamo` mono-repo. NVIDIA's Blossom-CI `Vulnerability scan` job cannot resolve a pinned version that isn't on PyPI, so declaring `"gpu-memory-service>=0.9.0,<0.10.0"` as an `extras_require` entry in TRT-LLM's `setup.py` causes the scan to fail (exit code 255), blocking the entire L0 pipeline from starting. This is the primary reason PR #13045's `[gms]` / `[dynamo]` extras were removed from `setup.py`.

**Our workaround:** `[mx]` / `[gms]` / `[dynamo]` extras removed from `setup.py` with inline comment documenting the rationale and manual install instructions (`git clone ... && pip install ./dynamo/lib/gpu_memory_service`). Restoring one-line `pip install tensorrt_llm[gms]` / `tensorrt_llm[dynamo]` ergonomics is a single-hunk revert once this package is published to PyPI **and** onboarded into NVIDIA's OSS allowlist (see MX-7 for the allowlist step).

---

### Summary table

| ID | To | Title | Priority | Workaround in PR #13045? | Blocks merge? |
|---|---|---|---|---|---|
| MX-1 | MX | `LoadFormat.PRESHARDED` vs per-module flag | High | `mx_preshard_strategy='global'` raises until upstream lands | No |
| MX-2 | MX | Promote `_build_trtllm_identity` to public | Medium | `MODEL_EXPRESS_URL` + `MODEL_NAME` env-var dance in `publish_as_source` | No |
| ~~MX-3~~ | MX | ~~Per-rank addressing in identity/metadata~~ | ~~Medium~~ | **Resolved (non-issue)** — upstream uses global MPI rank on both sides | No |
| MX-4 | MX | Adopt `RdmaStrategy` immediate-fallback in `MxLiveWeightLoader` | Medium | `MX_SOURCE_QUERY_TIMEOUT=30` defensive `setdefault` (30 s poll vs correct try-once) | No |
| MX-5 | MX | Clarify NIXL ownership for MX+GMS composition | Medium | MX P2P bypassed in GMS-RW path | **Yes for MX+GMS validation** |
| MX-6 | MX | MLA model support (informational) | Low | — | No |
| MX-7 | MX / TRT-LLM | Onboard `modelexpress` into NVIDIA OSS allowlist | Medium | `[mx]`/`[dynamo]` extras removed from `setup.py`; manual install | No |
| GMS-1 | GMS | Document non-monkey-patch integration path | High | We avoid `setup_gms()` and own the integration in `GMSBackend` | No |
| GMS-2 | GMS | Promote `_move_untracked_params` to public | High | Re-implemented in `GMSBackend.move_untracked_params()` | No (high maintenance burden) |
| GMS-3 | GMS | Lightweight peek RPC | Medium | None — `has_committed_weights()` requires prior `connect()` | No |
| GMS-4 | GMS | Document tag-name conventions | Low | `gms_tag` default is `"weights"` | No |
| GMS-5 | GMS | Confirm `device_index` kwarg contract (informational) | Low | — | No |
| GMS-6 | GMS | Publish `gpu-memory-service` to PyPI | Medium | `[gms]`/`[dynamo]` extras removed from `setup.py`; manual install from source | No |

**MX-1 and MX-5 are the items most likely to come up in PR review** — both have working workarounds today, but both are real design conversations the MX team would want to weigh in on before TRT-LLM lands a stable design.

**MX-7 and GMS-6 are the blockers for restoring one-line `pip install tensorrt_llm[dynamo]` ergonomics** — both are administrative/publish steps, not design issues.

---

### 🔭 Recommended Next Step

The API-alignment blocker is resolved. The next gates are environment-side:

1. **Force-push the rebased + aligned branch to PR #13045.** `dynamo-integration-prototype-rebased` (commit `62ac40f6b`) replaces the current PR head (`84dfb2aa7`). Two new commits in the PR diff vs the prior version: (a) the rebase itself and (b) the API-alignment commit.
2. **GMS-only end-to-end test** on this node (no MX needed): start a `gpu-memory-service` daemon for one GPU, launch `trtllm-serve --load-format gms` (RW), inspect committed bytes, then launch a second `trtllm-serve --load-format gms --gms-mode ro` and verify zero-copy import + correct inference. Exercises B2 and B6 (shadow memory + failover floor) — the prototype's most novel claims.
3. **MX setup on a separate node** with Rust + Docker available; then B1 (bit-exactness), B3 (P2P throughput), B4 (cold-start headline), and the MX+GMS composition can run.
4. **M1 baseline regression** on `dynamo-integration-prototype-rebased` (single §11 v3 config — Qwen 7B TP=1 S2) is still a cheap (~15 min) sanity check that the AUTO/HF path didn't regress through the rebase + alignment work. The pre-alignment smoke (Qwen 7B TP=1 warm NFS: 67.6s server / 36.3s worker) is consistent with no regression but ran on warm NFS, not the §11 cold protocol.

---

## Reuse §11 Baselines (Don't Re-Measure)

The PR rebased onto `upstream/main @ 4a848ccce`, which is exactly the same codebase used for the v3 dataset in §11. **There is no need to re-run baseline measurements** — the §11 v3 numbers are the "before" for our comparison.

Reusable baselines (from [§11 Part 1 v3 results](11-results-analysis.md#part-1--model-size-scaling-s2-nfs-cold--production-baseline)):

| Config | Qwen 7B TP=1 | Qwen 72B TP=8 | DS 7B TP=1 | DS 70B TP=8 |
|--------|------------:|--------------:|------------:|--------------:|
| **S2 (NFS cold) total** | 77.4s | **306.3s** | 94.4s | **389.8s** |
| &nbsp;&nbsp;checkpoint prefetch | 16.9 | 233.2 | 35.7 | 318.4 |
| &nbsp;&nbsp;warmup (1st + 2nd) | 39.1 | 42.7 | 37.9 | 40.9 |
| **S3 (warm) total** | — | **74.6s** | — | **77.7s** |
| &nbsp;&nbsp;checkpoint prefetch | — | 3.5 | — | 6.0 |

Steady-state inference floor (from [§11 Part 5 / Test 4a](11-results-analysis.md#part-5--failover-latency-floor-test-4a)):

| Config | TTFT median | E2E median |
|--------|------------:|-----------:|
| Qwen 72B TP=8 | 63 ms | 108 ms |
| DS 70B TP=8 | 56 ms | 98 ms |

These are the comparison anchors. Verification tests below produce numbers compared directly against these.

---

## Test Matrix

The PR's two-axis design gives 4 configurable modes:

| Mode | `checkpoint_format` | `LoadFormat` | Use case |
|------|--------------------:|-------------:|----------|
| **M1: Baseline** | HF | AUTO | Current behavior — baseline (= §11 v3) |
| **M2: GMS-only** | HF | GMS | Within-node weight sharing + crash resilience |
| **M3: MX-only** | MX | AUTO | Cross-node P2P weight transfer |
| **M4: MX + GMS** | MX | GMS | Full vision — cross-node P2P + within-node sharing |

---

## Verification Tests (Priority-Ordered)

Tests are ordered so failures stop us early before wasting time on later tests.

### B1 — Bit-Exactness (Correctness Gate)

**Why first:** if MX-loaded weights diverge from HF-loaded weights, all subsequent perf measurements are meaningless.

**Protocol:**
1. Start two servers: M1 (`--checkpoint-format HF`) and M3 (`--checkpoint-format MX`)
2. Send identical prompts with greedy decoding (`temperature=0`)
3. Compare output token IDs

**Pass criterion:** Identical token IDs across all prompts.

**Failure mode:** MX is not applying post-load transforms (quant, weight-mapper, etc.) identically to HF. This is a correctness bug in the MX path, not a perf issue.

**Cost:** ~20 min (one Qwen 72B TP=8 server pair).

---

### B2 — GMS Shadow Memory Overhead

**Why second:** zero-copy shadow import is the foundational claim of GMS. If shadow adds full weight-bytes of memory, everything else GMS-related is suspect.

**Protocol:**
1. Start GMS daemon: `gpu-memory-service --socket /tmp/gms-0.sock &`
2. Start primary (`--load-format GMS --gms-mode rw`), wait until ready
3. Record GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
4. Start shadow (`--load-format GMS --gms-mode ro`) on same node
5. Re-record GPU memory after shadow ready
6. Time the GMS RO import phase from the profiler (`executor.load_model_weights` should be near-instant)

**Pass criteria:**
- Shadow adds **~0 GB** of weight memory (some bookkeeping in 10s of MB is OK)
- GMS RO import latency: **<500ms** per [§04](04-implementation-plan.md) success criteria; [§11 Impact Projection](11-results-analysis.md#mxgms-impact-projection) says ~100ms

**Failure mode:** if shadow adds anything close to "weights / TP" bytes (e.g., ~14 GB per rank for Qwen 72B TP=8), the RO zero-copy import is broken.

**Cost:** ~30 min (Qwen 72B TP=8, three runs, plus daemon setup).

---

### B3 — P2P Transfer Throughput

**Why:** anchors whether any shortfall in B4 (cold-start) comes from MX itself or from surrounding TRT-LLM glue. Sub-measurement of B4.

**Protocol:**
- Same node (NVLink): donor and receiver on different ranks of same node
- Cross-node (whatever fabric available — IB HDR, RoCE 100G)
- Measure: weight bytes transferred ÷ MX `_try_p2p_transfer()` duration
- If MX SDK emits transfer telemetry, use that as ground truth

**Pass criteria** (from [§10 Test 2](10-methodology.md#test-2-p2p-transfer-throughput--not-yet-executed)):
- Same node (NVLink): **>50 GB/s**
- Cross-node IB HDR: **>20 GB/s**
- Cross-node RoCE 100G: **>10 GB/s**

**Cost:** ~10 min (sub-test of B4; reuses same servers).

---

### B4 — Cold-Start Headline (the demo)

**Why:** this is the headline number that justifies the whole MX integration. Compares M3 / M4 against §11 baseline.

**Protocol:**
1. Set up MX donor: start an M1 instance (`--checkpoint-format HF`) and let it fully load — this IS the §11 baseline, no separate measurement needed
2. Start MX server: `modelexpress-server --port 8001 &`
3. Profile M3 receiver:
   ```bash
   TRTLLM_PROFILE_STARTUP=1 \
   TRTLLM_STARTUP_PROFILE_OUTPUT=/tmp/mx_b2_run1.json \
   trtllm-serve Qwen/Qwen2.5-72B-Instruct \
       --backend pytorch --tensor_parallel_size 8 \
       --max_batch_size 4 --max_num_tokens 1024 --max_seq_len 4096 \
       --checkpoint-format MX --mx-server-url http://localhost:8001 \
       --port 8002
   # Then drive with benchmark_serving.py --save-startup-metrics (per §10)
   ```
4. 3 runs per config, median-representative protocol (matches §11)
5. Repeat for M4 (`--checkpoint-format MX --load-format GMS`)

**Configs:** B2 (Qwen 72B TP=8) and B4 (DS 70B TP=8). Skip small models — prefetch is already cheap there.

**Pass criteria:**

| Mode | Qwen 72B S2 baseline | M3 target | M4 target |
|------|---------------------:|----------:|----------:|
| Cold start total | 306.3s | **~75–80s** | **~64s** |
| `executor.checkpoint_prefetch` | 233.2s | **~10–15s** | **~0.1s** (zero-copy) |
| CPU memory peak | ~9× model size | ~1× model size | ~1× model size |

**Cost:** ~1 hr (4 configs × 3 runs each, including server setup overhead).

---

### B5 — Throughput Regression

**Why:** ensures the loaded model is bit-identical regardless of how it got into GPU memory. Cheap to run, high-value for sign-off.

**Protocol:**
1. Run steady-state throughput benchmark (sharegpt or random dataset) on M1
2. Run same benchmark on M3 (and M4 if relevant)
3. Compare tokens/sec

**Pass criterion:** **<2% throughput delta** vs M1.

**Failure mode:** weights from MX path haven't gone through the same post-load transforms — this is a correctness bug surfacing as a perf gap. Shouldn't happen if B1 passes, but worth verifying.

**Cost:** ~20 min.

---

### B6 — Failover Latency E2E

**Why:** full validation of the failover story. Only meaningful if Phase 2 / shadow is in scope for the prototype.

**Protocol** (from [§10 Test 4](10-methodology.md#test-4-shadow-failover-latency--partially-executed)):
1. Start primary + GMS shadow on same node
2. Send warmup requests to primary (populates compile cache if implemented)
3. `kill -9` primary
4. Orchestrator routes to shadow; measure time to first response

**Pass criterion:** **<5s** from primary kill → shadow responds. Note: this assumes warm compile cache. Without compile cache, warmup adds ~43s ([§11 Insight #6](11-results-analysis.md#6-warmup-overhead-regression-from-pr-12407-new-in-v3)) and the budget is blown.

**Cost:** ~30 min.

---

### Skipped Tests (Don't Need Re-Measurement)

| Test | Why skipped |
|------|-------------|
| M1 baseline cold-start re-run | Reuse [§11 v3 numbers](11-results-analysis.md#part-1--model-size-scaling-s2-nfs-cold--production-baseline) — same codebase |
| Test 4b (cold-restart failover) | Same as M1 baseline — already in §11 |
| S1 (remote cold) | Already deprioritized in §11 |
| Small-model S2/S3 with MX | Prefetch is already cheap; less compelling demo |
| vLLM comparison ([§10 Test 6](10-methodology.md#test-6-vllm-comparison--not-yet-executed)) | Only meaningful once B4 is stable; follow-up work |

---

## Critical Diagnostic: MX/Worker-Init Overlap

The §11 projection assumes MX P2P transfer **overlaps with the ~21s worker init** (Python imports, CUDA ctx, NCCL setup) — similar to how S1 hides worker init behind the HF download (see [§11 Insight #1](11-results-analysis.md#1-cold-nfs-io-dominates-production-cold-start) and the [Worker Init Investigation](11-results-analysis.md#worker-init-investigation-results)).

If this overlap doesn't actually happen — e.g., MX waits for worker init to complete before starting transfer — we lose ~20s of the projected win and B4 will come in closer to **~95s instead of ~75s**.

**This is not a prototype bug per se**, but it is the most likely explanation for any shortfall vs projection.

### Verification

After each B4 run, check the hierarchical profile:

```python
import json
p = json.load(open('mx_b2_run1.json'))
records = p['attached_profiles']['executor_workers']['ranks'][0]['records']
for rec in records:
    name = rec['name']
    if 'checkpoint_prefetch' in name or 'worker.initialize' in name or 'load_model_weights' in name:
        print(f"{name}: starts {rec['start_offset_s']:.1f}s, dur {rec['duration_s']:.1f}s")
```

**Expected (overlap working):**
```
executor_worker.initialize: starts 0.0s, dur 21.5s   ← worker init happens
executor.load_model_weights: starts 5.0s, dur 15.0s  ← MX transfer overlaps
executor.checkpoint_prefetch: starts 5.5s, dur 10.0s ← P2P starts during init
```

**Shortfall (no overlap):**
```
executor_worker.initialize: starts 0.0s, dur 21.5s   ← worker init first
executor.load_model_weights: starts 21.5s, dur 15.0s ← MX transfer waits
executor.checkpoint_prefetch: starts 22.0s, dur 10.0s ← lost 20s of overlap
```

If shortfall is observed, document it explicitly in the §11 update and treat it as a follow-up optimization (separate from the prototype itself).

---

## Service Setup Reference

### GMS daemon (M2, M4)

```bash
gpu-memory-service --socket /tmp/gms-0.sock &
```

### MX server + donor (M3, M4)

```bash
# 1. MX server
modelexpress-server --port 8001 &

# 2. Donor instance — this IS the §11 v3 baseline
trtllm-serve Qwen/Qwen2.5-72B-Instruct \
    --backend pytorch --tensor_parallel_size 8 \
    --max_batch_size 4 --max_num_tokens 1024 --max_seq_len 4096 \
    --checkpoint-format HF \
    --port 8001 &
# Wait for donor ready before launching M3/M4 receivers
```

### M3 / M4 receiver (the measured instance)

```bash
TRTLLM_PROFILE_STARTUP=1 \
TRTLLM_STARTUP_PROFILE_OUTPUT=/tmp/mx_b2_run<N>.json \
trtllm-serve Qwen/Qwen2.5-72B-Instruct \
    --backend pytorch --tensor_parallel_size 8 \
    --max_batch_size 4 --max_num_tokens 1024 --max_seq_len 4096 \
    --checkpoint-format MX --mx-server-url http://localhost:8001 \
    [--load-format GMS --gms-socket-path /tmp/gms-0.sock --gms-mode auto] \
    --port 8002 &
```

---

## Benchmark Script Changes

`run_startup_bench.sh` needs ~5–10 lines added to forward the new flags:

```bash
# In argument parsing:
        --checkpoint-format) CHECKPOINT_FORMAT="$2"; shift 2 ;;
        --load-format)       LOAD_FORMAT="$2";       shift 2 ;;
        --mx-server-url)     MX_SERVER_URL="$2";     shift 2 ;;
        --gms-socket-path)   GMS_SOCKET_PATH="$2";   shift 2 ;;
        --gms-mode)          GMS_MODE="$2";          shift 2 ;;

# In the trtllm-serve invocation:
        ${CHECKPOINT_FORMAT:+--checkpoint-format $CHECKPOINT_FORMAT} \
        ${LOAD_FORMAT:+--load-format $LOAD_FORMAT} \
        ${MX_SERVER_URL:+--mx-server-url $MX_SERVER_URL} \
        ${GMS_SOCKET_PATH:+--gms-socket-path $GMS_SOCKET_PATH} \
        ${GMS_MODE:+--gms-mode $GMS_MODE} \
```

`run_failover_floor_bench.sh` works as-is — it measures steady-state response time, which is independent of how weights were loaded.

---

## Execution Plan & Time Budget

| Phase | Duration | Output |
|-------|---------:|--------|
| **Phase A**: Branch integration + rebuild | ~30 min | Working `dynamo/proto-bench-integration` branch |
| **B1**: Bit-exactness | ~20 min | Pass/fail correctness gate |
| **B2**: GMS shadow memory overhead | ~30 min | Pass/fail GMS zero-copy gate |
| **B3 + B4**: Cold-start headline + P2P throughput | ~1 hr | The demo numbers + diagnostic profile data |
| **B5**: Throughput regression | ~20 min | <2% sign-off measurement |
| **B6**: Failover E2E | ~30 min | <5s validation (if Phase 2 is in scope) |
| **Total** | **~3.5 hr** | Full validation dataset |

---

## Documentation Outputs (Post-Validation)

After execution, fold results back into the design doc:

1. **Add to [§11 Results & Analysis](11-results-analysis.md):**
   - **Part 6: Prototype Validation** — measured M2/M3/M4 numbers in standard format
   - **Part 7: Bit-exactness verification** — confirms M3 = M1 outputs

2. **Update [§11 Impact Projection](11-results-analysis.md#mxgms-impact-projection):**
   - Replace projected scenarios 3–7 with measured numbers from M2/M3/M4
   - Color-code which lines are now measured vs still projected (e.g., bold for measured)
   - Note any shortfall vs projection with link to the prefetch-overlap diagnostic

3. **Update [§10 Methodology](10-methodology.md):**
   - Mark Test 1 MX/GMS rows as Completed with link to Part 6
   - Mark Test 2 (P2P throughput) as Completed
   - Mark Test 3 (memory efficiency) as Completed
   - Mark Test 4 as Completed (or partially, if only B6 runs)
   - Mark Test 5 (throughput regression) as Completed

4. **This file:** retire or move to a `completed-plans/` archive once Section 11 is updated.

---

## Recommendation

**Start with Phase A + B1.** It's the smallest commitment that gives a meaningful go/no-go signal:

- Phase A (~30 min) confirms branches integrate cleanly + build is good
- B1 (~20 min) is a pure correctness test
- If B1 fails → stop, fix prototype, no perf work wasted
- If B1 passes → B4 (the headline) becomes a 1-hour demo with high impact

After B1 passes, B4 + B3 are the highest-impact next steps; B2 and B5 are validation/sign-off; B6 depends on Phase 2 scope.
