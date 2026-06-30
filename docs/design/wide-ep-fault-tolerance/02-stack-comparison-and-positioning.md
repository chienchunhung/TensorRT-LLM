# 2. Stack Comparison & TRT-LLM's Unique Position

[< Back to Overview](README.md)

## 2.1 Layer-by-layer comparison: TRT-LLM vs vLLM vs SGLang

[§1.2](01-user-journey-and-stack.md#12-the-stack-at-each-layer) establishes TRT-LLM's three-layer stack. Competitors make different choices at each layer, which largely explains why their FT designs look different from ours.

| Layer | TRT-LLM (default) | TRT-LLM (opt-in) | vLLM (multi-node default) | SGLang |
|:---|:---|:---|:---|:---|
| **L1 — Process orchestration** | MPI (`mpirun` / `srun`, `MPIPoolExecutor`) | Ray actors + KubeRay | Ray actors (Ray is the standard multi-node path) | Custom scheduler over Python multiprocessing; Ray for production deploys |
| **L2 — Control plane** | `mpi4py` over `MPI.COMM_WORLD` | `torch.distributed` (cuda:nccl, cpu:gloo) | `torch.distributed` | Custom Python scheduler; `torch.distributed` for collectives |
| **L3 — AlltoAll for MoE (primary)** | **MNNVL shared CUDA memory + custom kernels** (`NVLinkOneSided`) | Same | Elastic-EP baseline `allgather_reducescatter`, optional NIXL-EP; separate in-flight FT work also treats DeepEP timeout handling | **Mooncake EP** |
| **L3 — AlltoAll (cross-node fallback)** | DeepEP / NVSHMEM | Same | DeepEP / NVSHMEM | DeepEP / NVSHMEM |
| **L3 — AlltoAll (generic fallback)** | `AllGatherReduceScatter` / NCCL | Same (via `torch.distributed` path) | NCCL collectives | NCCL collectives |
| **EPLB equivalent** | Mature: online weight migration, host-side POSIX shm, C++ `MoeLoadBalancer`, per-layer replica tracking | Same | Pluggable balancer; less mature; primarily per-batch re-routing | Built-in; re-routing focus rather than weight migration |
| **FT primitive** | **Must add to our own kernel** (no library-level API exists for MNNVL) | Same | DeepEP timeout handling or NIXL-EP `connect_ranks` / `disconnect_ranks`; generic fallback rebuild is NCCL-based | Mooncake `activeRanks` |

The critical row is **L3 AlltoAll ownership**. Competitor recovery is backend-specific: SGLang's Mooncake path exposes `activeRanks`; vLLM's current in-flight FT work treats DeepEP timeout handling and NIXL-EP `connect_ranks` / `disconnect_ranks` topology mutation as distinct mechanisms. Those are library integrations. TRT-LLM's MNNVL path is an owned kernel protocol, so it requires launch-time masking plus a running-kernel escape and coordinated generation commit.

TRT-LLM's primary production path does not use Mooncake. On GB200/NVL72, the AlltoAll is `NVLinkOneSided`: custom CUDA kernels written in TRT-LLM that allocate MNNVL fabric memory via `cuMemCreate(..., CU_MEM_HANDLE_TYPE_FABRIC, ...)` and spin on a `completion_flags[kMaxRanks][kMaxRanks]` table using raw inline PTX. There is no library between TRT-LLM and the hardware. There is also no API to call. **The masking primitive has to live inside our kernel** — we own the synchronization protocol end-to-end.

## 2.2 What makes TRT-LLM's position unique

Four structural properties that matter for FT design. Each has a direct consequence for what FT approach fits TRT-LLM and what would not translate from vLLM / SGLang.

### 2.2.1 Kernel ownership of the performance-critical AlltoAll

TRT-LLM owns the `NVLinkOneSided` / `NVLinkTwoSided` kernels in `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.{h,cu}` and `fusedMoeCommKernels.cu`. No third-party library wraps them.

**Advantage.** We can add fault-tolerance semantics — rank masking, kernel-side abort, completion-flag health telemetry — at the lowest possible level. We are not rate-limited by what Mooncake or DeepEP exposes through its API. Future enhancements (partial-batch completion, adaptive per-peer timeouts) are all in scope.

**Cost.** Kernel modification is harder work than API integration. Multi-GPU memory ordering, PTX memory consistency, and race-free mask propagation across surviving ranks are non-trivial systems problems. The design in [§5.1](05-phase-1-immediate-survival.md) accounts for this explicitly.

**Why this doesn't translate.** SGLang delegates to Mooncake; vLLM delegates to its generic collective, DeepEP, or NIXL-EP backend. TRT-LLM cannot substitute an `activeRanks` or topology-mutation call for its MNNVL path because no such library owns that data path; the mask/abort protocol lives in our kernel.

### 2.2.2 EPLB maturity

TRT-LLM's EPLB (`tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py` + `cpp/tensorrt_llm/runtime/moeLoadBalancer/`) is substantially more developed than vLLM's or SGLang's equivalents:

| Capability | TRT-LLM EPLB | vLLM / SGLang balancers |
|:---|:---|:---|
| Online weight migration at iteration boundary | ✓ (`cudaMemcpy2D` + gdrcopy, <0.3ms/expert) | Partial: vLLM RFC proposes it; SGLang's focus is re-routing rather than migration |
| Host-side shared memory for all expert weights | ✓ (`HostMoeTensorSharer` via POSIX shm, node-local) | Not equivalent |
| Expert replication with per-slot tracking | ✓ (`MoePlacementCpuInfo`, `oldRankExpertIds` for rollback) | Simpler models in others |
| C++ implementation for low overhead | ✓ (worker + compute threads) | Python-dominated in others |
| Per-layer state machine for update coordination | ✓ (`MoeLoadBalanceSingleLayerSignal`) | Not equivalent |

**Advantage for FT.** When a rank dies and its experts need redistribution, much of the placement machinery is already built. MVP can use a placement-table rewrite (< 10 ms) without an H2D copy only after item 1b.2a proves, for every layer and expert, that the admitted failure leaves a resident copy on a distinct failure domain. `HostMoeTensorSharer` is node-local and aggregate spare-slot count is not that proof. If admission fails, serving fails closed rather than zero-filling a missing expert. This is detailed in [§5.2](05-phase-1-immediate-survival.md#52-eplb-topology-adaptation).

**Why this doesn't translate.** Competitors would need to build EPLB first before they could use the slot-remap approach. Their FT designs reflect this — SGLang's Elastic EP handles redistribution by re-routing (no weight move needed because the routing table decides where tokens go), which works but sacrifices the ability to handle the zero-replica case cheaply.

### 2.2.3 MX-GMS integration roadmap

TRT-LLM has an active integration with MX (weight streaming) and GMS (crash-resilient GPU memory), designed by a separate workstream. This is a unique capability that no competitor has:

- **GMS crash-resilient memory** means a process crashing on a GPU *leaves its GPU memory intact*. A replacement process on the same GPU can import those weights in ~100 ms via GMS zero-copy, rather than reloading from disk (minutes).
- **MX P2P RDMA** streams expert shards cross-node at ~20 GB/s; a DS-V3 expert shard (~9.5 GB) transfers in under 0.5s.
- **Shadow EP ranks** can pre-load weights read-only via GMS and activate on failure in sub-second time.

**Advantage for FT.** Phase 2 full-restoration recovery goes from minutes (disk reload) to sub-second. This is the capability that lets TRT-LLM offer a full-restoration path, which neither vLLM nor SGLang can match today.

**Why this doesn't translate.** GMS is an NVIDIA integration; the crash-resilient property is not something vLLM or SGLang could casually adopt without significant platform-level work.

### 2.2.4 NVL72-native design

TRT-LLM's AlltoAll backends are tuned for the MNNVL fabric that defines GB200 NVL72. `NVLinkOneSided` is the backend for a reason—on NVL72, it is the owned performance path. Competitors use different backend-specific mechanisms; the design does not assume Mooncake or NIXL behavior is equivalent to TRT-LLM's raw MNNVL protocol.

**Advantage for FT.** Our design can lean into NVL72-specific properties: 72 is the production rank count, `kMaxRanks` must accommodate it, and the MNNVL workspace teardown problem is scoped to the fabric protocol we own. Host POSIX shared memory remains node-local; rack-wide coordination uses the MNNVL FABRIC/IMEX path and must be validated separately by 1d.4a.

## 2.3 Implications for FT design strategy

These four properties dictate three design choices that shape the rest of this document:

1. **Kernel-level masking is the natural path.** Because TRT-LLM owns the kernel (§2.2.1), the masking primitive belongs inside the kernel. The alternative — an API-level masking wrapper — has no API to wrap. See [§5.1](05-phase-1-immediate-survival.md#51-rank-masking-in-communication-kernels) for the kernel changes.

2. **MVP recovery uses slot-remap only for admitted placements.** Item 1b.2a checks the actual per-layer placement, not an assumed replication factor. When every expert has a surviving resident copy on a distinct admitted failure domain, MVP rewrites `MoePlacementInfo` with zero H2D copies. Otherwise the configuration is rejected or requires the later weight-migration path. See [§5.2](05-phase-1-immediate-survival.md#52-eplb-topology-adaptation).

3. **Full restoration via shadow EP ranks is a differentiator.** Because MX-GMS is on the roadmap (§2.2.3), Phase 2 can investigate sub-second recovery via pre-staged shadow ranks while retaining a correct slower baseline. This is a capability gap that no competitor fills today. See [§6.3](06-phase-2-full-restoration.md#63-shadow-rank--gms-roles).

The flip side of these advantages is that **TRT-LLM has to do more work at the lowest layers** than competitors. SGLang's Elastic EP was mostly integration effort; vLLM's RFC is mostly integration effort. Our equivalent is kernel work, EPLB threading-safety work, and custom consensus protocols. [§3](03-failure-modes-and-gaps.md) details exactly which gaps this translates to, and [§5](05-phase-1-immediate-survival.md) scopes the engineering.
