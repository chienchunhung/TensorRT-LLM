# Chunked KV Transfer: Early Block Release and Prefill-Transfer Pipelining

| | |
|---|---|
| **JIRA** | [TRTLLM-11608](https://jirasw.nvidia.com/browse/TRTLLM-11608) |
| **PRs** | [#12602](https://github.com/NVIDIA/TensorRT-LLM/pull/12602) (Phase 1a), [#12469](https://github.com/NVIDIA/TensorRT-LLM/pull/12469) (Phase 1c), [#12781](https://github.com/NVIDIA/TensorRT-LLM/pull/12781) (Phase 2) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-24 |
| **Last Updated** | 2026-04-09 |
| **Status** | Phase 1a in review; Phase 1b-1c planned; Phase 2 prototype |

## Problem

In disaggregated serving, the context server holds all KV cache blocks for a request from prefill start until the RDMA transfer to the generation server completes. For long-context requests (e.g., 128K tokens, 256 blocks), this creates GPU memory pressure that blocks new prefill requests and degrades throughput.

## Architecture: KV Cache Manager vs Transceiver

The work spans two independent components, each with two implementations:

- **KV Cache Manager** (V1 C++ or V2 Python): owns GPU memory blocks, provides `release_prefix_blocks` API for early block release. This API is transceiver-agnostic.
- **Transceiver** (C++ default or Python): handles RDMA transfer of KV data. Chunking logic (slice creation, `chunk_block_offset`, per-chunk callbacks) lives in the transceiver.

| | C++ Transceiver (default) | Python Transceiver |
|---|---|---|
| **V1 KV Cache (default)** | Phase 1b: chunked transfer | Phase 1a: chunked transfer + Phase 2: pipelined |
| **V2 KV Cache** | Not planned | Phase 1c: chunked transfer + Phase 2: pipelined |

**C++ transceiver:** Default production path. Supports all backends (NIXL, UCX, MPI, MOONCAKE). Allocates a contiguous staging buffer; chunking would allow a smaller buffer. Uses context-first flow only — pipelining is not possible because the generation executor is not allocated until after full prefill.

**Python transceiver:** Auto-selected when `chunk_size_blocks` is set (NIXL/DEFAULT only). GPUDirect RDMA directly from KV cache blocks (no staging buffer). Supports gen-first flow, enabling pipelined prefill-transfer.

## Phased Roadmap

| Phase | What | KV Cache | Transceiver | Scheduling | PR | Status |
|-------|------|----------|-------------|------------|-----|--------|
| **1a** | Chunked transfer + early release | V1 (C++) | Python | Any | [#12602](https://github.com/NVIDIA/TensorRT-LLM/pull/12602) | In review |
| **1b** | Chunked transfer + early release | V1 (C++) | C++ | Context-first | Future | Planned |
| **1c** | Chunked transfer + early release | V2 (Py) | Python | Any | [#12469](https://github.com/NVIDIA/TensorRT-LLM/pull/12469) | Follow-up |
| **2** | Pipelined prefill-transfer | V1+V2 | Python | Gen-first | [#12781](https://github.com/NVIDIA/TensorRT-LLM/pull/12781) | Prototype |

**Why no Phase 2 for C++ transceiver?** The C++ transceiver uses context-first flow where the generation executor is not allocated until the full context forward completes. There is no receiver to send to during prefill. Pipelining requires gen-first flow, which is only supported in the Python transceiver.

## Timeline Comparison

**Baseline (no chunking):**

| Step | GPU (Prefill) | RDMA (Transfer) | Blocks Held |
|------|---------------|-----------------|-------------|
| 1 | Chunk 0 | | 256 |
| 2 | Chunk 1 | | 256 |
| 3 | Chunk 2 | | 256 |
| 4 | Chunk 3 | | 256 |
| 5 | | All 256 blocks | 256 |
| 6 | | (complete) | 0 |

**Phase 1 (chunked + early release, both transceivers):**

| Step | GPU (Prefill) | RDMA (Transfer) | Blocks Held |
|------|---------------|-----------------|-------------|
| 1 | Chunk 0 | | 256 |
| 2 | Chunk 1 | | 256 |
| 3 | Chunk 2 | | 256 |
| 4 | Chunk 3 | | 256 |
| 5 | | Chunk 0 xfer | 256 -> 192 |
| 6 | | Chunk 1 xfer | 192 -> 128 |
| 7 | | Chunk 2 xfer | 128 -> 64 |
| 8 | | Chunk 3 xfer | 64 -> 0 |

**Phase 2 (pipelined, Python transceiver + gen-first only):**

| Step | GPU (Prefill) | RDMA (Transfer) | Blocks Held |
|------|---------------|-----------------|-------------|
| 1 | Chunk 0 | | 64 |
| 2 | Chunk 1 | Chunk 0 xfer | 128 -> 64 |
| 3 | Chunk 2 | Chunk 1 xfer | 128 -> 64 |
| 4 | Chunk 3 | Chunk 2 xfer | 128 -> 64 |
| 5 | | Chunk 3 xfer | 64 -> 0 |

## Performance Impact Summary

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Single-request TTFT | Neutral (transfer latency unchanged) | Improved (transfer overlapped with prefill) |
| System-level TTFT (under load) | Improved (memory freed sooner, new prefills unblocked) | Further improved |
| TPOT | Neutral (decode path unchanged) | Neutral (decode path unchanged) |
| Peak GPU memory | Reduced (incremental release during transfer) | Further reduced (~2 chunks vs C chunks) |
| Context server throughput | Improved (more prefill concurrency) | Further improved |
| C++ staging buffer | Reduced to 1 chunk (Phase 1b) | N/A (Python transceiver has no staging buffer) |

## Implementation Details

### Phase 1a: V1 + Python Transceiver (PR #12602)

- Shared chunking infrastructure: `_create_kv_slices`, `KVSlice.chunk_block_offset`, sender-only dst slicing
- V1 C++ `releasePrefixBlocks` on `WindowBlockManager` / `BlockManager` / `KVCacheManager`
- Thread-safe release queue + `hasattr`-based callback gate
- Auto-selection of Python transceiver when `chunk_size_blocks` is set (NIXL/DEFAULT)
- Warning when `chunk_size_blocks` set with unsupported backend

### Phase 1b: V1 + C++ Transceiver (Future)

- Modify `CacheFormatter::format` to partition blocks into chunk ranges (~500 lines C++)
- Per-chunk callback in `CacheSender` calling existing `releasePrefixBlocks` API
- Reduce staging buffer from `max_tokens_in_buffer` to one chunk's worth
- Enables chunking for UCX/MPI/MOONCAKE backends

### Phase 1c: V2 + Python Transceiver (PR #12469)

- `_KVCache.release_prefix` + `_num_released_prefix_blocks` + `_check_sanity` update
- `KVCacheManagerV2.release_prefix_blocks` (auto-activated by `hasattr` gate)
- ~170 lines, no shared infrastructure changes

### Phase 2: Pipelined Prefill-Transfer (PR #12781)

- CUDA event per chunk for GPU memory ordering
- `send_prefill_chunk()` for incremental session creation
- `_maybe_send_prefill_chunk()` hook in executor after `move_to_next_context_chunk()`
- Requires gen-first scheduling (Python transceiver only)
- Works with both V1 and V2 KV cache managers

### Dependency

    Phase 1a: PR #12602 (V1 + Python transceiver)
      |
      +---> Phase 1b (V1 + C++ transceiver)
      |       |-- CacheFormatter chunking
      |       |-- Staging buffer reduction
      |
      +---> Phase 1c: PR #12469 (V2 + Python transceiver)
      |       |-- _KVCache.release_prefix
      |       |-- hasattr gate auto-activates
      |
      +---> Phase 2: PR #12781 (Pipelined, Python transceiver)
              |-- CUDA event sync per chunk
              |-- Incremental session creation
              |-- Gen-first scheduling required

## Key Metrics to Track

| Metric | How to Measure | Phase 1 | Phase 2 |
|--------|---------------|---------|---------|
| TTFT P50/P99 | SLURM disagg benchmark | Indirect (more prefill capacity) | Direct (transfer overlapped) |
| Context server throughput | `benchmark_serving` ISL >= 32K | Higher (blocks freed sooner) | Higher (freed even sooner) |
| `free_num_blocks` over time | `KvCacheStats` | Step-wise reclamation | Smoother, lower peak |
| Per-chunk transfer throughput | `TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO` | Baseline | Validate no degradation |
| GPU memory high-water mark | `torch.cuda.mem_get_info()` | Reduced | Further reduced |
| C++ staging buffer size | Config | 1 chunk (Phase 1b) | N/A |

## Risks and Mitigations

| Risk | Phase | Mitigation |
|------|-------|------------|
| VSWA counter sharing in V1 C++ | 1a | Assert non-variable-window; documented at call site |
| Python transceiver perf gap vs C++ | 1a | Benchmark before/after; staging buffer savings may offset |
| C++ transceiver touches production path | 1b | Separate PR; thorough testing |
| Stale RDMA reads from GPU memory | 2 | CUDA event synchronization before RDMA |
| Receiver not ready (context-first) | 2 | Gen-first only; context-first not supported for pipelining |
| Request cancellation mid-pipeline | 2 | Existing fail-fast session semantics |

## Detailed Design Documents

- [Phase 1: Chunked KV Transfer with Early Block Release](phase1-early-block-release.md)
- [Phase 2: Pipelined Prefill-Transfer](phase2-pipelined-transfer.md)
