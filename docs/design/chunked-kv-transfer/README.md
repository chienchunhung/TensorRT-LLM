# Chunked KV Transfer: Early Block Release and Prefill-Transfer Pipelining

| | |
|---|---|
| **JIRA** | [TRTLLM-11608](https://jirasw.nvidia.com/browse/TRTLLM-11608) |
| **PRs** | [#12602](https://github.com/NVIDIA/TensorRT-LLM/pull/12602) (shared infra + V1), [#12469](https://github.com/NVIDIA/TensorRT-LLM/pull/12469) (V2 follow-up) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-24 |
| **Last Updated** | 2026-04-06 |
| **Status** | Phase 1 in review; Phase 2 design only |

## Problem

In disaggregated serving, the context server holds all KV cache blocks for a request from prefill start until the RDMA transfer to the generation server completes. For long-context requests (e.g., 128K tokens, 256 blocks), this creates GPU memory pressure that blocks new prefill requests and degrades throughput.

## Approaches

### Phase 1: Chunked KV Transfer with Early Block Release

Split the monolithic RDMA transfer into chunks. Free each chunk's GPU blocks as soon as its RDMA completes. All prefill still finishes before any transfer begins.

**Result:** Memory freed incrementally during transfer. Latency unchanged.

### Phase 2: Pipelined Prefill-Transfer

Start transferring each chunk's KV immediately after its prefill completes, overlapping GPU compute with RDMA. Builds on Phase 1 infrastructure.

**Result:** Memory freed even earlier. Transfer latency partially hidden behind prefill.

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

**Phase 1 (chunked + early release):**

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

**Phase 2 (pipelined):**

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

## Scope

| | Phase 1 | Phase 2 |
|---|---------|---------|
| V1 KV cache (C++) | Supported (PR #12602) | Supported (future) |
| V2 KV cache (Py) | Follow-up PR (#12469) | Supported (future) |
| Complexity | Moderate | Higher |
| Dependencies | None | Phase 1 |

## Implementation Roadmap

### Phase 1 (Foundation) — Two PRs

**PR #12602 (V1 + shared infrastructure):**
- All shared chunking infrastructure ported to `transceiver.py` layout
- V1 C++ `releasePrefixBlocks` on `WindowBlockManager` / `BlockManager` / `KVCacheManager`
- Nanobind binding + Python wrapper
- `hasattr`-based callback gate (activates V1 immediately, V2 when follow-up lands)
- All tests (chunking logic, session state machine, V1+V2 e2e)

**PR #12469 (V2 follow-up, ~170 lines):**
- `_KVCache.release_prefix` + `_num_released_prefix_blocks` + `_check_sanity` update
- `KVCacheManagerV2.release_prefix_blocks`
- V2 type stubs + release_prefix unit tests
- No shared infrastructure changes needed

### Phase 2 (Extension)

- CUDA event recording after each chunk's forward
- Incremental session creation (post-first-chunk, not post-full-prefill)
- Integration with generation-first scheduling mode

### Dependency

    Phase 1, PR #12602 (shared infra + V1)
      |
      |-- KVSlice chunking logic
      |-- chunk_block_offset in _build_kv_write_meta
      |-- Thread-safe release queue
      |-- hasattr-based callback gate
      |-- V1: KVCacheManager::releasePrefixBlocks (C++)
      |
      +---> Phase 1, PR #12469 (V2 follow-up)
      |       |-- V2: _KVCache.release_prefix (Python)
      |       |-- KVCacheManagerV2.release_prefix_blocks
      |       |-- hasattr gate auto-activates V2
      |
      +---> Phase 2 (Pipelined, extends Phase 1)
              |-- CUDA event sync per chunk
              |-- Incremental session creation
              |-- Executor loop integration

## Key Metrics to Track

| Metric | How to Measure | Phase 1 | Phase 2 |
|--------|---------------|---------|---------|
| TTFT P50/P99 | SLURM disagg benchmark | Indirect (more prefill capacity) | Direct (transfer overlapped) |
| Context server throughput | `benchmark_serving` ISL >= 32K | Higher (blocks freed sooner) | Higher (freed even sooner) |
| `free_num_blocks` over time | `KvCacheStats` | Step-wise reclamation | Smoother, lower peak |
| Per-chunk transfer throughput | `TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO` | Baseline | Validate no degradation |
| GPU memory high-water mark | `torch.cuda.mem_get_info()` | Reduced | Further reduced |

## Risks and Mitigations

| Risk | Phase | Mitigation |
|------|-------|------------|
| VSWA counter sharing in V1 C++ | 1 | Assert non-variable-window; disagg doesn't use VSWA |
| Stale RDMA reads from GPU memory | 2 | CUDA event synchronization before RDMA |
| Receiver not ready (context-first mode) | 2 | Pair with generation-first; defer context-first to follow-up |
| Request cancellation mid-pipeline | 2 | Existing fail-fast session semantics |
| RDMA bandwidth contention with compute | 2 | Attention is compute-bound; PCIe contention minimal |

## Detailed Design Documents

- [Phase 1: Chunked KV Transfer with Early Block Release](phase1-early-block-release.md)
- [Phase 2: Pipelined Prefill-Transfer](phase2-pipelined-transfer.md)
