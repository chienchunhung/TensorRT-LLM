# Phase 1: Chunked KV Transfer with Early Block Release

[< Back to Overview](README.md)

| | |
|---|---|
| **JIRA** | [TRTLLM-11608](https://jirasw.nvidia.com/browse/TRTLLM-11608) |
| **PRs** | [#12602](https://github.com/NVIDIA/TensorRT-LLM/pull/12602) (Phase 1a: V1 + Python transceiver), [#12469](https://github.com/NVIDIA/TensorRT-LLM/pull/12469) (Phase 1c: V2 + Python transceiver) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-24 |
| **Last Updated** | 2026-04-09 |
| **Status** | Phase 1a in review; Phase 1b (C++ transceiver) planned; Phase 1c follow-up |

## Problem Statement

### Background

In TensorRT-LLM's disaggregated serving architecture, the context (prefill) server and generation (decode) server run on separate GPUs. After the context server completes prefill for a request, it transfers the KV cache to the generation server via RDMA (typically using NIXL with GPUDirect). During this transfer, the context server holds all KV cache blocks allocated for that request in GPU memory until the entire RDMA transfer completes and `free_resources` is called.

For long-context requests (e.g., 128K tokens), this creates severe GPU memory pressure on the context server:

1. **Block budget exhaustion.** The KV cache pool has a fixed number of blocks (`max_num_blocks`). A single 128K-token request can occupy hundreds of blocks. While those blocks are held during transfer, the scheduler cannot allocate them for new prefill requests, even if the RDMA for earlier portions of the data has already completed.

2. **Head-of-line blocking.** The capacity scheduler checks `enough_available_blocks(req)` before admitting new context requests. When in-flight transfers hold most of the block budget, new prefill requests queue up, reducing the context server's throughput and increasing end-to-end TTFT for downstream requests.

3. **Monolithic transfer latency.** A single large RDMA transfer with thousands of NIXL descriptors can be slower than multiple smaller transfers due to NIC descriptor pressure and internal queuing.

### Goal

Split the monolithic KV cache transfer into smaller chunks and release each chunk's GPU blocks on the context server as soon as its RDMA completes. This has two complementary benefits:

- **Reduced GPU memory pressure.** For a 128K-token request split into 4 chunks, the context server reclaims ~75% of KV memory before the full transfer finishes, allowing new prefill requests to proceed sooner.

- **Reduced per-transfer NIXL descriptor pressure.** Smaller RDMA operations with fewer descriptors can achieve better NIC utilization.

### Key Metrics

| Metric | Expected Impact |
|--------|-----------------|
| `free_num_blocks` during transfer | Higher (blocks freed incrementally) |
| Context server prefill throughput | Higher (new requests scheduled sooner) |
| Time-to-first-token (TTFT) P99 | Lower (reduced head-of-line blocking) |
| Context server GPU memory utilization | Smoother (less bursty allocation) |
| Per-transfer NIXL descriptor count | Lower (bounded by chunk size) |

### Impact on Serving Metrics

- **Single-request TTFT:** Neutral. The total transfer time for an individual request is unchanged — all C chunks still transfer sequentially after prefill completes.
- **System-level TTFT under load:** Improved. Freed blocks become available for new prefill requests before the full transfer finishes. Under high concurrency with long-context requests, this reduces head-of-line blocking and lowers TTFT P99 for queued requests.
- **TPOT:** Neutral. The decode path on the generation server is completely unaffected. No changes to the generation-side KV cache, attention kernel, or sampling.
- **Peak GPU memory:** Reduced. For C chunks, (C-1)/C of memory is reclaimed before the transfer finishes.


## Current Status and Limitations

### Existing Transfer Flow (Before This Work)

    Context Server                              Generation Server
    ─────────────                               ─────────────────
    1. Prefill completes
    2. store_blocks_for_reuse(pin=True)
    3. respond_and_send_async(req)
       └─ _create_kv_slice(req)               4. request_and_receive_async(req)
          → 1 KVSlice (all blocks)               └─ _create_kv_slice(req)
       └─ session.send(slice)                       → 1 KVSlice (all blocks)
          → 1 KVSendTask                         └─ session.receive(slice)
          → 1 monolithic RDMA                       → 1 KVRecvTask
                                                     → 1 monolithic RecvReqInfo
          ... entire transfer in flight ...
          ... ALL blocks held ...

    5. check_context_transfer_status()
       └─ session.wait_complete()
    6. end_transfer() → unpin
    7. free_resources() → remove_sequence()
       └─ ALL blocks freed atomically

**Problems with this flow:**

- **All-or-nothing block lifetime.** Blocks are allocated in `prepare_resources` and freed atomically in `free_resources`. There is no intermediate release point.
- **Single-slice design.** `_create_kv_slice` produces one `KVSlice` with all block IDs. The session tracks one `KVSendTask` / `KVRecvTask`.
- **Monolithic RDMA descriptor pressure.** A single transfer with thousands of descriptors can underutilize NIC resources.

### Why plain chunking alone doesn't reduce GPU memory pressure

Splitting a monolithic RDMA transfer into smaller chunks — without other changes — does not reduce memory usage. In the Python/native path, there are no staging buffers; NIXL performs GPUDirect RDMA directly between registered KV cache pools. The KV cache blocks for a request are all allocated upfront in `prepare_resources()` before the transfer begins, and freed atomically in `remove_sequence()` / `close()` at termination. Merely breaking the transfer into pieces doesn't change when blocks are allocated or freed.

The memory benefit comes from early block release: after each chunk's RDMA completes, those source blocks are released back to the storage pool via `release_prefix_blocks`. For a 128K-token request with 4 chunks, the context server reclaims ~75% of KV memory before the full transfer finishes, allowing new prefill requests to proceed sooner.


## Design

### Architecture

The design has three layers:

1. **Shared chunking infrastructure** — works with both V1 and V2 KV cache managers
2. **V1-specific early release** — C++ `releasePrefixBlocks` API on `BlockManager` (PR #12602)
3. **V2-specific early release** — Python `_KVCache.release_prefix` (PR #12469, follow-up)

A capability-detection gate (`hasattr`) connects layers 1 and 2/3, so each can be added independently.

### Implementation Strategy

The work is split into two PRs for independent review and merge:

| Component | PR | Status |
|-----------|-----|--------|
| Shared chunking infrastructure | #12602 | V1 + shared infra |
| V1 C++ `releasePrefixBlocks` | #12602 | V1 early release |
| V2 `_KVCache.release_prefix` | #12469 | Follow-up, V2 early release |
| `hasattr` callback gate | #12602 | Auto-activates V2 when #12469 merges |


### Shared Infrastructure (V1 and V2)

#### Chunking at KVSlice Level

Transfers are split into multiple `KVSlice` objects, one per chunk. Each slice contains a subset of block IDs per layer group. The chunking boundary is driven by `chunk_size_blocks` (max blocks per layer group per chunk) and is determined by the largest layer group:

    Original:  [b0, b1, b2, b3, b4, b5, b6, b7]  (8 blocks, chunk_size=4)

    Slice 0:   [b0, b1, b2, b3]  is_last_slice=False
    Slice 1:   [b4, b5, b6, b7]  is_last_slice=True

The `chunk_size_blocks` field specifies the maximum blocks per layer group per chunk. The total data volume per chunk is approximately `chunk_size_blocks * num_layer_groups * slot_bytes`.

A reassembly assertion verifies every block appears exactly once across all slices.

**Why chunking is per-layer-group, not across flattened blocks:** Block IDs from different layer groups refer to different physical memory pools. A block ID in layer group 0 maps to `pool_0.base + id * pool_0.slot_bytes`; the same integer in layer group 1 maps to a completely different address in `pool_1`. The entire transfer infrastructure (`KVRegionExtractor`, `RegionMapper`, `_build_kv_write_meta`) iterates over pool mappings and extracts regions per layer group. Additionally, layer groups can have different block counts (e.g., sliding window attention), so uniform flat chunking would not produce meaningful partitions.

#### Sender-Only Chunking

The receiver always sends a single monolithic `RecvReqInfo` with all destination block IDs. Only the sender chunks. In `_build_kv_write_meta`, each sender task uses `chunk_block_offset` to slice the receiver's full destination block list to extract the matching subset. One `RecvReqInfo` arrival triggers one `_respond_with_kv` invocation that dispatches N correctly-paired tasks.

On the receiver side, the sender maps all chunks to `receiver_slice_id=0`. Only the last chunk carries `is_last_slice=True` so the receiver knows when all data has arrived. Intermediate chunk results (`is_last_slice=False`) are intentionally sent (not suppressed) so that RDMA failures propagate to the receiver immediately rather than requiring a timeout.

See **Alternatives Considered** below for the detailed rationale behind sender-only chunking, including the N-squared dispatch bug that rules out receiver-side chunking.

#### Session Multi-Slice Support

- `TxSession.send()` accepts `chunk_block_offset` and creates a `KVSendTask` per slice.
- `TxSession.status` checks **all** `kv_tasks` (not just `[0]`): `ERROR` if any fails, `KV_TRANSFERRED` only when all succeed.
- `TxSession.wait_complete` iterates all task futures.
- `RxSession.status` and `wait_complete` similarly check all tasks.
- The receiver-side `slice_id` is always 0 (the sender maps all chunks to the receiver's single task).

#### Thread-Safe Release Queue

The sender worker thread cannot access the KV cache manager directly (thread safety). A `queue.Queue` (`_pending_prefix_releases`) bridges the two threads:

**Sender worker thread:**

1. Chunk RDMA completes
2. `on_chunk_transferred(req_id, chunk_offset, num_blocks)` is invoked
3. Computes `cumulative_blocks = chunk_offset + num_blocks`
4. Enqueues `(req_id, cumulative_blocks)` into the release queue

**Main executor thread:**

1. `check_context_transfer_status()` is called
2. `_drain_pending_releases()` runs first
3. Dequeues all pending `(req_id, cumulative_blocks)` entries
4. Calls `kv_cache_manager.release_prefix_blocks(req_id, cumulative_blocks)` for each entry

`_drain_pending_releases` is called at the start of `check_context_transfer_status`, ensuring releases are processed before session completion checks.

#### Capability-Detection Callback Gate

`_make_chunk_callback` uses `hasattr(self._kv_cache_manager, 'release_prefix_blocks')` to determine whether early release is available:

| Scenario | `hasattr` result | Behavior |
|----------|---------------|----------|
| V1 PR merged, V2 not yet | V1: `True`, V2: `False` | V1 gets early release; V2 gets chunking only |
| V2 PR merged, V1 not yet | V1: `False`, V2: `True` | V2 gets early release; V1 gets chunking only |
| Both merged | Both: `True` | Both get early release |
| Neither (shared infra only) | Both: `False` | Both get chunking only (reduced descriptor pressure) |

This follows the open/closed principle: adding early release to a new manager type requires only implementing `release_prefix_blocks`, not modifying `_make_chunk_callback`.

#### Configuration

`CacheTransceiverConfig.chunk_size_blocks` (optional `PositiveInt`, default `None`). When set with NIXL/DEFAULT backend, the Python transceiver is auto-selected. `None` produces identical single-slice behavior (backward compatible).

Enable via YAML config:

    cache_transceiver_config:
      backend: "DEFAULT"
      chunk_size_blocks: 64

Or via Python API:

    config = CacheTransceiverConfig(backend="NIXL", chunk_size_blocks=64)

| `chunk_size_blocks` | Backend | Effect |
|---------------------|---------|--------|
| `None` (default) | Any | No chunking, C++ transceiver (unchanged) |
| 64 | NIXL / DEFAULT | Python transceiver auto-selected, chunked transfer + early release |
| 64 | UCX / MPI / MOONCAKE | Warning logged, ignored (C++ transceiver has no chunking support) |

Recommended chunk sizes:

| `chunk_size_blocks` | Granularity | RDMA Overhead | Recommendation |
|-------------------|-------------|---------------|----------------|
| `None` (default) | No chunking | Baseline | Use when memory is not a bottleneck |
| 64-128 | Moderate | Low | Good default for long-context workloads |
| 16-32 | Fine | Moderate | Use when memory is severely constrained |
| 1-4 | Very fine | High | Not recommended (RDMA overhead dominates) |


### V1 KV Cache Manager: C++ `releasePrefixBlocks`

The V1 path wraps a C++ `KVCacheManager` whose `BlockManager` owns `WindowBlockManager` instances (one per attention window size). The Python `KVCacheManager` class accesses it via `self.impl` (nanobind binding).

#### Existing Mechanism: `detachFrontBlock`

The C++ `WindowBlockManager` already has a per-block prefix release mechanism for Sliding Window Attention (SWA):

    WindowBlockManager::detachFrontBlock(sequence):
      1. blockIdx = sequence.getNumFrontBlocksRemoved()
      2. block = allocatedBlocks[blockIdx]
      3. block.decRefCount()
      4. if !block.hasRefs() -> evictionPolicy.releaseBlock(block)
      5. sequence.removeFrontBlock(windowSize)  // increments mNumFrontBlocksRemoved

The `mNumFrontBlocksRemoved` counter tells `releaseBlocks` (called during `removeSequence`) to skip already-freed prefix blocks:

    // In WindowBlockManager::releaseBlocks:
    for (auto it = allocatedBlocks.rbegin();
         it != allocatedBlocks.rend() - sequence.getNumFrontBlocksRemoved();
         ++it) { /* decRefCount + releaseBlock */ }

#### New API: `releasePrefixBlocks`

Three new methods form the release chain:

    KVCacheManager::releasePrefixBlocks(requestId, numBlocks)
      └─ acquires mSequencesMtx, no-op if sequence not found
      └─ BlockManager::releasePrefixBlocks(sequence, numBlocks)
           └─ iterates WindowBlockManagers
           └─ WindowBlockManager::releasePrefixBlocks(sequence, numBlocks)
                └─ per-block: decRefCount → releaseBlock → removeFrontBlock

`WindowBlockManager::releasePrefixBlocks` loops the `detachFrontBlock` logic from `getNumFrontBlocksRemoved()` up to `min(numBlocks, allocatedBlocks.size())`, with cumulative semantics — calling with 3 then 5 releases blocks 0-4 total.

    void WindowBlockManager::releasePrefixBlocks(
        GenerationRequest& sequence, SizeType32 numBlocks)
    {
        TLLM_CHECK(sequence.getBeamWidth() == 1);
        auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(
            sequence.getRequestId());
        SizeType32 target = std::min(numBlocks,
            static_cast<SizeType32>(allocatedBlocks.size()));

        while (sequence.getNumFrontBlocksRemoved() < target)
        {
            SizeType32 blockIdx = sequence.getNumFrontBlocksRemoved();
            auto& block = allocatedBlocks.at(blockIdx);
            if (block->hasRefs()) block->decRefCount();
            if (!block->hasRefs())
                mEvictionPolicy->releaseBlock(block);
            sequence.removeFrontBlock(mWindowSize);
        }
    }

#### Nanobind Binding

Exposed on the concrete `KVCacheManager` class (not the `BaseKVCacheManager` virtual interface) since this is specific to the V1 implementation:

    .def("release_prefix_blocks",
        &KVCacheManager::releasePrefixBlocks,
        nb::arg("request_id"), nb::arg("num_blocks"),
        nb::call_guard<nb::gil_scoped_release>())

#### Python V1 Wrapper

    # In KVCacheManager (V1) in resource_manager.py
    def release_prefix_blocks(self, request_id: int,
                              num_blocks: int) -> None:
        self.impl.release_prefix_blocks(request_id, num_blocks)


### V2 KV Cache Manager: Python-Side `release_prefix` (Follow-Up PR #12469)

The V2 path uses pure-Python `_KVCache` objects managed by `KVCacheManagerV2`.

#### `_KVCache.release_prefix(num_blocks)`

- Nulls out `_PageHolder` references in each `SeqBlock`'s `beam_pages` for the first `num_blocks` blocks.
- When a `_PageHolder` reference is dropped, its `__del__` method transitions the underlying page from `HELD` to `DROPPABLE` and calls `schedule_for_eviction`, placing the page into the eviction queue at its current cache level (GPU).
- Sets `_base_page_indices` entries to `BAD_PAGE_INDEX`.
- Updates `_num_released_prefix_blocks` watermark so that `_check_sanity` skips holder-type assertions for blocks whose holders were intentionally nulled.
- The `_blocks` list is **not** shortened — block ordinals and `close()` / `stop_committing()` invariants are preserved.

#### Interaction with Multi-Level Cache

Released pages go through the normal cache hierarchy:

| Scenario | Behavior |
|----------|----------|
| No memory pressure | Pages sit in GPU eviction queue, reusable via radix tree |
| GPU pressure only | Pages demoted GPU -> host via `_batched_migrate`, GPU slot freed |
| GPU + host pressure | Pages evicted from host -> `Page.__del__` -> slot freed |
| Single-level (GPU only) | Pages evicted from GPU -> `Page.__del__` -> slot freed |

#### `KVCacheManagerV2.release_prefix_blocks(request_id, num_blocks)`

Thin wrapper that looks up the `_KVCache` in `kv_cache_map` and calls `release_prefix`. No-op if the request is not found (already freed).

Once this method exists, the `hasattr` gate in `_make_chunk_callback` (from PR #12602) auto-activates V2 early release — no other code changes needed.


### Transfer Flow (After This Work)

    Context Server                                Generation Server
    ─────────────                                 ─────────────────
    1. Prefill completes (128K tokens, 256 blocks)
    2. store_blocks_for_reuse(pin=True)
    3. respond_and_send_async(req)
       └─ _create_kv_slices(req)                 4. request_and_receive_async(req)
          → 4 KVSlices (64 blocks each)             └─ _collect_block_ids(req)
       └─ session.send(slice_0, offset=0)              → 1 KVSlice (all 256 blocks)
       └─ session.send(slice_1, offset=64)           └─ session.receive(full_slice)
       └─ session.send(slice_2, offset=128)
       └─ session.send(slice_3, offset=192)

       [slice_0 RDMA completes]
       → on_chunk_transferred(req, 0, 64)
       → queue.put((req, 64))

       [main thread: _drain_pending_releases]
       → release_prefix_blocks(req, 64)
       → 64 blocks freed!                        5. Receiver sees is_last_slice=False
       → New prefill can use those blocks!           (continues waiting)

       [slice_1 RDMA completes]
       → release_prefix_blocks(req, 128)
       → 64 more blocks freed!

       [slice_2 RDMA completes]
       → release_prefix_blocks(req, 192)
       → 64 more blocks freed!

       [slice_3 RDMA completes]                  6. Receiver sees is_last_slice=True
       → release_prefix_blocks(req, 256)            → KV_TRANSFERRED
       → last 64 blocks freed!

    7. check_context_transfer_status()
       → session completed
    8. end_transfer() → unpin
    9. free_resources() → remove_sequence()
       → releaseBlocks() skips already-freed
         prefix blocks (mNumFrontBlocksRemoved)


## Alternatives Considered

### Chunking within a single KVSlice (rejected)

**Approach:** One `KVSlice` with all blocks; the transfer agent internally batches pointer lists into smaller RDMA calls.

**Rejected because:**
- Requires a second layer of sub-task tracking hidden inside the transfer agent, duplicating the session layer's existing `Future`/`TaskStatus`/`slice_id` tracking.
- Error boundaries are unclear — which sub-batch failed?
- Per-chunk completion callbacks don't plug into the existing task lifecycle.
- The `KVSlice` abstraction was designed for multi-slice transfers: `token_range`, `layer_range`, and `is_last_slice` fields exist precisely for this.

### Chunking on both sender and receiver (rejected)

**Approach:** The receiver sends N `RecvReqInfo` messages (one per chunk, each containing that chunk's destination block IDs). The sender creates N `KVSendTask`s. Each `RecvReqInfo` arrival pairs with the corresponding sender task for a 1:1 dispatch.

**Why it fails — N-squared dispatch bug:**

The sender's message handler `_respond_with_kv` fires once per incoming `RecvReqInfo`. On each invocation, it iterates **all** `kv_tasks` in the session and dispatches every one of them:

    def _respond_with_kv(self, _send_id, message):
        info = RecvReqInfo.from_bytes(message[1])
        session = self._get_session(info.unique_rid)
        self._save_peer_req_info(info)
        tasks = list(session.kv_tasks)      # ALL tasks, not just the matching one
        for task in tasks:
            trans_meta = self._build_kv_write_meta(task, info)
            self._enqueue(trans_meta)

With N `RecvReqInfo` arrivals and N `kv_tasks`, this produces N x N dispatch attempts. Each dispatch pairs the wrong source chunk with the wrong destination chunk (e.g., sender chunk 2's blocks paired with receiver chunk 0's destination addresses).

**Root cause:** `RecvReqInfo` has no `slice_id` field. The `_peer_requests` store uses `(unique_rid, instance_rank)` as the key, so multiple `RecvReqInfo` messages from the same peer overwrite each other:

    _peer_requests[unique_rid][instance_rank] = info   # last one wins

When `_respond_with_kv` reads the stored info to build `WriteMeta`, it always sees the **last** `RecvReqInfo`'s destination blocks, regardless of which sender task is being processed. This means:

- `RecvReqInfo` 0 arrives: dispatches tasks 0..N-1, all using recv chunk 0's dst blocks
- `RecvReqInfo` 1 arrives: overwrites stored info; dispatches tasks 0..N-1 again, all using recv chunk 1's dst blocks
- Result: N-squared RDMA operations with mismatched src/dst pairings

**Why sender-only chunking avoids this:**

One `RecvReqInfo` arrival (with all destination blocks) triggers one `_respond_with_kv` invocation. That single invocation dispatches all N sender tasks. Each task uses its own `chunk_block_offset` to slice the correct subset from the receiver's full destination block list:

    # In _build_kv_write_meta:
    chunk_offset = task.chunk_block_offset
    dst_block_ids = full_dst_block_ids[
        chunk_offset : chunk_offset + len(src_block_ids)
    ]

This produces exactly N correctly-paired RDMA operations. No protocol changes, no new fields on `RecvReqInfo`, and the receiver code is completely unchanged.

**Alternative fix considered but rejected:** Adding a `slice_id` field to `RecvReqInfo` and changing `_peer_requests` to key on `(unique_rid, instance_rank, slice_id)`. This would fix the N-squared bug but requires protocol changes (`RecvReqInfo` serialization format), receiver-side changes, `_respond_with_kv` matching logic, and testing the full cross-product of sender/receiver chunk counts. The sender-only approach achieves the same result with zero protocol or receiver changes.

### Flattened block chunking across layer groups (rejected)

**Approach:** Flatten all block IDs across layer groups and chunk the flat list.

**Rejected because:**
- Block IDs from different layer groups refer to different physical memory pools; the same integer in group 0 and group 1 maps to completely different addresses.
- The entire transfer infrastructure (`KVRegionExtractor`, `RegionMapper`) iterates per layer group.
- Layer groups can have different block counts (e.g., sliding window), so flat chunking produces meaningless partitions.

### `is_v2_manager` gate (replaced with `hasattr`)

**Original design:** `_make_chunk_callback` checks `if not self.is_v2_manager` to gate early release.

**Replaced with** `hasattr(self._kv_cache_manager, 'release_prefix_blocks')` **because:**
- Decouples V1 and V2 PRs — either can merge first
- Follows the open/closed principle — new manager types activate automatically by implementing `release_prefix_blocks`
- No dead code paths for unsupported managers


## Performance Analysis

### Expected Gains

Memory pressure reduction is the primary benefit. For a request with N blocks split into C chunks, the context server reclaims (C-1)/C of KV memory before the full transfer finishes. With 4 chunks, ~75% of memory is reclaimed early.

The throughput improvement depends on the workload characteristics:

| Workload | Expected Benefit |
|----------|------------------|
| High concurrency, long context (ISL >= 32K) | Significant: long transfers hold blocks for longer, chunking enables overlap |
| Low concurrency, short context (ISL <= 4K) | Minimal: transfers complete quickly, memory pressure is not the bottleneck |
| Mixed ISL workloads | Moderate: long requests benefit, short requests see reduced head-of-line blocking |

NIXL descriptor pressure is reduced regardless of early release. Smaller per-transfer descriptor lists can improve NIC utilization on some hardware configurations.

### Overhead

- **Per-chunk callback overhead.** Each chunk completion invokes a Python callback on the sender worker thread that does a `queue.put()`. This is O(1) and negligible compared to RDMA latency.
- **Main-thread drain overhead.** `_drain_pending_releases` calls `release_prefix_blocks` for each queued entry. For V2, this is pure Python (null out page holders). For V1, this crosses into C++ via nanobind (acquire `mSequencesMtx`, iterate blocks). Both are fast relative to the RDMA transfer time.
- **Additional RDMA operations.** Splitting into C chunks means C separate RDMA submissions instead of 1. Each submission has fixed overhead (descriptor setup, NIC doorbell). For large chunk sizes (e.g., 64 blocks x 128 layers x 256KB slot = ~2GB per chunk), this overhead is negligible. For very small chunk sizes, the overhead could dominate.

### Measurement

Enable per-transfer perf logging with:

    export TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO=1
    export TLLM_KV_TRANSFER_PERF_LOG_FILE=/path/to/perf_log

This produces CSV records with `transfer_size_bytes`, `transfer_latency_ms`, and `throughput_mbs` per task. Compare monolithic vs. chunked transfer throughput to verify that chunking does not degrade RDMA bandwidth.

Monitor `KvCacheStats.free_num_blocks` over time (via `_update_iter_stats`) to observe the step-wise block reclamation pattern with chunking enabled.


## Further Discussion

### Interaction with Block Reuse

When `enable_block_reuse` is enabled, `start_transfer` calls `store_blocks_for_reuse(pin=True)` before the send begins. This commits blocks to the radix tree and pins them against eviction.

- **V2:** `release_prefix` drops `_PageHolder` references, triggering `schedule_for_eviction` on each committed page. Pages remain in the radix tree as eviction candidates. When memory pressure occurs, pages are demoted through the cache hierarchy (GPU -> host -> disk) rather than being destroyed.
- **V1:** `releasePrefixBlocks` calls `decRefCount` + `releaseBlock` on each block. If the block was stored in the radix tree (via `storeBlocksForReuse`), it remains reusable as an eviction candidate. The block is not destroyed — it enters the eviction queue managed by `LRUEvictionPolicy`.

In both cases, early release and block reuse coexist naturally. The per-block lifecycle is identical to what happens at `close()` / `removeSequence` time — `release_prefix` / `releasePrefixBlocks` simply triggers it earlier, per-chunk rather than all-at-once.

### Variable Sliding Window Attention (VSWA)

The V1 C++ implementation uses a shared `mNumFrontBlocksRemoved` counter on `GenerationRequest` across all `WindowBlockManager` instances. With multiple window sizes, `releasePrefixBlocks` on the first window manager increments the counter, causing the second window manager to see an already-advanced offset.

**Current scope:** Disaggregated serving does not support VSWA (confirmed by `should_store_blocks` gating: `not self.kv_cache_manager.is_vswa`). An assertion is added in `releasePrefixBlocks` for safety.

**Future:** If VSWA support is needed, the counter would need to be made per-window-size, or `releasePrefixBlocks` would need to save/restore the counter for each window manager.

### Beam Width > 1

Both V1 and V2 early release assume `beam_width == 1`:

- V1: `detachFrontBlock` (and by extension `releasePrefixBlocks`) asserts `beamWidth == 1` in C++.
- V2: `release_prefix` iterates `block.pages` which contains per-beam entries, so it handles multi-beam in principle, but disaggregated context requests are always `beam_width == 1`.

A Python-side guard in `respond_and_send_async` sets the early-release callback to `None` for `beam_width > 1`, so chunked transfer still works (reduced descriptor pressure) but early release is skipped. This prevents the C++ assertion from crashing the process if beam search is ever combined with chunked transfer.

This is not a practical limitation since disaggregated serving context-only requests always use beam width 1.

### Backward Compatibility

- `chunk_size_blocks=None` (default) produces identical single-slice behavior.
- No protocol changes, no new message types, no changes to the generation server.
- The `RecvReqInfo` format is unchanged; sender-side chunking is transparent to the receiver.
- Existing tests continue to pass with the default configuration.

### Error Handling

- **Fail-fast semantics.** If any chunk's RDMA fails, the sender worker sets the task's future to `ERROR`. `TxSession.status` reports `ERROR` if any task fails. The session is cleaned up the same way as the monolithic case.
- **Partial release on failure.** If chunks 0-2 succeed but chunk 3 fails, blocks from chunks 0-2 are already released. The remaining blocks (chunk 3) are freed during `removeSequence` / `close()` cleanup. No blocks are leaked.
- **Race with `removeSequence`.** `releasePrefixBlocks` acquires `mSequencesMtx` (V1) or checks `kv_cache_map` (V2) and returns early if the sequence was already removed. The release queue may contain stale entries for completed requests; these are harmlessly skipped.

### Python vs C++ Transceiver

Chunking is currently implemented in the Python transceiver only (`KvCacheTransceiverV2`). The C++ transceiver (`BindKvCacheTransceiver` / `CacheTransceiver`) is the production default but has no chunking support. The Python transceiver is auto-selected when `chunk_size_blocks` is set with NIXL/DEFAULT backend.

Key differences:

- **Python transceiver:** GPUDirect RDMA directly from KV cache blocks (no staging buffer). Fully extensible for chunking, callbacks, release queue. Only supports NIXL backend.
- **C++ transceiver:** Allocates a contiguous staging buffer of `max_tokens_in_buffer` size per transfer. Supports all backends (NIXL, UCX, MPI, MOONCAKE). Monolithic `respondAndSendAsync` with no per-chunk hook points.

The `releasePrefixBlocks` C++ API is transceiver-agnostic — adding chunking to the C++ transceiver (~500 lines) would call it directly. This is planned as future work.

### Potential Follow-Up Work

1. **Per-chunk retry.** Currently, if any chunk fails, the entire session fails (fail-fast). Per-chunk retry could improve resilience for transient RDMA errors without restarting the full transfer.
2. **Pipelined prefill-transfer.** Overlap chunk N's RDMA with chunk N+1's prefill computation. See [Phase 2 design](phase2-pipelined-transfer.md) for the detailed design.
3. **Adaptive chunk sizing.** Dynamically adjust `chunk_size_blocks` based on real-time `free_num_blocks` pressure. When memory is abundant, use larger chunks (less overhead); when memory is tight, use smaller chunks (faster reclamation).
4. **Receiver-side chunking.** The current design is sender-only. Receiver-side chunking could enable the generation server to start decode on partial KV data (speculative prefix decode), but this requires significant changes to the attention kernel and scheduler.
5. **Multi-threaded slice distribution.** Currently, all slices for a request are routed to the same sender worker thread via `unique_rid % num_threads`. Distributing slices across threads was considered but deferred — NIC bandwidth is typically the bottleneck, not Python thread overhead.
6. **C++ transceiver support.** The chunking infrastructure is Python-transceiver only (auto-selected for NIXL/DEFAULT when `chunk_size_blocks` is set). Extending to the C++ transceiver (~500 lines) would require modifying `CacheFormatter::format` to partition blocks into chunk ranges and adding per-chunk callbacks in `CacheSender` that call the existing `releasePrefixBlocks` API. This would enable chunking for UCX/MPI/MOONCAKE backends.
7. **VSWA support.** The `mNumFrontBlocksRemoved` counter is shared across window managers. Supporting VSWA would require making it per-window-size.

### Test Coverage

All unit tests call real production methods (via mock transceivers), not reimplemented logic.

| Test | Description |
|------|-------------|
| `test_create_kv_slices_basic` | 5 parametrized cases calling real `_create_kv_slices` (no_chunking, even_split, uneven_split, empty_blocks, chunk_larger_than_total) |
| `test_create_kv_slices_integrity_check` | Reassembled block IDs match original across layer groups |
| `test_create_kv_slices_multiple_layer_groups` | Asymmetric layer groups produce correct chunking |
| `test_transfer_worker_chunked[v1_tp1_pp1_chunked]` | E2E GPU test with actual NIXL transfer (V1) |
| `test_transfer_worker_chunked[v2_tp1_pp1_chunked]` | E2E GPU test with actual NIXL transfer (V2) |
| `test_chunked_transfer.py` | 19 tests for session state machine using real `TxSession`/`RxSession` classes |
| `test_make_chunk_callback_conditions` | 4 parametrized cases calling real `KvCacheTransceiverV2._make_chunk_callback` |
| `test_chunk_callback_enqueues_release` | Real callback from `_make_chunk_callback` enqueues correct entries |
| `test_chunk_callback_then_drain` | End-to-end: real `_make_chunk_callback` + real `_drain_pending_releases` |
| `test_drain_pending_releases` | Real `_drain_pending_releases` calls `release_prefix_blocks` correctly |
| `test_cache_transceiver_config_chunk_size_blocks` | Config field validation (valid, None, default, zero, negative) |
| `test_release_prefix_*` (V2 follow-up) | 7 unit tests for `_KVCache.release_prefix` (basic, zero, clamped, cumulative, close-after-release, negative) |
