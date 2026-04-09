# Phase 2: Pipelined Prefill-Transfer

[< Back to Overview](README.md)

| | |
|---|---|
| **JIRA** | [TRTLLM-11608](https://jirasw.nvidia.com/browse/TRTLLM-11608) |
| **PRs** | [#12781](https://github.com/NVIDIA/TensorRT-LLM/pull/12781) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-24 |
| **Last Updated** | 2026-04-09 |
| **Status** | Prototype |
| **Depends on** | [Phase 1](phase1-early-block-release.md) |
| **Transceiver** | Python only (gen-first flow required) |

## Problem Statement

### Background

With the chunked KV cache transfer work (Phase 1), the sender splits a monolithic RDMA transfer into chunks and releases blocks as each chunk completes. However, **all prefill computation must finish before any transfer begins.** For a 128K-token request chunked into 4 prefill iterations, the context server computes all 4 chunks sequentially, and only then starts the first RDMA transfer. The RDMA transfer latency is entirely additive to the prefill latency.

### Functional Goal

Start transferring each chunk's KV cache data as soon as its prefill completes, rather than waiting for the entire prompt to be prefilled. The context server should overlap GPU prefill computation with RDMA transfer.

### Performance Goal

Reduce the wall-clock time from "request arrives at context server" to "generation server has all KV data." For C chunks where transfer time per chunk (T_t) is less than prefill time per chunk (T_p), transfer latency is nearly hidden:

    Current:     C * T_p + C * T_t
    Pipelined:   C * T_p + T_t       (last chunk's transfer only)

Additionally, the memory high-water mark is reduced to ~2 chunks worth (one being prefilled, one being transferred) instead of all C chunks.

### Key Metrics

| Metric | Target |
|--------|--------|
| TTFT P50/P99 | Directly reduced (transfer overlapped with prefill) |
| Context server throughput | Higher (lower peak memory, faster turnaround) |
| GPU memory high-water mark | ~2 chunks instead of C chunks |
| Transfer throughput | No degradation vs Phase 1 |

### Impact on Serving Metrics

- **Single-request TTFT:** Improved. Transfer latency is partially or fully hidden behind prefill compute. The generation server receives all KV data up to `C * T_t` sooner (when `T_t < T_p`, nearly all transfer time is hidden).
- **System-level TTFT under load:** Further improved over Phase 1. Peak memory is ~2 chunks instead of C chunks, allowing even more prefill concurrency.
- **TPOT:** Neutral. The decode path is completely unaffected. Pipelining only changes the context server's prefill-to-transfer timing.
- **Peak GPU memory:** Further reduced compared to Phase 1. At steady state, only ~2 chunks worth of blocks are held (one being prefilled, one being transferred).

### Timeline Comparison

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


## Current Status and Limitations

### What Phase 1 Provides

Phase 1 (chunked transfer with early block release) provides:

- KVSlice chunking infrastructure (`_create_kv_slices`, `chunk_block_offset`)
- Per-chunk completion callbacks and thread-safe release queue
- V1 and V2 early block release APIs (`release_prefix_blocks`)
- Session multi-slice tracking (`TxSession.send` with `chunk_block_offset`)
- Receiver unchanged: single monolithic `RecvReqInfo`

### Remaining Limitation

All prefill computation must finish before any transfer begins. `_send_kv_async` only fires when `is_context_finished` is true:

    if req.is_context_only_request and (
            req.is_context_finished or req.is_finished_due_to_length
    ) and not req.is_finished_due_to_cancellation:
        self.async_transfer_manager.start_transfer(req)
        self.kv_cache_transceiver.respond_and_send_async(req)

For a 4-chunk prefill, the GPU is idle during transfer and the NIC is idle during prefill. The two operations are fully serialized.

### Opportunity: KV Block Stability

Analysis of the codebase confirms KV blocks for earlier chunks are safe to transfer during subsequent prefill:

1. **No recomputation.** The model engine slices `input_ids` to `[context_current_position : context_current_position + context_chunk_size]`. Chunk N+1's forward does not include chunk N's tokens. Both FlashInfer (`append_paged_kv_cache`) and the TRTLLM backend write KV only for the current chunk.

2. **No RoPE rewriting.** Each chunk uses absolute position IDs: `range(begin_compute, begin_compute + len(prompt_tokens))`. Earlier KV entries are not modified.

3. **Block allocation.** V1 allocates all blocks upfront (`add_sequence(request_id, prompt_len, ...)`). V2 grows incrementally via `resize_context`. In both cases, chunk N's block IDs are known after chunk N's prefill.

4. **Incremental KV commit.** `update_resources` commits prefix KV up to `context_current_position` after each chunk (when block reuse is enabled), confirming per-chunk finality.


## Design

### CUDA Memory Ordering

The most critical correctness concern. The GPU writes KV data on the default CUDA stream. GPUDirect RDMA reads from the same GPU memory. Without synchronization, RDMA could read stale or partial data.

**Solution:** Record a CUDA event after each chunk's forward. The sender worker thread calls `event.synchronize()` before initiating RDMA for that chunk.

    # Main executor thread (after chunk N's forward):
    cuda_event = torch.cuda.Event()
    cuda_event.record()  # records on the current (default) stream
    # Pass cuda_event to the sender worker along with the KVSlice

    # Sender worker thread:
    cuda_event.synchronize()  # blocks this thread until GPU work done
    agent.submit_transfer_requests(...)  # safe to read GPU memory

`event.synchronize()` blocks only the sender worker thread, not the GPU or the main executor thread. Chunk N+1's forward runs on the GPU concurrently.

### Why Python Transceiver Only (No C++ Transceiver)

The C++ transceiver uses context-first flow exclusively. In context-first, the generation executor is not allocated until the full context forward pass completes — there is no receiver to send to during prefill. Pipelining requires gen-first flow where the receiver is ready before the sender starts, and gen-first is only supported in the Python transceiver.

Phase 1 (chunked transfer without pipelining) applies to both transceivers. Phase 2 (pipelined) is Python transceiver only.

### Scheduling Mode Interaction

**Generation-first (required):** The generation server receives the request simultaneously with the context server. It allocates KV blocks and sends `RecvReqInfo` before the context server finishes prefill. By the time chunk 0's prefill completes, the receiver may already be ready.

**Context-first (not supported for pipelining):** The orchestrator only sends the gen request after full context completion. Pipelining requires either:

- The context server proactively notifying the gen server after the first chunk
- The orchestrator sending a "prepare" signal earlier

This is a more invasive change, deferred to future work.

### Session Lifecycle Changes

Currently, `TxSession` is created in `respond_and_send_async` after `is_context_finished`. For pipelined transfer:

1. **Create session after first chunk.** After chunk 0's forward, create the `TxSession` and send slice 0.
2. **Add slices incrementally.** After each subsequent chunk's forward, call `session.send(slice_N)` to enqueue additional slices.
3. **`start_transfer` timing.** `AsyncTransferManager.start_transfer` currently calls `store_blocks_for_reuse(pin=True)` and releases non-KV resources. For pipelined mode, pinning and resource release would need to happen after the first chunk or be deferred until full prefill completes.

### Integration in Executor Loop

The key change is in the executor main loop. After each context chunk's forward, check if the request is a disaggregated context-only request with pipelining enabled:

    # In _update_request_states or a new post-forward hook:
    for req in scheduled_requests.context_requests:
        if (req.is_context_only_request
                and self.pipelined_transfer_enabled
                and not req.is_first_context_chunk):
            # Chunk N-1's KV is now finalized
            self._start_chunk_transfer(req,
                chunk_index=current_chunk - 1)

The `is_first_context_chunk` edge case: after the first chunk's forward, we need to create the session and potentially wait for the receiver's `RecvReqInfo`.

### Block ID Availability

- **V1:** All block IDs known after the first chunk (`add_sequence` allocates for the full prompt). Chunk 0's block IDs can be extracted immediately.
- **V2:** Block IDs grow incrementally via `resize_context`. Each chunk's block IDs available after that chunk's `resize_context` call.

### Receiver Side

Two sub-approaches:

**A. Sender-only pipelining (recommended, simpler):** The receiver still sends a single monolithic `RecvReqInfo` with all destination blocks. The sender pipelines its chunks but uses `chunk_block_offset` to slice into the receiver's full destination list. Identical to Phase 1 protocol — only the timing of when the sender starts changes.

**B. Progressive receiver (deferred):** The receiver sends `RecvReqInfo` incrementally as it allocates blocks. Reduces receiver upfront memory but requires protocol changes.

### Pipelined Transfer Flow

    Context Server:
      [1] Prefill chunk 0
      [2] Record CUDA event
      [3] Create TxSession, send slice 0
          -> Worker: event.synchronize()
          -> RDMA for chunk 0 (overlaps step 4)
      [4] Prefill chunk 1
          [3a] Chunk 0 RDMA done
               -> release_prefix_blocks(64)
      [5] Record CUDA event, send slice 1
          -> RDMA for chunk 1 (overlaps step 6)
      [6] Prefill chunk 2
          [5a] Chunk 1 RDMA done
               -> release_prefix_blocks(128)
      ... (repeat)
      [N] Last chunk RDMA done
      [N+1] end_transfer, free_resources

    Generation Server (unchanged):
      Same as Phase 1

### Changes Required

| Component | Change |
|-----------|--------|
| `py_executor.py` | Post-forward hook for per-chunk transfer start |
| `transceiver.py` | Incremental session creation, CUDA event plumbing |
| `transfer.py` | CUDA event field on `KVSendTask`, worker `event.synchronize()` |
| `AsyncTransferManager` | Incremental `start_transfer` support |

### V1 and V2 Compatibility

Phase 2 builds on Phase 1's infrastructure. The `hasattr`-based callback gate ensures both V1 (C++ `releasePrefixBlocks`) and V2 (`_KVCache.release_prefix`) work with pipelined transfer without modification. The CUDA event synchronization and session lifecycle changes are in the Python transceiver layer, shared by both managers.


## Alternatives Considered

### CUDA stream-ordered RDMA (deferred)

Instead of `event.synchronize()` on the worker thread, use CUDA stream-ordered RDMA APIs (if supported by NIXL/UCX) to issue the RDMA from a GPU stream that depends on the forward stream via events. Eliminates worker thread blocking entirely. Deferred because it depends on NIXL API support.

### Multi-threaded slice distribution (rejected)

Distribute slices across sender worker threads instead of routing all slices for a request to the same thread (via `unique_rid % num_threads`). Rejected because NIC bandwidth is the bottleneck, not Python thread overhead. Would also complicate future pipelining support where slice ordering matters.

### Receiver-side incremental allocation (deferred)

Instead of allocating blocks for the full prompt upfront, the generation server allocates incrementally as chunks arrive. Reduces peak memory on the generation server but requires protocol changes to `RecvReqInfo`. Deferred to future work.


## Performance Analysis

### Expected Latency Reduction

Let `T_p` = prefill time per chunk, `T_t` = transfer time per chunk, `C` = number of chunks.

| Scenario | Current | Pipelined | Speedup |
|----------|---------|-----------|---------|
| `T_t << T_p` | `C*T_p + C*T_t` | `C*T_p + T_t` | Transfer nearly hidden |
| `T_t ~= T_p` | `C*T_p + C*T_t` | `C*T_p + T_t` | ~2x wall-clock reduction |
| `T_t >> T_p` | `C*T_p + C*T_t` | `T_p + C*T_t` | Minimal (transfer-dominated) |

For typical configurations with 100+ Gbps NIC and modern GPUs, transfer bandwidth often exceeds compute throughput for attention-heavy long-context prefill (`T_t < T_p` regime), where transfer can be substantially hidden.

### Memory Pressure Interaction

Pipelined transfer compounds with early block release:

    Without pipelining:  Prefill all -> Xfer chunk 0 -> Release 0 -> ...
    With pipelining:     Prefill 0 -> (Xfer 0 || Prefill 1) -> Release 0 -> ...

Peak memory is ~2 chunks (one prefilling, one transferring) instead of C chunks.

### Overhead

- **CUDA event recording:** < 1 microsecond per event
- **Worker `event.synchronize()`:** Blocks worker until GPU finishes chunk N. In the `T_t < T_p` regime, the GPU forward is likely done by the time the worker dequeues the task.
- **Per-chunk session bookkeeping:** Same as Phase 1

### Limitations

1. **Context-first scheduling.** Receiver not ready until orchestrator routes gen request after full prefill.
2. **Single-chunk prefill.** Nothing to pipeline when entire prompt fits in one chunk.
3. **GPU memory contention.** GPUDirect RDMA reads share PCIe bandwidth with GPU memory accesses. Attention is compute-bound for long contexts, so this is unlikely to be significant.


## Further Discussion

### Open Concerns

1. **GPUDirect RDMA memory ordering guarantees.** CUDA event synchronization should suffice, but the exact interaction between CUDA stream completion and GPUDirect RDMA visibility needs validation on target hardware (H100/B200 with ConnectX-7). A functional correctness test comparing transferred KV data with and without pipelining is essential.

2. **Interaction with speculative decoding.** If the context server uses speculative decoding with chunked prefill, draft/target token handling adds complexity to which blocks are "finalized" per chunk.

3. **Multi-GPU considerations.** With tensor parallelism, each GPU prefills its shard and transfers independently — pipelining should work per-GPU. With pipeline parallelism, chunk completion timing varies by PP stage, requiring stage-aware transfer initiation.

4. **Request cancellation mid-prefill.** If a request is cancelled after chunk 0's transfer starts but before chunk 1's prefill, the in-flight RDMA and session need clean cancellation. The existing fail-fast semantics (`TxSession.set_exception`) should handle this, but needs testing.

### Future Opportunities

1. **Orchestrator early routing.** For context-first mode, send a "prepare" signal to the gen server after the first chunk so the receiver allocates blocks early.
2. **Adaptive pipelining.** Only enable pipelining when `num_chunks > threshold` (e.g., >= 3). For 2-chunk prefill, the overhead may not justify the small overlap window.
3. **Cross-request pipelining.** Transfer completed request A's KV while request B is still prefilling. This is a simpler form of overlap that doesn't require intra-request pipelining and may capture most of the benefit in high-concurrency workloads. Could be a lower-hanging fruit than intra-request pipelining.
4. **Receiver-side incremental allocation.** Enable the generation server to start decode on partial KV data (speculative prefix decode). Requires significant changes to the attention kernel and scheduler.
5. **C++ transceiver support.** The pipelining logic is Python-transceiver only (`transceiver_runtime="PYTHON"`). Extending to the C++ transceiver would require mirroring the chunking and event logic in C++.
