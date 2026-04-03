# Background and Root Cause Analysis

## Disaggregated Serving Architecture

TensorRT-LLM supports [disaggregated serving](https://arxiv.org/pdf/2506.05508) where the prefill (context) and decode (generation) phases run on separate GPU pools. The CTX server computes KV cache for prompt tokens and transfers it to the GEN server via RDMA/NVLink (using NIXL, UCX, or MPI backends). The GEN server then generates tokens using the received KV cache.

```
User Request → CTX Server (prefill) → KV Transfer → GEN Server (decode) → Tokens
```

## Benchmark Disagg Mode

For performance measurement, a benchmark mode pre-loads a fixed number of requests (`benchmark_req_queues_size`, set via `TLLM_BENCHMARK_REQ_QUEUES_SIZE`) into the GEN executor before starting the forward pass. This ensures consistent batch sizes across measurements. The mode is activated when both `benchmark_req_queues_size > 0` and `kv_cache_transceiver is not None`.

## Prior Art: PR #12091 — Batched Fill Loop

[PR #12091](https://github.com/NVIDIA/TensorRT-LLM/pull/12091) (by @Tabrizian, merged March 11, 2026) was the first attempt to fix the CTX-side KV cache pressure deadlock. It changed the fill loop from fetching **all** `benchmark_req_queues_size` requests in a single blocking loop to fetching in smaller batches of `batch_size` per invocation of `_prepare_and_schedule_batch`:

```python
# PR #12091: batch_size = tp_size (ADP) or 1 (non-ADP)
fill_target = min(num_fetch_requests + batch_size, benchmark_req_queues_size)
while self.num_fetch_requests < fill_target:
    iter_requests = self._fetch_and_activate_new_requests()
    ...
    time.sleep(1)
```

This reduced the severity of the deadlock but **did not eliminate it**. The fill loop remained inside `_prepare_and_schedule_batch` and still blocked for up to `batch_size` requests. During that blocking wait, KV transfer processing (`_check_disagg_gen_transfer_status`) did not run — it only ran once before the fill loop, not during it.

## Prior Art: PR #12206 — Insufficient GEN-Side KV Cache

[PR #12206](https://github.com/NVIDIA/TensorRT-LLM/pull/12206) (by @Tabrizian, merged March 20, 2026) addressed a hang when the **GEN server's KV cache** is too small to hold all benchmark requests at once. In that scenario, some requests remain permanently in `DISAGG_GENERATION_INIT` state because the scheduler cannot allocate KV blocks for them while existing generation requests hold theirs indefinitely. PR #12206 detects this condition and returns an explicit error to all clients instead of hanging silently. This fix handles the GEN-side capacity failure cleanly, but does **not** address the CTX-side pipeline starvation.

## The Root Cause: Structural Starvation

The deadlock is **structural, not timing-based**. The code flow in `_prepare_and_schedule_batch` is:

```
_prepare_and_schedule_batch():
  Line 1: _check_disagg_gen_transfer_status()    ← runs ONCE
  Line 2: ── fill loop begins ──
  Line 3:   while num_fetch_requests < fill_target:
  Line 4:     _fetch_and_activate_new_requests()  ← only this runs in the loop
  Line 5:     time.sleep(X)                        ← X=10, 1, 0.1, or even 0
```

`_check_disagg_gen_transfer_status()` runs at line 1, **before** the fill loop. It is **never called inside the loop**. This is the function that processes completed KV transfers — transitioning requests from `TRANSMISSION_IN_PROGRESS` to `TRANSMISSION_COMPLETE`.

**Even `time.sleep(0)` causes the same deadlock** — the loop just spins faster while still never processing transfers.

## When the Deadlock Occurs (Post-#12091)

When CTX capacity is smaller than `batch_size` (e.g., `tp_size=32` but CTX can only hold 16 requests in KV cache), the fill loop waits for 32 requests while the CTX server is blocked:

```
GEN: fill loop waiting for 32      CTX: KV cache full (16 sent)
     requests, only has 16               needs GEN to process
     ↓                                   transfers to free blocks
     sleep(1)... fetch... still 16       ↑
     sleep(1)... fetch... still 16  ←────┘ deadlock
```

## Backend-Dependent Severity

The practical impact depends on how the transport backend handles send completion:

| Backend | CTX blocks on GEN? | Deadlock severity |
|---------|-------------------|-------------------|
| **MPI** | Yes — sends can block until receiver posts matching receive | **Hard deadlock** — no progress possible |
| **RDMA (NIXL/UCX)** | No — CTX frees blocks based on send completion independently | **Executor starvation** — no deadlock, but no transfer processing, timeout handling, or error detection during fill |

PR #12091 worked for the tested config (DeepSeek-R1 TP32 with large CTX KV cache) because CTX capacity exceeded `tp_size`, so the batch-size requests were always available. But this is configuration-dependent — the fix fails when `CTX capacity < tp_size`.

## Comparison: PR #12091 vs This PR (#12208)

| Aspect | PR #12091 | PR #12208 (this PR) |
|--------|-----------|---------------------|
| Fill loop | Still blocks inside `_prepare_and_schedule_batch` for `batch_size` requests | **Eliminated entirely** — single fetch per call, fully non-blocking |
| Transfer servicing during fill | Only before the loop, not during | Every executor loop iteration services transfers |
| Deadlock-free guarantee | Only when CTX capacity >= `batch_size` | **For all configurations** — no blocking, transfers always serviced |
| Gate in `_executor_loop` | Not added | Added to both `_executor_loop` and `_executor_loop_overlap` |
| ADP dummy counting | Counted all gen requests (including dummies) | Excludes `is_attention_dp_dummy` |
| Gate retry sleep | 10 seconds (in overlap loop) | 1 second |
| Unit tests | None | 44 tests |

This PR supersedes PR #12091's approach by eliminating the fill loop entirely. PR #12206's stuck-request detection remains intact and runs inside `_prepare_and_schedule_batch`, before the gate.
