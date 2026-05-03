# 01 — Background and Architecture

This file is the orientation read for anyone new to the disaggregated KV
cache transfer code path. It walks through:

1. The deployment topology
2. The `LlmRequestState` lifecycle (the disagg-relevant subset)
3. The end-to-end happy path
4. The C++ transceiver internals
5. The cancellation flow and where it can break
6. The Python ↔ C++ ↔ NIXL/UCX boundary
7. A quick reference of files / classes / state holders

Defaults assumed throughout: PyTorch backend, v1 KV cache manager, C++
transceiver (NIXL/UCX backend). All file paths are relative to the
TensorRT-LLM repo root.

---

## 1. Deployment topology

```mermaid
graph LR
    Client[HTTP client] -->|POST /v1/chat/completions| Router

    subgraph Disagg["trtllm-serve disaggregated (port 8000)"]
        Router["openai_disagg_server.py<br/>OpenAIDisaggServer"]
    end

    subgraph CtxNode["Context worker (port 8001, GPU 0)"]
        CtxFE["openai_server.py<br/>OpenAIServer<br/>--server_role context"]
        CtxLLM[LLM API]
        CtxExec["GenerationExecutor<br/>+ PyExecutor (event loop)"]
        CtxKVMgr[v1 KVCacheManager]
        CtxTrans["CacheTransceiver C++<br/>(CacheSender::Impl)"]
        CtxNIXL[NixlTransferAgent]
    end

    subgraph GenNode["Generation worker (port 8002, GPU 1)"]
        GenFE["openai_server.py<br/>OpenAIServer<br/>--server_role generation"]
        GenLLM[LLM API]
        GenExec["GenerationExecutor<br/>+ PyExecutor (event loop)"]
        GenKVMgr[v1 KVCacheManager]
        GenTrans["CacheTransceiver C++<br/>(CacheReceiver::Impl)"]
        GenNIXL[NixlTransferAgent]
    end

    Router -->|"1. POST ctx (prefill+register)"| CtxFE
    CtxFE --> CtxLLM --> CtxExec --> CtxTrans
    CtxExec <--> CtxKVMgr
    CtxTrans <--> CtxNIXL

    Router -->|"2. POST gen (with disagg_request_id)"| GenFE
    GenFE --> GenLLM --> GenExec --> GenTrans
    GenExec <--> GenKVMgr
    GenTrans <--> GenNIXL

    CtxNIXL <-.->|"NIXL UCX KV transfer<br/>(UCX_TLS=tcp,cuda_copy,self)"| GenNIXL
```

The router holds the request; the **context worker** runs the prefill and
publishes its KV cache; the **generation worker** receives that KV cache and
generates tokens. KV transfer happens *worker-to-worker via NIXL/UCX*, not
through the router. The request ID linking the two is `disagg_request_id`.

---

## 2. `LlmRequestState` lifecycle (the disagg-relevant subset)

State enum lives in `cpp/include/tensorrt_llm/batch_manager/llmRequest.h`.
Numeric values shown for the values that appear in trace logs.

```mermaid
stateDiagram-v2
    [*] --> kCONTEXT_INIT: ctx worker enqueue
    kCONTEXT_INIT --> kCONTEXT_IN_PROGRESS: scheduled for prefill
    kCONTEXT_IN_PROGRESS --> kDISAGG_CONTEXT_TRANS_IN_PROGRESS: prefill done,<br/>respondAndSendAsync()
    note right of kDISAGG_CONTEXT_TRANS_IN_PROGRESS
        kDISAGG_CONTEXT_TRANS_IN_PROGRESS = 21
        Sender enqueued in mReadyResponses,
        future pushed to mSenderFutures.
        KV blocks pinned via store_blocks_for_reuse.
    end note
    kDISAGG_CONTEXT_TRANS_IN_PROGRESS --> kDISAGG_CONTEXT_COMPLETE: future ready,<br/>peer ack received
    kDISAGG_CONTEXT_COMPLETE --> [*]: ctx worker terminates request
    kDISAGG_CONTEXT_TRANS_IN_PROGRESS --> kDISAGG_TRANS_ERROR: timeout / exception /<br/>cancel-after-ready
    note right of kDISAGG_TRANS_ERROR
        kDISAGG_TRANS_ERROR = -1
        Surface for sigs #1, #5, #7
    end note

    [*] --> kDISAGG_GENERATION_INIT: gen worker enqueue<br/>(carrying disagg_request_id)
    kDISAGG_GENERATION_INIT --> kDISAGG_GENERATION_TRANS_IN_PROGRESS: requestAndReceiveAsync()
    note left of kDISAGG_GENERATION_TRANS_IN_PROGRESS
        kDISAGG_GENERATION_TRANS_IN_PROGRESS = 9
        RequestAndPromise queued in
        CacheReceiver::Impl::mRequestsQueue;
        future pushed to mRequesterFutures.
    end note
    kDISAGG_GENERATION_TRANS_IN_PROGRESS --> kDISAGG_GENERATION_TRANS_COMPLETE: receiveSync done,<br/>future resolved
    kDISAGG_GENERATION_TRANS_COMPLETE --> kGENERATION_IN_PROGRESS: enter normal decode
    kGENERATION_IN_PROGRESS --> [*]: tokens streamed to client
    kDISAGG_GENERATION_TRANS_IN_PROGRESS --> kDISAGG_TRANS_ERROR: timeout / Broken promise /<br/>cancel
```

The `kDISAGG_CONTEXT_TRANS_IN_PROGRESS` and
`kDISAGG_GENERATION_TRANS_IN_PROGRESS` states are exactly the windows in
which signatures `#1`, `#5`, and `#6` fire. Both end in either
`kDISAGG_*_TRANS_COMPLETE` or `kDISAGG_TRANS_ERROR`.

---

## 3. End-to-end happy path

This is the full request flow with no errors and no cancellation. Steps
are numbered to make the cross-references in section 5 easier to follow.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Router as Disagg Router<br/>(openai_disagg_server.py)
    participant CtxFE as Ctx OpenAIServer
    participant CtxPy as Ctx PyExecutor<br/>(py_executor.py)
    participant CtxCpp as Ctx CacheSender::Impl<br/>(dataTransceiver.cpp)
    participant CtxNixl as Ctx NixlTransferAgent
    participant GenNixl as Gen NixlTransferAgent
    participant GenCpp as Gen CacheReceiver::Impl<br/>(dataTransceiver.cpp)
    participant GenPy as Gen PyExecutor
    participant GenFE as Gen OpenAIServer

    Client->>Router: POST /v1/chat/completions
    Router->>CtxFE: POST (ctx phase, disagg_request_id=R)
    CtxFE->>CtxPy: enqueue_requests (kCONTEXT_INIT)
    CtxPy->>CtxPy: schedule, run prefill forward
    Note over CtxPy: KV blocks populated in v1 KVCacheManager
    CtxPy->>CtxPy: setState(kDISAGG_CONTEXT_TRANS_IN_PROGRESS)
    CtxPy->>CtxCpp: respond_and_send_async(R)<br/>→ CacheSender::sendAsync(R)
    CtxCpp->>CtxCpp: emplace into mReadyResponses[R]<br/>= Response{ &llmReq, promise }
    CtxCpp-->>CtxPy: future
    CtxPy->>CtxPy: mSenderFutures.emplace_back(R, future)
    CtxFE-->>Router: 200 OK (ctx done)

    Router->>GenFE: POST (gen phase, disagg_request_id=R)
    GenFE->>GenPy: enqueue_requests (kDISAGG_GENERATION_INIT)
    GenPy->>GenPy: setState(kDISAGG_GENERATION_TRANS_IN_PROGRESS)
    GenPy->>GenCpp: request_and_receive_async(R)
    GenCpp->>GenCpp: queue RequestAndPromise<br/>onto mRequestsQueue
    GenCpp-->>GenPy: future
    GenPy->>GenPy: mRequesterFutures.emplace_back(R, future)

    Note over GenCpp,CtxCpp: --- KV transfer dance over NIXL ---

    GenCpp->>GenCpp: requestSync() picked up by worker
    GenCpp->>GenCpp: assignBufferIndexForRecv()<br/>(BaseTransBufferManager pool)
    GenCpp->>GenNixl: sendRequestInfo(R)<br/>→ AgentConnection::sendRequestAndBufferInfo
    GenNixl->>CtxNixl: notifySyncMessage (request info)
    CtxNixl->>CtxCpp: recvRequestInfo() returns RequestInfo{R}
    CtxCpp->>CtxCpp: response() worker picks Response from<br/>mReadyResponses[R], sets mCurrentRequest
    CtxCpp->>CtxNixl: sendSync (data transfer)
    CtxNixl->>GenNixl: data + ready notify (UCX RMA)
    GenCpp->>GenCpp: receiveReadySignal observes is_ready=true
    GenCpp->>GenCpp: receiveSync() → unformat() →<br/>freeBufferIndexForRecv()
    GenCpp->>GenCpp: promise.set_value() on receiver future
    CtxCpp->>CtxCpp: promise.set_value() on sender future

    Note over GenPy: Next poll cycle
    GenPy->>GenCpp: check_gen_transfer_status(at_least_num=1)
    GenCpp-->>GenPy: future ready, get() returns
    GenPy->>GenPy: setState(kDISAGG_GENERATION_TRANS_COMPLETE)<br/>→ kGENERATION_IN_PROGRESS

    Note over CtxPy: Next poll cycle
    CtxPy->>CtxCpp: check_context_transfer_status(0)
    CtxCpp-->>CtxPy: future ready
    CtxPy->>CtxPy: setState(kDISAGG_CONTEXT_COMPLETE)<br/>terminate_request, free KV blocks

    GenPy->>GenPy: decode loop, sample tokens
    GenPy-->>GenFE: token stream
    GenFE-->>Router: SSE stream
    Router-->>Client: response stream
```

Key participants by responsibility:

- **`PyExecutor`** owns the per-iteration loop. Calls
  `respond_and_send_async()` on ctx side and `request_and_receive_async()`
  on gen side, then polls `check_*_transfer_status()` until done.
- **`CacheTransceiver`** is the C++ object the Python side talks to via
  nanobind. It holds `mSenderFutures` (ctx) or `mRequesterFutures` (gen)
  — the per-request `(LlmRequest*, future)` pairs that drive the polls.
- **`CacheSender::Impl` / `CacheReceiver::Impl`** own the actual worker
  threads, queues (`mReadyResponses`, `mRequestsQueue`), and per-transfer
  bookkeeping (`mCurrentRequest`).
- **`NixlTransferAgent`** wraps the NIXL agent and is the boundary into
  the UCX/TCP transport.

---

## 4. C++ transceiver internals

This zooms into the dotted "KV transfer dance" of the previous diagram.
It shows the worker threads, the queues they consume, and the
buffer-pool manager.

```mermaid
graph TB
    subgraph CtxC["Context (sender) C++ side"]
        CtxFut["mSenderFutures<br/>vector&lt;LlmRequest*, future&gt;"]
        CtxRR["mReadyResponses<br/>map&lt;RequestId, Response&gt;"]
        CtxResp["response() worker thread<br/>(picks ready, sends)"]
        CtxCur["mCurrentRequest<br/>(in-flight slot)"]
        CtxBuf["BaseTransBufferManager<br/>send-buffer pool<br/>(default size 1)"]
        CtxSendQ["mSendQueue<br/>(staged sends)"]
    end

    subgraph GenC["Generation (receiver) C++ side"]
        GenFut["mRequesterFutures<br/>vector&lt;LlmRequest*, future&gt;"]
        GenQ["mRequestsQueue<br/>deque&lt;RequestAndPromise&gt;"]
        GenReq["request() worker thread<br/>(drains queue, calls requestSync)"]
        GenBuf["BaseTransBufferManager<br/>recv-buffer pool<br/>(default size 1)"]
        GenSession[TransferSession<br/>per in-flight transfer]
    end

    PyCtx[Python: respond_and_send_async] --> CtxFut
    PyCtx --> CtxRR
    CtxResp --> CtxRR
    CtxResp --> CtxCur
    CtxResp -->|sendSync uses<br/>buffer slot| CtxBuf
    CtxResp -->|format → AgentConnection::send| Nixl

    PyGen[Python: request_and_receive_async] --> GenFut
    PyGen --> GenQ
    GenReq --> GenQ
    GenReq -->|assignBufferIndexForRecv<br/>cv.wait if pool full| GenBuf
    GenReq -->|sendRequestInfo<br/>→ AgentConnection::send| Nixl
    GenReq --> GenSession
    GenSession -->|receiveSync → unformat<br/>→ freeBufferIndexForRecv| GenBuf

    Nixl[("NIXL UCX plugin<br/>libplugin_UCX.so<br/>nixlUcxThreadEngine")]

    CtxResp -.->|on completion<br/>promise.set_value| CtxFut
    GenReq -.->|on completion<br/>promise.set_value| GenFut

    classDef leak fill:#fee,stroke:#c00
    class GenBuf leak
```

Important things that are easy to miss but matter for the bug class:

- **Both buffer pools default to size 1** when
  `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM=1` (the customer config). Only
  one transfer can be in flight at a time. A single leaked slot wedges
  every subsequent call to `assignBufferIndex*` on the unbounded
  `cv.wait`. That is signature `#6`.
- **`mReadyResponses` and `mRequestsQueue` carry raw `LlmRequest*`** in
  `rc11`. PR `#13056` changes these to `shared_ptr<LlmRequest>`; PR
  `#13495` inherits the same change from `#13439`. This is the load-bearing
  UAF closure for the cleanup paths.
- **The receiver's request thread calls `requestSync()` which internally
  calls `assignBufferIndexForRecv()` *first*** (to reserve the slot for
  the eventual incoming data), then `sendRequestInfo()` over NIXL, then
  `receiveReadySignal()`, then `receiveSync()` which calls `unformat()`
  which calls `freeBufferIndexForRecv()`. Any early-return between those
  leaks the slot — that's the sig `#6` mechanism.
- **The sender's worker calls `recvRequestInfo()` to receive the gen-side
  request info**, which in turn calls `nixlAgent::getNotifs()`. The
  `pthread_mutex_lock` deadlock variants of sig `#7` surface in this
  call chain.

---

## 5. Cancellation flow + where it can break

This is the single most useful diagram for understanding NVBug 6104831,
because every signature except `#2` fires somewhere on this path.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Router
    participant GenPy as Gen PyExecutor
    participant GenCpp as Gen CacheReceiver::Impl
    participant GenNixl as Gen NIXL
    participant CtxNixl as Ctx NIXL
    participant CtxCpp as Ctx CacheSender::Impl
    participant CtxPy as Ctx PyExecutor

    Client->>Router: client closes connection<br/>(or kv_transfer_timeout fires)
    Router->>GenPy: cancel disagg_request_id=R
    Router->>CtxPy: cancel disagg_request_id=R

    GenPy->>GenCpp: cancel_request(R)<br/>→ CacheReceiver::cancelRequest(R)

    alt R is still queued in mRequestsQueue
        Note over GenCpp: Receiver worker hasn't dequeued yet.<br/>cancelRequest() erases entry.
        GenCpp->>GenCpp: erase RequestAndPromise from mRequestsQueue
        Note over GenCpp,GenPy: Sig #5: rc11 erases without<br/>fulfilling promise. promise dtor<br/>= std::future_error: Broken promise.<br/>Fix #13672: set_exception(kNETWORK_ERROR).
    else R is in flight, owned by request() worker
        Note over GenCpp: rc11: returns false, logs<br/>"Cannot cancel request".<br/>PR #13056: per-request cancel-flag<br/>flipped, worker observes & throws.
        GenCpp-->>GenPy: false
        GenPy->>GenPy: retry next iter, eventually<br/>marks DISAGG_TRANS_ERROR
    end

    CtxPy->>CtxCpp: cancel_request(R)<br/>→ CacheSender::cancelRequest(R)

    alt R is in mReadyResponses (waiting to send) and ready-signal sent
        Note over CtxCpp: Cancel-after-ready window.<br/>sendResponse() takes else branch,<br/>erases entry without fulfilling promise.
        CtxCpp->>CtxCpp: erase from mReadyResponses
        Note over CtxCpp,CtxPy: Sig #1: rc11 = Broken promise<br/>on consumer's future.get().<br/>Fix #13640 / #13495:<br/>set_exception(RequestSpecificException).
    else R is mCurrentRequest (sendSync in flight)
        Note over CtxCpp: rc11: returns false,<br/>"Cannot cancel request".<br/>PR #13056: cancel-flag set,<br/>AgentConnection::send poll loop<br/>checks flag, throws on next slice.<br/>PR #13495: TransferStatus::release()<br/>→ nixlAgent::releaseXferReq().
        CtxCpp-->>CtxPy: false
    end

    Note over GenCpp,GenNixl: --- The post-cancel leak path (sig #6) ---
    Note over GenCpp: After sig #1 fix lands on ctx side,<br/>ctx sends is_ready=false on cancel.<br/>Gen receives !isReady, requestSync<br/>takes early-return — but in rc11<br/>that path skips receiveSync()<br/>→ unformat() → freeBufferIndexForRecv().<br/>BUFFER SLOT LEAKED.

    Note over GenPy: Next gen request:
    GenPy->>GenCpp: request_and_receive_async(R')
    GenCpp->>GenCpp: requestSync() →<br/>assignBufferIndexForRecv()
    Note over GenCpp: With pool size 1 and slot leaked,<br/>cv.wait blocks forever.<br/>Sig #6. Fix #13673: try/catch<br/>release on early return.

    Note over CtxCpp,CtxNixl: --- The terminal sender wedge (sig #7) ---
    CtxCpp->>CtxNixl: recvRequestInfo() →<br/>nixlAgent::getNotifs()
    CtxNixl->>CtxNixl: pthread_mutex_lock<br/>inside CacheSender::Impl::*
    Note over CtxNixl: STUCK — sender lifecycle / contention class.<br/>Sig #7. Manifests as deadlock, SIGSEGV,<br/>or mpi4py executor exit depending on<br/>fix stack. Phase 12 falsified the<br/>"NIXL-only" classification.

    Note over GenPy: Sig #4 self-block:
    GenPy->>GenPy: _check_disagg_gen_cache_transfer_status(at_least_num=1)
    GenPy->>GenCpp: check_gen_transfer_status(1)
    Note over GenCpp: rc11 path picks unready entry<br/>by insertion order, calls<br/>future.get() unconditionally.<br/>Blocks forever.<br/>Fix #13671: wait_for(0) skip.<br/>#13056 deadline-hoist also evicts.
```

---

## 6. The Python ↔ C++ ↔ NIXL/UCX boundary

Three layers, each with its own thread / queue / future model:

| Layer | Lives in | Concurrency model | Ownership of `LlmRequest` |
|---|---|---|---|
| Python | `tensorrt_llm/_torch/pyexecutor/py_executor.py` and `tensorrt_llm/_torch/disaggregation/transceiver.py` | One main event-loop thread per worker; cooperative async tasks for HTTP. | `shared_ptr` exposed via nanobind; `_terminate_request` decides when the underlying `LlmRequest` is freed. |
| TRT-LLM C++ transceiver | `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp`, `dataTransceiver.cpp`, `baseTransBuffer.cpp` | One sender worker thread (`response()`), one receiver worker thread (`request()`), per-async-send drain worker (`handleAsyncSend`), buffer-pool `cv.wait`. | Pre-`#13056`/`#13439`: raw `LlmRequest*` in queues and futures. Post-fix: `shared_ptr<LlmRequest>`. |
| NIXL UCX plugin | `libplugin_UCX.so`, `nixlUcxThreadEngine`, `nixlAgent` | Internal worker threads: `nixl-comm-worker`, `nixl-ucx-shared`. Notification waits, transfer-state polls. | Doesn't know about `LlmRequest`; tracks `nixlXferReqHandle` opaque handles. |

The cleanup-path bugs span all three layers, which is why the fix
strategies disagree on where the right cancellation primitive lives:

- **PR `#13056`'s answer:** in TRT-LLM C++, via a per-request cancel-flag
  registry plumbed through `sendRequestInfo` / `receiveReadySignal` /
  `AgentConnection::send`'s polling loop.
- **PR `#13495`'s answer:** at the NIXL boundary, via a
  `TransferStatus::release()` hook that calls `nixlAgent::releaseXferReq()`
  to drop the backend handle.
- **The combo answer (Approach D):** both. They address different parts of
  the cancellation lifecycle and are not redundant — see
  [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md).

---

## 7. Quick-reference table

### Files

| Path | Role |
|---|---|
| `tensorrt_llm/serve/openai_disagg_server.py` | Disagg router HTTP layer. |
| `tensorrt_llm/serve/openai_server.py` | Per-role server (`--server_role context` or `--server_role generation`). |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py` | The PyTorch backend's per-worker event loop (`PyExecutor`). |
| `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py` | Python wrapper over the C++ `CacheTransceiver`. |
| `tensorrt_llm/_torch/disaggregation/transceiver.py` | The disagg-specific Python transceiver shim. |
| `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` | Top-level C++ transceiver: `mSenderFutures`, `mRequesterFutures`, `checkContextTransferStatus`, `checkGenTransferStatus`. |
| `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` | `CacheSender::Impl` and `CacheReceiver::Impl`: worker threads, queues, async-send. |
| `cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp` | `BaseTransBufferManager` (the buffer-index pool). |
| `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`, `mlaCacheFormatter.cpp`, `rnnCacheFormatter.cpp` | Format / unformat KV blocks for transfer. |
| `cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp` | `AgentConnection` (NIXL-level send / recv). |
| `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp` | `NixlTransferAgent`, `NixlTransferStatus`. |

### State holders that matter for the bug class

| Holder | Owner | Lifetime contract (rc11 baseline) |
|---|---|---|
| `mSenderFutures` | `CacheTransceiver` (ctx) | `vector<pair<LlmRequest*, future<void>>>`; entry erased on completion / timeout / exception. Raw pointer makes UAF possible until shared_ptr lands. |
| `mRequesterFutures` | `CacheTransceiver` (gen) | Symmetric to `mSenderFutures` on the gen side. |
| `mReadyResponses` | `CacheSender::Impl` | `map<RequestId, Response{LlmRequest*, promise<void>}>`; populated by `sendAsync`, drained by `response()` worker. Sig `#1` erase site. |
| `mRequestsQueue` | `CacheReceiver::Impl` | `deque<RequestAndPromise>`; populated by `request_and_receive_async`, drained by `request()` worker. Sig `#5` erase site. |
| `mCurrentRequest` | `CacheSender::Impl` | Single in-flight slot. `cancelRequest` returns false on this; sig `#1` / `#7` escape window. |
| `mAsyncSendResource.mSendQueue` | `CacheSender::Impl` | `deque<Response>` consumed by `handleAsyncSend`; this is where the eval-order regression lives once `Response::mRequest` becomes `shared_ptr`. |
| `BaseTransBufferManager` recv pool | gen-side | Default size 1. `assignBufferIndex` / `freeBufferIndex` pairing must hold on every exit path. Sig `#6` leak surface. |
| `BaseTransBufferManager` send pool | ctx-side | Symmetric on ctx side. Both `#13056` and `#13495` add RAII coverage. |

### Trace markers for cross-reference

When reading the run logs in the investigation archives, these are the
markers gated by `TRTLLM_DISAGG_TRACE_PROMISE=1` /
`TRTLLM_DISAGG_TRACE_BLOCK=1` and what they mean:

- `gen_request_enqueue` / `gen_request_dequeue` — queue movement on
  receiver side (`mRequestsQueue`).
- `gen_request_sync_begin` / `gen_request_sync_end` — `requestSync()`
  entry/exit on the receiver worker.
- `gen_send_assign_buffer_begin` / `_step` / `_end` — receiver-side
  `assignBufferIndexForRecv` lifecycle. Stalls here = sig `#6`.
- `gen_send_request_info_begin` / `_end` — `sendRequestInfo()` over NIXL
  on the receiver side.
- `gen_wait_ready_signal_begin` / `_end` — receiver waiting for ctx's
  ready notification.
- `gen_receive_sync_begin` / `_end` — `receiveSync()` (the actual data
  receive) on the receiver.
- `gen_request_promise_set_value` / `_set_exception` — receiver-side
  promise fulfillment.
- `gen_request_sync_not_ready_buffers_freed` — sig `#6` fix path firing
  on the `!isReady` early-return; `count=N` shows how many slots got
  released.
- `gen_future_skip_unready` — sig `#4` fix firing in `checkGenTransferStatus`.
- `gen_future_get_ok` / `gen_future_get_exception` — final outcome of the
  receiver-side future.

The corresponding ctx-side markers are `create`, `send_response_ready`,
`send_sync_begin`, `promise_set_value`, `future_get_ok`,
`mark_cancelled`, `cancel_rejected`, `promise_set_exception`,
`future_get_exception`. The async-send instrumentation added in Phase 14
adds: `enter_sendAsync`, `enqueue_ready`, `enter_sendResponse`,
`producer_move`, `enqueue_send`, `consumer_wake`, `consumer_dequeue`,
`preDeref`, `sendAndRemove_*`, `sendSync_*`.

---

## What to read next

- For a list of the seven discrete bugs that fire on this code path, see
  [`02-failure-signatures.md`](02-failure-signatures.md).
- For the *defect-class layering* that explains why the four candidate
  fixes have different coverage, see
  [`03-defect-class-stack.md`](03-defect-class-stack.md).
- For the actual reproducer, see [`04-reproduction.md`](04-reproduction.md).
