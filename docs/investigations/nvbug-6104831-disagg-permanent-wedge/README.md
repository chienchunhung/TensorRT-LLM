# NVBug 6104831: Permanent Disaggregated-Serving Wedge in `rc11`

- **Severity:** P0 / Critical
- **Affected component:** Disaggregated serving (`trtllm-serve` context worker + generation worker + disaggregated front-end), `rc11` baseline
- **Affected backend:** PyTorch executor, NIXL/UCX KV-cache transceiver
- **Symptom (customer-facing):** Local 1P1D `trtllm-serve` deployment serves
  the first burst of requests, then stops responding. All probes after the
  burst hit `ReadTimeout`. Workers stay alive (no crash, no exit), but the
  generation event loop never recovers.
- **Origin signal:** Dynamo + TRT-LLM `rc11` deployment hang, three apparent
  crash signatures observed in the field.
- **Branches in this worktree:**
  - `local/sig1-broken-promise-test` (signature #1 reproducer test)
  - `local/sig1-broken-promise-fix` (signature #1 fix)
  - `local/rc11-disagg-repro` (isolated `rc11` worktree with cumulative fixes
    + instrumentation)
- **Related PRs:**
  - [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) — signature #2
    reproducer test
  - [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) — signature #2
    fix
  - [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) — signature #1
    reproducer test
  - [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) — signature #1
    fix
- **Companion fixes in main (not in `rc11`):**
  - [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) — request-level
    error propagation (cleaner failure visibility, not a fix for the wedge)
  - [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — fatal engine
    detection / pod restart (mitigation for silent wedges, not a fix for the
    wedge)
- **Status:** Investigation in progress. Signatures #1, #2, #4, and #5 have
  fixes in flight or merged. Signature #6 has now been **root-caused** to a
  recv-buffer index leak in `BaseTransBufferManager::assignBufferIndex()`
  triggered by the `!isReady` early-return path in
  `CacheReceiver::Impl::requestSync()`; the targeted fix is built and an
  end-to-end validation run is currently in flight.

---

## Executive Summary

Disaggregated serving in `rc11` collapses into a permanent wedge after a
burst of long-prompt requests with retries and cancellations. The system
keeps every process alive — no crash, no `/health` failure, no orchestrator
failover — yet every subsequent request either times out or returns a generic
`400 Bad Request`. From the outside it looks like a single wedge; from the
C++ transceiver it is **six distinct, partially overlapping TRT-LLM bugs**
plus **a seventh signature that lives one architectural layer below TRT-LLM**
in the NIXL/UCX transfer plugin.

The investigation peeled them off one at a time. Each fix exposed the next
layer of failure, which is why the bug count grew over time rather than
shrank. The pattern below is consistent across signatures:

- **One disaggregated request is cancelled or times out mid-flight.**
- A `std::promise` / `std::future` pair on either the sender or the receiver
  side is mishandled.
- The error either becomes a `Broken promise` exception, or the gen event
  loop self-blocks on an unresolved future, or a downstream invariant
  (KV-block trie, decode-side `std::optional`) is violated.
- Because each of these is in the C++ transceiver path, neither the Python
  health checks ([#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718))
  nor the request-level error propagation
  ([#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119)) flips the
  pod or the request to "failed". The pod stays *healthy*, the request
  stays *in flight*, and the deployment stays *wedged*.

Throughout the report we use these signature labels, which match the
labelling we used in the chat session that produced this investigation:

| Signature | Short name | Where it lives | First-found via |
|---|---|---|---|
| **#1** | Sender-side `Broken promise` after ready signal | `CacheSender::Impl::sendResponse` (cancelled-after-ready path) | Production logs (Dynamo `rc11` deploy) |
| **#2** | Trie `cascade prune: parent did not find this node as a child` assertion | `templatedTrie.h::clearNode` / `KVCacheBlock` lifecycle | Field investigation report + C++ unit-test stress probe |
| **#3** | Decode-side `RuntimeError: bad optional access` | C++ `std::optional::value()` inside disagg gen path, surfaced through Python | Production logs (Dynamo `rc11` deploy) |
| **#4** | Gen-side blocking hang in `CacheTransceiver::checkGenTransferStatus()` with `atLeastNum=1` | `cacheTransceiver.cpp` unconditional `future.get()` on selected-but-unready future | Local `trtllm-serve` 1P1D repro + Python thread-stack dump |
| **#5** | Receiver-side `Broken promise` from queued cancel | `CacheReceiver::Impl::cancelRequest()` erasing queued request without fulfilling promise | Post-`#4`-fix C++ trace logs |
| **#6** | Recv-buffer index leak via `!isReady` early-return; subsequent receives block forever in `BaseTransBufferManager::assignBufferIndex()` | `cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp` (unbounded `cv.wait`) leaked from `CacheReceiver::Impl::requestSync()` (`!isReady` path) | Fine-grained C++ instrumentation across `sendRequestInfo()` body in `run7` |
| **#7** | NIXL UCX-internal `pthread_mutex_lock` deadlock surfacing as a wedged ctx-side `dataTransResp` thread + stranded gen-side receiver futures | `libplugin_UCX.so` (`nixlUcxThreadEngine::getNotifs()`); surfaces in TRT-LLM via `CacheSender::Impl::recvRequestInfo()` | `py-spy` + `gdb` post-mortem of the `run8` validation experiment |

The mapping of fixes is summarised at the end, in
[Signature ↔ PR Map](#signature--pr-map).

> **Read this caveat before reading anything else.** Signatures `#1`
> through `#6` are real TRT-LLM bugs and the chained PRs land their
> fixes. Signature `#7` is **the actual terminal wedge driver** under
> the customer load shape; it is **not** a TRT-LLM bug — it lives in
> the NIXL UCX plugin (`nixlUcxThreadEngine::getNotifs()` blocked on
> `pthread_mutex_lock`) and surfaces in TRT-LLM only because the
> blocked NIXL call is invoked from a TRT-LLM thread. The TRT-LLM-side
> deadline work (`kv_transfer_timeout_ms` enforcement, see Next Steps
> item 7) is best understood as a **fallback / mitigation** for
> signature `#7` — it converts the silent wedge into structured
> per-request errors and lets orchestration recover via pod restart,
> but it is **not** the ultimate solution. The ultimate solution for
> `#7` is a NIXL/UCX root-cause fix for the internal mutex deadlock
> (Next Steps item 8). Full evidence is in Phase 10 of the timeline.

> Use this report alongside the upstream investigation for
> [NVBug 6043291](../nvbug-6043291-zombie-worker-pods/README.md). That bug is
> about *the engine dying without anyone noticing*; this bug is about
> *no engine actually dying, but the disaggregated KV pipeline still
> deadlocking*. They share the disaggregated-serving HTTP path but their root
> causes are independent.

---

## How to Reproduce

### Topology

- 1 context worker (`trtllm-serve serve --server_role context --port 8001`)
- 1 generation worker (`trtllm-serve serve --server_role generation --port 8002`)
- 1 disaggregated front-end (`trtllm-serve disaggregated --port 8000`)
- All three colocated on a single node, two GPUs (`CUDA_VISIBLE_DEVICES=0`
  for context, `=1` for generation).
- Backend: PyTorch (`--backend pytorch`).
- Model: `Qwen/Qwen3-0.6B` (small enough to bring up quickly; the wedge
  pattern reproduces independent of model size).
- Transceiver: NIXL over UCX with TCP-only transport (the customer
  configuration). Relevant env:
  ```sh
  TRTLLM_USE_UCX_KVCACHE=1
  TRTLLM_NIXL_KVCACHE_BACKEND=UCX
  TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM=1
  UCX_TLS=tcp,cuda_copy,self
  ```
- Trace gating envs (set on workers and front-end while reproducing):
  ```sh
  TRTLLM_DISAGG_TRACE_PROMISE=1
  TRTLLM_DISAGG_TRACE_TRIE=1
  TRTLLM_DISAGG_TRACE_OPTIONAL=1
  TRTLLM_DISAGG_TRACE_BLOCK=1
  TRTLLM_DISAGG_TRACE_BLOCK_TIMEOUT_S=5
  ```

### Client load shape

The minimal harness that reliably reproduces the wedge is the
"long-prompt burst + recovery probes" script preserved in the local
disagg-repro worktree (under `.repro/harness/onepair/`).

Key parameters: `CONC=16`, `BURST_DUR_S=60`, prompt length sampled from
`gauss(8000, 2000)` tokens, `max_tokens=200`, `min_tokens=150`,
`temperature=0`. After the burst, the harness fires sanity probes at
`+30s`, `+60s`, `+90s`, `+120s`, and `+180s` of idle. If any probe returns
`ok200`, the system has recovered; if all probes time out, it is a permanent
wedge.

### Expected stock-`rc11` outcome

```text
[ 0.0s] CONC=16 SANITY PROBE
[PROBE-PRE] result=ok200 wall=8.8s
[BURST-1 90.0s] done ok200=8 errors=12 total=20
[PROBE-T+30] result=exc:ReadTimeout wall=60.1s
[PROBE-T+60] result=exc:ReadTimeout wall=60.1s
[PROBE-T+90] result=exc:ReadTimeout wall=60.1s
[PROBE-T+120] result=exc:ReadTimeout wall=60.1s
[PROBE-T+180] result=exc:ReadTimeout wall=60.1s
NO RECOVERY after 180s idle -- permanent wedge
```

This was confirmed both on stock `rc11` (run 4) and after the signature #4
fix in isolation (run 5). The system never recovers without process restart.

### Configurations that did *not* reproduce

- 1P1D with very short prompts (≤256 tokens): no wedge.
- 1P1D with overlap disabled: no wedge.
- 1P1D with no client-side timeouts (no cancels): no wedge.
- Single-process unit tests of the cache transceiver alone (without the
  disagg HTTP layer): only signature #1, #2, and #4 reproduce; #3, #5, #6
  require the full HTTP path with cancellation and retries.

---

## Failure Signatures

Each section below describes one signature: how it manifests, where it lives
in the code, what triggers it, and what the fix is (or what we still need
to learn).

### Signature #1 — Sender-side `Broken promise` after ready signal

**Symptom (log):**

```text
std::future_error: Broken promise
  ... (no exception path on the sender; the consumer's future.get() throws)
```

Originally surfaced on prefill workers in the field deployment as the most
visible Python-side traceback ("Hang detected on rank 0 in PyExecutor" was
ultimately a downstream consequence of the broken future).

**Where it lives:**
[`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp)
in `CacheSender::Impl::sendResponse(...)`.

**Root cause:** When a context request was cancelled while its KV-cache
ready-signal was already in flight, `sendResponse()` took the `else` branch
("not ready") and erased the corresponding `mReadyResponses` entry **without
fulfilling the promise**. The destructor of `std::promise<void>` then ran
with no value and no exception set, which causes the consumer's
`future.get()` to throw `std::future_error: Broken promise`.

That exception then propagated up into Python via the executor wrapper and
ultimately surfaced as a generic disagg failure that the upper layers were
not equipped to attribute to a specific request. From the outside, it looked
like a generic "context worker became flaky".

**Fix:** Set a structured `kNETWORK_ERROR` exception on the promise before
erasing the entry:

```cpp
auto cancelledException = TLLM_REQUEST_EXCEPTION(reqId,
    tensorrt_llm::common::RequestErrorCode::kNETWORK_ERROR,
    "Context KV cache transfer cancelled after ready-signal for request %zu",
    reqId);
it->second.mPromise.set_exception(std::make_exception_ptr(cancelledException));
```

The consumer's `future.get()` then throws a real `RequestSpecificException`
that the existing error path can attribute to the cancelled request.

**Reproducer:** `tests/unittest/others/test_kv_cache_transceiver.py::test_cancel_request_in_transmission_fulfills_sender_future`.
The test forces the cancel-after-ready path, then asserts that:
- the captured logs do **not** contain `"Broken promise"`
- the captured logs **do** contain `"cancelled_after_ready_signal"`
- the gen request reaches `LlmRequestState.DISAGG_TRANS_ERROR`

**PRs:** [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639)
(reproducer test) → [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640)
(fix). Chained on top of `#13639`.

### Signature #2 — Trie `cascade prune` assertion

**Symptom (log / assertion):**

```text
[TensorRT-LLM][ERROR] cascade prune: parent did not find this node as a child
TLLM_CHECK_WITH_INFO failed at templatedTrie.h:203 (or :249)
```

Caught most reliably as a hard assertion when running the disagg path under
sustained load with frequent eviction.

**Where it lives:**
[`cpp/include/tensorrt_llm/batch_manager/templatedTrie.h`](../../../cpp/include/tensorrt_llm/batch_manager/templatedTrie.h)
inside `clearNode()` and the cascade-prune walk it triggers, with the
trigger condition arising in
[`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`](../../../cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp)
during `removeNextBlock` / `attachToLookupNode` / `detachFromLookupNode` /
`freeBlockAndAllDescendants`.

**Root cause:** When a child node was removed from a parent's `mNextNodes`
without resetting the child's `mPrevNode`, the child still carried a stale
back-pointer to the parent. The next cascade-prune walk traversed from a
deeper subtree up through that stale `mPrevNode` and asserted because the
parent no longer recognised the child as one of its `mNextNodes`. This was
a true structural inconsistency in the trie, not a sporadic data race.

**Fix:** In `clearNode()`, reset the child's `mPrevNode` before erasing the
entry from the parent's `mNextNodes` map:

```cpp
itr->second->setPrevNode(NodePtr{});
mNextNodes.erase(itr);
```

This ensures every removed child becomes a root and the cascade-prune walk
cannot walk back into a parent that no longer holds the entry.

**Reproducer:** `cpp/tests/unit_tests/runtime/radixBlockTreeTest.cpp` —
four new stress cases that mirror the failing call path
(`addSequence → getFreeBlock → freeBlockAndAllDescendants →
detachDescendantsFromLookupTree`) under repeated insert/evict on
prefix-overlapping sequences. They reproduce the assertion deterministically
on stock `rc11`.

**PRs:** [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571)
(reproducer tests) → [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572)
(fix). Chained on top of `#13571`.

### Signature #3 — Decode-side `RuntimeError: bad optional access`

**Symptom (log):**

```text
RuntimeError: bad optional access
  ... raised through pybind into the Python decode-side event loop
```

Observed on **decode** (generation) workers in the original field deployment.
It is the only decode-side Python `RuntimeError` we saw before the wedge.

**Where it lives:** A C++ `std::optional::value()` call inside the
disaggregated gen path, surfaced through pybind11 / nanobind into Python.
We have not yet localised this to a single line, because in our local
1P1D reproductions it does **not** appear; only the field deployment hits
it. It is the only signature we have not yet caught with a unit test.

**Root cause (hypothesis):** `LlmRequest`'s `getContextPhaseParams()` and
several similar accessors return `std::optional<…>`. When a request is
cancelled or times out mid-transfer, certain disagg-only fields are left
empty even though downstream code unconditionally calls `.value()`. The
`Broken promise` from signature #1 may be the upstream trigger that leaves
the request in this half-initialised state.

**Status:** Not fixed yet. Likely to disappear (or change shape) once
signatures #1, #4, and #5 are all in place, because those are the conditions
under which the half-initialised state is reached. We have added Python-side
trace logs around the gen event loop's `_event_loop_wrapper` and
`_check_disagg_gen_cache_transfer_status` (gated on
`TRTLLM_DISAGG_TRACE_OPTIONAL=1`) so the next field hit will produce a
labelled stack with the active-request summary instead of just a bare
`RuntimeError`.

**Reproducer:** None yet. This signature is currently field-only.

### Signature #4 — Gen-side blocking hang in `checkGenTransferStatus(atLeastNum=1)`

**Symptom (Python thread-stack dump from the built-in hang detector):**

```text
[E] Hang detected after 300 seconds.
File ".../py_executor.py", line 671, in _event_loop_wrapper
  self.event_loop()
File ".../py_executor.py", line 2240, in _executor_loop_overlap
  scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
File ".../py_executor.py", line 1828, in _prepare_and_schedule_batch
  self._check_disagg_gen_transfer_status()
File ".../py_executor.py", line 2940, in _check_disagg_gen_transfer_status
  self._check_disagg_gen_cache_transfer_status(at_least_num)
File ".../py_executor.py", line 3284, in _check_disagg_gen_cache_transfer_status
  self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)
File ".../kv_cache_transceiver.py", line 200, in check_gen_transfer_status
  return self.impl.check_gen_transfer_status(at_least_request_num)
```

The gen worker's main event-loop thread is blocked inside the C++
transceiver, and never returns. From outside, the gen worker looks alive
(its HTTP server still accepts connections) but it never completes another
disagg transfer, so the front-end's requests to `:8002` time out.

**Where it lives:**
[`cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp)
inside `CacheTransceiver::checkGenTransferStatus(atLeastRequestNum)`.

**Root cause:** When `atLeastRequestNum=1`, the function fills
`toCompleteIdSet` up to `atLeastRequestNum` using either the ready-frequency
list or, as a fallback, the insertion order of `mRequesterFutures`. It then
unconditionally calls `it->second.get()` on every selected entry, even when
the entry was selected purely by insertion order and is **not yet ready**.
This is an unbounded blocking wait that can sit for the entire lifetime of
the deployment if the chosen receiver future never resolves (which is
exactly what signatures #1, #5, and #6 cause).

The Python transceiver in
[`tensorrt_llm/_torch/disaggregation/transceiver.py`](../../../tensorrt_llm/_torch/disaggregation/transceiver.py)
already has the intended semantics: only the explicit `block_all` case calls
the blocking variant; `at_least_request_num=1` calls
`wait_complete(blocking=False)` and skips entries that aren't ready. The C++
path was inconsistent.

**Fix:** Make the C++ path match the Python semantics. In the non-`blockAll`
path, probe each selected future with `wait_for(0)` first; if it is not
ready, log `gen_future_skip_unready` and skip it for this poll cycle
instead of blocking. `blockAll` semantics are preserved.

```cpp
auto status = it->second.wait_for(std::chrono::milliseconds(0));
if (!blockAll && status != std::future_status::ready) {
    if (tracePromiseLifecycle()) {
        TLLM_LOG_WARNING("[promise-trace] gen_future_skip_unready request=%zu status=%d",
            it->first->mRequestId, static_cast<int>(status));
    }
    ++it;
    continue;
}
it->second.get();
```

**Reproducer:** `tests/unittest/others/test_kv_cache_transceiver.py::test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`.
The test deterministically constructs an unresolved gen future, asserts
that `check_gen_transfer_status(0)` returns immediately, asserts that
`check_gen_transfer_status(1)` does **not** block on the unresolved future
(short-timeout watchdog thread), then unblocks the gen future and verifies
the transfer completes cleanly. The test fails on stock `rc11` and passes
post-fix.

**PRs:** [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674)
(test) → [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671)
(fix). Both PRs target `main`; `#13671` carries both the test and the
fix as 2 commits, so `#13674` lands first and `#13671`'s duplicate test
commit becomes a no-op.

### Signature #5 — Receiver-side `Broken promise` from queued cancel

**Symptom (log, post-`#4` fix):**

```text
[TensorRT-LLM][ERROR] Error occurred during generation transfer for
  request 4113: std::future_error: Broken promise
[TensorRT-LLM][ERROR] [promise-trace] gen_future_get_exception
  request=4113 msg=std::future_error: Broken promise
```

Observed only **after** the signature #4 fix: removing the gen-side
self-block exposed this receiver-side broken-promise pattern, which had
previously been hidden behind the unbounded `future.get()`.

**Where it lives:**
[`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp)
inside `CacheReceiver::Impl::cancelRequest(...)`.

**Root cause:** When a generation-side request was cancelled before the
worker thread picked it up off `mRequestsQueue`, `cancelRequest()` erased
the queued `RequestAndPromise` entry without fulfilling its promise. As the
`std::unique_ptr<std::promise<void>>` was destroyed, the consumer's
`future.get()` collapsed with `std::future_error: Broken promise`. This is
the structural mirror of signature #1, on the receiver side instead of the
sender side.

**Fix:** Extract the queued promise under the lock, then fulfill it with a
structured `kNETWORK_ERROR` exception once the lock is released:

```cpp
auto cancelledException = TLLM_REQUEST_EXCEPTION(cancelledId,
    tensorrt_llm::common::RequestErrorCode::kNETWORK_ERROR,
    "Generation KV cache request cancelled before send for request %zu",
    cancelledId);
queuedPromise->set_exception(std::make_exception_ptr(cancelledException));
```

**Reproducer:** New unit test
`test_cancel_queued_gen_request_fulfills_receiver_future` (in
`tests/unittest/others/test_kv_cache_transceiver.py`). It keeps the
receiver worker thread busy with a first orphan generation request whose
context counterpart will never respond, then enqueues a second orphan
request and cancels it while it is still queued. Pre-fix the test fails
because `Broken promise` appears on stderr; post-fix the cancelled
request lands in `kDISAGG_TRANS_ERROR` cleanly and stderr stays clean.

**PRs:** [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672)
(combined test + fix). Independent of the `#1` chain — the queued-cancel
path does not require the `#1` fix to be present.

### Signature #6 — Control-path stall inside `sendRequestInfo()` / `sendRequestAndBufferInfo()` *(suspected)*

**Symptom (gen-side request lifecycle trace, post-`#4` fix):**

The lifecycle markers we added show one specific request that reaches
`gen_request_sync_begin` but never reaches `gen_wait_ready_signal_begin`.
Every other gen request in the same window walks the full chain from
`gen_request_sync_begin → gen_wait_ready_signal_begin →
gen_wait_ready_signal_end → gen_receive_sync_begin → gen_receive_sync_end →
gen_request_promise_set_value`. Exactly one request stops between
`gen_request_sync_begin` and the next marker.

**Where it lives (most likely):** Either
[`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp)
inside `CacheReceiver::Impl::sendRequestInfo(LlmRequest const&)` (the
session-building variant, which loops over counterparts), or
[`cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp`](../../../cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp)
inside `AgentConnection::sendRequestAndBufferInfo(...)`, which performs the
control-path notify to the remote agent.

**Root cause (confirmed in `run7`):** Recv-buffer index exhaustion caused
by the `!isReady` early-return in `CacheReceiver::Impl::requestSync()`,
which leaks the buffer index that was reserved at the top of
`CacheReceiver::Impl::sendRequestInfo()`. The leak is then converted into
a permanent global wedge by an unbounded `cv.wait` inside
`BaseTransBufferManager::assignBufferIndex()`:

```cpp
// cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp
std::unique_lock lk(resource.mBuffersMutex);
resource.mBuffersCV.wait(
    lk, [&resource, bufferCount]() {
        return static_cast<size_t>(resource.mConcurrence) < bufferCount;
    });
```

`mRecvBufferCount` defaults to `1` (it is only larger when
`TRTLLM_REQUEST_KV_CACHE_CONCURRENT=1` is set), so a single leaked recv
buffer index is enough to wedge **every subsequent** receive forever.

The leak path itself is the cascade we expected from the earlier fixes:

1. The signature `#1` fix on the **sender** side now correctly sends
   `is_ready=false` for cancelled-after-ready requests.
2. On the **receiver** side this becomes `bool isReady = false` from
   `receiveReadySignal(session)`.
3. `requestSync()` then sets `kDISAGG_TRANS_ERROR` and `return`s **without
   calling `receiveSync()`**, so `unformat()` never runs, so
   `freeBufferIndexForRecv()` is never called, and the recv buffer index
   reserved at the top of `sendRequestInfo()` is leaked.
4. The next request to call `assignBufferIndexForRecv()` blocks forever
   inside the unbounded `cv.wait` above. The Python `optional-trace`
   markers showed exactly one in-progress generation request stuck
   between `gen_request_sync_begin` and `gen_send_request_buffer_info_*`
   in `run6`; the new fine-grained markers in `run7`
   (`gen_send_assign_buffer_begin / step / end`, plus
   `gen_send_compute_counterparts_begin / end` etc.) showed exactly
   `13` `assign_buffer_begin` events vs `12` `_step` and `_end` events
   for the same request — confirming the stall is **inside the very
   first `assignBufferIndexForRecv()` call** for the next request after
   a leak.

**Fix (Layer A — `sendRequestInfo` exception safety):** Track every
`(BaseTransBufferManager*, std::optional<size_t>)` pair returned by
`assignBufferIndexForRecv()`. Wrap the rest of `sendRequestInfo()` in a
`try { ... } catch (...) { freeAssignedRecvBuffers(); throw; }` block so
that any exception between assignment and the eventual `unformat()` call
releases the indices. On the success path the local tracking vector is
explicitly `clear()`-ed because ownership has been handed off to the
`AgentConnection`'s `mCacheBufferIds`, which `unformat()` will free.

**Fix (Layer B — `requestSync` `!isReady` cleanup):** Mirror what
`unformat()` does on the success path. In the `!isReady` early-return
branch of `CacheReceiver::Impl::requestSync()`, iterate the session's
connections, look up each pre-assigned recv buffer ID via
`agentConnection->getPreAssignedBufferId(static_cast<uint8_t>(mgr->getBufferKind()))`,
and free it via `mgr->freeBufferIndexForRecv(id)`. The number of indices
freed is logged as
`gen_request_sync_not_ready_buffers_freed request=R count=N` so each
post-fix run shows explicitly that the leak path is now closing.

**Reproducer:** Same end-to-end 1P1D + long-prompt burst harness. A
minimal unit test analogous to the signature `#1` reproducer is the
natural next step: queue a generation request whose matching context is
cancelled-after-ready, observe that the next generation request would
have blocked in `assignBufferIndexForRecv()` pre-fix and completes
normally post-fix.

**Status:** Combined test + fix opened as
[#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673), chained on
[#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) (the `#1`
fix is a prerequisite for the `!isReady` early-return path to be
reachable in production code). `run8` validated the fix at the C++
trace level before submission — every request now walks
`gen_send_assign_buffer_begin → step → end` cleanly (33/33), and the
new `gen_request_sync_not_ready_buffers_freed` marker fires on every
cancelled-after-ready request (3/3 in `run8`). The wedge however still
occurs at the harness level; the post-mortem stack dumps on the ctx
worker show the wedge has shifted **off the TRT-LLM transceiver
entirely** and into a NIXL UCX-internal `pthread_mutex_lock` inside
`recvRequestInfo()` (signature `#7`). See Phase 10 of the timeline
below for full details.

**PRs:** [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673)
(combined test + fix, chained on `#13640`).

### Signature #7 — NIXL UCX-internal `pthread_mutex_lock` deadlock surfacing as a transceiver-level wedge

**Symptom (run8 post-mortem, after every TRT-LLM-side signature is
fixed):**

The local burst-harness still reports
`NO RECOVERY after 180s idle -- permanent wedge`. Every TRT-LLM trace
marker shows clean lifecycle progression — 33/33 `requestSync_begin`
→ `requestSync_end`, 33/33 `assignBufferIndex` `begin → step → end`,
3/3 `gen_request_sync_not_ready_buffers_freed`, signature `#1`/`#5`
fix paths firing exactly the expected number of times — yet
`checkGenTransferStatus()` only ever observes 16 of 30 receiver-side
futures as ready, and recovery probes never succeed.

**Where it lives:** This is **not** in TRT-LLM. It is in the NIXL UCX
plugin (`/opt/nvidia/nvda_nixl/lib/.../libplugin_UCX.so`), inside
`nixlUcxThreadEngine::getNotifs()`. From a TRT-LLM perspective, the
deadlock surfaces in
[`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp)
inside `CacheSender::Impl::recvRequestInfo()`, which calls into the
NIXL agent and never returns.

**Definitive evidence (gdb thread dump, ctx-side `mpi4py.futures.server`
worker):**

```text
CacheSender::Impl::response()                                       [thread "dataTransResp"]
  → CacheSender::Impl::recvRequestInfo()
    → AgentConnectionManager::recvConnectionAndRequestInfo()
      → AgentConnectionManager::updateUnhandledNotifications()
        → NixlTransferAgent::getNotifiedSyncMessages()
          → nixlAgent::getNotifs()
            → nixlUcxThreadEngine::getNotifs()
              → pthread_mutex_lock                                  ← STUCK INDEFINITELY
```

Two further ctx-side threads confirm the deadlock is internal to the
NIXL UCX plugin (and not in TRT-LLM code):

| thread | top-most NIXL frame |
|---|---|
| `nixl-comm-worker` | `nixlAgentData::commWorkerInternal()` → `sched_yield()` (NIXL spin loop) |
| `nixl-ucx-shared` | `nixlUcxSharedThread::run()` → `nixlUcxWorker::arm()` → `ucp_worker_arm()` → `read(fd=129)` |
| `dataTransResp` | `pthread_mutex_lock` inside `nixlUcxThreadEngine::getNotifs()` |

The wedged mutex is owned by NIXL UCX-internal state; no TRT-LLM
thread holds it. There is no TRT-LLM API that can release it.

**Mechanism end-to-end (why an external wedge looks like a TRT-LLM
wedge):**

1. The first batch of disaggregated requests under the customer load
   shape (long prompts, high concurrency, mid-flight cancellations,
   retries) exercises the NIXL UCX path under a contention pattern
   that hits an internal lock-ordering problem.
2. NIXL deadlocks on its own internal mutex inside `getNotifs()`. The
   underlying UCX worker thread (which would normally release the
   mutex by progressing UCX events) is itself blocked.
3. The TRT-LLM-side `dataTransResp` thread that called `getNotifs()`
   is now stuck forever on `pthread_mutex_lock`. There is one such
   thread per `CacheSender::Impl`, so the entire ctx-side `CacheSender`
   becomes single-threaded-and-wedged.
4. Subsequent gen-side requests can complete the TRT-LLM-side
   handshake (`sendRequestInfo` → ready signal) up to the point where
   the ctx side would actually drain the next `RequestInfo` — at
   which point the gen side either silently waits forever or, with
   the signature `#6` fix in place, walks the in-progress
   request-sync path cleanly but never observes its receiver-side
   future become ready (the 14-of-30 stranding pattern in `run8`).
5. From the outside, the deployment is wedged. From the gen-side
   `py-spy` dump, every Python and TRT-LLM-level thread is in a
   correct idle position. From the ctx-side `gdb` dump, one thread is
   stuck on `pthread_mutex_lock` inside NIXL.

**Reproducer:** Same end-to-end 1P1D + long-prompt burst harness used
for the rest of this investigation. The `run8` archive at
`~/disagg-investigation-archive/run8_sig6_fix/pyspy/` contains the
canonical evidence: `gen_worker_*_gdb.txt` (gen worker, all-thread bt)
and `ctx_worker_*_gdb.txt` (ctx worker, all-thread bt — contains the
NIXL mutex frame).

**Fix (root cause — out of TRT-LLM scope):** Fix the
`pthread_mutex_lock` deadlock inside `nixlUcxThreadEngine::getNotifs()`.
This is the only path that makes the local reproducer pass *without*
relying on an external orchestrator restart loop. Owned by the
NIXL/UCX team; should be filed with the `ctx_worker_*_gdb.txt` stack
as the canonical reproducer (Next Steps item 8).

**Fallback / mitigation (TRT-LLM scope):** Enforce
`kv_transfer_timeout_ms` as a hard deadline on the C++ blocking
entry points (Next Steps item 7). This **does not** unwedge the NIXL
mutex — TRT-LLM has no way to interrupt a `pthread_mutex_lock` inside
NIXL — but it converts the silent global wedge into structured
per-request `kNETWORK_ERROR`s, bounds queue growth on the gen side,
and gives orchestration a real signal so a pod restart can clear the
NIXL-level state. Effort estimate, layered options, and trade-offs
are documented in the "Effort estimate for the deadline enforcement"
section below.

**Status:** Identified, classified, and documented in this report.
The TRT-LLM-side fallback (Layer A: Python-level deadline + cancel)
is the recommended next implementation step. The NIXL/UCX root-cause
fix should be filed in parallel.

**PRs:** No TRT-LLM PR for the root cause (out of scope). The
deadline-enforcement fallback PR will be sized per the "Effort
estimate" section below.

---

## Investigation Timeline

This is the chronological order in which the signatures were identified,
not the chronological order of the fixes (which generally followed a few
hours behind).

### Phase 0 — Field report (T0)

Customer report from Dynamo + TRT-LLM `rc11` deployment of a permanent
hang under sustained traffic. Three apparent failure signatures in the
crash dumps and Python tracebacks:

1. `std::future_error: Broken promise` on prefill workers.
2. `cascade prune: parent did not find this node as a child` C++ assertion
   under sustained load.
3. `RuntimeError: bad optional access` raised in the decode-side Python
   event loop.

These are signatures **#1, #2, #3** in the table above.

### Phase 1 — Layered C++ unit-test probes for signature #2 (T0 + a few hours)

Cheapest, fastest layer: extend `radixBlockTreeTest.cpp` with stress cases
that mirror the failing call path
(`addSequence → getFreeBlock → freeBlockAndAllDescendants →
detachDescendantsFromLookupTree`). Four new tests reproduce the
`cascade prune` assertion deterministically on stock `rc11`. This proves
that signature #2 is independent of Dynamo, NIXL, and disaggregated
networking.

→ Reproducer PR: [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571).

### Phase 2 — Layer-B fix for signature #2 (T+1 day)

Reset the child's `mPrevNode` in `clearNode()` before erasing the entry
from the parent. The new unit tests pass; existing radix-tree tests still
pass.

→ Fix PR: [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572),
chained on top of `#13571`.

### Phase 3 — Local 1P1D reproduction attempts (T+1 day)

Spin up a local two-GPU 1P1D `trtllm-serve` deployment on `rc11`. Light load
does not reproduce the hang. Switching to the customer's load shape (long
prompts, `CONC=16`, retries, cancellations) successfully reproduces the
permanent wedge. Confirms that signature #1 is real and reachable
end-to-end without Dynamo.

### Phase 4 — Signature #1 isolation, fix, and unit test (T+2 days)

- Targeted C++ instrumentation (`tracePromiseLifecycle()` gated on
  `TRTLLM_DISAGG_TRACE_PROMISE=1`) localises signature #1 to the
  cancelled-after-ready path in `CacheSender::Impl::sendResponse`.
- Fix: `set_exception(std::make_exception_ptr(...))` on the promise
  before the entry is erased.
- New unit test
  `test_cancel_request_in_transmission_fulfills_sender_future` reproduces
  the broken promise on stock `rc11` and passes post-fix.

→ Reproducer PR: [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639);
fix PR: [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640), chained
on top of `#13639`.

### Phase 5 — Post-`#1`-and-`#2` rerun: wedge persists, surfaces signature #4 (T+3 days)

With `#13572` and `#13640` applied to the isolated `rc11` worktree, the
1P1D repro **still wedges**. The Python hang detector dumps thread stacks
and shows the gen worker's main event loop blocked exactly inside
`self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)`.

Code reading of `cacheTransceiver.cpp` and the matching Python
implementation in `transceiver.py` shows that the C++ path takes an
unbounded blocking wait when `atLeastNum=1`, while the Python path does
not. This is signature **#4**.

### Phase 6 — Signature #4 fix and regression test (T+3 days)

- Fix: in the non-`blockAll` path, probe each selected future with
  `wait_for(0)` and skip if not ready.
- New unit test
  `test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
  fails on stock `rc11` (asserts the wrong behaviour) and passes post-fix.

This is the regression test for signature #4. It is currently isolated in
the `local/rc11-disagg-repro` worktree.

### Phase 7 — Post-`#4` rerun: wedge persists, surfaces signature #5 and suspected signature #6 (T+4 days)

Repro again with the `#4` fix applied. The gen event loop is no longer
self-blocked; `gen_future_skip_unready` markers appear repeatedly in
`gen.log`. But the end-to-end harness still reports `NO RECOVERY after 180s
idle -- permanent wedge`, and two new patterns appear in the C++ traces:

- **Signature #5:** several requests get `Broken promise` from the gen
  side that originate from the receiver-side cancel path
  (`CacheReceiver::Impl::cancelRequest`). This is a structural mirror of
  signature #1 on the receive side.
- **Suspected signature #6:** exactly one request reaches
  `gen_request_sync_begin` and never reaches `gen_wait_ready_signal_begin`,
  meaning it stalls inside `sendRequestInfo()` or
  `sendRequestAndBufferInfo()` before reaching the ready-signal wait. The
  current instrumentation is not granular enough to say which.

### Phase 8 — Signature #5 fix and signature #6 instrumentation (T+5 days)

- Receiver-side fix: extract the queued promise under the lock and fulfill
  it with a structured `kNETWORK_ERROR` exception once released. Same shape
  as signature #1.
- New trace markers around `sendRequestInfo()` (`gen_send_request_info_begin/end`)
  and `AgentConnection::sendRequestAndBufferInfo()`
  (`gen_send_request_buffer_info_begin/notify/end`).
- `run6` confirms signature #5 is gone (zero `Broken promise` events on the
  generation side, `gen_request_promise_set_exception type=cancelled_before_send`
  fires on the receiver's queued-cancel path) and confirms one in-progress
  request is still stuck after `gen_request_sync_begin`. The wedge persists.

### Phase 9 — Signature #6 root cause + fix (T+5 days)

- Add fine-grained instrumentation across the entire `sendRequestInfo()`
  body: `gen_send_validate_support_begin/end`, `gen_send_block_range_begin/end`,
  `gen_send_assign_buffer_begin / step / end`,
  `gen_send_compute_counterparts_begin/end`,
  `gen_send_get_kv_counterparts_begin/end`,
  `gen_send_get_connections_begin/end`,
  `gen_send_counterpart_iter`, `gen_send_pick_recv_connections_begin/end`,
  `gen_send_agent_dispatch_begin/end`, `gen_send_nonagent_dispatch_begin/end`.
  Plus a defensive `mTerminate` check at the top of `sendRequestInfo()` and
  inside the per-counterpart loop so receiver shutdown can interrupt the
  worker thread.
- `run7` shows exactly one in-progress generation request reaches
  `gen_send_assign_buffer_begin` and never reaches `_step` or `_end`. Code
  reading of `BaseTransBufferManager::assignBufferIndex()` confirms it does
  an unbounded `cv.wait` with no timeout, and `mRecvBufferCount` defaults
  to `1`. A single leaked recv buffer index permanently wedges every
  subsequent receive.
- The leak was a direct consequence of signature `#1`'s fix: the `!isReady`
  early-return path in `CacheReceiver::Impl::requestSync()` skips
  `receiveSync()` (and therefore `unformat()`'s `freeBufferIndexForRecv()`
  call) for every cancelled-after-ready request.
- Fix: RAII-style cleanup vector in `sendRequestInfo()` (Layer A), and
  explicit free in the `!isReady` early-return path of `requestSync()`
  (Layer B, mirrors `unformat()` via `getPreAssignedBufferId`). New marker
  `gen_request_sync_not_ready_buffers_freed request=R count=N` shows the
  leak path closing on every cancelled request.
- `run8` is the first end-to-end validation run with this fix.

### Phase 10 — `run8` validation: signature `#6` confirmed fixed, signature `#7` identified as a NIXL/UCX-layer deadlock (T+6 days, *current*)

`run8` was run end-to-end with the signature `#6` Layer-A + Layer-B fix in
place plus the new `[promise-trace]` markers. The harness reported
`NO RECOVERY after 180s idle -- permanent wedge`, but the C++ trace
counts and post-mortem stack dumps show that the wedge mechanism has
**shifted off the TRT-LLM transceiver entirely**.

#### Per-marker accounting (gen worker, `run8`)

| marker | count | meaning |
|---|---:|---|
| `gen_request_enqueue` / `gen_request_dequeue` | 33 / 33 | every request is queued and pulled by the receiver worker |
| `gen_request_sync_begin` / `gen_request_sync_end` | 33 / 33 | every request enters and exits `requestSync()` cleanly |
| `gen_send_assign_buffer_begin` / `step` / `end` | 33 / 33 / 33 | **no `assignBufferIndex()` `cv.wait` stalls** (signature `#6` fix verified) |
| `gen_request_sync_not_ready_buffers_freed` | 3 | new explicit free in the `!isReady` early return is firing on every cancelled-after-ready request |
| `gen_request_promise_set_value` | 33 | every requester-side promise is fulfilled |
| `gen_receive_sync_begin` / `gen_receive_sync_end` | 30 / 30 | 30 of 33 reach `receiveSync()`; the other 3 are the cancelled-after-ready cases |
| `gen_future_get_ok` / `gen_future_get_exception` | 16 / 0 | only 16 of the 30 receiver-side futures are observed ready by `checkGenTransferStatus()` |

#### Per-marker accounting (ctx worker, `run8`)

| marker | count | meaning |
|---|---:|---|
| `create` | 43 | sender-side promises created |
| `send_response_ready` | 33 | 33 requests reached the ready-signal dispatch |
| `send_sync_begin` / `promise_set_value` / `future_get_ok` | 30 / 30 / 30 | 30 successful sends |
| `mark_cancelled` / `cancel_rejected` | 12 / 40 | client-side cancels in flight; `cancel_rejected` counts the races where the request is no longer in the queue |
| `promise_set_exception` / `future_get_exception` | 3 / 3 | signature `#1` fix path firing on cancelled-after-ready (ctx side) |
| `drop_without_fulfill` | 3 | (see "trace marker correction" below — these are *not* drops) |

#### Stack-trace evidence: the wedge is one layer below TRT-LLM

Once the harness reported `NO RECOVERY`, `py-spy dump` and
`gdb -p ... thread apply all bt` were run against both the gen-side and
ctx-side mpi4py-spawned executor workers. The gen-side `CacheReceiver`
request thread is parked correctly on its own `cv` waiting for new
work — the receiver side is healthy. The actual wedge is on the
**ctx-side `dataTransResp` thread**, stuck inside NIXL UCX:

```text
CacheSender::Impl::response()                                       [ctx, "dataTransResp"]
  → CacheSender::Impl::recvRequestInfo()
    → AgentConnectionManager::recvConnectionAndRequestInfo()
      → AgentConnectionManager::updateUnhandledNotifications()
        → NixlTransferAgent::getNotifiedSyncMessages()
          → nixlAgent::getNotifs()
            → nixlUcxThreadEngine::getNotifs()
              → pthread_mutex_lock                 ← STUCK indefinitely
```

Two further ctx-side threads confirm the deadlock is internal to the
NIXL UCX plugin (and not in TRT-LLM code):

| thread | stack (top-most TRT-LLM/NIXL frame) |
|---|---|
| `nixl-comm-worker` | `nixlAgentData::commWorkerInternal()` → `sched_yield()` (NIXL spin) |
| `nixl-ucx-shared` | `nixlUcxSharedThread::run()` → `nixlUcxWorker::arm()` → `ucp_worker_arm()` → `read(fd=129)` |
| `dataTransResp` | `pthread_mutex_lock` inside `nixlUcxThreadEngine::getNotifs()` |

This explains the otherwise-puzzling marker accounting:

- The first 33 requests slipped through the request handshake before
  NIXL got stuck (hence `gen_send_request_info_end:33`,
  `gen_wait_ready_signal_end:33`).
- Once NIXL deadlocks on its own internal mutex, `recvRequestInfo` on
  the ctx side can't drain notifications anymore. From the gen side,
  every subsequent `sendRequestInfo` either blocks on the recv-buffer
  cv-wait (formerly signature `#6`, now fixed) or is silently never
  acknowledged by ctx.
- The harness's recovery probes (issued *after* the burst window) never
  succeed because the ctx-side NIXL agent stays deadlocked for the
  entire 180 s idle period.

Stack-trace artifacts:
`/home/.../disagg-investigation-archive/run8_sig6_fix/pyspy/`
contains `gen_worker_*_gdb.txt` (gen worker, all-thread bt) and
`ctx_worker_*_gdb.txt` (ctx worker, all-thread bt — contains the NIXL
mutex frame).

#### Promoting the NIXL deadlock to Signature #7

The NIXL UCX-internal `pthread_mutex_lock` deadlock is now tracked as
**Signature `#7`** in the canonical signature list above. This phase is
where it was identified. Two earlier hypotheses (`#7a` and `#7b`) that
appeared during the `run8` triage are explicitly *not* real bugs:

- **`#7a` (the `drop_without_fulfill` trace marker firing 3 times):**
  not a bug. The marker is a leftover name from before the signature
  `#1` fix landed, and currently fires on the line *immediately above*
  the new `set_exception(kNETWORK_ERROR)` call in
  `CacheSender::Impl::sendResponse()`. Every `drop_without_fulfill`
  event is followed in the next 3 lines by the correct
  `promise_set_exception` + `future_get_exception` pair. Counts match:
  3 / 3 / 3. The marker should be renamed `cancelled_after_ready_handled`
  in a follow-up cleanup (Next Steps item 9).
- **`#7b` (the 14-of-30 receiver-side future stranding pattern):** real
  symptom, but not a standalone TRT-LLM bug. With the NIXL-layer
  evidence above, it is best understood as the visible TRT-LLM-side
  surface of signature `#7`: `receiveSync()` returns once the UCX recv
  has been *posted*, but the UCX progress thread that would fire the
  completion is itself blocked by the same NIXL mutex contention,
  leaving the receiver-side future technically `not_ready` even though
  upstream code has already moved on. (A secondary contributor — the
  unresolved C++ ↔ Python lifetime ownership debt called out in the
  Architectural Reflections section — could in principle produce the
  same visible pattern, but a focused disambiguation is deferred until
  the NIXL-level deadlock is addressed.)

#### Updated wedge-driver picture

Before this investigation: the wedge looked like a TRT-LLM-only set of
five-to-six signatures (one assertion, three promise-lifetime issues,
one blocking-wait issue, one resource-leak issue). After this
investigation and the `run8` post-mortem the picture is cleaner:

- Signatures `#1`, `#2`, `#4`, `#5`, `#6` are real TRT-LLM bugs and are
  fixed by the chained PRs documented below. Their fixes are necessary.
- Signature `#3` is a visibility issue, mitigated by upstream PRs not
  in `rc11`.
- Signature `#7` (the NIXL UCX-internal `pthread_mutex_lock` deadlock)
  is the **terminal wedge driver** under the customer load shape and
  is **not a TRT-LLM bug**. It surfaces in TRT-LLM only because the
  blocked NIXL call is invoked from a TRT-LLM thread.

The chained TRT-LLM PRs close every TRT-LLM-side bug in the failure
class. They are necessary but not sufficient on their own to make the
field reproducer pass: the reproducer will continue to wedge until
either the NIXL/UCX root-cause fix lands (Next Steps item 8) or the
TRT-LLM-side `kv_transfer_timeout_ms` deadline fallback (Next Steps
item 7) is paired with an orchestrator-restart contract that converts
the wedge into a recoverable per-request error pattern.

The deadline enforcement work is therefore positioned in this report
as a **fallback / mitigation for signature `#7`**, not as an
ultimate fix.

---

## Why the Existing Tests Did Not Catch This

The disaggregated-serving test suite covered each of these surfaces in
isolation but not in combination. In particular:

- **Sender / receiver cancel paths are tested for happy-path completion,
  not for the cancel-after-ready and queued-cancel races.** Signatures #1
  and #5 both live in `cancelRequest()` paths that existing tests touch
  only superficially.
- **`checkGenTransferStatus` is tested with `atLeastNum=0` and with
  `block_all`, not with the mixed `atLeastNum=1` semantics over an
  outstanding-but-not-ready future.** Signature #4 lives exactly there.
- **The trie-eviction tests previously stressed insertion and lookup, not
  the `freeBlockAndAllDescendants → detachDescendantsFromLookupTree` walk
  on prefix-overlapping sequences.** Signature #2 is reachable only via
  that walk.
- **The end-to-end disagg integration tests use short prompts, low
  concurrency, no client-side timeouts, and no cancellations.** None of
  the signatures above are reproducible under that load shape — even
  signature #1, which exists on stock `rc11`, only fires when a request
  is actually cancelled while in flight.
- **The bugs partially mask each other and even create each other.** With
  signature #4 in place, the gen event loop self-blocks before signatures
  #5 and #6 can manifest. Removing signature #4 was a prerequisite for
  even seeing #5 and #6 in the logs. Signature #6 is more pointed: it is
  a **direct consequence** of the signature #1 fix — the new
  cancelled-after-ready path on the sender turned into a `!isReady`
  early-return on the receiver, which skipped the only `unformat()` call
  site that would have released the recv buffer index. This is also why
  "fix one bug, see another" is the dominant pattern in the timeline above.
- **Companion fixes #12718 and #13119 are not in `rc11`.** Even when one
  of these signatures fires in the field on `rc11`, the failure-visibility
  improvements that would have made attribution easier
  ([#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119)) and the
  pod-restart safety net that would have prevented an indefinite wedge
  ([#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718)) are
  absent. They both need to land into `rc11` (or the equivalent field
  branch) regardless of how this investigation finishes.

The single largest test-coverage gap is the **disaggregated, long-prompt,
high-concurrency, with cancellations and retries** scenario — exactly the
load shape used by the customer reproducer and by the long-prompt burst
harness in the local disagg-repro worktree. The new unit tests close part
of this gap (sender-side cancel-after-ready, gen-side `atLeastNum=1` with
unresolved future), but a true integration test that drives the full HTTP
path with cancellations is still missing.

---

## Architectural Reflections — What Was Missing in the First Place

A reasonable question to ask after this many cascading bugs is: *why are
there so many hidden bugs, and why are they only surfacing now?* This
section is the answer I arrived at while running the investigation.

### Why now

Three things converged. None of them are individually new, but together
they exercise a part of the disaggregated transceiver that prior workloads
never reached in volume:

1. **The subsystem is young.** Disaggregated serving is still flagged
   "experimental" in the docs. NIXL is newer still. The transceiver was
   built layer-by-layer (UCX → NIXL → cache-aware formatters → buffer
   pool manager) with each layer adding its own thread, queue, future,
   and condition-variable wait. The combined contract across the layers
   was never formalised.
2. **The customer load shape exercises the cleanup paths, not the happy
   path.** Long prompts plus high concurrency plus client-side cancels
   plus retries means almost every request can hit an abort, timeout, or
   eviction mid-transfer. That is the surface where every signature in
   this investigation lives. Most prior workloads — short prompts, low
   concurrency, no aggressive timeouts — never reach the cleanup paths
   in volume, so the bugs sat dormant.
3. **The test pyramid is shaped wrong for this surface.** Each subsystem
   has unit tests for happy-path completion. End-to-end disaggregated
   integration tests use short prompts, low concurrency, and no
   cancellations. There is essentially no test that drives the
   combination "cancel during transfer at scale", which is the single
   load shape every signature here requires.

That alone explains the "many latent bugs surface in two weeks" pattern.
But it does not explain why the bugs cluster so tightly on the same
handful of code paths. That part is design.

### The seven invariants the transceiver doesn't enforce

Every signature in this investigation can be re-described as a violation
of one of seven contracts that the transceiver doesn't actually have an
explicit enforcement point for. Each is a missing invariant, not a bug —
the bugs are individual instances, the invariant gaps are the architecture.

1. **Ownership across the C++ ↔ Python boundary.** `mSenderFutures` and
   `mRequesterFutures` hold raw `LlmRequest*` while Python (with
   `shared_ptr<LlmRequest>` semantics) decides when the underlying
   `LlmRequest` dies. That is a guaranteed use-after-free surface — Python
   only has to terminate a request mid-transfer once. The right
   architectural answer is `shared_ptr` all the way through; the fact
   that raw pointers ever crossed a language-managed lifetime boundary
   is the smell. **None of the six signatures here is the UAF, but every
   single fix here lives next to one.**
2. **Every promise must be fulfilled exactly once before destruction.**
   Signatures `#1` and `#5` are the same architectural omission on
   opposite sides: a code path erases a `(request, promise)` entry
   without first calling `set_value` or `set_exception`. There is no
   central invariant, no lint, no destructor that defaults to
   `set_exception(unfulfilled)`. Every new cleanup path is a fresh
   chance to forget. The two `set_exception(kNETWORK_ERROR)` fixes are
   correct but they are patching individual sites of a missing invariant.
3. **Every blocking wait must be interruptible.** The
   `BaseTransBufferManager::assignBufferIndex()` `cv.wait`, the gen-side
   `checkGenTransferStatus()` unconditional `future.get()`, the
   ready-signal recv, and the underlying NIXL/UCX waits all blocked
   unboundedly with no cancel-flag awareness. Signatures `#4` and `#6`
   live exactly here. There is no cross-cutting "all blocking calls take
   a cancel token / a deadline / a `mTerminate` check" rule.
4. **Every acquired resource must release on every exit path (RAII).**
   The recv-buffer pool slots had at least three exit paths from
   `requestSync()` and only the happy one (success → `unformat()`)
   released. The Layer-A and Layer-B fix for signature `#6` is a textbook
   RAII fix; the question is why the original code did manual
   `assignBufferIndex` / `freeBufferIndex` pairing instead of writing
   the holder on day one.
5. **Same operation, same semantics across language layers.** Signature
   `#4` — the C++ `checkGenTransferStatus(atLeastNum=1)` blocks while the
   Python `transceiver.py` wrapper for the same operation skips unready
   entries — is a pure contract divergence. Two implementations of one
   conceptual operation drifted; nothing checks they agree.
6. **A configuration knob without an enforcement point is debt.**
   `kv_transfer_timeout_ms` was plumbed all the way through config and
   was never enforced as a hard deadline for the C++ blocking calls.
   Signature `#6` would have surfaced as a per-request error long before
   it became a global wedge if the receiver-side `cv.wait` had honored
   that knob. This is symptomatic of feature-on-feature growth without
   a designated enforcement layer for newly-added knobs.
7. **Long-lived worker loops must be robust to any escape.** The
   receiver drain worker uses `catch (std::exception)` but no
   `catch (...)`. A non-`std` throw from NIXL or UCX strands the queue
   and silently kills the worker thread, which then looks identical to
   signature `#6` from outside. The investigation didn't end up needing
   this fix, but the same "no rule" pattern is the reason it exists.

### How to read these as a class

It is more accurate to think of the transceiver as a textbook example of
**inherited concurrency complexity without a unifying async contract**
than as "fundamentally bad design". This is a depressingly common pattern
in performance-focused C++ async code — not unique to TRT-LLM. The
transceiver works under the happy path because each subsystem is
individually correct. It breaks under cancel/timeout/exception paths
because there is no shared notion of:

- what it means to cancel a request mid-flight,
- who owns an in-flight request's lifetime,
- when a promise gets fulfilled,
- where a blocking wait checks for shutdown / cancel / deadline,
- which exits must release which resources, and
- how errors propagate from C++ back to Python.

In a more mature subsystem you would expect to see a single
`TransferSession`-like type that bundles request lifetime + cancel token
+ buffer holders + promise + timeout into one RAII-managed object, with
every send/receive path expressed as a method on it. The fixes in this
investigation are incrementally bending the code in that direction
(structured cancellation exceptions, the recv-buffer RAII guard, bounded
non-blocking polls), but they are a retrofit rather than a clean redesign.

### Why code review didn't catch any of this

Honestly: because the review surface for "you forgot to fulfill a promise
on this cleanup path" or "this `cv.wait` isn't cancellable" is invisible
without the contracts written down. A reviewer looking at a 50-line PR
adding a new cleanup branch has no way to spot that it violates an
unwritten invariant the rest of the file follows by accident. This is
exactly the failure mode that systematic invariants (or strong type-level
abstractions) are supposed to prevent — and the transceiver currently
has neither.

### What this implies for follow-up work

The actual remediation, in order of long-term value:

1. **Document the seven invariants above** in the disaggregated-serving
   developer guide, with a one-paragraph "if you're adding a new transfer
   path, here is the checklist" section. Cheap, high leverage, prevents
   the next field hit.
2. **Introduce a `TransferSession`-like abstraction** that is the only
   blessed way to start a disagg KV transfer, with the seven invariants
   baked into its type. Reviewers can then enforce by type, not by
   discipline.
3. **Add an integration test** specifically for the cancel-during-transfer
   surface: long prompts, high concurrency, aggressive client-side
   timeouts, retries. This is the single load shape that exercises every
   signature documented here, and the absence of such a test is the
   single biggest reason this bug class went undetected for so long.
4. **Audit other early-return paths** in the C++ disagg transceiver for
   leaks of similarly cv-waited resources (other concurrence resources,
   request-side state, etc.). The signature `#6` pattern — "fix the
   visible failure path on side A, surface a resource leak on side B" —
   is likely to repeat if other paths share the same RAII gap.

The point of this section is that the next contributor adding a new
transfer mode is one cleanup path away from re-introducing the same class
of bug if these invariants stay implicit. A short architectural note that
names the seven contracts above would pay for itself in one prevented
field hit.

---

## Signature ↔ PR Map

| Signature | Status | Test PR | Fix PR | Notes |
|---|---|---|---|---|
| **#1** Sender-side `Broken promise` after ready signal | Test merged; fix in review | [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) | [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) | Chained: `#13640` builds on `#13639`. |
| **#2** Trie `cascade prune` assertion | Test merged; fix in review | [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) | [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) | Chained: `#13572` builds on `#13571`. Independent of disagg networking. |
| **#3** Decode-side `RuntimeError: bad optional access` | Field-only; not yet localised | — | — | Python-side trace markers added; will localise on next field hit. |
| **#4** Gen-side blocking hang in `checkGenTransferStatus(atLeastNum=1)` | Test merged; fix in review | [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674) | [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671) | `#13671` carries both the test and the fix as 2 commits; both PRs target `main` so `#13674` lands first and `#13671`'s duplicate test commit becomes a no-op. |
| **#5** Receiver-side `Broken promise` from queued cancel | Combined test + fix in review | (combined into fix PR) | [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672) | Mirror of `#1` on the receiver side. New test `test_cancel_queued_gen_request_fulfills_receiver_future` keeps the receiver worker busy with a first orphan request, then enqueues and cancels a second; pre-fix `Broken promise` lands on stderr, post-fix the cancelled request reaches `kDISAGG_TRANS_ERROR` cleanly. |
| **#6** Recv-buffer index leak via `!isReady` early-return; subsequent receives block in `BaseTransBufferManager::assignBufferIndex()` | Combined test + fix in review (chained on `#13640`) | (combined into fix PR) | [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673) | Two-layer fix: RAII cleanup in `sendRequestInfo()` (Layer A) + explicit free in `requestSync()` `!isReady` path (Layer B). Direct cascade from the `#1` fix; chained on `#13640` because the `!isReady` branch is only reachable once the sender-side cancellation correctly sends `is_ready=false`. New test `test_cancelled_after_ready_does_not_leak_recv_buffer_index` uses the NIXL backend (the only backend that goes through `assignBufferIndexForRecv`). |
| **#7** NIXL UCX-internal `pthread_mutex_lock` deadlock (terminal wedge driver) | Identified, classified, documented; **not** a TRT-LLM bug | — (test would need to inject a NIXL mock or fault-inject the UCX layer; deferred) | NIXL/UCX root-cause fix is **out of TRT-LLM scope** (Next Steps item 8) | TRT-LLM-side **fallback / mitigation** is the `kv_transfer_timeout_ms` deadline work in Next Steps item 7 — converts silent wedge into per-request errors so orchestration can recover. **Not** the ultimate fix. |

Companion fixes (already in `main`, not in `rc11`):

- [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) — request-level
  error propagation in disagg serving. Does **not** fix the wedge, but makes
  the failure mode visible (replaces generic `400 Bad Request` with the real
  error body, regenerates `disagg_request_id` on retry). Strongly recommended
  as a backport target for `rc11`.
- [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — fatal engine
  detection / pod restart. Does **not** fix the wedge, but ensures that if
  the engine actually crashes (which is **not** what happens here), the pod
  restarts. Useful as a backstop, not as a fix.

---

## Next Steps

In rough priority order:

1. **Validate the signature #6 fix end-to-end with `run8`** *(done)*.
   Expectation was met at the C++ trace level (33/33 `assignBufferIndex`
   `begin → step → end`, 3/3 `gen_request_sync_not_ready_buffers_freed`),
   but the harness still reports `NO RECOVERY` because the wedge has
   shifted to signature `#7` (NIXL UCX-internal `pthread_mutex_lock`).
   See Phase 10 of the timeline for full details.
2. **Implement focused unit tests for signatures #5 and #6** *(done)*.
   - `#5` test (`test_cancel_queued_gen_request_fulfills_receiver_future`)
     keeps the receiver worker busy with a first orphan generation request,
     then enqueues and cancels a second; pre-fix `Broken promise` lands on
     stderr, post-fix the cancelled request reaches `kDISAGG_TRANS_ERROR`
     cleanly. Bundled into [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672).
   - `#6` test (`test_cancelled_after_ready_does_not_leak_recv_buffer_index`)
     uses the NIXL backend, drives a cancelled-after-ready transfer once,
     then issues a follow-up generation request on a worker thread with a
     10s probe timeout; pre-fix the worker thread stays alive past the
     timeout, post-fix the follow-up completes normally. Bundled into
     [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673).
3. **Split signatures #4, #5, and #6 fixes into reviewable PRs** *(done)*.
   - `#4`: chained pair [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674) (test) → [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671) (fix).
   - `#5`: combined test + fix in [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672).
   - `#6`: combined test + fix in [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673), chained on `#13640` (the `#1` fix is a prerequisite for the `!isReady` early-return path to be reachable).
4. **Backport** `#13119` (request-level error propagation) to the `rc11`
   field branch so future field hits are easier to attribute to a specific
   signature.
5. **Add an integration test** that drives the disagg HTTP path with the
   long-prompt + retries + cancels load shape used by the local burst
   harness. This is the single largest coverage gap surfaced by this
   investigation.
6. **Audit other early-return paths** in the C++ disagg transceiver for
   leaks of similarly cv-waited resources (other concurrence resources,
   request-side state, etc.). The `#6` pattern — "fix the visible failure
   path on side A, surface a resource leak on side B" — is likely to repeat
   if other paths share the same RAII gap.
7. **Enforce `kv_transfer_timeout_ms` as a hard deadline** on the
   transceiver's blocking entry points — *as the TRT-LLM-side
   fallback / mitigation for signature `#7`, not as the ultimate fix.*
   As of `rc11` the knob is fully plumbed through Python config, C++
   config class, serialization, getters/setters — but **never consumed
   in the request execution path** of `cacheTransceiver.cpp` /
   `dataTransceiver.cpp`. A deadline on the TRT-LLM side converts the
   silent wedge into structured per-request `kNETWORK_ERROR`s, but it
   cannot by itself unwedge the NIXL UCX-internal `pthread_mutex_lock`
   on the ctx side; the underlying mutex stays held, and the deadline
   only gives the *caller* an escape, not the *callee*. Concretely:

   | path | makes the local reproducer pass? | who owns it |
   |---|---|---|
   | TRT-LLM deadline alone | **no** — symptom shifts from silent hang to fast per-request errors; the wedge persists | TRT-LLM |
   | TRT-LLM deadline + orchestrator (e.g. K8s liveness, Dynamo) restart on sustained error rate | **yes** — restart cycle clears the wedge in seconds-to-minutes | TRT-LLM + orchestrator |
   | NIXL/UCX root-cause fix for the internal `pthread_mutex_lock` deadlock | **yes** — clean fix at the right layer | NIXL/UCX team |
   | TRT-LLM in-process NIXL agent reset on timeout | theoretically yes, but heavy and may not be safe to recreate the agent while a thread is stuck on its internal mutex | TRT-LLM (substantial design work) |

   The deadline is still worth landing because:
   - it bounds `mRequesterFutures` growth (no eventual OOM on the
     gen side under sustained NIXL deadlock);
   - it gives orchestration a real signal (high error rate) instead of
     a silent stall;
   - it is the prerequisite for any retry / restart loop above
     TRT-LLM.

   See "Effort estimate for the deadline enforcement" below.
8. **File a NIXL/UCX bug for signature `#7`** — *this is the ultimate
   fix.* The canonical reproducer is the `ctx_worker_*_gdb.txt` stack
   from the `run8` archive. The wedge originates inside the NIXL UCX
   plugin (`nixlUcxThreadEngine::getNotifs()` blocked on
   `pthread_mutex_lock`), not in TRT-LLM. The local burst harness will
   continue to wedge until this lands, regardless of how thorough the
   TRT-LLM-side defenses are. Item 7 above is a TRT-LLM-side fallback
   for the same signature; item 8 here is the only path that makes the
   reproducer pass cleanly without an external orchestrator restart
   loop.
9. **Rename the misleading `drop_without_fulfill` trace marker.** As
   noted in Phase 10, the marker fires immediately *before* the
   signature `#1` cancellation handler that already fulfills the
   promise correctly. The 3 events per `run8` are the fix path doing
   its job, not actual drops. Renaming to
   `cancelled_after_ready_handled` removes the false-positive in future
   forensic readings.

### Effort estimate for the deadline enforcement (Next Steps item 7)

This section sizes the **TRT-LLM-side fallback / mitigation for
signature `#7`** — not the ultimate fix. The ultimate fix is the
NIXL/UCX root-cause bug (Next Steps item 8). The deadline work
decomposes into four implementation layers, each with its own
trade-off between effort, blast radius, and how much of the wedge
class it actually covers. Calendar estimates assume one engineer
familiar with the disagg transceiver code path.

#### Layer A — Python-level deadline + structured cancel (1 engineer, ~1 week)

**Where:** `tensorrt_llm/_torch/pyexecutor/py_executor.py` and
`kv_cache_transceiver.py` — the existing per-iteration loops that
already call `check_context_transfer_status()` /
`check_gen_transfer_status()`.

**How:**
1. Track `req.kv_transfer_started_at` when the request enters the
   transceiver path.
2. After each non-blocking poll cycle, scan in-flight requests; if
   `now - started_at > kv_transfer_timeout_ms`, call
   `transceiver.cancel_request(req)`, mark the request
   `kDISAGG_TRANS_ERROR`, fulfil the Python-side completion future
   with a structured timeout exception, and remove from the in-flight
   tracker.
3. Surface the timeout cleanly to the OpenAI-style HTTP layer as a
   structured 5xx (e.g. `kNETWORK_ERROR` body), so the orchestrator
   sees real signal.

**Pros:**
- ~50–100 lines of Python; no C++ rebuild.
- Testable with the existing `test_kv_cache_transceiver.py` plus a
  small mock-NIXL fixture.
- Immediately surfaces the silent wedge as a clean per-request error.

**Cons:**
- The C++ side already restricts `cancelRequest()` to requests that
  are *not currently being processed* (see
  `dataTransceiver.cpp:431`); the request actually wedged inside
  `nixlAgent::getNotifs()` will return "Cannot cancel". Python
  effectively has to "abandon and report timeout" without the C++
  side cleaning up.
- The wedged C++ thread keeps consuming its slot. Layer A buys a
  bounded number of clean errors but does *not* buy sustained
  recovery; for that, the orchestrator must restart the wedged pod.

**Verdict:** This is the right starting point. Cheap, low risk,
unblocks orchestrator-driven recovery, immediate operability win.

#### Layer B — C++-side deadline on slice-able blocking paths (1 engineer, ~2 weeks)

**Where:** Every `cv.wait(...)` and unbounded `future.get()` in
`dataTransceiver.cpp`, `cacheTransceiver.cpp`, and
`baseTransBuffer.cpp`. The four obvious candidates are the
`assignBufferIndex()` `cv.wait`, the `CacheSender::Impl::response()`
outer `mSenderCv.wait`, the inner ready-signal recv in
`CacheReceiver::Impl::sendRequestInfo()`, and the
`CacheTransceiver::checkGenTransferStatus()` future probe (already
fixed for `atLeastNum=1` via the signature `#4` patch — extend to a
deadline-aware variant).

**How:**
1. Add a `std::chrono::steady_clock::time_point deadline` (or
   `std::optional<int> timeoutMs`) parameter to the relevant private
   methods.
2. Replace each `cv.wait(lk, predicate)` with a
   `cv.wait_for(lk, slice_ms, predicate)` loop that checks
   `mTerminate || past_deadline` on each slice.
3. On deadline expiry: set the request future with
   `kNETWORK_ERROR`, set state to `kDISAGG_TRANS_ERROR`, *free any
   reserved buffer indices via the same RAII helper used for the
   signature `#6` fix*, continue serving other requests.
4. Add unit tests for each deadline path (mirroring the
   `test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
   regression test added for signature `#4`).

**Pros:**
- Real per-request timeout behaviour across every TRT-LLM-owned
  blocking primitive.
- Cleans up properly on timeout — closes the same RAII gap that
  signature `#6` exposed.
- Defends against many slow-path hangs, not just NIXL deadlocks.

**Cons:**
- Does **not** cover the NIXL `pthread_mutex_lock` wedge: that's a
  single C call into NIXL, not a `cv.wait` in TRT-LLM code. Slicing
  only works on blocking primitives that TRT-LLM owns.
- Larger surface for race conditions; needs careful review.
- Requires C++ rebuild, full test sweep.

**Verdict:** Right follow-up to Layer A. Closes the architectural
"every blocking wait must be interruptible" invariant from the
Architectural Reflections section, and bounds the failure surface for
all TRT-LLM-owned blocking calls.

#### Layer C — `std::async` watchdog around NIXL calls (1 engineer, ~2 weeks, with caveats)

**Where:** Each call into a NIXL primitive that today blocks
indefinitely — primarily `nixlAgent::getNotifs()`,
`nixlAgent::genNotif()`, and the underlying UCX waits. Wrap with the
`std::async` + `future.wait_for(timeout)` pattern.

**How:**
```cpp
auto fut = std::async(std::launch::async, [&]() { return recvRequestInfoImpl(); });
if (fut.wait_for(timeout) == std::future_status::ready) { return fut.get(); }
throw TimeoutException(...); // detached worker keeps running until NIXL returns (or never)
```

**Pros:**
- Caller actually escapes; can serve other requests until thread pool
  exhaustion.

**Cons:**
- **Thread leak per timeout.** Eventually OOM on threads if NIXL
  truly stays wedged.
- Doesn't prevent further wedges; subsequent NIXL calls hit the same
  internal mutex.
- Higher per-call overhead than slicing.
- Detecting "NIXL agent is poisoned, refuse new traffic" requires a
  higher-order recovery contract — that itself is design work
  (Layer D).

**Verdict:** Pursue only if Layer A + Layer B aren't enough and the
NIXL fix is far away. Otherwise the cost-to-benefit is poor compared
with the orchestrator-restart contract.

#### Layer D — In-process NIXL agent reset on timeout (1 engineer, ~3–4 weeks, design-heavy)

**Where:** New "NIXL recovery" subsystem that detects sustained
timeouts, tears down the wedged NIXL agent, recreates it, and
re-establishes connections with all peers.

**Pros:**
- Actually unwedges the system without external orchestration.

**Cons:**
- Likely needs NIXL API support for clean shutdown of an agent that
  has threads stuck on internal mutexes (this API may not exist
  today).
- Potentially unsafe — recreating an agent while internal threads are
  blocked on its mutex risks state corruption.
- Significant design + cross-team coordination + extensive testing.

**Verdict:** Last resort. If NIXL ships a fix for the underlying
`pthread_mutex_lock` deadlock (Next Steps item 8), Layer D becomes
unnecessary.

#### Recommended order

1. **Layer A** (1 week) — land the Python-level deadline + structured
   cancel + per-request 5xx. Pair with the orchestrator-restart
   contract to deliver actual recovery on the field reproducer.
2. **Layer B** (2 weeks) — extend the deadline into C++ for every
   TRT-LLM-owned blocking primitive. Closes the architectural gap and
   prevents the "fix-on-side-A surfaces leak-on-side-B" pattern from
   recurring.
3. **NIXL/UCX bug** (Next Steps item 8) — the only path that makes
   the local reproducer pass *without* an orchestrator restart loop.
   Out of TRT-LLM scope but should be filed in parallel with Layer A.
4. Layer C and Layer D should not be pursued unless Layers A+B and
   the NIXL fix all fall through.

---

## File / Branch Index

- Reproducer harness: long-prompt burst + recovery-probe script in the
  `local/rc11-disagg-repro` worktree under `.repro/harness/onepair/`.
- Run logs:
  - `run4` (stock `rc11`): permanent wedge. Archived at
    `~/trtllm-experiment-archives/run4_final_20260429_211126/`.
  - `run5` (post-signature-#4 fix): permanent wedge persists; surfaces
    signatures #5 and #6. Archived at
    `~/trtllm-experiment-archives/run5_fixsig4_final_20260429_224332/`.
  - `run6` (post-signature-#5 fix + first round of signature-#6
    instrumentation): permanent wedge persists; pinpoints the stall to a
    single in-progress request stuck after `gen_request_sync_begin` but
    before `gen_send_request_buffer_info_*`. Archived at
    `~/trtllm-experiment-archives/run6_recvfix_final_20260429_233258/`.
  - `run7` (fine-grained signature-#6 instrumentation across the
    `sendRequestInfo()` body): permanent wedge persists; pinpoints the
    stall to the first `assignBufferIndexForRecv()` call for the next
    request after a leak. Logs preserved under `.repro/logs/` in the
    `local/rc11-disagg-repro` worktree (`run7_sig6_instr/`).
  - `run8` (post-signature-#6 fix end-to-end validation): completed.
    Signature `#6` fix verified at the C++ trace level; harness-level
    wedge persists with the wedge driver shifted to a NIXL UCX-internal
    mutex contention on the ctx side. Logs and post-mortem stack dumps
    archived at
    `~/disagg-investigation-archive/run8_sig6_fix/` (includes
    `pyspy/gen_worker_*_gdb.txt`, `pyspy/ctx_worker_*_gdb.txt` —
    the latter is the canonical NIXL deadlock evidence).
- New unit tests:
  - `cpp/tests/unit_tests/runtime/radixBlockTreeTest.cpp` (signature #2).
  - `tests/unittest/others/test_kv_cache_transceiver.py::test_cancel_request_in_transmission_fulfills_sender_future`
    (signature #1).
  - `tests/unittest/others/test_kv_cache_transceiver.py::test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
    (signature #4).
- Trace gating env summary:
  - `TRTLLM_DISAGG_TRACE_PROMISE` — sender / receiver promise lifecycle and
    `checkGenTransferStatus` selection / get markers.
  - `TRTLLM_DISAGG_TRACE_TRIE` — trie attach / detach / cascade-prune
    markers.
  - `TRTLLM_DISAGG_TRACE_OPTIONAL` — Python event-loop exception summaries
    around the optional accessors implicated in signature #3.
  - `TRTLLM_DISAGG_TRACE_BLOCK` — Python watchdog around blocking
    transceiver calls; `TRTLLM_DISAGG_TRACE_BLOCK_TIMEOUT_S` controls the
    threshold (default 10s, repro uses 5s).
