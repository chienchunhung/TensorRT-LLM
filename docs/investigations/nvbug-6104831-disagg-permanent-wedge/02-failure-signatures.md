# 02 — Failure Signatures (`#1` – `#7`)

The customer-visible wedge is the union of seven distinct failure
signatures in the disaggregated KV-cache transfer path. This file describes
each one in detail: how it manifests, where it lives in the code, what
triggers it, what the fix is (or what we still need to learn), and the
regression test if any.

For the *deeper* defect-class layering that explains *why* these seven
signatures exist, see [`03-defect-class-stack.md`](03-defect-class-stack.md).
For the chronological order in which they were discovered and isolated,
see [`05-investigation-timeline.md`](05-investigation-timeline.md).

For convenience, the signature labels at a glance:

| Signature | Short name | Where it lives | First-found via |
|---|---|---|---|
| **#1** | Sender-side `Broken promise` after ready signal | `CacheSender::Impl::sendResponse` (cancelled-after-ready path) | Production logs (Dynamo `rc11` deploy) |
| **#2** | Trie `cascade prune: parent did not find this node as a child` assertion | `templatedTrie.h::clearNode` / `KVCacheBlock` lifecycle | Field investigation report + C++ unit-test stress probe |
| **#3** | Decode-side `RuntimeError: bad optional access` | C++ `std::optional::value()` inside disagg gen path, surfaced through Python | Production logs (Dynamo `rc11` deploy) |
| **#4** | Gen-side blocking hang in `CacheTransceiver::checkGenTransferStatus()` with `atLeastNum=1` | `cacheTransceiver.cpp` unconditional `future.get()` on selected-but-unready future | Local `trtllm-serve` 1P1D repro + Python thread-stack dump |
| **#5** | Receiver-side `Broken promise` from queued cancel | `CacheReceiver::Impl::cancelRequest()` erasing queued request without fulfilling promise | Post-`#4`-fix C++ trace logs |
| **#6** | Recv-buffer index leak via `!isReady` early-return; subsequent receives wedge in `BaseTransBufferManager::assignBufferIndex()` | `cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp` (unbounded `cv.wait`) leaked from `CacheReceiver::Impl::requestSync()` (`!isReady` path) | Fine-grained C++ instrumentation across `sendRequestInfo()` body in `run7` |
| **#7** | A bug class in `CacheSender::Impl::*` C++ code, observed as four manifestations: deadlock in `response()`; ctx mpi4py executor exits; Python-`getattr` SIGSEGV; first-request SIGSEGV in `handleAsyncSend` (PR `#13056` + UCX) | `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp::CacheSender::Impl::*` | `gdb` post-mortems of `run8`, `pr13056_run1`, `rc11_ucx_run1`, `run9`, `run10`, `run14`/`run14c` |

> **Read this caveat before reading anything else.** Signatures `#1`
> through `#6` are real TRT-LLM bugs and the chained PRs land their
> fixes. Signature `#7` is the residual class of bugs that fires under
> the cancel-during-transfer load shape after `#1`–`#6` are individually
> fixed. Phases 10–11 of the timeline initially classified `#7` as a
> NIXL UCX-plugin internal mutex deadlock (out of TRT-LLM scope).
> Phase 12 falsified that classification: the same `pthread_mutex_lock`
> wedge frame inside `CacheSender::Impl::response()` fires identically
> on TRT-LLM's direct UCX backend (`rc11_ucx_run1`), in a process
> where `libnixl.so` is not loaded at all. Phase 13 broadened the
> framing further: `run9` exposes a Python `getattr` SIGSEGV downstream
> of our cancellation fixes and `run10` (PR `#13056` + direct UCX)
> exposes a synchronous C++ SIGSEGV inside
> `CacheSender::Impl::handleAsyncSend` on the *first* request. Phase 14
> showed that this was an argument-evaluation-order hazard introduced
> when PR `#13056` changed `Response::mRequest` from a raw pointer to
> a `std::shared_ptr`. Sig `#7` is therefore a **TRT-LLM-side bug class
> in `CacheSender::Impl::*`**, with at least four observed
> manifestations across two transports and three fix bundles, **fixable
> entirely in TRT-LLM**.

---

## Signature #1 — Sender-side `Broken promise` after ready signal

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

---

## Signature #2 — Trie `cascade prune` assertion

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

This signature is **independent** of the cancellation/cleanup bug class
that drives the rest of this investigation. It's an eviction-driven trie
invariant violation that can fire under any sustained workload with heavy
KV-block eviction; the customer load shape just happens to drive frequent
eviction.

---

## Signature #3 — Decode-side `RuntimeError: bad optional access`

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
`Broken promise` from signature `#1` may be the upstream trigger that
leaves the request in this half-initialised state.

**Status:** Not fixed yet. Likely to disappear (or change shape) once
signatures `#1`, `#4`, and `#5` are all in place, because those are the
conditions under which the half-initialised state is reached. After the
`run8` post-mortem there is a second candidate trigger to keep in mind:
a NIXL-layer wedge (signature `#7`) can also strand a request mid-transfer
and leave receiver-side state in the same half-initialised shape, so a
field hit *after* the chained PRs land should be checked against the
`#7` signature before being attributed to a fresh TRT-LLM bug. We have
added Python-side trace logs around the gen event loop's
`_event_loop_wrapper` and `_check_disagg_gen_cache_transfer_status`
(gated on `TRTLLM_DISAGG_TRACE_OPTIONAL=1`) so the next field hit will
produce a labelled stack with the active-request summary instead of
just a bare `RuntimeError`.

**Reproducer:** None yet. This signature is currently field-only.

---

## Signature #4 — Gen-side blocking hang in `checkGenTransferStatus(atLeastNum=1)`

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
exactly what signatures `#1`, `#5`, and `#6` cause).

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

---

## Signature #5 — Receiver-side `Broken promise` from queued cancel

**Symptom (log, post-`#4` fix):**

```text
[TensorRT-LLM][ERROR] Error occurred during generation transfer for
  request 4113: std::future_error: Broken promise
[TensorRT-LLM][ERROR] [promise-trace] gen_future_get_exception
  request=4113 msg=std::future_error: Broken promise
```

Observed only **after** the signature `#4` fix: removing the gen-side
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
the structural mirror of signature `#1`, on the receiver side instead of
the sender side.

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

---

## Signature #6 — Recv-buffer index leak via `!isReady` early-return

**Symptom (gen-side request lifecycle trace, post-`#4` fix):**

The lifecycle markers we added show one specific request that reaches
`gen_request_sync_begin` but never reaches `gen_wait_ready_signal_begin`.
Every other gen request in the same window walks the full chain from
`gen_request_sync_begin → gen_wait_ready_signal_begin →
gen_wait_ready_signal_end → gen_receive_sync_begin → gen_receive_sync_end →
gen_request_promise_set_value`. Exactly one request stops between
`gen_request_sync_begin` and the next marker.

**Where it lives (confirmed in `run7`):** The visible stall is on the
*caller* of
[`cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp`](../../../cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp)'s
`BaseTransBufferManager::assignBufferIndex()` `cv.wait`, namely
[`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp)
inside `CacheReceiver::Impl::sendRequestInfo()`'s recv-buffer
reservation loop. The *leak* is in the same file's
`CacheReceiver::Impl::requestSync()` `!isReady` early-return path,
which skips the `receiveSync()` → `unformat()` →
`freeBufferIndexForRecv()` chain that would normally release the slot.

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
   inside the unbounded `cv.wait` above.

**Fix (Layer A — `sendRequestInfo` exception safety):** Track every
`(BaseTransBufferManager*, std::optional<size_t>)` pair returned by
`assignBufferIndexForRecv()`. Wrap the rest of `sendRequestInfo()` in a
`try { ... } catch (...) { freeAssignedRecvBuffers(); throw; }` block so
that any exception between assignment and the eventual `unformat()` call
releases the indices.

**Fix (Layer B — `requestSync` `!isReady` cleanup):** Mirror what
`unformat()` does on the success path. In the `!isReady` early-return
branch of `CacheReceiver::Impl::requestSync()`, iterate the session's
connections, look up each pre-assigned recv buffer ID via
`agentConnection->getPreAssignedBufferId(static_cast<uint8_t>(mgr->getBufferKind()))`,
and free it via `mgr->freeBufferIndexForRecv(id)`. The number of indices
freed is logged as
`gen_request_sync_not_ready_buffers_freed request=R count=N` so each
post-fix run shows explicitly that the leak path is now closing.

**Status:** Combined test + fix opened as
[#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673), chained on
[#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) (the `#1`
fix is a prerequisite for the `!isReady` early-return path to be
reachable in production code). `run8` validated the fix at the C++
trace level — every request now walks `gen_send_assign_buffer_begin →
step → end` cleanly (33/33), and the new
`gen_request_sync_not_ready_buffers_freed` marker fires on every
cancelled-after-ready request (3/3 in `run8`). The harness still
reaches no-recovery in `run8`; that residual wedge is signature `#7`,
not a remaining `#6` problem.

**PRs:** [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673).

> **Note for fix-approach reviewers:** PR `#13056` and PR `#13495` (via
> its #13439 base) both implement the same RAII pattern more
> idiomatically (`BufferIndexHolder` move-only type; #13495 also adds
> `TransferSession`). When approach D (combo) lands, the regression
> test from `#13673` should be rebased on top of that implementation
> and the local try/catch implementation can be dropped.

---

## Signature #7 — `pthread_mutex_lock` wedge in `CacheSender::Impl::response()` under cancel-during-transfer load

**Symptom (consistent across `run8`, `pr13056_run1`, and `rc11_ucx_run1` —
NIXL backend, comprehensive-refactor variant, and direct-UCX variant
respectively):**

The local burst-harness reports `NO RECOVERY after 180s idle --
permanent wedge` after the `CONC=16` 60-second burst completes. All
five recovery probes (idle = 30 / 60 / 90 / 120 / 180 s) return silent
`ReadTimeout` — no structured 5xx, no error message, just no response.
TRT-LLM trace markers (where present) show clean lifecycle progression
through every signature `#1`–`#6` code path; the wedge fires *after*
those signatures are individually fixed.

**Where it lives (cross-variant analysis):** The wedge frame is
**always** the same:

```text
ctx-side dataTransResp thread:
  pthread_mutex_lock
    ← tensorrt_llm::batch_manager::CacheSender::Impl::response()
                                    [libtensorrt_llm.so]
```

What the deeper frames look like differs by transport:

| Variant | Deeper frames in stack | Other relevant threads in process |
|---|---|---|
| **NIXL backend** (`run8`, `pr13056_run1`) | `recvRequestInfo` → `AgentConnectionManager::recvConnectionAndRequestInfo` → `updateUnhandledNotifications` → `NixlTransferAgent::getNotifiedSyncMessages` → `nixlAgent::getNotifs` → `nixlUcxThreadEngine::getNotifs` → `pthread_mutex_lock` | NIXL plugin spin / UCX shared threads from `libplugin_UCX.so` and `libnixl.so` |
| **Direct UCX backend** (`rc11_ucx_run1`) | (deeper frames inlined; only `response()` and `pthread_mutex_lock` visible) | `ucxx::Worker::progressOnce` → `ucp_worker_progress` (UCX core) and `UcxConnectionManager`'s ZMQ control thread, **all from `libtensorrt_llm_ucx_wrapper.so`**. **No NIXL plugin loaded in this process at all.** |

So the NIXL plugin is **not** the locus of the bug; it was misidentified
in the `run8` analysis because all three runs available at that point
used the NIXL backend. The actual locus is `CacheSender::Impl::response()`
in TRT-LLM, and the mutex it's blocked on is shared between the two
transport paths.

**Most plausible mutex (source-code inference):** Looking at
`CacheSender::Impl::response()` in
[`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp:674-745),
the function opens its main loop with:

```cpp
void response() noexcept {
    while (!mTerminate || !mAnyReady) {
        if (!mAnyReady) {
            std::unique_lock lk(mCondMutex);   // ← pthread_mutex_lock(mCondMutex)
            mSenderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
        }
        ...
    }
}
```

The `std::unique_lock` constructor invokes `pthread_mutex_lock(mCondMutex)`.
The frame pattern `response() → pthread_mutex_lock` with no intermediate
frames matches this site best. **Confidence: medium-high** — confirmed
by code structure, not yet by runtime register inspection of the mutex
address.

### Variants

**Variant A: deadlock in `response()` `mCondMutex` cv-wait** (the
"canonical" sig `#7` manifestation, all NIXL `run8` and direct-UCX
`rc11_ucx_run1` evidence).

**Variant B: ctx-side mpi4py executor exits unexpectedly.** A separate
`rc11+UCX` run (`rc11_ucx_run2_diag`) showed a different but related
failure: the ctx-side mpi4py.futures.server worker process exited
during the burst, leaving the ctx-serve Python proxy alive (so `/health`
still returns 200) but with no executor backend. From the harness's
perspective the wedge is identical — recovery probes time out
silently — but the underlying mechanism is process exit rather than
deadlock. Both manifestations point at the same code region.

**Variant C: Python-`getattr` SIGSEGV downstream of cancellation
cleanup (Phase 13, `run9`).** With our chained fixes for sig
`#1`/`#4`/`#5`/`#6` applied and the same UCX harness, the ctx mpi4py
worker no longer deadlocks; instead it `SIGSEGV`s in
`_PyObject_GenericGetAttrWithDict` → `PyObject_GetAttr` →
`_PyEval_EvalFrameDefault` at iter 92 of the burst, with the sig `#1`
fix path (`promise_set_exception` for `cancelled_after_ready_signal`)
captured cleanly in the promise-trace log immediately before. This
looks like a Python wrapper around a C++ object being destructed by
one thread while another still holds a reference; the wedge in earlier
runs was masking it.

**Variant D: null `shared_ptr<LlmRequest>` deref in `handleAsyncSend`
on the first request (Phase 13, `run10` → root-caused in Phase 14).**
With PR `#13056`'s comprehensive `shared_ptr` lifetime + RAII +
deadline enforcement applied and the same UCX harness, the ctx mpi4py
worker `SIGSEGV`s synchronously inside
`CacheSender::Impl::handleAsyncSend(AsyncSendResource&)` on the
**first** sanity-probe request — no concurrency, no burst, no
cancellation. **Phase 14 root-caused this as a C++ argument-evaluation-order
hazard introduced by changing `Response::mRequest` from raw `LlmRequest*`
to `std::shared_ptr<LlmRequest>`:**

```cpp
// dataTransceiver.cpp::handleAsyncSend, line 514:
sendAndRemoveResponse(resp.mRequest->mRequestId, std::move(resp));
```

When `Response::mRequest` was a raw pointer, the move-construction
copied the pointer value, so `resp.mRequest->mRequestId` was safe under
either evaluation order. Once it became a `shared_ptr`, `std::move(resp)`
move-constructs the field and leaves `resp.mRequest` empty. C++ does
not guarantee left-to-right argument evaluation; if the compiler
evaluates `std::move(resp)` first, the second-argument read of
`mRequestId` dereferences a moved-from null `shared_ptr`. The minimal
fix:

```cpp
TLLM_CHECK(resp.mRequest != nullptr);
auto const reqId = resp.mRequest->mRequestId;
sendAndRemoveResponse(reqId, std::move(resp));
```

This is part of approach D (the combo) and was the breakthrough that
let the combo make recovery progress.

### Reproducer artifacts

Five independently-built TRT-LLM binaries reach a manifestation in this
bug class:

- `~/disagg-investigation-archive/run8_sig6_fix/pyspy/` (NIXL backend;
  this investigation's chained PRs) — variant A.
- `~/disagg-investigation-archive/pr13056_run1/pyspy/` (NIXL backend;
  PR `#13056` independent stack) — variant A.
- `~/disagg-investigation-archive/rc11_ucx_run1/pyspy/` (direct UCX
  backend; this investigation's chained PRs) — variant A. **This is
  the single most important artifact for the deadlock variant** —
  wedge frame in a process with no `libnixl.so` loaded.
- `~/disagg-investigation-archive/run9_rc11_ourfixes_ucx_segfault/`
  (direct UCX, this investigation's chained PRs) — variant C.
- `~/disagg-investigation-archive/run10_pr13056_ucx_segfault_handleAsyncSend/`
  (direct UCX, PR `#13056`) — variant D. **The cleanest evidence that
  the bug class is purely TRT-LLM-internal** (synchronous C++ crash,
  zero transport-library frames).

### Fix scope

Three layers of remediation, in priority order:

1. **Pin down the exact mutex in the deadlock variant** (~30–60 min)
   by attaching `gdb` to a live wedge, dumping `info registers` for
   the `dataTransResp` thread, examining the `pthread_mutex_t` at
   `$rdi`, and finding which other thread's TID matches the mutex's
   `__owner` field. Phase 13 attempted this but the wedge changed
   character to a SIGSEGV under the rc11+our-fixes+UCX configuration
   (`run9`); a follow-on attempt should re-run with the NIXL backend
   or a lower `CONC` to reach the deadlock variant cleanly.
2. **Fix the deadlock in `CacheSender::Impl`.** Once the holder is
   identified, the fix is most likely a lock-ordering or
   release-before-blocking-call change in the offending code path.
   This is **fixable in TRT-LLM** without needing a NIXL or UCX
   change.
3. **Fix the `handleAsyncSend` eval-order bug** (Variant D —
   confirmed in Phase 14, included in approach D's combo PR
   `#13713`).

**Status:** Re-classified in Phase 12 from "out-of-TRT-LLM-scope NIXL
plugin bug" to "TRT-LLM-side `CacheSender::Impl` mutex bug, exposed
by the cancel-during-transfer load shape across both NIXL and direct
UCX backends". Phase 13 broadened to "a class of `CacheSender::Impl::*`
bugs with at least four observed manifestations". Phase 14 confirmed
and fixed Variant D; Variant A's exact mutex address still requires a
live `gdb` capture session.

**PRs:** Variant D fix is included in approach D's combo
([#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713)). Variants
A, B, C are not yet individually addressed; combo D's other mechanisms
(per-request cancel-flag, RAII buffer holders, NIXL release hook,
Python idempotency guards, NIXL transport path) empirically eliminate
or work around the remaining wedges through `CONC=64` on NIXL and
`CONC=32` on direct UCX. Direct-UCX `CONC=64` still wedges.
