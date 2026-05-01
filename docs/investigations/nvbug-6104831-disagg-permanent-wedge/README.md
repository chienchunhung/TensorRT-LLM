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
  - `local/sig4-checkgen-nonblocking-test` (signature #4 reproducer test)
  - `local/sig4-checkgen-nonblocking-fix` (signature #4 fix)
  - `local/sig5-recv-cancelrequest-fulfill` (signature #5 combined test + fix)
  - `local/sig6-recv-buffer-leak` (signature #6 combined test + fix, chained on `local/sig1-broken-promise-fix`)
  - `local/rc11-disagg-repro` (isolated `rc11` worktree with cumulative fixes
    + instrumentation; the testbed used for `run4`–`run8`)
- **Related PRs:**
  - [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) — signature #2
    reproducer test
  - [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) — signature #2
    fix
  - [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) — signature #1
    reproducer test
  - [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) — signature #1
    fix
  - [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674) — signature #4
    reproducer test
  - [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671) — signature #4
    fix (carries the test commit too as 2 commits; targets `main` directly)
  - [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672) — signature #5
    combined test + fix
  - [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673) — signature #6
    combined test + fix (chained on `#13640` because the `!isReady` early-return
    path is only reachable once `#13640` sends `is_ready=false`)
- **Companion fixes in main (not in `rc11`):**
  - [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) — request-level
    error propagation (cleaner failure visibility, not a fix for the wedge)
  - [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — fatal engine
    detection / pod restart (mitigation for silent wedges, not a fix for the
    wedge)
- **Status:** All six TRT-LLM signatures (`#1`, `#2`, `#4`, `#5`, `#6`)
  have chained PRs in review. Signature `#3` is field-only and is
  expected to disappear or change shape once `#1`, `#4`, and `#5` land
  (a `CacheSender::Impl` mutex wedge from `#7` is also a candidate
  cause; see that signature's section). Signature `#7` (`pthread_mutex_lock`
  wedge in `CacheSender::Impl::response()`) was initially classified
  as a NIXL plugin bug in Phases 10–11 and reclassified in Phase 12
  as a TRT-LLM-side mutex bug exposed by both NIXL and direct UCX
  backends. The recommended next action is **Next Steps item 7** —
  pin down the exact mutex with a live `gdb` register inspection and
  fix the offending lock-ordering in `CacheSender::Impl`. Item 8
  (`kv_transfer_timeout_ms` deadline) becomes defence-in-depth that
  bounds this class of bug going forward.

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
| **#7** | `pthread_mutex_lock` wedge in `CacheSender::Impl::response()` under cancel-during-transfer load; also a variant where the ctx-side mpi4py executor exits unexpectedly | `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp::CacheSender::Impl::response()`; most plausibly on `mCondMutex` at the top of the response loop | `gdb` post-mortems of `run8` (NIXL backend), `pr13056_run1` (NIXL backend, comprehensive refactor), and `rc11_ucx_run1` (direct UCX backend) all show the same wedge frame |

The mapping of fixes is summarised at the end, in
[Signature ↔ PR Map](#signature--pr-map).

> **Read this caveat before reading anything else.** Signatures `#1`
> through `#6` are real TRT-LLM bugs and the chained PRs land their
> fixes. Signature `#7` is the residual wedge that fires under the
> cancel-during-transfer load shape after `#1`–`#6` are individually
> fixed. Phases 10 and 11 of the timeline initially classified `#7`
> as a NIXL UCX-plugin internal mutex deadlock (out of TRT-LLM
> scope). Phase 12 falsifies that classification: the same
> `pthread_mutex_lock` wedge frame inside
> `CacheSender::Impl::response()` fires identically on TRT-LLM's
> direct UCX backend (`rc11_ucx_run1`), in a process where
> `libnixl.so` is not loaded at all. Sig `#7` is therefore a
> **TRT-LLM-side mutex bug** in `CacheSender::Impl`, exposed by both
> NIXL and direct UCX backends, **fixable in TRT-LLM**. The exact
> mutex (most likely `mCondMutex`) and the holder thread still need
> to be pinned down via runtime register inspection. Full evidence is
> in Phases 10–12 of the timeline.

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
  disagg HTTP layer) reproduce signatures `#1`, `#2`, `#4`, `#5`, and
  `#6` (the latter two via the new tests added in `#13672` and
  `#13673` respectively). Signatures `#3` and `#7` are field-only:
  `#3` requires the full HTTP path with cancellation and retries, and
  `#7` requires the NIXL/UCX runtime under a contention pattern that
  hasn't been mock-injected yet.

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

### Signature #6 — Recv-buffer index leak via `!isReady` early return; subsequent receives wedge in `assignBufferIndexForRecv()`

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
The control-path notify in
[`cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp`](../../../cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp)'s
`AgentConnection::sendRequestAndBufferInfo(...)` was an early
suspect but was ruled out by the `run7` per-marker accounting.

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

### Signature #7 — `pthread_mutex_lock` wedge in `CacheSender::Impl::response()` under cancel-during-transfer load

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
This is the only direct `pthread_mutex_lock` callsite in `response()`'s
own body that would surface as a top-level frame (other lock
acquisitions are inside helper functions like `recvRequestInfo`,
`sendResponse`, `getCurrentResponse`). The frame pattern `response() →
pthread_mutex_lock` with no intermediate frames matches this site
best. **Confidence: medium-high** — confirmed by code structure,
not yet by runtime register inspection of the mutex address.

**What's likely holding `mCondMutex`:** Several TRT-LLM-side methods
acquire `mCondMutex` for short critical sections:
`notifyResponseReady()` (sets `mAnyReady=true` + notify), the
cancel-after-ready handler in `sendResponse()` (line 666: sets
`mAnyReady=false`), `terminate()`, and the `~Impl()` destructor. Under
the cancel-during-transfer load shape, multiple concurrent paths can
interact via these mutexes (`mCondMutex`, `mSenderMutex`,
`mMtxForMap`) plus the cv-wait re-acquisition that follows
`pthread_cond_wait`. Pinning down which thread holds the lock and why
requires runtime register inspection of the wedged thread's `%rdi`
mutex address and a corresponding scan of `pthread_mutex_t::__owner`
across all threads.

**Variant: ctx-side mpi4py executor exits unexpectedly.** A separate
`rc11+UCX` run (`rc11_ucx_run2_diag`) showed a different but related
failure: the ctx-side mpi4py.futures.server worker process exited
during the burst (the parent `orted` is still running but has zero
descendants), leaving the ctx-serve Python proxy alive (so `/health`
still returns 200) but with no executor backend. From the harness's
perspective the wedge is identical — recovery probes time out
silently — but the underlying mechanism is process exit rather than
deadlock. Both manifestations point at the same code region.

**Reproducer:** Same end-to-end 1P1D + long-prompt burst harness.
Three independently-built TRT-LLM binaries reach the same wedge:

- `~/disagg-investigation-archive/run8_sig6_fix/pyspy/` (NIXL
  backend; this investigation's chained PRs)
- `~/disagg-investigation-archive/pr13056_run1/pyspy/` (NIXL backend;
  comprehensive refactor variant)
- `~/disagg-investigation-archive/rc11_ucx_run1/pyspy/` (direct UCX
  backend; this investigation's chained PRs)

The `rc11_ucx_run1` archive is the **single most important artifact**
for this signature because it shows the wedge in a process where
`libnixl.so` is not loaded, falsifying the previous "NIXL plugin
internal mutex" hypothesis.

**Fix (TRT-LLM scope, with a NIXL-side caveat):** Two layers of
remediation, in priority order:

1. **Pin down the exact mutex** (~30–60 min) by attaching `gdb` to a
   live wedge, dumping `info registers` for the `dataTransResp`
   thread, examining the `pthread_mutex_t` at `$rdi`, and finding
   which other thread's TID matches the mutex's `__owner` field. This
   identifies whether the conflict is in `mCondMutex`,
   `mSenderMutex`, or somewhere else, and which code path is holding
   the lock while blocked downstream.
2. **Fix the deadlock in `CacheSender::Impl`.** Once the holder is
   identified, the fix is most likely a lock-ordering or
   release-before-blocking-call change in the offending code path.
   This is **fixable in TRT-LLM** without needing a NIXL or UCX
   change.

The previous narrative (file a NIXL/UCX bug, deploy `kv_transfer_timeout_ms`
as a fallback) was based on the assumption that the mutex was inside
the NIXL plugin. The `rc11_ucx_run1` evidence falsifies that
assumption. The fallback is still useful as defence-in-depth (see
Effort Estimate Layer B below — it now covers `#7` directly), but the
"NIXL/UCX bug filing" is **no longer the primary action**.

**There is still a possible NIXL/UCX-side issue.** While the wedge
itself is in TRT-LLM-owned code, the `run8` and `pr13056_run1`
backtraces showed deeper frames inside `nixlUcxThreadEngine::getNotifs()`
holding their own internal mutex. That may be a contributing factor
(NIXL plugin holding its lock while TRT-LLM holds `mCondMutex`,
producing a cross-library lock-ordering deadlock) or an unrelated
secondary issue. Worth noting in the NIXL/UCX bug filing but no
longer the main story.

**Status:** Identified, characterised, and documented. Re-classified
from "out-of-TRT-LLM-scope NIXL plugin bug" to "TRT-LLM-side
`CacheSender::Impl` mutex bug, exposed by the cancel-during-transfer
load shape across both NIXL and direct UCX backends". A targeted
runtime gdb session to pin down the exact mutex address and holder
is the recommended next investigation step.

**PRs:** No TRT-LLM PR open yet. Once the exact mutex and holder are
identified, the fix is expected to be a small, surgical change in
`CacheSender::Impl` (estimated ~30–80 lines).

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

(In hindsight, the underlying terminal driver of the Phase-5 wedge was
already signature `#7` — the NIXL UCX-internal `pthread_mutex_lock`
deadlock identified in Phase 10 — but `#4` was the *visible* TRT-LLM-side
symptom because the gen event loop was self-blocking before any of the
later layers could surface. Fixing `#4` was a prerequisite for *seeing*
`#5` and `#6`, which were prerequisites for *seeing* `#7`.)

### Phase 6 — Signature #4 fix and regression test (T+3 days)

- Fix: in the non-`blockAll` path, probe each selected future with
  `wait_for(0)` and skip if not ready.
- New unit test
  `test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
  fails on stock `rc11` (asserts the wrong behaviour) and passes post-fix.

This is the regression test for signature `#4`. It was subsequently
split out of `local/rc11-disagg-repro` and submitted as the chained
pair [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674)
(test) → [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671)
(fix).

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

### Phase 10 — `run8` validation: signature `#6` confirmed fixed, signature `#7` identified as a NIXL/UCX-layer deadlock (T+6 days)

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
  in a follow-up cleanup (Next Steps item 10).
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
- Signature `#7` was *initially* hypothesised in this phase as the NIXL
  UCX-internal `pthread_mutex_lock` deadlock (a non-TRT-LLM bug). Phase 12
  below revises this — `#7` is actually a wedge in
  `CacheSender::Impl::response()` itself, in TRT-LLM-owned code, exposed by
  both NIXL and direct UCX backends.

The chained TRT-LLM PRs close every TRT-LLM-side bug in the
five-signature class identified by Phases 1–9. They are necessary but
not sufficient on their own: under the cancel-during-transfer load
shape they reveal `#7`, an additional TRT-LLM-side mutex bug whose
root cause was clarified in Phase 12.

### Phase 11 — Cross-validation against an independent fix stack: signature `#7` survives a comprehensive deadline + RAII refactor (T+7 days)

After `run8` left signature `#7` (NIXL UCX-internal `pthread_mutex_lock`
deadlock) as the only remaining unexplained wedge, the next question
was whether that conclusion is robust against *how* the TRT-LLM-side
bugs are fixed. Concretely: do the surgical patches in the chained PRs
documented above (`#13571 / #13572`, `#13639 / #13640`, `#13674 /
#13671`, `#13672`, `#13673`) leave any window for a NIXL-adjacent bug
to be misclassified as `#7` when it might actually be a TRT-LLM-side
issue we missed?

To answer this, the same 1P1D long-prompt burst harness was run against
a **completely different TRT-LLM-side fix stack** developed
independently (a comprehensive refactor that introduces an end-to-end
`shared_ptr<LlmRequest>` lifetime through the transceiver, a
`BufferIndexHolder` RAII class for recv-buffer pool slots, a deadline
enforcement pass that consumes `kv_transfer_timeout_ms` at every
relevant blocking site, structured cancellation across ctx and gen
sides, and `catch (...)` hardening on the drain worker). Same `rc11`
base, same Qwen3-0.6B model, same NIXL/UCX backend, same env vars,
same harness parameters (`CONC=16`, `BURST_DUR_S=60`, recovery probes
at idle=30/60/90/120/180s).

#### Coverage map vs. the chained PRs

The independent stack addresses the same TRT-LLM-side bug class as
this investigation, just with different mechanisms:

| Sig | This investigation's fix | Independent stack's mechanism | Same bug? |
|---|---|---|---|
| `#1` | Single `set_exception(kNETWORK_ERROR)` in `CacheSender::Impl::sendResponse` cancel-after-ready branch | Multiple `set_exception` sites across all sender cancel paths + end-to-end `shared_ptr<LlmRequest>` that prevents the lifetime smell at the source | yes |
| `#2` | Reset child's `mPrevNode` in `templatedTrie.h::clearNode` | **Not addressed** (no diff in `templatedTrie.h` / `kvCacheManager.cpp`) | no — gap on this side |
| `#3` | Hypothesised downstream of `#1`/`#4`/`#5`; trace markers added | Hypothesised cleared by the comprehensive cancel-handling rewrite | indirect |
| `#4` | Single `wait_for(0)` probe in `checkGenTransferStatus` | Four `wait_for(0)` probes across `cacheTransceiver.cpp` covering both ctx and gen paths + deadline enforcement | yes |
| `#5` | Single `set_exception` in `CacheReceiver::Impl::cancelRequest` | Receiver-worker `set_exception` calls + `shared_ptr` lifetime so cancellation is structurally safer | yes |
| `#6` | Local `assignedRecvBuffers` vector + `try { ... } catch (...) { freeAssignedRecvBuffers(); throw; }` + explicit free in `requestSync !isReady` | A `BufferIndexHolder` RAII class in `baseTransBuffer.cpp/h`; comments in the source describe it as *"the core of the RAII fix … RAII covers all non-happy-path exits (not-ready, cancel, throw)"* | yes — different abstraction, same fix shape |
| `#7` | Out of TRT-LLM scope | Out of TRT-LLM scope | n/a |

So the two stacks **converge on the same TRT-LLM-side bug set** for
`#1`, `#4`, `#5`, `#6`. The independent stack is broader in two ways
(four `wait_for(0)` probes vs. one for `#4`; `shared_ptr` lifetime end
to end vs. local mutations) and narrower in one (no fix for `#2`
because `#2` lives in the trie, not the transceiver). For the local
1P1D harness, `#2` is irrelevant — that signature requires sustained
trie eviction with prefix-overlapping prompts, which the burst harness
doesn't drive. So the comparison is apples-to-apples for the wedge
under test.

#### Build + experiment setup

A fresh isolated worktree was set up at
`/home/.../TensorRT-LLM-pr13056-experiment/`, with its own
`--system-site-packages` venv (matching the recipe used for
`local/rc11-disagg-repro` to avoid the pip-torch-vs-container-torch
mismatch documented in earlier diagnostics). Build via
`scripts/build_wheel.py --build_dir=.repro/build --job_count=64
--use_ccache --skip_building_wheel` against the same NVIDIA PyTorch
container 26.02 (torch 2.11.0a0). The resulting wheel was archived
to `/home/.../trtllm_wheels/` for cross-node reuse.

#### Result

The harness reported `NO RECOVERY after 180s idle -- permanent wedge`
— **the same outcome as `run8`**. Side-by-side with the previous two
runs:

| Run | Pre-burst probe | Burst (`ok200` / errors / total) | Recovery probes | Verdict |
|---|---|---:|---|---|
| `run4` (stock `rc11`) | ok200 (8.8 s) | 8 / 12 / 20 | 5× `ReadTimeout` (60 s) | NO RECOVERY |
| `run8` (chained PRs above) | ok200 (8.3 s) | 12 / 9 / 21 | 5× `ReadTimeout` (60 s) | NO RECOVERY |
| `pr13056_run1` (independent stack) | ok200 (8.7 s) | 12 / 9 / 21 | 5× `ReadTimeout` (60 s) | NO RECOVERY |

#### Per-marker accounting (independent-stack ctx + gen workers)

The independent stack's diagnostic markers tell a striking story:

| marker | count | meaning |
|---|---:|---|
| `[buf] BufferIndexHolder AUTO_RELEASE` | 13 | RAII guard fired 13 times — but only on the happy path (request completion). |
| `[buf] assignBufferIndex CANCEL` | 0 | The new deadline-cancel path **never fired** on either worker. |
| `[buf] assignBufferIndex STILL_WAITING` | 0 | The cv-wait inside `assignBufferIndex` **never blocked long enough to even log a wait warning**. |
| `kNETWORK_ERROR` / `Broken promise` / `future_error` / `KvCacheTransferTimedOut` / `FUTURE_WAIT` / `FUTURE_JOIN` | 0 | No request-level error propagation, no broken futures, no deadline-driven failures anywhere in either log. |

The independent stack's defensive layers are **silent and idle during
the wedge**. Not because they're broken — RAII is firing 13 times for
happy-path completions — but because **the wedge is at a layer below
where any of them enforce**.

#### Stack-trace evidence (ctx worker, `pr13056_run1`)

`gdb` post-mortem of the `pr13056_run1` ctx worker shows the same
wedge-frame as `run8`:

```text
dataTransResp thread (LWP 669963):
#0  pthread_mutex_lock                      [libc.so]
#1  CacheSender::Impl::response()           [libtensorrt_llm.so]
```

In the same process, the NIXL plugin threads are present and active
in their typical positions:

| thread | top NIXL/UCX frame |
|---|---|
| nixl-comm-worker | `nixlAgentData::commWorkerInternal()` |
| nixl-ucx-shared | `nixlUcxSharedThread::run()` → `nixlUcxWorker::arm()` → `ucp_worker_arm()` (`/opt/nvidia/nvda_nixl/.../libplugin_UCX.so`) |
| `dataTransResp` | `pthread_mutex_lock` inside `CacheSender::Impl::response()` |

This is structurally identical to the `run8` post-mortem documented in
Phase 10. The `pthread_mutex_lock` frame is on a NIXL-layer mutex
(deeper frames are inlined out under the independent stack's refactor,
but the rest of the thread inventory is identical to `run8`'s
NIXL-deadlocked configuration).

#### What this proves

Two independent investigations of the same bug class — one via the
surgical chained patches documented above, one via a comprehensive
end-to-end refactor — converge on:

1. **The same set of TRT-LLM-side bugs** (`#1`, `#4`, `#5`, `#6`) that
   each stack fixes via different mechanisms.
2. **The same residual wedge** (`#7`) that neither stack can address
   because it lives in NIXL/UCX.

The convergence eliminates two alternative hypotheses:

- *"Maybe our fixes for `#1` / `#4` / `#5` / `#6` introduced a regression
  that masquerades as `#7`."* — Refuted: an independent fix stack
  produces the same wedge.
- *"Maybe `#13056`-style comprehensive deadline enforcement is enough
  to clear the field reproducer without needing a NIXL fix."* — Refuted:
  the deadline enforcement is in place, fully built into
  `CacheSender::Impl`, and it never fires during the wedge because the
  wedged thread is on a `pthread_mutex_lock` inside NIXL, not on a
  TRT-LLM-owned `cv.wait` that the deadline could interrupt.

This is the **third-outcome match** from the 3-outcome decision matrix
laid out before the experiment:

- ❌ `RECOVERY at idle=Xs` — would have meant cancellation cleanup
  paths were the bottleneck.
- ❌ `NO RECOVERY` but probes return structured 5xx — would have meant
  the deadline converts the wedge into per-request errors.
- ✅ `NO RECOVERY` with silent `ReadTimeout` — read at this phase as
  "`#7` is below where any TRT-LLM-side deadline enforcement can
  reach". *(Phase 12 reclassified this: the actual mutex IS in
  TRT-LLM-owned code; the deadline can reach it once it's
  implemented.)*

**Implication for follow-up at this phase** (later revised in Phase
12): NIXL/UCX bug filing was thought to be the right next step,
backed by two stack dumps from independently-fixed TRT-LLM binaries.
Phase 12's direct-UCX experiment refutes the "NIXL plugin bug"
classification and refocuses follow-up on Next Steps item 7 (pin
down the exact `CacheSender::Impl` mutex and fix the deadlock
in-tree).

#### Artifacts

- Worktree: `/home/scratch.chienchunh_coreai/dev/TensorRT-LLM-pr13056-experiment/`
- Run logs: `.repro/logs/pr13056_run1/{ctx,gen,front,client}.log`
- Wedge stack dumps: `.repro/logs/pr13056_run1/pyspy/worker_660952_gdb.txt` (ctx, contains the `pthread_mutex_lock` frame in `CacheSender::Impl::response()` plus the NIXL plugin threads)
- Reusable wheel: `/home/scratch.chienchunh_coreai/trtllm_wheels/pr13056-c9777c4ac2-nv26.02.whl` (2.7 GB, MD5 `99ff92bf5e43120c43abfe32e241df8b`) — re-installable on any node with NVIDIA PyTorch container 26.02 to skip the ~1.5 h rebuild.

### Phase 12 — Cross-backend validation: signature `#7` is *not* NIXL-specific (T+7 days, *current*)

After Phase 11 left an unresolved question — "is `#7` a NIXL-plugin
bug or something deeper?" — a team member suggested switching the
TRT-LLM cache transceiver from `backend: NIXL` to `backend: UCX` (the
direct UCX path that bypasses the NIXL plugin entirely). This was the
single most decisive experiment of the investigation: NIXL backend
loads `libnixl.so` and `libplugin_UCX.so` and routes through
`AgentConnectionManager`, while direct UCX backend goes through
`UcxConnectionManager` + `libtensorrt_llm_ucx_wrapper.so` and never
loads NIXL plugin code. If sig `#7` is a NIXL plugin bug, switching to
direct UCX should clear the wedge. If it's deeper, the wedge will
fire either way.

#### Three sub-experiments

The experiment was run as three sequential sub-runs with the same
1P1D long-prompt burst harness, same `Qwen3-0.6B`, same `CONC=16,
BURST_DUR_S=60`:

**1. `pr13056_ucx_run1`: comprehensive-refactor variant + UCX backend.**
Result: harness aborted at the pre-burst sanity probe with
`http_500 wall=7.5s`. The first request reached the gen worker,
which threw `Request canceled (kNETWORK_ERROR)` from the catch in
`TransferSession::recv()`'s UCX recv path. Frontend retried, second
attempt got `Connection reset by remote peer` from
`CacheReceiver request()`. **Conclusion**: the comprehensive-refactor
variant introduces an unrelated regression on the TRT-LLM direct UCX
path — first-request UCX wireup fails. Not a sig-`#7` data point;
documented for the sake of completeness because we observed it.

**2. `pr13056_ucx_run2_tlsall`: same as run 1 but with `UCX_TLS=all`.**
Same fast HTTP 500 at the pre-burst probe. **Conclusion**: TLS
mode is not the issue; the regression in run 1 is real, not a
configuration mismatch.

**3. `rc11_ucx_run1`: this investigation's chained-PR variant + UCX
backend.** Pre-burst probe completes (`ok200 wall=6.2s`). Burst
completes (`8/11/19`). All five recovery probes return silent
`ReadTimeout` for 60 s each. **Verdict: `NO RECOVERY after 180s
idle`** — *the same silent wedge as `run8` and `pr13056_run1`,
but in a process where `libnixl.so` is not loaded at all.*

#### Stack-trace evidence

`gdb` post-mortem of the wedged `rc11_ucx_run1` ctx-side worker
shows:

```text
Thread "dataTransResp" (LWP 706490):
#0  pthread_mutex_lock                                   [libc.so]
#1  tensorrt_llm::batch_manager::CacheSender::Impl::response()
                                                         [libtensorrt_llm.so]
```

— the **same TRT-LLM-side frame as the NIXL runs**. The other threads
in the same process show only:

| Thread | Top non-libc frame |
|---|---|
| (UCX progress) | `ucxx::Worker::progressOnce` → `ucp_worker_progress` (UCX core) — *via* `libtensorrt_llm_ucx_wrapper.so`, **not** `libplugin_UCX.so` |
| (ZMQ control channel) | `UcxConnectionManager::UcxConnectionManager()::{lambda()}` → `zmq_msg_recv` — *via* `libtensorrt_llm_ucx_wrapper.so` |
| (gen-side request, separate worker) | `UcxConnection::recv()` → `__atomic_futex_unsigned_base::_M_futex_wait_until` — stuck waiting for ctx to send data |

`grep nixl` on the entire thread dump returns zero matches. So the
process has **no NIXL plugin loaded** and is wedged with the same
`response() → pthread_mutex_lock` frame.

#### What this proves about sig `#7`

The Phase 10 + 11 framing of sig `#7` as a **NIXL UCX plugin
internal mutex deadlock** is **falsified** by this experiment. The
wedge fires identically with TRT-LLM's direct UCX backend. The
mutex is therefore **not** in NIXL plugin code. The two transport
paths (NIXL plugin → UCX vs TRT-LLM UCX wrapper → UCX) share only
two things below TRT-LLM:

1. **UCX core** (`libucx*.so`)
2. **Operating-system primitives** (kernel TCP, futex, etc.)

…and one thing **inside TRT-LLM**:

3. **`CacheSender::Impl::response()` itself** (the calling function in
   both cases)

Since the wedge frame is `response()` directly invoking
`pthread_mutex_lock` with no intermediate frames visible in the
direct-UCX dump, the most plausible mutex is one **owned by
`CacheSender::Impl`** — specifically `mCondMutex` at the top of
`response()`'s loop body
([`dataTransceiver.cpp:684`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp#L684)).
The NIXL run's deeper frames showed an additional NIXL-internal mutex
in `getNotifs()`, but that's now understood as a secondary/contributing
factor (or a parallel issue), not the primary wedge cause.

#### Variant: ctx-side mpi4py executor exit (`rc11_ucx_run2_diag`)

A follow-up rc11+UCX run intended to capture register-level mutex
diagnostics produced a different but related failure mode: the
ctx-side `mpi4py.futures.server` worker process exited mid-burst
(parent `orted` is still running but has zero descendants). The
ctx-serve Python proxy stayed alive — `/health` still returns 200 —
but with no executor backend. From the harness's perspective the
wedge is identical (silent `ReadTimeout`); the underlying mechanism
is process exit rather than deadlock. ctx.log shows multiple
`Context KV cache transfer cancelled after ready-signal` errors
(signature `#1` fix path firing repeatedly) plus a Python traceback
returning HTTP 400, but no `SIGSEGV` / `Aborted` signal trace. Both
this and the deadlock variant point at the same `CacheSender::Impl`
code region; together they suggest a class of bugs (deadlock + dirty
exit) rather than a single point fix.

#### Revised sig `#7` framing

Folded into the Failure Signatures section above. Headline change:

| Before this phase | After this phase |
|---|---|
| `#7` is a NIXL UCX-internal `pthread_mutex_lock` deadlock | `#7` is a `pthread_mutex_lock` wedge in `CacheSender::Impl::response()` (TRT-LLM-owned), exposed by both NIXL and direct UCX backends, most likely on `mCondMutex` |
| Out of TRT-LLM scope; file with NIXL/UCX team | **Fixable in TRT-LLM**; file with NIXL/UCX team only as a secondary `getNotifs` issue worth noting |
| The deadline enforcement work is a fallback / mitigation | The C++ deadline-on-blocking-primitives work (Effort Estimate Layer B) **directly addresses `#7`** since `mCondMutex` is a TRT-LLM-owned primitive |

The pragmatic effect is a substantial shift in priority: **the
recommended next investigation step is no longer "file a NIXL/UCX
bug" but "pin down which mutex is held by which thread with a live
`gdb` register inspection, and then fix the offending lock-ordering
in `CacheSender::Impl`."**

#### Artifacts

- All three UCX experiments archived under `/home/.../disagg-investigation-archive/` as `pr13056_ucx_run1/`, `pr13056_ucx_run2_tlsall/`, `rc11_ucx_run1/`, `rc11_ucx_run2_diag/`.
- The single most important artifact is `rc11_ucx_run1/pyspy/worker_705787_gdb.txt` — it shows the `dataTransResp → pthread_mutex_lock → CacheSender::Impl::response()` frame in a process with no NIXL plugin loaded. This is what falsified the original Phase 10 framing.
- `rc11_ucx_run2_diag/diag/gen_worker_714018_gdb.txt` shows the alternative failure mode (ctx-side executor exit) for completeness.

#### Side finding worth reporting back

The comprehensive-refactor variant (`#13056`-equivalent) introduces a
regression on the TRT-LLM direct UCX path: pre-burst sanity probe
fails with HTTP 500 at the very first request because UCX wireup
between ctx and gen produces `Connection reset by remote peer`. This
is **unrelated to sig `#7`** and is documented purely as feedback for
future review of that refactor. Both `pr13056_ucx_run1` and
`pr13056_ucx_run2_tlsall` reproduce it.

---

## Signature Taxonomy and Cascade Map

A natural high-level question after reading the seven signatures is:
*are they all triggered by the same thing, and do the fixes interact?*
Both halves of that question have a precise answer worth writing down,
because the answer governs how to split PRs, how to write regression
tests, and how to interpret a future field hit.

### What each signature is actually triggered by

It is tempting to summarise the whole investigation as "burst of traffic
→ timeouts → cancellations → mishandling". That is right for the
*majority* of signatures but is not literally true for all seven. The
honest taxonomy is:

| Sig | Category | Triggered by |
|---|---|---|
| `#1` | **Cancellation-handling** (direct) | Sender erases `(req, promise)` entry on cancel-after-ready without fulfilling the promise → `Broken promise`. Needs a real cancellation in flight. |
| `#3` | **Cancellation-handling** *(hypothesised)* | Field-only `bad optional access`. Hypothesised to be a downstream consequence of `#1`/`#4`/`#5` leaving a request half-initialised. NIXL-layer stranding (`#7`) is a second candidate trigger now that we know about it. |
| `#5` | **Cancellation-handling** (direct) | Receiver `cancelRequest()` erases queued `(req, promise)` without fulfilling the promise — exact mirror of `#1`. Needs a real cancellation against a still-queued request. |
| `#6` | **Cancellation-handling** (cascade) | Lives in the `!isReady` early return of the receiver. That branch is only reached *after the `#1` fix is in place* and the sender sends `is_ready=false`. Needs a cancel-after-ready in flight, plus the `#1` fix as a prerequisite. |
| `#4` | **Structural, exposed by cancellation** | Pure structural defect: unconditional `future.get()` on a future that may not be ready. Doesn't *need* cancellation to exist; any reason a receiver-side future stays unresolved trips it. In practice on `rc11` the unresolved future is created by `#1`/`#5`/`#6`, which all are cancellation-driven. |
| `#2` | **Eviction-driven, not cancellation-driven** | KV-block trie inconsistency surfaced via `freeBlockAndAllDescendants → detachDescendantsFromLookupTree`. Triggered by block eviction under memory pressure, *not* by cancellation. Burst traffic happens to drive frequent eviction (especially with prefix-overlapping prompts at high concurrency), so it shares the "burst exposes it" property, but it would also fire under any sustained workload with heavy eviction. |
| `#7` | **NIXL-internal contention** | NIXL UCX-internal `pthread_mutex_lock` deadlock inside `nixlUcxThreadEngine::getNotifs()`. The customer load shape that triggers it includes cancellations, but the NIXL bug could be a pure lock-ordering issue between concurrent send + receive paths that would also fire under a different high-concurrency pattern. We don't have NIXL-internal visibility to say for sure. Safe to call it *contention-driven*; not safe to call it *strictly cancellation-driven*. |

So the right one-sentence summary is: **four-of-seven (#1, #3, #5, #6)
are cancellation-handling bugs; #4 is a latent blocking bug that
cancellations expose; #2 is an eviction bug that burst traffic exposes
via memory pressure; #7 is a NIXL-internal deadlock that the same
load shape happens to trigger but which is not strictly a cancellation
bug**.

### Refined trigger chain

```text
long prompts                                         (~8K tokens, gauss(8000, 2000))
  + high concurrency                                 (CONC=16)
  + aggressive client-side timeouts                  (60 s wall, with retries)
  + retries on every timeout
        ↓
high cancellation rate    +    frequent eviction
        ↓                              ↓
cleanup paths exercised       trie eviction exercised
in volume                     in volume + memory pressure
        ↓                              ↓
sig #1 / #5 / #6 fire         sig #2 fires
        ↓
sig #4 exposed by unresolved futures from #1 / #5 / #6
        ↓
sig #3 exposed in decode by half-initialised state
        ↓
NIXL contention pattern                              (parallel path,
        ↓                                             same load shape)
sig #7 fires in NIXL UCX internal mutex
```

Two specific corrections to the naïve "burst → cancellation → bug"
chain are worth being precise about:

1. **Burst of traffic alone is not the trigger; aggressive client
   timeouts + retries are.** Without aggressive client-side timeouts,
   even a long burst at concurrency=16 would not produce the
   cancellation rate that exercises the cleanup paths in volume.
   Customers running production-grade serving with aggressive timeouts
   is what surfaces the cleanup-path bugs; CI's integration tests
   without aggressive timeouts is what hid them. (This is the same
   point as the "test pyramid is wrong" item in the Architectural
   Reflections section, said one layer up.)
2. **"Cancellation" is itself one of several entry points to cleanup
   paths.** Cancellations come from client HTTP disconnects,
   client-side timeouts that become server-side cancellations,
   server-side `kv_transfer_timeout_ms` *(not enforced today — see
   Next Steps item 8)*, internal aborts from downstream errors,
   block eviction under memory pressure (`#2`'s entry point), and
   the `CacheSender::Impl::response()` mutex contention (`#7`'s
   entry point — see Phase 12). The unifying property is
   **cleanup paths exercised in volume**, not "cancellation"
   specifically.

### How the signatures and their fixes tangle

The signatures are not independent. A natural follow-up question is
whether fixing one introduces another, or merely uncovers another
that was already there. Both happen in this investigation, and the
distinction matters for review and for how chained PRs are scoped.

#### Type 1 cascade — a fix *produces* a new signature

This is the strict case where a fix changes behaviour such that a
previously-unreachable code path becomes reachable in production, and
that code path has a latent bug. The fix isn't *wrong*, but it makes
a new bug newly observable.

There is **only one such case in this investigation: the `#1` fix
produces `#6`.**

- Pre-`#1`-fix: the sender on a cancelled-after-ready request just
  erased the entry without sending anything to the receiver. The
  receiver therefore never saw `is_ready=false` in production — the
  receiver's `if (!isReady)` early-return branch was structurally
  unreachable from the sender's behaviour.
- Post-`#1`-fix: the sender correctly sends `is_ready=false`. The
  receiver now hits the `!isReady` branch on every cancelled-after-ready
  request. That branch returns *without* calling `receiveSync()`, which
  means `unformat()` doesn't run, which means `freeBufferIndexForRecv()`
  doesn't run, which means the recv-buffer slot reserved at the top
  of `sendRequestInfo()` is leaked. With `mRecvBufferCount=1`, one
  leak permanently wedges the receiver — that's `#6`.

This is why the `#6` PR
([#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673)) is
**explicitly chained on the `#1` fix PR
([#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640))**: the bug
only exists in the world where `#1`'s fix is present.

#### Type 2 cascade — a fix *exposes* a pre-existing signature

Here the downstream bug already exists; it just wasn't observable
because something upstream was masking it. Fixing the upstream removes
the mask. The downstream bug is **not** a regression of the fix — it
was a latent pre-existing issue.

| Mask removed | Bug exposed |
|---|---|
| `#4` fix (gen event loop no longer self-blocks) | `#5` — the receiver-side broken-promise was already firing, but the polling loop wasn't running to surface it |
| `#4` fix | `#6` — the recv-buffer leak was already happening, but the gen worker wedged in `#4` before the next request could exhibit the wedge |
| `#6` fix (gen worker no longer wedges on `assignBufferIndex`) | `#7` — NIXL deadlock was already firing, but `#6` was wedging the gen worker before NIXL's wedge could surface as the *terminal* failure |

Type 2 cascades are what the timeline calls "fix one bug, see the
next one" — Phase 5 → 7 → 8 → 10 is essentially this pattern
repeating four times.

#### A subtler third relationship: `#4` as a defensive catcher

`#4` is structurally a *catcher* for any upstream bug that produces a
never-resolving receiver future. It would have stayed dormant forever
in a world where no upstream bug created stuck futures. In `rc11` the
upstream bugs (`#1`, `#5`, `#6`) all do create stuck futures, so `#4`
fires constantly. After fixing `#1`/`#5`/`#6`, `#4` would have stopped
firing on its own — but fixing `#4` is *independently valuable* as a
defensive measure: any *future* bug (yours or someone else's) that
creates a stuck receiver future would otherwise re-trigger `#4`. The
`#4` fix is therefore a deliberate piece of defence-in-depth, not just
a symptomatic patch.

#### Do the fixes overlap in code?

Not in any conflicting way. Mapping fixes to files:

| Sig | File | Function |
|---|---|---|
| `#1` | `dataTransceiver.cpp` | `CacheSender::Impl::sendResponse` |
| `#2` | `templatedTrie.h` | `clearNode` |
| `#4` | `cacheTransceiver.cpp` | `CacheTransceiver::checkGenTransferStatus` |
| `#5` | `dataTransceiver.cpp` | `CacheReceiver::Impl::cancelRequest` |
| `#6` Layer A | `dataTransceiver.cpp` | `CacheReceiver::Impl::sendRequestInfo` |
| `#6` Layer B | `dataTransceiver.cpp` | `CacheReceiver::Impl::requestSync` |

Same file (`dataTransceiver.cpp`) for `#1`, `#5`, and `#6`, but
different classes / functions. There are no merge conflicts; the PRs
apply cleanly to `main` independently. The only structural dependency
is the `#6` → `#1` chain documented in Type 1 above, and that's
enforced by the chained PR base, not by line-overlapping diffs. A
reviewer working through `#13639/#13640/#13674/#13671/#13672/#13673`
in any order will not see textual conflicts; only `#13673` will fail
to compile correctly without `#13640` first.

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

## What We Would Do Differently — A Retrospective Process Reflection

The investigation was sequential by necessity: one signature surfaced,
got a fix, and the next signature emerged from the post-fix behaviour.
Six rounds of "find bug → fix bug → find next bug" took ~6 days of
calendar time. With the end-to-end view in hand, it is worth asking:
*could we have done this differently from the start?* The honest answer
has two parts.

### What was not actually possible at T0

A "design one comprehensive fix" approach is the natural counterfactual.
Concretely: at T0, introduce a `TransferSession`-like abstraction that
encapsulates request lifetime + RAII buffer holders +
promise-fulfillment-on-destruct + deadline, and let it close all six
TRT-LLM signatures in one PR. This sounds clean in retrospect but had
two hard blockers:

1. **You cannot design an abstraction to fix bugs you have not found
   yet.** We knew about three signatures from the field at T0
   (`#1`, `#2`, `#3`). The other four (`#4`, `#5`, `#6`, `#7`)
   emerged from investigation. The architectural answer ("what
   invariants does this abstraction enforce?") is the *output* of
   finding the bugs, not the *input*. A `TransferSession` designed at
   T0 against only `#1`/`#2`/`#3` would not have prevented `#5` or
   `#6`, because we would not have known to enforce the invariants
   those signatures violate.
2. **The field was wedged.** Customers needed the smallest patches
   that work, not a multi-thousand-line refactor of a critical path.
   Review pressure on TRT-LLM also favours small focused changes; a
   refactor of `dataTransceiver.cpp` that touched all three backends
   (UCX/NIXL/MPI) would not have landed quickly.

A single coordinated fix was therefore strictly impossible given the
information state at T0. What we *should* have done is structurally
different: change the **meta-process** so each subsequent bug discovery
would have been faster, and so cascade relationships would have been
caught at design time instead of after a build-and-rerun cycle.

### What we should have done first (in priority order)

#### 1. Add deadline enforcement (Next Steps item 8) as PR #0

The single highest-leverage change. The `kv_transfer_timeout_ms` knob
is already plumbed through Python config, the C++ config class,
serialization, and getters/setters — it is just never consumed in the
request execution path. Even Layer A alone (the ~1-week Python-level
deadline) would have:

- **Converted every cleanup-path bug from a *silent wedge* into a
  *per-request error*.** Signatures `#1`, `#5`, `#6`, and `#7` all
  surface as `kNETWORK_ERROR` 5xx responses with a real exception
  message instead of the deployment going dark.
- **Given each subsequent bug an attributable failure point**
  (`request 4113 timed out in checkGenTransferStatus`) instead of
  requiring `py-spy` / `gdb` post-mortem to figure out which request
  was stuck and why.
- **Given orchestration a real signal** so customers' production
  wedges self-heal via pod restart while we work on root causes. Field
  urgency drops from P0 to P2.

The investigation would have shifted from "*the deployment is wedged,
dump stacks, find the wedged thread*" to "*these 14 requests timed
out, here are their lifecycle traces*". Phase 5 → Phase 10 of the
timeline (currently ~6 days) would plausibly have collapsed to ~2
days.

The deadline enforcement is also retrospectively justified by the
investigation itself: signature `#7` is the only signature TRT-LLM
cannot fix at the source, and the deadline is the *only* TRT-LLM-side
defence against it. We would have built this layer eventually anyway.
Building it first makes everything else trivially debuggable.

#### 2. Write down the seven invariants first, fix against them

If the seven contracts in the Architectural Reflections section had
been written down at T0 — even just as a paragraph in the
disaggregated-serving developer guide — three concrete cascade
relationships would have been caught at review time instead of after
days of reproduction:

- **`#5` would have been caught at `#1`'s PR review.** "This is the
  sender-side fix for the missing-promise-fulfillment invariant; the
  receiver-side mirror is structurally identical — fix both at once."
  Two PRs collapse into one.
- **`#6` would have been caught at `#1`'s PR review.** "The new
  `!isReady` path on the receiver — under the 'every acquired resource
  must release on every exit path' invariant, does it release every
  resource the success path releases?" Type 1 cascade prevented at
  design time, not after a 2-day reproduction cycle.
- **`#4` would have been visible to any code search for unconditional
  `future.get()`.** Under the "every blocking wait must be
  interruptible" invariant it is a bug regardless of whether it
  currently fires; a sweep against the invariants would have caught
  it as a latent issue before the field hit it.

The cost is a single document edit. The benefit is preventing the
entire Type 1 cascade and most of the Type 2 cascades documented in
the Signature Taxonomy and Cascade Map section above.

#### 3. Add the cancel-during-transfer integration test first

The single largest test-coverage gap surfaced by this investigation
is the cancel-during-transfer load shape (long prompts, high
concurrency, aggressive client timeouts, retries). If that test had
existed at T0, all six TRT-LLM signatures would have been visible
**as test failures in CI** instead of as a customer field hit. Even
signature `#3` (currently field-only) would plausibly have been
reproducible. The investigation would not have needed external
infrastructure (Dynamo, mpi4py worker dumps, NIXL trace correlation,
`py-spy` / `gdb` post-mortem).

The cost is moderate (a few hundred lines of integration test
infrastructure plus a CI lane to run it). The ongoing benefit is
huge: every future PR that touches the disaggregated path is gated
against this load shape.

### What this implies for next time

The cleaner approach is not a different *fix*; it is a different
**order of operations**:

```text
What we did:
    field hit → reproduce → find bug N → fix bug N → repeat 7 times
        ↓
    ~6 days, 6 cascading PRs, NIXL bug discovered last as a surprise

What we should have done:
    field hit → containment layer (deadline enforcement, ~1 week)
              → integration test for the load shape (~few days)
              → write down the seven invariants (~hours)
              → bugs become CI-visible and individually attributable
              → fix them in any order, each PR reviewable against invariants
              → Type 1 cascade caught at review time, not after rebuild + rerun
        ↓
    same 7 signatures, identified in parallel from one CI run,
    fixed individually but with no cascade surprises
```

The key insight is that the bottleneck of the investigation was
**observability and attribution**, not fix complexity. Each fix
individually is small (`#1` is ~5 lines, `#4` is ~17 lines, `#5` is
~20 lines, `#6` is the largest at ~80 lines including the RAII
helper). What ate the calendar time was not writing the fixes — it
was figuring out what was wedged, why, and which fix to write next.
The three meta-process changes above all attack that bottleneck
directly.

### What this section is *not* arguing

A few clarifications to avoid over-reading the retrospective:

- **It is not arguing for a `TransferSession` rewrite as PR #0.**
  That refactor is still the right long-term direction (see "What
  this implies for follow-up work" in Architectural Reflections), but
  it is a separate, larger project that should follow the
  per-signature fixes once the contracts are stable, not replace
  them.
- **It is not arguing that one PR could have fixed all seven
  signatures.** Six of them are real, distinct bugs in different
  functions, and `#7` lives in NIXL. They genuinely need separate
  fixes. The argument is about how *quickly* they would have been
  found and how *cleanly* they would have been reviewed, not about
  collapsing them into a single patch.
- **It is not arguing that the sequential discovery was avoidable in
  absolute terms.** It was avoidable *given the meta-process changes
  above*, but not avoidable given the meta-process we actually had.
  The retrospective is about what we should change for the *next*
  investigation of this shape, not about whether this one could have
  been done differently after T0 with the same tooling.

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
| **#6** Recv-buffer index leak via `!isReady` early return; subsequent receives wedge in `assignBufferIndexForRecv()` | Combined test + fix in review (chained on `#13640`) | (combined into fix PR) | [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673) | Two-layer fix: RAII cleanup in `sendRequestInfo()` (Layer A) + explicit free in `requestSync()` `!isReady` path (Layer B). Direct cascade from the `#1` fix; chained on `#13640` because the `!isReady` branch is only reachable once the sender-side cancellation correctly sends `is_ready=false`. New test `test_cancelled_after_ready_does_not_leak_recv_buffer_index` uses the NIXL backend (the only backend that goes through `assignBufferIndexForRecv`). |
| **#7** `pthread_mutex_lock` wedge in `CacheSender::Impl::response()` under cancel-during-transfer load (re-classified in Phase 12 from "NIXL plugin bug" to "TRT-LLM-side mutex bug") | Characterised; root cause not yet pinned down to a specific mutex/holder | — (unit test deferred until the exact mutex and holder are identified) | TRT-LLM-side fix expected once the live `gdb` register inspection identifies the offending mutex and holder; estimated ~30–80 lines | Three independently-built TRT-LLM binaries (this investigation's chained PRs on `rc11`, the comprehensive-refactor variant on `rc11`, and our chained PRs with **direct UCX backend** on `rc11`) all reach the same wedge frame. The direct UCX run proves the bug is in `CacheSender::Impl`, not in any transport plugin. |

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
7. **Pin down the exact mutex behind signature `#7` and fix the
   deadlock in `CacheSender::Impl`** — *this is the new highest-priority
   action after Phase 12 reclassified `#7` as a TRT-LLM-side bug, not
   a NIXL plugin bug.* Procedure:
   - Re-launch the rc11+UCX (or rc11+NIXL) burst harness; let it
     wedge.
   - Attach `gdb` to the wedged ctx-side mpi4py executor worker
     (`mpi4py.futures.server`) and locate the `dataTransResp`
     thread.
   - On that thread, `frame 0` and `info registers rdi` to read the
     mutex address; then `x/8x $rdi` to dump the
     `pthread_mutex_t::__owner` field — that gives the holder TID.
   - Match the holder TID to a thread in `info threads`, walk its
     backtrace to find the holding code path.
   - Fix the lock-ordering or release-before-blocking-call issue in
     `CacheSender::Impl`.

   The estimated fix is small (~30–80 lines of C++), confined to
   `dataTransceiver.cpp`. The diagnostic step itself is ~30–60 min;
   the fix + chained test PR is probably ~1–2 days.

8. **Enforce `kv_transfer_timeout_ms` as a hard deadline** on the
   transceiver's blocking entry points. As of `rc11` the knob is
   fully plumbed through Python config, C++ config class,
   serialization, and getters/setters — but **never consumed in the
   request execution path** of `cacheTransceiver.cpp` /
   `dataTransceiver.cpp`. With Phase 12's reclassification, the role
   of this work has shifted:

   - **Before Phase 12** (when `#7` was thought to be a NIXL plugin
     bug): the deadline was a fallback / mitigation only — it
     couldn't unwedge the NIXL-internal mutex.
   - **After Phase 12** (now that `#7` is a TRT-LLM-side
     `mCondMutex`-class wedge): the deadline **directly addresses
     `#7`** for any TRT-LLM-owned mutex. With Effort Estimate Layer B
     in place, every `cv.wait` and unbounded `future.get()` becomes
     interruptible by `kv_transfer_timeout_ms`, including the
     `mCondMutex` lock-acquisition that wedges today.

   Even after item 7's surgical fix lands, item 8 remains valuable
   defence-in-depth: the architectural invariant "every blocking
   wait must be interruptible" from the Reflections section is what
   prevents the next mutex bug of this shape from recurring. See
   "Effort estimate for the deadline enforcement" below.

9. **File a NIXL/UCX bug as a secondary issue** — Phase 11's `gdb`
   evidence on the NIXL backend shows that `nixlUcxThreadEngine::getNotifs()`
   was *also* parked on `pthread_mutex_lock` deep in the NIXL plugin's
   own internal lock. That may be a contributing factor to `#7` (a
   cross-library lock-ordering deadlock between TRT-LLM's `mCondMutex`
   and the NIXL plugin's internal mutex) or it may be an unrelated
   secondary issue. Either way it's worth filing with the NIXL/UCX
   team using the `pr13056_run1` ctx-worker stack as the canonical
   reproducer. **No longer the top-priority action** for resolving the
   wedge — that's now item 7 above.
10. **Rename the misleading `drop_without_fulfill` trace marker.** As
    noted in Phase 10, the marker fires immediately *before* the
    signature `#1` cancellation handler that already fulfills the
    promise correctly. The 3 events per `run8` are the fix path doing
    its job, not actual drops. Renaming to
    `cancelled_after_ready_handled` removes the false-positive in future
    forensic readings.

### Effort estimate for the deadline enforcement (Next Steps item 8)

This section sizes the deadline-enforcement work. After Phase 12
reclassified `#7` as a TRT-LLM-side mutex bug, this work has a
dual role: (a) it directly addresses `#7` once Layer B lands (every
TRT-LLM-owned blocking primitive becomes interruptible), and (b) it
remains valuable defence-in-depth even after the surgical fix in
Next Steps item 7 is applied, so the architectural invariant "every
blocking wait must be interruptible" is enforced going forward. The
work decomposes into four implementation layers, each with its own
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

**Verdict:** Last resort. With Phase 12's reclassification (the
mutex is in TRT-LLM, not NIXL), Layer D is even less likely to be
needed — the surgical fix in Next Steps item 7 should resolve the
wedge directly without requiring agent reset.

#### Recommended order (revised after Phase 12)

1. **Surgical mutex fix for `#7`** (Next Steps item 7, ~1–2 days) —
   live `gdb` inspection to identify the offending mutex and holder
   in `CacheSender::Impl`, then a small lock-ordering /
   release-before-blocking-call fix. This is the highest-leverage
   action; everything else below is defence-in-depth.
2. **Layer A** (1 week) — land the Python-level deadline + structured
   cancel + per-request 5xx. Useful even after item 1 lands, because
   the deadline gives orchestration a real signal for any *future*
   bug of similar shape.
3. **Layer B** (2 weeks) — extend the deadline into C++ for every
   TRT-LLM-owned blocking primitive. Closes the architectural gap and
   prevents the "fix-on-side-A surfaces leak-on-side-B" pattern from
   recurring. With Phase 12's reclassification, this also retroactively
   covers `#7` if the surgical fix in item 1 misses an edge case.
4. **NIXL/UCX bug** (Next Steps item 9) — file as a secondary issue
   for the deeper `getNotifs` mutex frame seen in NIXL-backend runs.
   No longer top priority, but worth filing for completeness.
5. Layer C and Layer D should not be pursued unless items 1–3 all
   fall through.

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
  - `cpp/tests/unit_tests/runtime/radixBlockTreeTest.cpp` (signature `#2`).
  - `tests/unittest/others/test_kv_cache_transceiver.py::test_cancel_request_in_transmission_fulfills_sender_future`
    (signature `#1`).
  - `tests/unittest/others/test_kv_cache_transceiver.py::test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
    (signature `#4`).
  - `tests/unittest/others/test_kv_cache_transceiver.py::test_cancel_queued_gen_request_fulfills_receiver_future`
    (signature `#5`).
  - `tests/unittest/others/test_kv_cache_transceiver.py::test_cancelled_after_ready_does_not_leak_recv_buffer_index`
    (signature `#6`; uses the NIXL backend, which is the only backend
    that goes through `assignBufferIndexForRecv()`).
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
