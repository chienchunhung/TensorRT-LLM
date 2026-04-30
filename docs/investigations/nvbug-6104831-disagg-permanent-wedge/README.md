# NVBug 6104831: Permanent Disaggregated-Serving Wedge in `rc11`

- **Severity:** P0 / Critical
- **Affected component:** Disaggregated serving (`trtllm-serve` context worker
  + generation worker + disaggregated front-end), `rc11` baseline
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
  fixes in flight or merged; signature #6 (control-path send stall) is the
  next target. The full end-to-end reproducer still wedges in a fresh post-fix
  run, so the chain is not yet root-caused at the system level.

---

## Executive Summary

Disaggregated serving in `rc11` collapses into a permanent wedge after a
burst of long-prompt requests with retries and cancellations. The system
keeps every process alive — no crash, no `/health` failure, no orchestrator
failover — yet every subsequent request either times out or returns a generic
`400 Bad Request`. From the outside it looks like a single wedge; from the
C++ transceiver it is **at least five distinct, partially overlapping bugs**
in the disaggregated KV-cache transfer path.

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
| **#6** *(suspected)* | Control-path stall inside `sendRequestInfo()` / `AgentConnection::sendRequestAndBufferInfo()` before ready-signal wait | `dataTransceiver.cpp` + `agent_utils/connection.cpp` | Post-`#4`-fix gen-side request-lifecycle trace |

The mapping of fixes is summarised at the end, in
[Signature ↔ PR Map](#signature--pr-map).

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

**PRs:** Currently isolated in the `local/rc11-disagg-repro` worktree
together with signature #5 fix. Will be split out into its own chained
test+fix PR pair before submission.

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

**Reproducer:** Currently observed only via the full 1P1D HTTP repro;
no dedicated unit test yet. A test analogous to the signature #1 reproducer
is the natural next step.

**PRs:** Currently isolated in the `local/rc11-disagg-repro` worktree
together with the signature #4 fix. Will be split out into its own chained
test+fix PR pair before submission.

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

**Root cause:** Not yet confirmed. The structural shape suggests one of:
- a `notifySyncMessage()` to a remote agent that has already gone away
  (e.g. the matching context request was cancelled and the sender side
  tore down its agent state),
- a counterpart that we are still waiting for in
  `mManager->getConnections(commState)`,
- or a control-path send into a transport that has already entered an
  error state.

**Status:** Not fixed yet. We added new gated trace markers around both
sites — `gen_send_request_info_begin/end`,
`gen_send_request_buffer_info_begin/notify/end` — so the next post-fix
repro will pinpoint exactly which step is blocking. The same
`TRTLLM_DISAGG_TRACE_PROMISE=1` env gates these; they cost nothing in
normal runs.

**Reproducer:** Same end-to-end 1P1D + long-prompt burst harness; no minimal
unit test yet. A first attempt at one will be made once we know which step
is hanging.

**PRs:** None yet.

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

### Phase 8 — Signature #5 fix and signature #6 instrumentation (T+5 days, *current*)

- Receiver-side fix: extract the queued promise under the lock and fulfill
  it with a structured `kNETWORK_ERROR` exception once released. Same shape
  as signature #1.
- New trace markers around `sendRequestInfo()` (`gen_send_request_info_begin/end`)
  and `AgentConnection::sendRequestAndBufferInfo()`
  (`gen_send_request_buffer_info_begin/notify/end`) so the next repro will
  pinpoint signature #6.
- Rebuild + rerun the repro — in progress at the time of writing.

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
- **The bugs partially mask each other.** With signature #4 in place, the
  gen event loop self-blocks before signatures #5 and #6 can manifest.
  Removing signature #4 was a prerequisite for even seeing #5 and #6 in
  the logs. This is also why "fix one bug, see another" is the dominant
  pattern in the timeline above.
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

## Signature ↔ PR Map

| Signature | Status | Test PR | Fix PR | Notes |
|---|---|---|---|---|
| **#1** Sender-side `Broken promise` after ready signal | Test merged; fix in review | [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) | [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) | Chained: `#13640` builds on `#13639`. |
| **#2** Trie `cascade prune` assertion | Test merged; fix in review | [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) | [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) | Chained: `#13572` builds on `#13571`. Independent of disagg networking. |
| **#3** Decode-side `RuntimeError: bad optional access` | Field-only; not yet localised | — | — | Python-side trace markers added; will localise on next field hit. |
| **#4** Gen-side blocking hang in `checkGenTransferStatus(atLeastNum=1)` | Fix + regression test in `local/rc11-disagg-repro` worktree | (pending split) | (pending split) | To be split into a chained test+fix pair before submission. |
| **#5** Receiver-side `Broken promise` from queued cancel | Fix in `local/rc11-disagg-repro` worktree | — (still needed) | (pending split) | Mirror of `#1`. Unit test analogous to `#1` reproducer is the next step. |
| **#6** *(suspected)* Control-path stall inside `sendRequestInfo()` / `sendRequestAndBufferInfo()` | Instrumentation only; not fixed | — | — | New trace markers added; next post-fix repro will pinpoint the blocking step. |

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

1. **Confirm signature #6 with the new instrumentation.** The post-fix repro
   currently in flight should produce `gen_send_request_info_*` /
   `gen_send_request_buffer_info_*` markers that pinpoint the exact blocking
   step.
2. **Implement and unit-test signature #5 fix.** Mirror of the signature #1
   reproducer; should be a small, focused test in
   `tests/unittest/others/test_kv_cache_transceiver.py`.
3. **Split signatures #4 and #5 fixes into chained PR pairs** matching
   `#13571 / #13572` and `#13639 / #13640`.
4. **Backport** `#13119` (request-level error propagation) to the `rc11`
   field branch so future field hits are easier to attribute to a specific
   signature.
5. **Add an integration test** that drives the disagg HTTP path with the
   long-prompt + retries + cancels load shape used by the local burst
   harness. This is the single largest coverage gap surfaced by this
   investigation.

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
  - `run6` (post-signature-#5 fix + signature-#6 instrumentation): in
    progress at the time of writing. Logs preserved under `.repro/logs/`
    in the `local/rc11-disagg-repro` worktree.
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
