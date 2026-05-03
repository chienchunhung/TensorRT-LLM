# 05 — Investigation Timeline

This is the chronological narrative of how the seven failure signatures
and the four fix approaches emerged. It is *not* required reading to
understand the current state — for that, read
[`02-failure-signatures.md`](02-failure-signatures.md),
[`03-defect-class-stack.md`](03-defect-class-stack.md), and
[`06-fix-approaches/README.md`](06-fix-approaches/README.md). This file
is for someone who wants to understand *how* the bug class was
discovered and *why* each fix exposed the next.

The phases are approximate calendar markers ("T+1 day"), not strict
sequencing — multiple things were often in flight simultaneously.

---

## Phase 0 — Field report (T0)

Customer report from Dynamo + TRT-LLM `rc11` deployment of a permanent
hang under sustained traffic. Three apparent failure signatures in the
crash dumps and Python tracebacks:

1. `std::future_error: Broken promise` on prefill workers.
2. `cascade prune: parent did not find this node as a child` C++
   assertion under sustained load.
3. `RuntimeError: bad optional access` raised in the decode-side Python
   event loop.

These are signatures `#1`, `#2`, `#3`.

---

## Phase 1 — C++ unit-test probes for sig `#2` (T0 + a few hours)

Cheapest, fastest layer: extend `radixBlockTreeTest.cpp` with stress
cases that mirror the failing call path (`addSequence → getFreeBlock →
freeBlockAndAllDescendants → detachDescendantsFromLookupTree`). Four new
tests reproduce the `cascade prune` assertion deterministically on
stock `rc11`. This proves that signature `#2` is independent of Dynamo,
NIXL, and disaggregated networking.

**Reproducer PR:** [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571).

## Phase 2 — Sig `#2` fix (T+1 day)

Reset the child's `mPrevNode` in `clearNode()` before erasing the entry
from the parent. The new unit tests pass; existing radix-tree tests
still pass.

**Fix PR:** [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572),
chained on `#13571`.

---

## Phase 3 — Local 1P1D reproduction attempts (T+1 day)

Spin up a local two-GPU 1P1D `trtllm-serve` deployment on `rc11`. Light
load does not reproduce the hang. Switching to the customer's load
shape (long prompts, `CONC=16`, retries, cancellations) successfully
reproduces the permanent wedge. Confirms that signature `#1` is real
and reachable end-to-end without Dynamo.

This establishes the harness used through Phase 14 (see
[`04-reproduction.md`](04-reproduction.md)).

---

## Phase 4 — Sig `#1` isolation, fix, and unit test (T+2 days)

- Targeted C++ instrumentation (`tracePromiseLifecycle()` gated on
  `TRTLLM_DISAGG_TRACE_PROMISE=1`) localises sig `#1` to the
  cancelled-after-ready path in `CacheSender::Impl::sendResponse`.
- Fix: `set_exception(std::make_exception_ptr(...))` on the promise
  before the entry is erased.
- New unit test
  `test_cancel_request_in_transmission_fulfills_sender_future`
  reproduces the broken promise on stock `rc11` and passes post-fix.

**PRs:** [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639)
(test) → [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640)
(fix), chained.

---

## Phase 5 — Post-`#1`-and-`#2` rerun: wedge persists, surfaces sig `#4` (T+3 days)

With `#13572` and `#13640` applied to the isolated `rc11` worktree, the
1P1D repro **still wedges**. The Python hang detector dumps thread
stacks and shows the gen worker's main event loop blocked exactly
inside `self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)`.

Code reading of `cacheTransceiver.cpp` and the matching Python
implementation in `transceiver.py` shows that the C++ path takes an
unbounded blocking wait when `atLeastNum=1`, while the Python path
does not. This is sig `#4`.

In hindsight, the underlying terminal driver of the Phase-5 wedge was
already sig `#7` (the deeper `CacheSender::Impl::response()` mutex
wedge identified in Phase 12), but `#4` was the *visible* TRT-LLM-side
symptom because the gen event loop was self-blocking before any of the
later layers could surface. Fixing `#4` was a prerequisite for *seeing*
`#5` and `#6`, which were prerequisites for *seeing* `#7`. This is the
first instance of the "Type 2 cascade" pattern (a fix exposes a
pre-existing latent signature).

---

## Phase 6 — Sig `#4` fix and regression test (T+3 days)

- Fix: in the non-`blockAll` path, probe each selected future with
  `wait_for(0)` and skip if not ready.
- New unit test
  `test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
  fails on stock `rc11` and passes post-fix.

**PRs:** [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674)
(test) → [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671)
(fix).

---

## Phase 7 — Post-`#4` rerun: wedge persists, surfaces sig `#5` and suspected sig `#6` (T+4 days)

Repro again with the `#4` fix applied. The gen event loop is no longer
self-blocked; `gen_future_skip_unready` markers appear repeatedly. But
the harness still reports `NO RECOVERY after 180s idle`, and two new
patterns appear:

- **Sig `#5`**: several requests get `Broken promise` from the gen side
  that originate from the receiver-side cancel path
  (`CacheReceiver::Impl::cancelRequest`). This is a structural mirror
  of sig `#1` on the receive side.
- **Suspected sig `#6`**: exactly one request reaches
  `gen_request_sync_begin` and never reaches
  `gen_wait_ready_signal_begin`. The current instrumentation is not
  granular enough to say which step is blocking.

---

## Phase 8 — Sig `#5` fix and sig `#6` instrumentation (T+5 days)

- Receiver-side fix for sig `#5`: extract the queued promise under the
  lock and fulfill it with a structured `kNETWORK_ERROR` exception
  once released. Same shape as sig `#1`. Submitted as
  [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672).
- New trace markers around `sendRequestInfo()` and
  `AgentConnection::sendRequestAndBufferInfo()`.
- `run6` confirms sig `#5` is gone; one in-progress request is still
  stuck after `gen_request_sync_begin`. The wedge persists.

---

## Phase 9 — Sig `#6` root cause + fix (T+5 days)

- Add fine-grained instrumentation across the entire `sendRequestInfo()`
  body (`gen_send_assign_buffer_begin / step / end`, etc.).
- `run7` shows exactly one in-progress generation request reaches
  `gen_send_assign_buffer_begin` and never reaches `_step` or `_end`.
- Code reading of `BaseTransBufferManager::assignBufferIndex()`
  confirms it does an unbounded `cv.wait` with no timeout, and
  `mRecvBufferCount` defaults to `1`. A single leaked recv buffer
  index permanently wedges every subsequent receive.
- The leak was a direct consequence of sig `#1`'s fix: the `!isReady`
  early-return path in `CacheReceiver::Impl::requestSync()` skips
  `receiveSync()` (and therefore `unformat()`'s
  `freeBufferIndexForRecv()` call) for every cancelled-after-ready
  request. **This is the only "Type 1 cascade" in the investigation —
  a fix that *produces* a new signature.**
- Fix: RAII-style cleanup vector in `sendRequestInfo()` (Layer A), and
  explicit free in the `!isReady` early-return path of `requestSync()`
  (Layer B).

**PR:** [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673),
chained on `#13640`.

---

## Phase 10 — `run8` validation: sig `#6` confirmed fixed, sig `#7` identified (T+6 days)

`run8` was run end-to-end with the sig `#6` fix in place plus
promise-trace markers. The harness still reported `NO RECOVERY`, but
the C++ trace counts and post-mortem stack dumps showed that the
wedge mechanism had **shifted off the TRT-LLM transceiver entirely**.

`py-spy` and `gdb -p ... thread apply all bt` against the ctx and
gen workers showed the gen-side `CacheReceiver` request thread parked
correctly. The actual wedge was on the **ctx-side `dataTransResp`
thread**:

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

This was initially classified as a **NIXL UCX-internal mutex
deadlock** — out of TRT-LLM scope. The classification was later
falsified by Phase 12; see below.

This phase also disambiguated two earlier hypotheses:

- `#7a` (the `drop_without_fulfill` trace marker firing 3 times) is
  not a bug — the marker is a leftover name from before the sig `#1`
  fix landed and currently fires on the line *immediately above* the
  new `set_exception(kNETWORK_ERROR)` call.
- `#7b` (the 14-of-30 receiver-side future stranding pattern) is a
  symptom of `#7`, not a standalone bug.

---

## Phase 11 — Cross-validation against an independent fix stack (T+7 days)

To check whether the chained signature fixes were the cause of any
misclassification, the same harness was run against an independent
TRT-LLM-side fix stack: PR `#13056`'s comprehensive refactor
(end-to-end `shared_ptr<LlmRequest>` lifetime, `BufferIndexHolder`
RAII, deadline enforcement, structured cancellation, `catch (...)`
hardening). Same `rc11` base, same harness.

**Result:** identical wedge. `pthread_mutex_lock` frame in
`CacheSender::Impl::response()` on a process built from completely
different fix mechanisms. Two independent investigations of the same
bug class converge on the same residual `#7`.

The independent stack's defensive layers (`BufferIndexHolder`
AUTO_RELEASE markers fired 13 times for happy-path completions;
`assignBufferIndex CANCEL` and `STILL_WAITING` markers fired zero
times) were silent and idle during the wedge — **not because they're
broken, but because the wedge is at a layer below where they
enforce**.

---

## Phase 12 — Direct-UCX experiment falsifies "NIXL plugin internal" classification (T+7 days)

Switched the cache transceiver from `backend: NIXL` to `backend: UCX`
(the direct UCX path that bypasses the NIXL plugin entirely).

**`rc11_ucx_run1`: this investigation's chained-PR variant + UCX
backend.** Pre-burst probe completes; burst completes; all five
recovery probes return silent `ReadTimeout`. The wedge fires
**identically** in a process where `libnixl.so` is **not loaded at
all**.

`gdb` post-mortem of the wedged ctx-side worker:

```text
Thread "dataTransResp" (LWP 706490):
#0  pthread_mutex_lock                                   [libc.so]
#1  tensorrt_llm::batch_manager::CacheSender::Impl::response()
                                                         [libtensorrt_llm.so]
```

— the **same TRT-LLM-side frame as the NIXL runs**. `grep nixl` on the
entire thread dump returns zero matches.

**This falsifies the Phase 10 framing.** Sig `#7` is reclassified from
"NIXL UCX-plugin internal mutex deadlock" to "**`pthread_mutex_lock`
wedge in `CacheSender::Impl::response()`**, exposed by the
cancel-during-transfer load shape across both NIXL and direct UCX
backends, **fixable in TRT-LLM**." The most plausible mutex (by source
code inference) is `mCondMutex` at the top of `response()`'s loop body.

Side effect: `pr13056_ucx_run1` (PR `#13056` + direct UCX) shows a
*different* failure — the pre-burst sanity probe fails with HTTP 500
because the ctx-side `mpi4py.futures.server` worker `SIGSEGV`s before
the first request completes. This is later root-caused in Phase 13/14
as a separate variant of the `#7` bug class.

---

## Phase 13 — `gdb` capture loop exposes new SIGSEGV manifestations (T+8 days)

Following Phase 12's recommendation to pin down the exact mutex with a
live `gdb` register inspection, two new runs were attempted with a
periodic `gdb` capture loop attached:

**`run9`: rc11 + our fixes + UCX.** The wedge changed character. Instead
of the silent `pthread_mutex_lock` deadlock from earlier phases, the
ctx mpi4py worker `SIGSEGV`s at iter 92 of the burst:

```text
_PyObject_GenericGetAttrWithDict
PyObject_GetAttr
_PyEval_EvalFrameDefault
```

The crash is in the **Python C-API**, not in C++ code directly. The
sig `#1` fix path (`promise_set_exception` for
`cancelled_after_ready_signal`) fired cleanly immediately before. This
looks like a Python wrapper around a C++ object being destructed by
one thread while another still holds a reference; the wedge in earlier
runs was masking it.

**`run10`: rc11 + PR `#13056` + UCX.** Result: ctx `mpi4py.futures.server`
worker `SIGSEGV`s on the **very first sanity-probe request** — single
chat completion, no concurrency, no burst, no cancellation. The MPI
segfault handler captures:

```text
tensorrt_llm::batch_manager::CacheSender::Impl::handleAsyncSend(AsyncSendResource&)
std::_Function_handler<...>::_M_invoke(...)
std::__future_base::_State_baseV2::_M_do_set(...)
std::__future_base::_Async_state_impl<...>::_M_run()
```

Top frame unambiguously inside `libtensorrt_llm.so`. **No UCX or NIXL
frames anywhere in the stack.** This is the cleanest evidence that
the bug class is TRT-LLM-internal: a synchronous crash, fully
deterministic, zero transport frames.

The initial source-level guess was that `resp.mRequest` was already null
when `handleAsyncSend` reached
`sendAndRemoveResponse(resp.mRequest->mRequestId, std::move(resp));`.
That guess was plausible but unproven. Phase 14 followed up with
instrumentation to confirm.

**Sig `#7` framing extended to a class of bugs in `CacheSender::Impl::*`** with
four observed manifestations:

| Variant | Where (top frame) | Triggering condition |
|---|---|---|
| Mutex-deadlock | `CacheSender::Impl::response()` → `pthread_mutex_lock` | Cancel-during-transfer under burst |
| ctx-mpi4py-exit | (parent `orted` survives, child exits, no signal trace) | Same burst + many `cancelled_after_ready_signal` |
| Python-getattr-SIGSEGV | `_PyObject_GenericGetAttrWithDict` (downstream of ctx C++ cleanup) | Same burst, mid-iter 92 |
| Async-send eval-order SIGSEGV | `CacheSender::Impl::handleAsyncSend` directly | First request — no concurrency required |

---

## Phase 14 — Eval-order root cause confirmed; idempotency gap identified; direct-UCX recovery boundary measured (T+8 days, *current*)

Phase 14 starts from the strongest Phase 13 artifact: `run10`'s
deterministic first-request `handleAsyncSend` SIGSEGV. The async-send
path was instrumented with the markers `enter_sendAsync`,
`enqueue_ready`, `enter_sendResponse`, `producer_move`, `enqueue_send`,
`consumer_wake`, `consumer_dequeue`, `preDeref`, `sendAndRemove_*`,
`sendSync_*`.

### `run14` falsifies the null-`shared_ptr` hypothesis

The first instrumented direct-UCX request reproduced the crash:

```text
[asyncSend-trace] enter_sendAsync reqId=876742104559616 llmReq=0xe037801ca08 useCount=1
[asyncSend-trace] enqueue_ready useCount=2
[asyncSend-trace] producer_move mReq=0xe037801ca08 useCount=2
[asyncSend-trace] consumer_wake front_mReq=0xe037801ca08 front_nonnull=1
[asyncSend-trace] consumer_dequeue mReq=0xe037801ca08 useCount=2
[asyncSend-trace] preDeref reqId=876742104559616 mReq=0xe037801ca08
!!!!!!! Segfault encountered !!!!!!!
```

The same `shared_ptr` value flows through the entire pipeline,
`useCount=2`, and the `preDeref` log successfully reads
`resp.mRequest->mRequestId`. So the request object is alive and
non-null immediately before the call. A second run with traces inside
`sendAndRemoveResponse()` showed that no `sendAndRemove_enter` trace
fired before the SIGSEGV. This places the crash **between `preDeref`
and the callee body** — in argument construction, not in the callee.

### Confirmed root cause: read/move argument evaluation order

The exact problematic expression is:

```cpp
sendAndRemoveResponse(resp.mRequest->mRequestId, std::move(resp));
```

Before PR `#13056`, `Response::mRequest` was a raw pointer. Moving the
`Response` copied the raw pointer value, so even an unlucky argument
evaluation order did not empty the source field. PR `#13056` changed
the field to `std::shared_ptr<LlmRequest>`. Moving the `Response` now
move-constructs the `shared_ptr` into the callee argument and leaves
the source `resp.mRequest` empty. C++ does not guarantee left-to-right
evaluation of function arguments, so the compiler is allowed to
evaluate `std::move(resp)` before `resp.mRequest->mRequestId`. In that
order, the first argument reads a moved-from `resp.mRequest` and can
SIGSEGV before the callee body runs.

The minimal fix:

```cpp
TLLM_CHECK(resp.mRequest != nullptr);
auto const reqId = resp.mRequest->mRequestId;
sendAndRemoveResponse(reqId, std::move(resp));
```

This is the **L7** layer in the defect-class stack — see
[`03-defect-class-stack.md`](03-defect-class-stack.md#l7--eval-order-ub).
It is required by any approach that closes L2 (`shared_ptr<LlmRequest>`),
i.e. approaches B, C, and D.

### `run14c`: first-request crash fixed; `CONC=4` and one `CONC=16` recovers

After rebuilding `libtensorrt_llm.so` with the sequencing fix, the
first single-request probe completed and the `CONC=4`, 30 s burst
recovered cleanly. The original stress shape (`CONC=16`,
`BURST_DUR_S=60`) recovered once as well at `idle=30s`.

### `emplaceDone` assertion → idempotency gap (`L8`)

A repeat run showed a new generation event-loop failure:

```text
RuntimeError: [TensorRT-LLM][ERROR] Assertion failed: emplaceDone
(.../cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:2992)
self.impl.add_sequence(req.py_request_id, ...)
```

The generation scheduler can surface the same logical
`DISAGG_GENERATION_INIT` request over many iterations while it waits
for KV transfer to complete. The Python operations
`_prepare_disagg_gen_init()` (calling each resource manager's
`prepare_resources()`) and `_recv_disagg_gen_cache()` (starting
`request_and_receive_async()`) **must be idempotent by logical
`py_request_id`**, not by Python object identity.

Two local guards close this:

- `_disagg_gen_init_prepared_ids`, keyed per `ResourceManagerType`.
- `_disagg_gen_kv_recv_started_ids`. The synchronous path discards the
  id on exception so a real failed start can still be retried.

This is **L8**. It applies regardless of which approach lands.

### Combo (Approach D) recovery boundary

After landing the eval-order fix and the idempotency guards on top of
PR `#13056` + PR `#13495` (the combo), the latest direct-UCX repeats
recovered:

| Test | Result |
|---|---|
| `CONC=16`, 60 s, 5 iter | 5/5 recovered |
| `CONC=24`, 60 s and 90 s, 5 iter | 5/5 recovered |
| `CONC=32`, 90 s, 5 iter | 5/5 recovered (after explicit stale-server cleanup) |
| `CONC=64`, 90 s | wedged, no recovery 180 s |

The remaining direct-UCX `CONC=64` wedge is consistent with backlog /
timeout interaction rather than a stuck `sendSync()` call. The marker
balances confirm this:

| Marker (combo, `CONC=24`, 90 s) | Count |
|---|---:|
| `sendSync_before_format` / `[transfer-send] begin` / `_end` / `sendSync_after_format` / `sendAndRemove_exit` | 37 each |
| gen `pre-sendRequestInfoDirect` / `post-sendRequestInfoDirect` / `post-requestSync` | 45 each |
| gen `post-receiveReadySignal isReady=1` | 37 |
| gen `post-receiveReadySignal isReady=0` | 8 |

A representative timed-out request reached
`post-receiveReadySignal ... isReady=0` because ctx had already timed
out the same request before the response worker selected it for
sending. Head-of-line backlog, not a stuck transfer.

### NIXL transport validation

Switching to the NIXL transceiver path (still using the UCX plugin
underneath) on the combo:

| Test | Result |
|---|---|
| `CONC=32`, 90 s, 5 iter | 5/5 recovered, zero burst-time errors |
| `CONC=64`, 90 s, 5 iter | 5/5 recovered, ~zero burst-time errors |

The customer's transport is **clean through `CONC=64`** on the combo.
The remaining direct-UCX `CONC=64` wedge becomes the next scope:
add a direct-UCX `TransferStatus::release()` analog to `#13495`'s
NIXL primitive, using `ucxx::Request::cancel()` /
`ucp_request_cancel()`. See
[`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md).

### Phase 14 conclusion

Three confirmed fix candidates, listed in defect-class order:

1. **L7**: materialize `reqId` before `std::move(resp)` in
   `CacheSender::Impl::handleAsyncSend`. Fixes the deterministic PR
   `#13056` + direct-UCX first-request SIGSEGV.
2. **L8**: make generation-init resource preparation and KV receive
   start idempotent by `py_request_id`. Not specific to PR `#13056`;
   repeated scheduler visits should not re-run non-idempotent side
   effects.
3. **Direct-UCX cancellation primitive**: add a `TransferStatus::release()`
   wrapper around `ucxx::Request::cancel()` to close the residual L6
   gap on the direct-UCX path. Open follow-up.

The sig `#7` mutex deadlock variant's exact mutex address still needs a
runtime `gdb` register capture in a configuration that produces the
deadlock variant rather than one of the SIGSEGV variants — open
follow-up.

---

## What this timeline shows

Six rounds of "find bug → fix bug → find next bug" took ~8 days of
calendar time. The pattern is essentially the Type 2 cascade
relationship from
[`03-defect-class-stack.md`](03-defect-class-stack.md): each fix
removed a mask that was hiding a downstream bug that was already
firing. The one Type 1 cascade (sig `#1` fix produces sig `#6`) is
called out explicitly in Phase 9.

The investigation could not have been a single comprehensive fix at
T0 because four of the seven signatures emerged from investigation
itself, not from the field report. See
[`07-architectural-reflections.md`](07-architectural-reflections.md)
for the retrospective on what we would do differently.

---

## What to read next

- For the *current state* of the bug class (without the chronological
  story), see [`02-failure-signatures.md`](02-failure-signatures.md)
  and [`03-defect-class-stack.md`](03-defect-class-stack.md).
- For the candidate fix stacks, see
  [`06-fix-approaches/README.md`](06-fix-approaches/README.md).
- For the retrospective, see
  [`07-architectural-reflections.md`](07-architectural-reflections.md).
