# Bounded Polling Admission Accounting Addendum

| | |
|---|---|
| **Date** | 2026-06-24 |
| **Related PRs** | <https://github.com/NVIDIA/TensorRT-LLM/pull/15181>, <https://github.com/NVIDIA/TensorRT-LLM/pull/15356>, <https://github.com/NVIDIA/TensorRT-LLM/pull/15238> |
| **Related tests** | `TestQwen3_8B::test_auto_dtype_with_helix[fifo_v2-cudagraph:with_padding-pp1tp1cp4]`, `TestQwen3_8B::test_auto_dtype_with_helix[fifo_v2-cudagraph:with_padding-pp1tp2cp2]` |
| **Status** | Design note before implementation |

## Why This Addendum Exists

The bounded-polling PR chain made KV transfer deadlines observable by
replacing long `future.get()` waits with bounded status polling. That
is necessary for cancellation and fail-closed cleanup, but the Qwen
HELIX failures showed an unintended scheduling side effect: removing
the blocking wait also removed an implicit admission throttle.

The failed tests admitted a burst of generation-side disaggregated
requests quickly enough that many KV transfers exceeded the default
60 s transfer deadline. Extending `kv_transfer_timeout_ms` to 300 s
for the two targeted tests allowed both variants to pass with all
1319 responses completed and no KV transfer timeout warnings. The
same runs still emitted `num_fitting_reqs=0 ... may not have enough
kvCache` warnings, which confirms resource pressure but separates it
from the 60 s deadline failure.

The root issue is not that bounded polling should stop admission. It
is that admission accounting must count requests that have already
entered KV-transfer progress, even though those requests are not yet
decode-ready.

## Self-Review Of The Plan

The implementation direction is sound:

- Keep bounded polling so timeout checks, cancellation, health checks,
  and cleanup can run while transfers are unresolved.
- Preserve admission when capacity remains; bounded polling must not
  become "wait for all current transfers before admitting anything."
- Count `DISAGG_GENERATION_TRANS_IN_PROGRESS` requests as occupying
  generation admission capacity.
- Do not treat `DISAGG_GENERATION_TRANS_IN_PROGRESS` requests as
  compute/decode-ready.
- Start the timeout clock only when the request is admitted into the
  transfer-in-progress state.

The main caveat is the definition of "capacity." Counting
transfer-in-progress requests against the existing scheduler budget is
necessary, but may not be sufficient if the model/KV budget is much
larger than practical KV-transfer backend capacity. If that happens,
we will still need an explicit transfer-admission window or credit
limit derived from transfer-buffer/backend capacity.

## Queueing vs Transfer Service Rate

The 300 s timeout experiment is important evidence: with bounded
polling enabled, the failing Qwen HELIX tests passed when
`kv_transfer_timeout_ms` was extended from 60 s to 300 s. That points
away from a hard deadlock. Transfers continued to make progress; the
60 s failure mode was primarily deadline accounting under burst
admission.

The precise distinction is:

```text
time in DISAGG_GENERATION_TRANS_IN_PROGRESS
  = queueing delay inside the transfer machinery
  + actual transport service time
  + completion / consensus observation delay
```

`DISAGG_GENERATION_TRANS_IN_PROGRESS` means the request has been handed
to the transfer machinery. It does not guarantee that the request's KV
bytes are actively moving on the transport at that instant. Admitting a
large burst can therefore start many timeout clocks for requests that
are effectively waiting behind the physical transfer service lane.

This means the downside of burst admission does **not** require proving
that the raw transport service rate degrades when more requests are
queued. Even if the service rate stayed roughly constant, requests near
the back of the burst can exceed the request-level transfer deadline
because their timeout starts before they receive service.

Current defaults make the mismatch visible:

- The C++ transceiver can store many generation receive futures in
  `mRequesterFutures`; that vector is not the transport concurrency
  limit.
- In the C++ data transceiver, request-side concurrency is disabled by
  default because `TRTLLM_REQUEST_KV_CACHE_CONCURRENT` is unset. The
  receive path therefore uses one shared request worker / receive
  resource by default.
- `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM` defaults to `1`, so the
  sender side also defaults to one request-level send worker.
- Preallocated receive buffers default to one slot unless
  `TRTLLM_REQUEST_KV_CACHE_CONCURRENT` is enabled. With concurrent mode
  enabled, `TRTLLM_KVCACHE_RECV_BUFFER_COUNT` defaults to `2`.
- In the Python/V2 transceiver, `max_concurrent_sessions` is very large
  (`max_batch_size * 20000`), while `TRTLLM_KV_TRANSFER_NUM_THREADS`
  defaults to `1`. The NIXL agent has its own internal thread count
  (`TRTLLM_NIXL_NUM_THREADS`, default `8`), but that is not the same
  thing as allowing thousands of request-level transfers to receive
  service concurrently.

So the current shape is:

```text
requests allowed into transfer-in-progress:
  many / effectively unbounded at the request-state layer

request-level transfer service resources:
  small by default, often approximately one lane
```

That mismatch explains why a bounded-polling design needs explicit
admission accounting even if the transport itself is healthy.

## Cross-Framework Principle

"Serialize request-level KV transfers by default" is not the general
principle across LLM inference systems. The broader principle is that
KV memory and KV movement are schedulable resources. The serving stack
should bound and account them the same way it bounds decode batch
slots, KV blocks, and prefill/decode work.

This is aligned with the direction of other LLM serving systems:

- vLLM's PagedAttention work made KV-cache memory management a first-
  class serving problem.
- SGLang's runtime uses RadixAttention to make KV reuse central to
  scheduling efficiency.
- Mooncake-style disaggregated serving treats KV cache as the center of
  the serving architecture and uses SLO/load-aware scheduling around it.
- LMCache-style designs emphasize batched KV movement and compute/I/O
  pipelining rather than treating transfer as an incidental side effect.

The better long-term shape is therefore not simply "keep transfer
concurrency at one." It is:

```text
transfer admission window:
  bounded and rank-consistent

actual transfer concurrency:
  configurable / autotuned per backend and topology

timeout accounting:
  starts when a request is admitted into the transfer window, not while
  it is merely waiting behind that window

metrics:
  report queued transfer-admission latency separately from active
  transport service time
```

Increasing physical transfer concurrency can be a valid performance
optimization, but it is a separate tuning track from the correctness
fix. The risks are GPU transfer-buffer memory growth, dynamic-buffer
fallback or NIXL failure paths, decode/prefill bandwidth contention,
worse tail latency, and more complex cancellation / cleanup races.

## Current Behavior

On the generation side, `CapacityScheduler` first collects
`DISAGG_GENERATION_INIT` requests into `pendingDisGenInitRequests`.
That vector is only the candidate list. The admitted subset is the
returned `fittingDisaggGenInitRequests`.

`TrtGptModelInflightBatching::prepareDisaggGenInitRequests()` then
does two things for every request in `fittingDisaggGenInitRequests`:

1. Allocate KV cache by treating the request as a context request.
2. Call `CacheTransceiver::requestAndReceiveAsync()`.

`requestAndReceiveAsync()` records the transfer start timestamp,
creates the receive future, stores it in `mRequesterFutures`, and sets
the request state to `DISAGG_GENERATION_TRANS_IN_PROGRESS`.

The surprising part is state classification:

- `DISAGG_GENERATION_TRANS_IN_PROGRESS` is not
  `isGenerationInProgressState()`.
- `isGenerationInProgressState()` currently covers
  `DISAGG_GENERATION_TRANS_COMPLETE`, `GENERATION_IN_PROGRESS`, and
  `GENERATION_TO_COMPLETE`.

That distinction is correct for compute scheduling: a transfer-in-
progress request is not decode-ready. But `CapacityScheduler` also
uses generation-in-progress style checks as part of resource
accounting. As a result, a transfer-in-progress request can be polled
by `checkDisaggGenTransferStatus()` without being counted as an
already-admitted generation request for the next wave of
`DISAGG_GENERATION_INIT` admissions.

The old blocking behavior accidentally throttled the loop:

```text
admit a scheduler-sized wave
if all useful active work is transfer-in-progress:
    wait until at least one transfer completes
loop again
```

Bounded polling changed this to:

```text
admit a scheduler-sized wave
poll transfers with a short wait
if none completed, continue the scheduler loop anyway
admit another wave if scheduler capacity appears available
```

The second shape is correct for liveness, but only if the scheduler's
capacity accounting includes already-admitted transfer work.

## Target Semantics

The design should split three concepts that are currently too easy to
collapse:

```text
admission/resource occupancy:
  DISAGG_GENERATION_TRANS_IN_PROGRESS
  DISAGG_GENERATION_TRANS_COMPLETE
  GENERATION_IN_PROGRESS
  GENERATION_TO_COMPLETE

compute/decode schedulable:
  DISAGG_GENERATION_TRANS_COMPLETE
  GENERATION_IN_PROGRESS
  GENERATION_TO_COMPLETE

transfer-admissible:
  DISAGG_GENERATION_INIT, limited by remaining admission capacity
```

This gives the intended bounded-polling behavior:

```text
poll active transfers
account active transfers as occupying capacity
compute remaining capacity
admit pending DISAGG_GENERATION_INIT requests only up to remaining capacity
```

Examples:

```text
capacity = 128
active transfer-in-progress = 80
decode-ready/in-generation = 20
remaining admission room = 28
=> admit up to 28 new transfer requests
```

```text
capacity = 128
active transfer-in-progress = 128
remaining admission room = 0
=> poll transfer progress and admit nothing new until capacity opens
```

## Implementation Sketch

Do not broaden `isGenerationInProgressState()` to include
`DISAGG_GENERATION_TRANS_IN_PROGRESS`. That helper is used by decode
paths and would make a transfer-in-progress request look compute-
schedulable.

Instead, add a scheduler-local admission-occupancy concept, for
example:

```text
occupiesGenerationAdmissionCapacity(req)
```

The helper should include `DISAGG_GENERATION_TRANS_IN_PROGRESS` plus
the existing decode-ready generation states.

Capacity scheduling should then keep separate notions of:

- requests that occupy admission capacity and reserve KV budget;
- requests returned to the microbatch scheduler as compute work;
- `DISAGG_GENERATION_INIT` requests returned as
  `fittingDisaggGenInitRequests` and then submitted to transfer.

For transfer-in-progress requests:

- count them against `mMaxNumRequests` or the selected admission
  budget;
- reserve their already-allocated KV blocks;
- do not return them as `fittingRequests`;
- do not let `MaxUtilizationScheduler` pause or evict them as ordinary
  started generation requests while their transfer future is unresolved.

For pending disaggregated generation-init requests:

- admit only up to the remaining admission capacity;
- call `requestAndReceiveAsync()` only for admitted requests;
- start `kvCacheTransferStart` only when the request enters
  `DISAGG_GENERATION_TRANS_IN_PROGRESS`.

For the idle/no-compute case:

- if no compute work is schedulable but transfers are active, perform a
  bounded wait slice for transfer progress;
- do not use an unbounded `future.get()` as the progress mechanism;
- after each bounded wait, re-run timeout checks and admission
  accounting.

## Risks And Gaps

1. **Existing scheduler capacity may not equal transfer capacity.**
   If KV/model capacity allows a very large wave, counting
   transfer-in-progress requests against that budget may still admit
   too many transfers. The fallback is an explicit transfer window or
   credit counter.

2. **Sender-side admission is separate.** The discussion above focuses
   on generation-side `requestAndReceiveAsync()`. Context-side
   `respondAndSendAsync()` can also create transfer pressure and may
   need analogous sender-side credits if logs show send-side flooding.

3. **Do not leak compute readiness into admission accounting.** A
   broad helper change can regress decode scheduling by treating
   transfer-in-progress requests as runnable. Keep the new predicate
   local to admission/resource accounting unless every call site is
   audited.

4. **Do not make transfer-in-progress evictable.** Max-utilization
   policies can pause started requests to free budget. A
   transfer-in-progress request has an unresolved future and should not
   be paused or have KV reclaimed without the cancellation/quiescence
   protocol.

5. **Rank consistency still matters.** Admission decisions that affect
   distributed batch composition must stay rank-consistent. This note
   complements, but does not replace, the Phase 1 consensus contract.

6. **Timeout metrics need clear semantics.** If we introduce a queued
   transfer-admission state later, report queued latency separately
   from in-progress transfer elapsed time.

## Validation Plan

Add focused scheduler tests:

- transfer-in-progress requests consume admission capacity;
- transfer-in-progress requests are not returned as compute-schedulable
  generation requests;
- pending disaggregated generation-init requests are still admitted
  when remaining capacity exists;
- no new disaggregated generation-init request is admitted when
  transfer-in-progress requests fill the admission budget;
- max-utilization scheduling does not pause or reclaim
  transfer-in-progress requests.

Add integration evidence:

- rerun the two Qwen HELIX tests that exposed the issue;
- confirm all 1319 responses complete;
- confirm no KV transfer timeout warnings at the intended timeout;
- classify `num_fitting_reqs=0 ... may not have enough kvCache` as
  resource pressure, not as transfer timeout failure;
- log active transfer count, admission capacity, remaining admission
  capacity, `fittingDisaggGenInitRequests.size()`, and
  `mRequesterFutures.size()` to make future regressions obvious.

## Open Decision

The first implementation should likely start by counting
`DISAGG_GENERATION_TRANS_IN_PROGRESS` against the existing scheduler
admission budget. If that still admits a burst larger than the transfer
backend can service safely, add a dedicated transfer-admission budget
as a second step rather than weakening bounded polling.
