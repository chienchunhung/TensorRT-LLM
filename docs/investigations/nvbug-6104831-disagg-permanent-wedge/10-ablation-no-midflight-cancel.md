# 10 — PR #13713 value proposition: why the cancel / RAII / lifetime / fail-closed surface is load-bearing

This section answers the three questions a PR #13713 reviewer typically
asks first:

1. **What fails, and where in the code does it fail?**
2. **Why does it fail?** (root cause, not symptom)
3. **How does PR #13713 help — and specifically, why is mid-flight
   cancellation necessary?**

The answers are backed by six A/B experiments comparing PR #13713 head
(`local/pr13713-rc13-clean`) against an ablation branch with the cancel
surface removed (`local/pr13713-no-midflight-cancel`).

---

## Why this section exists (value positioning)

PR #13713 does not close a single customer-reported regression. The
NVBug 6104831 wedge is closed at lower concurrency by simpler subsets
of the stack. What PR #13713 lands is a **comprehensive defensive
surface** over the disaggregated KV-cache transceiver — five
interlocking layers that close the latent invariant gaps in the cancel
/ cleanup / lifetime / quiescence semantics.

The temptation in reviewing work like this is to ask "what bug does
each line fix?" and to push back on anything that isn't a 1:1 customer
regression repro. The cost of that framing is exactly what this
investigation spent on the path from sig `#1` to sig `#8`: peeling one
signature, finding another behind it, peeling that, finding another.
Sigs `#1`–`#8` are eight sequential discoveries of the same underlying
gap, surfacing differently under different load shapes, transports, and
rc-level scheduler defaults. Every signature looked like its own bug
until we proved it was the same gap.

The value PR #13713 offers is **not** another rev of "fix the next
signature". It is **the close-out of the invariant class** — the
position from which future load shapes, transport backends, and
scheduler changes do not reopen this investigation. Future-proofing
the transceiver against a class of latent bugs is, in this codebase,
a higher-leverage spend of review effort than fixing them individually
as they surface.

The headline TL;DR, in three rows:

| Tier | Failure class | Closed by |
|---|---|---|
| **1 — Correctness (memory safety)** | Baseline `kv_transfer_timeout_ms` frees a buffer NIXL is still pinning → NIXL eventually writes into the reclaimed buffer → silent corruption of an unrelated request's response. | Layers 1 + 2 + 5 (mid-flight cancel + poison + fail-closed) |
| **2 — Operability (debuggability)** | `NixlTransferStatus::wait()` is unbounded. Python's `kv_transfer_timeout_ms` cannot reach the C++ worker. Terminal peer failures leak workers indefinitely. | Layer 1 (`release()`) — the only application-level exit from NIXL's submit/wait API |
| **3 — Quality of service** | `Broken promise` cascade (89–162 events per burst), buffer-pool starvation on cancel exit paths, unbounded recovery from peer slowdowns. | Layers 2, 3, 4 in combination |

---

## What fails, and where

Four concrete failure modes, each with code site and observable
symptom:

### 1. Permanent worker wedge — the NVBug 6104831 customer report

**Symptom:** generation pod stops responding after the first burst.
Workers stay alive (no crash, no exit). Generation event loop never
recovers. Probes hit `ReadTimeout`.

**Where:** a sender's C++ async thread is parked in
`NixlTransferStatus::wait`
(`cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp`)
called from `AgentConnection::send`
(`cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp`).
`nixlAgent::getXferStatus` keeps returning `NIXL_IN_PROG` because the
peer has gone silent. The Python event loop has already marked the
`LlmRequest` as error via `kv_transfer_timeout_ms` and called
`_terminate_request`, but the C++ worker has no application-level
signal to break out.

### 2. Broken-promise cascade

**Symptom:** 89–162 `std::future_error: Broken promise` exceptions per
burst on cancel-heavy workloads (Experiment 4: 89; Experiment 6: 162).

**Where:** `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` —
`receiveAsync` and the recv-side machinery. The race: cancel and
completion paths fulfill the same `std::promise<TransferResult>`
concurrently. Whichever loses sees a destroyed promise. The exception
trace points at the *consumer* of the future, not the producer race,
which is why this signature took multiple investigation phases to
root-cause.

### 3. Buffer-pool starvation (sig `#6`)

**Symptom:** receiver workers stuck in
`BaseTransBufferManager::assignBufferIndex`, parked on `mBuffersCV`
indefinitely. Permanent wedge of the receiver-side worker.

**Where:** `cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp` plus
the early-return path in
`cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp::CacheReceiver::Impl::receiveSync`.
The latter leaks the buffer index slot without returning it to the
pool; after enough cancellations, every slot is leaked and new
transfers wait forever.

### 4. Silent use-after-free of an NIXL-pinned buffer (the memory-safety hazard)

**Symptom:** HTTP 200 responses with garbled bytes. *Invisible to
throughput metrics.* The worst kind of failure: the orchestrator sees
healthy responses and routes more traffic at the corrupted worker.

**Where:** opened by the combination of `kv_transfer_timeout_ms`-driven
`_terminate_request` (`tensorrt_llm/_torch/pyexecutor/py_executor.py`)
and the absence of any application-level cancellation of the NIXL
request. NIXL retains the destination buffer pin; TRT-LLM reclaims the
buffer to the allocator; the next request gets the same buffer; NIXL's
queued push eventually writes the original payload into the new
request's buffer.

We did not observe (4) directly under AddressSanitizer (see "Honest
gaps"). The architectural timeline is given in "Why it fails"; the
observable proxy is that Experiment 6 directly observed Layer 5
shutting `PyExecutor` down by design exactly when (4)'s preconditions
held on head.

---

## Why it fails

The shared root cause is a missing invariant on the disaggregated
transceiver:

> Cancellation of a `LlmRequest` must drain — or at minimum
> *signal-to-NIXL* — any in-flight transport handle owned by that
> request **before** the request's destination buffer is returned to
> the allocator.

The baseline `rc11` / `rc13` code does not enforce this invariant.
Four contributing facts:

**(a) NIXL's `submitTransferRequests → getXferStatus` API has no
application-level wake-up.** Three ways out of `NIXL_IN_PROG`: peer
ack (`NIXL_SUCCESS`), NIC / agent error, or `releaseXferReq`. Without
`releaseXferReq`, the C++ thread parks in `getXferStatus` until the
peer behaves. `kv_transfer_timeout_ms` lives entirely on the Python
side — it changes Python state, not C++ thread state.

**(b) `std::promise<TransferResult>` is single-fulfillment.** Both the
cancel path and the natural-completion path can hold a reference. If
both fire, one wins and the other throws `Broken promise`.

**(c) `BaseTransBufferManager` uses raw buffer indices, not RAII.**
Every cancel exit path is its own opportunity to leak an index. Easy
to get right at write time; impossible to keep right as new exit paths
land.

**(d) C++ async workers reference `LlmRequest` by raw pointer /
reference.** If Python's `_terminate_request` destroys the
`LlmRequest` while the C++ worker is still unwinding from an
interrupted `NixlTransferStatus::wait`, the worker reads freed memory.

Each of (a)-(d) is benign in steady-state. Under cancellation pressure
(short timeouts, peer slowdowns, deployment kill signals, transport
backend changes) they compound into the four failure modes above. The
critical point: *new* sources of cancellation pressure — short timeouts
for SLA tightening, new transport backends, new schedulers that cancel
more eagerly — re-expose this gap automatically. The investigation
already saw this happen: rc13's default-on block reuse turned sig `#5`
back into sig `#8` overnight.

**The use-after-free timeline (failure mode #4) made concrete:**

```text
T+0   sender: submitTransferRequests(dst=X)        X pinned by NIXL
T+0   sender: status->wait()  (unbounded)
T+0   receiver: peer slow/stuck                    NIXL push queued, not yet written
T+5   deadline fires; "Marking as error"           X still pinned by NIXL
T+5   _terminate_request → LlmRequest destroyed    X returned to allocator
T+5–10 new request gets buffer at X                X holds new request's bytes
T+20+ peer recovers / NIXL drains                  NIXL writes original payload into X
                                                   ──► USE-AFTER-FREE
T+30+ client of new request sees HTTP 200          Response contains garbled bytes
```

A deployment with `kv_transfer_timeout_ms` set but without PR #13713's
defensive surface is **strictly less safe** than one without any
deadline-eviction at all — the latter at least doesn't free the buffer
out from under NIXL.

---

## How PR #13713 helps — and why mid-flight cancellation is the keystone

PR #13713 introduces five interlocking layers. **Layer 1 (mid-flight
NIXL cancellation) is the keystone** — without it, the rest of the
stack has nothing to act on.

### Layer 1 — `release()` on `NixlTransferStatus` + `AgentConnection::send` poll loop

Adds `NixlTransferStatus::release()` calling
`nixlAgent::releaseXferReq` under `mHandleMutex`. `AgentConnection::send`
(and `recv`) replaces the unbounded `status->wait()` with a poll loop
that checks a per-request cancel registry on each slice. On cancel
detection it calls `release()` and throws — unwinding the worker.

Code:
- `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.{h,cpp}`
  — `release()` + `mHandleMutex`
- `cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.{h,cpp}`
  — poll loop in `send()` / `recv()`, per-request cancel check

> **Why mid-flight cancellation is necessary, in one sentence:**
> without `release()`, `kv_transfer_timeout_ms` is a Python-level
> timeout that cannot actually interrupt the C++ thread doing the
> transfer — the C++ worker is stuck inside an NIXL API call with no
> application-level wake-up, and Python's "marked as error"
> disposition has no causal connection to the C++ worker's state.

**Proof.** Experiment 6, transient peer pause via SIGSTOP-gen-8004 for
20 s mid-burst:

| Branch | Worker recovery after SIGCONT |
|---|---|
| Ablation (no `release()`) | **NO RECOVERY** in 60 s probe window; HTTP 500 at +60 s |
| PR #13713 head | **HTTP 200 at +1.71 s** |

That's a ~50× differential under a *transient* failure. The
customer-reported NVBug 6104831 wedge is the canonical *terminal*
failure case: NIXL never returns, `release()` is the only exit.

### Layer 2 — `BufferIndexHolder` RAII + `poison()`

RAII guard around buffer-index acquisition; every exit path returns
the index to the pool. On a cancel-driven exception, the catch block
in `cacheFormatter.cpp` / `mlaCacheFormatter.cpp` calls
`sendHolder.poison()` — sets `mPoisoned` on `BaseTransBufferManager`.
`mPoisoned` is the "we cancelled but cannot prove the transport is
quiescent" flag.

Code:
- `cpp/tensorrt_llm/batch_manager/baseTransBuffer.{h,cpp}` — RAII + `poison()`
- `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp` — catch block
- `cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp` — catch block (MLA port)

**Depends on Layer 1.** The catch block only fires on a cancel-driven
throw. Without Layer 1's throw path, no signal.

**Proof.** Experiment 4 (1 s timeout, conc=64): ablation `NO RECOVERY`
count = 1 (iter 1 wedges, iters 2–5 abort at sanity probe); head `NO
RECOVERY` count = 0 plus 2/5 PASS. The RAII destructor returning
indices on every exit path is what unblocks subsequent iterations.

### Layer 3 — `std::shared_ptr<LlmRequest>` async lifetime

Changes `CacheTransceiver::sendAsync` / `receiveAsync` to take
`std::shared_ptr<LlmRequest>` instead of `LlmRequest&`. Pins the
request lifetime to the C++ async operation, independent of when
Python's `_terminate_request` runs. Closes the use-after-free race
that Layer 1 **exposes** — now that the worker can unwind mid-transfer,
it might unwind into a destroyed `LlmRequest`.

Code:
- `cpp/tensorrt_llm/batch_manager/dataTransceiver.{h,cpp}` — signature change

**Depends on Layer 1.** Layer 1 creates the race; Layer 3 closes it.
Asking "why not remove just the shared_ptr part?" gets: because Layer 1
created the race that the shared_ptr closes. Removing 3 reopens a race
that didn't exist before Layer 1.

### Layer 4 — Recv-side per-request idempotency

Guards `std::promise<TransferResult>` fulfillment against the
cancel / completion race that produces `Broken promise`.

Code:
- `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` — recv-side
  per-request guard

**Proof.** `Broken promise` counts: Experiment 4 ablation = 89, head =
0; Experiment 6 ablation = 162, head = 0. Cleanest single A/B signal
in the entire investigation.

### Layer 5 — `_fail_closed_for_unquiesced_disagg_transfer` (memory-safety policy)

Python reads `has_poisoned_transfer_buffer()` in
`_check_cache_transfer_errors`; if true, sets `shutdown_event` and
graceful-shuts-down `PyExecutor`. When Layer 2's `mPoisoned` is set,
we *know* we cancelled with non-zero probability of incomplete
quiescence; continuing to allocate buffers from a possibly-still-pinned
pool is unsafe. The only safe response is to take the pod out of
service explicitly so the orchestrator restarts it.

Code:
- `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py` —
  `has_poisoned_transfer_buffer()` shim
- `tensorrt_llm/_torch/pyexecutor/py_executor.py` —
  `_fail_closed_for_unquiesced_disagg_transfer`,
  `_check_cache_transfer_errors`
- `cpp/tensorrt_llm/batch_manager/cacheTransceiver.{h,cpp}` +
  `cpp/tensorrt_llm/nanobind/batch_manager/cacheTransceiver.cpp` —
  `hasPoisonedTransferBuffer` aggregation + binding

**Depends on Layers 1 and 2.** Layer 5 has no signal to act on without
Layer 2's poison flag; Layer 2 has no exception to catch without Layer
1's throw. Asking "can we land just Layer 5?" gets: no, Layer 5 has
no input.

**Proof.** Experiment 6 head log:

```text
[serve] Client error to http://localhost:8004/v1/chat/completions:
  400, message='Bad Request: PyExecutor has already been shutdown.'
```

`has_poisoned_transfer_buffer present: True` on head; `False` on
ablation. The fail-closed shutdown fired on head exactly when designed.
Ablation has no such guard and proceeds into the UAF window.

### Why the layers must land as a unit

```text
   Layer 1 (mid-flight cancel)
   release() + poll loop in send/recv
        │ throws on cancel
        ▼
   Layer 2 (RAII + poison)
   BufferIndexHolder catches, sets mPoisoned
        │ exposed as has_poisoned_transfer_buffer()
        ▼
   Layer 5 (Python fail-closed)
   _fail_closed_for_unquiesced_disagg_transfer
   → shutdown_event.set() → graceful PyExecutor exit → HTTP 400

   Independent races opened/closed by Layer 1:
   Layer 3 (shared_ptr<LlmRequest>): keeps the request alive while
            the unwinding C++ worker still references it.
   Layer 4 (recv-side idempotency): guards std::promise against
            double-fulfillment when cancel and completion race.
```

There is no consistent subset to ship. "Just Layer 1" leaves Layer 3's
race uncovered (UAF on `LlmRequest`) and re-introduces Layer 4's
broken-promise cascade. "Just Layer 5" has no input. "Layers 1+3+4 but
skip 5" leaves the buffer-pool corruption invisible to Python and lets
the worker keep serving from a possibly-corrupted pool.

---

## Empirical evidence — six experiments

| # | Workload | Branches | Headline result |
|---|---|---|---|
| 1 | conc=64 NIXL+UCX, 60 s timeout | ablation only | PASS, defenses dormant |
| 2 | conc=256 NIXL native, 60 s timeout | ablation only | PASS — concurrency alone doesn't trigger the failure class |
| 3 | conc=64 NIXL native, **1 s timeout** | ablation only | **WEDGE** in iter 1; 89 `Broken promise`; iters 2–5 abort |
| 4 | conc=64 NIXL native, 1 s timeout | A/B | Ablation: WEDGE; head: 2/5 PASS, **0 `Broken promise`** |
| 5 | conc=64 NIXL native, **5 s timeout** | A/B | Both PASS, identical marker counts (13/13/13/0) — head's defenses fire silently |
| 6 | conc=64, 5 s timeout, **SIGSTOP gen-8004 for 20 s** | A/B | Ablation stuck >82 s; head recovers in **1.71 s** + Layer 5 shuts down PyExecutor by design |

### Regime spectrum

| Timeout / Failure | Cancel-path pressure | Ablation | PR #13713 head |
|---|---|---|---|
| 60 s (production default) | dormant | PASS | PASS |
| 5 s | fires silently — natural completion bails ablation out | PASS | PASS |
| **1 s** | **saturated** | **permanent wedge** | 2/5 PASS, no permanent wedge |
| **5 s + SIGSTOP** | **natural completion cannot bail out** | NO RECOVERY in 60 s; UAF window opens | recovers in 1.71 s; **Layer 5 fail-closed fires** |

The regime spectrum is the central data point for the "future-proofing"
argument. Under today's production defaults the ablation looks fine.
Tighten the timeout to 1 s, or introduce a peer that pauses for 20 s,
and the gap is immediate and severe. A future load shape that didn't
exist when the code was written can reopen this without warning. The
PR #13713 surface caps that exposure.

---

## Tradeoff acknowledgement

PR #13713 deliberately trades **apparent availability** for **verified
safety + bounded operability**:

| Aspect | Ablation | PR #13713 head |
|---|---|---|
| Apparent availability during brief peer pause | Higher (degraded serving) | Lower (affected PyExecutor shut down) |
| Memory safety after cancel with unknown quiescence | None — possible silent UAF | Enforced — preemptive shutdown |
| Failure visibility to orchestrator | Silent (correct or corrupted HTTP 200) | Loud (explicit HTTP 400) |
| Recovery from terminal peer failure | None (unbounded `status->wait()`) | Bounded by `kv_transfer_timeout_ms` |
| `Broken promise` cascade | 89–162 per burst | 0 |
| Worker-level recovery from transient pause | >80 s | 1.71 s |

For orchestrated production (the customer scenario in NVBug 6104831),
head is strictly preferred: a shut-down worker can be restarted and
serves correct responses afterward; a worker silently serving
corrupted responses is a much harder correctness hazard to detect.

---

## Why we ship default-OFF

The empirical case above defends the layers' correctness *as a unit*.
It does not, on its own, defend defaulting them ON for every
deployment. After merging the candidate stack with `upstream/main`,
three CI failures surfaced that point to a deeper architectural gap:
**none of the new layers were designed with a consensus story across
the parallelism strategies the executor supports** (TP, PP, EP). The
five layers all make per-rank decisions about cancellation and
deferred cleanup. The V1 + C++ transceiver path the layers extend has
no consensus mechanism; once each rank decides independently, the
ranks split-brain on which requests are still scheduled and which
have been freed.

### Three CI failures, one root cause

| RC | Test | Surface | Per-rank decision that diverges |
|---|---|---|---|
| **RC-1** | `TestQwen3_5_35B_A3B.test_bf16_mtp[mtp_on]` | MTP scheduler vs `_can_terminate_request_now` | Deferred request stays in `active_requests`, but MTP speculative state is partially torn down. Next forward step's `_prepare_inputs` returns `None` instead of `(inputs, gather_ids)`. `mtp_off` variant passes — isolates MTP path as trigger. |
| **RC-2** | `TestDeepSeekV3Lite.test_auto_dtype_with_helix[pp1dp2cp2 / pp2tp1cp2]` | TP / CP allgather | `checkContextTransferStatus` deadline check fires at slightly different wall-clock times per rank. Rank 0 includes a request in its batch; rank 1 excludes it. Different batches → one rank enters `tp_cp_allgather`, the other doesn't → MPI collective blocks indefinitely → 300 s hang detector. |
| **RC-3** | `TestDisagg.test_asymmetric_executor[llama-4proc-mpi_kvcache-90]` | PP termination handler | `DisaggPPTerminationHandler` callback was changed to `_do_terminate_request_if_safe`, which defers on in-transmission requests but has no retry path wired up for the PP case. KV blocks pinned indefinitely → pool exhaustion → CUDA illegal memory access on the next scheduling attempt. |

All three are the same defect class: **the deferred-cleanup logic
treats "is this request in transmission?" as a per-rank query and
acts unilaterally on the answer.** A rank that answers "yes, defer"
and a rank that answers "no, terminate" diverge on the next
iteration's batch and on whether the request's KV blocks have been
returned to the pool — neither of which is a per-rank property in TP,
PP, or EP.

### Contrast with the V2 + Python transceiver

The V2 + Python transceiver (`KvCacheTransceiverV2` in
`tensorrt_llm/_torch/disaggregation/transceiver.py`) was designed
with this exact concern. Every per-iteration outcome flows through
`_consensus_outcome`:

```text
CANCELLED on any rank   → globally cancelled
FAILED on any rank      → globally failed
COMPLETED on every rank → globally completed (else: re-poll next iter)
```

with TP allgather first then PP allgather
(`_ctx_consensus_outcome`). The deferred-cleanup pattern is already
there — `cancel_request` returns `False` if any task is mid-write,
and the caller retries next iteration — but it's anchored on a
*globally* consistent decision, not a per-rank one.

The V1 + C++ transceiver path PR #13713 extends does not have this
machinery. Adding consensus to that path is a substantial design
effort that has to thread through:

- the per-rank C++ `checkContextTransferStatus` /
  `checkGenTransferStatus` deadline checks (rank-time skew),
- the `DisaggPPTerminationHandler` callback contract (one callback
  per request across all PP ranks vs N callbacks per rank),
- the MTP scheduler's coupling between `_can_terminate_request_now`
  and speculative state teardown (atomicity requirement),
- and the EP routing layer's assumption that every expert sees the
  same active-request set on each iteration (we have not exercised
  this combination yet; the design has to anticipate it).

Shipping the cancellation + poison + deferred-cleanup surface
default-ON would force this design to land *under time pressure*,
with the regression evidence already in hand. Shipping default-OFF
gives the architectural rethink room to happen properly. The
follow-up design doc
([`docs/design/disagg-inflight-cancel-poison/README.md`](../../design/disagg-inflight-cancel-poison/README.md))
captures the rethink: the goal becomes "architecturally correct
request cancellation across V1/V2 × C++/Python × NIXL/UCX × TP/PP/EP"
rather than "make the existing in-flight cancel less aggressive".

### What `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1` covers

Single env-var opt-in, all-or-nothing within the cancellation surface:

| Layer / point | Default (unset) behaviour | `=1` behaviour |
|---|---|---|
| C++ `CacheSender::Impl::cancelRequest` in-flight-flag branch | Skipped; `cancelRequest` returns `false` for in-flight requests | Flips the cancel flag; `AgentConnection::send` unwinds |
| C++ `CacheReceiver::Impl::cancelRequest` in-flight-flag branch | Symmetric to sender | Symmetric to sender |
| C++ `checkContextTransferStatus` / `checkGenTransferStatus` deadline force-evict | Logs once-per-request WARN; future left to complete naturally | Cancels + marks `kDISAGG_TRANS_ERROR` + erases future |
| C++ `catch (...)` poison call in formatters (Layer 2b) | Catch never fires (the in-flight cancel path that throws is gated upstream) | Catch fires; pool-wide `mPoisoned` set |
| C++ Layer 5 fail-closed (transitive on poison) | Inert | Active |
| Python `_can_terminate_request_now` deferral | Returns `True` immediately (no deferral) | Defers when request is in disagg transmission state |
| Python `_handle_errors` `deferred_requests` population | Always `[]` | Populated with in-transmission requests |
| Python `DisaggPPTerminationHandler` callback choice | `_do_terminate_request` (synchronous) | `_do_terminate_request_if_safe` (defers) |
| Python `_check_kv_transfer_timeout` active cancellation + error marking | Logs the timeout once per request per role; future left to complete naturally | Cancels + marks error |
| Python `cancel_request()` / `has_poisoned_transfer_buffer()` helpers | No-op (returns `False`) | Delegates to C++ |

Orthogonal always-on pieces (no flag): `BufferIndexHolder` RAII,
`shared_ptr<LlmRequest>` async lifetime, recv-side promise
idempotency, `handleAsyncSend` eval-order fix, Python idempotency
guards on `_prepare_disagg_gen_init` / `_recv_disagg_gen_cache`,
nanobind `nb::keep_alive` on the NIXL agent / status pair.

The flag's atomicity is the load-bearing property: every point above
either ALL becomes active or ALL stays dormant. Partial enabling
re-introduces the per-rank-decision-without-consensus failure mode
(e.g. enabling C++ cancel but not Python deferral re-opens the UAF
window because Python frees while C++ may still touch the buffer).

---

## Honest gaps

Three gaps in the empirical case. The architectural arguments hold
without them; the experiments would make the case more compact:

1. **Direct ASan observation of the UAF on ablation.** Failure mode
   #4 rests on the architectural timeline plus the observation that
   Layer 5 fires by design on head. A ~3–4 hour follow-up
   (`-fsanitize=address` rebuild + re-run Experiment 6) would catch
   the `heap-use-after-free` directly with addresses and stack trace.
2. **SIGKILL injection (terminal peer failure).** All injections in
   this study are *transient*. The strict-necessity claim for Layer 1
   under terminal failures rests on the NIXL API shape and the NVBug
   6104831 production report. A ~30 min SIGKILL follow-up would
   directly show ablation never recovers while head unwinds and
   surfaces HTTP 400.
3. **Production-default 60 s timeout run on rc13-clean head.** The
   README cites earlier rc11 / rc13 validation; we have not re-run
   on the current rc13-clean state. Sanity-check follow-up.

---

## Appendix — per-experiment data

### Experiment 1 — conc=64 NIXL+UCX, 60 s timeout (ablation)

All 5 iters: 715 ok / 0 err / RECOVERY at idle=30 s. All markers zero.

### Experiment 2 — conc=256 NIXL native, 60 s timeout (ablation)

All 5 iters: 715–716 ok / 0 err / RECOVERY at idle=30 s. All markers
zero. Confirms throughput pressure alone doesn't trigger this failure
class.

### Experiment 3 — conc=64 NIXL native, 1 s timeout (ablation only)

```text
iter 1: 216 ok / 499 err / 715 total — 70 % error rate
iter 2–5: ABORT at sanity probe
OVERALL: FAIL (permanent wedge)
```

Markers: `Cannot cancel`=7, `ExcTO`=596, `MarkErr`=507,
**`Broken promise`=89** (1:1 with receiver-side timeouts). Receiver
workers parked in `waitForNotification`; promises destroyed when
`LlmRequest` torn down → broken-future errors.

### Experiment 4 — A/B at 1 s timeout

| Marker | Ablation | Head | Δ |
|---|---|---|---|
| `Cannot cancel request` | 7 | 6 | -1 |
| `exceeded total timeout` | 596 | 961 | +365 (head processes more) |
| `Marking as error` | 507 | 965 | +458 |
| **`Broken promise`** | **89** | **0** | **-89** |
| **`NO RECOVERY`** | **1** | **0** | **-1** |
| Iteration verdicts | F/F/F/F/F | P/F/F/P/F | +2 PASS |

### Experiment 5 — A/B at 5 s timeout

Both branches PASS all 5 iters: 716 ok / 0 err / RECOVERY at idle=30 s.
Markers identical across branches: `Cannot cancel`=13, `ExcTO`=13,
`MarkErr`=13, `Broken promise`=0. The cancel path fires on both
branches (queue-drain misses); the behavioural difference is
*downstream* — head's worker unwinds in ms via the Layer 1 throw;
ablation's spins for ~30 s until NIXL drains naturally, confirmed by
`elapsed 30-32 s > limit 5000 ms` log lines.

### Experiment 6 — SIGSTOP-injected peer pause

Timing:
```text
T+150 s after launch: SIGSTOP gen-8004
T+170 s:              SIGCONT gen-8004
Injector probes at +0, +1, +2, +5, +10, +20, +30, +60 s after SIGCONT
```

| Branch | First HTTP 200 after SIGCONT |
|---|---|
| Ablation | NO RECOVERY in 60 s; HTTP 500 at wall=82.23 s |
| Head | HTTP 200 at wall=1.71 s |

| Marker | Ablation | Head |
|---|---|---|
| `Cannot cancel request` | 9 | **0** |
| `exceeded total timeout` | 1130 | 807 |
| `Marking as error` | 967 | 854 |
| **`Broken promise`** | **162** | **0** |
| `NO RECOVERY` | 0 | 1 (Layer 5 — by design) |
| Iter verdicts | 3 PASS, 2 ABORT | 0 PASS, 1 NO RECOVERY + 4 ABORT |

Layer 5 firing in head's `front.log`:

```text
[serve] Client error: 400 'PyExecutor has already been shutdown.'
```

Causal chain: `release()` → catch block → `sendHolder.poison()` →
`mPoisoned=true` → `has_poisoned_transfer_buffer()=True` →
`_fail_closed_for_unquiesced_disagg_transfer()` → `shutdown_event.set()`
→ PyExecutor shutdown → HTTP 400.

---

## Reproduction artefacts

Branches: `local/pr13713-no-midflight-cancel` at commit `e7b5931227`
([fork](https://github.com/chienchunhung/TensorRT-LLM/tree/local/pr13713-no-midflight-cancel))
and `local/pr13713-rc13-clean` (PR #13713 head).

Smoke check on each rebuild:

```python
from tensorrt_llm.bindings.internal.batch_manager import CacheTransceiver
hasattr(CacheTransceiver, 'has_poisoned_transfer_buffer')
# expected: False on ablation, True on head
```

Wheel archive: `/home/chienchunh/wheel-archive/pr13713-no-midflight-cancel-<TS>/`
with `RESTORE.md` checklist.

Harness variants in `.repro/`:

- `harness-aggressive-timeout/` — 1 s timeout, 500 ms poll slice
  (Experiments 3, 4)
- `harness-5s-timeout/` — 5 s timeout, 2500 ms poll slice
  (Experiments 5, 6)

SIGSTOP injector at `/tmp/sigstop-injector.sh`; A/B chain runner at
`/tmp/run-sigstop-ab-chain.sh`.

Most relevant run-log directories under each worktree's `.repro/logs/`:
- Experiment 4 (head): `run_pr13713_head_aggressive_timeout_conc64_20260513_223841`
- Experiment 6 (ablation): `run_sigstop_ablation_20260514_003635`
- Experiment 6 (head): `run_sigstop_head_20260514_004813`

---

## Cross-references

- [`02-failure-signatures.md`](02-failure-signatures.md) — sigs `#1`,
  `#4`, `#5`, `#6` are the four failure modes named in "What fails".
- [`03-defect-class-stack.md`](03-defect-class-stack.md) — L1–L10
  layering. The PR #13713 five-layer surface closes L2 / L3 / L5 / L7
  and gives Python an explicit signal to act on for L8.
- [`04-reproduction.md`](04-reproduction.md) — original reproduction
  recipe. Experiment 6's SIGSTOP scenario is the closest in-process
  simulation of the production failure mode (peer genuinely
  unresponsive).
- [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md) —
  the combo approach this section empirically defends.
- [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) —
  follow-ups (ASan build, SIGKILL injection, response-content
  validation).
- Cross-investigation:
  [NVBug 6043291 (zombie worker pods)](../nvbug-6043291-zombie-worker-pods/README.md)
  — operationally adjacent failure mode; Layer 1 eliminates the same
  root cause (workers stuck in unbounded waits) from a different
  direction.
