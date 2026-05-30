# 16 — Diagnostic instrumentation findings: the wedge is mobile

**Status:** Empirical finding from two CI runs of the diag-instrumentation branch.
**Trigger:** CI test results from the cross-rank-divergence diagnostic branch (the one carrying Layer A–F `[DIAG-*]` log markers on top of the full disagg cancellation fix).
**Public references:** instrumentation lives in `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`, `cpp/tensorrt_llm/executor/cache_transmission/{ucx_utils,agent_utils}/connection.cpp`, and `tensorrt_llm/_torch/pyexecutor/py_executor.py` on the diag branch on the author's fork.

---

## 1. Summary

Two CI runs of the same test on near-identical code produced two *different* wedge signatures:

| Run | Phase where the wedge sat | Evidence |
|---|---|---|
| Earlier run (Layer A–D logs only) | Phase 1 — request-info / ready-signal handshake | A specific reqId got `mRemainSendCount` stuck at 1; never reached `[DIAG-CTX-READY-PRE]`. Only 1 of 2 expected counterparts sent request-info. |
| Later run (Layer A–E logs, post-merge with `upstream/main`) | **Past** Phase 1 (Phase 4 — data transfer, formatter, or future completion) | Every Phase 1/2/3 marker paired up cleanly: 104 unique reqIds reached `CTX-READY-POST`, all 640 UCX sends/recvs paired. Test still timed out at the pytest `--timeout=3600` boundary. |

**Conclusion:** the wedge is flaky in *mechanism*, not just in *manifestation*. Different race orderings produce wedges at different code-path points. The two captured signatures are best modeled as two distinct race-ordering outcomes of the same family of cross-rank scheduling-decision divergence.

This document is a corrective addendum to documents 12 and 13 in two respects:

1. It supersedes the implicit framing that the wedge was a single deterministic mechanism rooted in either horizontal-consistency ABBA (doc 12) or a transport-layer mpi_kvcache hang (doc 13). Both framings remain valid for the *specific* signatures they capture, but neither is the full picture: each is one race-ordering outcome.
2. It introduces Layer F instrumentation, designed to disambiguate the next failure into one of three Phase-4 sub-categories.

---

## 2. The instrumentation layers and what each one tells us

The `[DIAG-*]` instrumentation was added in waves, each layer covering one segment of the request lifecycle. Layers A–E shipped on the branch before the later CI run; Layer F was added in response to that run's findings.

### Layer A — ctx-side ready-signal correlation
Fires inside `CacheSender::Impl` to bracket the ctx-side state machine that decides when to emit a ready signal.

```
[DIAG-CTX-EMPLACE-RESP]   — sendAsync inserted a Response into mReadyResponses
[DIAG-CTX-RECV-INFO]      — recvRequestInfo received a peer's request-info
[DIAG-CTX-INIT-COUNT]     — mRemainSendCount[reqId] = N (first arrival)
[DIAG-CTX-DECR]           — count decremented (countBefore → countAfter)
[DIAG-CTX-AWAIT]          — count > 0 after decrement; still waiting
[DIAG-CTX-READY-PRE]      — count == 0; about to send ready signal
[DIAG-CTX-READY-POST]     — ready signal sent
```

### Layer B — UCX transport bracket
Fires inside `UcxConnection::send` and `UcxConnection::recv` so a hung UCX call shows up as ENTRY without DONE.

```
[DIAG-UCX-SEND-ENTRY] / -WAIT / -DONE
[DIAG-UCX-RECV-ENTRY] / -WAIT / -DONE
```

### Layer C — ctx-side exception drain
Fires when an exception propagates out of the response worker so we can see which reqIds were in flight at that moment.

```
[DIAG-CTX-EXCEPT]                   — caught exception; mPending size
[DIAG-CTX-EXCEPT-DRAIN]             — one reqId being drained out
[DIAG-CTX-EXCEPT-UNKNOWN] / -DRAIN  — analogous for unknown-type exceptions
```

### Layer D — gen-side requestSync bracket
Fires inside `CacheReceiver::Impl` to bracket the gen-side request-info send and ready-signal recv.

```
[DIAG-GEN-REQSYNC-ENTRY]      — worker picked up the request
[DIAG-GEN-SEND-INFO-PRE]      — about to send request-info
[DIAG-GEN-SEND-INFO-POST]     — send done
[DIAG-GEN-AWAIT-READY]        — parking on UCX recv for ready-signal tag
[DIAG-GEN-RECV-READY]         — ready-signal received
```

Plus the Python-side bracket on the scheduler decision that hands the request to C++:

```
[DIAG-PY-CTX-SEND]            — respond_and_send_async (ctx Python entry)
[DIAG-PY-GEN-RECV-ASYNC] / -SYNC  — request_and_receive_{async,sync} (gen Python entry)
```

### Layer E — per-rank dedup-set tracing and counterpart visibility
Fires across the Python recv-side dedup logic plus the ctx/gen counterpart-list logs. Designed to distinguish "rank-divergent dedup decision" from "rank-divergent schedule input" from "missing counterpart".

```
[DIAG-GEN-PREPARE-INPUT]           — _prepare_disagg_gen_init entry, with fitting_py_ids and prepared-set snapshot
[DIAG-GEN-RECV-INPUT]              — _recv_disagg_gen_cache entry, with input_py_ids and dedup-set snapshot
[DIAG-GEN-DEDUP-DECISION]          — per-request dedup decision (SKIP / CALL_RECV) with state
[DIAG-GEN-DEDUP-DISCARD-ROLLBACK-{SYNC,ASYNC}]  — rollback on request_and_receive_* throw
[DIAG-GEN-DEDUP-DISCARD-TERMINATE] — discard at request termination
[DIAG-CTX-COUNTERPARTS]            — ctx side: who was expected, who just sent
[DIAG-GEN-SEND-INFO-COUNTERPARTS]  — gen side: full list of ctx peers this rank will notify
[DIAG-GEN-SEND-INFO-DEST]          — per-peer dest in the send loop
```

### Layer F — post-ready-signal data-transfer bracket *(new)*
Fires around the data-transfer phase and the future-completion site. Designed to identify whether a Phase-4 wedge sits inside the formatter (sendAllBuffers/unformat) or after it (release / future map / promise.set_value).

```
Ctx side:
[DIAG-CTX-SENDSYNC-ENTRY] / -EXIT     — around sendSync (called from sendAndRemoveResponse)
[DIAG-CTX-FORMAT-ENTRY] / -EXIT       — around mCacheTransferLayer.format(*session)
[DIAG-CTX-PROMISE-SET]                — resp.mPromise.set_value() reached (future completes)

Gen side:
[DIAG-GEN-RECVSYNC-ENTRY] / -EXIT     — around receiveSync (inside requestSync)
[DIAG-GEN-UNFORMAT-ENTRY] / -EXIT     — around mCacheTransferLayer.unformat(session)
[DIAG-GEN-PROMISE-SET]                — requestAndPromise.mPromise->set_value() reached
```

---

## 3. Expected event sequence for a healthy request

Per reqId, on the ctx side:

```
PY-CTX-SEND
CTX-EMPLACE-RESP
  ← N × (CTX-RECV-INFO, CTX-COUNTERPARTS, CTX-DECR, optionally CTX-AWAIT)
  with CTX-INIT-COUNT firing on the first one
CTX-READY-PRE
CTX-READY-POST
CTX-SENDSYNC-ENTRY
CTX-FORMAT-ENTRY
  ← many UCX-SEND-* events (per-peer / per-block sends)
CTX-FORMAT-EXIT
CTX-SENDSYNC-EXIT
CTX-PROMISE-SET
```

Per gen-side counterpart of the same reqId:

```
PY-GEN-RECV-ASYNC (or -SYNC)
GEN-REQSYNC-ENTRY
GEN-SEND-INFO-PRE
GEN-SEND-INFO-COUNTERPARTS
N × GEN-SEND-INFO-DEST
GEN-SEND-INFO-POST
GEN-AWAIT-READY
GEN-RECV-READY                ← ctx side has just sent the ready signal
GEN-RECVSYNC-ENTRY
GEN-UNFORMAT-ENTRY
  ← many UCX-RECV-* events
GEN-UNFORMAT-EXIT
GEN-RECVSYNC-EXIT
GEN-PROMISE-SET
```

A reqId that has an ENTRY but no matching EXIT at any phase identifies the wedge phase precisely.

---

## 4. Run-by-run findings

### 4.1 Earlier run (Layer A–D only)

- Failing test: `test_asymmetric_executor[llama-6proc-ucx_kvcache-90]` (the same parametrization across both runs).
- Wedged reqId (representative): `2175266568904705`.
- Marker count for the wedged reqId:
  - `CTX-RECV-INFO`: 2
  - `CTX-INIT-COUNT`: 2 with `initialCount=2`
  - `CTX-DECR`: 2 events, each `countBefore=2 countAfter=1`
  - `CTX-AWAIT`: 2 events, each `remainingCount=1`
  - `CTX-READY-PRE` / `-POST`: 0
  - `GEN-AWAIT-READY`: 1 (one gen counterpart parked forever)
  - `PY-GEN-RECV-ASYNC`: 1
- Interpretation: ctx expected 2 gen counterparts to each send a request-info. Only 1 of the 2 actually called `request_and_receive_async` on the Python side. Count never reached 0; ready signal never sent; the one gen counterpart that did call recv hung forever on `GEN-AWAIT-READY`.
- The two `CTX-INIT-COUNT` events correspond to two distinct `CacheSender::Impl` instances on rank 0 (two consecutive gtest cases reusing the same `mRequestId`), each of which independently received 1 of 2 expected counterparts.
- **Wedge phase:** Phase 1 — request-info / ready-signal handshake.
- **Suspected root cause at the time:** per-rank Python recv-side dedup state divergence (the `_disagg_gen_kv_recv_started_ids` set, populated only on the rank that actually called recv on the prior iteration, causing one gen rank to skip the call this iteration).

### 4.2 Later run (Layer A–E, post-merge with `upstream/main`)

- Failing test: same.
- Marker counts across all 104 unique reqIds:

| DIAG marker | Count | Implication |
|---|---|---|
| `CTX-EMPLACE-RESP` | 104 | 104 unique ctx-side reqIds entered `mReadyResponses` |
| `CTX-INIT-COUNT` | 104 | Each had its remaining-send-count initialized |
| `CTX-RECV-INFO` | 192 | Every gen-side counterpart sent its request-info (192 = 88 × 2 multi-counterpart + 16 × 1 single-counterpart) |
| `CTX-DECR` | 192 | All decrements happened |
| `CTX-AWAIT` | 88 | The 88 multi-counterpart reqs awaited on first decrement |
| `CTX-READY-PRE` / `-POST` | 104 / 104 | All 104 reqs reached `count==0`; all ready signals emitted |
| `GEN-AWAIT-READY` / `-RECV-READY` | 128 / 128 | All gen-side waiters armed and unblocked |
| `UCX-SEND-ENTRY` / `-DONE` | 640 / 640 | Every UCX send paired |
| `UCX-RECV-ENTRY` / `-DONE` | 640 / 640 | Every UCX recv paired |

Every Phase 1, 2, and 3 marker paired up internally consistent. Yet the test still timed out (pytest `--timeout=3600`).

Of 108 `[ RUN ]` markers across all gtest cases in the shard, only 24 received `[       OK ]`. The 84 unmatched RUN markers are **not 84 hung cases** — pytest's `--timeout-method=thread` killed the process after 1 hour during one specific case, so those 84 are cases that never got to run, not cases that ran and hung. **One** gtest case (the 25th in sequence) is the actual hung case, and it hung for ~57 minutes after the first 24 had finished in ~3 minutes total.

The 24 OK markers and their runtimes (visible because of stdout buffering, all flushed at timeout):
- 22 fast cases: 2–6 seconds each
- 2 long cases in `LlamaConTP2GenPP4`: ~318 s each (5.3 min)

The 5.3-minute cases ran near the timeout boundary, suggesting the actual hung case is the one that started just before or at the 1-hour mark.

- **Wedge phase:** **Past Phase 1 / 2 / 3.** Either inside the formatter on Phase 4 (`format` / `unformat`), or between phase 4 and promise fulfillment, or in test teardown.
- **Layer F was added in response** to disambiguate among those three sub-phases on the next run.

### 4.3 Comparing the two runs

The two runs were on functionally similar branches:
- Earlier run: Layer A–D logging on top of the cancellation-fix branch.
- Later run: same branch + Layer E logging + a clean merge of `upstream/main` (no behavioral changes to disagg paths beyond the merge).

A deterministic mechanism would produce the same wedge signature on both runs. They produce *different* signatures. This is the strongest evidence we have so far that the wedge is a flaky race with multiple manifestations, not a single deterministic bug.

This is also consistent with documents 12 §4 and the pre-existing observation that two byte-identical-tree runs (recorded in the gating inventory memory) produced different failure profiles across runs.

---

## 5. What we now know vs. don't know

**We know:**
1. The wedge is reproducible at the test level (every CI run of the diag branch on the same test hits a timeout).
2. The *mechanism* of the wedge differs across runs.
3. Phase 1 (request-info / ready-signal handshake) is *not* the only wedge site. At least one run wedges past it.
4. The per-rank Python recv-side dedup hypothesis from doc 12 / earlier investigation explains the Phase 1 signature in the earlier run, but does not by itself explain the later run.

**We don't know yet:**
1. Where in Phase 4 the later run wedged. Layer F was added precisely to answer this.
2. Whether the *underlying* race is a single common cause that manifests at different code-path points depending on timing, or whether there are multiple independent races each producing one signature.
3. Whether the wedge is dependent on the specific gtest case that runs 25th (case ordering effect) or whether any of the 14 cases in any of the suites could be the hung one given a different timing.

---

## 6. Hypotheses to evaluate against the next CI run

If Layer F captures the Phase 4 wedge:

| Signature | Suspected mechanism |
|---|---|
| `CTX-SENDSYNC-ENTRY` without `CTX-FORMAT-ENTRY` | Wedge inside the lookup or session-prepare prefix of `sendSync` (mutex contention, mRequestToSession lookup) |
| `CTX-FORMAT-ENTRY` without `CTX-FORMAT-EXIT` | Wedge inside the formatter — likely a per-peer send that blocks indefinitely or a buffer-slot acquire that never completes |
| `CTX-FORMAT-EXIT` without `CTX-PROMISE-SET` | Wedge between sendSync return and set_value — probably inside `release(id)` (futures-map eviction / lock contention) |
| `CTX-PROMISE-SET` without any further progress | Wedge on the consumer side of the promise (whoever called `future.get()` is stuck despite the promise being fulfilled — would be very odd) |
| `GEN-RECVSYNC-ENTRY` without `GEN-UNFORMAT-EXIT` | Wedge inside the per-block recv loop in the formatter unformat |
| `GEN-UNFORMAT-EXIT` without `GEN-PROMISE-SET` | Wedge between formatter return and gen-side promise fulfillment |

If **no** Phase 4 ENTRY fires for any reqId that completed Phase 1, the wedge is not in the disagg KV transfer code at all — it would be in test framework teardown (executor destructor, MPI finalization, gtest fixture cleanup).

---

## 7. Implications for the decomposition plan and the parallel verification branches

The flakiness finding sharpens the case for landing the always-on baseline as a separate, small change first, *without* the cancellation surface, and treating subsequent CI runs of that smaller change as the test for whether the wedge race depends on the cancellation surface.

If the always-on-only branch wedges on the same test, the race is in code that's *not* in the cancellation surface — and the focus shifts to the rank-symmetric collective entry, the per-rank dedup, or the post-ready-signal data-transfer code. If the always-on-only branch does *not* wedge on the same test, then by elimination the cancellation surface is implicated and Phase 4 instrumentation will pin the specific site.

The current parallel branches set up exactly this elimination:
- "C1 + A2 + A8" branch — buffer-pool RAII + agent lifetime + polling-cap; deliberately excludes A1, A3, A4, A5, A6, A7, A9, A10.
- "A4 + A6 + A7" branch — deadline observation only.
- "A1 + A2 + A4 + A7 + A8 (with A9/A10 reverted)" branch — the merge-intended baseline.

Each branch is a different ablation against the wedge. The first that fails CI on the same test, in the same way, identifies which always-on component participates in the race.

---

## 8. Next steps

1. Re-run CI on the diag branch with Layer F included. Wait for the test timeout to flush buffered stdout, then look at the Phase 4 markers for any reqId.
2. Based on which Phase 4 site has ENTRY-without-EXIT, narrow further. Likely-needed Layer G additions would be:
   - Per-peer breakdown inside `format` and `unformat` (which peer's send/recv is hung)
   - State of the response worker thread when the wedge is happening (mutex hold, CV wait, etc.)
3. Run the parallel verification branches through CI and compare which (if any) wedges.
4. Update document 13 once we have a name for the Phase-4 mechanism: doc 13 framed `asymmetric_executor[mpi_kvcache]` specifically as a transport-layer hang; the present finding suggests the same test can also wedge at the data-transfer or future-completion phase, depending on race timing.

---

## 9. Doc relationships

- Supersedes (partially): doc 12 §4 single-mechanism framing of the asymmetric_executor wedge.
- Supersedes (partially): doc 13 framing of the mpi_kvcache transport hang as the canonical wedge signature.
- Builds on: doc 15 (decomposition plan), doc 14 (cross-rank consistency).
- Inputs: two CI runs of the diag-instrumentation branch; the gating inventory memory note about byte-identical-tree-different-failure-profile across earlier builds.
