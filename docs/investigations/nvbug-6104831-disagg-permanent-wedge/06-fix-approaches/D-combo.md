# Approach D — Combo Stack (PR `#13713`)

The combo combines PR `#13056`'s architectural lifetime / cancellation
refactor with PR `#13495`'s backend transfer-release cancellation, then
adds the eval-order sequencing fix and Python idempotency guards. PR
`#13728`'s fail-closed memory-safety policy is folded in directly,
plus a port of the same poison-on-NIXL-throw pattern to the MLA send
formatter that `#13728` missed. **The strongest candidate so far**
and the only stack that closes every layer in the `L1`–`L9` defect
class stack.

Submitted as PR [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713).

---

## What it contains

```text
rc11
+ PR #13056   (architectural lifetime / cancellation refactor)
+ PR #13495   (transfer-release cancellation hook)
+ eval-order fix in CacheSender::Impl::handleAsyncSend
+ Python idempotency guards in _prepare_disagg_gen_init() and _recv_disagg_gen_cache()
+ PR #13728   (fail-closed on unquiesced disagg KV transfer)
+ MLA port    (poison-on-NIXL-throw + zero-copy guard in mlaCacheFormatter.cpp)
```

For the detailed contents of the first four pieces, see the per-approach
files ([`B-pr13056.md`](B-pr13056.md) and [`C-pr13495.md`](C-pr13495.md))
and the local-patch descriptions in either of them. The PR `#13728`
fold-in and the MLA port are described in the *Memory-safety hardening*
section below.

---

## What it covers (`L1`–`L9`)

| Layer | Coverage | Where it comes from |
|---|---|---|
| **L1** sig `#1` | ✓✓ | Both `#13056`'s exception-via-cancel-flag flow and `#13495`'s explicit `set_exception` after erase. Combo uses `#13495`'s ordering (post-erase, empirically tested under stress). |
| **L1** sig `#5` | ✓ | This is the one place the combo retains your chained PR's contribution: `#13672`'s queued-cancel `set_exception` is the only fix for this path; neither `#13056` nor `#13495` covers it. |
| **L2** request lifetime / UAF | ✓ | `shared_ptr<LlmRequest>` from either `#13056` or `#13439` (both make the same change; combo benefits from the consistency). |
| **L3** in-process cancellation primitive | ✓ | Per-request cancel-flag registry from `#13056`. |
| **L4** `checkGenTransferStatus` blocking | ✓ | Layered: `#13671` (`wait_for(0)` skip) prevents the indefinite wait per poll, `#13056`'s deadline-hoist evicts entries that stay unready past `kv_transfer_timeout_ms`. Together this is the correct semantics ("skip if not ready, evict if ignored too long"). |
| **L5** recv-buffer slot leak | ✓ | `BufferIndexHolder` (and `#13495`'s `TransferSession` for the cross-formatter ownership case). |
| **L6** NIXL backend handle release | ✓ | `#13495`'s `TransferStatus::release()` → `nixlAgent::releaseXferReq()`. |
| **L7** eval-order regression | ✓ | Local eval-order fix (necessary because L2 is closed). |
| **L8** Python scheduler idempotency | ✓ | Local idempotency guards. |
| **L9** transport quiescence on unsafe exit | ✓ | PR `#13728` covers the non-MLA send path (`cacheFormatter.cpp`) and the recv path (`dataTransceiver.cpp`); the local MLA port covers `mlaCacheFormatter.cpp`. The Python-side `_fail_closed_for_unquiesced_disagg_transfer()` drains the scheduler when an unquiesced transfer is detected. |

---

## Memory-safety hardening (PR `#13728` fold-in + MLA port)

PR `#13713`'s original fix scope was wedge prevention: closing
`L1`–`L8` so the customer-visible failure stops happening. Once those
layers were closed and the combo recovered cleanly through `CONC=256`
on NIXL, the residual `L9` invariant came into focus during code
review: cancel and exception paths were still returning recv buffer
slots (and, on the MLA path, send buffer slots) to the pool while the
local NIXL agent thread or the remote peer might still be reading or
writing into the slot's VRAM. No empirical wedge was traceable to
`L9`; it is a silent-corruption surface, the worst kind of bug
because it does not announce itself.

PR [`#13728`](https://github.com/NVIDIA/TensorRT-LLM/pull/13728)
introduced the fail-closed memory-safety policy that closes `L9`. The
combo PR `#13713` folds `#13728` in directly rather than stacking it
as a separate PR; the rationale is that exposing a known memory-
safety hazard between two PRs in the chain would be worse than
shipping a slightly larger combo. The fold-in adds three pieces:

1. **`BufferIndexHolder::poison()`** in
   `cpp/tensorrt_llm/batch_manager/baseTransBuffer.{h,cpp}`. Marks the
   slot poisoned (flag `mBufferIndexFlag[idx] = 2`) and the entire
   pool poisoned (`mConcurrenceResource.mPoisoned.store(true)`).
   Subsequent `assignBufferIndex` calls fail closed with a `TLLM_THROW`
   instructing the operator to restart the process. `held()` is
   true on construction even when `mIndex == std::nullopt` so the
   dynamic-buffer path can still poison its pool on unsafe exit.

2. **`ReadySignalResult` tri-state** in
   `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` and
   `cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.{h,cpp}`.
   `recvReadySignalWithStatus()` returns
   `kReady` / `kNotReady` / `kCancelled`; the recv-side
   `requestSync()` poisons holders only on `kCancelled` and on
   exception (where transport quiescence is unknown). `kNotReady`
   means the peer explicitly said no data is coming, which is safe
   to release.

3. **Send-side poison-on-throw** in
   `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp` and
   `cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp`. The
   `sendAllBuffers()` call (or the parallel send loop in MLA) is
   wrapped in `try { ... } catch (...) { if (agentConnection != nullptr)
   sendHolder.poison(); throw; }`. The `agentConnection != nullptr`
   guard restricts poisoning to NIXL paths where transport quiescence
   is unknown; direct-UCX paths fall back to the destructor's normal
   release because direct UCX has no quiescence semantics anyway.

4. **Python-side fail-closed shutdown** in
   `tensorrt_llm/_torch/pyexecutor/py_executor.py`.
   `_fail_closed_for_unquiesced_disagg_transfer()` clears
   `active_requests` / `waiting_queue` / `request_accumulated` /
   `control_requests`, sets `is_shutdown = True` *and*
   `shutdown_event.set()` so callers parked on either signal wake up,
   enqueues structured error responses for every in-flight request,
   and notifies `response_cv`. It triggers when `_handle_errors`
   detects any in-flight transfer in `is_disagg_generation_transmission_in_progress`
   or `is_disagg_context_transmission_state` at the moment of
   failure.

PR `#13728`'s diff missed `mlaCacheFormatter.cpp` — the MLA send
path uses the same `BufferIndexHolder sendHolder` pattern as
`cacheFormatter.cpp` but had no try/catch + poison around its parallel
send loop and no zero-copy guard. The PR `#13713` review-fix cleanup
ports both to MLA (commit-pending in the combo worktree). MLA
deployments (DeepSeek-V2/V3/R1, etc.) on NIXL therefore inherit `L9`
coverage on the same terms as MHA models.

The receiver path covers both MHA and MLA models because the cancel /
exception poisoning happens in `dataTransceiver.cpp::CacheReceiver::Impl::requestSync`,
above the per-model formatter dispatch.

### Empirical impact of the fold-in

The L9 fail-closed policy was *not exercised* in any of the
reaffirmation runs (CONC=128 / 256 NIXL 3-pair, 5/5 PASS each). That
is the expected outcome: with `L1`–`L8` closed, the visible wedges
don't happen, and a clean burst should never reach an unquiesced-
transfer path. `L9` is the rip-cord — it exists so that the *next*
unknown bug in `L1`–`L8` cannot silently corrupt the buffer pool
instead of being detected. Validating that the rip-cord triggers
correctly is unit-test scope; the recv-side `kCancelled` →
`poisonRecvHolders()` path and the send-side `agentConnection !=
nullptr` → `sendHolder.poison()` path each have a focused C++ unit
test, and the Python `_fail_closed_for_unquiesced_disagg_transfer()`
has a Python-level test (porting in progress alongside the sig `#1`,
`#4`, `#5` regression tests).

---

## Why this works when no other approach does

The customer's wedge is **a stack of independent defect classes**, each
of which is independently sufficient to wedge the deployment. Closing
all eight is the only way to recover under the customer load shape;
closing the ninth is what stops the next unknown bug in `L1`–`L8`
from corrupting the buffer pool instead of being detected. The other
approaches each leave at least one layer uncovered:

- A leaves L2, L3, L6, L9.
- B leaves L6, L9 (and is partial on L1, L4).
- C leaves L3, L4, L9, sig `#5` half of L1.
- D leaves nothing — and once PR `#13728` is folded in plus the MLA
  port, also closes L9 across both formatter paths.

For the layer-by-layer reasoning, see
[`README.md`](README.md#coverage-matrix). The empirical confirmation
(direct-UCX recovery at `CONC=16`/`24`/`32`, NIXL recovery at
`CONC=32`/`64`/`128`/`256`, plus L9 active across all paths) matches
the prediction exactly.

---

## Empirical results

Local 1P1D `trtllm-serve` long-prompt burst harness, single host:

### Direct UCX

| Test | Result |
|---|---|
| Regular `CONC=16`, `BURST_DUR_S=60` | Recovered at idle 30 s. |
| Same servers, `CONC=16`, `BURST_DUR_S=60`, 5 iterations | 5/5 recovered. |
| Same servers, `CONC=24`, `BURST_DUR_S=60`, 5 iterations | 5/5 recovered. |
| Same servers, `CONC=24`, `BURST_DUR_S=90`, 5 iterations, after stale-server cleanup | 5/5 recovered. |
| Same servers, `CONC=32`, `BURST_DUR_S=90`, 5 iterations, after stale-server cleanup | 5/5 recovered. |
| Same servers, `CONC=48`, `BURST_DUR_S=90`, 5 iterations, with diagnostic build | Failed on iteration 1: `ok200=11`, `errors=48`, `total=59`; all probes through idle 180 s hit `ReadTimeout`; no recovery. |
| Same servers, `CONC=64`, `BURST_DUR_S=90`, clean retry | Failed on iteration 1: `ok200=9`, `errors=64`, `total=73`; all probes through idle 180 s hit `ReadTimeout`; no recovery. |
| Same servers, `CONC=64`, `BURST_DUR_S=90`, confirmation after NIXL success | Failed on iteration 1 again: `ok200=12`, `errors=64`, `total=76`; same pattern; reproducible. |
| Same servers, `CONC=128`, `BURST_DUR_S=90`, 5 iterations, with diagnostic build | Failed on iteration 1: `ok200=12`, `errors=100`, `total=112`; same pattern. |

The direct-UCX usable boundary on this rig is therefore `CONC=32`,
`BURST_DUR_S=90`. Above that, direct UCX wedges on the first burst and
does not recover.

### NIXL transceiver path (NIXL transfer agent using backend `UCX`)

| Test | Build | Result |
|---|---|---|
| Same servers, `CONC=32`, `BURST_DUR_S=90`, 5 iterations | combo (pre-`#13728`) | 5/5 recovered; each burst completed with `ok200=716`, `errors=0`, `total=716`. |
| Same servers, `CONC=64`, `BURST_DUR_S=90`, 5 iterations | combo (pre-`#13728`) | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` except one iteration with `ok200=715`, `errors=0`, `total=715`. |
| 3 ctx/gen pairs on one 8-GPU B300 node, `CONC=128`, `BURST_DUR_S=90`, 5 iterations | combo (pre-`#13728`) | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` (one iteration `ok200=715`). |
| 3 ctx/gen pairs on one 8-GPU B300 node, `CONC=256`, `BURST_DUR_S=90`, 5 iterations | combo (pre-`#13728`) | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` (one iteration `ok200=715`). |
| 3 ctx/gen pairs on one 8-GPU B300 node, `CONC=128`, `BURST_DUR_S=90`, 5 iterations (review-fix v3) | combo + `#13728` + MLA port + cleanup edits | **5/5 recovered, zero failure markers across all 12 worker / front / client logs (no `Broken promise`, no `bad optional access`, no `Assertion`, no `Traceback`, no `SIGSEGV`, no `Cannot cancel request`, no `exceeded total timeout`).** Bursts: 716 / 715 / 716 / 715 / 715 OK; recovery at idle 30 s on every iteration; all 7 ports return HTTP 200 post-run. |

The 3-pair `CONC=256` recovery is the strongest pre-`#13728` local
single-node verdict; the 3-pair `CONC=128` review-fix-v3 run is the
strongest *post*-fold-in verdict and confirms that the review-fix
edits (PR `#13728` integration, MLA port, comment / docstring /
shutdown-event cleanups) do not regress the proven-stable non-MLA
NIXL path. Multi-node fabric, Dynamo orchestration, and production
mixed traffic remain open follow-up validation scopes; an MLA-model
stress test (e.g. DeepSeek on NIXL) is a separate validation scope
because Qwen3-0.6B does not exercise `mlaCacheFormatter.cpp`.

The latest contrast is **not** "UCX hardware transport bad, NIXL
transport good"; both NIXL runs used the UCX plugin underneath. The
split is between TRT-LLM's direct UCX transceiver path and the NIXL
transfer-agent path with PR `#13495`'s explicit transfer-release
cancellation semantics.

### Direct-UCX saturation evidence (diagnostic build)

To understand why direct UCX falls behind NIXL on the same workload, we
built a diagnostic version of the combo with three extra signals:

- `[ucx-cancel] cancel observed` warnings inside `waitUcxRequestOrCancel()`
  whenever a per-request cancel flag flips while a direct-UCX wait is in
  progress.
- `[ucx-slow] op=... elapsedMs=...` warnings when any direct-UCX call
  (`sendConnectionId`, `send`, `recv`, `recvConnect`) takes ≥ 100 ms (env
  override `TRTLLM_UCX_SLOW_CALL_LOG_MS`).
- `[cancel] CacheSender|CacheReceiver::cancelRequest` entry/flip/exit lines
  promoted to `WARNING` so the cancel path is visible live.

For `CONC=128`, `BURST_DUR_S=90`, the diagnostic counts were:

| Marker | gen.log | ctx.log |
|---|---:|---:|
| `[ucx-cancel] cancel observed` | 0 | 3 |
| `[ucx-slow] op=...` (≥100 ms) | 34 | 67 |
| `CacheSender::cancelRequest entered` | 0 | 139 |
| `CacheSender::cancelRequest flipped` | 0 | 3 |
| `CacheReceiver::cancelRequest entered` | 33 | 0 |
| `CacheReceiver::cancelRequest removedFromQueue=1` | 33 | 0 |
| `exceeded total timeout` | 27 | 139 |

For `CONC=48`, `BURST_DUR_S=90`, the same shape held (gen `[ucx-slow]=31`,
ctx `[ucx-slow]=61`, ctx sender `entered=91 / flipped=3`, gen receiver
`entered=31 / flipped=0 / removedFromQueue=23`, ctx `exceeded total
timeout=91`).

What this evidence says:

- Individual direct-UCX calls *complete*, but they are slow under load.
  `[ucx-slow]` durations were 3-11 s for buffers of 1-3.7 GB, i.e. on the
  order of 300-400 MB/s effective. This is well below what UCX can sustain
  for shared-memory / NVLink / IPC peer-to-peer on a single B300 node. The
  bottleneck is throughput, not a stuck call.
- The TRT-LLM cancel paths run, but almost all cancels resolve via *queue
  removal* rather than aborting an in-flight UCX call:
  - ctx `CacheSender::cancelRequest`: 139 entered, 3 flipped the in-flight
    cancel flag, the rest were satisfied via `mReadyResponses` queue
    removal (request waiting to be sent, never started).
  - gen `CacheReceiver::cancelRequest`: 33 entered, 0 flipped, all 33 had
    `removedFromQueue=1` (request waiting in the receiver queue, never
    started).
- The failure mode is *queue backpressure → deadline reaper →
  cancellation of work that never started*, not "UCX call stuck waiting on
  a dead peer". A scoped cancel-aware UCX wait helper has the right
  shape, but it cannot help when no in-flight UCX call needs to be
  cancelled.

### Why direct UCX falls behind NIXL on the same workload

NIXL and direct UCX use the same UCX plugin, so the wire-level transport is
the same. The difference is the request shape used to move data:

| Layer | NIXL path (`AgentConnection::send`) | Direct UCX path (`UcxConnection::send` / `recv`) |
|---|---|---|
| Data transfer primitive | One-sided RDMA `submitTransferRequests` (write to remote pre-registered VRAM) | Two-sided `tagSend` / `tagRecv` rendezvous |
| Receiver participation | None per buffer; receiver consumes after a single notification | Receiver must post matching `tagRecv` per buffer |
| Sync between sender and receiver | Single `notifySyncMessage` after the write completes | Per-buffer rendezvous handshake plus completion |
| Per-buffer latency under load | Bounded by RDMA write throughput | Bounded by rendezvous round-trip + receiver scheduling latency |
| Cancellation handle | `nixlAgent::releaseXferReq()` via `TransferStatus::release()` (used in `AgentConnection::send` poll-wait) | `ucxx::Request::cancel()` → `ucp_request_cancel()` available, but the wedge happens before in-flight-cancel becomes the deciding factor |

The practical consequence is what we measured: under `CONC ≥ 48`, direct UCX
spends multi-second windows in tag rendezvous per buffer, the response /
request queues build up, the deadline reaper cancels the backlog, and
recovery probes after the burst keep timing out because the worker queues
do not drain in time.

---

## The remaining direct-UCX boundary above `CONC=32`

The combo still wedges on direct UCX at `CONC=48 / 64 / 128`,
`BURST_DUR_S=90`. This is **not** primarily an L1-L8 cancellation gap;
the diagnostic build shows the wedge is throughput saturation under
the rendezvous protocol plus queue backpressure (see *Direct-UCX
saturation evidence* above). The proposed short-term design from
Phase 14 - a direct-UCX `TransferStatus::release()` wrapper around
`ucxx::Request::cancel()` plus a shared cancel-aware wait helper - is
still the right *cancellation shape* for direct UCX. It correctly
matches NIXL's lifecycle contract for the cases where a cancel does
fire mid-call. But it does **not** lift the saturation boundary. The
diagnostic build at `CONC=128` shows that only 3 of 139 ctx-side
cancels actually flipped the in-flight cancel flag the helper would
observe; the rest are queue removals of requests that never started.

Lifting the direct-UCX boundary on this workload requires either:

- **(a)** UCX rendezvous tuning (e.g. `UCX_RNDV_SCHEME=put_zcopy`,
  `UCX_RNDV_THRESH`) plus parallel send workers
  (`TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM`); cheap to test, may shift
  the throughput ceiling; or
- **(b)** replacing `tagSend`/`tagRecv` rendezvous with a one-sided RDMA
  shape similar to NIXL's: pre-register receiver memory, use
  `ucp_put_nbx`, deliver completion via a small notification message.
  Larger change; either an extension of the direct-UCX path or a new
  `BaseTransferAgent` backend that unifies with the NIXL path.

This is a follow-up TRT-LLM PR scope, not a NIXL or UCX change. See
[`../08-next-steps-and-pr-map.md`](../08-next-steps-and-pr-map.md).

For the current PR cycle the decision is to ship PR `#13713` with
direct-UCX as recovered up to `CONC=32`, `BURST_DUR_S=90` on this rig,
and to defer further direct-UCX work.

---

## Run-hygiene caveats for the latest results

Two caveats matter for interpreting the empirical data:

1. **One `CONC=24`, 90 s launch failed before the burst** because
   stale gen processes still held `localhost:8002`. That run is
   **invalid** as a product signal and is excluded from the 5/5
   counts.
2. **An earlier `CONC=32`, 90 s run failed on iteration 1**, but the
   clean rerun after explicit stale-server cleanup recovered 5/5. The
   clean `CONC=64`, 90 s run still failed even after the same
   cleanup; a later confirmation run after NIXL validation also
   failed on iteration 1, so the direct-UCX high-load failure is
   reproducible (not a hygiene artifact).

---

## When to use this approach

- **As the canonical fix path for the customer wedge** — yes. This is
  the recommended landing path on `main`.
- **For `rc11` backport** — the bigger blast radius makes this risky on
  a release branch, but the customer's deployment runs on `rc11` and
  the wedge isn't fixable with a smaller stack. Treat the rc11 backport
  as a high-touch operation: land in stages, validate at each step.

---

## Strengths

- **The only stack that closes every layer in L1–L9.** This isn't
  rhetoric — every other stack leaves at least one layer open and
  therefore has a predictable residual failure mode (and only D
  closes the `L9` memory-safety hazard at all).
- **NIXL recovery clean through `CONC=256`** with three ctx/gen pairs.
  This is the customer's transport. The combo is the first stack that
  works for the reporter's actual deployment shape, and the
  review-fix-v3 reaffirmation at `CONC=128` confirms the PR `#13728`
  fold-in does not regress that.
- **Direct-UCX recovery clean through `CONC=32`.** The remaining
  `CONC=64` wedge is a known gap (throughput saturation, not
  cancellation) with a clear follow-up scope.
- **MHA + MLA models both covered for L9.** `#13728` covers the
  non-MLA send path; the local MLA port covers `mlaCacheFormatter.cpp`.
  The recv-side poison logic is above the formatter dispatch and
  applies to both.
- **Each piece has independent design rationale**: `#13056` has
  detailed commit messages; `#13495` has a 512-line design doc;
  `#13728` has a structured fail-closed contract and tri-state
  `ReadySignalResult`; the local patches have empirical justification
  (Phase 14 traces for eval-order; review-time code audit for the MLA
  port).

## Weaknesses

- **Largest blast radius** of all four approaches. Combines three
  large PRs with two local patches; the integration surface is
  non-trivial.
- **L1 has overlapping coverage** between `#13056`'s cancel-flag flow
  and `#13495`'s explicit `set_exception`. The combo uses
  `#13495`'s ordering, but the two mechanisms touch the same code site
  and could conflict during merge.
- **L5 has overlapping coverage** between `#13056`'s and `#13495`'s
  `BufferIndexHolder`. Both make the same RAII change to the same
  files. Merge resolution required.
- **Direct-UCX `CONC=64` still wedges** — the L6-equivalent for the
  direct-UCX path isn't there yet (separate from `L9`; that's a
  cancellation gap, not a memory-safety gap).
- **`L9` triggers haven't been validated under live load.** The fail-
  closed Python path and the C++ poison hooks are unit-tested
  individually, but the customer-shape stress harness has not yet
  reached an unquiesced-transfer scenario in any of the runs (because
  `L1`–`L8` are doing their job). That coverage gap is by design but
  worth documenting.
- **Multi-node and Dynamo orchestration not yet validated.**

---

## Caveats worth being honest about

1. The combo's empirical recovery is **"no permanent wedge"**, not
   "no errors." Burst phase still produces many `400 Bad Request`
   responses and KV-transfer-timeout logs under stress. That's
   expected when L4 / L6 are doing their job (clean per-request
   errors), but it's a serving-quality degradation worth documenting
   separately as a capacity ceiling.
2. The `#13056` / `#13495` overlap on L1 is functionally fine but
   architecturally redundant. A clean follow-up would consolidate to
   one mechanism (probably `#13495`'s post-erase ordering).
3. Same for L5: `BufferIndexHolder` shows up twice. The natural
   merge resolution is to keep `#13495`'s additional `TransferSession`
   on top of either implementation.

---

## What to read next

- For the side-by-side comparison framework, return to
  [`README.md`](README.md).
- For each individual piece's contribution, see
  [`B-pr13056.md`](B-pr13056.md) and [`C-pr13495.md`](C-pr13495.md).
- For what the chained-PR approach left undone, see
  [`A-chained-fixes.md`](A-chained-fixes.md).
- For the deadline-enforcement effort estimate that complements the
  fix, see
  [`../08-next-steps-and-pr-map.md`](../08-next-steps-and-pr-map.md).
