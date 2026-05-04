# Approach D — Combo Stack (PR `#13713`)

The combo combines PR `#13056`'s architectural lifetime / cancellation
refactor with PR `#13495`'s backend transfer-release cancellation, then
adds the eval-order sequencing fix and Python idempotency guards. **The
strongest candidate so far** and the only stack that closes every layer
in the `L1`–`L8` defect class stack.

Submitted as PR [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713).

---

## What it contains

```text
rc11
+ PR #13056   (architectural lifetime / cancellation refactor)
+ PR #13495   (transfer-release cancellation hook)
+ eval-order fix in CacheSender::Impl::handleAsyncSend
+ Python idempotency guards in _prepare_disagg_gen_init() and _recv_disagg_gen_cache()
```

For the detailed contents of each piece, see the per-approach files
([`B-pr13056.md`](B-pr13056.md) and [`C-pr13495.md`](C-pr13495.md)) and
the local-patch descriptions in either of them.

---

## What it covers (`L1`–`L8`)

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

---

## Why this works when no other approach does

The customer's wedge is **a stack of independent defect classes**, each
of which is independently sufficient to wedge the deployment. Closing
all eight is the only way to recover under the customer load shape.
The other approaches each leave at least one layer uncovered:

- A leaves L2, L3, L6.
- B leaves L6 (and is partial on L1, L4).
- C leaves L3, L4 sig `#5` half of L1.
- D leaves nothing.

For the layer-by-layer reasoning, see
[`README.md`](README.md#coverage-matrix). The empirical confirmation
(direct-UCX recovery at `CONC=16`/`24`/`32` and NIXL recovery at
`CONC=32`/`64`) matches the prediction exactly.

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

| Test | Result |
|---|---|
| Same servers, `CONC=32`, `BURST_DUR_S=90`, 5 iterations | 5/5 recovered; each burst completed with `ok200=716`, `errors=0`, `total=716`. |
| Same servers, `CONC=64`, `BURST_DUR_S=90`, 5 iterations | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` except one iteration with `ok200=715`, `errors=0`, `total=715`. |
| 3 ctx/gen pairs on one 8-GPU B300 node, `CONC=128`, `BURST_DUR_S=90`, 5 iterations | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` (one iteration `ok200=715`). |
| 3 ctx/gen pairs on one 8-GPU B300 node, `CONC=256`, `BURST_DUR_S=90`, 5 iterations | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` (one iteration `ok200=715`). |

The 3-pair `CONC=256` recovery is the strongest local single-node
verdict for the combo+NIXL stack. Multi-node fabric, Dynamo
orchestration, and production mixed traffic remain open follow-up
validation scopes.

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

- **The only stack that closes every layer in L1–L8.** This isn't
  rhetoric — every other stack leaves at least one layer open and
  therefore has a predictable residual failure mode.
- **NIXL recovery clean through `CONC=64`.** This is the customer's
  transport. The combo is the first stack that works for the
  reporter's actual deployment shape.
- **Direct-UCX recovery clean through `CONC=32`.** The remaining
  `CONC=64` wedge is a known gap with a clear follow-up scope.
- **Each piece has independent design rationale**: `#13056` has
  detailed commit messages; `#13495` has a 512-line design doc; the
  local patches have empirical justification (Phase 14 traces).

## Weaknesses

- **Largest blast radius** of all four approaches. Combines two large
  PRs with two local patches; the integration surface is non-trivial.
- **L1 has overlapping coverage** between `#13056`'s cancel-flag flow
  and `#13495`'s explicit `set_exception`. The combo uses
  `#13495`'s ordering, but the two mechanisms touch the same code site
  and could conflict during merge.
- **L5 has overlapping coverage** between `#13056`'s and `#13495`'s
  `BufferIndexHolder`. Both make the same RAII change to the same
  files. Merge resolution required.
- **Direct-UCX `CONC=64` still wedges** — the L6-equivalent for the
  direct-UCX path isn't there yet.
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
