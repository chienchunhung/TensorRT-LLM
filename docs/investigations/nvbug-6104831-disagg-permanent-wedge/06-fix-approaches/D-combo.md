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
| Same servers, `CONC=64`, `BURST_DUR_S=90`, clean retry | Failed on iteration 1: `ok200=9`, `errors=64`, `total=73`; all probes through idle 180 s hit `ReadTimeout`; no recovery. |
| Same servers, `CONC=64`, `BURST_DUR_S=90`, confirmation after NIXL success | Failed on iteration 1 again: `ok200=12`, `errors=64`, `total=76`; same pattern; reproducible. |

### NIXL transceiver path (NIXL transfer agent using backend `UCX`)

| Test | Result |
|---|---|
| Same servers, `CONC=32`, `BURST_DUR_S=90`, 5 iterations | 5/5 recovered; each burst completed with `ok200=716`, `errors=0`, `total=716`. |
| Same servers, `CONC=64`, `BURST_DUR_S=90`, 5 iterations | 5/5 recovered; bursts completed with `ok200=716`, `errors=0`, `total=716` except one iteration with `ok200=715`, `errors=0`, `total=715`. |
| 3 ctx/gen pairs on one 8-GPU B300 node, `CONC=128`, `BURST_DUR_S=90`, 5 iterations | Running as the current local stress verdict candidate. |

The latest contrast is **not** "UCX hardware transport bad, NIXL
transport good"; both NIXL runs used the UCX plugin underneath. The
split is between TRT-LLM's direct UCX transceiver path and the NIXL
transfer-agent path with PR `#13495`'s explicit transfer-release
cancellation semantics.

---

## The remaining direct-UCX `CONC=64` wedge

The combo still wedges on direct UCX at `CONC=64`. This is consistent
with the L1–L8 framing: `#13495`'s `TransferStatus::release()` is a
NIXL-side primitive. The direct-UCX path doesn't have an equivalent
primitive yet, even though the underlying UCX library exposes one
(`ucp_request_cancel()` / `ucxx::Request::cancel()`). The proposed
short-term design (from Phase 14):

1. Add a direct-UCX `TransferStatus` wrapper around `ucxx::Request`.
2. Implement `wait(timeout_ms)` using `isCompleted()` /
   callback-future polling, returning `kIN_PROGRESS` on timeout.
3. Implement `release()` by calling `ucxx::Request::cancel()`, then
   continue progressing / waiting until the UCXX request reaches a
   terminal state. Only after that point can TRT-LLM safely unwind
   and release or reuse send / receive buffers.
4. Factor the NIXL polling policy into a shared helper used by both
   `AgentConnection::send()` and direct `UcxConnection::{send,recv}`:
   submit transfer, bounded wait, observe
   `DataContext::getTransferTerminate()`, call `release()` on cancel.

This is a follow-up TRT-LLM PR scope, not a NIXL or UCX change. See
[`../08-next-steps-and-pr-map.md`](../08-next-steps-and-pr-map.md).

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
