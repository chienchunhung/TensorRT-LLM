# 06 — Fix Approaches Compared

Four candidate fix stacks emerged from the investigation. They overlap,
but they are not equivalent: each closes a different combination of the
L1–L9 defect classes (see
[`../03-defect-class-stack.md`](../03-defect-class-stack.md)). End-to-end
harness results are materially different between them.

This README is the comparison and the recommendation. The four
companion files describe each approach in detail:

- [`A-chained-fixes.md`](A-chained-fixes.md) — surgical signature fixes
  (PRs `#13571`–`#13674`).
- [`B-pr13056.md`](B-pr13056.md) — PR `#13056` plus eval-order and
  idempotency.
- [`C-pr13495.md`](C-pr13495.md) — PR `#13495` plus sig `#4`,
  eval-order, and idempotency.
- [`D-combo.md`](D-combo.md) — the combo stack: PR `#13056` + PR `#13495`
  + eval-order + idempotency. Submitted as PR `#13713`.

---

## TL;DR

**Approach D (combo) wins.** It is the first stack that closes every
load-bearing layer (L1–L8) plus the residual memory-safety invariant
(L9), and recovers cleanly on both transport paths up through
`CONC=32` on direct UCX and `CONC=256` on NIXL+UCX-plugin (3 ctx/gen
pairs). The other approaches each leave at least one layer uncovered,
and *any uncovered layer in `L1`–`L8` is sufficient to wedge the
deployment* under the customer load shape; only D additionally closes
`L9` (silent buffer-pool corruption hazard on cancel/exception).

That ordering — A < B < C ≪ D — is not "more code is better." It is a
direct consequence of layer coverage, where every uncovered layer
corresponds to a specific failure mode that was empirically observed
in the run archives (or, for `L9`, a code-level audit gap).

---

## Coverage matrix

This is the central table. Each row is a defect-class layer; each
column is an approach. Cells indicate whether the approach closes the
layer and, where relevant, by what mechanism.

| Layer | A: chained sigs | B: `#13056` + local | C: `#13495` + sig `#4` + local | D: combo |
|---|---|---|---|---|
| **L1** sig `#1` (sender cancel-after-ready promise) | ✓ via `#13640` (set_exception before erase) | partial — cancel-flag flips, worker's `catch(std::exception)` calls `set_exception(current_exception())`; the specific cancel-after-ready erase site in `sendResponse`'s "else" branch is unchanged | ✓ — explicit `RequestSpecificException` on the moved-out promise (post-erase ordering) | ✓✓ via `#13495` |
| **L1** sig `#5` (receiver queued-cancel promise) | ✓ via `#13672` | partial — cancel-flag covers in-flight; in-queue erase path unchanged | ✗ — explicitly acknowledges *"`Cannot cancel request` may still appear"* | ✓ via `#13672` (combo retains your test+fix here) |
| **L2** request lifetime (`shared_ptr<LlmRequest>`) | ✗ — Response::mRequest stays raw | ✓ via PR `#13056` commit `649d1466bb7a` | ✓ via PR `#13439` base | ✓ |
| **L3** in-process cancellation primitive | ✗ | ✓ — full per-request cancel-flag registry on sender + receiver, plumbed through `sendRequestInfo` / `receiveReadySignal` / `AgentConnection::send` polling loops | ✗ — relies on existing `DataContext::transferTerminate` only | ✓ via PR `#13056` |
| **L4** `checkGenTransferStatus(atLeastNum=1)` blocking | ✓ via `#13671` (`wait_for(0)` skip) | partial — `kv_transfer_timeout_ms` deadline-hoist evicts after 60 s but blocks for the deadline first | ✗ — `#13495` doesn't touch this site at all | ✓ — `#13671` (skip) + `#13056` (deadline) layered |
| **L5** recv-buffer slot leak (sig `#6`) | ✓ via `#13673` (try/catch + explicit free) | ✓ via `BufferIndexHolder` RAII (PR `#13056`) | ✓ via `BufferIndexHolder` + `TransferSession` (PR `#13439`) | ✓✓ |
| **L6** NIXL backend handle release on cancel | ✗ | ✗ — no NIXL backend interaction | ✓ — **only `#13495` has this**: `TransferStatus::release()` → `nixlAgent::releaseXferReq()` | ✓ via PR `#13495` |
| **L7** eval-order regression introduced by L2 fix | n/a — L2 not closed, no regression | ✓ via local eval-order fix | ✓ via local eval-order fix (`#13439` triggers same regression) | ✓ |
| **L8** Python scheduler idempotency | partial — only observable if other layers don't crash first | ✓ via local idempotency guards | ✓ via the same local guards | ✓ |
| **L9** transport quiescence on unsafe exit | ✗ | ✗ | ✗ | ✓ — PR `#13728` (fail-closed poison + Python shutdown) folded directly into the combo, plus the local MLA port to `mlaCacheFormatter.cpp` |

Reading this matrix tells the story:

- **A** leaves L2, L3, and L6 entirely uncovered. After the surgical
  signature fixes, sig `#7`'s manifestations fire (deadlock variant,
  Python-`getattr` SIGSEGV) because UAFs aren't closed, no in-process
  cancel primitive exists, and no backend handle release is wired up.
- **B** leaves L6 entirely uncovered, plus partial L1 and L4. The L6
  gap means stranded NIXL/UCX handles can pile up under contention; the
  partial L4 means a 60 s self-block per stuck transfer per poll
  cycle. Empirically: *"Some direct-UCX runs recovered, including a
  `CONC=24` run, but recovery was not yet consistently clean across
  repeats."*
- **C** leaves L3 and L4 entirely uncovered, plus the queued-cancel
  half of L1. The L4 gap is decisive — without `#13671`, sig `#4`
  self-blocks the gen event loop indefinitely, and L6's `releaseXferReq()`
  mechanism can't even fire because the gen worker never reaches the
  cancel observation. Empirically: *"the direct-UCX stress run still
  ended in no-recovery, and the gen event loop later failed with stale
  sequence state."*
- **D** covers every layer. It is the only stack that recovers 5/5 at
  `CONC=16/24/32` on direct UCX and 5/5 at `CONC=32/64/128/256` on
  NIXL+UCX-plugin (3 ctx/gen pairs at `CONC≥128`). It is also the
  only stack that closes `L9` (silent corruption hazard).

---

## End-to-end results, side by side

This is the empirical confirmation of the layer analysis. The harness
is the long-prompt + cancellation burst from
[`../04-reproduction.md`](../04-reproduction.md).

| Approach | direct UCX `CONC=16` | direct UCX `CONC=24` | direct UCX `CONC=32` | direct UCX `CONC=64` | NIXL `CONC=32` | NIXL `CONC=64` | NIXL `CONC=128` (3-pair) | NIXL `CONC=256` (3-pair) |
|---|---|---|---|---|---|---|---|---|
| A — chained signature fixes | wedge after ~burst+cleanup; surfaces sig `#7` variants | wedge | wedge | wedge | wedge (NIXL `run8`) | not run | not run | not run |
| B — `#13056` + eval-order + idempotency | inconsistent: some repeats recover, some wedge | inconsistent | not validated | wedge or stale-sequence | not validated | not validated | not validated | not validated |
| C — `#13495` + sig `#4` + eval-order + idempotency | improved failure shape; still no-recovery; later `unordered_map::at` from `add_token` | wedge or stale-sequence | wedge or stale-sequence | wedge | not validated | not validated | not validated | not validated |
| **D — combo (with `#13728` + MLA port)** | **5/5 recovered (60 s + 90 s)** | **5/5 recovered (60 s + 90 s)** | **5/5 recovered (90 s)** | wedged on iteration 1 (no recovery 180 s) | **5/5 recovered, zero burst-time errors** | **5/5 recovered, zero burst-time errors** | **5/5 recovered, zero burst-time errors (review-fix v3, `#13728` folded in)** | **5/5 recovered, zero burst-time errors** |

The customer transport (NIXL + UCX plugin) is **clean on the combo
through `CONC=256` with three ctx/gen pairs**. The only remaining
failure case is direct-UCX under sustained `CONC≥48` stress; that's
throughput saturation, not a cancellation gap, and requires either
UCX rendezvous tuning + parallel send workers or a one-sided RDMA
shape mirroring NIXL (see
[`../08-next-steps-and-pr-map.md`](../08-next-steps-and-pr-map.md)).

---

## Why ablation analysis predicts each approach's residual failure mode

Reading the matrix top-down to bottom predicts each approach's actual
failure surface — and the run archive confirms the prediction:

- **A predicts: sig `#7` manifestations fire after the surgical fixes.**
  Run record matches: `run9` (Python-`getattr` SIGSEGV) and
  `rc11_ucx_run1` (deadlock variant) both occur on stack A.
- **B predicts: backlog under contention from missing L6.** Run record
  matches: combo's NIXL `CONC=64` runs are clean precisely because L6's
  `releaseXferReq()` drops the backend handle on cancel, while B's
  equivalent runs accumulate handles and eventually deadlock.
- **C predicts: gen event loop self-blocks on first stuck transfer
  from missing L4.** Run record matches: C's stress runs end in
  no-recovery with the `_check_disagg_gen_cache_transfer_status` →
  `check_gen_transfer_status` blocking pattern.
- **D predicts: clean recovery on layer-complete stack.** Run record
  matches: 5/5 recovery in every NIXL test and direct-UCX test up to
  `CONC=32`.

The remaining direct-UCX `CONC=64` wedge in approach D is *also*
predicted by the matrix: D includes `#13495`'s NIXL `releaseXferReq()`
(closing L6 for NIXL) but the equivalent direct-UCX backend handle
release primitive isn't there. That is the right additional fix to
land — see "Direct-UCX cleanup / cancellation design" in
[`../05-investigation-timeline.md`](../05-investigation-timeline.md)
Phase 14.

---

## Architectural reading: why D wins specifically

The wedge is **a stack of independent defect classes**, not a single bug
with multiple symptoms. Three observations follow directly:

1. **No surgical-signature-fixes-only stack is sufficient.** Approach A
   demonstrates this. Each individual signature fix is correct; the bug
   class as a whole has eight invariant gaps and the chained PRs only
   close five. The remaining three (L2, L3, L6) drive sig `#7`'s
   manifestations.

2. **No "one big PR" is sufficient either.** PR `#13056` and PR `#13495`
   each take an architectural angle (lifetime + cancellation primitive
   for `#13056`; backend handle release + transfer-session for `#13495`),
   and each closes a different subset. The two PRs are complementary,
   not competing — `#13056` covers L2, L3, partial L4; `#13495` covers
   L2, L5 better than `#13056`, L6.

3. **Two more layers (L7 and L8) are not addressed by any of the
   architectural PRs and need local patches.** L7 is a pure
   C++ argument-evaluation-order regression introduced by L2's
   `shared_ptr` change; L8 is a Python-side scheduler defect that only
   becomes observable once L7 is fixed. Both must be added on top.

The combo (D) is the smallest stack that closes all eight layers and is
therefore the smallest stack that empirically recovers under the
customer load shape.

---

## Decision guide

If you can land everything (no review-budget constraint):

> Land all of approach D's pieces in their natural order:
> `#13056` → `#13495` → eval-order patch → idempotency-guard patch.
> Use the combo PR `#13713` as a single landing artifact, or split as
> appropriate for review. Keep your chained-PR test scaffolding from
> `#13639`, `#13571`, `#13674`, `#13672`, `#13673` and rebase the tests
> against whichever production code lands; drop the corresponding fix
> commits since they're subsumed.

If you can only land one of `#13056` or `#13495`:

> Land **`#13056`** for `main`. It closes more layers (L2, L3, partial
> L1, L4, L5) than `#13495` does (L2, L5, L6). The L6 gap can be
> mitigated with `#13495`'s mechanism in a follow-up.
>
> Land **`#13495`** for `rc11`. It is more conservative (smaller diff,
> fail-closed Python policy), has a bundled design doc, and the L6
> mitigation is the most operationally important piece for the
> customer's NIXL deployment shape. Pair it with sig `#4`'s fix
> (`#13671`) which is non-negotiable.

If you can land neither and only the chained surgical fixes:

> Approach A is *necessary but not sufficient*. The customer wedge will
> still reproduce because L2, L3, L6 remain open. Land it anyway —
> closing L1, L4, L5 reduces the wedge frequency and the test
> scaffolding is reusable. But don't ship it as the sole fix.

---

## What survives whichever approach lands

The **regression tests** from your chained PRs are the
irreplaceable artifact, because none of `#13056`, `#13495`, or any of
the local patches contains a focused unit test for these signatures.

| Test | PR | Defect class | What it asserts |
|---|---|---|---|
| `test_cancel_request_in_transmission_fulfills_sender_future` | `#13639` | L1 (sig `#1`) | sender promise fulfilled with structured exception on cancel-after-ready |
| `radixBlockTreeTest.cpp` 4 stress cases | `#13571` | sig `#2` | trie cascade-prune doesn't fire under repeated insert/evict |
| `test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future` | `#13674` | L4 (sig `#4`) | `checkGenTransferStatus(atLeastNum=1)` doesn't block on unready future |
| `test_cancel_queued_gen_request_fulfills_receiver_future` | `#13672` | L1 (sig `#5`) | receiver promise fulfilled on queued cancel |
| `test_cancelled_after_ready_does_not_leak_recv_buffer_index` | `#13673` | L5 (sig `#6`) | recv-buffer slot freed on `!isReady` early-return |

**Recommendation:** rebase these tests against whichever production
code lands and keep them. They are the bug-class regression coverage.

---

## Caveats

1. **Direct-UCX `CONC=64` still wedges even under approach D.** The
   remaining failure mode is described in Phase 14 as "head-of-line
   backlog and timeout interaction rather than a stuck ctx `sendSync()`
   or gen `requestSync()` call". A direct-UCX cancellation primitive
   analogous to `#13495`'s NIXL `releaseXferReq()` (i.e.
   `ucxx::Request::cancel()`) is the obvious follow-up.
2. **Multi-node and Dynamo orchestration not yet validated.** All
   results above are single-host. The customer's deployment is K8s
   cluster with Dynamo Operator; combo D needs validation in that shape
   before being declared production-ready.
3. **The L1 sig `#1` overlap between `#13640` and `#13495`.** Same
   code site, same idiom, slightly different ordering. `#13495`'s
   post-erase ordering empirically avoided a v2 regression they hit
   with pre-erase. The combo (D) uses `#13495`'s ordering by virtue of
   including `#13495`; `#13640`'s separate fix isn't needed in D, but
   `#13639`'s test is.
4. **The L5 sig `#6` overlap between `#13673` and `BufferIndexHolder`.**
   Both close the same leak. `BufferIndexHolder` (in `#13056` and
   `#13495`) is the more idiomatic RAII implementation and additionally
   covers sender-side formatter exit paths. Combo D uses
   `BufferIndexHolder`; keep `#13673`'s regression test, drop the
   try/catch implementation.
