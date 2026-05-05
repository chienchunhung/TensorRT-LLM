# 00 — TL;DR (10-minute read)

Skim this first. If you want depth on any one part, every section ends
with a pointer to the file that has the full story.

---

## What is broken

A TensorRT-LLM disaggregated `trtllm-serve` deployment running rc11
**permanently wedges** after a single burst of long-prompt traffic with
client-side cancellations. The pod stays `1/1 Running`, the HTTP
server keeps accepting connections, but every request after the burst
times out indefinitely.

From a customer's perspective it looks like one bug. From the C++ KV
cache transceiver it is **a stack of eight independent wedge defects
plus one residual memory-safety gap** in the request cleanup path. Any
one of `L1`–`L8` is sufficient to wedge the deployment; `L9` is the
defense-in-depth invariant that closes the silent-corruption surface
left exposed once L1–L8 prevent the visible wedges.

The minimal trigger is:

- long prompts (~8K tokens),
- high concurrency (CONC ≥ 16),
- aggressive client-side timeouts that cause cancellations and retries,
- overlap scheduling enabled (the default).

That combination drives every request through the cleanup paths in
volume. The cleanup paths are where the bugs live.

> Full reproducer: [`04-reproduction.md`](04-reproduction.md).

---

## Where the bugs live: the disagg KV transfer flow

Disaggregated serving splits one request across two workers — a
**context worker** that runs prefill and publishes the KV cache, and a
**generation worker** that receives that KV cache and decodes tokens.
KV transfer happens worker-to-worker over NIXL/UCX, not through the
front-end router.

```mermaid
graph LR
    Client[HTTP client] --> Router["trtllm-serve disaggregated<br/>front-end"]

    subgraph Ctx["Context worker (prefill)"]
        CtxPy[PyExecutor] --> CtxCpp["CacheSender::Impl<br/>(C++)"]
        CtxCpp --> CtxNixl[NIXL/UCX]
    end

    subgraph Gen["Generation worker (decode)"]
        GenPy[PyExecutor] --> GenCpp["CacheReceiver::Impl<br/>(C++)"]
        GenCpp --> GenNixl[NIXL/UCX]
    end

    Router --> CtxPy
    Router --> GenPy
    CtxNixl <-.->|"KV transfer<br/>worker-to-worker"| GenNixl
```

The wedge surface is **inside the cancellation flow**: when a request
in flight gets cancelled (client disconnects, KV transfer times out,
retries fire), at least one of seven distinct things can go wrong in
the C++ cleanup path. Two of the seven were known from the field at
T0. Five more emerged from investigation, each exposed by fixing the
previous one.

| Sig | Where it lives | One-line symptom |
|---|---|---|
| **#1** | `CacheSender::Impl::sendResponse` (cancel-after-ready erase path) | `Broken promise` raised by the consumer's `future.get()` on the sender side |
| **#2** | `templatedTrie.h::clearNode` / `kvCacheManager.cpp` cascade-prune walk | `cascade prune: parent did not find this node as a child` C++ assertion under sustained eviction |
| **#3** | C++ `std::optional::value()` in the disagg gen path, surfaced via pybind | `RuntimeError: bad optional access` raised in the decode-side Python event loop *(field-only)* |
| **#4** | `CacheTransceiver::checkGenTransferStatus(atLeastNum=1)` unconditional `future.get()` | gen worker's main event loop blocks indefinitely on a not-yet-ready future |
| **#5** | `CacheReceiver::Impl::cancelRequest` (queued-cancel erase) | `Broken promise` raised by the consumer's `future.get()` on the receiver side |
| **#6** | `CacheReceiver::Impl::requestSync` `!isReady` early-return + `BaseTransBufferManager::assignBufferIndex` `cv.wait` | one cancelled-after-ready transfer leaks a recv-buffer slot; the next request wedges the receiver pool forever |
| **#7** | `CacheSender::Impl::*` (bug class with 4 manifestations) | mutex deadlock in `response()`; ctx mpi4py worker exits; Python `getattr` SIGSEGV; first-request SIGSEGV in `handleAsyncSend` |

Sig `#2` is independent of the cleanup-path bug class — it's an
eviction-driven trie invariant violation that just happens to fire
under the same high-concurrency load.

> Full per-signature root causes, fixes, regression tests, and code
> sites: [`02-failure-signatures.md`](02-failure-signatures.md).

---

## Root cause: nine invariant gaps

The seven signatures are the visible faces of **nine underlying
invariant gaps** that the rc11 transceiver doesn't enforce. We call
them L1 through L9. L1–L8 are wedge-prevention invariants — each is
independently sufficient to wedge the deployment under the customer
load shape. L9 is the residual memory-safety invariant that, even
with L1–L8 closed, prevents silent buffer-pool corruption when a
cancel or exception races an in-flight NIXL/UCX transfer.

| Layer | What's missing | Customer-visible failure |
|---|---|---|
| **L1** | Promises must be fulfilled before destruction | `Broken promise` thrown by consumer's `future.get()` |
| **L2** | Request lifetime must outlive C++ async workers | UAF on `LlmRequest*` after Python `_terminate_request` |
| **L3** | `cancelRequest` must interrupt in-flight workers | `Cannot cancel request` log accumulates; transfers pile up |
| **L4** | `future.get()` must be non-blocking when polled | Gen event loop self-blocks on first stuck transfer |
| **L5** | Acquired buffer slots must release on every exit | One leaked recv-buffer slot wedges all subsequent receives |
| **L6** | NIXL/UCX backend must release transfer handles on cancel | Stranded transfer handles → contention → mutex deadlock |
| **L7** | C++ argument evaluation order must be safe with `shared_ptr` | First-request SIGSEGV in `handleAsyncSend` after L2 fix |
| **L8** | Python scheduler must be idempotent by `py_request_id` | `KVCacheManager::addSequence` `emplaceDone` assertion |
| **L9** | Buffer slots must not return to the pool while transport quiescence is unknown | (silent) recv buffer reused mid-transfer → KV-cache corruption / sporadic decode garbage |

L1–L6 are C++ defects in the transceiver. L7 is a *regression*
introduced by L2's fix — closing the UAF requires `shared_ptr<LlmRequest>`,
which makes a previously-safe argument-evaluation pattern undefined
behaviour. L8 is a Python-side scheduler defect that only becomes
observable once L7 is removed and the system progresses far enough to
exercise the repeated init scheduling path. L9 is a defense-in-depth
memory-safety invariant: it never *fires* in a clean run, but on
cancel/exception paths where TRT-LLM cannot prove the transport has
quiesced it forces a fail-closed shutdown rather than handing pool
VRAM back to the next request.

> Code-level evidence for each layer plus the cascade map:
> [`03-defect-class-stack.md`](03-defect-class-stack.md).

---

## The fix

The fix is a single bundled stack — submitted as PR
[#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) — that
closes all nine layers. It composes six pieces:

```mermaid
graph TB
    rc11["rc11 baseline<br/>(broken)"]

    pr13056["PR #13056<br/>shared_ptr&lt;LlmRequest&gt; lifetime<br/>+ per-request cancel-flag<br/>+ kv_transfer_timeout_ms enforcement<br/>+ BufferIndexHolder RAII"]
    pr13495["PR #13495<br/>TransferStatus::release()<br/>→ nixlAgent::releaseXferReq()<br/>+ TransferSession ownership"]
    evalfix["Eval-order fix<br/>materialize reqId before std::move(resp)<br/>in handleAsyncSend"]
    pyguards["Python idempotency guards<br/>_disagg_gen_init_prepared_ids<br/>_disagg_gen_kv_recv_started_ids"]
    pr13728["PR #13728<br/>BufferIndexHolder::poison()<br/>+ ReadySignalResult tri-state<br/>+ _fail_closed_for_unquiesced_disagg_transfer"]
    mlaport["MLA port<br/>poison-on-NIXL-throw + zero-copy guard<br/>in mlaCacheFormatter.cpp"]

    rc11 --> pr13056 --> pr13495 --> evalfix --> pyguards --> pr13728 --> mlaport --> done["Combo PR #13713<br/>(fixed)"]

    pr13056 -.->|closes| L2[L2 lifetime]
    pr13056 -.->|closes| L3[L3 cancel primitive]
    pr13056 -.->|closes| L4[L4 blocking ✓ partial]
    pr13056 -.->|closes| L5[L5 buffer RAII]
    pr13495 -.->|closes| L1[L1 promise fulfillment]
    pr13495 -.->|closes| L6[L6 NIXL handle release]
    evalfix -.->|closes| L7[L7 eval-order]
    pyguards -.->|closes| L8[L8 idempotency]
    pr13728 -.->|closes for non-MLA| L9[L9 transport quiescence]
    mlaport -.->|closes for MLA| L9

    classDef done fill:#cfc,stroke:#0a0,stroke-width:2px
    classDef broken fill:#fcc,stroke:#a00,stroke-width:2px
    classDef pr13056_color fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef pr13495_color fill:#ffe5cc,stroke:#cc6600,stroke-width:2px
    classDef evalfix_color fill:#d4f4d4,stroke:#0a8a0a,stroke-width:2px
    classDef pyguards_color fill:#e8d4f4,stroke:#7030a0,stroke-width:2px
    classDef pr13728_color fill:#fce8d4,stroke:#a04020,stroke-width:2px
    classDef mlaport_color fill:#fce8d4,stroke:#a04020,stroke-width:2px
    classDef pr13056_layer fill:#e8f1ff,stroke:#0066cc,stroke-dasharray:3 3
    classDef pr13495_layer fill:#fff1e0,stroke:#cc6600,stroke-dasharray:3 3
    classDef evalfix_layer fill:#e8faea,stroke:#0a8a0a,stroke-dasharray:3 3
    classDef pyguards_layer fill:#f4e8fa,stroke:#7030a0,stroke-dasharray:3 3
    classDef pr13728_layer fill:#fdf0e0,stroke:#a04020,stroke-dasharray:3 3

    class done done
    class rc11 broken
    class pr13056 pr13056_color
    class pr13495 pr13495_color
    class evalfix evalfix_color
    class pyguards pyguards_color
    class pr13728 pr13728_color
    class mlaport mlaport_color
    class L2,L3,L4,L5 pr13056_layer
    class L1,L6 pr13495_layer
    class L7 evalfix_layer
    class L8 pyguards_layer
    class L9 pr13728_layer
```

The six fix components are color-coded. Each layer node (`L1`–`L9`) is
tinted with the lighter shade of whichever fix closes it, so the
`closes` arrows are reinforced by colour: blue for `#13056`, orange for
`#13495`, green for the eval-order fix, purple for the Python
idempotency guards, and brown for `#13728` + the MLA port (which
together close `L9` across both formatter paths).

Each layer needs at least one piece to close it; some pieces close
multiple layers. The minimum closure is exactly these six pieces.
Removing any of the first four re-opens at least one wedge layer
(empirically sufficient to wedge the deployment); removing `#13728`
and / or the MLA port re-opens `L9` (silent corruption hazard, no
visible wedge).

The contributions break down as:

- **PR #13056** (Yifan's architectural refactor) closes L2, L3, partial
  L4, L5. Largest single behavioural change: introduces
  `std::shared_ptr<LlmRequest>` end-to-end through the C++ transceiver
  and a per-request cancel-flag registry.
- **PR #13495** (codex-assisted; stacked on PR #13439) closes L6 (the
  unique contribution: `TransferStatus::release()` →
  `nixlAgent::releaseXferReq()`) plus reinforces L1 with the
  cancel-after-ready `set_exception` ordering.
- **Eval-order fix** closes L7. One-line patch in
  `CacheSender::Impl::handleAsyncSend`:
  ```cpp
  TLLM_CHECK(resp.mRequest != nullptr);
  auto const reqId = resp.mRequest->mRequestId;
  sendAndRemoveResponse(reqId, std::move(resp));
  ```
  Required because PR #13056's `shared_ptr` change makes the previous
  one-liner undefined behaviour under the C++ argument-evaluation rules.
- **Python idempotency guards** close L8. ~30 lines in
  `tensorrt_llm/_torch/pyexecutor/py_executor.py` that prevent
  `_prepare_disagg_gen_init` and `_recv_disagg_gen_cache` from running
  side effects twice on the same `py_request_id` when the scheduler
  re-presents an in-progress request.
- **PR #13728** closes L9 on the non-MLA send path and the recv path.
  Adds `BufferIndexHolder::poison()` (marks the slot poisoned, refuses
  to reuse without a process restart), a tri-state
  `ReadySignalResult{kReady, kNotReady, kCancelled}` so the recv side
  can distinguish "peer explicitly said not-ready" (safe to release)
  from "wait was cancelled / no signal" (transport quiescence
  unknown — must poison), and Python-side
  `_fail_closed_for_unquiesced_disagg_transfer()` to drain the
  scheduler and notify waiters when an unquiesced transfer is detected.
- **MLA port** closes L9 on the MLA send path. `mlaCacheFormatter.cpp`
  was missed by `#13728` and got the same try/catch + `sendHolder.poison()`
  pattern plus the zero-copy disable guard during the PR `#13713`
  review-fix cleanup.

The combo also retains your chained regression tests as the
load-bearing test scaffolding — neither PR #13056 nor PR #13495 has
focused unit tests for these signatures, and the L1 / L4 / L5 sig
tests now ship with the combo PR directly.

> The full per-piece breakdown and the L1-L9 → fix mapping:
> [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md).

---

## Does it work?

Yes, on the customer's transport. The combo recovers cleanly under the
local 1P1D `trtllm-serve` long-prompt burst harness on a single host
(plus a 3 ctx/gen pair stress shape on one 8-GPU B300 node). The
**bold** entries below are post-PR-`#13728`-fold-in reaffirmations on
the review-fix v3 build (current `c3e8659670` + the cleanup commits);
the rest pre-date the fold-in but ran on the same combo without `L9`
hardening:

| Transport | `CONC=16` | `CONC=24` | `CONC=32` | `CONC=48` | `CONC=64` | `CONC=128` (3-pair) | `CONC=256` (3-pair) |
|---|---|---|---|---|---|---|---|
| **NIXL + UCX plugin** *(customer transport)* | n/a | n/a | 5/5 recovered, zero burst errors | n/a | 5/5 recovered, zero burst errors | **5/5 recovered, zero burst errors (review-fix v3)** | **5/5 recovered, zero burst errors** |
| Direct UCX | 5/5 recovered | 5/5 recovered | 5/5 recovered | wedged on iter 1 | wedged on iter 1 | wedged on iter 1 | n/a |

The customer-reported failure shape is **fixed on NIXL+UCX-plugin
through at least `CONC=256` with 3 ctx/gen pairs, with the L9 fail-
closed policy active**. NIXL is the customer's transport, so the
reporter's deployment shape is covered by the combo. Direct UCX
recovers cleanly through `CONC=32` and saturates above that. The
diagnostic build shows the direct-UCX failure at high load is
throughput saturation plus queue backpressure (not a stuck UCX call):
individual `tagSend`/`tagRecv` calls take 3-11 s for 1-3.7 GB buffers
(~300-400 MB/s effective), the response/request queues build up, the
deadline reaper cancels backlog, and recovery probes after the burst
keep timing out. Most TRT-LLM cancels resolve via queue removal of
work that never started rather than aborting an in-flight UCX call.
Lifting that boundary needs either UCX rendezvous tuning + parallel
send workers, or a one-sided RDMA shape mirroring NIXL — both deferred
(see [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md)).

The L9 fail-closed policy was *not exercised* by any of these runs —
no log line shows a poison or fail-closed shutdown firing. That is
expected: L1–L8 close the visible wedges, and a clean burst should
never reach an unquiesced-transfer path. L9 is the rip-cord for the
case the rest of the stack misses, and validating that it triggers
correctly is a unit-test rather than stress-test scope.

A few honest caveats:

- **Multi-node and Dynamo orchestration are not yet validated.** All
  results above are single-host. The customer runs on K8s with the
  Dynamo Operator; the combo needs validation in that shape before
  being declared production-ready.
- **The combo's empirical recovery is "no permanent wedge", not "no
  errors."** The burst phase still produces `400 Bad Request`
  responses and KV-transfer-timeout logs under stress. Those are
  expected when L4 / L6 are doing their job (clean per-request errors
  rather than silent stalls), but they are a serving-quality
  degradation worth tracking separately as a capacity ceiling.
- **The sig `#7` mutex deadlock variant's exact mutex address** still
  needs a runtime `gdb` register capture to confirm it is `mCondMutex`
  (the source-code inference from Phase 12). The combo prevents the
  variant from manifesting under our test load, but a surgical
  in-`CacheSender::Impl` lock-ordering fix is still the right
  defence-in-depth follow-up.

> Per-test reproducer details, marker accounting, and run archives:
> [`04-reproduction.md`](04-reproduction.md) and
> [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md).

---

## Why this took eight days to find

The eight layers weren't visible at T0. Three were known from the
field (sigs `#1`, `#2`, `#3`); the other four signatures (and their
underlying L4–L7 layers) **emerged from investigation itself**, each
exposed by fixing the previous one. The cascade pattern is:

```mermaid
graph LR
    SIG1["Sig #1 fixed"] -->|"creates"| SIG6["Sig #6 (recv-buffer leak<br/>via the new !isReady path)"]
    SIG4["Sig #4 fix"] -->|"exposes"| SIG5["Sig #5 (was masked by<br/>self-blocking event loop)"]
    SIG4 -->|"exposes"| SIG6
    SIG6 -->|"exposes"| SIG7["Sig #7 (was masked by<br/>buffer-pool wedge)"]
    L2FIX["L2 fix (shared_ptr)"] -->|"creates"| L7["L7 (eval-order UB)"]
    L7FIX["L7 fix"] -->|"exposes"| L8["L8 (Python<br/>idempotency)"]
```

In retrospect, the meta-process bottleneck was **observability**, not
fix complexity (every fix is small — `#1` is ~5 lines, `#4` is ~17
lines, the eval-order fix is ~3 lines). What ate calendar time was
figuring out which request was wedged, why, and which fix to write
next. The retrospective in
[`07-architectural-reflections.md`](07-architectural-reflections.md)
argues that adding `kv_transfer_timeout_ms` enforcement as PR #0
(converting silent wedges into per-request errors) plus a
cancel-during-transfer integration test would have collapsed Phase 5
through Phase 10 from ~4 days to ~2 days.

This pattern also explains why no single fix bundle on its own was
sufficient. Approaches that close some layers but not others still
wedge under the customer load shape, in the layer they leave open. The
combo is the smallest stack that closes every layer, and the
emergence pattern means the smallest stack was not knowable at T0.

> The full chronological story:
> [`05-investigation-timeline.md`](05-investigation-timeline.md).

---

## What is left to do

In rough priority order:

1. **Land PR #13713 (combo, with `#13728` folded in and MLA hardened)**
   on `main`. The strongest current candidate.
2. **Pin down sig `#7` Variant A's mutex address** with a live `gdb`
   capture under NIXL backend. Estimated 30–60 min of diagnostic
   work, then ~1–2 days for a surgical lock-ordering fix.
3. **Add a direct-UCX `TransferStatus::release()` analog** using
   `ucxx::Request::cancel()` to close the residual `CONC=64`
   direct-UCX wedge.
4. **Lift the direct-UCX saturation boundary above `CONC=32`**. UCX
   rendezvous tuning + parallel send workers first; one-sided RDMA
   shape mirroring NIXL as a follow-up.
5. **Multi-node and Dynamo orchestration validation.**
6. **Backport [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119)**
   (request-level error propagation) to `rc11` for clean failure
   attribution on any future field hit.
7. **Add a cancel-during-transfer integration test** to CI. The single
   biggest reason this bug class went undetected until production.
8. **Document the seven invariants** in the disaggregated-serving
   developer guide so the next contributor adding a transfer mode
   doesn't reintroduce this bug class.

> The full PR map, deadline-enforcement effort estimate, and run
> archive index: [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md).

---

## Where to go from here

If you have 30 more minutes after this, the highest-leverage reads are:

- [`01-background.md`](01-background.md) — architecture diagrams and a
  full happy-path / cancellation walkthrough. Required reading if you
  haven't worked in this code path before.
- [`03-defect-class-stack.md`](03-defect-class-stack.md) — the L1–L8
  framework with code-level evidence and the cascade map. Required
  reading if you're reviewing PR #13713.
- [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md) — the
  detailed per-layer breakdown of what each piece of the combo
  contributes.

If you're investigating a *future* field hit on this code path:

- [`02-failure-signatures.md`](02-failure-signatures.md) — match the
  observed log lines / stack traces to a known signature.
- [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) — see
  whether the failing build includes the relevant fix from the combo.
- The trace-marker glossary at the end of
  [`01-background.md`](01-background.md) — read run logs against the
  known marker patterns.

If you're building intuition about the broader architectural debt:

- [`07-architectural-reflections.md`](07-architectural-reflections.md) —
  the seven invariants the transceiver doesn't enforce, why review
  didn't catch any of this, and what we would do differently.

If you want the *story* of how this was investigated:

- [`05-investigation-timeline.md`](05-investigation-timeline.md) —
  Phases 0 through 14, eight days of "find bug → fix bug → find next
  bug" in chronological order.
