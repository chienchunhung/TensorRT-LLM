# 08 — Next Steps and PR Map

This file is the operational view of the investigation: outstanding
work, the chained PRs in flight, companion fixes, and the
deadline-enforcement effort estimate. For the *strategic* fix-path
recommendation, read
[`06-fix-approaches/README.md`](06-fix-approaches/README.md).

---

## Signature ↔ PR map

| Signature | Status | Test PR | Fix PR | Notes |
|---|---|---|---|---|
| **#1** Sender-side `Broken promise` after ready signal | Test merged; fix in review | [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) | [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) | Chained: `#13640` builds on `#13639`. |
| **#2** Trie `cascade prune` assertion | Test merged; fix in review | [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) | [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) | Chained: `#13572` builds on `#13571`. Independent of disagg networking. |
| **#3** Decode-side `RuntimeError: bad optional access` | Field-only; not yet localised | — | — | Python-side trace markers added; will localise on next field hit. |
| **#4** Gen-side blocking hang in `checkGenTransferStatus(atLeastNum=1)` | Test merged; fix in review | [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674) | [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671) | `#13671` carries both the test and the fix as 2 commits; both PRs target `main` so `#13674` lands first and `#13671`'s duplicate test commit becomes a no-op. |
| **#5** Receiver-side `Broken promise` from queued cancel | Combined test + fix in review | (combined into fix PR) | [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672) | Mirror of `#1` on the receiver side. |
| **#6** Recv-buffer index leak via `!isReady` early return | Combined test + fix in review (chained on `#13640`) | (combined into fix PR) | [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673) | Two-layer fix: RAII cleanup in `sendRequestInfo()` (Layer A) + explicit free in `requestSync()` `!isReady` path (Layer B). Direct cascade from the `#1` fix. |
| **#7** `pthread_mutex_lock` wedge in `CacheSender::Impl::*` (bug class with 4 manifestations) | Variant D fixed; mutex deadlock variant still needs `gdb` capture | — (unit test deferred until exact mutex / holder identified) | Variant D fix included in [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) (combo PR) | Re-classified in Phase 12 from "NIXL plugin bug" to "TRT-LLM-side `CacheSender::Impl` mutex bug, exposed across both NIXL and direct-UCX backends". Phase 13 broadened to a 4-manifestation class. Phase 14 confirmed and fixed Variant D (eval-order). |
| **L9** Transport quiescence on unsafe exit (no signature — defense-in-depth) | Folded into combo; MLA port follow-up done | — (focused unit tests being ported alongside the sig regression tests) | [#13728](https://github.com/NVIDIA/TensorRT-LLM/pull/13728) folded directly into [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) | Adds `BufferIndexHolder::poison()`, tri-state `ReadySignalResult{kReady,kNotReady,kCancelled}`, send-side `try/catch` + poison around `sendAllBuffers`, and Python `_fail_closed_for_unquiesced_disagg_transfer()`. The MLA send path was missed by `#13728` and ported to `mlaCacheFormatter.cpp` as part of the PR `#13713` review-fix cleanup (zero-copy disable + try/catch + `sendHolder.poison()`). |
| **Combo (Approach D)** | In review | (multiple) | [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) | PR `#13056` + PR `#13495` + eval-order fix + Python idempotency guards + PR `#13728` (folded in) + MLA port. The strongest current candidate; see [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md). |

### Companion fixes (already in `main`, not in `rc11`)

- [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) —
  request-level error propagation in disagg serving. Does **not** fix
  the wedge, but makes the failure mode visible (replaces generic
  `400 Bad Request` with the real error body, regenerates
  `disagg_request_id` on retry). Strongly recommended as a backport
  target for `rc11`.
- [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — fatal
  engine detection / pod restart. Does **not** fix the wedge, but
  ensures that if the engine actually crashes (which is **not** what
  happens here), the pod restarts. Useful as a backstop, not as a
  fix.

---

## Outstanding work

In rough priority order:

### 1. Pin down the mutex behind sig `#7` (Variant A — mutex deadlock)

This is the open root-cause investigation that's blocked the surgical
fix for the deadlock variant. Procedure:

1. Re-launch the rc11 + chained-fix-stack burst harness with NIXL
   backend; let it wedge into the deadlock variant rather than the
   SIGSEGV variant. (Phases 10–11 evidence shows the NIXL backend
   reaches the deadlock cleanly; direct-UCX with our chained fixes
   tends to crash with Variant C SIGSEGV before reaching the mutex
   deadlock.)
2. Attach `gdb` to the wedged ctx-side `mpi4py.futures.server` worker.
3. Locate the `dataTransResp` thread.
4. `frame 0` and `info registers rdi` to read the mutex address; then
   `x/8x $rdi` to dump the `pthread_mutex_t::__owner` field — that
   gives the holder TID.
5. Match the holder TID to a thread in `info threads`, walk its
   backtrace to find the holding code path.
6. Fix the lock-ordering or release-before-blocking-call issue in
   `CacheSender::Impl`.

Estimated effort: ~30–60 min for the diagnostic; ~1–2 days for fix
+ chained test PR. Source code suggests the mutex is `mCondMutex` at
the top of `response()`'s loop body
([`dataTransceiver.cpp:684`](../../../cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp#L684)),
but this needs runtime confirmation.

### 2. Land the combo (Approach D) for the customer, with the rc13 stop-gap

Land [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) on
`main`. The combo includes the eval-order fix (Variant D of sig `#7`)
and idempotency guards already, plus PR `#13056` + PR `#13495`'s
architectural mechanisms, plus PR `#13728`'s `L9` fail-closed
memory-safety policy folded in directly with an MLA-formatter port.
Recovery is clean on NIXL+UCX-plugin through `CONC=256` with three
ctx/gen pairs, the customer's transport.

**Important: the combo regresses on rc13 without the L10 stop-gap.**
rc13 turns on block reuse by default, which surfaces the L10 dual-path
defect (sig `#8`). The combo must therefore land *with* a small
stop-gap in `_end_transfer_and_maybe_terminate`:

1. Remove the `if not should_store_blocks:` guard. Always call
   `_terminate_request` after `end_transfer()` returns true.
2. Add a `resources_freed` flag on the request's transfer metadata.
   Set inside `_do_terminate_request` after `free_resources` runs,
   check on entry to dedupe.
3. Add an integration test driving disagg + block_reuse +
   slow-transfer so `_handle_responses` runs while the request is in
   transmission. Tag the test as "covers the dual-path that Phase 2
   will simplify" so the test author of the Phase 2 PR knows to update
   it after the deletion.

The stop-gap is ~10 lines plus an integration test. Comment every
new field with `# STOP-GAP: remove with Phase 2 pin-elimination work`
so future contributors don't solidify it as the long-term contract.

### 2a. Land Phase 2 of the block-reuse-overlap-scheduler design (medium-term)

Follow-up PR after `#13713` lands. Implements the deletion documented
in
[`docs/design/block-reuse-overlap-scheduler/phase2-unify-reuse-mechanisms.md`](../../design/block-reuse-overlap-scheduler/phase2-unify-reuse-mechanisms.md):

- Replace `store_blocks_for_reuse(request, pin=True)` with `pin=False`
  in the disagg path.
- Delete the `should_store_blocks` flag and the conditional in
  `_end_transfer_and_maybe_terminate`.
- Delete `block_id` from `RequestTransferMetadata`; simplify to a
  bare counter.
- Delete `unpin_blocks_by_id` in `end_transfer`.
- Drop the `pp_size == 1` restriction.
- Delete the stop-gap fields (`resources_freed`) and the dual-path
  integration-test annotations.

This closes layer **L10** outright. Strictly smaller diff than the
stop-gap once measured by net-lines (deletes more than it adds).
Risks are documented in the Phase 2 design doc and need explicit
audit on rc13: (a) the "sequence alive → ref count > 0" invariant
must hold under PR `#13728`'s fail-closed paths; (b) the scheduler's
free-block accounting must keep treating in-transfer blocks as
allocated under memory pressure; (c) PP > 1 enablement should be
verified end-to-end through the long-prompt burst harness, not just
asserted because the restriction is gone.

### 3. Lift the direct-UCX saturation boundary above `CONC=32`

The combo recovers cleanly on direct UCX through `CONC=32`,
`BURST_DUR_S=90` and wedges above that on this single-host rig
(`CONC=48 / 64 / 128`). The diagnostic build (see
[`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md#direct-ucx-saturation-evidence-diagnostic-build))
shows the wedge is throughput saturation plus queue backpressure: per-
buffer `tagSend`/`tagRecv` calls take 3-11 s for 1-3.7 GB buffers
(~300-400 MB/s effective), the deadline reaper cancels backlog, and
recovery probes time out while queues drain. Almost all TRT-LLM
cancels under saturation resolve via queue removal of work that never
started (3 of 139 ctx-side cancels actually flipped the in-flight
cancel flag at `CONC=128`).

Two follow-up scopes lift the boundary:

**3a. Cheap test: UCX rendezvous tuning + parallel send workers**

- `UCX_RNDV_SCHEME=put_zcopy` (or `get_zcopy`), raise `UCX_RNDV_THRESH`,
  ensure GPU NVLink/IPC transports are enabled.
- Increase `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM` (currently 1) so
  multiple async-send worker threads can overlap sends.
- Possibly raise `kv_transfer_timeout_ms` to absorb burst-time variance.

This will not change architecture; it can lift throughput enough that
the queue stops backing up. Cheap to test, no rebuild. Recommended as
the first step.

**3b. One-sided RDMA shape for direct UCX (matches NIXL)**

If 3a plateaus, the direct-UCX path needs to replace `tagSend` /
`tagRecv` rendezvous with a one-sided RDMA shape similar to NIXL's:

1. Pre-register receiver memory (memory descriptor exchange).
2. Use `ucp_put_nbx` for one-sided writes (already exposed by UCXX).
3. Deliver completion via a small notification message.
4. Add a `TransferStatus` wrapper around the UCXX request and implement
   `release()` as `ucxx::Request::cancel()` plus terminal-state
   draining.
5. Factor the NIXL polling / cancel policy into a shared helper used by
   both `AgentConnection::send()` and the new direct-UCX path.

The cancellation primitive is independently useful for sub-saturation
cancellation correctness; it is **not** by itself sufficient to lift
the saturation boundary.

The larger architectural option is to turn direct UCX into a full
`BaseTransferAgent` backend and route it through `AgentConnection`,
which would unify the two paths. That removes more duplication
long-term but is materially larger and riskier.

For the current PR cycle, this work is **deferred**. PR `#13713` ships
with direct-UCX recovered up to `CONC=32`, `BURST_DUR_S=90` on this
rig, and direct-UCX above its boundary is documented as a known
limitation, not a regression introduced by the combo.

### 4. Multi-node / Dynamo orchestration validation

All current results are single-host. Customer's deployment is K8s
cluster with Dynamo Operator. Combo + NIXL needs validation in that
shape before being declared production-ready.

### 4a. MLA-model stress validation

All current stress runs use Qwen3-0.6B (GQA-style attention), which
exercises `cacheFormatter.cpp` rather than `mlaCacheFormatter.cpp`.
The MLA `L9` port should be exercised under the customer-shape stress
harness with a DeepSeek-class model (V2 / V3 / R1) before the combo
is declared safe for MLA deployments. Cost is one rebuild + one
5-iteration loop using the existing `.repro/run_validation_loop.sh`
with `MODEL` overridden.

### 5. Backport `#13119` to `rc11`

Request-level error propagation in disagg serving. Doesn't fix the
wedge, but makes failure modes visible. Strongly recommended for any
field hit even after the combo lands.

### 6. Add cancel-during-transfer integration test

The single largest test-coverage gap surfaced by this investigation.
A CI lane that drives the disagg HTTP path with the long-prompt +
retries + cancels load shape would have caught all six TRT-LLM
signatures as test failures rather than as a customer field hit.
Cost is moderate (a few hundred lines + a CI lane); benefit is huge
(every future PR touching the disagg path is gated against this load
shape).

### 7. Audit other early-return paths in the C++ disagg transceiver

The `#6` pattern — "fix the visible failure path on side A, surface a
resource leak on side B" — is likely to repeat if other paths share
the same RAII gap. Specific candidates: other `concurrence` resources
in `BaseTransBufferManager`, request-side state in `dataTransceiver.cpp`
helpers, formatter exit paths.

This audit is **strongly motivated** by the MLA-formatter finding
during the PR `#13713` review cycle: PR `#13728` originally fixed
`cacheFormatter.cpp` (MHA / GQA send path) but missed the structurally
identical `mlaCacheFormatter.cpp::format()` send loop. The MLA path
had the same `BufferIndexHolder sendHolder` pattern with no try/catch
+ poison around `sendBufferFun`, leaving DeepSeek-style MLA models
exposed to the same memory-safety hazard the rest of the combo
closes. The fix was mechanical (port the same try/catch + poison
pattern + zero-copy disable guard), but the gap existed for an entire
PR cycle because no one auditing `#13728` cross-checked the MLA
formatter. A short audit pass over every other RAII-protected path in
the disagg transceiver would catch the next instance of this pattern
before it ships.

### 8. Document the seven invariants

A short architectural note in the disaggregated-serving developer
guide naming the seven contracts from
[`07-architectural-reflections.md`](07-architectural-reflections.md).
Cheap, high leverage, prevents the next field hit.

### 9. Long-term: introduce a `TransferSession`-like unifying abstraction

The right long-term direction. Bundles request lifetime + cancel
token + buffer holders + promise + timeout into one RAII-managed
object, with every send/receive path expressed as a method on it. PR
`#13495`'s `TransferSession` is one explicit step in this direction;
a clean redesign would consolidate it with `#13056`'s per-request
cancel-flag + `BufferIndexHolder`.

### 10. File a NIXL/UCX bug as a secondary issue

Phase 11's `gdb` evidence on the NIXL backend showed that
`nixlUcxThreadEngine::getNotifs()` was *also* parked on
`pthread_mutex_lock` deep in the NIXL plugin's own internal lock.
That may be a contributing factor to `#7` (cross-library lock-ordering
deadlock) or unrelated. Worth filing using the `pr13056_run1` ctx-worker
stack as the canonical reproducer. **No longer the top-priority
action** since Phase 12 reclassified `#7` as a TRT-LLM-side bug.

### 11. Rename the misleading `drop_without_fulfill` trace marker

The marker fires immediately *before* the sig `#1` cancellation
handler that already fulfills the promise correctly. Rename to
`cancelled_after_ready_handled` to remove the false-positive in
future forensic readings.

---

## Effort estimate for the deadline enforcement

This section sizes the deadline-enforcement work
(`kv_transfer_timeout_ms` consumption in the C++ blocking sites). After
Phase 12 reclassified `#7` as a TRT-LLM-side mutex bug, this work has
a dual role:

- It directly addresses `#7` once Layer B lands (every TRT-LLM-owned
  blocking primitive becomes interruptible).
- It remains valuable defence-in-depth even after the surgical fix
  lands in item 1 above, so the architectural invariant "every
  blocking wait must be interruptible" is enforced going forward.

The work decomposes into four implementation layers, each with a
different effort / blast-radius / coverage trade-off. Calendar
estimates assume one engineer familiar with the disagg transceiver.

### Layer A — Python-level deadline + structured cancel (~1 week)

**Where:** `tensorrt_llm/_torch/pyexecutor/py_executor.py` and
`kv_cache_transceiver.py` — the existing per-iteration loops that
already call `check_context_transfer_status()` /
`check_gen_transfer_status()`.

**How:**

1. Track `req.kv_transfer_started_at` when the request enters the
   transceiver path.
2. After each non-blocking poll cycle, scan in-flight requests; if
   `now - started_at > kv_transfer_timeout_ms`, call
   `transceiver.cancel_request(req)`, mark the request
   `kDISAGG_TRANS_ERROR`, fulfil the Python-side completion future
   with a structured timeout exception, and remove from the in-flight
   tracker.
3. Surface the timeout cleanly to the OpenAI-style HTTP layer as a
   structured 5xx (e.g. `kNETWORK_ERROR` body), so the orchestrator
   sees real signal.

**Pros:** ~50–100 lines of Python; no C++ rebuild; testable with the
existing `test_kv_cache_transceiver.py`; immediately surfaces the
silent wedge as a clean per-request error.

**Cons:** The C++ side already restricts `cancelRequest()` to
requests that are *not currently being processed*; the request
actually wedged inside `nixlAgent::getNotifs()` will return "Cannot
cancel". Python effectively "abandons and reports timeout" without
the C++ side cleaning up. The wedged C++ thread keeps consuming its
slot. Layer A buys a bounded number of clean errors but does *not*
buy sustained recovery; for that, the orchestrator must restart the
wedged pod.

**Verdict:** Right starting point. Cheap, low risk, unblocks
orchestrator-driven recovery, immediate operability win.

### Layer B — C++-side deadline on slice-able blocking paths (~2 weeks)

**Where:** Every `cv.wait(...)` and unbounded `future.get()` in
`dataTransceiver.cpp`, `cacheTransceiver.cpp`, and
`baseTransBuffer.cpp`. Four obvious candidates: the
`assignBufferIndex()` `cv.wait`, the `CacheSender::Impl::response()`
outer `mSenderCv.wait`, the inner ready-signal recv in
`CacheReceiver::Impl::sendRequestInfo()`, and the
`CacheTransceiver::checkGenTransferStatus()` future probe (already
fixed for `atLeastNum=1` via the sig `#4` patch — extend to a
deadline-aware variant).

**How:**

1. Add a `std::chrono::steady_clock::time_point deadline` parameter
   to relevant private methods.
2. Replace each `cv.wait(lk, predicate)` with a
   `cv.wait_for(lk, slice_ms, predicate)` loop that checks
   `mTerminate || past_deadline` on each slice.
3. On deadline expiry: set the request future with `kNETWORK_ERROR`,
   set state to `kDISAGG_TRANS_ERROR`, free any reserved buffer
   indices via the same RAII helper used for the sig `#6` fix,
   continue serving other requests.
4. Add unit tests for each deadline path.

**Pros:** Real per-request timeout behaviour across every TRT-LLM-owned
blocking primitive; cleans up properly on timeout; defends against
many slow-path hangs.

**Cons:** Does **not** cover the underlying NIXL `getNotifs` mutex
(that's a single C call into NIXL, not a `cv.wait` in TRT-LLM).
Larger surface for race conditions; needs careful review.

**Verdict:** Right follow-up to Layer A. Closes the architectural
"every blocking wait must be interruptible" invariant.

### Layer C — `std::async` watchdog around NIXL calls (~2 weeks, with caveats)

Wrap each blocking NIXL call with `std::async` + `future.wait_for(timeout)`.

**Pros:** Caller actually escapes; can serve other requests until
thread pool exhaustion.

**Cons:** Thread leak per timeout. Doesn't prevent further wedges;
subsequent NIXL calls hit the same internal mutex.

**Verdict:** Pursue only if Layers A + B aren't enough. Otherwise the
cost-to-benefit is poor.

### Layer D — In-process NIXL agent reset on timeout (~3–4 weeks, design-heavy)

A new "NIXL recovery" subsystem that detects sustained timeouts, tears
down the wedged NIXL agent, recreates it, and re-establishes
connections.

**Verdict:** Last resort. With Phase 12's reclassification (the mutex
is in TRT-LLM, not NIXL), Layer D is even less likely to be needed.

### Recommended order

1. **Surgical mutex fix for `#7`** (item 1 above, ~1–2 days).
2. **Layer A** (1 week).
3. **Layer B** (2 weeks).
4. **NIXL/UCX bug filing** as secondary (item 10 above).
5. Layers C and D should not be pursued unless 1–3 fall through.

---

## File / branch index

### Reproducer harness

Long-prompt burst + recovery-probe script in the
`local/rc11-disagg-repro` worktree under `.repro/harness/onepair/`.

### Run logs (not in git)

| Run | Configuration | Archive |
|---|---|---|
| `run4` | stock `rc11` | `~/trtllm-experiment-archives/run4_final_20260429_211126/` |
| `run5` | post-sig-`#4` fix | `~/trtllm-experiment-archives/run5_fixsig4_final_20260429_224332/` |
| `run6` | + sig `#5` fix + first sig `#6` instrumentation | `~/trtllm-experiment-archives/run6_recvfix_final_20260429_233258/` |
| `run7` | fine-grained sig `#6` instrumentation | `.repro/logs/run7_sig6_instr/` in `local/rc11-disagg-repro` |
| `run8` | post-sig-`#6` fix end-to-end | `~/disagg-investigation-archive/run8_sig6_fix/` (includes `pyspy/gen_worker_*_gdb.txt`, `pyspy/ctx_worker_*_gdb.txt`) |
| `pr13056_run1` | independent: PR `#13056` (NIXL) | `~/disagg-investigation-archive/pr13056_run1/` |
| `rc11_ucx_run1` | our fixes + direct UCX | `~/disagg-investigation-archive/rc11_ucx_run1/` (canonical `#7` mutex deadlock evidence with no NIXL plugin loaded) |
| `rc11_ucx_run2_diag` | our fixes + UCX with `gdb` capture | `~/disagg-investigation-archive/rc11_ucx_run2_diag/` |
| `run9` | rc11 + our fixes + UCX | `~/disagg-investigation-archive/run9_rc11_ourfixes_ucx_segfault/` (Variant C SIGSEGV) |
| `run10` | rc11 + PR `#13056` + UCX | `~/disagg-investigation-archive/run10_pr13056_ucx_segfault_handleAsyncSend/` (Variant D SIGSEGV) |
| `run14`, `run14c` | PR `#13056` + UCX with eval-order instrumentation / fix | local logs in PR `#13056` worktree |
| `run_pr13713_reviewfix_v3` | combo + PR `#13728` + MLA port + cleanup edits, NIXL 3-pair, `CONC=128`, 5 iterations | `.repro/logs/run_pr13713_reviewfix_v3_20260504_172546/` in `local/pr13056-pr13495-combo` worktree (5/5 PASS, zero failure markers across all 12 worker / front / client logs, all 7 ports return HTTP 200 post-run) |

### New unit tests

- `cpp/tests/unit_tests/runtime/radixBlockTreeTest.cpp` (sig `#2`).
- `tests/unittest/others/test_kv_cache_transceiver.py::test_cancel_request_in_transmission_fulfills_sender_future`
  (sig `#1`).
- `tests/unittest/others/test_kv_cache_transceiver.py::test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future`
  (sig `#4`).
- `tests/unittest/others/test_kv_cache_transceiver.py::test_cancel_queued_gen_request_fulfills_receiver_future`
  (sig `#5`).
- `tests/unittest/others/test_kv_cache_transceiver.py::test_cancelled_after_ready_does_not_leak_recv_buffer_index`
  (sig `#6`; uses NIXL backend).

### Trace gating envs

- `TRTLLM_DISAGG_TRACE_PROMISE` — sender / receiver promise lifecycle
  and `checkGenTransferStatus` selection / get markers.
- `TRTLLM_DISAGG_TRACE_TRIE` — trie attach / detach / cascade-prune
  markers.
- `TRTLLM_DISAGG_TRACE_OPTIONAL` — Python event-loop exception
  summaries around the optional accessors implicated in sig `#3`.
- `TRTLLM_DISAGG_TRACE_BLOCK` — Python watchdog around blocking
  transceiver calls; `TRTLLM_DISAGG_TRACE_BLOCK_TIMEOUT_S` controls
  the threshold (default 10 s, repro uses 5 s).

### Reusable wheels

- `/home/scratch.chienchunh_coreai/trtllm_wheels/rc11-sigfixes-1-4-5-6-instrumented-4e69c14f73-2026-05-01-nv26.02.whl`
  — rc11 + this investigation's chained PRs + instrumentation.
- `/home/scratch.chienchunh_coreai/trtllm_wheels/pr13056-c9777c4ac2-nv26.02.whl`
  — rc11 + PR `#13056` (independent comprehensive refactor variant).

Both built against NVIDIA PyTorch container 26.02 (torch 2.11.0a0).
