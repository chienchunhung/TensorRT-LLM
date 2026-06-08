# Doc 19 — Exp 4 forensic: the wedge decomposes into three independent failures (F1 / F2 / F3)

| | |
|---|---|
| **Source experiment** | `experiments/exp4-14979-cachefix-subset.md` in `fengyul/dynamo-disagg` (gitlab-master). Reproduces the field 1P1D NIXL wedge on dev-01 with the `head-pr14979cachefix` image (tree = PR `#14979` + a single edit under test). Same harness as exp 1. |
| **Question asked** | What is the *smallest* slice of PR `#13713`, added on top of PR `#14979`, that converts the field wedge into a recovery? |
| **Headline answer** | **No small subset works**, and the reason is the most useful output of the experiment: **the wedge is not one bug — it is three independent failures stacked on the same decode-side code path.** PR `#14979` closes one (F1); each subsequent trial closed one more (F2); none of them touch the one that decides recovery on NIXL (F3). |
| **A/B clincher** | Same cluster, same loadgen, single hour: `#13713` recovers `+30 FAIL → 200/200/200/200`; the largest cachefix-subset trial wedges `5/5 FAIL → 000`. The differentiator lives in `#13713`'s full disagg redesign — specifically F3-done-safely — not in any cacheTransceiver-only subset. |
| **Status** | Confirmed for NIXL deployments (field transport). The customer-facing implication is in [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md). The architectural follow-up implication for the cancel-and-poison redesign is in [`../../design/disagg-inflight-cancel-poison/README.md`](../../design/disagg-inflight-cancel-poison/README.md). |

> **Why a new doc instead of updating `02-failure-signatures.md` or `03-defect-class-stack.md`.** The F1 / F2 / F3 lens is a **mechanism story for the customer-visible decode wedge**, validated against an external A/B test. The existing layer model (`L1`–`L11` in doc 03) and signature catalogue (`#1`–`#9` in doc 02) are still the canonical "what bugs are there" / "which invariants are missing" framing. F1 / F2 / F3 is a third, intentionally smaller lens — "how does the decode wedge manifest under field load, and what must each fix actually do?" — that maps cleanly onto layers and signatures but reads independently. The mapping table in §3 below makes that cross-reference explicit.

## 1. Mental model — the decode worker's three hats

A single decode request's life in disagg: the decode worker reserves KV blocks, asks the prefill worker to push that request's KV into them, and waits for the bytes to land before it can start generating. **Three different things can go wrong while it waits**, all on the same code path, each fixable with a different change.

It helps to think of the decode worker's single per-rank engine thread (the PyExecutor loop) as wearing two hats it must alternate between every iteration:

- **Hat A — "check if my KV arrived":** call `cacheTransceiver.checkGenTransferStatus`, which for each pending receive calls `future.get()` (rc11) or `future.wait_for(t)` (PR `#13713`) on the future handed out by `receiveAsync`.
- **Hat B — "do everything else":** schedule the next batch, run forward, sample, service health probes, process Python-level cancellations.

If hat A blocks indefinitely (unbounded `get()`), the thread never takes it off and never gets to hat B. That's F2.

But even with hat A bounded — so the loop keeps cycling — the *transfer itself* may still be stuck for reasons that have nothing to do with how long hat A waits. On NIXL, the transfer runs on a **separate UCX background progress thread** (`cpp/tensorrt_llm/runtime/utils/ucxCacheCommunicator.cpp:331`, `startProgressThread(true)`); the engine thread polls but doesn't drive bytes. So a stuck transfer is *its own problem*, distinct from the engine's responsiveness. Recovering it requires cancelling the transfer and freeing its buffers — **safely**, after the network has quiesced. That's F3.

And before either of those becomes the dominant problem, the request may already be freed underneath the transfer worker, which holds a raw `LlmRequest*`. That's F1.

So: F1 closes the use-after-free crash; F2 keeps the engine responsive; F3 makes recovery from a stuck transfer actually possible.

## 2. The three failures

### F1 — `Broken promise` use-after-free crash

**Mechanism.** `cacheReceiver->receiveAsync(req)` hands a **background transfer worker** the request and a `std::promise` it will fulfill once the bytes arrive. In `rc11`-and-before, the worker held the request by a raw `LlmRequest*`. If Python's `_terminate_request` freed the request first (timeout, client disconnect, error), the worker was left holding a dangling pointer, and the `promise` could be destroyed *before* anyone called `set_value` on it. The matching `future` then returns with `std::future_error: Broken promise`. **This is a crash/abort, not a hang.**

**Code site.** `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` — `Impl::receiveAsync`, the `Response` struct's `LlmRequest*` field, the `requestAndReceiveAsyncMultiThreads` worker. Same shape on the sender side (`sendAsync`, `RequestAndPromise`, `handleAsyncSend`).

**Fix.** Capture the request by `std::shared_ptr<LlmRequest>` so the transfer worker co-owns it for the duration of the transfer. The two layers of fix are:

1. **Outer layer** (`cacheTransceiver.cpp` `mSenderFutures` / `mRequesterFutures`) — landed in PR `#14768`.
2. **Inner layer** (`dataTransceiver.cpp` `Response` / `RequestAndPromise`) — landed in PR `#14979` (port from PR `#13713`).

**Layer / signature mapping.** Defect class L1 (lifetime/RAII), signature `#1` (Broken promise) in [`02-failure-signatures.md`](02-failure-signatures.md).

**Empirical confirmation.** `Broken promise = 0` in every build that has the shared_ptr port, including `#14979` head and every cachefix-subset trial.

### F2 — Engine-loop freeze on unbounded `future.get()`

**Mechanism.** `CacheTransceiver::checkGenTransferStatus` iterates the pending receives and calls `future.get()` on each. `get()` is **unbounded** — it blocks the calling thread until the future is satisfied. The calling thread is the **per-rank PyExecutor engine loop** (one thread per rank, doing everything serially: schedule, forward, sample, *and* this status check). If one receive is stuck, the engine thread parks inside `get()` and the **entire loop stops**: no scheduling, no forward, no token generation, no `/health` servicing, no Python-side cancellation processing. After ~300 s the `HangDetector` watchdog trips and logs `Hang detected on rank N in PyExecutor`.

**Key NIXL observation.** Both the reproducer and the prod shadow use the NIXL transport (`backend: NIXL`). NIXL/UCX runs its own background progress thread (verified in-tree at `ucxCacheCommunicator.cpp:331`). **So the transfer keeps advancing on the progress thread regardless of what the engine thread is doing.** The blocking `get()` does not stall the transfer itself — it freezes only the engine loop. Whatever made a given transfer stuck is a *separate* problem (F3); the unbounded `get()` just compounds it by taking the whole engine down with one stuck receive.

**Code site.** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp::checkGenTransferStatus` and `checkContextTransferStatus`.

**Fix.** Replace the unbounded `future.get()` with a bounded `future.wait_for(≤50 ms)` poll so the engine thread is never parked for more than 50 ms on any one receive. The thread falls through, does the rest of its work, and re-polls on the next iteration. This is the `cacheTransceiver.cpp` commit in PR `#13713`. **Not** in PR `#14979`.

**Layer / signature mapping.** Defect class L3 (cancellation pathway) — the unbounded wait is the *blocker* that prevents Python-side timeouts and cancels from ever running. Symptomatically presents as signature `#4` (engine-loop hang).

**Empirical confirmation.** Trial 1 (`#14979 + bounded poll`): `Hang detected = 0`. Engine remains responsive. **Still wedged 5/5** — un-freezing the loop does not progress the stuck transfer on NIXL.

### F3 — Stuck transfer is never cleaned up *safely* (the load-bearing one)

**Mechanism (originating trigger, inferred).** Under the repro's deliberate KV pressure (`free_gpu_memory_fraction=0.2`, conc-16, ISL ≈ 8000) the transfer genuinely stalls for a credit/buffer reason: completing the transfer needs free KV blocks on **both** ends simultaneously, but decode has none free to receive into (pinned by other in-flight receives that are themselves waiting), and prefill is holding computed KV it can't push yet. The log fingerprint is exactly this: `exceeded total timeout` → `cancelled before send` (prefill never pushed) → requests that never complete and blocks that never return.

**Mechanism (the trap, the load-bearing part).** To recover, the stuck transfer's KV blocks must be returned to the pool — but **only after the transport is truly done with those buffers.** `rc17`'s `py_executor.py` does already detect the timeout and call `_terminate_request` (the `_check_kv_transfer_timeout` → `py_kv_transfer_timed_out` → `cancel → free` path *fires* in logs). The bug is that it frees the buffers **eagerly**, the instant it cancels, *while the UCX progress thread may still be touching them*. That corrupts transport state, and from then on every subsequent transfer fails `cancelled before send` → **permanent wedge**.

**Code site.** `tensorrt_llm/_torch/pyexecutor/py_executor.py` — `_check_kv_transfer_timeout` / `_terminate_request` / `_release_kv_resources` and the `AsyncTransferManager`'s release path.

**Fix.** **Quiescence-gated (deferred) freeing.** Cancel the stuck transfer first; then free its blocks only once the transport confirms it has quiesced for that request. This is PR `#13713`'s `py_executor.py` redesign, anchored on two new predicates:

- `_is_unquiesced_disagg_transfer(request)` — "the C++ transfer status for this request is still pending; we cannot safely free yet."
- `_can_terminate_request_now(request)` — "every prerequisite for safe termination is satisfied: cancellation has propagated, transport has quiesced, no rank still claims this request is in flight."

Plus an `AsyncTransferManager` rewrite that tracks "resources safe to free" rather than "resources to free now." This is spread across `py_executor.py` *and* the transfer-manager API, and is tangled with rc16 → rc17 differences — it **cannot be cleanly cherry-picked** as a small patch.

**Layer / signature mapping.** Defect class L4 + L5 (fail-closed memory safety; deferred-cleanup invariant) and the underlying C4 invariant ("deferred cleanup is a globally consistent decision") in the design doc. Symptomatically presents as the customer wedge after F1 + F2 are closed.

**Empirical confirmation (Trial 2 — the instructive failure).** Trial 2 added an active drain (cancel → wait-ready → erase) on top of the bounded poll. The drain *engaged* — `Cannot cancel 243 → 1`, `exceeded total timeout; cancelling` fired — confirming that detection-and-termination was already working. **It still wedged 5/5.** The drain freed eagerly, exactly like `rc17`, and so poisoned the transport the same way. The conclusion: *detecting* the stuck transfer was never the gap; *freeing it safely* is.

## 3. Mapping back to the existing framework

| F-failure | Layer (doc 03) | Signature (doc 02) | Fix landed in |
|---|---|---|---|
| **F1** Broken promise UAF | L1 (lifetime / RAII) | `#1` Broken promise | PR `#14979` (inner `dataTransceiver.cpp`) + PR `#14768` (outer `cacheTransceiver.cpp`); both ports from PR `#13713` |
| **F2** Engine-loop freeze | L3 (cancellation pathway blocker) | `#4` engine-loop hang | PR `#13713` (`cacheTransceiver.cpp` bounded `wait_for` poll); **not** in `#14979` |
| **F3** Eager-free poisons transport | L4 + L5 (fail-closed + deferred-cleanup invariant); maps to C4 (deferred cleanup is a global decision) in the design doc | symptomatic: customer wedge after F1+F2 closed | PR `#13713` (`py_executor.py` + `AsyncTransferManager` redesign with `_is_unquiesced_disagg_transfer` / `_can_terminate_request_now`); not cleanly cherry-pickable |

## 4. Trial table from the experiment

| Trial | F1 crash | F2 freeze | F3 safe cleanup | Recovers on NIXL? |
|---|---|---|---|---|
| **PR `#14979`** (dataTransceiver `shared_ptr`) | ✅ | — | — | **no** |
| Trial 1 (+ bounded 50 ms poll) | ✅ (inherited) | ✅ | — | **no** |
| Trial 2 (+ active drain: cancel → wait-ready → erase) | ✅ | ✅ | ⚠️ attempted, but **eager free** (same mistake as rc17) | **no** |
| **PR `#13713`** (full) | ✅ | ✅ | ✅ quiescence-gated | **yes** |

## 5. Strategic implications

### 5.1 For PR `#14979`

PR `#14979` is **necessary but not sufficient** for the production wedge. F1 is a real and useful close — eliminates a co-occurring crash class — and the change is a strict subset of `#13713`, so there is no future merge friction. The PR description should not, however, imply it addresses the field decode wedge on NIXL. It does not, by Trial 1's evidence. Recommended PR description framing:

> Fixes F1 (`Broken promise` use-after-free) from the three-failure decomposition documented in `19-exp4-f1-f2-f3-decomposition.md`. F2 (engine freeze) and F3 (eager-free wedge) are out of scope — they need PR `#13713`'s bounded poll + quiescence-gated freeing, the latter of which is structurally tangled across `py_executor.py` and the transfer manager API and not cleanly portable. Deployments hitting the field wedge on NIXL need PR `#13713` in full; this PR removes a co-occurring crash class.

### 5.2 Why bounded polling alone is insufficient on NIXL

It is worth being precise about this because "make `get()` bounded" sounds like an obviously good fix and was the first thing tried. What bounded polling actually buys, and does not:

**Buys (Trial 1 confirmed):**

- Engine remains responsive; `/health` probes keep answering → orchestrator does not SIGKILL the worker.
- `_check_kv_transfer_timeout` is *reachable*; before bounded polling it is parked behind the unbounded `get()` and never runs.
- Cancellations from Python actually get processed within ≤50 ms of being queued.
- Healthy requests on the same rank keep generating; one stuck transfer does not punish every tenant.
- `HangDetector` no longer fires.

**Does not buy (Trial 1 confirmed wedge 5/5):**

- The transfer itself does not progress just because we polled — UCX's background progress thread is the byte-mover, and F3's credit/buffer deadlock is what stalled it.

**Subtle downside.** Bounded polling *uncorks* the cancellation path. `_terminate_request` runs, `rc17`'s eager-free executes, the transport gets poisoned. Without bounded polling, the wedge manifests as "engine frozen → watchdog SIGKILL → restart." With bounded polling, the wedge manifests as "transport poisoned, 5/5 FAIL, permanent." That is *better for diagnostics* (visible per-request errors, persistent state to inspect) but it is **not less wedged**. Bounded polling without F3 is a degradation-mode improvement, not a recovery mechanism.

The corollary for the PR chain: there is no value in adding the F2 fix to PR `#14979` in isolation. F2 on its own does not change production outcome, and shipping it without F3 makes the failure surface slightly bigger (Python-side timeouts that previously never fired now do, hitting the unsafe eager-free path). F2 should land with F3, as it does in PR `#13713`.

### 5.3 For the cancel-and-poison architectural redesign

The F3 finding is direct empirical evidence for the design doc's existing Phase 2 thesis (**deferred un-poison via status polling**) and for the consensus invariant C4 ("deferred cleanup is a globally consistent decision"). Trial 2 replicates `rc17`'s eager-free *exactly because* the per-rank cleanup machinery makes a unilateral "free now" decision; the design doc's Phase 1 cancellation contract is what makes the safe-vs-unsafe distinction structurally enforceable.

Specifically:

- **C4 (deferred cleanup is a global decision)** is what F3 violates in `rc17` and what Trial 2 also violates. The fix is not "wait longer before freeing"; it is "freeing is a decision that depends on transport quiescence, and that decision must be globally consistent."
- **Phase 2 (deferred un-poison via NIXL polling)** is the operational mechanism that realizes C4 for the V1 + C++ path. The exp4 evidence is that this mechanism is **load-bearing**, not optional: without it the wedge is permanent.
- **Phase 1 V1 alignment options (a) / (b) / (c)** — exp4 does not pick between them by itself, but it raises the cost of option (a) (duplicate consensus in Python on the V1 + C++ path). Any per-rank-decision shape that touches cleanup must satisfy C4; option (a)'s duplication of the consensus logic increases the surface for getting that wrong.

## 6. Recommendation

**Ship the full PR `#13713` for any deployment hitting the field decode wedge.** The "safe subset" path (`#14768` → `#14979`) is good hygiene and useful crash-class closure, but it does not fix the wedge. Per the conservative/cautious guidance, the entangled F3 redesign is **not** cherry-pickable in isolation; the deployable that passes the reproducer **is `#13713` itself**, gated for risk control via `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL` (default OFF), and flipped on by affected deployments.

## 7. Confidence and gaps

**Proven (source + logs):**

- F1 / F2 / F3 mechanisms, including NIXL's verified background progress thread (`ucxCacheCommunicator.cpp:331`) which establishes that bounded polling and transfer-stall recovery are *separate* concerns on NIXL.
- A/B clincher: PR `#13713` recovers 200/200/200/200, cachefix-trial-2 wedges 5/5 FAIL, identical harness, single hour.

**Inferred (config + log signatures, not directly traced):**

- F3's originating trigger inside the transfer — that KV-block / credit pressure is what makes the prefill fail to push. Pinning this down empirically needs **transceiver-level block-accounting logs** on both ends of the cancel path (free counts on the prefill side at `cancelled before send`; pinned-block counts on the decode side at `exceeded total timeout`). The `nvbug6104831-diag-logging` branch is the right vehicle; this is a candidate Layer G in that branch's log layering (Layers A–F already cover lifetime / cross-rank dedup / data-transfer phase). Closing this gap moves F3 from "inferred trigger" to "proven trigger" and is the most efficient single empirical follow-up.

**Out of scope here:**

- The behaviour of direct UCX vs NIXL with respect to F2 / F3. The exp4 report deliberately tests NIXL (field transport). Direct UCX has its own progress model and saturation characteristic (see `00-tldr.md` and `06-fix-approaches/D-combo.md#direct-ucx-saturation-evidence-diagnostic-build`); F1's fix transfers; F2 and F3 may need a separate empirical pass.

## 8. Reproduce / build artifacts referenced in the source experiment

- Patch carrying the largest cachefix-subset trial: `../patches/pr14979-cachefix-13713-boundedpoll.patch` (in `fengyul/dynamo-disagg`).
- Trial 2 worker logs: `../deploy/logs/pr14979cachefix-*.log`.
- Image: `head-pr14979cachefix` (rc17 tree + the edit under test, overwritten each trial); deployed as `repro-kvhang-14979cachefix` on dev-01.
- A/B comparison image: `repro-kvhang-13713` (PR `#13713` full).

## 9. Cross-references

- This investigation: [`00-tldr.md`](00-tldr.md) (start here), [`02-failure-signatures.md`](02-failure-signatures.md) (`#1`, `#4`), [`03-defect-class-stack.md`](03-defect-class-stack.md) (L1, L3, L4, L5), [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) (PR chain & landing plan), [`10-ablation-no-midflight-cancel.md`](10-ablation-no-midflight-cancel.md) (six-experiment ablation, complementary evidence).
- Design doc: [`../../design/disagg-inflight-cancel-poison/README.md`](../../design/disagg-inflight-cancel-poison/README.md) (cancellation contract C1–C5, Phase 1 V1 alignment options, Phase 2 deferred un-poison).
- Diag logging: branch `nvbug6104831-diag-logging` (candidate Layer G — KV-block accounting on cancel path).
