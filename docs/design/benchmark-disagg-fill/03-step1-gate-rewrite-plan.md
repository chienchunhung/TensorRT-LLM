# Step 1 — Gate-Condition Rewrite (v3)

[< Back to index](README.md)

**Prerequisite reading:** [`02-regression-investigation.md`](02-regression-investigation.md) — do not proceed without the full causal chain. This plan assumes you understand why the real-gen-count gate is unsatisfiable under ADP router skew, why dummy suppression creates a circular dependency, and why PR #12206's fail-fast fires as a secondary symptom.

Bug reference: nvbug 6071070, PR [#12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) regression. Failing test: `perf/test_perf_sanity.py::test_e2e[disagg-gen_only-wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_con8192_ccb-NIXL]`.

---

## 1. Goal

Replace the "real-gen-count ≥ threshold" gate-completion predicate with a **state-based** predicate that does not depend on any ADP-router balance invariant, while keeping the non-blocking single-threaded executor structure from PR #12208 intact.

After this change:

- Gate opens as soon as every request the benchmark asked for is past its KV-transfer phase, regardless of how the ADP router distributed them across ranks.
- Dummy-suppression-during-fill is preserved (no leak).
- PR #12206's fail-fast remains, but its trigger is decoupled from the fill-phase transient.

Out of scope for this PR: changing where the gate lives (that's step 2), changing the ADP router, or changing CTX-side behavior.

---

## 2. The new fill-complete predicate

### 2.1. Definition

Fill is complete on a given executor iteration iff **all three** of the following hold globally (across all TP ranks in an ADP deployment):

**(A)** The executor has fetched at least `benchmark_req_queues_size` requests cumulatively:
```
num_fetch_requests  ≥  benchmark_req_queues_size
```

**(B)** Every request currently in `active_requests` is *past* the KV-transfer phase. Concretely, no request is in:
- `DISAGG_GENERATION_INIT` (waiting to request a transfer), or
- `DISAGG_GENERATION_TRANS_IN_PROGRESS` (transfer in flight).

Equivalently: every active request is in one of `{TRANS_COMPLETE, GENERATION, ...}` — a state from which a normal generation forward pass is valid.

**(C)** There are no pending transfers held by the KV cache transceiver for requests still in INIT (i.e. nothing about to move a request *into* an earlier state).

Intuition: (A) says "we've admitted the full benchmark batch"; (B) says "none of them are still mid-transfer"; (C) says "and nothing is in the pipeline to add a new mid-transfer one."

### 2.2. What this replaces

Replaces the count-based criterion that was introduced by PR #12208:

```python
# OLD (PR #12208) — brittle
local_gen_count = sum(1 for req in scheduled_batch.generation_requests
                      if not req.is_attention_dp_dummy)
if self.enable_attention_dp:
    total_gen_count = sum(self.dist.tp_allgather(local_gen_count))
else:
    total_gen_count = local_gen_count
return total_gen_count >= self.benchmark_req_queues_size
```

The new predicate is **immune** to the per-rank distribution skew described in §5 of the investigation doc: it asks "are all admitted requests ready for generation?" not "did the router put ≥ threshold/tp_size on every rank?"

### 2.3. Why this is correct

- **Under perfect balance** (old invariant): when each rank has exactly `threshold / tp_size` real requests, every request has completed KV transfer before the gate check that would have opened the count-based gate. So (A) ∧ (B) ∧ (C) is true no later than the old criterion was.
- **Under imbalance**: the count-based gate never opens. The state-based gate opens correctly because (A) fires once all requests are fetched, (B) fires once all transfers complete, and (C) fires once the transceiver's inbound queue drains.
- **Under genuine KV-capacity insufficiency**: (B) is violated forever because some request is stuck in `DISAGG_GENERATION_INIT`. This is exactly the condition PR #12206's fail-fast detects (see §3.5), so that fires instead of the gate — which is the intended behavior.

### 2.4. ADP / collective correctness

For ADP, (B) and (C) must be evaluated *globally*, not per-rank, because different ranks hold different request subsets. Use the same allgather pattern as the existing code:

```python
local_all_past_transfer = all(
    not (req.is_disagg_generation_init_state
         or req.is_disagg_generation_transmission_in_progress)
    for req in self.active_requests
)
local_no_inflight_transfers = (self.kv_cache_transceiver is None
                               or self.kv_cache_transceiver.num_pending_transfers() == 0)
local_ok = local_all_past_transfer and local_no_inflight_transfers

if self.enable_attention_dp:
    # AND across ranks. Use tp_allgather of int and check min == 1.
    global_ok = min(self.dist.tp_allgather(int(local_ok))) == 1
else:
    global_ok = local_ok

return (self.num_fetch_requests >= self.benchmark_req_queues_size) and global_ok
```

All ranks enter this function on every iteration (same as today's gate), so the allgather is collective-safe. Use a single int allgather, not an all-reduce — symmetric with the current code's `sum(self.dist.tp_allgather(local_gen_count))` pattern.

**Verify** that `num_pending_transfers()` (or whatever the actual attribute is — see §5.2) exists on the transceiver. If not, fall back to the next-weakest form: (A) ∧ (B). Empirically (A) ∧ (B) is sufficient when the transceiver's completion callback updates request state atomically with the transfer finishing, which is the current design — condition (C) is belt-and-braces.

---

## 3. Dummy-suppression changes

### 3.1. Why the current suppression must stay (mostly)

Today, `_should_skip_dummy_for_benchmark_disagg` returns True whenever `_benchmark_fill_phase_active` is True. PR #12208's analysis doc (§adp-dummy-requests.md) is correct that inserting dummies *during* the fill phase causes a leak: a dummy inserted while the batch is < full never gets removed because the taper-down logic only fires when the rank empties. The suppression must stay for fill-phase correctness.

### 3.2. Where the new predicate opens the gate

Under the new predicate, the gate opens on the iteration where (A) ∧ (B) ∧ (C) first holds. At that point:

- All real requests are past KV transfer.
- `_benchmark_fill_phase_active` transitions False (same place as today: `_check_benchmark_disagg_gate`).
- On the *next* iteration, the normal dummy lifecycle applies; any rank that now has 0 requests (because the ADP router under-filled it) gets a normal dummy via `_pad_attention_dp_dummy_request`.

This matches the "taper-down safety" invariant listed in [`01-history-nonblocking-gate/analysis.md`](01-history-nonblocking-gate/analysis.md#correctness).

### 3.3. The single-iteration risk

There is a one-iteration window where the gate has opened but a rank may have 0 real requests (the router under-filled it). Two sub-cases:

**Case A: under-filled rank has 0 real requests at gate-open.** The scheduled_batch on that rank is empty; `_pad_attention_dp_dummy_request` runs with `_benchmark_fill_phase_active == False` and inserts a dummy. Forward pass proceeds. Subsequent iterations see the rank in the "has zero real reqs, has one dummy" state, which the existing taper-down code already handles.

**Case B: under-filled rank has ≥ 1 real request.** Normal behavior; no dummy needed.

Both cases already work under today's code for the taper-down phase; the gate-open transition simply activates that existing path one iteration earlier than today (because today, under imbalance, it never activates).

### 3.4. No change needed to `_should_skip_dummy_for_benchmark_disagg` itself

The predicate is still `return self._benchmark_fill_phase_active and not self.is_warmup`. What changed is **when** `_benchmark_fill_phase_active` flips False — now determined by the state-based predicate rather than the count-based one. No code change here.

---

## 4. PR #12206 fail-fast — update the trigger

### 4.1. What's wrong today

```python
# Current (PR #12206) — fires on transient imbalance
if (self.benchmark_req_queues_size > 0 and not self.is_warmup
        and not fitting_disagg_gen_init_requests):
    stuck_init_requests = [req for req in self.active_requests
                           if req.is_disagg_generation_init_state]
    if (stuck_init_requests and self.num_fetch_requests
            >= self.benchmark_req_queues_size):
        self._handle_errors("Insufficient KV cache ...")
        return None, None
```

The predicate treats "one iteration where scheduler couldn't fit any new INIT request" as evidence of permanent KV insufficiency. Under the regression, the overflow requests on overshooting ranks look like stuck INITs, but transfers are still in flight on other ranks — it's transient.

### 4.2. Fix — require multiple consecutive iterations of no progress

Change the trigger to "this has been true for N consecutive iterations with no progress":

```python
if (self.benchmark_req_queues_size > 0 and not self.is_warmup
        and not fitting_disagg_gen_init_requests):
    stuck_init_requests = [req for req in self.active_requests
                           if req.is_disagg_generation_init_state]
    if (stuck_init_requests and self.num_fetch_requests
            >= self.benchmark_req_queues_size):
        # Track forward progress: a completed transfer or a newly fit INIT
        # resets the counter.
        if self._disagg_fill_stall_signature == (len(stuck_init_requests),
                                                  num_transfers_completed_total):
            self._disagg_fill_stall_iters += 1
        else:
            self._disagg_fill_stall_iters = 0
            self._disagg_fill_stall_signature = (len(stuck_init_requests),
                                                  num_transfers_completed_total)

        if self._disagg_fill_stall_iters >= FILL_STALL_THRESHOLD:
            self._handle_errors(error_msg, requests=self.active_requests)
            return None, None
```

Where:
- `num_transfers_completed_total` is a monotonic counter of completed KV transfers (add to the transceiver or compute from request state transitions).
- `FILL_STALL_THRESHOLD` = e.g. 50 iterations with 0.1s sleep between gate checks = 5 s of zero progress. Tune based on the existing CTX-side 1 s timeout.

### 4.3. Alternative — predicate via request-state instead of counts

A cleaner variant: only fire if *no* active request has made a state transition for N iterations:

```python
self._last_state_snapshot_hash = ...  # hash of (req_id -> state) map
if current_snapshot_hash == self._last_state_snapshot_hash:
    self._no_progress_iters += 1
else:
    self._no_progress_iters = 0
    self._last_state_snapshot_hash = current_snapshot_hash

if self._no_progress_iters >= FILL_STALL_THRESHOLD and stuck_init_requests:
    # fire
```

Either form is acceptable. The key point is: **do not use `fitting_disagg_gen_init_requests == []` on a single iteration as the sole trigger**. That condition is true for every iteration where the transceiver hasn't completed a transfer in the last scheduler tick, which is most iterations during normal operation.

### 4.4. Error message

Keep the existing message but soften the "KV insufficient" framing — under the new logic, if it fires, the cause is *either* genuine KV insufficiency *or* a truly stuck transfer, both of which need investigation. Consider:

```
Gen-only benchmark fill has not progressed for {N} iterations with
{M} requests stuck waiting for KV cache. Probable causes:
  1. GEN free_gpu_memory_fraction too low (reduce concurrency or raise fraction).
  2. CTX→GEN KV transfer stuck (check NIXL/UCX transceiver logs).
```

---

## 5. Concrete code changes

### 5.1. File: `tensorrt_llm/_torch/pyexecutor/py_executor.py`

**(a) Replace `_is_benchmark_disagg_fill_complete` (lines ~1935–1972).**

New body:

```python
def _is_benchmark_disagg_fill_complete(self, scheduled_batch) -> bool:
    """State-based fill-complete predicate. Gate opens when:
    (A) num_fetch_requests >= benchmark_req_queues_size, AND
    (B) every active request is past KV-transfer states, AND
    (C) no KV transfers are in flight on this rank.
    For ADP, (B) and (C) are AND-ed across TP ranks via allgather.
    """
    if not self.is_benchmark_disagg or self.is_warmup:
        raise RuntimeError(
            "_is_benchmark_disagg_fill_complete called outside benchmark disagg")

    # (A): cumulative fetch count — local, no sync needed
    if self.num_fetch_requests < self.benchmark_req_queues_size:
        return False

    # (B): all active requests past transfer states
    local_all_past_transfer = all(
        not (req.is_disagg_generation_init_state
             or req.is_disagg_generation_transmission_in_progress)
        for req in self.active_requests
    )

    # (C): no inflight transfers
    local_no_inflight = (
        self.kv_cache_transceiver is None
        or not self.kv_cache_transceiver.has_pending_transfers()
    )

    local_ok = int(local_all_past_transfer and local_no_inflight)

    if self.enable_attention_dp:
        global_ok = min(self.dist.tp_allgather(local_ok)) == 1
    else:
        global_ok = local_ok == 1

    return global_ok
```

Note: the function signature still takes `scheduled_batch` for backwards compatibility with callers, but the parameter is no longer used. Remove the parameter in a follow-up if the compiler flags it.

**(b) Update `_check_benchmark_disagg_gate` (lines ~1974–2003).**

No logic change. The helper still calls `_is_benchmark_disagg_fill_complete`, clears `_benchmark_fill_phase_active` on True, sleeps and retries on False. The existing signature/shape is preserved — only the underlying predicate changes.

Verify this call site compiles given the parameter change in (a). If you keep the unused parameter, no call-site edit needed.

**(c) Update PR #12206 fail-fast (lines ~1884–1907).**

Add stall tracking. In `__init__`:

```python
self._disagg_fill_stall_iters = 0
self._disagg_fill_stall_signature = None
```

Replace the fail-fast block with the progress-tracking version from §4.2. Constant: `_DISAGG_FILL_STALL_THRESHOLD = 50` (~5 s at 0.1s sleep). Put it at module scope so tests can monkeypatch it down.

**(d) Verify `has_pending_transfers` / equivalent on the transceiver.**

If it doesn't exist, add it to the relevant classes. Candidates (verify actual names in the source):
- `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
- The C++ binding surface if the transceiver is a native object.

If exposing a new attribute is risky, drop condition (C) from the predicate and rely on (A) ∧ (B). This weakens the guarantee slightly but is acceptable if transceiver state transitions request state atomically.

### 5.2. File: `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`

Add `has_pending_transfers(self) -> bool` if not already present. Should return True iff the transceiver has any pending INIT → TRANS_IN_PROGRESS or TRANS_IN_PROGRESS → TRANS_COMPLETE transitions that have not yet fired.

### 5.3. File: `tensorrt_llm/_torch/pyexecutor/llm_request.py` (or wherever `is_disagg_generation_init_state` lives)

Verify the existence of `is_disagg_generation_init_state` and `is_disagg_generation_transmission_in_progress` property accessors. These are already used elsewhere in `py_executor.py`, so they should exist.

---

## 6. Invariants to preserve

Cross-reference with [`01-history-nonblocking-gate/analysis.md`](01-history-nonblocking-gate/analysis.md#correctness). After this change, every invariant in that table must still hold:

| Property | How preserved |
|---|---|
| No premature forwarding | (A) ensures threshold met; (B) ensures no request mid-transfer sees forward. Equivalent to original. |
| No deadlock | (B) ∧ (C) cannot be frozen by imbalance, only by genuine stuck transfers — those are caught by the updated fail-fast. |
| Warmup bypass | Unchanged. |
| Latching gate | Unchanged. |
| Fill phase flag | Still cleared once, in `_check_benchmark_disagg_gate`, only when new predicate returns True. |
| ADP allgather consistency | New predicate still allgathers under ADP; pattern unchanged. |
| GEN KV failure still detected | Updated predicate in §4 detects **persistent** stuck INIT; transient overflow no longer false-positives. |
| Taper-down safety | Unchanged — dummy lifecycle resumes at gate-open. |

---

## 7. Test plan

### 7.1. Update existing unit tests

File: `tests/unittest/_torch/executor/test_benchmark_disagg.py` (already has 40 tests across 8 classes from PR #12208).

**Modify:**
- `TestFillCompleteNonADP` and `TestFillCompleteADP`: Tests that asserted behavior based on `generation_requests` counts need to be rewritten to assert on request states.
- `TestFillCompleteADPDummyExclusion`: The dummy-exclusion tests are no longer directly meaningful (new predicate doesn't look at dummies). Replace with tests that verify dummies in `active_requests` don't count toward the threshold because they can't be in `DISAGG_GENERATION_*` states.

**Add: `TestFillCompleteStateBased`** — 5–7 tests:
- All ADP ranks have threshold/tp_size real requests, all in TRANS_COMPLETE → gate opens.
- One rank has one fewer (the regression case): `{255, 256, 256, ..., 256}` all in TRANS_COMPLETE → gate **opens** (vs. old behavior of never opening).
- One rank has one more: `{257, 256, ..., 256}` with the 257th in INIT → gate does NOT open (correct: that req is still admitting).
- Threshold not yet fetched → gate does NOT open regardless of state.
- Transceiver reports pending transfers → gate does NOT open.
- ADP allgather: one rank reports `local_ok=0`, gate does NOT open on any rank.

### 7.2. Add regression test for the imbalance hang

Add `TestADPRouterImbalanceHang` — end-to-end-ish test in a controlled fixture:
- Mock the ADP router to produce imbalanced distribution (one rank 255, one rank 257, rest 256).
- Run fill phase.
- Assert: gate opens within N iterations (where N is bounded by KV-transfer completion, not by achieving count == threshold).

This is the specific scenario that failed on `wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256`.

### 7.3. Update the PR #12206 dedicated test

File: `tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_benchmark_gen_only_insufficient_kv` (line 724).

This test uses `TLLM_BENCHMARK_REQ_QUEUES_SIZE=64` and validates fail-fast. With the new stall-count logic, verify:
- Genuine KV insufficiency still fails fast (within ~5s of stall, not 1 iter).
- The test still detects the correct error message.
- If the test relied on sub-second failure detection, adjust its timeout expectations.

### 7.4. Run the failing perf test

Reproduce the regression locally (if possible, on a DEP32 config — may require a GB200 node; the user's benchmark environment is `umb-b300-dp-186` and similar):

```
pytest tests/integration/defs/perf/test_perf_sanity.py::test_e2e -k \
    "disagg-gen_only-wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_con8192_ccb-NIXL"
```

Must: complete with all 8192 requests successful.

### 7.5. Regression check on smaller configs

Run the existing disagg tests that *don't* trigger the boundary condition, to confirm no regression:
- `tests/integration/defs/disaggregated/test_disaggregated.py` — all tests pass.
- `tests/unittest/_torch/executor/test_benchmark_disagg.py` — updated suite passes.

---

## 8. Rollout / commit strategy

Single PR, ready to merge once CI is clean:

**Commit 1:** Add `has_pending_transfers` to transceiver (if needed).
**Commit 2:** Rewrite `_is_benchmark_disagg_fill_complete` to state-based predicate. Update tests.
**Commit 3:** Update PR #12206 fail-fast to progress-tracking. Update PR #12206 dedicated test.
**Commit 4:** Add regression test for ADP router imbalance.

Squash on merge. PR title: `[NVBUG-6071070][bug] Fix disagg gen-only benchmark hang under ADP router imbalance`.

PR description must reference: PR #12091, PR #12206, PR #12208, nvbug 6071070, and this design doc lineage (via public URLs once the docs land on `main`).

---

## 9. What this PR explicitly does NOT do

- Does not remove the fill gate from `PyExecutor` (that's step 2).
- Does not change the ADP router's distribution algorithm.
- Does not change CTX-side transfer logic.
- Does not touch the non-benchmark disagg code path (which has always been `is_benchmark_disagg=False`).
- Does not change `TLLM_BENCHMARK_REQ_QUEUES_SIZE` semantics or the benchmark client contract.

---

## 10. Known limitations that step 2 addresses

Even with the state-based gate, `PyExecutor` still owns benchmark orchestration — a responsibility that doesn't belong there. The gate still runs in the critical executor loop; it still uses `time.sleep(0.1)` (busy-wait); it still requires the transceiver to expose implementation details (`has_pending_transfers`) to a benchmark-only code path; and the `_benchmark_fill_phase_active` lifecycle is still a distinct state machine riding alongside the real request lifecycle.

Step 2 removes the entire feature from `PyExecutor` by moving orchestration to the benchmark client. See [`04-step2-external-orchestrator-plan.md`](04-step2-external-orchestrator-plan.md).
