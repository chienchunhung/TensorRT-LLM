# Step 2 — External Benchmark Orchestrator (v4)

[< Back to index](README.md)

**Prerequisite reading:** [`02-regression-investigation.md`](02-regression-investigation.md) and [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md) in that order. This document assumes you understand what the gate does, why it regressed, and why the state-based predicate in step 1 is a bounded patch rather than a structural fix.

Bug reference (motivating the structural work): nvbug 6071070, PR [#12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) regression. The v3 gate patch (step 1) unblocks CI; v4 removes the class of bug entirely.

---

## 1. Thesis

The benchmark fill gate is **measurement orchestration**, not runtime serving behavior. It exists only so the benchmark harness can guarantee that "time starts when all N requests have landed on GEN" — a property the *client* wants, not a property the server needs. Putting it in `PyExecutor`:

- Pollutes the executor loop with a benchmark-only state machine (`_benchmark_fill_phase_active`, gate polling, dummy suppression).
- Couples the executor to the ADP router's distribution shape (the bug we just fixed).
- Couples the executor to the KV cache transceiver's internal state (the `has_pending_transfers` surface added in step 1).
- Entangles fail-fast for "bad benchmark config" with fail-fast for "bad serving config" — two different error domains that currently share code.
- Requires a feature flag (`TLLM_BENCHMARK_REQ_QUEUES_SIZE`) to toggle a codepath that the server should never know about.

Step 2 moves orchestration to where it belongs: the benchmark client. `PyExecutor` goes back to being a straightforward request-driven executor with no "benchmark mode."

---

## 2. Scope

**In scope:**

1. Define a client-side barrier protocol that lets the benchmark harness wait until N requests are admitted on GEN before the time-measurement window opens.
2. Delete `benchmark_req_queues_size`, `is_benchmark_disagg`, `_benchmark_fill_phase_active`, `_is_benchmark_disagg_fill_complete`, `_check_benchmark_disagg_gate`, `_should_skip_dummy_for_benchmark_disagg` from `PyExecutor`.
3. Simplify `_pad_attention_dp_dummy_request` to its non-benchmark form (remove the `_should_skip_dummy_for_benchmark_disagg` call).
4. Keep PR #12206's "GEN KV insufficient" fail-fast but generalize it — it's a legitimate serving-time check that doesn't depend on benchmark mode (an overloaded CTX/GEN pair should still fail cleanly).
5. Update all benchmark configs and documentation to use the new client-side mechanism.

**Out of scope:**

- Changing the disagg request lifecycle (INIT / TRANS_IN_PROGRESS / TRANS_COMPLETE / GENERATION).
- Changing the ADP router.
- Changing NIXL/UCX/MPI transceiver internals.
- Production (non-benchmark) disagg serving behavior.

---

## 3. Current responsibilities to relocate

| Responsibility | Today | In v4 |
|---|---|---|
| "Wait until all benchmark requests admitted before forwarding." | `_check_benchmark_disagg_gate` in executor loop | Client-side barrier; server never gates |
| "Suppress ADP dummies while admitting." | `_should_skip_dummy_for_benchmark_disagg` | N/A — no fill phase on server |
| "Detect benchmark-gen-only KV insufficiency and fail." | PR #12206 fail-fast gated by `benchmark_req_queues_size > 0` | Generic stuck-INIT watchdog, not benchmark-gated |
| "Measure time from 'all N on GEN' to completion." | Implicit: gate delays forward, client's timer starts at first response | Explicit: client barrier, client's timer starts at barrier release |

The third row is the subtle one. PR #12206's fail-fast was *designed* for the benchmark-gen-only scenario because that was the context where it was motivated. But "GEN can't admit all requested concurrent requests" is a legitimate failure even outside benchmarks — a misconfigured prod deployment hits the same wall. The watchdog should exist; it just shouldn't be predicated on `benchmark_req_queues_size`.

---

## 4. Client-side barrier design

### 4.1. Two candidate mechanisms

**Mechanism A — Poll GEN admission state via a new HTTP endpoint.**

`trtllm-serve` exposes e.g. `GET /v1/admission_state` returning:
```json
{
  "total_active_requests": 8192,
  "requests_in_kv_transfer": 0,
  "requests_in_generation": 8192,
  "num_pending_transfers": 0
}
```

Client algorithm:
```
submit all N requests in parallel (do not await responses)
loop:
    state = GET /v1/admission_state
    if state.requests_in_generation == N and state.num_pending_transfers == 0:
        break
    sleep(100ms)
START_TIMER()
await all N response streams
STOP_TIMER()
```

**Pros:** No server-side barrier, simple client change, no new state machine on the server.
**Cons:** Needs a new API surface (minor); slightly racier than a true barrier (the N could include not-yet-arrived requests in future benchmark variants).

**Mechanism B — Explicit barrier primitive in `trtllm-serve` benchmark mode.**

Client posts a `POST /v1/benchmark_barrier` with `{ "expected_requests": N }`. Server responds 200 only when admission state matches. Under the hood this is the same poll as A, but encapsulated server-side.

**Pros:** Clean API; no client-side polling loop; one round-trip.
**Cons:** Reintroduces benchmark-aware code on the server, just in a different place (HTTP handler vs. executor). Tempting but not clearly better than A.

**Recommendation:** Mechanism A. The server exposes raw state; the client decides what "ready" means. This keeps all benchmark semantics on the client and avoids any benchmark-specific state machine on the server.

### 4.2. Admission state endpoint

Add `/v1/admission_state` (or extend an existing debug/metrics endpoint) that queries the executor for:

- `num_active_requests`
- `num_requests_in_state_generation`
- `num_requests_in_state_disagg_generation_init`
- `num_requests_in_state_disagg_generation_trans_in_progress`
- `num_pending_kv_transfers` (from transceiver)

These are all read-only introspections that `PyExecutor` can compute without a new state machine. The `PyExecutor` side is a pure getter; no polling, no sleeping, no gating.

### 4.3. Client barrier implementation

Location: `tensorrt_llm/serve/scripts/benchmark_serving.py` (or whichever benchmark client drives the failing perf test — verify exact path).

Change sketch:

```python
async def await_full_admission(client, expected_n, timeout_s=120):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = await client.get("/v1/admission_state")
        if (state["requests_in_generation"] == expected_n
                and state["num_pending_transfers"] == 0):
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"GEN did not reach full admission ({expected_n} requests) within {timeout_s}s")

# In benchmark run():
async with aiohttp.ClientSession() as client:
    tasks = [submit_request(client, req) for req in all_requests]
    # Launch all but don't await responses yet.
    submission_futures = [asyncio.ensure_future(t) for t in tasks]

    await await_full_admission(client, len(all_requests))

    t0 = time.monotonic()
    responses = await asyncio.gather(*submission_futures)
    t1 = time.monotonic()

    report_metrics(t1 - t0, responses)
```

Two properties this gives the benchmark:
- **Measurement start is deterministic**: starts strictly after full admission.
- **Failure is local**: if full admission doesn't happen within timeout, the client knows *exactly* what happened (stuck transfers, insufficient GEN KV, etc.) via the admission_state response — not via an opaque 5xx.

### 4.4. Timeout / fail-fast on the client

The client's `timeout_s` replaces server-side PR #12206 fail-fast *for the benchmark context*. The server keeps a generic watchdog (next section), but the benchmark's domain-specific expectation ("I asked for N and I should see N land in generation within T seconds") lives where it semantically belongs.

---

## 5. Server-side cleanup

### 5.1. Deletions from `py_executor.py`

Remove:
- `self.benchmark_req_queues_size` attribute (and its CLI / config plumbing — see §5.4).
- `self.is_benchmark_disagg` attribute.
- `self._benchmark_fill_phase_active` attribute.
- `_is_benchmark_disagg_fill_complete` method.
- `_check_benchmark_disagg_gate` method.
- `_should_skip_dummy_for_benchmark_disagg` method.
- All call sites in `_executor_loop` and `_executor_loop_overlap` (the `can_forward` gate pattern).

Set `can_forward = True` unconditionally at the start of each loop — which is what it was before PR #12091 introduced the fill loop.

### 5.2. Simplify `_pad_attention_dp_dummy_request`

Remove the `_should_skip_dummy_for_benchmark_disagg` call. The normal dummy lifecycle (insert when a rank has zero active requests, remove when real requests fill it) applies unconditionally. This is correct because: under the new design, no benchmark fill phase exists on the server — the first forward pass on GEN happens after the client's barrier releases, by which time all requests have been admitted through the normal (non-benchmark) path.

### 5.3. Generalize the stuck-INIT watchdog

Keep the spirit of PR #12206 but remove the `benchmark_req_queues_size` predicate:

```python
# Generic: fire if INIT requests have been stuck for N consecutive iterations
# with no transfer progress.
if stuck_init_requests and no_progress_for_N_iters:
    self._handle_errors(
        "Stuck disagg requests: {M} request(s) cannot obtain KV cache "
        "allocation. Probable causes: GEN free_gpu_memory_fraction too low, "
        "or stuck CTX→GEN transfer.",
        requests=stuck_init_requests)
    return None, None
```

This runs in prod too, protecting against mis-sized deployments — a legitimate improvement independent of the benchmark.

### 5.4. Remove `TLLM_BENCHMARK_REQ_QUEUES_SIZE`

Delete the env var plumbing:
- Any call site that reads `TLLM_BENCHMARK_REQ_QUEUES_SIZE` from the environment.
- `benchmark_req_queues_size` from `PyExecutorConfig` (or equivalent config dataclass).
- CLI flag in `trtllm-serve` if one exists.
- Any references in `tests/integration/defs/disaggregated/` yaml configs.

Document the removal in a release note: "Benchmark gen-only orchestration has moved to the client. Use `benchmark_serving.py --await-full-admission` instead of `TLLM_BENCHMARK_REQ_QUEUES_SIZE`."

### 5.5. Expose admission-state getter

Add a thin method on `PyExecutor` that snapshots admission state without side effects:

```python
def get_admission_state(self) -> AdmissionState:
    return AdmissionState(
        num_active=len(self.active_requests),
        num_in_generation=sum(1 for r in self.active_requests
                              if r.is_generation_state),
        num_in_disagg_init=sum(1 for r in self.active_requests
                               if r.is_disagg_generation_init_state),
        num_in_disagg_trans=sum(1 for r in self.active_requests
                                 if r.is_disagg_generation_transmission_in_progress),
        num_pending_transfers=(
            self.kv_cache_transceiver.num_pending_transfers()
            if self.kv_cache_transceiver else 0),
    )
```

For ADP, the HTTP handler must aggregate across ranks via an existing cross-rank RPC or broadcast path. If no such path exists, the simplest safe route is for the handler to only expose rank-0's state and document that under ADP, the endpoint reports rank-0's view (not globally correct, but useful for the client barrier under the assumption that rank-0 is representative of "one rank is still admitting → barrier waits").

Better: return per-rank state. The client barrier aggregates client-side:
```python
state["ranks"] = [ {...}, {...}, ... ]  # one entry per TP rank
barrier_met = all(r["num_pending_transfers"] == 0 for r in state["ranks"]) \
              and sum(r["num_in_generation"] for r in state["ranks"]) >= N
```

### 5.6. HTTP endpoint

`tensorrt_llm/serve/openai_server.py` (or whichever file routes FastAPI/aiohttp endpoints — verify). Add:

```python
@app.get("/v1/admission_state")
async def admission_state():
    return JSONResponse(await executor.get_admission_state_async())
```

Thin wrapper; no benchmark-specific behavior.

---

## 6. Migration path

Step 1 must ship first and stabilize in CI. Step 2 is a larger refactor that depends on step 1's correctness (so CI stays green during the refactor window).

**Phase 2a (days, after step 1 lands):**
- Add `/v1/admission_state` endpoint.
- Add client-side barrier in `benchmark_serving.py` (opt-in via a new flag, e.g. `--await-full-admission`).
- Run both the old gate path (via `TLLM_BENCHMARK_REQ_QUEUES_SIZE`) and the new barrier path in CI side-by-side; verify equivalent results on the failing test and on existing benchmarks.

**Phase 2b (next PR):**
- Flip the benchmark configs to use the new barrier. Old env-var path still present but deprecated.
- Add a deprecation warning when `TLLM_BENCHMARK_REQ_QUEUES_SIZE` is set.

**Phase 2c (next release cycle):**
- Delete all server-side benchmark gate code (§5.1, §5.4).
- Delete any remaining references in tests and configs.
- Release note calling out the migration.

Three-phase rollout is defensive — the failing test has burned CI for weeks, and an atomic "delete the gate, replace with client barrier" PR has too much blast radius. Phases 2a and 2b are incremental; 2c is the payoff.

---

## 7. Test plan

### 7.1. New tests

**Server side:**
- `test_admission_state_endpoint.py` — `/v1/admission_state` returns correct fields; updates as requests transition.
- ADP path: state aggregation across ranks is correct.

**Client side:**
- `test_await_full_admission.py` — barrier waits until admission_state reports full; releases promptly when conditions met; raises on timeout.

### 7.2. Port the failing test

`perf/test_perf_sanity.py::test_e2e[disagg-gen_only-wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_con8192_ccb-NIXL]` currently relies on `TLLM_BENCHMARK_REQ_QUEUES_SIZE`. Port to use the new client barrier. Verify it passes end-to-end.

### 7.3. Regression

All `tests/integration/defs/disaggregated/` tests must pass with both old (still-present-in-2a) and new code paths.

### 7.4. Delete test-for-the-feature-being-deleted

`tests/integration/defs/disaggregated/test_disaggregated.py::test_disaggregated_benchmark_gen_only_insufficient_kv` (line 724) — this specifically exercises the PR #12206 fail-fast in its benchmark-gated form. In phase 2c, rewrite to exercise the generalized watchdog (§5.3) instead.

### 7.5. `tests/unittest/_torch/executor/test_benchmark_disagg.py`

Added by PR #12208 (40 tests). In phase 2c, delete the file entirely — the feature it tests no longer exists on the server.

---

## 8. Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Client barrier sees inconsistent cross-rank state under ADP (e.g. rank 0 fully admitted, rank 1 still mid-transfer, but endpoint only reports rank 0). | Medium | Expose per-rank state in the endpoint response (§5.5 option 2). |
| Existing users of `TLLM_BENCHMARK_REQ_QUEUES_SIZE` break silently. | Medium | Two-phase deprecation in phase 2b. Deprecation warning on first call. |
| The `/v1/admission_state` endpoint becomes a hot-polled path in production. | Low | Document it as debug-only; no SLO on the endpoint. Rate-limit at 10 Hz if needed. |
| Removing the gate subtly changes timing under pre-existing disagg deployments. | Low | No production code depends on the gate — `is_benchmark_disagg` is False without `TLLM_BENCHMARK_REQ_QUEUES_SIZE`. The code paths are unreachable in prod today. |
| Generalized watchdog (§5.3) false-positives on a legitimately slow CTX. | Medium | Tune `N iterations / stall threshold` conservatively; make it configurable; include the signal `num_transfers_completed_total` so progress is detected even when INIT count is flat. |

---

## 9. Design invariants

After step 2 ships, these must hold:

1. **`PyExecutor` has no knowledge of benchmark vs production.** No code path branches on a "we're in benchmark mode" boolean.
2. **No gate in the main loop.** `can_forward` is always True; `_executor_loop` proceeds as fast as it can.
3. **Request lifecycle is unchanged.** INIT / TRANS_IN_PROGRESS / TRANS_COMPLETE / GENERATION semantics are preserved.
4. **Dummy lifecycle is unchanged from the non-benchmark path.** Rank empties → insert dummy; rank fills → remove dummy. Nothing about this depends on a fill phase.
5. **Stuck-INIT detection runs unconditionally** (generalized from PR #12206). Produces a readable error with diagnostics regardless of whether we're benchmarking.
6. **Client owns measurement.** Any "wait until state X before starting the timer" logic lives on the client.

If any of (1)–(6) is violated in review, the design is wrong — go back and fix before merging.

---

## 10. What this design does NOT solve

This refactor removes a class of bug (server-side benchmark state machines). It does not address:

- **ADP router imbalance** itself. The router still produces ±several skew per distribution cycle. Nothing in v4 needs this to be balanced, so it's fine — but if the imbalance matters for production SLOs in the future (e.g. tail latency), it's a separate project.
- **CTX→GEN transfer stalls.** The 1-second CTX-side timeout still exists. If a deployment genuinely cannot complete transfers, the new watchdog surfaces it quickly, but the fix is CTX/transceiver-side.
- **Measurement methodology.** "What should the benchmark measure?" (from first token? from full admission? from submission?) is a separate, and arguably more important, question. This design just makes the mechanism match one choice cleanly; the choice itself is out of scope.
