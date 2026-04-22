# Regression Investigation — Post-PR-#12208 Hang

[< Back to index](README.md)

This document is the full root-cause analysis of the regression introduced by [PR #12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208). Read it in full before starting either step 1 or step 2 — the gate-rewrite plan assumes you have internalized the causal chain below.

---

## 1. Failing test and environment

**Test:**

```
perf/test_perf_sanity.py::test_e2e[disagg-gen_only-wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_con8192_ccb-NIXL]
```

**Similar failures:** nvbug 6071070 on K2.5 with matching signature.

**Commit window:**

| | Commit | Date | Test result |
|---|---|---|---|
| PASSED | `d71e8804fa` | 2026-04-10 | Pass |
| FAILED | `87299ffbda` | 2026-04-19 | All 8,192 requests fail |

Regression window = 124 commits. The AI pre-triage bisected to `d07dcd7588` = PR #12208.

**Environment:**
- Platform: GB200-LYRIS
- Docker: `urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.02-py3-sbsa-ubuntu24.04-trt10.15.1.29-skip-tritondevel-202604011104-12600`
- Wheel: `tensorrt_llm-1.3.0rc13-cp312-cp312-linux_aarch64.whl`

**Test config** (`tests/integration/defs/perf/disagg/test_configs/wideep/perf/kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_ccb-NIXL.yaml`):

| Setting | Value |
|---|---|
| `benchmark_req_queues_size` (from `TLLM_BENCHMARK_REQ_QUEUES_SIZE`) | 8192 |
| concurrency | 8192 |
| GEN `tensor_parallel_size` | 32 |
| GEN `enable_attention_dp` | true |
| GEN `max_batch_size` | 256 |
| GEN `max_num_tokens` | 256 |
| GEN `max_seq_len` | 9256 |
| GEN `kv_cache_config.free_gpu_memory_fraction` | 0.6 |
| GEN `moe_config.backend` | WIDEEP, `num_slots=416` |
| CTX servers | 8, each TP=4 |
| KV transfer backend | NIXL |
| Input / output length | 8192 / 1024 |

The critical boundary condition:

```
tp_size × max_batch_size  =  32 × 256  =  8192  =  benchmark_req_queues_size
```

---

## 2. Observed failure signature

From the bug report:

### Benchmark client

- `Total failed requests: 8192` / `Successful requests: 0`
- `UserWarning: All requests failed. This is likely due to a misconfiguration on the benchmark arguments.`
- Benchmark duration: 173.53 s

### GEN server logs (`trtllm-serve.GEN_0.0.log`)

- `tensorrt_llm.executor.utils.RequestError: Insufficient KV cache for gen-only benchmark mode: 37 request(s) are waiting for KV cache allocation but the scheduler could not fit any of them. Increase free_gpu_memory_fraction or reduce TLLM_BENCHMARK_REQ_QUEUES_SIZE (currently 8192).`
- Error repeated 16,384 times (8,192 requests × 2 tracebacks each)
- First error: `[04/19/2026-10:54:35]`
- GEN server does **not** crash — errors returned via the API

### DISAGG server logs (`trtllm-serve.DISAGG_SERVER.0.log`)

- No HTTP 5xx. Only 404s for `/energy_metrics` probes.
- `aiohttp ClientOSError / ServerDisconnectedError` on streaming responses — the client sees failed streams rather than 5xx.

### CTX server logs (`trtllm-serve.CTX_*.log`)

- Zero errors / tracebacks.
- `slurm-1520334.out` shows 320 occurrences of:
  `[W] num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache`
- Beginning at `[04/19/2026-10:54:01]`: `[WARNING] Timed out waiting for context KV cache transfer after 1000 milliseconds`

Interpretation: CTX completes prefill and tries to transfer KV to GEN, but GEN cannot admit the transfer within the 1 s timeout → CTX-side back-pressure, NOT CTX error.

### What the numbers tell us

The `37` in the GEN error is key: `len(stuck_init_requests) == 37`. This is the number of requests stuck in `DISAGG_GENERATION_INIT` state when the fail-fast check fired. 37 is far smaller than any rank-level or total capacity; it is an **imbalance residual**, not a capacity shortfall.

---

## 3. Code surface involved

All line numbers reference `tensorrt_llm/_torch/pyexecutor/py_executor.py` at commit `87299ffbda` unless otherwise noted.

### 3.1. Init

```python
# ~line 524
self.is_benchmark_disagg = (self.benchmark_req_queues_size > 0
                            and self.kv_cache_transceiver is not None)
self._benchmark_fill_phase_active = self.is_benchmark_disagg
```

Two state variables: `is_benchmark_disagg` (config, immutable) and `_benchmark_fill_phase_active` (runtime, transitions True → False exactly once).

### 3.2. The fill-complete predicate (introduced by PR #12208)

```python
# ~lines 1935–1972
def _is_benchmark_disagg_fill_complete(self, scheduled_batch) -> bool:
    ...
    local_gen_count = sum(1 for req in scheduled_batch.generation_requests
                          if not req.is_attention_dp_dummy)
    if self.enable_attention_dp:
        total_gen_count = sum(self.dist.tp_allgather(local_gen_count))
    else:
        total_gen_count = local_gen_count

    if total_gen_count >= self.benchmark_req_queues_size:
        return True
    return False
```

**Two consequential decisions:**

1. Counts `scheduled_batch.generation_requests` — i.e. what the scheduler picked for the next forward pass this iteration. Capped by `max_batch_size` per rank.
2. Excludes `is_attention_dp_dummy` — only **real** generation requests count.

### 3.3. The gate helper

```python
# ~lines 1974–2003
def _check_benchmark_disagg_gate(self, scheduled_batch, can_forward):
    if not self.is_warmup and not can_forward:
        can_forward = self._is_benchmark_disagg_fill_complete(scheduled_batch)
        if can_forward:
            self._benchmark_fill_phase_active = False   # (★) only place it's cleared
        else:
            time.sleep(0.1)
            return can_forward, True
    return can_forward, False
```

`_benchmark_fill_phase_active` transitions False **only** when `_is_benchmark_disagg_fill_complete` returns True. This is the load-bearing coupling.

### 3.4. Dummy suppression during fill

```python
# ~lines 3004–3033
def _should_skip_dummy_for_benchmark_disagg(self, num_schedulable_requests):
    if not self._benchmark_fill_phase_active or self.is_warmup:
        return False     # not in fill — normal dummy lifecycle applies
    return True          # in fill — skip insertion (would leak)
```

Called from `_pad_attention_dp_dummy_request` (~line 3035). When `True`, no dummy is inserted on an empty rank.

### 3.5. PR #12206's fail-fast (the thing that actually raises the error)

```python
# ~lines 1884–1907
if (self.benchmark_req_queues_size > 0 and not self.is_warmup
        and not fitting_disagg_gen_init_requests):
    stuck_init_requests = [
        req for req in self.active_requests
        if req.is_disagg_generation_init_state
    ]
    if (stuck_init_requests and self.num_fetch_requests
            >= self.benchmark_req_queues_size):
        error_msg = (
            f"Insufficient KV cache for gen-only benchmark mode: "
            f"{len(stuck_init_requests)} request(s) are waiting for "
            f"KV cache allocation ...")
        self._handle_errors(error_msg, requests=self.active_requests)
        return None, None
```

Predicate: `stuck_init_requests NOT empty  AND  num_fetch_requests ≥ threshold  AND  scheduler couldn't fit any new INIT request this iteration`.

**This predicate is what fires under the regression** — but it fires *incorrectly* in this case, because "couldn't fit any new INIT *this iteration*" is a transient condition (transfers in flight), not "KV cache truly insufficient."

### 3.6. Fetch path (what changed in behavior)

`_fetch_new_requests` → `_pop_from_waiting_queue` (~lines 2655–2738). Two caps matter:

```python
# ~line 2663
if self.enable_attention_dp:
    total_max = self.dist.tp_size * self.max_num_active_requests   # 32 × 256 = 8192
else:
    total_max = self.max_num_active_requests
max_new_requests = total_max - total_num_active_requests
```

Each `_fetch_new_requests` call pulls up to `max_new_requests` requests from the waiting queue in a single shot. In the NEW code path (post-PR #12208), this can be up to `total_max` on the first iteration.

### 3.7. ADP router

`DefaultADPRouter.route_requests` at `tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py:198`. Heap-based token-weighted distribution, capped at `expected_num_active_requests` per rank where:

```python
expected_num_active_requests = max(
    (total_num_active_requests + num_new_requests_all_ranks + tp_size - 1) // tp_size,
    max(all_ranks_num_active_requests),
)
```

**The balance is per-call, not global.** With requests arriving asynchronously from 8 CTX servers, per-call rounding produces ±1 skew that can persist.

---

## 4. Why the OLD code (pre-#12208) worked — and it was *not* dummies

Before PR #12208, `_prepare_and_schedule_batch` contained a batched fill loop that fetched exactly `tp_size` requests per outer-loop iteration:

```python
# Removed by PR #12208
batch_size = min(self.dist.tp_size if self.enable_attention_dp else 1,
                 self.benchmark_req_queues_size)
fill_target = min(self.num_fetch_requests + batch_size,
                  self.benchmark_req_queues_size)
while self.num_fetch_requests < fill_target:
    iter_requests = self._fetch_and_activate_new_requests()
    ...
    if self.num_fetch_requests < fill_target:
        time.sleep(0.1)
```

What this quietly did: on each call, it handed the ADP router exactly `tp_size = 32` new requests with ~32 ranks. `expected_num_active_requests = ceil(32 / 32) = 1`, so the router distributed **exactly one request to each rank, every single call**. Repeat 256 times and every rank has exactly 256 real requests **by construction** — the batching rhythm was enforcing perfect balance.

PR #12208 removed the batched loop. `_fetch_new_requests` now pulls whatever is enqueued at that moment (up to 8192), and the router runs once against a much less controlled input. The ±1 skew per call accumulates across iterations instead of resetting every 32 requests. The design doc for PR #12208 did not notice this because the doc frames the router as unchanged (it was — the invariant the router *relied on* changed).

**Important:** the OLD code's dummy insertion (`num_active_request == 0` guard) was *not* what made balance work. A rank with 255 real requests would not have gotten a dummy in OLD code either. It's the batched fetch rhythm, not the dummy logic, that made the old gate always satisfiable.

---

## 5. Why the NEW code hangs

### 5.1. The imbalance

With 8192 requests arriving from 8 CTX servers and the router running once against the bulk, the final per-rank distribution settles with small skew. A representative failure shape:

```
31 ranks × 256 real requests + 1 rank × 255 real requests = 8191
```

(The actual failure could also have a rank with 257 and several at 255 — the error counts show 37 stuck INIT, meaning several ranks overshot. See §6 for how overshoot happens even with the router cap.)

### 5.2. The circular dependency that freezes the fill phase

```
(1) Gate opens only if   total_real_gen_count ≥ 8192
(2) Dummies suppressed   while _benchmark_fill_phase_active is True
(3) _benchmark_fill_phase_active clears only when gate opens

If distribution is 255+256+256+...+256:
    total_real_gen_count = 8191 < 8192  →  gate does NOT open
    →  _benchmark_fill_phase_active stays True
    →  dummies stay suppressed on the underfilled rank
    →  total_real_gen_count stays at 8191
    →  goto gate check
```

No exit condition other than "real count hits threshold." There is no "we've fetched everything and we're waiting too long" branch. The fill phase is stuck forever in the sense of its own state machine.

### 5.3. The trigger — PR #12206's fail-fast mercy-kills it

While the gate is stuck, `num_fetch_requests` has long since reached 8192. On any iteration where the scheduler finds it cannot fit more INIT requests (because KV cache is being held by the already-TRANS_COMPLETE 8155+), the PR #12206 fail-fast predicate matches:

```
stuck_init_requests (the 37 overflow reqs on overshooting ranks)  ≠ []
num_fetch_requests (8192) ≥ benchmark_req_queues_size (8192)
fitting_disagg_gen_init_requests (this iteration) = []
```

→ `_handle_errors` is called with all `active_requests` (8192 of them) → every client sees a `RequestError` → benchmark reports all failed.

### 5.4. The CTX-side `1000ms` timeout confirms the picture

CTX servers try to push KV to GEN but cannot because GEN's KV cache is saturated (holding the already-transferred reqs). Transfers time out at 1 s and retry. This is back-pressure, not CTX error — the log pattern (`Timed out waiting for context KV cache transfer after 1000 milliseconds`, no tracebacks) matches exactly this scenario.

---

## 6. Where the 37 overflow requests come from

The ADP router caps per-rank requests at `expected_num_active_requests`, which adapts as more requests are routed. It is **not** a hard per-rank cap of 256 across the whole fill phase — it's a per-call computed target. Concretely:

- Iter 1: pull `N1` requests, compute `expected = ceil(N1 / 32)`, distribute to all ranks up to that per-rank ceiling.
- Iter 2: pull `N2` more, compute new `expected` based on updated totals, distribute — but the heap already has asymmetric starting counts from iter 1.

Over many iterations, some ranks can end up above the nominal 256 even while others are below 256 — the `max(...)` in the `expected_num_active_requests` formula explicitly allows this:

```python
expected_num_active_requests = max(
    (total + new + tp_size - 1) // tp_size,   # target (256-ish)
    max(all_ranks_num_active_requests),        # floor = current max
)
```

So final state can be, e.g., `{256, 257, 258, ..., 255, 254, ...}`. The 37 overflow counted in the error are on one or more ranks that pushed past 256, and their excess cannot be admitted because per-rank KV allocator is full. The remainder make it, but the gate still cannot reach 8192 because the under-filled ranks contribute < 256.

This compounds the failure: not only is balance ±1 off, it can be ±several, and the overflow is what *actually* manifests as `stuck_init_requests` in the fail-fast path.

---

## 7. Why neither "count dummies" nor "insert dummies during fill" alone would fix it

| Fix attempt | Why it fails |
|---|---|
| Revert gate to count `is_attention_dp_dummy` as well (like OLD) | During fill, `_should_skip_dummy_for_benchmark_disagg` suppresses insertion. No dummies to count. No effect. |
| Revert dummy insertion to happen during fill (like OLD) | OLD code only inserted dummies on ranks with `num_active_request == 0`. A rank with 255 real reqs is *not* empty, so no dummy is inserted. No effect. (And if you relax that guard to "insert whenever rank < 256", dummies inserted mid-fill never terminate — you reintroduce the leak #12208 was closing.) |
| Revert both together | Same as above: if the insertion guard still requires `num_active_request == 0`, nothing changes for the 255-of-256 case. |
| Rely on the ADP router to produce exact balance | It doesn't, and no per-call adjustment inside the router fixes the global skew without changing the router contract. |

None of these address the actual issue: **the gate-complete condition is a coincidence condition, not a causation condition**. It was satisfiable only because the OLD fill-batching rhythm forced exact balance; with that removed, a fundamentally different criterion is needed. That is the step-1 plan in `03-step1-gate-rewrite-plan.md`.

---

## 8. Summary — the invariants that were never explicit

PR #12208 locally-correctly changed two things:

1. Gate criterion: `count of REAL (non-dummy) gen_requests ≥ threshold`.
2. Dummy lifecycle: `do not insert during fill to avoid leaks`.

Both assume — but do not state or check — a third invariant:

3. **ADP router produces a distribution in which every rank has at least `threshold / tp_size` real requests, exactly.**

Invariant (3) was silently enforced by pre-#12208 batched fill. With batched fill removed, (3) becomes unenforceable in general. The consequence is that (1) becomes unsatisfiable, (2) makes (1) unrecoverable, and PR #12206's fail-fast — designed for a different failure mode (true KV insufficiency) — now fires on the transient state instead.

The step-1 plan replaces (1) with a criterion that does not depend on (3), and restructures (2) accordingly. The step-2 plan removes (1) from `PyExecutor` entirely.
