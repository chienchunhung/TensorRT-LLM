# Finding — Fill-Phase Stuck-State Deadlock (Latent)

[< Back to index](README.md)

**Status:** **Latent bug, separate from PR #13347's v2.1 + v2.2 fixes.** Documented here because it was uncovered while validating v2.2 locally.
**Severity:** Affects benchmark-mode runs where the scheduler cannot fit all in-flight INIT requests after the per-rank capacity is reached. In practice this means: any deployment whose KV-cache pool is undersized relative to `max_batch_size × max_seq_len × tp_size`.
**Bug references:** TBD (file as a separate NVBug)

---

## 1. Symptom

Reproduced in a local 8-GPU B200 run of:

- `disagg-gen_only-gb200_deepseek-v32-fp4_8k1k_con1024_ctx1_dep4_gen1_dep4_bs256_fill-pressure_ccb-UCX`
- 4-rank CTX, 4-rank GEN, attention-DP, `max_batch_size=256`, `concurrency=1024 = tp_size × max_batch_size`

Run timeline:

| t | CTX | GEN |
|---|---|---|
| 0:00 | Loading model | Loading model |
| 6:00 | Started serving | Started serving |
| 8:00 | Receiving 1024 prefill requests | Admitting requests; gate closed (fill phase) |
| 16:00 | **All 1024 prefills complete** (`200 OK × 1024`) | Per-rank state: `256/256 active, 144 schedulable` — gate still closed |
| 65:00 | Idle for 49 minutes | **Still stuck**: identical per-rank state, gate still closed, 30,321 `Skipped adding dummy requests` iterations |

The benchmark client sees `0/1024` completed throughout. GPU memory steady at ~92 % of B200 budget. No errors logged. Both before-fix (no v2.2 patch) and after-fix (v2.2 patch applied) runs hit the **identical** plateau, confirming the deadlock is independent of v2.2.

---

## 2. Diagnosis

The state-based gate predicate (v2.1) requires:

1. `num_fetched_requests >= benchmark_req_queues_size` — satisfied (1024 fetched).
2. **All active requests on every rank are not in `DISAGG_GENERATION_INIT` or `DISAGG_GENERATION_TRANS_IN_PROGRESS`** — *not* satisfied: each rank has 256 active requests of which 144 are schedulable and 112 remain in INIT/TRANS.
3. No inflight transceiver sessions — likely also not satisfied.

So the gate keeps polling forever. Forward never executes, no decode happens, no slot frees, and the 112 INIT requests per rank cannot make progress past INIT.

The puzzle is that **CTX completed all 1024 prefill HTTP responses 49 minutes earlier**, so from the CTX's perspective every KV transfer should already be initiated or completed. Yet GEN reports 112 per rank still in INIT.

The most likely mechanism (not yet root-caused with code-level evidence):

- The transceiver only handles a small number of concurrent receives per rank (bounded by `max_tokens_in_buffer / per_request_kv_size`). With `max_tokens_in_buffer=16384` and 8 K input length, that's ~2 concurrent transfers per rank.
- 256 INIT requests per rank are admitted at once (or at `tp_size` per iteration with v2.2). Only ~2 are actively transferring; the remaining 254/rank are queued in the transceiver's pending list.
- Buffer rotation (releasing a transfer's buffer once the receive completes, then assigning it to the next pending transfer) appears to require either: (a) periodic transceiver polling that is starved by the tight `time.sleep(0.1)` outer-loop polling pattern, or (b) some signal that only fires from a forward-step path that the gate is currently preventing.

This is consistent with the design assumption that "during fill, transfers complete in the background and the gate eventually opens" — but the assumption breaks when transfer buffer rotation is itself coupled to the forward path that the gate suppresses.

---

## 3. Why this is independent of PR #13347 and v2.2

| Patch | Affects gate predicate? | Affects admission rate? | Affects transceiver buffer rotation? |
|---|---|---|---|
| v2.1 state-based gate | ✓ | ✗ | ✗ |
| v2.1 ADP router cap | ✗ (admits ≤ max_batch_size per rank) | ✗ | ✗ |
| v2.1 fail-fast suppression | ✗ | ✗ | ✗ |
| v2.2 admission cap | ✗ | ✓ (slows burst) | ✗ |

The deadlock can only be fixed at the transceiver layer (decoupling buffer rotation from forward steps) or at the gate layer (firing the gate even when some INIT requests remain, on the grounds that the system can make progress on the schedulable subset). Neither is in scope for the v2.1+v2.2 patch series.

---

## 4. Why it doesn't manifest in the original Kimi failure

The Kimi failure (`dep32 × bs256 == con8192` on GB200) **OOMs before reaching this deadlock**. The burst-admission memory spike fires the OS OOM-killer within the first second of the fill phase, long before the transceiver buffer rotation issue would have a chance to manifest. Once v2.2 prevents the burst-OOM, the Kimi setup will likely either:

- Complete normally, because Kimi's `cache_transceiver_config` (`max_tokens_in_buffer=8448`) and per-request KV size in the production config happen to keep ≥ all-active concurrent transfers within transceiver capacity, and the deadlock is a function of the *ratio* of admitted requests to transceiver capacity rather than absolute counts; or
- Exhibit this deadlock too, in which case it would surface as a benchmark-side timeout rather than an OOM.

CI validation of the original Kimi config (post-v2.2) will determine which of these holds. Either way, the v2.2 patch is necessary to even reach that determination — currently the run dies before any meaningful fill-phase behavior is observable.

---

## 5. Repro instructions

```bash
cd <worktree>
LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
bash tests/integration/defs/perf/run_local_disagg_perf_sanity.sh \
  --test-case 'disagg-gen_only-gb200_deepseek-v32-fp4_8k1k_con1024_ctx1_dep4_gen1_dep4_bs256_fill-pressure_ccb-UCX' \
  --output-dir /tmp/repro \
  --ctx-gpus 0,1,2,3 --gen-gpus 4,5,6,7 --control-gpus 0 \
  --ctx-ranks 1 --gen-ranks 1
```

Observe in `/tmp/repro/local_GEN_0.log`:

- `Skipped adding dummy requests: num_fetch_requests=1024, num_schedulable_requests=144` — repeats indefinitely.
- `currank_total_requests = 256/1024` — never advances.
- `host_step_time = 101 ms` — confirms the executor is in the gate's `time.sleep(0.1)` loop, not doing forward.

Observe in `/tmp/repro/local_CTX_0.log`:

- 1024 lines of `INFO: ... POST /v1/completions HTTP/1.1 200 OK`.
- `iter` log shows CTX stops after request 1024 is processed.

Observe in `/tmp/repro/local_BENCHMARK.log`:

- `Benchmarking: 0/1024 [00:00<?, ?it/s]` — frozen.

---

## 6. Recommended next steps

1. **File as a separate NVBug.** Title: "PyExecutor benchmark-disagg fill phase deadlocks when admitted requests exceed transceiver concurrent-receive capacity."
2. **Instrument GEN's transceiver state.** Add per-iteration counters for: requests in INIT, requests in TRANS_IN_PROGRESS, requests in TRANS_COMPLETE, transceiver active sessions, transceiver pending sessions, transceiver completed sessions. This will confirm whether the 112 stuck per rank are in INIT or TRANS_IN_PROGRESS.
3. **Test the transceiver-decoupled hypothesis.** Add a forced periodic flush of the transceiver state in the gate-closed loop, separately from the forward path. If the deadlock clears, the buffer rotation is indeed coupled to forward.
4. **Consider the v4 redesign** (`04-step2-external-orchestrator-plan.md`). Moving fill orchestration to the client eliminates the in-executor gate and the entire class of "gate × X" interaction bugs.
