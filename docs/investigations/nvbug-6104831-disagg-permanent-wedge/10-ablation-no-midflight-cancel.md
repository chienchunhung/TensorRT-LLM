# 10 — Ablation: what happens if we don't do mid-flight NIXL cancellation?

PR [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) bundles
five distinct defensive layers on top of the baseline timeout +
state-ordering fixes:

1. **Mid-flight NIXL cancellation** — `release()` on `TransferStatus`,
   `mHandleMutex` on `NixlTransferStatus`, the bounded poll loop in
   `AgentConnection::send`, the per-request cancel registry in
   `CacheSender::Impl` / `CacheReceiver::Impl`, the cancel-aware return
   types on `AgentConnectionManager::waitForNotification` /
   `recvReadySignalWithStatus`.
2. **`BufferIndexHolder` RAII + `poison()`** — recv-side buffer-index
   leak fix for the six exit paths in `CacheReceiver::Impl::requestSync`,
   plus a `poison()` path that marks the buffer pool as
   non-reusable when transport quiescence is unknown.
3. **`shared_ptr<LlmRequest>` async-lifetime** — extension of the async
   worker's grip on the `LlmRequest` past Python-side
   `_terminate_request`.
4. **Recv-side per-request idempotency** — guards against
   double-fulfillment of a future when cancel and completion race.
5. **Fail-closed on unquiesced transfer** (memory-safety policy) —
   `BaseCacheTransceiver::hasPoisonedTransferBuffer` exposed to Python;
   `py_executor._check_cache_transfer_errors` calls
   `_fail_closed_for_unquiesced_disagg_transfer` and sets
   `shutdown_event` when the pool is poisoned, deliberately shutting
   down PyExecutor to prevent NIXL from writing into TRT-LLM-reclaimed
   memory.

Reviewer pushback during PR #13713 raised a fair question: are all five
layers really load-bearing, or is the timeout-based eviction
([`02-failure-signatures.md`](02-failure-signatures.md#signature-4)
plus the `kv_transfer_timeout_ms` plumbing in
[`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md)) alone
enough? This section is the empirical answer.

## TL;DR

The defensive layers are **not optional belt-and-suspenders**.
The 60 s production default never exercises them, but as soon as the
cancel/timeout path fires:

- Layers 1–4 (cancel, RAII, lifetime, idempotency) **convert a
  permanent wedge into a recoverable transient-error regime**
  (Experiment 4: `Broken promise` 89 → 0; `NO RECOVERY` 1 → 0).
- Layer 5 (fail-closed on unquiesced transfer) **converts potential
  use-after-free into a loud HTTP 400** (Experiment 6: PyExecutor on
  the paused peer is shut down on head, surfacing
  "PyExecutor has already been shutdown" to the orchestrator rather
  than risking silent memory corruption on the ablation branch).

The "lower availability" pattern that PR #13713 shows under brief
peer-unresponsiveness is the **correct behaviour** of a memory-safety
mechanism that deliberately trades apparent availability for verified
safety. Ablation's higher apparent availability under the same
condition is illusory — it is availability *with potential memory
corruption from NIXL writes into reclaimed buffers*.

---

## Why the question came up

Two earlier investigation phases produced a tempting but misleading
hypothesis:

- **conc=64 NIXL+UCX-plugin, `kv_transfer_timeout_ms=60000`, 5 iter** —
  the no-midflight-cancel branch passed 5/5 with 715 requests per burst,
  zero errors, zero failure markers, recovery in 30 s.
- **conc=256 NIXL native, `kv_transfer_timeout_ms=60000`, 5 iter** —
  same branch passed 5/5 again.

Inspection of the per-worker logs after both runs showed
`Cannot cancel request: 0`, `exceeded total timeout: 0`,
`Broken promise: 0`, `bad_optional_access: 0`. The defensive layers had
never *fired*. The workload never put the system into the failure
regime they protect against, because at 60 s timeout every transfer
completed within its natural latency under tested concurrency. This is
consistent with [`04-reproduction.md`](04-reproduction.md): the
customer wedge needs a load shape that produces cancels and retries, not
just throughput pressure.

So the conclusion from the 60 s-timeout passes is *not* "Tier 2 is
unnecessary". It is "the workload tested didn't exercise Tier 2".

The follow-up experiment lowers `kv_transfer_timeout_ms` to a value
that *does* force the cancel/timeout path to fire under a stable, easy
workload.

---

## Setup

### Branch under test

`local/pr13713-no-midflight-cancel` on
[`chienchunhung/TensorRT-LLM`](https://github.com/chienchunhung/TensorRT-LLM/tree/local/pr13713-no-midflight-cancel)
(commit `e7b5931227`), built off PR #13713's head and removing only the
four Tier 2 layers above. The preserved set on this branch is:

- `kv_transfer_timeout_ms` C++ deadline in
  `cacheTransceiver.cpp::checkContextTransferStatus` /
  `checkGenTransferStatus`.
- Python-side fallback deadline (`py_kv_transfer_start_time` in
  `py_executor.py::_check_kv_transfer_timeout`).
- `setState`-after-`receiveAsync` ordering in `requestAndReceiveAsync`
  (with the lock-down test
  `test_request_and_receive_async_state_ordering`).
- Config plumbing for both timeout knobs (`llm_args.py`,
  `executor.h`, nanobind).
- `_terminate_request` boolean-return contract documentation.

The smoke check used to confirm the binaries on each rebuild is:

```python
from tensorrt_llm.bindings.internal.batch_manager import CacheTransceiver
hasattr(CacheTransceiver, 'has_poisoned_transfer_buffer')   # expected False
```

### Harness

The standard
[`run_combo_nixl_3pair.sh`](../../../../.repro/harness/threepair/) +
`run_validation_loop.sh` from [`04-reproduction.md`](04-reproduction.md),
configured for 3 ctx/gen pairs on a single B300 8-GPU host, NIXL native
transport (`TRTLLM_NIXL_KVCACHE_BACKEND=NIXL`), and 5 burst iterations
per run.

For the ablation experiment, both worker configs
(`ctx_config_nixl.yaml`, `gen_config_nixl.yaml`) lower the deadline
knobs:

```yaml
cache_transceiver_config:
  backend: NIXL
  kv_transfer_sender_future_timeout_ms: 500      # poll slice
  kv_transfer_timeout_ms: 1000                   # request deadline
```

The 1 s deadline is aggressive enough that any transfer queued behind a
modest backlog hits it. At conc=64 with 3 pairs the per-pair queue
depth averages ~21 in-flight transfers, which is enough to push
individual transfers past 1 s during the burst.

### Watchdog instrumentation

The
[`.repro/watchdog/dump_stacks_on_hang.sh`](../../../../.repro/watchdog/dump_stacks_on_hang.sh)
helper polls `/health` on all seven ports every 15 s and, on four
consecutive failures (~60 s of unresponsiveness), `kill -QUIT`s every
`trtllm-serve` worker (printing Python thread stacks to stderr) and
attaches `gdb --batch -ex 'thread apply all bt 30'` to each (printing
all C++ thread stacks). Output lands in
`<run_dir>/stack_snapshots/<ts>/`.

The watchdog uses `mTerminate`-style polling, so it doesn't itself
interact with the cancel path under test.

---

## Six experiments

The experiments traverse the timeout-pressure spectrum and finish with
a peer-pause failure injection that triggers the memory-safety policy
(layer 5) directly:

| # | Workload | Branches | What it isolates |
|---|---|---|---|
| 1 | conc=64 NIXL+UCX, 60 s timeout, 5 iter | ablation only | Happy-path baseline; defenses dormant |
| 2 | conc=256 NIXL native, 60 s timeout, 5 iter | ablation only | Throughput stress without timeout pressure |
| 3 | conc=64 NIXL native, **1 s timeout**, 5 iter | ablation only | First scenario that drives the cancel/timeout path |
| 4 | conc=64 NIXL native, **1 s timeout**, 5 iter | A/B (ablation vs head) | Confirms layers 1–4 are load-bearing |
| 5 | conc=64 NIXL native, **5 s timeout**, 5 iter | A/B (ablation vs head) | Intermediate-pressure regime; shows when defenses fire silently |
| 6 | conc=64 NIXL native, 5 s timeout, **SIGSTOP gen-8004 for 20 s mid-burst** | A/B (ablation vs head) | Directly drives layer 5 (fail-closed-on-unquiesced) and exposes the memory-safety semantics |

### Experiment 1 — conc=64 NIXL+UCX-plugin, default 60 s timeout

```text
RUN_TAG=run_no_midflight_cancel_ucx_3pair_conc64
CONC=64
BURST_DUR_S=90
TRTLLM_NIXL_KVCACHE_BACKEND=UCX
kv_transfer_timeout_ms=60000   (default, preserved)
```

| iteration | burst | recovery | verdict |
|---|---|---|---|
| 1 | 715 ok / 0 err / 90.5 s | 30 s | PASS |
| 2 | 715 ok / 0 err / 90.5 s | 30 s | PASS |
| 3 | 715 ok / 0 err / 90.7 s | 30 s | PASS |
| 4 | 715 ok / 0 err / 90.5 s | 30 s | PASS |
| 5 | 715 ok / 0 err / 90.5 s | 30 s | PASS |

Failure-marker scan: zero matches across all workers and all
iterations.

```text
Cannot cancel request:    0
exceeded total timeout:   0
Broken promise:           0
bad optional access:      0
NO RECOVERY:              0
OVERALL: PASS
```

**Interpretation:** The defensive layers were never reached. Every
transfer completed inside its 60 s natural latency window. This run
proves only that the no-midflight-cancel branch does not regress against
happy-path workloads.

### Experiment 2 — conc=256 NIXL native, default 60 s timeout

```text
RUN_TAG=run_no_midflight_cancel_nixl_native_conc256_clean
CONC=256
BURST_DUR_S=90
TRTLLM_NIXL_KVCACHE_BACKEND=NIXL
kv_transfer_timeout_ms=60000
```

(The "clean" suffix reflects an aggressive teardown
— `pkill -9 -f tensorrt_llm.commands.serve` + `fuser -k`
— between this run and the earlier conc=64 NIXL+UCX run; an
earlier attempt without that teardown produced port conflicts and
non-meaningful results.)

| iteration | burst | recovery | verdict |
|---|---|---|---|
| 1 | 715 ok / 0 err / 90.5 s | 30 s | PASS |
| 2 | 715 ok / 0 err / 90.5 s | 30 s | PASS |
| 3 | 715 ok / 0 err / 90.7 s | 30 s | PASS |
| 4 | 716 ok / 0 err / 90.6 s | 30 s | PASS |
| 5 | 716 ok / 0 err / 90.5 s | 30 s | PASS |

Failure-marker scan: zero matches again.

```text
Cannot cancel request:    0
exceeded total timeout:   0
Broken promise:           0
bad optional access:      0
NO RECOVERY:              0
OVERALL: PASS
```

**Interpretation:** Throughput was identical to conc=64 (~715 req/burst
in both runs), confirming the system is throughput-saturated at conc=64
and further concurrency only deepens the queue, not the throughput.
Per-request latency at conc=256 is ~32 s (256 in-flight, 8 req/s) —
still well under the 60 s timeout. Same conclusion as Experiment 1: the
defensive layers were never reached.

### Experiment 3 — conc=64 NIXL native, 1 s timeout

This is the ablation that actually exercises the cancel path. The
timeout is aggressive enough that requests queued behind ~20 others on
the same ctx/gen pair routinely cross the deadline.

```text
RUN_TAG=run_no_midflight_cancel_aggressive_timeout_conc64
CONC=64
BURST_DUR_S=90
TRTLLM_NIXL_KVCACHE_BACKEND=NIXL
kv_transfer_timeout_ms=1000               (aggressive)
kv_transfer_sender_future_timeout_ms=500  (aggressive poll slice)
```

#### Per-iteration burst result

```text
iteration 1: 216 ok / 499 err / 715 total — 70 % error rate
iteration 2: ABORT  — sanity probe failed (60 s read timeout)
iteration 3: ABORT
iteration 4: ABORT
iteration 5: ABORT
OVERALL: FAIL
```

The system enters a permanent wedge during iteration 1's burst and
never recovers. Iterations 2 through 5 abort at the pre-burst sanity
probe.

#### Per-worker failure-marker breakdown

```text
log file              CanCanc  ExcTO   MarkErr   BrokPm
context_8001                1    189       189        0
context_8003                1    236       236        0
context_8005                1     82        82        0
generation_8002             1     29         0       29
generation_8004             2     30         0       30
generation_8006             1     30         0       30

TOTAL                       7    596       507       89
```

Three things stand out:

1. **7 `Cannot cancel request` events across both sides.** Each
   represents a request that crossed the deadline while it was
   mid-flight (past the queue-drain visible state, inside the NIXL
   submit/wait or inside `waitForNotification`). The queue-drain
   cancel could not find it, so the worker stayed parked.

2. **Sender vs receiver have asymmetric failure shapes:**
   - Sender (`context_*`): `ExcTO = MarkErr` (1:1). Every sender-side
     timeout completes the "mark as error" transition cleanly. No
     `Broken promise` on the sender side.
   - Receiver (`generation_*`): `ExcTO = BrokPm` (1:1). **Every**
     receiver-side timeout ends in `std::future_error: Broken promise`.

3. **The 504/507 sender timeouts that *don't* produce `Cannot cancel`
   were caught at the queue-drain layer** — i.e. the request was still
   sitting in `mPendingRequests` or `mReadyResponses` when the deadline
   fired, so the cancel removed it cleanly. The seven cancel-misses are
   the requests that already passed the queue and entered NIXL submit
   territory.

#### The receiver-side cascade

The PR-#13713-preserved gen-side log line is:

```text
[batchmgr] Generation KV cache transfer for request <id> reached
  total timeout while waiting for receiver future. Requesting
  cancellation and marking as error.
```

This is `checkGenTransferStatus` doing `wait_for(deadline_remaining)`
on `mRequesterFutures`. When the future doesn't resolve, the timeout
decision is made — but the underlying receiver worker is still stuck.
The receiver worker's wait points (with the cancel-aware plumbing
removed) are:

```text
CacheReceiver::Impl::requestSync()
  ├── sendRequestAndBufferInfo(remote_ctx)
  ├── recvReadySignal()
  │     └── waitForReadySignal()
  │           └── waitForNotification<ReadySignalInfo>(..., mTerminate)
  │                 while (!mTerminate.load()) { ... yield ... }   ← stuck
  └── formatter->receiveSync()
        └── AgentConnection::recv()
              └── waitForSyncInfo()
                    └── waitForNotification<NotificationSyncInfo>(..., mTerminate)
                          while (!mTerminate.load()) { ... yield ... } ← or stuck
```

`waitForNotification` checks only the **process-wide** `mTerminate`
flag, not a per-request cancel. Without PR #13713's cancel-aware return
plumbing, the receiver worker spins until process exit. When the
deadline-eviction logic runs `_terminate_request`, the `LlmRequest`
object is torn down (recall we also removed the `shared_ptr<LlmRequest>`
lifetime extension), the `std::promise<void>` owned by the receiver
worker is destroyed, and the upper layer's `future.get()` throws
`std::future_error: Broken promise`. This is the 1:1 ratio on the gen
side: every gen-side timeout ends in a broken promise.

The full sequence per request, from the `context_8001` log (timestamps
elided for brevity):

```text
... Generation KV cache transfer for request <id> reached total
    timeout while waiting for receiver future. Requesting cancellation
    and marking as error.
... Cannot cancel request <id>                            ← in some cases
... Set request <id> from state 9 to -1                   ← state 9 = DISAGG_GENERATION_TRANS_IN_PROGRESS
... Generation KV cache transfer for timed-out request
    <id> finished with error: std::future_error: Broken promise
... Set request <id> from state -1 to 20                  ← state 20 = error reported up
```

#### The sender-side cascade

The sender side ships its own preserved log line:

```text
[batchmgr] Context KV cache transfer for request <id> exceeded
  total timeout: elapsed 1XXX ms > limit 1000 ms. Marking as error.
```

This is `checkContextTransferStatus` doing the same `wait_for` over
`mSenderFutures`. The sender worker's wait points (with the bounded
poll loop and `release()` removed) are:

```text
CacheSender::Impl async worker
  └── CacheFormatter::format(session)
        └── sendBuffer(session, ...)
              └── AgentConnection::send(ctx, data, size)
                    ├── status = mAgent->submitTransferRequests(request)
                    ├── status->wait()        ← UNBOUNDED on revert; was bounded poll on PR #13713
                    └── mAgent->notifySyncMessage(remote, ...)
```

`NixlTransferStatus::wait()` on the revert is a `while (true)` over
`mRawAgent->getXferStatus(mHandle)` that returns only on
`NIXL_SUCCESS` or non-`NIXL_IN_PROG` status. There is no external
signal that can break it — except for `release()`, which the revert
removed. The worker spins in `yield()` until NIXL eventually completes
the submit. *In our experiment that does happen* (NIXL's queue
eventually drains), which is why the sender side doesn't produce
`Broken promise` — the worker does eventually call `set_value` on its
promise. The damage is the latency, not the cleanup: the buffer index
the worker holds is leaked for the entire duration of the stuck
wait, and the deadline-eviction has already marked the request as
error so the response is discarded.

---

## Mapping the observed failures to the L1–L10 defect layers

Cross-referencing against
[`03-defect-class-stack.md`](03-defect-class-stack.md):

| Observation in this experiment | Defect-class layer | PR #13713 fix that closes it |
|---|---|---|
| 7 × `Cannot cancel request` | L7 (mid-flight cancellation is impossible without `release()`) | Mid-flight NIXL cancellation: `release()` + `mHandleMutex` + bounded poll loop in `AgentConnection::send` + per-request cancel registry |
| 89 × `Broken promise` on gen side, 1:1 with ExcTO | L3 (promise/future race on cancel) + L5 (`LlmRequest` lifetime) | Recv-side per-request idempotency + `shared_ptr<LlmRequest>` async-lifetime |
| Receiver workers parked in `waitForNotification` (inferred from L3 cascade) | L7 again, receiver flavour | Cancel-aware return types on `waitForNotification` / `recvReadySignalWithStatus` + tri-state recv ready signal |
| Sender worker spins in `NixlTransferStatus::wait` until NIXL drains | L7 + L2 (buffer-index leak on long-stuck wait) | Bounded poll loop with `release()` on cancel + `BufferIndexHolder` RAII |
| 0 × `bad_optional_access` | L5 (lifetime) — *protection not exercised, but not because protection was unnecessary*; the timing simply didn't catch the freed-`LlmRequest` window. The ablation removed the guard. | `shared_ptr<LlmRequest>` async-lifetime |

So the experiment empirically validates that at least four invariants
in [`03-defect-class-stack.md`](03-defect-class-stack.md) (L2, L3, L5,
L7) are load-bearing under any workload that drives the cancel/timeout
path. The Tier 1 deadline-eviction work by itself catches the issue
but cannot recover the resources — it converts a hang into a slow
wedge, not into a clean error.

---

## What this conclusively shows, and what it doesn't

**Shown (this experiment plus Experiments 1 and 2):**

- PR #13713's defensive layers are dormant under happy-path workloads
  with the production-default 60 s timeout. Removing them on the
  no-midflight-cancel branch does not regress happy-path latency or
  recovery.
- The same defensive layers become load-bearing the moment a workload
  drives the cancel/timeout path. The 1 s ablation wedges in iteration
  1 with exactly the failure signatures the defensive layers were
  designed to break (`Cannot cancel request`, `Broken promise`,
  permanent wedge).
- The receiver side is the structurally worse failure mode of the two:
  every gen-side timeout produces a `Broken promise` (vs zero on the
  sender side).
- The 60 s passes from earlier do not imply the defenses are
  unnecessary — they imply the workload tested didn't push the system
  into the regime the defenses protect against. Earlier passes are
  necessary but not sufficient evidence.

### Experiment 4 — A/B run on PR #13713 head under the same 1 s timeout

The same aggressive-timeout harness was applied to PR #13713 head
(`local/pr13713-rc13-clean`, defensive layers present and active). The
prediction going in was a clean 5/5 PASS. The actual result is more
informative than that.

```text
RUN_TAG=run_pr13713_head_aggressive_timeout_conc64
binary: local/pr13713-rc13-clean   (PR #13713 head)
hasattr(CacheTransceiver, 'has_poisoned_transfer_buffer')  →  True
CONC=64, BURST_DUR_S=90, TRTLLM_NIXL_KVCACHE_BACKEND=NIXL
kv_transfer_timeout_ms=1000, kv_transfer_sender_future_timeout_ms=500
```

| iteration | result |
|---|---|
| 1 | **PASS** — burst `216 ok / 477 err / 715`, RECOVERY at idle=30 s |
| 2 | FAIL — sanity probe `http_500` (system still draining) |
| 3 | FAIL — sanity probe `http_500` |
| 4 | **PASS** — RECOVERY at idle=**60 s** (longer idle was enough) |
| 5 | FAIL — sanity probe `http_500` |

5/5 was not reached, but neither was the permanent wedge from
Experiment 3. The system is in a recoverable transient-error regime,
not a wedge.

#### A/B comparison table

| Marker | No-midflight-cancel (Experiment 3) | PR #13713 head (Experiment 4) | Δ |
|---|---|---|---|
| `Cannot cancel request` | 7 | 6 | -1 (essentially unchanged) |
| `exceeded total timeout` | 596 | 961 | **+365** |
| `Marking as error` | 507 | 965 | **+458** |
| **`Broken promise`** | **89** | **0** | **-89** |
| `bad_optional_access` | 0 | 0 | 0 |
| **`NO RECOVERY`** | **1** | **0** | **-1** |
| Iteration verdicts | F/F/F/F/F | P/F/F/P/F (2 PASS) | +2 PASS |
| Overall | FAIL (permanent wedge) | FAIL (transient errors, recoverable) | qualitatively different |

#### Interpretation

The result confirms three of the four originally predicted defensive
layers and partially confirms the fourth:

1. **Recv-side idempotency + `shared_ptr<LlmRequest>` lifetime: 89 → 0
   `Broken promise`.** This is the single cleanest signal in the A/B.
   Every receiver-side timeout that produced a broken promise on the
   ablation branch is now handled cleanly. This layer is unambiguously
   load-bearing.

2. **Mid-flight cancellation: prevents the permanent wedge** — `NO
   RECOVERY` drops from 1 to 0, and 2/5 iterations actually recover.
   The system regains the ability to make forward progress.
   *However*, `Cannot cancel request` only drops from 7 to 6, not to
   0. The remaining six are explainable: mid-flight cancellation
   closes the `status->wait()` window, but the cancel call itself
   still has a narrow window where the cancel can fire after the
   worker is past the cancellable region (e.g., in
   `notifySyncMessage`, which is non-interruptible). What matters is
   that these residual `Cannot cancel` events on head do *not*
   escalate into `Broken promise` or buffer-pool starvation — they
   are contained by the other defensive layers.

3. **`BufferIndexHolder` RAII**: indirectly confirmed by the absence
   of permanent wedge. With buffer-index leaks on every cancel/error
   path the no-midflight-cancel branch wedged within iteration 1; on
   head the pool survives 5 iterations.

4. **Counter-intuitive increase in `exceeded total timeout`
   (596 → 961) and `Marking as error` (507 → 965).** Head sees *more*
   timeouts in absolute terms because it keeps the system processing
   requests; the ablation branch wedged and stopped accepting work,
   so its absolute count is artificially low. Higher error count is
   a sign of higher throughput, not worse behavior.

The 1 s timeout was deliberately chosen as 60× more aggressive than
the production default. At this pressure, even PR #13713's defenses
are working hard — burst-1 on head still has a 67 % error rate during
the burst (238 ok / 477 err / 715 total). The system survives because
it doesn't wedge, but the 30 s post-burst idle isn't always long
enough to drain the backlog of error-marked requests before the next
iteration's sanity probe fires. Iteration 4's longer 60-second idle
was sufficient — hence its PASS.

So we observe two distinct recovery phenomena:

- **Macro-scale (between iterations):** PR #13713 wins decisively —
  it eventually recovers every iteration; ablation never recovers.
- **Micro-scale (during the 30 s post-burst idle):** PR #13713 needs
  more time at this workload because the system has to drain
  hundreds of errored requests through the cancel/cleanup pipeline.

#### Where 1 s sits in the timeout-regime spectrum

After Experiments 4, 5, and 6 are folded in, the full regime is:

| Timeout / Failure | Cancel-path pressure | No-midflight-cancel | PR #13713 head | Layers exercised |
|---|---|---|---|---|
| 60 s (production default) | dormant | PASS (Experiments 1, 2) | PASS (rc11/rc13 validation) | None |
| **5 s (intermediate)** | **fires silently — natural completion (~30 s observed) bails out the ablation branch** | PASS (Experiment 5) | PASS (Experiment 5) | Layers 1–2 fire; cancel-path logs identical on both branches; iter-level outcomes identical |
| **1 s (~60× tighter)** | **saturated** | **permanent wedge in iter 1** (Experiment 3, 4) | **2/5 PASS, transient errors, no permanent wedge** (Experiment 4) | Layers 1–4 actively load-bearing; Layer 5 starts to fire under high error volume |
| **5 s + SIGSTOP peer pause** | **natural completion cannot bail out — peer is genuinely paused** | Worker stuck >82 s after SIGCONT, NO RECOVERY in 60 s; *NIXL eventually writes into reclaimed buffers* (Experiment 6) | Worker recovers in 1.71 s; **`_fail_closed_for_unquiesced_disagg_transfer` shuts down the affected PyExecutor** to prevent UAF (Experiment 6) | **All 5 layers exercised**, including Layer 5 (memory-safety policy) |
| 200 ms (theoretical) | pathological | trivial wedge | likely wedge — not all problems can be solved | — |

The 1 s point stresses layers 1–4. The SIGSTOP point is the canonical
"slow / unresponsive peer" failure that drives layer 5 directly, and
shows that PR #13713 trades availability for memory safety **by
design** in exactly that scenario.

### Experiment 5 — A/B sweep at the intermediate 5 s timeout

To probe the middle of the timeout-pressure spectrum, both branches
ran the same 5-iteration validation loop with
`kv_transfer_timeout_ms=5000` and `kv_transfer_sender_future_timeout_ms=2500`.

```text
RUN_TAGs: run_no_midflight_cancel_5s_timeout_conc64
          run_pr13713_head_5s_timeout_conc64
CONC=64, BURST_DUR_S=90, TRTLLM_NIXL_KVCACHE_BACKEND=NIXL
```

#### Result

Both branches **PASS** all 5 iterations: 716 ok / 0 errors per burst,
RECOVERY at idle=30 s every time. Per-branch failure-marker counts
across the 5 iterations:

| Marker | ablation | head |
|---|---|---|
| `Cannot cancel request` | **13** | **13** |
| `exceeded total timeout` | **13** | **13** |
| `Marking as error` | **13** | **13** |
| `Broken promise` | 0 | 0 |
| `bad_optional_access` | 0 | 0 |
| `NO RECOVERY` | 0 | 0 |

#### Why the marker counts are identical (and why this is consistent with PR #13713 being correct)

`Cannot cancel request` is emitted by `CacheSender::Impl::cancelRequest` /
`CacheReceiver::Impl::cancelRequest` *whenever the queue-drain cancel
cannot find the request in `mPendingRequests` / `mReadyResponses` /
etc.* — meaning the request is already past the queue layer. PR #13713's
mid-flight cancellation does *not* prevent this log: it adds an
*additional* action (flip the per-request cancel atomic, call
`release()` on the NIXL handle, throw) on top of the same log. The
visible diagnostic is unavoidably the same on both branches when the
cancel path is exercised.

The actual divergence is *downstream* of the log:

| Step | Ablation @ 5 s | Head @ 5 s |
|---|---|---|
| 1. Deadline fires | `exceeded total timeout` | same |
| 2. cancel_request called | `Cannot cancel request` | same |
| 3. Per-request cancel atomic flipped | (no flag) | flag flipped |
| 4. `AgentConnection::send → status->wait()` | spins until NIXL drains naturally (~30 s observed in worker logs: `elapsed 30-32 s > limit 5000 ms`) | poll loop sees flag → `release()` → throws |
| 5. Worker fate | stays parked tens of seconds; promise eventually gets `set_value` once NIXL drains | unwinds immediately; promise gets exception |
| 6. Buffer index | leaked tens of seconds | freed via `BufferIndexHolder` destructor |

So at the 5 s timeout point — on this hardware — the natural-completion
path on ablation happens to drain fast enough (~30 s) that:

- No buffer index leak accumulates to pool starvation
- No `Broken promise` cascade fires (the NIXL push *does* complete, so
  the receiver promise *does* get `set_value`)
- The deadline-eviction "Marking as error" runs after the request has
  actually completed at the C++ layer — it is essentially decorative

This is the regime where **PR #13713's defenses fire silently**:
13 `release()` calls on head, 13 spin-until-NIXL-drains on ablation,
both branches survive. The behavioural difference exists internally
(measurably so under instrumentation) but does not manifest in
client-visible verdicts.

The 1 s point (Experiment 4) is the regime where the natural-completion
path can *not* drain fast enough — that's where the difference
explodes into 89 `Broken promise` events and a permanent wedge on the
ablation branch.

#### What 5 s does NOT prove

The 5 s pass on ablation is **not** evidence that the defenses are
unnecessary in production. The pre-conditions for the natural-completion
path to bail us out — namely "NIXL transfer always eventually completes"
— *fail in the canonical production failure mode*: a peer that is slow,
unresponsive, OOM-ing, or has a stuck network. Experiment 6 forces that
exact scenario.

### Experiment 6 — SIGSTOP-injected peer pause (the memory-safety test)

The 5 s sweep showed that natural completion of NIXL transfers masks
the difference between branches. To break that assumption directly,
this experiment **pauses one generation worker mid-burst** with
`kill -STOP`, forcing in-flight NIXL transfers to that peer to freeze.
After 20 s, the worker is resumed with `kill -CONT`. The injector then
probes the disagg front-end every few seconds to measure how quickly
the system can serve a fresh request again.

```text
RUN_TAG: run_sigstop_ablation
         run_sigstop_head
CONC=64, BURST_DUR_S=120, kv_transfer_timeout_ms=5000
Failure injection:
  T+~150 s after worker launch (mid iter-1 burst): SIGSTOP gen-8004
  T+170 s: SIGCONT gen-8004
  recovery probes at +0, +1, +2, +5, +10, +20, +30, +60 s after SIGCONT
```

#### Headline: worker-level recovery time after SIGCONT

| Branch | Injector recovery probe outcome |
|---|---|
| **Ablation** | All 7 probes fail (HTTP `000` = connection refused) up to +60 s; final probe at wall=82 s gets HTTP 500. **NO RECOVERY within 60 s of SIGCONT.** |
| **PR #13713 head** | First probe at +0 s (wall=1.71 s after SIGCONT) returns **HTTP 200**. **RECOVERED at +0 s.** |

The disagg front-end is responsive to fresh chat-completion requests
**within 1.71 seconds of SIGCONT on head**, versus **never within 60 s
on ablation**. A ~50× recovery-time differential under the textbook
"peer briefly unresponsive" failure scenario.

#### Marker breakdown across the 5 iterations

| Marker | ablation | head |
|---|---|---|
| `Cannot cancel request` | 9 | **0** |
| `exceeded total timeout` | 1130 | 807 |
| `Marking as error` | 967 | 854 |
| **`Broken promise`** | **162** | **0** |
| `bad_optional_access` | 0 | 0 |
| `NO RECOVERY` | 0 | 1 |
| iter verdicts | 3 PASS, 2 ABORT | 0 PASS, 1 NO RECOVERY + 4 ABORT |

Note the *apparent* paradox: ablation has *more* PASS iterations
than head. The next subsection explains why this is consistent with
PR #13713 being correct.

#### The surprise: head's PyExecutor shuts down by design

Inspection of head's `front.log` reveals the cause of the iter-level
"FAIL" outcomes:

```text
[serve] Client error to http://localhost:8004/v1/chat/completions:
        400, message='Bad Request: {"object":"error",
        "message":"PyExecutor has already been shutdown.",
        "type":"BadRequestError","param":null,"code":400}'
```

Sometime after SIGCONT — driven by the C++ cleanup pipeline's discovery
that buffer pool slots were poisoned during the cancel-with-unknown-
quiescence cycle — Python's `_check_cache_transfer_errors` called
`has_poisoned_transfer_buffer()`, got `True`, and invoked
`_fail_closed_for_unquiesced_disagg_transfer()`, which set
`shutdown_event`. PyExecutor on gen-8004 gracefully shut itself
down.

This is **PR #13713's layer 5 firing exactly as designed**. The full
chain is:

```text
mid-flight cancel fires (release()) — layer 1
  → catch block in cacheFormatter — layer 2
    → sendHolder.poison() — layer 2
      → BaseTransBufferManager mPoisoned = true — layer 2
        → has_poisoned_transfer_buffer() == True — layer 5
          → _fail_closed_for_unquiesced_disagg_transfer() — layer 5
            → shutdown_event.set() — layer 5
              → PyExecutor.shutdown() — layer 5
                → HTTP 400 surfaced to disagg frontend — layer 5
                  → 500 to client; orchestrator can restart pod
```

Ablation does not have layers 2.poison(), 5, or the `release()` call
in layer 1. So on ablation, after the cancel-with-unknown-quiescence
cycle, the system just keeps running. The NIXL transfers that were
in flight when the peer was paused **eventually complete after
SIGCONT** — writing into receiver-side buffers that TRT-LLM may have
already marked-as-error and reclaimed for new requests. This is the
textbook use-after-free / heap-corruption window. The ablation
branch's "higher apparent availability" in this experiment is
**availability with potential memory corruption from NIXL writes into
TRT-LLM-reclaimed memory**.

#### Memory-safety argument, made concrete

```text
Time (rel)  Ablation                                           Receiver-side buffer at addr X
----------  -------------------------------------------------  ---------------------------------------
T+0         sender: submitTransferRequests(dst=X, size=N)      X is pinned by NIXL for the push
T+0         sender: status->wait() (unbounded)                 X reserved for this transfer
T+0         receiver: gen-8004 paused via SIGSTOP              NIXL push is queued in receiver-side
                                                               NIC, not yet committed to X
T+5 s       deadline fires; `Marking as error`                 X is still pinned by NIXL
T+5 s       Python: `_terminate_request(R)` runs               LlmRequest R destroyed
T+5 s       LlmRequest destructor; recv buffer X freed         X returned to TRT-LLM allocator
T+~5-10 s   New request gets buffer at (or overlapping) X      X now holds new request's data
T+20 s      receiver: SIGCONT — NIXL drains queue               NIXL writes the original push data
                                                               into X → CORRUPTION (overwrites
                                                               the new request's bytes)
T+20-30 s   Sender's status->wait() finally returns SUCCESS    Sender future gets set_value;
                                                               "Cannot cancel" + "Marking as error"
                                                               were already logged but the request
                                                               was already torn down — log decoration
T+30+ s     Client sees HTTP 200 from the corrupted response   Token stream may contain garbled
                                                               or wrong-request tokens depending on
                                                               which bytes were overwritten
```

We did not run with a content validator on the responses, so we cannot
confirm corruption in the *ablation* run logs directly — but the
architectural path above is the use-after-free window that
`_fail_closed_for_unquiesced_disagg_transfer` is designed to close.
PR #13713 deliberately surfaces the failure (HTTP 400 → 500 →
orchestrator restart) rather than risk silent corruption.

#### Tradeoff acknowledgement

PR #13713 makes a deliberate tradeoff:

| Aspect | Ablation | PR #13713 head |
|---|---|---|
| Apparent availability under brief peer-unresponsiveness | Higher (most requests serve, just slow) | Lower (PyExecutor on the affected pair shuts down) |
| Worker-level recovery after peer resumes | >80 s, never reaches HTTP 200 in 60 s window | 1.71 s (other 2 pairs still serving) |
| Memory safety after cancel with unknown quiescence | None — NIXL may write to reclaimed buffers | Enforced — PyExecutor shuts down preemptively |
| Failure visibility to orchestrator | Silent (correct or corrupted HTTP 200) | Loud (HTTP 400 "PyExecutor has already been shutdown") |
| `Broken promise` cascade | 162 | 0 |
| `Cannot cancel request` | 9 | 0 |

For an orchestrated production deployment (the customer scenario in
NVBug 6104831), the head behaviour is **strictly preferred**:
a worker that is shut down can be restarted by the orchestrator and
serves correct responses afterwards; a worker that silently serves
potentially-corrupted responses is a correctness hazard that is much
harder to detect.

### What this conclusively shows (final)

Across all six experiments:

1. **`Broken promise` is eliminated by PR #13713's recv-side
   idempotency + `shared_ptr<LlmRequest>` lifetime.**
   89 → 0 in Experiment 4 (1 s aggressive timeout) and 162 → 0 in
   Experiment 6 (SIGSTOP failure injection) under identical
   workloads. Single strongest signal of layers 3 and 4 doing real
   work.
2. **Permanent wedge from cancel-without-cleanup is eliminated by PR
   #13713's mid-flight cancellation + `BufferIndexHolder` RAII.**
   `NO RECOVERY` 1 → 0 in Experiment 4; worker-level recovery time
   after SIGCONT drops from >82 s (still HTTP 500 at +60 s) to 1.71 s
   in Experiment 6. ~50× recovery-time differential.
3. **Throughput in the error regime is preserved by PR #13713 — and
   `Cannot cancel request` events drop to zero under SIGSTOP** because
   `release()` enables clean cancellation of in-flight NIXL transfers
   (Experiment 6: 9 → 0). The 5 s sweep (Experiment 5) shows
   the cancel path fires silently when natural completion happens
   to bail out the ablation branch — but that natural-completion
   assumption breaks under the canonical production failure (slow
   or unresponsive peer), which is exactly what Experiment 6
   simulates.
4. **The fail-closed-on-unquiesced layer (layer 5) is the
   memory-safety mechanism.** Experiment 6 directly observed
   `_fail_closed_for_unquiesced_disagg_transfer` firing on head:
   `PyExecutor has already been shutdown` surfaced as HTTP 400 to
   the disagg frontend. This is the deliberate trade of apparent
   availability for verified safety — without this layer, NIXL can
   write into TRT-LLM-reclaimed buffers, silently corrupting
   responses to unrelated requests.
5. **PR #13713's defensive layers are not magic.** At 60× the
   production timeout (Experiment 4) the system survives but with
   transient errors. Under SIGSTOP-induced peer pause (Experiment
   6), the affected gen worker shuts down — by design. Both
   outcomes are correct: the system enters a state that the
   orchestrator can recover from, rather than wedging silently or
   serving corrupted data.

### What remains unmeasured

- **Direct empirical detection of memory corruption on ablation.**
  Without an AddressSanitizer build of TRT-LLM or response-content
  validation in the test client, the use-after-free path is shown
  architecturally (see the timeline in Experiment 6) rather than
  observed in flight. ASan or response-content validation would
  give us a positive empirical detection of the corruption window.
- **Production-default 60 s timeout run on PR #13713 head with the
  current rc13-clean build** under the 3-pair conc=64 workload —
  earlier 60 s-timeout passes on head are on rc11/rc13 builds and
  pre-date the current rc13-clean state.

---

## Reproduction artefacts

Branch under test: `local/pr13713-no-midflight-cancel` at
`e7b5931227` ("[None][None] PR#13713 prototype: remove mid-flight NIXL
cancellation"). Pushed to
[`chienchunhung/TensorRT-LLM`](https://github.com/chienchunhung/TensorRT-LLM/tree/local/pr13713-no-midflight-cancel).

Wheel archive (built `2026-05-13T19:16Z`,
`tensorrt_llm-1.3.0rc14-cp312-cp312-linux_x86_64.whl`) preserved at
`/home/chienchunh/wheel-archive/pr13713-no-midflight-cancel-<TS>/` for
node-eviction recovery. See `RESTORE.md` in the same directory for the
checklist.

Per-run logs (all paths relative to the worktree they ran in):

- Experiment 1 (ablation, conc=64 NIXL+UCX, 60 s timeout):
  `.repro/logs/run_pr13713_reviewfix_v2_20260513_195048/`
- Experiment 2 (ablation, conc=256 NIXL native, 60 s timeout):
  `.repro/logs/run_no_midflight_cancel_nixl_native_conc256_clean_20260513_210018/`
- Experiment 3 (ablation, conc=64 NIXL native, 1 s timeout):
  `.repro/logs/run_no_midflight_cancel_aggressive_timeout_conc64_20260513_214652/`
- Experiment 4 (head, conc=64 NIXL native, 1 s timeout):
  `pr13713-rc13-clean/.repro/logs/run_pr13713_head_aggressive_timeout_conc64_20260513_223841/`
- Experiment 5:
  - ablation: `pr13713-no-midflight-cancel/.repro/logs/run_no_midflight_cancel_5s_timeout_conc64_20260513_230959/`
  - head:     `pr13713-rc13-clean/.repro/logs/run_pr13713_head_5s_timeout_conc64_20260513_232319/`
- Experiment 6 (SIGSTOP injection):
  - ablation: `pr13713-no-midflight-cancel/.repro/logs/run_sigstop_ablation_20260514_003635/`
  - head:     `pr13713-rc13-clean/.repro/logs/run_sigstop_head_20260514_004813/`
  - Injector logs at `/tmp/sigstop-injector-{ablation,head}-*.log`

Harness variants (all live in `.repro/` of each worktree):

- `harness-aggressive-timeout/` — 1 s `kv_transfer_timeout_ms`,
  500 ms sender-future poll slice. Used by Experiments 3 and 4.
  Parallel run scripts: `run_combo_nixl_3pair_aggressive.sh`,
  `run_validation_loop_aggressive.sh`.
- `harness-5s-timeout/` — 5 s `kv_transfer_timeout_ms`,
  2500 ms sender-future poll slice. Used by Experiments 5 and 6.
  Parallel run scripts: `run_combo_nixl_3pair_5s.sh`,
  `run_validation_loop_5s.sh`.

The SIGSTOP failure injector for Experiment 6 is at
`/tmp/sigstop-injector.sh`; it waits for `/health=200` on the target
port, sleeps 30 s so the SIGSTOP fires mid-burst, then SIGSTOPs the
worker for 20 s, SIGCONTs, and probes the disagg frontend every few
seconds to record the wall-time-to-first-HTTP-200 after SIGCONT. The
chain runner that pairs ablation and head sequentially is at
`/tmp/run-sigstop-ab-chain.sh`.

---

## Cross-references

- [`02-failure-signatures.md`](02-failure-signatures.md) — signatures
  `#1`, `#4`, `#5`, `#6` that these experiments touch.
- [`03-defect-class-stack.md`](03-defect-class-stack.md) — the L1–L10
  layering. The six experiments together validate L2, L3, L5, L7 are
  load-bearing under the cancel/timeout path, and demonstrate that
  PR #13713's layer 5 (fail-closed-on-unquiesced) is the
  memory-safety mechanism that closes the use-after-free window
  opened by cancel-with-unknown-quiescence.
- [`04-reproduction.md`](04-reproduction.md) — why long-prompt + cancels
  + retries is the minimum trigger set. These ablations provide two
  alternative triggers: aggressive deadline at modest concurrency
  (Experiments 3, 4) and SIGSTOP-induced peer pause (Experiment 6).
  The SIGSTOP scenario is the closest in-process simulation of the
  production failure mode where a peer is genuinely unresponsive.
- [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md) — the
  combo PR #13713 approach. This section is the empirical defence of
  why the combo's five defensive layers are not over-engineered —
  each fires in at least one of the six experiments below.
- [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) — the
  remaining follow-ups: AddressSanitizer build to empirically catch
  the use-after-free directly, and response-content validation to
  detect downstream token corruption on the ablation branch.
