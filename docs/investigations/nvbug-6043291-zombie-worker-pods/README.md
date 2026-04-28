# NVBug 6043291: Zombie Worker Pods After CUDA Engine Crash

- **Severity:** P0 / Critical
- **Reported by:** Astra (customer)
- **Affected model:** gpt-oss-120b
- **Date:** 2026-03-18
- **Branch:** `fix-zombie-worker-health-check`
- **PRs:** [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718)
  (fatal engine / health detection),
  [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119)
  (request-level error propagation in disaggregated serving)
- **Status:** In review — all reviewer comments addressed, squashed to 1 commit

---

## Executive Summary

Two worker pods suffered fatal CUDA crashes but remained registered in service
discovery, silently black-holing all routed requests. The Dynamo runtime (Rust)
stayed alive after the TRT-LLM engine (Python/C++) crashed, so pods continued
accepting TCP connections and incrementing `dynamo_component_inflight_requests`
but never produced tokens. The frontend queue grew at ~30 req/min with no
recovery until replacement pods came online ~1 hour later.

While the root routing failure is a Dynamo-side issue (it should detect dead
engines and stop routing), TRT-LLM must add defensive mechanisms to prevent
zombie processes.  Two TensorRT-LLM fixes now cover complementary parts of the
failure surface:

- **PR #12718**: system-level detection and handling.  Fatal engine / worker
  failures are promoted into executor health, `/health` returns unhealthy, and
  the serving process exits so the pod restarts.
- **PR #13119**: request-level propagation.  Per-request or postprocessing
  failures keep their real error messages through `GenerationResult`,
  postprocessing, OpenAI/disaggregated HTTP clients, and response formatting so
  callers see the real failure instead of a malformed response or generic
  `400 Bad Request`.

Together they establish the core split used throughout this investigation:
**request failed → return the real error; engine died → mark unhealthy and
restart.**

---

## Root Cause Analysis

### The Crash

The crash was caused by a **CUDA OOM error** during bursty traffic spikes with
the 120B model. The model nearly saturates GPU memory, so under extreme spikes
the KV cache runs out. A CUDA allocation fails and the engine hits a fatal CUDA
error. The engine catches the exception but transitions into a state where the
Python process stays alive while the C++ engine underneath is dead.

### Why Health Checks Didn't Catch It

The health check call chain:

```
GET /health (openai_server.py:584)
  -> OpenAIServer._check_health() (openai_server.py:431-437)
    -> BaseLLM._check_health() (llm.py:946-955)
      -> GenerationExecutor.is_shutdown() (executor.py:298-299)
        -> returns self.doing_shutdown  # <-- Python bool, only set on intentional shutdown
```

**`doing_shutdown` is only set during intentional shutdown flows** (e.g.,
`pre_shutdown()`, explicit `shutdown()` calls). It is **never set when the
engine crashes**.

### The Error Queue Gap

When an MPI worker crashes:
1. The MPI future completes with an exception
2. `mpi_done_callback` (proxy.py:229-234) puts the exception into `_error_queue`
3. **Nobody drains `_error_queue` from the health check path**
4. `_handle_background_error()` is only called during `generate()` calls or
   shutdown
5. Health returns 200 because `doing_shutdown` is still `False`

### The PyExecutor Gap

In `_torch/pyexecutor/py_executor.py`, the `_handle_errors()` method
(line 3344):
- Fails only the **current batch** of active requests
- Does **not** set any fatal/shutdown state on the executor
- The executor loop continues to the next iteration
- If the CUDA context is corrupted, it loops forever failing every batch but
  never shuts down

### The Request-Path Workaround (Insufficient)

Serving endpoints catch `CppExecutorError` and call
`signal.raise_signal(signal.SIGINT)` (e.g., openai_server.py:1031-1034). But
this only triggers when a request actually **hits the dead engine**. If health
checks keep returning 200 and the orchestrator keeps routing, requests may queue
up but the error path may not trigger quickly enough.

### The `/health_generate` Endpoint

A deeper health check exists at `/health_generate` (openai_server.py:594) that
runs an actual generation. This **would** catch the zombie state but:
- It's at a different URL path
- Kubernetes probes don't typically hit it
- It's not what Dynamo or standard k8s liveness probes use

---

## Affected Components

| Component | File | Issue |
|-----------|------|-------|
| Health endpoint (HTTP) | `tensorrt_llm/serve/openai_server.py` | Returns 200 when engine is dead |
| Health endpoint (gRPC) | `tensorrt_llm/grpc/grpc_request_manager.py` | Same: only checks `is_shutdown()` |
| LLM health check | `tensorrt_llm/llmapi/llm.py` | Delegates to `is_shutdown()` only |
| Executor base | `tensorrt_llm/executor/executor.py` | No fatal error state; `is_shutdown()` only checks `doing_shutdown` |
| Executor proxy | `tensorrt_llm/executor/proxy.py` | No MPI worker liveness monitoring |
| PyExecutor | `tensorrt_llm/_torch/pyexecutor/py_executor.py` | `_handle_errors()` never triggers shutdown |

---

## Fix: Defense-in-Depth Strategy

Four independent layers ensure that a crashed engine results in pod termination,
regardless of which detection mechanism fires first.

### Layer 1: `check_health()` on `GenerationExecutor`

**File:** `tensorrt_llm/executor/executor.py`

- Add `_fatal_error: Optional[BaseException]` field to track unrecoverable errors
- Add `_set_fatal_error(error)`: records the first fatal error (first-error-wins)
- Add `check_health() -> bool`: returns `False` if `doing_shutdown` or
  `_fatal_error` is set; drains `_error_queue` directly via `get_nowait()`
  (not via `_handle_background_error()` which is documented for main-thread
  use and would cause re-entrancy issues from health-check / event-loop threads)
- Per-request errors (`RequestError`, `str`) in the queue are skipped — a single
  bad request should not crash the server
- Modify `_handle_background_error()`: call `_set_fatal_error(error)` before
  `self.shutdown()` for serious errors; the queue drain path also filters
  `RequestError`/`str` to avoid poisoning `_fatal_error` from per-request errors
- Update `is_shutdown()` to also return `True` when `_fatal_error` is set

**Why this helps:** Health probes now detect errors that were sitting in the
queue unprocessed, and return 503 so the orchestrator stops routing traffic.

### Layer 2: MPI Worker Liveness Check + Background Monitor

**File:** `tensorrt_llm/executor/proxy.py`

- Extract shared `_check_mpi_futures()` and `_drain_error_queue()` helpers,
  used by both `check_health()` and `_error_monitor_loop()` to avoid code
  duplication
- Both helpers use `pre_shutdown()` (non-blocking, not `shutdown()` which
  blocks on `f.result()`) and drain-all patterns (all queued errors processed
  in one call, so a fatal behind per-request errors is detected immediately)
- Per-request errors (`RequestError`, `str`) are skipped in the drain
- Fix `pre_shutdown()` sentinel condition:
  `all(not f.done())` → `not self.mpi_futures or any(not f.done())`.  The
  empty-list branch is required for `RemoteMpiCommSessionClient` /
  `trtllm-llmapi-launch` where workers run in a separate `mgmn_leader_node`
  process and no local future handles exist; the `any(...)` branch keeps the
  partial-crash fix so surviving workers still get the quit signal when one has
  died
- Join `_error_monitor_thread` during `shutdown()` with a 5-second timeout,
  guarded by `threading.current_thread() is not self._error_monitor_thread`
  to prevent a self-join deadlock when the monitor thread initiates shutdown

**Why this helps:** Even if no health checks or `generate()` calls arrive, the
monitor thread auto-detects worker crash within ~5 seconds and shuts down.

### Layer 3: Fatal Error Detection in PyExecutor

**Files:** `tensorrt_llm/_torch/pyexecutor/error_classification.py` (standalone,
no CUDA/C++ dependencies) and `tensorrt_llm/_torch/pyexecutor/py_executor.py`

Three-tier error classification with token-bucket error budget:

| Tier | Patterns | Behavior |
|------|----------|----------|
| **Immediate fatal** | `cudaErrorIllegalAddress`, `cudaErrorLaunchFailure`, `illegal memory access`, `device-side assert`, `unrecoverable` | Crash on first occurrence — CUDA context is corrupted |
| **Severe** | `CUDA out of memory`, `CUDA error` (excluding illegal memory access), `NCCL error` | Costs 5× budget (0.5) per error — two rapid OOMs crash, one recoverable OOM is tolerated |
| **Transient** | Everything else | Costs 1× budget (0.1) per error — ~10 rapid errors before crash |

The `ErrorBudget` dataclass (`error_classification.py`) encapsulates the
token-bucket state:
- `budget`: starts at 1.0, capped at 1.0
- `cost`: 0.1 per transient error, 0.5 (5×) per severe error
- `recovery_rate`: 0.1 per second of error-free wall time
- `consume(error_msg)`: classifies and deducts; returns True if exhausted
- Immediate-fatal errors bypass the budget entirely

**Design rationale:** A simple consecutive counter (crash after N errors) was
replaced because it couldn't distinguish between "10 transient errors over an
hour" (fine) and "10 errors in 100ms" (engine is broken).  CUDA OOM is
classified as severe (not immediate-fatal) because the CUDA context remains
valid after a failed allocation — the engine can recover if the next batch
is smaller.

`_handle_errors()` accepts a `charge_budget` flag (default `True`).  Request-
scoped call sites pass `charge_budget=False`:
- `_validate_request()` failures (parameter/format errors)
- `_check_cache_transfer_errors()` (KV-transfer errors)
- KV cache transfer timeout in `_handle_responses()`
- `_handle_guided_decoder_errors()` (guided decoding failures)

These per-request errors only fail the affected request and are propagated
back to the client — they don't consume the error budget or affect server
health.  System-level call sites (forward, decode, sample, hang detector)
keep the default `True`.

On the fatal path, `_handle_errors()` also:
- Sets `is_shutdown = True` immediately (prevents the executor loop from
  scheduling more requests on a corrupted CUDA context)
- Copies `active_requests` to a local list before calling `clear()` (fixes an
  aliased-list bug where `_terminate_request` never ran because the list was
  emptied first)
- Drains both `waiting_queue` and `executor_request_queue` so queued-but-not-
  yet-activated requests are failed immediately

**Why this helps:** PyExecutor stops looping forever on corrupted CUDA context.
It self-terminates instead of silently failing every batch.

### Layer 4: Process Self-Kill from Health Endpoint

**File:** `tensorrt_llm/serve/openai_server.py`

- In `health()`: when `_check_health()` returns `False`, check if the
  executor's `_fatal_error` is set. If so, raise `signal.SIGINT` to trigger
  uvicorn shutdown (same established pattern as `CppExecutorError` handlers)

**Why this helps:** Final backstop ensuring the process terminates and the pod
restarts, even if the orchestrator keeps polling health.

### Supporting Changes

- **`tensorrt_llm/llmapi/llm.py`**: Change `_check_health()` to call
  `self._executor.check_health()` instead of `not self._executor.is_shutdown()`
- **`tensorrt_llm/grpc/grpc_request_manager.py`**: Change `health_check()` to
  use `check_health()` and report a sanitized `_fatal_error` summary
  (`"TypeName: first line"`) to gRPC clients; full error logged server-side
- **`tests/unittest/llmapi/apps/_test_openai_metrics.py`**: Update existing
  `test_health` to patch `check_health()` instead of `is_shutdown()` to match
  the new delegation path

---

## Related Fix: PR #13119 (Request-Level Error Propagation)

PR #12718 and PR #13119 address adjacent failure layers, not duplicate code.

| Layer | PR #12718 | PR #13119 |
|---|---|---|
| Scope | Executor / worker / pod health | Individual request / disaggregated serving response |
| Question answered | "Is this engine fatally broken and should the pod restart?" | "If this request failed, can the caller see the real error?" |
| Main primitives | `_fatal_error`, `check_health()`, `_error_monitor_loop`, `ErrorBudget`, health-endpoint `SIGINT` | `GenerationResultBase.error`, `ErrorResponse` from postprocessing, preserved HTTP error bodies, disagg ID regeneration |
| Expected outcome | Pod exits and restarts | Request fails with the real error; server can stay healthy |

PR #13119 fixes several paths where a request-level error was being lost or
mutated:

- `GenerationResultBase` now stores `_error_msg` and exposes `result.error`.
  When a response has `has_error()`, it records the error and returns early
  instead of falling through to `response.result`.
- `PostprocWorker` catches postprocessing exceptions and emits `ErrorResponse`
  instead of crashing the postprocess worker.
- `OpenAIServer` checks `promise.error` / `response.error` before formatting
  chat/completion responses.
- `OpenAIHttpClient` preserves HTTP response bodies for non-2xx disaggregated
  calls instead of reducing everything to a generic `raise_for_status()`
  message.
- Disaggregated retries regenerate `disagg_request_id` to avoid worker-side
  ID collisions, and `_verify_ctx_response()` messages now include
  `finish_reason`, `disagg_request_id`, and `ctx_request_id`.

The key interaction is that PR #13119 makes request errors more explicit, while
PR #12718 must decide whether an error is request-scoped or process-fatal.
That is why PR #12718 filters `RequestError` / `str` in executor queue drains
and uses `_handle_errors(..., charge_budget=False)` for validation,
KV-transfer timeout, guided-decoder, and cache-transfer request paths.  Without
that separation, PR #13119's improved propagation could accidentally become a
server-crash trigger under a burst of bad client requests.

In short:

```text
PR #13119: request failed  -> preserve and return the real error
PR #12718: engine died     -> mark unhealthy and restart the pod
```

---

## Detection Timeline (After Fix)

```
T+0s    CUDA OOM crash in C++ executor / MPI worker dies
T+0s    MPI future completes with exception -> _error_queue populated
T+0-5s  _error_monitor_loop detects dead future -> _set_fatal_error + shutdown
T+next  GET /health -> check_health() returns False -> 503
T+next  Health endpoint raises SIGINT -> uvicorn shutdown -> pod terminates
T+k8s   Kubernetes detects unhealthy pod -> restarts
```

For PyExecutor (in-process, immediate-fatal error):
```
T+0s    cudaErrorIllegalAddress in _forward_step() -> _handle_errors()
T+0s    ErrorBudget.consume() -> classify_error() returns "immediate_fatal"
T+0s    _fatal_error set, enqueue_shutdown_request() called
T+0s    Executor loop breaks, shutdown proceeds
```

For PyExecutor (in-process, severe error / budget exhaustion):
```
T+0s    CUDA OOM in _forward_step() -> _handle_errors()
T+0s    ErrorBudget.consume() -> "severe" -> budget -= 0.5 (budget=0.5)
T+0s    Engine retries next iteration (budget > 0)
T+0s    CUDA OOM again -> budget -= 0.5 (budget=0.0)
T+0s    Budget exhausted -> _fatal_error set, shutdown
```

---

## Test Coverage

All unit tests are in
`tests/unittest/executor/test_fatal_error_health_check.py` (74 tests total,
heavily parametrized).  Tests use the **real** `classify_error()` function and
`ErrorBudget` dataclass imported from `error_classification.py` via `importlib`
(avoids C++ extension loading).

| Test class | Count | What's covered |
|---|---|---|
| `TestClassifyError` | 15 | Real `classify_error()`: three-tier classification, case insensitivity |
| `TestErrorBudget` | 11 | Real `ErrorBudget` dataclass: immediate-fatal bypass, severe/transient exhaustion, time recovery, aliased-list fix, `is_shutdown` set on fatal, `waiting_queue` drain, `executor_request_queue` drain, `charge_budget=False` skips budget / never triggers fatal |
| `TestGenerationExecutor` | 10 | `_set_fatal_error` first-wins, `is_shutdown` (4 states), `check_health` drain-all with per-request skip |
| `TestProxyCheckHealth` | 6 | MPI future states via shared `_check_mpi_futures`/`_drain_error_queue` helpers |
| `TestPreShutdownSentinel` | 6 | Empty-`mpi_futures` / `RemoteMpiCommSessionClient` sentinel regression, all-alive, all-done, partial-crash, idempotency, workers-not-started |
| `TestErrorMonitorLoop` | 4 | Worker crash, error queue, per-request string skip, shutdown flag |
| `TestGrpcHealthCheck` | 5 | Parametrized: healthy, fatal, no executor, no LLM, shutdown |
| `TestOpenAIHealthEndpoint` | 3 | Parametrized: 200, 503, 503+SIGINT |
| `TestBaseLLMCheckHealth` | 4 | Parametrized delegation |

### Issues Found During Development

During CI validation and code review (Superjomn, hchings, pcastonguay,
CodeRabbit), twelve issues were found and fixed:

1. **Thread leak (`proxy_error_monitor`)**: `pytest-threadleak` detected the
   daemon thread surviving past test teardown. **Fix:** join the thread during
   `shutdown()` with a 5-second timeout.
2. **`test_health[False-503]` assertion failure**: The existing test patched
   `is_shutdown()`, but `BaseLLM._check_health()` now delegates to
   `check_health()`. **Fix:** patch `check_health` instead.
3. **Self-join deadlock**: `_error_monitor_loop` calls `shutdown()`, which joins
   the monitor thread. **Fix:** `threading.current_thread()` guard.
4. **Aliased-list bug in `_handle_errors()`** (pre-existing, made worse by PR):
   `failed_requests` aliased `self.active_requests`, then `clear()` emptied it
   before `_terminate_request` ran. **Fix:** `list(self.active_requests)` copy.
5. **Per-request errors treated as fatal**: Monitor loop and `check_health()`
   promoted `RequestError`/string errors from the queue to fatal. **Fix:** skip
   with `isinstance(e, (str, RequestError))` check.
6. **`check_health`/monitor called `shutdown()` inline**: `shutdown()` blocks
   on `f.result()` for surviving workers.  **Fix:** use `pre_shutdown()` which
   is non-blocking.
7. **`pre_shutdown` sentinel used `all()`**: When one worker was already dead,
   `all(not f.done())` was False, so the quit sentinel was never sent to
   surviving workers. **Fix:** `any(not f.done())`.
8. **Duplicate MPI future / queue drain code**: `check_health()` and
   `_error_monitor_loop` had near-identical logic.  **Fix:** extract shared
   `_check_mpi_futures()` and `_drain_error_queue()` helpers.
9. **Error budget fields loose on PyExecutor**: 4 separate float/Optional
   fields cluttered the class.  **Fix:** `ErrorBudget` dataclass in
   `error_classification.py` with a `consume()` method.
10. **Proxy single-drain vs base drain-all**: Proxy drained one item per call
    while the base class drained all.  **Fix:** both proxy helpers now use
    drain-all `while True` loops.
11. **Silent `except Exception: pass` in monitor**: Hard to debug.  **Fix:**
    `logger.debug(...)` instead of silent pass.
12. **Request-scoped errors consumed the error budget** (pcastonguay): A burst
    of malformed requests could exhaust the budget and crash a healthy server.
    **Fix:** `charge_budget=False` on validation, KV-transfer timeout,
    guided-decoder, and cache-transfer call sites.

### Remaining Test Gap

- **Integration test**: Start serving, kill MPI worker externally, verify
  `/health` returns 503 within ~10 seconds. Not yet automated — requires
  multi-GPU environment with K8s liveness probes.
- **Disaggregated end-to-end error body test**: Cause a context server to fail,
  verify that the disaggregated frontend returns the original error body and
  request IDs from PR #13119 instead of a generic `400 Bad Request`.
- **Health vs request-error separation test**: Send a burst of malformed
  requests through the OpenAI/disaggregated server and assert (a) each request
  receives its real error, and (b) `/health` remains healthy because
  `charge_budget=False` was used for request-scoped paths.

---

## Open Questions (Resolved)

- ~~Should `_max_consecutive_errors` be configurable via `TorchLlmArgs`?~~
  **Resolved:** Replaced with token-bucket error budget. Configuration is not
  exposed — most open-source inference engines (vLLM, TGI, Triton) follow a
  "crash fast, let the orchestrator restart" pattern with no user-facing error
  tolerance knobs. If a specific customer needs tuning, a simple
  `fail_fast_on_error: bool` flag can be added later.

## Open Questions (Deferred)

- Should we add a CUDA context health probe (e.g., small allocation test) to
  `check_health()`? **Deferred** — the current pattern matching catches known
  fatal CUDA errors; a runtime probe adds latency to health checks.
- Should the `/health` endpoint distinguish between "shutting down gracefully"
  (503) and "engine crashed" (500)? **Deferred** — this is a TRT-LLM-side
  change (in `openai_server.py`'s `health()` method). Implementation is
  straightforward but value depends on whether Dynamo inspects the status code
  difference for routing decisions.
- Dynamo-side fix: Dynamo should implement circuit-breaker / health-aware
  routing independently. This TRT-LLM fix is defensive only.

## Remaining Gaps Across Detection, Propagation, and Handling

The two PRs substantially improve the failure story, but they do not close
every gap.  The remaining gaps fall into three buckets:

### Detection gaps

- **Hung rank without process exit**: PR #12718 detects completed MPI futures
  and queued background errors.  It does **not** detect a rank that is still
  alive but stuck inside a CUDA/NCCL/MPI collective.  WideEP FT's AlltoAll
  watchdog / main-thread polling work is still required for that class.
- **`RemoteMpiCommSessionClient` visibility**: in the `trtllm-llmapi-launch`
  / `mgmn_leader_node` path, `submit()` returns `[]`, so
  `_check_mpi_futures()` has no local future handles to inspect.  This is why
  the `pre_shutdown()` empty-list sentinel branch is required, and why
  process-death detection for that deployment still depends on the remote
  manager / queue path rather than local futures.
- **GPU-context liveness probe**: no active CUDA context probe is run from
  `/health`.  Known fatal CUDA messages are classified, but silent context
  corruption with no queued error can still require a request or monitor signal
  to surface.
- **External orchestrator routing**: TRT-LLM now marks itself unhealthy, but
  Dynamo / Kubernetes must consume that signal promptly and stop routing.
  This investigation still treats Dynamo-side circuit breaking as an external
  gap.

### Propagation gaps

- **Streaming SSE error format consistency**: PR #13119 adds checks before
  normal response formatting, but streaming paths must consistently emit a
  structured SSE error event and terminate with `[DONE]`.  CodeRabbit noted
  this risk during PR #13119 review; audit current streaming helpers before
  relying on this path operationally.
- **Sanitized vs diagnostic detail**: gRPC health responses intentionally
  return `TypeName: first line` while logging full tracebacks server-side.
  That is correct for clients, but support/debug tooling must know where to
  retrieve the full traceback.
- **Postprocessing worker failures**: PR #13119 converts postprocessing
  exceptions to `ErrorResponse`, but if the postprocessing process itself dies
  hard (SIGKILL / OOM), it still becomes a worker/process-liveness problem
  rather than a clean request error.

### Handling gaps

- **Serving in degraded mode**: PR #12718 is fail-fast at the pod level.  It
  does not attempt rank masking, communicator rebuild, or serving on a reduced
  EP group.  That work belongs to the WideEP FT design.
- **Retry semantics after fatal**: after `_fatal_error` is set, queued and
  active requests are failed.  Client-side retry policy is outside TRT-LLM and
  must be handled by the caller/orchestrator.
- **User-facing tuning knobs**: error-budget thresholds are intentionally not
  exposed.  This is the right default for crash-fast serving, but deployments
  with unusual retry/orchestration semantics may eventually want a coarse
  `fail_fast_on_error`-style knob.

## Backend Coverage

| Backend | Executor class | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|---|---|---|---|---|---|
| **PyTorch (default)** | `GenerationExecutorProxy` | Yes | Yes | Yes | Yes |
| **PyTorch + RPC** | `GenerationExecutorRpcProxy` | Base only | No | Yes | Yes |
| **PyTorch + Ray** | `RayExecutor` | Base only | No | Yes | Yes |
| **TensorRT (legacy)** | `GenerationExecutorProxy` | Yes | Yes | No (C++ executor) | Yes |
| **AutoDeploy** | Same as PyTorch | Same | Same | Same | Same |
| **ModelRunnerCpp** | No `GenerationExecutor` | No | No | No | No |

Primary use case (PyTorch backend via `trtllm-serve` with MPI) has all four
layers active — this is the configuration that hit the original zombie pod.
