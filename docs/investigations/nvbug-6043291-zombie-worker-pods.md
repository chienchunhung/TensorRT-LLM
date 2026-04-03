# NVBug 6043291: Zombie Worker Pods After CUDA Engine Crash

- **Severity:** P0 / Critical
- **Reported by:** Astra (customer)
- **Affected model:** gpt-oss-120b
- **Date:** 2026-03-18
- **Branch:** `fix-zombie-worker-health-check`
- **PR:** [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718)
- **Status:** In review — CI passing after fix-up commits

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
zombie processes.

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

- Override `check_health()` in `GenerationExecutorProxy`: after the base check,
  verify MPI worker futures — if any future is `.done()`, extract its exception,
  set fatal error, call `pre_shutdown()` (not `shutdown()`, which would block
  on `f.result()` for surviving workers), return `False`
- Add `_error_monitor_loop()` daemon thread: every ~5 seconds checks MPI futures
  and error queue, calls `pre_shutdown()` on detection
- Both `check_health()` and the monitor drain the error queue directly
  (`get_nowait`) instead of calling `_handle_background_error()` (documented for
  main-thread use); per-request errors (`RequestError`, `str`) are skipped
- Fix `pre_shutdown()` sentinel condition: `all(not f.done())` → `any(not
  f.done())` so surviving workers still get the quit signal when one has died
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

Token-bucket parameters (hardcoded, not user-facing):
- `_error_budget`: starts at 1.0, capped at 1.0
- `_error_budget_cost`: 0.1 per transient error, 0.5 per severe error
- `_error_budget_recovery_rate`: 0.1 per second of error-free wall time
- Immediate-fatal errors bypass the budget entirely

**Design rationale:** A simple consecutive counter (crash after N errors) was
replaced because it couldn't distinguish between "10 transient errors over an
hour" (fine) and "10 errors in 100ms" (engine is broken).  CUDA OOM is
classified as severe (not immediate-fatal) because the CUDA context remains
valid after a failed allocation — the engine can recover if the next batch
is smaller.

On the fatal path, `_handle_errors()` also:
- Sets `is_shutdown = True` immediately (prevents the executor loop from
  scheduling more requests on a corrupted CUDA context)
- Copies `active_requests` to a local list before calling `clear()` (fixes an
  aliased-list bug where `_terminate_request` never ran because the list was
  emptied first)

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
T+0s    _classify_error() returns "immediate_fatal" -> budget bypassed
T+0s    _fatal_error set, enqueue_shutdown_request() called
T+0s    Executor loop breaks, shutdown proceeds
```

For PyExecutor (in-process, severe error / budget exhaustion):
```
T+0s    CUDA OOM in _forward_step() -> _handle_errors()
T+0s    _classify_error() returns "severe" -> budget -= 0.5 (budget=0.5)
T+0s    Engine retries next iteration (budget > 0)
T+0s    CUDA OOM again -> budget -= 0.5 (budget=0.0)
T+0s    Budget exhausted -> _fatal_error set, shutdown
```

---

## Test Coverage

All unit tests are in
`tests/unittest/executor/test_fatal_error_health_check.py` (56 tests total,
heavily parametrized).  `TestClassifyError` tests the **real**
`classify_error()` function directly (imported from `error_classification.py`
via `importlib` to avoid triggering C++ extension loading).

| Test class | Count | What's covered |
|---|---|---|
| `TestClassifyError` | 15 | Tests the real `classify_error()` function: three-tier classification, case insensitivity |
| `TestErrorBudget` | 9 | Token-bucket budget: immediate-fatal bypass, severe exhausts in 2, transient exhaustion, time recovery, aliased-list fix (all requests terminated on fatal), `is_shutdown` set on fatal, shutdown enqueue |
| `TestGenerationExecutor` | 10 | `_set_fatal_error` first-wins, `is_shutdown` (4 states), `check_health` (4 states incl. queue drain) |
| `TestProxyCheckHealth` | 6 | MPI worker future states (running / crashed / exited / cancelled), empty futures, parent-unhealthy short-circuit |
| `TestErrorMonitorLoop` | 4 | Worker crash, error queue, per-request string error skip, shutdown flag stop |
| `TestGrpcHealthCheck` | 5 | Parametrized: healthy, fatal, no executor, no LLM, shutdown |
| `TestOpenAIHealthEndpoint` | 3 | Parametrized: 200, 503, 503+SIGINT |
| `TestBaseLLMCheckHealth` | 4 | Parametrized delegation: healthy, fatal, no executor, missing attr |

### Issues Found During Development

During CI validation and code review, seven issues were found and fixed:

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

### Remaining Test Gap

- **Integration test**: Start serving, kill MPI worker externally, verify
  `/health` returns 503 within ~10 seconds. Not yet automated — requires
  multi-GPU environment with K8s liveness probes.

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
