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
- Add `_set_fatal_error(error)`: records the first fatal error
- Add `check_health() -> bool`: returns `False` if `doing_shutdown` or
  `_fatal_error` is set; also **drains `_error_queue`** by calling
  `_handle_background_error()` when the queue is non-empty
- Modify `_handle_background_error()`: call `_set_fatal_error(error)` before
  `self.shutdown()` for serious errors and errors drained from the queue
- Update `is_shutdown()` to also return `True` when `_fatal_error` is set

**Why this helps:** Health probes now detect errors that were sitting in the
queue unprocessed, and return 503 so the orchestrator stops routing traffic.

### Layer 2: MPI Worker Liveness Check + Background Monitor

**File:** `tensorrt_llm/executor/proxy.py`

- Override `check_health()` in `GenerationExecutorProxy`: after the base check,
  verify MPI worker futures — if any future is `.done()`, extract its exception,
  set fatal error, trigger shutdown, return `False`
- Add `_error_monitor_loop()` daemon thread: every ~5 seconds checks MPI futures
  and error queue, triggers shutdown on detection
- Join `_error_monitor_thread` during `shutdown()` (before joining other
  threads) with a 5-second timeout to prevent thread leaks detected by
  `pytest-threadleak`

**Why this helps:** Even if no health checks or `generate()` calls arrive, the
monitor thread auto-detects worker crash within ~5 seconds and shuts down.

### Layer 3: Fatal Error Detection in PyExecutor

**File:** `tensorrt_llm/_torch/pyexecutor/py_executor.py`

- Add `_is_fatal_error(error_msg) -> bool`: pattern-matches against known CUDA
  fatal errors (OOM, illegal address, launch failure, NCCL error, device-side
  assert)
- Modify `_handle_errors()`: if `_is_fatal_error()` returns true, set
  `_fatal_error` and call `enqueue_shutdown_request()` to break the executor
  loop
- Add consecutive error counting: if `_max_consecutive_errors` (default 10)
  errors occur in a row without a successful iteration, treat as fatal
- Reset counter on successful iteration completion

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
  use `check_health()` and report `_fatal_error` details in the error message
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

For PyExecutor (in-process):
```
T+0s    CUDA OOM in _forward_step() -> _handle_errors() called
T+0s    _is_fatal_error("CUDA out of memory") returns True
T+0s    _fatal_error set, enqueue_shutdown_request() called
T+0s    Executor loop breaks, shutdown proceeds
```

---

## Test Coverage

All unit tests are in
`tests/unittest/executor/test_fatal_error_health_check.py` (52 tests total).

| Test class | Count | What's covered |
|---|---|---|
| `TestIsFatalError` | 15 | Pattern matching for CUDA OOM, NCCL, device-side assert, launch failure, case insensitivity; non-fatal errors (tokenizer, KV cache, timeout, etc.) correctly pass through |
| `TestHandleErrors` | 6 | Fatal errors fail all active requests and enqueue shutdown; non-fatal only fails specified requests; consecutive threshold promotion; counter reset; edge cases (no active requests, default error message) |
| `TestSetFatalError` | 2 | First-error-wins semantics; initial `None` state |
| `TestCheckHealth` | 5 | Healthy default; unhealthy after fatal error, shutdown, or error queue item; healthy with empty queue |
| `TestIsShutdown` | 3 | Returns `True` for `doing_shutdown` or `_fatal_error` set |
| `TestProxyCheckHealth` | 6 | Running workers healthy; crashed/exited/cancelled workers detected; no mpi_futures; parent unhealthy short-circuits |
| `TestErrorMonitorLoop` | 3 | Background thread detects worker crash, error queue items, and stops on shutdown flag |
| `TestGrpcHealthCheck` | 5 | Healthy executor, fatal error message surfaced, no executor, no LLM, shutdown state |
| `TestOpenAIHealthEndpoint` | 4 | 200 when healthy, 503 when unhealthy, SIGINT raised on fatal error, no SIGINT without fatal error |
| `TestBaseLLMCheckHealth` | 3 | Delegation to executor `check_health()`; no executor returns `False` |

### CI Findings

During CI validation (PR #12718, build 32444), two issues surfaced:

1. **Thread leak (`proxy_error_monitor`)**: `pytest-threadleak` detected the
   daemon thread surviving past test teardown. **Fix:** join the thread during
   `shutdown()` with a 5-second timeout.
2. **`test_health[False-503]` assertion failure**: The existing test patched
   `is_shutdown()` to simulate an unhealthy executor, but `BaseLLM._check_health()`
   now delegates to `check_health()`. **Fix:** patch `check_health` instead.

### Remaining Test Gap

- **Integration test**: Start serving, kill MPI worker externally, verify
  `/health` returns 503 within ~10 seconds. Not yet automated — requires
  multi-GPU environment with K8s liveness probes.

---

## Open Questions

- Should `_max_consecutive_errors` be configurable via `TorchLlmArgs`?
- Should we also add a CUDA context health probe (e.g.,
  `torch.cuda.is_available()` or a small allocation test) to `check_health()`?
- Should the `/health` endpoint distinguish between "shutting down gracefully"
  (503) and "engine crashed" (500)?
- Dynamo-side fix: Dynamo should implement circuit-breaker / health-aware
  routing independently. This TRT-LLM fix is defensive only.
