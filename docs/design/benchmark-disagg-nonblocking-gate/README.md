# Fixing the Benchmark Disaggregated Serving Deadlock: Replacing the Blocking Fill Loop with a Non-blocking Gate

| | |
|---|---|
| **JIRA** | [TRTLLM-11492](https://jirasw.nvidia.com/browse/TRTLLM-11492) |
| **PR** | [#12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) |
| **Author** | Chien-Chun Hung |
| **Created** | 2026-03-13 |
| **Last Updated** | 2026-04-13 |
| **Status** | Merged |

## Problem

In TensorRT-LLM's disaggregated serving benchmark mode, the GEN executor's `_prepare_and_schedule_batch` contained a blocking fill loop that held control until a batch of requests was fetched. During this blocking wait, the executor could not service KV transfers, check timeouts, detect errors, or handle control requests.

The severity depends on the transport backend:
- **MPI:** Sends can block until the receiver posts a matching receive. Since `_check_disagg_gen_transfer_status` never runs inside the fill loop, GEN never receives, CTX's sends block, and CTX cannot free KV cache blocks — causing a **hard deadlock** when CTX capacity < `batch_size`.
- **RDMA (NIXL/UCX):** CTX frees blocks based on send completion independently, so the deadlock is less likely. However, the fill loop still starves all other executor work (timeout handling, error detection, hang detector, control requests) for the duration of the wait.

## Solution

Replace the blocking fill loop with a non-blocking `can_forward` gate in the executor loops. Each main-loop iteration now performs a full processing cycle — fetch, service transfers, check timeouts, handle errors — then checks readiness via `_is_benchmark_disagg_fill_complete`. This eliminates the deadlock for all backends and ensures the executor remains responsive during the fill phase.

## Prior Art

This work builds on two related PRs:

| PR | What it fixes | Relationship |
|---|---|---|
| [#12091](https://github.com/NVIDIA/TensorRT-LLM/pull/12091) | First attempt at the CTX-side deadlock — batched fill loop (`batch_size = tp_size`) | This PR **supersedes** #12091 by eliminating the loop entirely |
| [#12206](https://github.com/NVIDIA/TensorRT-LLM/pull/12206) | GEN-side KV cache insufficiency — explicit error when requests can't fit | **Complementary** — runs inside `_prepare_and_schedule_batch`, before the gate |

## Justification

This PR is justified on three grounds:

1. **Correctness across all backends.** The fill loop never calls `_check_disagg_gen_transfer_status` in its inner loop. With MPI (a supported backend), this creates a genuine deadlock. The PR eliminates the structural issue for all backends.

2. **Robustness beyond the deadlock.** Even when RDMA avoids the deadlock, the fill loop prevents the GEN executor from doing any other work while it waits: no timeout handling, no error detection, no hang detector progress, no control request handling.

3. **Code quality.** Consolidates duplicated gating logic, adds a missing `can_forward` gate to `_executor_loop`, fixes the ADP dummy counting bug, and provides 40 unit tests for a code path that previously had zero test coverage.

## Detailed Design Documents

- [Background and Root Cause Analysis](background.md)
- [Design and Implementation](design.md)
- [ADP Dummy Request Handling](adp-dummy-requests.md)
- [Analysis and Test Coverage](analysis.md)
