# WideEP Fault Tolerance JIRA Work-Item Ledger

[< Back to WideEP Fault Tolerance](../README.md) · [MVP graph](mvp-dependency-graph.md) · [V1 graph](v1-dependency-graph.md)

**JIRA snapshot:** supplied by the user on 2026-06-29; time not provided.

**Coverage:** 22 user-supplied tickets plus eight corrected-MVP work items whose JIRA keys are still `TBD`. All supplied issues are type `Task`; no ticket or workflow state is invented for the new items.

The planning views deliberately keep three state domains separate:

- **JIRA workflow** is recorded in this ledger.
- **PR delivery state** drives node fill color and review/CI text in the dependency graphs.
- **Prerequisite satisfaction** drives edge color.
- **All hard parents merged** drives the gold `★` dependency-ready marker.

JIRA status never changes a graph's PR color or dependency-readiness calculation.

| Plan ID | Milestone | Action item | JIRA | JIRA workflow | Assignee | Delivery node / upstream PR |
|:---|:---|:---|:---|:---|:---|:---|
| 1a.1 | MVP | EPGroupHealth thread-safe rank-mask primitive | [TRTLLM-12199](https://jirasw.nvidia.com/browse/TRTLLM-12199) | Done | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `A1_1`; [#13302](https://github.com/NVIDIA/TensorRT-LLM/pull/13302) |
| 1a.2 | MVP | Launch-time NVLinkOneSided kernel mask (CUDA) | [TRTLLM-12200](https://jirasw.nvidia.com/browse/TRTLLM-12200) | In Review | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `A1_2`; [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) |
| 1a.3 | MVP | Committed-mask NVLinkOneSided Python binding | [TRTLLM-12556](https://jirasw.nvidia.com/browse/TRTLLM-12556) | To Do | Unassigned | `A1_34`; [#15524](https://github.com/NVIDIA/TensorRT-LLM/pull/15524), shared with 1a.4 |
| 1a.4 | MVP | Detection-only AlltoAllWatchdog host thread | [TRTLLM-12557](https://jirasw.nvidia.com/browse/TRTLLM-12557) | In Progress | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `A1_34`; [#15524](https://github.com/NVIDIA/TensorRT-LLM/pull/15524), shared with 1a.3 |
| 1a.5 | V1 | NVLinkTwoSided kernel mask (CUDA) | [TRTLLM-12558](https://jirasw.nvidia.com/browse/TRTLLM-12558) | To Do | Unassigned | `A1_5`; no upstream PR mapped |
| 1a.6 | V1 | NVLinkTwoSided Python binding update | [TRTLLM-12559](https://jirasw.nvidia.com/browse/TRTLLM-12559) | To Do | Unassigned | `A1_6`; no upstream PR mapped |
| 1a.7 | MVP | Coordinator-driven NCCL abort/rebuild primitive + AllGatherReduceScatter wiring | [TRTLLM-12560](https://jirasw.nvidia.com/browse/TRTLLM-12560) | To Do | Unassigned | `A1_7`; [#15789](https://github.com/NVIDIA/TensorRT-LLM/pull/15789) |
| 1a.8 | **MVP (promoted)** | Running-kernel abort + mask-generation primitive; recoverable return replaces `trap;` | [TRTLLM-12561](https://jirasw.nvidia.com/browse/TRTLLM-12561) | To Do | Unassigned | `A1_8`; no upstream PR mapped |
| 1a.11 | **MVP (promoted)** | Eager fallback + generation-scoped CUDA graph invalidation/recapture | JIRA: TBD | Not created | Unassigned | `A1_11`; no upstream PR mapped |
| MVP integration prototype | MVP validation aid | No-mock production-component vertical slice on physical hardware | [TRTLLM-12728](https://jirasw.nvidia.com/browse/TRTLLM-12728) | To Do | Unassigned | `NEW_PROTO`; branch `WideEP-FT/e2e-mvp-prototype`. [#14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) is the historical mock-heavy predecessor. |
| 1b.1 | MVP | `reconfigure_mask_only` C++ entry point | [TRTLLM-13543](https://jirasw.nvidia.com/browse/TRTLLM-13543) | In Progress | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `B1_12`; [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525) |
| 1b.2 | MVP | Python wrapper for mask-only reconfigure | [TRTLLM-13544](https://jirasw.nvidia.com/browse/TRTLLM-13544) | To Do | Unassigned | `B1_12`; [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525), shared with 1b.1 |
| 1b.2a | **MVP (new)** | Per-layer/per-expert FT placement invariant + startup/recovery admission | JIRA: TBD | Not created | Unassigned | `B1_2A`; no upstream PR mapped |
| 1b.3 | MVP | Iteration-boundary EPLB prepare/commit integration | [TRTLLM-13545](https://jirasw.nvidia.com/browse/TRTLLM-13545) | To Do | Unassigned | `B1_3`; no upstream PR mapped |
| 1c.1 | MVP | EP-specific error classification patterns | [TRTLLM-13546](https://jirasw.nvidia.com/browse/TRTLLM-13546) | In Progress | Unassigned | `C1_1`; [#15677](https://github.com/NVIDIA/TensorRT-LLM/pull/15677) |
| 1c.2 | MVP | EPRankHealthTracker per-rank budgets | [TRTLLM-13547](https://jirasw.nvidia.com/browse/TRTLLM-13547) | To Do | Unassigned | `C1_2`; no upstream PR mapped |
| 1c.3 | MVP | Failure-notification MPI subcommunicator + broadcast thread | [TRTLLM-13548](https://jirasw.nvidia.com/browse/TRTLLM-13548) | To Do | Unassigned | `C1_3`; [#15785](https://github.com/NVIDIA/TensorRT-LLM/pull/15785) |
| 1c.3a | **MVP (new)** | Survivor control communicator + immutable `ActiveRankMap` | JIRA: TBD | Not created | Unassigned | `C1_3A`; no upstream PR mapped |
| 1c.4 | MVP | Model-engine recovery hook joining detection/tracker/coordinator | [TRTLLM-13549](https://jirasw.nvidia.com/browse/TRTLLM-13549) | To Do | Unassigned | `C1_4`; no upstream PR mapped |
| 1c.4a | **MVP (new)** | Survivor-aware attention-DP/PyExecutor membership | JIRA: TBD | Not created | Unassigned | `C1_4A`; no upstream PR mapped |
| 1c.4b | **MVP (new)** | Atomic recovery coordinator and common generation commit | JIRA: TBD | Not created | Unassigned | `C1_4B`; no upstream PR mapped |
| 1c.4c | **MVP (new)** | Failed epoch + request disposition; no partial logits | JIRA: TBD | Not created | Unassigned | `C1_4C`; no upstream PR mapped |
| 1d.0 | MVP | MPI signal handler replacement | [TRTLLM-13550](https://jirasw.nvidia.com/browse/TRTLLM-13550) | Done | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `D1_0`; [#14160](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) |
| 1d.0a | **MVP (new)** | Poisoned-MPI lifecycle + deterministic shutdown | JIRA: TBD | Not created | Unassigned | `D1_0A`; no upstream PR mapped |
| 1d.1 | MVP | Unified feature + deployment admission gate (`TLLM_FAULT_TOLERANCE_MODE`) | [TRTLLM-13551](https://jirasw.nvidia.com/browse/TRTLLM-13551) | To Do | Unassigned | `D1_1`; no upstream PR mapped |
| 1d.2 | MVP | `check_health()` degraded reporting | [TRTLLM-13552](https://jirasw.nvidia.com/browse/TRTLLM-13552) | To Do | Unassigned | `D1_2`; no upstream PR mapped |
| 1d.3 | MVP | Passive committed-membership telemetry / metrics | [TRTLLM-13553](https://jirasw.nvidia.com/browse/TRTLLM-13553) | To Do | Unassigned | `D1_3`; [#15788](https://github.com/NVIDIA/TensorRT-LLM/pull/15788) |
| 1d.4 | **MVP (expanded)** | Real-component 4+ GPU E2E fault-injection harness + realistic model/workload | [TRTLLM-13554](https://jirasw.nvidia.com/browse/TRTLLM-13554) | To Do | Unassigned | `D1_4`; no upstream PR mapped |
| 1d.4a | **MVP (new ship gate)** | NVL72 FABRIC/IMEX process-death + peer-memory-containment E2E acceptance | JIRA: TBD | Not created | Unassigned | `D1_4A`; requires approved inaccessible-peer-memory/device-loss injection or Q3 remains fail-closed; no upstream PR mapped |
| 1d.5 | MVP | Steady-state overhead regression test | [TRTLLM-13555](https://jirasw.nvidia.com/browse/TRTLLM-13555) | To Do | Unassigned | `D1_5`; no upstream PR mapped |

## Coverage gaps

JIRA keys are required for the eight corrected-MVP items currently marked `TBD`: 1a.11, 1b.2a, 1c.3a, 1c.4a–1c.4c, 1d.0a, and 1d.4a. No mapping was supplied for 1a.9–1a.10, 1b.4–1b.7, 1c.5–1c.6, 1d.6–1d.7, Phase 1-DS, Phase 1-IB, or V2 implementation units. Plan nodes remain valid without a JIRA key; the ledger must not fabricate workflow state.

## Tracking drift to reconcile

- TRTLLM-12556 / 1a.3 is `To Do`, but its implementation is carried in draft PR #15524 with 1a.4.
- TRTLLM-13543 / 1b.1 remains `In Progress` and TRTLLM-13544 / 1b.2 remains `To Do`, but their shared implementation PR #15525 is merged.
- TRTLLM-12200 / 1a.2 is `In Review` in the supplied JIRA snapshot, while PR #13404 is merged.
- TRTLLM-12560 / 1a.7, TRTLLM-13548 / 1c.3, and TRTLLM-13553 / 1d.3 remain `To Do` and unassigned in the supplied JIRA snapshot, but draft PRs #15789, #15785, and #15788 now carry their implementation.
