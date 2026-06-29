# WideEP Fault Tolerance JIRA Work-Item Ledger

[< Back to WideEP Fault Tolerance](../README.md) · [MVP graph](mvp-dependency-graph.md) · [V1 graph](v1-dependency-graph.md)

**JIRA snapshot:** supplied by the user on 2026-06-29; time not provided.

**Coverage:** 22 supplied tickets across MVP, the MVP prototype, and V1. All supplied issues are type `Task`.

The planning views deliberately keep three state domains separate:

- **JIRA workflow** is recorded in this ledger.
- **PR delivery state** drives node fill color and review/CI text in the dependency graphs.
- **Prerequisite satisfaction** drives edge color.
- **All hard parents merged** drives the gold `★` dependency-ready marker.

JIRA status never changes a graph's PR color or dependency-readiness calculation.

| Plan ID | Milestone | Action item | JIRA | JIRA workflow | Assignee | Delivery node / upstream PR |
|:---|:---|:---|:---|:---|:---|:---|
| 1a.1 | MVP | EPGroupHealth thread-safe rank-mask primitive | [TRTLLM-12199](https://jirasw.nvidia.com/browse/TRTLLM-12199) | Done | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `A1_1`; [#13302](https://github.com/NVIDIA/TensorRT-LLM/pull/13302) |
| 1a.2 | MVP | NVLinkOneSided kernel mask (CUDA) | [TRTLLM-12200](https://jirasw.nvidia.com/browse/TRTLLM-12200) | In Review | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `A1_2`; [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) |
| 1a.3 | MVP | NVLinkOneSided Python binding | [TRTLLM-12556](https://jirasw.nvidia.com/browse/TRTLLM-12556) | To Do | Unassigned | `A1_34`; [#15524](https://github.com/NVIDIA/TensorRT-LLM/pull/15524), shared with 1a.4 |
| 1a.4 | MVP | AlltoAllWatchdog host thread | [TRTLLM-12557](https://jirasw.nvidia.com/browse/TRTLLM-12557) | In Progress | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `A1_34`; [#15524](https://github.com/NVIDIA/TensorRT-LLM/pull/15524) |
| 1a.5 | V1 | NVLinkTwoSided kernel mask (CUDA) | [TRTLLM-12558](https://jirasw.nvidia.com/browse/TRTLLM-12558) | To Do | Unassigned | `A1_5`; no upstream PR mapped |
| 1a.6 | V1 | NVLinkTwoSided Python binding update | [TRTLLM-12559](https://jirasw.nvidia.com/browse/TRTLLM-12559) | To Do | Unassigned | `A1_6`; no upstream PR mapped |
| 1a.7 | MVP | NCCL fault-tolerant wrapper + AllGatherReduceScatter mask wiring | [TRTLLM-12560](https://jirasw.nvidia.com/browse/TRTLLM-12560) | To Do | Unassigned | `A1_7`; no upstream PR mapped |
| 1a.8 | V1 | Tighten kernel-side `check_timeout` + replace `trap;` with host-visible flag | [TRTLLM-12561](https://jirasw.nvidia.com/browse/TRTLLM-12561) | To Do | Unassigned | `A1_8`; no upstream PR mapped |
| MVP prototype | MVP prototype | De-risk inter-component interaction | [TRTLLM-12728](https://jirasw.nvidia.com/browse/TRTLLM-12728) | To Do | Unassigned | `PROTO`; [#14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) |
| 1b.1 | MVP | `reconfigure_mask_only` C++ entry point | [TRTLLM-13543](https://jirasw.nvidia.com/browse/TRTLLM-13543) | In Progress | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `B1_12`; [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525) |
| 1b.2 | MVP | Python wrapper for mask-only reconfigure | [TRTLLM-13544](https://jirasw.nvidia.com/browse/TRTLLM-13544) | To Do | Unassigned | `B1_12`; [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525), shared with 1b.1 |
| 1b.3 | MVP | Iteration-boundary reconfigure integration | [TRTLLM-13545](https://jirasw.nvidia.com/browse/TRTLLM-13545) | To Do | Unassigned | `B1_3`; no upstream PR mapped |
| 1c.1 | MVP | EP-specific error classification patterns | [TRTLLM-13546](https://jirasw.nvidia.com/browse/TRTLLM-13546) | In Progress | Unassigned | `C1_1`; [#15677](https://github.com/NVIDIA/TensorRT-LLM/pull/15677) |
| 1c.2 | MVP | EPRankHealthTracker per-rank budgets | [TRTLLM-13547](https://jirasw.nvidia.com/browse/TRTLLM-13547) | To Do | Unassigned | `C1_2`; no upstream PR mapped |
| 1c.3 | MVP | MPI FT subcomm + broadcast thread | [TRTLLM-13548](https://jirasw.nvidia.com/browse/TRTLLM-13548) | To Do | Unassigned | `C1_3`; no upstream PR mapped |
| 1c.4 | MVP | Model engine health-check hook | [TRTLLM-13549](https://jirasw.nvidia.com/browse/TRTLLM-13549) | To Do | Unassigned | `C1_4`; no upstream PR mapped |
| 1d.0 | MVP | MPI signal handler replacement | [TRTLLM-13550](https://jirasw.nvidia.com/browse/TRTLLM-13550) | Done | [Chien-Chun Hung](https://jirasw.nvidia.com/secure/ViewProfile.jspa?name=chienchunh) | `D1_0`; [#14160](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) |
| 1d.1 | MVP | Feature flag + config gating (`enable_wide_ep_fault_tolerance`) | [TRTLLM-13551](https://jirasw.nvidia.com/browse/TRTLLM-13551) | To Do | Unassigned | `D1_1`; no upstream PR mapped |
| 1d.2 | MVP | `check_health()` degraded reporting | [TRTLLM-13552](https://jirasw.nvidia.com/browse/TRTLLM-13552) | To Do | Unassigned | `D1_2`; no upstream PR mapped |
| 1d.3 | MVP | Per-rank health telemetry / metrics | [TRTLLM-13553](https://jirasw.nvidia.com/browse/TRTLLM-13553) | To Do | Unassigned | `D1_3`; no upstream PR mapped |
| 1d.4 | MVP | 4-GPU E2E fault-injection harness + test | [TRTLLM-13554](https://jirasw.nvidia.com/browse/TRTLLM-13554) | To Do | Unassigned | `D1_4`; no upstream PR mapped |
| 1d.5 | MVP | Steady-state overhead regression test | [TRTLLM-13555](https://jirasw.nvidia.com/browse/TRTLLM-13555) | To Do | Unassigned | `D1_5`; no upstream PR mapped |

## Coverage gaps

No JIRA mapping was supplied for 1a.9–1a.11, 1b.4–1b.7, 1c.5–1c.6, 1d.6–1d.7, Phase 1-DS, Phase 1-IB, or V2 implementation units. Their graph nodes remain valid plan items without a JIRA key until mappings are provided.

## Tracking drift to reconcile

- TRTLLM-12556 / 1a.3 is `To Do`, but its implementation is carried in in-review PR #15524 with 1a.4.
- TRTLLM-13544 / 1b.2 is `To Do`, but its implementation is carried in in-review PR #15525 with 1b.1.
- TRTLLM-12200 / 1a.2 is `In Review`, while PR #13404 is already approved and waiting on CI.
