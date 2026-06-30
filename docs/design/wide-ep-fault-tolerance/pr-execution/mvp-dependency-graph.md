# MVP PR Dependency Graph

[< Back to WideEP Fault Tolerance](../README.md) · [Implementation plan](08-implementation-plan.md) · [Correction checklist](source-of-truth-correction-checklist.md)

**Status snapshot:** 2026-06-30 12:15 PDT

**Scope:** corrected Phase 1 MVP merge units and acceptance gates.

The implementation plan is the item-definition source of truth; this graph is the proof that those definitions, live PR states, and hard dependencies agree. A solid arrow is a hard merge dependency. A dashed gray arrow is supporting or historical context and does not affect readiness.

JIRA workflow is planning metadata from the user-provided 2026-06-29 snapshot. PR fill colors and edge state come from live GitHub state and the declared dependency model, not from JIRA status.

## Status colors

| Color | Status | Rule |
|:---|:---|:---|
| Green (`#dcfce7`) | **Merged** | The upstream PR is merged. |
| Orange (`#ffedd5`) | **Draft** | A draft PR has not been opened for official review. |
| Blue (`#dbeafe`) | **Inflight — review required** | The PR is open and non-draft, but GitHub reports `REVIEW_REQUIRED`. CI/base state is shown in the node. |
| Purple (`#ede9fe`) | **Inflight — approved** | GitHub reports `APPROVED`; the node states whether CI still blocks merge. |
| Gray (`#f3f4f6`) | **Planned** | The roadmap has a named production item, but no upstream PR is open for it. |

## Dependency-state colors

| Edge or marker | Meaning |
|:---|:---|
| Green solid edge (`#16a34a`) | The source prerequisite is merged; this edge is satisfied. |
| Red solid edge (`#dc2626`) | The source prerequisite is not merged; this edge blocks the target. |
| Gray dashed edge (`#6b7280`) | Supporting, prototype, or historical relationship; excluded from readiness. |
| Gold outline + `★` | **Candidate action:** a non-merged production item whose hard parents are all merged. Root production items qualify; prototype nodes do not. |

The gold outline is orthogonal to PR status. It means dependency-unblocked, not review-ready or CI-green.

## Production merge graph

```mermaid
flowchart LR
    subgraph PREREQS["Merged supporting prerequisites"]
        direction TB
        P12718["#12718 · fatal-error classification<br/>merged"]
        P13119["#13119 · request-error propagation<br/>merged"]
    end

    subgraph COMM["1a · Communication and rank masking"]
        direction TB
        A1_1["1a.1 · EPGroupHealth primitive<br/>PR: #13302 · merged<br/>JIRA: TRTLLM-12199"]
        A1_2["1a.2 · launch-time NVLinkOneSided mask<br/>PR: #13404 · merged 2026-06-30 PDT<br/>JIRA: TRTLLM-12200"]
        A1_34["1a.3 + 1a.4 · Python binding + detection-only watchdog<br/>PR: #15524 · review required · dirty base · blossom-ci failed<br/>JIRA: TRTLLM-12556 / TRTLLM-12557<br/>★ dependency-ready action"]
        A1_7["1a.7 · coordinator-driven NCCL recovery primitive<br/>PR: #15789 · draft · blossom-ci failed<br/>JIRA: TRTLLM-12560<br/>★ dependency-ready action"]
        A1_8["1a.8 · running-kernel abort + generation primitive<br/>PR: not opened · planned · promoted to MVP<br/>JIRA: TRTLLM-12561<br/>★ dependency-ready action"]
        A1_11["1a.11 · eager fallback + graph invalidation/recapture<br/>PR: not opened · planned · promoted to MVP<br/>JIRA: TBD"]
    end

    subgraph EPLB["1b · Placement and topology adaptation"]
        direction TB
        B1_12["1b.1 + 1b.2 · mask-only C++ + Python API<br/>PR: #15525 · merged 2026-06-29 PDT<br/>JIRA: TRTLLM-13543 / TRTLLM-13544"]
        B1_3["1b.3 · iteration-boundary reconfigure integration<br/>PR: not opened · planned<br/>JIRA: TRTLLM-13545<br/>★ dependency-ready action"]
        B1_3A["1b.3a · FT placement invariant + admission<br/>PR: not opened · planned · new MVP item<br/>JIRA: TBD<br/>★ dependency-ready action"]
    end

    subgraph CONTROL["1c · Detection, survivor membership, and recovery"]
        direction TB
        C1_1["1c.1 · EP error patterns<br/>PR: #15677 · review required · blossom-ci pending<br/>JIRA: TRTLLM-13546<br/>★ dependency-ready action"]
        C1_2["1c.2 · per-rank health budgets<br/>PR: not opened · planned<br/>JIRA: TRTLLM-13547"]
        C1_3["1c.3 · failure-notification subcomm + broadcast<br/>PR: #15785 · draft · blossom-ci pending<br/>JIRA: TRTLLM-13548<br/>★ dependency-ready action"]
        C1_3A["1c.3a · survivor control communicator + ActiveRankMap<br/>PR: not opened · planned · new MVP item<br/>JIRA: TBD"]
        C1_4A["1c.4a · degraded attention-DP/PyExecutor membership<br/>PR: not opened · planned · new MVP item<br/>JIRA: TBD"]
        C1_4["1c.4 · recovery coordinator + atomic generation commit<br/>PR: not opened · planned · expanded MVP item<br/>JIRA: TRTLLM-13549"]
        C1_4B["1c.4b · failed epoch/request disposition<br/>PR: not opened · planned · new MVP item<br/>JIRA: TBD"]
    end

    subgraph INTEGRATE["1d · Gating, lifecycle, and validation"]
        direction TB
        D1_0["1d.0 · MPI signal-handler replacement<br/>PR: #14160 · merged<br/>JIRA: TRTLLM-13550"]
        D1_0A["1d.0a · poisoned-MPI lifecycle + shutdown<br/>PR: not opened · planned · new MVP item<br/>JIRA: TBD"]
        D1_1["1d.1 · unified feature + admission gate<br/>PR: not opened · planned · expanded MVP item<br/>JIRA: TRTLLM-13551"]
        D1_2["1d.2 · degraded health reporting<br/>PR: not opened · planned<br/>JIRA: TRTLLM-13552"]
        D1_3["1d.3 · passive rank-health telemetry<br/>PR: #15788 · draft · DCO action · blossom-ci pending<br/>JIRA: TRTLLM-13553<br/>★ dependency-ready action"]
        D1_4["1d.4 · real-component 4+ GPU E2E harness<br/>PR: not opened · planned · expanded MVP item<br/>JIRA: TRTLLM-13554"]
        D1_4A["1d.4a · NVL72 FABRIC/IMEX E2E acceptance<br/>PR: not opened · planned · new MVP item<br/>JIRA: TBD"]
        D1_5["1d.5 · steady-state overhead regression<br/>PR: not opened · planned<br/>JIRA: TRTLLM-13555"]
        MVP_EXIT["MVP exit<br/>real single-rank death · no failed-epoch output<br/>correct survivor serving · physical hardware"]
    end

    OLD_PROTO["Historical seam prototype<br/>PR: #14198 · draft · paused · mock-heavy<br/>JIRA: TRTLLM-12728 · non-production"]
    NEW_PROTO["No-mock integration prototype<br/>branch: WideEP-FT/e2e-mvp-prototype<br/>stacked PR heads · validation aid · non-merge node"]

    P12718 --> C1_1
    P13119 --> C1_4B

    A1_1 --> A1_34
    A1_2 -->|merged stack base| A1_34
    A1_1 --> A1_7
    A1_2 --> A1_8
    A1_7 --> A1_11

    B1_12 --> B1_3
    B1_12 --> B1_3A

    C1_1 --> C1_2
    A1_1 --> C1_3
    C1_3 --> C1_3A
    C1_3A --> C1_4A

    A1_34 --> C1_4
    A1_7 --> C1_4
    A1_8 --> C1_4
    B1_3 --> C1_4
    B1_3A --> C1_4
    C1_3 --> C1_4
    C1_3A --> C1_4
    C1_4A --> C1_4
    C1_4 --> C1_4B

    D1_0 --> D1_0A
    C1_3 --> D1_0A

    B1_3A --> D1_1
    A1_11 --> D1_1
    C1_4 --> D1_1
    D1_0A --> D1_1
    C1_4 --> D1_2
    A1_1 --> D1_3

    A1_34 --> D1_4
    A1_7 --> D1_4
    A1_8 --> D1_4
    A1_11 --> D1_4
    B1_3 --> D1_4
    B1_3A --> D1_4
    C1_2 --> D1_4
    C1_3 --> D1_4
    C1_3A --> D1_4
    C1_4 --> D1_4
    C1_4A --> D1_4
    C1_4B --> D1_4
    D1_0A --> D1_4
    D1_1 --> D1_4
    D1_2 --> D1_4
    D1_3 --> D1_4

    A1_34 --> D1_5
    A1_8 --> D1_5
    A1_11 --> D1_5
    D1_4 -.->|shared harness| D1_5
    D1_4 --> D1_4A

    D1_1 --> MVP_EXIT
    D1_2 --> MVP_EXIT
    D1_3 --> MVP_EXIT
    D1_4 --> MVP_EXIT
    D1_4A --> MVP_EXIT
    D1_5 --> MVP_EXIT

    A1_1 -.-> OLD_PROTO
    D1_0 -.-> OLD_PROTO
    OLD_PROTO -.->|findings only| D1_4

    A1_2 -.-> NEW_PROTO
    A1_34 -.-> NEW_PROTO
    A1_7 -.-> NEW_PROTO
    C1_3 -.-> NEW_PROTO
    D1_3 -.-> NEW_PROTO
    NEW_PROTO -.->|reference path + timing evidence| D1_4

    linkStyle default stroke:#dc2626,stroke-width:3px;
    linkStyle 0,1,2,3,4,5,7,8,10,22,29 stroke:#16a34a,stroke-width:3px;
    linkStyle 49,57,58,59,60,61,62,63,64,65 stroke:#6b7280,stroke-width:2px,stroke-dasharray:6 4;

    classDef merged fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef draft fill:#ffedd5,stroke:#f97316,color:#7c2d12,stroke-width:2px;
    classDef reviewing fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px;
    classDef approved fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:2px;
    classDef planned fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:5 5;
    classDef context fill:#f8fafc,stroke:#94a3b8,color:#475569,stroke-dasharray:3 3;
    classDef gate fill:#ffffff,stroke:#111827,color:#111827,stroke-width:3px;
    classDef candidate stroke:#d97706,stroke-width:5px,stroke-dasharray:0;

    class P12718,P13119,A1_1,A1_2,B1_12,D1_0 merged;
    class A1_7,C1_3,D1_3 draft;
    class A1_34,C1_1 reviewing;
    class A1_8,A1_11,B1_3,B1_3A,C1_2,C1_3A,C1_4A,C1_4,C1_4B,D1_0A,D1_1,D1_2,D1_4,D1_4A,D1_5 planned;
    class OLD_PROTO,NEW_PROTO context;
    class MVP_EXIT gate;
    class A1_34,A1_7,A1_8,B1_3,B1_3A,C1_1,C1_3,D1_3 candidate;
```

## Candidate actions now

| Node | Delivery state | Why dependency-unblocked | Immediate action |
|:---|:---|:---|:---|
| **1a.3 + 1a.4 / #15524** | Review required; dirty base; `blossom-ci` failed | 1a.1 / #13302 and 1a.2 / #13404 are merged | Rebase onto current `main`, enforce detection-only watchdog publication, diagnose CI, then request review. |
| **1a.7 / #15789** | Draft; `blossom-ci` failed | 1a.1 / #13302 is merged | Diagnose CI, align the coordinator/generation contract, finish validation, then mark ready. |
| **1a.8** | Planned; promoted to MVP | 1a.2 / #13404 is merged | Implement a running-kernel-observable abort/generation primitive and recoverable return path. |
| **1b.3** | Planned | 1b.1 + 1b.2 / #15525 is merged | Implement iteration-boundary placement reconfiguration, but publish only through the coordinator commit. |
| **1b.3a** | Planned; new MVP item | 1b.1 + 1b.2 / #15525 is merged | Implement per-layer/per-expert survivor admission and distinct-failure-domain validation. |
| **1c.1 / #15677** | Review required; `blossom-ci` pending | #12718 is merged | Complete review and CI; keep scope limited to classification patterns. |
| **1c.3 / #15785** | Draft; `blossom-ci` pending | 1a.1 / #13302 is merged | Finish failure-notification validation and document the boundary to 1c.3a. |
| **1d.3 / #15788** | Draft; DCO action; `blossom-ci` pending | 1a.1 / #13302 is merged | Repair sign-off and keep telemetry passive; then finish draft validation. |

An action can be dependency-ready while still blocked by code correctness, review, CI, DCO, or hardware. Items downstream of any red edge must not receive the gold marker.

## Live PR snapshot

| Plan ID | Upstream PR | Live state at snapshot | Corrected delivery role |
|:---|:---|:---|:---|
| Foundation | [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) | Merged 2026-04-27 | Classification foundation for 1c.1 and failed-request handling. |
| Supporting | [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) | Merged 2026-04-24 | Request-error propagation used by 1c.4b. |
| 1a.1 | [#13302](https://github.com/NVIDIA/TensorRT-LLM/pull/13302) | Merged 2026-06-17 PDT | Committed-mask primitive; detection state must be separate. |
| 1a.2 | [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) | **Merged 2026-06-30 PDT** | Launch-time/next-launch rank mask. A running kernel still requires 1a.8. |
| 1a.3 + 1a.4 | [#15524](https://github.com/NVIDIA/TensorRT-LLM/pull/15524) | Review required; dirty base; `blossom-ci` failed | Python mask wiring plus watchdog; must report suspicion without directly committing the data-plane mask. |
| 1b.1 + 1b.2 | [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525) | Merged 2026-06-29 PDT | Mask-only reconfigure APIs; they fail closed on a zero-survivor expert but do not prove admission. |
| 1c.1 | [#15677](https://github.com/NVIDIA/TensorRT-LLM/pull/15677) | Review required; `blossom-ci` pending | Pattern-only classifier slice. |
| 1c.3 | [#15785](https://github.com/NVIDIA/TensorRT-LLM/pull/15785) | Draft; `blossom-ci` pending | Failure evidence/broadcast; does not replace normal MPI/ADP collectives. |
| 1a.7 | [#15789](https://github.com/NVIDIA/TensorRT-LLM/pull/15789) | Draft; `blossom-ci` failed | Manual NCCL abort/rebuild primitive; coordinator and graph recovery remain separate items. |
| 1d.3 | [#15788](https://github.com/NVIDIA/TensorRT-LLM/pull/15788) | Draft; DCO action; `blossom-ci` pending | Passive telemetry; it must not drive recovery. |
| Historical prototype | [#14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) | Draft, paused, `DO NOT SUBMIT` | Mock-heavy seam-finding evidence only. It is not an MVP implementation dependency. |

## Tracking decisions

- Existing JIRA-backed IDs are unchanged. New suffix items are `JIRA: TBD` until tickets are assigned.
- #15524 and #15525 each remain one merge node containing two coherent work items. Their individual item identities and JIRA tickets remain visible.
- Detection state and committed communication state are separate. Only 1c.4 may atomically publish a common mask and generation after placement and communicator readiness.
- 1a.8 and 1a.11 are MVP ship gates, not V1 polish. They are removed from the V1 delivery set.
- 1c.3 is notification/consensus; 1c.3a and 1c.4a own survivor-only control and attention-DP/PyExecutor collectives.
- 1d.4 is an intra-node real-component E2E gate. 1d.4a is the rack-fabric/IMEX acceptance gate. Neither the old nor new prototype node can satisfy those production gates by itself.

## Updating this graph

1. Refresh live PR state from upstream GitHub and timestamp the snapshot.
2. Apply node fill in this order: merged → draft → approved → review required → planned.
3. Keep CI, DCO, and base state as qualifiers; they do not change review-status color.
4. Recompute every hard edge from the source node: green only when the source PR is merged, red otherwise.
5. Recompute the gold frontier after every merge or dependency edit.
6. Recount zero-based `linkStyle` indices whenever an edge is inserted, removed, or reordered.
