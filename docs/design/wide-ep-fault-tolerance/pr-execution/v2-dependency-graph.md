# V2 PR Dependency Graph — Phase 2 Restoration

[< Back to WideEP Fault Tolerance](../README.md) · [Implementation plan](08-implementation-plan.md)

**Status snapshot:** 2026-06-30 14:23 PDT

**Scope mapping:** The roadmap does not define a product milestone named “V2.” In this graph, **V2 means §8.2 Phase 2 Restoration**: replacing the dead rank, rebuilding process groups, and restoring full N-rank capacity. It does not mean the “Draft v2” revision of the design document itself.

**JIRA coverage:** no V2 ticket mapping was included in the supplied snapshot; see the [JIRA ledger coverage gaps](jira-work-item-ledger.md#coverage-gaps).

All V2 implementation units are **planned** at this snapshot. Sizes and several edges remain provisional until the teardown audits complete.

## Status colors

| Color | Status |
|:---|:---|
| Green (`#dcfce7`) | Merged |
| Orange (`#ffedd5`) | Draft / not officially opened for review |
| Blue (`#dbeafe`) | Inflight — review and approvals required |
| Purple (`#ede9fe`) | Inflight — approved / merge-ready or CI-blocked |
| Gray (`#f3f4f6`) | Planned |

White, heavy-border nodes are milestone, hardware, or external-project gates rather than PR-status nodes.

## Dependency-state colors

| Edge or marker | Meaning |
|:---|:---|
| Green solid edge (`#16a34a`) | The prerequisite is satisfied and does not block the target. |
| Red solid edge (`#dc2626`) | The prerequisite is unsatisfied and blocks the target. |
| Red dashed edge | Conditional, final-ship, or optional-track prerequisite that is currently unsatisfied. |
| Gray dashed edge (`#6b7280`) | Resource or out-of-scope context; reported separately from PR dependency readiness. |
| Gold outline + `★` | Dependency-ready implementation node. Resource availability may still be required. |

## Restoration dependency graph

Phase 2 extends, rather than replaces, the corrected-MVP recovery contracts.
The graph therefore makes the graph-invalidation contract, survivor-rank map,
and atomic recovery coordinator explicit inputs to communicator reconstruction.

```mermaid
flowchart LR
    V1_GATE["Corrected Phase 1 V1 complete<br/>includes corrected MVP · release gate not satisfied"]
    M_A7["Phase 1 1a.7 · NCCL FT wrapper<br/>PR: #15789 · draft · blossom-ci pending<br/>JIRA: TRTLLM-12560<br/>★ dependency-ready MVP action"]
    M_GRAPH["Corrected MVP 1a.11 · graph-safe recovery contract<br/>planned · JIRA: TBD"]
    M_SURV["Corrected MVP 1c.3a · survivor control communicator<br/>ActiveRankMap + generation · planned · JIRA: TBD"]
    M_COORD["Corrected MVP 1c.4b · atomic recovery coordinator<br/>planned · JIRA: TBD"]
    M_B5["Phase 1 V1 1b.5 · full EPLB reconfigure<br/>PR: not opened · planned<br/>JIRA: not mapped"]
    NVL_NODE["≥4-GPU NVLink-connected node<br/>audit hardware"]
    NVL72["NVL72 or equivalent<br/>hardware access"]
    MX1["MX-GMS Phase 1<br/>MX P2P · soft accelerator"]
    MX2["MX-GMS Phase 2<br/>GMS · soft accelerator"]

    subgraph REBUILD["2a · Process-group reconstruction"]
        direction TB
        A0A["2a.0a · intra-node teardown audit<br/>PR: not opened · planned<br/>JIRA: not mapped<br/>★ dependency-ready V2 action · hardware required"]
        A0B["2a.0b · rack-fabric validation<br/>planned"]
        A1["2a.1 · coordinated NCCL teardown<br/>planned"]
        A2["2a.2 · MNNVL teardown and reallocate<br/>planned · audit-dependent"]
        A3["2a.3 · NVSHMEM safe deallocation<br/>planned · deferred"]
        A4["2a.4 · DeepEP destroy sequencing<br/>planned · deferred"]
        A5["2a.5 · NVLink workspace deallocation<br/>planned"]
        A6["2a.6 · N-rank PG creation<br/>planned"]
        A7["2a.7 · full EPLB rebalance<br/>planned"]
        A8["2a.8 · second failure during rebuild<br/>planned"]
    end

    subgraph SHADOW["2b · Shadow EP ranks"]
        direction TB
        B1["2b.1 · shadow lifecycle and GMS pre-load<br/>planned"]
        B2["2b.2 · shadow health loop<br/>planned"]
        B3["2b.3 · activate, join PG, and serve<br/>planned"]
        B4["2b.4 · MX P2P fallback<br/>planned"]
    end

    subgraph ORCH["2c · Orchestrator integration"]
        direction TB
        C1["2c.1 · replacement provisioning API<br/>planned"]
        C2["2c.2 · replacement-rank join protocol<br/>planned"]
        C3["2c.3 · Phase 1 + 2 lifecycle E2E test<br/>planned"]
    end

    NVL_NODE -.->|resource required| A0A
    A0A --> A0B
    NVL72 -.->|rack resource required| A0B
    A0A -->|sizes implementation| A2
    A0B -.->|final ship gate| A2
    A0A --> A3

    V1_GATE --> A1
    M_A7 --> A1
    M_GRAPH --> A1
    A1 --> A4
    A1 --> A5

    A1 --> A6
    A2 --> A6
    A3 -.->|only if DeepEP remains in scope| A6
    A4 -.->|only if DeepEP remains in scope| A6
    A5 --> A6
    M_SURV -->|extends survivor map to replacement rank| A6
    M_COORD -->|reuses prepare/commit/resume transaction| A6

    A6 --> A7
    M_B5 --> A7
    A6 --> A8

    MX2 -.->|required for shadow track| B1
    B1 --> B2
    B1 --> B3
    A6 --> B3
    MX1 -.->|required for P2P fallback| B4

    A6 --> C1
    C1 --> C2
    A6 --> C2
    B3 -.->|optional GMS pre-load adapter| C2
    C2 --> C3
    A7 --> C3

    linkStyle 1,3,5,6,7,8,9,10,11,12,15,16,17,18,19,20,22,23,24,26,27,28,30,31 stroke:#dc2626,stroke-width:3px;
    linkStyle 0,2,29 stroke:#6b7280,stroke-width:2px,stroke-dasharray:6 4;
    linkStyle 4,13,14,21,25 stroke:#dc2626,stroke-width:3px,stroke-dasharray:6 4;

    classDef draft fill:#ffedd5,stroke:#f97316,color:#7c2d12,stroke-width:2px;
    classDef planned fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:5 5;
    classDef gate fill:#ffffff,stroke:#111827,color:#111827,stroke-width:3px;
    classDef candidate stroke:#d97706,stroke-width:5px,stroke-dasharray:0;

    class M_A7 draft;
    class M_GRAPH,M_SURV,M_COORD,M_B5,A0A,A0B,A1,A2,A3,A4,A5,A6,A7,A8,B1,B2,B3,B4,C1,C2,C3 planned;
    class V1_GATE,NVL_NODE,NVL72,MX1,MX2 gate;
    class M_A7,A0A candidate;
```

## Candidate actions now

**2a.0a — intra-node teardown audit** is the only V2 action-frontier node. It has no parent implementation PR and the roadmap explicitly allows it to start before V1 completes. A suitable ≥4-GPU NVLink-connected node is still a resource requirement, shown by the gray dashed edge; the gold marker means dependency-ready, not resource-booked.

The gold 1a.7 / #15789 prerequisite anchor is a dependency-ready **MVP** draft action reused in this view; it is not a V2 candidate.

The 1a.11, 1c.3a, and 1c.4b anchors are corrected-MVP contracts, not
independent V2 candidates. Their red edges make it explicit that Phase 2 may
not invent a second graph, membership, or recovery-transaction contract.

The MX-GMS-specific 2b.1 and 2b.4 nodes are not candidates: their external MX-GMS interfaces are unsatisfied optional-track prerequisites, shown as red dashed edges.

## Critical sequencing

1. **Audit in parallel with Phase 1:** 2a.0a can start before V1 completes. It sizes the rebuild work; 2a.0b requires rack hardware and gates the final 2a.2 ship decision.
2. **Baseline rebuild:** corrected V1 completion, merged 1a.7 / #15789, and the corrected-MVP 1a.11 graph contract unlock 2a.1. The baseline path then converges through 2a.2 and 2a.5 into 2a.6.
3. **Conditional DeepEP work:** 2a.3 and 2a.4 are deferred. Their edges into 2a.6 are dashed so they do not accidentally block the MNNVL baseline when DeepEP is out of scope.
4. **Capacity restoration:** 2a.6 extends the corrected-MVP `ActiveRankMap` and atomic coordinator to the replacement rank, then unlocks EPLB rebalance, second-failure handling, shadow activation, and replacement provisioning.
5. **Acceleration is soft:** MX-GMS shortens weight-load latency but must not gate a correct minutes-class disk-reload path.

## Replacement-join dependency decision

The baseline 2c.2 join protocol depends on **2c.1 + 2a.6** and must support a
disk-loaded replacement without MX-GMS. Item 2b.3 is an optional GMS pre-load
adapter to that same protocol, shown by the gray dashed edge. The Phase 1 + 2
lifecycle test, 2c.3, depends on the completed baseline join (2c.2) and the
post-rebuild EPLB rebalance (2a.7); 2c.1 is satisfied transitively through
2c.2.

## Updating this graph

When a V2 PR opens, replace its `planned` suffix with the PR number and apply the same five-state rules used by the [MVP graph](mvp-dependency-graph.md). Recompute red/green edges and the gold action frontier after every merge or external-gate decision. Do not promote provisional audit-dependent work to inflight solely because an audit prototype branch exists.
