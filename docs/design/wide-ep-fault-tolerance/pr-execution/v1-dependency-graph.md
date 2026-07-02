# V1 PR Dependency Graph

[< Back to WideEP Fault Tolerance](../README.md) · [Implementation plan](08-implementation-plan.md)

**Status snapshot:** 2026-07-02 16:34 PDT

**Scope:** Phase 1 v1 plus the two post-MVP parallel tracks, Phase 1-DS and conditional Phase 1-IB.

**JIRA mapping snapshot:** user-provided 2026-06-29. Ticket keys appear on mapped nodes; workflow and assignee are tracked in the canonical [JIRA work-item ledger](jira-work-item-ledger.md), not in PR-status colors or dependency readiness.

All V1 implementation units are **planned** at this snapshot; no upstream PR has been opened for a V1 plan ID. MVP prerequisite anchors carry their live PR/planned colors so blocked edges can be read correctly. See the [MVP graph](mvp-dependency-graph.md) for full context.

## Status colors

| Color | Status |
|:---|:---|
| Green (`#dcfce7`) | Merged |
| Orange (`#ffedd5`) | Draft / not officially opened for review |
| Blue (`#dbeafe`) | Inflight — review and approvals required |
| Purple (`#ede9fe`) | Inflight — approved / merge-ready or CI-blocked |
| Gray (`#f3f4f6`) | Planned |

White, heavy-border nodes are milestone, audit, or external prerequisite gates rather than PR-status nodes.

## Dependency-state colors

| Edge or marker | Meaning |
|:---|:---|
| Green solid edge (`#16a34a`) | The prerequisite path is satisfied and does not block the target. |
| Red solid edge (`#dc2626`) | The prerequisite path is unsatisfied and blocks the target. |
| Red dashed edge | Conditional-path prerequisite that is currently unsatisfied. |
| Gray dashed edge (`#6b7280`) | Informational or deployment context outside the current committed path. |
| Gold outline + `★` | Dependency-ready action. No V1 implementation node qualifies; reused MVP prerequisite anchors may still be gold. |

## Core V1 graph

The global prerequisite for every node is the **corrected MVP**: it includes the
promoted 1a.8 in-flight-kernel escape, promoted 1a.11 graph-safe recovery,
survivor control-plane rebuild, the atomic recovery coordinator,
poisoned-MPI shutdown, and real-model E2E acceptance. The graph expands only
the more specific MVP merge unit each V1 PR consumes. The 1a.7 and 1a.8
delivery PRs [#15789](https://github.com/NVIDIA/TensorRT-LLM/pull/15789) and
[#15895](https://github.com/NVIDIA/TensorRT-LLM/pull/15895) remain draft with
`blossom-ci` pending; those states are folded into the unsatisfied global gate
rather than represented as otherwise-unconnected V1 anchors.

```mermaid
flowchart LR
    MVP["Corrected MVP landed<br/>1a.8 + 1a.11 + survivor-control/recovery contracts<br/>global release gate · not satisfied"]

    subgraph MVP_ANCHORS["MVP prerequisite anchors"]
        direction TB
        M_A1["MVP 1a.1 · EPGroupHealth<br/>PR: #13302 · merged<br/>JIRA: TRTLLM-12199"]
        M_A2["MVP 1a.2 · kernel-mask pattern<br/>PR: #13404 · merged 2026-06-30 PDT<br/>JIRA: TRTLLM-12200"]
        M_B1["MVP 1b.1 + 1b.2 · mask-only EPLB<br/>PR: #15525 · merged 2026-06-29 PDT<br/>JIRA: TRTLLM-13543 / TRTLLM-13544"]
        M_C3["MVP 1c.3 · FT broadcast<br/>PR: #15785 · draft · blossom-ci pending<br/>JIRA: TRTLLM-13548<br/>★ dependency-ready action"]
        M_D4["MVP 1d.4 · E2E harness<br/>PR: not opened · planned<br/>JIRA: TRTLLM-13554"]
    end

    AUDIT3["Audit 3 positive<br/>NIXL-EP gate"]

    subgraph COMM["1a · Communication coverage"]
        direction TB
        A1_5["1a.5 · NVLinkTwoSided kernel mask<br/>JIRA: TRTLLM-12558<br/>planned"]
        A1_6["1a.6 · NVLinkTwoSided Python binding<br/>JIRA: TRTLLM-12559<br/>planned"]
        A1_9["1a.9 · NIXL-EP strategy and factory<br/>planned · conditional"]
        A1_10["1a.10 · NIXL-EP FT integration<br/>planned · conditional"]
    end

    subgraph EPLB["1b · Full EPLB reconfiguration"]
        direction TB
        B1_4["1b.4 · mutable EPLB metadata<br/>planned"]
        B1_5["1b.5 · full online reconfigure<br/>planned"]
        B1_6["1b.6 · weight migration<br/>planned"]
        B1_7["1b.7 · zero-replica handling<br/>planned"]
    end

    subgraph DETECT["1c · Multi-failure detection"]
        direction TB
        C1_5["1c.5 · barrier-piggyback broadcast<br/>planned"]
        C1_6["1c.6 · two-phase multi-failure consensus<br/>planned"]
    end

    subgraph VALIDATE["1d · Production validation"]
        direction TB
        D1_6["1d.6 · multi-failure chaos suite<br/>planned"]
        D1_7["1d.7 · cross-model matrix<br/>planned"]
    end

    MVP --> A1_5
    MVP --> A1_9
    MVP --> B1_4
    MVP --> C1_5
    MVP --> C1_6
    MVP --> D1_7

    M_A2 --> A1_5
    A1_5 --> A1_6
    M_A1 --> A1_9
    AUDIT3 -.->|conditional ship gate| A1_9
    A1_9 --> A1_10

    M_B1 --> B1_4
    B1_4 --> B1_5
    B1_5 --> B1_6
    B1_5 --> B1_7

    M_C3 --> C1_5
    M_C3 --> C1_6
    C1_6 --> D1_6
    B1_6 --> D1_6
    B1_7 --> D1_6
    M_D4 --> D1_7

    linkStyle 0,1,2,3,4,5,7,10,12,13,14,15,16,17,18,19,20 stroke:#dc2626,stroke-width:3px;
    linkStyle 6,8,11 stroke:#16a34a,stroke-width:3px;
    linkStyle 9 stroke:#dc2626,stroke-width:3px,stroke-dasharray:6 4;

    classDef merged fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef draft fill:#ffedd5,stroke:#f97316,color:#7c2d12,stroke-width:2px;
    classDef reviewing fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px;
    classDef approved fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:2px;
    classDef planned fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:5 5;
    classDef gate fill:#ffffff,stroke:#111827,color:#111827,stroke-width:3px;
    classDef candidate stroke:#d97706,stroke-width:5px,stroke-dasharray:0;

    class M_A1,M_A2,M_B1 merged;
    class M_C3 draft;
    class M_D4,A1_5,A1_6,A1_9,A1_10,B1_4,B1_5,B1_6,B1_7,C1_5,C1_6,D1_6,D1_7 planned;
    class MVP,AUDIT3 gate;
    class M_C3 candidate;
```

## Parallel Phase 1-DS graph

This track starts after MVP and can run in parallel with core V1. It is not part of the core V1 release gate unless scope is explicitly changed.

```mermaid
flowchart LR
    MVP["Corrected MVP landed<br/>global release gate · not satisfied"]
    M_D4["MVP 1d.4 · E2E harness<br/>PR: not opened · planned<br/>JIRA: TRTLLM-13554"]
    M_C1["MVP 1c.1 · error classification<br/>PR: #15677 · merged 2026-07-02 PDT<br/>JIRA: TRTLLM-13546"]
    RAY_GAP["Ray + disagg + NIXL support<br/>required only for a Ray deployment"]

    DS1["DS.1 · per-pool FT harness<br/>planned"]
    DS2["DS.2 · KV failure-surface audit<br/>planned"]
    DS3["DS.3 · cross-pool notification<br/>planned"]
    DS4["DS.4 · retry and reroute policy<br/>planned"]
    DS5["DS.5 · KV transfer cancellation<br/>planned"]
    DS6["DS.6 · disagg E2E fault injection<br/>planned"]

    MVP --> DS1
    MVP --> DS2
    M_D4 --> DS1
    M_C1 --> DS2
    DS2 --> DS3
    DS3 --> DS4
    DS2 --> DS5
    DS1 --> DS6
    DS4 --> DS6
    DS5 --> DS6
    RAY_GAP -.->|deployment-conditional| DS6

    linkStyle 0,1,2,4,5,6,7,8,9 stroke:#dc2626,stroke-width:3px;
    linkStyle 3 stroke:#16a34a,stroke-width:3px;
    linkStyle 10 stroke:#6b7280,stroke-width:2px,stroke-dasharray:6 4;

    classDef merged fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef planned fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:5 5;
    classDef gate fill:#ffffff,stroke:#111827,color:#111827,stroke-width:3px;

    class M_C1 merged;
    class M_D4,DS1,DS2,DS3,DS4,DS5,DS6 planned;
    class MVP,RAY_GAP gate;
```

## Conditional Phase 1-IB graph

The two incoming edges to `IB path selected` are alternatives: choose the DeepEP timeout interim **or** the NIXL-EP path. They are not cumulative requirements.

```mermaid
flowchart LR
    MVP["MVP landed"]
    AUDIT3["Audit 3 positive"]
    STORE["TCPStore with MPI<br/>or Ray pivot"]
    COORD["Topology-aware EPLB coordination"]

    IB1["IB.1 · DeepEP timeout interim<br/>planned"]
    IB2["IB.2 · alias gate for core 1a.9–1a.10<br/>not an additional PR"]
    IB_PATH{"IB path selected<br/>IB.1 OR IB.2"}
    IB3["IB.3 · DeepEP lifecycle hardening<br/>planned"]
    IB4["IB.4 · B200 + IB fault harness<br/>planned"]
    IB5["IB.5 · deployment guide<br/>planned"]
    IB6["IB.6 · topology-aware EPLB<br/>planned"]

    MVP --> IB1
    MVP --> IB2
    AUDIT3 -.-> IB2
    STORE -.-> IB2
    IB1 --> IB_PATH
    IB2 --> IB_PATH
    IB1 --> IB3
    IB_PATH --> IB4
    IB4 --> IB5
    IB_PATH --> IB6
    COORD -.-> IB6

    linkStyle 0,1,4,5,6,7,8,9 stroke:#dc2626,stroke-width:3px;
    linkStyle 2,3,10 stroke:#dc2626,stroke-width:3px,stroke-dasharray:6 4;

    classDef planned fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:5 5;
    classDef gate fill:#ffffff,stroke:#111827,color:#111827,stroke-width:3px;

    class IB1,IB3,IB4,IB5,IB6 planned;
    class MVP,AUDIT3,STORE,COORD,IB2,IB_PATH gate;
```

## Candidate actions now

None of the V1, Phase 1-DS, or Phase 1-IB implementation nodes qualify yet. Every implementation path has an unsatisfied corrected-MVP release/component dependency, shown by at least one red ancestral edge. The merged #13404 component edge into 1a.5, #15525 edge into 1b.4, and #15677 edge into DS.2 are green, but the global corrected-MVP release edges still block those nodes. Gold prerequisite anchors are dependency-ready **MVP** actions reused here, not V1 candidates. Audit and external-integration gates may proceed as preparatory work, but they are not PR action-frontier nodes.

## JIRA work-item mapping

See the [canonical JIRA work-item ledger](jira-work-item-ledger.md) for the supplied tickets and newly identified `TBD` mappings. The V1 nodes shown here retain 1a.5 / TRTLLM-12558 and 1a.6 / TRTLLM-12559. Item 1a.8 / TRTLLM-12561 is now a corrected-MVP item and is intentionally absent from this graph.

## Tracking decisions

- Core V1 contains **10 unconditional plan IDs plus 2 Audit-3-conditional NIXL-EP IDs**. Promoted items 1a.8 and 1a.11 belong to corrected MVP and must not be counted or rendered as V1 work.
- Reused MVP anchors reflect the live delivery state: #13404, #15525, and #15677 are merged, while #15785 is draft. The corrected-MVP gate also includes #15789 and #15895 (both draft with `blossom-ci` pending), 1a.11, and the new survivor/recovery contracts. Only merged anchors turn their outgoing component edge green.
- `1d.6` requires multi-failure consensus and the completed full-EPLB branch. The graph shows terminal dependencies 1c.6, 1b.6, and 1b.7; 1b.4 and 1b.5 are satisfied transitively.
- Phase 1-DS and Phase 1-IB are visible here because both are scheduled after MVP and parallel to V1, but neither silently expands the core V1 ship gate.
- IB.2 is an umbrella alias for core PR units 1a.9 and 1a.10, not an additional implementation PR.
