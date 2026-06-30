# WideEP Fault Tolerance PR Execution

[< Back to WideEP Fault Tolerance](../README.md)

This folder is the single workspace for implementation sequencing, dependency state, PR delivery status, and JIRA execution tracking.

| Artifact | Purpose |
|:---|:---|
| [Implementation plan](08-implementation-plan.md) | Canonical PR breakdown, scope, dependencies, sizing, and timelines. |
| [MVP dependency graph](mvp-dependency-graph.md) | Live MVP PR status, blocked/unblocked edges, merged prerequisites, and dependency-ready actions. |
| [V1 dependency graph](v1-dependency-graph.md) | Core V1 plus parallel Phase 1-DS and conditional Phase 1-IB execution paths. |
| [V2 dependency graph](v2-dependency-graph.md) | Phase 2 restoration sequencing, audits, external gates, and candidate actions. |
| [JIRA work-item ledger](jira-work-item-ledger.md) | User-supplied JIRA workflow, assignee, graph-node, milestone, and PR mapping. |
| [Source-of-truth correction checklist](source-of-truth-correction-checklist.md) | Control record for the 2026-06-30 MVP contract, graph, PR, and validation corrections. |

## State ownership

- GitHub PR state drives node fill color and review/CI text.
- Prerequisite satisfaction drives green/red dependency edges.
- All hard parents merged drives the gold `★` dependency-ready marker.
- JIRA workflow remains separate planning metadata in the ledger.
