# WideEP Fault Tolerance PR Execution

[< Back to WideEP Fault Tolerance](../README.md)

This folder is the single workspace for implementation sequencing, dependency state, PR delivery status, and JIRA execution tracking.

| Artifact | Purpose |
|:---|:---|
| [2026-06-30 correction checklist](source-of-truth-correction-checklist.md) | Gated control record for design repairs, promoted/new MVP items, PR corrections, validation, and publication. |
| [Implementation plan](08-implementation-plan.md) | Canonical PR breakdown, scope, dependencies, sizing, and timelines. |
| [MVP dependency graph](mvp-dependency-graph.md) | Live MVP PR status, blocked/unblocked edges, merged prerequisites, and dependency-ready actions. |
| [V1 dependency graph](v1-dependency-graph.md) | Core V1 plus parallel Phase 1-DS and conditional Phase 1-IB execution paths. |
| [V2 dependency graph](v2-dependency-graph.md) | Phase 2 restoration sequencing, audits, external gates, and candidate actions. |
| [JIRA work-item ledger](jira-work-item-ledger.md) | User-supplied JIRA workflow, assignee, graph-node, milestone, and PR mapping. |

## State ownership

- GitHub PR state drives node fill color and review/CI text.
- Prerequisite satisfaction drives green/red dependency edges.
- All hard parents merged drives the gold `★` dependency-ready marker.
- JIRA workflow remains separate planning metadata in the ledger.
- Detection/suspicion state is owned by the watchdog, classifier, tracker, and 1c.3 notification plane. It never directly changes the committed data-plane mask.
- Committed membership is owned by the 1c.4b recovery coordinator. It publishes the common `ActiveRankMap`, mask, and generation only after placement, survivor control/data communicators, and graph policy are ready.

## Source-of-truth precedence

1. The design invariants and recovery state machine define correct behavior.
2. The implementation plan defines work-item scope and hard dependencies.
3. The dependency graphs prove live status, edge state, and the current action frontier.
4. The JIRA ledger preserves planning workflow and assignee metadata without overriding code/PR state.
5. Live GitHub is authoritative for the current PR head, draft/review/merge decision, CI, and DCO status.

If any view diverges, stop downstream PR creation, record the mismatch in the correction checklist, and update every affected view before treating the plan as review guidance.
