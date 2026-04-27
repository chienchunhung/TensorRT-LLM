# Benchmark Disaggregated-Serving Fill Mechanism

This directory tracks the full lineage of the benchmark disaggregated-serving fill/gate mechanism in `PyExecutor`: what exists today, why it regressed, how to fix it short-term, and how to eliminate the class of bug long-term.

## Current state (2026-04-26)

Status: **three-part fix in review in [PR #13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347)**. The wide-EP Kimi-K2-Thinking gen-only test exposed three coupled issues:

1. The original count-based fill gate depended on exact ADP router balance.
2. The ADP router could set `expected_num_active_requests` above `max_batch_size` during bulk arrivals.
3. The PR #12206 fail-fast could fire during the benchmark fill phase before the state-based gate had a chance to open.

The current PR fixes all three: state-based gate predicate, ADP router per-rank cap, and fill-phase fail-fast suppression.

See [`02-regression-investigation.md`](02-regression-investigation.md) for the full causal chain.

## Lineage of the feature

| Version | PR | What it introduced | State |
|---|---|---|---|
| v0 | prior to #12091 | Single blocking fill loop — fetched all requests at once before first forward | Deadlocks under small-CTX configs |
| v1a | [#12091](https://github.com/NVIDIA/TensorRT-LLM/pull/12091) | Batched fill (`tp_size` per iteration) | Fixed some deadlock cases; still starved transfer servicing |
| v1b | [#12206](https://github.com/NVIDIA/TensorRT-LLM/pull/12206) | Explicit fail-fast when GEN-side KV cache insufficient | Kept — needed to avoid silent hangs |
| **v2** | [#12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) | Eliminated fill loop; non-blocking `can_forward` gate; dummy suppression during fill | Merged. Docs in [`01-history-nonblocking-gate/`](01-history-nonblocking-gate/README.md) |
| **v2.1** | [#13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347) | State-based fill gate, ADP router cap, fill-phase fail-fast suppression | **In review.** Fixes nvbug 6071070 / nvbug 6093911. Docs in [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md), [`05-router-cap-fix.md`](05-router-cap-fix.md), and [`06-fill-phase-fail-fast.md`](06-fill-phase-fail-fast.md) |
| v3 | follow-up if needed | Separate admission control from routing; harden orchestration boundaries | Planned follow-up, not required for the current PR |
| v4 | this plan, step 2 | Remove fill gate from `PyExecutor`; orchestrate from benchmark client | Planned |

## Documents

| File | Purpose | Audience |
|---|---|---|
| [`01-history-nonblocking-gate/`](01-history-nonblocking-gate/README.md) | Original v2 design (PR #12208). Preserved as-is for historical context. | Review / archaeology |
| [`02-regression-investigation.md`](02-regression-investigation.md) | Root-cause analysis of the post-v2 regression. | Engineer or AI implementing step 1 — read first, in full |
| [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md) | Code-level plan for v3. New fill-complete predicate, dummy handling changes, fail-fast trigger update, test additions. | Implementer |
| [`04-step2-external-orchestrator-plan.md`](04-step2-external-orchestrator-plan.md) | Architecture-level plan for v4. Deletes the feature from `PyExecutor` entirely; moves orchestration to the benchmark client. | Design reviewer first, then implementer |
| [`05-router-cap-fix.md`](05-router-cap-fix.md) | v2.1: ADP router cap fix — deterministic one-line fix at the admission layer. | Reviewer / implementer |
| [`06-fill-phase-fail-fast.md`](06-fill-phase-fail-fast.md) | v2.1: why PR #12206 fail-fast must be suppressed during benchmark fill. | Reviewer / implementer |

## Reading order

For context only: skim `01-history-nonblocking-gate/README.md`.

For reviewing the current PR: read `02` for the regression, then `03`, `05`, and `06` for the three production fixes.

For the structural redesign: read `03` (so you know what's being removed) → read `04` → discuss before starting.

## Relationship between the two steps

Step 1 (v3) is a bounded patch to `PyExecutor`. It keeps the feature in the same place it has lived through versions 0–2 and fixes its correctness. Ships as a normal bug-fix PR.

Step 2 (v4) deletes the feature from `PyExecutor`. The gate becomes a client-side barrier in the benchmark harness, because that is what the gate logically is — measurement orchestration. Ships as a larger refactor once step 1 is stable in CI.

Step 1 ships first for two reasons: it unblocks current CI failures, and it stabilizes the contract that step 2 is going to remove, making the removal cleaner.
