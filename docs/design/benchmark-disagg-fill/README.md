# Benchmark Disaggregated-Serving Fill Mechanism

This directory tracks the full lineage of the benchmark disaggregated-serving fill/gate mechanism in `PyExecutor`: what exists today, why it regressed, how to fix it short-term, and how to eliminate the class of bug long-term.

## Current state (2026-04-27)

Status: **v2.1 + v2.2 four-part fix in review in [PR #13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347).** The wide-EP Kimi-K2-Thinking gen-only test exposed four coupled issues that the PR collectively fixes:

1. The original count-based fill gate depended on exact ADP router balance. *(v2.1 — state-based gate.)*
2. The ADP router could set `expected_num_active_requests` above `max_batch_size` during bulk arrivals. *(v2.1 — router cap.)*
3. The PR #12206 fail-fast could fire during the benchmark fill phase before the state-based gate had a chance to open. *(v2.1 — fill-phase fail-fast suppression.)*
4. With the gate predicate fixed, a latent issue surfaced: the executor admits up to `tp_size × max_batch_size` requests in a single iteration, which spikes peak KV-cache + recv-buffer reservations and OOM-kills the GEN server on tight memory budgets. *(v2.2 — fill-phase admission cap.)*

A separate latent issue, `08-fill-phase-stuck-state-finding.md`, was discovered while validating v2.2 locally. It is independent of v2.1 / v2.2 and is filed as a follow-up.

See [`02-regression-investigation.md`](02-regression-investigation.md) for the v2.1 causal chain and [`07-fill-phase-flow-control.md`](07-fill-phase-flow-control.md) for the v2.2 causal chain.

## Lineage of the feature

| Version | PR | What it introduced | State |
|---|---|---|---|
| v0 | prior to #12091 | Single blocking fill loop — fetched all requests at once before first forward | Deadlocks under small-CTX configs |
| v1a | [#12091](https://github.com/NVIDIA/TensorRT-LLM/pull/12091) | Batched fill (`tp_size` per iteration) | Fixed some deadlock cases; still starved transfer servicing |
| v1b | [#12206](https://github.com/NVIDIA/TensorRT-LLM/pull/12206) | Explicit fail-fast when GEN-side KV cache insufficient | Kept — needed to avoid silent hangs |
| **v2** | [#12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) | Eliminated fill loop; non-blocking `can_forward` gate; dummy suppression during fill | Merged. Docs in [`01-history-nonblocking-gate/`](01-history-nonblocking-gate/README.md) |
| **v2.1** | [#13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347) | State-based fill gate, ADP router cap, fill-phase fail-fast suppression | **In review** as part of PR #13347. Fixes nvbug 6071070 / nvbug 6093911. Docs in [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md), [`05-router-cap-fix.md`](05-router-cap-fix.md), and [`06-fill-phase-fail-fast.md`](06-fill-phase-fail-fast.md) |
| **v2.2** | [#13347](https://github.com/NVIDIA/TensorRT-LLM/pull/13347) | Fill-phase admission cap (`tp_size` per iteration) reintroduced as explicit memory-pressure regulator | **In review** as part of PR #13347 (4th commit). Fixes burst-admission OOM that v2.1 uncovered on tight-memory configs. Docs in [`07-fill-phase-flow-control.md`](07-fill-phase-flow-control.md) |
| v3 | follow-up if needed | Separate admission control from routing; harden orchestration boundaries | Planned follow-up |
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
| [`07-fill-phase-flow-control.md`](07-fill-phase-flow-control.md) | v2.2: per-iteration admission cap (`tp_size`) to bound fill-phase peak memory. | Reviewer / implementer |
| [`08-fill-phase-stuck-state-finding.md`](08-fill-phase-stuck-state-finding.md) | Latent finding: gate-vs-transceiver deadlock observed during local v2.2 validation. Independent of v2.1 / v2.2. | Reviewer; file as separate NVBug |

## Reading order

For context only: skim `01-history-nonblocking-gate/README.md`.

For reviewing PR #13347 (which includes both v2.1 and v2.2): read `02` for the regression, then `03`, `05`, `06`, and `07` for the four production fixes (in commit order). Skim `08` to understand the latent gate-vs-transceiver finding that was discovered during v2.2 validation but is *not* part of this PR.

For the structural redesign: read `03` (so you know what's being removed) → read `04` → discuss before starting.

## Relationship between the steps

Step 1 (v2.1 + v2.2) is a bounded patch to `PyExecutor`. It keeps the feature in the same place it has lived through versions 0–2 and fixes its correctness and resource-budget behavior. Ships as a single PR (#13347) that contains all four commits — v2.1 first (state correctness), then v2.2 (memory-pressure regulator) on top — because the PR's CI gate cannot pass without v2.2 anyway.

Step 2 (v4) deletes the feature from `PyExecutor`. The gate becomes a client-side barrier in the benchmark harness, because that is what the gate logically is — measurement orchestration. Ships as a larger refactor once Step 1 is stable in CI.

Step 1 ships first for two reasons: it unblocks current CI failures, and it stabilizes the contract that Step 2 is going to remove, making the removal cleaner.
