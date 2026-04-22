# Benchmark Disaggregated-Serving Fill Mechanism

This directory tracks the full lineage of the benchmark disaggregated-serving fill/gate mechanism in `PyExecutor`: what exists today, why it regressed, how to fix it short-term, and how to eliminate the class of bug long-term.

## Current state (2026-04-22)

Status: **regressing in post-merge CI**. The wide-EP Kimi-K2-Thinking gen-only test (`perf/test_perf_sanity.py::test_e2e[disagg-gen_only-wideep_kimi-k2-thinking-fp4_8k1k_ctx8_gen1_dep32_bs256_eplb416_mtp0_con8192_ccb-NIXL]`) and the K2.5 variant (nvbug 6071070) hang and fail with `RequestError: Insufficient KV cache for gen-only benchmark mode`. Root cause traced to [PR #12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) (author: Chien-Chun Hung). The regression is not a KV-capacity shortage — it is an unsatisfiable gate-completion condition interacting with `±1` ADP router imbalance.

See [`02-regression-investigation.md`](02-regression-investigation.md) for the full causal chain.

## Lineage of the feature

| Version | PR | What it introduced | State |
|---|---|---|---|
| v0 | prior to #12091 | Single blocking fill loop — fetched all requests at once before first forward | Deadlocks under small-CTX configs |
| v1a | [#12091](https://github.com/NVIDIA/TensorRT-LLM/pull/12091) | Batched fill (`tp_size` per iteration) | Fixed some deadlock cases; still starved transfer servicing |
| v1b | [#12206](https://github.com/NVIDIA/TensorRT-LLM/pull/12206) | Explicit fail-fast when GEN-side KV cache insufficient | Kept — needed to avoid silent hangs |
| **v2** | [#12208](https://github.com/NVIDIA/TensorRT-LLM/pull/12208) | Eliminated fill loop; non-blocking `can_forward` gate; dummy suppression during fill | **Current. Regressing.** Docs in [`01-history-nonblocking-gate/`](01-history-nonblocking-gate/README.md) |
| v3 | this plan, step 1 | Gate-condition rewrite based on request state, not counts | To be implemented |
| v4 | this plan, step 2 | Remove fill gate from `PyExecutor`; orchestrate from benchmark client | To be implemented |

## Documents

| File | Purpose | Audience |
|---|---|---|
| [`01-history-nonblocking-gate/`](01-history-nonblocking-gate/README.md) | Original v2 design (PR #12208). Preserved as-is for historical context. | Review / archaeology |
| [`02-regression-investigation.md`](02-regression-investigation.md) | Root-cause analysis of the post-v2 regression. | Engineer or AI implementing step 1 — read first, in full |
| [`03-step1-gate-rewrite-plan.md`](03-step1-gate-rewrite-plan.md) | Code-level plan for v3. New fill-complete predicate, dummy handling changes, fail-fast trigger update, test additions. | Implementer |
| [`04-step2-external-orchestrator-plan.md`](04-step2-external-orchestrator-plan.md) | Architecture-level plan for v4. Deletes the feature from `PyExecutor` entirely; moves orchestration to the benchmark client. | Design reviewer first, then implementer |

## Reading order

For context only: skim `01-history-nonblocking-gate/README.md`.

For implementing the fix: read `02` in full → read `03` in full → execute `03`.

For the structural redesign: read `03` (so you know what's being removed) → read `04` → discuss before starting.

## Relationship between the two steps

Step 1 (v3) is a bounded patch to `PyExecutor`. It keeps the feature in the same place it has lived through versions 0–2 and fixes its correctness. Ships as a normal bug-fix PR.

Step 2 (v4) deletes the feature from `PyExecutor`. The gate becomes a client-side barrier in the benchmark harness, because that is what the gate logically is — measurement orchestration. Ships as a larger refactor once step 1 is stable in CI.

Step 1 ships first for two reasons: it unblocks current CI failures, and it stabilizes the contract that step 2 is going to remove, making the removal cleaner.
