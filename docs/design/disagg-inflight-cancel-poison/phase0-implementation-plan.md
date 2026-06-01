# Phase 0 — Implementation Plan

|     |     |
| --- | --- |
| **Phase** | 0 (prerequisite to Phases 1–4) |
| **JIRA**  | [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) tied to [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Spec**  | [`phase0-stress-test-suite.md`](phase0-stress-test-suite.md) |
| **Owner** | Chien-Chun Hung |
| **Status**| Skeleton landed in `upstream/main` (steps 1–3 below). Thread bodies are implemented incrementally (step 4): log_scanner is live; metrics_thread is next, followed by injector, canary, load. |

## Purpose of this document

The spec ([`phase0-stress-test-suite.md`](phase0-stress-test-suite.md))
describes **what** to build and **what counts as pass/fail**. This
document captures **how** we plan to build it: open-question
resolutions, PR split, build sequencing, risk-mitigation choices, and
any decisions that go beyond what the spec explicitly states.

This is a working document. Open questions resolve over time as
implementation progresses. Each entry should be dated when answered so
a re-pickup or another implementer can reconstruct the timeline.

## Open implementer questions

The spec ends with seven explicit questions ("Open questions for the
implementing agent to resolve"). This section answers them as they're
resolved, plus four additional questions raised in the implementation
conversation.

### Q1 (spec) — `run_cancel_stress_test` integration

> The function exits when its inner `asyncio.run(...)` completes. For
> a marathon, it must loop until `stop_event` is set.

**Spec recommendation:** wrap the existing function in
`while not stop_event:` in the harness's `load_thread`.

**Decision:** TBD — verify the recommended option works as-is by
reading the function. Default to the recommendation unless there's a
clean reason not to (e.g., the function carries per-invocation state
that conflicts with restart-in-a-loop).

### Q2 (spec) — burst vs. steady-state mixing

> The current `run_cancel_stress_test` is burst-only. Either extend it
> to also send steady-state traffic between bursts, or add a separate
> `run_steady_state_load` companion.

**Decision:** TBD — leaning toward a sibling `run_steady_state_load`
that the harness alternates with the existing burst function. The
existing function is used by `test_disaggregated_cancel_large_context_requests`
in CI; we should not change its observable semantics under that test.

### Q3 (spec) — V1/V2 KV cache manager field name

> Verify the exact field name and supported values by reading
> `tensorrt_llm/_torch/pyexecutor/resource_manager.py` and the
> surrounding config classes.

**Decision:** TBD — to be resolved during step 1 of the build
sequencing (read-only exploration). Update this entry with the
concrete field name + accepted values once known.

### Q4 (spec) — C++ vs Python transceiver selection

> Controls whether the NIXL agent goes through C++ bindings or pure
> Python. Verify the exact selection mechanism (env var? config
> field?).

**Decision:** TBD — to be resolved during step 1. Likely candidates:
an env var, a `cache_transceiver_config.runtime` field, or dispatch
logic in `_torch/disaggregation/nixl/_agent_{cpp,py}.py`. Update with
the concrete mechanism once known.

### Q5 (spec) — `setup_disagg_cluster` worker-handle for SIGKILL + relaunch

> Verify that the handle supports SIGKILL + relaunch, or that it can
> be extended to.

**Decision:** TBD. Two paths:
- **(a) Full respawn.** Extend the worker-handle class with a
  `relaunch()` method that records launch command + env + log file at
  startup. Higher fidelity to the spec's `respawn_within_s: 60`
  contract.
- **(b) Kill-only fallback.** Spec explicitly permits this:
  > If this turns out to be too invasive, the alternative is to limit
  > the SIGKILL injection to the kill-only step (skip the respawn) and
  > verify that the *remaining* workers absorb the load.

**Default to (a) if the existing handle exposes the launch command +
env + log path (a few-hour extension). Fall back to (b) if (a) would
require a significant refactor.** Document the choice taken in the
test README.

### Q6 (spec) — model selection for both marathons

Constraints:
1. Available in `$LLM_MODELS_ROOT` on CI runners.
2. Big enough to make KV transfer non-trivial (so the cancel-mid-flight
   race window is reachable).
3. Fits 3P3D + clients on a single 8-GPU node at TP=1 per worker.
4. Matches archetype: Marathon A = "DeepSeek-class MLA"; Marathon B =
   "Qwen-class non-MLA".

**Candidates to evaluate** (TBD — pending CI model cache enumeration):

- Marathon A (MLA): `DeepSeek-V3-Lite-bf16` may be too small;
  `DeepSeek-R1-Distill-Llama-8B` is a reasonable size but isn't MLA.
  A medium-sized MLA model is the open question.
- Marathon B (non-MLA): `Qwen2-7B-Instruct`, `Qwen3-8B-Instruct`,
  `Qwen3-Coder-Mini`.

### Q7 (spec) — greedy-decode determinism

> Confirm that the TRT-LLM PyTorch backend produces deterministic
> outputs under greedy decoding + fixed seed across runs.

**Decision:** TBD — confirm empirically by running the same prompt
twice on the same engine and comparing token IDs. Fallback chain if
non-deterministic — see Q10.

### Q8 (impl) — serial vs. parallel marathon execution

Spec says "single 8-GPU node" and the test budget is "4 h total"
(2 × 2 h). Each marathon uses 6 GPUs for 3P3D + spare for clients;
two in parallel exceeds an 8-GPU node.

**Decision: serial execution.** Pytest parametrize runs serially by
default; the parametrized test (over the two YAML configs) takes the
documented 4 h end-to-end on one node. No deviation from the spec
needed; this captures the implicit assumption explicitly.

### Q9 (impl) — initial-PR scope for SIGKILL+respawn

Tied to Q5. The spec defines the test contract assuming respawn
within 60 s. If Q5 lands on path (b) (kill-only), the test asserts
"remaining 5 workers absorb load" instead of "respawned worker rejoins
within 60 s" for the T+60 injection.

**Decision:** Document the choice explicitly in the test README and
in this doc when Q5 resolves. If we ship (b), file a follow-up issue
for (a).

### Q10 (impl) — canary token-equivalence fallback if Q7 is non-deterministic

Fallback ordering, preferred → least preferred:
1. **(a) Exact text equivalence after detokenize.** Preserves most of
   the UAF-detection signal (catches arbitrary mid-string corruption).
2. **(b) BLEU / ROUGE threshold.** Catches gross corruption but not
   single-token UAF.
3. **(c) Length-only sanity check.** Catches only response-truncation
   or response-explosion.

**Decision: prefer (a).** If forced to (b) or (c), document the
weakened UAF-detection guarantee in the test README.

### Q11 (impl) — PR split

**Two PRs:**

**PR1 — Phase 0 harness + Marathon A end-to-end.**
- New directory `tests/integration/defs/stress_test/disagg_cancel/`.
- `harness.py` with `DisaggCancellationStressHarness` and all five
  thread implementations.
- `test_disagg_cancel_stress.py` with the parametrized pytest test
  (parametrize list initially contains Marathon A only).
- `configs/marathon_a_v1_cpp_deepseek.yaml`.
- `configs/stress_canary_prompts.json` (canary prompts + reference
  token IDs for Marathon A's model only).
- `tools/generate_canary_references.py`.
- Directory-level `README.md`.
- Test ID registered in `tests/integration/test_lists/qa/llm_function_stress.txt`.
- Acceptance: ≥1 full 2-h Marathon A run on developer machine, PASS,
  attached to PR description.

**PR2 — Marathon B (V2 + Python transceiver).**
- `configs/marathon_b_v2_py_qwen.yaml`.
- Canary references for Marathon B's model.
- Any V2- or Python-transceiver-specific branches in `harness.py`
  (e.g., different metric name, different log markers).
- Second test ID in `llm_function_stress.txt`.
- Acceptance: ≥1 full 2-h Marathon B run, PASS, attached to PR
  description.

PR1 is the load-bearing change (harness module, schema, infrastructure,
canary tooling). PR2 is a parametric add-on.

**Subsequent follow-up PRs** (per the spec's "Deferred" section): each
of 1P1D, 4P2D, V1+Python, UCX, block-reuse-off, overlap-off, 1 s
aggressive timeout, multi-node lands as a single new YAML + test-list
registration with no Python changes required.

## Build sequencing (PR1)

| Step | Action | Outputs | Status |
|------|--------|---------|--------|
| 1 | Read-only exploration: resolve Q3, Q4, Q5 by reading `resource_manager.py`, `_torch/disaggregation/nixl/_agent_{cpp,py}.py`, `ProcessWrapper` / `run_ctx_worker` / `run_gen_worker`. | Updated Q3/Q4/Q5 entries in this doc; brief findings summary in the implementation conversation. | Done |
| 2 | Model selection: enumerate `$LLM_MODELS_ROOT`, pick Marathon A candidate, sanity-check VRAM + KV transfer time. | Updated Q6 entry. | Done |
| 3 | Harness skeleton: directory structure, stub `DisaggCancellationStressHarness` class with thread fields + lifecycle methods (`start`, `stop`, `wait_until_done`), stub YAML schema parsing, stub pytest test. | Compiles; passes a smoke `harness.start(); harness.stop()` with no threads doing real work. | Done (landed in `upstream/main`) |
| 4 | Implement thread bodies (simplest first): log_scanner → metrics → injector → canary → load. | Each thread independently testable; harness assembles them. | In progress — `log_scanner` done; `metrics` next |
| 5 | Marathon A YAML + canary references: write YAML using resolved Q3/Q4 values; run `generate_canary_references.py` once against Marathon A's model; commit the JSON. | `marathon_a_v1_cpp_deepseek.yaml`, `stress_canary_prompts.json`, generator tool. | Not started |
| 6 | Validation: full 2-h Marathon A run on developer machine. | PASS log attached. | Not started |
| 7 | Registration: register both marathons in `llm_function_stress.txt`. | Test IDs picked up by weekly stress CI. | Not started |

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| 2-h marathon too long for local iteration | Optional `--smoke` mode that runs a 10-min marathon with the same shape (1 burst + 1 injection). Documented in test README; not registered in CI. |
| SIGKILL+respawn API surface is invasive | Q5 fallback to kill-only variant; spec explicitly permits. |
| Greedy-decode non-determinism breaks canary | Q10 fallback chain; document weakened guarantee if forced to (b) or (c). |
| Prometheus metric scrape format changes | Use `prometheus_client.parser.text_string_to_metric_families`; pin metric name (`trtllm_kv_cache_utilization`) and fail loudly if absent. |
| Worker log paths / formats change across releases | Scan all stdout/stderr captured by `setup_disagg_cluster`'s worker handles; do not hard-code paths. |
| 3P3D × 8-GPU node insufficient VRAM headroom | Q6 model selection sanity-checks this. Fallback: smaller model. |
| Pre-commit hooks (D205, ruff-legacy, clang-format) | Write code to lint-clean from the start; run pre-commit locally before each commit. |

## Decisions beyond the spec

These are choices the implementation makes that the spec does not
state explicitly. Captured here so a re-pickup can find them.

- **Serial marathon execution** on a single 8-GPU node (Q8).
- **Two-PR split** (Q11). Spec's acceptance checklist treats
  everything as one PR; we split for review tractability.
- **`--smoke` mode** for developer iteration (under Risks). Optional
  flag; not in CI.
- **Kill-only SIGKILL fallback** (Q5 / Q9) is an acceptable initial
  shape per the spec; promotion to full respawn is a follow-up if
  needed.

## Out of scope for PR1 + PR2

Per the spec's "What's deferred" section, all of these are tracked as
future YAML + test-list additions with no Python changes:

- 1P1D, 4P2D, V1+Python, UCX, block-reuse-off, overlap-off, 1 s
  aggressive timeout, multi-node cross-node.

## Cross-references

- [`phase0-stress-test-suite.md`](phase0-stress-test-suite.md) — the
  spec this implements.
- [`README.md`](README.md) — overall TRTLLM-12721 design (Phases 0–4).
- [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/10-ablation-no-midflight-cancel.md)
  — the §10 ablation experiments this suite generalises.
- <https://github.com/NVIDIA/TensorRT-LLM/pull/13713> — the
  disaggregated cancellation / poison bug fix this suite gates
  regressions against.
- [TRTLLM-12648](https://jirasw.nvidia.com/browse/TRTLLM-12648) —
  weekly stress CI ticket.
- [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) —
  cancellation/poison improvement initiative this is Phase 0 of.
