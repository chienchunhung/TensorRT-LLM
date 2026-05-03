# NVBug 6104831 — Permanent Disaggregated-Serving Wedge in `rc11`

- **Severity:** P0 / Critical
- **Affected component:** Disaggregated serving (`trtllm-serve` context worker
  + generation worker + disaggregated front-end), `rc11` baseline
- **Affected backend:** PyTorch executor, NIXL/UCX KV-cache transceiver
- **Symptom (customer-facing):** Local 1P1D `trtllm-serve` deployment serves
  the first burst of requests, then stops responding. All probes after the
  burst hit `ReadTimeout`. Workers stay alive (no crash, no exit), but the
  generation event loop never recovers.
- **Origin signal:** Dynamo + TRT-LLM `rc11` deployment hang, three apparent
  crash signatures observed in the field.

---

## Status (current)

The strongest candidate stack is the **combo approach** (Approach D):

```text
rc11
+ PR #13056   (architectural lifetime / cancellation refactor)
+ PR #13495   (transfer-release cancellation hook)
+ eval-order fix in CacheSender::Impl::handleAsyncSend
+ Python idempotency guards in _prepare_disagg_gen_init() and _recv_disagg_gen_cache()
```

Submitted as PR [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713).

Latest local results, 1P1D `trtllm-serve` long-prompt burst harness on a
single host:

| Transport | `CONC=16` | `CONC=24` | `CONC=32` | `CONC=64` |
|---|---|---|---|---|
| Direct UCX | 5/5 recovered (60 s + 90 s) | 5/5 recovered (60 s + 90 s) | 5/5 recovered (90 s) | wedged (no recovery 180 s) |
| NIXL + UCX plugin | n/a | n/a | 5/5 recovered, zero burst-time errors | 5/5 recovered, zero burst-time errors |

The customer-reported failure mode is *fixed on NIXL+UCX-plugin through
`CONC=64`*, with the only remaining failure being on TRT-LLM's *direct UCX*
path under sustained `CONC=64` stress. NIXL is the customer transport, so
the reporter's deployment shape is covered by the candidate stack.
Multi-node and full Dynamo orchestration validation are still pending.

---

## How to read this investigation

The investigation grew large because it traversed seven failure signatures,
four candidate fix approaches, and fourteen documented experiment phases.
The single-file form became hard to navigate, so the report is now split
across sections that are each meant to be readable on its own:

| File | When to read it |
|---|---|
| [`01-background.md`](01-background.md) | Read first if you are new to this code. Architecture diagrams of the disagg deployment, request lifecycle walkthrough, `LlmRequestState` state machine, key files / classes. |
| [`02-failure-signatures.md`](02-failure-signatures.md) | The seven concrete failure signatures (`#1`–`#7`): symptom, code site, root cause, fix, regression test. The "what bugs are there" view. |
| [`03-defect-class-stack.md`](03-defect-class-stack.md) | The eight-layer `L1`–`L8` defect-class model that emerged from the investigation. Re-frames the seven signatures as the customer-visible faces of underlying invariant gaps. **This is the framework used in the four-approach comparison.** |
| [`04-reproduction.md`](04-reproduction.md) | How to reproduce the wedge locally with `trtllm-serve` 1P1D, the load shape that matters, and what does *not* reproduce it. |
| [`05-investigation-timeline.md`](05-investigation-timeline.md) | Chronological journey: Phases 0 (field report) through 14 (current). Useful for understanding *how* the bug class was discovered and why each fix exposed the next, but not required to understand the current state. |
| [`06-fix-approaches/`](06-fix-approaches/) | The four candidate fix stacks (A, B, C, D), one file each, plus a comparison `README.md` that scores them against the `L1`–`L8` defect class stack. **The `README.md` here is the most important file in the report for picking a fix path.** |
| [`07-architectural-reflections.md`](07-architectural-reflections.md) | Why so many bugs were latent until now; the seven invariants the transceiver doesn't enforce; what we would do differently in retrospect. Reader-orientation for the long-term remediation conversation. |
| [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) | The chained PRs in flight, companion fixes, deadline-enforcement effort estimate, and outstanding work. Operational view for landing fixes. |

---

## Suggested reading paths

- **"I'm picking up this investigation cold and have 30 minutes."** Read this
  README, then [`01-background.md`](01-background.md), then
  [`06-fix-approaches/README.md`](06-fix-approaches/README.md). You will
  understand the topology, the defect-class layering, and the fix
  trade-offs. Skim the rest as needed.
- **"I need to land a fix this week."** Read
  [`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md) and
  [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md). You will
  know what is in the candidate PR stack, what is still required, and
  which validation gaps remain.
- **"I'm reviewing a single PR."** Read
  [`02-failure-signatures.md`](02-failure-signatures.md) for the signature
  the PR addresses, then
  [`03-defect-class-stack.md`](03-defect-class-stack.md) for which
  invariant layer it closes, then
  [`06-fix-approaches/README.md`](06-fix-approaches/README.md) for the
  surrounding context.
- **"I want to understand the design debt."** Read
  [`03-defect-class-stack.md`](03-defect-class-stack.md), then
  [`07-architectural-reflections.md`](07-architectural-reflections.md).
  Skip the timeline; the reflections are organised by invariant, not
  chronology.
- **"I'm doing a postmortem of how this was investigated."** Read
  [`05-investigation-timeline.md`](05-investigation-timeline.md) end-to-end,
  then the "What We Would Do Differently" section in
  [`07-architectural-reflections.md`](07-architectural-reflections.md).

---

## Branches and PRs at a glance

**Local worktrees** (cumulative repro under `local/rc11-disagg-repro`):

- `local/sig1-broken-promise-test`, `local/sig1-broken-promise-fix`
- `local/sig4-checkgen-nonblocking-test`, `local/sig4-checkgen-nonblocking-fix`
- `local/sig5-recv-cancelrequest-fulfill`
- `local/sig6-recv-buffer-leak` (chained on `local/sig1-broken-promise-fix`)

**Chained PRs** (one signature per pair where applicable):

| PR | Role |
|---|---|
| [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) | sig `#2` reproducer test |
| [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) | sig `#2` fix |
| [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) | sig `#1` reproducer test |
| [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) | sig `#1` fix |
| [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674) | sig `#4` reproducer test |
| [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671) | sig `#4` fix |
| [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672) | sig `#5` test + fix |
| [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673) | sig `#6` test + fix (chained on `#13640`) |

**Combo PR**: [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) — Approach D.

**Companion fixes** in `main` but not in `rc11`:

- [#13119](https://github.com/NVIDIA/TensorRT-LLM/pull/13119) — request-level
  error propagation (cleaner failure visibility, not a wedge fix).
- [#12718](https://github.com/NVIDIA/TensorRT-LLM/pull/12718) — fatal engine
  detection / pod restart (mitigation for silent wedges, not a wedge fix).

---

## A note on the cross-investigation reference

This bug shares the disaggregated-serving HTTP path with
[NVBug 6043291 (zombie worker pods)](../nvbug-6043291-zombie-worker-pods/README.md)
but has independent root causes:

- NVBug 6043291 — *the engine dies without anyone noticing*. Fix: detect
  fatal errors and surface them to the orchestrator.
- NVBug 6104831 (this investigation) — *no engine actually dies, but the
  disaggregated KV pipeline still deadlocks*. Fix: close the cancellation
  / lifetime / cleanup gaps in the transceiver.

Both bugs need their respective fixes for a fully healthy production
deployment. They do not block each other.
