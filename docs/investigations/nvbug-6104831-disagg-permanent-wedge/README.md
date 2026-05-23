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

The strongest candidate stack is the **combo approach** (Approach D),
now with the PR `#13728` fail-closed memory-safety policy folded in
and ported to the MLA send path:

```text
rc11
+ PR #13056   (architectural lifetime / cancellation refactor)
+ PR #13495   (transfer-release cancellation hook)
+ eval-order fix in CacheSender::Impl::handleAsyncSend
+ Python idempotency guards in _prepare_disagg_gen_init() and _recv_disagg_gen_cache()
+ PR #13728   (fail-closed on unquiesced disagg KV transfer)
+ MLA port    (poison-on-NIXL-throw + zero-copy guard in mlaCacheFormatter.cpp)
```

Submitted as PR [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713).

> **Default-OFF after merge with `upstream/main`.** The cancellation +
> poison + fail-closed surface, plus the Python-side timeout
> enforcement and the deferred-cleanup machinery that depends on it,
> are now gated behind a single opt-in environment variable
> `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL`. Default unset → pre-PR
> baseline behaviour for every gated point; customers hitting NVBug
> 6104831 opt in explicitly.
>
> The orthogonal lifetime / idempotency / RAII / eval-order fixes
> (sigs `#1`, `#5`, `#6`, `#7` and the always-on portion of `#4`)
> remain unconditional because they close baseline races that exist
> regardless of mid-flight cancellation.
>
> Why default-OFF, not default-ON: the deferred-cleanup logic in
> particular ("don't free Python resources while C++ transfer status
> is still in progress") is a per-rank decision in the V1 + C++
> transceiver path, which has no consensus story across TP, PP, or
> EP. Per-rank deferral conflicts with the consistency invariants
> those parallelism strategies depend on (TP allgather rank-batch
> divergence, PP termination retry, MTP scheduler state). The V2 +
> Python transceiver already enforces consensus via
> `_consensus_outcome` (CANCELLED/FAILED on any rank → global,
> COMPLETED only when all ranks agree); the V1 + C++ path does not.
> Architecturally correct deferred cleanup needs to be designed
> *with* the consensus story, not retrofitted on top of a per-rank
> decision. See
> [`10-ablation-no-midflight-cancel.md`](10-ablation-no-midflight-cancel.md#why-we-ship-default-off)
> for the empirical CI evidence (RC-1 MTP / RC-2 TP allgather / RC-3
> PP) and
> [the follow-up design doc](../../design/disagg-inflight-cancel-poison/README.md)
> for the architectural rethink.

Latest local results, 1P1D `trtllm-serve` long-prompt burst harness on a
single 8-GPU B300 host. Bold cells are the post-PR-#13728 reaffirmations;
the rest pre-date the fold-in:

| Transport | `CONC=16` | `CONC=24` | `CONC=32` | `CONC=64` | `CONC=128` (3-pair) | `CONC=256` (3-pair) |
|---|---|---|---|---|---|---|
| NIXL + UCX plugin | n/a | n/a | 5/5 recovered, zero burst errors | 5/5 recovered, zero burst errors | **5/5 recovered, zero burst errors (review-fix v3)** | **5/5 recovered, zero burst errors** |
| Direct UCX | 5/5 recovered (60 s + 90 s) | 5/5 recovered (60 s + 90 s) | 5/5 recovered (90 s) | wedged (no recovery 180 s) | wedged | n/a |

The customer-reported failure mode is *fixed on NIXL+UCX-plugin through
`CONC=256` with three ctx/gen pairs* on `rc11`, with the only remaining
failure being on TRT-LLM's *direct UCX* path above `CONC=32` (throughput
saturation, not a cancellation gap — see
[`06-fix-approaches/D-combo.md`](06-fix-approaches/D-combo.md#direct-ucx-saturation-evidence-diagnostic-build)).
NIXL is the customer transport, so the reporter's deployment shape is
covered by the candidate stack. Multi-node and full Dynamo orchestration
validation are still pending.

> **rc13 caveat (sig `#8` / L10):** the same combo regresses to a
> server hang when applied on top of `rc13`, because rc13 turns block
> reuse on by default and that surfaces a redundant cleanup-mechanism
> defect (the L10 layer in
> [`03-defect-class-stack.md`](03-defect-class-stack.md)). PR
> `#13713` must therefore land *with* a small stop-gap that always
> calls `_terminate_request` after `end_transfer()` returns true and
> dedupes via a `resources_freed` flag. The architectural follow-up
> is Phase 2 of the existing
> [block-reuse-overlap-scheduler design doc](../../design/block-reuse-overlap-scheduler/),
> which deletes the dual-path entirely (replaces
> `store_blocks_for_reuse(pin=True)` with `pin=False`, drops the
> `should_store_blocks` flag and the `unpin_blocks_by_id` call).
> Phase 15 of the timeline has the empirical evidence and the full
> plan. See
> [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) item 2
> for the staged landing plan.

> **Helix CI caveat (sig `#9` / L11):** PR `#13713` build `#39529`
> exposed a previously-latent rank-asymmetric Python gate defect in
> the disagg gen-side scheduling path. Three Python gates
> (`_prepare_disagg_gen_init`, `_recv_disagg_gen_cache`,
> `_check_disagg_gen_transfer_status`) each read rank-local state
> while guarding the only call chain into the C++ `gatherRequestIds`
> cross-rank allgather. Once PR `#13713`'s `shared_ptr<LlmRequest>`
> lifetime extension closes L2, the per-rank state divergence that
> used to crash loudly becomes silent — and the gates produce an
> ABBA deadlock against any downstream unconditional collective on
> the same gen-side ranks (`_can_queue::tp_allgather` under
> attention-DP, PP step-boundary collective under `gen_pp > 1`).
> Helix CI fired on the two parametrizations that meet both
> conditions (`pp1dp2cp2`, `pp2tp1cp2`); the other two
> (`pp1tp1cp4`, `pp1tp2cp2`) lack a downstream unconditional
> collective and so didn't deadlock despite carrying the same
> latent divergence. **Fix** (commit
> [`bdfdf8be02`](https://github.com/NVIDIA/TensorRT-LLM/pull/13713)
> on PR `#13713`): drop all three rank-asymmetric gates so every
> gen-side rank enters the C++ call together. The C++ side handles
> empty `mRequesterFutures` cheaply (one empty allgather, no
> inner-loop work). The same fix likely also resolves the
> `TIMEOUT (60)` masking on
> `TestQwen3NextInstruct::test_auto_dtype[use_py_transceiver=False]`
> (gen pp=2, NIXL → C++ transceiver — meets the trigger). See
> sig `#9` in [`02-failure-signatures.md`](02-failure-signatures.md)
> and L11 in [`03-defect-class-stack.md`](03-defect-class-stack.md).

---

## How to read this investigation

The investigation grew large because it traversed seven failure signatures,
four candidate fix approaches, and fourteen documented experiment phases.
The single-file form became hard to navigate, so the report is now split
across sections that are each meant to be readable on its own:

| File | When to read it |
|---|---|
| **[`00-tldr.md`](00-tldr.md)** | **Start here. 10-minute read.** Summarises the wedge symptom, the eight-layer root cause, the combo fix (PR `#13713`), and the empirical recovery results. Has the architecture and fix-mapping figures inline. Inspires the deeper reads if you want detail. |
| **[`09-executive-summary-rc11-to-rc13.md`](09-executive-summary-rc11-to-rc13.md)** | **15-minute read covering the full rc11 → rc13 journey.** Extends `00-tldr.md` with the rc13 regression chapter: why block reuse triggers it, what the short-term stop-gap covers and doesn't, why the design doc's Phase 2 deletion is the right long-term answer. Use this for an exec briefing or a new teammate joining mid-investigation. |
| [`01-background.md`](01-background.md) | Read first if you are new to this code. Architecture diagrams of the disagg deployment, request lifecycle walkthrough, `LlmRequestState` state machine, key files / classes. |
| [`02-failure-signatures.md`](02-failure-signatures.md) | The nine concrete failure signatures (`#1`–`#9`, with `#8` being the rc13 server hang under disagg + block reuse + in-flight cancel, and `#9` being the helix CI hang from rank-asymmetric Python gates over a cross-rank C++ collective): symptom, code site, root cause, fix, regression test. The "what bugs are there" view. |
| [`03-defect-class-stack.md`](03-defect-class-stack.md) | The eleven-layer `L1`–`L11` defect-class model that emerged from the investigation. Re-frames the nine signatures as the customer-visible faces of underlying invariant gaps. **This is the framework used in the four-approach comparison.** |
| [`04-reproduction.md`](04-reproduction.md) | How to reproduce the wedge locally with `trtllm-serve` 1P1D, the load shape that matters, and what does *not* reproduce it. |
| [`05-investigation-timeline.md`](05-investigation-timeline.md) | Chronological journey: Phases 0 (field report) through 14 (current). Useful for understanding *how* the bug class was discovered and why each fix exposed the next, but not required to understand the current state. |
| [`06-fix-approaches/`](06-fix-approaches/) | The four candidate fix stacks (A, B, C, D), one file each, plus a comparison `README.md` that scores them against the `L1`–`L8` defect class stack. **The `README.md` here is the most important file in the report for picking a fix path.** |
| [`07-architectural-reflections.md`](07-architectural-reflections.md) | Why so many bugs were latent until now; the seven invariants the transceiver doesn't enforce; what we would do differently in retrospect. Reader-orientation for the long-term remediation conversation. |
| [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md) | The chained PRs in flight, companion fixes, deadline-enforcement effort estimate, and outstanding work. Operational view for landing fixes. |
| **[`10-ablation-no-midflight-cancel.md`](10-ablation-no-midflight-cancel.md)** | **PR `#13713` value proposition with six-experiment empirical defence.** Organised around the three questions a reviewer asks first: **what fails, where** (four concrete failure modes with code sites — wedge in `NixlTransferStatus::wait`, broken-promise cascade in `dataTransceiver`, buffer-pool starvation in `baseTransBuffer`, silent UAF of an NIXL-pinned buffer); **why it fails** (a single missing invariant: cancellation must signal NIXL before the destination buffer is returned to the allocator, with four contributing API/lifecycle gaps); **how PR #13713 helps and why mid-flight cancellation is the keystone** (Layer 1 `release()` is the only application-level exit from NIXL's `getXferStatus` loop — without it, `kv_transfer_timeout_ms` is Python-only and cannot interrupt the C++ worker; Layers 2–5 close races and hazards that Layer 1 exposes/enables, including the fail-closed memory-safety policy via `has_poisoned_transfer_buffer`). Also frames the **value position**: PR #13713 is preventive, not reactive — it closes out an invariant class so that future load shapes, transports, and scheduler changes don't reopen this thirty-engineer-week investigation. Backed by six A/B experiments traversing the timeout-pressure spectrum (60 s production default through 1 s aggressive down to a SIGSTOP-induced peer pause), with Experiment 6 directly observing Layer 5 firing on head (HTTP 400 `PyExecutor has already been shutdown`). Honest-gaps section acknowledges what was not directly measured (ASan UAF, SIGKILL terminal failure, 60 s rc13-clean re-run). |

---

## Suggested reading paths

- **"I have 10 minutes."** Read [`00-tldr.md`](00-tldr.md). It
  has the wedge symptom, the L1–L8 root cause, the combo fix
  (PR `#13713`), and the empirical recovery results, with two
  inline figures. Pointers to deeper reads at the end of every
  section.
- **"I have 15 minutes and need to brief someone on the full rc11 → rc13
  journey."** Read [`09-executive-summary-rc11-to-rc13.md`](09-executive-summary-rc11-to-rc13.md).
  Adds the rc13 regression chapter, the short-term stop-gap, and the
  Phase 2 long-term plan on top of the 10-minute story.
- **"I'm picking up this investigation cold and have 30 minutes."** Read
  [`00-tldr.md`](00-tldr.md) first, then
  [`01-background.md`](01-background.md), then
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
- **"A reviewer asked whether PR `#13713`'s mid-flight cancellation,
  RAII, lifetime, idempotency, and fail-closed-on-unquiesced layers
  are really necessary on top of the deadline-eviction work."** Read
  [`10-ablation-no-midflight-cancel.md`](10-ablation-no-midflight-cancel.md).
  It answers, in this order: **what fails and where in the code**,
  **why it fails** (one missing invariant + four contributing API /
  lifecycle gaps), and **how PR #13713 helps — and why mid-flight
  cancellation is the keystone layer**. The keystone claim is that
  `release()` is the only application-level exit from NIXL's
  `getXferStatus` loop; without it, `kv_transfer_timeout_ms` is a
  Python-side timeout that cannot interrupt the C++ thread doing the
  transfer, and Layers 2–5 (RAII / lifetime / idempotency / fail-
  closed) close races and hazards that Layer 1 exposes or enables.
  The empirical backing is six A/B experiments traversing the timeout
  pressure spectrum, the strongest single piece being Experiment 6:
  on PR #13713 head, the fail-closed memory-safety policy was directly
  observed firing — converting a potential use-after-free into an
  explicit `PyExecutor has already been shutdown` HTTP 400. The
  section is also explicit about the **value position**: PR #13713 is
  preventive, not reactive — it closes out an invariant class so that
  future load shapes, transports, and scheduler changes don't reopen
  the investigation. Honest-gaps section acknowledges what the
  experiments do *not* directly show (no AddressSanitizer UAF
  detection, no SIGKILL injection for terminal failures) and frames
  those as follow-up gaps rather than load-bearing claims.

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
| [#13728](https://github.com/NVIDIA/TensorRT-LLM/pull/13728) | fail-closed on unquiesced disagg KV transfer (folded into `#13713`) |

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
