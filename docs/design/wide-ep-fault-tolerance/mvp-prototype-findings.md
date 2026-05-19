# WideEP FT MVP Prototype — Findings

[< Back to Overview](README.md) • [Prototype plan](mvp-prototype-plan.md) • [Audit 1a findings](audit-1a-findings.md)

**Status:** Living document — updated as the prototype runs. • **Owner:** WideEP FT track • **Last updated:** 2026-05-18

This file collects the seam-contract issues, performance surprises, and integration-risk discoveries surfaced by running the throwaway scaffolding at `prototypes/wide_ep_ft_mvp/` on the [`WideEP-FT/mvp-prototype`](https://github.com/chienchunhung/TensorRT-LLM/tree/WideEP-FT/mvp-prototype) branch (preview draft [PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198)).

Per [`mvp-prototype-plan.md` §9](mvp-prototype-plan.md#9-after-the-prototype), each finding here feeds back into the design of a specific production PR — the prototype's value is exactly this list.

---

## F1. The watchdog's completion-flag view must NOT require peer participation

**Surfaced during:** initial scaffolding review (before first run).

**Where:** [`prototypes/wide_ep_ft_mvp/stubs/shm_completion_flags.py`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/stubs/shm_completion_flags.py) (the prototype's substitute) replaces an initial design that used `MPI.COMM_WORLD.allgather` as the completion-flag view.

**Why it matters.** If the watchdog's "read every peer's flag" path requires a collective involving the dead peer, the dead peer's non-participation blocks the watchdog and the watchdog can never fire. The MNNVL fabric-memory property "host can read peer flags without peer participation" is *load-bearing*, not incidental. This was almost lost in the scaffolding pass because `allgather` is the cheapest single-node substitute for "every rank's view of every other rank's counter" — but it has fundamentally wrong fault-tolerance semantics.

**Implication for production [PR 1a.4](08-implementation-plan.md#phase-1-pr-breakdown).** Whatever the production `AlltoAllWatchdog` uses to read the completion-flag table — MNNVL fabric memory, NVSHMEM-equivalent, or any future substitute — the read path **must be zero-collective**. Any alternative that requires peer participation must be rejected during PR 1a.4 design review. The MNNVL fabric-memory access pattern is the contract, not just an optimization.

**Prototype mitigation.** POSIX shared memory (`/dev/shm/wide_ep_ft_proto/run_<id>/rank_<i>.counter`); each rank `mmap`s its own counter file rw and peers' files ro. Single-node only, but preserves the zero-collective read property.

---

## F2. Survivors hang in `MPI_Finalize` after a peer SIGKILL

**Surfaced during:** first Level A smoke run on 8× B300 node, 2026-05-18.

**Symptom.** Driver injects SIGKILL on the victim at iteration N. Survivors complete their main loop, return from `main()`. Python interpreter shuts down. `mpi4py`'s `atexit` hook calls `MPI_Finalize`. **`MPI_Finalize` hangs forever** because it is a collective and the dead victim cannot participate. The driver never sees `loop_end` events from any survivor and times out.

This is exactly the [audit-1a Day 2 F4 finding](audit-1a-findings.md#day-2--mpi-signal-handler--exit-mitigation) ("survivors hang in their next collective after a peer death") manifesting at *process-shutdown time* rather than at the next in-loop collective. The audit prototype exercised it within an `Allreduce` loop; this prototype shows it also fires through `MPI_Finalize` on normal shutdown.

**Why it matters.** Even with the [1d.0 signal-handler replacement (PR #14160)](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) installed, even with `--mca orte_enable_recovery 1` on `mpirun`, even with the watchdog → broadcast → reconfigure cascade working perfectly, **survivors will silently hang on shutdown**. There is no error message, no signal, no diagnostic — just `MPI_Finalize` never returning. From the operator's perspective the cluster simply stops responding when the inference job tries to end.

**Implication for production PRs.** The fix is a Python-side coordinated shutdown path that detects "we are in a poisoned-world state" and calls `_exit(0)` instead of letting `MPI_Finalize` run. Distinct ownership questions:

- **[PR 1d.0](08-implementation-plan.md#phase-1-pr-breakdown) (signal-handler replacement, in flight as PR #14160).** The current C++ handler installs `_exit(137)` on `SIGABRT`/`SIGSEGV` — but normal post-cascade shutdown doesn't go through a signal. 1d.0's scope is *signal-time* shutdown; the *atexit-time* shutdown belongs to a different PR. **No 1d.0 change required for F2** — flagging here so reviewers don't conflate the two.
- **[PR 1c.3](08-implementation-plan.md#phase-1-pr-breakdown) (MPI FT subcomm + broadcast thread).** Likely owns the "is the world poisoned?" check, since it already tracks active vs. failed ranks in the FT subcomm. **Suggest adding:** `MpiFtSubcomm.world_is_poisoned() -> bool` for the shutdown path to consult.
- **[PR 1c.4](08-implementation-plan.md#phase-1-pr-breakdown) (model engine health-check hook).** The cleanest place to install the shutdown handler is wherever the model engine wires its main loop — the same place that registers the iteration-boundary hook can register an `atexit` (or equivalent) that consults `world_is_poisoned()` and `_exit(0)`s if true, otherwise lets normal shutdown proceed. **This is where the production fix for F2 should live.**

**Prototype mitigation.** The worker calls `os._exit(0)` unconditionally after the main loop's `loop_end` event. See [`prototypes/wide_ep_ft_mvp/scripts/kill_and_survive_worker.py`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/scripts/kill_and_survive_worker.py) — the inline comment explicitly cites this finding. Production must be conditional (`_exit` only if `world_is_poisoned`, else `return` normally) so legitimate shutdowns still run `MPI_Finalize` cleanly.

**Open sub-questions for production design.**

- Should the "skip `MPI_Finalize`" decision happen at every rank independently (based on local `EPGroupHealth.has_failures()`), or via a broadcast from one designated rank? Independent decisions are simpler but assume every survivor has seen the same broadcast — already a PR 1c.3 invariant for single-failure but may not hold under multi-failure (PR 1c.6).
- Does `--mca orte_enable_recovery 1` change `MPI_Finalize`'s behavior in any way? Audit 1a Day 2 did not test this specifically; worth a focused micro-prototype if PR 1c.3 doesn't already validate it.
- Are there other implicit collectives in the inference shutdown path (e.g. CUDA context destroy on driver-mapped MNNVL memory)? This prototype only exercises the MPI shutdown path; the CUDA-driver path is Audit 1a Day 3's territory but wasn't tested under "survivor with poisoned MNNVL mapping" conditions. **Defer to [Audit 1b](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access).**

---

## Pending findings

The Level A smoke run is incomplete (driver timed out before writing the JSON timeline; F2 was the symptom). After patching F2's prototype mitigation, the following are still pending:

- [ ] Successful end-to-end Level A run: full per-event timeline (`t_kill` → `t_first_new_request_completed`) for `--np 4` and `--np 8`.
- [ ] Watchdog vs. NCCL collective ordering (Open Question 1 from [mvp-prototype-plan.md §8](mvp-prototype-plan.md#8-open-questions-for-the-prototype-to-answer)).
- [ ] Iteration-boundary semantics (Open Question 2): does the engine check `EPGroupHealth.generation` before or after fwd setup?
- [ ] Detection latency reality check (Open Question 4): is the 5 s default watchdog timeout right for this hardware?
- [ ] Seam-stressing kill points (during dispatch / combine / routing / EPLB-stride) — blocked on the kernel-side 1a.2/1a.3 integration; see [`prototypes/wide_ep_ft_mvp/kernel/README.md`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/kernel/README.md).
