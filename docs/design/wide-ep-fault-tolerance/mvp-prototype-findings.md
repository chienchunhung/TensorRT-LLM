# WideEP FT MVP Prototype — Findings

[< Back to Overview](README.md) • [Prototype plan](mvp-prototype-plan.md) • [Audit 1a findings](audit-1a-findings.md)

**Status:** Living document — updated as the prototype runs. • **Owner:** WideEP FT track • **Last updated:** 2026-05-19

This file collects the seam-contract issues, performance surprises, and integration-risk discoveries surfaced by running the throwaway scaffolding at `prototypes/wide_ep_ft_mvp/` on the [`WideEP-FT/mvp-prototype`](https://github.com/chienchunhung/TensorRT-LLM/tree/WideEP-FT/mvp-prototype) branch (preview draft [PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198)).

Per [`mvp-prototype-plan.md` §9](mvp-prototype-plan.md#9-after-the-prototype), each finding here feeds back into the design of a specific production PR — the prototype's value is exactly this list.

---

## F1. The watchdog's completion-flag view must NOT require peer participation

**Surfaced during:** initial scaffolding review (before first run).

**Where:** [`prototypes/wide_ep_ft_mvp/stubs/shm_completion_flags.py`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/stubs/shm_completion_flags.py) (the prototype's substitute) replaces an initial design that used `MPI.COMM_WORLD.allgather` as the completion-flag view.

**Why it matters.** If the watchdog's "read every peer's flag" path requires a collective involving the dead peer, the dead peer's non-participation blocks the watchdog and the watchdog can never fire. The MNNVL fabric-memory property "host can read peer flags without peer participation" is *load-bearing*, not incidental. This was almost lost in the scaffolding pass because `allgather` is the cheapest single-node substitute for "every rank's view of every other rank's counter" — but it has fundamentally wrong fault-tolerance semantics.

**Implication for production [PR 1a.4](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown).** Whatever the production `AlltoAllWatchdog` uses to read the completion-flag table — MNNVL fabric memory, NVSHMEM-equivalent, or any future substitute — the read path **must be zero-collective**. Any alternative that requires peer participation must be rejected during PR 1a.4 design review. The MNNVL fabric-memory access pattern is the contract, not just an optimization.

**Prototype mitigation.** POSIX shared memory (`/dev/shm/wide_ep_ft_proto/run_<id>/rank_<i>.counter`); each rank `mmap`s its own counter file rw and peers' files ro. Single-node only, but preserves the zero-collective read property.

---

## F2. Survivors hang in `MPI_Finalize` after a peer SIGKILL

**Surfaced during:** first Level A smoke run on 8× B300 node, 2026-05-18.

**Symptom.** Driver injects SIGKILL on the victim at iteration N. Survivors complete their main loop, return from `main()`. Python interpreter shuts down. `mpi4py`'s `atexit` hook calls `MPI_Finalize`. **`MPI_Finalize` hangs forever** because it is a collective and the dead victim cannot participate. The driver never sees `loop_end` events from any survivor and times out.

This is exactly the [audit-1a Day 2 F4 finding](audit-1a-findings.md#day-2--mpi-signal-handler--exit-mitigation) ("survivors hang in their next collective after a peer death") manifesting at *process-shutdown time* rather than at the next in-loop collective. The audit prototype exercised it within an `Allreduce` loop; this prototype shows it also fires through `MPI_Finalize` on normal shutdown.

**Why it matters.** Even with the [1d.0 signal-handler replacement (PR #14160)](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) installed, even with `--mca orte_enable_recovery 1` on `mpirun`, even with the watchdog → broadcast → reconfigure cascade working perfectly, **survivors will silently hang on shutdown**. There is no error message, no signal, no diagnostic — just `MPI_Finalize` never returning. From the operator's perspective the cluster simply stops responding when the inference job tries to end.

**Implication for production PRs.** The fix is a Python-side coordinated shutdown path that detects "we are in a poisoned-world state" and calls `_exit(0)` instead of letting `MPI_Finalize` run. Distinct ownership questions:

- **[PR 1d.0](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (signal-handler replacement, in flight as PR #14160).** The current C++ handler installs `_exit(137)` on `SIGABRT`/`SIGSEGV` — but normal post-cascade shutdown doesn't go through a signal. 1d.0's scope is *signal-time* shutdown; the *atexit-time* shutdown belongs to a different PR. **No 1d.0 change required for F2** — flagging here so reviewers don't conflate the two.
- **[PR 1c.3](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (MPI FT subcomm + broadcast thread).** Likely owns the "is the world poisoned?" check, since it already tracks active vs. failed ranks in the FT subcomm. **Suggest adding:** `MpiFtSubcomm.world_is_poisoned() -> bool` for the shutdown path to consult.
- **[PR 1c.4](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (model engine health-check hook).** The cleanest place to install the shutdown handler is wherever the model engine wires its main loop — the same place that registers the iteration-boundary hook can register an `atexit` (or equivalent) that consults `world_is_poisoned()` and `_exit(0)`s if true, otherwise lets normal shutdown proceed. **This is where the production fix for F2 should live.**

**Prototype mitigation.** The worker calls `os._exit(0)` unconditionally after the main loop's `loop_end` event. See [`prototypes/wide_ep_ft_mvp/scripts/kill_and_survive_worker.py`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/scripts/kill_and_survive_worker.py) — the inline comment explicitly cites this finding. Production must be conditional (`_exit` only if `world_is_poisoned`, else `return` normally) so legitimate shutdowns still run `MPI_Finalize` cleanly.

**Open sub-questions for production design.**

- Should the "skip `MPI_Finalize`" decision happen at every rank independently (based on local `EPGroupHealth.has_failures()`), or via a broadcast from one designated rank? Independent decisions are simpler but assume every survivor has seen the same broadcast — already a PR 1c.3 invariant for single-failure but may not hold under multi-failure (PR 1c.6).
- Does `--mca orte_enable_recovery 1` change `MPI_Finalize`'s behavior in any way? Audit 1a Day 2 did not test this specifically; worth a focused micro-prototype if PR 1c.3 doesn't already validate it.
- Are there other implicit collectives in the inference shutdown path (e.g. CUDA context destroy on driver-mapped MNNVL memory)? This prototype only exercises the MPI shutdown path; the CUDA-driver path is Audit 1a Day 3's territory but wasn't tested under "survivor with poisoned MNNVL mapping" conditions. **Defer to [Audit 1b](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access).**

---

## F3. Detection is parallel, not serial — broadcast is consensus backup, not primary spreading

**Surfaced during:** first successful Level A run, 2026-05-19. Raw event distribution from [`prototypes/wide_ep_ft_mvp/results/np4-iter40.json`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/results/np4-iter40.json):

| Event | Rank 0 | Rank 1 | Rank 3 |
|---|---|---|---|
| `watchdog_marked_failed(peer=2)` | t=88624.804 | t=88624.806 | t=88624.804 |
| `broadcast_received(peer=2)` | — | — | — |

**All three survivors detected the dead peer independently within 2 ms.** Zero `broadcast_received` events fired because by the time the MPI broadcast arrived on any survivor, that survivor's local watchdog had already called `mark_failed` (which is idempotent), so the recv-side `mark_failed` returned `False` and the broadcast-received callback was correctly skipped.

**Why it matters.** The implicit assumption "rank A detects, broadcasts to B/C/D, they apply" is *not* the actual data path. The actual data path is "every rank's local zero-collective watchdog detects independently; the broadcast is a consensus backup for the small skew window where one rank's watchdog is slower than another's." This matches the production [§5.3 design](05-phase-1-immediate-survival.md#layer-1--alltoall-watchdog-the-host-side-abort-hook) (Layer 1 = primary, scale-independent), but is worth flagging explicitly so reviewers and operators don't model the system as broadcast-driven.

**Implication for production PRs.**

- **[PR 1c.3](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (MPI FT subcomm).** The broadcast must continue to exist — there is still a ~ms skew window where one survivor has detected and another hasn't, and during that window the AlltoAll kernel could race. But the broadcast's *latency budget* is relaxed: it doesn't need to be on the critical path. Production can prefer slower-but-more-reliable primitives (e.g. a barrier+broadcast at iteration boundaries, per [PR 1c.5](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown)) without affecting recovery time.
- **[PR 1c.4](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (model engine hook).** The `EPGroupHealth.generation` check at iteration boundary picks up *any* survivor's local mark_failed; the broadcast is only needed to handle the case where rank A's hook ran *before* its watchdog fired but *after* peer's watchdog fired. The broadcast handles this case but is not on the critical path for the common case where every survivor's watchdog has already fired by the next iteration boundary.

**Driver fix.** The driver's `t_mark_failed_propagated` measurement initially required a `broadcast_received` event on every survivor; updated to count "ranks whose local `mark_failed` succeeded (via either watchdog or broadcast)" so the parallel-detection case is correctly measured.

---

## F4. Detection latency dominates the recovery budget; relationship is linear

**Surfaced during:** OQ4 watchdog-timeout sweep, 2026-05-19. Identical workload (4 ranks, kill at iter 40, 400 iters total), varying `--watchdog-timeout-sec`:

| Watchdog timeout | Total recovery (t_first_new_request_completed) | Budget verdict |
|---|---|---|
| 1 s | 1.10 s | ✓ PASS |
| 2 s | 2.10 s | ✓ PASS |
| **5 s (default)** | **5.11 s** | **✓ PASS (default)** |
| 10 s | 10.13 s | ✗ **FAIL** |

`recovery ≈ watchdog_timeout + 100 ms`. The 100 ms tail is one poll interval (50 ms in this run) + iteration boundary delay + reconfigure (~10 µs).

**Why it matters.** The watchdog timeout is the *only* meaningful tuning knob for recovery latency. Everything else (broadcast, EPLB reconfigure, iteration boundary) is in the noise.

**Implication for production PRs.**

- **[PR 1a.4](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) default value.** 5 s default is well-chosen — fits the 10 s recovery budget with ~50% headroom for everything else (NCCL surfaces, model engine drain, first new request latency).
- **[PR 1d.1](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (LLMArgs).** The watchdog timeout **must** be exposed as a tunable config field (not buried as a constant). Deployments with stricter latency SLAs (e.g. < 5 s recovery) should be able to dial it down to 1-2 s at the cost of higher false-positive risk on noisy systems.

**Open sub-questions.**

- **False-positive floor.** At what timeout does spurious detection become a problem in production? The single-node prototype has near-zero noise; the 72-rank NVL72 case may show natural completion-flag pauses (load imbalance, EPLB stride, GC pauses) that fire a 1 s watchdog spuriously. Validation belongs to [Audit 1b](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access).
- **Poll-interval scaling.** Default 100 ms poll wastes ~99% of CPU samples; this prototype uses 50 ms with no measurable cost, but at 72 ranks × 1 watchdog thread/rank the steady-state cost may matter. Worth profiling in PR 1a.4 to find the right default.

---

## F5. Recovery time is scale-independent across 4 vs 8 ranks (validates §5 claim)

**Surfaced during:** Level A end-to-end runs at `--np 4` and `--np 8`, 2026-05-19. Identical kill timing (iter 40) and identical detection config (5 s timeout, 100 ms poll), measured wall-clock from kill to first new request completed at N−1:

| `--np` | `t_kill` | `t_watchdog_fires` | `t_propagated` | `t_reconfigure_done` | **Total recovery** |
|---|---|---|---|---|---|
| 4 | 2.056 s | 7.108 s | 7.110 s | 7.120 s | **7.168 s** |
| 8 | 2.056 s | 7.109 s | 7.111 s | 7.121 s | **7.168 s** |

**Identical to within 1 ms across every measured event.** Doubling the EP size adds zero recovery latency.

**Why it matters.** The plan [§5 "What this prototype validates"](mvp-prototype-plan.md#5-what-the-prototype-validates--does-not-validate) claims "Order-of-magnitude on the < 10 s recovery target. Detection dominates the budget, and detection is scale-independent." This is now empirically confirmed at small N; the claim's extrapolation to 72 ranks is justified by the same property (every rank's local watchdog is independent — there's nothing in the design that scales with N).

**Caveats.** The prototype does not exercise:

- The 72×72 completion-flag table itself (`kMaxRanks` 64→128 register-pressure question is still pending for [PR 1a.2](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown)).
- The MPI broadcast at scale (would scale ~O(N) but is off the critical path per F3).
- The EPLB reconfigure at 58 layers vs the prototype's 2 layers (~30× more work, but the prototype's per-layer cost is ~6 µs so 58 layers ≈ 350 µs, still well in the noise).

These three pending items are [Audit 1b](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access) territory; the prototype's "scale-independent" claim is robust within its scope.

---

## OQ2. Iteration-boundary semantics — answered

The plan asked: "Where exactly does the model engine check `EPGroupHealth.generation` — top of iteration before any kernel launches, or after fwd setup? The latter risks launching one more iteration's kernels with the old mask."

**Empirical answer from Level A runs.** The iteration-boundary hook fires within `t_iteration_boundary - t_mark_failed_propagated = 7-8 ms` of detection, well below the `iter_sleep_sec = 50 ms` cadence. Even an iteration-boundary check placed "wrong" (after fwd setup instead of before) would add at most one `iter_sleep_sec` of stale-mask exposure — ~50 ms in the prototype, likely 100-500 ms in production depending on token batch sizes.

**Implication.** The "where in the iteration to check" question becomes a near-noise design choice compared to the 5 s detection budget. Production [PR 1c.4](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) can prioritize *cleanest integration point* over *earliest possible check point*; the cost difference is small.

---

## Pending findings

Successfully closed: **F1, F2, F3, F4, F5, OQ2, OQ4.**

Still pending:

- [ ] **OQ1: Watchdog vs. NCCL collective ordering.** Not validated empirically — requires adding `torch.distributed` (NCCL backend) initialization + a real allreduce to the worker, plus GPU contexts. Significant new work. *Quick a-priori analysis:* the watchdog should fire first because NCCL's async-error scan interval is typically ≥ 1 s while the watchdog is bounded by `timeout + poll_interval`. Validated empirically once we have a kernel-stub integration (currently blocked alongside the kernel-side 1a.2/1a.3 work).
- [ ] **Seam-stressing kill points** (during dispatch / combine / routing / EPLB-stride) — blocked on the kernel-side 1a.2/1a.3 integration; see [`prototypes/wide_ep_ft_mvp/kernel/README.md`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/kernel/README.md).
- [ ] **False-positive floor characterization** (F4 sub-question) — needs NVL72 noisy-workload data; deferred to Audit 1b.
- [ ] **Failure-during-recovery stress case** (multi-failure ordering) — out of MVP scope per PR 1c.6.

---

## Status: paused (2026-05-19)

The prototype's primary mandate ([§1 of the plan](mvp-prototype-plan.md#1-why-this-exists)) is empirically discharged: the MVP integration story works, the < 10 s recovery target is achievable in principle (5.11 s with default config), and the seam contracts are correct. Six findings + two open questions closed; the remaining four pending items all hit diminishing returns vs. continuing work on the production PRs that they unblock.

[Draft PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) is left in `Draft (DO NOT SUBMIT)` state. The branch `WideEP-FT/mvp-prototype` continues to carry private cherry-picks of PR #13302 (1a.1) and PR #14160 (1d.0) plus the throwaway scaffolding under `prototypes/wide_ep_ft_mvp/`; if either parent PR lands on `main` while the prototype is paused, rebasing this branch will drop the cherry-pick as already-applied with no manual intervention.

### When to resume

Resume the prototype when *any* of the following events unblock a pending item:

| Trigger | Unblocks | Action |
|---|---|---|
| **[PR 1a.2](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (#13404, NVLinkOneSided kernel mask) lands or reaches stable review state** | Seam-stressing kill points (dispatch / combine / routing / EPLB-stride) + OQ1 (NCCL ordering, which becomes free once the kernel is in the loop) | Cherry-pick #13404 onto `WideEP-FT/mvp-prototype` per Path A in [`kernel/README.md`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/kernel/README.md); replace the pseudo-AlltoAll loop in `kill_and_survive_worker.py` with the real kernel; rerun with `--kill-during {dispatch,combine,routing,eplb-stride}` variants. |
| **[PR 1a.4](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (AlltoAllWatchdog production) lands** | Validates that the production watchdog reproduces the prototype's F3/F4/F5 numbers under real MNNVL fabric memory (not POSIX shm). | Swap `stubs/alltoall_watchdog.py` for the production watchdog in the worker; rerun Level A; diff timeline against the regression baseline JSONs already committed under `prototypes/wide_ep_ft_mvp/results/`. |
| **[PR 1c.3](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown) (MPI FT subcomm) lands** | Lets the prototype use the production FT subcomm instead of the `Isend/Irecv` stub on `COMM_WORLD`. Also exposes `MpiFtSubcomm.world_is_poisoned()` which the prototype's F2 mitigation can be tightened against. | Swap `stubs/mpi_ft_subcomm.py` for the production component; rerun Level A; document any new propagation-time delta vs. F3 baseline. |
| **NVL72 access becomes available** | False-positive floor characterization (F4 sub-question) + the actually-72-rank scale validation that F5 cannot extrapolate to with certainty. | Coordinate with [Audit 1b](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access); prototype runs as written should port to NVL72 once IMEX is configured per [mvp-prototype-plan.md §3](mvp-prototype-plan.md#3-hardware). |
| **PR 1d.4 (fault-injection harness) starts** | The prototype's `kill_and_survive_driver.py` becomes the reference implementation for the production harness; timeline JSONs become the regression baseline. | Hand off `scripts/kill_and_survive_driver.py` + `results/*.json` to PR 1d.4 author; archive the prototype dir per [mvp-prototype-plan.md §9](mvp-prototype-plan.md#9-after-the-prototype). |

### How to resume (mechanical steps)

1. `cd /home/scratch.chienchunh_coreai/dev/TensorRT-LLM-mvp-prototype` (or recreate the worktree from `WideEP-FT/mvp-prototype`).
2. `git fetch fork && git rebase fork/WideEP-FT/mvp-prototype` — picks up any cherry-pick drops if a parent PR landed.
3. `git fetch upstream && git rebase upstream/main` — picks up upstream churn.
4. Apply the trigger-specific action above.
5. Rerun the Level A baseline (`--np 4 --kill-at-iteration 40`) and diff the resulting `np4-iter40.json` against the regression baseline; verify F3/F4/F5 still hold before adding new variants.
6. Append the new findings to this file as F6+, following the F1-F5 template.
