# MVP End-to-End Prototype Plan

[< Back to Overview](README.md)

**Status:** Draft v1 — scaffolding shipped as preview draft [PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198) on branch `WideEP-FT/mvp-prototype` (DO NOT SUBMIT; not for merge). First runs in progress; findings collected in [mvp-prototype-findings.md](mvp-prototype-findings.md). • **Owner:** WideEP FT track • **Last updated:** 2026-05-18

## 1. Why this exists

The MVP is shipped as 14 separate PRs across four tracks ([§8.1](pr-execution/08-implementation-plan.md#81-phase-1-pr-breakdown)). Each PR is reviewable in isolation, but the *integration contracts between them* (who calls `EPGroupHealth.mark_failed`, when the kernel re-reads the mask, what triggers `reconfigure_mask_only`, how the survivor-side NCCL collective interacts with the watchdog) only become visible end-to-end. Discovering an interface mismatch at the integration stage — after six PRs have landed — is expensive.

A **3–5 day throwaway end-to-end prototype** on a 4 or 8-GPU node validates those seams ahead of the production PRs. It does **not** replace the MVP; every prototype component is stubbed to the absolute minimum needed to exercise the seam, and the prototype code is discarded once the production PRs land.

The prototype's claim is narrow but high-value: *the MVP integration story works, the < 10 s recovery target is achievable in principle, and the seam contracts are correct.*

## 2. Prototype vs. MVP (vertical slice, not a subset)

The prototype is a **vertical slice** through every MVP track, not a subset of tracks. Each component is present, but heavily stubbed.

| MVP component | Prototype shape | What's deferred to the real PR |
|:---|:---|:---|
| 1a.1 `EPGroupHealth` | Use the real one from PR #13302 (already mergeable) | — |
| 1a.2 NVLinkOneSided kernel mask | Add a single rank-test branch in dispatch + combine; no `kMaxRanks` 64→128 bump; no perf gate | Production kernel reasoning, register pressure analysis, < 0.1 % overhead validation |
| 1a.3 NVLinkOneSided Python binding | Hard-coded plumbing of the mask to the kernel; no factory integration | Clean `CommunicationFactory` integration |
| 1a.4 `AlltoAllWatchdog` | Python timer thread, 100 ms poll of host-visible `completion_flags`; 5 s timeout; calls `mark_failed` directly | Three-layer detection, error-classification integration, telemetry |
| 1a.7 NCCL FT wrapper | Set `NCCL_ASYNC_ERROR_HANDLING=1` env var; a single watchdog goroutine in Python calling `ncclCommGetAsyncError` | `ncclCommAbort` + reinit; `abort_and_reinit(active_ranks)` API; PR #12718 classifier integration |
| 1b.1-3 EPLB `reconfigure_mask_only` | Zero out dead-rank slot in one or two layers; called directly from iteration-boundary hook | All 58 layers, < 10 ms target, thread-safe pause/resume of EPLB worker |
| 1c.3 MPI FT subcomm | Global Python state + one `MPI_Isend`/`Irecv` pair on a dedicated thread; no `MPI_Comm_split` | Proper FT subcomm with `MPI_ERRORS_RETURN`, ULFM, consensus protocol |
| 1c.4 Model engine health-check hook | `if health.generation != cached: reconfigure_mask_only()` at top of every iteration | Backpressure, drain, telemetry, integration with `check_health()` |
| **1d.0 MPI signal-handler replacement** | **Needed in real form** — `_exit(N)` + `--mca orte_enable_recovery 1` + `MPI_ERRORS_RETURN`. Small enough to keep | Feature-flag gating |
| 1d.4 Fault-injection harness | `os.kill(rank_pid, SIGKILL)` from a Python test driver | Pytest fixture, mid-collective abort points, per-token correctness assertions |

**Calendar cost.** ~3–5 days for the prototype itself, assuming the hardware is configured. Compare with ~7 weeks for the full MVP. The 1d.0 work is reused as-is into the MVP, so it's not wasted.

## 3. Hardware

### Recommended platforms

The prototype needs **one node with ≥ 4 NVLink-connected GPUs of NVL or NVL72 class**. Three viable options:

| Platform | GPUs / node | Form factor | IMEX setup needed? | What it validates |
|:---|:---|:---|:---|:---|
| **DGX/HGX B200** | 8 | NVLink5 between GPUs, no NVSwitch chip mediating | No — `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` works | Kernel mask + EPLB + watchdog seams; **does not** test the fabric-handle path |
| **DGX/HGX B300** | 8 | Same as B200 (NVLink5, no NVSwitch chip mediating) | No — same as B200 | Same as B200; Blackwell Ultra refresh, same FT story |
| **DGX/HGX H100** | 8 | NVLink4 between GPUs | No | Same; older but adequate for seam validation |
| **GB200 NVL72 tray** | 4 (compute tray) | Grace+Blackwell, NVSwitch fabric chip | **Yes** | Real fabric-handle path; matches production NVL72 |
| **GB300 NVL72 tray** | 4 | Grace+Blackwell-Ultra, same fabric story as GB200 | **Yes** (identical to GB200) | Same as GB200 |

**Recommendation.** If the goal is fastest path to seam validation, an 8-GPU HGX/DGX B200 or B300 box is the easiest setup. If the goal is closest match to NVL72 production hardware, a single GB200 or GB300 tray is preferred at the cost of one-time IMEX setup work.

### IMEX configuration (only for GB200/GB300 NVL72 trays)

IMEX (`nvidia-imex`) is the user-space daemon that programs the NVSwitch chip's permission table so processes can grant each other cross-process P2P access to fabric memory. It's required even on a single GB200/GB300 tray because the NVSwitch chip mediates intra-tray P2P; on HGX/DGX B200/B300 boxes (which use direct NVLink without an NVSwitch chip in the way) it isn't in the path.

**Five-step setup on the host:**

1. **Install the IMEX package.** Ships with R575+ NVIDIA datacenter drivers. Confirm with `which nvidia-imex` and `systemctl status nvidia-imex`. Package name varies by distribution.
2. **Write `/etc/nvidia-imex/` configuration.** Single-node prototype: just localhost in the nodes config. Exact file format shifts between R575 / R580; consult the [NVIDIA IMEX guide](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/gettingstarted.html) for the current driver version. Same guide is cited at `cpp/tensorrt_llm/runtime/mcastDeviceMemory.h:34-35`.
3. **Enable + start the daemon.** `systemctl enable --now nvidia-imex`. Verify with `journalctl -u nvidia-imex` (clean output) and `ls /dev/nvidia-caps-imex-channels/` (channel device files present).
4. **Container: mount the IMEX devices.** `docker run --device=/dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels …`. Production CI already does this via `--container-env=NVIDIA_IMEX_CHANNELS` (`jenkins/scripts/perf/local/submit.py:316`).
5. **Verify the round-trip.** Run a tiny `cuMemCreate(... CU_MEM_HANDLE_TYPE_FABRIC ...)` + `cuMemExportToShareableHandle(... CU_MEM_HANDLE_TYPE_FABRIC ...)` from two processes. TRT-LLM's auto-detection at `cpp/tensorrt_llm/runtime/ipcNvlsMemory.cu:397-412` does the same probe and falls back to posix-FD if fabric handles fail; running the same probe by hand confirms IMEX is working.

**Failure signal in TRT-LLM.** When IMEX isn't configured, `cacheTransBuffer.cpp:193` logs `"Try to creat fabric memory failed, setting imex channel may be required"` and the runtime silently falls back to posix-FD. The fallback is functional but does not exercise the fabric-handle path.

**Honest caveat on the steps above.** The five-step shape is durable across releases; exact paths/commands/config-file format shift between R575, R580, and downstream. Confirm against the IMEX guide for the exact driver version on the prototype node before starting setup. Audit 1a Days 4–5 stalled on this exact issue — getting IMEX configured + matched to the right NGC container image is hands-on ops work.

## 4. Test recipe

A concrete kill-and-survive sequence on the prototype node:

### 4.1 Workload

A small MoE that exercises `NVLinkOneSided` with as few moving parts as possible. Two options:

- **DeepSeek-V2-Lite** (16 experts, 2 active per token; fits comfortably on 4-GPU MNNVL with `ep=4`).
- **A toy 4-expert MoE** on 4 ranks, replication ≥ 2 (so every expert has a surviving copy).

The point is the AlltoAll kernel + EPLB + MoE forward — not the model. Either choice is fine.

### 4.2 Configuration

- `ep_size = 4` (or 8), `slot_count_per_rank = 2`, replication ≥ 2 → every expert has a surviving copy on another rank.
- `NVLinkOneSided` forced as the AlltoAll backend (via `CommunicationFactory` priority or explicit config).
- Prototype `AlltoAllWatchdog` running with 100 ms poll, 5 s timeout.
- Prototype kernel mask and `reconfigure_mask_only` wired through `EPGroupHealth`.
- 1d.0 signal-handler replacement active.

### 4.3 Sequence

1. **Steady-state baseline.** Run 30 s of serving (small prompt stream). Record throughput. Verify `EPGroupHealth.generation == 0`, all ranks active, no watchdog firings.
2. **Kill-and-survive.** From the test driver, `os.kill(rank2_pid, SIGKILL)`. Wall-clock timestamps:
   - `t_kill` — SIGKILL issued.
   - `t_watchdog_fires` — first survivor's watchdog calls `mark_failed(2)`.
   - `t_mark_failed_propagated` — every survivor's `EPGroupHealth.generation` reflects the failure.
   - `t_iteration_boundary` — next iteration boundary reached.
   - `t_reconfigure_done` — `reconfigure_mask_only` returns on all survivors.
   - `t_first_new_request_completed` — first request completed at N-1.
3. **Assertions.**
   - Survivors do not die from MPI signal propagation (1d.0 working).
   - `t_watchdog_fires - t_kill` is within the configured 5 s window.
   - `t_reconfigure_done - t_iteration_boundary` is small (< 100 ms is fine for a prototype; the production target is < 10 ms).
   - `t_first_new_request_completed - t_kill < 10 s`.
   - Steady-state throughput recovers to ≈ (N-1)/N of baseline.
   - No data corruption — every token in completed requests has the correct logit output.
4. **Seam-stressing variants.** Kill at different points in the iteration:
   - During dispatch (kernel actively spinning on dead peer's `completion_flags`).
   - During combine (similar).
   - During routing (between dispatch and combine).
   - During EPLB worker stride (off-iteration cleanup work).
   - All four should converge to the same final state and timing budget.

### 4.4 Reusable artifact

The per-event timeline (`t_kill` → `t_first_new_request_completed`) becomes the regression baseline for the eventual MVP harness (PR 1d.4). Log to JSON; the prototype's correctness assertions become the harness's correctness assertions.

## 5. What the prototype validates / does not validate

### Validated (high-leverage for MVP integration risk)

- **MNNVL data-plane code path.** `NVLinkOneSided` runs the same kernel on 4 ranks as on 72; `cuMemCreate(... FABRIC ...)` (or posix-FD) works the same; the completion-flag table exists.
- **The MVP integration seams.** Who calls what, in what order, on what thread. This is where MVP integration risk concentrates and where the prototype has the most leverage.
- **Mode A signal-handler fix (1d.0).** Orchestrator behavior doesn't care about rank count.
- **Detection latency for Mode B.** Watchdog logic is scale-independent.
- **EPLB `reconfigure_mask_only` mechanism.** Per-layer iteration just smaller (a few layers vs. 58); the operation is the same shape.
- **Throughput drop to ≈ (N-1)/N.** Math holds at any N.
- **Order-of-magnitude on the < 10 s recovery target.** Detection dominates the budget, and detection is scale-independent.
- **Watchdog ↔ NCCL collective interaction.** Does the *survivors'* next NCCL collective hang before the AlltoAll watchdog fires? — answerable at 4 ranks.

### Not validated (needs NVL72 / Audit 1b)

- **`kMaxRanks` 64→128 register-pressure behavior at 72-rank polling loop.**
- **NVSwitch fabric manager behavior when a rack member disappears** — not in the path on HGX/DGX form factors, and only intra-tray on a single GB200/GB300 tray.
- **IMEX dynamic re-grant under rank death** at multi-node fabric scale.
- **72-rank scaling tail** (e.g., 72×72 completion-flag table page-cache interaction).
- **Per-iteration steady-state overhead at 72 ranks vs. 4 ranks.** The < 0.1 % gate has to be re-measured at scale.
- **ULFM multi-failure paths.** Single-failure consensus is trivial; multi-failure isn't, and isn't in MVP anyway.

These are exactly the items [Audit 1b](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access) is scoped to cover with NVL72 time. The prototype does not substitute for Audit 1b.

## 6. Sequencing

Two ordering constraints:

1. **PR 1d.0 (MPI signal-handler replacement) must land before the prototype.** Without it, the SIGKILL'd rank's signal handler aborts every other rank — the prototype can't survive a single kill. 1d.0 is the smallest PR in MVP and is already started; finishing it first costs little and unblocks both the prototype and the rest of the MVP.
2. **PR #13302 (`EPGroupHealth`) ideally lands before the prototype**, since the prototype reuses it directly. If #13302 is still in review when the prototype starts, the prototype can carry a private copy and rebase to the merged version at the end.

**Current status (2026-05-15):** [PR #13302 (1a.1)](https://github.com/NVIDIA/TensorRT-LLM/pull/13302) and [PR #14160 (1d.0)](https://github.com/NVIDIA/TensorRT-LLM/pull/14160) are both in review, not yet merged, so the prototype branch (`WideEP-FT/mvp-prototype`, [draft PR #14198](https://github.com/NVIDIA/TensorRT-LLM/pull/14198)) carries private cherry-picks of both. When either lands on `main`, rebase this branch onto the new main — git will treat the cherry-picked commit as already-applied and drop it cleanly. The kernel-side 1a.2 + 1a.3 work is deferred; see [`prototypes/wide_ep_ft_mvp/kernel/README.md`](https://github.com/chienchunhung/TensorRT-LLM/blob/WideEP-FT/mvp-prototype/prototypes/wide_ep_ft_mvp/kernel/README.md) on the prototype branch for the two integration paths (cherry-pick PR #13404 vs. inline minimal stub).

Beyond those two, the prototype runs **in parallel with PR 1a.2 (kernel mask), PR 1c.3 (MPI FT subcomm), and PR 1d.4 (fault-injection harness)**. The prototype's stub versions of those components are different code from the production PRs, so the two streams of work don't block each other. The prototype's findings feed back into all three PR designs.

```
Now ────────────────────────────────────────────────────────────────────────────────► MVP ship
       1a.1 (PR #13302) ──┐
       1d.0   ────────────┤
                          ▼
                  ┌───────────────────────────────┐
                  │ 3–5 day e2e prototype         │
                  │ (single-rank kill, 4/8 GPU)   │
                  │ ─ stub 1a.2/1a.4/1b.1-3/      │
                  │   1c.3/1c.4/1d.4              │
                  │ ─ validates MVP seams         │
                  │ ─ produces timing baseline    │
                  └─────────┬─────────────────────┘
                            │
                            ▼  (findings feed back)
                  ┌────────────────────────────────────────────┐
                  │ Production PR tracks run in parallel:      │
                  │ 1a.2 (kernel mask) — PR #13404             │
                  │ 1a.3/1a.4/1a.7  — Python + watchdog +NCCL  │
                  │ 1b.1-3 — EPLB                              │
                  │ 1c.1-4 — Detection + broadcast             │
                  │ 1d.1-5 — Integration + tests               │
                  └────────────────────────────────────────────┘
```

## 7. Exit criteria

The prototype is "done" when:

1. **Single-rank kill survives.** SIGKILL on rank K leaves the other N-1 ranks serving; total wall-clock from kill to first new request completed at N-1 is < 10 s.
2. **All four seam-stressing kill points converge to the same final state.** Dispatch / combine / routing / EPLB-stride kills all produce a consistent recovery.
3. **The per-event timeline JSON is logged for every run.** This becomes the MVP regression baseline.
4. **The findings are written up.** A short report covering: what worked, what surprised, which seam contracts changed during prototyping, which production PRs need design adjustments based on prototype findings. Lives at [mvp-prototype-findings.md](mvp-prototype-findings.md) alongside [audit-1a-findings.md](audit-1a-findings.md); each finding routes to the specific production PR that should incorporate it.

## 8. Open questions for the prototype to answer

Items the prototype is specifically scoped to surface answers for:

- **Watchdog vs. NCCL collective ordering.** When rank K dies during a non-MoE NCCL collective (e.g., TP allreduce in attention projection), does PR 1a.7's `NCCL_ASYNC_ERROR_HANDLING` actually surface the error before the AlltoAll watchdog times out? Or does the AlltoAll watchdog fire first because the next AlltoAll comes faster than NCCL's async-error scan?
- **Iteration-boundary semantics.** Where exactly does the model engine check `EPGroupHealth.generation` — top of iteration before any kernel launches, or after fwd setup? The latter risks launching one more iteration's kernels with the old mask.
- **Three-part 1d.0 fix interaction.** Does `_exit(N)` + `--mca orte_enable_recovery 1` + `MPI_ERRORS_RETURN` actually prevent `mpirun` from killing the survivors? Audit 1a Day 2 validated this in isolation; the prototype validates it under a real MoE workload with NCCL collectives also flowing.
- **Detection latency reality check.** Is the 5 s default watchdog timeout right for the configured deployment? Too low → false positives; too high → blows the 10 s budget.

## 9. After the prototype

When the prototype is complete and the findings are written up:

- The prototype code is **discarded** (it's all stubs).
- Each production PR design ingests the prototype findings — interface contract changes go into the PR description before the PR is opened.
- The per-event timeline JSON gets pulled into PR 1d.4's fault-injection harness as the reference baseline.
- The prototype's hardware setup (especially IMEX, if configured) is reused for the rest of Audit 1a (Days 4–5: intra-node MNNVL teardown prototype) and for the eventual MVP integration testing.
- The findings inform what changes (if any) need to be made to the MVP critical-path Gantt chart in [§8.1](pr-execution/08-implementation-plan.md#phase-1-mvp-critical-path).

## 10. What this does *not* replace

- **Audit 1b** ([§9.1](09-risks-and-open-questions.md#audit-1b--rack-fabric-validation-pending-nvl72-access)). Rack-fabric behavior at 72 ranks needs NVL72 time. The prototype + Audit 1a together cover everything non-rack-fabric-specific; Audit 1b covers what's left.
- **The MVP PRs themselves.** The prototype is throwaway scaffolding to validate the seams. The production PRs still need to land with full test coverage, error handling, telemetry, and feature-flag gating.
- **Performance regression testing.** Steady-state overhead at 4 ranks is not predictive of 72-rank overhead. PR 1d.5 (steady-state overhead regression test) still has to run on the right hardware.
