# 7. Phase 3: Beyond Failover

[< Back to Overview](README.md)

Phase 3 is the *resilience* phase — preventing failures, adapting capacity, and handling the **soft-failure** cases that Phases 1 and 2 don't address. Phases 1+2 treat failure as binary (alive vs dead, recover from death). In practice, ranks can also be *alive but degraded* — thermal throttling, ECC correctable spikes, NVLink lane retries, sustained routing imbalance — without ever crossing into Mode A or Mode B. At WideEP scale, where AlltoAll is N-way synchronous and tail-latency-dominated, **a single straggler costs every rank, for every MoE layer, for every iteration**. Phase 3 widens the failure model from binary to graded.

Phase 3 splits into two threads:

- **Hard-failure precursors** (§7.1, §7.2, §7.4) — detect ranks heading toward Mode A/B and mitigate before failure lands. Reuses Phase 1+2 primitives.
- **Soft-failure / straggler mitigation** (§7.5, forward-looking) — rank stays alive indefinitely but slow; the goal is to bound tail latency, not to recover from death. This is genuinely new design surface, sketched at radar level here; detailed design is a follow-up.

Phase 3 is the lowest-priority phase, not staffed for MVP, and treated at discussion level; [§8.3](08-implementation-plan.md#83-phase-3-rough-plan) sizes the work at work-track granularity rather than per-PR detail.

The capabilities below share one design property: they all reuse primitives built in Phases 1 and 2 (with §7.5 also introducing kernel-level changes that go beyond what 1+2 land). Phase 3 doesn't introduce a new communication layer or new storage medium; it adds policy on top of the mechanisms already in place.

## 7.1 Latency anomaly detection

Most GPU failures aren't instantaneous. ECC memory errors accumulate; thermal throttling starts before a full stall; NVLink link-quality degrades over minutes. If we can detect degradation before the GPU becomes Mode B (silent hang) or Mode A (signal abort), we can act before the failure lands in Phase 1's recovery path.

**Mechanism.** Per-rank AlltoAll latency measured via CUDA events, stored in a circular buffer on each rank. Every N iterations, compute the median across all ranks and flag any rank whose local median exceeds `3×` the global median.

This is the approach vLLM's RFC #27774 uses (3× median threshold, 100–1000 iteration window) and it's been validated empirically for catching thermal / ECC degradation.

**Telemetry foundation (shared across §7.1, §7.4, §7.5).** Latency anomaly is the first use case but the underlying instrumentation is broader. The same per-rank ring buffer ingests:

- AlltoAll latency (CUDA events around dispatch + combine).
- Per-rank MoE compute time (CUDA events around expert kernels).
- NVML signals: `gpu_clocks_throttle_reasons`, junction temperature, ECC correctable counts, NVLink retry / replay counts, power state.
- EPLB routing-imbalance stats (per-rank token count, hot-expert distribution — already exposed by EPLB).

Building this telemetry once is ~3–4 weeks; §7.1, §7.4 (predictive failure detection), and §7.5 (straggler classification) all consume different views of the same data. Cross-correlated signals (rising latency *and* rising thermal *and* increasing NVLink retries → high-confidence "thermal-throttle straggler") are what let downstream classifiers separate routing imbalance from rank-level degradation.

**Key properties:**
- **Non-intrusive.** CUDA events around existing AlltoAll calls; no new kernels, no new collectives. Overhead is event recording + a circular-buffer update.
- **Relative, not absolute.** 3× median adapts to workload — a heavy-traffic iteration and a light-traffic iteration both get flagged on the same relative scale.
- **Per-rank.** Each rank's timing is independent; a rank that's consistently slow stands out.

**Signal output.** A "rank R is degrading" notification on the FT subcomm. Downstream actions are Phase 3.2 (preemptive migration) and/or orchestrator-level alerting.

**What it doesn't catch.** Sudden failures — a GPU that goes from healthy to dead in one iteration triggers Phase 1, not Phase 3. Phase 3 is for the slow-degradation class of failures.

## 7.2 Preemptive expert migration

Once a rank is flagged as degrading (7.1), the natural next step is to move its experts off it before it fails. The work reuses the Phase 1 v1 weight-migration path:

1. Mark rank R as "draining" in a new state on `EPGroupHealth` (between Active and Failed).
2. For each MoE layer, redistribute R's expert slots to surviving ranks using the same `reconfigure` flow from §5.2 v1. Weight migration runs from R's host-shm segment to the target ranks' GPUs.
3. Once R's slots are empty of unique experts, mark R as "drained." At this point R can be safely decommissioned — either voluntarily removed (Phase 3.3 scale-down), or left in place until it eventually fails (at which point Phase 1 mask-only recovery is trivially cheap — no zero-replica case).

**Why this is cheaper than Phase 1 recovery.** Phase 1 happens under time pressure (dead peer is hanging the collective). Phase 3.2 happens at iteration boundaries with full control — no 5s detection budget, no consensus problem, no kernel hang.

**Hot-expert prioritization.** During degradation, routing traffic is asymmetric — the degrading rank's slots get less load because their latency signal is already flagged. Preemptive migration should prioritize hot experts (highest replica counts, highest routing traffic) to minimize the load spike on surviving ranks during the transition.

**Reuses existing primitives.** `reconfigure` and the `cudaMemcpy2D` path from Phase 1 v1 (PR 1b.6). Phase 3.2's code change is largely policy — deciding when and what to migrate, not how.

## 7.3 Elastic scaling

Two directions:

### Scale-up: adding capacity to a healthy group

New hardware becomes available and the operator wants to grow a 64-rank EP group to 72. Mechanism:

1. Provision the new ranks (orchestrator).
2. Load the expert shards for the new ranks (GMS / MX / disk, per §6.3).
3. **Reuse Phase 2 rebuild** to transition the group from N to M > N ranks. EPLB redistributes to use the new capacity.

This is a scheduled operation, not an emergency recovery, so time pressure is lower. Phase 2's minutes-class disk-reload path is acceptable; sub-second is unnecessary.

### Scale-down: gracefully reducing capacity

Load is lower than capacity; the operator wants to shrink a 72-rank EP group to 64. Mechanism:

1. Identify the 8 ranks to remove (ideally least loaded, or ones flagged as degrading by §7.1).
2. **Reuse Phase 3.2 preemptive migration** to drain those ranks' experts.
3. Mask the drained ranks (Phase 1 primitive).
4. Release the hardware (orchestrator).

Scale-down is cleanly equivalent to "plan the failures that haven't happened yet" — the ranks are decommissioned in a controlled way using the same primitives that would handle them as failures.

**Why elasticity belongs in Phase 3.** It isn't strictly fault tolerance, but it uses exactly the same primitives and is natural to land once the FT path is complete. Operationally, elasticity and FT share most of the cost; adding elastic scaling on top of a working Phase 1+2 is mostly policy + orchestrator integration, not new mechanisms.

**Orchestrator interface.** Needs to be defined: how does the user request a scale event? Via `trtllm-serve` admin API? Via K8s CRD? Via direct LLM API call? Phase 3.3 design picks one; the mechanism is backend-neutral.

## 7.4 Predictive failure detection

Moving from reactive (Phase 1, react to a dead peer) through preventive (Phase 3.1, react to degradation) to predictive (Phase 3.4, react to the *trend* that predicts degradation). The data needed:

- Historical per-rank latency distributions (beyond the current-iteration window).
- ECC correctable error counts over time (via NVML).
- Thermal readings (via NVML).
- NVLink link quality metrics (via NVML / DCGM).
- Correlations between signals (e.g., rising ECC counts + rising thermal + rising latency = high failure probability within minutes).

**This depends on telemetry infrastructure that this design doesn't build.** A prediction model needs historical time-series storage, access to NVML/DCGM metrics (TRT-LLM doesn't surface these today), and a place to run the prediction logic (PyExecutor? an external monitoring service?). Phase 3.4 is therefore the furthest-out capability — useful, but blocked on cross-cutting infrastructure decisions.

**Starting point.** Before building a prediction model, instrument the existing signals (latency from 7.1, ECC from NVML, thermal) and publish them as telemetry. Downstream consumers (alerting, capacity planning, operations dashboards) get value from the raw signals before any model is trained. Only once the telemetry is in production do we have the data to train a predictor on.

**Model class.** A simple rule-based predictor (e.g., "ECC rate doubled in last 10 minutes AND latency > 2× global median AND thermal > 85 °C → flag as imminent failure") is a reasonable starting point and much more legible than an ML model. Graduation to an ML model, if needed, is a later decision.

## 7.5 Straggler mitigation (forward-looking)

**Status: radar-level sketch.** Detailed design is a follow-up; this section names the design space and the open questions, not the chosen solution.

Today's design treats failure as binary. Stragglers are the case where a rank is *alive and contributing* but slowly enough that it dominates AlltoAll's tail latency. Concretely: an N-way synchronous collective is bottlenecked by `max(per_rank_latency)` — every rank waits for the slowest. With 58 MoE layers per iteration on DeepSeek-V3, a single straggler at +20 % latency compounds across the iteration. Common straggler sources on NVL72:

| Source | Surfaces as | Detectable via |
|:---|:---|:---|
| Thermal throttling | Per-kernel slowdown 1.5–3× | NVML `gpu_clocks_throttle_reasons`, junction temp |
| ECC correctable spikes | µs–ms latency spikes | NVML `volatile_corr_ecc` counters |
| NVLink lane degradation | Reduced effective bandwidth | NVML `nvlink_replay_errors` |
| Power capping / DVFS | Frequency excursions | NVML `power_state` |
| Routing imbalance | One rank's experts hot | EPLB stats (already exposed) |
| Software jitter (GC, OS, contention) | Tail-only spikes | Per-iteration timing residuals |

### Sub-tasks

The work breaks into three interlocking pieces:

**(1) Monitoring worker progress / performance.** The shared telemetry foundation introduced in §7.1. ~3–4 weeks of instrumentation work; reused across §7.1, §7.4, and §7.5.

**(2) Identifying stragglers.** Classifier that distinguishes routing imbalance (EPLB problem — rebalance the placement table) from rank-level degradation (hardware problem — preemptive migration or speculative replication) from transient jitter (do nothing) from sustained degradation (act). Cross-correlated signals (latency *and* thermal *and* ECC) provide confidence weighting. Distinct from §7.1's binary 3×-median rule. ~2–3 weeks.

**(3) Speculative copies / proactive replication.** The novel piece. Several options with very different cost profiles, none settled here:

| Option | Mechanism | Cost | Tail-latency benefit | Detailed-design status |
|:---|:---|:---|:---|:---|
| **A. Latency-aware routing** | EPLB routing kernel reads per-replica latency; biases token dispatch toward fast replicas | Low (kernel + host weight update) | Probabilistic; only helps for replicated experts | Follow-up; lightest first cut |
| **B. Speculative redundant compute** | Mirror straggler rank's MoE work to a healthy rank; combine takes whichever responds first | High (~1/N extra compute, kernel-level race semantics) | **Strict tail-latency bound** (`min(slow, fast)` instead of `max`) | Follow-up; novel design surface |
| **C. Shadow rank as performance hot-spare** | Reuse §6.3 shadow-EP-rank lifecycle with a degradation trigger; promote shadow + demote slow rank | Medium (reuses §6.3 infra) | Reset, not bound — trades a slow rank for a fast one | Follow-up; depends on §6.3 |
| **D. Tail-cutting timeout** | AlltoAll combine treats unresponsive slot as zero-contribution after deadline | Low (reuses §5.1 mask infra) | Strict tail bound at quality cost | Follow-up |

**A + D is the natural lightweight first cut.** Option C arrives "for free" once §6.3 lands. Option B is the most expensive and the most novel — the only one with strict tail-latency guarantees, and the only one without prior straggler-mitigation precedent in inference systems (see Prior art note below).

### Open design questions (settle before detailed design)

- **EPLB invariant.** Latency-aware routing (Option A) changes EPLB from "balance token load across slots" to "balance load *and* prefer fast replicas." Two axes can fight each other. Does load-balance still get final say, or does latency bias dominate?
- **Quality vs latency.** Tail-cutting timeout (Option D) accepts quality degradation when it fires (a token's expert contribution is zeroed). With replication ≥ 2 + routing fallback this is mostly invisible; in pathological cases it's measurable. Is per-token p99 token-quality SLA an acceptable framing?
- **Combine kernel race semantics.** Speculative redundant compute (Option B) requires "first valid response wins" in the AlltoAll combine accumulator. The MNNVL combine zero-fills `dst_idx == -1` (§5.1) — extending to "first valid response wins, ignore the second" is non-trivial bookkeeping at PTX level and is the central technical challenge for B.
- **Phase 3 priority.** Currently P2. If straggler events at WideEP scale (daily thermal / ECC at 72-rack) cost more goodput than monthly hard failures (the assumption Phase 1's prioritization rests on), parts of §7.5 may justify promotion to P1.

### Prior art note

Speculative execution for stragglers has a deep classical-systems literature in batch / data-parallel scheduling: MapReduce backup tasks (OSDI 2004), LATE (OSDI 2008), Mantri (OSDI 2010), Dolly (NSDI 2013), GRASS (NSDI 2014), Wrangler (SoCC 2014), Hopper (SIGCOMM 2015). All target *independent tasks* in batch jobs (you can clone a single MapReduce task without affecting other tasks). **Synchronous all-to-all collectives in tightly-coupled inference are a different setting** — every rank's contribution is needed by every other rank in the same iteration, the timescale is sub-millisecond not minutes, and replicas already exist as EPLB placement state rather than needing fresh task spawns. Translating speculation into AlltoAll combine semantics is the unsolved part.

### Why this section is on the radar but not the implementation plan

Two reasons. (1) Detailed design needs to settle the open questions above before sizing — the choice between A, A+D, A+D+C, or A+B+D materially changes the engineering scope from ~5 weeks to ~3+ months. (2) Option B in particular is research-grade: the prior art is in classical batch-job scheduling, translating to AlltoAll combine semantics is novel architecture work, and the path to real numbers needs a real WideEP deployment. Both warrant a focused mini-design doc rather than scope creep into this one.

[§8.3](08-implementation-plan.md#83-phase-3-rough-plan) lists straggler mitigation as a Phase 3.5 follow-on track without per-PR sizing.

### Unified variability framing — connection to §7.1, §7.3, and the research arm

§7.1 (latency anomaly detection), §7.3 (elastic scaling), and this §7.5 (straggler mitigation) are different responses to the *same* underlying observation: per-rank execution-time variability. The variability has multiple causes that all manifest the same way at the AlltoAll barrier:

| Cause | Section that responds | Response timescale |
|:---|:---|:---|
| **Workload-induced load imbalance** (MoE expert skew) | EPLB rebalance (production) | sub-iteration to seconds |
| **Topology asymmetry** (intra-tray NVLink vs cross-tray NVSwitch on NVL72; NVLink vs IB on B200 NVL8 — see [§1.1 note on topology symmetry](01-user-journey-and-stack.md#other-deployment-models-summary)) | Topology-aware EPLB placement (forward-looking, coordinated with Peiheng/Dongxu) | minutes to hours |
| **Hardware degradation / soft failures** (thermal, ECC, NVLink lane, DVFS) | §7.1 latency anomaly + §7.2 preemptive migration | seconds to minutes |
| **Full rank failure** (Phase 1 recovery) | §5 rank masking + EPLB slot remap | < 10 s |
| **FT recovery transient state** (right after a rank dies and before Phase 2 restoration) | §6 rebuild + §6.3 shadow rank | < 1 s to minutes |
| **Software jitter** (GC, OS scheduling, contention) | none (or D's tail-cutting timeout) | µs to ms |

All collapse to the same observable: **elevated per-rank execution time at the AlltoAll barrier**. The Phase 3 control loop is the unification — a shared observation pipeline (§7.1 telemetry foundation) feeding three response timescales:

- **Within-iteration:** speculative redundant compute (§7.5 Option B), latency-aware routing (Option A), tail-cutting timeout (Option D).
- **Cross-iteration (seconds–minutes):** preemptive expert migration (§7.2), latency-aware EPLB rebalance.
- **Cross-iteration (minutes–hours):** elastic scaling up / down (§7.3), topology-aware placement.

**The research arm** ([straggler-speculation-research/](straggler-speculation-research/README.md)) frames the joint formulation — placement + within-iteration scheduling + (optionally) cross-iteration capacity adaptation as a coordinated control loop over the unified variability signal. FT is not a separate axis; it's one of the inputs to the same controller. Auto-scaling (§7.3) is a temporal extension of the same controller, not a separate technique. This framing is what makes the research story stronger than "speculative compute in synchronous AlltoAll" alone — it generalizes the contribution to a setting (heterogeneous-topology WideEP, multi-source variability) that prior work in classical batch speculation does not cover.

## Phase 3 vs Phase 2 vs Phase 1

| | Phase 1 | Phase 2 | Phase 3 |
|:---|:---|:---|:---|
| Entry signal | Rank failure detected | Replacement available | Degradation or scale event |
| Time pressure | Very high (serving down) | Medium (serving degraded, rebuilding) | Low (operational, no emergency) |
| Primary primitives | Rank masking, slot remap | PG rebuild, weight load | Reuses Phase 1 + 2 primitives |
| New code surface | Large (kernels, EPLB, detection, MPI) | Medium (rebuild sequencing per backend) | Small (policy + orchestrator glue) |
| External dependencies | PR #12718 | Orchestrator + optional MX-GMS | Telemetry (for 7.4) |

Phase 3 is the smallest incremental investment per capability because it reuses what Phases 1 and 2 build. That's also why it's the lowest-priority phase — the first-order availability wins come from Phases 1 and 2; Phase 3 is the polish on top.

## Scope expectations for MVP / v1

**MVP: nothing from Phase 3.** Phase 3 is explicitly deferred until Phases 1 and 2 are in a stable production state. Timeline guesstimate: post-MVP + post-Phase 2 v1, so at least 6+ months from MVP ship.

**After Phase 2 lands:**
- 7.1 latency anomaly detection + telemetry foundation is the natural first item (small scope, high value, foundation for everything else in Phase 3).
- 7.2 preemptive migration follows (reuses 7.1 + Phase 1 v1 primitives).
- 7.3 elastic scaling is a separate decision with product / operations input.
- 7.4 predictive is gated on telemetry infrastructure that probably isn't in scope for this design at all.
- **7.5 straggler mitigation** — radar-level only here; detailed design is a Phase 3.5 follow-on, with options A and D the natural lightweight first cut. Option B (speculative redundant compute) is research-grade and warrants its own design doc.

[§8.3](08-implementation-plan.md#83-phase-3-rough-plan) sizes this in weeks of work-track effort (not per-PR) because Phase 3 scope will meaningfully refine once Phase 2 is done and production experience is informing what matters most.
