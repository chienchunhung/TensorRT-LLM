# Problem Statement — Variability-Aware Scheduling in Heterogeneous-Topology WideEP Serving

[< Back to Sub-Directory](README.md) | [↑ Up to WideEP FT](../README.md)

**Status:** Research exploration. Not committed engineering work.
**Connects to:** [§7.5 Straggler mitigation (forward-looking)](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking) and [§7.5 Unified variability framing](../07-phase-3-beyond-failover.md#unified-variability-framing--connection-to-71-73-and-the-research-arm) in the parent design.
**Last revised:** 2026-05-18 — broadened from "speculative compute in synchronous AlltoAll" to "variability-aware scheduling in heterogeneous-topology WideEP serving" after Peiheng Hu's May 2026 B200+IB perf work surfaced that (a) heterogeneous topology is a property of every WideEP deployment, not just B200+IB, and (b) FT is a *cause* of variability, not a third axis alongside topology and scheduling.

## 1. Setting

**Heterogeneous-topology WideEP MoE serving.** EP ≥ 32 across a fabric whose effective per-rank-pair bandwidth is not uniform. Concrete instances of this setting:

| Deployment | Topology asymmetry source | Magnitude |
|:---|:---|:---|
| NVL72 single rack | intra-tray NVLink vs cross-tray NVSwitch (multi-hop) | 1.5–2× peak-BW spread across pairs |
| Multi-rack via NVLink + IB | intra-rack NVLink vs cross-rack IB | 5–10× depending on cross-rack share |
| **B200 NVL8 + IB** | intra-node NVLink (~900 GB/s) vs cross-node IB (~50 GB/s) | **18× peak BW** (Peiheng Hu, May 2026) |

DeepSeek-V3-class workload: 256 experts, top-8, 58 MoE layers per forward iteration, attention-DP. Every MoE layer runs an N-way AlltoAll for token dispatch and combine; the collective is the critical path of every iteration.

Two structural properties of the setting matter:

1. **Synchronous N-way collective.** AlltoAll is bottlenecked by `max(per_rank_latency)`. Every rank waits for the slowest. There is no per-token or per-pair short-circuit — the iteration advances together.
2. **Replicated experts as routing state.** EPLB maintains expert replication for load balancing (replication ≥ 2 in production DeepSeek). Replicas are not spare task instances waiting to be spawned — they are pre-allocated GPU memory, mapped into the routing table. Tokens dispatched to "expert E" can already land on rank A's slot or rank B's slot depending on routing.

These two properties create a setting that prior straggler-mitigation work did not address. Crucially, heterogeneous topology — once it exists — multiplies the consequences of both: asymmetric BW makes the synchronous-collective tail worse, *and* it makes replica placement matter (which replica you pick changes which fabric link carries the token).

## 2. The variability problem (unified formulation)

A *straggler* in the classical sense is a single rank running slower than peers, holding up a synchronous collective. In WideEP MoE serving on heterogeneous topology, the same observable — elevated per-rank execution time at the AlltoAll barrier — arises from many causes, all of which actually appear in production:

| Cause | Surfaces as | Detection signal | Timescale |
|:---|:---|:---|:---|
| Workload-induced load imbalance (MoE expert skew) | Some ranks finish FFN late, enter AlltoAll combine late (Peiheng slide 10) | EPLB stats; per-iteration timing residuals | sub-ms to seconds |
| Topology asymmetry (NVLink vs cross-tray vs IB) | Some rank-pairs systematically slower than others; combine BW utilization low for cross-fabric peers (Peiheng slide 8: 24% combine BW vs 95% DLSim) | Fabric BW telemetry; A2A model regression | static, but workload-dependent |
| Thermal throttling | Per-kernel slowdown 1.5–3× | NVML `gpu_clocks_throttle_reasons`, junction temp | seconds to minutes |
| ECC correctable spikes | µs–ms latency spikes at irregular intervals | NVML `volatile_corr_ecc` | µs spikes, persistent at hours |
| NVLink lane degradation | Reduced effective BW, higher transfer latency | NVML `nvlink_replay_errors` | persistent until repair |
| Power capping / DVFS | Frequency excursions, sustained slowdown | NVML `power_state` | seconds to minutes |
| Full rank failure recovery in progress | One rank dropped; surviving ranks running with new mask; placement temporarily suboptimal | EPLB generation counter; Phase 1 mask state | minutes |
| Software jitter (GC, OS scheduling, contention) | Tail-only spikes, otherwise normal | Per-iteration timing residuals | µs to ms |

**All collapse to one observable: elevated per-rank execution time at the AlltoAll barrier.** This is the central insight of the unified formulation. A scheduler that responds to *the observable* responds correctly to *all causes* — workload, topology, degradation, and FT-recovery transients are not orthogonal axes the controller must distinguish; they are one signal with multiple sources.

The cost of variability is non-linear in depth. A single rank at +20 % AlltoAll latency, across 58 MoE layers, **compounds end-to-end iteration latency well beyond +20 %** because iteration tail is dominated by collective tail. At p99 / p99.9 token-latency SLAs, even a transient cause (a rank slow for 200 iterations before clearing) shows up as a goodput cliff for the whole serving instance.

## 3. Why classical speculation doesn't directly apply

Classical straggler-mitigation literature (LATE, Mantri, Dolly, GRASS, Wrangler, Hopper — see [02-literature-survey.md](02-literature-survey.md)) operates on a setting that differs from this one in five ways:

| Classical setting | This setting |
|:---|:---|
| Independent tasks. Cloning task X has no effect on tasks Y, Z. | Every rank's contribution is consumed by every other rank in the same iteration. The collective is the unit of work. |
| Task-level granularity (seconds to minutes). | AlltoAll-level granularity (sub-millisecond to milliseconds). |
| Replicas don't exist; speculation spawns clones. | Replicas already exist as EPLB routing state. |
| Job-completion-time tail as objective. | Per-token p99 / p99.9 inference latency under SLA. |
| Homogeneous topology assumed. | **Heterogeneous topology is the rule, not the exception** — even on NVL72. |
| Above the per-task scheduler; no kernel-level changes. | Speculation correctness must be guaranteed *inside* the AlltoAll combine kernel. |

The last three rows are the differentiators. Four of these mismatches are quantitative (timescales, granularity, objective, topology magnitude). Two are qualitative (synchronous collective + kernel-level commit semantics). Translating speculation to this setting is not a parameter-tuning exercise; it requires re-engineering what speculation *means* when the unit of work is a collective rather than a task, and the cost-of-tail is multiplied by topology heterogeneity rather than ignorable.

## 4. The novel research questions

Stating these precisely so the research and engineering decisions stay anchored. The first three are the *technical* questions; the fourth and fifth are about *placement* and *temporal coordination*; the last is the *FT-as-input* question.

### Q1. What does "speculate" mean when the unit of work is a collective?

In classical speculation, the cloned task can race the original because either's output is independently usable — the scheduler picks whichever finishes first. In a synchronous AlltoAll, the *combine* kernel needs every rank's contribution to produce correct output. "First-wins" semantics has to live inside the combine accumulator: when both the slow rank R and the speculative rank R' produce contributions for the same expert slot, the combine kernel commits the first valid response and drops (does not double-count) the second.

This is a kernel-level systems problem, not a scheduler-level policy problem. The combine accumulator's existing `dst_idx == -1` zero-fill path (§5.1 in parent doc) is the closest extant primitive. Extending it to "first valid response wins, ignore later" requires:

- Atomic commit semantics on the combine accumulator slot (compare-and-set on a per-slot completion flag).
- Sender-side detection that the slot is already committed → skip the write to avoid wasting fabric bandwidth (especially valuable when the wasted write would go over IB).
- Correctness proof that no token's contribution is lost or doubled under concurrent first-wins commits across the slot space.

This is **the central technical contribution** of any paper in this space.

### Q2. When is replica-routing speculation enough vs when is full speculative compute needed?

Latency-aware replica routing (Option A in §7.5) is a much cheaper form of speculation: it doesn't duplicate compute, it just biases token dispatch toward fast replicas. For a workload where most experts have replicas, A captures most of the available speculation benefit at near-zero cost.

Open question: for what variability severity / what workload skew does A's expected-value benefit dominate B's strict-tail-bound benefit? At low severity (e.g., +10 % rank latency on one rank), A probably wins on cost-effectiveness. At high severity (one rank at 3× latency due to thermal throttling, or one rank-pair at 18× BW asymmetry due to IB vs NVLink), B's strict bound may be needed to meet SLA.

A 2D characterization — variability severity vs replica coverage — gives a phase diagram of which policy dominates. **Producing this diagram empirically is a clean structural contribution.**

### Q3. What detection signal triggers speculation, and over what window?

Speculation has cost. False-positive triggers (speculate when the rank is fine) waste compute. False-negative triggers (don't speculate when the rank is slow) waste tail. The classifier needs to:

1. Distinguish routing imbalance (EPLB problem — rebalance the placement table) from rank-level degradation (hardware problem — preemptive migration or speculative replication) from transient jitter (do nothing) from sustained degradation (act).
2. Provide confidence weighting from cross-correlated signals (latency *and* thermal *and* ECC *and* fabric BW).
3. Fire fast enough to act before the next AlltoAll launches, but slow enough that one slow iteration doesn't trigger speculation forever.
4. **Operate uniformly across variability sources** (Q6) — the classifier doesn't need to know whether the slowness is from workload, topology, hardware, or FT recovery, only that it's persistent enough to warrant action.

This is a sequential-decision problem with prior art in queueing theory (e.g., quartile-based stragglers). The novelty is in adapting it to sub-millisecond decision windows over GPU-fabric signals.

### Q4. How does speculation interact with EPLB's load-balancing invariant? *(joint placement + scheduling)*

EPLB optimizes "balance token load across replicas." Latency-aware routing adds "prefer fast replicas." **Topology-aware placement** adds "minimize cross-node IB traffic by placing replicas across nodes" (Peiheng's next-step item; coordinated with Dongxu Yang). Three axes can fight.

Open question: what's the constraint structure? Some natural candidates:

- **Load balance as hard constraint, latency bias as soft.** Equal-cost replicas → pick fastest. Unequal-cost replicas → load balance dominates.
- **Topology placement as static optimization, latency bias as runtime adjustment.** Placement minimizes *expected* tail given workload statistics; runtime bias responds to *observed* tail.
- **Joint optimization with a tunable knob.** Trades expected-case efficiency against tail mitigation aggressiveness.

The joint-formulation choice is itself a paper section. **This is where the topology-aware placement axis enters the research story — not as a separate paper, but as a structural input to the speculative-scheduling formulation.** Coordinated with Peiheng/Dongxu's production topology-aware EPLB work; the paper cites this as related/concurrent work or as a co-contribution depending on the agreed authorship arrangement.

### Q5. Cross-iteration capacity adaptation as a temporal extension *(auto-scaling)*

Speculative compute (Q1) is *within-iteration* tail mitigation. **Auto-scaling** (§7.3 in parent doc) is *cross-iteration* capacity adaptation that responds to the same variability signal at a longer timescale: if a rank is consistently slow for many iterations, speculation pays redundant compute every iteration; the correct response is to add capacity (scale up) or evict (scale down) and re-place.

Auto-scaling fits the joint-formulation framing as a **temporal extension** of the same control loop. The observation pipeline (per-rank execution time + fabric BW + ECC + thermal + EPLB stats) is shared across timescales:

- **Within-iteration response:** speculative redundant compute, latency-aware routing, tail-cutting timeout.
- **Seconds-to-minutes response:** preemptive expert migration (§7.2), latency-aware EPLB rebalance.
- **Minutes-to-hours response:** elastic scaling up / down (§7.3), topology-aware placement re-evaluation.

**Scope decision for first paper:** Q5 is **out of scope** for the first paper. Tightening to within-iteration + placement (Q1–Q4) keeps the contribution focused and lands at NSDI/MLSys scale. Auto-scaling lands as the second paper in a research agenda — extending the same controller to longer timescales is a cleaner standalone contribution than trying to publish three axes in one paper.

### Q6. FT recovery state as a variability source *(not a separate technique)*

When a rank fails and the Phase 1 mask + EPLB slot remap activates, the system is operating in a *transient* state: some experts have only one surviving replica, placement is locally suboptimal, and the surviving ranks running the dead rank's load are *systematically slower* until the next EPLB rebalance. From the AlltoAll barrier's perspective, this looks identical to any other variability source: per-rank execution time elevated on a subset of ranks for a transient period.

**The contribution: the speculative scheduler responds identically to FT-recovery transient state and any other variability source.** It does not require a separate "FT-aware" code path. The unified observation pipeline (per-rank execution time, fabric BW telemetry, EPLB generation counter) is sufficient.

This is the conceptually-cleanest payoff of the unified variability formulation: FT, topology, workload, and hardware variability all collapse to the same control problem. **Treating FT as one input to the same controller — rather than as a separate axis — is itself a structural contribution.**

## 5. Working assumptions and constraints

Choices the design / paper takes as given:

- **WideEP, not standard EP.** Results are about EP ≥ 32. Smaller EP doesn't have the AlltoAll tail-domination problem to the same degree.
- **Heterogeneous topology as the *general* setting.** The kernel-level combine work is over MNNVL (the backend TRT-LLM owns); the *findings* generalize to NCCL- or NVSHMEM-based AlltoAll on B200+IB and similar.
- **Replication ≥ 2.** The DeepSeek production default. Lower replication eliminates Option A (no replicas to choose between) and makes Option B's overhead more punishing.
- **Inference, not training.** Tail latency under SLA is the objective. Training tolerates per-iteration tail differently and is out of scope.
- **Single variability source dominant at a time.** Multi-source compounding (e.g., one thermally-throttled rank *and* one degraded NVLink lane *and* in-progress FT recovery) is an open question for follow-on work.

## 6. Success criteria

For the *engineering* deliverable (Phase 3.5 production work):

- Median p99 token latency reduction by ≥ 25 % under realistic variability injection (one rank at 1.5–2× thermal-throttle slowdown; or one rank-pair on cross-node fabric).
- Steady-state overhead under low-variability conditions ≤ 1 %.
- Stability: no false-positive speculation in 1000 consecutive iterations of routine load.

For the *research* deliverable (paper):

- Theoretical framing: unified variability formulation; placement-vs-routing-vs-speculation phase diagram.
- Empirical characterization on at least two heterogeneous-topology deployments (e.g., NVL72 cross-tray + B200 NVL8 + IB) with controlled variability injection.
- Ablation showing each component (telemetry, classifier, topology-aware placement, latency-bias routing, kernel first-wins combine) contributes measurably.
- Comparison against classical straggler-mitigation policies adapted to the setting (the most likely reviewer challenge).
- Demonstration that the same controller responds correctly to FT-recovery transient state without a separate FT-aware code path.

## 7. What this research is and isn't

**Is:** Variability-aware scheduling in heterogeneous-topology WideEP serving. The kernel-level first-wins combine is the central technical contribution; the unified variability formulation (workload + topology + degradation + FT-recovery collapsing to one observable) is the conceptual contribution; the joint placement + scheduling formulation is the structural contribution.

**Isn't:**
- Generic ML serving optimization. There is no contribution in "speculation generally helps inference." The contribution is precisely about the AlltoAll collective constraint + the topology asymmetry that amplifies its cost.
- Topology-aware EPLB on its own (that's a production track owned by Peiheng/Dongxu; the paper cites or co-authors).
- Auto-scaling for serving on its own (separate, second-paper extension to longer timescales).
- FT framework redesign (FT is an *input* to this controller, not a thing this controller fixes — the FT work lives in Phases 1–2).

**Risk:** A parallel research effort (Microsoft Research, Google DeepMind, Berkeley, CMU, or one of the inference startups) may already be working on adjacent problems. The literature survey ([02-literature-survey.md](02-literature-survey.md)) maps prior art and proposes a search query list to verify novelty before committing engineering investment. Heightened risk after Peiheng's deck — once B200+IB perf work is public, the variability-driven motivation is no longer proprietary; anyone working on WideEP serving with a similar deployment can construct the same problem framing.

## 8. Coordination requirements

Two adjacent production tracks need alignment before the paper framing locks in:

1. **Topology-aware EPLB (Peiheng / Dongxu).** If their team ships topology-aware placement as a production feature before the paper publishes, "placement" stops being a research contribution. Two outcomes are acceptable; pick before drafting starts:
   - **Co-author the paper with Peiheng / Dongxu.** Stronger contribution story; slows decision-making; some risk of differing publication appetites between research and production tracks.
   - **Scope the paper to joint formulation + scheduling.** Cite TRT-LLM's topology-aware EPLB as prior/concurrent work. Cleaner but slightly weaker placement-side novelty.

2. **Auto-scaling production work (Phase 3.3 in parent doc, ~3–4 weeks engineering).** Once production auto-scaling lands, the "temporal extension" framing for the second paper has a concrete baseline to compare against. Keeps the second-paper option open without blocking the first.

Both coordination points should be settled in a focused sync (~30 min) before the literature search runs in earnest.
