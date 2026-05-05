# Problem Statement — Speculative Execution in Synchronous AlltoAll

[< Back to Sub-Directory](README.md) | [↑ Up to WideEP FT](../README.md)

**Status:** Research exploration. Not committed engineering work.
**Connects to:** [§7.5 Straggler mitigation (forward-looking)](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking).

## 1. Setting

WideEP MoE serving on NVL72 — 72 GPUs, single fabric domain, DeepSeek-V3-class model with 256 experts, EP=72, attention-DP. Every MoE layer runs an N-way AlltoAll (`NVLinkOneSided` over MNNVL fabric memory) for token dispatch and combine. 58 MoE layers per forward iteration; one AlltoAll dispatch + one AlltoAll combine per layer. The collective is the critical path of every iteration.

Two structural properties matter for what follows:

1. **Synchronous N-way collective.** AlltoAll is bottlenecked by `max(per_rank_latency)`. Every rank waits for the slowest. There is no per-token or per-pair short-circuit — the whole iteration advances together.
2. **Replicated experts as routing state.** EPLB maintains expert replication for load balancing (replication ≥ 2 in production). Replicas are not spare task instances waiting to be spawned — they are pre-allocated GPU memory, mapped into the routing table. Tokens dispatched to "expert E" can already land on rank A's slot or rank B's slot depending on routing.

These two properties create a setting that prior straggler-mitigation work did not address.

## 2. The straggler problem

A *straggler* in this context is a rank that is alive and contributing correctly but slower than peers. It does not crash, does not hang the kernel (Mode B), does not trip MPI signal handlers (Mode A). The forward pass completes; the iteration just takes longer because every other rank waits.

Common sources, all of which actually appear at scale on NVL72:

| Source | Surfaces as | Detection signal |
|:---|:---|:---|
| Thermal throttling | Per-kernel slowdown 1.5–3× | NVML `gpu_clocks_throttle_reasons`, junction temp |
| ECC correctable spikes | µs–ms latency spikes at irregular intervals | NVML `volatile_corr_ecc` counters |
| NVLink lane degradation | Reduced effective bandwidth, higher transfer latency | NVML `nvlink_replay_errors` |
| Power capping / DVFS | Frequency excursions, sustained slowdown | NVML `power_state` |
| Routing imbalance | One rank's experts hot for the workload | EPLB stats (already exposed) |
| Software jitter (GC, OS scheduling, contention) | Tail-only spikes, otherwise normal | Per-iteration timing residuals |

The cost of a straggler is non-linear in slowdown depth. A single rank at +20 % per-AlltoAll latency, across 58 MoE layers, **compounds end-to-end iteration latency well beyond +20 %** because iteration tail is dominated by collective tail. At p99 / p99.9 token-latency SLAs, even a transient straggler (a rank slow for 200 iterations before clearing) shows up as a goodput cliff for the whole serving instance.

## 3. Why classical speculation doesn't directly apply

The classical literature on speculation for stragglers (LATE, Mantri, Dolly, GRASS, Wrangler, Hopper — venues / years in [02-literature-survey.md](02-literature-survey.md)) operates on a setting with two properties this setting lacks:

| Classical setting | This setting |
|:---|:---|
| Independent tasks. Cloning task X has no effect on tasks Y, Z. | Every rank's contribution is consumed by every other rank in the same iteration. The collective is the unit of work. |
| Task-level granularity (seconds to minutes). | AlltoAll-level granularity (sub-millisecond to milliseconds). |
| Replicas don't exist; speculation spawns clones. | Replicas already exist as EPLB routing state. |
| Job-completion-time tail as objective. | Per-token p99 / p99.9 inference latency under SLA. |
| Above the per-task scheduler; no kernel-level changes. | Speculation correctness must be guaranteed *inside* the AlltoAll combine kernel. |

Three of these mismatches are quantitative (timescales, granularity, objective). One — the synchronous collective — is qualitative. Translating speculation to this setting is not a parameter-tuning exercise; it requires re-engineering what speculation *means* when the unit of work is a collective rather than a task.

## 4. The novel research questions

Stating these precisely so the research and engineering decisions stay anchored:

### Q1. What does "speculate" mean when the unit of work is a collective?

In classical speculation, the cloned task can race the original because either's output is independently usable — the scheduler picks whichever finishes first. In a synchronous AlltoAll, the *combine* kernel needs every rank's contribution to produce correct output. "First-wins" semantics has to live inside the combine accumulator: when both the slow rank R and the speculative rank R' produce contributions for the same expert slot, the combine kernel commits the first valid response and drops (does not double-count) the second.

This is a kernel-level systems problem, not a scheduler-level policy problem. The combine accumulator's existing `dst_idx == -1` zero-fill path (§5.1) is the closest extant primitive. Extending it to "first valid response wins, ignore later" requires:

- Atomic commit semantics on the combine accumulator slot (compare-and-set on a per-slot completion flag).
- Sender-side detection that the slot is already committed → skip the write to avoid wasting fabric bandwidth.
- Correctness proof that no token's contribution is lost or doubled under concurrent first-wins commits across the slot space.

This is the central technical contribution of any paper in this space.

### Q2. When is replica-routing speculation enough vs when is full speculative compute needed?

Latency-aware replica routing (Option A in §7.5) is a much cheaper form of speculation: it doesn't duplicate compute, it just biases token dispatch toward fast replicas. For a workload where most experts have replicas, A captures most of the available speculation benefit at near-zero cost.

Open question: for what straggler severity / what workload skew does A's expected-value benefit dominate B's strict-tail-bound benefit? At low straggler severity (e.g., +10 % rank latency on one rank), A probably wins on cost-effectiveness. At high severity (one rank at 3× latency due to thermal throttling), B's strict bound may be needed to meet SLA.

A 2D characterization — straggler severity vs replica coverage — gives a phase diagram of which speculation policy dominates. **Producing this diagram empirically is a clean structural contribution.**

### Q3. What detection signal triggers speculation, and over what window?

Speculation has cost. False-positive triggers (speculate when the rank is fine) waste compute. False-negative triggers (don't speculate when the rank is slow) waste tail. The classifier needs to:

1. Distinguish routing imbalance (EPLB problem) from rank-level degradation (hardware problem).
2. Distinguish transient jitter (don't act) from sustained degradation (act).
3. Provide confidence weighting from cross-correlated signals (latency *and* thermal *and* ECC).
4. Fire fast enough to act before the next AlltoAll launches, but slow enough that one slow iteration doesn't trigger speculation forever.

This is a sequential-decision problem with prior art in queueing theory (e.g., quartile-based stragglers). The novelty lies in adapting it to sub-millisecond decision windows over GPU-fabric signals — the inputs and timescales differ from any prior straggler classifier.

### Q4. How does speculation interact with EPLB's load-balancing invariant?

EPLB optimizes "balance token load across replicas." Latency-aware routing adds "prefer fast replicas." These two axes can fight: if all top-K tokens for hot experts pile onto the fast replica, it becomes the new straggler. The optimal policy is a constrained optimization (minimize tail latency subject to load-balance constraints), but the constraint structure depends on how aggressive the latency bias is.

Open question: does the latency-aware bias get final say, or does load-balance? An adaptive policy that dynamically weights both is plausible; characterizing when it's worth the complexity is a paper section.

## 5. Working assumptions and constraints

Choices the design / paper takes as given:

- **WideEP, not standard EP.** The paper's results are about EP ≥ 32. Smaller EP doesn't have the AlltoAll tail-domination problem to the same degree.
- **MNNVL fabric memory as the data-plane substrate.** Findings should generalize to NCCL or NVSHMEM-based AlltoAll, but the kernel-level work is over MNNVL specifically.
- **Replication ≥ 2.** The DeepSeek production default. Lower replication eliminates Option A entirely (no replicas to choose between) and makes Option B's overhead more punishing.
- **Inference, not training.** Tail latency under SLA is the objective. Training tolerates per-iteration tail differently and is out of scope.
- **Single failure / straggler at a time.** Multi-straggler scenarios are an open question for follow-on work.

## 6. Success criteria

For the *engineering* deliverable (Phase 3.5 production work):

- Median p99 token latency reduction by ≥ 25 % under realistic straggler injection (one rank at 1.5–2× thermal-throttle slowdown).
- Steady-state overhead under no-straggler conditions ≤ 1 %.
- Stability: no false-positive speculation in 1000 consecutive iterations of routine load (no straggler).

For the *research* deliverable (paper):

- Theoretical framing of the speculation-vs-routing-vs-timeout phase diagram.
- Empirical characterization on real WideEP deployment with controlled straggler injection.
- Ablation showing each component (telemetry, classifier, routing-bias, kernel first-wins) contributes measurably.
- Comparison against classical straggler-mitigation policies adapted to the setting (the most likely reviewer challenge).

## 7. What this research is and isn't

**Is:** Translation of speculation primitives from independent-task scheduling to synchronous-collective inference, with the kernel-level combine semantics as the central technical content.

**Isn't:** Generic ML serving optimization. There is no contribution in "speculation generally helps inference." The contribution is precisely about the AlltoAll collective constraint and what it forces.

**Risk:** A parallel research effort (Microsoft Research, Google DeepMind, Berkeley, CMU, or one of the inference startups) may already be working on adjacent problems. The literature survey ([02-literature-survey.md](02-literature-survey.md)) maps prior art and proposes a search query list to verify novelty before committing engineering investment.
