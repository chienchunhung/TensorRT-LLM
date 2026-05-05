# Literature Survey — Speculation, Stragglers, and ML Inference

[< Back to Sub-Directory](README.md) | [↑ Up to WideEP FT](../README.md)

**Status:** Working document. Initial map of adjacent work + a search plan to verify novelty before committing engineering investment.
**Pairs with:** [01-problem-statement.md](01-problem-statement.md).
**Last updated:** 2026-05-05.

This is a structured map of where the speculation-for-stragglers idea lives in the literature, organized by adjacency to the specific problem of *speculation in synchronous AlltoAll for MoE inference*. The goal is to (a) precisely locate what's been done, (b) identify what hasn't, and (c) give a concrete search plan for novelty verification before any submission.

## 1. Classical batch-job speculation (most heavily cited prior art)

The foundational work, all in distributed batch-job / data-parallel scheduling. Each addresses a different sub-problem of speculation in independent-task settings. None addresses synchronous collectives or sub-millisecond decision windows; most operate at task granularity (seconds to minutes).

| Paper | Venue / year | One-line characterization |
|:---|:---|:---|
| MapReduce (Dean & Ghemawat) | OSDI 2004 | Backup-task mechanism for stragglers in batch reducers — speculation as a system primitive |
| LATE (Zaharia et al.) | OSDI 2008 | Longest Approximate Time to End — speculative scheduling that prioritizes by remaining time, not progress |
| Mantri (Ananthanarayanan et al.) | OSDI 2010 | Cause-aware mitigation: predict stragglers from observed input skew / runtime behavior, choose between restart and duplicate |
| Dolly (Ananthanarayanan et al.) | NSDI 2013 | Proactive cloning of small batch jobs to bound completion-time tail; argues against waiting for stragglers to appear |
| GRASS (Ananthanarayanan et al.) | NSDI 2014 | Speculation under deadlines; approximation-aware — when do you cut a slow task vs wait? |
| Wrangler (Yadwadkar et al.) | SoCC 2014 | ML-based predictive straggler mitigation using node-level signals |
| Hopper (Ren et al.) | SIGCOMM 2015 | Joint speculation + scheduling across competing jobs |
| Sparrow / Tetris / others | NSDI / SIGCOMM 2013–2017 | Various scheduling-side approaches; speculation as one of several mechanisms |

**Common framing.** Independent units of work (map / reduce / data-parallel tasks). Speculation = clone a task; race the clones; commit whichever finishes first; cancel the slow one. The decision is made by a centralized or distributed scheduler that observes per-task progress.

**What none of these address (and the WideEP work would):**

- Synchronous collectives where every participant's contribution is consumed by every other participant in the same iteration. No prior speculation paper addresses speculation in a setting where the unit of work is *a collective rather than a task*.
- Sub-millisecond decision windows. All prior work operates at hundreds of milliseconds to seconds.
- Replica-as-routing-state. Classical speculation spawns clones. EPLB has pre-allocated replicas in routing state. Speculation reduces to *replica selection*, not *clone spawning* — qualitatively different.
- Inference-SLA (per-token p99 / p99.9) rather than job-completion-time tail.
- GPU-fabric-specific signals (NVLink retry counts, NVML thermal, ECC) as predictive features.

This is the section a paper would need to ground precisely; the contribution is "translating ideas from this body of work to a setting these papers explicitly didn't address."

## 2. ML inference / serving systems

The most adjacent body of work — ML serving systems that have addressed stragglers, latency tails, or routing in large-scale inference. None directly addresses synchronous-collective speculation in MoE AlltoAll, but several are within citation distance.

| Paper / system | Venue / year | Why relevant |
|:---|:---|:---|
| Clipper (Crankshaw et al.) | NSDI 2017 | Predictive serving, model selection, latency bounds — the granddaddy of ML serving systems |
| InferLine (Crankshaw et al.) | SoCC 2020 | ML serving pipeline orchestration with latency / cost trade-offs |
| AlpaServe (Li et al.) | OSDI 2023 | Statistical multiplexing for parallelism in LLM serving — pioneered ideas around dynamic placement |
| Orca (Yu et al.) | OSDI 2022 | Continuous batching for LLM serving — establishes the iteration as the unit of scheduling |
| vLLM (Kwon et al.) | SOSP 2023 | PagedAttention; foundational for current LLM serving |
| Splitwise (Patel et al.) | ISCA 2024 | Disaggregated prefill / decode (the parent design's Phase 1-DS context) |
| DistServe (Zhong et al.) | OSDI 2024 | Disaggregated serving with phase-specific resource allocation |
| Mooncake (Qin et al.) | FAST 2025 | KV-cache-centric serving; introduces `activeRanks` masking that SGLang now uses |

**What's explored.** Disaggregation, batching, KV cache management, model parallelism strategies, request-level routing. **What isn't.** Speculation as a primitive for tail-latency mitigation in tightly-coupled MoE collectives. The serving systems above mostly assume a non-MoE or single-node MoE workload; tail latency for them is dominated by request-mix variance and KV-pressure, not by collective straggler dynamics.

**The closest adjacent paper that needs careful handling.** SGLang's Elastic EP (LMSYS blog post, March 2026) addresses *failure* in WideEP via Mooncake's `activeRanks` masking. It does not address *stragglers* — Elastic EP is binary (rank in or rank out), not graded. A paper in this space should cite Elastic EP and clearly frame its contribution as orthogonal: Elastic EP handles hard failures, our work handles soft ones.

## 3. MoE-specific systems

MoE-aware systems work that is closer to the actual data plane this research touches.

| Paper / system | Venue / year | Why relevant |
|:---|:---|:---|
| GShard (Lepikhin et al.) | ICLR 2021 | First large-scale MoE training infrastructure; established AlltoAll as the dominant communication primitive |
| Switch Transformer (Fedus et al.) | JMLR 2022 | Top-1 routing, infrastructure scaling for sparsely-activated models |
| DeepSpeed-MoE (Rajbhandari et al.) | ICML 2022 | Production MoE training with expert parallelism |
| Megatron-MoE (Shoeybi et al.) | various | TP × PP × EP for very large MoE training |
| FastMoE / Tutel | various 2021–2023 | Optimized MoE kernels and AlltoAll implementations |
| DeepEP (DeepSeek) | open source 2025 | NVSHMEM-based AlltoAll with low-latency variant |
| AnchorTP | arxiv 2024 (2511.11617) | Resilient EP with unequal-width partitioning — adjacent FT angle, not straggler-focused |
| MoC-System (Lin et al.) | ASPLOS 2025 | Efficient FT for MoE training (training-focused; addresses checkpointing / recovery, not stragglers) |

**What's explored.** Throughput optimization for MoE training and inference — kernel fusion, AlltoAll algorithms, expert placement, communication-computation overlap, training-time fault tolerance. **What isn't.** Straggler mitigation in AlltoAll. The closest is MoC-System (ASPLOS 2025) which addresses MoE training failure recovery; it does not address steady-state straggler dynamics or speculation.

## 4. Replication-aware FT and adjacent angles

A smaller body of work on replication-aware fault tolerance and speculative execution in HPC settings — closer in spirit to the "replicas-as-routing-state" framing.

| Topic / paper | Venue / year | Why relevant |
|:---|:---|:---|
| FA-MPI / ULFM literature | various 2010–2020 | Fault-tolerant MPI extensions — adjacent to L2 control-plane FT work, not directly straggler |
| Imitate (Liu et al.) | NSDI 2017 | Replicate operations across data center backups for tail mitigation in storage |
| Tail at scale (Dean & Barroso) | CACM 2013 | Foundational essay on tail-latency techniques in large-scale services — the philosophical parent |
| Storage-side speculation (Vulimiri et al.) | NSDI 2013 | Hedged requests in datacenter storage; closest analog to "issue both replicas, take first" in inference |

The hedged-requests pattern (Vulimiri NSDI 2013) is the closest spiritual cousin to Option B in §7.5. **Its limit there: hedged requests work on independent storage operations; our setting requires the collective combine kernel to handle the race.**

## 5. What's NOT in the literature (the precise novelty claim)

A paper in this space has an opportunity to make a precise novelty claim, defensible against any of the above:

1. **Speculation in synchronous collectives.** No prior paper formalizes speculation when the unit of work is a synchronous N-way collective rather than an independent task. The closest parallel is hedged requests in datacenter storage, but those operate over independent operations, not consensus collectives.

2. **First-wins kernel semantics in MNNVL combine.** No prior paper addresses combine-accumulator atomicity for first-valid-response wins over symmetric-memory PTX synchronization. The combine kernel is unique to TRT-LLM's NVLinkOneSided implementation; comparable kernels in NCCL or NVSHMEM exist but their internals are not tunable in the same way.

3. **Replica-as-routing speculation, not clone-spawn speculation.** Classical work spawns clones. We bias routing among pre-allocated replicas. The two are mathematically equivalent under specific conditions (replication ≥ K, all replicas equally-loaded) but the engineering trade-offs are very different. No prior paper makes this distinction.

4. **GPU-fabric signals as straggler predictors.** Wrangler used ML on host-level signals; no public work uses NVLink retry counts, NVML thermal, or ECC correctable counts as features for sub-millisecond speculation decisions on the kernel critical path.

5. **Phase diagram of speculation-vs-routing-vs-timeout policies.** Each of A / B / C / D in [§7.5](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking) dominates in a different operating regime (straggler severity × replica coverage × SLA tightness). A characterization paper that produces this diagram empirically is a clean structural contribution.

A paper that argues all five points cleanly, with measurements on real WideEP hardware, is a credible MLSys / NSDI / EuroSys submission.

## 6. Search plan

Before any commitment to publication-driven engineering, run a focused literature search to confirm the novelty claim. The plan below targets ~3 days of someone's time and is designed to be exhaustive enough to defend the claim against a thorough reviewer.

### Search queries (ArXiv + Google Scholar + DBLP)

| Query | Why |
|:---|:---|
| `speculative execution AlltoAll` | Direct hit on the central concept |
| `speculation MoE inference` | Recent ML systems work directly adjacent |
| `straggler mitigation expert parallelism` | Synonym pivot |
| `tail latency AlltoAll collective` | Tail-focused angle |
| `redundant computation distributed inference` | Alternative phrasing for speculation |
| `hedged request inference serving` | Datacenter / storage analog adapted to inference |
| `synchronous collective fault tolerance` | Classical-systems angle (FT, but adjacent) |
| `MoE serving tail latency` | Recent inference systems papers |
| `GPU thermal throttling distributed training/inference` | Hardware-signal angle |
| `dynamic replica selection collective communication` | Alternative phrasing for Option A |
| `first-wins combine kernel` | The specific kernel-level technical content |
| `latency-aware load balancing MoE` | Adjacent to Option A |

### Venues / pages to scan

- ArXiv `cs.DC`, `cs.LG`, `cs.AR` — last 18 months
- MLSys 2024, 2025, 2026 proceedings
- NSDI 2024, 2025, 2026 proceedings
- OSDI 2024, 2025, 2026 proceedings
- SOSP 2024, 2025 proceedings
- EuroSys 2024, 2025, 2026 proceedings
- ATC 2024, 2025, 2026
- ASPLOS 2024, 2025, 2026
- HPDC 2024, 2025, 2026
- ICML / NeurIPS systems-track papers 2024–2026

### Authors and groups to track

Researchers / labs that have published in adjacent areas and may be working on related problems. Worth scanning their recent ArXiv submissions specifically.

| Group / individual | Why track |
|:---|:---|
| Microsoft Research Systems & Networking | DeepSpeed, MS inference work — adjacent MoE infrastructure |
| Berkeley Sky / RISELab | AlpaServe, vLLM authors — ML serving research with systems flavor |
| CMU Catalyst | LLM serving, MoE work |
| Stanford MAST | ML systems |
| UCSD CSE (Voelker, Snoeren) | Datacenter networking, inference systems |
| MIT CSAIL | Various, DistServe authors |
| Tsinghua / Beijing Institute of AI / DeepSeek | DeepEP authors; closest to the actual deployment context |
| Moonshot AI | Mooncake authors; production WideEP infrastructure |
| LMSYS / SGLang team | Elastic EP authors; will know the WideEP FT space well |
| NVIDIA Research | Internal cousin work; check published papers and ArXiv |
| Inflection / Anthropic / OpenAI / Meta | Less public but worth scanning if any ML systems papers appear |

### What disqualifies the novelty claim

If the search returns:

- An MLSys / NSDI / SOSP paper from 2024–2026 that addresses speculation specifically in synchronous AlltoAll — the contribution is gone unless it's a different angle (different speculation policy, different correctness proof, etc.).
- A paper on MoE-AlltoAll tail-latency optimization that uses hedged-routing or first-wins combine semantics — even partial overlap requires reframing.
- A paper that produces the speculation-vs-routing-vs-timeout phase diagram — the structural contribution is gone.

If the search returns *adjacent* work (Elastic EP, MoC-System, DeepEP, etc.) but nothing on speculation in synchronous collectives, the novelty claim holds and the paper is differentiated by precisely citing each adjacent work and naming what's distinct.

### How to record findings

A simple table per paper found:

| Paper | Venue / year | Addressed | Not addressed (vs our claim) | Citation distance |
|:---|:---|:---|:---|:---|

That table, populated honestly from the search, drives the related-work section of the eventual paper and the novelty defense in the design doc.

## 7. Recommendation

Run the search plan in §6 before committing engineering investment in Option B. ~3 days. Outcome shapes whether the work is:

- **Engineering-only** (A + D + C in §7.5 production track, no paper) — if literature search uncovers parallel work that subsumes the novelty.
- **Engineering + paper** (A + D + C in production *and* B as research arm) — if novelty is confirmed and the team has appetite for the additional ~4–5 months of dedicated research engineering.

The search plan also doubles as the literature review for the eventual paper, so the cost is not wasted in either outcome.

[03-publication-venues.md](03-publication-venues.md) is the next document — concrete venue analysis with timelines if the engineering + paper path is pursued.
