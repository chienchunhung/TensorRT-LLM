# Straggler Speculation — Research Exploration

[↑ Up to WideEP FT](../README.md)

**Status:** Forward-looking research exploration. Not committed engineering work.
**Connects to:** [§7.5 Straggler mitigation (forward-looking)](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking) in the parent design.
**Created:** 2026-05-05.

## What's here

This sub-directory captures the *research arm* of straggler mitigation — the speculative-execution direction that may merit a publication in addition to its production value. The parent design ([§7.5](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking)) sketches four options (A latency-aware routing, B speculative redundant compute, C shadow rank as performance hot-spare, D tail-cutting timeout) at radar level. **Option B is the research-grade piece**: speculation in synchronous AlltoAll inference is genuinely under-explored in the literature.

These three documents are the work product needed *before* committing engineering investment in Option B:

| File | What it covers |
|:---|:---|
| [01-problem-statement.md](01-problem-statement.md) | Precise framing of the problem and the novel research questions. Why classical speculation doesn't directly apply, what the actual contribution would be, and what success criteria look like. |
| [02-literature-survey.md](02-literature-survey.md) | Map of prior art in classical batch speculation, ML serving, MoE systems, and adjacent FT work. **Includes a concrete search plan to verify novelty before publication-driven engineering kicks off.** |
| [03-publication-venues.md](03-publication-venues.md) | Conference / workshop targets with timelines (submission deadlines, conference dates, locations), acceptance-rate / page-length / audience details, and a recommended submission strategy. |

## How this relates to the production engineering

The production engineering for straggler mitigation (Options A + D in §7.5, possibly + C once §6.3 lands) is independent of any publication path. It ships as part of Phase 3.5 regardless of whether a paper happens. **Option B is the only one that requires publication-driven engineering** — its kernel-level first-wins combine semantics is the central technical contribution and warrants its own design once the research path is committed.

So the timeline is roughly:

```
Phase 1+2 production work  ──────────────────┐
                                              │
Phase 3 (3a–3e in §8.3)                      ──┐
                                                │
Lit search (~3 days)              ──────────────────┐
                                                    │
A + D production track  ────────────────────────────────┐
                                                        │
   ┌─ Option B research arm (only if novelty confirmed) ─┘
   │                                                      
   │                                                      
   └─ Paper writing + submission                          
```

Lit search runs *before* committing to Option B. If novelty is not confirmed, A + D + C ship in production and there's no paper.

## Read order

For someone reading this for the first time:

1. **Start with [01-problem-statement.md](01-problem-statement.md)** — establishes what the problem is and what the contribution would be.
2. **Then [02-literature-survey.md](02-literature-survey.md)** — see what's already out there and what the search plan looks like.
3. **Then [03-publication-venues.md](03-publication-venues.md)** — only relevant if the research path is being seriously considered.

## When to open this work track

This research is on hold pending two gates:

1. **Phase 1 MVP ships** — the production critical path doesn't compete with research engineering until MVP is in.
2. **Lit search confirms novelty** — ~3 days of focused search per the plan in 02. If parallel work covers the contribution, the research arm is shelved (production A + D + C still ships).

Until both gates pass, this sub-directory is preserved as the parking-lot for the thinking. The framing and venue analysis don't decay quickly; they remain useful when the team is ready.
