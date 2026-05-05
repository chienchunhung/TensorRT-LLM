# Publication Venues — Targets, Timelines, Strategy

[< Back to Sub-Directory](README.md) | [↑ Up to WideEP FT](../README.md)

**Status:** Strategy document for the research-arm option in [02-literature-survey.md](02-literature-survey.md). Use only if the literature search confirms the novelty claim and the team commits to publishing.
**Last updated:** 2026-05-05.

> **Note on dates.** Conference deadlines and dates vary year to year. The timelines below are based on historical patterns (each venue's submission has been in roughly the same calendar slot for several years running) but **specific deadlines and dates must be verified at each venue's website before any submission planning.** Each entry includes the canonical URL.

## How to read this document

Five tier-1 venues are realistic primary targets. Three tier-2 venues are reasonable fallbacks or for slightly different framings. Three workshop venues are low-cost de-risking options if the research arm wants early external feedback.

Each entry: typical timeline pattern, location pattern, what gets published there, fit-for-this-work assessment, acceptance / page / format notes.

The comparison table at §4 collapses everything into a single planning view. The strategy section at §5 picks a recommended target sequence given the engineering timeline.

---

## 1. Tier-1 venues — primary targets

### MLSys (Conference on Machine Learning and Systems)

| | |
|:---|:---|
| **Canonical URL** | https://mlsys.org/ |
| **Frequency** | Annual |
| **Typical conference month** | May |
| **Typical submission deadline** | Late October / early November (≈ 6 months before) |
| **Recent locations** | Santa Clara CA (2024, 2025), Boston (2026) — usually US, sometimes Asia |
| **Fit for this work** | **Highest.** ML systems is the home venue. Both kernel-level systems work and ML serving are core topics. Reviewers expect MoE-aware contributions. |
| **Acceptance rate** | ~20-25 % typical |
| **Page length** | ~10-12 pages (excluding references) |
| **Audience blend** | Roughly 50/50 academic / industry. NVIDIA, Google, MS, OpenAI, etc. all attend |
| **Notes** | Strong fit specifically for the speculation-in-AlltoAll framing. The kernel-level combine-semantics work plays to the audience's strengths. |

### NSDI (USENIX Symposium on Networked Systems Design and Implementation)

| | |
|:---|:---|
| **Canonical URL** | https://www.usenix.org/conference/nsdi |
| **Frequency** | Annual |
| **Typical conference month** | April |
| **Typical submission deadlines** | **Two rounds**: spring (~April-May for next-year's conference) and fall (~September). Both feed into the same conference. |
| **Recent locations** | Renton / Boston / Santa Clara / Philadelphia — usually US |
| **Fit for this work** | **High** if framed as a distributed systems paper. The user's prior NSDI community gives the framing leverage. The "translating speculation from independent tasks to synchronous collectives" angle plays well here. |
| **Acceptance rate** | ~18-20 % typical |
| **Page length** | ~13 pages (excluding references) |
| **Audience blend** | More academic, but datacenter-systems-aware industry presence |
| **Notes** | Fall round (Sept submission, April conference) is typically less competitive than spring. If targeting NSDI, fall round is the strategic choice. |

### OSDI (USENIX Symposium on Operating Systems Design and Implementation)

| | |
|:---|:---|
| **Canonical URL** | https://www.usenix.org/conference/osdi |
| **Frequency** | Annual |
| **Typical conference month** | July |
| **Typical submission deadline** | December (≈ 7 months before) |
| **Recent locations** | Boston (2025, 2026), Santa Clara (2024) — usually US |
| **Fit for this work** | **Plausible** if results are exceptional. ML systems papers have appeared (AlpaServe 2023, DistServe 2024) but compete with classical OS / distributed systems heavyweights. Higher bar than MLSys or NSDI. |
| **Acceptance rate** | ~15-18 % typical |
| **Page length** | ~13 pages (excluding references) |
| **Audience blend** | Strongly academic, top-bar systems conference |
| **Notes** | Bring-the-best-version-of-this-work venue. Expect heavy reviewer scrutiny on baseline coverage and theoretical framing. Worth attempting only if the implementation has shipped and produced numbers from a real WideEP deployment. |

### SOSP (ACM Symposium on Operating Systems Principles)

| | |
|:---|:---|
| **Canonical URL** | https://sosp.org/ |
| **Frequency** | Now annual (historically biennial) |
| **Typical conference month** | November |
| **Typical submission deadline** | April-May (≈ 6 months before) |
| **Recent locations** | Seoul (2025), Austin (2024), Koblenz (2023) — international |
| **Fit for this work** | **Plausible**, similar story to OSDI. Slightly more theoretical / structural papers than OSDI. The phase-diagram contribution from §1's Q2 plays well here. |
| **Acceptance rate** | ~15-20 % typical |
| **Page length** | ~13 pages (excluding references) |
| **Audience blend** | Top systems venue, strongly academic |
| **Notes** | Same bar as OSDI. International location historically gives slightly different reviewer pool. |

### EuroSys

| | |
|:---|:---|
| **Canonical URL** | https://www.eurosys.org/ |
| **Frequency** | Annual |
| **Typical conference month** | April-May |
| **Typical submission deadlines** | **Two rounds**: fall (~October for next-year's conference) and winter (~January). Both feed into the same conference. |
| **Recent locations** | Rotterdam (2025), Athens (2024), Rome (2023) — European |
| **Fit for this work** | **High.** European-flavored systems audience appreciates MoE work, and your prior EuroSys community presence is an asset. Slightly easier bar than OSDI/SOSP. Multiple rounds give resubmission flexibility. |
| **Acceptance rate** | ~20-22 % typical |
| **Page length** | ~13 pages (excluding references) |
| **Audience blend** | Strongly academic, European industry presence |
| **Notes** | Strategic fit for second submission if NSDI / MLSys round 1 doesn't land. The two-round structure makes EuroSys an excellent fallback. |

---

## 2. Tier-2 venues — secondary targets / specialized framings

### ASPLOS (Architectural Support for Programming Languages and Operating Systems)

| | |
|:---|:---|
| **Canonical URL** | https://www.asplos-conference.org/ |
| **Frequency** | Annual |
| **Typical conference months** | March-April |
| **Typical submission deadlines** | **Three or four rounds per year** spread across the calendar; cycling submission |
| **Recent locations** | Rotterdam (2025), San Diego (2024) — alternates US / international |
| **Fit for this work** | **Plausible.** ASPLOS is architecture-leaning; the kernel-level combine-semantics work plays well, but the paper would need to lean into the GPU-memory-system angle to fit ASPLOS culture. |
| **Acceptance rate** | ~20 % typical |
| **Notes** | The rolling submission is a real flexibility advantage. Worth considering if the paper's strongest framing is "novel CUDA / GPU-fabric kernel work" rather than "ML systems work." |

### ATC (USENIX Annual Technical Conference)

| | |
|:---|:---|
| **Canonical URL** | https://www.usenix.org/conference/atc |
| **Frequency** | Annual |
| **Typical conference month** | July |
| **Typical submission deadline** | January (≈ 6 months before) |
| **Recent locations** | Co-located with OSDI in some years; usually US |
| **Fit for this work** | **Plausible**, but ATC tends to favor practical / deployment papers over novelty-claim papers. Better fit for the engineering-only paper variant. |
| **Acceptance rate** | ~22-25 % typical |
| **Notes** | Co-location with OSDI (when applicable) gives reviewer-pool overlap. Worth as a fallback after primary submission. |

### SoCC (ACM Symposium on Cloud Computing)

| | |
|:---|:---|
| **Canonical URL** | https://acmsocc.org/ |
| **Frequency** | Annual |
| **Typical conference month** | October-November |
| **Typical submission deadline** | June-July |
| **Recent locations** | Various US / international |
| **Fit for this work** | **Plausible** for the cloud-serving framing. Less competitive than OSDI / SOSP. Wrangler appeared here originally. |
| **Acceptance rate** | ~22-25 % typical |
| **Notes** | Reasonable fallback; cloud-serving angle plays. |

---

## 3. Workshops — low-cost early feedback

These are useful if the team wants external review without committing to a full conference paper. Each is roughly 4-page paper, lower acceptance bar, primary value is feedback and visibility.

### HotOS (Workshop on Hot Topics in Operating Systems)

| | |
|:---|:---|
| **Canonical URL** | https://sigops.org/s/conferences/hotos/ |
| **Frequency** | Biennial (alternates with HotCloud) |
| **Typical month** | June-July |
| **Typical deadline** | February-March |
| **Page length** | ~5-6 pages |
| **Fit** | Position-paper venue. Good for "we think this problem is real and here's a sketch of approach" — exactly the radar-level contribution of §7.5. |
| **Notes** | If the literature survey confirms novelty and the team wants community signal before committing to engineering, HotOS is the right venue. |

### MLSys workshops

| | |
|:---|:---|
| **Frequency** | Co-located with main MLSys conference (annual) |
| **Typical deadlines** | February-March (workshop selection happens at MLSys) |
| **Page length** | ~4-6 pages |
| **Fit** | Workshop on ML systems specifically. Lower visibility than the main conference but a real audience. |

### ICML / NeurIPS systems-track workshops

| | |
|:---|:---|
| **Frequency** | Annual |
| **Typical deadlines** | Mid-year (depends on workshop) |
| **Page length** | ~4-8 pages |
| **Fit** | Less ideal — these tend to favor ML-leaning papers. The systems-track workshops at NeurIPS / ICML can work but the audience is less aligned. |

---

## 4. Comparison table — single planning view

| Venue | Tier | Conference month | Submission deadline | Acceptance | Best fit for our framing | Strategic role |
|:---|:---:|:---|:---|:---:|:---|:---|
| **MLSys** | 1 | May | Oct-Nov (prev year) | ~20-25 % | ML systems + kernel-level work | **Primary target** for round 1 |
| **NSDI** | 1 | April | Apr-May / Sept-Oct | ~18-20 % | Distributed systems + collective speculation | **Primary target**, fall round if MLSys doesn't land |
| **OSDI** | 1 | July | December | ~15-18 % | Highest-bar systems venue | Aspirational; only if real numbers are exceptional |
| **SOSP** | 1 | November | April-May | ~15-20 % | Top systems venue | Aspirational; phase-diagram contribution plays here |
| **EuroSys** | 1 | April-May | Oct / Jan | ~20-22 % | European systems audience | **Strong fallback** — two rounds, manageable bar |
| **ASPLOS** | 2 | March-April | Rolling (3-4/yr) | ~20 % | Architecture / GPU-fabric framing | Flexibility advantage from rolling submission |
| **ATC** | 2 | July | January | ~22-25 % | Practical / deployment papers | Engineering-only fallback |
| **SoCC** | 2 | Oct-Nov | June-July | ~22-25 % | Cloud-serving framing | Decent fallback |
| **HotOS** | Workshop | June-July | Feb-March | ~30 % | Position paper | **Early feedback** before committing |
| **MLSys workshops** | Workshop | May (co-located) | Feb-March | varies | ML systems | Visibility |

---

## 5. Strategic submission plan

### Engineering timeline assumed for this plan

Based on [§7.5](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking) sizing:

- **A + D production track**: ~6-7 weeks. Lands as part of Phase 3.5.
- **Telemetry foundation + classifier**: ~5-7 weeks. Foundation for both production and research arms.
- **Option B research arm**: ~10-14 weeks of dedicated implementation, plus ~4-6 weeks of experiments / paper writing. Total ~4-5 months.
- **Lit search**: ~3 days. Confirms novelty before commitment.

Earliest realistic paper-ready timeline: ~6-7 months from kickoff if Option B implementation goes smoothly.

Two viable target sequences, depending on which framing is strongest and how much risk the team wants in a single conference cycle. The team picks one or the other; **they cannot run in parallel** because of concurrent-submission policy. A workshop submission (HotOS) is the only safe parallel option.

#### Variant A — MLSys primary, NSDI next-cycle fallback (default recommendation)

Strategy: take a single best-fit shot at the strongest-aligned venue; if rejected, fall back to NSDI in the next conference cycle. Conservative — preserves NSDI as a clean second cycle rather than burning two venues in one.

**Phase 1 — Position paper at HotOS (optional, low cost).**

If HotOS is in the calendar window (biennial; check whether next instance lines up with the engineering kick-off), submit a position paper based on the §7.5 framing + literature survey. ~2 weeks of writing on top of the existing material.

- **Deadline:** Whichever calendar slot is closest after the literature survey completes.
- **Cost:** Low. The §7.5 sketch + 01-problem-statement.md are most of a HotOS paper already.
- **Value:** Community signal. If accepted, the talk is at OSDI/SOSP-co-located workshop, surfacing the work to the right audience before the full paper lands.

**Phase 2 — Primary submission at MLSys.**

Time the engineering arm to land before the MLSys submission deadline (typically late Oct / early Nov for May conference).

- **If kickoff is mid-2026:** target MLSys 2028 submission Oct/Nov 2027, conference May 2028.
- **If kickoff is later:** target MLSys 2029, slip by one year.

MLSys is the strongest fit and has the most aligned reviewer pool. Acceptance gives a top-tier ML systems publication.

**Phase 3 — Fallback at NSDI next-cycle spring, then EuroSys.**

If MLSys 2028 reject (~Jan/Feb 2028):
- **NSDI 2029 spring round** (typically April-May submission for next-April conference). Sufficient time after MLSys notification to revise; the framing pivots from MLSys's ML-systems angle to NSDI's distributed-systems angle.
- **EuroSys winter round** (typically Jan for April-May conference). Strong further fallback if NSDI rejects.

If all three reject, the paper has gone through three rounds of review with feedback at each — strong candidate for resubmission to OSDI or SOSP with revisions.

> **Note: same-cycle fallback is not available with Variant A.** MLSys 2028 deadline (Oct/Nov 2027) lands *after* the NSDI 2028 fall-round deadline (~Sept 2027), and MLSys 2028 notification (~Jan/Feb 2028) lands *after* the NSDI 2028 spring-round deadline (~April-May 2027 — already passed by then). So MLSys-then-NSDI in the same conference cycle is mechanically impossible. NSDI 2029 spring (~April-May 2028 deadline) is the earliest fallback.

#### Variant B — NSDI primary, MLSys same-cycle fallback

Strategy: lead with NSDI (distributed-systems framing); if rejected, use the ~6-10-week window between NSDI notification and MLSys deadline for revision-and-resubmit in the *same* conference cycle. Aggressive — two shots in 2028 cycle, at the cost of a tighter revision turnaround.

**Phase 1 — Position paper at HotOS (optional, same as Variant A).**

**Phase 2 — Primary submission at NSDI spring round.**

Time the engineering arm to land before the NSDI spring deadline (typically late April / early May for the next-April conference).

- **If kickoff is mid-2026:** target NSDI 2028 spring submission ~April-May 2027, conference April 2028.
- **If kickoff is later:** target NSDI 2029 spring, slip by one year.

NSDI rewards the "novel distributed systems primitive" angle. The framing leans into "translating speculation from independent tasks to synchronous collectives" — your prior NSDI community gives reviewer / audience advantages.

**Phase 3 — Same-cycle fallback at MLSys.**

If NSDI 2028 spring reject (~mid-late August 2027), revise based on reviewer comments and submit to **MLSys 2028** (deadline late Oct / early Nov 2027). Same-cycle conference (May 2028).

Critical constraints in this fallback:
- **Revision window: ~6-10 weeks** between NSDI notification and MLSys deadline. Sufficient for prose revision, addressing reviewer comments, and one ablation experiment on existing data. **Not** sufficient for major new experiments requiring real WideEP deployment time.
- **No concurrent submission.** Cannot submit to MLSys until NSDI rejection is final. If NSDI offers "revise and resubmit" instead of clean reject, the MLSys window is forfeit until the NSDI revision concludes.
- **Reframing cost.** NSDI optimizes for distributed-systems contribution claim; MLSys optimizes for ML-systems contribution. Revision must absorb framing changes on top of reviewer comments — the introduction, related-work, and contribution-claim sections often need substantive rewrites, not just polish.
- **Rebuttal cycle.** USENIX venues including NSDI have a rebuttal phase. Typical timeline: deadline ~April-May → reviews ~July → rebuttal ~late July → final notification ~mid-late August. If NSDI's review cycle slips, the MLSys window tightens further.

**Phase 4 — Further fallback if both reject.**

If both NSDI 2028 + MLSys 2028 reject:
- **EuroSys 2029 winter round** (~Jan 2029 deadline) — ~2-3 months after MLSys notification, sufficient for substantive revision.
- **NSDI 2029 spring round** is also an option but the same-conference repeat may face explicit-or-implicit deflation in the reviewer pool.

#### When to pick which

| | Variant A (MLSys primary) | Variant B (NSDI primary) |
|:---|:---|:---|
| First-shot venue | MLSys 2028 (May) | NSDI 2028 spring (April) |
| First-shot fit | Strongest for ML systems framing | Strong for distributed-systems framing |
| Same-cycle fallback available? | **No** (mechanically blocked by deadline ordering) | **Yes** (NSDI Aug → MLSys Oct/Nov) |
| Earliest fallback if first reject | NSDI 2029 spring (next cycle) | MLSys 2028 (same cycle, tight revision window) |
| Total shots in 2028 cycle | One (MLSys 2028) | Two (NSDI 2028 + MLSys 2028) |
| Total shots through 2029 | Two (MLSys 2028, NSDI 2029 spring) | Three (NSDI 2028, MLSys 2028, EuroSys 2029 winter) |
| Reframing burden if first reject | Significant (MLSys → NSDI angle) | Significant (NSDI → MLSys angle) and **on a tight schedule** |
| Risk of two-rejection year | Low (single shot per cycle) | Real (both venues in one cycle if both reject) |
| Best when | Confidence is high; framing leans ML-systems; want to preserve venue diversity | Want two attempts in same cycle; framing leans distributed-systems; have the bandwidth for fast revision |

**Defensibility check.** Both variants are defensible. Variant A is the slightly more conservative default — it doesn't burn two top venues in a single bad cycle and gives more time per revision. Variant B trades that conservation for an extra at-bat in the same year, which can be the right call if the team's prior is "we have a paper, let's try multiple shots quickly."

The doc retains Variant A as the recommended default but Variant B is fully scoped above for the team that prefers it.

### Concurrent submission policy

**None of these venues allow concurrent submission.** Most require a 6-month exclusivity for substantive papers. A workshop submission (HotOS) does not count as a prior publication for full conferences and is the only safe "parallel" option.

The sequential plan above respects this — only one full conference submission active at a time.

### Industry vs academic considerations

- **MLSys and NSDI:** mixed industry / academic; NVIDIA Research papers have appeared at both. Industry audience appreciates real-deployment numbers.
- **OSDI / SOSP:** academic-leaning; reviewer pool will be skeptical of "vendor numbers." Need to lean on the open-source TRT-LLM repo as the credibility anchor.
- **EuroSys:** balanced; European industry presence (DeepMind UK, Mistral, Aleph Alpha, etc.).
- **ATC:** practical / industry-friendly. The engineering-only paper variant fits here.

### Travel / location considerations

This is sometimes a meaningful factor in venue choice. Most of the venues above rotate location annually; only NSDI / OSDI / ATC are consistently US-based. SOSP and EuroSys are international. If the team has location preferences, these vary year-to-year and should be checked at submission time.

### What "success" looks like

Three honest outcome scenarios per variant:

**Variant A (MLSys primary):**

1. **Best case.** MLSys 2028 acceptance. Talk at MLSys May 2028. Industry follow-up. Production deployment validates the approach. Total elapsed: ~22 months from mid-2026 kickoff.
2. **Middle case.** MLSys 2028 reject (~Jan/Feb 2028); NSDI 2029 spring acceptance (~Aug 2028). Conference April 2029. Total elapsed: ~34 months from kickoff.
3. **Reasonable failure case.** MLSys 2028 + NSDI 2029 both reject; submit to EuroSys 2029 winter round or SoCC. Paper lands at less-prestigious venue. Total elapsed: ~3 years from kickoff.

**Variant B (NSDI primary, same-cycle fallback):**

1. **Best case.** NSDI 2028 spring acceptance (~Aug 2027). Talk at NSDI April 2028. Total elapsed: ~22 months from mid-2026 kickoff. (Same conference cycle as Variant A's best case, just at a different venue.)
2. **Middle case.** NSDI 2028 reject (~Aug 2027); MLSys 2028 fallback acceptance (~Jan/Feb 2028). Conference May 2028. Total elapsed: ~22 months. **Two papers' worth of work in one cycle, but only one publication.**
3. **Reasonable failure case.** NSDI 2028 + MLSys 2028 both reject in the same cycle; EuroSys 2029 winter as further fallback. Paper lands at EuroSys (~April 2029). Total elapsed: ~34 months. **The two-rejection-in-one-year scenario is a bigger morale hit than Variant A's equivalent**, and reviewer feedback overlap may force more substantial revision.

In every scenario across both variants, the **production engineering value** (Phase 3.5 A + D, plus telemetry foundation) is independently realized. Publication is a side benefit, not a dependency for production progress.

---

## Action items if this path is pursued

1. Run literature search per [02-literature-survey.md §6](02-literature-survey.md#6-search-plan) — ~3 days. Determines whether the novelty claim holds.
2. Confirm internal NVIDIA publication clearance for the work area. Usually permissive for systems papers tied to open-source TRT-LLM, but worth checking.
3. **Pick Variant A or Variant B**, then pick the target submission deadline based on engineering timeline:
   - **Variant A:** MLSys late-Oct/Nov is the primary target. NSDI 2029 spring (~April-May 2028) is the next-cycle fallback.
   - **Variant B:** NSDI spring (~April-May) is the primary target. MLSys late-Oct/Nov is the same-cycle fallback if NSDI rejects.
4. Decide on HotOS position-paper as parallel low-cost option, conditional on calendar alignment. Same in either variant.
5. Begin paper outlining 4-6 weeks before primary deadline. Reuse problem statement (01) and lit survey (02) as direct inputs to related-work and introduction sections.
6. **For Variant B specifically:** be ready to begin revision-and-reframe work the day NSDI notification arrives. Pre-stage the MLSys-framed introduction / related-work sections during the NSDI review period (without changing the paper itself, which is in review) so the ~6-10 week revision window is spent on reviewer-comment work rather than reframing from scratch.

These steps map cleanly to the engineering plan in [§7.5](../07-phase-3-beyond-failover.md#75-straggler-mitigation-forward-looking) and don't require additional planning infrastructure.
