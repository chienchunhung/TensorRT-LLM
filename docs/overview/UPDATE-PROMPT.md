# Periodic Update Prompt — TRT-LLM Architecture & Codebase Learning Overview

> **How to invoke (in Cursor / Claude Code):**
>
> ```
> Read and execute @docs/overview/UPDATE-PROMPT.md against the latest TRT-LLM main.
> Run autonomously — do not pause for permission. Request sandbox/network perms proactively.
> ```
>
> Optional flags you can append to the invocation:
> - `MODE=full`     (default) — refresh all 16 files
> - `MODE=targeted FILES="04,05-01,05-03"` — only touch listed files
> - `MODE=competitive-only` — skip codebase pass; only refresh §4 + §5 + §6
> - `SINCE=2026-01-15` — override the auto-detected "last update" anchor
> - `NO_PUSH=1` — commit locally but skip `git push`
> - `NO_TAG=1` — skip creating the `docs-overview/YYYY-MM-DD` tag

---

## 0. Goal & Deliverables

**Goal.** Refresh the TRT-LLM Architecture & Codebase Learning Overview (16 files under `docs/overview/`) so it accurately reflects:

1. **Current TRT-LLM behavior** on `upstream/main` (or `origin/main`) — features, defaults, deprecated paths, new modules, recent design notes, and shipped PRs since the last update.
2. **Re-assessed challenges, opportunities, innovations** — promote/demote priorities as the landscape shifts.
3. **Competitive landscape** — vLLM, SGLang, LMCache, NVIDIA Dynamo, TGI, Mooncake, and any newcomer that materially affects positioning.
4. **Hardware advancement** — NVIDIA roadmap (Blackwell refresh, Rubin/Rubin Ultra), AMD MI3xx/MI4xx, Google TPU v6e/v7, Groq LPU, Cerebras WSE-3, AWS Trainium 3, Intel Gaudi 3+, Etched Sohu, d-Matrix Corsair, SambaNova SN40L, etc.
5. **Latest open-source / academic work** — relevant arXiv papers (≤6 months old) and notable OSS releases that change what's achievable.

**Deliverables (all must exist when this prompt finishes):**

- [ ] All 16 files under `docs/overview/` updated, with inline `[UPDATED YYYY-MM]` markers on changed sections.
- [ ] `docs/overview/README.md` "Last updated" date and version pins refreshed.
- [ ] `docs/overview/CHANGELOG.md` — appended with a new dated entry summarizing what changed (TRT-LLM, competitors, hardware, academic) plus per-file highlights with citations.
- [ ] Snapshot directory `docs/overview/.snapshots/<previous-date>/` — copy of the prior version for offline diff (only if not already snapshotted).
- [ ] Single git commit on `docs-and-plans` with DCO sign-off.
- [ ] Tag `docs-overview/<YYYY-MM-DD>` at HEAD (unless `NO_TAG=1`).
- [ ] Branch + tag pushed to `origin` (unless `NO_PUSH=1`).

---

## 1. Autonomy Directives (read first, obey throughout)

1. **Do not pause for the user.** Never ask clarifying questions mid-run. Resolve ambiguity using the rules below or pick the most defensible default and note it in the changelog.
2. **Request perms proactively.** When a tool needs `network` / `full_network` / `all`, request it on the first call rather than failing and retrying.
3. **Run independent tool calls in parallel** — codebase greps, web searches, doc reads — to keep wall-clock time low.
4. **Only stop on hard errors** (e.g., merge conflict you can't resolve, network outage, branch divergence requiring human review). On stop, write what you tried into `CHANGELOG.md` under a `### Blocked / Skipped` heading and continue with the rest.
5. **No vaporware.** Every claim about TRT-LLM must cite a `path:line` or PR/issue number. Every competitor / hardware / academic claim must cite a URL or arXiv ID. If you can't cite it, either find a citation or remove the claim.
6. **Preserve user-authored prose** unless it is factually wrong or stale. Prefer additive `[UPDATED YYYY-MM]` callouts over wholesale rewrites.
7. **Respect repo rules** in `AGENTS.md` / `CLAUDE.md`:
   - `git commit -s` (DCO sign-off, no AI co-authors).
   - PR / commit title format: `[None][docs] overview refresh YYYY-MM` (or `[TRTLLM-XXXX][docs] ...` if a JIRA exists).
   - If `pre-commit` modifies files, re-stage and amend.

---

## 2. Workflow

### Phase A — Setup, snapshot, and anchor detection

1. **Identify worktree.**
   - If currently on `docs-and-plans`, work in place.
   - Otherwise, prefer a sibling worktree to avoid disturbing the user's current branch:
     ```
     git fetch origin docs-and-plans
     git worktree add ../TRTLLM-worktree-docs docs-and-plans
     cd ../TRTLLM-worktree-docs
     ```
     Remember to `git worktree remove` at the end.
2. **Sync.**
   ```
   git fetch origin
   git fetch upstream  # if 'upstream' remote exists; ignore failure
   git pull --ff-only origin docs-and-plans
   ```
   If `--ff-only` fails, stop and report — manual rebase is required.
3. **Determine the previous-update anchor (`PREV_TAG`, `PREV_SHA`, `PREV_DATE`).** In order:
   - Latest tag matching `docs-overview/*` → `PREV_TAG`, `PREV_SHA`, derive `PREV_DATE` from tag name.
   - Else parse `Last updated: <Month YYYY>` line from `docs/overview/README.md` and find the latest commit modifying `docs/overview/` before that month.
   - Else use the oldest commit that touches `docs/overview/`.
   - Honor `SINCE=` override if provided.
4. **Determine the previous code anchor.** Find the `tensorrt_llm/version.py` value at `PREV_SHA` and the corresponding `upstream/main` (or `origin/main`) commit at `PREV_DATE`. Save as `PREV_CODE_SHA`.
5. **Snapshot.** If `docs/overview/.snapshots/<PREV_DATE>/` does not exist, create it and copy every current `docs/overview/*.md` there (excluding `.snapshots/` itself, this prompt, and `CHANGELOG.md`). This is the offline diff anchor for next time, even if tags are pruned.
6. **Compute today's anchor (`TODAY = YYYY-MM-DD`).** All inline markers and the new tag use this date.

### Phase B — Codebase delta inspection

Use the `trtllm-codebase-exploration` skill where useful. Run searches in parallel.

1. **Repo-wide change scan.**
   ```
   git log --since="<PREV_DATE>" --pretty='%h %ad %s' --date=short -- tensorrt_llm/_torch tensorrt_llm/llmapi tensorrt_llm/executor tensorrt_llm/serve cpp/tensorrt_llm
   ```
   Categorize commits by feature area; cap to ~200 most-impactful (filter out `[chore]`, dependency bumps, test-only, format-only).
2. **Version + headline features.**
   - Read `tensorrt_llm/version.py`.
   - Skim `docs/source/release-notes.md` (or whatever the current release-notes file is) for entries since `PREV_DATE`.
   - Note any breaking changes flagged in `docs/source/developer-guide/api-change.md`.
3. **Per-feature deep-dive.** For each `02-*` doc, run targeted explorations and confirm or update the claims. Anchor files to start from:

   | Doc                           | Anchor source paths                                                                                                                            |
   |:------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
   | `02-01-in-flight-batching.md` | `tensorrt_llm/_torch/pyexecutor/scheduler.py`, `_torch/pyexecutor/py_executor.py`, `cpp/tensorrt_llm/batch_manager/`                            |
   | `02-02-overlap-scheduler.md`  | `_torch/pyexecutor/py_executor.py` (overlap scheduler / early-exit), `_torch/pyexecutor/sampler.py`                                            |
   | `02-03-kv-cache-manager.md`   | `_torch/pyexecutor/resource_manager.py`, `cpp/tensorrt_llm/batch_manager/kvCacheManager*`, `_torch/pyexecutor/kv_cache_manager*`               |
   | `02-04-block-reuse.md`        | `_torch/pyexecutor/resource_manager.py` (radix tree), block-reuse design docs in `docs/design/`                                                |
   | `02-05-disaggregated-serving.md` | `_torch/pyexecutor/disagg_*`, `cpp/tensorrt_llm/batch_manager/cacheTransceiver*`, NIXL / UCX / Mooncake transceiver impls                  |
   | `02-06-speculative-decoding.md` | `_torch/speculative/`, `_torch/pyexecutor/spec_*`                                                                                            |
   | `02-07-parallelism-strategies.md` | `_torch/distributed/`, `mapping.py`, Wide-EP / EPLB / DWDP code, `_torch/modules/fused_moe/`                                              |
   | `02-08-other-features.md`     | CUDA graphs (`_torch/utils/cuda_graph*`), chunked prefill, guided decoding (`_torch/pyexecutor/guided_decoder*`), LoRA, multimodal, visual gen |

   For each: spot-check 1–3 claims by opening current code; correct any drifted defaults, paths, or capabilities; note new capabilities added since `PREV_DATE` and capabilities that were removed/renamed.
4. **Architecture pass (`01-`).** Verify backend list, default sampler, `PyExecutor` boundaries, and the `_torch/auto_deploy/` shim path. Update the architecture diagram if the boundaries shifted.
5. **User journey pass (`03-`).** Replay startup sequence by tracing `tensorrt_llm/llmapi/llm.py` → executor instantiation. Update any call paths that moved.
6. **Record findings** in a working notes scratchpad (in memory; do not commit). Mark each as: `Confirmed`, `Updated`, `New`, `Removed`, `Unknown`.

### Phase C — Competitive landscape research

Run web searches in parallel. For each project, fetch the **release notes / GitHub releases page** and the **roadmap / docs**, not just blog posts.

| Target                  | What to capture                                                                                                                            |
|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| **vLLM**                | Latest version, EngineCore changes, scheduler updates, new spec-dec algos, multi-vendor backends (TPU/ROCm), elastic features, KV connector |
| **SGLang**              | Latest version, RadixAttention/HiSparse changes, elastic EP, FA4 spec-dec, DSL features, EPD disagg, cache-aware scheduling                |
| **LMCache**             | Latest version, GDS support, NIXL backend, cross-instance fabric, supported runtimes                                                       |
| **NVIDIA Dynamo**       | Architecture, what it owns vs. TRT-LLM, KV router, disagg integration, deployment story                                                    |
| **TGI (HuggingFace)**   | Major architectural moves, multi-vendor, OpenAI-compat                                                                                      |
| **Mooncake**            | Latest KV-pool features and integrations                                                                                                    |
| **llama.cpp / MLX**     | Only if relevant to a positioning point in the doc                                                                                          |
| **DeepSpeed-MII / FT**  | Only if their roadmap intersects with TRT-LLM areas                                                                                         |

For every claim added to the docs, record the **URL + access date** in the changelog (you don't have to inline every URL into the doc itself, but the changelog must let a reader trace it).

### Phase D — Hardware & academic landscape

Run web searches in parallel. Time-box: ~10 minutes total.

1. **Hardware roadmap deltas since `PREV_DATE`.** Anything announced or shipped:
   - NVIDIA: Blackwell refresh SKUs, Rubin / Rubin Ultra, NVL576+ rack designs, NVLink generation bumps, NVSwitch, GB300 NVL72.
   - AMD: MI325X/MI355X/MI400 (Instinct), ROCm inference stack progress.
   - Google: TPU v6e (Trillium), v7, Pathways inference path.
   - Groq: LPU2/LPU3, GroqCloud capacity, deterministic-latency claims.
   - Cerebras: WSE-3 / WSE-4, inference cloud throughput claims.
   - AWS: Trainium 2/3, Inferentia 3, Neuron SDK inference features.
   - Intel: Gaudi 3 / Falcon Shores delta.
   - Etched: Sohu (transformer-only ASIC) availability/benchmarks.
   - d-Matrix: Corsair shipping status.
   - SambaNova: SN40L throughput claims.
   - Tenstorrent: Wormhole / Blackhole inference angle.
2. **Memory / interconnect.** CXL 3.x / 4.0 GPU access, GPU Direct Storage updates, UALink / Ultra Ethernet, NVL fabric updates, PCIe 6/7.
3. **Academic / OSS papers (≤6 months).** Sample queries:
   - LLM serving optimization, prefix caching, prefill-decode disaggregation
   - Speculative decoding (new tree algos, EAGLE-3 successors, lossless variants)
   - Attention sparsity / linear attention / state-space hybrids
   - Quantization (FP4, MXFP4, MX-FP, NVFP4) for inference
   - Multi-modal / video generation serving
   - Agentic workflow infra (KV forking, persistent sessions, speculative tool calls)
   - Inference-time compute / test-time scaling

### Phase E — Update the docs

Apply edits file-by-file. Style rules:

- **Inline change marker.** When you change or add a section, append a small italic note at the end of the section: `*[Updated YYYY-MM: <one-line reason>]*`. When you add an entirely new section, prepend `*[New YYYY-MM]*`. When you remove content, leave a one-line note in the changelog only — do not leave tombstones in the doc.
- **Versions and dates.** Update the `Last updated:` line in `README.md` and any version pins (TRT-LLM, vLLM, SGLang, LMCache, NVIDIA Dynamo) to the latest releases observed in Phase C/D.
- **Mermaid.** If a diagram is now wrong (e.g., a backend was renamed, sampler default flipped), update it. Keep diagrams syntactically valid (no nested code fences inside ``` ```mermaid ``` blocks).
- **Tables.** Keep column ordering stable across edits. Add new rows; do not silently drop rows (move retired rows to a `### Deprecated / Removed` subsection).
- **Tone.** Match the existing voice: dense, technical, direct. No marketing language. No emojis.
- **Citations in body.** Use `path/to/file.py` style for code refs. Use `[name](url)` for external refs but only when essential — otherwise hold the citation in the changelog.

Per-file checklist (each must be touched if anything material changed in its area; otherwise add a `*[Reviewed YYYY-MM: no changes]*` line at the bottom):

- `README.md` — TOC, last-updated, version pins.
- `01-high-level-architecture.md` — backend status, default sampler, request-flow diagram.
- `02-01` … `02-08` — feature accuracy from Phase B.
- `03-user-journey.md` — startup / failover / scaling story.
- `04-framework-comparison.md` — feature matrix, perf positioning, gap analysis.
- `05-01-feature-gaps.md` — new / closed / re-prioritized gaps.
- `05-02-bugs-and-issues.md` — fixed since last update vs. still-open vs. newly-discovered.
- `05-03-innovative-features.md` — promote ideas that competitors / academia have started executing on; demote/revise ideas that no longer differentiate.
- `06-strategic-prioritization.md` — re-rank quadrant chart and Tier 1–4 lists; explain shifts.

### Phase F — Diff review (CHANGELOG.md)

Append a new entry to `docs/overview/CHANGELOG.md` (create the file if missing). Format:

```
## YYYY-MM-DD — overview refresh

**Anchors.** Previous: `<PREV_TAG or PREV_SHA>` (`<PREV_DATE>`). Code anchor on main: `<PREV_CODE_SHA>` → today's `<HEAD_CODE_SHA>` (TRT-LLM `<old version>` → `<new version>`).

### What changed in TRT-LLM since last update
- <bulleted, with PR numbers / `path:line` cites>

### What changed in competitors
- vLLM <old> → <new>: …
- SGLang <old> → <new>: …
- LMCache <old> → <new>: …
- NVIDIA Dynamo: …
- (others as relevant)

### What changed in hardware / academic
- <bullet>, citing announcements / arXiv IDs

### Per-file diff highlights
- `01-high-level-architecture.md`
  - <change> — reason — cite
- `02-01-in-flight-batching.md`
  - …
- (one block per file you touched; one line per material change)

### Priority shifts (vs. last `06-strategic-prioritization.md`)
- <Item> moved from Tier <N> to Tier <M> because <reason>.

### Sources
- <URL or arXiv ID> (accessed YYYY-MM-DD) — what it backs.

### Blocked / Skipped (if any)
- <what you couldn't verify and why>
```

**Quality bar for the changelog entry.** A reader who has not seen this run should be able to answer:
- "What does TRT-LLM do today that it didn't last quarter?"
- "What did the competitors ship that I should react to?"
- "What changed in the priority list and why?"

If your entry can't answer all three, expand it before committing.

### Phase G — Commit, tag, push

1. Stage only `docs/overview/`:
   ```
   git add docs/overview/
   ```
2. Commit (HEREDOC for clean formatting):
   ```
   git commit -s -m "$(cat <<'EOF'
   [None][docs] overview refresh YYYY-MM-DD

   Refreshed 16 files under docs/overview/ to reflect TRT-LLM <new version>,
   vLLM <new>, SGLang <new>, LMCache <new>, plus current hardware/academic
   landscape. See docs/overview/CHANGELOG.md for the dated entry.
   EOF
   )"
   ```
   If pre-commit modifies files, re-stage and create a new commit (do NOT amend unless this run created the previous commit and it has not been pushed).
3. Tag (skip if `NO_TAG=1`):
   ```
   git tag -a docs-overview/YYYY-MM-DD -m "overview refresh YYYY-MM-DD"
   ```
4. Push (skip if `NO_PUSH=1`):
   ```
   git push origin docs-and-plans
   git push origin docs-overview/YYYY-MM-DD
   ```
5. **Tear down the worktree if you created one:**
   ```
   cd <original repo>
   git worktree remove ../TRTLLM-worktree-docs
   ```

---

## 3. Quality Gates (self-check before declaring done)

Run through this list explicitly. Do not declare success until every item is `Yes`.

- [ ] All 16 `docs/overview/*.md` files were either updated or carry a `*[Reviewed YYYY-MM: no changes]*` line.
- [ ] `README.md` "Last updated" + version pins reflect today.
- [ ] `CHANGELOG.md` has a new dated entry that answers the three questions in Phase F.
- [ ] Every TRT-LLM claim in changed sections has a `path:line` or PR/issue cite (in body or in the changelog Sources block).
- [ ] Every competitor / hardware / academic claim has a URL or arXiv ID in the Sources block.
- [ ] No Mermaid block has a syntax error (sanity-check by skimming for unmatched braces / unbalanced subgraphs).
- [ ] No file has `TODO` / `XXX` / `???` left in this run (search before commit).
- [ ] `git diff --stat HEAD~1` only touches `docs/overview/`.
- [ ] Single commit, signed-off-by line present, no AI co-authors.
- [ ] Tag created (unless `NO_TAG=1`).
- [ ] Pushed to `origin docs-and-plans` (unless `NO_PUSH=1`).
- [ ] Worktree cleaned up.

---

## 4. Failure-mode playbook

| Symptom                                                                                | Action                                                                                                                                                                  |
|:---------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `git pull --ff-only` rejected (branch diverged)                                         | Stop. Write a CHANGELOG block under "Blocked / Skipped" naming both heads. Do not force-push.                                                                            |
| `git worktree add` fails with "Operation not permitted"                                 | Re-run with sandbox `all`. If still fails, fall back to in-place on `docs-and-plans` after `git stash`-ing any uncommitted work.                                         |
| Web search rate-limited / blocked                                                      | Reduce parallelism, retry with backoff. If still blocked, mark affected sections as "carried forward unchanged" and note in CHANGELOG.                                   |
| Code path you cited no longer exists                                                   | Search for the renamed / moved symbol. If found, update the cite. If gone, mark the corresponding feature as "Removed in vX.Y" and update §02-XX accordingly.            |
| Pre-commit hook modifies files                                                          | `git add -u && git commit -s --amend --no-edit` is allowed only if this run created the prior commit AND it hasn't been pushed yet. Otherwise, make a follow-up commit. |
| Two `02-XX` files would contradict each other after edits                               | Treat the deeper-dive file as authoritative; reduce the other to a one-line summary that links to it.                                                                    |
| Found a major new feature that doesn't fit any existing 02-XX bucket                    | Add a new `02-09-<slug>.md`, link from `README.md`, and note the addition in the CHANGELOG.                                                                              |
| Conflicting evidence on competitor capability (blog says X, release notes say Y)       | Prefer release notes / source code over blogs. Note the discrepancy in the changelog Sources block.                                                                      |

---

## 5. Targeted-mode shortcuts

If invoked with `MODE=targeted FILES="..."`:

- Skip Phase B's repo-wide change scan; instead run only the per-file deep-dives for the listed files.
- Skip Phase C/D unless one of the listed files is `04-*`, `05-*`, or `06-*`.
- Still produce a CHANGELOG entry, but title it `## YYYY-MM-DD — overview refresh (targeted: <files>)`.
- Still tag (so the diff anchor stays consistent for next time).

If invoked with `MODE=competitive-only`:

- Skip Phase B entirely.
- Run Phase C + D fully.
- Update only `04-framework-comparison.md`, `05-01-feature-gaps.md`, `05-03-innovative-features.md`, `06-strategic-prioritization.md`, and `README.md` (version pins + last-updated).

---

## 6. Notes for future maintenance of this prompt

- This file lives on `docs-and-plans` only. If the broader `main` ever absorbs the overview docs, also move this file and adjust the `git pull` / push targets.
- If you add new docs under `docs/overview/`, update the per-file checklist in §2 Phase E and the anchor table in §2 Phase B.
- If `CLAUDE.md` / `AGENTS.md` rules change (e.g., new commit-title format, new code-owner gate), reflect them in §1 and §2 Phase G.
- If a new specialist skill is added that materially helps (e.g., a `competitive-intel` skill), reference it in Phase C/D.

---

*This prompt itself is intentionally code-free. It is the operating instruction; the agent is expected to use the available tools (codebase exploration, web search, shell, edit, git) to execute it.*
