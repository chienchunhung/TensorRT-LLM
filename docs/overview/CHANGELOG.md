# Overview Refresh Changelog

This file is appended to by `UPDATE-PROMPT.md` on each periodic refresh of the
`docs/overview/` learning guide. Newest entries on top.

Each entry follows the schema documented in `UPDATE-PROMPT.md` §2 Phase F.
Tags of the form `docs-overview/YYYY-MM-DD` mark the commit at which each
entry was created, so a reader can `git diff docs-overview/<old>..docs-overview/<new> -- docs/overview/`
to see the literal text changes between two refreshes.

---

## Baseline — 2026-04 (pre-changelog)

The first dated `Last updated:` value in `docs/overview/README.md` was
**April 2026**, reflecting TensorRT-LLM v1.3.0, vLLM v0.19.0, SGLang v0.5.10,
and LMCache v0.4.2. No `docs-overview/*` tag exists for this baseline; the
first periodic refresh that runs `UPDATE-PROMPT.md` is responsible for
creating the first tag and the first dated changelog entry above this line.

If you are running the refresh prompt for the first time:
1. Treat the baseline as the previous-update anchor (`PREV_DATE = 2026-04-30`,
   `PREV_SHA = HEAD at the time of the first run`).
2. Snapshot the current `docs/overview/*.md` into
   `docs/overview/.snapshots/2026-04-30/` so future diffs have a real anchor
   even if no tag is created.
3. Append your first dated entry above this baseline note.
