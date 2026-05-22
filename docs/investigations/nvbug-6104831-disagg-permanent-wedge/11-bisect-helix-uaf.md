# 11 — Bisect plan: which PR #13713 commit introduces the helix CUDA illegal memory access?

**Status:** Open. Drives the next decision on whether PR #13713's always-on changes need scope reduction, transport gating, or a targeted fix.

**Reference build:** `LLM/main/L0_Test-x86_64-Multi-GPU/1118` (PR #13713 build #39481).
**Reference signature:** rank 0 wedged in `kv_cache_transceiver.check_gen_transfer_status`; CUDA illegal memory access raised from `cudaStreamSynchronize` inside `CacheReceiver::Impl::request()` for request `1298819601432577`. 38 per-iteration timeout warnings, 0 cancel attempts, 0 deadline triggers, 0 poison events.
**Transport:** direct UCX (`cache_transceiver_config.backend: UCX`), not NIXL.
**Reproducer:**
`tests/integration/defs/accuracy/test_disaggregated_serving.py::TestDeepSeekV3Lite::test_auto_dtype_with_helix[fifo_v1-cudagraph:with_padding-pp1dp2cp2]`

## What we know

- main builds #1114 and #1115 pass this test — the regression is on the PR-#13713 side.
- The PR currently has 21 first-parent commits, of which 12 are functional (touch C++ or runtime Python) and 9 are tests/lint/chore/merge. Bisection space = 12 commits.
- Helix uses **direct UCX**, but PR #13713's quiescence and deferred-cleanup work was designed around NIXL semantics. The hypothesis is that one of the always-on changes mishandles UCX's lack-of-NIXL-style status, freeing or unblocking memory while the UCX transfer is still touching the receive buffer.
- Confidence that PR #13713 caused this: medium-high (test deltas are clean across main, fails reliably on PR HEAD). Confidence on **which commit**: unknown — that's what this bisect resolves.

## Functional commits (chronological)

Test-only commits and merges are not bisect points; they cannot change runtime behavior. The 12 candidates:

| # | SHA | Title | Family |
|---|-----|-------|--------|
| C1 | `630fa3b4f2` | [fix] Disagg request cancellation fix | Layer 1 (cancel surface) |
| C2 | `1c321cc0b0` | Fail closed on unquiesced disagg KV transfer | Layer 5 (Python fail-closed) |
| C3 | `9a1af0bfd4` | [fix] Incorporate PR#13728's improvements | misc improvements |
| C4 | `78fa9d712a` | Narrow Python disagg transfer timeout handling | Python timeout |
| C5 | `89001e3062` | Defer context transfer cleanup after timeout cancel | Layer 2 (deferred cleanup) |
| C6 | `62102e8317` | [fix] Complete deferred disagg cleanup after transfer | Layer 2 follow-on |
| C7 | `a201be28a0` | [fix] Cleanup PR for review | review cleanup |
| C8 | `36aef04054` | [fix] Address review feedback | review fixes |
| C9 | `0a2d5268e6` | [feat] Gate disagg mid-flight cancel surface behind `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL` | flag gating |
| C10 | `6a9869b751` | [refactor] bundle disagg KV transfer timeout enforcement under inflight-cancel flag | timeout refactor |
| C11 | `3259c8fb3a` | [fix] keep NIXL agent alive while its TransferStatus exists | NIXL agent lifetime |
| C12 | `f79d8b7a61` | [fix] gate deferred-cleanup paths under `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL` | deferred-cleanup gating |

Reference points:
- `c20b192ee5` — parent of C1 on main; should reproduce main's pass.
- `5234311282` (current PR HEAD) — reproduces the helix failure in CI.

## Bisect order (binary search, ≤4 iterations)

```
Step 1: test C6 (62102e83 — "Complete deferred disagg cleanup after transfer")
  ↳ fails  → regression is in C1..C6 → Step 2A
  ↳ passes → regression is in C7..C12 → Step 2B

Step 2A: test C3 (9a1af0bf — "Incorporate PR#13728's improvements")
  ↳ fails  → C1..C3 → Step 3A: test C2 (1c321cc0)
  ↳ passes → C4..C6 → Step 3B: test C5 (89001e30)

Step 2B: test C10 (6a9869b7 — "bundle timeout enforcement")
  ↳ fails  → C7..C10 → Step 3C: test C8 (36aef040)
  ↳ passes → C11..C12 → Step 3D: test C11 (3259c8fb)

Step 3A/B/C/D narrows to ≤2 commits; one final test resolves it.
```

A priori hypothesis ranking (where regression most likely sits, given the UCX-not-NIXL transport and `cudaStreamSynchronize` failure mode):

1. **C5 / C6** (deferred cleanup) — most architecturally invasive change on the receive path; the deferred-cleanup state machine doesn't know about UCX's transfer lifetime.
2. **C1** (cancel surface) — original cancellation entry point; could change ordering of buffer reclamation vs UCX progress.
3. **C11** (NIXL agent lifetime) — labelled NIXL-only but touches shared transceiver lifetime code; verify it doesn't accidentally affect direct UCX path.
4. **C2 / C9 / C12** — Python-side / flag-gating; less likely to cause C++ UAF, but worth ruling out.

## Test command (on the 8×Blackwell GPU node)

The failing variant requires 8 Blackwell GPUs (`@skip_pre_blackwell`, `@skip_less_device(8)`).

```bash
export LLM_MODELS_ROOT=<path-to-models>
pytest -v -s \
  tests/integration/defs/accuracy/test_disaggregated_serving.py::TestDeepSeekV3Lite::test_auto_dtype_with_helix \
  -k "fifo_v1 and with_padding and pp1dp2cp2" \
  2>&1 | tee bisect-${SHA}.log
```

Pass/fail signal:
- **PASS**: pytest returns 0 and `bisect-${SHA}.log` contains `1 passed`.
- **FAIL (regression)**: pytest returns nonzero, log contains `CUDA error: an illegal memory access was encountered` or `cudaStreamSynchronize`. Stack trace should match build #39481: `_check_disagg_gen_cache_transfer_status` → `kv_cache_transceiver.check_gen_transfer_status` → `impl.check_gen_transfer_status` → `CacheReceiver::Impl::request()` → `cudaStreamSynchronize`.

If the test fails for an **unrelated reason** (build break, hang in a different place, MPI Allgather not blamed on rank 0) — mark that commit as **skip** in `git bisect`, not bad.

## `git bisect run` automation (optional)

```bash
# On the GPU node, after checkout of PR #13713:
git fetch upstream pull/13713/head:pr-13713-head
git checkout pr-13713-head

cat > /tmp/run_helix.sh <<'EOF'
#!/usr/bin/env bash
set -e
# Build (skip if you have ccache and incremental builds work across these commits)
./scripts/build_wheel.py --clean --cuda_architectures "100" || exit 125  # 125 = skip if build breaks
pip install --force-reinstall build/tensorrt_llm-*.whl

# Run the helix repro
pytest -v -s \
  tests/integration/defs/accuracy/test_disaggregated_serving.py::TestDeepSeekV3Lite::test_auto_dtype_with_helix \
  -k "fifo_v1 and with_padding and pp1dp2cp2" \
  --timeout=900
EOF
chmod +x /tmp/run_helix.sh

# Drive the bisect manually (not git bisect — the PR's merge commits confuse first-parent walking).
# Use the 12-commit list above and test in the order: C6 → (C3 or C10) → final.
```

> Skip `git bisect start` over the full PR range: the 3 merge commits (a3230d7e, 4c79ba44, 5234311282) make automatic first-parent walking unreliable. Drive the 12 functional commits manually using the Step 1 → 2 → 3 decision tree above.

## When you have the result

Report back with:
1. Which commit is the first **bad** one (regression introducer).
2. Stack trace from that commit (to confirm same signature as build #39481).
3. Whether earlier commits **also** showed any related anomaly (e.g., hang without illegal-memory-access).

That tells us:
- **If C5 or C6**: deferred cleanup needs UCX-aware quiescence (or restrict deferred-cleanup to NIXL via the existing flag — extend `f79d8b7a`'s gating semantically to "NIXL-only" rather than "flag-only").
- **If C1**: the cancellation entry point itself reclaims state in a way UCX doesn't tolerate. May need to gate Layer 1 to NIXL too.
- **If C11**: revert C11 or scope its lifetime change to NIXL handles.
- **If C2 / C9 / C12 / C7 / C8**: surprising — investigate deeper. (Low prior.)

If two adjacent commits both fail (e.g., C5 and C6 are needed together), tag both as "regression set."

## After the bisect: optional second-pass diagnostics

If reproduction is flaky or the stack trace is ambiguous, re-run the bad commit under:
- `CUDA_LAUNCH_BLOCKING=1` — pins the failing kernel to the launch site.
- `compute-sanitizer --tool memcheck` — reports the offending pointer + access size.
- Increase `kv_transfer_timeout_ms` to 600000 — rules out the timeout/cancel surface as a trigger (helps confirm whether failure is in the steady-state UCX progress path vs the cancel/cleanup path).
