# 15 — PR #13713 Decomposition Plan

**Status:** Draft (2026-05-28). Proposal for breaking PR #13713 into smaller,
independently-shippable PRs to enable incremental progress and clean bug
attribution.

**Motivation:** PR #13713 has accumulated 8+ failed CI cycles over ~4 weeks
with overlapping bug surfaces that prevent attribution of which always-on
change caused which regression. Each retry of the combined PR re-tests the
entire delta, making the debug loop slow and the failure-to-fix
correspondence ambiguous. Decomposing into smaller PRs gives each landing a
tight blast radius and clean revertable scope.

**Scope:**

- Define the components currently in PR #13713 with code names and short
  descriptions (§1)
- Lay out the proposed tier sequence with risk and code-change estimates (§2)
- Present the submission dependency graph (§3)
- Document the decision checkpoint after Tier 3 (§4)

---

## 1. Component inventory

PR #13713 contains 18 functionally-distinct components (after excluding the
reverted `A11`). They fall into three families:

- **Ax — always-on**: behavior changes that ship unconditionally
- **C1 — standalone**: the single targeted polling-cap commit landed
  after the upstream merge
- **Gx — gated**: behind `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1`
  (default `"0"` → all behave as if absent)

Risk legend: **L** = low, **M** = medium, **H** = high.

### 1.1 Always-on components (Ax)

#### A1 · `shared-llmreq` · LlmRequest async lifetime via shared_ptr

- **What:** Replace raw `LlmRequest*` with `std::shared_ptr<LlmRequest>` in
  `mSenderFutures` / `mRequesterFutures` and the 3 async API methods
  (`respondAndSendAsync`, `requestAndReceiveAsync`, `cancelRequest`).
  **Bundles the eval-order discipline at every call site**: use `.get()` for
  raw-pointer APIs, do all `->` derefs before any `std::move`, move the
  shared_ptr last when emplacing into the futures map.
- **Closes:** LlmRequest-object UAF when Python destroys the request while
  C++ async workers still reference it. (Different UAF from the original
  NVBug 6104831 wedge bug, which is on KV blocks.)
- **Files:** `cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h`
  (signatures + futures map types) + every call site in `cacheTransceiver.cpp`,
  `dataTransceiver.cpp`, `trtGptModelInflightBatching.cpp`,
  `nanobind/batch_manager/cacheTransceiver.cpp`, plus test file updates.
- **Code size:** ~150–250 changed lines across ~6 files
- **Risk:** **H** — widest blast radius. Signature changes flagged by API
  stability tests. Eval-order discipline must be reviewed at every call
  site. **The eval-order fix is inseparable from A1** — it only exists
  because A1 introduces movable shared_ptr.

#### A2 · `buf-holder-raii` · BufferIndexHolder RAII

- **What:** Add `BufferIndexHolder` class with destructor-fallback
  `release()`. Prevents send/recv buffer-pool slot leaks when exception
  unwinds between `assignBufferIndexForRecv` and the explicit release in
  `requestSync`. `poison()` method is exposed but only called from G3.
- **Closes:** Real buffer-pool slot-leak bug — pre-PR, exception paths
  leaked one slot each, eventually wedging the size-1 default pool.
- **Files:** `cpp/tensorrt_llm/batch_manager/baseTransBuffer.h` (new class),
  `baseTransBuffer.cpp` (impl), narrow usage updates in `cacheFormatter.cpp`,
  `mlaCacheFormatter.cpp`, `dataTransceiver.cpp`
- **Code size:** ~150–200 lines
- **Risk:** **L** — additive, no behavior change on happy path,
  destructor-fallback only fires on exception

#### A3 · `py-recv-dedup` · Python recv-side idempotency

- **What:** Add per-rank dedup sets `_disagg_gen_init_prepared_ids` and
  `_disagg_gen_kv_recv_started_ids` in PyExecutor. Skip duplicate
  `prepare_resources` / `request_and_receive_async` calls. Prune on
  `_terminate_request`.
- **Closes:** Double-invocation of recv-side setup when scheduler retries
  the same request.
- **Files:** `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- **Code size:** ~50 lines
- **Risk:** **L** — pure Python, narrow, idempotency-only

#### A4 · `cpp-timeout-dedup` · C++ per-rank timeout warning dedup

- **What:** Add `mTimedOutSenderIds` and `mTimedOutRequesterIds` sets in
  `CacheTransceiver`. Ensures a single hung request produces at most one
  timeout warning regardless of which loop site observes the deadline.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` and `.h`
- **Code size:** ~30 lines
- **Risk:** **L** — observability-only; insertion always-on; the action
  taken is gated (G2)

#### A5 · `state-reorder` · State-transition reorder in `requestAndReceiveAsync`

- **What:** Move `setState(IN_PROGRESS)` from before `receiveAsync` to
  after. Tightens the "IN_PROGRESS ↔ entry in `mRequesterFutures`"
  invariant: if `receiveAsync` throws, the request stays out of both
  atomically. Pre-PR had the torn-state-on-throw bug.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` (single
  function body)
- **Code size:** ~5–15 lines (pure reorder)
- **Risk:** **M** — narrow but behavior-affecting. **Currently the
  strongest suspect for the `numNewOutputTokens > numGeneratedTokens`
  decoder assertion** seen in `asymmetric_executor[6proc-nixl_kvcache]`
  build #40124.

#### A6 · `eager-xfer-start` · Unconditional `setKvCacheTransferStart` in caller

- **What:** Add one line at top of `requestAndReceiveAsync`:
  `llmRequest->setKvCacheTransferStart(now())`. Required so the new
  deadline check (A7's observation) sees a valid start time rather than
  epoch for entries whose worker thread hasn't run yet.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` (single
  function body)
- **Code size:** ~5 lines
- **Risk:** **L** — single field write; only consumer is A7's deadline
  check
- **Dependency:** Must land with or before A7

#### A7 · `obs-timeout-warn` · Observe-only timeout warnings + deadline observation

- **What:** Add deadline-observation block in `checkContextTransferStatus`
  and `checkGenTransferStatus`: compute elapsed, compare to
  `kvTransferTimeoutMs`, dedup via A4, log warning if exceeded. Python
  sibling at `py_executor.py:flag_if_kv_transfer_timed_out`. The warning
  includes a tail explaining whether action will be taken (depends on env
  var); the action itself (G2) is omitted here.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp`
  (deadline check blocks at 4 sites),
  `tensorrt_llm/_torch/pyexecutor/py_executor.py` (Python timeout flagger)
- **Code size:** ~80–120 lines
- **Risk:** **L** — observation-only; no behavior change. Surgical split
  needed: ships the observation block but omits the inner
  `if (inflightCancelEnabled) { /* G2 action */ }`.
- **Dependency:** Requires A4 (dedup) and A6 (valid start times)

#### A8 · `nixl-agent-life` · NIXL agent lifetime fix

- **What:** Keep NIXL `BaseTransferAgent` alive while any `TransferStatus`
  references it. Closes UAF when agent is destroyed before in-flight
  TransferStatus objects.
- **Files:**
  `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp`
  and `.h`
- **Commit ref:** `3259c8fb3a`
- **Code size:** ~20 lines
- **Risk:** **L** — narrow, defensive, no behavior change

#### A9 · `gen-sym-collective` · Gen-side rank-symmetric collective entry

- **What:** Remove `if need_check:` and `if not recv_reqs: return` early
  exits from `_check_disagg_gen_transfer_status` and
  `_recv_disagg_gen_cache`. Every gen-side rank enters
  `_check_disagg_gen_cache_transfer_status → gatherRequestIds` Allgather
  on every iteration.
- **Motivation:** Helix hang in CI build #39529 (rank-asymmetric ABBA on
  the gen-side collective).
- **Files:** `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- **Commit ref:** `bdfdf8be02`
- **Code size:** ~50 lines
- **Risk:** **M** — changes coordination cadence; one of the suspected
  contributors to Class B (`_pp_retry_until_can_schedule` RuntimeError)
  failures

#### A10 · `ctx-sym-collective` · Ctx-side rank-symmetric collective entry (Python)

- **What:** Remove `if num_fitting_reqs == 0 ...:` gate around
  `_check_disagg_ctx_cache_transfer_status` in `_executor_loop_pp` and
  `_executor_loop`. `at_least_num` defaults to 0, escalates to 1 on
  nothing-fits-locally.
- **Motivation:** Build #39569 hang on
  `TestQwen3_8B::test_auto_dtype_with_helix[pp1tp1cp4, pp1tp2cp2]`.
- **Files:** `tensorrt_llm/_torch/pyexecutor/py_executor.py`
- **Commit ref:** `53a0692aa4`
- **Code size:** ~60 lines
- **Risk:** **M** — same class as A9; coordination cadence shift

#### A11 · `pp-retry-sym` · `_pp_retry` + C++ ctx-drain rank-symmetric (REVERTED)

- **Status:** **NOT IN PR HEAD.** Reverted by `e8f194f728` after it
  regressed Qwen3 helix and `ctxpp2_genpp2` in build #39634 without fixing
  `asymmetric_executor`. Listed here only for completeness.
- **Commit refs:** `dbaf7a1106` (introduced), `e8f194f728` (revert)

### 1.2 Standalone

#### C1 · `poll-slice-cap` · 50ms `wait_for` slice cap

- **What:** Cap `future.wait_for` slice at 50ms in
  `checkContextTransferStatus` and `checkGenTransferStatus`. Forces app
  thread to cycle back periodically so MPI/UCX progress functions inside
  the worker get called rather than blocking indefinitely.
- **Closes:** The deterministic 4proc-mpi wedge. Proven fix — 4proc-mpi
  started passing in build #40124 after this landed.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` (2 sites)
- **Commit ref:** `9a1a0329fb`
- **Code size:** 28 lines (1 file)
- **Risk:** **L** — proven beneficial, narrow

### 1.3 Gated components (Gx)

All behind `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1`. Default `"0"` → all
behave as if absent.

#### G1 · `py-cancel-req` · Python `cancel_request` Layer 1

- **What:** Python `cancel_request` forwards to C++ `cancel_request`. When
  env off: returns False; natural-completion path takes over (pre-PR
  behavior).
- **Files:** `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`
- **Code size:** ~10 lines
- **Risk:** **L** (gated)

#### G2 · `cpp-deadline-evict` · C++ deadline-driven eviction

- **What:** On deadline exceeded (observation done by A7):
  `mCacheSender->cancelRequest(*request) → setState(DISAGG_TRANS_ERROR) →
  erase from futures map → insert into errorRequestIds`. The inner
  `if (inflightCancelEnabled) { ... }` block of A7's observation site.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` (4 sites,
  the inner action blocks)
- **Code size:** ~80 lines (4 nearly-identical sites)
- **Risk:** **L** (gated); the action is the riskier counterpart of A7's
  observation

#### G3 · `send-holder-poison` · `sendHolder.poison()` on cancel-class exception

- **What:** Catch block in `cacheFormatter`/`mlaCacheFormatter` calls
  `sendHolder.poison()` (provided by A2) to quarantine the buffer-pool
  slot. When off: RAII returns slot to pool normally.
- **Files:** `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`,
  `mlaCacheFormatter.cpp`
- **Code size:** ~30 lines
- **Risk:** **L** (gated)

#### G4 · `defer-terminate` · `_can_terminate_request_now` deferred-termination

- **What:** Returns False for in-progress disagg states; installs
  `mark_termination_requested` handoff via `async_transfer_manager`.
  Termination retries after C++ surfaces transfer completion. **This is
  the only mechanism that prevents the original NVBug 6104831 KV-block
  UAF.** When off (default): returns True immediately → same KV-block UAF
  risk as pre-PR.
- **Files:** `tensorrt_llm/_torch/pyexecutor/py_executor.py:4532-4546`
- **Code size:** ~30 lines (function body) plus the handoff plumbing
- **Risk:** **L** (gated); the load-bearing fix when enabled

#### G5 · `fail-closed` · Layer 5 fail-closed via `has_poisoned_transfer_buffer`

- **What:** Poisoned buffer surfaces as unrecoverable transfer error;
  `_handle_errors` called with `charge_budget=True` → executor shutdown so
  orchestrator can restart pod. When off: always reports False; no
  shutdown.
- **Files:** `kv_cache_transceiver.py:260-267`, `py_executor.py:4127-4139`
- **Code size:** ~30 lines
- **Risk:** **L** (gated)

#### G6 · `handle-err-defer` · `_handle_errors` deferred-cleanup for unquiesced transfers

- **What:** Failed requests still in unquiesced transfer state are
  deferred from immediate termination. DISAGG_TRANS_ERROR transition left
  to C++ deadline or Python fallback floor.
- **Files:** `py_executor.py:4470-4495`
- **Code size:** ~30 lines
- **Risk:** **L** (gated)

#### G7 · `pp-term-safe` · Disagg-PP termination handler choice

- **What:** PP termination uses `_do_terminate_request_if_safe` instead of
  direct `_do_terminate_request`.
- **Files:** `py_executor.py:681-686`
- **Code size:** ~6 lines
- **Risk:** **L** (gated)

#### G8 · `promise-idempotent` · Promise idempotency in cancel path

- **What:** `catch (std::future_error const&)` around
  `mPromise->set_exception` in cancel paths. Swallows the error if the
  worker already fulfilled the promise. Effectively gated because the call
  path enters only via `cancelRequest`.
- **Files:**
  `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp:1240-1254` and a few
  sibling sites
- **Code size:** ~20 lines (try/catch blocks)
- **Risk:** **L** (gated)

### 1.4 Component summary table

| Code | Short name | Always-on? | Risk | Approx. lines | Key dependency |
|---|---|---|---|---|---|
| A1 | `shared-llmreq` | yes | H | ~150–250 | (includes eval-order discipline) |
| A2 | `buf-holder-raii` | yes | L | ~150–200 | — |
| A3 | `py-recv-dedup` | yes | L | ~50 | — |
| A4 | `cpp-timeout-dedup` | yes | L | ~30 | needed by A7 |
| A5 | `state-reorder` | yes | M | ~10 | — |
| A6 | `eager-xfer-start` | yes | L | ~5 | needed by A7 |
| A7 | `obs-timeout-warn` | yes | L | ~80–120 | needs A4 + A6 |
| A8 | `nixl-agent-life` | yes | L | ~20 | — |
| A9 | `gen-sym-collective` | yes | M | ~50 | — |
| A10 | `ctx-sym-collective` | yes | M | ~60 | — |
| A11 | `pp-retry-sym` | **REVERTED** | — | — | not in PR HEAD |
| C1 | `poll-slice-cap` | yes | L | ~30 | — |
| G1 | `py-cancel-req` | gated | L | ~10 | needs A1 |
| G2 | `cpp-deadline-evict` | gated | L | ~80 | needs A7 (obs) |
| G3 | `send-holder-poison` | gated | L | ~30 | needs A2 |
| G4 | `defer-terminate` | gated | L | ~30 | needs A1 |
| G5 | `fail-closed` | gated | L | ~30 | needs G3 |
| G6 | `handle-err-defer` | gated | L | ~30 | — |
| G7 | `pp-term-safe` | gated | L | ~6 | — |
| G8 | `promise-idempotent` | gated | L | ~20 | needs G1 |

---

## 2. Tier sequence

Six tiers ordered by ship-readiness. Earlier tiers depend on nothing
post-merge; later tiers depend on earlier ones.

### Tier 1 — Standalone bug fixes

| Code | Short name | Lines | Risk |
|---|---|---|---|
| C1 | `poll-slice-cap` | ~30 | L |
| A2 | `buf-holder-raii` | ~150–200 | L |
| A8 | `nixl-agent-life` | ~20 | L |
| | **Total** | **~200–250** | **L** |

**PR title:** `[https://nvbugs/6104831][fix] disagg KV transport progress
+ buffer/agent lifetime fixes`

**Why first:** Three narrowly-scoped bug fixes, each independently
defensible. C1 is proven (fixes 4proc-mpi wedge). A2 closes a real
buffer-slot leak. A8 closes a NIXL agent UAF. None touch LlmRequest
signatures, none change call-site patterns, none affect coordination
cadence. Lowest review friction, lowest CI risk.

### Tier 2 — Observability + Python idempotency

| Code | Short name | Lines | Risk |
|---|---|---|---|
| A3 | `py-recv-dedup` | ~50 | L |
| A4 | `cpp-timeout-dedup` | ~30 | L |
| A6 | `eager-xfer-start` | ~5 | L |
| A7 | `obs-timeout-warn` | ~80–120 | L |
| | **Total** | **~165–205** | **L** |

**PR title:** `[https://nvbugs/6104831][feat] disagg KV transfer
observability + recv-side idempotency`

**Surgical note for A7:** The deadline-observation block in
`cacheTransceiver.cpp` includes both the warning (always-on) and the
eviction action (G2, gated). Tier 2 ships only the observation+warning.
The inner `if (inflightCancelEnabled) { mCacheSender->cancelRequest(...);
... }` is omitted here and added in Tier 6.

**Why second:** Pure observability + idempotency. No behavior change to
disagg semantics. A6 must ship with A7 for the deadline check to compute
valid elapsed times. A4 must ship with A7 to dedup the warnings.

### Tier 3 — State-transition invariant

| Code | Short name | Lines | Risk |
|---|---|---|---|
| A5 | `state-reorder` | ~10 | M |
| | **Total** | **~10** | **M** |

**PR title:** `[https://nvbugs/6104831][fix] reorder requestAndReceiveAsync
state transition to fix torn-state-on-throw`

**Why third:** Tiny diff, isolated for clean attribution. Currently the
strongest suspect for the `numNewOutputTokens > numGeneratedTokens`
decoder assertion seen in `asymmetric_executor[6proc-nixl_kvcache]`
build #40124. Landing it standalone means:

- If the assertion fires post-merge, you can revert just this PR
- If CI stays clean, you've confirmed A5 is benign and can move on

**Keep explicitly revertable.** No dependencies above it.

### CHECKPOINT — after Tier 3

Let Tier 1+2+3 bake in CI for several runs. Decide:

- **CI clean →** continue to Tier 4 (if shared_ptr is desired) or stop
  here (Tiers 1-3 cover the meaningful bug fixes without the shared_ptr
  blast)
- **Decoder assertion fires →** revert Tier 3, investigate further, ship
  without A5
- **Helix / Class B `_pp_retry_until_can_schedule` RuntimeError fires →**
  add Tier 5 as a probe (see below)

### Tier 4 — shared_ptr foundation (optional)

| Code | Short name | Lines | Risk |
|---|---|---|---|
| A1 | `shared-llmreq` (+ eval-order discipline at all call sites) | ~150–250 | H |
| | **Total** | **~150–250** | **H** |

**PR title:** `[https://nvbugs/6104831][fix] use shared_ptr<LlmRequest> for
disagg KV transfer async lifetime`

**Why fourth and optional:** Big blast radius. Closes the LlmRequest-object
UAF (a defensive fix for a pre-existing UAF that hasn't been observed
firing). Required by Tier 6's cancel surface, so must land before Tier 6
if cancel is going to ship. Without Tier 6, A1 is purely defensive.

**Eval-order discipline cannot be split out** — every call site must use
`.get()` for raw-pointer APIs and dereference before `std::move`,
otherwise A1 introduces null-deref bugs. The eval-order pattern only
matters when `llmRequest` is a `shared_ptr` that can be moved, so it has
no meaning before A1.

### Tier 5 — Horizontal-consistency fix (conditional)

Pick one of two strategies:

#### Strategy A — minimal: rank-symmetric gate removal

| Code | Short name | Lines | Risk |
|---|---|---|---|
| A9 | `gen-sym-collective` | ~50 | M |
| A10 | `ctx-sym-collective` | ~60 | M |
| | **Total** | **~110** | **M** |

#### Strategy B — principled: V2-style consensus port

- Port `_consensus_outcome` mechanism from V2 transceiver
  (`tensorrt_llm/_torch/disaggregation/transceiver.py`) into V1 path
- Adds union/intersection consensus across cancelled/failed/completed
  outcomes via extra allgathers
- Lines: ~150–250 in `cacheTransceiver.cpp` + Python wrapper
- Risk: **M-H** (more invasive but more principled)

**Why fifth and conditional:** Only land Tier 5 if post-Tier 3
(or post-Tier 4) CI surfaces the `_pp_retry_until_can_schedule`
RuntimeError or similar Class B failures (helix tests, Qwen3-Next).
Otherwise skip. The decomposition's smaller per-tier delta may reduce the
timing-coordination shift enough that A9/A10 aren't needed at all.

### Tier 6 — Cancel surface (gated, defer last)

| Code | Short name | Lines | Risk |
|---|---|---|---|
| G1 | `py-cancel-req` | ~10 | L (gated) |
| G2 | `cpp-deadline-evict` | ~80 | L (gated) |
| G3 | `send-holder-poison` | ~30 | L (gated) |
| G4 | `defer-terminate` | ~30 | L (gated) |
| G5 | `fail-closed` | ~30 | L (gated) |
| G6 | `handle-err-defer` | ~30 | L (gated) |
| G7 | `pp-term-safe` | ~6 | L (gated) |
| G8 | `promise-idempotent` | ~20 | L (gated) |
| | Env var introduction + tests | ~150 | L |
| | **Total** | **~400** | **L (gated)** |

**PR title:** `[https://nvbugs/6104831][feat] add opt-in disagg mid-flight
cancellation surface (TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL)`

**Why last:** Default off → no behavior change in default CI. Easy review
because functionally additive when flag is off. Requires Tier 4 (A1
shared_ptr) for safe LlmRequest access in cancel paths. Requires Tier 1
(A2 BufferIndexHolder) for G3's `poison()`. Requires Tier 2 (A7
observation block) for G2's action block.

**The original NVBug 6104831 KV-block UAF fix is G4 specifically**, which
only operates when the env var is on.

---

## 3. Dependency graph

```
                        ┌──────────────────┐
                        │  Pre-PR baseline │
                        └────────┬─────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
        ┌────────────────────┐      ┌────────────────────┐
        │  Tier 1            │      │  Tier 2            │
        │  C1, A2, A8        │      │  A3, A4, A6, A7    │
        │  (~200-250 LOC, L) │      │  (~165-205 LOC, L) │
        └─────────┬──────────┘      └────────┬───────────┘
                  │                          │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌────────────────────┐
                  │  Tier 3            │
                  │  A5                │
                  │  (~10 LOC, M)      │
                  └────────┬───────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │  CHECKPOINT        │
                  │  CI bake; decide   │
                  └────────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌──────────┐  ┌──────────────┐  ┌──────────────┐
      │ Stop     │  │ Tier 5       │  │ Tier 4       │
      │ here     │  │ (conditional)│  │ A1           │
      │          │  │ A9+A10  OR   │  │ (~150-250    │
      │          │  │ consensus    │  │  LOC, H)     │
      │          │  │ (~110+, M)   │  │              │
      └──────────┘  └──────┬───────┘  └──────┬───────┘
                           │                 │
                           │                 ▼
                           │      ┌────────────────────┐
                           │      │  Tier 6            │
                           │      │  G1-G8 (gated)     │
                           │      │  (~400 LOC, L)     │
                           │      └────────────────────┘
                           ▼
                   (lands independently
                    if/when needed)
```

### 3.1 Dependency rules in text

**Strict hard dependencies (Y must precede X):**

- `Tier 6 → Tier 4`: G1, G4 require A1 (safe LlmRequest access in cancel
  path)
- `Tier 6 → Tier 2`: G2 is the action block inside A7's observation site
- `Tier 6 → Tier 1`: G3 calls `poison()` provided by A2

**Soft ordering (recommended but not strict):**

- Tier 3 after Tier 2 (so A5's behavior change lands on top of
  observability — easier to diagnose if A5 surfaces a bug)
- Tier 4 after Tier 3 (so shared_ptr type change applies cleanly to the
  new reordered code)
- Tier 5 only if needed (conditional on post-Tier 3/4 CI behavior)

**Parallel possibilities:**

- Tier 1 and Tier 2 can be authored / reviewed in parallel (touch
  disjoint files mostly)
- Tier 5 (if chosen) and Tier 6 can be authored in parallel after Tier 4
  lands

### 3.2 Submission order

```text
1. Tier 1 ──────►  ship → CI bake → confirm green
2. Tier 2 ──────►  ship → CI bake → confirm green
3. Tier 3 ──────►  ship → CI bake → decision point
                                    │
                                    ├─→ stop (Option A)
                                    │
                                    ├─→ Tier 4 (Option B)
                                    │       │
                                    │       └─→ Tier 6 (full cancel surface)
                                    │
                                    └─→ Tier 5 only if Class B failures appear
                                            (independent of Tier 4 ordering)
```

### 3.3 Scope estimates

- **Conservative path** (ship Tier 1 → Tier 2 → Tier 3 → stop):
  **~380–470 LOC total**, three small PRs, mostly L risk with one M.
  Covers the meaningful defensive fixes without the shared_ptr blast or
  the cancel surface. Original wedge bug (KV-block UAF) remains unfixed
  in default config — same as pre-PR.

- **Full path** (ship through Tier 6): **~1200–1550 LOC total** across 5–6
  PRs. Full PR #13713 coverage, decomposed for clean attribution.

---

## 4. Checkpoint criteria

After Tier 3 lands and CI has had several runs to settle, evaluate:

### 4.1 Decoder assertion test

`numNewOutputTokens > numGeneratedTokens` assertion in `decoderSync`.

- **Fires:** Tier 3 (A5) is implicated. Revert just Tier 3. Investigate
  further before re-shipping.
- **Does not fire:** A5 is exonerated. Continue.

### 4.2 Class B RuntimeError test

`_pp_retry_until_can_schedule` RuntimeError: "No context cache
transmission is in progress, but current rank cannot run first PP's
schedule result due to limited KV cache resources. This is not expected."

- **Fires:** Add Tier 5 as a follow-up PR. Pick Strategy A or B based on
  team appetite for invasiveness.
- **Does not fire:** Tier 5 is not needed. Original A9/A10 were addressing
  a problem that the smaller per-tier landing pattern may have avoided.

### 4.3 V2 transceiver setup race test

`test_kv_cache_transceiver_single_process[PYTHON-mha-ctx_fp16_gen_fp16]`
TxSession timeout at 1000ms.

- **Fires:** Indicates V2 transceiver init slowdown from one of the
  upstream-merged commits (most likely PR #14060 `ab08ffd03c` hybrid
  mamba). Out of scope for this decomposition — file as a separate
  upstream-bisect.
- **Does not fire:** No action needed.

### 4.4 Class A wedge tests

`test_asymmetric_executor[6proc-ucx_kvcache]`, helix tests, request hangs
in `disaggregated_gpt_oss_120b_harmony` / `deepseek_v3_lite_fp8_nixl`.

These are transport-layer wedges (per doc 13 forensics) — the
`AgentConnectionManager::waitForNotification` busy-spin where the
notification never arrives from the peer. None of Tiers 1–4 address this
directly. Tier 6 can't help either (cancel surface only kicks in when env
var is on).

- **Fires:** Persistent wedge problem. Requires either:
  - Adding a deadline to `waitForNotification` (Option A "wider cap")
  - Diagnostic-logging probe to identify whether ctx side fails to send /
    fails to notify / NIXL drops the notification (see doc 13 §3)
  - Both — diagnostic first, then targeted fix
- **Does not fire:** Class A is dodged by the smaller per-tier timing
  shift. Track for post-merge follow-up.

### 4.5 Recommended decision matrix

| Tier 3 CI result | Action |
|---|---|
| All targeted tests pass | Continue to Tier 4 (if cancel surface is the goal) or stop here |
| Decoder assertion only | Revert Tier 3, ship without A5, stop or continue to Tier 4 minus A5 |
| Class B RuntimeError only | Add Tier 5, then continue or stop |
| Class A wedge only | Investigate wedge separately, may still continue to Tier 4 |
| Multiple failure classes | Pause; investigate one at a time; do not advance tiers until each is understood |

---

## 5. Open questions

1. **Do we actually need Tier 4 (A1 shared_ptr)?** If we stop at Tier 3,
   we ship the meaningful defensive improvements without the shared_ptr
   blast radius. Future cancel-surface work will need A1, but if the team
   defers the cancel surface indefinitely, A1's marginal benefit may not
   justify its review burden.

2. **Strategy A vs B for Tier 5?** Strategy A is faster but treats
   symptoms. Strategy B is more principled (matches V2 transceiver's
   consensus design) but more invasive. Decide based on whether Class B
   failures actually fire and how much engineering bandwidth is available.

3. **Should the V2-vs-V1 disagg path consolidation be in scope?** V2 +
   Python transceiver already has the consensus mechanism we're
   considering porting in Tier 5 Strategy B. Long-term, deprecating V1 +
   C++ transceiver may be the cleanest path. Out of scope for this PR
   decomposition but worth flagging for follow-up planning.

4. **How to handle PR #13713's existing reviewers?** Decomposing into
   smaller PRs requires re-collecting review approvals per PR. Some
   reviewers prefer one big PR with one merge event. Socialize the plan
   with reviewers before starting Tier 1 to avoid friction.

---

## 6. References

- [`10-ablation-no-midflight-cancel.md`](../10-ablation-no-midflight-cancel.md)
  — analysis of cancel-surface default-OFF rationale and V2 + Python
  transceiver consensus contrast
- [`11-bisect-helix-uaf.md`](../11-bisect-helix-uaf.md) — bisect plan for
  helix CUDA illegal memory access regression
- [`12-horizontal-consistency-and-layer3-gating.md`](../12-horizontal-consistency-and-layer3-gating.md)
  — original theory and Path A/B/C analysis
- [`13-cpp-gtest-transport-hang-finding.md`](../13-cpp-gtest-transport-hang-finding.md)
  — forensics finding that the cpp gtest `mpi_kvcache` wedge is a
  transport-layer hang in `waitForNotification`, not horizontal-consistency
  ABBA. Drives Tier 6's interaction with the C1 polling cap.
- [`14-cross-rank-consistency-enforcement.md`](../14-cross-rank-consistency-enforcement.md)
  — Tier 5 Strategy B reference: V2 `_consensus_outcome` port to V1 C++
  transceiver as a principled fix for Class B failures.
- PR [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) —
  combined PR being decomposed
- PR [#14726](https://github.com/NVIDIA/TensorRT-LLM/pull/14726) —
  diagnostic logging probe branch (off PR #13713 head)
