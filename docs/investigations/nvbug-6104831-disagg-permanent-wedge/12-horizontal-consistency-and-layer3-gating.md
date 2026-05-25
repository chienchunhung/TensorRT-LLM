# 12 — Horizontal consistency, Layer 3 lifetime, and the path to merging PR #13713

**Status:** Active investigation, post-rc13.
**Trigger:** Multiple rounds of CI failures (builds #39529, #39569, #39604, #39634, #39661) on PR #13713 that the rank-asymmetric-collective patches (commits `bdfdf8be02`, `53a0692aa4`, `dbaf7a1106`) did not converge.
**Public PR:** https://github.com/NVIDIA/TensorRT-LLM/pull/13713

---

## 1. Why this doc exists

The investigation captured in docs 00–11 traced the original wedge to a UAF on the gen-side recv worker and proposed PR #13713 as the fix. After PR #13713 entered CI, a new class of failures appeared in disagg tests — primarily helix-CP and asymmetric-parallelism configurations. Three rounds of "fix one rank-asymmetric collective, watch another fail" patches followed, with the failure profile *changing* between identical code revisions.

This doc captures:

- The diagnostic finding that drove the rethink: **the failures are flaky, not deterministic regressions, and identical code can produce different failure profiles across CI runs**.
- The **vertical-vs-horizontal consistency theory** that explains the failure pattern.
- The **why** behind the pattern: which tests fail, which don't, and what they share.
- The **fix options** considered, with concrete scope estimates.

It is meant as a hand-off artifact for whoever picks up the lifecycle/cancel-flag work after PR #13713 lands.

## 2. Decisive observation: identical code, different failure profiles

`dbaf7a1106` was reverted via `e8f194f728`. `git diff 53a0692aa4 e8f194f728 -- '*.py' '*.cpp' '*.h' '*.cu'` returns **empty** — the trees are byte-for-byte identical.

Yet:

| Test | Build #39604 (HEAD = `53a0692aa4`) | Build #39661 (HEAD = `e8f194f728`, same tree) |
|---|---|---|
| Qwen3 helix `pp1tp2cp2` | passed | **FAILED** |
| `ctxpp2_genpp2` (TinyLlama) | (not in failure list) | **FAILED** |
| DSV3Lite helix `pp1dp2cp2` | (not in failure list) | **FAILED** |
| DSV3Lite helix `pp2tp1cp2` | FAILED | (not in failure list) |
| Nemotron3Super120B `test_auto_dtype` | FAILED | (not in failure list) |
| `asymmetric_executor[llama-4proc-mpi_kvcache-90]` | FAILED | FAILED |

**Same code, completely different failure profile.** The only consistently-failing test across every PR #13713 build we have data for is `asymmetric_executor[llama-4proc-mpi_kvcache-90]`. Everything else rotates.

This finding invalidates several rounds of prior reasoning. We had been treating these as deterministic regressions and attributing each "fix" to the commit that immediately preceded a CI pass. With flake variance this large, that attribution was noise.

The real signal: the bug is **timing-sensitive and probabilistic**. Tests that have the right preconditions (see §4) hit it some fraction of the time.

## 3. The vertical-vs-horizontal consistency theory

### 3.1 Statement

Pre-PR #13713 used **raw pointers** for `LlmRequest` in `mRequesterFutures` and `mSenderFutures` (verified by reading the pre-PR header at blob `8f833060`):

```cpp
// pre-PR cacheTransceiver.h
virtual void respondAndSendAsync(LlmRequest* llmRequest) = 0;
virtual void requestAndReceiveAsync(LlmRequest* llmRequest) = 0;
virtual bool cancelRequest(LlmRequest* llmRequest) = 0;

std::vector<std::pair<LlmRequest*, std::future<void>>> mSenderFutures;
std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
```

**Vertical consistency** = Python and C++ agree on whether a request is alive.
**Horizontal consistency** = ranks agree on each request's state.

Pre-PR design **broke vertical consistency** (Python could destroy the `LlmRequest` while C++ still held the raw pointer in `mRequesterFutures`/`mSenderFutures`, leading to UAF) **but accidentally preserved horizontal consistency** (when Python terminated a request, all ranks did so based on the same deterministic timeout and at the same iteration, so all ranks' C++ entries became dead-pointers at roughly the same time).

PR #13713 introduced `shared_ptr<LlmRequest>` everywhere on this path. This **fixed vertical consistency** (C++ keeps the request alive until C++ is done with it) **but broke the implicit horizontal consistency** — each rank's C++ side now persists the request entry independently. Differences in network timing across ranks (especially under helix-CP, where per-rank state already differs by design) cause the C++ entry's state to diverge between ranks. Downstream rank-local conditional code (e.g., `if num_fitting_reqs == 0` gating an MPI collective) then enters collectives asymmetrically and deadlocks.

### 3.2 Evidence

**The pre-PR cleanup mechanism existed and worked.** `kv_transfer_timeout_ms` (default 60000ms) was added to main in October 2025 by commit `879039f6d58336c8208f082550757686ece29ae7` ("[feat] Kv transfer timeout (#8459)"). It sets `req.py_kv_transfer_timed_out = True` on timeout; downstream code calls `kv_cache_transceiver.cancel_request` then `_handle_errors`, which terminates the request and frees blocks. This pre-dates PR #13713 by ~7 months.

**On main, the cleanup actually completed because:**
- Python termination eagerly freed `LlmRequest`.
- The C++ raw pointer in `mRequesterFutures` became dangling.
- Subsequent C++ access either crashed (process restart, blocks freed implicitly) or got "lucky" garbage (often returned quickly because the future was treated as done).
- Either way, the entry was effectively removed from the C++ side at roughly the same wall-clock across ranks.

**On PR #13713 the same chain breaks because:**
- Python termination still runs, but the C++ `shared_ptr` keeps `LlmRequest` alive.
- `mRequesterFutures` / `mSenderFutures` retain the zombie entry indefinitely.
- Each rank's `checkGenTransferStatus` / `checkContextTransferStatus` keeps iterating the zombie, observing per-rank-different network states.
- Helix-CP rank state divergence (each CP rank holds a different KV-cache shard, so per-rank occupancy differs by design) now propagates into rank-local conditional code paths that enter MPI collectives asymmetrically.

### 3.3 Code anchors

- pre-PR `LlmRequest*` field at `cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h` blob `8f833060`
- post-PR `std::shared_ptr<LlmRequest>` field at the same path on `pr-13713-head`
- timeout mechanism at `tensorrt_llm/_torch/pyexecutor/py_executor.py` `_check_kv_transfer_timeout`, introduced by `879039f6d5`
- cancel flow at `py_executor.py` line ~4436: `if request.py_kv_transfer_timed_out: is_cancelled = self.kv_cache_transceiver.cancel_request(request) ...`

## 4. Failure pattern: why only some tests fail

The bug requires **two conditions simultaneously**:

### 4.1 Condition 1 — cross-rank state divergence opportunity

| Configuration | Why ranks diverge |
|---|---|
| **Helix CP > 1** | Each CP rank holds a different shard of KV cache (different tokens). Per-rank occupancy differs by design. |
| **Asymmetric ctx vs gen** (`asymmetric_executor`) | Different parallelism on each side → different connection mesh → different per-rank transfer ordering. |
| **PP > 1 on both sides** (`ctxpp2_genpp2`) | Per-rank pipeline state diverges. |
| **DP** (helix `pp1dp2cp2`) | Per-DP-rank request distribution differs. |

Tests **without** this property (single-GPU, TP-only with matching ctx/gen configs) cannot trigger the bug because all ranks always observe identical state.

### 4.2 Condition 2 — a stuck or slow transfer

A transfer must be stuck long enough that ranks observe divergent KV-cache state. Triggers:
- Network jitter / head-of-line blocking on the size-1 send/recv pool (default `mRecvBufferCount = mSendBufferCount = 1`)
- Transient peer slowness
- Specific request-size patterns that hit edge cases in the protocol

This is the **probabilistic** part — same code, same workload, different result.

### 4.3 Cross-check: the one consistently-failing test

`test_asymmetric_executor[llama-4proc-mpi_kvcache-90]` fails on every PR #13713 build. The same test on `ucx_kvcache` and `nixl_kvcache` passes. This is the cleanest evidence that the bug is timing-sensitive at the transport layer:

- `mpi_kvcache` uses MPI for KV transfer — slower / more synchronous → reliably reproduces condition 2.
- `ucx_kvcache` / `nixl_kvcache` are faster → dodge condition 2 in most runs.

Condition 1 (asymmetric ctx/gen) is satisfied by all three transports; only the slower transport reliably hits condition 2.

## 5. Why patch-by-patch rank-asymmetric fixes didn't converge

The commits `bdfdf8be02` (gen-side gate removal) and `53a0692aa4` (ctx-side gate removal) addressed real rank-asymmetric collective entries:

```python
# before — rank-local gate, ABBA risk
if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
    if not all_gen_first:
        self._check_disagg_ctx_cache_transfer_status(1)   # enters MPI collective

# after — always enter
at_least_num = 0
if (num_fitting_reqs == 0 and ...):
    at_least_num = 1
self._check_disagg_ctx_cache_transfer_status(at_least_num)
```

These fixes are correct in isolation. But they address a *downstream symptom*: the rank-asymmetric collective entry fires *because* the upstream condition (a stuck transfer + diverging rank states) already exists. Removing one gate just routes the wedge to the next rank-local conditional.

The patch-by-patch loop produced apparent progress that was largely flake variance (§2). Three rounds of fixes added complexity without converging.

Commit `dbaf7a1106` (my attempt at a generalized rank-symmetric `_pp_retry_until_can_schedule` + C++ ctx-drain symmetrization) was net-negative: it did not fix `asymmetric_executor` (identical failure signature), regressed Qwen3 helix and `ctxpp2_genpp2` in build #39634, and was reverted as `e8f194f728`.

## 6. Solution paths

### 6.1 Path A — gate Layer 3 (shared_ptr) behind the existing cancel flag

**Idea:** when `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=0` (default), C++ holds a non-owning reference; behavior matches main. When the flag is on, full `shared_ptr` ownership and the cancel/poison machinery.

**Scope inventory (verified from headers):**

- 5 public virtual methods in `BaseCacheTransceiver` whose signature was `LlmRequest*` pre-PR and is `shared_ptr<LlmRequest>` post-PR:
  - `respondAndSendAsync`
  - `requestAndReceiveSync`
  - `requestAndReceiveAsync`
  - `cancelRequest`
  - `setContextState`
- 2 fields:
  - `std::vector<std::pair<..., std::future<void>>> mSenderFutures`
  - `std::vector<std::pair<..., std::future<void>>> mRequesterFutures`
- Internal worker struct `RequestAndPromise::mRequest` in `dataTransceiver.cpp`
- 2 internal async APIs `CacheSender::sendAsync` and `CacheReceiver::receiveAsync`
- Nanobind trampolines + bindings at `cpp/tensorrt_llm/nanobind/batch_manager/cacheTransceiver.cpp` (4 `NB_OVERRIDE_PURE` macros + 4 `.def(...)` entries)

**Implementation options:**

**A1. Wrapper class (cleanest):**

```cpp
class TransceiverLlmRequestRef {
public:
    static TransceiverLlmRequestRef make(std::shared_ptr<LlmRequest> req) {
        return common::getEnvDisaggEnableInflightCancel()
            ? TransceiverLlmRequestRef{std::move(req)}   // strong
            : TransceiverLlmRequestRef{req.get()};       // weak
    }
    LlmRequest& operator*()  const { return mStrong ? *mStrong : *mRaw; }
    LlmRequest* operator->() const { return mStrong ? mStrong.get() : mRaw; }
    LlmRequest* get() const { return mStrong ? mStrong.get() : mRaw; }
    // ... copy / move constructors ...
private:
    std::shared_ptr<LlmRequest> mStrong;
    LlmRequest* mRaw{nullptr};
};
```

Replace `shared_ptr<LlmRequest>` with `TransceiverLlmRequestRef` in the 2 field types and the 5 API methods. Most existing call sites (`request->mRequestId`, `*request`, etc.) work unchanged.

**Effort:** ~2-3 engineer-days for the change and verification. Most of the time is verifying that all access paths — including error paths and lambda captures inside `Impl::receiveAsync` and `Impl::requestAndReceiveAsyncMultiThreads` — handle the weak mode correctly.

**Risk:** with flag off, the UAF window from main is restored. Accepted — same risk as main today. **Important caveat:** PR #13713 introduced always-on deadline detection that accesses `request->getKvCacheTransferStart()` and `request->mRequestId`. With flag off, these can dereference a dangling pointer if Python has freed the request. Mitigation: hold a `weak_ptr` on the C++ side and lock for the duration of the deadline check, OR accept the same UAF window as main on this path.

**A2. Prune on timeout (simplest):**

Keep `shared_ptr<LlmRequest>` always. When flag is OFF, in `checkContextTransferStatus`'s and `checkGenTransferStatus`'s deadline blocks, **immediately erase the entry from the futures vector** when the timeout fires, instead of waiting for the future to resolve:

```cpp
if (elapsedMs > kvTransferTimeoutMs.value()) {
    if (common::getEnvDisaggEnableInflightCancel()) {
        // existing flag-on path: cancelRequest, mark errored, erase
    } else {
        // mimic pre-PR: forcibly drop the entry so iteration doesn't
        // see a zombie. Python termination handles KV block release.
        TLLM_LOG_WARNING("Pruning timed-out request %ld (flag off)", request->mRequestId);
        request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
        it = mRequesterFutures.erase(it);
        continue;
    }
}
```

**Effort:** ~0.5 engineer-day for the change. Probably 30-40 net lines of C++ across `checkContextTransferStatus` and `checkGenTransferStatus`.

**Risk:** less semantically equivalent to main than A1 — main relied on Python's `cancel_request` → `_terminate_request` flow, not the C++ deadline. There's a window where Python may still hold a reference. Need careful state-machine handling to avoid double-free.

**A3. Status-quo with Python-side aggressive cleanup (not recommended):**

Don't change the storage type. Add a new C++ method `CacheTransceiver::forceEraseTimedOutRequest(uint64_t)` and call it from Python's `_check_kv_transfer_timeout` when flag is off. Smallest C++ diff but adds a "back door" cleanup API that bypasses the future-driven state machine. Listed only for completeness — A1 or A2 is preferable.

### 6.2 Path B — explicit horizontal consistency layer (long-term)

Add an explicit cross-rank synchronization of per-request state once per scheduling iteration, before any rank-divergent decision:

- All-gather `{request_id: state}` across the relevant comm (`mGroupComm` or `mGroupTensorParaComm` depending on context).
- Take the conservative consensus (e.g., a transfer is "complete" only when *all* ranks see it complete; matches `gatherRequestIds`' existing freq-based logic).
- All subsequent local decisions key off the consensus snapshot, not rank-local state.

**Pros:**
- Architecturally cleanest answer to the entire class of horizontal-consistency bugs.
- Robust to *any* future change that introduces per-rank state divergence — not just PR #13713's shared_ptr.

**Cons:**
- One additional MPI all-gather per iteration. For helix at 4 ranks, ~few KB; latency cost is small but non-zero.
- Needs careful design of what state to sync, how to handle disagreement, and where to invalidate the snapshot.

**Effort:** ~1-2 engineer-weeks for design + implementation + testing.

This is the "right" long-term answer but is too big to block PR #13713 merging.

### 6.3 Path C — waive the flaky tests and merge

**Tactical option** to unblock PR #13713 merge given:
- The fixes in PR #13713 (UAF fix via Layer 3, cancellation API surface) are real improvements.
- The failures are flaky (§2) and concentrated in helix-CP + asymmetric configurations, which are uncommon in production.
- The single deterministic failure (`asymmetric_executor[mpi_kvcache]`) is transport-specific (UCX / NIXL variants pass).
- The patch-by-patch loop is not converging.

**Discipline required:**
- Open a tracking ticket (proposed: TRTLLM-XXXXX) for the proper fix before merging.
- Waive with documented root-cause comments in `tests/integration/test_lists/waives.txt`.
- Scope the `asymmetric_executor` waiver narrowly: only `[llama-4proc-mpi_kvcache-90]`, not the whole test. Preserve coverage of ucx_kvcache and nixl_kvcache variants.
- Acknowledge the deferral in PR #13713's description.
- Treat the waiver as time-bounded — re-enable when the proper fix lands.

## 7. Recommendation

**Sequence:**

1. **Now:** waive (Path C) with proper documentation and the tracking ticket open. Get PR #13713 merged to ship the UAF fix.

2. **Immediately after merge:** ship Path A2 (prune-on-timeout) as a small follow-up PR. This is ~0.5 engineer-day and restores main-equivalent behavior when the flag is off, allowing the waivers to be lifted.

3. **Medium-term (folded into TRTLLM-12721 or new ticket):** Path A1 (wrapper class). Cleaner architecture, no schedule pressure.

4. **Long-term (separate ticket):** Path B (explicit horizontal consistency layer). The structural answer that prevents the entire class of bugs from recurring, regardless of future changes.

## 8. Related docs

- [03-defect-class-stack.md](03-defect-class-stack.md) — the original L1–L10 defect classification.
- [06-fix-approaches/](06-fix-approaches/) — the original RC11–RC13 fix approach exploration.
- [08-next-steps-and-pr-map.md](08-next-steps-and-pr-map.md) — covers TRTLLM-12721 follow-up scope.
- [10-ablation-no-midflight-cancel.md](10-ablation-no-midflight-cancel.md) — empirical exploration of the cancel-off behavior.
- [11-bisect-helix-uaf.md](11-bisect-helix-uaf.md) — earlier bisection plan (superseded by this doc's diagnosis).
- [README.md](README.md) — investigation index.

## 9. Open questions / follow-up

- **Quantify the flake rate.** We have anecdotal evidence (§2) that the failures are flaky. A proper measurement would run the failing tests N times on PR #13713 HEAD and on main, computing pass rates. Estimated effort: 1 day with GPU access.
- **Verify Path A1's UAF window assumption.** PR #13713 added always-on deadline detection that accesses `request->getKvCacheTransferStart()`. With flag off and a weak ref, this can UAF if Python freed the request first. Either accept (matches main) or add a `weak_ptr.lock()` guard. Decision deferred until A1 is implemented.
- **`asymmetric_executor[mpi_kvcache]` deterministic case.** Even with Path A2 prune-on-timeout, this transport-specific failure may persist. Worth a targeted investigation once the flaky cases are off the critical path.
- **Effect on production users.** If anyone in production runs disagg with helix CP, they will hit this same class of bug. Path A2 with cancel-off is roughly main-equivalent and should be safe. Worth confirming with deployment teams which configurations are actually in use.
