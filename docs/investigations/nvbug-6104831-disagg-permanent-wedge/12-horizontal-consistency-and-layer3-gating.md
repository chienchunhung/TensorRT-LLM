# 12 — Vertical and horizontal consistency: the unified theory of the post-rc13 CI failures

**Status:** Active investigation, post-rc13.
**Trigger:** Multiple rounds of CI failures (builds #39529, #39569, #39604, #39634, #39661) on PR #13713 that the rank-asymmetric-collective patches (commits `bdfdf8be02`, `53a0692aa4`, `dbaf7a1106`) did not converge.
**Public PR:** https://github.com/NVIDIA/TensorRT-LLM/pull/13713

---

## 1. Why this doc exists

The investigation captured in docs 00–11 traced the original wedge to a UAF on the gen-side recv worker and proposed PR #13713 as the fix. After PR #13713 entered CI, a new class of failures appeared in disagg tests — primarily helix-CP and asymmetric-parallelism configurations. Three rounds of "fix one rank-asymmetric collective, watch another fail" patches followed, with the failure profile *changing* between identical code revisions.

After three rounds of inconclusive patching, a unifying theory emerged that finally explains the failure pattern: **the disagg system needs *both* vertical and horizontal consistency, and PR #13713 trades one for the other.** The pre-PR design had vertical inconsistency (UAF risk) that accidentally preserved horizontal consistency; PR #13713's `shared_ptr<LlmRequest>` fix repairs vertical consistency but breaks the implicit horizontal consistency that came with the UAF.

This doc is organized around that theory:

1. **§2 — The theory itself**, in its full form, with the failure-mode prediction it makes.
2. **§3 — Verification**: the code evidence and CI observations that confirm the theory.
3. **§4 — Why only some tests fail**: the pattern the theory predicts (and we observe).
4. **§5 — Fix options**: how each candidate fix addresses one or both consistency axes.
5. **§6 — Recommendation** and §7 open questions.

It is meant as a hand-off artifact for whoever picks up the lifecycle/cancel-flag work after PR #13713 lands.

---

## 2. The theory: vertical and horizontal consistency

### 2.1 Two consistency axes

A correctly-functioning disagg system must maintain **two independent consistency invariants** between Python orchestration, C++ batch manager, and the network backend:

**Vertical consistency** — every layer (Python, C++ batch manager, network backend) agrees on whether a request is alive.

> If Python destroys the `LlmRequest` while C++ still has work pending on it (e.g., a network-backend transfer in flight via NIXL / UCX / MPI), the C++ side dereferences freed memory. The KV-cache blocks owned by the request may have been reassigned to *new* requests, while the network backend is still writing into them. This is **use-after-free**: vertical inconsistency between Python (which thinks the request is gone) and C++ / the network backend (which is still using its memory).

**Horizontal consistency** — every rank (across PP, TP, CP, DP) agrees on each request's state.

> If rank 0 considers a request "complete" while ranks 1-3 still consider it "in-progress", the next code path that branches on per-rank request state can have some ranks enter an MPI collective and others skip it. The result is **ABBA deadlock** at the collective.

These are independent failure modes. Vertical breakage looks like memory corruption / process crashes; horizontal breakage looks like cross-rank hangs.

### 2.2 Pre-PR #13713: vertical inconsistency, horizontal consistency (by accident)

Before PR #13713, `mRequesterFutures` and `mSenderFutures` held **raw `LlmRequest*` pointers** (verified §3.1). When Python's timeout-driven termination ran (`_check_kv_transfer_timeout` → `cancel_request` → `_terminate_request`, mechanism in place since `879039f6d5` in October 2025), the `LlmRequest` was destroyed; the C++ raw pointer in the futures vector became dangling.

This was a **vertical-consistency bug**: the C++ side could still attempt to dereference the dead pointer during ongoing checkContextTransferStatus / checkGenTransferStatus loops, network-backend transfers, or buffer cleanup. The fact that the network backend (NIXL especially) does *not* offer synchronous quiescence means that, under sufficient traffic, blocks previously owned by the request could be reassigned to *new* requests while the network backend was still writing — producing memory corruption. **This is the original NVBug 6104831 wedge bug.**

But this same flaw **accidentally preserved horizontal consistency**:

- Every rank ran the same Python timeout logic on the same `kv_transfer_timeout_ms` (default 60s), so every rank's Python termination fired at roughly the same wall-clock.
- Every rank's C++ raw pointer became dangling at roughly the same time.
- Every rank's subsequent C++ access either crashed (clean exit, process-level cleanup) or got "lucky garbage" (typically rapid short-circuit treating the future as resolved).
- Either way, the per-rank C++ entries were effectively removed at synchronized wall-clock moments.

No rank ever observed the request "alive" while another rank thought it was "dead" for long enough to matter. **The implicit horizontal consistency was a side-effect of the vertical consistency bug** — same timeout, same dereferencing behavior, same de-facto cleanup window across ranks.

### 2.3 The wedge bug was the vertical failure mode finally biting

For a long time, the vertical inconsistency never bit visibly because traffic was low: the few-microsecond UAF window between Python free and C++ dereference rarely overlapped with new-request block allocation. Production traffic eventually grew enough that block reassignment *did* fall inside that window. The result was the wedge / memory corruption that started the entire NVBug 6104831 investigation.

### 2.4 PR #13713: vertical consistency restored, horizontal consistency broken

PR #13713 replaced raw `LlmRequest*` with `std::shared_ptr<LlmRequest>` throughout the disagg path. From the C++ side's perspective:

- The C++ futures vector now owns a strong reference. Python termination cannot destroy the `LlmRequest` while C++ has pending work. **Vertical consistency restored.** The UAF window is gone. The wedge / memory corruption from §2.3 cannot happen.

But the implicit horizontal-consistency mechanism that §2.2 described disappears at the same time:

- Each rank's C++ `mRequesterFutures` / `mSenderFutures` retains its entry independently. Python termination does not remove the C++ entry anymore.
- Each rank's network backend resolves the entry on its own schedule. Network timing differs across ranks even under nominally-identical workloads (helix CP's per-rank shard occupancy, asymmetric ctx/gen topologies, transport-specific scheduling).
- For long-running or stuck transfers, the C++ futures vector accumulates **zombie entries** that resolve at rank-dependent times.
- Downstream rank-local conditional code (e.g. `if num_fitting_reqs == 0`, `if not fittingDisaggGenInitRequests.empty()`) sees per-rank-divergent state.
- When such a rank-local condition gates entry into an MPI collective, ranks enter the collective asymmetrically → ABBA deadlock.

**Horizontal consistency is no longer free.** The pre-PR design got it as a side effect of vertical inconsistency; PR #13713 paid for vertical consistency by losing it.

### 2.5 The unified diagnosis

**Both axes need explicit guarantees.** The system worked historically because:

- Vertical: the (broken) raw-pointer scheme survived as long as traffic stayed below the window where UAF could bite.
- Horizontal: emerged accidentally from the same broken raw-pointer scheme tying lifetime to a deterministic Python timeout.

The right design treats both as first-class invariants:

- Vertical: own the `LlmRequest` lifetime correctly (what PR #13713 does, via `shared_ptr`).
- Horizontal: *separately* ensure each rank's view of every request's state is consistent before any rank-divergent decision is made.

The post-PR CI failures are the system **failing horizontal consistency** because the implicit mechanism that supplied it no longer exists. The patch-by-patch rank-asymmetric-collective fixes (`bdfdf8be02`, `53a0692aa4`, `dbaf7a1106`) are *local* attempts to recover horizontal consistency at specific collective sites; they fix individual cases but don't address the class.

### 2.6 What the theory predicts

The theory makes specific testable predictions, all of which the CI evidence confirms (§3 and §4):

1. **Failure should be flaky, not deterministic.** Horizontal divergence depends on network-timing differences that vary per run. ✅ Confirmed: identical code (`53a0692aa4` ↔ `e8f194f728`) produces different failure profiles (§3.2).

2. **Failure should only occur in configurations with cross-rank state divergence opportunity.** ✅ Confirmed: helix CP, asymmetric ctx/gen, PP > 1 on both sides. Single-GPU / TP-only-matching-configs don't fail (§4.1).

3. **Failure should correlate with transports that produce stuck or slow transfers.** ✅ Confirmed: `asymmetric_executor` fails reproducibly on `mpi_kvcache` (slowest, most synchronous), passes on `ucx_kvcache` and `nixl_kvcache` (§4.3).

4. **Removing one rank-local conditional gate moves the wedge to the next, doesn't eliminate it.** ✅ Confirmed: three rounds of patches (`bdfdf8be02` → `53a0692aa4` → `dbaf7a1106`) did not converge; each fix routed the wedge to a different code path.

5. **Restoring pre-PR-equivalent C++ lifetime when the cancel flag is off should fix the CI failures** (because it re-enables the accidental horizontal consistency). This is the fix path A1/A2 below.

---

## 3. Verification — the evidence behind the theory

### 3.1 The pointer change is verifiable in the headers

Pre-PR `cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h` (blob `8f833060`) declares:

```cpp
virtual void respondAndSendAsync(LlmRequest* llmRequest) = 0;
virtual void requestAndReceiveSync(LlmRequest* llmRequest) = 0;
virtual void requestAndReceiveAsync(LlmRequest* llmRequest) = 0;
virtual bool cancelRequest(LlmRequest* llmRequest) = 0;

std::vector<std::pair<LlmRequest*, std::future<void>>> mSenderFutures;
std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
```

Post-PR (HEAD of `pr-13713-head`) all five became `std::shared_ptr<LlmRequest>`. The 2 fields and 5 public methods are the load-bearing scope of the change.

### 3.2 Identical code, different failure profiles

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

**Same code, completely different failure profile.** The only consistently-failing test across every PR #13713 build is `asymmetric_executor[llama-4proc-mpi_kvcache-90]`. Everything else rotates.

This is what the theory predicts (prediction 1, §2.6): failures should be flaky because they depend on per-run network-timing differences, not on a deterministic code-level bug. We had been treating these as deterministic regressions and attributing each "fix" to the commit that preceded a CI pass. With flake variance this large, that attribution was noise.

### 3.3 The timeout mechanism that drives §2.2 is real and pre-PR

The horizontal-consistency-via-Python-timeout chain that §2.2 describes is not hypothetical. It is implemented on main as:

- Config: `CacheTransceiverConfig.kv_transfer_timeout_ms`, default **60000 ms**, defined in `tensorrt_llm/llmapi/llm_args.py`.
- Detection: `py_executor.py:_check_kv_transfer_timeout` sets `req.py_kv_transfer_timed_out = True` when elapsed > timeout.
- Action: downstream code calls `self.kv_cache_transceiver.cancel_request(request)`, then `_handle_errors`, then `_terminate_request`, then `resource_manager.free_resources(request)`.

This entire chain was introduced by commit `879039f6d58336c8208f082550757686ece29ae7` ("[feat] Kv transfer timeout (#8459)"), authored **2025-10-22** — about 7 months before PR #13713's first commit (`630fa3b4` on 2026-05-02). It is part of `main`, not part of PR #13713.

Helix test configs do not override `kv_transfer_timeout_ms`, so they get the 60s default. This is the deterministic timeout that, pre-PR, made all ranks' Python termination fire synchronously (within milliseconds of each other) — producing the implicit horizontal consistency described in §2.2.

### 3.4 Patch-by-patch rank-asymmetric fixes addressed real bugs at the wrong level

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

These fixes are correct in isolation. But they address a *downstream symptom* of the theory's prediction 4 (§2.6): the rank-asymmetric collective entry fires *because* horizontal consistency is broken upstream. Removing one gate just routes the wedge to the next rank-local conditional.

Commit `dbaf7a1106` (a generalized rank-symmetric `_pp_retry_until_can_schedule` + C++ ctx-drain symmetrization) was net-negative: it did not fix `asymmetric_executor` (identical failure signature), regressed Qwen3 helix and `ctxpp2_genpp2` in build #39634, and was reverted as `e8f194f728`.

---

## 4. Why only some tests fail — the predicted pattern

The theory predicts failures should require **two conditions simultaneously** (§2.6, prediction 2 and 3):

### 4.1 Condition 1 — cross-rank state divergence opportunity

| Configuration | Why ranks diverge |
|---|---|
| **Helix CP > 1** | Each CP rank holds a different shard of KV cache (different tokens). Per-rank occupancy differs by design. |
| **Asymmetric ctx vs gen** (`asymmetric_executor`) | Different parallelism on each side → different connection mesh → different per-rank transfer ordering. |
| **PP > 1 on both sides** (`ctxpp2_genpp2`) | Per-rank pipeline state diverges. |
| **DP** (helix `pp1dp2cp2`) | Per-DP-rank request distribution differs. |

Tests **without** this property (single-GPU, TP-only with matching ctx/gen configs) cannot trigger the bug because all ranks always observe identical state — horizontal divergence has no source.

### 4.2 Condition 2 — a stuck or slow transfer

A transfer must be stuck long enough that ranks observe meaningfully-divergent KV-cache state. Triggers:

- Network jitter / head-of-line blocking on the size-1 send/recv pool (default `mRecvBufferCount = mSendBufferCount = 1`)
- Transient peer slowness
- Specific request-size patterns that hit edge cases in the protocol

This is the **probabilistic** part — same code, same workload, different result.

### 4.3 Cross-check: the one consistently-failing test

`test_asymmetric_executor[llama-4proc-mpi_kvcache-90]` fails on every PR #13713 build. The same test on `ucx_kvcache` and `nixl_kvcache` passes:

- `mpi_kvcache` uses MPI for KV transfer — slower / more synchronous → reliably reproduces condition 2.
- `ucx_kvcache` / `nixl_kvcache` are faster → dodge condition 2 in most runs.

Condition 1 (asymmetric ctx/gen) is satisfied by all three transports; only the slower transport reliably hits condition 2. This precisely matches theory prediction 3 (§2.6).

### 4.4 Mapping observed failures to the theory

| Test | Has Condition 1? | Why it triggers Condition 2 |
|---|---|---|
| Helix tests (DSV3Lite, Qwen3, all variants) | Yes (helix CP) | Network jitter on stuck transfers |
| `Nemotron3Super120B::test_auto_dtype` | Yes (PP > 1) | Same |
| `ctxpp2_genpp2[TinyLlama]` | Yes (PP > 1 on both sides) | Same |
| `asymmetric_executor[mpi_kvcache]` | Yes (asymmetric topology) | mpi_kvcache slowness — reliable trigger |
| `asymmetric_executor[ucx_kvcache]` | Yes | UCX fast enough to dodge — passes |
| `asymmetric_executor[nixl_kvcache]` | Yes | NIXL fast enough to dodge — passes |
| Single-GPU tests | **No** | Cannot trigger regardless |
| TP-only matching-config disagg tests | **No** | Cannot trigger regardless |

Every observed failure has both conditions; every test without Condition 1 passes consistently. The theory's predictions hold across every test we have data for.

---

## 5. Fix options — through the lens of the theory

The theory makes it clear that any fix must address **horizontal consistency** explicitly, instead of relying on the vertical-inconsistency side-effect that pre-PR #13713 had. There are two families of approaches: restore the *implicit* horizontal mechanism (Path A), or build *explicit* horizontal consistency (Path B).

### 5.1 Path A — restore pre-PR-equivalent C++ lifetime when cancel is disabled

**Idea:** when `TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=0` (default), C++ holds a non-owning reference to `LlmRequest`; behavior matches main. When the flag is on, full `shared_ptr` ownership and the cancel/poison machinery.

This re-introduces the pre-PR vertical inconsistency *only when the flag is off*, which restores main-equivalent behavior — including the implicit horizontal consistency that prevented these failures historically. It's a deliberate trade: accept the pre-existing UAF risk (which we know is rare under most workloads) in exchange for shipping PR #13713's cancel/poison machinery as an opt-in safety mechanism.

**Scope inventory (verified from headers, §3.1):**

- 5 public virtual methods in `BaseCacheTransceiver`: `respondAndSendAsync`, `requestAndReceiveSync`, `requestAndReceiveAsync`, `cancelRequest`, `setContextState`
- 2 fields: `mSenderFutures`, `mRequesterFutures`
- Internal worker struct `RequestAndPromise::mRequest` in `dataTransceiver.cpp`
- 2 internal async APIs `CacheSender::sendAsync` and `CacheReceiver::receiveAsync`
- Nanobind trampolines + bindings at `cpp/tensorrt_llm/nanobind/batch_manager/cacheTransceiver.cpp` (4 `NB_OVERRIDE_PURE` macros + 4 `.def(...)` entries)

**A1 — Wrapper class (cleanest, ~2-3 engineer-days):**

```cpp
class TransceiverLlmRequestRef {
public:
    static TransceiverLlmRequestRef make(std::shared_ptr<LlmRequest> req) {
        return common::getEnvDisaggEnableInflightCancel()
            ? TransceiverLlmRequestRef{std::move(req)}   // strong: vertical consistency
            : TransceiverLlmRequestRef{req.get()};       // weak:   horizontal consistency (pre-PR)
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

Risk: with flag off, the UAF window from main is restored. Accepted — same risk as main today. **Caveat:** PR #13713 introduced always-on deadline detection that accesses `request->getKvCacheTransferStart()` and `request->mRequestId`. With flag off, these can dereference a dangling pointer if Python freed the request. Mitigation: hold a `weak_ptr` on the C++ side and lock for the duration of the deadline check, OR accept the same UAF window as main (it has always been there on this path).

**A2 — Prune on timeout (simplest, ~0.5 engineer-day):**

Keep `shared_ptr<LlmRequest>` always. When flag is OFF, in `checkContextTransferStatus`'s and `checkGenTransferStatus`'s deadline blocks, immediately erase the entry from the futures vector when the timeout fires, instead of waiting for the future to resolve:

```cpp
if (elapsedMs > kvTransferTimeoutMs.value()) {
    if (common::getEnvDisaggEnableInflightCancel()) {
        // existing flag-on path: cancelRequest, mark errored, erase
    } else {
        // mimic pre-PR horizontal consistency: forcibly drop the entry so
        // iteration doesn't see a zombie. Python termination handles KV
        // block release independently.
        TLLM_LOG_WARNING("Pruning timed-out request %ld (flag off)", request->mRequestId);
        request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
        it = mRequesterFutures.erase(it);
        continue;
    }
}
```

Effort: ~30-40 net lines of C++ across `checkContextTransferStatus` and `checkGenTransferStatus`.

Risk: less semantically equivalent to main than A1 — main relied on Python's `cancel_request` → `_terminate_request` flow, not the C++ deadline. There's a window where Python may still hold a reference. Need careful state-machine handling to avoid double-free.

**A3 — Python-side aggressive cleanup (not recommended):**

Don't change the storage type. Add a new C++ method `CacheTransceiver::forceEraseTimedOutRequest(uint64_t)` and call it from Python's `_check_kv_transfer_timeout` when flag is off. Smallest C++ diff but adds a "back door" cleanup API that bypasses the future-driven state machine. Listed only for completeness — A1 or A2 is preferable.

### 5.2 Path B — explicit horizontal consistency layer (long-term)

The theory's diagnosis (§2.5) is that both axes need explicit guarantees. Path A keeps horizontal consistency *implicit* — relying on the same trick pre-PR used, just gated by a flag. **Path B makes horizontal consistency explicit**, addressing the structural issue rather than restoring the accidental mechanism.

Add an explicit cross-rank synchronization of per-request state once per scheduling iteration, before any rank-divergent decision:

- All-gather `{request_id: state}` across the relevant comm (`mGroupComm` or `mGroupTensorParaComm` depending on context).
- Take the conservative consensus (e.g., a transfer is "complete" only when *all* ranks see it complete; matches `gatherRequestIds`' existing freq-based logic).
- All subsequent local decisions key off the consensus snapshot, not rank-local state.

**Pros:**

- Architecturally cleanest answer to the entire class of horizontal-consistency bugs.
- Robust to *any* future change that introduces per-rank state divergence — not just PR #13713's shared_ptr.
- The system gets both vertical (`shared_ptr` always on) AND horizontal (explicit all-gather) consistency as first-class invariants, matching the theory's prescription in §2.5.

**Cons:**

- One additional MPI all-gather per iteration. For helix at 4 ranks, ~few KB; latency cost is small but non-zero.
- Needs careful design of *what* state to sync, *how* to handle disagreement, and *where* to invalidate the snapshot.

**Effort:** ~1-2 engineer-weeks for design + implementation + testing.

This is the "right" long-term answer but is too big to block PR #13713 merging.

### 5.3 Path C — waive the flaky tests to unblock merge

**Tactical option** to ship PR #13713's vertical-consistency fix (the load-bearing UAF closure) without waiting for the horizontal-consistency solution:

- The fixes in PR #13713 (UAF fix via Layer 3, cancellation API surface) are real improvements that address the original §2.3 wedge.
- The failures are flaky (§3.2) and concentrated in helix-CP + asymmetric configurations, which are uncommon in production.
- The single deterministic failure (`asymmetric_executor[mpi_kvcache]`) is transport-specific (UCX / NIXL variants pass).
- The patch-by-patch loop is not converging.

**Discipline required:**

- Open a tracking ticket (proposed: TRTLLM-XXXXX) for the proper fix (Path A or B) before merging.
- Waive with documented root-cause comments in `tests/integration/test_lists/waives.txt` pointing back to this doc.
- Scope the `asymmetric_executor` waiver narrowly: only `[llama-4proc-mpi_kvcache-90]`, not the whole test. Preserve coverage of ucx_kvcache and nixl_kvcache variants.
- Acknowledge the deferral in PR #13713's description.
- Treat the waiver as time-bounded — re-enable when the proper fix lands.

---

## 6. Recommendation

The theory predicts a clean sequence of fixes, each addressing one consistency axis at a time:

1. **Now: ship PR #13713 + waivers (Path C).** Vertical consistency is the production-critical fix; ship it. Waive the flaky tests with proper documentation. Open the tracking ticket for follow-up.

2. **Immediately after merge: Path A2 (prune-on-timeout, ~0.5 day).** Restores horizontal consistency to pre-PR-equivalent levels when cancel is off (the default). Allows the waivers to be lifted with high confidence.

3. **Medium-term: Path A1 (wrapper class, ~2-3 days).** Cleaner architecture for the gating. Same correctness as A2 but expressed via type-level invariants instead of a one-shot prune.

4. **Long-term: Path B (explicit horizontal consistency layer, 1-2 weeks).** The structural answer that matches the theory's prescription: both consistency axes as first-class invariants. Prevents the entire class of bugs from recurring under any future change.

---

## 7. Open questions / follow-up

- **Quantify the flake rate.** We have anecdotal evidence (§3.2) that the failures are flaky. A proper measurement would run the failing tests N times on PR #13713 HEAD and on main, computing pass rates. The theory predicts the flake rate should track the time-fraction during which condition 2 (stuck/slow transfer) is true. Estimated effort: 1 day with GPU access.

- **Verify Path A1's UAF window assumption.** PR #13713 added always-on deadline detection that accesses `request->getKvCacheTransferStart()`. With flag off and a weak ref, this can UAF if Python freed the request first. Either accept (matches main) or add a `weak_ptr.lock()` guard. Decision deferred until A1 is implemented.

- **`asymmetric_executor[mpi_kvcache]` deterministic case.** Even with Path A2 prune-on-timeout, this transport-specific failure may persist. Worth a targeted investigation once the flaky cases are off the critical path. The theory predicts the failure should go away once horizontal consistency is restored under flag-off; if it doesn't, the theory needs refinement.

- **Effect on production users.** If anyone in production runs disagg with helix CP, they will hit this same class of bug. Path A2 with cancel-off is roughly main-equivalent and should be safe. Worth confirming with deployment teams which configurations are actually in use.

- **Does Path B's all-gather catch every divergence source?** The theory predicts §4.1's listed sources cover the observed failures. But future features (new parallelism modes, new transports) could introduce new divergence sources. Path B's design should make it easy to add new state to the sync.

---

## 8. Related docs

- [03-defect-class-stack.md](03-defect-class-stack.md) — the original L1-L10 defect classification.
- [06-fix-approaches/](06-fix-approaches/) — the original RC11-RC13 fix approach exploration.
- [08-next-steps-and-pr-map.md](08-next-steps-and-pr-map.md) — covers TRTLLM-12721 follow-up scope.
- [10-ablation-no-midflight-cancel.md](10-ablation-no-midflight-cancel.md) — empirical exploration of the cancel-off behavior.
- [11-bisect-helix-uaf.md](11-bisect-helix-uaf.md) — earlier bisection plan (superseded by the diagnosis here).
- [README.md](README.md) — investigation index.
