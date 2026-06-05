# Phase 1 — V1 disagg cancellation consensus collective design

| | |
|---|---|
| **Phase** | 1 (architectural design — precedes implementation) |
| **JIRA** | [TRTLLM-12721](https://jirasw.nvidia.com/browse/TRTLLM-12721) |
| **Owner** | Chien-Chun Hung |
| **Status** | Design proposal; not yet implemented. Implementation tracked separately as the in-flight cancellation surface follow-up to <https://github.com/NVIDIA/TensorRT-LLM/pull/14768>. |
| **Depends on** | (1) <https://github.com/NVIDIA/TensorRT-LLM/pull/14768> already merged (always-on lifetime / RAII / dedup / NIXL keep-alive); (2) the upcoming `dataTransceiver` `shared_ptr<LlmRequest>` follow-up at <https://github.com/NVIDIA/TensorRT-LLM/pull/14979>; (3) the V1 `_consensus_outcome` port from doc 14 (env-gated, currently default OFF) as a structural reference point. |

## Why this exists

The investigation captured in [`docs/investigations/nvbug-6104831-disagg-permanent-wedge/18-pr-14746-prior-art-and-v1-two-layer-gap.md`](../../investigations/nvbug-6104831-disagg-permanent-wedge/18-pr-14746-prior-art-and-v1-two-layer-gap.md) (especially §3) established that the V1 disagg path has a two-layer cross-rank-consistency problem:

- **L1** — *Which* requests are flagged for action (timed-out, cancellation-eligible, etc.). PR #14746 closes this for the timeout flag via `dist.allgather → union`. The in-flight cancellation follow-up needs to do the same for the recv-side dedup state (the A3 problem from doc 17).
- **L2** — *What state transitions* the ranks apply in response. V2 closes this via `KvCacheTransceiverV2._consensus_outcome` (three allgathers — one each for cancelled / failed / completed rids). V1 has no equivalent; this is the gap exercised by `cacheTransceiver.cpp:689-690` where every rank independently does `cancelRequest` + `setState(kDISAGG_TRANS_ERROR)` with no allgather between detection and action.

This document specifies the recommended **consensus collective shape** for the V1 follow-up. It does not specify *where* in the V1 code the collective fires (that is a separate implementation-plan concern); it specifies how the collective should be structured to minimize round-trips, reuse existing V1 infrastructure, and stay reviewable.

## The decision

**Use a single allgather of packed `(rid, state)` values for both L1 and L2 consensus, rather than V2's pattern of three separate allgathers per category.**

Rationale and full design follow.

## Encoding

Each per-rank intent is encoded as a single `uint64`:

```cpp
// state in the high 4 bits, rid in the low 60 bits — request ids fit comfortably
// in 60 bits (we don't generate 2^60 requests); 4 bits encodes the 4-state enum
// (IN_PROGRESS / COMPLETED / FAILED / CANCELLED) with headroom.
constexpr uint64_t kStateShift = 60;
constexpr uint64_t kRidMask = (1ULL << kStateShift) - 1;

inline uint64_t pack(RequestIdType rid, RequestState s)
{
    return (uint64_t(s) << kStateShift) | (rid & kRidMask);
}

inline std::pair<RequestIdType, RequestState> unpack(uint64_t v)
{
    return {v & kRidMask, RequestState(v >> kStateShift)};
}
```

### Priority-ordering invariant

The `RequestState` enum values **must be assigned so the numeric ordering equals the consensus priority** (CANCELLED > FAILED > COMPLETED > IN_PROGRESS):

```cpp
enum class RequestState : uint8_t
{
    IN_PROGRESS = 0,
    COMPLETED = 1,
    FAILED = 2,
    CANCELLED = 3,
};

static_assert(static_cast<uint8_t>(RequestState::CANCELLED) > static_cast<uint8_t>(RequestState::FAILED));
static_assert(static_cast<uint8_t>(RequestState::FAILED) > static_cast<uint8_t>(RequestState::COMPLETED));
static_assert(static_cast<uint8_t>(RequestState::COMPLETED) > static_cast<uint8_t>(RequestState::IN_PROGRESS));
```

The `static_assert` block is load-bearing — if a future enum reorder breaks the priority ordering, the consensus reduce silently produces wrong outputs (e.g. a CANCELLED request would be downgraded to COMPLETED). The asserts force the compiler to catch the breakage.

## Per-rank build

Each rank scans its current request batch and produces a list of non-IN_PROGRESS packed values:

```cpp
std::vector<uint64_t> local;
local.reserve(currentBatch.size());
for (auto& req : currentBatch)
{
    if (req.state != RequestState::IN_PROGRESS)
    {
        local.push_back(pack(req.id, req.state));
    }
}
```

**IN_PROGRESS is implicit by absence.** A rid not present in any rank's contribution stays IN_PROGRESS in the consensus. This matches V2's `_consensus_outcome` semantic and keeps the wire payload small (only state transitions are sent).

## The collective

V1 already has `gatherRequestIds(syncComm, vector<uint64_t>)` in `cacheTransceiver.cpp:482`. **No new collective primitive is needed** — we reuse the existing helper as-is:

```cpp
auto global = gatherRequestIds(syncComm, local);  // returns vector<uint64_t> aggregated from all ranks
```

The helper already handles variable-length payloads (allgather of the per-rank size, then allgatherv of the data). Behavior unchanged.

## Consensus reduce

A single pass over the gathered values builds a per-rid tally:

```cpp
struct Tally
{
    RequestState maxState = RequestState::IN_PROGRESS;
    int completedCount = 0;
};

std::unordered_map<RequestIdType, Tally> tally;
for (uint64_t packed : global)
{
    auto [rid, state] = unpack(packed);
    auto& t = tally[rid];
    t.maxState = std::max(t.maxState, state);  // priority-encoded: CANCELLED > FAILED > COMPLETED > IN_PROGRESS
    if (state == RequestState::COMPLETED)
    {
        t.completedCount++;
    }
}

// CANCELLED / FAILED follow UNION semantics — maxState already captures them.
// COMPLETED follows INTERSECTION semantics — only consensus-COMPLETED when ALL ranks reported COMPLETED;
// otherwise some rank still sees IN_PROGRESS and the global view must be IN_PROGRESS.
for (auto& [rid, t] : tally)
{
    if (t.maxState == RequestState::COMPLETED && t.completedCount < nRanks)
    {
        t.maxState = RequestState::IN_PROGRESS;
    }
}
```

The reduce is **O(N)** where N = total entries in the gathered vector (sum over ranks of non-IN_PROGRESS rids in this iteration). Two passes: one to tally, one to apply the INTERSECTION downgrade for COMPLETED.

## Cost comparison vs the 3-list (V2 shape) alternative

| Aspect | V2 shape (3 lists) | Packed `{rid, state}` (this design) |
|---|---|---|
| Collectives per iter | 3 (`cancelled`, `failed`, `completed` lists, separately) | 1 |
| Multiplier for ADP or PP > 1 | × 2 (V2 does TP allgather + PP allgather) → 6 | × 2 → 2 |
| Underlying allgatherv "size sync + data sync" | × 2 again → 12 actual MPI/NCCL calls in the worst case | × 2 again → 4 |
| Synchronization barriers | 3 (or 6 for PP) | 1 (or 2 for PP) |
| Wire bytes for non-IN_PROGRESS rids | sum of three `vector<uint64>` | one `vector<uint64>` of the same total |
| Existing V1 infrastructure | would need a new typed `gatherRequestIds`-style helper per category | reuses `gatherRequestIds(syncComm, vector<uint64>)` as-is — zero new collective code |
| Code clarity | high (the three list names self-document the intent) | medium (priority reduce + COMPLETED-count downgrade) |

The latency savings are most visible in cross-node disagg over TCP/IB (where per-allgather latency is 10× higher than on NVLink), under PP > 1 (collective count doubles for V2's shape), and on hot paths exercised at high request rates.

## Open design questions for the implementation PR

The following remain to be answered in code, not in this design doc:

1. **Combine L1 (dedup) and L2 (cancel outcome) into one packed gather, or two distinct gathers?** They fire at different lockstep call sites (the dedup gather inside `_recv_disagg_gen_cache`; the cancel-outcome gather at `_handle_responses` / `_check_disagg_ctx_cache_transfer_status`). Folding into one would require co-locating the call sites, which may not be natural. Probably easiest to start with two separate packed gathers — still 2 collectives total, vs. 4+ for the three-list shape — then fold only if call-site co-location turns out to make sense.

2. **Where to define `pack` / `unpack` and the `RequestState` enum?** Likely in `cacheTransceiver.h` next to `gatherRequestIds`, since both are V1 disagg consensus primitives. The enum should live with the existing `LlmRequestState` to keep state-related types together; the helpers should be free functions in the `tensorrt_llm::batch_manager` namespace.

3. **Should we widen `RequestState` to 8 bits to leave more headroom?** With 8 bits we get 256 possible states; with 4 bits we get 16. Either fits comfortably in the high bits of a `uint64`. 4 bits is enough for the current design but constrains future state additions. Cost of 8 bits: rid is restricted to 56 bits (still trivially sufficient). Decision is a style call; leaning toward 8 bits for future-proofing.

4. **Integration with V1's existing readiness consensus.** `cacheTransceiver.cpp:610` already calls `gatherRequestIds(syncComm, contextCompleteRequestIds)` for the readiness allgather. The new packed consensus is structurally a generalization — readiness is the COMPLETED-INTERSECTION case. Could the existing readiness consensus eventually be folded into the packed one? Probably yes, but **not as part of this follow-up** — touching `checkContextTransferStatus` / `checkGenTransferStatus` is a separate concern, and the readiness gather already works.

## V2 propagation — deferred decision, contingent on V1 measurements

If the V1 follow-up's packed-state allgather lands and we observe the predicted latency reduction (especially on multi-node / PP > 1 / ADP configs), **the same encoding is mechanically applicable to V2's `_consensus_outcome`** at `tensorrt_llm/_torch/disaggregation/transceiver.py:280-298`:

- V2 currently calls `_allgather_or_passthrough` three times (one each for `cancelled`, `failed`, `completed` lists)
- The packed variant collapses to one call, with the same priority-reduce + COMPLETED-count logic
- V2's two `_consensus_outcome` call sites (`_ctx_consensus_outcome`, `_gen_consensus_outcome`) would each go from 3 → 1 collective, and the ctx-side TP-then-PP chain (already 2 sub-collectives per call) goes from 6 → 2

### Reasons to defer the V2 propagation behind the V1 work

1. V2's existing code is reviewable and works; the cost of refactoring V2 without measured benefit is non-trivial (touches V2 test harness, multi-rank integration tests, doc 14's V1 port that mirrors V2's shape).
2. V2 deployments are typically intra-node (TP within an NVLink domain), where allgather latency is sub-100μs and the 3→1 savings amount to maybe 50μs/iter — real but not transformative.
3. Measuring on V1 first gives us a concrete data point under realistic disagg load to justify the V2 refactor or shelve it. Without measurement, the V2 refactor would be guessing.

### Triggers to revisit V2

Any one of:
- V1 follow-up measurements show >5% wall-clock improvement on a representative disagg config (e.g., the [Phase-0 stress test suite's](phase0-stress-test-suite.md) marathon configs).
- A V2 deployment on cross-node disagg surfaces synchronization overhead as a top-3 bottleneck in NVTX traces.
- Independent maintenance work on V2's `_consensus_outcome` opens the door for a structural change at low marginal cost.

Track as a follow-up under TRTLLM-12721, separate from the V1 cancellation PR.

## Cross-references

- [`README.md`](README.md) — the overarching TRTLLM-12721 cancellation re-design document.
- [Doc 18 §3](../../investigations/nvbug-6104831-disagg-permanent-wedge/18-pr-14746-prior-art-and-v1-two-layer-gap.md#3-concrete-l2-evidence--v1-vs-v2-with-line-citations) — the L1 / L2 framing and the V1-vs-V2 line citations that motivate this design.
- [Doc 14](../../investigations/nvbug-6104831-disagg-permanent-wedge/14-cross-rank-consistency-enforcement.md) — the env-gated V1 `_consensus_outcome` port that mirrors V2's three-list shape. This design proposes the packed alternative for the cancellation-specific consensus; doc 14's port remains useful as a structural reference but its three-list shape is not the recommended target.
- [PR #14746](https://github.com/NVIDIA/TensorRT-LLM/pull/14746) — concrete L1 consensus implementation in the `feat/deepseek_v4` branch (single allgather UNION of timeout-flagged rids). The packed design generalizes this to multi-state consensus while preserving the single-collective shape.
- [phase0-stress-test-suite.md](phase0-stress-test-suite.md) — the regression gate this design will be measured against once implemented.
