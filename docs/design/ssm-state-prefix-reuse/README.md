# SSM State Prefix Reuse for Hybrid Linear-Attention Models

**Status:** Pitch Draft
**Created:** 2026-06-26

## Summary

TensorRT-LLM supports KV cache block reuse for softmax attention, allowing
requests with shared token prefixes to skip redundant prefill work. Hybrid
models such as Qwen3-Next and Nemotron-H include linear-attention or SSM layers
alongside softmax attention layers. Reusing only the attention KV blocks is not
enough for these models: the SSM layers must also resume from the matching
recurrent state and causal convolution history.

[PR #12896](https://github.com/NVIDIA/TensorRT-LLM/pull/12896)
("[TRTLLM-12026][feat] Support MTP with block reuse enabled for hybrid models")
merged the majority of the implementation foundation for this problem. It added
unified KV + recurrent-state cache management, SSM snapshot intervals, recurrent
state cache blocks, placeholder blocks, radix-tree lookup for snapshot
positions, memory-budget accounting, forced context chunking at snapshot
boundaries, and tests for hybrid block reuse with MTP.

PR #12896 was motivated by hybrid MTP, where draft and verification work happen
inside one request. The project scoped here is broader: reuse SSM state for a
new request whose prompt shares a prefix with earlier requests. The merged
foundation is still relevant because it puts recurrent-state snapshots into the
same prefix-indexed cache machinery used by attention KV blocks.

This document scopes the remaining project work as a follow-up to that
foundation: validate the supported surface, close major integration gaps, add
production observability, and decide when hybrid SSM block reuse can be enabled
more broadly.

## Motivation

Shared-prefix workloads are common in production serving:

- Agentic workflows repeatedly resend a system prompt, tool schemas, policy
  text, conversation history, retrieved documents, and then append a small tool
  result or next user turn.
- Branching and best-of-N workflows explore multiple continuations from the same
  prefix.
- RAG and coding agents often fan out requests that share a long document or
  repository context.
- Multi-tenant serving sees repeated chat templates and common instruction
  prefixes across requests.

The agentic benchmark draft that motivates this work has the same shape. It
models multi-turn trajectories where each turn can carry a large precomputed or
cached input prefix plus a much smaller fresh input suffix. It also permits
reuse within a trajectory and system-prompt reuse across trajectories. That
means the target workload is not only "continue one live request"; it is many
new requests that repeatedly replay a long prefix.

For attention-only models, KV block reuse avoids recomputing the shared prefix.
For hybrid SSM/linear-attention models, the same optimization requires matching
SSM state reuse. PR #12896 provides a first implementation path for this in the
single-instance PyTorch backend; the remaining question is how far that support
extends and what gaps block production default enablement.

## Current State and Prior Art

Historically, the PyTorch backend had two separate concepts:

- **KV block reuse** is prefix-indexed. The KV cache manager uses token-prefix
  lookup to reuse committed KV blocks and adjust the remaining context work.
- **Mamba/SSM state caching** is request-slot based. The Mamba cache manager
  stores per-request convolution state and SSM recurrent state, then updates
  those tensors in place as the request advances.

PR #12896 added a third path for hybrid models:

- **`CppMambaHybridCacheManager`** stores attention KV blocks and Mamba recurrent
  state blocks in one C++ KV cache manager pool.
- **`LinearCacheType.RECURRENT_STATES`** represents the recurrent-state cache
  window for SSM/linear-attention layers.
- **`mamba_state_cache_interval`** controls how often SSM/conv snapshots are
  saved for prefix reuse.
- **Placeholder recurrent-state blocks** preserve block positions that do not
  carry a real SSM snapshot.
- **Radix-tree lookup at all positions** allows the recurrent-state cache to
  distinguish real snapshot blocks from placeholder positions.
- **Forced context chunking at snapshot boundaries** ensures prefill stops at
  positions where SSM state can be materialized safely.
- **Affine memory budgeting** accounts for both attention KV bytes per token and
  amortized recurrent-state snapshot bytes.

That is most of the hard implementation foundation. Remaining work should avoid
reimplementing this path and instead assess, harden, and extend it.

## Problem Scope

The project should answer one narrow question:

> Given a new request whose token prefix has already been processed by the same
> model instance, can TRT-LLM resume both attention and SSM layers from the
> cached prefix state and process only the suffix?

PR #12896 answers "yes" for an important subset. The follow-up scope is to make
that support pitchable and production-ready:

- PyTorch backend.
- Hybrid models using the unified `CppMambaHybridCacheManager` path.
- Exact snapshot-boundary reuse first.
- Aggregate serving first.
- Inter-request reuse for repeated prompts inside one serving instance.
- Single model identity, single tokenizer, same quantization and runtime layout.
- Explicit opt-in validation before changing model defaults.

Still out of scope for the first production milestone:

- Reusing SSM state across different model weights, revisions, LoRA adapters, or
  incompatible TP/PP/DP layouts.
- Cross-instance or persistent SSM state reuse.
- Partial-token or partial-block SSM resume.
- Changing SSM kernel math.

## Design Foundation

SSM prefix reuse makes SSM state a first-class companion to reusable KV blocks.
PR #12896 implements this direction by storing recurrent-state blocks in the
same cache-management ecosystem as attention KV blocks. Conceptually, each
reusable prefix boundary needs:

1. **Prefix identity**
   Token IDs plus the existing cache identity inputs that affect hidden states,
   such as LoRA/task identity and model/runtime identity.

2. **Attention KV blocks**
   Existing block reuse keeps the KV blocks for full-attention layers.

3. **SSM recurrent state**
   The per-layer recurrent state after processing the prefix. In current Mamba
   cache terms, this corresponds to the `temporal` / SSM state tensors.

4. **Causal convolution window**
   The per-layer short history window needed by the next convolution update. For
   Mamba-like layers this is the last `d_conv - 1` inputs in the convolution
   state.

5. **Metadata**
   Prefix length, layer mask, state dtype, tensor-parallel partitioning, and
   enough ownership metadata to evict or copy the snapshot safely.

On a cache hit, the scheduler and cache manager should:

1. Find the longest reusable prefix using the existing prefix index.
2. Attach or restore the matching attention KV blocks.
3. Restore the matching SSM and convolution state.
4. Set the request's current context position to the matched prefix length.
5. Run prefill only for the unmatched suffix.

PR #12896 covers this through the unified manager and snapshot-boundary chunking
rather than a separate Python-side SSM cache. The remaining design work is to
document exactly where that path applies and where the system falls back.

## Integration Points and Gaps

### KV Cache Manager

PR #12896 chooses the preferred direction: keep the existing KV reuse machinery
as the source of truth and store recurrent-state snapshots as another cache
window. This avoids introducing a second prefix tree for SSM state.

Remaining gaps:

- Confirm eviction and reuse accounting for recurrent-state blocks under memory
  pressure.
- Document the relationship between attention block positions, recurrent-state
  placeholder blocks, and `mamba_state_cache_interval`.
- Validate whether the snapshot interval should be workload-tunable or model
  defaulted.

### Mamba Cache Manager

`CppMambaHybridCacheManager` is now the main implementation foundation. It
stores SSM and convolution state as views into the recurrent-state cache pool and
uses cache block offsets to select the state slot for each request.

Remaining gaps:

- Clarify when the unified manager is selected versus the mixed manager.
- Preserve a correctness fallback when the unified path is not supported.
- Add higher-level diagnostics for invalid or missing recurrent-state blocks.

### Scheduler

The scheduler must ensure that attention and SSM reuse agree on the same prefix
length. PR #12896 forces context chunks to stop at snapshot boundaries for hybrid
block reuse.

Remaining gaps:

- Quantify latency impact from additional forced chunk boundaries.
- Confirm behavior with overlap scheduling, chunked prefill, and admission
  accounting at high concurrency.
- Define fallback behavior when the longest KV match lacks a compatible SSM
  snapshot.

### Model Execution

Hybrid model forward paths already read `conv_states`, `ssm_states`, and
`state_indices` from the cache manager. A restored state should look like any
other live request state. This keeps model kernels largely unchanged and pushes
the feature into cache lifecycle code.

Remaining gaps:

- Add tests proving full-prefill and snapshot-resume outputs match across
  realistic model configurations.
- Document which speculative decoding variants use the unified path and which
  still fall back.

### Metrics

New metrics should distinguish KV-only reuse from full hybrid reuse:

- SSM snapshot entries and bytes.
- SSM snapshot hits, misses, and evictions.
- Prefix tokens skipped because both KV and SSM state were reusable.
- Fallback reasons, such as missing SSM snapshot or incompatible runtime key.

This is one of the largest remaining gaps. Without these counters, it is hard to
pitch the feature beyond correctness because we cannot easily quantify SSM
snapshot hit rate, memory overhead, or saved prefill work.

For the motivating agentic benchmark, the key metric should be cached-prefix
tokens skipped for hybrid layers. Reporting only KV hit rate is insufficient
because a hybrid model can hit attention KV blocks while still missing the SSM
snapshot needed to skip the same prefix.

## Major Gaps

### Default Enablement

Hybrid model defaults may still keep `enable_block_reuse=False`. The safe path
is to keep opt-in behavior until supported configurations and fallback reasons
are observable.

### Disaggregated Serving

The merged foundation is primarily a unified-pool single-instance path. Mixed
manager, cache transceiver, and cross-worker SSM state transfer remain major
gaps.

### KV Cache Connector and External Cache

External cache/offload flows need an SSM snapshot format and identity contract
before recurrent-state reuse can cross instance boundaries.

### Partial Reuse

PR #12896 disables partial reuse for hybrid Mamba paths. Exact snapshot-boundary
reuse is the right first milestone, but partial reuse may matter for workloads
where shared prefixes do not align with the snapshot interval.

### Observability

The system needs metrics and logs that separate:

- attention KV reuse,
- recurrent-state snapshot reuse,
- hybrid reuse fallback,
- forced chunking overhead,
- memory consumed by recurrent-state snapshots.

### Agentic Workload Evidence

The tests added by PR #12896 validate important correctness and MTP surfaces, but
the pitch still needs workload evidence for agent loops, branching, RAG, and
long shared-prefix serving.

The motivating benchmark requires a specific validation mode: multiple requests
or turns in the same trajectory reuse a previously processed prefix, with only a
small fresh suffix. This should be measured separately from MTP draft/verify
reuse so the project can prove it addresses the shared-prefix prompt scenario.

### Session-Aware Routing and Eviction

Agentic workloads often introduce delays between turns. Prefix reuse therefore
depends on the cache surviving scheduling gaps and on routing follow-up turns to
an instance that owns the relevant KV and SSM snapshots.

Remaining gaps:

- Confirm session-aware routing can preserve both KV and recurrent-state cache
  locality.
- Measure snapshot eviction under realistic inter-turn delays and high
  concurrency.
- Decide whether recurrent-state snapshots need a different eviction priority
  from attention KV blocks.

## Phased Plan

### Phase 0: Audit PR #12896 and Document Supported Surface

- Map the current `CppMambaHybridCacheManager` implementation against this
  design.
- Identify which model/config/runtime combinations use the unified manager.
- Document fallback paths where block reuse is still disabled or warned.
- Capture known limitations: disaggregated serving, mixed manager paths,
  connectors, partial reuse, and default enablement.

### Phase 1: Measurement and Validation

- Add tracing for repeated-prefix opportunities in hybrid models.
- Estimate SSM snapshot sizes for Qwen3-Next and Nemotron-H configurations.
- Compare full-prefill output against restore-from-snapshot output for exact
  prefix matches.
- Measure latency, throughput, cache hit rate, and memory overhead on
  shared-prefix agentic workloads.
- Measure inter-request reuse separately from intra-request MTP reuse.

### Phase 2: Production Hardening

- Add observability for recurrent-state snapshot hits, misses, bytes, and
  fallback reasons.
- Expand correctness tests for decode continuation, cache miss fallback,
  snapshot intervals, and high-concurrency eviction.
- Integrate with chunked prefill and overlap scheduling.
- Add model-specific validation for Qwen3-Next and Nemotron-H.
- Define behavior for LoRA, prompt tuning, multimodal inputs, and quantized SSM
  cache dtypes.

### Phase 3: Broader Serving Integration

- Evaluate disaggregated serving and cache transfer support.
- Consider host offload or connector support for SSM snapshots.
- Explore partial-block SSM snapshots if full-block reuse leaves meaningful
  performance on the table.

## Impact Assessment

The impact is highest for workloads with long shared prefixes and short
suffixes:

- Agentic tool loops.
- Branching or best-of-N inference.
- RAG over shared context.
- Multi-turn chat clients that resend full history each turn.

The feature is less valuable for short prompts, low prefix-hit workloads, or
requests that remain live and continue decoding without being reissued. Existing
request-local Mamba state already handles the live-request continuation case.

For speculative decoding, SSM prefix reuse is complementary. PR #12896
specifically targets hybrid MTP with block reuse enabled, so the foundation is
stronger than a greenfield proposal. Two-model speculation still cannot share
SSM snapshots between draft and target models unless their weights and state
layout are identical; each model needs its own cache identity.

For agentic workflows, the expected impact remains high because these workloads
often create new requests that resend long shared prefixes. The key missing
evidence is an agentic benchmark showing how often PR #12896's SSM snapshot path
hits and what prefill work it actually saves.

## Risks and Open Questions

- **Correctness:** The restored SSM state must match the same hidden history as
  the reused KV blocks. Any mismatch produces silent output drift.
- **Prefix boundary:** SSM state is naturally "after token N", while KV reuse is
  block-oriented. Snapshot intervals make this explicit but can introduce forced
  chunking overhead.
- **Memory pressure:** SSM snapshots are compact compared with KV history, but
  storing them at many boundaries and layers can still affect capacity.
- **State mutability:** After resume, SSM state is updated in place. Sharing
  snapshots directly would require copy-on-write or strict immutability.
- **Default enablement:** Hybrid model defaults need an explicit decision backed
  by metrics and compatibility tests.
- **Disaggregated serving:** Mixed manager, cache transceiver, and cross-worker
  SSM state transfer remain major gaps.
- **KV Cache Connector:** External cache/offload flows need an SSM snapshot
  format and identity contract before recurrent-state reuse can cross instance
  boundaries.
- **Feature interactions:** Chunked prefill, speculative decoding, LoRA,
  multimodal inputs, PP/TP/DP layouts, and disaggregated serving all need
  explicit compatibility rules.
- **Cache identity:** The key must include every input that changes hidden
  states. Reusing a snapshot across incompatible adapter, model, or runtime
  state would be incorrect.

## Pitch Recommendation

This is no longer a greenfield implementation proposal. PR #12896 delivered the
majority of the core implementation foundation. The project should be pitched as
the productionization and expansion of hybrid SSM block reuse.

The recommended pitch is a staged investigation:

1. Audit PR #12896 and document the supported runtime surface.
2. Measure hit-rate, latency, throughput, and memory tradeoffs on agentic
   shared-prefix workloads.
3. Add observability and compatibility tests for the major fallback paths.
4. Decide which gaps are required for default enablement: disaggregated serving,
   connector/offload, partial reuse, or workload-specific tuning.

If shared-prefix agentic workloads are important for Qwen3-Next or Nemotron-H,
the next step is not to prototype from scratch. It is to validate and harden the
merged foundation, then close the integration gaps that matter for the target
deployment.
