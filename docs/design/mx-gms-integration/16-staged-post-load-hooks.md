# 16. Staged Post-Load Hooks for MX / GMS Weight Sharing

**Status:** Historical migration plan tracked under [TRTLLM-11901](https://jirasw.nvidia.com/browse/TRTLLM-11901).
Core staging through PR #15288 and the fixture fix in PR #15471 are merged; remaining model-family/MX work is separate.
**Created:** 2026-05-19
**Last updated:** 2026-06-30

> [§18](18-dynamo-pr11000-gaps.md) is the current source for GMS loader blockers, reversible lifecycle,
> and follow-up PR ordering. This section remains the detailed rationale for staged post-load hooks.

## Reader Summary

TensorRT-LLM currently uses one hook, `post_load_weights()`, for several different jobs that do not always belong together:

- Structural alias wiring, such as `layer.next_attn = next_layer.self_attn`.
- Weight-content transforms, such as FP8 / NVFP4 packing, fused QKV layout, and MoE weight packing.
- Derived Python state recomputation from loaded tensors.
- Process-local finalization, such as MoE load-balancer and EP topology setup.

That works for ordinary disk loading because every job runs in one process after real tensors are present. It breaks down for MX and GMS weight sharing:

- **GMS read-only import** needs aliases before it can materialize the module tree, but it must not run tensor-derived work on meta tensors.
- **MX publish-after-transform** needs receivers to reuse already-transformed bytes, but receivers still need alias setup and derived-state refresh.
- **Both paths** need a compatibility gate so a receiver never consumes weights transformed for a different backend, quantization mode, dtype, or parallel layout.

The fix is to split `post_load_weights()` into staged hooks with narrow contracts:

| Stage | Purpose | Receiver of post-transform weights |
|:--|:--|:--|
| `setup_aliases()` | Structural Python references only. No tensor reads or writes. | Always run |
| `transform_weights()` | One-shot tensor layout/content transforms, guarded by `_weights_transformed`. | Skip |
| `cache_derived_state()` | Recompute Python-side state from the currently-bound real tensors. | Run after weights arrive |
| Orchestrator-managed finalization | Process-local state, for example `MoeLoadBalancer.finalize_model()`. | Always run |

The rollout is intentionally staged. Existing full-load behavior remains available through the backward-compatible `post_load_weights()` orchestrator while modules migrate.

## Current Status - 2026-06-15

This status snapshot was refreshed from the GitHub PR metadata and commit status checks on 2026-06-15.

### Landed Foundation

| PR | Status | What it provides |
|:--|:--|:--|
| [PR 14770](https://github.com/NVIDIA/TensorRT-LLM/pull/14770) / [TRTLLM-13077](https://jirasw.nvidia.com/browse/TRTLLM-13077) | Merged on 2026-06-03 | Adds default staged-hook methods, `ModelLoader` walkers, and `_weights_transformed` lifecycle support. No model behavior migrated. |
| [PR 14878](https://github.com/NVIDIA/TensorRT-LLM/pull/14878) / [TRTLLM-13141](https://jirasw.nvidia.com/browse/TRTLLM-13141) | Merged on 2026-06-12 | Adds `SourceIdentity`, source compatibility policies, MX pre-transfer gate, and GMS strict pre-materialize gate plumbing. |

### In-Flight PRs

| Wave | PR | Current state | Scope | Notes |
|:--|:--|:--|:--|:--|
| Wave 1 | [PR 15014](https://github.com/NVIDIA/TensorRT-LLM/pull/15014) / [TRTLLM-13246](https://jirasw.nvidia.com/browse/TRTLLM-13246) | Jira: In Progress. Open, non-draft, mergeable, and approved by Funatiq on 2026-06-15. Current head is `6ba212d`; `blossom-ci` is pending in PR_Github #54327. | Moves alias-only model hooks to `setup_aliases()` and cuts GMS RO over to `setup_aliases()` -> SourceIdentity check -> materialize -> `cache_derived_state()`. | Previous full L0 on head `4352612` passed in PR_Github #53930 / L0 #43023 on 2026-06-12. The pending rerun validates final review-fix churn on the new head. |
| Wave 2 | [PR 15288](https://github.com/NVIDIA/TensorRT-LLM/pull/15288) / [TRTLLM-13247](https://jirasw.nvidia.com/browse/TRTLLM-13247) | Jira: To Do. Open draft, mergeable, and stacked on [PR 15014](https://github.com/NVIDIA/TensorRT-LLM/pull/15014). Current head is `bfebf3a`; `blossom-ci` is pending in PR_Github #54338. | Moves Linear and Attention/MLA tensor-layout work into `transform_weights()` with `_weights_transformed` guards. | Older full CI attempts on head `67267df` failed, while targeted B200 coverage in PR_Github #53990 passed on 2026-06-13. The pending run validates the latest stack move. |
| MX delegation | [PR 14151](https://github.com/NVIDIA/TensorRT-LLM/pull/14151) | Open, not draft, currently not mergeable. Current head is `088aa36`; no combined commit status is posted for that head. | Delegates low-level ModelExpress checkpoint loading to the `modelexpress` package. | Adds end-to-end ModelExpress + TRT-LLM validation notes for Kimi-K2.5-NVFP4. This affects where MX carries source identity and publish-after-transform metadata, but the staged-hook design is agnostic to that ownership split. |

### Wave Jira Tasks

| Wave | Jira | Jira state | Scope |
|:--|:--|:--|:--|
| Wave 1 | [TRTLLM-13246](https://jirasw.nvidia.com/browse/TRTLLM-13246) | In Progress | Alias migration + GMS RO cutover |
| Wave 2 | [TRTLLM-13247](https://jirasw.nvidia.com/browse/TRTLLM-13247) | To Do | Linear / Attention transform migration |
| Wave 3 | [TRTLLM-13248](https://jirasw.nvidia.com/browse/TRTLLM-13248) | To Do | MoE + Mamba transform migration |
| Wave 4 | [TRTLLM-13249](https://jirasw.nvidia.com/browse/TRTLLM-13249) | To Do | TRT-LLM-side MX receiver cutover + per-model allow-list framework |
| Wave 5 | [TRTLLM-13250](https://jirasw.nvidia.com/browse/TRTLLM-13250) | To Do | MX publisher flip + first model in allow-list |

### What Is Still Not Done

- Until Wave 1 lands, GMS RO still relies on the old meta-tensor workaround on `main`.
- Wave 2 starts the transform migration, but MX receivers still do not skip transforms at runtime.
- MoE, Mamba, min-latency Llama, and sparse-attention transform migrations remain Wave 3 work.
- The MX receiver cutover and per-model allow-list are still future work.
- The MX publisher flip to publish post-transform bytes is still future work and should happen only after receiver-side safeguards and allow-listing are in place.

## Problem Statement

### Why The Current Hook Is Too Coarse

`ModelLoader` currently walks the model and calls `post_load_weights()` after weights are loaded. Across the codebase, that hook means different things:

| Category | Examples | Safe to rerun? | Needed by post-transform receivers? |
|:--|:--|:--|:--|
| A. Structural alias wiring | `next_attn`, shared embedding refs, fused-module refs | Yes | Yes |
| B. Weight-data transforms | FP8/NVFP4 conversion, QKV fusion, MoE packing | No | No |
| C. Process-local setup | MoE routing tables, EP topology bookkeeping | Yes, orchestrator-owned | Yes |
| D. Derived Python state | cached scale/state, dtype-validation booleans, fingerprints | Yes, if recomputed on real tensors | Yes |

The problem is not that any one use is wrong. The problem is that consumers cannot ask for only A or only D. They get A + B + D as a bundle.

### Failure Mode 1: GMS RO Alias Ordering

GMS read-only import materializes storage by walking a module-keyed catalog. Some catalog paths require structural aliases such as `model.layers[i].next_attn` to exist before materialization.

The existing workaround runs `post_load_weights()` before materialization so alias paths exist. That avoids an `AttributeError`, but it also runs transform and derived-state code while tensors are still meta tensors. Any module that reads tensor data for cached scales, validation booleans, or fingerprints can produce NaN, zero, or otherwise divergent state on the RO peer.

Wave 1 fixes the order:

```text
GMS RO reader:
  setup_aliases()
  check SourceIdentity under STRICT policy
  materialize_module()
  cache_derived_state()
```

That preserves the alias requirement while moving data-dependent work after real CUDA storage is bound.

### Failure Mode 2: MX Publish-After-Transform

The merged MX path currently publishes **pre-transform** bytes. That is safe because each receiver runs the normal full `post_load_weights()` path. It is also wasteful because every receiver repeats the same transform work.

A future publish-after-transform path can avoid that duplicated work, but only if receivers skip `transform_weights()` while still running `setup_aliases()` and `cache_derived_state()`. The monolithic hook cannot express that.

The staged hooks make the desired receiver path explicit:

```text
MX receiver with compatible post-transform source:
  setup_aliases()
  skip transform_weights()
  cache_derived_state()
```

This path is only safe when SourceIdentity proves that source and receiver agree on every layout-affecting choice.

## Goals And Non-Goals

### Goals

- Make GMS RO materialization correct without running tensor-derived work on meta tensors.
- Make MX publish-after-transform possible without double-transforming receiver weights.
- Keep normal disk/HF loading behavior backward compatible during migration.
- Keep process-local finalization orchestrator-owned.
- Gate all transformed-weight reuse with SourceIdentity.
- Roll out runtime behavior per model so risky transform-skip behavior is opt-in and revertable.

### Non-Goals

- Do not redesign MX or GMS transport internals.
- Do not change the legacy TensorRT backend.
- Do not make MX publish-after-transform default for every model at once.
- Do not move MoE process-local topology finalization into a per-module hook.

## Proposed Protocol

Each participating module can implement any subset of these methods:

```python
def setup_aliases(self) -> None:
    """Wire structural Python references only. No tensor reads or writes."""

def transform_weights(self) -> None:
    """Apply one-shot tensor layout/content transforms."""
    if getattr(self, "_weights_transformed", False):
        return
    ...
    self._weights_transformed = True

def cache_derived_state(self) -> None:
    """Recompute Python-side derived state from currently-bound real tensors."""

def post_load_weights(self) -> None:
    """Backward-compatible full-load path."""
    self.setup_aliases()
    self.transform_weights()
    self.cache_derived_state()
```

`ModelLoader` owns the walks:

- `_setup_aliases(model)` invokes structural alias setup.
- `_walk_transform(model)` walks modules for `transform_weights()`.
- `_walk_cache_state(model)` walks modules for `cache_derived_state()`.
- `_walk_full_post_load(model)` preserves the existing `post_load_weights()` behavior.
- `_reset_weights_transformed(...)` clears transform guards only when fresh untransformed bytes are rebound.

Wave 1 currently uses a recursive alias walk, matching the implementation in [PR 15014](https://github.com/NVIDIA/TensorRT-LLM/pull/15014).

## Per-Path Behavior

### Normal Disk/HF Load

No behavior change:

```text
load raw weights
apply weights to modules
walk full post_load_weights() path
run orchestrator-owned finalization
```

### GMS RW Writer

The writer owns real tensors and publishes the final post-transform layout:

```text
enter GMS memory pool
load or receive weights
apply weights
walk full post_load_weights() path
finalize_write(model)
```

### GMS RO Reader

The reader imports already-published storage:

```text
setup_aliases()
check SourceIdentity under STRICT policy
materialize_module()
cache_derived_state()
post_load_publish / process-local follow-up as needed
```

The RO reader never runs `transform_weights()` because the RW writer already baked the transform into the GMS pool.

### MX Receiver Today

The current MX path publishes pre-transform bytes, so receivers still do the full normal path:

```text
receive raw/pre-transform bytes
apply weights
walk full post_load_weights() path
```

### MX Receiver Target State

After Waves 2-5, a compatible allow-listed receiver can consume post-transform bytes:

```text
fetch source identity
if compatible and model is allow-listed:
    setup_aliases()
    skip transform_weights()
    cache_derived_state()
else:
    fall back to normal full-load path
```

## Hard Preconditions

### P1. SourceIdentity Must Cover Layout-Affecting Choices

Skipping `transform_weights()` is safe only when source and receiver agree on every choice that affects transformed weight layout. SourceIdentity must cover at least:

- Model architecture and model revision.
- Weight dtype.
- Quantization config and quant backend lists.
- FP8 / NVFP4 scale and packing strategy.
- Attention backend (`TRTLLM`, `FlashInfer`, `FlashAttention`) and any backend-specific fusion.
- TP / PP / EP / CP sizes and this rank's shard identity.
- Any future quant scheme, fusion pass, or layout-affecting knob.

If identity is missing or mismatched:

- MX should fall back to the full disk/load path.
- GMS RO should fail closed because it has no disk fallback in that branch.

### P2. `_weights_transformed` Reset Is Orchestrator-Owned

Subclasses set `_weights_transformed = True` only after a successful transform. They do not reset it.

Reset belongs to the orchestrator that introduces fresh untransformed bytes, for example full reload or a partial fallback that rebinds specific tensors. Resetting too much can cause unnecessary re-transform; resetting too little can skip a required transform.

## Migration Inventory

| Bucket | Files / modules | Target stage |
|:--|:--|:--|
| Alias-only model hooks | `modeling_llama.py`, `modeling_deepseekv3.py`, `modeling_glm.py`, `modeling_exaone_moe.py`, `modeling_qwen3_moe.py`, `modeling_qwen3_next.py`, `modeling_gpt_oss.py` | `setup_aliases()` |
| Dense transforms | `modules/linear.py`, `modules/attention.py` / MLA | `transform_weights()` |
| MoE transforms | `fused_moe/quantization.py`, `fused_moe_triton.py`, `configurable_moe.py` | `transform_weights()` |
| Other transform paths | `mamba2_mixer.py`, `modeling_llama_min_latency.py`, `attention_backend/sparse/dsa.py` | `transform_weights()` or split if mixed |
| Derived state | Any module that reads real tensor content to cache Python-side state | `cache_derived_state()` |
| Process-local finalization | MoE load-balancer, EP topology setup | Keep in `ModelLoader` / orchestrator |

Migration rule: when a class moves logic into staged hooks, delete its old `post_load_weights()` override unless it is intentionally a compatibility shim. A stale override can shadow the base orchestrator and silently prevent staged methods from running.

## Rollout Plan

| Phase | Jira | Vehicle | Status | Risk | Runtime behavior change |
|:--|:--|:--|:--|:--|:--|
| Prep | [TRTLLM-13077](https://jirasw.nvidia.com/browse/TRTLLM-13077) | [PR 14770](https://github.com/NVIDIA/TensorRT-LLM/pull/14770) | Merged | Low | None; adds contract surface and walkers |
| Source identity | [TRTLLM-13141](https://jirasw.nvidia.com/browse/TRTLLM-13141) | [PR 14878](https://github.com/NVIDIA/TensorRT-LLM/pull/14878) | Merged | Medium | Adds compatibility gates; existing paths remain safe/fallback-oriented |
| Wave 1 | [TRTLLM-13246](https://jirasw.nvidia.com/browse/TRTLLM-13246) | [PR 15014](https://github.com/NVIDIA/TensorRT-LLM/pull/15014) | Jira: In Progress; PR approved; latest CI pending | Low | Fixes GMS RO ordering and migrates alias-only model hooks |
| Wave 2 | [TRTLLM-13247](https://jirasw.nvidia.com/browse/TRTLLM-13247) | [PR 15288](https://github.com/NVIDIA/TensorRT-LLM/pull/15288) | Jira: To Do; PR draft; latest CI pending | High | Migrates Linear and Attention/MLA transforms; no MX transform-skip yet |
| Wave 3 | [TRTLLM-13248](https://jirasw.nvidia.com/browse/TRTLLM-13248) | TBD | Jira: To Do; planned | High | Migrates MoE, Mamba, min-latency, and sparse transform paths |
| Wave 4 | [TRTLLM-13249](https://jirasw.nvidia.com/browse/TRTLLM-13249) | TBD | Jira: To Do; planned | Low | Adds MX receiver cutover and empty per-model allow-list; no model enabled yet |
| Wave 5 | [TRTLLM-13250](https://jirasw.nvidia.com/browse/TRTLLM-13250) | TBD | Jira: To Do; planned | Medium | MX publisher emits post-transform bytes and first model enters allow-list |

### Wave 1: Alias Migration + GMS RO Cutover ([TRTLLM-13246](https://jirasw.nvidia.com/browse/TRTLLM-13246))

Wave 1 is the first end-to-end correctness slice:

- Move alias-only model hooks into `setup_aliases()`.
- Replace the GMS RO meta-tensor workaround with staged ordering.
- Add/adjust unit tests for GMS RO sequencing and staged walkers.

Gate to merge:

- GMS RO unit tests validate `setup_aliases()` -> materialize -> `cache_derived_state()` ordering.
- A migrated model proves alias setup is idempotent.
- CI is green for the affected model families.

### Wave 2: Linear And Attention Transform Migration ([TRTLLM-13247](https://jirasw.nvidia.com/browse/TRTLLM-13247))

Wave 2 moves dense transform logic into `transform_weights()`:

- `Linear.transform_weights()` owns FP8/NVFP4 and quant-method driven layout work.
- Attention/MLA transform logic gets `_weights_transformed` guards.
- Unit tests cover idempotency and reset behavior.

This has wide blast radius because almost every model uses Linear and Attention modules. It makes dense models eligible for future publish-after-transform, but runtime MX still uses the full path until Wave 4/5.

### Wave 3: MoE, Mamba, And Remaining Transform Paths ([TRTLLM-13248](https://jirasw.nvidia.com/browse/TRTLLM-13248))

Wave 3 migrates the high-risk remaining transform bodies:

- MoE quant variants and Triton/configurable MoE paths.
- Mamba mixer post-load transforms.
- Min-latency Llama and sparse-attention transform paths.

The main review risk is keeping Category B transforms separate from process-local Category C finalization.

### Wave 4: MX Receiver Cutover Infrastructure ([TRTLLM-13249](https://jirasw.nvidia.com/browse/TRTLLM-13249))

Wave 4 adds the receiver-side branch without enabling any real model:

- If source identity matches and the model is allow-listed, run the staged receiver path.
- Otherwise, fall back to the existing full-load path.
- Ship the allow-list empty so production behavior remains unchanged.

This keeps the risky path testable but dormant.

### Wave 5: First Publish-After-Transform Model ([TRTLLM-13250](https://jirasw.nvidia.com/browse/TRTLLM-13250))

Wave 5 is the first runtime speedup:

- MX publisher carries source identity and publishes post-transform bytes.
- One production model, likely a Llama-family model, is added to the allow-list.
- Integration tests compare `cache_derived_state()` output against disk-loaded reference and verify mismatch fail/fallback behavior.

Additional models roll out through small allow-list PRs after bit-equivalence validation.

## Coordination With MX And GMS

### MX

MX needs to carry enough metadata for the receiver to know whether incoming bytes are raw or post-transform and whether they are layout-compatible.

Open coordination point: [PR 14151](https://github.com/NVIDIA/TensorRT-LLM/pull/14151) proposes delegating source discovery, identity construction, RDMA transfer, fallback behavior, source publication, and metadata lifecycle to ModelExpress. The staged-hook design works either way:

- If ModelExpress owns the full checkpoint-loading flow, TensorRT-LLM should provide a SourceIdentity provider/checker that ModelExpress calls.
- If TensorRT-LLM keeps orchestration and ModelExpress stays transport-focused, TensorRT-LLM calls SourceIdentity directly and passes opaque bytes through MX metadata.

The safety requirement is the same in both models: no receiver skips `transform_weights()` without a compatible SourceIdentity and an allow-listed model.

### GMS

GMS needs no protocol change for Wave 1. The RO reader still materializes from the published catalog, but TensorRT-LLM changes the local order:

```text
setup aliases before catalog walk
materialize storage
refresh derived state from real tensors
```

SourceIdentity remains important for fail-closed protection before RO materialization.

## Testing Expectations

Minimum coverage by phase:

- Wave 1: GMS RW/RO unit tests for alias/materialize/cache ordering; affected model alias tests; regression test for skipped `_weights_removed` modules if relevant.
- Wave 2: Linear/Attention transform idempotency tests; reset tests for full reload and partial fallback; representative FP8/NVFP4/BF16/INT8 integration coverage.
- Wave 3: MoE quant-method idempotency tests; MoE and Mamba integration coverage; manual sweep for stale `post_load_weights()` overrides.
- Wave 4: Synthetic allow-list test proving staged receiver path and fallback path.
- Wave 5: End-to-end MX publish-after-transform test for the first allow-listed model; SourceIdentity mismatch test; bit-equivalence of receiver-derived state.

Longer-term, add one MX + GMS integration regression that covers:

- Compatible source -> alias setup + materialize/receive + cache-derived-state.
- MX incompatible source -> fallback.
- GMS incompatible source -> strict failure.

## Open Questions

| Question | Current default |
|:--|:--|
| Where should the protocol live long term? | Duck-typed `ModelLoader` walkers, with optional mixin/type helpers only if useful. |
| Should quant-method callbacks be renamed from `post_load_weights(module)`? | Keep callback names unless renaming materially improves the Wave 2/3 implementation. The no-arg module hook and `(self, module)` quant callback are distinguishable. |
| How should the per-model allow-list be stored? | Key by `(model_class, transform_protocol_version)`; each model addition is a small, revertable PR with integration proof. |
| How do we enforce SourceIdentity completeness as new layout knobs appear? | Add lint/registration/test coverage so new transform-affecting config fields cannot silently bypass the fingerprint. |
| How expensive is `cache_derived_state()` on very large models? | Expected to be cheap because most state is scalar/dtype metadata; profile during Wave 1 and first allow-listed MX cutover. |

## References

- [TRTLLM-11901](https://jirasw.nvidia.com/browse/TRTLLM-11901) - Parent Jira epic for ModelExpress and GPU Memory Service integration.
- [PR 13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531) / [TRTLLM-11851](https://jirasw.nvidia.com/browse/TRTLLM-11851) - MX-only P2P checkpoint loading support, merged. Establishes current publish-pre-transform behavior.
- [PR 13926](https://github.com/NVIDIA/TensorRT-LLM/pull/13926) / [TRTLLM-12440](https://jirasw.nvidia.com/browse/TRTLLM-12440) - GMS-only weight sharing support, merged. Introduces the GMS RO workaround Wave 1 replaces.
- [PR 13045](https://github.com/NVIDIA/TensorRT-LLM/pull/13045) - End-to-end MX + GMS prototype.
- [PR 14770](https://github.com/NVIDIA/TensorRT-LLM/pull/14770) / [TRTLLM-13077](https://jirasw.nvidia.com/browse/TRTLLM-13077) - Staged-hook prep, merged.
- [PR 14878](https://github.com/NVIDIA/TensorRT-LLM/pull/14878) / [TRTLLM-13141](https://jirasw.nvidia.com/browse/TRTLLM-13141) - SourceIdentity gate, merged.
- [PR 15014](https://github.com/NVIDIA/TensorRT-LLM/pull/15014) / [TRTLLM-13246](https://jirasw.nvidia.com/browse/TRTLLM-13246) - Wave 1, approved and waiting on latest CI for head `6ba212d`.
- [PR 15288](https://github.com/NVIDIA/TensorRT-LLM/pull/15288) / [TRTLLM-13247](https://jirasw.nvidia.com/browse/TRTLLM-13247) - Wave 2, draft and waiting on latest CI for head `bfebf3a`.
- [TRTLLM-13248](https://jirasw.nvidia.com/browse/TRTLLM-13248) - Wave 3 planned task for MoE + Mamba transform migration.
- [TRTLLM-13249](https://jirasw.nvidia.com/browse/TRTLLM-13249) - Wave 4 planned task for TRT-LLM-side MX receiver cutover and per-model allow-list framework.
- [TRTLLM-13250](https://jirasw.nvidia.com/browse/TRTLLM-13250) - Wave 5 planned task for MX publisher flip and first model allow-list entry.
- [PR 14151](https://github.com/NVIDIA/TensorRT-LLM/pull/14151) - ModelExpress delegation proposal, open and currently not mergeable.
- [ai-dynamo/dynamo PR 7053](https://github.com/ai-dynamo/dynamo/pull/7053) - Upstream GMS prototype that surfaced the alias-resolution ordering issue.
- [05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) - Earlier write-up of the GMS module-path resolution problem.
- [12-risks.md](12-risks.md) - Risk row for module path resolution and aliased layers.
