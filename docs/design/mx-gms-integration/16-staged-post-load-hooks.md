# 16. Staged Post-Load Hooks (Holistic MX + GMS Fix)

**Status:** Locked (2026-05-31) — prep PR awaiting review as TRTLLM-13077 (`[TRTLLM-13077][feat] Decompose post_load_weights()`); Wave 1 begins once it merges. Full migration sequenced as Waves 1–4 below.
**Created:** 2026-05-19
**Last updated:** 2026-05-31
**Drives:**
- TRTLLM-12440 — `[TRTLLM-12440][feat] Add GMS-only weight sharing support` (merged): review feedback on RO post-load ordering. The §7 ad-hoc mitigation (run full `post_load_weights()` on meta tensors before `materialize_module()`) is the workaround the staged-hook protocol replaces.
- TRTLLM-11851 — `[TRTLLM-11851][feat] Add MX-only P2P checkpoint loading support for TRTLLM` (merged): establishes the current MX behavior (publish-PRE-transform; receivers run full `post_load_weights()` on disk-loaded bytes). Wave 4 below changes this contract.
- The inflight MX-team `[None][refactor] Delegate MX checkpoint loading to ModelExpress` refactor proposal: publish-pre vs publish-post-transform discussion, receivers selectively skipping `module.post_load_weights()`, and the homogeneity-assumption hazard surfaced on the review thread.

---

## TL;DR

TensorRT-LLM's `module.post_load_weights()` hook does four categorically different things in one call: structural alias wiring, weight-data transforms (FP8 / NVFP4 / fused-QKV / fused-MoE), per-process state setup, and recomputation of weight-derived Python state. The recently merged GMS integration (TRTLLM-12440) and any publish-post-transform MX receiver path each need to override **some but not all** of those operations. Because today's hook is monolithic, the GMS integration shipped an ad-hoc workaround that leaves a correctness bug (derived state recomputed on meta tensors instead of real tensors), and any publish-post-transform MX path runs into a symmetric hazard (transforms re-applied onto already-transformed bytes).

The plan is to decompose `post_load_weights()` into three explicit per-module stages with clear contracts, keep the existing per-process finalization where it is, and migrate the in-tree overrides in four sequenced waves. The protocol is non-breaking; a back-compat orchestrator preserves today's behavior for any code path that hasn't migrated.

| Stage | Idempotent? | Receiver of post-transform weights needs it? |
|:------|:------------|:---------------------------------------------|
| `setup_aliases()` | Yes (by `nn.Module.__setattr__` dedupe) | **Always** — module-tree resolution depends on it |
| `transform_weights()` | One-shot (gated by `_weights_transformed`) | **Skip** — bytes already transformed |
| `cache_derived_state()` | Yes (by recomputation on real tensors) | **Recompute** after weights arrive |
| `MoeLoadBalancer.finalize_model()` + EP topology setup (orchestrator-managed, unchanged) | Yes | **Always** (process-local) |

---

## Foundation references

### Merged foundation work

These tickets established the current integration shape; the decomposition proposed here is the structural fix that lets their residual workarounds be removed cleanly.

- **TRTLLM-11851** — `[TRTLLM-11851][feat] Add MX-only P2P checkpoint loading support for TRTLLM` (merged). Establishes the current MX behavior on the receiver side: publish-PRE-transform, with the receiver running the full `post_load_weights()` on disk-loaded bytes. Wave 4 below changes this contract.
- **TRTLLM-12440** — `[TRTLLM-12440][feat] Add GMS-only weight sharing support` (merged). Establishes the current GMS RW / RO contract, including the meta-tensor workaround on the RO branch (see [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific)) that Wave 1 replaces with the staged-hook protocol.
- **End-to-end MX + GMS integration prototype** (open). Demonstrates the integrated MX + GMS data path and surfaces the shared architectural gap that this document addresses.

### Inflight

- **TRTLLM-13077** — `[TRTLLM-13077][feat] Decompose post_load_weights()` (awaiting review). The prep PR for this design: introduces the contract surface (default no-op `setup_aliases()` / `transform_weights()` / `cache_derived_state()`, the `_weights_transformed` lifecycle flag, helper walkers on `ModelLoader`) without migrating any model. Wave 1 begins once this merges.
- **MX-team Delegate-to-ModelExpress refactor proposal** (`[None][refactor] Delegate MX checkpoint loading to ModelExpress`, under review). Surfaces the publish-pre vs publish-post-transform discussion and the homogeneity-assumption hazard from the MX side. This document is largely agnostic to whether that refactor ships as proposed; the scope-of-delegation question is discussed under "Coordination with MX and GMS" below.

---

## Audience and intent

This document is intended for two readers:

1. **MX (ModelExpress) and GMS (GPU Memory Service) upstream engineers.** You are the source of the weight-sharing protocols TRT-LLM is integrating. You don't need to know every TRT-LLM module path, but you do need to know (a) what TRT-LLM is asking your protocol to express on the publisher side (notably the backend fingerprint that **P1** below requires from MX), and (b) why the receive-side ordering matters for GMS RO catalog resolution.
2. **TensorRT-LLM code owners** in the disagg-serving and PyExecutor areas, who will review the migration PRs and own the resulting code surface.

---

## Background

### What `post_load_weights()` is today

After TRT-LLM's `ModelLoader` finishes binding parameters to GPU memory and copying weight bytes into them, it walks every `nn.Module` in the model tree and calls a no-arg `post_load_weights(self)` on those that override it. The hook is the per-module "finalize me" step. In practice, different module classes use it for very different purposes:

- `LlamaForCausalLM.post_load_weights()` (and six other top-level model classes) only wires structural cross-references — for example, `layer.next_attn = self.model.layers[idx + 1].self_attn` so the FlashInfer / TRTLLM attention backends can reach the next layer's projections during decode. No tensor math.
- `Linear.post_load_weights()` performs the FP8 weight conversion, packs scales, and dispatches to the active `QuantMethod.post_load_weights(self, module)` callback for quantization-specific layout changes. Heavy tensor math.
- `Attention.post_load_weights()` fuses Q/K/V projections, packs RoPE scales, and rebuilds attention-mask buffers. Tensor math, dependent on which attention backend (`TRTLLM`, `FlashInfer`, `FlashAttention`) is selected.
- `FusedMoE.post_load_weights()` (six quant variants in `fused_moe/quantization.py` plus several in `fused_moe_triton.py`, `configurable_moe.py`) packs expert weights according to the EP topology. Heavy.
- `Mamba2Mixer.post_load_weights()` recomputes the SSM A-matrix discretization. Tensor math.

There are also modules whose `post_load_weights()` is a mix of structural and tensor work, or empty / pass-through. The point is that one method name carries four distinct semantic intents:

| # | Category | Examples | Idempotent? | Receiver of post-transform weights needs it? |
|:--|:---------|:---------|:------------|:---------------------------------------------|
| **A** | **Structural alias wiring** | `layer.next_attn = self.model.layers[idx + 1].self_attn`; shared embedding refs; fused-module references | Yes, naturally — `nn.Module.__setattr__` dedupes by object identity in `_modules` | **Always** — module-tree resolution depends on it |
| **B** | **Weight-data transforms** | Fused QKV, quantization scales, FP8 weight conversion, MoE shared-weight loading | No, one-shot and irreversible | **Skip** — bytes already transformed |
| **C** | **Per-process state setup** | MoE routing tables, EP topology bookkeeping, expert weight slot registration | Yes, but must always run | **Always** — process-local state cannot be transferred |
| **D** | **Derived Python-side state** | `self._scale_cache = self.weight.std()`; dtype-validation booleans; cached fingerprints | Yes, by recomputation | **Recompute** on real tensors after weights arrive |

Category C is already orchestrator-managed today: `MoeLoadBalancer.finalize_model()` is invoked from `ModelLoader.load`, not from a per-module hook. That's the right shape and stays as-is. The decomposition proposed here only touches A, B, and D.

### Why this matters for MX and GMS integration

A naive integration of either weight-sharing technology calls `post_load_weights()` on the receiver side after the bytes arrive. That naively does all of A + B + D. Both integrations need a subset:

- **MX P2P receiver** under any publish-after-transform path needs A and D, but must **skip B** — the sender already applied the transforms; re-running them would double-transform (e.g., FP8-convert an already-FP8 buffer, fuse already-fused QKV). The currently-merged MX path (TRTLLM-11851) sidesteps this by publishing PRE-transform; the price is that every receiver redoes the transform work the publisher already did.
- **GMS RO reader** needs A *before* `materialize_module_from_gms()` walks the module-keyed catalog, and needs D *after* the catalog walk binds real CUDA storage. Running the monolithic `post_load_weights()` at either point is wrong:
  - **Pre-materialize** (today's workaround in TRTLLM-12440) runs B and D against meta tensors, silently producing NaN / zero scales and divergent dtype-validation booleans on the RO peer.
  - **Post-materialize** crashes on alias resolution because the catalog walk encounters paths like `model.layers[i].next_attn` that haven't been wired yet.

The two failure modes have a shared root cause: A, B, D travel together inside one method name with no way for a consumer to ask for one without the others.

---

## The two concrete failure modes today

### Failure 1: the `LlamaForCausalLM.next_attn` AttributeError (GMS RO catalog walk)

GMS's read-only peer materializes weight storage by walking a module-keyed catalog published by the read-write peer. The walk traverses qualified attribute paths like `model.layers[3].next_attn.q_proj.weight`. In TRT-LLM, `next_attn` is a cross-layer reference assigned by `LlamaForCausalLM.post_load_weights()`:

```python
class LlamaForCausalLM(...):
    def post_load_weights(self):
        # Wire cross-references that the attention backend uses during decode.
        for idx, layer in enumerate(self.model.layers[:-1]):
            layer.next_attn = self.model.layers[idx + 1].self_attn
            layer.next_layer_layernorm = ...
        # (no tensor work in this override)
```

If the catalog walk runs *before* this method, `model.layers[3].next_attn` does not exist on `model.layers[3]` and the walk raises `AttributeError`. The upstream ai-dynamo prototype ([ai-dynamo/dynamo PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053)) discovered this and fixed it with the contract "call `model.post_load_weights()` (top-level only) before `materialize_module_from_gms()`."

That contract works for `LlamaForCausalLM` specifically because *its* `post_load_weights()` happens to be only structural Python — no tensor work, no meta-tensor concern. But the same contract applied generically to all modules' `post_load_weights()` (which is what TRT-LLM's GMS-RO path currently does — see [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) for the in-tree mitigation write-up) drags in category B+D, which is the source of the second residual bug (cached scales becoming NaN/0 on the RO peer because they were recomputed against meta tensors).

Decomposing the hook so that `setup_aliases()` is a separate, top-level-only, structural-only stage cleanly removes the contradiction. The GMS RO peer calls `setup_aliases()` → `materialize_module_from_gms()` → per-module `cache_derived_state()`, and never touches `transform_weights()` because the RW peer already did.

### Failure 2: the homogeneity hazard for any MX publish-post-transform receiver

The currently-merged MX integration (TRTLLM-11851) takes the safe path: publish-PRE-transform. The publisher writes raw disk-loaded bytes to the receiver, and the receiver runs the full `post_load_weights()` on those bytes, just like a regular disk load. This is correct but expensive: every receiver re-does all the FP8 conversions, QKV fusion, MoE expert packing, etc., even though it's CPU-bound, GPU-stream-bound, and identical work to what the publisher just did.

A publish-POST-transform path — whether shipped by the MX-team Delegate-to-ModelExpress refactor or by some other future MX evolution — would have the publisher send already-transformed bytes and the receiver skip its own transforms. This is a 1–10× speedup depending on model. But naively skipping the receiver-side `post_load_weights()` skips A and D as well, which is wrong. And skipping only "the transform parts" of `post_load_weights()` is not expressible in the current API — there is no per-stage selector.

Worse, a receiver that blindly accepts pre-transformed bytes inherits the publisher's backend choice silently. If the publisher used `FlashInfer` and the receiver was configured for `TRTLLM`, the receiver gets weights laid out for `FlashInfer` (different fused-QKV permutation, different attention-mask packing) but runs `TRTLLM` kernels on them. This is the "homogeneity-assumption hazard" flagged on the MX-team refactor's review thread.

Decomposing the hook lets a publish-post-transform receiver call `setup_aliases()` and `cache_derived_state()` while explicitly skipping `transform_weights()`. The homogeneity hazard is addressed orthogonally by precondition **P1** below — a backend-identity check that the receiver runs before allowing the skip.

---

## The proposed staged-hook protocol

Decompose the single `post_load_weights()` into three per-module stages plus orchestrator-managed per-process finalization (category C). Each per-module stage has a default no-op body and a clear contract:

```python
class DecoderModelForCausalLM(nn.Module):  # and applicable submodule base classes
    def setup_aliases(self) -> None:
        """Wire structural Python references only — no tensor operations.

        Idempotent by ``nn.Module.__setattr__`` semantics: re-assigning the
        same module reference is a no-op. May be called any number of times.
        """

    def transform_weights(self) -> None:
        """Apply weight-content transforms (fused QKV, quantization, FP8 conversion).

        ONE-SHOT: gated by ``self._weights_transformed`` flag. Receivers of
        post-transform weights (MX P2P, GMS RO) MUST skip this stage.
        """
        if getattr(self, '_weights_transformed', False):
            return
        # subclass-specific transform logic
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        """Recompute Python-side state from current CUDA weight tensors.

        Reserved for data-dependent Python-side state where it exists —
        cached scalars, fingerprints, dtype-validation booleans computed
        from weight content. Many existing modules will have an empty
        ``cache_derived_state()`` after migration; the stage exists so
        consumers like GMS RO have a correctness path on the modules
        that DO carry such state, without re-running ``transform_weights``.
        Idempotent by recomputation. Always safe to call after any source
        of weight bytes (disk, MX P2P, GMS materialize). Runs on real
        tensors only.
        """

    # finalize_per_process_state stays orchestrator-managed (MoeLoadBalancer,
    # EP topology setup, etc.) — no per-module method needed today.

    # Backward-compat orchestrator. Existing callers see no change.
    def post_load_weights(self) -> None:
        self.setup_aliases()
        self.transform_weights()
        self.cache_derived_state()
```

The default `post_load_weights()` orchestrator preserves today's behavior 1:1 for any code path that hasn't migrated, so the protocol can land as a non-breaking base-class addition.

### Why `setup_aliases()` is top-level-only while `transform_weights()` and `cache_derived_state()` walk per-module

Alias wiring conventionally lives on the top-level model class (e.g., `LlamaForCausalLM.post_load_weights` walks `self.model.layers` and assigns `next_attn` / `next_layer_layernorm`). It does not need to fire per submodule. By contrast, `transform_weights()` (FP8 conversion, quant-scale fusion) and `cache_derived_state()` (data-dependent Python state) live on the submodules that own the weights, so those walks must visit each module. Calling `setup_aliases()` only at the top level mirrors the documented §7 mitigation and avoids over-broadly invoking a hook on submodules whose alias contract is "no-op."

### Lifecycle of `_weights_transformed`

The new flag governs `transform_weights()` idempotency. Without crisp set/reset rules, a stale `True` would silently skip a legitimate transform on new untransformed bytes.

- **Set** at the end of a successful `transform_weights()` call (after the subclass-specific transform body completes without raising).
- **Reset** by any code path that overwrites the underlying tensor with new untransformed bytes:
  - `ModelLoader.reload()` — rebinds parameters with fresh disk-loaded weights; transform must run again.
  - Partial-fallback merging — when a checkpoint loader returns a non-empty `weights` dict after MX P2P (the size-mismatched fallback path), the merged tensors are pre-transform and need a transform pass.
  - Any future sleep/wake path that rebinds tensors instead of just releasing them. (Today's sleep/wake clears `_parameters` and sets `_weights_removed=True` without re-binding, so this case is hypothetical until that changes.)
- **Orthogonal** to the existing `_weights_removed` flag: the two flags track different lifecycles. `_weights_removed` is the sleep/wake "weights are not currently allocated" signal; `_weights_transformed` is the "current weights have already been through `transform_weights()`" signal. All four combinations are valid:

| `_weights_removed` | `_weights_transformed` | Meaning |
|:---:|:---:|:---|
| False | False | Normal pre-load state, or post-`reload()` before transform runs again. |
| False | True | Normal post-load steady state. |
| True | False | Sleep state (weights released); transform flag is moot. |
| True | True | Sleep state from a fully-loaded engine; on wake-with-rebind, reset `_weights_transformed` first. |

Reset is the responsibility of the orchestrator that introduces new bytes (e.g., `ModelLoader.reload()` resets the flag on every affected module before invoking the standard mapper). Subclasses do not manage reset themselves — see precondition **P2** below.

---

## Per-path orchestration

```python
# AUTO + HF disk (today's behavior, unchanged)
weights = checkpoint_loader.load_weights(...)
self._call_load_weights(...)
for m in model.modules():
    m.post_load_weights()                    # default orchestrator: A → B → D


# AUTO + MX, publish-post-transform target state (Wave 4; receivers skip B)
weights = mx_loader.load_weights(...)        # P2P writes post-transform bytes
if mx_p2p_succeeded:
    model.setup_aliases()                    # A: top-level only — alias wiring lives
                                             #    on the model class (LlamaForCausalLM,
                                             #    etc.), not on layer submodules.
                                             #    Matches the §7 mitigation contract.
    # SKIP transform_weights — sender already baked it
    for m in model.modules():
        m.cache_derived_state()              # D: recompute from received tensors
                                             #    (per-module: data-dependent state may
                                             #    live anywhere in the tree)
else:
    for m in model.modules():
        m.post_load_weights()                # full path on disk-loaded weights


# GMS RW writer (post-TRTLLM-12440 widening, slight refinement)
with mem_pool_scope():
    init_meta_tensors(); model.to("cuda")
    weights = checkpoint_loader.load_weights(...)
    self._call_load_weights(...)
    for m in model.modules():
        m.post_load_weights()                # full path inside pool: A → B → D all in pool
gms_backend.finalize_write(model)            # commits post-transform layout for RO peers


# GMS RO reader (closes §7 alias trap AND closes the (D) divergence)
model.setup_aliases()                        # A: top-level only — alias wiring lives on
                                             #    the model class. This is exactly the
                                             #    "pre-materialize alias hook" the
                                             #    ai-dynamo GMS prototype does externally;
                                             #    we own it in-tree so non-Dynamo TRT-LLM
                                             #    users benefit too.
gms_backend.materialize_module(model)        # zero-copy bind storage to wired-up tree
for m in model.modules():
    # SKIP m.transform_weights — RW already baked transforms into the pool
    m.cache_derived_state()                  # D: recompute Python state from real tensors
                                             #    (per-module: data-dependent state may live
                                             #    on any submodule in the tree)
```

After this, every consumer can pick the exact subset it needs, and each individual stage has a single, easy-to-reason-about contract. Category C (`MoeLoadBalancer.finalize_model()`, EP topology) continues to run unconditionally from `ModelLoader.load` regardless of weight provenance — it was always orchestrator-managed and the staged hook protocol does not move it.

---

## Hard preconditions

The staged-hook design relies on two correctness preconditions that the design itself cannot enforce. If either is unmet, receivers can silently consume incompatible weights.

### P1. Source-identity matching must cover all transform-affecting parameters

Letting any receiver (MX or GMS RO) skip `transform_weights()` is only safe if the source and the receiver agree on every choice that affects how transformed weights are laid out. The identity comparison must cover at least:

- `attn_backend` (`TRTLLM`, `FlashInfer`, `FlashAttention`) — drives weight layout for fused QKV and attention masks.
- Quant backend list, e.g. `nvfp4_allowed_backends`, `fp8_allowed_backends` — drives FP8 / NVFP4 fusion strategy.
- FP8 / NVFP4 scale-fusion strategy (per-tensor, per-channel, blocked, …).
- TP / PP / EP layout (sizes and ranks).
- Model revision and quantization config.
- Any future quant scheme or fusion pass that introduces a new transform variant.

Identity comparison is fundamentally a **TRT-LLM concern**: only TRT-LLM knows which knobs affect layout, and the set grows as new fusion / quant strategies are added. So the API surface lives in TRT-LLM, and both MX and GMS consume it:

- `tllm.disagg.compute_source_identity(config) -> bytes` — TRT-LLM computes a canonical fingerprint over the local config. The weight-sharing publisher (MX or GMS RW) calls this and embeds the result as opaque bytes in its payload.
- `tllm.disagg.is_source_compatible(local_config, remote_identity) -> bool` — the receiver (MX P2P or GMS RO) calls this before consuming. A False result must fall back to a full disk-load path or raise, per consumer policy. The `remote_identity` is opaque to MX / GMS — they only transport it.

This contract protects MX and GMS from churn: when TRT-LLM adds a new layout-affecting knob, only `compute_source_identity` / `is_source_compatible` change. The transport libraries carry the bytes unchanged.

Without complete identity coverage, the staged-hook design lets a misconfigured replica pull weights laid out for the wrong backend and run silently — exactly the homogeneity-assumption hazard raised on the MX-team refactor's review. **This is a hard precondition, not an open question.** No receiver path may skip `transform_weights()` until `is_source_compatible()` is wired into that receiver path.

### P2. Reset of `_weights_transformed` is the orchestrator's responsibility

Subclasses set the flag at the end of `transform_weights()`. Resets happen in `ModelLoader.reload()` and in any other code path that re-binds tensors to untransformed bytes. Subclasses MUST NOT reset the flag themselves; that would couple transform idempotency to subclass-specific semantics and break the orchestrator-level contract.

---

## Migration scope

Inventory of `post_load_weights(self)` overrides in `tensorrt_llm/_torch/` (excluding visual-gen pipelines, which are not in the LLM weight-load path):

| Bucket | Files | Action |
|:-------|:------|:-------|
| **A — pure alias wiring** | `models/modeling_llama.py`, `models/modeling_deepseekv3.py`, `models/modeling_glm.py`, `models/modeling_exaone_moe.py`, `models/modeling_qwen3_moe.py`, `models/modeling_qwen3_next.py`, `models/modeling_gpt_oss.py` (7 files) | Move body verbatim into `setup_aliases()`. Trivial mechanical migration. |
| **B — weight-data transforms** | `modules/attention.py`, `modules/linear.py`, `modules/mamba/mamba2_mixer.py`, `modules/fused_moe/fused_moe_triton.py`, `modules/fused_moe/quantization.py` (1 file with 6 quant variants), `models/modeling_llama_min_latency.py` (5+1 files, 11 method bodies) | Move body into `transform_weights()` with `_weights_transformed` guard. Mechanical with care for the multi-override quantization file. |
| **Mixed A + B** | Likely 0–2 files; needs deeper read (e.g. `modeling_llama_min_latency.py` may have both) | Split into `setup_aliases()` + `transform_weights()`. |
| **Trivial / pass-through** | `models/modeling_speculative.py`, `models/modeling_nemotron_h.py`, `attention_backend/sparse/dsa.py`, `modules/fused_moe/{interface,configurable_moe,fused_moe_wide_ep,fused_moe_trtllm_gen,fused_moe_densegemm,fused_moe_cutlass}.py`, `modules/fused_moe/mega_moe/mega_moe_deepgemm.py` (10 files) | Either inherit defaults (no migration) or migrate trivially. |

Two distinct method signatures exist in the codebase. The protocol targets only the no-arg `Module.post_load_weights(self)` form that the `ModelLoader` walker invokes:

```python
# IN-SCOPE (called by ModelLoader's `for m in model.modules(): m.post_load_weights()`)
class Module(nn.Module):
    def post_load_weights(self): ...

# OUT-OF-SCOPE (internal callback inside Linear / MoE; comes along for the ride
# when the outer Linear / MoE.post_load_weights() migrates to transform_weights())
class QuantMethod:
    def post_load_weights(self, module): ...
```

So the migration target is **~13 substantive files** plus 7 trivial alias-only files. Largest single body is `fused_moe/quantization.py` at ~147 LOC across 6 quant-method overrides — mechanical renames, not redesigns. `modeling_llama.py` (the §7 example) is ~20 LOC of pure alias wiring.

---

## Implementation plan

### Phase status (2026-05-31)

| Phase | Status | Vehicle | Risk | Est. LOC | MX receiver value |
|:------|:-------|:--------|:-----|:---------|:------------------|
| Prep | Open, review-required | TRTLLM-13077 | n/a | ~70 | — |
| **Wave 1** | **Next** | TBD | LOW | ~165 | 0 (MX still publishes PRE-transform) |
| Wave 2 | Queued | TBD | HIGH | ~80 | partial (~60% of models become receiver-ready) |
| Wave 3 | Queued | TBD | HIGH | ~280 | full (100% of models receiver-ready) |
| Wave 4 | Queued | TBD (MX-side publisher flip + TRT-LLM receiver cutover) | MEDIUM | ~80 in-tree + MX-side flip | flip + per-model rollout |

**Migration callout (applies to every wave below that migrates an override):** when a subclass moves from overriding `post_load_weights()` to overriding `setup_aliases()` / `transform_weights()` / `cache_derived_state()`, the old `post_load_weights()` override **must be removed**. Leaving it in place silently shadows the base-class orchestrator and the new staged calls become no-ops. The pattern is: (a) move each block of the old body into the appropriate new method, (b) delete the `def post_load_weights(self):` line, (c) verify by grepping the diff for any remaining `def post_load_weights` in the migrated class.

### Prep PR — TRTLLM-13077 (awaiting review)

`[TRTLLM-13077][feat] Decompose post_load_weights()` introduces the contract surface without migrating any model: default no-op `setup_aliases()` / `transform_weights()` / `cache_derived_state()`, the `_weights_transformed` flag with the lifecycle documented above, helper walkers on `ModelLoader` (`_walk_transform`, `_walk_cache_state`, plus a backward-compat orchestrator), and protocol unit tests. Neither TRTLLM-12440's GMS RO branch nor the inflight MX-team refactor is blocked on the migration: each carries a `TODO(STAGED-HOOKS)` against this section and continues with its ad-hoc workaround until its respective wave cuts it over. Wave 1 begins once this prep PR merges.

Prep-PR scope:

1. **Define the contract via duck-typed helpers, not via inheritance.** `ModelLoader` already invokes `post_load_weights()` through `getattr(module, 'post_load_weights', None)` + `hasattr` checks — see [model_loader.py](../../../tensorrt_llm/_torch/pyexecutor/model_loader.py). The walkers follow the same pattern so that `Linear`, `Attention`, MoE, Mamba submodules can opt in by simply defining the method, without forced inheritance changes. Optionally provide a `StagedHooksMixin` for type-checking convenience, but do not require it.
2. Define the three per-module stages — `setup_aliases()`, `transform_weights()` (with `_weights_transformed` guard), `cache_derived_state()` — as documented method names with default no-ops on the existing base classes (`DecoderModelForCausalLM`, plus `nn.Module` defaults via the helper). Subclasses opt in by overriding any subset.
3. Provide a backward-compat `post_load_weights()` orchestrator on the base class that calls the three stages in order, so non-migrated subclasses see no behavior change. NOTE: this only helps subclasses that don't currently override `post_load_weights()`. Subclasses with existing overrides keep their old behavior until they explicitly migrate.
4. `_weights_transformed` flag introduced alongside the existing `_weights_removed` flag — orthogonal semantics:
   - `_weights_removed` = sleep / wake lifecycle (existing meaning).
   - `_weights_transformed` = transform-step idempotency (new).
   See "Lifecycle of `_weights_transformed`" above for full set/reset rules.
5. Helper functions on `ModelLoader` for the per-path dispatch:
   - `_setup_aliases(model)` — top-level only.
   - `_walk_transform(model)` — per-module walk honoring the `_weights_transformed` guard.
   - `_walk_cache_state(model)` — per-module walk for `cache_derived_state()`.
   - `_walk_full_post_load(model)` — current behavior, equivalent to `for m in model.modules(): m.post_load_weights()`.
6. Migration tracker (this doc) listing the LLM-relevant files and their target bucket so subsequent waves have a checklist.
7. Tests for the orchestrator default behavior + the per-stage walks (mock-based, no model touched).

### Wave 1 — Alias migration + GMS-RO cutover (~165 LOC, LOW risk)

**Scope:**
- Migrate the 7 alias-wiring model classes from `post_load_weights()` to `setup_aliases()`: `modeling_llama`, `modeling_deepseekv3`, `modeling_glm`, `modeling_exaone_moe`, `modeling_qwen3_moe`, `modeling_qwen3_next`, `modeling_gpt_oss`.
- Cut over the GMS RO branch in `tensorrt_llm/_torch/pyexecutor/model_loader.py` from the §7 workaround (full `post_load_weights()` on meta tensors before `materialize_module_from_gms()`) to the staged-hook protocol: `model.setup_aliases()` → `gms_backend.materialize_module(model)` → per-module `cache_derived_state()` walk.

**Bundling rationale (why W1 is one PR, not two):**
- Alias migration alone produces no user-visible change — the top-level `setup_aliases()` invoked by the orchestrator default is bit-identical to the current top-level `post_load_weights()` walk.
- GMS-RO cutover alone is broken without `setup_aliases()` carrying real wiring — an empty default no-op would not reproduce the §7 mitigation, and `materialize_module_from_gms()` would AttributeError as it does in the pre-mitigation state.
- Bundled, they form one cohesive vertical slice with end-to-end observable correctness on the GMS RO path. The change touches code paths that GMS-RO functional tests exercise directly.

**Blast radius:** GMS RO load path; alias wiring on the listed 7 model classes. Non-GMS paths (AUTO + HF, AUTO + MX) continue to use the backward-compat `post_load_weights()` orchestrator, which is bit-identical to today's behavior.

**Risk: LOW.**
- Default orchestrator preserves behavior for non-GMS paths.
- GMS-RO functional tests (introduced in TRTLLM-12440) catch regressions immediately.
- Alias wiring is structural Python — no tensor math, no numerical risk.

**MX-side value: 0.** The merged TRTLLM-11851 MX behavior is publish-PRE-transform; receivers correctly run the full `post_load_weights()` on disk-loaded bytes. Wave 1 does not change this. No MX receiver cutover happens in Wave 1.

**Gate to Wave 2:**
- GMS RW/RO functional tests green.
- CI integration tests for the 7 migrated models green.
- Manual verification of `nn.Module.__setattr__` dedupe semantics on at least one model (alias re-assignment is the new idempotency contract).

### Wave 2 — Linear / Attention transform migration (~80 LOC, HIGH risk)

**Scope:**
- Migrate `tensorrt_llm/_torch/modules/linear.py` and `tensorrt_llm/_torch/modules/attention.py` from `post_load_weights()` to `transform_weights()` + `_weights_transformed` guard.
- **Quant-method callback decision** (deferred from prep PR): the existing internal callback `QuantMethod.post_load_weights(self, module)` is invoked from `Linear.post_load_weights(self)`. Default is (a) keep the quant-method callback name unchanged and have the migrated `Linear.transform_weights(self)` invoke `quant_method.post_load_weights(module)`. The no-arg vs `(self, module)` signature distinction already disambiguates them; renaming touches ~10 quant-method overrides for cosmetic gain. Revisit only if (b) renaming materially helps clarity during implementation.

**Blast radius:** Every model with a `Linear` or `Attention` module — i.e., the whole model zoo. Highly exercised code path; a regression here surfaces in essentially every CI integration test.

**Risk: HIGH.** Mitigations:
- Strict `_weights_transformed` idempotency contract: set on successful return of the transform body; reset only by orchestrator-managed code (`ModelLoader.reload()`, partial-fallback merge). Subclasses MUST NOT reset.
- Standalone PR. No other migrations bundled.
- CI must run the full integration suite on representative models per backend (FP8, NVFP4, BF16, INT8) before merge.
- Add a per-module idempotency unit test: calling `transform_weights()` twice produces a no-op on the second call.

**MX-side value: PARTIAL — ~60% of the model zoo.** Once Wave 2 lands, models whose only transform-affecting modules are Linear + Attention (Llama, Qwen, Mistral, and similar dense models — no MoE, no Mamba, no sparse attention) become *eligible* for publish-after-transform on the receiver side. Eligibility is gated by Wave 4 plumbing; no MX runtime behavior changes in Wave 2.

**Gate to Wave 3:**
- Full integration suite green across FP8, NVFP4, BF16, INT8 for at least one Llama-style and one Qwen-style model.
- Bit-equivalence check on transformed weight hashes for a fixed checkpoint, pre-Wave-2 vs post-Wave-2.

### Wave 3 — MoE + Mamba transform migration (~280 LOC, HIGH risk)

**Scope:**
- `tensorrt_llm/_torch/modules/fused_moe/quantization.py` — 6 quant-method overrides (largest single body in the migration; ~147 LOC of method renames + guards).
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_triton.py`.
- `tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py`.
- `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`.
- `tensorrt_llm/_torch/models/modeling_llama_min_latency.py`.
- `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`.

**Blast radius:** Every MoE-bearing model (DeepSeek V3, Qwen3-MoE, GPT-OSS, Mixtral, etc.), every Mamba-bearing model, the min-latency Llama path, and the sparse-attention path. Less broadly exercised than Wave 2 but still production-critical.

**Risk: HIGH.**
- Six quant-method overrides multiply the surface area for typos.
- MoE expert-slot bookkeeping (Category C, orchestrator-managed via `MoeLoadBalancer.finalize_model()`) is adjacent to and easy to confuse with weight transforms (Category B). Reviewers must explicitly verify the boundary.
- Mitigations: standalone PR; per-quant-method idempotency unit test; manual review on the Linear/Attention vs MoE callback boundary; full integration suite on at least one MoE model and one Mamba model per backend.

**MX-side value: FULL.** After Wave 3, every model is receiver-ready for publish-after-transform.

**Gate to Wave 4:**
- Full integration suite green on at least one MoE and one Mamba model per backend.
- Idempotency tests green for all 6 quant-method overrides.
- Manual sweep to confirm no model class is left with a stale `def post_load_weights(self)` that would shadow the staged hooks.

### Wave 4 — MX publish-after-transform flip + P1 fail-safe + receiver cutover (~80 LOC TRT-LLM + MX-side, MEDIUM risk)

**Scope:**
- **TRT-LLM source-identity API (~80 LOC):** land `tllm.disagg.compute_source_identity()` and `tllm.disagg.is_source_compatible()` in a `disagg` module callable by both MX and GMS receiver paths. Identity covers the parameters listed in **P1** above. Used by Wave 4's MX receiver cutover; consumed by the GMS RO path opportunistically once `transform_weights()`-skip plumbing exists there too.
- **MX publisher flip (MX-side, ~5 LOC):** the publisher embeds `compute_source_identity()` output in its payload and writes post-transform bytes. Whether this lands inside ModelExpress or stays in TRT-LLM's MX path depends on the "scope of delegation" discussion below — the staged-hook design is agnostic.
- **MX receiver cutover in `model_loader.py` (~30 LOC):** cut over the MX path from the current full `post_load_weights()` walk to the staged-hook protocol: call `is_source_compatible()`; on True, run `model.setup_aliases()` → skip `transform_weights()` walk → per-module `cache_derived_state()` walk; on False, fall back to the current full-load path. The cutover is also gated on (b) below.
- **Per-model enable allow-list:** the receiver does not unconditionally trust an identity-compatible source. It consults a TRT-LLM allow-list keyed by `(model_class, transform_protocol_version)`. Models migrate into the allow-list one-by-one as integration testing validates them. Models not on the list run the full path regardless of compatibility.

**Blast radius:** MX-only path. No GMS impact. Affects only deployments that enable MX P2P checkpoint loading. Default-off for any model not in the allow-list — a deployment that upgrades to the Wave-4 code while running a not-yet-allow-listed model continues to receive PRE-transform bytes and runs the full receiver-side `post_load_weights()`, identical to today.

**Risk: MEDIUM.** The publish-after-transform flip is the very change that the MX-team refactor's review flagged as unsafe; the P1 fail-safe is the in-tree answer to that critique. Risk is bounded by:
- Receiver-side fingerprint check raises before any transform-skip happens.
- Per-model allow-list prevents accidental cutover for models that did not undergo a transform-equivalence sweep.
- Independent of GMS — failure here does not regress GMS or HF loading paths.

**Dependencies:** Waves 2 and 3 must be complete. Without them, the `transform_weights()` migration is incomplete and the receiver cannot safely skip per-module hooks for every model.

**Gate to closing the staged-hook initiative:**
- At least one production model (typically Llama-3-70B) cut over end-to-end on a multi-host MX deployment.
- Bit-equivalence verification of `cache_derived_state()` output between disk-loaded and MX-received tensors on that model.
- P1 fail-safe verified by deliberately running a config mismatch (e.g., publisher on FlashInfer, receiver on TRTLLM) and observing the raise.

### Per-model incremental rollout strategy (post-Wave-2 / Wave-3 staging into Wave 4)

Waves 2 and 3 are repo-wide module migrations and cannot be done per-model without leaving the codebase in a split state. But Wave 4's *consumption* of those migrations is gated per-model:

- The MX publisher's identity payload carries a `transform_protocol_version` field.
- The MX receiver in TRT-LLM consults a per-model allow-list and only allows publish-after-transform receive for models on the list at the matching protocol version.
- After Wave 2 lands, dense Linear/Attention-only models (Llama, Qwen, Mistral, etc.) become candidates. After Wave 3, MoE and Mamba models become candidates.
- Each model's addition to the allow-list is its own small change, independently revertable, gated by an integration test that verifies bit-equivalence of `cache_derived_state()` output between disk-loaded and MX-received tensors.
- A model that fails the bit-equivalence sweep stays off the allow-list — Wave 4 ships and is useful even if not every model is in the allow-list on day one.

### Cleanup (post-Wave-4)

Once all subclasses have migrated and Wave 4 has shipped:

- Remove the orchestrator's transitional `post_load_weights()` back-compat path.
- Consider collapsing `_weights_removed` into `_weights_transformed` if their lifecycles converge in practice (they do not today — `_weights_removed` is sleep/wake, `_weights_transformed` is one-shot — but reassess after the migration settles).
- Close out the `TODO(STAGED-HOOKS)` markers left by TRTLLM-12440 and the MX-team refactor.

Total full-migration LOC (Waves 1–4): ~600 in-tree + MX-side publisher flip.

---

## Coordination with MX and GMS

The staged-hook decomposition is largely an in-tree TensorRT-LLM change. Two interface points need explicit coordination with the upstream weight-sharing libraries.

### Source-identity API (TRT-LLM-owned, used by both MX and GMS)

The receiver-side `transform_weights()` skip is only safe when the remote source agrees with the local consumer on every layout-affecting choice (precondition **P1**). Identity comparison is fundamentally a TRT-LLM concern — only TRT-LLM knows which knobs affect layout, and the set grows as new fusion / quant strategies are added. So the API surface lives in TRT-LLM, and both MX and GMS consume it:

- **TRT-LLM provides** `tllm.disagg.compute_source_identity(config) -> bytes` and `tllm.disagg.is_source_compatible(local_config, remote_identity) -> bool`. Both functions are stable, versioned, and exhaustively cover the parameters listed under P1 above (plus any future layout-affecting knob — change-detection is the API's job).
- **MX adopts** the API by calling `compute_source_identity` on the publisher and embedding the opaque bytes in its payload, and by calling `is_source_compatible` on the receiver before allowing a `transform_weights()` skip.
- **GMS adopts** the API by the same pattern: RW writer calls `compute_source_identity` and stores the bytes in the catalog metadata; RO reader calls `is_source_compatible` before binding storage and choosing the staged-hook path. Wave 1 does not require this yet (GMS RO will not skip `transform_weights()` until Wave 4 plumbing exists), but the API contract can land alongside Wave 1.

The opaque-bytes design protects MX and GMS from churn: when TRT-LLM adds a new layout-affecting parameter, only `compute_source_identity` / `is_source_compatible` change. The transport libraries carry the bytes unchanged.

### Scope of MX checkpoint loading: delegation vs. retained custom logic

The inflight MX-team Delegate-to-ModelExpress refactor proposes moving the entire MX checkpoint-loading flow into the ModelExpress library, with TRT-LLM only consuming the API. The wholesale delegation is appealing for separation of concerns, but it is not necessarily the safe path for TRT-LLM:

- **Fallback logic.** When MX P2P fails or returns size-mismatched data, TRT-LLM's current MX path falls back to disk load. The fallback decision uses TRT-LLM-internal state (which weights matched, what's left to load, what the local cache looks like) and is best owned by TRT-LLM, not by a generic transport library.
- **Custom validation.** TRT-LLM applies its own per-tensor validation on loaded weights (dtype checks, shape checks, sparsity-pattern sanity) before they hit the transform stage. These checks are TRT-LLM-internal and should not be delegated.
- **Telemetry and error recovery.** TRT-LLM emits its own metrics on the load path and has its own error-classification taxonomy. A wholesale delegation forces ModelExpress to carry TRT-LLM-aware logic.

A safer division of labor would be: ModelExpress provides the transport (P2P protocol, buffer management, RDMA primitives), TRT-LLM keeps the orchestration (fallback decision, custom validation, telemetry, retry policy). This is the model already used for NIXL and UCX in TRT-LLM's disagg KV-cache transceiver — the transport library does transport, and TRT-LLM does orchestration.

This document is **agnostic** on which division of labor the MX-team refactor ultimately ships with. Wave 4's receiver-side cutover from "full `post_load_weights()`" to "`setup_aliases()` + `cache_derived_state()`" applies regardless. What changes between the two models is only **where** the publisher embeds `compute_source_identity()` output and **where** the receiver calls `is_source_compatible()`:

- *Wholesale delegation*: both calls live in ModelExpress; TRT-LLM hands ModelExpress a `SourceIdentityProvider` object at registration.
- *Transport-only delegation*: both calls live in TRT-LLM's MX path, and ModelExpress sees only opaque bytes alongside the weight payload.

### What this asks of MX

- Decide on the scope of delegation in the MX-team refactor proposal (wholesale vs. transport-only), with TRT-LLM advocating transport-only for the safety reasons above.
- Adopt the TRT-LLM source-identity API on both publisher and receiver, regardless of which scope wins.
- Provide a publish-pre / publish-post toggle on the MX payload so the receiver-side allow-list (Wave 4) can roll out gradually per model.

### What this asks of GMS

- No new asks for Wave 1. The existing materialize-from-catalog contract from the ai-dynamo prototype is preserved; Wave 1 only moves the in-tree alias-wiring step from inside the monolithic `post_load_weights()` to a dedicated `setup_aliases()` method, which is functionally equivalent on the GMS RW writer side.
- Confirm RO-peer behavior on `cache_derived_state()`. Wave 1 will start calling `cache_derived_state()` on the RO peer against real CUDA tensors after `materialize_module_from_gms()` binds storage. This is the correctness fix for the cached-scales-NaN / dtype-validation-lies bugs in the current workaround. No behavior change on the GMS protocol itself, but the RO peer now does meaningful per-module work after materialize. Profile guidance for large models (70B+) welcomed; expected to be cheap (most cached state is small scalars / dtypes).
- Adopt the TRT-LLM source-identity API alongside Wave 1 (compute_source_identity on RW writer, is_source_compatible on RO reader) so that the same path is ready when Wave 4 plumbing lands.

---

## Open questions to resolve during implementation

| # | Question | Default if unresolved |
|:--|:---------|:-----------------------|
| 1 | Where the protocol lives — `DecoderModelForCausalLM` only? `nn.Module` mixin? Per-class trait? | Duck-typed via `getattr/hasattr` on the `ModelLoader` walkers, mirroring the existing `post_load_weights()` walker pattern. Optionally add a mixin for type-checking convenience, but do not require subclasses to inherit from it — most existing overrides are on `Linear`, `Attention`, MoE, and Mamba submodules that don't share a common base. |
| 2 | Naming finalization — `setup_aliases` / `transform_weights` / `cache_derived_state`. | Keep these names. Alternatives considered: `_post_load_setup` / `_post_load_transform` / `_post_load_finalize` (more parallel but vaguer). |
| 3 | Should the orchestrator default invoke an existing override of `post_load_weights()` if a subclass has one but no granular methods? | Yes during transition, with a deprecation warning. After full migration, remove the back-compat path. |
| 4 | Speculative decoding draft model — is the `model.draft_model` walk handled correctly by the same per-stage walkers? | Yes — `for m in model.modules()` recurses into `draft_model`. No special-casing needed. (Applies to `transform_weights()` and `cache_derived_state()` walks; `setup_aliases()` is top-level so the draft model's own alias wiring is invoked separately if it has any.) |
| 5 | `ModelLoader.reload()` interaction. After `reload()` rebinds tensors, which stages need to re-run? | Reset `_weights_transformed=False` on every affected module; then run `transform_weights()` and `cache_derived_state()`. `setup_aliases()` is idempotent and need not be re-run unless the module tree itself changed. |
| 6 | (D) recomputation cost. For a 70B model, how expensive is `cache_derived_state` on real tensors? | Expected to be cheap (most cached state is small scalars / dtypes). Profile during the cutover PR. |

(Open question 6 from the prior revision — "Identity matching on the MX side" — has been promoted to hard precondition P1 above and removed from this table.)

---

## References

- **TRTLLM-13077** — `[TRTLLM-13077][feat] Decompose post_load_weights()` (open, awaiting review). Prep PR that introduces the staged-hook contract surface, helper walkers, and `_weights_transformed` lifecycle without migrating any model. Vehicle for the "Prep" row in the phase-status table above.
- **TRTLLM-12440** — `[TRTLLM-12440][feat] Add GMS-only weight sharing support` (merged). RO-branch ordering trade-off documented inline; `TODO(STAGED-HOOKS)` to be added in the Wave 1 cutover.
- **TRTLLM-11851** — `[TRTLLM-11851][feat] Add MX-only P2P checkpoint loading support for TRTLLM` (merged). Establishes the current MX publish-PRE-transform contract that Wave 4 flips.
- **End-to-end MX + GMS prototype** — first integration demonstration combining both technologies, motivating the architectural fix in this design.
- Inflight MX-team `[None][refactor] Delegate MX checkpoint loading to ModelExpress` proposal — the publish-pre vs publish-post-transform discussion that surfaced the same architectural gap from the MX side. The homogeneity-assumption hazard raised on its review thread is the motivation for the P1 fail-safe in Wave 4.
- ai-dynamo/dynamo PR [#7053](https://github.com/ai-dynamo/dynamo/pull/7053) — upstream GMS prototype that originally surfaced and fixed the alias-resolution `AttributeError`. The mitigation contract ("call `model.post_load_weights()` before `materialize_module_from_gms()` to set up structural cross-references") is the source of TRT-LLM's current GMS RO ordering and the motivation for splitting alias wiring out as its own stage.
- [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) — the existing in-tree write-up of the alias bug and the per-PR mitigation. After Wave 1 lands, §7 forward-links here for the holistic fix.
- [Module Path Resolution risk in §12-risks.md](12-risks.md) — risk row "Module path resolution (aliased layers)". The staged-hook protocol moves this risk from "MEDIUM, mitigated per PR" to "LOW, structurally fixed."
