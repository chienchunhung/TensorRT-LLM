# 16. Staged Post-Load Hooks (Holistic MX + GMS Fix)

**Status:** Draft — discussion captured, scope locked in for a Tiny prep PR; full migration deferred to follow-up family PRs.
**Created:** 2026-05-19
**Drives:** Review feedback on [PR #13926](https://github.com/NVIDIA/TensorRT-LLM/pull/13926) (`@hhzhang16` — RO post-load ordering) and [PR #14151](https://github.com/NVIDIA/TensorRT-LLM/pull/14151) (`@chienchunhung` — publish-pre vs publish-post-transform; receivers selectively skipping `module.post_load_weights()`).

---

## Background

Both PR #13926 (GMS-only) and PR #14151 (MX shim refactor) independently bumped into the same architectural gap when integrating weight-sharing technologies:

- **GMS RO reader** needs to wire structural aliases before `materialize_module_from_gms()` walks the catalog (proven in [ai-dynamo/dynamo PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053); see [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) for the `LlamaForCausalLM.next_attn` `AttributeError`), but ALSO needs to recompute Python-side derived state on real CUDA tensors after materialization. With `post_load_weights()` overloaded as a single hook, neither order is correct: pre-materialize hides Python-state divergence; post-materialize crashes on alias resolution.

- **MX receiver** with publish-post-transform delivery (#14151) inherits weight tensors that are already transformed. Re-running the full `post_load_weights()` re-applies transforms (FP8 conversion, QKV fusion, quant-scale fusion) on already-transformed bytes — silently corrupting weights, or silently inheriting the publisher's backend choice when the receiver is configured differently (the homogeneity-assumption hazard flagged in the review).

The shared root cause is that `module.post_load_weights()` glues four categorically different operations together with one `_weights_removed` flag governing all of them.

## The four categories `post_load_weights()` conflates today

| Category | Example | Idempotent? | Receiver of post-transform weights needs it? |
|:---------|:--------|:------------|:---------------------------------------------|
| **A. Structural alias wiring** | `layer.next_attn = self.model.layers[idx + 1].self_attn`; shared embedding refs; fused-module references | **Yes**, naturally — `nn.Module.__setattr__` dedupes by object identity in `_modules` | **Always** — module-tree resolution depends on it (GMS catalog walk, `post_load_apply`, etc.) |
| **B. Weight-data transforms** | Fused QKV, quantization scales, FP8 weight conversion, MoE shared-weight loading | **No**, one-shot and irreversible | **Skip** — bytes already transformed |
| **C. Per-process state setup** | MoE routing tables, EP topology bookkeeping, expert weight slot registration | Yes, but must always run | **Always** — process-local state cannot be transferred |
| **D. Derived Python-side state** | `self._scale_cache = self.weight.std()`; dtype-validation booleans; cached fingerprints | **Yes**, by recomputation | **Recompute** on real tensors after weights arrive (Python state is not transferred over the wire) |

The current single-method `post_load_weights()` glues A+B+D together with one `_weights_removed` flag governing all of them. Category C (per-process state) is largely orchestrator-managed today (`MoeLoadBalancer.finalize_model()` is called from `ModelLoader.load`, not from per-module hooks), which is the right shape and stays unchanged.

## Why each PR's ad-hoc workaround leaves a residual bug

| PR | Workaround | Residual bug |
|:---|:-----------|:-------------|
| **#13926 (GMS RO)** | Run full `post_load_weights()` on meta tensors before `materialize_module()` (per [§7 mitigation](05-challenges.md#7-module-path-resolution-gms-specific)) | Categories B and D run against meta tensors — silent divergence between RW and RO peers for any module whose hook reads weight data (cached scales become NaN/0, dtype validation lies, fingerprints diverge). |
| **#14151 (MX)** | Publish-post-transform; receivers re-run their own `post_load_weights()` after P2P delivery | Category B re-runs on already-transformed bytes — re-applies FP8 conversion, double-fuses QKV, etc. Receivers also have no way to opt out of transforms while still keeping per-process state setup (the EP+MoE example in the review thread). |

## Proposed staged-hook protocol

Decompose the single `post_load_weights()` into four explicit stages on the base class, each with a default no-op body and clear contracts:

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

        Idempotent by recomputation. Always safe to call after any source of
        weight bytes (disk, MX P2P, GMS materialize). Runs on real tensors only.
        """

    # finalize_per_process_state stays orchestrator-managed (MoeLoadBalancer,
    # EP topology setup, etc.) — no per-module method needed today.

    # Backward-compat orchestrator. Existing callers see no change.
    def post_load_weights(self) -> None:
        self.setup_aliases()
        self.transform_weights()
        self.cache_derived_state()
```

The default `post_load_weights()` orchestrator preserves current behavior 1:1 for any code path that hasn't migrated, so this can land as a non-breaking base-class addition.

## Per-path orchestration

```python
# AUTO + HF disk (today's behavior, unchanged)
weights = checkpoint_loader.load_weights(...)
self._call_load_weights(...)
for m in model.modules():
    m.post_load_weights()                    # default orchestrator: A → B → D


# AUTO + MX (publish-post-transform per #14151, receivers skip B)
weights = mx_loader.load_weights(...)        # P2P writes post-transform bytes
if mx_p2p_succeeded:
    for m in model.modules():
        m.setup_aliases()                    # A: alias wiring (idempotent on real tensors)
        # SKIP m.transform_weights — sender already baked it
        m.cache_derived_state()              # D: recompute from received tensors
else:
    for m in model.modules():
        m.post_load_weights()                # full path on disk-loaded weights


# GMS RW writer (post-#13926 widening, slight refinement)
with mem_pool_scope():
    init_meta_tensors(); model.to("cuda")
    weights = checkpoint_loader.load_weights(...)
    self._call_load_weights(...)
    for m in model.modules():
        m.post_load_weights()                # full path inside pool: A → B → D all in pool
gms_backend.finalize_write(model)            # commits post-transform layout for RO peers


# GMS RO reader (closes §7 alias trap AND closes the (D) divergence)
for m in model.modules():
    m.setup_aliases()                        # A: structural-only, idempotent (this is exactly
                                             #    the "pre-materialize alias hook" the upstream
                                             #    GMS shim does externally; we own it in-tree
                                             #    so non-Dynamo TRT-LLM users benefit too)
gms_backend.materialize_module(model)        # zero-copy bind storage to wired-up tree
for m in model.modules():
    # SKIP m.transform_weights — RW already baked transforms into the pool
    m.cache_derived_state()                  # D: recompute Python state from real tensors
```

After this, every consumer can pick the exact subset it needs, and each individual stage has a single, easy-to-reason-about contract. Category C (`MoeLoadBalancer.finalize_model()`, EP topology) continues to run unconditionally from `ModelLoader.load` regardless of weight provenance — it was always orchestrator-managed and the staged hook protocol does not move it.

## Migration scope

Inventory of `post_load_weights(self)` overrides in `tensorrt_llm/` (excluding visual-gen pipelines, which are not in the LLM weight-load path):

| Bucket | Files | Action |
|:-------|:------|:-------|
| **A — pure alias wiring** | `modeling_llama.py`, `modeling_deepseekv3.py`, `modeling_glm.py`, `modeling_exaone_moe.py`, `modeling_qwen3_moe.py`, `modeling_qwen3_next.py`, `modeling_gpt_oss.py` (7 files) | Move body verbatim into `setup_aliases()`. Trivial mechanical migration. |
| **B — weight-data transforms** | `modules/attention.py`, `modules/linear.py`, `modules/mamba/mamba2_mixer.py`, `modules/fused_moe/fused_moe_triton.py`, `modules/fused_moe/quantization.py` (1 file with 6 quant variants), `modeling_llama_min_latency.py` (5+1 files, 11 method bodies) | Move body into `transform_weights()` with `_weights_transformed` guard. Mechanical with care for the multi-override quantization file. |
| **Mixed A + B** | Likely 0–2 files; needs deeper read (e.g. `modeling_llama_min_latency.py` may have both) | Split into `setup_aliases()` + `transform_weights()`. |
| **Trivial / pass-through** | `modeling_speculative.py`, `modeling_nemotron_h.py`, `attention_backend/sparse/dsa.py`, `modules/fused_moe/{interface,configurable_moe,fused_moe_wide_ep,fused_moe_trtllm_gen,fused_moe_densegemm,fused_moe_cutlass}.py`, `modules/fused_moe/mega_moe/mega_moe_deepgemm.py` (10 files) | Either inherit defaults (no migration) or migrate trivially. |

**Two distinct method signatures** exist in the codebase. The protocol targets only the no-arg `Module.post_load_weights(self)` form that the `ModelLoader` walker invokes:

```python
# IN-SCOPE (called by ModelLoader's `for m in model.modules(): m.post_load_weights()`)
class Module(nn.Module):
    def post_load_weights(self): ...

# OUT-OF-SCOPE (internal callback inside Linear/MoE; comes along for the ride
# when the outer Linear/MoE.post_load_weights() migrates to transform_weights())
class QuantMethod:
    def post_load_weights(self, module): ...
```

So the migration target is **~13 substantive files** plus 7 trivial alias-only files. Largest single body is `fused_moe/quantization.py` at 147 LOC across 6 quant-method overrides — but those are mechanical renames, not redesigns. `modeling_llama.py` (the §7 example) is 20 LOC of pure alias wiring.

## Implementation plan

### Tiny prep PR (lock in scope)

**Goal:** add the protocol contract without migrating any model. Both #13926 and #14151 declare a TODO dependency on the new contract.

**Scope:**
1. Add four method stubs to `DecoderModelForCausalLM` (and any submodule base class — `nn.Module` mixin or per-class trait, to be confirmed during implementation): `setup_aliases()`, `transform_weights()` with `_weights_transformed` guard, `cache_derived_state()`. All default to no-op.
2. Default `post_load_weights()` orchestrator on the base class that calls all three in order (preserves current behavior for any non-migrated subclass).
3. `_weights_transformed` flag introduced alongside the existing `_weights_removed` flag — they have orthogonal semantics:
   - `_weights_removed` = sleep/wake lifecycle (existing meaning).
   - `_weights_transformed` = transform-step idempotency (new).
4. Helper functions on `ModelLoader` for the per-path dispatch (alias-only walk, transform-skipping walk, full walk). Both #13926's GMS branches and #14151's MX receiver branch consume these helpers.
5. Migration tracker doc / docstring listing the 23 LLM-relevant files and their target bucket so subsequent PRs have a checklist.
6. Tests for the orchestrator default behavior + the per-stage walks (mock-based, no model touched).

**Estimated size:** 50–100 LOC.

**Outcome:** #13926 and #14151 both gain a `TODO(STAGED-HOOKS)` marker and continue with their current ad-hoc workarounds in their respective PRs. Neither is blocked on the migration.

### Follow-up family PRs (full migration)

Sequenced to keep each PR small, reviewable, and independently mergeable:

1. **Alias-wiring family**: migrate the 7 pure-A files (`modeling_llama`, `modeling_deepseekv3`, `modeling_glm`, `modeling_exaone_moe`, `modeling_qwen3_moe`, `modeling_qwen3_next`, `modeling_gpt_oss`). Each subclass overrides `setup_aliases()` instead of `post_load_weights()`. ~150 LOC total.
2. **Linear / Attention transform family**: migrate `modules/linear.py`, `modules/attention.py`. Activates the `_weights_transformed` guard for all leaf-level FP8 / quant transforms. ~80 LOC.
3. **MoE transform family**: migrate `fused_moe/quantization.py` (6 overrides), `fused_moe/fused_moe_triton.py`, `fused_moe/configurable_moe.py`. ~200 LOC.
4. **Mamba / misc**: `mamba2_mixer.py`, `modeling_llama_min_latency.py`, `attention_backend/sparse/dsa.py`. ~80 LOC.
5. **Switch consumers**: GMS RO branch (in #13926 follow-up) and MX receiver branch (#14151 follow-up) cut over to using `setup_aliases()` + `cache_derived_state()` directly instead of calling the orchestrator. Removes the §7-style ad-hoc workarounds.
6. **Cleanup**: remove the orchestrator's transitional behavior once all subclasses have migrated; collapse `_weights_removed` semantics if they end up redundant with `_weights_transformed` (probably not — they have different lifecycles).

Total full-migration LOC: ~600–800 across 4–5 family PRs over a few weeks.

## Open questions to resolve during implementation

| # | Question | Default if unresolved |
|:--|:---------|:-----------------------|
| 1 | Where the protocol lives — `DecoderModelForCausalLM` only? `nn.Module` mixin? Per-class trait? | Add to `DecoderModelForCausalLM`; add a sibling mixin/trait for submodules (Linear, Attention, MoE) since most current overrides live on submodules, not the top-level model. |
| 2 | Naming finalization — `setup_aliases` / `transform_weights` / `cache_derived_state`. | Keep these names. Alternatives considered: `_post_load_setup` / `_post_load_transform` / `_post_load_finalize` (more parallel but vaguer). |
| 3 | Should the orchestrator default invoke an existing override of `post_load_weights()` if a subclass has one but no granular methods? | Yes during transition, with a deprecation warning. After full migration, remove the back-compat path. |
| 4 | Speculative decoding draft model — is the `model.draft_model` walk handled correctly by the same per-stage walkers? | Yes — `for m in model.modules()` recurses into `draft_model`. No special-casing needed. |
| 5 | `ModelLoader.reload()` interaction. After `reload()` rebinds tensors, which stages need to re-run? | `cache_derived_state` (Python state may go stale); `transform_weights` is gated by `_weights_transformed` (no re-run); `setup_aliases` is idempotent (safe to re-run but unneeded). |
| 6 | Identity matching on the MX side — does upstream MX cover all weight-content-affecting config (attn_backend, quant backend list, FP8 fusion strategy, future quant schemes)? | Per [#14151 thread reply from @zhengluo-nv](https://github.com/NVIDIA/TensorRT-LLM/pull/14151#issuecomment-4462681671): "MX must guarantee to incorporate all sources of customization into identity building / matching." This is a separate upstream commitment; the staged-hook design is compatible with either outcome. |
| 7 | (D) recomputation cost. For a 70B model, how expensive is `cache_derived_state` on real tensors? | Expect to be cheap (most cached state is small scalars / dtypes). Profile during the cutover PR. |

## References

- TRT-LLM PR [#13926](https://github.com/NVIDIA/TensorRT-LLM/pull/13926) — `[TRTLLM-12440][feat] Add GMS-only weight sharing support`. RO-branch ordering trade-off documented inline; `TODO(STAGED-HOOKS)` will be added once this section lands.
- TRT-LLM PR [#14151](https://github.com/NVIDIA/TensorRT-LLM/pull/14151) — `[None][refactor] Delegate MX checkpoint loading to ModelExpress`. Publish-pre vs publish-post-transform discussion captures the same architectural gap from the MX side.
- ai-dynamo/dynamo PR [#7053](https://github.com/ai-dynamo/dynamo/pull/7053) — upstream GMS prototype that originally surfaced and fixed the alias-resolution `AttributeError`. The mitigation contract ("call `model.post_load_weights()` before `materialize_module_from_gms()` to set up structural cross-references") is the source of TRT-LLM's current GMS RO ordering and the motivation for splitting alias wiring out as its own stage.
- [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) — the existing in-tree write-up of the alias bug and the per-PR mitigation. After this section lands, §7 forward-links here for the holistic fix.
- [Module Path Resolution risk in §12-risks.md](12-risks.md) — risk row "Module path resolution (aliased layers)". The staged-hook protocol moves this risk from "MEDIUM, mitigated per PR" to "LOW, structurally fixed."
