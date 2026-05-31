# 16. Staged Post-Load Hooks (Holistic MX + GMS Fix)

**Status:** Locked (2026-05-30) — prep PR landed as TRTLLM-13077 (`[TRTLLM-13077][feat] Deocmpose post_load_weights()`). Wave 1 (alias migration + GMS-RO cutover) is the immediate next step. Full migration is sequenced as Waves 1–4 below.
**Created:** 2026-05-19
**Drives:**
- TRTLLM-12440 — `[TRTLLM-12440][feat] Add GMS-only weight sharing support` (merged): review feedback on RO post-load ordering. The §7 ad-hoc mitigation (run full `post_load_weights()` on meta tensors before `materialize_module()`) is the workaround the staged-hook protocol replaces.
- The inflight MX-team `[None][refactor] Delegate MX checkpoint loading to ModelExpress` proposal: publish-pre vs publish-post-transform discussion, receivers selectively skipping `module.post_load_weights()`, and the homogeneity-assumption hazard surfaced on the review thread.
- TRTLLM-11851 — `[TRTLLM-11851][feat] Add MX-only P2P checkpoint loading support for TRTLLM` (merged): establishes the current MX behavior (publish-PRE-transform; receivers run full `post_load_weights()` on disk-loaded bytes). Wave 4 below changes this contract.

---

## Background

Both the GMS-only weight sharing work (TRTLLM-12440, merged) and the inflight MX-team Delegate-to-ModelExpress refactor proposal independently bumped into the same architectural gap when integrating weight-sharing technologies:

- **GMS RO reader** needs to wire structural aliases before `materialize_module_from_gms()` walks the catalog (proven in [ai-dynamo/dynamo PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053); see [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) for the `LlamaForCausalLM.next_attn` `AttributeError`), but ALSO needs to recompute Python-side derived state on real CUDA tensors after materialization. With `post_load_weights()` overloaded as a single hook, neither order is correct: pre-materialize hides Python-state divergence; post-materialize crashes on alias resolution.

- **MX receiver** under the inflight publish-post-transform proposal would inherit weight tensors that are already transformed. Re-running the full `post_load_weights()` re-applies transforms (FP8 conversion, QKV fusion, quant-scale fusion) on already-transformed bytes — silently corrupting weights, or silently inheriting the publisher's backend choice when the receiver is configured differently (the homogeneity-assumption hazard flagged in the review). The currently-merged TRTLLM-11851 MX path side-steps this by publishing PRE-transform bytes; Wave 4 below is the controlled flip to publish-post-transform.

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

| Path | Workaround | Residual bug |
|:-----|:-----------|:-------------|
| **GMS RO (TRTLLM-12440)** | Run full `post_load_weights()` on meta tensors before `materialize_module()` (per [§7 mitigation](05-challenges.md#7-module-path-resolution-gms-specific)) | Categories B and D run against meta tensors — silent divergence between RW and RO peers for any module whose hook reads weight data (cached scales become NaN/0, dtype validation lies, fingerprints diverge). |
| **MX shim refactor (inflight)** | Publish-post-transform; receivers re-run their own `post_load_weights()` after P2P delivery | Category B re-runs on already-transformed bytes — re-applies FP8 conversion, double-fuses QKV, etc. Receivers also have no way to opt out of transforms while still keeping per-process state setup (the EP+MoE example in the review thread). |

## Proposed staged-hook protocol

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
        ``cache_derived_state()`` after migration; the stage exists to
        give consumers like GMS RO a correctness path on the modules
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

The default `post_load_weights()` orchestrator preserves current behavior 1:1 for any code path that hasn't migrated, so this can land as a non-breaking base-class addition.

### Lifecycle of `_weights_transformed`

The new flag governs `transform_weights()` idempotency. Without crisp set/reset rules a stale `True` would silently skip a legitimate transform on new untransformed bytes. The contract:

- **Set** at the end of a successful `transform_weights()` call (after the subclass-specific transform body completes without raising).
- **Reset** by any code path that overwrites the underlying tensor with new untransformed bytes:
  - `ModelLoader.reload()` — rebinds parameters with fresh disk-loaded weights; transform must run again.
  - Partial-fallback merging — when a checkpoint loader returns a non-empty `weights` dict after MX P2P (the size-mismatched fallback path), the merged tensors are pre-transform and need a transform pass.
  - Any future sleep/wake path that rebinds tensors instead of just releasing them. (Today's sleep/wake clears `_parameters` and sets `_weights_removed=True` without re-binding, so this case is hypothetical until that changes.)
- **Orthogonal** to `_weights_removed`: the two flags track different lifecycles. `_weights_removed` is the sleep/wake "weights are not currently allocated" signal; `_weights_transformed` is the "current weights have already been through `transform_weights()`" signal. Any combination is valid:

| `_weights_removed` | `_weights_transformed` | Meaning |
|:---:|:---:|:---|
| False | False | Normal pre-load state, or post-`reload()` before transform runs again. |
| False | True | Normal post-load steady state. |
| True | False | Sleep state (weights released); transform flag is moot. |
| True | True | Sleep state from a fully-loaded engine; on wake-with-rebind, reset `_weights_transformed` first. |

Reset is the responsibility of the orchestrator that introduces new bytes (e.g., `ModelLoader.reload()` resets the flag on every affected module before invoking the standard mapper). Subclasses do not manage reset themselves.

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
                                             #    the model class (LlamaForCausalLM, etc.),
                                             #    not on layer submodules. Matches the §7
                                             #    mitigation from ai-dynamo/dynamo PR #7053
                                             #    ("Call model.post_load_weights() (top-level
                                             #    only) before materialize_module_from_gms()").
                                             #    This is exactly the "pre-materialize alias
                                             #    hook" the upstream GMS shim does externally;
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

**Why `setup_aliases()` is top-level-only while `transform_weights()` and `cache_derived_state()` walk per-module.** Alias wiring conventionally lives on the top-level model class (e.g., `LlamaForCausalLM.post_load_weights` walks its own `model.layers` and assigns `next_attn` / `next_layer_layernorm` cross-references). It does not need to fire per submodule. By contrast, `transform_weights()` (FP8 conversion, quant-scale fusion) and `cache_derived_state()` (data-dependent Python state) live on the submodules that actually own the weights, so those walks must visit each module. Calling `setup_aliases()` only at the top level mirrors the documented §7 mitigation and avoids over-broadly invoking a hook on submodules whose alias contract is "no-op."

## Hard preconditions

The staged-hook design relies on two correctness preconditions that the design itself cannot enforce. If either is unmet, receivers can silently consume incompatible weights.

### P1. MX source-identity matching must cover all transform-affecting parameters

Letting an MX receiver skip `transform_weights()` is only safe if the source and the receiver agree on every choice that affects how transformed weights are laid out. The publisher's identity hash MUST cover at least:

- `attn_backend` (TRTLLM, FlashInfer, …) — drives weight layout for fused QKV and attention masks.
- Quant backend list, e.g. `nvfp4_allowed_backends`, `fp8_allowed_backends` — drives FP8 / NVFP4 fusion strategy.
- FP8 / NVFP4 scale-fusion strategy (per-tensor, per-channel, blocked, …).
- TP / PP / EP layout (sizes and ranks).
- Model revision and quantization config (already covered in `MXSourceIdentity`).
- Any future quant scheme or fusion pass that introduces a new transform variant.

Without complete identity coverage, the staged-hook design lets a misconfigured replica pull weights laid out for the wrong backend and run silently — exactly the "homogeneity-assumption hazard" raised on the review of the inflight MX-team Delegate-to-ModelExpress refactor proposal.

This is a hard precondition, not an open question. If upstream MX cannot guarantee identity completeness, TRT-LLM MUST add an in-tree fail-safe before the staged-hook receivers ship. Concretely (P1 fallback): TRT-LLM stores a backend-fingerprint (the canonical projection of the parameters above) on the publisher's side and re-checks it against the receiver's local config before the receiver skips `transform_weights()`. Mismatch raises rather than silently proceeds. The fallback is roughly ~50 LOC and lives in the MX checkpoint loader; it is preferable to skipping the precondition entirely.

### P2. Reset of `_weights_transformed` is the orchestrator's responsibility

See the "Lifecycle of `_weights_transformed`" subsection above. Subclasses set the flag at the end of `transform_weights()`. Resets happen in `ModelLoader.reload()` and in any other code path that re-binds tensors to untransformed bytes. Subclasses MUST NOT reset the flag themselves; that would couple transform idempotency to subclass-specific semantics and break the orchestrator-level contract.

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

### Phase status (2026-05-30)

| Phase | Status | Vehicle | Risk | Est. LOC | MX receiver value |
|:------|:-------|:--------|:-----|:---------|:------------------|
| Prep | ✅ Landed | TRTLLM-13077 | n/a | ~70 (delivered) | — |
| **Wave 1** | **Next** | TBD | LOW | ~165 | 0 (MX still publishes PRE-transform) |
| Wave 2 | Queued | TBD | HIGH | ~80 | partial (~60% of models become receiver-ready) |
| Wave 3 | Queued | TBD | HIGH | ~280 | full (100% of models receiver-ready) |
| Wave 4 | Queued | TBD (MX-side publisher flip + TRT-LLM receiver cutover) | MEDIUM | ~80 in-tree + MX-side flip | flip + per-model rollout |

**Migration callout (applies to every wave below that migrates an override):** when a subclass migrates from overriding `post_load_weights()` to overriding `setup_aliases()` / `transform_weights()` / `cache_derived_state()`, the old `post_load_weights()` override **must be removed**. Leaving it in place silently shadows the base-class orchestrator and causes the new staged calls to do nothing. The pattern is: (a) move each block of the old body into the appropriate new method, (b) delete the `def post_load_weights(self):` line, (c) verify by grepping the diff for any remaining `def post_load_weights` in the migrated class.

### Prep PR — ✅ landed as TRTLLM-13077

`[TRTLLM-13077][feat] Deocmpose post_load_weights()` introduced the contract surface without migrating any model: default no-op `setup_aliases()` / `transform_weights()` / `cache_derived_state()`, the `_weights_transformed` flag with the lifecycle documented above, helper walkers on `ModelLoader` (`_walk_transform`, `_walk_cache_state`, plus a backward-compat orchestrator), and protocol unit tests. Neither TRTLLM-12440's GMS RO branch nor the inflight MX-team refactor is blocked on the migration: each carries a `TODO(STAGED-HOOKS)` against this section and continues with its ad-hoc workaround until its respective wave below cuts it over.

Original prep-PR scope (kept for record):

1. **Define the contract via duck-typed helpers, not via inheritance.** `ModelLoader` already invokes `post_load_weights()` through `getattr(module, 'post_load_weights', None)` + `hasattr` checks — see [model_loader.py](../../../tensorrt_llm/_torch/pyexecutor/model_loader.py). The walkers follow the same pattern so that `Linear`, `Attention`, MoE, Mamba submodules can opt in by simply defining the method, without forced inheritance changes. Optionally provide a `StagedHooksMixin` for type-checking convenience, but do not require it.
2. Define the three per-module stages — `setup_aliases()`, `transform_weights()` (with `_weights_transformed` guard), `cache_derived_state()` — as documented method names with default no-ops on the existing base classes (`DecoderModelForCausalLM`, plus `nn.Module` defaults via the helper). Subclasses opt in by overriding any subset.
3. Provide a backward-compat `post_load_weights()` orchestrator on the base class that calls the three stages in order, so non-migrated subclasses see no behavior change. NOTE: this only helps subclasses that don't currently override `post_load_weights()`. Subclasses with existing overrides keep their old behavior until they explicitly migrate.
4. `_weights_transformed` flag introduced alongside the existing `_weights_removed` flag — orthogonal semantics:
   - `_weights_removed` = sleep/wake lifecycle (existing meaning).
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
- Migrate 7 model classes from `post_load_weights()` to `setup_aliases()`: `modeling_llama`, `modeling_deepseekv3`, `modeling_glm`, `modeling_exaone_moe`, `modeling_qwen3_moe`, `modeling_qwen3_next`, `modeling_gpt_oss`.
- Cut over the GMS RO branch in `tensorrt_llm/_torch/pyexecutor/model_loader.py` from the §7 workaround (full `post_load_weights()` on meta tensors before `materialize_module_from_gms()`) to the staged-hook protocol: `model.setup_aliases()` → `gms_backend.materialize_module(model)` → per-module `cache_derived_state()` walk.

**Bundling rationale (why W1 is one PR, not two):**
- Alias migration alone produces no user-visible change (top-level `setup_aliases()` invoked by the orchestrator default is bit-identical to the current top-level `post_load_weights()` walk).
- GMS-RO cutover alone is broken without `setup_aliases()` carrying real wiring — an empty default no-op would not reproduce today's §7 mitigation, and `materialize_module_from_gms()` would AttributeError as it does in the pre-mitigation state.
- Bundled, they form one cohesive vertical slice with end-to-end observable correctness on the GMS RO path. The change touches code paths that GMS-RO functional tests exercise directly.

**Blast radius:** GMS RO load path; alias wiring on the listed 7 model classes. Non-GMS paths (AUTO + HF, AUTO + MX) continue to use the backward-compat `post_load_weights()` orchestrator, which is bit-identical to today's behavior.

**Risk:** LOW.
- Default orchestrator preserves behavior for non-GMS paths.
- GMS-RO functional tests (introduced in TRTLLM-12440) catch regressions immediately.
- Alias wiring is structural Python — no tensor math, no numerical risk.

**MX-side value:** **0**. The merged TRTLLM-11851 MX behavior is publish-PRE-transform; receivers correctly run the full `post_load_weights()` on disk-loaded bytes. Wave 1 does not change this. No MX receiver cutover happens in Wave 1.

**Gate to Wave 2:**
- GMS RW/RO functional tests green.
- CI integration tests for the 7 migrated models green.
- Manual verification of `nn.Module.__setattr__` dedupe semantics on at least one model (alias re-assignment is the new idempotency contract).

### Wave 2 — Linear / Attention transform migration (~80 LOC, HIGH risk)

**Scope:**
- Migrate `tensorrt_llm/_torch/modules/linear.py` and `tensorrt_llm/_torch/modules/attention.py` from `post_load_weights()` to `transform_weights()` + `_weights_transformed` guard.
- **Quant-method callback decision** (deferred from prep PR): the existing internal callback `QuantMethod.post_load_weights(self, module)` is invoked from `Linear.post_load_weights(self)`. Default is (a) keep the quant-method callback name unchanged and have the migrated `Linear.transform_weights(self)` invoke `quant_method.post_load_weights(module)`. The no-arg vs `(self, module)` signature distinction already disambiguates them; renaming touches ~10 quant-method overrides for cosmetic gain. Revisit only if (b) renaming materially helps clarity during implementation.

**Blast radius:** Every model with a `Linear` or `Attention` module — i.e., the whole model zoo. Highly exercised code path; a regression here surfaces in essentially every CI integration test.

**Risk:** HIGH. Mitigations:
- Strict `_weights_transformed` idempotency contract: set on successful return of the transform body; reset only by orchestrator-managed code (`ModelLoader.reload()`, partial-fallback merge). Subclasses MUST NOT reset.
- Standalone PR. No other migrations bundled.
- CI must run the full integration suite on representative models per backend (FP8, NVFP4, BF16, INT8) before merge.
- Add a per-module idempotency unit test: calling `transform_weights()` twice produces a no-op on the second call.

**MX-side value:** **PARTIAL — ~60% of the model zoo.** Once Wave 2 lands, models whose only transform-affecting modules are Linear + Attention (Llama, Qwen, Mistral, and similar dense models — no MoE, no Mamba, no sparse attention) become *eligible* for publish-after-transform on the receiver side. Eligibility is gated by Wave 4 plumbing; no MX runtime behavior changes in Wave 2.

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

**Risk:** HIGH.
- Six quant-method overrides multiply the surface area for typos.
- MoE expert-slot bookkeeping (Category C, orchestrator-managed via `MoeLoadBalancer.finalize_model()`) is adjacent to and easy to confuse with weight transforms (Category B). Reviewers must explicitly verify the boundary.
- Mitigations: standalone PR; per-quant-method idempotency unit test; manual review on the Linear/Attention vs MoE callback boundary; full integration suite on at least one MoE model and one Mamba model per backend.

**MX-side value:** **FULL.** After Wave 3, every model is receiver-ready for publish-after-transform.

**Gate to Wave 4:**
- Full integration suite green on at least one MoE and one Mamba model per backend.
- Idempotency tests green for all 6 quant-method overrides.
- Manual sweep to confirm no model class is left with a stale `def post_load_weights(self)` that would shadow the staged hooks.

### Wave 4 — MX publish-after-transform flip + P1 fail-safe + receiver cutover (~80 LOC TRT-LLM + MX-side, MEDIUM risk)

**Scope:**
- **MX-side (out of TRT-LLM):** flip the MX publisher from publish-PRE-transform (current TRTLLM-11851 behavior) to publish-POST-transform. Tracked in the inflight MX-team Delegate-to-ModelExpress refactor proposal. ~5 LOC in MX checkpoint loader.
- **P1 fail-safe (~50 LOC, in TRT-LLM `mx_checkpoint_loader`):** add a backend-fingerprint check on the receiver side. Stores the publisher's canonical fingerprint (`attn_backend`, quant backend list, FP8 / NVFP4 fusion strategy, TP / PP / EP layout, model revision) alongside the weight payload; receiver compares against its local config before allowing any `transform_weights()` skip. Mismatch raises immediately rather than silently consuming incompatible weights. This is the in-tree answer to the homogeneity-assumption hazard surfaced on the MX-team refactor review.
- **MX receiver branch (~30 LOC) in `model_loader.py`:** cut over the MX path from the current full `post_load_weights()` walk to the staged-hook protocol: `model.setup_aliases()` → skip `transform_weights()` walk → per-module `cache_derived_state()` walk. The skip is conditional on (a) the per-model allow-list entry and (b) the P1 fingerprint match.
- **Per-model enable allow-list:** the receiver does not unconditionally skip `transform_weights()`. It consults an allow-list keyed by `(model_class, transform_protocol_version)`. Models migrate into the allow-list one-by-one as integration testing validates them.

**Blast radius:** MX-only path. No GMS impact. Affects only deployments that enable MX P2P checkpoint loading. Default-off for any model not in the allow-list — a deployment that upgrades to the Wave-4 code while running a not-yet-allow-listed model continues to receive PRE-transform bytes and runs the full receiver-side `post_load_weights()`, identical to today.

**Risk:** MEDIUM. The publish-after-transform flip is the very change that the inflight MX-team refactor's review flagged as unsafe; the P1 fail-safe is the in-tree answer to that critique. Risk is bounded by:
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

## Open questions to resolve during implementation

| # | Question | Default if unresolved |
|:--|:---------|:-----------------------|
| 1 | Where the protocol lives — `DecoderModelForCausalLM` only? `nn.Module` mixin? Per-class trait? | Duck-typed via `getattr/hasattr` on the `ModelLoader` walkers, mirroring the existing `post_load_weights()` walker pattern. Optionally add a mixin for type-checking convenience, but do not require subclasses to inherit from it — most existing overrides are on `Linear`, `Attention`, MoE, and Mamba submodules that don't share a common base. |
| 2 | Naming finalization — `setup_aliases` / `transform_weights` / `cache_derived_state`. | Keep these names. Alternatives considered: `_post_load_setup` / `_post_load_transform` / `_post_load_finalize` (more parallel but vaguer). |
| 3 | Should the orchestrator default invoke an existing override of `post_load_weights()` if a subclass has one but no granular methods? | Yes during transition, with a deprecation warning. After full migration, remove the back-compat path. |
| 4 | Speculative decoding draft model — is the `model.draft_model` walk handled correctly by the same per-stage walkers? | Yes — `for m in model.modules()` recurses into `draft_model`. No special-casing needed. (Applies to `transform_weights()` and `cache_derived_state()` walks; `setup_aliases()` is top-level so the draft model's own alias wiring is invoked separately if it has any.) |
| 5 | `ModelLoader.reload()` interaction. After `reload()` rebinds tensors, which stages need to re-run? | Reset `_weights_transformed=False` on every affected module; then run `transform_weights()` and `cache_derived_state()`. `setup_aliases()` is idempotent and need not be re-run unless the module tree itself changed. |
| 6 | (D) recomputation cost. For a 70B model, how expensive is `cache_derived_state` on real tensors? | Expect to be cheap (most cached state is small scalars / dtypes). Profile during the cutover PR. |

(Open question 6 from the prior revision — "Identity matching on the MX side" — has been promoted to hard precondition P1 above and removed from this table.)

## References

- **TRTLLM-13077** — `[TRTLLM-13077][feat] Deocmpose post_load_weights()` (merged). Prep PR that introduced the staged-hook contract surface, helper walkers, and `_weights_transformed` lifecycle without migrating any model. Vehicle for the "Prep" row in the phase-status table above.
- **TRTLLM-12440** — `[TRTLLM-12440][feat] Add GMS-only weight sharing support` (merged). RO-branch ordering trade-off documented inline; `TODO(STAGED-HOOKS)` to be added in the Wave 1 cutover.
- **TRTLLM-11851** — `[TRTLLM-11851][feat] Add MX-only P2P checkpoint loading support for TRTLLM` (merged). Establishes the current MX publish-PRE-transform contract that Wave 4 flips.
- Inflight MX-team `[None][refactor] Delegate MX checkpoint loading to ModelExpress` proposal — the publish-pre vs publish-post-transform discussion that surfaced the same architectural gap from the MX side. The homogeneity-assumption hazard raised on its review thread is the motivation for the P1 fail-safe in Wave 4.
- ai-dynamo/dynamo PR [#7053](https://github.com/ai-dynamo/dynamo/pull/7053) — upstream GMS prototype that originally surfaced and fixed the alias-resolution `AttributeError`. The mitigation contract ("call `model.post_load_weights()` before `materialize_module_from_gms()` to set up structural cross-references") is the source of TRT-LLM's current GMS RO ordering and the motivation for splitting alias wiring out as its own stage.
- [§7 of 05-challenges.md](05-challenges.md#7-module-path-resolution-gms-specific) — the existing in-tree write-up of the alias bug and the per-PR mitigation. After Wave 1 lands, §7 forward-links here for the holistic fix.
- [Module Path Resolution risk in §12-risks.md](12-risks.md) — risk row "Module path resolution (aliased layers)". The staged-hook protocol moves this risk from "MEDIUM, mitigated per PR" to "LOW, structurally fixed."
