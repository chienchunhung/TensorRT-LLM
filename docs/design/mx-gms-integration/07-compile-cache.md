# 7. Tiered Compile Cache

[< Back to Overview](README.md)

> **This section is new** — originally a sub-section of [§06 Executor Failover](06-executor-failover.md), the compile cache design was promoted to its own section because v3 benchmarks ([§11 Results](11-results-analysis.md)) show the compile cache is **required**, not optional, for MX+GMS to meet the <5s shadow-failover target.

## The Problem

The shadow activation budget in [§06](06-executor-failover.md#shadow-worker-lifecycle) targets <5s, but **warmup is not accounted for in that budget**. Benchmark data ([§11 Results & Analysis](11-results-analysis.md)) shows warmup takes **~43s on v3 code** (post-PR #12407) for Qwen 72B TP=8. This is a ~2.6× growth vs the pre-PR #12407 baseline (~16s) driven primarily by a new general warmup pass added by that PR:

| Warmup Phase | v3 Duration | v2 Duration | What It Does |
|:-------------|:---------|:---------|:-------------|
| General warmup (1st pass, shape specialization) | ~25s | ~0s (did not exist) | Forward pass over multiple shapes before CUDA graph capture (added by PR #12407) |
| Autotuner forward (1st pass) | ~1.5s | ~12s | Kernel selection (most work shifted to CUDA graphs on v3) |
| CUDA graphs (1st pass) | ~11s | ~0.7s | Graph capture (absorbs work that used to happen during autotuner on v2) |
| 2nd pass warmup | ~5s | ~4s | Second-shape specialization pass |
| **Total** | **~43s (v3)** | **~16s (v2)** | |

Without a compile cache, shadow activation on v3 code would take **~65s** (21s worker init + ~1-3s KV cache + ~43s warmup) — an order of magnitude over the <5s target. This gap was ~17-19s on v2 code; the v3 warmup regression makes the cache essentially mandatory.

**Why the shadow can't pre-warm during shadow mode:** Warmup executes model forward passes, which require KV cache to be allocated. The shadow intentionally does NOT allocate KV cache (to minimize GPU memory footprint). No KV cache → no forward passes → no warmup.

## Solution: Tiered Compile Cache

A two-tier cache hierarchy, analogous to CPU cache (fast/volatile) backed by disk (slow/durable):

```
Shadow activation compile lookup:
  Tier 1: GMS compile_cache tag (GPU memory, ~ms import)   → fast, volatile (survives process crash, not node reboot)
  Tier 2: Disk compile cache (filesystem, ~0.5-2s load)    → slow, durable (survives node reboot)
  Tier 3: Full recompile (~43s on v3, ~16s on v2)          → cold start, last resort
```

The primary writes to **both tiers** during its initial warmup. The shadow reads from whichever is available, in priority order.

## GMS Tag Model (Extended)

This adds a third GMS tag per GPU, fitting naturally into the existing per-GPU per-tag architecture:

```
Per-GPU GMS tags:
  weights        → model parameters (RW/RO sharing, long-lived)
  kv_cache       → KV cache blocks (released on sleep, allocated on activation)
  compile_cache  → compiled kernels + autotuner results (written once by primary, imported by shadow)
```

On an 8-GPU node, this means 24 GMS processes (8 GPUs × 3 tags) instead of 16.

| Tag | Written by | Read by | Lifecycle | Survives |
|:----|:-----------|:--------|:----------|:---------|
| `weights` | Primary (RW) | Shadow (RO import) | Long-lived; shared continuously | Process crash ✅, node reboot ❌ |
| `kv_cache` | Active worker | Same worker only | Released on sleep/demotion; allocated on activation | Process crash ✅, node reboot ❌ |
| `compile_cache` | Primary after warmup | Shadow on activation | Written once; imported on demand | Process crash ✅, node reboot ❌ |

The `materialize_with_tag("compile_cache")` / `release_with_tag("compile_cache")` mapping is documented alongside the other sleep/wake operations in [§06 Executor Integration](06-executor-failover.md#mapping-to-existing-sleep-wake).

## What Goes in Each Tier

| Artifact | GMS Tier (Tier 1) | Disk Tier (Tier 2) | Notes |
|:---------|:-------------------|:-------------------|:------|
| `torch.compile` compiled kernels | Serialized kernel objects | `~/.cache/torch/inductor/` (automatic) | Deterministic given same model + config |
| Autotuner results (kernel configs) | Serialized config map | `TRTLLM_AUTOTUNER_CACHE_DIR` | Map from op signature → optimal kernel config |
| CUDA graph templates | **Cannot share** | **Cannot share** | Tied to specific memory addresses; must recapture on activation |

**Key insight:** CUDA graphs must always be recaptured after KV cache allocation because they encode specific GPU memory addresses. The compile cache eliminates the expensive compilation cost (v3: ~25s general warmup + ~1.5s autotuner; v2: ~12s autotuner); CUDA graph recapture remains at ~0.5-1s with pre-compiled kernels regardless of version.

## Activation Warmup Budget (Cache-Warm)

v3 numbers are primary. v2 numbers shown for reference.

| Step | Without Cache (v3 / v2) | With Disk Cache (Tier 2) | With GMS Cache (Tier 1) |
|:-----|:-------------|:------------------------|:------------------------|
| Load compiled kernels + general warmup artifacts | ~27s (v3 recompile) / ~12s (v2) | ~0.5-1s (disk read) | ~10ms (GMS import) |
| CUDA graph recapture | ~11s (v3) / ~4s (v2) | ~0.5-1s (compiled kernels ready) | ~0.5-1s (compiled kernels ready) |
| 2nd pass warmup | ~5s (v3) / ~4s (v2) | ~0.5s (shapes cached) | ~0.1s |
| **Warmup total** | **~43s (v3) / ~16s (v2)** | **~1.5-2.5s** | **~0.6-1.1s** |

The compile cache collapses v3's ~43s warmup to ~1.5-2.5s on disk or ~0.6-1.1s on GMS — the ratio is roughly the same as on v2, so the cache design works equally well against both code generations.

The cache-warm warmup step slots into the [shadow activation sequence](06-executor-failover.md#shadow-mode-implementation) as step 3, immediately after KV cache allocation.

## Implementation Phasing

| Phase | Scope | Compile Cache |
|:------|:------|:-------------|
| Phase 2 (GMS integration) | Shadow holds weights only | **Tier 2 (disk) only** — relies on shared filesystem between primary and shadow on same node |
| Phase 3+ (extension) | Shadow imports compile artifacts from GMS | **Tier 1 (GMS) + Tier 2 (disk)** — full hierarchy |

Tier 2 (disk cache) is sufficient for the initial implementation because primary and shadow are always co-located on the same node and share the filesystem. Tier 1 (GMS) is a performance optimization that tightens the failover budget and provides resilience against filesystem latency.

## Open Design Questions

See [§14 Open Questions](14-open-questions.md) for the full list. Compile-cache-specific items:

- **PR #12407 confirmation with TRT-LLM team.** Whether the ~27s warmup regression is intended or can be reverted/gated. If reverted, Tier 2 becomes optional and Tier 1 becomes unnecessary.
- **Serialization format for Tier 1.** `torch.compile` artifacts aren't designed for cross-process import; may need a wrapper.
- **Cache invalidation on model config change.** Key the cache on `(model_hash, config_hash, torch_version, TP/PP/EP shape)`.
- **Phase 3 scope.** Whether Tier 1 ships together with KV cache extension ([§09](09-kv-cache-extension.md)) or earlier.
