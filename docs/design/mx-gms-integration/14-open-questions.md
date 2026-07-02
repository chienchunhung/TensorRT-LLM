# 14. Open Questions & Discussion

[< Back to Overview](README.md)

> **Status note:** [§18](18-dynamo-pr11000-gaps.md) resolves the current GMS lifecycle, ownership, and PR
> ordering. GMS API names and lock-upgrade questions below predate the pinned contract and should be treated as
> historical unless §18 still lists them as open.

Consolidated list of open questions, pending verifications, and discussion items surfaced across the design. Each item notes the section it connects back to and what it blocks or would change.

---

## A. Performance & Measurement

### A1. PR #12407 confirmation with TRT-LLM team

**What:** Confirm whether the ~27s warmup regression introduced by [PR #12407](https://github.com/NVIDIA/TensorRT-LLM/pull/12407) ("Refactor warmup orchestration in MTP") is an intended correctness cost or an unintended side effect.

**Why it matters:** If the new general warmup pass can be gated (e.g., only when `torch.compile` is enabled, or only for MTP models), then on v3 code (a) the warmup floor returns to ~16s, (b) compile cache reverts to being a "nice extension" rather than required, and (c) Tier 1 (GMS-backed cache) becomes unnecessary for Phase 2.

**See:** [§11 Analysis §6](11-results-analysis.md#6-warmup-overhead-regression-from-pr-12407-new-in-v3), [§07 Open Design Questions](07-compile-cache.md#open-design-questions).

**Action:** Raise with PR authors / TRT-LLM team. Pending.

### A2. v2-on-v3-node isolation run

**What:** Run the v2 binary (TRT-LLM 1.3.0rc11, pre-#12407) on the v3 benchmark node `umb-b300-dp-186`, or the v3 binary on the v2 node `umb-b300-dp-199`, to cleanly separate **environment** (NFS throughput) from **code** (PR #12407 warmup change) in the v2→v3 delta.

**Why it matters:** Currently we attribute the +168s S2 prefetch delta to environment and +26s warmup delta to code, but this is inference, not a controlled measurement. A small fraction of the prefetch delta could be code-induced (e.g., different mmap pattern). Needed before publishing v2-vs-v3 numbers externally.

**See:** [§11 Dataset Summary](11-results-analysis.md#dataset-summary) and [Insight #6](11-results-analysis.md#6-warmup-overhead-regression-from-pr-12407-new-in-v3) for the current v2→v3 attribution.

**Action:** One benchmark run on either node with the cross-version binary.

### A3. v3 re-runs for pending configurations

Configs measured on v2 but not yet re-verified on v3:

| Config | Expected v3 behavior | Priority |
|:-------|:---------------------|:---------|
| **Qwen 72B S3 / DS 70B S3** (warm cache) | ~73s / ~76s (v2 + warmup delta) | High — S3 is the cache-warm baseline that MX/GMS+compile cache target |
| **S1 remote cold** (all models) | Same structural story; S1 < S2 on v3 even with slow CDN | Low — HF rate-limiting concerns on v3 node |
| **Autotuner OFF** (Parts 3) | Smaller effect than v2 (autotuner already only ~1.5s on v3) | Medium — confirms finding generalizes |
| **Serving config sensitivity** (Part 4 D1/D2) | Similar relative effect; absolute numbers scale with v3 warmup floor | Medium |
| **S4 local NVMe cold** | 20–40s prefetch for 145GB; ceiling for MX comparison | Medium — first v3 run, needed for MX benchmark comparison |

**See:** [§10 Summary](10-methodology.md#summary) for what's been run, [§10 Scenario Coverage and Gaps](10-methodology.md#scenario-coverage-and-gaps).

## B. Compile Cache (§07)

### B1. Serialization format for Tier 1 (GMS-backed)

**What:** `torch.compile` artifacts aren't designed for cross-process import. Need a wrapper that can (a) serialize compiled kernel objects + autotuner config maps into a GMS memory region, and (b) import them in the shadow process without re-triggering compilation.

**Why it matters:** Tier 2 (disk cache) uses the existing `~/.cache/torch/inductor/` and `TRTLLM_AUTOTUNER_CACHE_DIR` mechanisms. Tier 1 needs custom marshalling. If infeasible, Tier 1 reduces to ~ms-import of a buffer that still triggers ~0.5s deserialization — closer to Tier 2 disk speed.

**See:** [§07 Open Design Questions](07-compile-cache.md#open-design-questions).

### B2. Cache invalidation

**What:** Cache key must include `(model_hash, config_hash, torch_version, TP/PP/EP shape, serving_config)`. Open: is `serving_config` fine-grained enough (max_batch_size, max_num_tokens, max_seq_len all change CUDA graph variants)? And do we need to version by TRT-LLM commit hash for robustness?

**See:** [§07 Open Design Questions](07-compile-cache.md#open-design-questions).

### B3. Phase 3 scope for Tier 1

**What:** Whether Tier 1 (GMS compile_cache tag) ships in Phase 3 alongside the KV cache extension ([§09](09-kv-cache-extension.md)), or earlier as a compile-cache-only follow-up.

**Why it matters:** Dependent on A1 (PR #12407 outcome). If warmup reverts to ~16s, Tier 2 disk alone is sufficient and Tier 1 becomes optional.

---

## C. Integration & API

### C1. GMS API stability

**What:** The [GMS prototype PR #7053](https://github.com/ai-dynamo/dynamo/pull/7053) demonstrates the per-GPU per-tag model but the API surface (`gms_client.import`, `upgrade_lock`, `release_lock`, `heartbeat`) is not yet frozen. Integration code will need to track API changes.

**See:** [§05 Challenges](05-challenges.md), [§12 Risk Assessment](12-risks.md).

### C2. vLLM comparison plan

**What:** Test 6 in [§10](10-methodology.md#test-6-vllm-comparison--not-yet-executed) compares `TRT-LLM --checkpoint-format mx` against `vLLM --load-format mx`. Open: which vLLM version to benchmark against (MX support landed in a specific release), and whether comparison should include the post-load warmup too or just weight-load time (vLLM doesn't have the ~25s general warmup pass).

**See:** [§10 Test 6](10-methodology.md#test-6-vllm-comparison--not-yet-executed), [§12 vLLM comparison](12-risks.md).

### C3. Multi-instance sharing relevance

**What:** UC4 (multi-model / LoRA sharing via GMS zero-copy) is listed as a niche use case for smaller models. Open: is there concrete demand, or does this stay as a design-intent-only feature?

**Why it matters:** Test 3 (memory overhead validation) currently measures 1 active + 1 shadow. Multi-instance (N>2) testing is only relevant if UC4 has a concrete consumer.

**See:** [§02 UC4](02-problem-and-goals.md#uc4-multi-model--lora-sharing-niche).

---

## D. Operational & Deployment

### D1. Shadow co-location constraint

**What:** Phase 2 assumes primary and shadow are always co-located on the same node (shared filesystem for Tier 2 disk cache, same-node GMS tags for weights). Open: how does this compose with Kubernetes pod scheduling, where primary and shadow may land on different nodes unless explicit anti-affinity is expressed as "same-node affinity"?

**See:** [§06 Executor Integration](06-executor-failover.md), [§07 Implementation Phasing](07-compile-cache.md#implementation-phasing).

### D2. MPI pool cold-start optimization

**What:** The ~21s MPI worker cold start is on the critical path for all S2/S3 scenarios without concurrent server work. [§11 Worker Init Investigation](11-results-analysis.md#worker-init-investigation-results) showed simple warm-up dispatch doesn't help. Open: is the underlying `mpi4py.futures.MPIPoolExecutor` lazy-spawn behavior modifiable (config? custom pool?), and is it worth the effort given MX P2P would hide most of it organically?

**See:** [§11 Worker Init Investigation](11-results-analysis.md#worker-init-investigation-results).

---

## E. Items Deferred from Earlier Drafts

### E1. KV cache sharing — deferred to KVBM, not GMS

Earlier drafts proposed GMS-backed KV cache options; those are superseded. KV cache is out of GMS's scope — Dynamo's KVBM already covers tiered KV storage and cross-node sharing via the KV Cache Connector API. Phases 1–3 are scoped to ensure non-interference with a future KVBM connector; the KVBM integration itself is Phase 4+ and tracks the Dynamo roadmap. See [§09 KV Cache Extension Path](09-kv-cache-extension.md).

### E2. Compile cache sharing via MX

Noted in [§02 Non-Goals](02-problem-and-goals.md#non-goals) as future work. If compile cache needs to be shared across nodes (not just within a node via GMS), MX-style P2P distribution of the cache artifacts would be a natural extension but is not in the current scope.

### E3. Legacy TensorRT engine backend

Explicitly out of scope ([§02](02-problem-and-goals.md#non-goals)). MX/GMS integration targets PyTorch backend only.
