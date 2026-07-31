Source URL: https://raw.githubusercontent.com/chienchunhung/TensorRT-LLM/docs-and-plans/docs/design/mx-gms-integration/21-mx-readiness-gaps-and-model-family-plan.md
Title: 21. ModelExpress Readiness Gaps and Model-Family Expansion Plan

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 21. ModelExpress Readiness Gaps and Model-Family Expansion Plan

[< Back to README](README.md)

**Status:** Active delivery plan

**Last updated:** 2026-07-31

**Epic:** [TRTLLM-11901](https://jirasw.nvidia.com/browse/TRTLLM-11901)

**Umbrella task:** [TRTLLM-14727](https://jirasw.nvidia.com/browse/TRTLLM-14727)

**Execution runbook:** [§20 ModelExpress End-to-End Verification Plan](20-mx-e2e-verification-plan.md)

The [model-family delivery tracker](#3-model-family-delivery-tracker) is the focal point of this document. Update its
PR, status, dependency, and date columns whenever work advances. The surrounding sections explain why each row exists
and what evidence is required to close it.

---

## 1. Current Position

| Milestone | State | What it means |
|:--|:--|:--|
| Shared MX integration | **Landed** | #15641 supplies optional packaging, MX 0.4.1 integration, local-server convenience, and complete-disk fallback. |
| Artifact and layout identity | **Landed** | #16159 binds immutable checkpoint content; #16458 binds exact profiles and transform-layout ABI in SourceIdentity v3. |
| Llama profile | **Enabled on `main`** | `llama-for-causal-lm-target-v1` is the only landed exact profile. Repeat §20 on current `main` before making the R1 live-preview claim. |
| Qwen2/Qwen2.5 dense | **In flight** | #16974 is the first non-Llama family PR and introduces the reusable runtime-constraint matrix. |
| Qwen3 dense | **In flight, stacked** | #17142 adds a distinct Qwen3 profile/ABI and Q/K-normalized lifecycle coverage on top of #16974. |
| Next sequential target | **Mixtral** | Use the basic MoE canary to prove expert packing and TP/EP before shared-expert Qwen MoE. |

An open PR, staged-hook migration, or passing unit suite does not by itself make a profile supported. Count a profile
only after its exact registry row is merged and its required real donor/receiver evidence is retained.

## 2. Why Per-Family Work Remains

The basic integration answers a transport question: **can TRT-LLM discover a compatible donor, transfer
post-transform tensors, bind them into a receiver, and fall back safely?** It does not establish that every model
produces the same post-transform tensor contract or can complete the same receiver lifecycle.

Post-transform MX payloads are runtime layouts, not portable Hugging Face checkpoint files. Their names, shapes,
aliases, semantics, and required finalization depend on the exact root model and runtime configuration. A receiver that
reruns an irreversible transform can corrupt already-transformed weights; one that skips an alias or derived-state
step can load successfully and still produce incorrect output.

| Source of the gap | Why shared infrastructure is insufficient | Representative examples |
|:--|:--|:--|
| Model-specific transforms | Families fuse, pack, transpose, or requantize different tensors. | Qwen fused QKV/gate-up, MoE expert packing, DeepSeek MLA projections. |
| Root lifecycle differences | Shared modules may be staged while the root still has aliases, legacy `post_load_weights()`, derived buffers, or local finalization. | DeepSeek V4 structural work, Qwen3 Q/K norm, Mamba cache reconstruction. |
| Ambiguous architecture identity | One implementation class can serve multiple architectures, while subclasses can have different layouts. Class resemblance must not imply support. | DeepSeek V3/V3.2, Kimi, and GLM DSA share paths; Qwen3.5 subclasses Qwen3-Next. |
| Runtime-dependent layout | Dtype, quantization, backend, parallel mapping, RoPE, tied embeddings, and speculation can change transferred state. | TP/EP rank-local tensors, alternate MoE backends, YaRN, MTP. |
| Component ownership | Multimodal and target-plus-draft models load independent components that need separate identity and atomic fallback. | Qwen-VL vision/language components, Kimi K2.5 MoonViT plus language model. |
| Evidence boundary | Tiny lifecycle tests do not prove publication, rank matching, NIXL writes, no-disk reception, or failure recovery. | Real exact-token donor/receiver and negative-control runs remain mandatory. |

This creates three distinct stages:

1. **Migration:** put lifecycle work in `setup_aliases()`, `transform_weights()`, and `cache_derived_state()`.
2. **Qualification:** prove full loading and staged reception are equivalent for one exact profile and runtime row.
3. **Enablement:** add only that proven row to the registry and public support matrix.

Most shared migration is complete. The remaining work is primarily per-family audit, targeted cleanup where the audit
finds a gap, exact-profile definition, and qualification. Some roots, such as DeepSeek V4 and multimodal wrappers,
still require larger lifecycle or component-protocol work before ordinary qualification can begin.

## 3. Model-Family Delivery Tracker

**How to read this table:** `Delivery status` combines actual GitHub and dependency state. Jira workflow state is
summarized here instead of repeated in every row because, as of 2026-07-31, all 20 tasks remain `To Do` and unassigned
even though two draft PRs exist. `Updated` is the latest verified Jira/PR update date. `Not opened` means no
implementation PR has been identified.

| Wave / item | Jira | PR | Delivery status | Scope and purpose | Depends on | Updated |
|:--|:--|:--|:--|:--|:--|:--|
| A1 - Qwen2/Qwen2.5 dense | [TRTLLM-14879](https://jirasw.nvidia.com/browse/TRTLLM-14879) | [#16974](https://github.com/NVIDIA/TensorRT-LLM/pull/16974) (draft) | **In flight** | First non-Llama profile, `qwen2-for-causal-lm-bf16-target-v1`; adds the shared runtime-constraint matrix. | #16458 foundation | 2026-07-31 |
| A2 - Qwen3 dense | [TRTLLM-14880](https://jirasw.nvidia.com/browse/TRTLLM-14880) | [#17142](https://github.com/NVIDIA/TensorRT-LLM/pull/17142) (stacked draft) | **In flight** | `qwen3-for-causal-lm-bf16-target-v1`; distinct ABI and Q/K-normalized attention lifecycle. | A1 | 2026-07-31 |
| A-parallel - Mistral dense | [TRTLLM-14881](https://jirasw.nvidia.com/browse/TRTLLM-14881) | Not opened | **Planned, parallel** | Independent dense root; audit mapper, sliding-window, and RoPE behavior instead of inheriting Llama support. | #16458 foundation | 2026-07-31 |
| B1 - Mixtral MoE canary | [TRTLLM-14882](https://jirasw.nvidia.com/browse/TRTLLM-14882) | Not opened | **Next** | First basic MoE profile; isolate expert packing, router state, backend choice, and TP/EP mapping. | A1/A2 runtime matrix | 2026-07-31 |
| B2 - Qwen3 MoE | [TRTLLM-14883](https://jirasw.nvidia.com/browse/TRTLLM-14883) | Not opened | **Planned** | Strategic shared-expert profile; cover router, shared experts, next-layer aliases, and local finalization. | A2 and B1 | 2026-07-31 |
| B3 - Qwen2 MoE | [TRTLLM-14884](https://jirasw.nvidia.com/browse/TRTLLM-14884) | Not opened | **Planned** | Separate Qwen2 mapper, expert layout, ABI, and runtime matrix. | A1 and B2 | 2026-07-31 |
| B3 - GLM4 MoE | [TRTLLM-14885](https://jirasw.nvidia.com/browse/TRTLLM-14885) | Not opened | **Planned** | Custom loader, Q/K-normalized attention, expert/shared-head layout, aliases, and derived state. | B2 | 2026-07-31 |
| C1 - DeepSeek V3 | [TRTLLM-14886](https://jirasw.nvidia.com/browse/TRTLLM-14886) | Not opened | **Planned** | First narrow target-only MLA/MoE profile with explicit architecture identity and TP/EP rows. | B2 | 2026-07-31 |
| C2 - DeepSeek V3.2 | [TRTLLM-14887](https://jirasw.nvidia.com/browse/TRTLLM-14887) | Not opened | **Planned** | Distinct config profile despite sharing `DeepseekV3ForCausalLM`; keep Kimi/GLM disabled. | C1 | 2026-07-31 |
| C3 - Kimi K2 text | [TRTLLM-14888](https://jirasw.nvidia.com/browse/TRTLLM-14888) | Not opened | **Planned** | Explicit Kimi identity, mapper, router/config, YaRN, artifact, and output qualification. | C2 | 2026-07-31 |
| C3 - GLM DSA | [TRTLLM-14889](https://jirasw.nvidia.com/browse/TRTLLM-14889) | Not opened | **Planned** | Preserve GLM identity and qualify sparse indexer/attention state without collapsing into DeepSeek V3.2. | C1/C2 | 2026-07-31 |
| D1 - Qwen3-Next | [TRTLLM-14891](https://jirasw.nvidia.com/browse/TRTLLM-14891) | Not opened | **Planned** | First hybrid attention/Mamba profile; reconstruct caches and prove repeated-decode stability. | A2 and B2 as applicable | 2026-07-31 |
| D2 - Qwen3.5 dense | [TRTLLM-14892](https://jirasw.nvidia.com/browse/TRTLLM-14892) | Not opened | **Planned** | Exact wrapper profile for Qwen3.5 config normalization, mapper, aliases, and hybrid state. | D1 | 2026-07-31 |
| D3 - Qwen3.5 MoE | [TRTLLM-14893](https://jirasw.nvidia.com/browse/TRTLLM-14893) | Not opened | **Planned** | Combine exact Qwen3.5 hybrid-state and shared-expert MoE qualification. | D1 and B2 | 2026-07-31 |
| E - DeepSeek V4 | [TRTLLM-14894](https://jirasw.nvidia.com/browse/TRTLLM-14894) | Not opened | **Planned** | Migrate remaining root lifecycle, then qualify V4 sparse indexer, engram, HC mapping, MLA, and MoE state. | C1 and C3 GLM DSA | 2026-07-31 |
| F0 - Multimodal component contract | [TRTLLM-14895](https://jirasw.nvidia.com/browse/TRTLLM-14895) | Not opened | **Planned, parallel** | Define ordered component identity, transfer scope, atomic commit, strict result, and complete fallback. | #16159/#16458 foundation | 2026-07-31 |
| F1 - Qwen2-VL | [TRTLLM-14896](https://jirasw.nvidia.com/browse/TRTLLM-14896) | Not opened | **Planned** | First multimodal canary; qualify exact outer/component scope with text/image and no-disk evidence. | A1 and F0 | 2026-07-31 |
| F2 - Qwen3-VL | [TRTLLM-14897](https://jirasw.nvidia.com/browse/TRTLLM-14897) | Not opened | **Planned** | Distinct Qwen3-VL outer root, vision mapper, language component, aliases, and media outputs. | A2, F0, and F1 | 2026-07-31 |
| F3 - Qwen3-VL MoE | [TRTLLM-14898](https://jirasw.nvidia.com/browse/TRTLLM-14898) | Not opened | **Planned** | Combine multimodal atomicity with Qwen3 shared-expert state and TP/EP evidence. | B2 and F2 | 2026-07-31 |
| F3 - Kimi K2.5 | [TRTLLM-14900](https://jirasw.nvidia.com/browse/TRTLLM-14900) | Not opened | **Planned** | Declare MoonViT/language scope and qualify atomic image/video reception for every transferred component. | C3 Kimi K2, F0, and VL canary | 2026-07-31 |

The umbrella task TRTLLM-14727 tracks the overall expansion, but each implementation PR should cite its itemized task.
Update Jira status and assignee when work starts so Jira tracking and actual delivery state converge.

### Integration watch

[#17029](https://github.com/NVIDIA/TensorRT-LLM/pull/17029) is a parallel MX transport-delegation refactor. It does
not replace TRT-LLM-owned profile gating, transform ABI, lifecycle finalization, or family evidence. If it lands first,
rebase the active family stack and re-prove those contracts.

## 4. What It Takes to Support a Family

Each tracker row should deliver one or more exact support profiles, not a family-wide wildcard.

| Step | Required work | Exit evidence |
|:--|:--|:--|
| 1. Freeze the claim | Record root class, pre-construction architecture/model type, artifact, transfer scope, dtype/quantization, backends, parallel mapping, layout flags, and speculation/components. | A small, explicit runtime row that can be matched before publication and P2P. |
| 2. Audit and clean up lifecycle | Inventory every reachable staged/legacy hook, alias, transform, derived buffer, and process-local finalizer, including backend- and quantization-selected modules. | All structural, irreversible, derived, and local work is assigned to the correct phase. |
| 3. Define compatibility | Add an exact registry profile, structured rejection reasons, immutable ArtifactIdentity, and a new transform ABI when tensor semantics/finalization differ. | Unsupported roots, subclasses, runtime rows, artifacts, ranks, scopes, protocols, and ABIs reject before P2P. |
| 4. Prove lifecycle equivalence | Compare deterministic full load with staged reception for tensor layout/value, aliases, guards, derived state, and outputs; assert receiver transforms are skipped. | Family-level positive tests plus an unregistered-root and runtime-negative matrix. |
| 5. Prove real transfer | Run HF baseline, donor, full receiver, and blocked-shard receiver on real GPUs; inject identity, partial-transfer, and service failures. | Exact outputs, no-disk proof, per-rank transfer evidence, and one complete declared fallback or required failure. |
| 6. Enable and retain | Add the exact profile and public support row only after Steps 1-5, then add durable CI. | Merged code, focused/full CI, permanent small-fixture GPU gate, scheduled representative checkpoint, and archived evidence. |

### Initial support boundary

Start conservatively: target-only, one canonical dtype or production quantization, default selected backends, TP1/TP2,
and no MTP/speculation. Add TP/EP for MoE. Quantization variants, alternate backends, PP/CP/attention DP, tied
embeddings, YaRN, MTP, and multimodal scopes are separate rows unless directly tested.

### Fail-closed invariants

1. Use one qualification decision before donor publication and receiver P2P.
2. Bind immutable checkpoint content and transform ABI into SourceIdentity.
3. Never rerun irreversible transforms on post-transform tensors.
4. Rebuild aliases, derived state, and process-local state on the receiver.
5. Never mix a partial transfer with raw checkpoint tensors; discard staged state and perform one complete fallback.
6. Count support only with exact-output and no-disk real-transfer evidence.

## 5. Cross-Cutting Readiness Work

These items apply across families and should not be duplicated in every family PR.

| Track | Remaining work | Readiness effect |
|:--|:--|:--|
| Current-main Llama evidence | Repeat §20 with production wheel, canonical snapshot, ArtifactIdentity/ABI controls, and normal publisher liveness. | Closes R1 preview evidence. |
| Persistent GPU qualification | Pre-merge small fixtures plus scheduled representative checkpoints for each advertised category. | Prevents one-time family qualification from drifting. |
| Transaction and observability | Stage the full tensor set before commit; failure-inject transfer/alias/derived phases; emit terminal reason and add `required` mode. | Makes fallback safe and visible. |
| Cross-node | Two-node production NIC/NIXL run with rank mapping, timeout, and failure controls. | Required before any RDMA or cross-node claim. |
| MX API and concurrency | Replace private/process-global compatibility state with a public versioned per-load API; stress concurrent models/ranks. | Required for maintainable version ranges and multi-model use. |
| Operations and SLOs | Define external-service ownership, authentication, cleanup, liveness, startup/resource measurements, and p50/p95 targets. | Required for R3 production support. |
| MX-to-GMS composition | Qualify only after §18's native GMS committed-layout contract exists. | Separate track; does not block standalone family work. |

## 6. Readiness Gates

| Level | Claim | Exit gate |
|:--|:--|:--|
| R0 - Foundation | Optional MX integration is installed and guarded safely. | #15641, #16159, and #16458 merged; unsupported or mismatched loads reject before P2P. **Complete.** |
| R1 - Llama preview | One content-bound Llama profile works end to end. | Current-main §20 with exact output, no-disk receiver, artifact/ABI mismatch, and fallback controls. |
| R2 - Multi-family beta | Representative dense, MoE, and MLA/DSA families work. | Llama, Qwen dense, one Qwen MoE, one DeepSeek/MLA, and one GLM or Kimi text profile pass declared matrices. |
| R3 - Supported feature | MX has a maintainable production contract. | Stable API, persistent GPU CI, cross-node qualification, transaction/observability, concurrency proof, and SLOs. |

Speculative decoding, multimodal wrappers, managed lifecycle, and MX-to-GMS composition remain off unless separately
qualified.

<details>
<summary>Historical Llama evidence</summary>

The 2026-07-15 §20 experiment used TinyLlama-1.1B-Chat-v1.0 BF16 on B300 with TRT-LLM `752c05c9af` and MX 0.4.1 plus
a local canonical-wire-catalog patch. TP1 and TP2 donor/receiver, no-shards, exact-output, rank matching, and
parallel-mismatch fallback passed. TP4 was slower than HF and is not a supported performance claim.

- TP1 baseline and mismatch-fallback hash: `24f0cd36473e4b1a53156c26abdcc4a9db78f662aa51fad47373b1c8387a9b8d`
- TP2 baseline/donor/receivers hash: `45841007c85b4496a6388cbf52d433e741ffb8dfb3048ae78b1de1e14241cabc`

This predates the final merged foundation and remains historical evidence, not the current-main R1 proof.

</details>

## 7. References

- [§16 Staged Post-Load Hooks](16-staged-post-load-hooks.md)
- [§18 GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-concrete-pr-plan.md)
- [§20 ModelExpress End-to-End Verification Plan](20-mx-e2e-verification-plan.md)
- [#15014 - Wave 1](https://github.com/NVIDIA/TensorRT-LLM/pull/15014)
- [#15288 - Wave 2](https://github.com/NVIDIA/TensorRT-LLM/pull/15288)
- [#15386 - Wave 3](https://github.com/NVIDIA/TensorRT-LLM/pull/15386)
- [#15387 - Wave 4](https://github.com/NVIDIA/TensorRT-LLM/pull/15387)
- [#15432 - Wave 5](https://github.com/NVIDIA/TensorRT-LLM/pull/15432)
- [#15641 - Optional standalone MX integration](https://github.com/NVIDIA/TensorRT-LLM/pull/15641)
- [#16159 - ArtifactIdentity and SourceIdentity v2](https://github.com/NVIDIA/TensorRT-LLM/pull/16159)
- [#16458 - Exact-profile and transform-ABI foundation](https://github.com/NVIDIA/TensorRT-LLM/pull/16458)
- [#16974 - Qwen2 dense qualification](https://github.com/NVIDIA/TensorRT-LLM/pull/16974)
- [#17029 - ModelExpress strategy delegation](https://github.com/NVIDIA/TensorRT-LLM/pull/17029)
- [#17142 - Qwen3 dense qualification](https://github.com/NVIDIA/TensorRT-LLM/pull/17142)
