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

**Tracking:** [TRTLLM-11901](https://jirasw.nvidia.com/browse/TRTLLM-11901)

**Execution runbook:** [§20 ModelExpress End-to-End Verification Plan](20-mx-e2e-verification-plan.md)

This document is the source of truth for three questions:

1. Which exact ModelExpress (MX) model profiles are supported today?
2. Which profile should be implemented next, and what does it depend on?
3. What evidence is required before a profile counts as supported?

It is not a generic model-support matrix. Every claim here is limited to an exact root, artifact, runtime layout,
transfer scope, and transform ABI.

---

## 1. Executive Snapshot

| Area | Status | Next decision or gate |
|:--|:--|:--|
| Integration and identity foundation | **Landed** in #15641, #16159, and #16458 | Keep unsupported profiles fail-closed and preserve exact artifact/layout identity. |
| Llama baseline | **Enabled on `main`**, but current-main live evidence is incomplete | Repeat §20 on current `main` before making the R1 preview claim. |
| Qwen2/Qwen2.5 dense | **Draft #16974**, tracked by TRTLLM-14879 | Finish review and real TP1/TP2 donor/receiver qualification. |
| Qwen3 dense | **Stacked draft #17142**, tracked by TRTLLM-14880 | Rebase after #16974 settles, then run the same real-GPU gate with Q/K-normalized lifecycle checks. |
| Next sequential family | **Mixtral**, tracked by TRTLLM-14882 | Start after the dense Qwen stack establishes the reusable runtime matrix. |
| Parallel dense family | **Mistral**, tracked by TRTLLM-14881 | May proceed independently from the MoE sequence. |
| Production readiness | **Open** | Permanent GPU CI, two-node RDMA, structured results, strict mode, stable MX API, concurrency, and SLOs. |

**Support accounting rule:** a profile is not supported merely because its modules have staged hooks, its PR is open,
or its unit tests pass. Count it only after the exact profile is merged and the required real donor/receiver evidence is
retained.

### Readiness levels

| Level | Claim | Exit gate |
|:--|:--|:--|
| R0 - Foundation | Optional MX integration is installed and guarded safely. | #15641, #16159, and #16458 are merged; unsupported or mismatched loads reject before P2P. **Complete.** |
| R1 - Llama preview | One content-bound Llama profile works end to end. | Current-main §20 run with exact output, no-disk receiver, ArtifactIdentity mismatch, ABI mismatch, and fallback controls. |
| R2 - Multi-family beta | Representative dense, MoE, and MLA/DSA families work. | Llama, Qwen dense, one Qwen MoE, one DeepSeek/MLA, and one GLM or Kimi text profile pass their declared matrices. |
| R3 - Supported feature | MX has an operable, maintainable support contract. | Stable API/version policy, persistent GPU CI, cross-node qualification, observability, concurrency proof, and documented SLOs. |

Speculative decoding, multimodal wrappers, managed Kubernetes lifecycle, and MX-to-GMS composition require separate
qualification. R2 or R3 does not imply them unless their exact rows are enabled.

## 2. Landed Foundation and Current Scope

### Foundation timeline

| PR | Merged | Contribution |
|:--|:--|:--|
| [#15641](https://github.com/NVIDIA/TensorRT-LLM/pull/15641) at `6967d0eaf2` | 2026-07-18 | Optional `tensorrt_llm[mx]` packaging, external MX 0.4.1 integration, local Docker/Redis convenience path, and complete-disk fallback. |
| [#16159](https://github.com/NVIDIA/TensorRT-LLM/pull/16159) at `5a69240c9e` | 2026-07-23 | ArtifactIdentity v1 and content-bound SourceIdentity v2. |
| [#16458](https://github.com/NVIDIA/TensorRT-LLM/pull/16458) at `731d293dd7` | 2026-07-31 | Exact qualification profiles, reusable lifecycle harness, transform-layout ABI, and SourceIdentity v3. |

The five staged-hook migration waves are merged. They separate `setup_aliases()`, `transform_weights()`, and
`cache_derived_state()` so a receiver can bind post-transform tensors without rerunning irreversible transforms.
That migration enables qualification; it does not qualify every model automatically.

### Non-negotiable safety contract

Every enabled profile must satisfy all of the following:

1. Match the exact root class and the architecture/model type captured before model construction.
2. Match transfer scope, protocol version, speculative mode, declared features, and every layout-affecting runtime
   constraint.
3. Bind an immutable ArtifactIdentity and a versioned transform-layout ABI into SourceIdentity.
4. Apply the same qualification decision before donor publication and receiver P2P.
5. Reject unsupported, incomplete, old, or mismatched metadata before any tensor write.
6. On transfer or finalization failure, discard staged state and perform one complete disk load. Never combine partial
   post-transform state with raw checkpoint tensors.
7. Rebuild aliases, derived state, and process-local runtime state on the receiver without rerunning transforms.
8. Prove exact output and no-disk reception with a real donor/receiver run.

The capability registry answers, "Has this kind of model been audited?" SourceIdentity answers, "Do these two
concrete executions have the same artifact and layout?" Both checks are required.

### Exact profiles

| Profile | State | Initial envelope | Important exclusions |
|:--|:--|:--|:--|
| `llama-for-causal-lm-target-v1` | Enabled on `main`; R1 rerun pending | Exact `LlamaForCausalLM`, target-only, protocol v1, `trtllm-llama-target-layout-v1` | No separately loaded draft model; no claim beyond the recorded dtype/parallel rows. |
| `qwen2-for-causal-lm-bf16-target-v1` | Draft [#16974](https://github.com/NVIDIA/TensorRT-LLM/pull/16974) / [TRTLLM-14879](https://jirasw.nvidia.com/browse/TRTLLM-14879) | Exact `Qwen2ForCausalLM`, BF16, unquantized weights/KV cache, TRTLLM attention, TP1/TP2, PP1/CP1/EP1, default fused RoPE, untied embeddings | FP16, quantization, alternate attention, TP>2, PP/CP/EP expansion, LoRA, sparse attention, attention DP, multi-node, YaRN, tied embeddings, and speculation. |
| `qwen3-for-causal-lm-bf16-target-v1` | Stacked draft [#17142](https://github.com/NVIDIA/TensorRT-LLM/pull/17142) / [TRTLLM-14880](https://jirasw.nvidia.com/browse/TRTLLM-14880) | Same initial dense envelope, with a distinct Qwen3 ABI and Q/K-normalized attention lifecycle | Same exclusions as Qwen2; no support inherited from the Qwen2 profile. |

Only the Llama row is present on landed `main`. Draft rows remain unsupported until merge and qualification.

## 3. Delivery Roadmap

The first family PR is larger than later family PRs because #16974 also introduces the reusable runtime-constraint
matrix. Follow-on PRs should add a profile, family-specific lifecycle evidence, and only the generic dimensions that
their audit proves are missing.

### Active stack

```text
#16458  exact-profile and transform-ABI foundation [merged]
`-- #16974 / TRTLLM-14879  Qwen2 and Qwen2.5 dense [draft]
    `-- #17142 / TRTLLM-14880  Qwen3 dense [stacked draft]
```

[#17029](https://github.com/NVIDIA/TensorRT-LLM/pull/17029) is a parallel, ready-for-review MX transport-delegation
refactor. If it lands first, rebase the family stack and re-prove TRT-LLM-owned profile gating, ABI identity, staged
finalization, full fallback, and tests. It does not replace model-family qualification.

### Immediate next actions

1. Finish #16974 review and retain real Qwen2/Qwen2.5 TP1/TP2 donor, receiver, no-disk, exact-output, identity, and
   negative-fallback evidence.
2. Rebase #17142 onto the settled Qwen2 change, then run the equivalent Qwen3 evidence with Q/K norm and fused-layout
   probes.
3. Start Mixtral as the first narrow MoE canary. It isolates expert packing and TP/EP behavior before shared-expert and
   next-layer alias complexity.
4. Allow Mistral dense to proceed in parallel as independent dense-root coverage.

### Itemized work under TRTLLM-11901

| Wave | Work item | Tracking | Dependency and purpose |
|:--|:--|:--|:--|
| A1 | Qwen2/Qwen2.5 dense | [TRTLLM-14879](https://jirasw.nvidia.com/browse/TRTLLM-14879), [PR #16974](https://github.com/NVIDIA/TensorRT-LLM/pull/16974) | First family extension and shared runtime matrix. |
| A2 | Qwen3 dense | [TRTLLM-14880](https://jirasw.nvidia.com/browse/TRTLLM-14880), [PR #17142](https://github.com/NVIDIA/TensorRT-LLM/pull/17142) | Stacked on A1; adds Q/K-normalized lifecycle and a distinct ABI. |
| A-parallel | Mistral dense | [TRTLLM-14881](https://jirasw.nvidia.com/browse/TRTLLM-14881) | Independent exact dense root; does not inherit Llama support. |
| B1 | Mixtral MoE canary | [TRTLLM-14882](https://jirasw.nvidia.com/browse/TRTLLM-14882) | First basic expert-layout and TP/EP qualification. |
| B2 | Qwen3 MoE | [TRTLLM-14883](https://jirasw.nvidia.com/browse/TRTLLM-14883) | Strategic shared-expert profile after the MoE canary. |
| B3 | Qwen2 MoE | [TRTLLM-14884](https://jirasw.nvidia.com/browse/TRTLLM-14884) | Separate Qwen2 expert layout and loader path. |
| B3 | GLM4 MoE | [TRTLLM-14885](https://jirasw.nvidia.com/browse/TRTLLM-14885) | Custom mapping, QK normalization, aliases, and MoE state. |
| C1 | DeepSeek V3 | [TRTLLM-14886](https://jirasw.nvidia.com/browse/TRTLLM-14886) | First narrow target-only MLA plus MoE profile. |
| C2 | DeepSeek V3.2 | [TRTLLM-14887](https://jirasw.nvidia.com/browse/TRTLLM-14887) | Distinct configuration profile after V3. |
| C3 | Kimi K2 text | [TRTLLM-14888](https://jirasw.nvidia.com/browse/TRTLLM-14888) | Explicit Kimi profile; do not inherit DeepSeek support by class. |
| C3 | GLM DSA | [TRTLLM-14889](https://jirasw.nvidia.com/browse/TRTLLM-14889) | Preserve canonical pre-normalization identity and sparse state. |
| D1 | Qwen3-Next | [TRTLLM-14891](https://jirasw.nvidia.com/browse/TRTLLM-14891) | Hybrid attention/Mamba state reconstruction. |
| D2 | Qwen3.5 dense | [TRTLLM-14892](https://jirasw.nvidia.com/browse/TRTLLM-14892) | Exact wrapper and config-normalization profile. |
| D3 | Qwen3.5 MoE | [TRTLLM-14893](https://jirasw.nvidia.com/browse/TRTLLM-14893) | Hybrid plus expert state; no subclass-based enablement. |
| E | DeepSeek V4 | [TRTLLM-14894](https://jirasw.nvidia.com/browse/TRTLLM-14894) | Migrate remaining root lifecycle, then qualify V4-specific sparse/MLA/MoE state. |
| F0 | Multimodal component contract | [TRTLLM-14895](https://jirasw.nvidia.com/browse/TRTLLM-14895) | Define component identity, scope, atomic installation, and fallback first. |
| F1 | Qwen2-VL | [TRTLLM-14896](https://jirasw.nvidia.com/browse/TRTLLM-14896) | Depends on A1 and F0. |
| F2 | Qwen3-VL | [TRTLLM-14897](https://jirasw.nvidia.com/browse/TRTLLM-14897) | Depends on A2 and F0. |
| F3 | Qwen3-VL MoE | [TRTLLM-14898](https://jirasw.nvidia.com/browse/TRTLLM-14898) | Depends on Qwen3 MoE, Qwen3-VL, and F0. |
| F3 | Kimi K2.5 | [TRTLLM-14900](https://jirasw.nvidia.com/browse/TRTLLM-14900) | Depends on Kimi K2 text and F0; qualify image/video output and atomic fallback. |

The umbrella task [TRTLLM-14727](https://jirasw.nvidia.com/browse/TRTLLM-14727) remains useful for the overall model
family expansion, but each implementation PR should cite its itemized task above.

## 4. Qualification Contract

### 4.1 Freeze one exact profile

Record the following before code changes:

- Root class, pre-construction architecture/model type, checkpoint, and transfer scope.
- Producer and receiver TRT-LLM commits plus MX client/server version.
- Dtype, weight/KV quantization, attention and MoE backends, and layout-affecting flags.
- TP/PP/EP/CP sizes and ranks, attention DP, tied embeddings, speculative mode, and component scope.
- Immutable ArtifactIdentity scheme/digest and a new transform ABI whenever transferred tensor semantics or receiver
  finalization changes.

Never infer support through `isinstance`, architecture resemblance, or a shared marketing family. Multiple checkpoints
may share one profile only when an audit proves identical root/config semantics, tensor layout, ABI, feature envelope,
and component scope.

### 4.2 Audit the complete lifecycle

From the constructed root, inventory every reachable override of `post_load_weights()`, `setup_aliases()`,
`transform_weights()`, `cache_derived_state()`, and `_weights_transformed`, including backend- and quantization-selected
modules.

Classify each action as:

- structural aliasing or runtime wiring, which belongs in `setup_aliases()`;
- irreversible packing, fusion, or requantization, which belongs in guarded `transform_weights()`;
- cache, scale alias, validation state, or non-persistent buffer reconstruction, which belongs in
  `cache_derived_state()`; or
- process-local communicator, stream, event, or load-balancer setup, which remains orchestrator-owned.

Use a compatibility `post_load_weights()` shim only when it deliberately invokes these stages. A family PR should
change lifecycle code only when this audit finds missing or misplaced work.

### 4.3 Required unit evidence

For deterministic tiny models, compare a normal full load with a staged receiver that binds post-transform tensors:

1. Parameter and buffer names, shapes, dtypes, and values match.
2. Required aliases have the same object identity.
3. Transform guards and family-specific layout state match.
4. Derived state and deterministic forward outputs match.
5. No receiver `transform_weights()` method is called.
6. An unregistered root and every unsupported runtime dimension reject with a structured reason.
7. Artifact, rank/layout, protocol, scope, and ABI mismatches reject before P2P.

### 4.4 Required real-GPU evidence

Clone §20 for the exact candidate profile:

- Run an independent HF baseline, MX donor, full receiver, and receiver with checkpoint weight shards blocked.
- Require exact token IDs and retain publication, discovery, transfer, and staged-finalization evidence for every rank.
- Run unsupported-profile, artifact mismatch, ABI mismatch, parallel mismatch, partial-transfer, and server-failure
  controls. Each control must take the declared full-fallback or required-failure path.
- Run TP1 and TP2 for the initial dense profile. MoE profiles must also exercise the declared expert-parallel row.
- Record every claimed dtype, quantization, backend, and topology. Untested combinations stay unsupported.

Use pairwise expansion for a large matrix, but never advertise an untested combination.

### 4.5 Enable and keep it qualified

Add the exact registry row only with its positive and negative tests. In the same PR, update the public support table,
limitations, and fallback reasons. Add a permanent small-fixture GPU gate and a scheduled representative-checkpoint
run so qualification does not become one-time evidence.

## 5. Family-Specific Audit Focus

| Model category | Additional proof required before enablement |
|:--|:--|
| Qwen dense | Fused QKV/gate-up layout; Qwen3 Q/K normalization; RoPE mode; tied versus untied head. Keep YaRN, quantization, alternate attention, CP/attention DP, and speculation as later rows. |
| Mistral dense | Distinct root and loader path. Run the dense procedure rather than inheriting Llama evidence. |
| Mixtral and Qwen/GLM MoE | Expert packing, shared experts, router weights, aliases, process-local MoE finalization, TP/EP rank mapping, selected MoE backend, and production quantization. |
| DeepSeek V3/V3.2 | MLA projection/scale transforms, routed/shared experts, exact config identity, selected attention/MoE backends, and separate rows for CP, attention DP, and MTP. |
| Kimi K2 text and GLM DSA | Preserve distinct canonical identity despite a DeepSeek-style root. Verify Kimi config/YaRN choices or GLM sparse-indexer derived state with real family checkpoints. |
| Qwen3-Next/Qwen3.5 | Mamba cache reconstruction, stable derived-buffer behavior, repeated decode, dense versus MoE wrappers, and exact subclass profiles. |
| DeepSeek V4 | First migrate the root's remaining structural `post_load_weights()` work, then audit sparse indexer, engram, HC mapping, MLA, and MoE state. |
| Multimodal | Define language-only versus complete-wrapper scope, ordered component identities, all-or-fallback installation, blocked-shard proof for every transferred component, and text plus media output. |

## 6. Readiness Gap Register

| ID | State | Closure required |
|:--|:--|:--|
| MX-R1 - Model families | Active | Land and qualify the itemized profiles in roadmap order. Qwen drafts do not count until merged and proven live. |
| MX-R2 - Exact capability gate | Base closed; per-profile | #16458 landed exact profiles and structured reasons; #16974 adds runtime constraints. Keep registry and docs synchronized. |
| MX-R3 - Artifact identity | Base closed; components open | ArtifactIdentity v1 binds one target checkpoint. Composite target/draft/language/vision/adapter identity belongs to F0. |
| MX-R4 - Transform ABI | Base closed; per-profile | Preserve immutable ABI IDs and add a new family/layout ID whenever transferred semantics or receiver finalization changes. |
| MX-R5 - MX API/version policy | Open for R3 | Keep `modelexpress==0.4.1` while private APIs and process-global state are used. Require a public versioned API and compatibility CI before widening the pin. |
| MX-R6 - Runtime matrix | Active | Publish only evidenced dtype, quantization, backend, TP/PP/EP/CP, attention-DP, and rank combinations. |
| MX-R7 - Permanent GPU gate | Open | Add pre-merge or scheduled donor/receiver jobs with exact outputs, no-disk proof, and negative controls for every family. |
| MX-R8 - Cross-node RDMA | Open | Run two-node qualification with production NIC/NIXL routing, rank mapping, timeout handling, and failure injection before any cross-node claim. |
| MX-R9 - Observability/strict mode | Partial | Emit one terminal result and reason per rank plus an aggregate summary; add a required mode that cannot silently fall back. |
| MX-R10 - Target plus draft/MTP | Open, feature-specific | Track identity/layout per submodel and make multi-component transfer atomic before enabling separate draft or MTP rows. |
| MX-R11 - Multimodal scope | Open | Complete F0 before qualifying any VL wrapper. |
| MX-R12 - Managed lifecycle | Local-only | Treat Docker/Redis startup as a convenience path. Define external-service ownership, authentication, cleanup, and isolation before a managed claim. |
| MX-R13 - Performance/SLOs | Open | Measure identity construction, donor publication, discovery, transfer, receiver finalization, peak resources, and p50/p95 startup against HF. |
| MX-R14 - MX-to-GMS | Separate track | Qualify composition only after §18's native GMS committed-layout contract. Do not block standalone family work on it. |
| MX-R15 - Transactional receiver | Partial | Stage and validate the complete tensor set before commit; failure-inject transfer, alias, and derived-state phases and prove one complete fallback. |
| MX-R16 - Concurrent loads | Open | Replace serialized process-global compatibility state with a per-load public client context; meanwhile stress model/rank concurrency for cross-talk. |

## 7. Production Policy

### Package and API

Keep MX optional and exactly pinned until it exports public APIs for identity construction, exact source queries,
publication metadata, and terminal transfer results. URL, model name, timeout, and credentials must eventually become
per-load arguments rather than temporary process-wide environment state. Any future version range needs CI for every
admitted client/server pair.

### Result and fallback

Every rank should emit one terminal record and every model instance one aggregate startup summary:

```text
source=mx_p2p | hf_disk
result=success | fallback | required_failure
reason=source_missing | unsupported_profile | identity_mismatch |
       artifact_mismatch | protocol_mismatch | partial_transfer |
       local_server_failure | transfer_error | none
profile=<canonical profile id>
source_instance=<sanitized instance id>
bytes=<transferred bytes>
duration_ms=<transfer duration>
```

`best_effort` preserves complete disk fallback. `required` fails startup whenever MX is not used and is mandatory for
positive CI/E2E. Partial transfer is never success unless a separate atomic mixed-layout protocol is designed and
qualified.

## 8. Qualification Evidence and Exit Gates

### Historical Llama evidence

The 2026-07-15 §20 experiment used TinyLlama-1.1B-Chat-v1.0 BF16 on B300 with TRT-LLM `752c05c9af` and MX 0.4.1 plus
a local canonical-wire-catalog patch. TP1 and TP2 donor/receiver, no-shards, exact-output, rank matching, and
parallel-mismatch fallback passed. TP4 was slower than HF and is not a supported performance claim.

Retained output hashes:

- TP1 baseline and mismatch-fallback receiver: `24f0cd36473e4b1a53156c26abdcc4a9db78f662aa51fad47373b1c8387a9b8d`
- TP2 baseline, donor, full receiver, and no-shards receiver: `45841007c85b4496a6388cbf52d433e741ffb8dfb3048ae78b1de1e14241cabc`

This is valuable historical evidence, not proof for final `main`. The current-main rerun must confirm that the
canonical catalog fix is in the selected MX release, normal publisher liveness works without the test-only lease, and
ArtifactIdentity plus transform-ABI controls pass.

### R1 Llama preview

- [x] #15641, #16159, and #16458 merged.
- [x] The exact Llama root/protocol/ABI profile is documented and registered.
- [ ] A production Linux wheel from current `main` installs with `[mx]` and leaves the base install unchanged.
- [ ] §20 passes on current `main` with exact output and canonical-snapshot no-disk reception.
- [ ] Artifact, ABI, parallel, and unsupported-profile mismatches reject before P2P and take the expected fallback.

### R2 multi-family beta

- [ ] Qwen2/Qwen2.5 and Qwen3 dense merge after focused CI and real-GPU evidence.
- [ ] Llama, Qwen dense, one Qwen MoE, one DeepSeek/MLA, and one GLM or Kimi text profile pass.
- [ ] Every enabled row has full-versus-staged unit evidence and live no-disk donor/receiver evidence.
- [ ] Receiver installation is transactional across transfer, alias, and derived-state failures.
- [ ] Structured result/reason reporting makes every fallback visible.

### R3 supported feature

- [ ] Public versioned MX API and dependency compatibility CI.
- [ ] Permanent GPU CI for each advertised model category.
- [ ] Single-node and two-node RDMA qualification on supported environments.
- [ ] Concurrent model/rank stress with no identity, endpoint, model-name, or credential cross-talk.
- [ ] Startup performance, resource, timeout, retry, cleanup, and liveness SLOs.
- [ ] Multimodal, speculative, managed lifecycle, and MX+GMS remain off unless separately qualified.

<details>
<summary>Per-profile evidence record</summary>

```text
profile_id:
root_class:
architecture/model_type:
transfer_scope:
trtllm_producer_sha:
trtllm_receiver_sha:
modelexpress_client/server:
transform_layout_abi:
artifact_identity_format/scheme/digest:
checkpoint:
dtype/quantization:
attention/moe_backend:
tp/pp/ep/cp/attention_dp:
speculative_mode:
hardware/topology:
unit_equivalence:
single_node_e2e:
cross_node_e2e:
exact_output_result:
no_disk_receiver_result:
negative_controls:
performance_summary:
evidence_location:
known_exclusions:
```

</details>

## 9. References

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
