<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 19. ModelStreamer and Weight-Loading Integration Assessment

[< Back to Overview](README.md)

**Status:** Draft assessment and recommended integration direction

**Created:** 2026-07-08

**Last Updated:** 2026-07-19

> [§18 GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-concrete-pr-plan.md) remains the
> implementation source of truth for the GMS lifecycle and PR ordering. This assessment defines storage-ingress and
> weight-materialization composition; it does not add a GMS delivery gate.

The implemented native host-policy prototype, policy-selection guidance, and four-treatment cold-start experiment are
specified in [Native Hybrid Weight Loader](../hybrid-weight-loader/README.md).

## Executive Summary

Run:ai Model Streamer, ModelExpress (MX), GPU Memory Service (GMS), GMS storage snapshots, and Dynamo process Snapshot
solve different stages of model startup. They should be composed rather than treated as competing alternatives.

The recommended direction is:

> Use ModelStreamer inside ModelExpress as the cold-storage ingestion backend, while TensorRT-LLM owns a rank-aware
> weight-loading plan, dependency-safe weight application, transformations, artifact identity, and startup metrics.

For a fresh process, the preferred source cascade is:

1. Reattach compatible GMS-resident weights.
2. Receive or restore a compatible post-transform artifact from an MX donor or GMS storage snapshot.
3. Read raw SafeTensors through ModelStreamer.
4. Fall back to GDS or the native Hugging Face loader.

A Dynamo process snapshot remains above this pipeline. When a complete warmed-process snapshot is valid, it can skip
model construction, weight loading, transformations, warmup, autotuning, and CUDA graph capture.

ModelStreamer should initially target the TensorRT-LLM PyTorch SafeTensors path. Accelerating legacy TensorRT
`.engine` deserialization is explicitly out of scope for this integration and should be evaluated separately.

## Motivation and Evidence

The weight-loading proposal identifies two independently significant startup costs:

| Workload | Checkpoint preparation | Weight population/application | Interpretation |
|:--|--:|--:|:--|
| GLM-5, TP8 | ~86.6 s | ~84.0 s | Storage ingestion and model-specific application are both major bottlenecks. |
| GPT-OSS, TP8, warm storage | ~2.9 s | ~31.6 s | Once storage is warm, weight application and transformation dominate. |

The existing Qwen 72B TP8 measurements show the opposite extreme: cold NFS startup takes about 306 seconds, of which
about 233 seconds is checkpoint prefetch, while warm-cache startup takes about 75 seconds and prefetch falls to about
3.5 seconds. See [§11 Results and Analysis](11-results-analysis.md).

These measurements are workload- and environment-specific, but they establish the architectural point: faster storage
reads alone do not solve startup. ModelStreamer addresses checkpoint discovery and byte ingestion; TensorRT-LLM must
also make mapping, sharding, conversion, fusion, and placement streamable and dependency-safe.

## Component Responsibilities

### Run:ai Model Streamer

Owns parallel local or object-storage reads, multi-file concurrency, bounded reusable buffers, and optional distributed
read sharing. It does not own model-specific mapping, transform correctness, runtime-artifact compatibility, or process
restoration.

### ModelExpress

Owns source selection, retry and fallback policy, warm-donor discovery, cross-node GPU transfer, and artifact
publication. It should host the ModelStreamer and GDS source strategies rather than requiring TensorRT-LLM to
reimplement each transport.

### TensorRT-LLM

Owns rank requirements, weight mapping and application, transform dependencies, destination placement, correctness
validation, artifact identity, and startup metrics. It exposes engine-specific adapter hooks without absorbing cloud
authentication or storage-transport policy.

### GPU Memory Service

Owns GPU allocation lifetime, same-physical-GPU reattachment, zero-copy reuse, and durable snapshots of committed GPU
layouts. It does not provide general object-storage ingestion or restore complete process state.

### Dynamo Process Snapshot

Owns restoration of a complete warmed process and the privileged lifecycle orchestration around CRIU, CUDA
checkpointing, placement, and resume. It sits above, rather than inside, the weight-source abstraction. See
[§17 Snapshot Integration Assessment](17-snapshot-assessment.md).

### AutoDeploy

Owns graph-based weight transformation and backend-aware materialization for the AutoDeploy execution path. It can
share source, plan, identity, artifact, and metric contracts with the PyTorch backend without sharing one loader
implementation.

## Overlap and Integration Opportunities

### Producer-Consumer Loading

ModelStreamer already supplies the parallel storage producer: it can read multiple SafeTensors files concurrently,
stream from supported object stores, and bound staging memory. TensorRT-LLM should consume it through a source or
iterator interface rather than building a second storage reader first.

A node-wide pinned-memory producer should be considered only after measurement shows that page cache plus per-process
ModelStreamer is insufficient. Such a service would require NUMA placement, bounded queues, crash-safe reference
counts, cancellation, and rank-failure handling.

### Rank-Aware Weight Manifest

The proposal's weight-use manifest is its strongest architectural idea. It should become a backend-neutral
`RankWeightManifest` that describes what each rank requires before storage reads begin. The manifest can then drive
selective SafeTensors reads, ModelStreamer scheduling, MX source matching, GMS layout construction, and artifact
identity.

### Transformed Weight Cache

The proposed transformed-weight cache overlaps directly with MX post-transform publication and GMS storage snapshots.
TensorRT-LLM should define one semantic `RuntimeWeightArtifact` contract rather than a third cache format. GMS storage
snapshots should be the first durable backend for post-transform artifacts in GMS deployments; MX should publish the
same semantic layout from a live donor.

### Preallocated GPU Destinations

Preallocation should be expressed as a destination-allocation policy that can target ordinary CUDA memory or a
GMS-managed pool. A universal `uint8` arena is not sufficient without rules for alignment, strides, tied storage,
aliases, quantization packing, transforms that replace storage, NIXL registration, and GMS lifecycle behavior.

### Startup Metrics

The weight-loading proposal's coarse measurements are useful, but they should extend the hierarchical profiler in
[§10 Methodology and Test Plan](10-methodology.md) rather than create another incompatible schema. Preserve separate
stages for discovery, prefetch/read, host staging, device transfer, mapper initialization, application, transformation,
post-load processing, synchronization, warmup, and first-request readiness.

## Target Architecture

```text
Dynamo process snapshot
    +-- restore complete warmed process when valid

Fresh process
    +-- GMS resident-weight attach
    +-- compatible post-transform source
    |     +-- local GMS storage snapshot
    |     +-- ModelExpress warm donor
    +-- raw-weight source
          +-- ModelStreamer
          +-- GDS
          +-- native Hugging Face loader
                    |
                    v
          RankWeightManifest
                    |
                    v
       destination allocation policy
          +-- normal CUDA allocation
          +-- GMS-managed allocation
                    |
                    v
       alias setup -> apply/transform -> derived-state caching
                    |
                    v
        publish through MX / commit to GMS
                    |
                    v
        warmup, autotuning, and CUDA graph capture
```

The fallback chain must be deterministic, observable, and correctness-preserving. Backend availability alone must not
silently change behavior.

## Core Architecture Contracts

### `RankWeightManifest`

> Terminology update: earlier revisions called this richer tensor/rank object `WeightLoadPlan`. PR #16562 uses that name
> for an ordered policy tuple. This assessment now uses `RankWeightManifest`, following the
> [native loader design](../hybrid-weight-loader/README.md#terminology-policy-plan-versus-rank-manifest), so the two
> contracts do not share a public name.

A `RankWeightManifest` describes rank-local requirements without binding them to one storage backend. It should
include:

- Immutable source identity and checkpoint revision.
- Source tensor name, object or file extent, slice, dtype, and checksum information.
- Destination parameter or storage range and owning rank.
- TP, PP, EP, and CP projection.
- Transform operations such as split, concatenate, pack, quantize, pad, or swizzle.
- Aliases, tied storage, shared inputs, and consumer counts.
- Dependency ordering and legal parallel-execution groups.
- Maximum in-flight host, pinned, and device memory.
- Cancellation, completion, retry, and fallback semantics.
- Artifact stage: raw, rank-sharded, or post-transform.

The model and mapper must exist before source reads are scheduled so the selected source can fetch only the ranges
needed by the current rank. The existing dictionary-returning loader API should remain available during migration,
while new source and sink capabilities enable true streaming and direct placement.

### `RuntimeWeightArtifact`

MX publication, GMS reuse, and transformed-weight persistence should share one semantic artifact envelope containing:

- Exact checkpoint revision, signed manifest, or immutable object generation.
- Existing TensorRT-LLM `SourceIdentity` data.
- Rank and TP/PP/EP/CP projection.
- Raw, rank-sharded, or post-transform stage.
- Canonical load/transform-plan hash.
- Transform protocol, producer ABI, and backend or kernel-format versions.
- Tensor names, shapes, dtypes, strides, storage offsets, and alias groups.
- Payload checksums and atomic generation/commit metadata.

The compatible post-transform receiver sequence is:

```text
setup_aliases
-> validate artifact and layout identity
-> bind or copy post-transform bytes
-> mark weights transformed
-> cache_derived_state
```

This extends the staged lifecycle in [§16 Staged Post-Load Hooks](16-staged-post-load-hooks.md) and the identity work
in [§18 GMS Integration Gaps and Concrete PR Plan](18-gms-integration-gaps-and-concrete-pr-plan.md).

## Configuration and Dependency Model

ModelStreamer must be an explicitly selected optional backend, not a path activated merely because its package is
installed. Configuration should keep checkpoint serialization and source transport orthogonal. A possible internal
model is:

- `weights_uri`: local path, S3, GCS, or Azure URI.
- `weight_io_backend`: `auto`, `model_streamer`, `gds`, or `native`.
- `weight_source_policy`: ordered fallback policy.

These names are illustrative rather than a committed public API.

A genuine soft dependency requires ModelStreamer and cloud-provider integrations to be optional, preferably through
provider-specific package extras. Import failure or unsupported configuration must produce a clear diagnostic and
either follow the configured fallback policy or fail explicitly.

Phase 1 should use per-process or per-rank ModelStreamer loading. Distributed ModelStreamer depends on a Torch process
group, which may not be initialized at the relevant point in TensorRT-LLM's MPI worker lifecycle.

## Key Risks and Required Mitigations

1. **Loader API mismatch.** Current loaders tend to materialize a complete dictionary before application. Add planning
   and sink-oriented capabilities while preserving the existing path during migration.
2. **Unsafe custom-loader parallelism.** Model-specific loaders can mutate sibling parameters or depend on ordering.
   Parallelize only dependency-independent groups and qualify model families individually.
3. **Artifact identity cost.** Do not reread and hash every shard during startup. Prefer immutable Hugging Face
   revisions, signed manifests, object generations or ETags, or precomputed digests.
4. **Destination-layout assumptions.** Treat allocation as a policy interface and validate alignment, alias, stride,
   replacement-storage, VMM, NIXL, and GMS rules before claiming direct-placement or TLB benefits.
5. **Ambiguous metrics.** Do not combine reading and application into one timer. Report the selected source, hit or
   fallback reason, bytes, throughput, per-rank values, distributed critical span, maximum local-rank duration, and
   rank skew.
6. **Silent backend selection.** Package presence must not change correctness or startup behavior unexpectedly.
7. **Distributed-runtime coupling.** Validate Torch process-group and MPI lifecycle compatibility before enabling
   distributed ModelStreamer loading.
8. **Cache compatibility.** Reject stale, partially committed, or identity-mismatched artifacts before binding storage.

## Phased Recommendation

### Phase 0: Measurement and Contracts

- Standardize hierarchical startup metrics and cold/warm baselines.
- Define correctness, memory-pressure, and fallback acceptance criteria.
- Draft `RankWeightManifest` and `RuntimeWeightArtifact` schemas.

### Phase 1: Low-Risk Cold-Storage Integration

- Implement the TensorRT-LLM ModelExpress adapter for SafeTensors.
- Add per-process ModelStreamer ingestion with bounded memory.
- Preserve GDS and native-loader fallbacks.
- Use explicit backend configuration and optional packaging.
- Validate local NVMe, a shared filesystem, and one object-store provider.

### Phase 2: Streaming Application

- Add planning and sink-oriented loader capabilities without removing the existing dictionary API.
- Refactor one representative model family first.
- Validate numerical equivalence, peak memory, cancellation, and fallback.

### Phase 3: Reusable Post-Transform Artifacts

- Implement the shared `RuntimeWeightArtifact` envelope.
- Publish compatible layouts through MX and bind or restore them through GMS.
- Use GMS storage snapshots as the first durable transformed-artifact backend.

### Phase 4: Selective and Direct Placement

- Add range-selective reads based on `RankWeightManifest`.
- Evaluate direct placement into registered or GMS-managed destinations.
- Expand object-store and provider-native authentication support.
- Evaluate distributed ModelStreamer only after process-group compatibility is proven.

### Phase 5: Full Lifecycle Composition

- Keep process Snapshot above the loading stack.
- Define fallback behavior for Snapshot, GMS, MX, ModelStreamer, and artifact-validation failures.
- Establish p50 and p95 targets by model, topology, storage state, and source path rather than universal thresholds.

## Validation Matrix

At minimum, validation should cover:

- Cold and warm local NVMe.
- NFS, Lustre, or another shared parallel filesystem.
- Object storage with provider-native authentication.
- Single-GPU and TP8 execution, plus representative PP and EP configurations.
- No MX donor, compatible donor, incompatible donor, and donor failure.
- GMS attach hit, miss, stale identity, and partial artifact.
- Native fallback after ModelStreamer initialization or read failure.
- Numerical equivalence and first-inference correctness.
- Peak CPU, pinned-host, and HBM usage.
- Sample-supported p50/p95 phase timing, distributed critical span, and maximum local-rank duration.

## Explicit Non-Goal: Legacy TensorRT `.engine` Loading

This assessment targets the PyTorch backend's SafeTensors/checkpoint-loading and weight-materialization pipeline.
Run:ai Model Streamer's high-level model-loading interface is SafeTensors-oriented. Although lower-level range reads may
be adaptable to arbitrary files, that does not constitute first-class TensorRT engine support.

Accelerating legacy `.engine` deserialization remains a separate, maintenance-scoped investigation. It must account
for TensorRT stream-reader semantics, existing optional GDS support, deserialization behavior, and the limited
strategic investment appropriate for the legacy backend.

## Decision

Proceed with a composed design:

- ModelStreamer supplies cold bytes.
- ModelExpress selects and orchestrates sources.
- TensorRT-LLM plans and materializes valid runtime weights.
- MX distributes reusable GPU-resident artifacts.
- GMS owns their GPU-memory lifetime and durable layout snapshots.
- Dynamo Snapshot restores the complete process when possible.

This separation addresses first-replica cold start and subsequent restart, scale-out, and failover without creating
competing caches or embedding storage-provider logic in TensorRT-LLM.

## References

- [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)
- [Run:ai Model Streamer usage](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/usage.md)
- [ModelExpress loading strategy][mx-loading-strategy]
- [ModelExpress CI test plan](https://github.com/ai-dynamo/modelexpress/blob/main/ci/TEST_PLAN.md)
- [TRT-LLM weight-loading proposal][weight-loading-proposal]
- [AutoDeploy weight-materialization proposal][autodeploy-weight-proposal]

[mx-loading-strategy]: https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/load_strategy/__init__.py
[weight-loading-proposal]: https://docs.google.com/document/d/1DA_beHXOb3A_fdC2hjSdXeUsi9MdF1Q8WD3bFjfHY0M/edit
[autodeploy-weight-proposal]: https://docs.google.com/document/d/1Il3_CSq3IyfA4AjAgqQE0j0PaBQGA1CEQ4zuL75YqcE/edit
