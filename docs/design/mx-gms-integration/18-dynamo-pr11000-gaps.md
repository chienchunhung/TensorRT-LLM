# 18. Dynamo GMS Standalone Failover Gap Analysis

[< Back to Overview](README.md)

**Status:** Working notes
**Created:** 2026-06-26
**Last updated:** 2026-06-26

## Summary

[ai-dynamo/dynamo PR #11000](https://github.com/ai-dynamo/dynamo/pull/11000) is a useful standalone GMS
operational reference, but it is not a TRT-LLM integration PR. It adds external-facing documentation and a runnable
vLLM example for non-Kubernetes shadow-engine failover with plain inference-engine processes. The PR body explicitly
scopes out library, runtime, and packaging changes.

For TRT-LLM, the immediate conclusion is:

```text
PR #11000 documents the target GMS failover shape.
TRT-LLM still needs packaging, launch UX, executor lifecycle, KV/cache mechanics, and validation.
```

This note records the gaps identified from reading PR #11000 against the TRT-LLM MX/GMS design and the current local
prototype state.

## Sources Reviewed

| Source | What it contributes |
|:--|:--|
| [ai-dynamo/dynamo PR #11000](https://github.com/ai-dynamo/dynamo/pull/11000) | Standalone GMS shadow-engine failover guide and recipe. |
| [`lib/gpu_memory_service/docs/standalone-usage.md`](https://github.com/ai-dynamo/dynamo/pull/11000/files) | Engine requirements for GMS-backed failover: load, sleep/wake, scratch KV, memory accounting, promotion. |
| [`lib/gpu_memory_service/examples/shadow_failover/run.sh`](https://github.com/ai-dynamo/dynamo/pull/11000/files) | vLLM-only runnable example using Dynamo vLLM wrapper, etcd, NATS, frontend, and file-lock promotion. |
| [§04 Implementation & API Design](04-implementation-plan.md) | TRT-LLM two-axis MX/GMS design and the MX+GMS destination-buffer limitation. |
| [§06 Executor Integration and Failover](06-executor-failover.md) | Desired TRT-LLM shadow lifecycle, activation budget, and executor responsibilities. |
| [§07 Tiered Compile Cache](07-compile-cache.md) | Why the <5s activation target depends on warm compile/cache reuse. |
| [§16 Staged Post-Load Hooks](16-staged-post-load-hooks.md) | Required hook decomposition for safe transformed-weight reuse. |
| [§17 Snapshot Integration Assessment](17-snapshot-assessment.md) | Ownership boundary between TRT-LLM engine hooks and Dynamo orchestration. |

## What PR #11000 Covers

The PR describes GMS as an out-of-process per-GPU memory server that owns CUDA VMM mappings. Engines attach to
resident weights instead of reloading them, enabling:

- restart without weight reload when the GMS process survives
- warm shadow engines that can be promoted after a primary failure
- standalone operation without Kubernetes

The documented single-node flow is:

1. Start a GMS server with `GMS_SOCKET_DIR=/tmp/gms python -m gpu_memory_service.cli.server`.
2. Launch a primary and one or more shadows on the same GPUs with `--load-format gms`.
3. Let the first engine load and publish weights in RW mode.
4. Let shadows import the same weights in RO mode.
5. Use a kernel/file lock to decide which process may act as primary.
6. Promote a shadow by acquiring the lock, materializing runtime memory, and registering it for traffic.

The multi-node flow is conceptually a whole-rank-group version of the same model: run one full primary group plus one
or more full shadow groups, then promote the entire group together.

The PR also states the engine requirement table clearly:

| Engine | PR #11000 status | TRT-LLM implication |
|:--|:--|:--|
| vLLM | Full path through GMS patches and Dynamo vLLM wrapper. | Reference implementation for expected behavior. |
| TensorRT-LLM | Weight load only, prototype. | Not yet a failover-ready integration. |
| SGLang | GMS weight/memory-saver integration via Dynamo runtime. | Useful comparison point, but not a TRT-LLM launch recipe. |

## Packaging and Launch Gaps

### 1. No TRT-LLM GMS install extra

TRT-LLM has a ModelExpress dependency path, but not a GMS dependency path. The current prototype has an MX extra such
as:

```text
tensorrt_llm[mx] -> modelexpress>=0.5.0,<0.6.0
```

There is no equivalent:

```text
tensorrt_llm[gms]
tensorrt_llm[mx-gms]
```

The TRT-LLM GMS adapter currently treats `gpu_memory_service` as an optional library and tells users to install it
manually from the Dynamo source tree. PR #11000 also says the broader framework-integration packaging split is
intentionally deferred. That means a user cannot yet get a complete TRT-LLM GMS environment with one predictable
package install.

**Needed:** define the GMS packaging contract: package name, version range, extras, container content, and whether
`gpu_memory_service` is vendored, pinned from PyPI, or installed from the Dynamo repo.

### 2. No TRT-LLM standalone GMS guide

The existing TRT-LLM design has an MX feature doc and a local MX server flow, but no user-facing GMS equivalent.
PR #11000's runnable recipe uses:

```text
python -m dynamo.vllm ...
```

with vLLM-specific flags and environment variables. It does not show how to run:

```text
trtllm-serve <model> --config gms.yaml
```

or how TRT-LLM users should set `load_format`, `gms_config`, socket paths, tags, RW/RO mode, and process roles.

**Needed:** add a TRT-LLM-owned GMS feature page with at least:

- one-process RW weight publish
- second-process RO attach on the same GPU
- `trtllm-serve` config examples
- expected logs and failure modes
- clear non-goals for KV preservation and GPU/node loss

### 3. No GMS daemon lifecycle wrapper for TRT-LLM

PR #11000 requires users to start GMS daemons explicitly. That is acceptable for a low-level GMS guide, but rough for
TRT-LLM. MX already has a local-server concept in the design; GMS needs a comparable operational story.

**Needed:** decide whether TRT-LLM should provide:

- a helper command to start one GMS daemon per local GPU/tag
- a `trtllm-serve` option to launch or validate local GMS daemons
- container images that include both TRT-LLM and GMS entrypoints
- health checks for socket existence, GPU UUID matching, and tag availability

### 4. No TRT-LLM shadow failover launch script

PR #11000's example script is vLLM-specific. TRT-LLM needs its own recipe because its worker topology, PyExecutor
startup, MPI/rank handling, server registration, and disaggregated serving interactions differ from vLLM.

**Needed:** add a TRT-LLM example that covers:

- single-node TP shadow group
- primary/shadow process role assignment
- lock path or GMS lock ownership
- deferred router/frontend registration for shadows
- promotion and demotion commands
- cleanup of failed or orphaned ranks

### 5. Multi-node and disaggregated launch are still conceptual

PR #11000 describes group failover, but does not provide a TRT-LLM recipe for multi-node TP/PP/EP groups or
prefill/decode disaggregation. The MX/GMS design already calls out role-qualified tags for disaggregated serving, but
the operational wiring remains open.

**Needed:** document separate launch flows for:

- aggregate single-node serving
- multi-node TP/PP serving
- prefill-only and decode-only disaggregated roles
- mixed MX+GMS deployments where MX populates per-rank GMS pools

## Functional Gaps

### 1. TRT-LLM only has GMS weight loading, not failover

The current TRT-LLM path has a `LoadFormat.GMS` concept with RW and RO weight materialization. That is necessary but
not sufficient for PR #11000-style failover.

Missing engine behaviors include:

- shadow state that imports weights but does not serve
- hold-until-promoted behavior
- lock-gated activation
- GMS-aware sleep/wake
- scratch/unbacked KV for shadow CUDA graph capture
- promotion-time KV allocation
- router registration only after activation

### 2. PyExecutor needs an explicit shadow lifecycle

[§06](06-executor-failover.md) proposes `SHADOW` and `ACTIVATING` states. PR #11000 assumes this kind of engine-level
state exists, but the TRT-LLM prototype has not implemented it yet.

**Needed:** add executor states and transitions for:

```text
INITIALIZING -> SHADOW_READY -> ACTIVATING -> ACTIVE
```

The shadow must keep model weights attached while avoiding scheduler startup, request admission, and full KV cache
allocation until promotion.

### 3. GMS-aware sleep/wake is missing

PR #11000 requires sleep to unmap GMS virtual addresses and release physical backing, and wake to reconnect/remap the
same virtual addresses. TRT-LLM has memory lifecycle concepts, but it still needs a GMS-specific bridge that is safe
for model weights, KV cache, and any runtime buffers that must survive graph assumptions.

**Needed:** implement and test:

- RO -> RW upgrade or close/reconnect semantics
- `unmap_all_vas()` / remap behavior behind a TRT-LLM abstraction
- failure handling when a GMS daemon disappears
- compatibility with normal TRT-LLM sleep mode

### 4. Scratch KV and stable virtual addresses are not implemented

Warm shadows should not allocate full physical KV memory while the primary is serving, but CUDA graph capture may
still require stable virtual addresses. PR #11000 calls for scratch or placeholder KV during capture, followed by
real GMS-backed KV at the same virtual addresses on promotion.

**Needed:** define the TRT-LLM KV allocation path for shadows:

- reserve virtual address space
- optionally use placeholder physical backing for capture
- release physical backing while idle
- materialize real KV backing on promotion
- preserve graph validity assumptions

### 5. Compile cache remains mandatory for the failover target

The design's measured warmup floor means GMS alone cannot deliver the target activation time. PR #11000 discusses
engine failover mechanics, but TRT-LLM still needs the compile/autotuner cache path from [§07](07-compile-cache.md)
to avoid replaying tens of seconds of warmup.

**Needed:** ensure the failover plan has a concrete cache tier:

- disk cache as the first practical implementation
- GMS-backed compile cache only if serialization is viable
- cache keys that include model, runtime config, parallelism, TRT-LLM version, and relevant Torch/CUDA versions

### 6. Source identity metadata is not complete

The staged-hook design depends on a compatibility gate before RO processes reuse transformed weight memory. The
current prototype has a source-identity check path, but GMS metadata still needs a durable way to publish and read the
writer's identity.

**Needed:** define and implement GMS metadata for:

- model identity
- checkpoint source
- TP/PP/EP shape
- dtype and quantization settings
- post-load transform version
- TRT-LLM compatibility version

Without this, RO attach can be fast but not safely reusable across configuration drift.

### 7. True MX+GMS zero-copy is blocked by MX destination-buffer support

The design's critical MX+GMS limitation remains: the current MX SDK allocates its own CUDA buffers and does not accept
a caller-provided destination pointer. Therefore MX cannot directly populate GMS-managed memory.

Current combined behavior degenerates to:

```text
first process: MX or HF loads weights, then GMS publishes local GPU memory
later same-GPU processes: GMS RO attach
```

That is still useful, but it is not the final design:

```text
MX streams directly into GMS-managed destination buffers.
```

**Needed:** track upstream MX support for preallocated destination buffers before claiming full MX+GMS integration.

### 8. MoE load-balancer compatibility is unresolved

The TRT-LLM prototype rejects GMS with MoE load balancer today because some load-balancer allocations happen outside
the GMS memory pool. This is a practical gap for large MoE deployments.

**Needed:** move MoE load-balancer registration/finalization under the GMS memory-pool scope, or explicitly document
unsupported model families and serving modes until that work lands.

### 9. Router and request semantics need TRT-LLM ownership

GMS does not preserve in-flight requests or KV cache state after a primary crash. PR #11000 says GMS preserves
resident weights and lock semantics; the engine and router decide when to admit traffic.

For TRT-LLM this means failover must define:

- when a shadow is visible to the OpenAI server/router
- how failed streaming requests are surfaced or replayed
- whether request replay is Dynamo-only or also supported by standalone `trtllm-serve`
- how disaggregated prefill/decode failures are coordinated

## Validation Gaps

PR #11000 is documentation and example code. The TRT-LLM path still needs tests that prove the gap is closed.

At inspection time, PR #11000 also had a failing docs link-check job (`Docs link check | lychee`). That is not a
TRT-LLM integration blocker, but the referenced Dynamo guide should be treated as draft until its documentation gate
is clean.

Recommended validation sequence:

1. **Package smoke:** install `tensorrt_llm[gms]` in a clean environment and import `gpu_memory_service`.
2. **Single-GPU weight sharing:** start GMS, run one TRT-LLM RW process, attach one TRT-LLM RO process.
3. **Source-identity rejection:** prove a mismatched RO process refuses incompatible GMS memory.
4. **Shadow idle memory:** verify a shadow holds weights but does not allocate full KV.
5. **Promotion:** kill or demote the primary and activate the shadow.
6. **Compile-cache timing:** measure activation with and without cache.
7. **Multi-rank group promotion:** repeat with TP > 1.
8. **Disaggregated role promotion:** validate prefill/decode tags and router behavior separately.
9. **MoE compatibility:** cover at least one MoE model after the load-balancer allocation issue is fixed.

## Recommended Work Items

| Priority | Work item | Why |
|:--|:--|:--|
| P0 | Define `tensorrt_llm[gms]` / `tensorrt_llm[mx-gms]` packaging. | Removes manual install ambiguity. |
| P0 | Add a TRT-LLM GMS feature doc and minimal `trtllm-serve` recipe. | Makes current weight-sharing behavior usable. |
| P0 | Implement durable GMS source-identity metadata. | Makes RO attach safe. |
| P1 | Add same-GPU RW/RO GPU smoke test. | Locks down the basic integration. |
| P1 | Implement PyExecutor shadow and activation states. | Converts weight sharing into failover. |
| P1 | Add GMS-aware sleep/wake and lock promotion. | Required by PR #11000's failover model. |
| P1 | Add scratch KV / stable-VA allocation path. | Keeps shadows memory-efficient and graph-compatible. |
| P1 | Wire disk compile/autotuner cache into activation. | Required for the <5s target. |
| P2 | Add TRT-LLM multi-rank and disagg launch recipes. | Moves from single-process demo to deployment shape. |
| P2 | Track MX destination-buffer support. | Required for full MX -> GMS zero-copy. |
| P2 | Fix MoE load-balancer allocations under GMS. | Required for MoE production coverage. |

## Positioning

PR #11000 should be treated as a reference for the GMS operational contract, not as evidence that TRT-LLM failover is
done. It clarifies the engine responsibilities that TRT-LLM must own:

```text
GMS owns resident memory and lockable attachment.
TRT-LLM owns engine lifecycle, cache/KV behavior, launch UX, and safe compatibility checks.
Dynamo can orchestrate the cluster-level version once those TRT-LLM hooks exist.
```

That matches the ownership boundary in [§17](17-snapshot-assessment.md): standalone `trtllm-serve` should have a
first-class MX/GMS path, while Dynamo consumes those hooks for cluster-level orchestration.
