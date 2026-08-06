<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-06 Progress Reflection

[< Back to design package](README.md)

**Subject:** Rank-cooperative checkpoint-loading prototype in
[TensorRT-LLM PR #16562](https://github.com/NVIDIA/TensorRT-LLM/pull/16562)

**Evidence revision:** `0fe10ac670b821fe634c27ad24cd1315b2ad7a39`

**Status:** Research conclusion and proposed landing plan

## Executive Reflection

PR #16562 has succeeded as an experimental prototype: it identified a broadly applicable, relatively low-risk
latency optimization and exposed the costs of the first correctness-oriented bounded-streaming contract. The evidence
supports separating those paths rather than landing the prototype as one unit.

**RANK-STRIPED is the strongest near-term landing candidate.** It accelerates checkpoint acquisition while preserving
TensorRT-LLM's mature HF SafeTensors mmap, concurrent materialization, transformation, and H2D path. It improved both
measured model families: weight-session time fell by 17.1% for Qwen3.5 397B FP8 and 31.5% for Llama 4 Maverick FP8.

NODE-STREAM and RANK-STREAM remain useful bounded-memory research directions. Their common consumer currently
serializes incremental materialization, copies quantized payloads into rank-local staging, and synchronizes at every
dependency group. That shared path must approach native materialization performance before one versus multiple storage
producers can be compared meaningfully.

The recommended next step is a focused, explicit-opt-in RANK-STRIPED PR from current upstream. RANK-STRIPED should be
implemented as an I/O/read-ahead policy for the existing HF SafeTensors loader, **not** as a checkpoint format alongside
HF or MX.

## 1. Progress So Far

The prototype now provides four independently selectable paths:

- **NATIVE:** the existing checkpoint loader and regression control;
- **RANK-STRIPED:** node-local ranks issue disjoint background `pread()` extents into the Linux page cache while the
  unchanged native mmap/materialization/H2D path runs;
- **NODE-STREAM:** one node-local producer fills a bounded shared-memory stream consumed by all local ranks; and
- **RANK-STREAM:** several node-local rank producers cooperatively fill the same bounded stream.

The work also established:

- preflight policy eligibility and distributed policy agreement;
- coordinated error handling and cleanup mechanisms;
- policy, read, staging, materialization, and startup telemetry;
- the Yijin-style single-producer bounded stream as NODE-STREAM;
- a multi-producer TRT-LLM extension as RANK-STREAM;
- initial bounded-stream qualification for Qwen 3.5 and Llama 4; and
- same-node, true-cold measurements through first token rather than isolated file-read microbenchmarks.

DeepSeek V4 correctly rejects bounded incremental streaming before payload I/O because its bespoke loader does not
provide a safe partial-load transaction. It is design-eligible for RANK-STRIPED because RANK-STRIPED preserves the
whole-checkpoint loader contract, but the latest Appendix B does not contain a completed RANK-STRIPED DeepSeek
qualification.

This distinction is important: **model-family generality is not source-format universality.** RANK-STRIPED is
model-neutral within the supported filesystem-visible HF SafeTensors path because it does not change mapping,
quantization, model traversal, or parallelism semantics. The current prototype does not automatically cover every
checkpoint source or format, such as `.bin`/`.pth`, object-store URIs, Mistral-specific preprocessing, MX, or another
loader that bypasses the HF weight session.

## 2. Latest Evidence

Only the latest comparable rounds from the
[benchmark plan](benchmark-plan.md#appendix-b-current-streaming-prototype-results) are summarized here. Positive reductions
mean faster than NATIVE.

| Model | Policy | N | Weight-session median | Process-to-first-token median |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5 397B FP8 | NATIVE | 2 | 549.55 s | 764.23 s |
|  | RANK-STRIPED | 2 | 455.33 s | 666.15 s |
|  | NODE-STREAM | 2 | 465.69 s | 683.22 s |
|  | RANK-STREAM | 2 | 458.94 s | 669.54 s |
| Llama 4 Maverick FP8 | NATIVE | 2 | 218.88 s | 394.86 s |
|  | RANK-STRIPED | 2 | 149.33 s | 317.05 s |
|  | NODE-STREAM | 3 | 421.59 s | 586.64 s |
|  | RANK-STREAM | 3 | 424.05 s | 586.34 s |

The reductions below use comparable blocks only. Qwen uses its two complete blocks. Maverick uses the two complete
paired blocks for every policy and excludes the extra unpaired NODE-STREAM and RANK-STREAM observations.

| Model | Policy | Comparable N | Weight-session reduction | Process-to-first-token reduction |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5 397B FP8 | RANK-STRIPED | 2 | 17.1% | 12.8% |
|  | NODE-STREAM | 2 | 15.3% | 10.6% |
|  | RANK-STREAM | 2 | 16.5% | 12.4% |
| Llama 4 Maverick FP8 | RANK-STRIPED | 2 | 31.5% | 19.7% |
|  | NODE-STREAM | 2 | -93.4% | -48.0% |
|  | RANK-STREAM | 2 | -93.4% | -49.1% |

Qwen has two complete true-cold blocks. Maverick has two complete paired blocks plus one extra unpaired observation for
each stream; its reductions use only the two paired blocks. These samples establish a direction and reveal mechanism
costs, but they do not provide stable confidence intervals or justify a universal default policy.

The separate 806.80 GB Qwen BF16 diagnostic showed bounded RANK-STREAM completing while RANK-STRIPED was killed for
host OOM during materialization. It demonstrates a bounded-memory advantage, but it is an unmatched capacity
observation rather than comparative performance evidence. It also does not by itself prove that the read-ahead
admission heuristic caused the OOM.

## 3. What the Results Teach Us

### 3.1 RANK-STRIPED preserves the mature critical path

RANK-STRIPED collectively warms one logical checkpoint copy in each node's Linux page cache while the normal
SafeTensors mmap path continues. Its main advantages are:

- native concurrent module materialization remains intact;
- no model-specific partial-load or dependency-group contract is required;
- quantization and TP/PP/CP/EP semantics remain owned by the existing loader; and
- storage acquisition can overlap mmap demand faults, transformation, and H2D.

The overlap is opportunistic rather than readiness-gated: foreground demand faults can overtake the background
reader. Each node still acquires the complete checkpoint, and effectiveness depends on shard geometry, filesystem
behavior, issuer concurrency, page-cache state, and available host-memory headroom.

### 3.2 The streaming consumer currently hides producer topology

NODE-STREAM and RANK-STREAM differ in who issues storage reads, but share batch planning, shared slots, rank-local
staging, incremental dispatch, materialization, H2D synchronization, completion consensus, and cleanup. Their Maverick
weight-session medians differ by only 2.46 seconds, or approximately 0.6%. That result does not prove whether one
producer already saturates NFS; it shows that the common consumer dominates the comparison as currently implemented.

The confirmed shared costs are:

1. Incremental loading repeatedly selects destination roots and takes a serial traversal path, while NATIVE and
   RANK-STRIPED use native concurrent module materialization.
2. The current quantization safety rule rejects borrowed shared-buffer views. Maverick therefore reported
   `direct_bytes=0` and copied the complete 402.80 GB logical payload into rank-local staging on every rank—roughly
   3.2 TB of logical host copying at TP8 before H2D.
3. Maverick had 243 dependency groups and 243 transport batches. Independent groups were not packed together.
4. Every completed group introduced a CUDA synchronization and completion epoch; publication and completion also
   required world-wide consensus for each transport batch.
5. Stream materialization alone took approximately 294–304 seconds, longer than NATIVE's complete 218.88-second
   Maverick weight session.

A dependency group is a correctness unit: tensors participating in one fusion, quantization, or ordering dependency
must become available together. It need not be the transport, dispatch, synchronization, and lifetime unit.
Independent complete groups from unrelated roots can be packed into one bounded **materialization super-batch** and
processed concurrently. The current planner does not do this, so merely increasing slot size cannot reduce
Maverick's 243-batch count.

### 3.3 Confirmed observations versus open diagnoses

The benchmark confirms full rank-local staging, one group per batch for Maverick, per-batch coordination, a common
consumer for both streams, and a stream materialization span longer than the native complete session. It does not yet
isolate:

- how much of the regression comes from lost module concurrency;
- the cost of repeated partial-load invocation and root selection;
- the exposed cost of CUDA waits and collectives;
- whether one storage producer already saturates the measured NFS path; or
- the exact causal chain behind the BF16 host OOM.

Warm-source and resident-buffer ablations, native-serial versus native-concurrent runs, randomized treatment order,
and per-phase rank-max telemetry should answer those questions before more producer-topology work.

## 4. Current Limitations

### 4.1 RANK-STRIPED

- The prototype is limited to POSIX-visible HF/AUTO SafeTensors sessions with coherent distributed rank setup.
- Each node reads a complete logical checkpoint; it does not deduplicate storage I/O across nodes.
- Its full-checkpoint read-ahead is not bounded by model-consumption progress.
- The current 90%-of-`MemAvailable` admission rule is not cgroup-aware and reserves no explicit headroom for model
  construction, transformations, pinned allocations, or later startup work.
- Worker allocation gives every ordinary GPU rank at least one worker, but a strict node-wide cap needs
  quotient/remainder assignment for unusual nodes with more ranks than the nominal worker budget.
- Requested, selected, activated, and effective policy are not yet distinct. A selected policy can decline read-ahead
  internally and silently behave like NATIVE.
- That internal decline does not advance to the next item in the configured fallback list. The current ordered plan is
  therefore a capability selector, not an ordered sequence of performance treatments; both stream modes remain de
  facto explicit experiments for currently eligible HF SafeTensors checkpoints.
- A background read error is currently surfaced when the session finishes even if foreground native materialization
  succeeded. A focused implementation should instead distinguish an advisory read-ahead failure from checkpoint
  corruption and support a rank-coherent effective downgrade for the former.
- Strict benchmark requests and best-effort production requests do not yet have sufficiently explicit failure
  semantics.

### 4.2 NODE-STREAM and RANK-STREAM

- Model and mapper implementations must expose safe incremental dependency groups and destination roots.
- Incremental traversal has not recovered native module concurrency.
- The current quantization lifetime gate causes whole-payload rank-local staging.
- Dependency groups are overloaded as transport, dispatch, H2D, synchronization, and collective boundaries.
- Immediate CUDA synchronization prevents deeper CPU/I/O/H2D overlap.
- Multiple producers cannot be evaluated fairly until the shared consumer is competitive.
- Shared-memory, MPI, and model-specific lifecycle changes make the current prototype much larger and harder to land
  than the focused read-ahead mechanism.

## 5. Optimization Opportunities

| Opportunity | RANK-STRIPED | NODE-STREAM | RANK-STREAM |
| --- | ---: | ---: | ---: |
| Multi-group materialization super-batches | Not applicable | Yes | Yes |
| Restore concurrent materialization | Already native | Yes | Yes |
| Event-managed slot and staging leases | Not applicable | Yes | Yes |
| Per-tensor borrow/copy lifetime classes | Not applicable | Yes | Yes |
| Rank-aware source-range selection | Future | Future | Future |
| Bounded rolling read-ahead | Yes | Not applicable | Not applicable |

Super-batching keeps each dependency group atomic while co-scheduling several independent groups in one mapper/model
invocation and completion epoch. It can amortize Python dispatch, root traversal, H2D fencing, collectives, and
executor overhead for both shared streaming policies; it does not apply to RANK-STRIPED, which already uses native
materialization.

Event-managed leases replace immediate CUDA synchronization with fence-bound resource lifetimes. Recording an event
alone is insufficient: a slot generation must not be overwritten until all consuming CUDA streams retire. A borrowed
view therefore holds the shared slot to its aggregate CUDA fence; a staged path may release the shared slot after the
CPU copy while retaining the local staging allocation until H2D completes. Double and triple buffering should be
benchmarked after that lifecycle is explicit.

Per-tensor lifetime classes can replace the blanket quantized-profile staging rule: lease a direct view for inputs
consumed synchronously, copy only retained scales or temporaries, and fully stage only tensors whose ownership truly
escapes the load call.

For RANK-STRIPED, the most promising extension is bounded rolling read-ahead: preserve native materialization while a
moving read window follows weight-consumption progress and obeys memory pressure. This may keep the winning latency
path while reducing full-checkpoint page-cache risk.

## 6. Recommended Two-Track Plan

### Track A: focused RANK-STRIPED landing PR

Create a new PR from current upstream with:

- filesystem-visible HF SafeTensors only;
- explicit opt-in, leaving NATIVE as the initial default;
- fixed-size, disjoint extent assignment across node-local ranks;
- a fair, enforceable node-level worker budget;
- unchanged native mmap, mapper, materialization, transformation, and H2D;
- cgroup-aware memory admission and configurable reserved headroom;
- rank-coherent activation, degradation, error, and cleanup behavior;
- requested/selected/activated/effective-policy telemetry;
- read bytes, span, throughput, overlap, exposed tail, worker assignment, and memory telemetry; and
- unit, distributed, and lifecycle tests; DeepSeek correctness qualification; and randomized true-cold
  Qwen/Maverick validation.

Do not carry the bounded-stream dependency manifests, shared-memory transport, quantization/MoE lifecycle changes,
super-batches, event leases, or model-specific incremental-loading work into this PR.

### Track B: bounded-stream research

Freeze the current PR head as evidence and evolve the stream consumer in this order:

```text
instrument missing phases
    -> pack independent groups into super-batches
    -> restore native-style materialization concurrency
    -> introduce event-managed lifetimes
    -> adopt per-tensor borrow/copy ownership
    -> add rank-aware source ranges
    -> compare one versus multiple producers
```

The producer-topology comparison becomes meaningful when same-node warm-source stream materialization is within an
agreed threshold—initially 10%—of NATIVE.

## 7. RANK-STRIPED Is a Policy, Not a Checkpoint Format

A checkpoint format answers: **what representation supplies the weights, how are tensors named and laid out, and
which mapper interprets them?** HF, Mistral, and MX are source/representation choices.

RANK-STRIPED answers a different question: **how should node-local ranks schedule reads for an existing file-backed
source?** It creates no new artifact, serialization, tensor namespace, sharding contract, transformed layout, or
mapper. After background `pread()`, the same HF SafeTensors mmap dictionary and native materializer are used.

Encoding it as a new format would conflate source semantics with I/O scheduling, duplicate or misroute mapper
selection, and encourage a combinatorial set such as `HF_RANK_STRIPED`, `MX_RANK_STRIPED`, and
`MODELSTREAMER_RANK_STRIPED`. The intended composition is instead:

```text
Startup restoration and weight-source resolution
    complete process restoration available -> Snapshot
    reusable materialized weights available -> GMS / MX
    raw checkpoint required -> HF SafeTensors

HF file-backed acquisition policy
    RANK-STRIPED, when explicitly requested and admitted
    otherwise NATIVE

Existing HF mapping/materialization/H2D
```

This diagram separates layers; it does not prescribe one universal Snapshot/GMS/MX fallback order.

The first focused PR should keep `checkpoint_format=HF` and add a separate private or experimental read-ahead policy,
for example `native`, `rank_striped_read_ahead`, or later `auto`. Source integration can reuse this policy seam only
when it has compatible file-backed semantics.

MX requires special care. An MX miss or failure may currently be rank-local, whereas RANK-STRIPED enters distributed
coordination. MX can use RANK-STRIPED as its disk fallback only after all participating ranks agree that every rank and
node is taking the HF disk path. Otherwise some ranks could enter RANK-STRIPED collectives while peers remain on MX.

## 8. Proposed Phase-Aware Fallback and Cleanup

“Fall back to HF after cleanup” is safe only before model mutation. RANK-STRIPED already wraps the native HF path, so
after foreground materialization begins the correct response is normally to degrade the advisory read-ahead in place,
not restart the loader. The current prototype propagates a background read error at session completion; the focused
Track A implementation must add the advisory-downgrade semantics proposed below.

| Failure point | Required behavior |
| --- | --- |
| Eligibility or memory admission rejects before activation | Select NATIVE before policy-specific I/O or collectives. An explicit strict request reports rejection; AUTO or best-effort may fall back. |
| Reader/session setup fails before foreground materialization | Coordinate the outcome across ranks; cancel and join any work, close file descriptors, free the node communicator, then enter NATIVE together. |
| Background read-ahead fails after native materialization starts | Cancel and join remaining readers. If the error is advisory and native mmap/materialization succeeds, record an effective downgrade and continue without reloading. Treat source-integrity errors separately and consistently across ranks. |
| Native mmap, mapping, transformation, H2D, or integrity fails after mutation | Do not retry on the same model instance. Fail collectively or reconstruct a fresh model/process because parameters may be partially mutated. |

Cleanup must:

- cancel and join every reader before closing its descriptors or communicator;
- drain worker exceptions and classify advisory read-ahead failure separately from checkpoint-data failure;
- ensure all ranks agree before entering a native or policy-specific collective sequence;
- close executors and descriptors on every path;
- free the node communicator exactly once and never enter an unmatched barrier; and
- emit requested, selected, activated, effective, and fallback-reason telemetry.

Pages already inserted into the Linux page cache cannot be transactionally rolled back and normally should not be
evicted: they are shared, reclaimable, and may already be useful to peers or foreground mmap. “Proper cleanup” refers
to owned userspace and communicator resources, not undoing page-cache warming.

## 9. Immediate Next Steps

1. Freeze `0fe10ac670b821fe634c27ad24cd1315b2ad7a39` as the experimental evidence revision.
2. Create a focused RANK-STRIPED branch from current upstream rather than extracting the entire stream prototype.
3. Specify strict versus best-effort behavior and the phase-aware degradation contract before implementation.
4. Implement cgroup-aware admission, reserved headroom, fair worker assignment, and complete policy telemetry.
5. Verify parameter correctness, repeated initialization, injected-failure cleanup, absence of collective hangs, and
   DeepSeek V4 correctness qualification.
6. Run randomized, true-cold NATIVE versus RANK-STRIPED blocks on Qwen and Maverick from one binary.
7. Keep bounded rolling read-ahead as a follow-up after the focused mechanism lands.
8. Continue bounded-stream work separately, beginning with warm-source materialization ablations and super-batching.

## Decision

Proceed with a separate RANK-STRIPED PR, but implement it as an HF SafeTensors read-ahead policy—not a new checkpoint
format—and make fallback depend on whether foreground model mutation has begun.

## References

- [Rank-Cooperative Checkpoint Loading design](design.md)
- [Experiment and benchmark plan](benchmark-plan.md)
- [TensorRT-LLM PR #16562](https://github.com/NVIDIA/TensorRT-LLM/pull/16562)
- [ModelStreamer and Weight-Loading Integration Assessment](../mx-gms-integration/19-model-streamer-weight-loading-assessment.md)
