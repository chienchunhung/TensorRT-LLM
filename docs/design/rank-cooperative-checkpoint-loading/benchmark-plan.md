<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rank-Cooperative Checkpoint Loading: 8xB300 Experiment and Handoff Plan

[< Back to design package](README.md) | [Main design](design.md)

**Status:** Execution handoff specification; node campaign is blocked until the instrumentation gate below passes

**Created:** 2026-07-19

**Last Updated:** 2026-07-22

## 0. Handoff Contract

The execution owner needs exclusive access to one eight-GPU B300 node, the production shared-storage mount, permission
to perform an administrator-approved client page-cache reset, and access to the pinned model artifacts. The owner is
responsible for producing measured artifacts, not for proving a predetermined conclusion.

Use the reviewed Campaign 0 implementation from
[TensorRT-LLM PR #16562](https://github.com/NVIDIA/TensorRT-LLM/pull/16562):

```text
6065ccb57dd1ceff28278fee019a5412fcf19646
```

The last reviewed two-mechanism baseline, retained for implementation-diff reference only, was:

```text
aa7a616b0add9ffceab5bf72cb5ae35e0f81e64a
```

Build the pinned Campaign 0 SHA unless a later revision of this document names a reviewed successor. If the PR head
has advanced, record both SHAs and review the range before substituting it. Verify that `rank_cooperative_stream` is
selectable, and run all five treatments from the same binary.
Never mix results from different feature commits, containers, models, or benchmark instrumentation revisions.

The current handoff is **Campaign 0 only**:

1. **NATIVE -- Native Checkpoint Loader** (`legacy_fallback`)
2. **RANK-STRIPED -- Rank-Striped Read-Ahead** (`direct_rank_read`)
3. **NODE-STREAM -- Node-Shared Weight Streaming** (`shared_host_producer`)
4. **RANK-STREAM -- Rank-Cooperative Weight Streaming** (`rank_cooperative_stream`)
5. **AUTO -- Automatic Capability Selection** (environment variable unset)

AUTO is ordered capability fallback, not a performance-adaptive policy. For eligible HF/AUTO SafeTensors checkpoints
it is expected to select RANK-STRIPED and match RANK-STRIPED within noise. A future performance-adaptive selector is
out of scope for this run.

This campaign isolates checkpoint-loading policy inside the larger TensorRT-LLM startup path. Snapshot, MX/GMS,
ModelStreamer, compilation, autotuning, KV-cache setup, and CUDA graph improvements are not treatment variables in
Campaign 0. Keep those phases and their caches fixed across NATIVE, RANK-STRIPED, NODE-STREAM, RANK-STREAM, and AUTO; preserve their
timing boundaries; and report when accelerating checkpoint work merely moves the startup bottleneck into another
phase. The same process-to-first-token measurement hierarchy should remain usable by later campaigns that compose or
compare those mechanisms.

### Required Deliverables

The execution owner returns one self-contained campaign directory with:

- an immutable software, model, configuration, and hardware manifest;
- every command, environment variable, treatment order, exit status, and exclusion reason;
- raw per-rank logs and structured interval events;
- cache-state, CPU, host-memory, storage, and GPU telemetry;
- correctness and clean-lifecycle results;
- Nsight Systems traces for representative NATIVE, RANK-STRIPED, NODE-STREAM, and RANK-STREAM runs;
- aggregate tables with paired uncertainty, not isolated best runs; and
- a short `report.md` that answers the questions in the next section.

Do not invent missing measurements. If storage-side counters, cache-reset authority, model access, or instrumentation
are unavailable, mark the affected claim as not measured and continue only with conclusions the evidence can support.

## 1. Questions and Claim Boundaries

The campaign answers:

1. Does RANK-STRIPED reduce model initialization and process-to-first-token latency by overlapping checkpoint I/O with
   existing materialization/H2D work?
2. Does NODE-STREAM reduce startup on the target NFS with one producer per node, a bounded MPI shared double buffer,
   and overlap of batch N+1 I/O with materialization/H2D for the atomic group completed by batch N?
3. Does RANK-STREAM improve the same bounded stream when multiple node-local rank processes collectively fill each
   next batch, and when does that beat a single producer?
4. Which checkpoint geometries and storage states favor NATIVE, RANK-STRIPED, NODE-STREAM, or RANK-STREAM?
5. Do qualified Qwen 3.5 and Llama 4 profiles load correctly and exit cleanly in all three strict optimized modes,
   and does DeepSeek V4 behave correctly in strict RANK-STRIPED while both strict shared-buffer streams fail early
   with the documented unsupported reason?
6. How much of the storage-stage improvement survives model construction, transformation, H2D, warmup, and first
   inference?

The implementation provides three different producer schedules. RANK-STRIPED warms the OS page cache while existing
mmap-driven materialization/H2D proceeds. NODE-STREAM uses one storage producer per node to publish batches for
dependency-safe atomic tensor groups through a bounded, double-buffered shared arena. RANK-STREAM uses the identical
arena, batches, consumer, and materialization path but distributes each batch's nonoverlapping read extents across
multiple local rank producers. All active ranks participate in
publication, completion, and error consensus, so a node producer reuses a slot only after active-world completion.
CUDA registration is node-local: when every node-local rank registers the arena and the source-lifetime contract
permits borrowing, consumers use immutable shared tensor views directly; otherwise affected groups use rank-local
pinned or pageable staging. NODE-STREAM neither retains transformed weights nor keeps its producer alive for restart
reuse. Neither optimized treatment uses `O_DIRECT`, GPUDirect Storage, final-parameter destination reads,
TP/PP/EP-selective reads, or GPU fan-out.

NATIVE already assigns whole SafeTensors files across local ranks and prefetches assigned files with threads. The new
policies are not guaranteed to win for every shard layout or filesystem. In particular:

- RANK-STRIPED can win through finer extent balancing, more useful I/O concurrency, and hidden read time.
- NODE-STREAM can win when one issuer avoids storage contention and bounded shared streaming hides producer I/O without
  excessive slot, staging, synchronization, or module-dispatch overhead.
- RANK-STREAM can win when the bounded stream is beneficial and aggregate storage throughput scales across local rank
  processes, NIC queues, or NUMA domains; it can lose when one producer already saturates the mount.
- NATIVE can remain competitive when many uniform shards already provide enough file-level parallelism.
- AUTO should be equivalent to RANK-STRIPED on eligible cells; it cannot credibly beat RANK-STRIPED merely by selecting
  it.

## 2. Treatments

Run every treatment with the same feature binary and otherwise identical environment.

| Benchmark ID | Display name | Implementation token | Current behavior |
| --- | --- | --- | --- |
| NATIVE | Native Checkpoint Loader | `legacy_fallback` | Existing whole-file assignment, synchronous prefetch/barrier, then mmap/materialization |
| RANK-STRIPED | Rank-Striped Read-Ahead | `direct_rank_read` | Disjoint 256 MiB extents read in the background while `ModelLoader` maps and materializes weights |
| NODE-STREAM | Node-Shared Weight Streaming | `shared_host_producer` | One producer per node streams batches through a bounded, double-buffered MPI shared arena; completed atomic groups are materialized while the producer reads ahead, with direct CUDA-registered views preferred and rank-local staging fallback |
| RANK-STREAM | Rank-Cooperative Weight Streaming | `rank_cooperative_stream` | Multiple node-local rank producers collectively fill disjoint extents of the same bounded shared batches; consumer, direct-view/staging, mapper, and H2D semantics match NODE-STREAM |
| AUTO | Automatic Capability Selection | Environment variable unset | Capability-ordered plan; expected to select RANK-STRIPED for eligible native HF/AUTO SafeTensors |

**Single-Reader Page-Cache Warmup** (`single_producer_page_cache_prefetch`) is an optional diagnostic only. It is the
old synchronous rank-0 prefetch and barrier path; do not substitute it for either shared-buffer stream or include it among the five
main treatments.

Set the common controls before every run:

```bash
export TRTLLM_HF_WEIGHT_CACHE=0
export TLLM_OVERRIDE_LAYER_NUM=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Set exactly one treatment:

```bash
# NATIVE -- Native Checkpoint Loader
export TRTLLM_HF_WEIGHT_LOAD_PLAN=legacy_fallback

# RANK-STRIPED -- Rank-Striped Read-Ahead
export TRTLLM_HF_WEIGHT_LOAD_PLAN=direct_rank_read

# NODE-STREAM -- Node-Shared Weight Streaming
export TRTLLM_HF_WEIGHT_LOAD_PLAN=shared_host_producer

# RANK-STREAM -- Rank-Cooperative Weight Streaming
export TRTLLM_HF_WEIGHT_LOAD_PLAN=rank_cooperative_stream

# AUTO -- test the actual default, not a locally reconstructed order
unset TRTLLM_HF_WEIGHT_LOAD_PLAN
```

Structured policy telemetry is a hard instrumentation gate for all five treatments: every rank must emit requested
plan, selected policy, eligibility/fallback reasons, and run ID. RANK-STRIPED must retain assigned/completed bytes, read
span, and exposed tail. NODE-STREAM must retain configured/effective slot bytes, buffer budget, largest group, groups
fitting one slot, producer workers, `all_ranks_host_registered`, direct-view groups/bytes, staged groups/bytes,
batches/bytes published, and registration detail. RANK-STREAM additionally needs producer count, node/local worker
budgets, assigned/completed bytes per producer, slowest-producer time, fill imbalance, and fill-quorum-to-publication
tail. NATIVE needs an unambiguous benchmark-only `legacy_fallback`
selection/path event; `command.json` alone is insufficient.

### Pre-Feature Baseline Check

NATIVE is a feature-branch proxy for the pre-feature implementation. Before using it as the only baseline, run five
paired pilot blocks comparing the PR base commit with feature-branch NATIVE in one representative cold cell and one
warm cell. Exclude these pilots from confirmation. Compute a paired confidence interval for the log-time ratio and
require the whole interval to lie inside `[log(0.98), log(1.02)]`; an equivalent paired TOST procedure is acceptable if
declared in advance. If the fixed pilot is inconclusive or shows a material difference, retain the unmodified base as
an additional reported control; do not hide feature-branch overhead inside NATIVE.

### Yijin Comparability Contract

NODE-STREAM tests the core Yijin mechanism: one storage producer per node, a bounded double-buffered node-shared weight
stream, parallel consumers, and overlap of producer I/O with consumer materialization/H2D. Compare its numeric results
with Yijin's measurements only when model/checkpoint bytes, storage and cache state, node topology, rank count,
precision, startup boundary, trial count, and correctness gates are equivalent. Otherwise compare mechanisms and
report the TRT-LLM NATIVE/RANK-STRIPED/NODE-STREAM effect sizes without asserting numerical reproduction.

RANK-STREAM is a TRT-LLM extension, not part of the Yijin comparability claim. Its stream planner, shared arena,
consumer, mapper, direct/staged path, and H2D behavior must match NODE-STREAM; only the producer executor changes. The
NODE-STREAM versus RANK-STREAM comparison therefore measures whether the target storage benefits from one issuing
process or multiple rank processes, without crediting unrelated consumer-path changes.

NODE-STREAM always tests the core one-producer/bounded-double-buffer/parallel-consumer pipeline. A producer schedules
batch N+1 after publishing batch N. Split intermediate batches only assemble or stage bytes; model
transformation/H2D starts when a batch completes its atomic group. Publication, completion, and error consensus spans
the active world, so each node producer waits for active-world completion before slot reuse; CUDA registration remains
node-local. Its closest direct-shared-buffer comparison additionally requires CUDA registration by every node-local
rank, full direct borrowed-view byte/group coverage, zero staged bytes, and measured peak host memory.
Quantized profiles currently add a rank-local staging copy for source-lifetime safety; report those as a separate
conservative NODE-STREAM staging-fallback stratum rather than presenting them as a direct-path reproduction.
Transformed-weight caching, producer persistence across restarts, restart reuse, rank-selective source reads, GPU
fan-out, and cross-node deduplication are outside this comparison and must not be credited to NODE-STREAM.

## 3. Before Reserving the Expensive Node

### 3.1 Resolve the Exact Software Revision

From a clean TensorRT-LLM checkout:

```bash
gh pr view 16562 --repo NVIDIA/TensorRT-LLM --json headRefOid,baseRefOid,url
gh pr checkout 16562 --repo NVIDIA/TensorRT-LLM
git status --short --branch
git rev-parse HEAD
git submodule status
```

On a checkout with an `upstream` remote, an equivalent detached setup is:

```bash
git fetch upstream pull/16562/head:refs/remotes/upstream/pr/16562
git worktree add --detach ../trtllm-pr-16562 upstream/pr/16562
```

Record the PR head, PR base, any local benchmark-instrumentation commit, container digest, CUDA, driver, NCCL, Python,
PyTorch, SafeTensors, mpi4py, Nsight Systems, and TensorRT-LLM versions. A local instrumentation commit is acceptable
only if it is identical across NATIVE, RANK-STRIPED, NODE-STREAM, RANK-STREAM, and AUTO; its patch is archived; and it does not alter
policy scheduling.

Pin the image named by `jenkins/current_image_tags.properties` and record its immutable digest. Query the actual GPU
compute capability rather than assuming it:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

The repository currently builds B300 for `103-real`. After confirming that value on the assigned node, use this
source-backed release build:

```bash
python3 scripts/build_wheel.py \
  --build_type Release \
  --cuda_architectures 103-real \
  --use_ccache \
  --nvtx \
  --install \
  --yes
```

Do not use `--fast_build` for headline startup: omitted kernels or JIT work can change initialization. Then run the
repository-prescribed smoke workflow for the target container and save all build commands and logs. Verify that Python
imports the newly installed checkout before running the authoritative focused suite:

```bash
python3 -c 'import tensorrt_llm; print(tensorrt_llm.__file__)'
git rev-parse HEAD
pytest -q \
  tests/unittest/_torch/models/checkpoints/hf/test_weight_load_plan.py \
  tests/unittest/_torch/models/checkpoints/hf/test_incremental_weight_groups.py \
  tests/unittest/_torch/models/checkpoints/hf/test_shared_host_stream.py \
  tests/unittest/_torch/modeling/test_qwen3_5_incremental_loading.py \
  tests/unittest/_torch/modeling/test_llama4_incremental_loading.py \
  tests/unittest/_torch/test_mmap_utils.py

# Production MPI-window epoch smoke; do not substitute the single-process skip.
mpiexec -n 2 pytest -q \
  tests/unittest/_torch/models/checkpoints/hf/test_shared_host_stream.py \
  -k production_mpi_shared_window_epoch_smoke
```

A pre-build pure-Python test is optional and non-authoritative because it can use stale installed bindings.

### 3.2 Inventory Models and Checked-In Configurations

Set `LLM_MODELS_ROOT` and inspect local artifacts rather than assuming public names map to mounted paths:

```bash
rg -n 'Qwen3.5|DeepSeek-V4|Llama-4' \
  tests/integration/defs/perf/_model_paths.py \
  tests/test_common/llm_data.py \
  examples/configs/curated/lookup.yaml

MODEL="$LLM_MODELS_ROOT/Qwen3.5-397B-A17B-NVFP4"
test -f "$MODEL/config.json"
jq '{architectures,model_type,quantization_config}' "$MODEL/config.json"
find -L "$MODEL" -maxdepth 1 -type f -name '*.safetensors' -printf '%s %p\n' | sort -n
```

For each candidate, pin an immutable model revision or internal artifact identity and record a manifest of every
SafeTensors file. Prefer publisher/artifact-system digests. If local hashes are required, compute them once before the
verified client-cache reset; hashing a checkpoint reads it fully and can warm an uncontrolled NFS backend. If backend
cache cannot subsequently be reset, use the immutable revision plus size/stat manifest for trial identity and disclose
the hashing-induced backend warming. Require a nonzero SafeTensors count and confirm the intended load is not `.bin`,
`.pth`, MX, GMS, or a format-specific checkpoint loader.

Use `pytest --collect-only` to obtain exact test IDs from the checked-out revision instead of guessing parameter IDs:

```bash
pytest --collect-only -q tests/integration/defs/test_e2e.py | rg 'Llama-4-(Scout|Maverick)'
pytest --collect-only -q tests/integration/defs/accuracy/test_llm_api_pytorch.py \
  | rg 'Qwen3_5_397B|DeepSeekV4'
```

### 3.3 Node and Storage Preflight

Record before the first trial:

- eight visible B300 GPUs, NVLink/NVSwitch topology, per-GPU HBM, persistence and clock state;
- CPU sockets, cores, NUMA nodes, memory capacity, `MemAvailable`, and CPU affinity;
- mount source, filesystem type/options, client version, network interfaces, routing, and link speed;
- local NVMe identity, capacity, filesystem, and stripe/RAID layout when used;
- whether the storage administrator can reset and verify backend cache state; and
- whether the job has approved authority to reset the node's client page cache.

At minimum archive:

```bash
git rev-parse HEAD
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
lscpu
numactl -H
free -b
findmnt -T "$MODEL" -o SOURCE,FSTYPE,OPTIONS
```

All ranks must resolve the checkpoint to the same node-local backing files and `TLLM_OVERRIDE_LAYER_NUM` must be zero.
RANK-STRIPED additionally requires its full-checkpoint read-ahead guard to pass. Both shared-buffer streams instead
require room for two effective shared slots, rank-local staging if exercised, and normal model construction. The arena budget caps only
the two shared slots; rank-local staging can add as much as one full atomic group per local rank and has no separately
enforced staging budget. Predeclare a total host-memory safety limit and reject a stream performance run that exceeds
it; do not apply RANK-STRIPED's full-checkpoint page-cache rule to either stream.

### 3.4 Instrumentation Gate

Do not reserve the headline node campaign until the execution owner checks in or attaches a campaign-local supervisor,
probe, parser, and benchmark-only instrumentation patch; unit-tests them; and demonstrates one smoke run with all
required primary boundaries. PR #16562 currently emits useful human-readable policy logs and `Model init total`, but
final attribution needs structured events. The instrumentation must use `perf_counter_ns` without inserting new
headline CUDA synchronizations or barriers. Archive the exact scripts, patch, tests, and digests in the campaign.

Required events are:

```text
supervisor_spawn_start                              # external supervisor
llm_constructor_start / llm_constructor_end
model_init_start / model_init_end
weight_session_enter / weight_session_exit
read_ahead_start / read_ahead_end                 # NATIVE and RANK-STRIPED
shared_stream_start / shared_stream_end           # NODE-STREAM and RANK-STREAM lifecycle
shared_batch_read_start / shared_batch_publish    # stream producer(s), per batch
shared_batch_consume_start / shared_batch_ack     # stream consumer, per rank and batch
safetensors_map_start / safetensors_map_end
materialization_start / materialization_end
session_finish_start / session_finish_end         # includes exposed read/stream tail and consensus
final_existing_model_load_cuda_sync
service_ready                                     # server campaign only
first_request_start / first_token
shutdown_complete                                  # emitted before rank exit
supervisor_observed_process_exit                    # external supervisor
```

Record events per rank with monotonic timestamps and a shared run ID. The RANK-STRIPED read interval crosses map and
materialization; it is not a child phase that can be added to them. All headline measurements must preserve production
overlap. Diagnostic runs may add NVTX ranges and synchronizations, but must be labeled and excluded from headline
statistics.

The `ModelLoader` or complete `LLM(...)` path is mandatory. Direct calls to `HfWeightLoader.load_weights()` do not
exercise the same RANK-STRIPED/NODE-STREAM/RANK-STREAM pipelined model-materialization session.

The PR already prints per-rank `Model init total -- <seconds>` after an existing final model-load CUDA synchronization;
parse and report the rank maximum even if narrower structured instrumentation is not yet available. It does not yet
provide a structured requested/selected-policy record, a narrow checkpoint-to-final-CUDA timer, a page-residency tool,
or a parameter-fingerprint harness. These are experiment prerequisites or explicitly reported measurement gaps, not
fields to synthesize after the run.

Before either shared-buffer-stream run is accepted, parse its final telemetry on every rank. Require
`direct_view_groups + staged_groups == group_count` and
`direct_view_bytes + staged_bytes == logical manifest tensor bytes`. Report effective slot count/bytes and
largest-group bytes, and capture node plus per-rank peak host memory. A producer/consumer stream comparison remains
valid when a qualified profile uses rank-local staging, but report it as a policy-specific staging-fallback stratum
because the extra CPU copy is a conservative difference from a direct shared-buffer H2D path. For the NODE-STREAM
direct-view subanalysis, require every node-local rank record to report `all_ranks_host_registered=true`,
`staged_bytes=0`, and full direct byte coverage. CUDA registration and borrowed-source lifetime safety determine
direct-view versus staging behavior; they are not overall stream eligibility requirements. Do not pool staging-fallback
and direct-view results. Do not use `bytes_published` as the coverage denominator because
it may include alignment padding.
Also retain the group/batch count, batch-payload distribution, and time spent between read completion, publication,
active-world acknowledgement, and slot reuse. The initial implementation does not pack multiple independent groups into
one batch, so this coordination cost is part of both streams rather than noise to subtract after the run.

## 4. Metrics

### 4.1 Primary Product Metrics

| Metric | Boundary | Purpose |
| --- | --- | --- |
| `llm_init_e2e_s` | Immediately before `LLM(...)` through constructor return | Programmatic model readiness |
| `model_init_total_s` | `ModelLoader.load()` model-init region through its existing final CUDA completion | Model construction plus weights to usable CUDA state |
| `weight_session_s` | Weight-session enter through exit after mapper/materialization | Inclusive loader/materialization critical path |
| `cold_start_to_first_token_s` | External supervisor timestamp immediately before process spawn through receipt of the first-token event over IPC | Primary user-visible headline |
| `probe_start_to_first_token_s` | Probe timestamp before imports/configuration through the first token | In-process diagnostic; excludes process-spawn latency |

For `trtllm-serve`, also report process-to-health, health-to-first-token, and process-to-first-token. The full campaign
may use an in-process `LLM` harness for lower-variance mechanism measurement and a smaller server subset for operational
validation. Never reuse an `LLM` or server process across timed trials.

Do not include remote model download in the primary loader comparison. Pre-stage one immutable checkpoint on the target
mount. Measure object-store or Hugging Face acquisition separately because the current policies begin after files are
visible through the filesystem.

### 4.2 Read-Ahead and Shared-Stream Overlap Metrics

For RANK-STRIPED derive:

```text
read_elapsed_s = read_ahead_end - read_ahead_start

materialization_overlap_s =
    max(0, min(read_ahead_end, materialization_end)
           - max(read_ahead_start, materialization_start))

hidden_read_fraction =
    max(0, read_elapsed_s - exposed_read_tail_s) / read_elapsed_s

exposed_tail_fraction = exposed_read_tail_s / read_elapsed_s
```

`hidden_read_fraction` is a scheduling-overlap measure, not proof that all overlapped time was removed from the critical
path. Compare paired E2E/model-init results for the causal product effect.

For NODE-STREAM and RANK-STREAM, derive per-batch producer/consumer overlap from the structured events and aggregate
only over batches
with both boundaries present:

```text
shared_io_materialization_overlap_s(completing batch N) =
    overlap(read(batch N+1), materialize_and_h2d(group_completed_by_batch N))

logical_materialized_bytes = direct_view_bytes + staged_bytes
direct_byte_coverage = direct_view_bytes / logical_materialized_bytes
staged_byte_fraction = staged_bytes / logical_materialized_bytes
```

Require `all_ranks_host_registered=true`, direct group and byte coverage of the complete logical manifest, and
`staged_bytes=0` for the primary Yijin-style direct-view NODE-STREAM comparison. Report other NODE-STREAM runs
separately as registration or staging-fallback strata. Always report effective slot bytes, largest group bytes, groups
fitting one slot, batch count, and peak host memory. The shared arena remains bounded to two slots, but rank-local
staging can add up to one full atomic group per local rank and is not covered by the arena budget; report both so the
boundedness claim is scoped correctly.

For RANK-STREAM, additionally derive per batch:

```text
fill_quorum_s = max(local_producer_read_end) - min(local_producer_read_start)
producer_imbalance = max(local_producer_read_s) / mean(local_producer_read_s)
publish_tail_s = batch_publish - max(local_producer_read_end)
```

Report assigned and completed extents/bytes for every active producer, the slowest producer rank, effective producer
count, local worker counts, and aggregate node worker count. Verify the union of producer extents exactly covers each
batch and that no extent is written twice. Compare NODE-STREAM and RANK-STREAM using the same slot size, read-chunk
size, aggregate node worker budget, mapper profile, and direct/staged stratum.

For a representative NATIVE, RANK-STRIPED, direct-view NODE-STREAM, and direct-view RANK-STREAM cell, use Nsight Systems with CUDA, NVTX, and
OS-runtime tracing. The RANK-STRIPED mechanism gate is at least one weight-materialization H2D copy inside both
materialization and background read. The NODE-STREAM gate is at least one completed atomic group from batch N whose
H2D copy is concurrent with producer I/O for batch N+1, plus evidence that the consumed tensor address belongs to the
registered shared arena rather than a rank-local staging allocation.
The RANK-STREAM trace must additionally show multiple rank processes contributing disjoint reads to the same next
batch before its publication.

```text
read_ahead_start
    < weight_materialization_h2d_start
    < read_ahead_end
```

Scope the H2D event with the materialization NVTX/event range. A CUDA copy that occurs before read-ahead starts does not
prove overlap, even if it is earlier than the final read.

Use CUDA events or trace timestamps for GPU-only copy duration; do not label a synchronized wall span containing CPU
transforms and page faults as pure H2D. Verify the installed `nsys` options on the node before recording the command.
An expected diagnostic shape, subject to `nsys profile --help` on the installed version, is:

```bash
nsys profile \
  -f true \
  -t cuda,nvtx,osrt,python-gil \
  --sample=none \
  --gpu-metrics-devices=none \
  --trace-fork-before-exec=true \
  --export=sqlite \
  -o "$ARTIFACT_DIR/startup" \
  -- python "$CAMPAIGN_DIR/startup_probe.py" --run-manifest "$RUN_MANIFEST"
```

Preserve `.nsys-rep` and SQLite output. Current PR code lacks dedicated NVTX ranges around the background reader and
materialization. Add diagnostic-only ranges before claiming an automatically computed overlap percentage; otherwise
report visual/timestamp evidence plus the logged `read_elapsed - exposed_tail` proxy, with that limitation.

Before accepting a diagnostic trace, run an untimed trace preflight and verify that the installed-version launcher
arrangement captures all eight ranks, RANK-STRIPED read-ahead, NODE-STREAM's producer workers, RANK-STREAM's
rank-cooperative producer workers, `pread` activity, MPI shared publication/acknowledgement ranges, module dispatch,
and CUDA H2D events.
Tracing only the probe parent is insufficient. Record the exact MPI/worker launch arrangement and adjust the
installed-version all-process or fork/exec options before the diagnostic campaign if any worker is absent.

### 4.3 Distributed Aggregation

Startup completes when the slowest required rank is ready. Report:

- externally observed process-to-first-token time;
- distributed span `max(rank_end) - min(rank_start)`;
- rank minimum, median, maximum, and start/end skew;
- slowest-rank identity and rank-local assigned/completed bytes; and
- node aggregate prefetch rate computed as total policy-assigned/completed prefetch bytes divided by
  `max(read_end) - min(read_start)`.

The current RANK-STRIPED log reports a rank-local read rate. Do not sum rank-local rates or present one rank's value as
node throughput. The numerator above excludes demand reads/page faults during mmap and materialization, which can
approach world-size times the checkpoint. Do not average rank durations and call that the startup critical path.

### 4.4 I/O, Memory, and Resource Telemetry

Capture per run:

- checkpoint logical/on-disk bytes, shard count, min/median/max shard size, skew, and largest-shard fraction;
- assigned/completed/cancelled extents and bytes by rank when available;
- process `read_chars`, `read_bytes`, major/minor faults, CPU time, RSS, and peak RSS;
- node `MemAvailable`, page-cache residency before/after, and cache growth;
- filesystem/client counters, server bytes/requests/latency, and network bytes when authoritative;
- peak HBM by rank, H2D count/bytes/duration and copy-engine activity in diagnostics; and
- selected policy, prefetch guard, extent/read size, worker caps/actual workers, and fallback reasons.
- for NODE-STREAM and RANK-STREAM, configured/effective slot capacity, total arena allocation, largest atomic group, groups fitting one
  slot, `all_ranks_host_registered`, direct/staged groups and bytes, published batches/bytes, and node/per-rank peak
  host memory; for RANK-STREAM also record producer count, each producer's worker quota, extent/byte assignment, and
  fill-time imbalance.

`read_chars` is syscall-visible logical traffic, while process `read_bytes` is not a portable NFS/Lustre authority.
Claim physical throughput, request amplification, or reduced backend bytes only when storage-side counters or an
isolated OS I/O trace support it.

## 5. Controlled Initial State

### 5.1 Cache-State Classes

| State | Definition | Use |
| --- | --- | --- |
| Client cold, backend controlled | Checkpoint residency below 1%; backend cache reset and verified | Strongest storage claim |
| Client cold, backend warm/unknown | Client residency below 1%; backend state not controlled | Primary production observation if backend control is unavailable |
| Client warm | Checkpoint pages intentionally remain resident | Restart negative control |

For every cold trial:

1. Terminate prior server/worker processes and verify GPUs are empty.
2. Flush dirty data and use only the administrator-approved client-cache reset mechanism.
3. Verify below 1% residency across the exact SafeTensors manifest with `mincore`, `vmtouch`, or equivalent telemetry.
4. Start a fresh process group and record cache/storage counters before workload launch.
5. Run exactly one timed initialization and first-token probe.
6. Shut down all ranks and verify cleanup before resetting for the next treatment.

On an exclusive node, the standard Linux client reset is shown below only as the expected administrator-approved
operation; do not execute it on a shared host or without authorization:

```bash
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
find -L "$MODEL" -maxdepth 1 -type f -name '*.safetensors' -print0 \
  | xargs -0 -r vmtouch -v -f -m "$VMTOUCH_MAX_FILE_SIZE"
```

`vmtouch` is an external prerequisite. Set `VMTOUCH_MAX_FILE_SIZE` above the largest shard using syntax supported by the
installed version; its default maximum can skip multi-gigabyte SafeTensors. Parse and archive summed resident and total
page counts, assert accounted bytes/pages equal the frozen manifest, and require the resident ratio to be below 1%. A
purpose-built `mincore` helper is preferable when this accounting cannot be made exact. Non-root eviction remains a
weaker fallback and still requires post-eviction verification.

Dropping Linux client page cache does not flush an NFS server or appliance cache. If backend reset is unavailable,
label it uncontrolled. Inode-distinct replicas can reduce client reuse but do not prove a cold backend and can change
stripe placement. If replicas are used, balance treatment-to-replica assignment and record layout metadata.

For a warm-cache block, independently populate the exact SafeTensors manifest before **each** treatment and verify at
least 99% client residency. Use a fresh model process, randomize treatment order, and record whether backend state is
controlled. Do not run NATIVE, RANK-STRIPED, NODE-STREAM, RANK-STREAM, and AUTO sequentially against progressively
warmer pages and call that a balanced warm block.

### 5.2 Other Caches and Runtime State

Use one prebuilt image and identical compile, autotune, tokenizer, kernel, and CUDA-graph cache snapshots for all
treatments. The primary loader-isolation campaign may prepopulate non-weight caches. A secondary full-cold campaign may
clear them, but must time them separately. Keep GPU clocks, power, CPU affinity, worker counts, KV-cache settings,
prompt, and serving configuration fixed inside each cell.

## 6. Model and Topology Campaign

RANK-STRIPED is model-neutral for native HF/AUTO SafeTensors. Both shared-buffer streams additionally require an explicitly qualified
atomic manifest plus top-level and nested-module partial loading. A model using the exact generic HF mapper remains
ineligible unless its concrete class declares the audited opt-in; custom mappers and derived classes must qualify
independently. CUDA registration and the source-tensor lifetime contract do not determine stream eligibility;
they select the direct-view path when both permit borrowing, with rank-local staging otherwise. The staged matrix
below avoids spending the full node on a broken model/configuration and avoids a full parallelism Cartesian product.

Preflight must record whether the selected Linear/MoE backend and quant method advertise partial-load support. Dynamic
EPLB and Llama 4 min-latency mode are expected strict NODE-STREAM and RANK-STREAM rejections in this revision and must
not be relabeled as stream measurements after an ordered fallback. Quantized Qwen 3.5/Llama 4 cells are expected to use
stream staging fallback unless a later per-group source-lifetime audit enables direct views; retain direct/staged
telemetry rather than assuming one path from the model name.

Expected internal paths on the checked revision are listed only as discovery aids; verify them during preflight:

| Artifact | Expected local path |
| --- | --- |
| Qwen 3.5 smoke | Resolve the smallest available BF16 Qwen 3.5 checkpoint; prefer `Qwen3.5-4B` when present |
| Qwen 3.5 flagship | `$LLM_MODELS_ROOT/Qwen3.5-397B-A17B-NVFP4` |
| DeepSeek V4 Flash/Pro | `$LLM_MODELS_ROOT/DeepSeek-V4-Flash`, `$LLM_MODELS_ROOT/DeepSeek-V4-Pro` |
| Llama 4 Scout FP8 | `$LLM_MODELS_ROOT/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8` |
| Llama 4 Maverick FP8 | `$LLM_MODELS_ROOT/llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8` |

### Stage 0: Harness and Loader Smoke

1. Run the focused unit suite.
2. Resolve the smallest available BF16 Qwen 3.5 checkpoint, preferring Qwen3.5-4B, and run it through the normal model
   loader to validate the qualified Qwen 3.5 mapper and instrumentation.
3. After an untimed NATIVE TP8 bring-up succeeds for that same checkpoint, run three complete NATIVE, RANK-STRIPED,
   NODE-STREAM, RANK-STREAM, and AUTO cold blocks and one independently prepared warm block. Do not substitute a plain
   Qwen 3 checkpoint: it has no bounded-stream qualification in this revision.

The small TP8 checkpoint validates rank collaboration, cache control, event aggregation, policy assertions, timeout,
first-token correctness, and process cleanup. TP8 is a new smoke qualification for this model on the tested revision,
not checked-in topology coverage; first require an untimed NATIVE bring-up. It is not a flagship performance result.

### Stage 1: Flagship Qualification

Run one untimed or diagnostic NATIVE bring-up first, then strict RANK-STRIPED and, only where supported, both strict
shared-buffer streams. For DeepSeek V4, verify strict NODE-STREAM and RANK-STREAM reject before payload I/O or
parameter mutation and run performance with NATIVE, RANK-STRIPED, and AUTO only; do not label either expected
rejection as a stream benchmark result.

| Family | First gate | Full-node target | Source-backed starting point and caveat |
| --- | --- | --- | --- |
| Qwen 3.5 | Qwen3.5-35B-A3B BF16, then 397B TP4/EP4 | `Qwen3.5-397B-A17B-NVFP4`, TP8/EP8, attention-DP off; qualify attention-DP/MTP separately | PR-head `examples/configs/curated/qwen3.5.yaml`; TP8 stanza `qwen3_5_397b_fp4_tep8_1k1k` in `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml`. That file currently lists B200, so B300 is a qualification result, not a pre-existing claim. |
| DeepSeek V4 | `DeepSeek-V4-Flash`, TP4/EP4, attention-DP, TRTLLM MoE; strict RANK-STRIPED plus both strict stream early-rejection gates | `DeepSeek-V4-Pro`, TP8/EP8; NATIVE/RANK-STRIPED/AUTO only; MTP off then MTP1 | Both shared-buffer streams are intentionally unsupported because the bespoke loader requires a complete checkpoint transaction. B300 CI covers Flash in `l0_dgx_b300.yml`. Pro starts from `examples/configs/curated/deepseek-v4-pro-{throughput,latency}.yaml`; archive the exact derived MTP-off config. |
| Llama 4 | Scout FP8 | Maverick FP8, TP8/EP8, text-only request first | `examples/configs/curated/llama-4-scout.yaml`, `examples/models/core/llama4/README.md`, and checked eight-GPU Scout/Maverick tests. Aggregate construction already creates and loads the vision encoder even for text-only prompts; an image-input follow-up changes preprocessing/first inference, not weight-loading coverage. Full-node Llama 4 PP remains unqualified. |

Use the exact checkpoint paths present under `LLM_MODELS_ROOT`; do not silently substitute a different model, revision,
precision, converted format, topology, or MoE backend. If the largest target fails a gate, report the failure and move
to the next-largest pinned variant as a separately named cell.

Kimi K2.5 may be added as a stretch family only after the required three families complete. It must not displace the
Qwen 3.5, DeepSeek V4, or Llama 4 coverage requested here.

### Stage 2: Five-Block Screening

Before screening, predeclare every production cell and its aggregate weight. For each bounded-stream-qualified
Qwen/Llama target, run five randomized complete NATIVE, RANK-STRIPED, NODE-STREAM, RANK-STREAM, and AUTO blocks on the
production shared mount with verified client-cold state. DeepSeek V4 runs balanced NATIVE/RANK-STRIPED/AUTO blocks
plus both untimed strict stream rejection gates. Freeze one natural checkpoint layout and one production topology per
model.

Screening decides which cells merit confirmation. It does not establish tail latency or broad production support.
Promote a cell when:

- all functional and lifecycle gates pass;
- cache and read-ahead preconditions are valid;
- the paired effect is large enough to matter operationally; and
- variability is low enough for a fixed confirmatory campaign to be informative.

Record cells where any optimized policy loses. Do not prune unfavorable valid results.

### Stage 3: Ten-Block Confirmation

For every promoted cell, predeclare and run ten new randomized complete blocks. Screening observations are excluded.
The primary storage is the target NFS/shared filesystem. Add local NVMe and warm-page-cache controls for at least one
model that passed screening. Promotion makes this a conditional per-cell follow-up. Do not make a fleet-wide
confirmatory aggregate claim unless every predeclared production cell is confirmed; any exploratory screening
aggregate must include valid losing and inconclusive cells as well as winners.

Ten blocks support a median-effect estimate, not a p95 claim. A tail campaign needs a separately predeclared,
precision-driven sample count; use at least 50 observations for exploratory p90 and at least 100 before presenting a
useful p95 estimate.

### Stage 4: Mechanism Diagnostics

For one large model on the target shared mount:

- capture one or two NATIVE, RANK-STRIPED, direct-view NODE-STREAM, and direct-view RANK-STREAM traces with Nsight Systems;
- preserve NVTX/event alignment, H2D copies, OS reads, page faults, and CPU threads;
- for NODE-STREAM, preserve producer batch reads, publication, atomic-group completion, module dispatch, H2D,
  active-world acknowledgement, slot reuse, and
  shared-arena address/registration evidence;
- for RANK-STREAM, preserve those same events plus per-rank extent assignment, local producer completion, fill quorum,
  slowest-producer rank, and node-level publication;
- compare natural shard layout with controlled 1-, 8-, and 64-shard copies only if exact tensor equivalence and storage
  layout are documented; and
- exclude all traced/synchronized runs from headline timing statistics.

Natural publisher shard layout remains the headline. Resharded copies explain mechanism only.

### Stage 5: Targeted Parallelism Coverage

Use a covering set rather than a Cartesian product:

| Concern | Targeted comparison | Interpretation |
| --- | --- | --- |
| TP/rank count | Use the smallest available bounded-stream-qualified Qwen 3.5 BF16 checkpoint, preferring Qwen3.5-4B, then qualify TP1/2/4/8 points not already covered on the checked revision | I/O issuer scaling, RANK-STREAM fill balance, and coordination cost; do not substitute an unqualified generic-mapper model or assume mathematical divisibility implies runtime support |
| PP | Qualify the same Qwen 3.5 checkpoint at TP2xPP4 before timing; if it cannot run PP, report the gap instead of substituting a stream-ineligible model | Current policies still read the full node checkpoint; this measures overlap and ownership correctness, not selective PP reads |
| EP and attention DP | One flagship MoE profile with ADP off/on | Downstream correctness and replicated consumer behavior; raw-byte eligibility is unchanged |
| MTP/speculation | Qwen 3.5 integrated MTP off/on for both shared-buffer streams; DeepSeek V4 is RANK-STRIPED-only; separately opened draft checkpoints must pass both documented stream early-rejection gates | Integrated MTP dependency coverage versus unsupported multi-checkpoint transaction boundaries |
| CP | Only a checked bounded-stream-qualified Qwen 3.5 or Llama 4 configuration; otherwise report RANK-STRIPED-only coverage | Qualification; no source-backed flagship eight-GPU CP example was found during plan authoring |
| Replica DP | Two TP4 or four TP2 independent replicas launched together | Node/storage cold-start storm; each communicator may stage its own checkpoint |
| VLM | Llama 4 text-only versus image-input first inference | Aggregate startup loads both vision and text weights in either case; this isolates preprocessing/first-inference effects, not extra loader coverage |

RANK-STRIPED assigns raw storage extents without final tensor ownership. Both streams use dependency-safe mapper groups
and targeted module dispatch, but every node still reads the complete logical checkpoint; a PP/EP/CP result
must not be reported as rank-selective source acquisition or reduced logical checkpoint bytes.

## 7. Trial Protocol

### 7.1 Blocked Randomization

For a bounded-stream-qualified cell, one block contains one valid cold trial of NATIVE, RANK-STRIPED, NODE-STREAM,
RANK-STREAM, and AUTO on the same node and immutable checkpoint. Randomize order within every block and balance order
over the campaign. A five-period cyclic Latin-square rotation is an acceptable starting schedule:
`NATIVE -> RANK-STRIPED -> NODE-STREAM -> RANK-STREAM -> AUTO`,
`RANK-STRIPED -> NODE-STREAM -> RANK-STREAM -> AUTO -> NATIVE`,
`NODE-STREAM -> RANK-STREAM -> AUTO -> NATIVE -> RANK-STRIPED`,
`RANK-STREAM -> AUTO -> NATIVE -> RANK-STRIPED -> NODE-STREAM`, and
`AUTO -> NATIVE -> RANK-STRIPED -> NODE-STREAM -> RANK-STREAM`. DeepSeek V4 uses a separately balanced
NATIVE/RANK-STRIPED/AUTO schedule after both strict stream early-rejection gates. Generate and archive the complete
schedule before running.

If the backend cache is controlled, use one checkpoint replica and reset/verify it before every treatment. If backend
cache is uncontrolled and multiple replicas are used, balance treatment, order, replica, and stripe layout; include
those factors in analysis and label the storage state honestly.

Set a fixed per-model job timeout after an untimed NATIVE pilot and before comparative trials. A timeout, policy
mismatch, cache-precondition failure, or infrastructure failure invalidates the run but remains in the exclusion log.
Do not replace it silently or append samples until an interval becomes favorable.

### 7.2 One-Run Harness Contract

The execution owner may use the existing cluster launcher or create benchmark-only harness scripts outside the
production package. Every one-run invocation must:

1. consume a frozen JSON/YAML run manifest;
2. set exactly one treatment and launch the normal multi-rank LLM/ModelLoader path;
3. emit process and rank events to a unique run directory;
4. construct the model once;
5. issue one deterministic greedy request with exactly one output token;
6. stop the first-token timer at the first token, before heavyweight correctness probes;
7. run post-timing correctness/resource checks;
8. shut down cleanly and return a meaningful exit status; and
9. write a completion sentinel only after every rank and child process exits.

The repository does not currently contain a cold-start probe with these semantics; the checked quickstart performs an
untimed warmup. Before node execution, add a campaign-local `startup_supervisor.py` and `startup_probe.py`. The
supervisor records time immediately before spawning the frozen launcher argv, receives structured rank/first-token
events over a pipe or local socket, enforces the manifest timeout, and records PID exit timestamps/statuses. The probe
records a timestamp before LLM import/configuration, constructs
`LLM(model=<local-path>, backend="pytorch", **frozen_config)`, and times constructor return.

Freeze one short text prompt per model before the campaign, tokenize it once with the pinned local tokenizer, and
archive both exact text and token IDs in the manifest. Validate every ID against the model vocabulary. The probe passes
the archived `prompt_token_ids` directly; it must not tokenize inside the timed path. Set a fixed seed of 12345 and use:

```python
SamplingParams(
    max_tokens=1,
    min_tokens=1,
    end_id=manifest_eos_token_id,
    pad_id=manifest_pad_or_eos_token_id,
    temperature=0,
    seed=12345,
    ignore_eos=True,
    detokenize=False,
    add_special_tokens=False,
)
```

The manifest must contain the complete resolved LLM arguments, launcher argv, environment, timeout, event socket/path,
and output directory. The runnable contract is:

```bash
timeout "$RUN_TIMEOUT" python "$CAMPAIGN_DIR/startup_supervisor.py" \
  --run-manifest "$RUN_MANIFEST" \
  --output-dir "$RUN_OUTPUT_DIR"
```

Use `llm.generate_async(prompt_token_ids, sampling_params, streaming=True)` as shown by the checked-in async-streaming
example, iterate it with `async for`, and emit the first-token event on the first yielded output. Archive the pinned EOS
and pad IDs with the prompt token IDs. If the checked API cannot provide a streaming result for the profile, label the
metric `request_start_to_one_token_completion_s` and do not report it as TTFT.
The probe must emit exact generated token IDs, run identity, PR/instrumentation commits, policy, init duration, and the
accurately named request metric before `llm.shutdown()`, then emit `shutdown_complete`. The supervisor owns
process-launch and process-exit timing. Archive both scripts, their tests, and their digests with the campaign.

For server validation, the normal entry point is:

```bash
trtllm-serve <pinned-local-model-path> --config <frozen-config.yaml>
```

Use checked-in model configurations as starting points, but materialize a frozen campaign copy with all implicit values
expanded. Do not pass a nested perf-sanity suite file directly to `trtllm-serve`; extract its named `server_configs`
stanza or use the suite's own harness.

For a server-only first-request cross-check, the checked-in `benchmark_serving.py` supports a one-prompt random workload
with one output token and detailed saved results. Preserve its exact invocation after validating flags against
`--help`; an external driver is still required for process-start-to-health and process-start-to-first-token timing.

### 7.3 Correctness and Lifecycle Gates

After first-token timing, every measured run must verify:

- requested and selected policy on every rank, with no unexpected fallback;
- read-ahead guard and checkpoint manifest identity;
- for both shared-buffer streams, exact manifest coverage, host-registration result for all node-local ranks,
  effective slots/largest group, direct/staged byte accounting, and immutable borrowed-view lifetime checks;
- for RANK-STREAM, exact disjoint producer-extent coverage and consistent producer/worker telemetry on every rank;
- parameter count and peak HBM/RSS;
- deterministic first token and additional fixed greedy outputs matching NATIVE;
- selected logits within a predeclared dtype-specific tolerance when available;
- no missing-weight, collective, CUDA, I/O, or silent mapper errors;
- all ranks and child processes exit before timeout;
- no orphan server or read-ahead process/thread remains;
- GPU memory returns to the pre-run baseline.

Bit-exact sampled parameter fingerprints are an optional stronger gate until a benchmark-only worker-rank hook exists.
If implemented, freeze parameter names, offsets, raw-byte hash, and per-rank JSON schema and run it after the headline
timer so device-to-host probes do not contaminate startup. Do not require an external probe to inspect inaccessible
worker parameters.

During qualification, run one immediate same-profile warm restart after the first successful strict RANK-STRIPED,
NODE-STREAM, and RANK-STREAM bring-up per cell. Record it as a non-headline lifecycle probe, verify cleanup, and perform the full
verified cold reset before the next randomized treatment. Do not double every confirmatory run with an ambiguous
restart.

The checked-in unit suite covers direct-session error ordering. If a safe multi-rank fault hook exists, add optional
diagnostics for read start, read, mmap, and materialization failures. Expected RANK-STRIPED cleanup is: coordinate body
error, cancel peer reads, join readers, coordinate read errors, skip the success barrier, free the communicator once,
and keep the original body error primary. Do not claim runtime fault injection if only unit coverage was executed.

### 7.4 Steady-State Non-Regression

After startup checks, run a short fixed throughput and TTFT workload. Compare output correctness, throughput, and TTFT
with NATIVE. Loader policy should not alter steady-state weights or runtime state; a persistent delta suggests
unintended cache, placement, or synchronization effects.

## 8. Analysis

For each complete paired block and treatment `M` compute:

```text
cold_start_speedup_pct(M) =
    100 * (T_NATIVE_first_token - T_M_first_token) / T_NATIVE_first_token

llm_init_speedup_pct(M) =
    100 * (T_NATIVE_llm_init - T_M_llm_init) / T_NATIVE_llm_init

model_init_speedup_pct(M) =
    100 * (T_NATIVE_model_init - T_M_model_init) / T_NATIVE_model_init

weight_session_speedup_pct(M) =
    100 * (T_NATIVE_weight_session - T_M_weight_session) / T_NATIVE_weight_session
```

Report individual observations, paired medians, ranges, and paired-block bootstrap 95% confidence intervals. Bootstrap
complete blocks, not independent treatment observations. If the fixed campaign is inconclusive, report that result;
do not add samples in response to the observed interval.

Also report:

- baseline fraction of startup in model init and the weight session;
- RANK-STRIPED read elapsed, exposed tail, overlap interval, hidden/exposed fractions, and traced H2D overlap;
- NODE-STREAM producer read/consumer overlap, batch publication and acknowledgement tails, effective slot sizing,
  direct-versus-staged coverage, and peak shared plus rank-local host memory;
- RANK-STREAM producer assignment, fill-quorum time, slowest-rank/imbalance, publication tail, aggregate worker budget,
  and NODE-STREAM-parity consumer metrics;
- logical node read rate and physical throughput only at their supported accounting layers;
- rank imbalance, page faults, peak host memory, and peak HBM; and
- AUTO selection overhead/parity with RANK-STRIPED.

Do not use `prefetch_speedup_x` as a headline comparison: RANK-STRIPED overlaps page-cache read-ahead with the mmap
path, while both shared-buffer streams overlap next-batch producer I/O with completed-group materialization/H2D. Their read spans
have different semantics; paired critical-path latency is the product outcome.

Use Amdahl's law as a sanity check. When isolated storage wait is unavailable, label cold-minus-warm delta as an
estimate rather than measured I/O time.

### Aggregate Deployment Result

Predefine the production cells and their weights before screening. Aggregate positive time ratios with a geometric
mean, then convert the ratio to a percentage. Never geometrically average signed speedup percentages. A confirmatory
deployment aggregate requires confirmation of every predeclared cell; otherwise label an all-cell screening aggregate
exploratory and keep conditional confirmation results per-cell.

Campaign 0 has no adaptive-oracle comparison. AUTO's expected result is parity with RANK-STRIPED in eligible cells and
compatibility fallback elsewhere.

## 9. Acceptance and Decision Gates

### Functional Gates

- In bounded-stream-qualified cells, all five treatments emit matching requested/selected policy telemetry on every
  rank; all three strict optimized policies select exactly as requested, NATIVE selects `legacy_fallback`, and AUTO
  selects `direct_rank_read`. DeepSeek V4 instead passes both documented strict stream pre-I/O rejection gates and emits
  matching NATIVE/RANK-STRIPED/AUTO telemetry.
- Headline NODE-STREAM/Yijin direct-view runs have `all_ranks_host_registered=true`, complete direct group/byte
  coverage, zero staged bytes, and recorded effective slots, largest group, and peak host memory. NODE-STREAM
  staging-fallback strata are reported separately; they remain valid NODE-STREAM policy runs.
- The full-read-ahead guard is active for NATIVE/RANK-STRIPED performance cells; both streams instead stay within their
  predeclared total host-memory safety limit. The implementation enforces the shared-arena budget but does not enforce
  a separate rank-local staging budget.
- Correctness matches NATIVE.
- All ranks complete without deadlock, divergence, timeout, or leak; qualification lifecycle restarts pass.
- Host and HBM peaks remain inside predeclared safety budgets.

### Performance Gates

A positive result has a paired-bootstrap 95% confidence interval excluding zero. Initial materiality targets are:

- at least 10% median model-init or weight-session reduction; and
- at least 5% median process-to-first-token reduction.

These are decision thresholds, not promised outcomes. To claim a static treatment beats NATIVE overall, its lower
confidence bound for the predefined production-cell aggregate must be positive. One synthetic shard-layout win is
insufficient.

AUTO passes when it selects correctly and remains within 2% of RANK-STRIPED on eligible cells, subject to the observed
noise floor. It is not expected to outperform RANK-STRIPED. Steady-state throughput and TTFT should remain within 2%
of NATIVE unless pilot variance justifies a wider predeclared equivalence band.

### Decision Table

| Evidence | Next action |
| --- | --- |
| RANK-STRIPED wins and traces show useful I/O/materialization/H2D overlap | Keep RANK-STRIPED as the primary policy for the measured deployment profile. |
| RANK-STRIPED reads quickly but exposes a large tail | Tune issuer count/NUMA or implement finer bounded streaming and destination placement. |
| NODE-STREAM wins on target NFS with full direct coverage | Keep the bounded shared producer/consumer path and tune slot/workers from traces. |
| NODE-STREAM loses despite full direct coverage | Attribute producer, dispatch, H2D, acknowledgement, or slot tail before changing policy order. |
| NODE-STREAM requires substantial staging or registration fails | Treat it as a staging-fallback stratum; fix registration or source-lifetime constraints before comparing with Yijin's direct shared path. |
| RANK-STREAM beats NODE-STREAM with the same stream parameters | Preserve cooperative producers for that storage profile; use fill imbalance and issuer scaling to choose producer count. |
| RANK-STREAM matches or loses to NODE-STREAM | Prefer the simpler single producer for that profile unless multi-process resilience or topology evidence justifies the overhead. |
| RANK-STREAM has a long fill-quorum tail | Rebalance extents/workers or reduce producer count; the slowest producer gates publication. |
| NATIVE is competitive on natural many-shard checkpoints | Preserve NATIVE and include shard geometry in future selection. |
| I/O improves but first-token barely changes | Prioritize transforms, H2D, warmup, compilation, or reusable MX/GMS/Snapshot artifacts. |
| Full read-ahead fails the host-memory guard | Move to bounded selective streaming; do not tune the guard around unsafe memory pressure. |
| A flagship profile fails correctness/lifecycle | Keep the exact profile unqualified and fix downstream integration before performance claims. |
| AUTO matches RANK-STRIPED | Capability fallback works as designed; performance adaptation remains future work. |

## 10. Artifact Layout

Store one directory per immutable campaign:

```text
campaign/
+-- manifest.json
+-- instrumentation.patch
+-- schedule.csv
+-- environment/
|   +-- software.json
|   +-- hardware.json
|   +-- topology.txt
|   +-- storage.json
|   +-- model-manifests/
|   +-- versions.txt
+-- cells/<cell-id>/<treatment>/<run-id>/
|   +-- command.json
|   +-- environment.json
|   +-- cache-state.json
|   +-- startup-profile.json
|   +-- rank-events.jsonl
|   +-- resource-telemetry.jsonl
|   +-- stdout.log
|   +-- stderr.log
|   +-- correctness.json
|   +-- lifecycle.json
|   +-- exit-status.json
|   +-- nvidia-smi-samples.csv
|   +-- complete
+-- nsys/
|   +-- <run-id>.nsys-rep
|   +-- <run-id>-overlap.json
+-- exclusions.csv
+-- aggregate.csv
+-- aggregate.json
+-- report.md
```

Never silently discard an outlier. Classify it as valid variation, infrastructure failure, invalid cache state,
unexpected fallback, correctness failure, timeout, or lifecycle failure.

### Required Headline Table

| Model/revision | Topology | Storage/cache | Treatment | Selected policy | N | First-token median [CI] | Model-init median [CI] | Weight-session median [CI] | Stream direct/staged bytes | Effective slots/largest group | Peak host/HBM | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| | | | NATIVE | | | | | | | | | |
| | | | RANK-STRIPED | | | | | | | | | |
| | | | NODE-STREAM | | | | | | | | | |
| | | | RANK-STREAM | | | | | | | | | |
| | | | AUTO | | | | | | | | | |

### Required Mechanism Table

| Cell/run | Treatment | Producer count | Node/local workers | Read span | Node logical rate | Materialization span | I/O/materialization overlap | Fill imbalance/quorum | Arena CUDA-registered on all node-local ranks? | Direct/staged coverage | Slot/batch tail | Storage counter scope |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| | | | | | | | | | | | | |

## 11. Copy/Paste Brief for the Execution Agent

> Execute Campaign 0 in `docs/design/rank-cooperative-checkpoint-loading/benchmark-plan.md` against TensorRT-LLM PR
> #16562 on one exclusive 8xB300 node. Resolve and pin the PR head, container, model revisions, configs, and
> instrumentation patch.
> Do not change loader scheduling. First satisfy the unit, model-inventory, node/storage, cache-control, and structured
> timing gates. Run NATIVE, RANK-STRIPED, NODE-STREAM, RANK-STREAM, and AUTO from the same binary through the full ModelLoader/LLM
> path for bounded-stream-qualified cells; run NATIVE/RANK-STRIPED/AUTO plus both strict stream rejection gates for
> DeepSeek V4. Use fresh processes, verified cache state, randomized complete blocks, deterministic one-token
> correctness, clean-shutdown checks, and the staged
> smoke -> flagship qualification -> 5-block screening -> 10-block confirmation sequence. Prioritize Qwen3.5-397B
> NVFP4, DeepSeek V4 Pro, and Llama 4 Scout/Maverick with the source-backed configs and caveats in the plan. Capture
> representative NATIVE/RANK-STRIPED/NODE-STREAM/RANK-STREAM Nsight traces to prove or refute I/O/materialization/H2D overlap.
> Report both shared-buffer streams for every qualified profile, but require CUDA registration by all node-local
> ranks and full direct byte/group coverage for the direct-view Yijin subanalysis; report quantized staging and peak
> host memory separately. DeepSeek V4
> strict NODE-STREAM and RANK-STREAM are expected to reject before mutation, while RANK-STRIPED remains supported.
> Return all raw artifacts and an evidence-backed report; do not claim AUTO is performance-adaptive, do not assume
> any optimized treatment must win, and do not invent unavailable measurements.

## 12. Future Campaign: Performance-Adaptive Selection

After a real pre-I/O selector exists, run a new NATIVE/RANK-STRIPED/NODE-STREAM/RANK-STREAM/ADAPTIVE campaign from one new binary.
Calibrate storage profiles on independent objects, freeze the selector/profile, and validate on held-out cells. Define
the cell oracle from static medians:

```text
oracle_cell_time = min(
    median(T_NATIVE),
    median(T_RANK_STRIPED),
    median(T_NODE_STREAM),
    median(T_RANK_STREAM))
adaptive_regret = (median(T_ADAPTIVE) - oracle_cell_time) / oracle_cell_time
```

Recollect NATIVE/RANK-STRIPED/NODE-STREAM/RANK-STREAM with the ADAPTIVE binary; do not reuse Campaign 0 timings. A chooser can
approach the static oracle but cannot intrinsically beat it unless it implements a genuinely new mixed/overlapped data
path.

## Appendix A: Historical Pre-Streaming Benchmark Results

> These results are reproduced verbatim for historical reference. At PR #16562 commit
> `e836be13846dd7055c3d889cdb1510e71ce25d63`, `shared_host_producer` meant synchronous Single-Reader Page-Cache
> Warmup: node-local rank 0 read the checkpoint, all ranks crossed a barrier, and then every rank mmap/materialized it.
> It did not use bounded shared-memory slots, batch N/N+1 producer-consumer overlap, direct shared views, or cooperative
> multi-rank slot filling. Its rows therefore measure historical CACHE-WARMUP, not current NODE-STREAM or RANK-STREAM.

These July 20-21, 2026 measurements are a reference comparison for the native TRT-LLM checkpoint loaders from
PR #16562 at `e836be13846dd7055c3d889cdb1510e71ce25d63`. They do not exercise MX and are not evidence for any MX gate.

The runs used one balanced L/D/S/A0 block on eight B300 GPUs at TP=8. Before every measured run, the checkpoint was
copied through `O_DIRECT` to fresh inodes on a writable NFS volume. `mincore` then verified zero resident client pages
for every SafeTensors shard before timed startup. The copy itself was outside the timed interval; NFS server-side cache
remained uncontrolled. Qwen used the Triton GDN fallback consistently across treatments. DeepSeek used batch size 1,
`max_seq_len=512`, `max_num_tokens=64`, and a 10% KV-cache fraction.

| Model | Checkpoint | Policy | Model init (s) | Reduction | LLM init (s) | Reduction | Process to first token (s) | Reduction |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Qwen3.5-397B-A17B-FP8 | 406.15 GB | Legacy fallback | 466.06 | — | 594.05 | — | 645.45 | — |
| Qwen3.5-397B-A17B-FP8 | 406.15 GB | Direct rank read | 438.71 | 5.9% | 568.64 | 4.3% | 620.66 | 3.8% |
| Qwen3.5-397B-A17B-FP8 | 406.15 GB | Shared host producer | 519.55 | -11.5% | 645.54 | -8.7% | 696.56 | -7.9% |
| Qwen3.5-397B-A17B-FP8 | 406.15 GB | Default plan (direct) | 424.77 | 8.9% | 554.56 | 6.6% | 615.69 | 4.6% |
| DeepSeek-V4-Pro | 864.72 GB | Legacy fallback | 514.01 | — | 826.10 | — | 959.31 | — |
| DeepSeek-V4-Pro | 864.72 GB | Direct rank read | 349.50 | 32.0% | 688.61 | 16.6% | 848.71 | 11.5% |
| DeepSeek-V4-Pro | 864.72 GB | Shared host producer | 601.49 | -17.0% | 895.62 | -8.4% | 1007.99 | -5.1% |
| DeepSeek-V4-Pro | 864.72 GB | Default plan (direct) | 358.67 | 30.2% | 667.07 | 19.3% | 787.59 | 17.9% |

All eight measured runs selected the expected policy on every rank, generated one token, and shut down cleanly.
Negative reductions mean the treatment was slower than legacy. Because there is only one block per model, these values
show direction and magnitude but do not provide statistical confidence. Direct rank read improved the larger
DeepSeek checkpoint substantially; shared host producer was slower than legacy for both models.

The initial DeepSeek treatments above were interrupted and resumed across different node allocations. Their absolute
results remain valid individual observations, but their cross-policy ranking is confounded by node and NFS-path
variation. The controlled same-node rerun below supersedes that DeepSeek policy comparison.

#### Same-node instrumented DeepSeek rerun

The rerun kept legacy, direct rank read, and shared host producer on `umb-b300-dp-147` for the entire sequence. Every
event recorded the same hostname; each run used the same TP=8 configuration and passed the zero-resident-page gate.

| Policy | Model init (s) | Reduction | LLM init (s) | Reduction | Process to first token (s) | Reduction |
|:--|--:|--:|--:|--:|--:|--:|
| Legacy fallback | 532.65 | — | 822.75 | — | 886.50 | — |
| Direct rank read | 394.65 | 25.9% | 681.43 | 17.2% | 745.96 | 15.9% |
| Shared host producer | 582.68 | -9.4% | 866.72 | -5.3% | 931.61 | -5.1% |

| Policy | Checkpoint read/prefetch (s) | Aggregate read rate | SafeTensors mapping (s) | Materialization (s) | Weight session (s) |
|:--|--:|--:|--:|--:|--:|
| Legacy fallback | 314.92 | 2.56 GiB/s | 27.27 | 172.37 | 514.19 |
| Direct rank read | 315.75 | 2.55 GiB/s | 203.98 | 169.97 | 374.76 |
| Shared host producer | 374.37 | 2.15 GiB/s | 27.72 | 162.27 | 564.58 |

Direct rank read overlapped checkpoint I/O with mapping and materialization; every rank reported zero exposed read tail.
Shared host producer assigned the complete 864.72 GB checkpoint to one local producer with 16 workers, but achieved
lower aggregate NFS throughput than the distributed legacy/direct paths on this node. For this controlled run, direct
rank read is the only policy that improves startup; shared host producer should not be a default candidate without
further redesign or storage-specific qualification.

## Appendix B: Current Streaming Prototype Results

> **Result-provenance warning:** these results came from multiple independent experiment rounds, not one continuous
> campaign. The rounds used different model sizes, quantization formats, PR #16562 revisions, loader implementations,
> nodes, run counts, and cache controls. Compare a treatment only with the baseline from the same round, model,
> checkpoint, node/cache protocol, and runtime configuration. Do not aggregate superseded, cross-node, ineligible, or
> failed-OOM observations.

| Round | PR/head | Model/checkpoint | Node/control | Comparability |
| --- | --- | --- | --- | --- |
| Historical reference (Appendix A) | `e836be1384` | Qwen3.5-397B FP8 and DeepSeek-V4-Pro | One block per model; `O_DIRECT`; initial DeepSeek crossed nodes | Early direction only; the controlled DeepSeek rerun supersedes its cross-node ranking. |
| Updated stream qualification | `aa7a616b0a` | DeepSeek-V4-Pro | Same-node strict qualification | DeepSeek-V4 rejected incremental streaming because its loader lacked `allow_partial_loading`. |
| Four-policy Qwen qualification | `0fe10ac670` | Qwen3.5-397B FP8 (406.15 GB) | `umb-b300-dp-199`; one true-cold run per policy | First current direct/node-stream/rank-stream/native comparison; one block only. |
| Repeated instrumented Qwen campaign | `0fe10ac670` | Qwen3.5-397B FP8 (406.15 GB) | `umb-b300-dp-186`; two complete true-cold blocks | Latest results; eight per-rank-verified same-node runs. Block 3 was interrupted. |
| Excluded capacity diagnostic | `0fe10ac670` | Qwen3.5-397B BF16 (806.80 GB) | B300 TP=8; roughly 2 TiB host RAM | Rank-stream completed, but rank-striped read-ahead triggered host OOM. Excluded from speed comparisons. |

### Updated-head eligibility result

At `aa7a616b0add9ffceab5bf72cb5ae35e0f81e64a`, strict shared streaming rejected
`DeepseekV4ForCausalLM` before payload I/O because its `load_weights` path did not support incremental
`allow_partial_loading`. Qwen3.5-397B-A17B-FP8 was selected for the fair four-policy comparison because its mapper
provides qualified atomic weight groups.

### One-block four-policy qualification

At `0fe10ac670b821fe634c27ad24cd1315b2ad7a39`, one true-cold TP=8 run per strict policy completed on
`umb-b300-dp-199`. `O_DIRECT` staging and `mincore` verified zero resident client pages before every timed startup.
All ranks selected the requested policy, generated one token, and shut down cleanly.

| Policy | Model init (s) | Reduction | Weight session (s) | Reduction | LLM init (s) | Reduction | Process to first token (s) | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NATIVE (`legacy_fallback`) | 349.97 | — | 346.99 | — | 459.99 | — | 528.14 | — |
| RANK-STRIPED (`direct_rank_read`) | 306.34 | 12.5% | 303.35 | 12.6% | 418.59 | 9.0% | 487.36 | 7.7% |
| NODE-STREAM (`shared_host_producer`) | 344.93 | 1.4% | 340.52 | 1.9% | 457.33 | 0.6% | 525.79 | 0.4% |
| RANK-STREAM (`rank_cooperative_stream`) | 318.17 | 9.1% | 310.02 | 10.7% | 466.98 | -1.5% | 538.69 | -2.0% |

The one-block rank-stream loading gain did not improve full startup, but one observation provides no statistical
confidence and fixed treatment order can expose JIT, runtime-cache, and storage-order effects.

### Preliminary repeated instrumented qualification

A follow-up added timing for sampler and KV-cache initialization, executor construction, attention JIT, autotuning,
CUDA graph capture, model warmup, worker startup, and proxy READY wait. Blocks 1 and 2 completed on
`umb-b300-dp-186`; every event from all eight runs records that hostname. Orders were NATIVE/RANK-STRIPED/NODE-STREAM/
RANK-STREAM and RANK-STRIPED/NODE-STREAM/RANK-STREAM/NATIVE. Reallocation interrupted Block 3.

| Policy | Model init median (s) | Reduction | Weight session median (s) | Reduction | LLM init median (s) | Reduction | Process to first token median (s) | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NATIVE | 552.33 | — | 549.55 | — | 685.91 | — | 764.23 | — |
| RANK-STRIPED | 459.02 | 16.9% | 455.33 | 17.1% | 586.72 | 14.4% | 666.15 | 12.8% |
| NODE-STREAM | 469.76 | 15.0% | 465.69 | 15.3% | 593.80 | 13.4% | 683.22 | 10.6% |
| RANK-STREAM | 462.83 | 16.2% | 458.94 | 16.5% | 592.51 | 13.6% | 669.54 | 12.4% |

| Policy | Block 1 LLM init (s) | Block 2 LLM init (s) | Model warmup median (s) | Proxy READY wait median (s) |
| --- | ---: | ---: | ---: | ---: |
| NATIVE | 695.31 | 676.50 | 38.15 | 683.16 |
| RANK-STRIPED | 582.00 | 591.45 | 35.39 | 583.99 |
| NODE-STREAM | 593.78 | 593.82 | 36.16 | 590.92 |
| RANK-STREAM | 593.33 | 591.69 | 38.21 | 589.95 |

All optimized policies improved loading and full initialization in both completed blocks. The earlier rank-stream e2e
regression did not reproduce: its warmup median matched NATIVE while its shorter weight session propagated through
proxy READY. Two blocks show consistency but do not provide robust confidence intervals.

### Excluded BF16 capacity diagnostic

The 806.80 GB BF16 checkpoint is not operationally eligible for a four-policy speed comparison on this roughly 2 TiB
host. During RANK-STRIPED materialization, eight ranks each consumed roughly 226-286 GiB of anonymous host memory and
Linux SIGKILLed one rank. Bounded RANK-STREAM completed individually, demonstrating a memory-scalability advantage,
but the unmatched observation is excluded from performance ranking.

## References

- [Rank-Cooperative Checkpoint Loading](design.md)
- [TensorRT-LLM PR #16562](https://github.com/NVIDIA/TensorRT-LLM/pull/16562)
- [Startup Methodology and Test Plan](../mx-gms-integration/10-methodology.md)
- [ModelStreamer and Weight-Loading Integration Assessment](../mx-gms-integration/19-model-streamer-weight-loading-assessment.md)
- [TensorRT-LLM performance benchmarking guide](../../source/developer-guide/perf-benchmarking.md)
- [TensorRT-LLM supported models](../../source/models/supported-models.md)
- [vmtouch upstream](https://github.com/hoytech/vmtouch)
