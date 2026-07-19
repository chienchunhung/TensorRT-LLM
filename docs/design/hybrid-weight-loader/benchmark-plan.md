<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hybrid Weight Loader Benchmark and Qualification Plan

[< Back to Native Hybrid Weight Loader](README.md)

**Status:** Draft experiment design

**Created:** 2026-07-19

**Last Updated:** 2026-07-19

## 1. Objective

Measure whether the native host-staging policies reduce cold-start latency relative to the pre-feature TensorRT-LLM
loader, explain where the gain comes from, and establish the evidence required for a real adaptive policy.

The experiment answers four questions:

1. Does `direct_rank_read` reduce checkpoint-loading and end-to-end startup on storage that scales with concurrent
   readers?
2. Does `shared_host_producer` reduce those times on storage that penalizes multiple reader processes?
3. How much of a storage-stage gain survives model mapping, transformation, H2D, warmup, and first inference?
4. Can a pre-I/O adaptive selector stay close to the fastest static policy across a heterogeneous deployment mix?

The benchmark must test these hypotheses rather than arrange the matrix to prove a predetermined ordering. Legacy
loading is already parallel at whole-file granularity, so neither new policy is guaranteed to win for every checkpoint
layout or filesystem.

## 2. Four-Treatment Campaigns

Each campaign has exactly four treatments and runs them with the same binary. This prevents unrelated commit
differences from being mistaken for loader speedup. The three static treatments are common to both campaigns:

| ID | Treatment | Configuration | Distinct behavior today |
| --- | --- | --- | --- |
| L | Feature-branch legacy proxy | `legacy_fallback` | Existing whole-file assignment and prefetch, plus feature-branch consensus/dispatch overhead |
| D | Direct rank read | `direct_rank_read` | 256 MiB extents striped across local ranks |
| S | Shared host producer | `shared_host_producer` | Node-local rank 0 stages all extents |

The fourth treatment depends on the campaign; A0 and A1 are never run as two treatments in the same campaign:

| ID | Treatment | Configuration | Distinct behavior |
| --- | --- | --- | --- |
| A0 | Ordered capability fallback | Default ordered plan today | Equals D on every eligible cell; measures selection/fallback overhead only |
| A1 | Future performance-adaptive plan | Pre-I/O selector implemented later | Chooses L, D, or S from a frozen deployment profile |

Use explicit values in every trial:

```bash
export TRTLLM_HF_WEIGHT_CACHE=0
export TLLM_OVERRIDE_LAYER_NUM=0

# L: legacy
export TRTLLM_HF_WEIGHT_LOAD_PLAN=legacy_fallback

# D: direct
export TRTLLM_HF_WEIGHT_LOAD_PLAN=direct_rank_read

# S: shared
export TRTLLM_HF_WEIGHT_LOAD_PLAN=shared_host_producer

# A0: current ordered fallback
export TRTLLM_HF_WEIGHT_LOAD_PLAN=direct_rank_read,shared_host_producer,gpu_broadcast,legacy_fallback
```

Also fix `LoadFormat.AUTO`, the HF checkpoint loader, model revision, dtype, and all serving options. The raw HF weight
cache is disabled because it uses a different lifecycle and the implicit plan preserves it by selecting legacy.

Treatment L is a proxy, not automatically the pre-feature baseline: the feature branch adds manifest consensus, plan
coordination, policy resolution, and communicator handling around legacy I/O. Before using L as the baseline, use at
least five paired base-commit-versus-L pilot trials in one representative cold cell and one warm cell to estimate
variance; exclude those pilots from confirmation. Freeze the confirmatory sample count before collecting it. For each
metric, test whether the paired feature/base log-time ratio lies wholly inside the predeclared equivalence interval
`[log(0.98), log(1.02)]`. Do not inspect an ordinary confidence interval and append samples until it crosses a boundary;
an inconclusive fixed-size campaign remains inconclusive unless a separate, newly designed campaign or a predeclared
group-sequential design with alpha spending is used. If L is not equivalent or equivalence remains inconclusive, report
the unmodified base as an additional control and compute treatment speedups against both baselines.

A0 and A1 belong to separate campaigns:

- **Campaign 0:** run L/D/S/A0 with the current feature binary. A0 validates deterministic capability fallback.
- **Campaign 1:** after implementing A1, rerun L/D/S/A1 with the same new binary. Do not reuse Campaign 0 static-policy
  times, because selector implementation or rebasing can change the other paths.

### Required Naming in Results

Until a performance selector is implemented, label treatment A0 **ordered fallback/direct-equivalent**, not adaptive.
Direct and shared currently have identical qualification rules, direct appears first, and GPU fan-out is unavailable.
Therefore A0 must resolve to direct on an eligible checkpoint and cannot honestly outperform it except through noise.

For a future A1 selector, define the oracle from cell-level static-policy estimators:

```text
oracle_cell_time = min(median(legacy_time), median(direct_time), median(shared_time))
adaptive_cell_regret = (median(adaptive_time) - oracle_cell_time) / oracle_cell_time
```

A chooser can approach the oracle; it cannot intrinsically beat the oracle unless it adds a genuinely new mixed or
overlapped data path.

## 3. Primary Metrics

Report two product-level critical paths:

| Metric | Boundary | Interpretation |
| --- | --- | --- |
| `llm_init_e2e_s` | External driver immediately before `LLM(...)` through constructor return | Programmatic LLM readiness, including worker and executor setup |
| `cold_start_to_first_token_s` | Process launch through completion of a deterministic one-token request | User-visible cold start and primary headline |

If `trtllm-serve` is the deployment target, also report:

- `process_start_to_health_s`: launch through the first successful readiness response;
- `health_to_first_token_s`: ready service through the first successful token; and
- `process_start_to_first_token_s`: their end-to-end critical path.

Do not combine remote model acquisition with raw checkpoint loading in the primary comparison. Pre-stage one immutable
checkpoint for all treatments. Report Hugging Face or object-store download separately because the current native
policies begin after files exist locally or on a mounted filesystem.

## 4. Phase Timing

The implementation PR currently logs cooperative prefetch and mmap setup durations, but that is insufficient for
end-to-end attribution. Port the hierarchical profiler described in
[Startup Methodology and Test Plan](../mx-gms-integration/10-methodology.md), or add an equivalent structured profiler,
before collecting headline results.

Required nested timers are:

```text
startup
├── configuration_and_tokenizer
├── worker_and_executor_initialization
│   ├── model_config_and_defaults
│   ├── model_init_total
│   │   ├── model_config_validation
│   │   ├── model_construction
│   │   ├── parameter_allocation
│   │   └── checkpoint_to_cuda_total
│   │       ├── raw_checkpoint_load
│   │       │   ├── checkpoint_discovery
│   │       │   ├── policy_selection
│   │       │   ├── storage_prefetch
│   │       │   └── safetensors_map_setup
│   │       ├── weight_mapper_initialization
│   │       ├── weight_application_enqueue
│   │       └── post_load_and_final_cuda_completion
│   ├── kv_cache_and_executor_tail
│   └── warmup_compile_autotune_cuda_graphs
├── service_ready
└── first_successful_token
```

Recommended code boundaries on PR #16562 are:

| Timer | Boundary |
| --- | --- |
| `worker_executor_init` | `BaseWorker.setup_engine()` entry through return |
| `create_py_executor` | `create_py_executor()` entry through return |
| `model_init_total` | `ModelLoader.load()` entry through its final CUDA synchronization and cleanup |
| `checkpoint_to_cuda_total` | Immediately before `checkpoint_loader.load_weights()` through final post-load CUDA completion |
| `checkpoint_discovery` | HF file discovery and distributed manifest consensus |
| `policy_selection` | Plan coordination, eligibility evaluation, and policy resolution |
| `storage_prefetch` | Whole-file or extent prefetch, on every rank |
| `safetensors_map_setup` | `safetensors.torch.load_file()` calls and cooperative mmap setup |
| `raw_checkpoint_load` | Inclusive outer `checkpoint_loader.load_weights()` call; contains discovery, selection, prefetch, and mmap setup |
| `weight_mapper_initialization` | Mapper creation/initialization after raw checkpoint loading and before model application |
| `weight_application_enqueue` | Model `_call_load_weights()` entry through return without adding a headline synchronization |
| `post_load_and_final_cuda_completion` | Model post-load hooks through the existing final model-load CUDA synchronization |

Use `time.perf_counter_ns()` and emit structured JSON or JSONL after measured regions. Headline runs must preserve
production overlap. Extra barriers and `torch.cuda.synchronize()` calls are allowed only in breakdown runs because
they can alter the critical path. The existing final synchronization makes `model_init_total` and
`checkpoint_to_cuda_total` safe primary spans.

Emit two distinct `profile_kind` values:

- `production_unperturbed`: authoritative E2E and inclusive totals. Intermediate H2D can complete during a later span,
  so subphases are not a pure hardware-work decomposition and inclusive parents must not be added to their children.
- `diagnostic_synchronized` or `diagnostic_nsys`: explicit CUDA completion boundaries and memcpy attribution. These
  runs explain mechanism but are excluded from headline E2E statistics.

Do not sum diagnostic phase maxima to reconstruct production startup.

### mmap and Page-Fault Attribution

SafeTensors uses mmap. Mapping a file does not prove its tensor pages were physically read; demand faults can occur
later during weight application. At each phase boundary capture:

- process `read_bytes` and `read_chars`;
- major and minor page faults;
- RSS, peak RSS, and node page-cache growth;
- logical checkpoint bytes;
- assigned file, extent, and byte counts per rank; and
- storage-side byte and request counters where available.

This separates a fast `load_file()` call from the physical reads it may defer.

### Distributed Aggregation

Startup completes after every required rank is ready. For durations report:

- `max_local_duration`: the largest process-local phase duration;
- `distributed_span`: `max(rank_end) - min(rank_start)` using a synchronized epoch or coordinator-observed events;
- externally observed process-to-ready and process-to-first-token latency;
- rank minimum, median, maximum, and start/end skew; and
- node-local sum for bytes and I/O operations.

Do not call `max_local_duration` the distributed critical path: a late-starting shorter rank can still determine
readiness. Do not add headline barriers merely to align timers; gather events after the measured region. Never sum
overlapping per-rank durations or use their mean as startup latency.

## 5. Policy and Resource Telemetry

Every result must include:

- `profile_kind`: `production_unperturbed`, `diagnostic_synchronized`, or `diagnostic_nsys`;

### Policy identity

- requested plan and strict/ordered mode;
- selected policy on every rank;
- eligibility decision and every skipped-policy reason;
- prefetch enabled or disabled;
- extent size, read size, worker caps, and actual workers;
- checkpoint manifest and backing-file identity; and
- selector/profile version for A1.

### Checkpoint geometry

- total logical and on-disk bytes;
- number of SafeTensors shards;
- minimum, median, maximum, and coefficient of variation of shard size;
- largest-shard fraction; and
- bytes assigned and completed by each rank.

### Host and storage

- filesystem type, mount source, options, and storage class;
- `MemAvailable` before load and checkpoint-to-available-memory ratio;
- logical bytes assigned and completed by the policy;
- process syscall-visible bytes such as `read_chars`;
- client block/filesystem-accounted bytes such as `read_bytes`, NFS, Lustre, or block-device counters;
- authoritative storage/server bytes when the platform exposes them;
- effective prefetch throughput from logical bytes divided by the distributed prefetch span;
- physical throughput, read amplification, request rate, request size, queue depth, and latency only when their
  authoritative counters are available;
- per-rank CPU utilization and aggregate peak RSS;
- page-cache residency before and after the run.

`read_chars` is logical syscall traffic, and process `read_bytes` is not a portable authoritative measure of NFS,
Lustre, or storage-appliance traffic. Gate physical-throughput and read-amplification claims on storage-side counters or
an isolated OS I/O trace. Otherwise report the metric at its actual accounting layer.

### GPU

- peak HBM by rank;
- H2D memcpy count, bytes, and duration in diagnostic runs;
- copy-engine utilization; and
- future NCCL/NVLink bytes for GPU fan-out.

Use Nsight Systems with CUDA, NVTX, and OS-runtime tracing for one or two representative diagnostic runs. Exclude those
runs from headline timing statistics. A synchronized wall timer around weight application includes CPU transforms,
residual page faults, and H2D completion; it must not be labeled pure H2D.

## 6. Controlled Initial Conditions

### Reproduction Identity

Record:

- TensorRT-LLM base and feature commit SHAs;
- container digest, CUDA, driver, NCCL, Python, PyTorch, and SafeTensors versions;
- immutable model revision and checkpoint file digests or manifest identity;
- GPU, CPU, NUMA, RAM, NIC, filesystem, and mount topology;
- GPU persistence mode, power limit, application clocks, and boost state;
- CPU affinity and worker configuration; and
- serving configuration, quantization, and parallel mapping.

An HGX B300 node provides eight NVLink-connected GPUs and approximately 2.30 TB aggregate HBM, but model admission
must use measured per-rank peak HBM rather than aggregate parameter arithmetic. See the
[NVIDIA HGX B300 component specification](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html).

### Weight-Cache State

Define at least these conditions:

| State | Meaning | Use |
| --- | --- | --- |
| Client cold, backend controlled | Client residency is below 1%; backend cache is reset and the reset is verified | Primary storage claim |
| Client cold, backend warm/unknown | Client cache is cold but NFS or object backend cache is not controlled | Useful production observation, labeled exactly |
| Client warm | Checkpoint pages remain resident | Restart/failover negative control |

For every cold trial on a dedicated node:

1. Terminate all prior server and worker processes and verify GPUs are empty.
2. Flush dirty data and use the approved privileged mechanism to drop client page cache.
3. Verify less than 1% of checkpoint pages are resident with `mincore`, `vmtouch`, or equivalent telemetry.
4. Start a fresh process group; never reuse an `LLM` instance.
5. Record client and storage-side counters.
6. Reset the client cache before the next treatment and, for the controlled-backend condition, reset and verify the
   backend cache too.

Dropping Linux page cache does not flush an NFS server or storage appliance cache. Prefer an administrator-supported
backend reset and verify it. If unavailable, inode-distinct checkpoint replicas can reduce client reuse but do not
prove that the backend is cold, and can change Lustre stripe placement or local physical layout. Orthogonally balance
treatment-to-replica assignment, record stripe/layout metadata, and label the backend state as uncontrolled.

### Compilation and Runtime Caches

The primary loader-isolation suite keeps non-weight compile, autotune, and kernel caches identical across treatments.
Use one prebuilt image and the same prepopulated or empty cache snapshot for every run. A secondary full-cold suite may
clear those caches to measure operational startup, but must report their time separately because loader policy should
not affect them.

### Full-Prefetch Guard

The cooperative prototype prefetches only when checkpoint bytes are below 90% of node-local `MemAvailable`. Reject or
separately classify a trial when:

- the guard disables prefetch;
- policies observe different available-memory decisions; or
- another process creates material memory pressure during the run.

Without this check, a nominal direct/shared trial may actually measure the same demand-mmap path as legacy.

## 7. Storage and Checkpoint Geometry Matrix

Storage behavior is the principal independent variable.

| Storage class | Expected use | Hypothesis |
| --- | --- | --- |
| Local NVMe or NVMe RAID, client cold | Node-local staging | Direct should benefit if throughput scales with queue depth and readers. |
| Production NFS, client cold | Common first load on a fresh node | Shared may benefit when multiple client processes contend; direct may win on a scalable server. |
| Lustre or another parallel filesystem, client cold | High-bandwidth cluster deployment | Direct should benefit when extents distribute across targets and queues. |
| Page-cache warm | Rapid restart control | All modes should converge; material overhead is a regression. |
| Object storage through ModelStreamer | Future source experiment | Tests source integration, not the current native mounted-file implementation. |

Use the publisher's natural shard layout for every headline claim. Add a diagnostic copy of one checkpoint with the
same tensor values resharded into 1, 8, and 64 approximately uniform files. This explains mechanism:

- one large shard stresses legacy whole-file assignment and should expose chunk-striping benefit;
- eight shards approximately match an eight-rank node;
- many uniform shards give legacy abundant file-level parallelism and may reduce the new policies' advantage.

Controlled layouts are explanatory and must not replace an unfavorable natural-layout result.

## 8. Tiered Model Plan

The benchmark is split into current qualification and future family qualification. Do not run a strict direct/shared
comparison on a model that the policy rejects and then report fallback timing as an optimized result.

### Tier 0: Smoke and Harness Validation

| Model | Precision | Topology | Storage | Runs |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-8B` | BF16 | TP8 | Production shared filesystem | 3 cold runs per treatment |

Purpose:

- validate timer schema, cache reset, strict selection, distributed completion, and correctness;
- estimate variance and choose the headline repetition count; and
- contribute observations to the separate paired base-versus-L equivalence gate.

### Tier 1: Current-Qualified Core

| Model | Why | Topologies | Storage |
| --- | --- | --- | --- |
| `Qwen/Qwen3-32B` BF16 | Dense model for rank-count and PP sweeps | TP1, TP2, TP4, TP8; TP4xPP2, TP2xPP4, TP1xPP8 | NVMe plus production NFS/Lustre |
| `meta-llama/Meta-Llama-3.1-405B` BF16 | Large dense checkpoint that exercises an 8xB300 node | TP8 and TP4xPP2 | NVMe plus production NFS/Lustre |

Use the largest accessible dense Llama revision that passes the exact class, mapper, HBM, and host-prefetch gates. If
the gated 405B artifact is unavailable, select the largest qualified Llama checkpoint and record the substitution.

The core headline subset is:

```text
2 models
x 2 eight-rank topologies (TP8, TP4xPP2)
x 2 cold storage classes (local NVMe, production shared storage)
x 4 treatments
x 10 paired repetitions
```

At the ten-repetition planning floor, this is 320 confirmatory startup trials per campaign. Run Campaign 0 first. Use a
separate five-repetition screening campaign to estimate variance, freeze the confirmatory sample count, and exclude
those screening observations from confirmatory intervals. Campaign 1 reruns all four treatments after A1 exists. The
broader TP/PP sweep uses five exploratory repetitions per cell.

### Tier 2: Parallelism Isolation

Use one qualified dense model to isolate how assignment changes with local rank count and ownership:

- TP1, TP2, TP4, and TP8;
- TP4xPP2, TP2xPP4, and TP1xPP8; and
- two simultaneous TP4 replicas and four simultaneous TP2 replicas as a separate replica-startup storm.

Independent serving replicas are not the same as one `Mapping.dp_size` dimension. Each communicator can independently
stage the same checkpoint, so launch-wave tests must report aggregate node and storage load as well as per-replica
readiness.

CP, EP, attention-DP, and DWDP are outside the current cooperative qualification envelope. They belong in Tier 3
after correctness support is added.

### Tier 3: Largest Practical Model in Each Requested Family

An 8xB300 fit does not imply cooperative-loader support. The PR base already has TensorRT-LLM model implementations
for all four families, including DeepSeek V4 on Blackwell. PR #16562 still rejects them because its cooperative
allowlist contains only dense Llama/Qwen2/Qwen3 and because these flagship paths add MoE/EP, attention DP, VLM,
compressed or quantized formats, MTP, and model-specific mapping requirements.

| Family | Exact campaign candidate and precision | Initial topology | Cooperative qualification required |
| --- | --- | --- | --- |
| DeepSeek V4 | [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), official mixed FP4/FP8 Instruct checkpoint | Checked-in TP8/EP8 shape, MTP disabled for loader isolation first | Model-class allowlist, mixed precision, MoE/EP, mapper, then MTP separately; use V4 Flash only if Pro fails a recorded fit gate. |
| Qwen 3.5 | [`nvidia/Qwen3.5-397B-A17B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4), NVFP4 | Checked-in TP4/EP4 with attention DP; then validated TP8/EP8 or two concurrent TP4 replicas | Qualify `Qwen/Qwen3.5-27B` BF16 as the dense stepping-stone, then class, NVFP4, MoE/EP, mapper, and attention DP. |
| Kimi K2 family | [`moonshotai/Kimi-K2.5`](https://huggingface.co/moonshotai/Kimi-K2.5), native INT4/compressed SafeTensors | Validated text-only TP8/EP8 first; keep the vision path separate | Kimi K2.5 class, compressed format, MoE/EP, mapper/custom code, text/VLM boundary, and attention DP. Use checked-in `nvidia/Kimi-K2-Thinking-NVFP4` as a separate K2 deployment control, not a silent substitute. |
| Llama | [`nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8`](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8), FP8 | Validated TP8/EP8 first | Llama 4 class, FP8, MoE/EP, mapper, and multimodal boundary. |

Before admission, the campaign manifest must replace all floating revisions with immutable repository commit IDs and
record total SafeTensors bytes, file geometry, and measured `MemAvailable`. Never benchmark `main` as an identity.
HBM fit and host-prefetch fit are independent: a checkpoint can fit in 2.30 TB aggregate HBM but fail the
`checkpoint_bytes < 0.9 * MemAvailable` guard.

At execution time, pin the exact model revision and choose the largest variant satisfying all gates:

1. TRT-LLM model and checkpoint mapping are supported on the tested commit.
2. Strict direct and shared policies select without fallback.
3. The exact SafeTensors and quantization format is qualified.
4. Checkpoint bytes satisfy the host full-prefetch guard.
5. Peak HBM stays below the predefined per-rank safety budget after KV-cache and warmup allocation.
6. Deterministic first inference matches legacy.

If the largest family model fails a gate, record the failure as a qualification result and test the next-largest
supported variant. Do not silently change precision or use a converted checkpoint only for one treatment.

### Tier 3 Parallelism Covering Set

Avoid a full Cartesian product. Use one dense and one MoE representative:

| Concern | Eight-GPU examples after qualification | What it reveals |
| --- | --- | --- |
| TP | TP8 | Rank slicing and I/O issuer scaling |
| PP | TP4xPP2, TP2xPP4 | Layer ownership; current full-checkpoint staging inefficiency |
| CP | TP4xCP2, or a validated TP/PP/CP product of eight | Replicated weight consumers and future producer groups |
| EP | A checked-in eight-rank TP/EP mapping such as TP8/EP8, plus TP8/EP1 control | Expert ownership and selective-read opportunity; EP does not multiply world size independently of TP/PP/CP |
| Attention DP/DWDP | On/off in a supported MoE configuration | Duplicated attention-weight demand and communicator semantics |
| Replica DP | 2xTP4 and 4xTP2 launched concurrently | Cold-start storms across independent communicators |

Use checked-in validated model configurations when possible. Do not invent a mathematically valid product that the
model or runtime does not support.

## 9. Execution Protocol

### Blocked Randomization

Treat one trial for each of L, D, S, and A as a block, where A is A0 in Campaign 0 and A1 in Campaign 1. Never mix A0
and A1 observations. Keep the same node and software image within a block, and balance treatment order across blocks.

Use one of two explicit storage protocols:

- **Controlled backend:** use the same immutable checkpoint replica for all four treatments and perform a verified
  client-and-backend cache reset before every treatment. Randomize treatment order within the block.
- **Uncontrolled backend:** do not claim backend-cold results. Use equivalent immutable checkpoint replicas and a
  Latin-square schedule that balances both treatment order and treatment-to-replica assignment across blocks. Record
  replica identity, stripe or physical layout, observed backend state, and time trend; include replica, layout, and
  order effects in the analysis.

The uncontrolled protocol cannot make backend state cold, but it prevents one treatment or one replica layout from
being systematically favored.

### Adaptive Calibration and Held-Out Validation

Campaign 1 has two disjoint stages:

1. **Calibration:** measure the I/O issuer curve on independent calibration objects and designated mounts; train or tune
   the selector; then freeze the profile contents, version, expiry, thresholds, and deployment weights.
2. **Held-out validation:** run L/D/S/A1 on predeclared mount identities, checkpoint geometries, or a later time window
   that was not used to tune the selector.

Compute oracle regret and any "adaptive is best" result only on the held-out set. If only one physical mount exists,
use disjoint calibration files and model geometries plus a time-separated validation campaign, and state the resulting
generalization limitation.

### Repetitions

| Stage | Repetitions per cell | Report |
| --- | --- | --- |
| Smoke | 3 | Individual observations and range; no distributional claim |
| Screening matrix | 5 | Individual observations, median, and range; intervals are exploratory only |
| Headline median-effect cells | 10 | Individual points, median, range, and paired median-speedup interval; no tail claim |
| Exploratory p90 campaign | Predeclared precision-driven count, at least 50 | p50/p90 with order-statistic uncertainty |
| Useful p95 campaign | Predeclared precision-driven count, at least 100 | p50/p90/p95; larger samples are required for SLO claims |
| Nsight diagnostic | 1-2 | Timeline and bytes; excluded from headline statistics |

Use pilot data to estimate variance, then freeze the sample count for each confirmatory campaign before collecting its
observations. If the resulting interval is inconclusive, report it as such; do not append observations in response to
the observed interval. A separately registered follow-up campaign may use a newly fixed sample count. Bootstrap
complete randomized blocks, not independent treatment observations. Do not select one "representative" run for the
headline comparison; preserve paired blocks and report the distribution. A representative run may be used only for an
internally consistent phase waterfall after the distribution is reported.

### Correctness Probe

Immediately after readiness, issue the fixed deterministic one-token request and stop every headline timer. Only then
run parameter fingerprinting, additional prompts and logits, log scanning, and the steady-state probe. The timed token
output remains part of the correctness comparison, but benchmark-only GPU reads or device-to-host copies must not
contaminate process-to-first-token latency.

Every measured run must then:

- confirm the requested and selected policy on every rank;
- confirm manifest and model revision identity;
- record parameter count and peak HBM;
- compare deterministic sampled post-load parameter fingerprints with L;
- compare the recorded timed token, then execute additional fixed greedy prompts and compare their output;
- compare selected logits within the dtype-specific tolerance; and
- scan logs for missing weights, unexpected fallback, collective, CUDA, or I/O errors.

Because load ordering should not alter tensor values, bit-exact sampled weight fingerprints are the preferred gate.
Tolerance applies to inference output only where later kernels are nondeterministic.

### Steady-State Non-Regression

After startup, run a short fixed throughput and TTFT workload. Loader policy should not alter steady-state weights or
runtime state. Report throughput, TTFT, and output correctness; a regression suggests unintended cache, placement, or
synchronization effects.

## 10. Analysis

For every paired block compute the L-relative effects:

```text
startup_speedup_pct = 100 * (T_L_e2e - T_mode_e2e) / T_L_e2e

checkpoint_to_cuda_speedup_pct =
    100 * (T_L_checkpoint_to_cuda - T_mode_checkpoint_to_cuda)
        / T_L_checkpoint_to_cuda

prefetch_speedup_x = T_L_prefetch / T_mode_prefetch
```

Define the oracle at the held-out **cell** level, not as a clairvoyant per-run minimum:

```text
oracle_cell_time = min(median(T_L), median(T_D), median(T_S))

adaptive_cell_regret_pct =
    100 * (median(T_A1) - oracle_cell_time) / oracle_cell_time
```

Bootstrap complete blocks and recompute the three static medians, oracle choice, and adaptive regret inside every
replicate. A minimum over L/D/S within each individual block is biased by run noise and must not be used for the A1
acceptance gate.

Report medians with paired-block bootstrap 95% confidence intervals. Report tails only for campaigns meeting the
predeclared precision-driven sample count. Across a predefined deployment mix, define aggregate improvement as the
geometric mean of positive time ratios `T_L / T_mode`, then convert the ratio to a percentage. Never take a geometric
mean of signed speedup percentages. An explicitly weighted mean must use weights frozen before results are inspected.

Also report:

- baseline fraction of E2E time spent in `checkpoint_to_cuda_total` and `model_init_total`;
- delta by phase, to show where time moved rather than disappeared;
- logical/effective throughput, plus physical throughput and read amplification only when authoritative counters exist;
- rank imbalance and slowest-rank identity;
- peak host memory and HBM deltas; and
- A1 decision accuracy and selector overhead.

Use Amdahl's law as a sanity check. Exact storage-wait fraction requires isolated storage-side or OS I/O tracing. When
that is unavailable, label cold-minus-warm delta or prefetch-span contribution as an estimate rather than measured I/O
time. If estimated checkpoint I/O is 40% of baseline startup, even infinite I/O speedup can reduce E2E by at most 40%
unless phases are overlapped or removed.

## 11. Hypotheses and Acceptance Gates

### Mechanism Hypotheses

- D should beat L when storage throughput scales with disjoint readers, especially for a few large or skewed shards.
- S may beat L and D when the storage client penalizes multiple processes or connections.
- L may remain competitive with many uniform shards because it already parallelizes whole files across ranks.
- All policies should converge in warm-cache cells; material extra overhead is a regression.
- A1 should select D on scalable storage, S on client-limited storage, and L when unsupported or empirically fastest.

### Functional Gates

- Strict D and S select the exact requested policy on every rank; any fallback invalidates the trial.
- All ranks complete without deadlock, timeout, or divergent manifest/plan.
- Weight fingerprints and deterministic inference match L.
- Full-prefetch and memory-guard state are present in the artifact.
- No mode exceeds the agreed host or HBM budget.

### Suggested Performance Gates

A result is statistically positive when the paired-bootstrap 95% confidence interval for speedup excludes zero. A
material target-cell win is:

- at least 10% median `checkpoint_to_cuda_total` reduction; and
- at least 5% median process-to-first-token reduction.

These thresholds are initial decision gates, not facts about expected hardware performance.

To claim that both static modes beat legacy overall, each static mode must have a positive lower confidence bound for
`GM(T_L / T_mode) - 1` across the predefined production cells. A favorable synthetic one-shard diagnostic is not
sufficient.

To claim A1 is the best general policy:

- median adaptive regret is at most 5%; p90 regret is at most 10% only in a separately funded tail campaign;
- selector overhead is below 1% of E2E startup;
- no qualified cell materially regresses relative to L without an explicit fallback; and
- aggregate startup across the frozen deployment mix beats both fixed D and fixed S with a confidence interval that
  excludes parity.

Per-cell dominance over the oracle is neither required nor credible for a selector. In Campaign 0, where A is the
current A0 ordered fallback, the correct expected result is parity with D on eligible cells and parity with L where
direct is not eligible.

Steady-state throughput and TTFT should remain within 2% of L unless normal run-to-run variance justifies a wider
predeclared bound.

## 12. Result Tables

### Headline

| Model | Topology | Storage/cache | Campaign | Treatment | Selected policy | Profile kind | First-token median [CI] | E2E speedup | Checkpoint-to-CUDA speedup | Peak RSS/HBM | Result |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| | | | 0 or 1 | L | | production_unperturbed | | baseline | baseline | | |
| | | | | D | | production_unperturbed | | | | | |
| | | | | S | | production_unperturbed | | | | | |
| | | | 0 | A0 | | production_unperturbed | | | | | |
| | | | 1 | A1 | | production_unperturbed | | | | | |

### Phase Attribution

| Cell | Treatment | Profile kind | Model-init total | Checkpoint-to-CUDA total | Raw checkpoint load | Discovery | Selection | Prefetch | mmap/setup | Apply enqueue | Post-load/final CUDA | Warmup/tail | E2E |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | | production_unperturbed | | | | | | | | | | | |

Inclusive parents overlap their children, and production H2D completion can land in the post-load span. Do not add
columns to reconstruct E2E. Publish synchronized/Nsight H2D attribution in a separate diagnostic table.

### Adaptive Selection

| Held-out cell | Profile version | Oracle policy | Adaptive policy | Oracle time | Adaptive time | Regret | Decision reason |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| | | | | | | | |

## 13. Decision Outcomes

The experiment leads to one of these actions:

| Evidence | Decision |
| --- | --- |
| D wins consistently on target scalable storage | Keep D as primary for those profiled deployments. |
| S wins on client-limited storage | Implement and test the full bounded pinned producer/consumer pipeline. |
| L is competitive for natural many-shard layouts | Preserve L and include shard geometry in selection. |
| A1 stays near the oracle on held-out cells | Promote adaptive selection behind an explicit experimental configuration. |
| A1 has high regret or unstable decisions | Use deployment-static policy profiles rather than per-start auto selection. |
| I/O improves but E2E barely moves | Prioritize mapping, transforms, H2D overlap, post-load work, or reusable artifacts. |
| Largest models fail the host-memory guard | Move to bounded streaming/selective reads instead of full page-cache prefetch. |
| MoE/CP/DP qualification fails | Keep fallback and fix rank-ownership/communicator contracts before performance work. |

## 14. Required Artifacts

Store one directory per immutable run campaign:

```text
campaign/
├── manifest.json
├── environment/
│   ├── software.json
│   ├── hardware.json
│   ├── topology.txt
│   └── storage.json
├── cells/<cell-id>/<treatment>/<run-id>/
│   ├── startup-profile.json
│   ├── rank-events.jsonl
│   ├── resource-telemetry.jsonl
│   ├── stdout.log
│   ├── correctness.json
│   └── command.json
├── aggregate.csv
├── aggregate.json
└── report.md
```

The report must list excluded runs and reasons. Never silently discard an outlier; distinguish infrastructure failure,
invalid cache state, unexpected fallback, correctness failure, and valid performance variation.

## References

- [Native Hybrid Weight Loader](README.md)
- [TensorRT-LLM PR #16562](https://github.com/NVIDIA/TensorRT-LLM/pull/16562)
- [Startup Methodology and Test Plan](../mx-gms-integration/10-methodology.md)
- [ModelStreamer and Weight-Loading Integration Assessment](../mx-gms-integration/19-model-streamer-weight-loading-assessment.md)
- [TensorRT-LLM performance benchmarking guide](../../source/developer-guide/perf-benchmarking.md)
- [TensorRT-LLM supported models](../../source/models/supported-models.md)
