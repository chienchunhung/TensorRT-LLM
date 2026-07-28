<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rank-Cooperative Checkpoint Loading

*Parallel read-ahead and pipelined materialization for faster cold starts.*

**Status:** Draft design; host-staging prototype in
[TensorRT-LLM PR #16562](https://github.com/NVIDIA/TensorRT-LLM/pull/16562)

This package treats fast, efficient TensorRT-LLM startup as the product objective and checkpoint loading as the
current, bounded optimization problem. The checkpoint-loading interfaces remain composable with process snapshots,
MX/GMS artifact reuse, ModelStreamer and other raw-weight sources, future rank-aware materialization, GPU placement,
compilation, warmup, and other startup improvements.

## Documents

| Document | Purpose |
| --- | --- |
| [Main design](design.md) | Architecture, policy semantics, integration boundaries, and rollout plan. |
| [Experiment and benchmark plan](benchmark-plan.md) | Reproducible 8xB300 campaign, metrics, qualification matrix, acceptance gates, and measured results in Appendices A-B. |

The latest measured evidence is the preliminary two-block, same-node Qwen3.5-397B-A17B-FP8 qualification in
`benchmark-plan.md` Appendix B. Results from different models, PR revisions, nodes, and cache protocols are listed as
separate rounds and must not be combined.

## Scope

The current implementation focuses on filesystem-visible SafeTensors checkpoints and the path from raw checkpoint
bytes through materialization and existing H2D placement. It separates three mechanisms:

- **Rank-Striped Read-Ahead**: ranks issue disjoint background reads into the Linux page cache while the unchanged
  mmap-driven loader proceeds. This is opportunistic read-ahead, not a bounded data stream.
- **Node-Shared Weight Streaming**: one producer process per node fills a bounded shared-memory double buffer while
  every local rank consumes the previously published batch. This is the Yijin-style single-producer treatment.
- **Rank-Cooperative Weight Streaming**: multiple node-local rank processes collectively fill disjoint regions of the
  same bounded shared slots while consumers materialize the previous batch.

The two streaming mechanisms deliberately share batch planning, mapper contracts, consumer materialization, shared
memory, and H2D behavior; only their I/O producer assignment differs. This makes their benchmark comparison isolate
single-process versus multi-rank storage issuance. The package measures end-to-end process-to-first-token latency so a
local checkpoint-loading gain is accepted only when it improves the larger startup critical path.

Higher-level mechanisms remain separate and complementary:

- a valid process snapshot can skip most startup work;
- MX or GMS can reuse compatible materialized weights;
- ModelStreamer can provide a parallel local or object-storage raw-byte source;
- rank-aware selective reads and topology-aware GPU fan-out can optimize future materialization and placement; and
- compilation, autotuning, KV-cache initialization, CUDA graph capture, and first-request readiness remain distinct
  phases with their own optimization opportunities.

The design therefore exposes seams between source selection, I/O policy, materialization, placement, and reusable
artifacts instead of turning the checkpoint loader into a monolithic startup orchestrator.
