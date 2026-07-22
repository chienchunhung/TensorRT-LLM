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
| [Experiment and benchmark plan](benchmark-plan.md) | Reproducible 8xB300 campaign, metrics, qualification matrix, and acceptance gates. |

## Scope

The current implementation focuses on filesystem-visible SafeTensors checkpoints and the path from raw checkpoint
bytes through materialization and existing H2D placement. It compares rank-owned page-cache read-ahead with a
Yijin-style, one-producer-per-node bounded shared-memory stream that pipelines I/O, dependency-group materialization,
and H2D. It measures end-to-end process-to-first-token latency so a local checkpoint-loading gain is accepted only when
it improves the larger startup critical path.

Higher-level mechanisms remain separate and complementary:

- a valid process snapshot can skip most startup work;
- MX or GMS can reuse compatible materialized weights;
- ModelStreamer can provide a parallel local or object-storage raw-byte source;
- rank-aware selective reads and topology-aware GPU fan-out can optimize future materialization and placement; and
- compilation, autotuning, KV-cache initialization, CUDA graph capture, and first-request readiness remain distinct
  phases with their own optimization opportunities.

The design therefore exposes seams between source selection, I/O policy, materialization, placement, and reusable
artifacts instead of turning the checkpoint loader into a monolithic startup orchestrator.
