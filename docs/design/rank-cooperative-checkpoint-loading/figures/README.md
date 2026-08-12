<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rank-Cooperative Checkpoint Loading Figures

`checkpoint-loading-policy-comparison.svg` is the editable vector source for the
three-policy architecture comparison. The matching PNG is the raster asset used
when an embedding surface does not accept SVG.

`rank-cooperative-streaming-pipeline.svg` is the editable vector source for the
RANK-STREAM steady-state workflow. It emphasizes the concurrent producer and
consumer lanes, the node-local double buffer, and the consensus-gated slot swap.
The matching PNG is the raster asset used by Google Docs.

`startup-to-first-token-timeline.svg` is the editable vector source for the
end-to-end startup timeline. It distinguishes the model-weight-loading phase,
the checkpoint I/O directly accelerated by RANK-STRIPED, and the unchanged
native materialization work that can overlap with read-ahead. The matching PNG
is the raster asset used by Google Docs.
