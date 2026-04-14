# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU Memory Backend protocol and GMS implementation.

Defines a thin protocol to insulate TRT-LLM from potential GMS API changes.
If the GMS API evolves, only the concrete implementation needs updating.

The protocol supports two operating modes:
- **RW (Read-Write)**: First worker loads weights via the normal checkpoint
  loader pipeline, but allocations go into a GMS-managed CUDA memory pool.
  After loading, the weights are committed for read-only access by others.
- **RO (Read-Only)**: Subsequent workers zero-copy import already-committed
  weights from the GMS pool.  ``post_load_weights()`` must run BEFORE
  materialization so that module aliases are set up correctly.
"""

from typing import Optional, Protocol, runtime_checkable

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@runtime_checkable
class GPUMemoryBackend(Protocol):
    """Thin abstraction over GPU memory services for API stability.

    Any concrete backend (GMS, CUDA IPC fallback, etc.) should implement
    these methods to work with TRT-LLM's ``ModelLoader.load()`` GMS branch.
    """

    def has_committed_weights(self, tag: str) -> bool:
        """Check if weights with the given tag are already committed (RO-ready)."""
        ...

    def get_mem_pool(self) -> torch.cuda.MemPool:
        """Return a CUDA MemPool for RW-path allocations."""
        ...

    def materialize_module(self, model: nn.Module) -> None:
        """Zero-copy import committed weights into model params (RO path)."""
        ...

    def finalize_write(self, model: nn.Module, tag: str) -> None:
        """Register model tensors and commit them for RO readers (RW path)."""
        ...

    def release(self, tag: str) -> None:
        """Release committed memory for a given tag."""
        ...

    def cleanup(self) -> None:
        """Release all resources and disconnect."""
        ...


class GMSBackend:
    """Concrete GPUMemoryBackend using the GPU Memory Service (GMS) library.

    GMS provides a shared GPU memory pool that multiple inference instances
    can map model weights from.  Weights are loaded once by a RW worker and
    served to RO consumers as read-only CUDA memory regions.

    This implementation delegates to ``gpu_memory_service.client`` for all
    heavy lifting (CUDA VMM, FD passing, zero-copy tensor construction).
    TRT-LLM's role is orchestration — calling these APIs at the correct
    points in the model loading lifecycle.
    """

    def __init__(self, socket_path: Optional[str], mapping: Mapping,
                 mode: str = "auto", tag: str = "model_weights"):
        """Initialize the GMS backend.

        Args:
            socket_path: Unix domain socket path for GMS. If None, uses
                default ``/tmp/gms-{device_id}.sock``.
            mapping: Distributed mapping for TP/PP rank info.
            mode: Operating mode — "auto", "rw", or "ro".
            tag: Tag identifying the weight set in GMS.
        """
        if socket_path is None:
            device_id = torch.cuda.current_device()
            socket_path = f"/tmp/gms-{device_id}.sock"

        self._socket_path = socket_path
        self._mapping = mapping
        self._mode = mode
        self._tag = tag
        self._client = None
        self._is_rw = None  # Resolved during connect

    def connect(self) -> bool:
        """Establish connection to the GMS server.

        In "auto" mode, checks whether committed weights exist to decide
        between RW (first writer) and RO (read-only) modes.

        Returns:
            True if connection succeeded, False otherwise.
        """
        try:
            from gpu_memory_service import client as gms_client  # type: ignore[import-not-found]

            if self._mode == "rw":
                self._client = gms_client.connect(
                    self._socket_path, mode="rw")
                self._is_rw = True
            elif self._mode == "ro":
                self._client = gms_client.connect(
                    self._socket_path, mode="ro")
                self._is_rw = False
            else:
                # Auto mode: try RO first (fast path), fall back to RW.
                self._client = gms_client.connect(self._socket_path)
                self._is_rw = not self._client.has_committed_weights(
                    self._tag)

            logger.info(
                "Connected to GMS at %s (mode=%s, resolved=%s)",
                self._socket_path, self._mode,
                "rw" if self._is_rw else "ro")
            return True
        except ImportError:
            logger.warning(
                "gpu_memory_service library not installed; cannot use GPU "
                "memory serving. Install with: pip install nvidia-gms")
            return False
        except Exception as e:
            logger.warning("Failed to connect to GMS at %s: %s",
                           self._socket_path, e)
            return False

    @property
    def is_rw(self) -> Optional[bool]:
        """Whether this backend is in RW mode. None if not yet connected."""
        return self._is_rw

    def has_committed_weights(self, tag: str) -> bool:
        if self._client is None:
            return False
        try:
            return self._client.has_committed_weights(tag)
        except Exception:
            return False

    def get_mem_pool(self) -> torch.cuda.MemPool:
        """Return a GMS-managed CUDA MemPool for RW-path allocations.

        Torch allocations made inside ``torch.cuda.use_mem_pool(pool)``
        are intercepted by GMS's CUDAPluggableAllocator and placed in the
        shared memory region.
        """
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")

        from gpu_memory_service import client as gms_client  # type: ignore[import-not-found]
        return gms_client.get_mem_pool(self._client)

    def materialize_module(self, model: nn.Module) -> None:
        """Zero-copy import committed weights into model params (RO path).

        The GMS library walks the model's parameters and buffers, creating
        tensors backed by GPU pointers from the shared memory region.
        This replaces meta-initialized parameters with real CUDA tensors
        without any data copies.

        **Important**: ``post_load_weights()`` must be called on the model
        BEFORE this method, so that module aliases and derived parameters
        are set up correctly before materialization.
        """
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")

        from gpu_memory_service import client as gms_client  # type: ignore[import-not-found]
        gms_client.materialize_module_from_gms(self._client, model)

        # Mark Linear modules as presharded — GMS provides per-rank
        # weights that are already TP-sliced.
        from tensorrt_llm._torch.modules.linear import Linear
        for module in model.modules():
            if isinstance(module, Linear):
                module._weights_presharded = True

        logger.info(
            "GMS RO: materialized weights from %s (tag=%s, tp_rank=%d/%d)",
            self._socket_path, self._tag,
            self._mapping.tp_rank, self._mapping.tp_size)

    def finalize_write(self, model: nn.Module, tag: str = None) -> None:
        """Register model tensors and commit for RO readers (RW path).

        After the standard weight loading pipeline completes (under the
        GMS mem pool), this method tells GMS which tensors belong to the
        model and commits them for read-only access.
        """
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")
        if tag is None:
            tag = self._tag

        from gpu_memory_service import client as gms_client  # type: ignore[import-not-found]
        gms_client.register_module_tensors(self._client, model)
        gms_client.commit(self._client, tag)
        logger.info(
            "GMS RW: committed weights at %s (tag=%s)", self._socket_path, tag)

    def release(self, tag: str = None) -> None:
        if self._client is None:
            return
        if tag is None:
            tag = self._tag
        try:
            from gpu_memory_service import client as gms_client  # type: ignore[import-not-found]
            gms_client.release(self._client, tag)
        except Exception as e:
            logger.warning("GMS release error: %s", e)

    def cleanup(self) -> None:
        """Disconnect from GMS."""
        if self._client is not None:
            try:
                from gpu_memory_service import client as gms_client  # type: ignore[import-not-found]
                gms_client.disconnect(self._client)
                logger.info("GMS: disconnected from %s", self._socket_path)
            except Exception as e:
                logger.warning("GMS cleanup error: %s", e)
            finally:
                self._client = None
