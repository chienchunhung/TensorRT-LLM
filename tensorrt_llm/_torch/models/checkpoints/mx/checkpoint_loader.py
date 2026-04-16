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

import os
from typing import Any, Optional

from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import \
    HfCheckpointLoader
from tensorrt_llm._torch.models.modeling_utils import \
    register_checkpoint_loader
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@register_checkpoint_loader("MX")
class MXCheckpointLoader(HfCheckpointLoader):
    """Checkpoint loader for MX (Model eXchange) P2P weight transfer.

    When an MX server is available, weights are transferred directly from a
    source instance via P2P (GPU-to-GPU or NVLink), bypassing disk I/O.
    The source publishes weights *before* post_load_weights() runs, so the
    target receives raw loaded state and applies its own transforms.

    When no MX server is reachable, this loader transparently falls back to
    standard HuggingFace checkpoint loading (disk -> CPU -> GPU).
    """

    def __init__(
        self,
        *,
        weight_loader: Optional[BaseWeightLoader] = None,
        weight_mapper: Optional[BaseWeightMapper] = None,
        config_loader: Optional[BaseConfigLoader] = None,
        mx_server_url: Optional[str] = None,
    ):
        super().__init__(
            weight_loader=weight_loader,
            weight_mapper=weight_mapper,
            config_loader=config_loader,
        )
        # Align the backing attribute with the property override, so code
        # that reads self._checkpoint_format directly (instead of the
        # property) also sees "MX".
        self._checkpoint_format = "MX"
        self._mx_server_url = mx_server_url or os.environ.get(
            "MODEL_EXPRESS_URL")
        self._p2p_succeeded = False
        self._identity = None

    @property
    def checkpoint_format(self) -> str:
        """Override parent's checkpoint_format to return 'MX'."""
        return "MX"

    @property
    def mx_server_url(self) -> Optional[str]:
        return self._mx_server_url

    @property
    def p2p_succeeded(self) -> bool:
        """Whether the last load_weights() call used P2P transfer."""
        return self._p2p_succeeded

    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> dict[str, Any]:
        """Load weights, preferring MX P2P transfer when available.

        When P2P succeeds, weights are written directly into the model's
        parameter buffers by the MX library.  This method returns an empty
        dict so the caller knows to skip the normal weight-mapping pipeline.

        When P2P is unavailable, falls back to the parent HF weight loader.

        Args:
            checkpoint_dir: Path to the HF checkpoint directory.
            mapping: Distributed mapping configuration.
            **kwargs: Additional keyword arguments. When ``model`` is passed,
                it is used as the target for direct P2P weight writes.

        Returns:
            A weights dict.  Empty when P2P succeeded (weights already in
            model params); populated when falling back to disk loading.
        """
        model = kwargs.pop("model", None)
        self._p2p_succeeded = False

        if self._mx_server_url is not None and model is not None:
            if self._try_p2p_transfer(model, mapping, checkpoint_dir):
                self._p2p_succeeded = True
                logger.info(
                    "MX P2P weight transfer succeeded from %s",
                    self._mx_server_url,
                )
                return {}

        # Fallback: load from disk using the standard HF weight loader.
        logger.info(
            "MX P2P unavailable, falling back to HF checkpoint loading "
            "from %s",
            checkpoint_dir,
        )
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    def _build_identity(self, mapping: Mapping, checkpoint_dir: str) -> Any:
        """Map TRT-LLM's Mapping to MX's SourceIdentity protobuf.

        This encodes the full distributed topology (TP, PP, EP) so the MX
        server can match compatible sources for P2P transfer.
        """
        try:
            from modelexpress import proto as mx_proto  # type: ignore[import-not-found]

            return mx_proto.SourceIdentity(
                model_name=checkpoint_dir,
                dtype=str(mapping.dtype) if hasattr(mapping, 'dtype') else "",
                extra_params={
                    "tp_size": str(mapping.tp_size),
                    "pp_size": str(mapping.pp_size),
                    "ep_size": str(mapping.moe_ep_size),
                    "worker_rank": str(mapping.tp_rank),
                    "pp_rank": str(mapping.pp_rank),
                },
            )
        except ImportError:
            return None

    def _try_p2p_transfer(self, model, mapping: Mapping,
                          checkpoint_dir: str) -> bool:
        """Attempt P2P weight transfer from the MX server.

        Uses the modelexpress client SDK to discover compatible sources
        and perform NIXL-based GPU-to-GPU transfer.

        Returns:
            True if transfer succeeded, False otherwise.
        """
        try:
            from modelexpress import client as mx_client  # type: ignore[import-not-found]

            identity = self._build_identity(mapping, checkpoint_dir)
            if identity is None:
                logger.warning(
                    "Could not build MX identity (modelexpress.proto "
                    "not available). Skipping P2P transfer.")
                return False
            self._identity = identity

            connection = mx_client.connect(self._mx_server_url)
            sources = connection.list_sources(identity)

            # Find a source that matches this rank's topology.
            compatible = [
                s for s in sources
                if s.worker_rank == mapping.tp_rank
                and s.extra_params.get("pp_rank") == str(mapping.pp_rank)
            ]

            if compatible:
                connection.receive(compatible[0])
                return True
            else:
                logger.info(
                    "No compatible MX source found for tp_rank=%d, "
                    "pp_rank=%d. Falling back to disk loading.",
                    mapping.tp_rank, mapping.pp_rank)
                return False

        except ImportError:
            logger.warning(
                "modelexpress library not installed; cannot use P2P weight "
                "transfer. Install with: pip install nvidia-modelexpress")
            return False
        except Exception as e:
            logger.warning("MX P2P transfer failed: %s. "
                           "Falling back to disk loading.", e)
            return False

    def publish_as_source(self, model, mapping: Mapping = None,
                          checkpoint_dir: str = None) -> None:
        """Publish this instance's weights so other ranks can pull via P2P.

        Called *before* post_load_weights() so targets receive the raw
        loaded state and can apply their own post-load transforms.

        Args:
            model: The model whose weights to publish.
            mapping: Distributed mapping (used to build identity if not
                already cached from load_weights).
            checkpoint_dir: Checkpoint directory (used for model identity).
        """
        if self._mx_server_url is None:
            return

        try:
            from modelexpress import client as mx_client  # type: ignore[import-not-found]

            # Build identity if not cached from a prior load_weights() call.
            identity = self._identity
            if identity is None and mapping is not None and checkpoint_dir is not None:
                identity = self._build_identity(mapping, checkpoint_dir)

            if identity is None:
                logger.warning(
                    "Cannot publish MX source: no identity available. "
                    "Provide mapping and checkpoint_dir arguments.")
                return

            connection = mx_client.connect(self._mx_server_url)
            connection.register_source(model, identity)
            logger.info("Published weights to MX server at %s",
                        self._mx_server_url)
        except ImportError:
            logger.debug(
                "modelexpress library not installed; skipping publish.")
        except Exception as e:
            logger.warning("Failed to publish weights to MX server: %s", e)
