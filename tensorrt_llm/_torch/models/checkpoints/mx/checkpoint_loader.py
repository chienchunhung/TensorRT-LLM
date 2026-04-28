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
    """Checkpoint loader for MX (ModelExpress) P2P weight transfer.

    When an MX server is available with a published source, weights are
    transferred directly via NIXL RDMA (GPU-to-GPU), bypassing disk I/O.
    The source publishes weights *before* post_load_weights() runs, so the
    target receives raw loaded state and applies its own transforms.

    When no MX server or source is reachable, falls back to standard
    HuggingFace checkpoint loading (disk -> CPU -> GPU).
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
        self._checkpoint_format = "MX"
        self._mx_server_url = mx_server_url or os.environ.get(
            "MODEL_EXPRESS_URL")
        self._p2p_succeeded = False

    @property
    def checkpoint_format(self) -> str:
        return "MX"

    @property
    def mx_server_url(self) -> Optional[str]:
        return self._mx_server_url

    @property
    def p2p_succeeded(self) -> bool:
        return self._p2p_succeeded

    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> dict[str, Any]:
        """Load weights, preferring MX P2P transfer when available.

        When P2P succeeds, weights are written directly into the model's
        parameter buffers by the MX library.  Returns an empty dict so the
        caller skips the normal weight-mapping pipeline.

        Auto-detect logic: probe the MX server for existing sources.
        If sources exist, attempt RDMA receive (target mode).
        If no sources, fall back to disk loading (source mode).
        """
        model = kwargs.pop("model", None)
        self._p2p_succeeded = False
        self._mapping = mapping
        self._checkpoint_dir = checkpoint_dir

        is_target = os.environ.get("MODEL_EXPRESS_TARGET", "") == "1"

        if self._mx_server_url and model is not None:
            if is_target or self._has_existing_sources():
                logger.info("MX sources detected, attempting P2P transfer...")
                if self._try_p2p_transfer(model, mapping, checkpoint_dir):
                    self._p2p_succeeded = True
                    logger.info(
                        "MX P2P weight transfer succeeded from %s",
                        self._mx_server_url,
                    )
                    return {}
            else:
                logger.info(
                    "No MX sources found — loading from disk "
                    "(this instance will become a source).")

        logger.info(
            "Falling back to HF checkpoint loading from %s",
            checkpoint_dir,
        )
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    def _has_existing_sources(self) -> bool:
        """Quick probe: are there any MX sources already published?

        Used to auto-detect target mode. A short timeout avoids blocking
        source instances that should fall through to disk loading.
        """
        try:
            from modelexpress.client import MxClient
            from modelexpress.trtllm_live_transfer import \
                _build_trtllm_identity

            model_name = os.environ.get(
                "MODEL_NAME",
                os.path.basename(self._checkpoint_dir
                                 if hasattr(self, '_checkpoint_dir')
                                 else "unknown"))
            identity = _build_trtllm_identity(model_name=model_name)
            client = MxClient(server_url=self._mx_server_url)
            try:
                resp = client.list_sources(identity=identity)
                return bool(resp.instances)
            finally:
                client.close()
        except Exception as e:
            logger.debug("MX source probe failed: %s", e)
            return False

    def _try_p2p_transfer(self, model, mapping: Mapping,
                          checkpoint_dir: str) -> bool:
        """Attempt P2P weight transfer using MxLiveWeightLoader.

        Delegates to the battle-tested MxLiveWeightLoader from
        modelexpress.trtllm_live_transfer, which handles source
        discovery, NIXL initialization, RDMA transfer, dtype casting,
        and size-mismatch fallback.
        """
        try:
            from modelexpress.trtllm_live_transfer import MxLiveWeightLoader
        except ImportError:
            logger.warning(
                "modelexpress library not installed; cannot use P2P. "
                "Install with: pip install nvidia-modelexpress")
            return False

        try:
            loader = MxLiveWeightLoader(mx_server=self._mx_server_url)
            fallback_weights = loader.load_weights(
                checkpoint_dir=checkpoint_dir,
                mapping=mapping,
                model=model,
            )
            if fallback_weights:
                logger.info(
                    "MX P2P: %d tensors need disk fallback after RDMA",
                    len(fallback_weights))
            return True
        except TimeoutError:
            logger.info(
                "No MX source found within timeout — this instance "
                "will load from disk and become a source.")
            return False
        except Exception as e:
            logger.warning("MX P2P transfer failed: %s", e)
            return False

    def publish_as_source(self, model, mapping: Mapping = None,
                          checkpoint_dir: str = None) -> None:
        """Publish this instance's weights so other ranks can pull via P2P.

        Called from model_loader.py *before* post_load_weights() so targets
        receive the raw loaded state.  Delegates to the proven
        publish_model_params() from modelexpress.trtllm_live_transfer.
        """
        if not self._mx_server_url:
            return

        try:
            from modelexpress.trtllm_live_transfer import publish_model_params
        except ImportError:
            logger.debug(
                "modelexpress library not installed; skipping publish.")
            return

        os.environ.setdefault("MODEL_EXPRESS_URL", self._mx_server_url)
        if checkpoint_dir:
            os.environ.setdefault(
                "MODEL_NAME", os.path.basename(checkpoint_dir))

        try:
            publish_model_params(model)
            logger.info("Published weights to MX server at %s",
                        self._mx_server_url)
        except Exception as e:
            logger.warning(
                "Failed to publish weights to MX server: %s", e)
