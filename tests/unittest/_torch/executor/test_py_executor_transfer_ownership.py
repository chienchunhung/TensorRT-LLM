# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def test_resource_manager_shutdown_waits_for_transfer_drain() -> None:
    executor = PyExecutor.__new__(PyExecutor)
    transceiver = Mock()
    transceiver.requires_physical_drain_before_request_release = True
    transceiver.shutdown.return_value = False
    manager = Mock()
    executor.kv_cache_transceiver = transceiver
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": manager})

    with pytest.raises(RuntimeError, match="still owns physical accessors"):
        executor._shutdown_resource_managers()

    transceiver.shutdown.assert_called_once_with()
    manager.shutdown.assert_not_called()


def test_resource_manager_shutdown_follows_transfer_drain() -> None:
    calls = []
    executor = PyExecutor.__new__(PyExecutor)
    transceiver = Mock()
    transceiver.requires_physical_drain_before_request_release = True
    transceiver.shutdown.side_effect = lambda: calls.append("transceiver") or True
    manager = Mock()
    manager.shutdown.side_effect = lambda: calls.append("manager")
    executor.kv_cache_transceiver = transceiver
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": manager})

    executor._shutdown_resource_managers()

    assert calls == ["transceiver", "manager"]


def test_in_doubt_python_transfer_crashes_without_freeing_requests() -> None:
    executor = PyExecutor.__new__(PyExecutor)
    fault = RuntimeError("backend completion unknown")
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = True
    executor.kv_cache_transceiver.get_physical_ownership_fault.return_value = fault
    executor._handle_errors = Mock()
    executor.resource_manager = Mock()

    with pytest.raises(RuntimeError, match="IN_DOUBT") as raised:
        executor._handle_disagg_cache_errors_synced()

    assert raised.value.__cause__ is fault
    executor._handle_errors.assert_not_called()
    executor.resource_manager.free_resources.assert_not_called()


def test_logical_error_cannot_free_request_with_active_transfer_owner() -> None:
    request = SimpleNamespace(py_request_id=17)
    executor = PyExecutor.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = True
    executor.kv_cache_transceiver.cancel_request.return_value = False
    executor.resource_manager = Mock()

    with pytest.raises(RuntimeError, match="still owns physical accessors"):
        executor._do_terminate_request(request)

    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)
    executor.resource_manager.free_resources.assert_not_called()


def test_logical_error_frees_request_only_after_transfer_owner_drains() -> None:
    calls = []
    request = SimpleNamespace(py_request_id=19)
    executor = PyExecutor.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.requires_physical_drain_before_request_release = True
    executor.kv_cache_transceiver.cancel_request.side_effect = lambda _request: (
        calls.append("drain") or True
    )
    executor.resource_manager = Mock()
    executor.resource_manager.free_resources.side_effect = lambda _request: calls.append("free")
    executor._prefetched_request_ids = set()
    executor._disagg_timed_out_ctx_cancelled_ids = set()
    executor._disagg_timed_out_gen_cancelled_ids = set()
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=1)

    executor._do_terminate_request(request)

    assert calls == ["drain", "free"]
