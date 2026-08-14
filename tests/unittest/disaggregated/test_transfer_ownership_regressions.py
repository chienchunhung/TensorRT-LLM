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
"""Regressions for receive-side KV transfer ownership boundaries."""

from __future__ import annotations

import gc
import queue
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
import tensorrt_llm._torch.disaggregation.transceiver as transceiver_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    KVSendTask,
    MessageType,
    PeerIncompatibleError,
    Receiver,
    RxSession,
    Sender,
    TaskStatus,
    TransferWorker,
    TxSession,
    WriteMeta,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

_GLOBAL_RID_BASE = 1 << 40


class _BounceProbe:
    """No-bounce probe that records physical-owner cleanup decisions."""

    def __init__(self) -> None:
        self.failed_writers: list[tuple[tuple[int, int], int]] = []
        self.orphaned: list[tuple[int, int]] = []

    def record_failure(self, rid_slice: tuple[int, int], peer_rank: int) -> None:
        self.failed_writers.append((rid_slice, peer_rank))

    def reserve(self, _receiver_req, _num_writers: int, *, extra_bytes: int = 0) -> bool:
        del extra_bytes
        return False

    def release_idle_reservation(self, _rid_slice: tuple[int, int]) -> None:
        return

    def orphan_reservation(self, rid_slice: tuple[int, int]) -> None:
        self.orphaned.append(rid_slice)

    def is_bounced(self, _rid_slice: tuple[int, int]) -> bool:
        return False


class _TrackingLock:
    """Expose whether cancellation found the publication gate held."""

    def __init__(self, race_outcomes: queue.Queue[str]) -> None:
        self._lock = threading.Lock()
        self._race_outcomes = race_outcomes
        self.tracked_thread_id: int | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if threading.get_ident() == self.tracked_thread_id:
            if self._lock.acquire(blocking=False):
                return True
            self._race_outcomes.put("blocked")
            if not blocking:
                return False
        if timeout == -1:
            return self._lock.acquire(blocking)
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_TrackingLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


class _GatedEntryLock:
    """Pause one lock holder so a deterministic contender can queue behind it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tracked_thread_id: int | None = None
        self.tracked_entered = threading.Event()
        self.contender_blocked = threading.Event()
        self.resume_tracked = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        is_tracked = threading.get_ident() == self.tracked_thread_id
        if is_tracked:
            acquired = (
                self._lock.acquire(blocking)
                if timeout == -1
                else self._lock.acquire(blocking, timeout)
            )
            if acquired:
                self.tracked_entered.set()
                assert self.resume_tracked.wait(timeout=10)
            return acquired

        if self._lock.acquire(blocking=False):
            return True
        self.contender_blocked.set()
        if not blocking:
            return False
        if timeout == -1:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_GatedEntryLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


class _RacingBounceProbe(_BounceProbe):
    """Create a reservation only after cancellation has already inspected it."""

    def __init__(self) -> None:
        super().__init__()
        self.reserve_started = threading.Event()
        self.finish_reserve = threading.Event()
        self.reserved = False
        self.release_count = 0

    def reserve(self, receiver_req, _num_writers: int, *, extra_bytes: int = 0) -> bool:
        del extra_bytes
        self.reserve_started.set()
        assert self.finish_reserve.wait(timeout=10)
        receiver_req.bounce_dst_base = 0x1000
        self.reserved = True
        return True

    def release_idle_reservation(self, _rid_slice: tuple[int, int]) -> None:
        if self.reserved:
            self.reserved = False
            self.release_count += 1


class _ReceiverProbe:
    """Minimal receiver that publishes one destination to two writers."""

    def __init__(self) -> None:
        self._bounce = _BounceProbe()
        self._enforce_physical_ownership = True
        self._physical_ownership_fault: BaseException | None = None
        self._physical_ownership_fault_lock = threading.Lock()
        self._session: RxSession | None = None
        self._next_owner_generation = 1
        self.clear_count = 0
        self.cancel_count = 0

    def setup_session(self, session: RxSession) -> None:
        self._session = session

    def allocate_owner_generation(self) -> int:
        generation = self._next_owner_generation
        self._next_owner_generation += 1
        return generation

    def dispatch_task(self, task: KVRecvTask) -> None:
        assert self._session is not None
        task.expected_transfers = 2
        assert self._session.try_begin_transfer(task.slice_id, set(), {0, 1})

    def send_cancel_to_senders(self, _unique_rid: int, _sender_endpoints: set[str]) -> None:
        self.cancel_count += 1

    def clear_session(self, _unique_rid: int) -> None:
        self.clear_count += 1

    def _record_physical_ownership_fault(self, error: BaseException) -> None:
        with self._physical_ownership_fault_lock:
            if self._physical_ownership_fault is None:
                self._physical_ownership_fault = error

    @property
    def physical_ownership_fault(self) -> BaseException | None:
        with self._physical_ownership_fault_lock:
            return self._physical_ownership_fault


class _OneSlotAllocator:
    """Model the caller that releases a request allocation on a True result."""

    def __init__(self, owner: int) -> None:
        self.owner: int | None = owner
        self.release_count = 0

    @property
    def is_reusable(self) -> bool:
        return self.owner is None

    def apply_reuse_decision(self, safe_to_reuse: bool) -> None:
        if safe_to_reuse and self.owner is not None:
            self.owner = None
            self.release_count += 1


def _make_rank_info(
    physical_ownership_protocol: int,
    *,
    sender_endpoints: list[str] | None = None,
) -> RankInfo:
    return RankInfo(
        instance_name="peer",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[1],
        sender_endpoints=sender_endpoints or [],
        self_endpoint="tcp://peer",
        transfer_engine_info=b"agent",
        physical_ownership_protocol=physical_ownership_protocol,
    )


def _make_rx_session(receiver: object, rid: int) -> RxSession:
    return RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid),
        receiver=receiver,
    )


def _make_owned_tx_session(
    rid: int,
    task: KVSendTask,
    *,
    lock: object | None = None,
) -> TxSession:
    sender = SimpleNamespace(
        clear_session=Mock(),
        capture_receiver_endpoints=Mock(return_value=set()),
        send_cancel_to_receivers=Mock(),
    )
    session = object.__new__(TxSession)
    session._base_args = SimpleNamespace(
        params=DisaggregatedParams(disagg_request_id=rid),
    )
    session._timeout_s = 0.01
    session._overall_timeout_s = 0.01
    session._deadline_monotonic_s = None
    session._need_aux = False
    session._enforce_physical_ownership = True
    session._terminal_status = None
    session._exception = None
    session.receiver_ready = True
    session.kv_tasks = [task]
    session.aux_task = None
    session.lock = lock if lock is not None else threading.Lock()
    session._closed = False
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = sender
    session.request_id = rid
    session.transfer_start_time = None
    session.transfer_end_time = None
    return session


def _make_owned_transceiver(
    rid: int,
    session: TxSession,
    request: object,
) -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {rid: session}
    transceiver._send_reqs = {rid: request}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    return transceiver


@pytest.mark.cpu_only
def test_failed_writer_cannot_authorize_reuse_while_sibling_is_active() -> None:
    sibling_started = threading.Event()
    release_sibling = threading.Event()
    sibling_errors: list[Exception] = []
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid=41)
    close = Mock(wraps=session.close)
    session.close = close
    session.receive(KVSlice(is_last_slice=True))
    request = SimpleNamespace(
        request_id=41,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=41),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {41: session}
    transceiver._recv_reqs = {41: request}
    allocator = _OneSlotAllocator(owner=41)

    def finish_sibling_writer() -> None:
        try:
            sibling_started.set()
            release_sibling.wait()
            session.process_kv_agent_result(
                peer_rank=1,
                sender_slice_id=0,
                is_last_slice=True,
                status=AgentResult.SUCCESS,
            )
        except Exception as error:
            sibling_errors.append(error)

    sibling_thread = threading.Thread(target=finish_sibling_writer, daemon=True)
    sibling_thread.start()

    try:
        assert sibling_started.wait(timeout=10)

        # Writer 0 is terminal, but writer 1 has not reported a terminal result
        # and may still write to the same receive-side KV allocation.
        session.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.FAILED,
        )

        safe_to_reuse = transceiver.cancel_request(request)
        allocator.apply_reuse_decision(safe_to_reuse)

        assert safe_to_reuse is False, (
            "cancel_request() authorized receive-side KV reuse before every "
            "published writer reached a terminal physical state"
        )
        assert not allocator.is_reusable
        assert 41 in transceiver._recv_sessions
        assert receiver.clear_count == 0
        close.assert_not_called()
    finally:
        release_sibling.set()
        sibling_thread.join(timeout=10)

    assert not sibling_thread.is_alive()
    assert sibling_errors == []

    # The allocation becomes reusable only after the remaining writer reports
    # a terminal physical result. Repeated cancellation must not retire the
    # transceiver session more than once.
    safe_to_reuse = transceiver.cancel_request(request)
    allocator.apply_reuse_decision(safe_to_reuse)
    assert safe_to_reuse is True
    assert allocator.is_reusable
    assert allocator.release_count == 1
    assert 41 not in transceiver._recv_sessions
    assert receiver.clear_count == 1
    close.assert_called_once_with()
    repeated_decision = transceiver.cancel_request(request)
    assert repeated_decision is True
    assert allocator.release_count == 1
    assert receiver.clear_count == 1
    close.assert_called_once_with()


@pytest.mark.cpu_only
def test_pre_cancelled_rx_session_never_publishes_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 73
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = {rid}
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = threading.Lock()
    receiver._shutdown = True
    receiver.dispatch_task = Mock()
    receiver.send_cancel_to_senders = Mock()

    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )

    session = _make_rx_session(receiver, rid)
    assert session.status == SessionStatus.CANCELLED

    session.receive(KVSlice(is_last_slice=True))

    assert (len(session._kv_tasks), receiver.dispatch_task.call_count) == (0, 0), (
        "a pre-cancelled receive session created and published a destination task"
    )


@pytest.mark.cpu_only
def test_cancel_after_publication_cannot_overtake_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 79
    request_data_started = threading.Event()
    finish_request_data = threading.Event()
    race_outcomes: queue.Queue[str] = queue.Queue()
    protocol_order: list[str] = []
    receive_errors: list[Exception] = []
    cancel_errors: list[Exception] = []
    initial_cancel_outcome: str | None = None

    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = threading.Lock()
    receiver._next_owner_generation = 1
    receiver._shutdown = True
    receiver._dealers = {}
    receiver._build_recv_req_info = Mock(
        return_value=SimpleNamespace(
            unique_rid=rid,
            slice_id=0,
            mamba_state_index=None,
            bounce_dst_base=None,
            to_bytes=Mock(return_value=b"receiver-request"),
        )
    )
    overlap = SimpleNamespace(ranks=[0])
    receiver._registrar = SimpleNamespace(
        get_peer_overlap=Mock(return_value=overlap),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={0: "tcp://sender-0"},
            page_table=None,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            attention=None,
            physical_ownership_protocol=1,
        )
    )

    def request_sender_data(_endpoint: str, _receiver_info_bytes: bytes) -> None:
        protocol_order.append("request_data_started")
        request_data_started.set()
        assert finish_request_data.wait(timeout=10)
        protocol_order.append("request_data_sent")

    def send_cancel_to_senders(_unique_rid: int, _sender_endpoints: set[str]) -> None:
        protocol_order.append("cancel_sent")
        race_outcomes.put("cancel_sent")

    receiver._request_sender_data = request_sender_data
    receiver.send_cancel_to_senders = send_cancel_to_senders

    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )

    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid, ctx_dp_rank=0),
        receiver=receiver,
    )
    session_lock = _TrackingLock(race_outcomes)
    session.lock = session_lock

    def receive() -> None:
        try:
            session.receive(KVSlice(is_last_slice=True))
        except Exception as error:
            receive_errors.append(error)

    def cancel() -> None:
        session_lock.tracked_thread_id = threading.get_ident()
        try:
            session.cancel()
        except Exception as error:
            cancel_errors.append(error)

    receive_thread = threading.Thread(target=receive, daemon=True)
    cancel_thread = threading.Thread(target=cancel, daemon=True)
    receive_thread.start()
    try:
        assert request_data_started.wait(timeout=10)
        cancel_thread.start()
        initial_cancel_outcome = race_outcomes.get(timeout=10)
    finally:
        finish_request_data.set()
        receive_thread.join(timeout=10)
        cancel_thread.join(timeout=10)

    assert not receive_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert receive_errors == []
    assert cancel_errors == []
    assert initial_cancel_outcome == "blocked"
    assert protocol_order == ["request_data_started", "request_data_sent", "cancel_sent"], (
        "cancellation overtook an already-authorized REQUEST_DATA publication"
    )


@pytest.mark.cpu_only
def test_stale_terminal_generation_cannot_settle_current_receive_owner() -> None:
    receiver = _ReceiverProbe()
    receiver._next_owner_generation = 2
    session = _make_rx_session(receiver, rid=81)
    session.receive(KVSlice(is_last_slice=True))
    task = session._kv_tasks[0]
    assert task.owner_generation == 2

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
        owner_generation=1,
    )

    assert task.status == TaskStatus.TRANSFERRING
    assert not task.resources_drained
    assert receiver.physical_ownership_fault is None


@pytest.mark.cpu_only
def test_owner_generation_wire_extension_is_flag_gated() -> None:
    legacy = transfer_mod._make_kv_result_msg(0, 81, 0, True, AgentResult.FAILED)
    owned = transfer_mod._make_kv_result_msg(
        0,
        81,
        0,
        True,
        AgentResult.FAILED,
        owner_generation=7,
    )

    assert len(legacy[1]) == transfer_mod._KV_RESULT_PREFIX.size
    assert len(owned[1]) == transfer_mod._KV_RESULT_PREFIX_V1.size
    assert transfer_mod._KV_RESULT_PREFIX_V1.unpack(owned[1])[3] == 7


@pytest.mark.cpu_only
def test_sender_dispatch_rejects_missing_owner_generation_before_admission() -> None:
    rid = 811
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task._unique_rid = rid
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._begin_task_operations = Mock()
    sender._build_kv_write_meta = Mock()
    sender._enqueue = Mock()
    info = SimpleNamespace(
        unique_rid=rid,
        instance_rank=0,
        owner_generation=None,
    )

    with pytest.raises(ValueError, match="positive integer owner_generation"):
        sender.dispatch_task(task, {0: info})

    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    assert sender.physical_ownership_fault is not None
    sender._begin_task_operations.assert_not_called()
    sender._build_kv_write_meta.assert_not_called()
    sender._enqueue.assert_not_called()


@pytest.mark.cpu_only
def test_request_data_rejects_invalid_owner_generation_before_saving_or_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 812
    session = SimpleNamespace(set_exception=Mock())
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._save_peer_req_info = Mock()
    sender._begin_task_operation = Mock()
    info = SimpleNamespace(
        unique_rid=rid,
        instance_rank=0,
        owner_generation=0,
    )
    monkeypatch.setattr(
        transfer_mod.RecvReqInfo,
        "from_bytes",
        Mock(return_value=info),
    )

    with pytest.raises(ValueError, match="positive integer owner_generation"):
        sender._respond_with_kv(
            b"peer",
            [MessageType.REQUEST_DATA, b"malformed-request"],
        )

    assert sender.physical_ownership_fault is not None
    session.set_exception.assert_called_once()
    sender._save_peer_req_info.assert_not_called()
    sender._begin_task_operation.assert_not_called()


@pytest.mark.cpu_only
def test_worker_exception_after_local_drain_still_poisoned_and_cancels_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 813
    error = RuntimeError("terminal result was not published")
    task = SimpleNamespace(
        fail=Mock(),
        resources_drained=True,
        has_in_doubt_operations=False,
        mark_physical_operation_in_doubt=Mock(),
    )
    write_meta = SimpleNamespace(
        task=task,
        unique_rid=rid,
        peer_rank=0,
        meta_type=transfer_mod.WriteMetaType.KV,
    )
    task_queue = SimpleNamespace(get=Mock(side_effect=[write_meta, None]))
    sender = object.__new__(Sender)
    sender._device_id = 0
    sender._send_task_queues = [task_queue]
    sender._thread_local = threading.local()
    sender._enforce_physical_ownership = True
    sender._deliver_kv_to_agent = Mock(side_effect=error)
    sender._record_physical_ownership_fault = Mock()
    sender.send_cancel_to_receivers = Mock()
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())

    sender._process_task_queue(0)

    task.fail.assert_called_once_with(error)
    sender._record_physical_ownership_fault.assert_called_once_with(error)
    sender.send_cancel_to_receivers.assert_called_once_with(rid)
    task.mark_physical_operation_in_doubt.assert_not_called()


@pytest.mark.cpu_only
def test_worker_exception_after_terminal_publication_does_not_poison_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 814
    error = RuntimeError("post-publication bookkeeping failed")
    task = SimpleNamespace(fail=Mock())
    write_meta = SimpleNamespace(
        task=task,
        unique_rid=rid,
        peer_rank=0,
        meta_type=transfer_mod.WriteMetaType.KV,
        terminal_result_published=True,
    )
    task_queue = SimpleNamespace(get=Mock(side_effect=[write_meta, None]))
    sender = object.__new__(Sender)
    sender._device_id = 0
    sender._send_task_queues = [task_queue]
    sender._thread_local = threading.local()
    sender._enforce_physical_ownership = True
    sender._deliver_kv_to_agent = Mock(side_effect=error)
    sender._record_physical_ownership_fault = Mock()
    sender.send_cancel_to_receivers = Mock()
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())

    sender._process_task_queue(0)

    task.fail.assert_called_once_with(error)
    sender._record_physical_ownership_fault.assert_not_called()
    sender.send_cancel_to_receivers.assert_not_called()


@pytest.mark.cpu_only
def test_failed_terminal_result_publication_poisoned_and_cancels_receiver() -> None:
    rid = 815
    error = RuntimeError("terminal send failed")
    dealer = SimpleNamespace(send=Mock(side_effect=error))
    write_meta = SimpleNamespace(
        meta_type=transfer_mod.WriteMetaType.KV,
        unique_rid=rid,
        slice_id=0,
        is_last_slice=True,
        owner_generation=1,
        peer_endpoint="tcp://receiver",
        peer_rank=0,
        terminal_result_published=False,
    )
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._enforce_physical_ownership = True
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    sender._record_physical_ownership_fault = Mock()
    sender.send_cancel_to_receivers = Mock()

    sender._send_failed_write_meta_result(write_meta)

    sender._record_physical_ownership_fault.assert_called_once_with(error)
    sender.send_cancel_to_receivers.assert_called_once_with(
        rid,
        {"tcp://receiver"},
    )
    assert not write_meta.terminal_result_published


@pytest.mark.cpu_only
def test_failed_terminal_result_publication_preserves_flag_off_behavior() -> None:
    error = RuntimeError("terminal send failed")
    dealer = SimpleNamespace(send=Mock(side_effect=error))
    write_meta = SimpleNamespace(
        meta_type=transfer_mod.WriteMetaType.KV,
        unique_rid=816,
        slice_id=0,
        is_last_slice=True,
        owner_generation=None,
        peer_endpoint="tcp://receiver",
        peer_rank=0,
        terminal_result_published=False,
    )
    sender = object.__new__(Sender)
    sender._instance_rank = 0
    sender._enforce_physical_ownership = False
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    sender._record_physical_ownership_fault = Mock()
    sender.send_cancel_to_receivers = Mock()

    sender._send_failed_write_meta_result(write_meta)

    sender._record_physical_ownership_fault.assert_called_once_with(error)
    sender.send_cancel_to_receivers.assert_not_called()
    assert not write_meta.terminal_result_published


@pytest.mark.cpu_only
def test_worker_wide_fault_blocks_opposite_direction_admission() -> None:
    state = transfer_mod._PhysicalOwnershipFaultState()
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault_state = state
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = state.lock
    receiver._next_owner_generation = 1

    rx_session = _make_rx_session(receiver, rid=82)
    rx_task = rx_session.prepare_receive(KVSlice(is_last_slice=True))
    assert rx_task is not None
    rx_task.expected_transfers = 1

    opposite_direction_fault = RuntimeError("sender completion became ambiguous")
    state.record(opposite_direction_fault)
    with pytest.raises(RuntimeError, match="poisoned"):
        rx_session.try_begin_transfer(0, {"tcp://ctx"}, {0})
    assert not rx_task.resources_drained

    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault_state = state
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = state.lock
    sender._sessions_lock = threading.Lock()
    tx_session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
    )
    sender._sessions = {82: tx_session}
    tx_task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=82),
        slice_id=0,
    )
    tx_task._unique_rid = 82

    admission = sender._begin_task_operations(tx_task, {0})
    assert admission.newly_started == frozenset()
    assert admission.rejected_unsubmitted == frozenset({0})
    assert tx_task.resources_drained


@pytest.mark.cpu_only
def test_cancel_wins_atomic_publication_gate_and_releases_late_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 91
    bounce = _RacingBounceProbe()
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = bounce
    receiver._enforce_physical_ownership = True
    receiver._shutdown = True
    receiver._dealers = {}
    receiver._request_sender_data = Mock()
    receiver.send_cancel_to_senders = Mock()
    receiver._build_recv_req_info = Mock(
        return_value=SimpleNamespace(
            unique_rid=rid,
            slice_id=0,
            mamba_state_index=None,
            bounce_dst_base=None,
            to_bytes=Mock(return_value=b"receiver-request"),
        )
    )
    overlap = SimpleNamespace(ranks=[0])
    receiver._registrar = SimpleNamespace(
        get_peer_overlap=Mock(return_value=overlap),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={0: "tcp://sender-0"},
            page_table=None,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            attention=None,
            physical_ownership_protocol=1,
        )
    )

    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )

    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid, ctx_dp_rank=0),
        receiver=receiver,
    )
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {rid: session}
    transceiver._recv_reqs = {rid: request}
    receive_errors: list[Exception] = []

    def receive() -> None:
        try:
            session.receive(KVSlice(is_last_slice=True))
        except Exception as error:
            receive_errors.append(error)

    receive_thread = threading.Thread(target=receive, daemon=True)
    receive_thread.start()
    try:
        assert bounce.reserve_started.wait(timeout=10)

        assert transceiver.cancel_request(request) is False
        assert rid in transceiver._recv_sessions
    finally:
        bounce.finish_reserve.set()
        receive_thread.join(timeout=10)

    assert not receive_thread.is_alive()
    assert receive_errors == []
    assert receiver._request_sender_data.call_count == 0
    assert bounce.release_count == 1
    assert not bounce.reserved
    assert transceiver.cancel_request(request) is True
    assert rid not in transceiver._recv_sessions


@pytest.mark.cpu_only
def test_failed_session_cleanup_waits_for_distinct_writer_drain() -> None:
    rid = 102
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid)
    session.receive(KVSlice(is_last_slice=True))
    request = SimpleNamespace(state=None)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    sessions = {rid: session}
    requests = {rid: request}

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )
    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )

    assert transceiver._close_failed_sessions(sessions, requests, [rid]) == []
    assert rid in sessions
    assert receiver.clear_count == 0

    session.process_kv_agent_result(
        peer_rank=1,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
    )

    assert transceiver._close_failed_sessions(sessions, requests, [rid]) == [rid]
    assert sessions == {}
    assert requests == {}
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert receiver.clear_count == 1


@pytest.mark.cpu_only
def test_contradictory_writer_result_poison_retains_destination() -> None:
    rid = 103
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid)
    session.receive(KVSlice(is_last_slice=True))

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )
    with pytest.raises(RuntimeError, match="contradictory terminal evidence"):
        session.process_kv_agent_result(
            peer_rank=0,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
        )

    session.process_kv_agent_result(
        peer_rank=1,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
    )

    assert receiver.physical_ownership_fault is not None
    assert not session.resources_drained()
    assert not session.close()
    assert receiver.clear_count == 0


@pytest.mark.cpu_only
def test_out_of_cohort_writer_result_poison_retains_destination() -> None:
    rid = 104
    receiver = _ReceiverProbe()
    session = _make_rx_session(receiver, rid)
    session.receive(KVSlice(is_last_slice=True))

    with pytest.raises(RuntimeError, match="outside the sealed writer cohort"):
        session.process_kv_agent_result(
            peer_rank=99,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
        )

    assert receiver.physical_ownership_fault is not None
    assert not session.resources_drained()
    assert not session.close()
    assert receiver.clear_count == 0


@pytest.mark.cpu_only
def test_send_failure_does_not_hide_sibling_source_operation() -> None:
    params = DisaggregatedParams(disagg_request_id=113)
    task = KVSendTask(KVSlice(is_last_slice=True), params, slice_id=0)
    task.begin_physical_operation(0)
    task.begin_physical_operation(1)
    task.status = TaskStatus.TRANSFERRING
    task.fail(RuntimeError("writer 0 failed"))
    task.finish_physical_operation(0)

    session = object.__new__(TxSession)
    session.kv_tasks = [task]
    session.aux_task = None
    session._terminal_status = None
    session._enforce_physical_ownership = True
    session.receiver_ready = True

    assert session.has_failed()
    assert not session.resources_drained()
    assert session.has_transferring_tasks()

    task.finish_physical_operation(1)

    assert session.has_failed()
    assert session.resources_drained()
    assert not session.has_transferring_tasks()


@pytest.mark.cpu_only
def test_source_evidence_is_retained_and_not_polled_during_wait() -> None:
    class EvidenceStatus:
        def __init__(self) -> None:
            self.completed = False
            self.poll_count = 0

        def is_completed(self) -> bool:
            self.poll_count += 1
            return self.completed

    class EvidenceRequest:
        pass

    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=119),
        slice_id=0,
    )
    task.begin_physical_operation(0)
    status = EvidenceStatus()
    request = EvidenceRequest()
    status_ref = weakref.ref(status)
    request_ref = weakref.ref(request)
    task.attach_physical_operation_evidence(0, status, request)  # type: ignore[arg-type]

    assert not task.resources_drained
    assert status.poll_count == 0
    del status, request
    gc.collect()
    assert status_ref() is not None
    assert request_ref() is not None

    task.mark_physical_operation_in_doubt(0)
    assert not task.resources_drained
    assert status_ref().poll_count == 1  # type: ignore[union-attr]
    status_ref().completed = True  # type: ignore[union-attr]
    assert task.resources_drained

    gc.collect()
    assert status_ref() is None
    assert request_ref() is None


@pytest.mark.cpu_only
def test_prepublication_exception_closes_gate_without_publication() -> None:
    rid = 127
    receiver = _ReceiverProbe()
    receiver.dispatch_task = Mock(side_effect=RuntimeError("peer discovery failed"))
    receiver._bounce.release_idle_reservation = Mock()
    session = _make_rx_session(receiver, rid)

    with pytest.raises(RuntimeError, match="peer discovery failed"):
        session.receive(KVSlice(is_last_slice=True))

    assert session.has_failed()
    assert session.resources_drained()
    assert len(session._kv_tasks) == 1
    receiver._bounce.release_idle_reservation.assert_called_once_with((rid, 0))


@pytest.mark.cpu_only
def test_incompatible_peer_closes_unpublished_ownership_gate() -> None:
    rid = 128
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = threading.Lock()
    receiver._shutdown = True
    receiver._build_recv_req_info = Mock(return_value=SimpleNamespace())
    receiver._get_sender_info = Mock(
        side_effect=PeerIncompatibleError("synthetic peer-layout mismatch")
    )
    session = _make_rx_session(receiver, rid)

    session.receive(KVSlice(is_last_slice=True))

    assert session.wait_complete(blocking=False) == WaitResult.FAILED
    assert session.wait_complete(blocking=True) == WaitResult.FAILED
    assert session.resources_drained()
    assert session.close()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("peer_protocol", "peer_overrides", "ctx_dp_rank", "error_match"),
    [
        (0, {}, 0, "protocol mismatch"),
        (1, {"tp_size": 2}, 0, "tp_size=2"),
        (1, {"pp_size": 2}, 0, "pp_size=2"),
        (1, {"cp_size": 2}, 0, "cp_size=2"),
        (1, {"dp_size": 2}, 0, "dp_size=2"),
        (
            1,
            {"attention": SimpleNamespace(enable_attention_dp=True)},
            0,
            "attention_dp",
        ),
        (1, {}, None, "unknown_ctx_dp_rank"),
    ],
    ids=[
        "protocol",
        "tp",
        "pp",
        "cp",
        "dp",
        "attention-dp",
        "unknown-context-dp-rank",
    ],
)
def test_unsupported_remote_contract_is_rejected_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    peer_protocol: int,
    peer_overrides: dict[str, object],
    ctx_dp_rank: int | None,
    error_match: str,
) -> None:
    rid = 129
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = threading.Lock()
    receiver._shutdown = True
    receiver._request_sender_data = Mock()
    receiver.send_cancel_to_senders = Mock()
    receiver._build_recv_req_info = Mock(
        return_value=SimpleNamespace(
            unique_rid=rid,
            slice_id=0,
            mamba_state_index=None,
        )
    )
    receiver._registrar = SimpleNamespace(get_peer_overlap=Mock())
    peer_contract = {
        "sender_endpoints": {0: "tcp://sender-0", 1: "tcp://sender-1"},
        "page_table": None,
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "dp_size": 1,
        "attention": None,
        "physical_ownership_protocol": peer_protocol,
    }
    peer_contract.update(peer_overrides)
    receiver._get_sender_info = Mock(return_value=SimpleNamespace(**peer_contract))
    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )
    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid, ctx_dp_rank=ctx_dp_rank),
        receiver=receiver,
    )

    with pytest.raises(ValueError, match=error_match):
        session.receive(KVSlice(is_last_slice=True))

    assert session.has_failed()
    assert session.resources_drained()
    receiver._registrar.get_peer_overlap.assert_not_called()
    receiver._request_sender_data.assert_not_called()
    assert receiver.physical_ownership_fault is None


@pytest.mark.cpu_only
def test_writer_cohort_seal_requires_exact_cardinality() -> None:
    task = KVRecvTask(
        unique_rid=130,
        kv_slice=KVSlice(is_last_slice=True),
        slice_id=0,
        params=DisaggregatedParams(disagg_request_id=130),
        aux_slot=None,
    )
    task.expected_transfers = 2

    with pytest.raises(ValueError, match="expected exactly 2"):
        task.seal_writer_cohort({0, 1, 2})


@pytest.mark.cpu_only
def test_receiver_protocol_mismatch_stops_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_info = _make_rank_info(0, sender_endpoints=["tcp://sender"])
    info_messenger = SimpleNamespace(
        send=Mock(),
        receive=Mock(return_value=[peer_info.to_bytes()]),
        stop=Mock(),
    )
    monkeypatch.setattr(
        transfer_mod,
        "ZMQMessenger",
        Mock(return_value=info_messenger),
    )
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sender_ep_instance_map = {}
    receiver._incompatible_peers = {}
    receiver._registrar = SimpleNamespace(
        self_rank_info=_make_rank_info(1),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_or_connect_dealer = Mock()
    params = SimpleNamespace(ctx_info_endpoint="tcp://info")

    with pytest.raises(ValueError, match="protocol mismatch"):
        receiver._get_sender_info(params)

    info_messenger.send.assert_called_once_with([MessageType.REQUEST_INSTANCE_INFO])
    info_messenger.stop.assert_called_once()
    receiver._get_or_connect_dealer.assert_not_called()
    assert receiver._sender_ep_instance_map == {}


@pytest.mark.cpu_only
def test_receiver_matching_protocol_registers_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_info = _make_rank_info(1, sender_endpoints=["tcp://sender"])
    info_messenger = SimpleNamespace(
        send=Mock(),
        receive=Mock(return_value=[peer_info.to_bytes()]),
        stop=Mock(),
    )
    registration_dealer = SimpleNamespace(send=Mock())
    monkeypatch.setattr(
        transfer_mod,
        "ZMQMessenger",
        Mock(return_value=info_messenger),
    )
    monkeypatch.setattr(
        transfer_mod.MambaPolicy,
        "validate_peer_compatible",
        Mock(),
    )
    receiver = object.__new__(Receiver)
    receiver._enforce_physical_ownership = True
    receiver._sender_ep_instance_map = {}
    receiver._incompatible_peers = {}
    receiver._registrar = SimpleNamespace(
        self_rank_info=_make_rank_info(1),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_or_connect_dealer = Mock(return_value=registration_dealer)
    params = SimpleNamespace(ctx_info_endpoint="tcp://info")

    resolved_peer = receiver._get_sender_info(params)

    registration_dealer.send.assert_called_once()
    message = registration_dealer.send.call_args.args[0]
    assert message[0] == MessageType.REGISTER_RANK_INFO
    assert resolved_peer.physical_ownership_protocol == 1
    assert receiver._sender_ep_instance_map["tcp://info"] is resolved_peer


@pytest.mark.cpu_only
def test_sender_protocol_mismatch_stops_before_agent_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = object.__new__(Sender)
    sender._shutdown = False
    sender._device_id = 0
    sender._enforce_physical_ownership = True
    sender._registrar = SimpleNamespace(register=Mock())
    sender._agent = SimpleNamespace(load_remote_agent=Mock())
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._loaded_remote_agents = set()
    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())
    peer_info = _make_rank_info(0)

    with pytest.raises(ValueError, match="protocol mismatch"):
        sender._register_peer_rank(
            b"peer",
            [MessageType.REGISTER_RANK_INFO, peer_info.to_bytes()],
        )

    sender._registrar.register.assert_not_called()
    sender._agent.load_remote_agent.assert_not_called()


@pytest.mark.cpu_only
def test_sender_build_failure_settles_every_unsubmitted_writer() -> None:
    rid = 131
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._build_kv_write_meta = Mock(side_effect=RuntimeError("metadata failed"))
    sender._enqueue = Mock()
    sender._send_failed_task_result_to_receiver = Mock()
    recv_infos = {
        0: SimpleNamespace(instance_rank=10, owner_generation=1),
        1: SimpleNamespace(instance_rank=11, owner_generation=1),
    }

    with pytest.raises(RuntimeError, match="metadata failed"):
        sender.dispatch_task(task, recv_infos)

    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    assert sender._enqueue.call_count == 0
    assert {
        call.args[1].instance_rank
        for call in sender._send_failed_task_result_to_receiver.call_args_list
    } == {10, 11}


@pytest.mark.cpu_only
def test_sender_cancel_wins_before_operation_admission() -> None:
    rid = _GLOBAL_RID_BASE + 132
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    gate = _GatedEntryLock()
    session = _make_owned_tx_session(rid, task, lock=gate)
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    transceiver = _make_owned_transceiver(rid, session, request)
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._build_kv_write_meta = Mock()
    sender._enqueue = Mock()
    sender._send_failed_task_result_to_receiver = Mock()
    sender.capture_receiver_endpoints = Mock(return_value={"tcp://gen-0"})
    sender.send_cancel_to_receivers = Mock()
    sender.clear_session = Mock(
        side_effect=lambda request_id: sender._sessions.pop(request_id, None)
    )
    session._sender = sender
    info = SimpleNamespace(instance_rank=0, owner_generation=1)
    cancel_results: list[bool] = []
    errors: list[BaseException] = []

    def cancel() -> None:
        gate.tracked_thread_id = threading.get_ident()
        try:
            cancel_results.append(transceiver.cancel_request(request))
        except BaseException as error:
            errors.append(error)

    def dispatch() -> None:
        try:
            sender.dispatch_task(task, {0: info})
        except BaseException as error:
            errors.append(error)

    cancel_thread = threading.Thread(target=cancel, daemon=True)
    dispatch_thread = threading.Thread(target=dispatch, daemon=True)
    cancel_thread.start()
    assert gate.tracked_entered.wait(timeout=10)
    dispatch_thread.start()
    try:
        assert gate.contender_blocked.wait(timeout=10)
    finally:
        gate.resume_tracked.set()
        cancel_thread.join(timeout=10)
        dispatch_thread.join(timeout=10)

    assert not cancel_thread.is_alive()
    assert not dispatch_thread.is_alive()
    assert errors == []
    assert cancel_results == [True]
    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    sender._build_kv_write_meta.assert_not_called()
    sender._enqueue.assert_not_called()
    sender._send_failed_task_result_to_receiver.assert_called_once_with(
        task,
        info,
        transient=True,
    )
    assert rid not in transceiver._send_sessions
    assert rid not in transceiver._send_reqs
    sender.clear_session.assert_called_once_with(rid)
    assert rid not in sender._sessions


@pytest.mark.cpu_only
def test_backend_submission_wins_cancel_retains_source_until_evidence_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = _GLOBAL_RID_BASE + 133
    peer_rank = 3
    submitted = threading.Event()
    settle = threading.Event()
    delivery_errors: list[BaseException] = []
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    session = _make_owned_tx_session(rid, task)
    close = Mock(wraps=session.close)
    session.close = close
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    transceiver = _make_owned_transceiver(rid, session, request)

    wait_status = SimpleNamespace(is_completed=Mock(return_value=True))

    def wait() -> bool:
        submitted.set()
        assert settle.wait(timeout=10)
        return True

    wait_status.wait = Mock(side_effect=wait)
    dealer = SimpleNamespace(send=Mock())
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._device_id = 0
    sender._instance_rank = 0
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=0)
    )
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(return_value=wait_status))
    sender._bounce = SimpleNamespace(release_send=Mock())
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender.capture_receiver_endpoints = Mock(return_value={"tcp://gen-3"})
    sender.send_cancel_to_receivers = Mock()
    sender.clear_session = Mock(
        side_effect=lambda request_id: sender._sessions.pop(request_id, None)
    )
    sender._send_failed_write_meta_result = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    session._sender = sender
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))
    one = transfer_mod.np.array([1], dtype=transfer_mod.np.int64)
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen3",
        peer_rank=peer_rank,
        peer_endpoint="tcp://gen-3",
        unique_rid=rid,
        src_ptrs=one,
        dst_ptrs=one,
        sizes=one,
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
        owner_generation=1,
    )
    sender._build_kv_write_meta = Mock(return_value=write_meta)
    enqueued: list[WriteMeta] = []

    def enqueue(meta: WriteMeta) -> None:
        assert task.status == TaskStatus.TRANSFERRING
        assert not task.resources_drained
        enqueued.append(meta)

    sender._enqueue = Mock(side_effect=enqueue)
    info = SimpleNamespace(instance_rank=peer_rank, owner_generation=1)

    sender.dispatch_task(task, {peer_rank: info})

    sender._build_kv_write_meta.assert_called_once_with(task, info)
    sender._enqueue.assert_called_once_with(write_meta)
    assert enqueued == [write_meta]
    assert task.status == TaskStatus.TRANSFERRING
    assert not task.resources_drained

    def deliver() -> None:
        try:
            sender._deliver_kv_to_agent(write_meta)
        except BaseException as error:
            delivery_errors.append(error)

    delivery_thread = threading.Thread(target=deliver, daemon=True)
    delivery_thread.start()
    try:
        assert submitted.wait(timeout=10)

        assert transceiver.cancel_request(request) is False
        assert rid in transceiver._send_sessions
        assert rid in transceiver._send_reqs
        assert not task.resources_drained
        close.assert_not_called()
    finally:
        settle.set()
        delivery_thread.join(timeout=10)

    assert not delivery_thread.is_alive()
    assert delivery_errors == []
    assert task.resources_drained
    assert task.status == TaskStatus.TRANSFERRED

    assert transceiver.cancel_request(request) is True
    assert rid not in transceiver._send_sessions
    assert rid not in transceiver._send_reqs
    close.assert_called_once_with()
    sender.clear_session.assert_called_once_with(rid)
    assert rid not in sender._sessions


@pytest.mark.cpu_only
def test_terminal_session_replay_does_not_report_owned_writer_quiesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 133
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task.begin_physical_operation(0)
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        kv_tasks=[task],
        status=SessionStatus.CANCELLED,
        _closed=False,
        _has_logical_failure=Mock(return_value=True),
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._save_peer_req_info = Mock()
    sender._send_failed_task_result_to_receiver = Mock()
    sender._enqueue = Mock()
    info = SimpleNamespace(unique_rid=rid, instance_rank=0, owner_generation=1)
    monkeypatch.setattr(
        transfer_mod.RecvReqInfo,
        "from_bytes",
        Mock(return_value=info),
    )

    sender._respond_with_kv(b"peer", [MessageType.REQUEST_DATA, b"request"])

    sender._save_peer_req_info.assert_called_once_with(info)
    sender._send_failed_task_result_to_receiver.assert_not_called()
    sender._enqueue.assert_not_called()
    assert not task.resources_drained


@pytest.mark.cpu_only
def test_terminal_session_reports_failed_only_for_never_submitted_writer() -> None:
    rid = 134
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task.begin_physical_operation(0)
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=True),
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._enqueue = Mock()
    sender._send_failed_task_result_to_receiver = Mock()
    infos = {
        0: SimpleNamespace(instance_rank=0, owner_generation=1),
        1: SimpleNamespace(instance_rank=1, owner_generation=1),
    }

    sender.dispatch_task(task, infos)

    sender._enqueue.assert_not_called()
    sender._send_failed_task_result_to_receiver.assert_called_once_with(
        task,
        infos[1],
        transient=True,
    )
    assert not task.resources_drained


@pytest.mark.cpu_only
def test_sticky_sender_fault_rejects_new_writer_before_submission() -> None:
    rid = 135
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    session = SimpleNamespace(
        lock=threading.Lock(),
        _closed=False,
        _has_logical_failure=Mock(return_value=False),
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = RuntimeError("earlier ambiguous operation")
    sender._physical_ownership_fault_lock = threading.Lock()
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._enqueue = Mock()
    sender._send_failed_task_result_to_receiver = Mock()
    info = SimpleNamespace(instance_rank=0, owner_generation=1)

    sender.dispatch_task(task, {0: info})

    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    sender._enqueue.assert_not_called()
    sender._send_failed_task_result_to_receiver.assert_called_once_with(
        task,
        info,
        transient=True,
    )


@pytest.mark.cpu_only
def test_partial_publication_retains_ambiguous_writer_and_settles_later_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 137
    receiver = object.__new__(Receiver)
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = set()
    receiver._bounce = _BounceProbe()
    receiver._enforce_physical_ownership = True
    receiver._physical_ownership_fault = None
    receiver._physical_ownership_fault_lock = threading.Lock()
    receiver._shutdown = True
    receiver._fanin_bounce_safe = Mock(return_value=False)
    receiver._request_sender_data = Mock(side_effect=[None, RuntimeError("ambiguous send failure")])
    receiver.send_cancel_to_senders = Mock()
    receiver._build_recv_req_info = Mock(
        return_value=SimpleNamespace(
            unique_rid=rid,
            slice_id=0,
            mamba_state_index=None,
            bounce_dst_base=None,
            to_bytes=Mock(return_value=b"receiver-request"),
        )
    )
    overlap = SimpleNamespace(ranks=[0, 1, 2])
    receiver._registrar = SimpleNamespace(
        get_peer_overlap=Mock(return_value=overlap),
        self_extractor=SimpleNamespace(page_table=None),
    )
    receiver._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={
                0: "tcp://sender-0",
                1: "tcp://sender-1",
                2: "tcp://sender-2",
            },
            page_table=None,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            attention=None,
            physical_ownership_protocol=1,
        )
    )
    monkeypatch.setattr(
        transfer_mod.tensorrt_llm.bindings,
        "global_steady_clock_now",
        lambda: 0,
    )
    session = RxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid, ctx_dp_rank=0),
        receiver=receiver,
    )

    with pytest.raises(RuntimeError, match="ambiguous send failure"):
        session.receive(KVSlice(is_last_slice=True))

    task = session._kv_tasks[0]
    assert session.has_failed()
    assert not task.resources_drained
    assert receiver._request_sender_data.call_count == 2
    assert receiver._bounce.failed_writers == [((rid, 0), 2)]
    assert receiver.physical_ownership_fault is not None

    session.process_kv_agent_result(
        peer_rank=0,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )
    assert not task.resources_drained

    session.process_kv_agent_result(
        peer_rank=1,
        sender_slice_id=0,
        is_last_slice=True,
        status=AgentResult.FAILED,
    )
    assert task.resources_drained


@pytest.mark.cpu_only
def test_backend_wait_exception_retains_source_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 139
    peer_rank = 7
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task.begin_physical_operation(peer_rank)
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(side_effect=lambda reason: task.fail(RuntimeError(reason))),
    )
    wait_status = SimpleNamespace(
        wait=Mock(side_effect=RuntimeError("wait failed")),
        is_completed=Mock(return_value=False),
    )
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._device_id = 0
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(return_value=wait_status))
    sender._bounce = SimpleNamespace(release_send=Mock())
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender.send_cancel_to_receivers = Mock()
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))
    one = transfer_mod.np.array([1], dtype=transfer_mod.np.int64)
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen7",
        peer_rank=peer_rank,
        peer_endpoint="tcp://gen-7",
        unique_rid=rid,
        src_ptrs=one,
        dst_ptrs=one,
        sizes=one,
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
    )

    with pytest.raises(RuntimeError, match="wait failed"):
        sender._deliver_kv_to_agent(write_meta)

    assert task.status == TaskStatus.ERROR
    assert not task.resources_drained
    assert sender.physical_ownership_fault is not None
    session.set_exception.assert_called_once()
    sender.send_cancel_to_receivers.assert_called_once_with(rid)

    wait_status.is_completed.return_value = True
    assert task.resources_drained


@pytest.mark.cpu_only
def test_backend_success_retires_source_owner_before_task_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 143
    peer_rank = 3
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task.begin_physical_operation(peer_rank)
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
        transfer_end_time=None,
    )
    wait_status = SimpleNamespace(is_completed=Mock(return_value=True))

    def wait() -> bool:
        assert not task.resources_drained
        wait_status.is_completed.assert_not_called()
        return True

    wait_status.wait = Mock(side_effect=wait)

    def publish_result(_message) -> None:
        assert task.resources_drained
        assert task.status == TaskStatus.TRANSFERRING

    dealer = SimpleNamespace(send=Mock(side_effect=publish_result))
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._device_id = 0
    sender._instance_rank = 0
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=0)
    )
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(return_value=wait_status))
    sender._bounce = SimpleNamespace(release_send=Mock())
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender.send_cancel_to_receivers = Mock()
    sender._send_failed_write_meta_result = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))
    one = transfer_mod.np.array([1], dtype=transfer_mod.np.int64)
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen3",
        peer_rank=peer_rank,
        peer_endpoint="tcp://gen-3",
        unique_rid=rid,
        src_ptrs=one,
        dst_ptrs=one,
        sizes=one,
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
    )

    sender._deliver_kv_to_agent(write_meta)

    assert task.status == TaskStatus.TRANSFERRED
    assert task.resources_drained
    assert sender.physical_ownership_fault is None
    wait_status.wait.assert_called_once_with()
    wait_status.is_completed.assert_not_called()
    dealer.send.assert_called_once()
    sender.send_cancel_to_receivers.assert_not_called()
    sender._send_failed_write_meta_result.assert_not_called()


@pytest.mark.cpu_only
def test_phase1_timeout_retains_source_until_physical_operation_drains() -> None:
    rid = _GLOBAL_RID_BASE + 144
    peer_rank = 3
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task.begin_physical_operation(peer_rank)
    task.status = TaskStatus.TRANSFERRING
    session = _make_owned_tx_session(rid, task)
    session._deadline_monotonic_s = 0.0
    close = Mock(wraps=session.close)
    session.close = close
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    transceiver = _make_owned_transceiver(rid, session, request)
    transceiver._ever_had_send_session = True
    transceiver._ctx_consensus = Mock(return_value=[])
    transceiver._ctx_consensus_outcome = Mock(return_value=([], [], []))
    transceiver._transfer_worker = SimpleNamespace(sweep_stale_req_infos=Mock())
    transceiver.kv_transfer_timeout_ms = 10

    completed, failed = transceiver.check_context_transfer_status(None)

    assert completed == []
    assert failed == []
    assert rid in transceiver._send_sessions
    assert rid in transceiver._send_reqs
    assert not task.resources_drained
    close.assert_not_called()

    assert transceiver.cancel_request(request) is False
    assert rid in transceiver._send_sessions
    assert rid in transceiver._send_reqs
    close.assert_not_called()

    task.finish_physical_operation(peer_rank)
    assert transceiver.cancel_request(request) is True
    assert rid not in transceiver._send_sessions
    assert rid not in transceiver._send_reqs
    close.assert_called_once_with()


@pytest.mark.cpu_only
def test_ambiguous_result_publication_poison_never_sends_contradictory_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rid = 145
    peer_rank = 3
    task = KVSendTask(
        KVSlice(is_last_slice=True),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    task.begin_physical_operation(peer_rank)
    task.status = TaskStatus.TRANSFERRING
    session = SimpleNamespace(
        lock=threading.Lock(),
        status=SessionStatus.TRANSFERRING,
        kv_tasks=[task],
        _enforce_physical_ownership=True,
        set_exception=Mock(),
    )
    wait_status = SimpleNamespace(
        wait=Mock(return_value=True),
        is_completed=Mock(return_value=True),
    )
    dealer = SimpleNamespace(send=Mock(side_effect=RuntimeError("raise after delivery")))
    sender = object.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: session}
    sender._device_id = 0
    sender._instance_rank = 0
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(return_value=wait_status))
    sender._bounce = SimpleNamespace(release_send=Mock())
    sender._enforce_physical_ownership = True
    sender._physical_ownership_fault = None
    sender._physical_ownership_fault_lock = threading.Lock()
    sender.send_cancel_to_receivers = Mock()
    sender._send_failed_write_meta_result = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    monkeypatch.setattr(Sender, "_make_agent_request", Mock(return_value=object()))
    one = transfer_mod.np.array([1], dtype=transfer_mod.np.int64)
    write_meta = WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen3",
        peer_rank=peer_rank,
        peer_endpoint="tcp://gen-3",
        unique_rid=rid,
        src_ptrs=one,
        dst_ptrs=one,
        sizes=one,
        dst_device_id=0,
        slice_id=0,
        is_last_slice=True,
    )

    with pytest.raises(RuntimeError, match="result publication is ambiguous"):
        sender._deliver_kv_to_agent(write_meta)

    assert task.status == TaskStatus.ERROR
    assert task.resources_drained
    assert sender.physical_ownership_fault is not None
    dealer.send.assert_called_once()
    sender._send_failed_write_meta_result.assert_not_called()
    sender.send_cancel_to_receivers.assert_called_once_with(rid)


@pytest.mark.cpu_only
def test_phase1_ownership_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(transceiver_mod._PHYSICAL_OWNERSHIP_ENV, raising=False)
    monkeypatch.setenv(transceiver_mod._DISAGG_NO_RETRY_ENV, "1")
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)
    mapping = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        cp_size=1,
        dp_size=1,
        enable_attention_dp=False,
    )
    config = SimpleNamespace(kv_cache_bounce_size_mb=0)
    manager = SimpleNamespace(max_draft_len=0)

    assert not KvCacheTransceiverV2._supports_phase1_physical_ownership(
        mapping,
        manager,
        config,
    )


@pytest.mark.cpu_only
def test_phase1_ownership_opt_in_validates_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(transceiver_mod._PHYSICAL_OWNERSHIP_ENV, "1")
    monkeypatch.setenv(transceiver_mod._DISAGG_NO_RETRY_ENV, "1")
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)
    mapping = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        cp_size=1,
        dp_size=1,
        enable_attention_dp=False,
    )
    config = SimpleNamespace(kv_cache_bounce_size_mb=0)
    manager = SimpleNamespace(max_draft_len=0)

    assert KvCacheTransceiverV2._supports_phase1_physical_ownership(
        mapping,
        manager,
        config,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("unsupported_case", "error_match"),
    [
        ("tp", "tp_size=2"),
        ("pp", "pp_size=2"),
        ("cp", "cp_size=2"),
        ("dp", "dp_size=2"),
        ("attention_dp", "attention_dp"),
        ("dwdp", "dwdp_size=2"),
        ("bounce", "kv_cache_bounce_size_mb=1"),
        ("synchronous", "synchronous_transfer"),
        ("retry", f"{transceiver_mod._DISAGG_NO_RETRY_ENV}!=1"),
        ("hybrid_manager", "MambaHybridCacheManagerForTest"),
        ("separate_mamba_manager", "SeparateMambaManager"),
        ("draft", "draft_tokens"),
        ("connector", "kv_connector_manager"),
        ("linear_attention", "linear_attention"),
        ("layerwise", "layerwise_transfer"),
        ("v1_offload", "host_kv_cache_offload"),
        ("v2_offload", "kv_cache_offload_tiers"),
    ],
)
def test_phase1_rejects_unqualified_cache_and_parallel_modes(
    monkeypatch: pytest.MonkeyPatch,
    unsupported_case: str,
    error_match: str,
) -> None:
    monkeypatch.setenv(transceiver_mod._PHYSICAL_OWNERSHIP_ENV, "1")
    monkeypatch.setenv(transceiver_mod._DISAGG_NO_RETRY_ENV, "1")
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_LAYERWISE", raising=False)
    mapping = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        cp_size=1,
        dp_size=1,
        enable_attention_dp=False,
        dwdp_enabled=False,
        dwdp_size=1,
    )
    manager = SimpleNamespace(
        max_draft_len=0,
        is_linear_attention=False,
        kv_connector_manager=None,
        blocks_per_window={},
        kv_cache_manager_py_config=SimpleNamespace(cache_tiers=[object()]),
    )
    config = SimpleNamespace(kv_cache_bounce_size_mb=0)
    mamba_cache_manager = None
    if unsupported_case in {"tp", "pp", "cp", "dp"}:
        setattr(mapping, f"{unsupported_case}_size", 2)
    elif unsupported_case == "attention_dp":
        mapping.enable_attention_dp = True
    elif unsupported_case == "dwdp":
        mapping.dwdp_enabled = True
        mapping.dwdp_size = 2
    elif unsupported_case == "bounce":
        config.kv_cache_bounce_size_mb = 1
    elif unsupported_case == "synchronous":
        monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    elif unsupported_case == "retry":
        monkeypatch.delenv(transceiver_mod._DISAGG_NO_RETRY_ENV, raising=False)
    elif unsupported_case == "hybrid_manager":

        class MambaHybridCacheManagerForTest:
            pass

        monkeypatch.setattr(
            transceiver_mod,
            "MambaHybridCacheManager",
            MambaHybridCacheManagerForTest,
        )
        manager = MambaHybridCacheManagerForTest()
    elif unsupported_case == "separate_mamba_manager":

        class SeparateMambaManager:
            pass

        mamba_cache_manager = SeparateMambaManager()
    elif unsupported_case == "draft":
        manager.max_draft_len = 1
    elif unsupported_case == "connector":
        manager.kv_connector_manager = object()
    elif unsupported_case == "linear_attention":
        manager.is_linear_attention = True
    elif unsupported_case == "layerwise":
        monkeypatch.setenv("TRTLLM_DISAGG_LAYERWISE", "1")
    elif unsupported_case == "v1_offload":
        manager.blocks_per_window = {128: (10, 1)}
    else:
        manager.kv_cache_manager_py_config.cache_tiers.append(object())

    with pytest.raises(ValueError, match=error_match):
        KvCacheTransceiverV2._supports_phase1_physical_ownership(
            mapping,
            manager,
            config,
            mamba_cache_manager,
        )


@pytest.mark.cpu_only
def test_phase1_rejects_generation_first_before_session_creation() -> None:
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(
            disagg_request_id=157,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        )
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock())

    with pytest.raises(ValueError, match="context-first"):
        transceiver.request_and_receive_async(request)

    transceiver._transfer_worker.create_rx_session.assert_not_called()


@pytest.mark.cpu_only
def test_phase1_rejects_synchronous_receive_before_session_creation() -> None:
    rid = _GLOBAL_RID_BASE + 157
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock())

    with pytest.raises(ValueError, match="synchronous KV transfer"):
        transceiver.request_and_receive_sync(request)

    transceiver._transfer_worker.create_rx_session.assert_not_called()


@pytest.mark.cpu_only
def test_phase1_rejects_local_or_fallback_request_id() -> None:
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=157),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True

    with pytest.raises(ValueError, match="coordinator-issued"):
        transceiver._validate_phase1_request(request)


@pytest.mark.cpu_only
def test_shutdown_closes_receive_admission_before_session_creation() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_started = True
    transceiver._validate_phase1_request = Mock()
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock())

    with pytest.raises(RuntimeError, match="shutting down"):
        transceiver.request_and_receive_async(object())

    transceiver._transfer_worker.create_rx_session.assert_not_called()


@pytest.mark.cpu_only
def test_cancel_before_admission_leaves_zero_publication_tombstone() -> None:
    rid = _GLOBAL_RID_BASE + 158
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_started = False
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(create_rx_session=Mock())
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
        state=LlmRequestState.DISAGG_GENERATION_INIT,
    )

    assert transceiver.cancel_request(request) is True
    assert getattr(request, transceiver_mod._ADMISSION_CANCELLED_ATTR)

    transceiver.request_and_receive_async(request)

    transceiver._transfer_worker.create_rx_session.assert_not_called()
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


@pytest.mark.cpu_only
def test_cancel_snapshots_tx_target_before_drained_session_close() -> None:
    rid = 159
    dealer = SimpleNamespace(send=Mock())
    sender = object.__new__(Sender)
    sender._peer_requests_lock = threading.Lock()
    sender._peer_requests = {
        rid: {
            0: SimpleNamespace(instance_name="gen", instance_rank=0),
        }
    }
    sender._peer_requests_timestamps = {rid: 0.0}
    sender._sessions_lock = threading.Lock()
    sender._sessions = {rid: object()}
    sender._registrar = SimpleNamespace(
        get_peer_rank_info=Mock(return_value=SimpleNamespace(self_endpoint="tcp://gen-0"))
    )
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    session = SimpleNamespace(disagg_request_id=rid)
    session.cancel_local = Mock(return_value=True)
    session.capture_cancel_targets = Mock(
        side_effect=lambda: sender.capture_receiver_endpoints(rid)
    )
    session.has_transferring_tasks = Mock(return_value=False)
    session.close = Mock(side_effect=lambda: sender.clear_session(rid) is None)
    session.notify_cancel = Mock(
        side_effect=lambda targets: sender.send_cancel_to_receivers(rid, targets)
    )
    request = SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {rid: session}
    transceiver._send_reqs = {rid: request}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}

    assert transceiver.cancel_request(request) is True

    assert sender._peer_requests == {}
    session.notify_cancel.assert_called_once_with({"tcp://gen-0"})
    dealer.send.assert_called_once_with([MessageType.CANCEL_SESSION, str(rid).encode("ascii")])


@pytest.mark.cpu_only
def test_admitted_receive_does_not_block_shutdown_drain_check() -> None:
    rid = _GLOBAL_RID_BASE + 159
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    receive_errors: list[Exception] = []
    session = SimpleNamespace(
        disagg_request_id=rid,
        prepare_receive=Mock(return_value=object()),
        cancel_local=Mock(return_value=True),
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=True),
    )

    def dispatch_prepared_receive(_task) -> None:
        dispatch_started.set()
        assert release_dispatch.wait(timeout=10)

    session.dispatch_prepared_receive = Mock(side_effect=dispatch_prepared_receive)
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._ever_had_recv_session = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(
        create_rx_session=Mock(return_value=session),
        shutdown=Mock(return_value=True),
    )
    transceiver._create_kv_slice = Mock(return_value=KVSlice(is_last_slice=True))
    transceiver._slice_num_bytes = Mock(return_value=64)
    transceiver._kv_size_rank_factor = 1
    request = SimpleNamespace(
        request_id=rid,
        py_request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
        set_kv_cache_transfer_start=Mock(),
        state=None,
    )

    def receive() -> None:
        try:
            transceiver.request_and_receive_async(request)
        except Exception as error:
            receive_errors.append(error)

    receive_thread = threading.Thread(target=receive, daemon=True)
    receive_thread.start()
    try:
        assert dispatch_started.wait(timeout=10)

        assert transceiver.shutdown() is False
        assert transceiver._shutdown_started
        assert rid in transceiver._recv_sessions
        transceiver._transfer_worker.shutdown.assert_not_called()
    finally:
        release_dispatch.set()
        receive_thread.join(timeout=10)
        transceiver_mod._NON_DRAINED_TRANSCEIVERS.discard(transceiver)

    assert not receive_thread.is_alive()
    assert receive_errors == []
    session.resources_drained.return_value = True
    assert transceiver.shutdown() is True


@pytest.mark.cpu_only
def test_shutdown_refuses_to_deregister_active_owners() -> None:
    rid = 149
    session = SimpleNamespace(
        disagg_request_id=rid,
        cancel_local=Mock(return_value=True),
        resources_drained=Mock(return_value=False),
        close=Mock(return_value=True),
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._send_sessions = {rid: session}
    transceiver._send_reqs = {rid: object()}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(shutdown=Mock(return_value=True))

    assert transceiver.shutdown() is False

    assert rid in transceiver._send_sessions
    assert not transceiver._shutdown
    transceiver._transfer_worker.shutdown.assert_not_called()

    session.resources_drained.return_value = True
    assert transceiver.shutdown() is True

    assert transceiver._shutdown
    assert transceiver._send_sessions == {}
    session.close.assert_called_once()
    transceiver._transfer_worker.shutdown.assert_called_once()


@pytest.mark.cpu_only
def test_transceiver_retains_roots_when_worker_shutdown_proof_fails() -> None:
    rid = 163
    session = SimpleNamespace(
        disagg_request_id=rid,
        cancel_local=Mock(return_value=True),
        resources_drained=Mock(return_value=True),
        close=Mock(return_value=True),
    )
    request = object()
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._send_sessions = {rid: session}
    transceiver._send_reqs = {rid: request}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = SimpleNamespace(
        shutdown=Mock(side_effect=[False, True]),
    )
    try:
        assert transceiver.shutdown() is False
        assert transceiver._send_sessions == {rid: session}
        assert transceiver._send_reqs == {rid: request}
        assert transceiver._shutdown_started
        assert not transceiver._shutdown

        assert transceiver.shutdown() is True
        assert transceiver._send_sessions == {}
        assert transceiver._send_reqs == {}
    finally:
        transceiver_mod._NON_DRAINED_TRANSCEIVERS.discard(transceiver)


@pytest.mark.cpu_only
def test_transfer_worker_shutdown_is_single_transaction_under_concurrency() -> None:
    entered = threading.Event()
    release = threading.Event()

    def stop_rank_info(*, strict: bool) -> bool:
        assert strict
        entered.set()
        assert release.wait(timeout=10)
        return True

    worker = object.__new__(TransferWorker)
    worker._shutdown_lock = threading.Lock()
    worker._shutdown = False
    worker._shutdown_started = False
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    worker._rank_info_server = SimpleNamespace(shutdown=Mock(side_effect=stop_rank_info))
    worker._sender = SimpleNamespace(shutdown=Mock(return_value=True))
    worker._receiver = SimpleNamespace(shutdown=Mock(return_value=True))
    worker._bounce = SimpleNamespace(close=Mock())
    worker._registered_mem = []
    agent = SimpleNamespace(shutdown=Mock())
    worker._agent = agent
    results: list[bool] = []

    first = threading.Thread(target=lambda: results.append(worker.shutdown()), daemon=True)
    second = threading.Thread(target=lambda: results.append(worker.shutdown()), daemon=True)
    first.start()
    assert entered.wait(timeout=10)
    second.start()
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert results == [True, True]
    worker._rank_info_server.shutdown.assert_called_once_with(strict=True)
    worker._sender.shutdown.assert_called_once_with(strict=True)
    worker._receiver.shutdown.assert_called_once_with(strict=True)
    worker._bounce.close.assert_called_once_with()
    agent.shutdown.assert_called_once_with()


@pytest.mark.cpu_only
def test_transceiver_shutdown_is_single_transaction_under_concurrency() -> None:
    entered = threading.Event()
    release = threading.Event()

    def stop_worker() -> bool:
        entered.set()
        assert release.wait(timeout=10)
        return True

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._shutdown_lock = threading.Lock()
    transceiver._lifecycle_lock = threading.Lock()
    transceiver._shutdown_started = False
    transceiver._shutdown = False
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    worker = SimpleNamespace(shutdown=Mock(side_effect=stop_worker))
    transceiver._transfer_worker = worker
    results: list[bool] = []

    first = threading.Thread(target=lambda: results.append(transceiver.shutdown()), daemon=True)
    second = threading.Thread(target=lambda: results.append(transceiver.shutdown()), daemon=True)
    first.start()
    assert entered.wait(timeout=10)
    second.start()
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    try:
        assert not first.is_alive()
        assert not second.is_alive()
        assert results == [True, True]
        worker.shutdown.assert_called_once_with()
    finally:
        transceiver_mod._NON_DRAINED_TRANSCEIVERS.discard(transceiver)


@pytest.mark.cpu_only
def test_sender_strict_shutdown_retries_live_worker_thread() -> None:
    worker_thread = SimpleNamespace(
        name="kv-send-worker",
        join=Mock(),
        is_alive=Mock(side_effect=[True, False]),
    )
    task_queue = SimpleNamespace(put=Mock())
    sender = object.__new__(Sender)
    sender._shutdown = False
    sender._shutdown_started = False
    sender._stop_signals_sent = False
    sender._messenger = SimpleNamespace(stop=Mock(), _listener_thread=None)
    sender._send_task_queues = [task_queue]
    sender._worker_threads = [worker_thread]
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._loaded_remote_agents = set()
    sender._dealers = {}

    assert sender.shutdown(strict=True) is False
    assert not sender._shutdown
    task_queue.put.assert_called_once_with(None)

    assert sender.shutdown(strict=True) is True
    assert sender._shutdown
    task_queue.put.assert_called_once_with(None)


def _make_strict_transfer_worker(agent) -> TransferWorker:
    worker = object.__new__(TransferWorker)
    worker._config = SimpleNamespace(enforce_physical_ownership=True)
    worker._shutdown = False
    worker._shutdown_started = False
    worker._rank_info_server = SimpleNamespace(shutdown=Mock(return_value=True))
    worker._sender = SimpleNamespace(shutdown=Mock(return_value=True))
    worker._receiver = SimpleNamespace(shutdown=Mock(return_value=True))
    worker._bounce = SimpleNamespace(close=Mock())
    worker._registered_mem = [object()]
    worker._agent = agent
    return worker


@pytest.mark.cpu_only
def test_worker_strict_shutdown_retains_failed_registration_for_retry() -> None:
    agent = SimpleNamespace(
        deregister_memory=Mock(side_effect=[RuntimeError("deregister failed"), None]),
        shutdown=Mock(),
    )
    worker = _make_strict_transfer_worker(agent)
    descriptor = worker._registered_mem[0]

    assert worker.shutdown() is False
    assert worker._registered_mem == [descriptor]
    assert worker._agent is agent
    agent.shutdown.assert_not_called()

    assert worker.shutdown() is True
    assert worker._registered_mem == []
    assert worker._agent is None


@pytest.mark.cpu_only
def test_worker_strict_shutdown_retains_agent_after_shutdown_failure() -> None:
    agent = SimpleNamespace(
        deregister_memory=Mock(),
        shutdown=Mock(side_effect=[RuntimeError("agent shutdown failed"), None]),
    )
    worker = _make_strict_transfer_worker(agent)

    assert worker.shutdown() is False
    assert worker._registered_mem == []
    assert worker._agent is agent

    assert worker.shutdown() is True
    assert worker._agent is None


@pytest.mark.cpu_only
def test_in_doubt_transceiver_is_quarantined_before_executor_fail_stop() -> None:
    fault = RuntimeError("completion unknown")
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._physical_ownership_enabled = True
    transceiver._transfer_worker = SimpleNamespace(physical_ownership_fault=fault)
    try:
        assert transceiver.get_physical_ownership_fault() is fault
        assert transceiver in transceiver_mod._NON_DRAINED_TRANSCEIVERS
    finally:
        transceiver_mod._NON_DRAINED_TRANSCEIVERS.discard(transceiver)
