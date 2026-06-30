# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric-handle variant of cumem_unmap_dead_peer.py for Audit 1a Day 4-5.

Same shape as the posix-FD variant but uses CU_MEM_HANDLE_TYPE_FABRIC
(the Grace/aarch64 NVL72 MNNVL mode, not the current x86_64 mode). The cross-process IPC carries a 64-byte
CUmemFabricHandle struct instead of a file descriptor.

Question being answered:
    Does the answer from the posix-FD variant ("cuMemUnmap on dead-peer
    region: CUDA_SUCCESS in ~0.25 ms") generalize to fabric-typed
    allocations on intra-node NVLink? If yes, intra-node MNNVL teardown
    cost is ~ms-scale and PR 2a.2 sizing is bounded above.

Hardware: tested on B300 SXM6 single node (intra-node NVLink). Cross-node
rack-fabric behavior remains a separate question for Audit 1b.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from cuda.bindings import driver, runtime

DESIRED_SIZE = 4 * 1024
FABRIC_HANDLE_SIZE = 64  # CUmemFabricHandle is a 64-byte opaque ID.


def _ck(result):
    err = result[0]
    rest = result[1:] if len(result) > 1 else ()
    if err != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(err)
        _, desc = driver.cuGetErrorString(err)
        nm = name.decode() if isinstance(name, bytes) else str(name)
        ds = desc.decode() if isinstance(desc, bytes) else str(desc)
        raise RuntimeError(f"CUDA error: {nm}: {ds}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


def cuda_init(device_id: int = 0):
    _ck(driver.cuInit(0))
    dev = _ck(driver.cuDeviceGet(device_id))
    ctx = _ck(driver.cuCtxCreate(None, 0, dev))
    return dev, ctx


def make_alloc_prop_fabric(device_id: int):
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    return prop


def aligned_size(prop, desired: int) -> int:
    gran = _ck(
        driver.cuMemGetAllocationGranularity(
            prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
    )
    return ((desired + gran - 1) // gran) * gran


def make_access_desc(device_id: int):
    desc = driver.CUmemAccessDesc()
    desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def fabric_handle_to_bytes(fh) -> bytes:
    """Serialize CUmemFabricHandle.data (64 bytes) for cross-process transfer."""
    # cuda-python exposes the opaque .data field as a list/array of 64 ints.
    # Robust extraction works whether it's bytes-like or int-list.
    raw = bytes(fh.data) if hasattr(fh.data, "__bytes__") else bytes(list(fh.data))
    if len(raw) != FABRIC_HANDLE_SIZE:
        # Fall back: iterate
        raw = bytes(int(b) & 0xFF for b in fh.data)
    assert len(raw) == FABRIC_HANDLE_SIZE, f"got {len(raw)} bytes, want {FABRIC_HANDLE_SIZE}"
    return raw


def bytes_to_fabric_handle(buf: bytes):
    fh = driver.CUmemFabricHandle()
    assert len(buf) == FABRIC_HANDLE_SIZE
    for i, b in enumerate(buf):
        fh.data[i] = b
    return fh


# ---------------------------------------------------------------------------
# Owner
# ---------------------------------------------------------------------------


def owner_main(sock_path: str, ready_sock_path: str, log_path: Path, device_id: int) -> int:
    def log(ev):
        log_path.open("a").write(
            json.dumps({**ev, "role": "owner", "pid": os.getpid(), "t": time.monotonic()}) + "\n"
        )

    log({"event": "owner_start", "device_id": device_id})

    cuda_init(device_id)
    prop = make_alloc_prop_fabric(device_id)
    size = aligned_size(prop, DESIRED_SIZE)
    log({"event": "alloc_size", "size": size})

    handle = _ck(driver.cuMemCreate(size, prop, 0))
    log({"event": "cuMemCreate_ok", "handle": int(handle)})

    va = _ck(driver.cuMemAddressReserve(size, 0, 0, 0))
    _ck(driver.cuMemMap(va, size, 0, handle, 0))
    _ck(driver.cuMemSetAccess(va, size, [make_access_desc(device_id)], 1))
    log({"event": "cuMemMap_ok", "va": int(va)})

    pattern = (b"FAB" + b"\x00" * 13) * (size // 16)
    _ck(runtime.cudaMemcpy(int(va), pattern, size, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice))
    _ck(driver.cuCtxSynchronize())
    log({"event": "wrote_pattern", "first_bytes_sent": pattern[:16].hex()})

    fh = _ck(
        driver.cuMemExportToShareableHandle(
            handle,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
    )
    fh_bytes = fabric_handle_to_bytes(fh)
    log({"event": "exported_fabric_handle", "handle_first_8_bytes": fh_bytes[:8].hex()})

    # Send the 64-byte fabric handle struct over Unix socket.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    sock.sendall(struct.pack("!I", FABRIC_HANDLE_SIZE) + fh_bytes)
    log({"event": "fabric_handle_sent_to_peer"})

    ready_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_sock.connect(ready_sock_path)
    ready_sock.send(b"OWNER_READY")
    ready_sock.recv(64)  # block until launcher kills us

    log({"event": "owner_unexpected_exit"})
    return 0


# ---------------------------------------------------------------------------
# Peer
# ---------------------------------------------------------------------------


def peer_main(
    sock_path: str, ready_sock_path: str, log_path: Path, device_id: int, post_kill_wait_s: float
) -> int:
    def log(ev):
        log_path.open("a").write(
            json.dumps({**ev, "role": "peer", "pid": os.getpid(), "t": time.monotonic()}) + "\n"
        )

    log({"event": "peer_start", "device_id": device_id})

    cuda_init(device_id)
    prop = make_alloc_prop_fabric(device_id)
    size = aligned_size(prop, DESIRED_SIZE)

    listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen.bind(sock_path)
    listen.listen(1)
    conn, _ = listen.accept()
    hdr = conn.recv(4)
    n = struct.unpack("!I", hdr)[0]
    fh_bytes = b""
    while len(fh_bytes) < n:
        chunk = conn.recv(n - len(fh_bytes))
        if not chunk:
            break
        fh_bytes += chunk
    log({"event": "fabric_handle_received", "size": len(fh_bytes), "first_8": fh_bytes[:8].hex()})

    fh = bytes_to_fabric_handle(fh_bytes)
    handle = _ck(
        driver.cuMemImportFromShareableHandle(
            fh,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        )
    )
    log({"event": "cuMemImportFromShareableHandle_ok", "handle": int(handle)})

    va = _ck(driver.cuMemAddressReserve(size, 0, 0, 0))
    _ck(driver.cuMemMap(va, size, 0, handle, 0))
    _ck(driver.cuMemSetAccess(va, size, [make_access_desc(device_id)], 1))
    log({"event": "cuMemMap_ok", "va": int(va), "size": size})

    buf = bytearray(size)
    _ck(runtime.cudaMemcpy(buf, int(va), size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    _ck(driver.cuCtxSynchronize())
    cross_process_share_verified = buf[:3] == b"FAB"
    log(
        {
            "event": "cross_process_read",
            "first_bytes_seen": bytes(buf[:16]).hex(),
            "cross_process_share_verified": cross_process_share_verified,
        }
    )

    ready_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_sock.connect(ready_sock_path)
    ready_sock.send(b"PEER_READY")
    ready_sock.close()

    time.sleep(post_kill_wait_s)
    log({"event": "post_kill_wait_done", "post_kill_wait_s": post_kill_wait_s})

    # Read after owner is dead.
    try:
        t0 = time.monotonic()
        buf2 = bytearray(size)
        rc = runtime.cudaMemcpy(buf2, int(va), size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        sync_rc = driver.cuCtxSynchronize()
        elapsed = (time.monotonic() - t0) * 1000.0
        log(
            {
                "event": "post_kill_read_attempt",
                "rc": str(rc[0]),
                "sync_rc": str(sync_rc[0]),
                "elapsed_ms": elapsed,
                "first_bytes_seen": bytes(buf2[:16]).hex(),
            }
        )
    except Exception as e:
        log(
            {
                "event": "post_kill_read_exception",
                "exc_type": type(e).__name__,
                "exc_msg": str(e)[:200],
            }
        )

    for fn_name, fn_call in [
        ("cuMemUnmap", lambda: driver.cuMemUnmap(va, size)),
        ("cuMemRelease", lambda: driver.cuMemRelease(handle)),
        ("cuMemAddressFree", lambda: driver.cuMemAddressFree(va, size)),
    ]:
        try:
            t0 = time.monotonic()
            rc = fn_call()
            elapsed = (time.monotonic() - t0) * 1000.0
            _, name = driver.cuGetErrorName(rc[0])
            _, desc = driver.cuGetErrorString(rc[0])
            log(
                {
                    "event": f"{fn_name}_attempt",
                    "rc": int(rc[0]),
                    "rc_name": (name.decode() if isinstance(name, bytes) else str(name)),
                    "rc_desc": (desc.decode() if isinstance(desc, bytes) else str(desc)),
                    "elapsed_ms": elapsed,
                }
            )
        except Exception as e:
            log(
                {
                    "event": f"{fn_name}_exception",
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e)[:200],
                }
            )

    log({"event": "peer_done"})
    return 0


# ---------------------------------------------------------------------------
# Launcher (same as posix variant)
# ---------------------------------------------------------------------------


def launcher_main(args) -> int:
    out_dir = Path(args.log_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    owner_log = out_dir / "owner.jsonl"
    peer_log = out_dir / "peer.jsonl"
    fd_sock_path = str(out_dir / "fab.sock")
    ready_sock_path = str(out_dir / "ready.sock")

    ready_listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_listen.bind(ready_sock_path)
    ready_listen.listen(2)

    py = sys.executable
    common_env = os.environ.copy()

    peer_proc = subprocess.Popen(
        [
            py,
            __file__,
            "--role",
            "peer",
            "--sock-path",
            fd_sock_path,
            "--ready-sock-path",
            ready_sock_path,
            "--log-path",
            str(peer_log),
            "--device-id",
            str(args.device_id),
            "--post-kill-wait-s",
            str(args.post_kill_wait_s),
        ],
        env=common_env,
    )
    time.sleep(0.5)

    owner_proc = subprocess.Popen(
        [
            py,
            __file__,
            "--role",
            "owner",
            "--sock-path",
            fd_sock_path,
            "--ready-sock-path",
            ready_sock_path,
            "--log-path",
            str(owner_log),
            "--device-id",
            str(args.device_id),
        ],
        env=common_env,
    )

    saw_owner = saw_peer = False
    deadline = time.monotonic() + 30.0
    conns = []
    while not (saw_owner and saw_peer) and time.monotonic() < deadline:
        ready_listen.settimeout(max(0.5, deadline - time.monotonic()))
        try:
            conn, _ = ready_listen.accept()
            msg = conn.recv(32)
            conns.append(conn)
            if msg == b"OWNER_READY":
                saw_owner = True
                print("[launcher] owner ready")
            elif msg == b"PEER_READY":
                saw_peer = True
                print("[launcher] peer ready")
        except socket.timeout:
            pass

    if not (saw_owner and saw_peer):
        print(f"[launcher] timed out waiting for ready (owner={saw_owner}, peer={saw_peer})")
        owner_proc.kill()
        peer_proc.kill()
        return 1

    print(f"[launcher] SIGKILL owner pid={owner_proc.pid}")
    owner_proc.kill()
    print(f"[launcher] owner exit={owner_proc.wait()}")

    for c in conns:
        try:
            c.close()
        except OSError:
            pass

    try:
        peer_rc = peer_proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        peer_proc.kill()
        peer_rc = -1
        print("[launcher] peer timed out (likely hang on cuMemUnmap)")
    print(f"[launcher] peer exit={peer_rc}")

    if peer_log.exists():
        print("\n--- peer log ---")
        for line in peer_log.read_text().splitlines()[-20:]:
            print(line)
    return 0 if peer_rc == 0 else peer_rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["launcher", "owner", "peer"], default="launcher")
    p.add_argument("--sock-path", default="")
    p.add_argument("--ready-sock-path", default="")
    p.add_argument("--log-path", default="")
    p.add_argument("--log-dir", default="/tmp/audit-1a-prototypes/cumem_unmap_dead_peer_fabric")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--post-kill-wait-s", type=float, default=2.0)
    args = p.parse_args()

    if args.role == "owner":
        return owner_main(args.sock_path, args.ready_sock_path, Path(args.log_path), args.device_id)
    elif args.role == "peer":
        return peer_main(
            args.sock_path,
            args.ready_sock_path,
            Path(args.log_path),
            args.device_id,
            args.post_kill_wait_s,
        )
    else:
        return launcher_main(args)


if __name__ == "__main__":
    sys.exit(main())
