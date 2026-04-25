# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuMemUnmap-on-dead-peer prototype for WideEP FT Audit 1a (Day 3).

Question (per §9.1 Audit 1a Day 3):
    `cuMemCreate` with posix-FD handle type (not fabric), map cross-process,
    SIGKILL the owning process, then on the surviving peer test:
      - cuMemUnmap on the mapping
      - cuMemRelease on the imported handle
      - cuMemAddressFree on the VA reservation
    Does each succeed, hang, or segfault?

This isolates the "what happens to driver-side mappings when the owner
process dies" question from anything fabric-specific. The rack-fabric
variant (Audit 1b) inherits this behavior plus whatever NVSwitch fabric
manager does.

Setup
-----
Three processes coordinated by a launcher:
  - Owner: cuMemCreate, cuMemMap, write a known pattern, export FD.
  - Peer:  receive FD, cuMemImportFromShareableHandle, cuMemMap, read &
           verify the pattern. Signal "ready". Then wait. After the
           owner is killed, run the unmap/release/free sequence and log
           each call's return code + elapsed time.
  - Launcher: spawns owner + peer, hands the peer a Unix socket FD to
           receive the GPU FD, SIGKILLs the owner once peer signals
           "ready", waits for peer to finish & write its log.

Outputs
-------
JSONL log from the peer with one line per CUDA call attempted, recording
return code (`CUresult`) and elapsed milliseconds.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from cuda.bindings import driver, runtime

# 4 KiB allocation, single page on most systems. Granularity is platform
# dependent so we round up to whatever cuMemGetAllocationGranularity says
# at runtime.
DESIRED_SIZE = 4 * 1024


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------


def _ck(result):
    """Unpack the (CUresult, *return_values) tuple a cuda-python call returns.

    Raises RuntimeError on non-success; returns the single value, the value
    tuple, or None depending on what the underlying call produced.
    """
    err = result[0]
    rest = result[1:] if len(result) > 1 else ()
    if err != driver.CUresult.CUDA_SUCCESS:
        # cuGetErrorName / cuGetErrorString themselves return (CUresult, value).
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
    # cuCtxCreate signature in CUDA 13: (ctxCreateParams, flags, dev).
    # None for ctxCreateParams uses defaults.
    ctx = _ck(driver.cuCtxCreate(None, 0, dev))
    return dev, ctx


def make_alloc_prop(device_id: int):
    """CUmemAllocationProp asking for a posix-FD shareable allocation."""
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = (
        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.allocFlags.gpuDirectRDMACapable = 0
    prop.allocFlags.usage = 0
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


# ---------------------------------------------------------------------------
# FD passing via SCM_RIGHTS
# ---------------------------------------------------------------------------


def send_fd(sock: socket.socket, fd: int) -> None:
    sock.sendmsg([b"FD"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])


def recv_fd(sock: socket.socket) -> int:
    msg, ancdata, _, _ = sock.recvmsg(2, socket.CMSG_LEN(struct.calcsize("i")))
    for level, typ, data in ancdata:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            return struct.unpack("i", data[:4])[0]
    raise RuntimeError("no FD received")


# ---------------------------------------------------------------------------
# Owner role
# ---------------------------------------------------------------------------


def owner_main(sock_path: str, ready_sock_path: str, log_path: Path, device_id: int) -> int:
    def log(ev):
        log_path.open("a").write(
            json.dumps({**ev, "role": "owner", "pid": os.getpid(), "t": time.monotonic()}) + "\n"
        )

    log({"event": "owner_start", "device_id": device_id})

    cuda_init(device_id)
    prop = make_alloc_prop(device_id)
    size = aligned_size(prop, DESIRED_SIZE)
    log({"event": "alloc_size", "size": size})

    # Allocate physical memory.
    handle = _ck(driver.cuMemCreate(size, prop, 0))
    log({"event": "cuMemCreate_ok", "handle": int(handle)})

    # Reserve VA + map.
    va = _ck(driver.cuMemAddressReserve(size, 0, 0, 0))
    _ck(driver.cuMemMap(va, size, 0, handle, 0))
    _ck(driver.cuMemSetAccess(va, size, [make_access_desc(device_id)], 1))
    log({"event": "cuMemMap_ok", "va": int(va)})

    # Write a known pattern.
    pattern = (b"OWN" + b"\x00" * 13) * (size // 16)
    _ck(runtime.cudaMemcpy(int(va), pattern, size, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice))
    _ck(driver.cuCtxSynchronize())
    log({"event": "wrote_pattern", "first_bytes_sent": pattern[:16].hex()})

    # Export to shareable FD.
    fd = _ck(
        driver.cuMemExportToShareableHandle(
            handle,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
    )
    log({"event": "exported_fd", "fd": int(fd)})

    # Send the FD to the peer.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    send_fd(sock, int(fd))
    log({"event": "fd_sent_to_peer"})

    # Block on the ready_sock until the peer says it's done verifying. Then
    # the launcher will SIGKILL us.
    ready_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_sock.connect(ready_sock_path)
    ready_sock.send(b"OWNER_READY")
    ready_sock.recv(64)  # block here until launcher kills us

    # Should never reach here normally.
    log({"event": "owner_unexpected_exit"})
    return 0


# ---------------------------------------------------------------------------
# Peer role
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
    prop = make_alloc_prop(device_id)
    size = aligned_size(prop, DESIRED_SIZE)

    # Receive FD from owner.
    listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen.bind(sock_path)
    listen.listen(1)
    conn, _ = listen.accept()
    fd = recv_fd(conn)
    log({"event": "fd_received", "fd": fd})

    # Import to handle.
    handle = _ck(
        driver.cuMemImportFromShareableHandle(
            fd,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
    )
    log({"event": "cuMemImportFromShareableHandle_ok", "handle": int(handle)})

    # Reserve VA + map locally.
    va = _ck(driver.cuMemAddressReserve(size, 0, 0, 0))
    _ck(driver.cuMemMap(va, size, 0, handle, 0))
    _ck(driver.cuMemSetAccess(va, size, [make_access_desc(device_id)], 1))
    log({"event": "cuMemMap_ok", "va": int(va), "size": size})

    # Read back and verify the owner's pattern.
    buf = bytearray(size)
    _ck(runtime.cudaMemcpy(buf, int(va), size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    _ck(driver.cuCtxSynchronize())
    cross_process_share_verified = buf[:3] == b"OWN"
    log(
        {
            "event": "cross_process_read",
            "first_bytes_seen": bytes(buf[:16]).hex(),
            "cross_process_share_verified": cross_process_share_verified,
        }
    )

    # Tell launcher we're ready (launcher will then SIGKILL the owner).
    ready_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_sock.connect(ready_sock_path)
    ready_sock.send(b"PEER_READY")
    ready_sock.close()

    # Wait long enough for launcher to kill the owner.
    time.sleep(post_kill_wait_s)
    log({"event": "post_kill_wait_done", "post_kill_wait_s": post_kill_wait_s})

    # Try to read AFTER the owner is dead: does the mapping still hold?
    try:
        t0 = time.monotonic()
        buf2 = bytearray(size)
        rc = runtime.cudaMemcpy(buf2, int(va), size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        rc_str = str(rc[0])
        sync_rc = driver.cuCtxSynchronize()
        sync_rc_str = str(sync_rc[0])
        elapsed = (time.monotonic() - t0) * 1000.0
        log(
            {
                "event": "post_kill_read_attempt",
                "rc": rc_str,
                "sync_rc": sync_rc_str,
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

    # The headline test: cuMemUnmap, cuMemRelease, cuMemAddressFree on a
    # mapping whose owner is dead. Each call's CUresult + elapsed gets logged.
    for fn_name, fn_call in [
        ("cuMemUnmap", lambda: driver.cuMemUnmap(va, size)),
        ("cuMemRelease", lambda: driver.cuMemRelease(handle)),
        ("cuMemAddressFree", lambda: driver.cuMemAddressFree(va, size)),
    ]:
        try:
            t0 = time.monotonic()
            rc = fn_call()
            elapsed = (time.monotonic() - t0) * 1000.0
            err_, name = driver.cuGetErrorName(rc[0])
            err__, desc = driver.cuGetErrorString(rc[0])
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
# Launcher
# ---------------------------------------------------------------------------


def launcher_main(args) -> int:
    out_dir = Path(args.log_dir)
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    owner_log = out_dir / "owner.jsonl"
    peer_log = out_dir / "peer.jsonl"

    # Two unix sockets: one for FD passing (peer is server, owner client),
    # one for ready signaling (launcher is server).
    fd_sock_path = str(out_dir / "fd.sock")
    ready_sock_path = str(out_dir / "ready.sock")

    # Set up the ready listener BEFORE spawning children.
    ready_listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_listen.bind(ready_sock_path)
    ready_listen.listen(2)

    py = sys.executable
    common_env = os.environ.copy()

    # Spawn peer first (it's the FD-passing server) so the owner has someone
    # to connect to.
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
    time.sleep(0.5)  # let peer bind the FD socket

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

    # Wait for both children to signal ready.
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

    # SIGKILL the owner; the peer will then attempt unmap/release/free.
    print(f"[launcher] SIGKILL owner pid={owner_proc.pid}")
    owner_proc.kill()
    owner_rc = owner_proc.wait()
    print(f"[launcher] owner exit={owner_rc}")

    # Release the owner's ready connection (it would have been blocked on recv).
    for c in conns:
        try:
            c.close()
        except OSError:
            pass

    # Wait for peer to finish.
    try:
        peer_rc = peer_proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        peer_proc.kill()
        peer_rc = -1
        print("[launcher] peer timed out (likely hang on cuMemUnmap)")
    print(f"[launcher] peer exit={peer_rc}")

    # Print peer log tail
    if peer_log.exists():
        print("\n--- peer log ---")
        for line in peer_log.read_text().splitlines()[-20:]:
            print(line)
    return 0 if peer_rc == 0 else peer_rc


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["launcher", "owner", "peer"], default="launcher")
    p.add_argument("--sock-path", default="")
    p.add_argument("--ready-sock-path", default="")
    p.add_argument("--log-path", default="")
    p.add_argument("--log-dir", default="/tmp/audit-1a-prototypes/cumem_unmap_dead_peer")
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
