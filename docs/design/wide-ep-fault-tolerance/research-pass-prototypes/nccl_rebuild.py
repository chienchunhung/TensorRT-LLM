# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL rebuild prototype for WideEP FT Audit 1 (Day 1).

Question: when one rank dies via SIGKILL during an NCCL collective, can the
surviving N-1 ranks abort the broken communicator and create a new one
without restarting the whole job? What's the latency budget?

Setup
-----
4 worker processes, one per GPU, joined via a TCPStore-backed NCCL process
group. A designated victim rank self-`SIGKILL`s after several iterations of a
warm-up `all_reduce` loop. Survivors detect the failure via NCCL's watchdog
timeout, call `destroy_process_group()` (which internally calls
`ncclCommAbort`), then rendezvous with each other via the same TCPStore and
re-`init_process_group()` with `world_size = N - 1`.

The script is invoked once with `--launch`; the launcher spawns the workers
as separate processes (so SIGKILL semantics are realistic), waits for them,
and aggregates per-rank JSONL logs into a summary.

Measurements (each in ms, on the survivors)
-------------------------------------------
- detect_latency: t_detect - t_victim_kill
    (how long after the victim's kill the survivors first see an NCCL error)
- abort_latency: t_abort_done - t_detect
    (destroy_process_group on the broken comm)
- rebuild_latency: t_rebuild_done - t_abort_done
    (init_process_group with N-1)
- e2e_recovery: t_first_new_collective_ok - t_victim_kill
    (rank dies → first successful all_reduce on the new comm)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure NCCL behavior. There are two relevant modes:
#   ASYNC_ERROR_HANDLING=1, BLOCKING_WAIT=0 (PyTorch default):
#     - The collective returns immediately, work is async.
#     - A separate watchdog thread polls for timeouts and, on timeout,
#       calls std::terminate(), which raises SIGABRT and kills the entire
#       process. The main thread never observes the failure.
#     - Empirically observed in run #1: a single peer death killed all
#       survivors with exit code -6.
#   BLOCKING_WAIT=1:
#     - The collective blocks the main thread until completion or timeout.
#     - On timeout, raises a Python exception (DistBackendError) that the
#       main thread can catch -> the survivor recovery path is reachable.
# The choice is set via the launcher's --blocking-wait flag.
# These environment variables MUST be set before torch imports NCCL.
# Disable BOTH the watchdog terminate behavior AND BLOCKING_WAIT. We rely
# entirely on our own main-thread polling of the Work handle — that's the
# only PT 2.11 mode that lets a survivor catch a peer-death timeout AND
# proceed to call dist.shrink_group(...) without the watchdog thread
# concurrently calling std::terminate() on the process.
#
# Empirical findings from earlier runs:
#   ASYNC=1 (default): watchdog fires terminate() at timeout -> SIGABRT
#   BLOCKING=1 alone:  watchdog still fires; survivors crash silently
#   ASYNC=1 + custom polling + shrink_group: watchdog races with main;
#                       SIGABRT mid-shrink (run #5)
#   ASYNC=0 + custom polling + shrink_group: WORKS (this run)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
os.environ.setdefault("NCCL_DEBUG", "WARN")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def now_ms() -> float:
    return time.monotonic() * 1000.0


def log_event(log_path: Path, event: dict) -> None:
    event["t_ms"] = now_ms()
    event["pid"] = os.getpid()
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker(
    rank: int,
    world_size: int,
    victim_rank: int,
    master_port: int,
    victim_iter: int,
    log_path: Path,
    watchdog_sec: int,
) -> int:
    log_event(
        log_path,
        {
            "event": "worker_start",
            "rank": rank,
            "world_size": world_size,
            "victim_rank": victim_rank,
            "master_port": master_port,
            "victim_iter": victim_iter,
        },
    )

    torch.cuda.set_device(rank)

    # Build a TCPStore. is_master must be True on exactly one rank — pick rank 0.
    # Survivors rebuild a NEW process group from this same store after the
    # victim dies, so the store needs to outlive the broken NCCL group.
    is_master = rank == 0
    store = dist.TCPStore(
        host_name="127.0.0.1",
        port=master_port,
        world_size=world_size,
        is_master=is_master,
        timeout=datetime.timedelta(seconds=30),
    )
    log_event(log_path, {"event": "tcpstore_ready", "rank": rank})

    # ---- Phase A: initial process group with all N ranks ----
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        store=store,
        timeout=datetime.timedelta(seconds=watchdog_sec),
    )
    log_event(log_path, {"event": "init_pg_done", "phase": "A", "rank": rank})

    # Tag tensors with rank so any cross-rank corruption shows up.
    data = torch.full((1024,), float(rank), device="cuda")

    # Steady-state loop, victim self-kills mid-iteration. Each call uses
    # async_op=True so we get a Work handle we can poll with an explicit
    # bounded wait — in PT 2.11, neither the watchdog (which terminate()s
    # the process) nor BLOCKING_WAIT=1 (which doesn't surface a Python
    # exception) gives us a recoverable failure path. Polling the Work
    # handle with a per-iteration deadline is the closest pattern to what
    # a real out-of-band failure detector would do (see §5.3 / §1c.3).
    for it in range(20):
        try:
            work = dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=True)
            deadline = time.monotonic() + watchdog_sec
            while not work.is_completed():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"all_reduce iter {it}: peer did not respond within {watchdog_sec}s"
                    )
                time.sleep(0.01)
            torch.cuda.synchronize()
            if it < 3 or it % 5 == 0:
                log_event(
                    log_path,
                    {
                        "event": "all_reduce_ok",
                        "phase": "A",
                        "rank": rank,
                        "it": it,
                        "sum": float(data.sum().item()),
                    },
                )
        except Exception as e:
            t_detect = now_ms()
            log_event(
                log_path,
                {
                    "event": "all_reduce_failed",
                    "phase": "A",
                    "rank": rank,
                    "it": it,
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e)[:200],
                    "t_detect_ms": t_detect,
                },
            )
            # No explicit pg.abort() needed here — dist.shrink_group(... ,
            # shrink_flags=SHRINK_ABORT) below atomically aborts the in-flight
            # collective on the parent group as part of the shrink.
            break

        if rank == victim_rank and it == victim_iter:
            log_event(
                log_path,
                {
                    "event": "victim_self_killing",
                    "rank": rank,
                    "it": it,
                    "t_victim_kill_ms": now_ms(),
                },
            )
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(0.05)
    else:
        # No exception caught — victim never died (only happens if victim_rank
        # is out of [0, world_size) or victim_iter > 19).
        log_event(log_path, {"event": "loop_done_no_failure", "rank": rank})
        dist.destroy_process_group()
        return 0

    # ---- Phase B: survivor handles failure via dist.shrink_group ----
    # PT 2.11 introduced dist.shrink_group(ranks_to_exclude, shrink_flags=
    # SHRINK_ABORT) which atomically aborts the in-flight collective on the
    # parent group and creates a new, smaller group with the surviving ranks.
    # This subsumes the older "destroy_process_group + init_process_group with
    # smaller world_size" pattern and is the natural API for Phase 2 PR 2a.1.
    #
    # Note from the doc: "Only non-excluded ranks should call this function;
    # excluded ranks must not participate." The victim is dead so this is
    # automatic. In production, the dead set comes from §5.3 EPGroupHealth /
    # 1c.3 broadcast; for this audit, survivors know it a priori.
    SHRINK_ABORT = dist.distributed_c10d.SHRINK_ABORT
    t_shrink_start = now_ms()
    try:
        new_pg = dist.shrink_group(
            ranks_to_exclude=[victim_rank],
            shrink_flags=SHRINK_ABORT,
        )
        t_shrink_done = now_ms()
        log_event(
            log_path,
            {
                "event": "shrink_group_ok",
                "rank": rank,
                "shrink_latency_ms": t_shrink_done - t_shrink_start,
                "new_world_size": dist.get_world_size(group=new_pg),
                "new_rank": dist.get_rank(group=new_pg),
            },
        )
    except Exception as e:
        t_shrink_done = now_ms()
        log_event(
            log_path,
            {
                "event": "shrink_group_failed",
                "rank": rank,
                "exc_type": type(e).__name__,
                "exc_msg": str(e)[:300],
                "shrink_latency_ms": t_shrink_done - t_shrink_start,
            },
        )
        return 1

    # Step 2: run a collective on the shrunken group.
    new_rank = dist.get_rank(group=new_pg)
    new_world_size = dist.get_world_size(group=new_pg)
    data_b = torch.full((1024,), float(new_rank), device="cuda")
    try:
        # Use the same async-poll pattern; if shrink_group worked, this should
        # complete fast, but we keep the safety belt to avoid a silent hang
        # if the new group is itself broken.
        work = dist.all_reduce(data_b, op=dist.ReduceOp.SUM, async_op=True, group=new_pg)
        deadline = time.monotonic() + watchdog_sec
        while not work.is_completed():
            if time.monotonic() > deadline:
                raise TimeoutError("post-shrink all_reduce hung")
            time.sleep(0.01)
        torch.cuda.synchronize()
        t_first_collective_ok = now_ms()
        expected = float(sum(range(new_world_size)))
        actual = float(data_b[0].item())
        log_event(
            log_path,
            {
                "event": "first_post_failure_collective_ok",
                "rank": rank,
                "new_rank": new_rank,
                "expected_sum": expected,
                "actual_sum": actual,
                "t_first_new_collective_ms": t_first_collective_ok,
                "correct": expected == actual,
            },
        )
    except Exception as e:
        log_event(
            log_path,
            {
                "event": "first_post_failure_collective_failed",
                "rank": rank,
                "exc_type": type(e).__name__,
                "exc_msg": str(e)[:200],
            },
        )
        dist.destroy_process_group()
        return 1

    dist.destroy_process_group()
    log_event(log_path, {"event": "worker_clean_exit", "rank": rank})
    return 0


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch(
    world_size: int,
    victim_rank: int,
    victim_iter: int,
    master_port: int,
    watchdog_sec: int,
    log_dir: Path,
) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    log_paths = []
    env_base = os.environ.copy()
    # Make all `world_size` GPUs visible to every child; each child picks its
    # own GPU via torch.cuda.set_device(rank). Restricting per-child would
    # cause set_device(rank) to fail since the device renumbers to 0.
    env_base["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
    for rank in range(world_size):
        log_path = log_dir / f"rank_{rank}.jsonl"
        if log_path.exists():
            log_path.unlink()
        log_paths.append(log_path)
        env = env_base.copy()
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--victim-rank",
            str(victim_rank),
            "--master-port",
            str(master_port),
            "--victim-iter",
            str(victim_iter),
            "--watchdog-sec",
            str(watchdog_sec),
            "--log-path",
            str(log_path),
        ]
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
        time.sleep(0.05)  # stagger spawn so the master TCPStore is up first

    # Wait for all (the victim will exit -9 from SIGKILL; survivors should exit 0).
    rcs = []
    for rank, p in enumerate(procs):
        rc = p.wait()
        rcs.append(rc)
        print(f"rank {rank} exit code: {rc}")

    # Aggregate results
    summary = aggregate(log_paths, victim_rank, world_size)
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("recovery_succeeded") else 1


def aggregate(log_paths: list[Path], victim_rank: int, world_size: int) -> dict:
    per_rank = {}
    for p in log_paths:
        if not p.exists():
            continue
        rank = int(p.stem.split("_")[1])
        events = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        per_rank[rank] = events

    # Find t_victim_kill from victim's log.
    t_victim_kill = None
    for ev in per_rank.get(victim_rank, []):
        if ev.get("event") == "victim_self_killing":
            t_victim_kill = ev["t_victim_kill_ms"]

    survivor_summaries = {}
    recovery_ok = True
    for r in sorted(per_rank.keys()):
        if r == victim_rank:
            continue
        evs = per_rank[r]
        s = {"rank": r}
        for ev in evs:
            if ev.get("event") == "all_reduce_failed" and ev.get("phase") == "A":
                s["t_detect_ms"] = ev.get("t_detect_ms")
                s["detected_at_iter"] = ev.get("it")
                s["exc_type"] = ev.get("exc_type")
            if ev.get("event") == "shrink_group_ok":
                s["shrink_latency_ms"] = ev.get("shrink_latency_ms")
                s["new_rank"] = ev.get("new_rank")
                s["new_world_size"] = ev.get("new_world_size")
            if ev.get("event") == "shrink_group_failed":
                s["shrink_failed"] = True
                s["shrink_latency_ms"] = ev.get("shrink_latency_ms")
                s["shrink_exc_type"] = ev.get("exc_type")
                s["shrink_exc_msg"] = ev.get("exc_msg")
                recovery_ok = False
            if ev.get("event") == "first_post_failure_collective_ok":
                s["t_first_new_collective_ms"] = ev.get("t_first_new_collective_ms")
                s["correct"] = ev.get("correct")
            if ev.get("event") == "first_post_failure_collective_failed":
                s["first_collective_failed"] = True
                recovery_ok = False
            if ev.get("event") == "worker_clean_exit":
                s["clean_exit"] = True
        if t_victim_kill is not None and "t_detect_ms" in s:
            s["detect_latency_ms"] = s["t_detect_ms"] - t_victim_kill
        if t_victim_kill is not None and "t_first_new_collective_ms" in s:
            s["e2e_recovery_ms"] = s["t_first_new_collective_ms"] - t_victim_kill
        survivor_summaries[r] = s
        if not s.get("clean_exit"):
            recovery_ok = False

    return {
        "world_size": world_size,
        "victim_rank": victim_rank,
        "t_victim_kill_ms": t_victim_kill,
        "survivors": survivor_summaries,
        "recovery_succeeded": recovery_ok,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--worker", action="store_true", help="run as a worker (internal; --launch spawns these)"
    )
    p.add_argument("--launch", action="store_true", help="run as the launcher")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--victim-rank", type=int, default=2)
    p.add_argument("--victim-iter", type=int, default=5)
    p.add_argument("--master-port", type=int, default=29501)
    p.add_argument("--watchdog-sec", type=int, default=15)
    p.add_argument("--log-path", type=str, default="")
    p.add_argument("--log-dir", type=str, default="/tmp/audit-1a-prototypes/nccl_rebuild")
    args = p.parse_args()

    if args.worker:
        return worker(
            rank=args.rank,
            world_size=args.world_size,
            victim_rank=args.victim_rank,
            master_port=args.master_port,
            victim_iter=args.victim_iter,
            log_path=Path(args.log_path),
            watchdog_sec=args.watchdog_sec,
        )
    elif args.launch:
        return launch(
            world_size=args.world_size,
            victim_rank=args.victim_rank,
            victim_iter=args.victim_iter,
            master_port=args.master_port,
            watchdog_sec=args.watchdog_sec,
            log_dir=Path(args.log_dir),
        )
    else:
        print("Use --launch (default) or --worker", file=sys.stderr)
        return launch(
            world_size=args.world_size,
            victim_rank=args.victim_rank,
            victim_iter=args.victim_iter,
            master_port=args.master_port,
            watchdog_sec=args.watchdog_sec,
            log_dir=Path(args.log_dir),
        )


if __name__ == "__main__":
    sys.exit(main())
