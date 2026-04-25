# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher for the MPI signal-handler prototype.

Runs the worker (mpi_signal_handler.py) under `mpirun -np 4` once per
death mode, captures per-rank logs and the launcher's view of exit
codes, and writes a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "mpi_signal_handler.py"


def run_one(
    mode: str,
    results_dir: Path,
    n_ranks: int,
    victim_rank: int,
    victim_iter: int,
    total_timeout_sec: int,
) -> dict:
    out_dir = results_dir / mode
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    env = os.environ.copy()
    # Strip the recover_ prefix if present (it's just for launcher branching);
    # the worker should still see one of the canonical death modes.
    worker_death_mode = mode.removeprefix("recover_") if mode.startswith("recover_") else mode
    env["AUDIT_DEATH_MODE"] = worker_death_mode
    env["AUDIT_VICTIM_RANK"] = str(victim_rank)
    env["AUDIT_VICTIM_ITER"] = str(victim_iter)
    # Each rank gets its own log file, by rank id (mpi4py exposes RANK via
    # OMPI_COMM_WORLD_RANK or similar; we let the worker read MPI rank
    # *after* MPI.Init and write to a log derived from rank). We pre-create
    # the path here; the worker just appends.
    # Use a single dir; each rank derives its filename from its rank.
    # Easiest: pass a directory and have each rank pick its filename.
    env["AUDIT_LOG_DIR"] = str(out_dir)

    # We need a tiny shim: have each rank set AUDIT_LOG_PATH from rank id
    # before importing mpi4py. Easiest: prepend a one-liner via -c is awkward
    # because the worker is a file. Instead: pass a wildcard log path that
    # the worker interpolates.
    env["AUDIT_LOG_PATH"] = str(out_dir / "rank_${OMPI_COMM_WORLD_RANK}.jsonl")
    # OpenMPI sets OMPI_COMM_WORLD_RANK; set the env var here so it's
    # available pre-MPI-init (it's set by orted before exec).

    # Allow per-mode mpirun extras (e.g., enable_recovery experiment).
    extra_mpirun_args = []
    if mode.startswith("recover_"):
        extra_mpirun_args = [
            "--mca",
            "orte_enable_recovery",
            "1",
        ]
    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",  # in case 4 ranks > available hw cores
        *extra_mpirun_args,
        "-np",
        str(n_ranks),
        "--mca",
        "btl",
        "^openib",  # avoid IB warnings on a single node
        # Keep mpirun's child-watching behavior at default so we observe
        # what TRT-LLM users would see in production.
        "-x",
        "AUDIT_DEATH_MODE",
        "-x",
        "AUDIT_VICTIM_RANK",
        "-x",
        "AUDIT_VICTIM_ITER",
        "-x",
        "AUDIT_LOG_DIR",
        "-x",
        "AUDIT_LOG_PATH",
        sys.executable,
        # Tiny shim: substitute the rank into AUDIT_LOG_PATH at process
        # start. We do it via a small inline preamble that runs before the
        # worker imports mpi4py.
        "-c",
        (
            "import os, sys, runpy; "
            "os.environ['AUDIT_LOG_PATH'] = "
            "  os.environ['AUDIT_LOG_DIR'] + '/rank_' + os.environ.get('OMPI_COMM_WORLD_RANK', '?') + '.jsonl'; "
            f"runpy.run_path({str(WORKER)!r}, run_name='__main__')"
        ),
    ]

    t_start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=total_timeout_sec
        )
        rc = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc = -1

        # On some Python versions, e.stdout/stderr are bytes even with text=True.
        def _to_str(x):
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return x

        stdout = _to_str(e.stdout)
        stderr = _to_str(e.stderr) + f"\n[launcher: timeout after {total_timeout_sec}s]"
        timed_out = True
        # Make sure orphan mpirun/worker processes don't linger.
        os.system("pkill -9 -f mpi_signal_handler 2>/dev/null; pkill -9 -f mpirun 2>/dev/null")
    elapsed = time.monotonic() - t_start

    # Aggregate per-rank logs
    per_rank = {}
    for log_file in sorted(out_dir.glob("rank_*.jsonl")):
        rank = int(log_file.stem.split("_")[1])
        events = []
        for line in log_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        per_rank[rank] = events

    # Per-rank summary
    rank_summary = {}
    for r in sorted(per_rank.keys()):
        evs = per_rank[r]
        s = {"rank": r}
        max_iter = -1
        for ev in evs:
            if ev.get("event") == "allreduce_ok":
                max_iter = max(max_iter, ev["it"])
            if ev.get("event") == "victim_about_to_die":
                s["victim_about_to_die"] = True
            if ev.get("event") == "loop_done":
                s["loop_done"] = ev["iters_completed"]
            if ev.get("event") == "signal_caught":
                s.setdefault("signals_caught", []).append(ev["name"])
            if ev.get("event") == "allreduce_failed":
                s["allreduce_failed_at"] = ev["it"]
                s["exc_type"] = ev["exc_type"]
        s["max_completed_iter"] = max_iter
        rank_summary[r] = s

    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "launcher_returncode": rc,
        "launcher_timed_out": timed_out,
        "per_rank": rank_summary,
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="/tmp/audit-1a-prototypes/mpi_signal_handler")
    p.add_argument("--n-ranks", type=int, default=4)
    p.add_argument("--victim-rank", type=int, default=2)
    p.add_argument("--victim-iter", type=int, default=3)
    p.add_argument("--per-mode-timeout-sec", type=int, default=20)
    p.add_argument("--modes", nargs="+", default=["default", "exit2", "abort"])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    all_runs = {}
    for mode in args.modes:
        print(f"\n==== Mode: {mode} ====")
        run = run_one(
            mode,
            results_dir,
            args.n_ranks,
            args.victim_rank,
            args.victim_iter,
            args.per_mode_timeout_sec,
        )
        all_runs[mode] = run
        print(
            f"  launcher_returncode={run['launcher_returncode']}, "
            f"timed_out={run['launcher_timed_out']}, "
            f"elapsed={run['elapsed_s']:.1f}s"
        )
        for r, s in run["per_rank"].items():
            print(
                f"  rank {r}: max_iter={s['max_completed_iter']} "
                f"loop_done={s.get('loop_done')} "
                f"signals={s.get('signals_caught')}"
            )

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_runs, indent=2))
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
