# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MPI signal handler / `_exit` mitigation prototype for WideEP FT Audit 1.

Question (per design §3 "Mode A" + §5.4 / PR 1d.0):
    When one MPI rank dies abnormally, does it:
      A) Take down all surviving ranks via MPI_Abort propagation?
      B) Exit alone, letting survivors continue MPI operations on a
         smaller communicator?

The default in TRT-LLM today (mpiUtils.cpp lines 195–215, plus mpi4py's
atexit hook) is (A). PR 1d.0 proposes a custom signal handler that calls
`_exit(2)` to get behavior (B), so that downstream failure-detection and
rank-masking can do their jobs.

Test rig
--------
4 MPI ranks (`mpirun -np 4`). Each runs a small loop of MPI_Allreduce.
The victim (rank 2) "dies" at iter 3 in one of three ways selected by
`AUDIT_DEATH_MODE`:

  default   - raise Python RuntimeError; let mpi4py / Python's default
              exit path run. mpi4py registers an atexit hook that calls
              MPI_Abort on uncaught exceptions; this is the unmitigated
              baseline.
  exit2     - call os._exit(2) directly. Skips Python finalizers and
              mpi4py's atexit hook entirely. The OS process exits with
              code 2; MPI_Abort is never called.
  abort     - call MPI.COMM_WORLD.Abort(2) explicitly (the worst case).

Survivors run until they either complete N=20 iters of the loop OR
detect a peer death via the iteration's allreduce hanging.

Outputs
-------
JSONL log per rank, plus a launcher-side aggregation. Key signals:
  - victim exit code (0 / 1 / 2 / -6 / -9)
  - survivor exit codes (0 = clean, anything else = killed)
  - survivor's last completed iter
  - whether mpirun killed the survivors after victim death
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

# Mode is set in env (so the launcher can vary it without code edits).
DEATH_MODE = os.environ.get("AUDIT_DEATH_MODE", "default")
LOG_PATH = Path(os.environ["AUDIT_LOG_PATH"])

# Install a SIGTERM / SIGINT handler BEFORE mpi4py is imported. mpi4py's
# initialize hook installs its own handlers; we want to know whether ours
# are still effective post-init or whether mpi4py overrides them.
import signal  # noqa: E402  (intentional non-top: must precede mpi4py import)

_SIGNALS_SEEN: list[int] = []


def _signal_handler(signum: int, frame) -> None:  # noqa: ARG001
    _SIGNALS_SEEN.append(signum)
    log({"event": "signal_caught", "signum": signum, "name": signal.Signals(signum).name})
    # Emulate the PR 1d.0 mitigation: exit cleanly without MPI_Abort.
    if DEATH_MODE == "exit2":
        os._exit(2)
    # else: let the default handler run (which may invoke MPI_Abort)


def log(event: dict) -> None:
    event["t"] = time.monotonic()
    event["pid"] = os.getpid()
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()


# Install our handlers BEFORE importing mpi4py, so we can see what mpi4py
# overrides.
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, _signal_handler)

from mpi4py import MPI  # noqa: E402

# Did mpi4py override our handlers?
post_init_handlers = {sig.name: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}


def main() -> int:
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    victim_rank = int(os.environ.get("AUDIT_VICTIM_RANK", "2"))
    victim_iter = int(os.environ.get("AUDIT_VICTIM_ITER", "3"))
    n_iters = int(os.environ.get("AUDIT_N_ITERS", "20"))

    log(
        {
            "event": "start",
            "rank": rank,
            "size": size,
            "victim_rank": victim_rank,
            "victim_iter": victim_iter,
            "death_mode": DEATH_MODE,
            "host": socket.gethostname(),
            "post_init_handlers": {k: str(v) for k, v in post_init_handlers.items()},
        }
    )

    import numpy as np

    sendbuf = np.full(64, float(rank), dtype=np.float64)
    recvbuf = np.zeros_like(sendbuf)

    for it in range(n_iters):
        try:
            MPI.COMM_WORLD.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
        except Exception as e:
            log(
                {
                    "event": "allreduce_failed",
                    "rank": rank,
                    "it": it,
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e)[:200],
                }
            )
            break

        if it < 3 or it % 5 == 0:
            log({"event": "allreduce_ok", "rank": rank, "it": it, "sum": float(recvbuf[0])})

        if rank == victim_rank and it == victim_iter:
            log({"event": "victim_about_to_die", "rank": rank, "it": it, "death_mode": DEATH_MODE})
            if DEATH_MODE == "default":
                # Raise a RuntimeError. mpi4py's atexit hook will catch the
                # uncaught exception and call MPI_Abort on COMM_WORLD.
                raise RuntimeError("victim raising RuntimeError")
            elif DEATH_MODE == "exit2":
                # Bypass Python finalizers + mpi4py atexit hook. Non-zero
                # exit code: mpirun treats this as abnormal termination.
                os._exit(2)
            elif DEATH_MODE == "exit0":
                # Same as exit2 but with success exit code, to test whether
                # mpirun's "child died, kill rest" propagation depends on
                # exit code or any exit at all.
                os._exit(0)
            elif DEATH_MODE == "abort":
                # The worst case: explicitly call MPI_Abort.
                MPI.COMM_WORLD.Abort(2)
            elif DEATH_MODE == "sigkill":
                # Realistic case: external SIGKILL (e.g., OOM-killer, hardware
                # failure). Skips all clean exit paths.
                import signal as _sig

                os.kill(os.getpid(), _sig.SIGKILL)
            else:
                raise RuntimeError(f"unknown DEATH_MODE: {DEATH_MODE}")

        time.sleep(0.05)

    log({"event": "loop_done", "rank": rank, "iters_completed": it + 1})
    return 0


if __name__ == "__main__":
    sys.exit(main())
