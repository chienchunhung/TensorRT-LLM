# Research-pass prototypes

Throw-away prototype scripts that produced the empirical numbers folded into
[Item 7 of the research-pass report](../redesign-research-pass-report.md#-empirical-follow-up--audit-1a-partial-item-7)
(Audit 1a Days 1–3). Kept here so the report's claims are reproducible on any
≥ 4-GPU node with comparable software (the hardware / software bill is in the
report).

These are **not** part of TRT-LLM proper — no API, no test harness integration,
no Pydantic, no production sanity. They exist to answer one focused question
each, log to JSONL, exit. Everything is single-file and self-contained.

## Scripts

| Script | Question answered | Hardware required | Software required |
|:---|:---|:---|:---|
| [`nccl_rebuild.py`](nccl_rebuild.py) | Does PyTorch's `torch.distributed` provide a recoverable failure path after a peer SIGKILL? What's the rebuild latency? | ≥ 4 GPUs, NCCL 2.x | PyTorch 2.x |
| [`mpi_signal_handler.py`](mpi_signal_handler.py) + [`mpi_signal_launcher.py`](mpi_signal_launcher.py) | When one MPI rank dies abnormally, does it take down the world via `MPI_Abort` propagation, or can survivors continue? Does `_exit(N)` in `mpiUtils.cpp` change the answer? | None (CPU-only OK) | OpenMPI + mpi4py |
| [`cumem_unmap_dead_peer.py`](cumem_unmap_dead_peer.py) | When the process that allocated a `cuMemCreate` region is SIGKILLed, what happens to a peer that has cross-process-imported it via posix-FD? Can it cleanly `cuMemUnmap` / `cuMemRelease` / `cuMemAddressFree`? | 1 GPU (CUDA 12+) | `cuda-python` |
| [`cumem_unmap_dead_peer_fabric.py`](cumem_unmap_dead_peer_fabric.py) | Same as above but with `CU_MEM_HANDLE_TYPE_FABRIC` (the actual MNNVL handle type). | 1 GPU + active `nvidia-imex` daemon (typically NVL72) | `cuda-python` |

## Reproduction

Default log directory is `/tmp/audit-1a-prototypes/<test_name>/` for every
script. Override with `--log-dir` (or `--results-dir` for the MPI launcher) if
you want results elsewhere.

```bash
# Day 1 — NCCL rebuild (single best run from the 6 in the report):
python3 nccl_rebuild.py --launch --world-size 4 --victim-rank 2 \
    --victim-iter 5 --watchdog-sec 5 --master-port 29508

# Day 2 — MPI signal handler / _exit mode sweep (7 modes):
python3 mpi_signal_launcher.py --per-mode-timeout-sec 45 \
    --modes default exit2 exit0 abort sigkill recover_exit2 recover_sigkill

# Day 3 — driver-side cuMemUnmap, posix-FD variant (passes on any CUDA 12+ node):
python3 cumem_unmap_dead_peer.py

# Day 3 — fabric-handle variant (gated on nvidia-imex; runs on NVL72-class hosts):
python3 cumem_unmap_dead_peer_fabric.py
```

Each script writes per-rank / per-process JSONL log lines, plus a summary
JSON for the launchers. To inspect a result:

```bash
cat /tmp/audit-1a-prototypes/nccl_rebuild/summary.json | python3 -m json.tool
cat /tmp/audit-1a-prototypes/mpi_signal_handler/summary.json | python3 -m json.tool
ls /tmp/audit-1a-prototypes/cumem_unmap_dead_peer/
```

## Sample results

Two representative output files from the original Audit 1a runs are
checked in under [`sample-results/`](sample-results/) so readers can see
the JSON / JSONL shape without having to re-run anything:

- [`mpi_signal_handler-summary.json`](sample-results/mpi_signal_handler-summary.json)
  — launcher summary from the last `mpi_signal_launcher.py` invocation
  (`recover_exit2` + `recover_sigkill` modes). Shows the schema and the
  "survivors stay alive but hang in collective" outcome with
  `--mca orte_enable_recovery 1`.
- [`cumem_unmap_dead_peer-peer.jsonl`](sample-results/cumem_unmap_dead_peer-peer.jsonl)
  — one-process event log from the Day 3 posix-FD test. Shows the full
  success path: receive FD → import → map → verify cross-process share
  pre-kill → wait 2 s → verify share survives kill → cuMemUnmap (0.25 ms)
  → cuMemRelease (1.27 ms) → cuMemAddressFree (0.008 ms).

## What's intentionally not here

- Full per-run JSONL logs from the original test host. Logs are tied to a
  specific machine + invocation timestamp, and re-running the script
  regenerates them.
- The intra-node MNNVL `MnnvlMemory` rebuild prototype (Days 4–5 in the §9.1
  plan). Gated on the `nvidia-imex` daemon being active; left to Audit 1b
  validation.
- DeepEP destructor exercise (Day 3 secondary). Blocked here on a stale
  `tensorrt_llm` package import; needs an NGC container with matching C++
  binary.
- NVSHMEM teardown test (Day 5). Needs `nvshmem` Python module.
