# 04 — Reproduction

How to reproduce the wedge locally with `trtllm-serve` 1P1D and what
specifically about the load shape causes it.

---

## Topology

- 1 context worker (`trtllm-serve serve --server_role context --port 8001`)
- 1 generation worker (`trtllm-serve serve --server_role generation --port 8002`)
- 1 disaggregated front-end (`trtllm-serve disaggregated --port 8000`)
- All three colocated on a single node, two GPUs (`CUDA_VISIBLE_DEVICES=0`
  for context, `=1` for generation).
- Backend: PyTorch (`--backend pytorch`).
- Model: `Qwen/Qwen3-0.6B` (small enough to bring up quickly; the wedge
  pattern reproduces independent of model size).
- Transceiver: NIXL over UCX with TCP-only transport (the customer
  configuration). Relevant env:
  ```sh
  TRTLLM_USE_UCX_KVCACHE=1
  TRTLLM_NIXL_KVCACHE_BACKEND=UCX
  TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM=1
  UCX_TLS=tcp,cuda_copy,self
  ```
- Trace gating envs (set on workers and front-end while reproducing):
  ```sh
  TRTLLM_DISAGG_TRACE_PROMISE=1
  TRTLLM_DISAGG_TRACE_TRIE=1
  TRTLLM_DISAGG_TRACE_OPTIONAL=1
  TRTLLM_DISAGG_TRACE_BLOCK=1
  TRTLLM_DISAGG_TRACE_BLOCK_TIMEOUT_S=5
  ```

---

## Client load shape

The minimal harness that reliably reproduces the wedge is the
"long-prompt burst + recovery probes" script preserved in the local
disagg-repro worktree (under `.repro/harness/onepair/`).

Key parameters:

- `CONC=16` — concurrent in-flight requests during the burst.
- `BURST_DUR_S=60` — burst duration.
- Prompt length sampled from `gauss(8000, 2000)` tokens.
- `max_tokens=200`, `min_tokens=150`, `temperature=0`.

After the burst, the harness fires sanity probes at `+30s`, `+60s`,
`+90s`, `+120s`, and `+180s` of idle. If any probe returns `ok200`, the
system has recovered; if all probes time out, it is a permanent wedge.

---

## Expected stock-`rc11` outcome (the "before" state)

```text
[ 0.0s] CONC=16 SANITY PROBE
[PROBE-PRE] result=ok200 wall=8.8s
[BURST-1 90.0s] done ok200=8 errors=12 total=20
[PROBE-T+30] result=exc:ReadTimeout wall=60.1s
[PROBE-T+60] result=exc:ReadTimeout wall=60.1s
[PROBE-T+90] result=exc:ReadTimeout wall=60.1s
[PROBE-T+120] result=exc:ReadTimeout wall=60.1s
[PROBE-T+180] result=exc:ReadTimeout wall=60.1s
NO RECOVERY after 180s idle -- permanent wedge
```

This was confirmed both on stock `rc11` (run 4) and after the signature
`#4` fix in isolation (run 5). The system never recovers without
process restart.

---

## Configurations that did *not* reproduce

These narrow down what about the load shape matters:

- **1P1D with very short prompts (≤256 tokens)** — no wedge.
- **1P1D with overlap disabled** — no wedge.
- **1P1D with no client-side timeouts (no cancels)** — no wedge.
- **Single-process unit tests of the cache transceiver alone** (without
  the disagg HTTP layer) — reproduce signatures `#1`, `#2`, `#4`, `#5`,
  and `#6` (the latter two via the new tests added in `#13672` and
  `#13673` respectively). Signatures `#3` and `#7` are field-only:
  `#3` requires the full HTTP path with cancellation and retries, and
  `#7` requires the NIXL/UCX runtime under a contention pattern that
  hasn't been mock-injected yet.

The minimum trigger set is therefore:

- long prompts (~8K tokens),
- high concurrency (`CONC ≥ 16`),
- aggressive client-side timeouts that cause cancels and retries,
- a real HTTP path with NIXL or direct UCX transport,
- and overlap scheduling enabled.

Drop any one of these and the wedge typically does not reproduce.

---

## Run archive index

The investigation maintains a numbered series of runs at
`~/disagg-investigation-archive/` (and `.repro/logs/` in the repro
worktree for the most recent ones). Each archive contains the full
gen.log, ctx.log, harness output, and where applicable `pyspy` /
`gdb` post-mortems.

| Run | Configuration | Outcome | What it proved |
|---|---|---|---|
| `run4` | stock `rc11` | permanent wedge | baseline reproduction |
| `run5` | post-sig-`#4`-fix only | permanent wedge | sig `#4` is necessary but not sufficient; surfaces sig `#5` and `#6` |
| `run6` | + sig `#5` fix + first-round sig `#6` instrumentation | permanent wedge | localizes sig `#6` to one in-progress request stuck after `gen_request_sync_begin` |
| `run7` | fine-grained sig `#6` instrumentation across `sendRequestInfo` body | permanent wedge | pinpoints stall to first `assignBufferIndexForRecv` after a leak |
| `run8` | + sig `#6` Layer-A + Layer-B fix | permanent wedge | sig `#6` confirmed fixed; ctx-side `pthread_mutex_lock` wedge in `recvRequestInfo` surfaces (originally classified as NIXL bug, later reclassified as sig `#7`) |
| `pr13056_run1` | independent: PR `#13056` (NIXL backend) | permanent wedge with same `pthread_mutex_lock` frame | sig `#7` reproduces independently of our fix stack |
| `rc11_ucx_run1` | our fixes + direct UCX | permanent wedge with same frame, no NIXL plugin loaded | falsifies "NIXL plugin internal" classification of sig `#7`; reclassifies as TRT-LLM `CacheSender::Impl` bug |
| `rc11_ucx_run2_diag` | our fixes + direct UCX with `gdb` capture | ctx mpi4py worker exits | sig `#7` Variant B |
| `run9` | rc11 + our fixes + direct UCX | Python-`getattr` SIGSEGV at iter 92 | sig `#7` Variant C |
| `run10` | rc11 + PR `#13056` + direct UCX | first-request `handleAsyncSend` SIGSEGV | sig `#7` Variant D |
| `run14` | PR `#13056` + direct UCX with async-send instrumentation | confirmed eval-order hazard | falsifies null-`shared_ptr` hypothesis; pinpoints the L7 bug |
| `run14c` | + eval-order fix | `CONC=4` recovery; one `CONC=16` recovery | first run that recovers post-burst on direct UCX |
| direct-UCX `CONC=24` 60s/90s, 5 iter | combo (D) | 5/5 recovered | combo first stack to cleanly recover at moderate concurrency |
| direct-UCX `CONC=32` 90s, 5 iter | combo (D) | 5/5 recovered | direct-UCX recovery boundary at `CONC=32` |
| direct-UCX `CONC=64` 90s | combo (D) | wedged | direct-UCX still has remaining backlog/timeout interaction at high concurrency |
| NIXL+UCX-plugin `CONC=32` 90s, 5 iter | combo (D), NIXL transport | 5/5 recovered, zero burst-time errors | NIXL path validation |
| NIXL+UCX-plugin `CONC=64` 90s, 5 iter | combo (D), NIXL transport | 5/5 recovered, ~zero burst-time errors | the customer's transport path is clean through `CONC=64` |

Reading this table, the operational story is: combo (approach D) is the
first stack that recovers cleanly on the customer's transport (NIXL +
UCX plugin) up through `CONC=64`. Direct-UCX still has a remaining
high-concurrency wedge at `CONC=64` that needs an additional fix in
TRT-LLM's direct-UCX cancellation path (see
[`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md)).

---

## What to read next

- For the chronological story of how each run informed the next, see
  [`05-investigation-timeline.md`](05-investigation-timeline.md).
- For the candidate fix stacks evaluated against the L1–L8 framework,
  see [`06-fix-approaches/README.md`](06-fix-approaches/README.md).
