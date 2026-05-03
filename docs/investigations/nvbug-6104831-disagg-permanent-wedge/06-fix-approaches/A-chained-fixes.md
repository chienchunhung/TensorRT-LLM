# Approach A — Chained Signature Fixes

The forensic stack developed from the individual signatures during the
investigation, one chained PR per discovered signature.

---

## What it contains

PRs (each pair: reproducer test → fix, or combined test+fix):

| Sig | Test PR | Fix PR | Notes |
|---|---|---|---|
| `#1` | [#13639](https://github.com/NVIDIA/TensorRT-LLM/pull/13639) | [#13640](https://github.com/NVIDIA/TensorRT-LLM/pull/13640) | Chained on `#13639` |
| `#2` | [#13571](https://github.com/NVIDIA/TensorRT-LLM/pull/13571) | [#13572](https://github.com/NVIDIA/TensorRT-LLM/pull/13572) | Chained on `#13571`. Independent of disagg networking. |
| `#4` | [#13674](https://github.com/NVIDIA/TensorRT-LLM/pull/13674) | [#13671](https://github.com/NVIDIA/TensorRT-LLM/pull/13671) | `#13671` carries both test and fix as 2 commits |
| `#5` | combined into fix PR | [#13672](https://github.com/NVIDIA/TensorRT-LLM/pull/13672) | |
| `#6` | combined into fix PR | [#13673](https://github.com/NVIDIA/TensorRT-LLM/pull/13673) | Chained on `#13640` (the `#1` fix is a prerequisite) |

What each fix does in one line:

- `#13640` — sets a structured `kNETWORK_ERROR` exception on the
  sender promise before erasing the cancelled-after-ready entry from
  `mReadyResponses`.
- `#13572` — resets the child's `mPrevNode` in `clearNode()` before
  erasing it from the parent's `mNextNodes` (eviction-driven trie
  invariant).
- `#13671` — adds a `wait_for(0)` non-blocking probe to
  `CacheTransceiver::checkGenTransferStatus` so that selected-but-unready
  futures are skipped rather than blocked on.
- `#13672` — extracts the queued promise under the lock in
  `CacheReceiver::Impl::cancelRequest` and fulfills it with a structured
  exception once released.
- `#13673` — wraps `sendRequestInfo()` body in `try/catch` and adds
  explicit `freeBufferIndexForRecv()` calls in the `requestSync()`
  `!isReady` early-return path.

---

## What it covers (`L1`–`L8`)

| Layer | Coverage | Mechanism |
|---|---|---|
| **L1** sig `#1` | ✓ | `#13640` |
| **L1** sig `#5` | ✓ | `#13672` |
| **L2** request lifetime / UAF | ✗ | not addressed |
| **L3** in-process cancellation primitive | ✗ | not addressed |
| **L4** `checkGenTransferStatus` blocking | ✓ | `#13671` |
| **L5** recv-buffer slot leak | ✓ | `#13673` (try/catch) |
| **L6** NIXL backend handle release | ✗ | not addressed |
| **L7** eval-order regression | n/a | (no L2 fix → no L7 hazard) |
| **L8** Python scheduler idempotency | partial | only observable if other layers don't crash first |

---

## What it leaves uncovered

L2, L3, L6 are entirely uncovered. After the chained signature fixes
are individually applied:

- L2 means `Response::mRequest` and `RequestAndPromise::mRequest` stay
  raw `LlmRequest*`. Python's `_terminate_request` can free the
  underlying request while the C++ async send / receive worker is
  still dereferencing it. UAF surface remains open.
- L3 means `cancelRequest` returns `false` on in-flight requests. The
  log line `Cannot cancel request <id>` accumulates under contention
  and wedged transfers pile up.
- L6 means cancelled-mid-transfer NIXL handles stay registered until
  the underlying transfer completes naturally. Stranded handles
  accumulate; the deadlock variant of sig `#7` becomes more likely.

---

## Empirical result

The chained signature fixes prove out the individual root causes and
expose the residual sender / cleanup class. After applying them, the
local 1P1D UCX stress harness still reaches:

- **`run9`** — Python-`getattr` SIGSEGV at iter 92 of the burst, with
  the sig `#1` fix path captured cleanly in the promise-trace log
  immediately before. The wedge in earlier runs was masking it.
- **`rc11_ucx_run1`** — `pthread_mutex_lock` wedge in
  `CacheSender::Impl::response()` on the direct UCX backend (with no
  `libnixl.so` loaded in the process at all). This is the run that
  falsified the "NIXL plugin internal" classification of sig `#7`.

Both are sig `#7` manifestations, predicted by the L2 / L3 / L6 gaps.

---

## When to use this approach

- **As individual PRs landing into `main`** — yes, every chained PR is
  worth landing. They each fix a real bug with a focused unit test, the
  test scaffolding is reusable in any other approach, and they don't
  conflict with `#13056` or `#13495`.
- **As the sole fix for the customer wedge** — no. L2, L3, and L6 will
  still be open and the wedge will still reproduce.

The right framing for approach A is "the necessary regression tests
plus a partial fix." The fix portion gets superseded by approach B, C,
or D; the regression tests survive any landing path and should be
kept.

---

## Strengths

- Smallest individual PRs, easiest to review.
- Each fix has an explicit failing-then-passing unit test.
- Fully orthogonal to `main` — no merge conflicts with anything else.
- Works incrementally: any subset of the chained PRs landing reduces
  the wedge frequency without requiring the whole stack.

## Weaknesses

- Doesn't close the architectural lifetime / cancellation /
  backend-release gaps that underlie sig `#7`.
- "Fix one bug, find the next" pattern repeats — Phases 5, 7, 10 of
  the timeline document this exact discovery cycle.
- Regression risk under the customer load shape because sig `#7`
  variants still fire.

---

## What to read next

- For the fully-architectural alternatives, see
  [`B-pr13056.md`](B-pr13056.md) and [`C-pr13495.md`](C-pr13495.md).
- For the recommended composition, see [`D-combo.md`](D-combo.md).
- For the side-by-side comparison framework, return to
  [`README.md`](README.md).
