# 07 — Architectural Reflections

This file is the long-term reading on the bug class. Two parts:

1. **Why so many bugs were latent until now** — the seven invariants the
   transceiver doesn't enforce, and how they explain the cluster of
   failures documented in the rest of this report.
2. **What we would do differently** — process retrospective on whether
   this could have been investigated faster, and what to change for the
   next investigation of this shape.

For the customer-facing bug summary read
[`02-failure-signatures.md`](02-failure-signatures.md). For the actionable
fix path read [`06-fix-approaches/README.md`](06-fix-approaches/README.md).
This file is for the conversation *after* the fix lands: how to prevent
the next instance of this bug class.

---

## Why so many bugs were latent until now

### Three converging conditions

Three things converged. None individually new, but together they
exercise a part of the disaggregated transceiver that prior workloads
never reached in volume:

1. **The subsystem is young.** Disaggregated serving is still flagged
   "experimental" in the docs. NIXL is newer still. The transceiver was
   built layer-by-layer (UCX → NIXL → cache-aware formatters → buffer
   pool manager) with each layer adding its own thread, queue, future,
   and condition-variable wait. The combined contract across the
   layers was never formalised.
2. **The customer load shape exercises the cleanup paths, not the
   happy path.** Long prompts + high concurrency + client-side cancels
   + retries means almost every request can hit an abort, timeout, or
   eviction mid-transfer. That is the surface where every signature in
   this investigation lives. Most prior workloads — short prompts, low
   concurrency, no aggressive timeouts — never reach the cleanup paths
   in volume, so the bugs sat dormant.
3. **The test pyramid is shaped wrong for this surface.** Each
   subsystem has unit tests for happy-path completion. End-to-end
   disaggregated integration tests use short prompts, low concurrency,
   and no cancellations. There is essentially no test that drives the
   combination "cancel during transfer at scale", which is the single
   load shape every signature here requires.

That alone explains the "many latent bugs surface in two weeks"
pattern. But it does not explain why the bugs cluster so tightly on
the same handful of code paths. That part is design.

### The seven invariants the transceiver doesn't enforce

Every signature in this investigation can be re-described as a
violation of one of seven contracts that the transceiver doesn't
actually have an explicit enforcement point for. Each is a missing
invariant, not a bug — the bugs are individual instances; the
invariant gaps are the architecture.

1. **Ownership across the C++ ↔ Python boundary.** `mSenderFutures`
   and `mRequesterFutures` hold raw `LlmRequest*` while Python (with
   `shared_ptr<LlmRequest>` semantics) decides when the underlying
   `LlmRequest` dies. That is a guaranteed use-after-free surface —
   Python only has to terminate a request mid-transfer once. The right
   architectural answer is `shared_ptr` all the way through; the fact
   that raw pointers ever crossed a language-managed lifetime boundary
   is the smell. **None of the six signatures here is the UAF, but
   every single fix here lives next to one.** Maps to L2 in the
   defect class stack.
2. **Every promise must be fulfilled exactly once before destruction.**
   Signatures `#1` and `#5` are the same architectural omission on
   opposite sides: a code path erases a `(request, promise)` entry
   without first calling `set_value` or `set_exception`. There is no
   central invariant, no lint, no destructor that defaults to
   `set_exception(unfulfilled)`. Every new cleanup path is a fresh
   chance to forget. Maps to L1.
3. **Every blocking wait must be interruptible.** The
   `BaseTransBufferManager::assignBufferIndex()` `cv.wait`, the
   gen-side `checkGenTransferStatus()` unconditional `future.get()`,
   the ready-signal recv, and the underlying NIXL/UCX waits all
   blocked unboundedly with no cancel-flag awareness. Signatures `#4`
   and `#6` live exactly here. There is no cross-cutting "all blocking
   calls take a cancel token / a deadline / a `mTerminate` check"
   rule. Maps to L3 + L4.
4. **Every acquired resource must release on every exit path (RAII).**
   The recv-buffer pool slots had at least three exit paths from
   `requestSync()` and only the happy one (success → `unformat()`)
   released. The Layer-A and Layer-B fix for signature `#6` is a
   textbook RAII fix; the question is why the original code did manual
   `assignBufferIndex` / `freeBufferIndex` pairing instead of writing
   the holder on day one. Maps to L5.
5. **Same operation, same semantics across language layers.**
   Signature `#4` — the C++ `checkGenTransferStatus(atLeastNum=1)`
   blocks while the Python `transceiver.py` wrapper for the same
   operation skips unready entries — is a pure contract divergence.
   Two implementations of one conceptual operation drifted; nothing
   checks they agree.
6. **A configuration knob without an enforcement point is debt.**
   `kv_transfer_timeout_ms` was plumbed all the way through config and
   was never enforced as a hard deadline for the C++ blocking calls.
   Signature `#6` would have surfaced as a per-request error long
   before it became a global wedge if the receiver-side `cv.wait` had
   honored that knob. Symptomatic of feature-on-feature growth without
   a designated enforcement layer for newly-added knobs. Maps to L6.
7. **Long-lived worker loops must be robust to any escape.** The
   receiver drain worker uses `catch (std::exception)` but no
   `catch (...)`. A non-`std` throw from NIXL or UCX strands the queue
   and silently kills the worker thread, which then looks identical
   to signature `#6` from outside. The investigation didn't end up
   needing this fix, but the same "no rule" pattern is the reason it
   exists.

The L7 (eval-order) and L8 (Python idempotency) defect classes from
[`03-defect-class-stack.md`](03-defect-class-stack.md) are not on the
seven-invariants list because they are *consequences of fixing* the
above invariants rather than separate architectural omissions. L7 is
"once L2's `shared_ptr<LlmRequest>` is in place, the existing
`handleAsyncSend` callsite is argument-evaluation-order-unsafe" — an
auditable implication. L8 is "the scheduler can re-present the same
request many times" — a known property the disagg paths happened to
not handle idempotently.

### How to read these as a class

It is more accurate to think of the transceiver as a textbook example
of **inherited concurrency complexity without a unifying async
contract** than as "fundamentally bad design". This is a depressingly
common pattern in performance-focused C++ async code — not unique to
TRT-LLM. The transceiver works under the happy path because each
subsystem is individually correct. It breaks under
cancel/timeout/exception paths because there is no shared notion of:

- what it means to cancel a request mid-flight,
- who owns an in-flight request's lifetime,
- when a promise gets fulfilled,
- where a blocking wait checks for shutdown / cancel / deadline,
- which exits must release which resources, and
- how errors propagate from C++ back to Python.

In a more mature subsystem you would expect to see a single
`TransferSession`-like type that bundles request lifetime + cancel
token + buffer holders + promise + timeout into one RAII-managed
object, with every send/receive path expressed as a method on it. The
fixes in this investigation are incrementally bending the code in
that direction (PR `#13495`'s `TransferSession` is one explicit step;
PR `#13056`'s per-request cancel-flag + RAII `BufferIndexHolder` are
others), but they are a retrofit rather than a clean redesign.

### Why code review didn't catch any of this

Honestly: because the review surface for "you forgot to fulfill a
promise on this cleanup path" or "this `cv.wait` isn't cancellable" is
invisible without the contracts written down. A reviewer looking at a
50-line PR adding a new cleanup branch has no way to spot that it
violates an unwritten invariant the rest of the file follows by
accident. This is exactly the failure mode that systematic invariants
(or strong type-level abstractions) are supposed to prevent — and the
transceiver currently has neither.

### What this implies for follow-up work

The actual remediation, in order of long-term value:

1. **Document the seven invariants above** in the
   disaggregated-serving developer guide, with a one-paragraph "if
   you're adding a new transfer path, here is the checklist" section.
   Cheap, high leverage, prevents the next field hit.
2. **Introduce a `TransferSession`-like abstraction** that is the only
   blessed way to start a disagg KV transfer, with the seven
   invariants baked into its type. Reviewers can then enforce by type,
   not by discipline. PR `#13495` is one step in this direction.
3. **Add an integration test** specifically for the
   cancel-during-transfer surface: long prompts, high concurrency,
   aggressive client-side timeouts, retries. The single load shape
   that exercises every signature documented here, and the absence of
   such a test is the single biggest reason this bug class went
   undetected for so long.
4. **Audit other early-return paths** in the C++ disagg transceiver
   for leaks of similarly cv-waited resources. The signature `#6`
   pattern — "fix the visible failure path on side A, surface a
   resource leak on side B" — is likely to repeat if other paths
   share the same RAII gap.

The point is that the next contributor adding a new transfer mode is
one cleanup path away from re-introducing the same class of bug if
these invariants stay implicit. A short architectural note that names
the seven contracts above would pay for itself in one prevented field
hit.

---

## What we would do differently — retrospective

The investigation was sequential by necessity: one signature surfaced,
got a fix, and the next signature emerged from the post-fix behaviour.
Eight rounds of "find bug → fix bug → find next bug" took ~8 days of
calendar time. With the end-to-end view in hand: *could we have done
this differently from the start?* The honest answer has two parts.

### What was not actually possible at T0

A "design one comprehensive fix" approach is the natural counterfactual.
At T0, introduce a `TransferSession`-like abstraction that encapsulates
request lifetime + RAII buffer holders + promise-fulfillment-on-destruct
+ deadline, and let it close all six TRT-LLM signatures in one PR. This
sounds clean in retrospect but had two hard blockers:

1. **You cannot design an abstraction to fix bugs you have not found
   yet.** We knew about three signatures from the field at T0
   (`#1`, `#2`, `#3`). The other four (`#4`, `#5`, `#6`, `#7`)
   emerged from investigation. The architectural answer ("what
   invariants does this abstraction enforce?") is the *output* of
   finding the bugs, not the *input*. A `TransferSession` designed at
   T0 against only `#1`/`#2`/`#3` would not have prevented `#5` or
   `#6`, because we would not have known to enforce the invariants
   those signatures violate.
2. **The field was wedged.** Customers needed the smallest patches
   that work, not a multi-thousand-line refactor of a critical path.
   Review pressure on TRT-LLM also favours small focused changes; a
   refactor of `dataTransceiver.cpp` that touched all three backends
   (UCX/NIXL/MPI) would not have landed quickly.

A single coordinated fix was therefore strictly impossible given the
information state at T0. What we *should* have done is structurally
different: change the **meta-process** so each subsequent bug
discovery would have been faster, and so cascade relationships would
have been caught at design time instead of after a build-and-rerun
cycle.

### What we should have done first (in priority order)

#### 1. Add deadline enforcement as PR #0

The single highest-leverage change. The `kv_transfer_timeout_ms` knob
is already plumbed through Python config, the C++ config class,
serialization, and getters/setters — it is just never consumed in the
request execution path. Even Layer A alone (the ~1-week Python-level
deadline; see
[`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md)) would
have:

- **Converted every cleanup-path bug from a *silent wedge* into a
  *per-request error*.** Signatures `#1`, `#5`, `#6`, and `#7` all
  surface as `kNETWORK_ERROR` 5xx responses with a real exception
  message instead of the deployment going dark.
- **Given each subsequent bug an attributable failure point**
  (`request 4113 timed out in checkGenTransferStatus`) instead of
  requiring `py-spy` / `gdb` post-mortem.
- **Given orchestration a real signal** so customers' production
  wedges self-heal via pod restart while we work on root causes.
  Field urgency drops from P0 to P2.

The investigation would have shifted from "*the deployment is wedged,
dump stacks, find the wedged thread*" to "*these 14 requests timed
out, here are their lifecycle traces*". Phase 5 → Phase 10 of the
timeline (currently ~4 days) would plausibly have collapsed to ~2
days.

The deadline enforcement is also retrospectively justified by the
investigation itself: signature `#7` is the only signature TRT-LLM
cannot fix at the source until the exact mutex is identified, and the
deadline is the *only* TRT-LLM-side defence against it that converts
the wedge into a recoverable error. We would have built this layer
eventually anyway. Building it first makes everything else trivially
debuggable.

#### 2. Write down the seven invariants first, fix against them

If the seven contracts above had been written down at T0 — even just
as a paragraph in the disaggregated-serving developer guide — three
concrete cascade relationships would have been caught at review time
instead of after days of reproduction:

- **`#5` would have been caught at `#1`'s PR review.** "This is the
  sender-side fix for the missing-promise-fulfillment invariant; the
  receiver-side mirror is structurally identical — fix both at once."
  Two PRs collapse into one.
- **`#6` would have been caught at `#1`'s PR review.** "The new
  `!isReady` path on the receiver — under the 'every acquired
  resource must release on every exit path' invariant, does it
  release every resource the success path releases?" Type 1 cascade
  prevented at design time, not after a 2-day reproduction cycle.
- **`#4` would have been visible to any code search for
  unconditional `future.get()`.** Under the "every blocking wait
  must be interruptible" invariant it is a bug regardless of whether
  it currently fires; a sweep against the invariants would have
  caught it as a latent issue before the field hit it.

The cost is a single document edit. The benefit is preventing the
entire Type 1 cascade and most of the Type 2 cascades documented in
[`03-defect-class-stack.md`](03-defect-class-stack.md).

#### 3. Add the cancel-during-transfer integration test first

The single largest test-coverage gap surfaced by this investigation is
the cancel-during-transfer load shape (long prompts, high concurrency,
aggressive client timeouts, retries). If that test had existed at T0,
all six TRT-LLM signatures would have been visible **as test failures
in CI** instead of as a customer field hit. Even signature `#3`
(currently field-only) would plausibly have been reproducible. The
investigation would not have needed external infrastructure (Dynamo,
mpi4py worker dumps, NIXL trace correlation, `py-spy` / `gdb`
post-mortem).

The cost is moderate (a few hundred lines of integration test
infrastructure plus a CI lane to run it). The ongoing benefit is huge:
every future PR that touches the disaggregated path is gated against
this load shape.

### What this implies for next time

The cleaner approach is not a different *fix*; it is a different
**order of operations**:

```text
What we did:
    field hit → reproduce → find bug N → fix bug N → repeat 7 times
        ↓
    ~8 days, 6 cascading PRs, sig #7 discovered last as a surprise

What we should have done:
    field hit → containment layer (deadline enforcement, ~1 week)
              → integration test for the load shape (~few days)
              → write down the seven invariants (~hours)
              → bugs become CI-visible and individually attributable
              → fix them in any order, each PR reviewable against invariants
              → Type 1 cascade caught at review time, not after rebuild + rerun
        ↓
    same 7 signatures, identified in parallel from one CI run,
    fixed individually but with no cascade surprises
```

The key insight is that the bottleneck of the investigation was
**observability and attribution**, not fix complexity. Each fix
individually is small (`#1` is ~5 lines, `#4` is ~17 lines, `#5` is
~20 lines, `#6` is the largest at ~80 lines including the RAII
helper). What ate the calendar time was not writing the fixes — it
was figuring out what was wedged, why, and which fix to write next.
The three meta-process changes above all attack that bottleneck
directly.

### What this section is *not* arguing

A few clarifications to avoid over-reading the retrospective:

- **It is not arguing for a `TransferSession` rewrite as PR #0.** That
  refactor is still the right long-term direction (item 2 in the
  follow-up work above), but it is a separate, larger project that
  should follow the per-signature fixes once the contracts are
  stable, not replace them.
- **It is not arguing that one PR could have fixed all seven
  signatures.** Six of them are real, distinct bugs in different
  functions, and `#7` is a TRT-LLM-side bug class with at least four
  manifestations. They genuinely need separate fixes. The argument is
  about how *quickly* they would have been found and how *cleanly*
  they would have been reviewed, not about collapsing them into a
  single patch.
- **It is not arguing that the sequential discovery was avoidable in
  absolute terms.** It was avoidable *given the meta-process changes
  above*, but not avoidable given the meta-process we actually had.
  The retrospective is about what we should change for the *next*
  investigation of this shape, not about whether this one could have
  been done differently after T0 with the same tooling.

---

## What to read next

- For the actionable fix path, see
  [`06-fix-approaches/README.md`](06-fix-approaches/README.md).
- For outstanding work and the deadline-enforcement effort estimate,
  see [`08-next-steps-and-pr-map.md`](08-next-steps-and-pr-map.md).
