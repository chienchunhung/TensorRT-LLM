# 2.6 Speculative Decoding

[< Back to Overview](README.md)

## What It Is

Speculative decoding accelerates autoregressive generation by proposing multiple candidate tokens via a lightweight draft mechanism, then verifying them in parallel via a single target model forward pass. Matched tokens are accepted, reducing sequential forward passes.

## Why It Exists

Autoregressive decoding is inherently sequential — each token depends on the previous one. At low batch sizes, GPU utilization is poor because each forward pass processes only one token per sequence. Speculative decoding proposes K draft tokens and verifies them in one forward of K+1 positions, potentially generating K+1 tokens in ~2 forward passes.

## Supported Algorithms

```mermaid
graph TB
    subgraph "Speculative Decoding Algorithms"
        E3["EAGLE 3<br/>Separate draft model"]
        MTP["MTP<br/>Built-in model heads<br/>DeepSeek-specific"]
        NGram["NGram<br/>Pattern matching<br/>no draft model"]
        PARD["PARD<br/>Parallel mask prediction<br/>one-model + two-model"]
        SA["Suffix Automaton<br/>GPU pattern matching"]
        DT["Draft/Target<br/>Arbitrary smaller model"]
        UP["User-Provided<br/>Custom Drafter"]
    end

    subgraph "Combinable — SA Enhancement"
        SA_E["SA + EAGLE 3"]
        SA_M["SA + MTP"]
        SA_P["SA + PARD"]
    end

    SA --> SA_E
    SA --> SA_M
    SA --> SA_P
    E3 --> SA_E
    MTP --> SA_M
    PARD --> SA_P
```

| Algorithm | Draft Source | Draft Model Required? | Key Characteristics |
|:----------|:-----------|:---------------------|:--------------------|
| **EAGLE 3** | Lightweight trained model | Yes | Two-model or one-model; best with SA combination; MLA target + GQA draft support |
| **MTP** | Built-in prediction heads | No (embedded) | DeepSeek-specific; relaxed acceptance for reasoning; MTP>1 for DeepSeek v3.2 |
| **NGram** | Prompt/generation history | No | Prompt lookup decoding; zero extra model overhead |
| **PARD** | Parallel mask-token prediction | Yes | All K drafts in one forward; one-model + two-model paths; target-independent |
| **SA** | GPU suffix automaton | No | Model-free; very accurate on repetitive content; on-device processing |
| **Draft/Target** | Arbitrary smaller model | Yes | Simplest form; requires same tokenizer |

## Draft-Verify Loop

```mermaid
sequenceDiagram
    participant D as Drafter
    participant T as Target Model
    participant V as Verifier

    Note over D,V: Step N
    D->>D: Generate K draft tokens [d1, d2, ..., dk]
    D->>T: Forward with [input, d1, ..., dk] — K+1 positions
    T-->>V: Logits for all K+1 positions
    V->>V: Sample target tokens [t1, t2, ..., tk+1]
    V->>V: Compare — find longest prefix where di == ti

    alt All K drafts accepted
        V-->>D: Accept all K+1 tokens — K drafts + 1 bonus
    else M less than K drafts accepted
        V-->>D: Accept M+1 tokens — M drafts + 1 correction
    end
```

**What's new (v1.2-v1.3):**
- **PARD one-model path** — single-model speculative decoding without a separate draft model.
- **Dynamic draft length** across all spec decode algorithms (expanding from one-model path).
- **MTP>1** for DeepSeek v3.2 — multiple prediction heads for higher acceptance.
- **Guided decoding + speculative decoding** combination now works.
- **Suffix automaton on device** — GPU-side SA processing for lower latency.
- **Eagle MLA target with GQA draft** support for mixed-architecture speculation.

**GPU-side acceptance** (`py_executor.py`): Uses `torch.cumprod` of equality comparisons to find the longest matching prefix, then gathers the final accepted next token.

**Speculation gate** (`speculation_gate.py`): Dynamically disables speculation when acceptance rates drop below threshold, preventing throughput regression at high batch sizes.

**Key files:** `_torch/speculative/` directory — `model_drafter.py` (two-model loop), `eagle3.py`, `mtp.py`, `ngram.py`, `pard.py`, `suffix_automaton.py`, `speculation_gate.py`.

## Framework Comparison

| Framework | Support | Distinctive Feature |
|:----------|:--------|:-------------------|
| **TensorRT-LLM** | EAGLE3, MTP, NGram, PARD, SA, Draft/Target, user-provided; SA+neural combos | Richest algorithm set; dynamic draft length; SA hybrid approach; guided decoding combo |
| **vLLM** | EAGLE, draft models, NGram (GPU), rejection sampler with greedy/logprobs | Zero-bubble async scheduling + spec decode; multimodal embeddings for spec decode |
| **SGLang** | EAGLE, spec-dec with FlashAttention 4 | FA4 integration for spec decode verification |
