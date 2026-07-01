# Tiny Transformer Training Report — Phase 4

> **All generated text in this report is labelled:**
> *Tiny Transformer experimental generation — not a factual historical answer*

---

## Overview

This report covers Phase 4 of the Lost Melbourne AI project: building, training, and
evaluating a decoder-only Transformer language model from scratch on the Melbourne
historical corpus.

The tiny Transformer is an **educational experiment**. It must not replace the
evidence-grounded retrieval system and must not be presented as a reliable historical
source.

---

## Tokenizer statistics

| Metric | Character | BPE (1000 merges) |
| --- | ---: | ---: |
| Vocabulary size | 115 | 1,065 |
| Train tokens | 389,157 | 137,535 |
| Compression ratio (tokens/char) | 1.000 | 0.354 |
| Average tokens per word | 4.77 | 1.72 |
| Encoding time | <1 ms | <1 ms |
| Save/load consistent | Yes | Yes |
| Unknown token (OOV) behaviour | UNK per character | UNK per subword |

BPE provides 2.83× compression over character tokenisation, reducing sequence length
and allowing the same context window to cover more text.

---

## Language model baselines

Evaluated on the validation set (12,185 BPE tokens / 33,009 char tokens).

| Model | Perplexity (BPE) | Perplexity (char) |
| --- | ---: | ---: |
| Unigram | 401.8 | 28.2 |
| Bigram | 70.3 | 12.3 |
| Tiny Transformer (best) | **52.9** (BPE) | **7.3** (char) |

The Transformer outperforms the bigram baseline, confirming it captures patterns
beyond simple 2-gram statistics.  The lower char perplexity is a different measurement
(character-level vs subword-level) and is not directly comparable to BPE perplexity.

---

## Transformer configuration and parameter counts

| Run | Layers | Heads | Embed | FF | Ctx | Vocab | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bpe_small_ctx128_v2` | 2 | 4 | 128 | 512 | 128 | 1,065 | 548,480 |
| `char_small_ctx128` | 2 | 4 | 128 | 512 | 128 | 115 | 426,880 |
| `bpe_large_ctx256_v2` | 4 | 4 | 256 | 1,024 | 256 | 1,065 | 3,493,632 |

---

## Training results

All runs: 3,000 steps, AdamW, linear warmup (200 steps) + cosine decay, grad clip 1.0,
batch size 32, seed 42. Hardware: **Apple Silicon MPS**.

| Run | Best val loss | Final train loss | Overfitting? | Best step |
| --- | ---: | ---: | --- | ---: |
| `bpe_small_ctx128_v2` | 3.9690 | 3.6925 | Mild | 3000 |
| `char_small_ctx128` | 1.9868 | 2.0560 | None (val still improving) | 3000 |
| `bpe_large_ctx256_v2` | **4.0733** | **0.9550** | **Severe** (best at step 600) | 600 |

### Validation loss curves (BPE small vs BPE large)

```
Step   BPE-small-val   BPE-large-val
 300       5.0724          4.3479
 600       4.4277          4.0733  ← large model best
 900       4.2093          4.2524
1200       4.1058          4.5027
1500       4.0338          4.7643
1800       4.0043          4.9622
2100       3.9854          5.1281
2400       3.9736          5.2324
2700       3.9699          5.3030
3000       3.9690          5.3604
```

The large model (3.5M parameters, 137K training tokens) begins memorising by step 900.
Its training loss reaches 0.96 while validation loss climbs to 5.36 — a clear sign of
overfitting on this small corpus.

---

## Hardware backend

Both models trained on **Apple Silicon MPS** (Metal Performance Shaders).
CPU fallback is automatic when MPS is unavailable.  CUDA was not present.

---

## Overfitting and memorisation analysis

### n-gram overlap with training corpus (BPE small, best checkpoint)

18 generated samples (6 prompts × 3 temperatures):

| Overlap level | Count |
| --- | ---: |
| 3-gram overlap > 30% (likely memorised) | 1 |
| 5-gram overlap > 0% | 1 |

The one high-overlap sample ("Carlton and Fitzroy", temp=0.7, 3-gram=76.4%, 5-gram=48.6%)
reproduced a Wikipedia table structure (`| | |` pipe separators) present in the training
corpus.  This is memorisation of formatting rather than historical facts.

No sample reproduced multi-sentence factual claims verbatim at the 5-gram level.

### Assessment

The model shows:
- **Grammatically plausible generation** — recognisable English, Melbourne vocabulary
- **Surface-level memorisation** — repeated phrases, table formatting, familiar bigrams
- **Invented statements** — dates and names mixed in ways not supported by any source
- **Loop collapse under greedy decoding** — common failure of small models

---

## Sample generations

> *Tiny Transformer experimental generation — not a factual historical answer*

### BPE small, best checkpoint (val loss 3.97), prompt: "Flinders Street Station was"

**Greedy (deterministic):**
> Flinders Street Station was replaced by the station . The first of the 1 9 0 s , and
> the 1 9 0 s , the 1 9 0 s ...  *(loops)*

**Temperature 0.7:**
> Flinders Street Station was a interroof the nsheach office 1 8 5 , and other hotels .
> There are , and Seph service of the famility of the original buildings were town in
> 1 8 9 9 8 9 1 8 9 0 s ...

**Temperature 1.0:**
> Flinders Street Station was near has an centre , and the Eastern 3 there was landmarkets
> and design of and controration of Architman in 1 8 8 ...

### BPE large (overfit, step 600 best), prompt: "The history of Melbourne"

> (Not shown — overfit model is not used for generation; its best checkpoint is at step 600
> before severe overfitting begins.)

---

## Attention visualisation

An educational attention weight visualisation for head 0, block 0 is saved at:
`reports/attention_weights.png`

The causal mask (all future positions blocked) is at: `reports/causal_mask.png`

Positional embeddings heatmap: `reports/pos_embeddings.png`

**Caveat:** Attention weights are one window into the model and do not fully explain its
behaviour.  A head attending strongly to a token does not guarantee that token caused the
output.

---

## Test results

```
Ran 99 tests in 1.761s   OK
```

Tests added in Phase 4:
- `tests/test_tokenizer.py` — 24 tests: char + BPE round-trips, determinism, save/load,
  unknown chars, empty input, type-mismatch rejection
- `tests/test_transformer.py` — 20 tests: causal masking, attention shapes, invalid heads,
  forward pass, loss computation, generation determinism, checkpoint save/load, context
  truncation, very short input, CPU operation

All existing 55 collection + preparation + retrieval tests continue to pass.

---

## Why these three components serve different purposes

| Component | Purpose | Suitable for |
| --- | --- | --- |
| Tiny Transformer | Educational language modelling experiment | Learning architecture, observing overfitting, generating grammatically plausible but unreliable text |
| Retrieval system (Phase 3) | Evidence retrieval from verified corpus | Finding passages that support or deny a factual claim |
| Production generator | Factual answering | Requires pretrained LLM (e.g. Claude) + retrieval + claim validation against source passages |

The tiny Transformer **must not** be connected to the production retrieval answerer and
**must not** be used to generate historical answers presented as verified facts.

---

## Experiment log

Full JSONL log: `experiments/transformer_runs.jsonl` (5 runs)

Each entry records: run_id, git_commit, tokenizer_type, config, n_params, seed,
dataset_fingerprint, train/val metrics, checkpoint paths, tokens_processed,
elapsed_seconds, device, val_curve.
