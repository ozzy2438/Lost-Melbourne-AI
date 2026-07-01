#!/usr/bin/env python3
"""
Memorisation analysis for the TinyTransformer.

Generates samples and measures n-gram overlap with the training corpus.
Clearly distinguishes:
  - Grammatically plausible generation
  - Memorised training text (high n-gram overlap)
  - Invented / unsupported statements

The tiny Transformer must NOT be described as factually reliable.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from collections import Counter
from model import TinyTransformer
from tokenization import BPETokenizer, CharTokenizer


DISCLAIMER = "[Tiny Transformer experimental generation — not a factual historical answer]"


def make_ngrams(text: str, n: int) -> Counter:
    words = text.lower().split()
    return Counter(zip(*[words[i:] for i in range(n)]))


def overlap_score(generated: str, reference: str, n: int = 3) -> float:
    gen_ngrams = make_ngrams(generated, n)
    ref_ngrams = make_ngrams(reference, n)
    if not gen_ngrams:
        return 0.0
    matched = sum(min(gen_ngrams[g], ref_ngrams[g]) for g in gen_ngrams)
    return matched / sum(gen_ngrams.values())


def analyse(checkpoint_path: str, n_samples: int = 6) -> dict:
    model, payload = TinyTransformer.load_checkpoint(checkpoint_path, device="cpu")
    model.eval()
    cfg = model.cfg
    tok_dir = REPO_ROOT / "data" / "model" / "tokenizers"
    if cfg.tokenizer_type == "char":
        tokenizer = CharTokenizer.load(tok_dir / "char_tokenizer.json")
    else:
        tokenizer = BPETokenizer.load(tok_dir / "bpe_tokenizer.json")

    train_text = (REPO_ROOT / "data" / "model" / "train.txt").read_text(encoding="utf-8")

    prompts = [
        "The history of Melbourne",
        "Flinders Street Station was",
        "The Eastern Market in Melbourne",
        "Carlton and Fitzroy",
        "The heritage buildings of",
        "Melbourne was founded in",
    ]

    results = []
    for i, prompt in enumerate(prompts[:n_samples]):
        for temp in [0.0, 0.7, 1.0]:
            seed = 42 + i
            prompt_ids = tokenizer.encode(prompt)
            gen_ids = model.generate(
                prompt_ids, max_new_tokens=80, temperature=temp,
                seed=seed, eos_id=tokenizer.eos_id, device="cpu"
            )
            generated = tokenizer.decode(gen_ids)
            new_text = generated[len(prompt):]

            overlap_3 = overlap_score(new_text, train_text, n=3)
            overlap_5 = overlap_score(new_text, train_text, n=5)

            # Simple memorisation threshold: 3-gram overlap > 0.3 = likely memorised
            memorised = overlap_3 > 0.30

            results.append({
                "prompt": prompt,
                "temperature": temp,
                "generated": generated[:200],
                "3gram_overlap": round(overlap_3, 3),
                "5gram_overlap": round(overlap_5, 3),
                "likely_memorised": memorised,
            })

    return {
        "checkpoint": checkpoint_path,
        "step": payload.get("step"),
        "val_loss": payload.get("val_loss"),
        "samples": results,
    }


def print_report(analysis: dict) -> None:
    print(f"\n{'='*70}")
    print(f"MEMORISATION ANALYSIS — {DISCLAIMER}")
    print(f"Checkpoint: {analysis['checkpoint']}")
    print(f"Step: {analysis['step']}  Val loss: {analysis['val_loss']:.4f}")
    print(f"{'='*70}")

    for r in analysis["samples"]:
        label = "greedy" if r["temperature"] == 0 else f"temp={r['temperature']}"
        mem_flag = " ⚠ HIGH OVERLAP" if r["likely_memorised"] else ""
        print(f"\nPrompt: {r['prompt']!r}  [{label}]")
        print(f"3-gram overlap with training corpus: {r['3gram_overlap']:.1%}{mem_flag}")
        print(f"5-gram overlap: {r['5gram_overlap']:.1%}")
        print(f"Text: {r['generated'][:150]}")

    high_overlap = [r for r in analysis["samples"] if r["likely_memorised"]]
    print(f"\n{'─'*70}")
    print(f"Samples with high training overlap (3-gram > 30%): {len(high_overlap)}/{len(analysis['samples'])}")
    print(f"{'─'*70}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=6)
    args = p.parse_args()

    result = analyse(args.checkpoint, args.n_samples)
    print_report(result)
