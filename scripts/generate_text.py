#!/usr/bin/env python3
"""
Generate text samples from a trained TinyTransformer checkpoint.

Usage:
    python3 scripts/generate_text.py --checkpoint artifacts/checkpoints/<run>/best.pt \
        --prompt "Melbourne" --max-tokens 200 --temperature 0.7

All output is labelled:
    Tiny Transformer experimental generation — not a factual historical answer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import TinyTransformer
from tokenization import BPETokenizer, CharTokenizer

DISCLAIMER = "\n[Tiny Transformer experimental generation — not a factual historical answer]\n"

PROMPTS = [
    "The history of Melbourne",
    "Flinders Street Station was",
    "The Eastern Market in Melbourne",
]


def load_tokenizer(tok_type: str, tokenizer_dir: Path):
    if tok_type == "char":
        return CharTokenizer.load(tokenizer_dir / "char_tokenizer.json")
    return BPETokenizer.load(tokenizer_dir / "bpe_tokenizer.json")


def generate_sample(
    model: TinyTransformer,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: str,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        eos_id=tokenizer.eos_id,
        device=device,
    )
    return tokenizer.decode(generated_ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-prompts", action="store_true",
                   help="Run all built-in Melbourne prompts with multiple temperatures")
    args = p.parse_args()

    model, payload = TinyTransformer.load_checkpoint(args.checkpoint)
    model.eval()
    cfg = model.cfg
    tok_dir = REPO_ROOT / "data" / "model" / "tokenizers"
    tokenizer = load_tokenizer(cfg.tokenizer_type, tok_dir)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {cfg.param_summary()}")
    print(f"Step: {payload.get('step', '?')}  Val loss: {payload.get('val_loss', '?')}")
    print(DISCLAIMER)

    if args.all_prompts:
        for prompt in PROMPTS:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt!r}")
            for temp in [0.0, 0.7, 1.0]:
                label = "greedy" if temp == 0 else f"temp={temp}"
                text = generate_sample(model, tokenizer, prompt, args.max_tokens, temp, args.top_k, args.seed, "cpu")
                print(f"\n--- {label} ---")
                print(text[:500])
    else:
        prompt = args.prompt or PROMPTS[0]
        text = generate_sample(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.seed, "cpu")
        print(f"Prompt: {prompt!r}")
        print(f"\n--- temperature={args.temperature}, top_k={args.top_k} ---")
        print(text)

    print(DISCLAIMER)


if __name__ == "__main__":
    main()
