#!/usr/bin/env python3
"""
Train the tiny decoder-only Transformer on the Melbourne corpus.

Usage:
    python3 scripts/train_transformer.py [options]

Options:
    --tokenizer   char|bpe  (default: bpe)
    --n-layers    int       (default: 2)
    --n-heads     int       (default: 4)
    --embed-dim   int       (default: 128)
    --context-len int       (default: 128)
    --max-steps   int       (default: 3000)
    --seed        int       (default: 42)
    --resume      path      resume from checkpoint
    --run-id      str       experiment label
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from model import BigramLM, TransformerConfig, TinyTransformer, UnigramLM
from tokenization import BPETokenizer, CharTokenizer

# ──────────────────────────────────────────────────────────────────────────────
TOKENIZER_DIR  = REPO_ROOT / "data" / "model" / "tokenizers"
DATA_DIR       = REPO_ROOT / "data" / "model"
ARTIFACT_DIR   = REPO_ROOT / "artifacts" / "checkpoints"
EXPERIMENT_LOG = REPO_ROOT / "experiments" / "transformer_runs.jsonl"
REPORT_PATH    = REPO_ROOT / "reports" / "transformer_training_report.md"


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
        print("Hardware backend: Apple Silicon MPS")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        print("Hardware backend: CUDA GPU")
    else:
        dev = torch.device("cpu")
        print("Hardware backend: CPU")
    return dev


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(tok_type: str):
    if tok_type == "char":
        path = TOKENIZER_DIR / "char_tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"Run scripts/train_tokenizer.py first: {path}")
        return CharTokenizer.load(path)
    else:
        path = TOKENIZER_DIR / "bpe_tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"Run scripts/train_tokenizer.py first: {path}")
        return BPETokenizer.load(path)


def text_to_ids(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text)


def make_batches(ids: list[int], context_len: int, batch_size: int):
    """Yield (input_ids, targets) tensors of shape (B, T)."""
    T = context_len
    # Stride by T so no overlap
    chunks = [ids[i: i + T + 1] for i in range(0, len(ids) - T, T)]
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        if not batch:
            break
        # Pad shorter sequences
        padded = [c + [0] * (T + 1 - len(c)) for c in batch if len(c) >= 2]
        if not padded:
            continue
        tensor = torch.tensor(padded, dtype=torch.long)
        yield tensor[:, :T], tensor[:, 1: T + 1]


def corpus_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: TinyTransformer, ids: list[int], cfg: TransformerConfig, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in make_batches(ids, cfg.context_len, cfg.batch_size):
        x, y = x.to(device), y.to(device)
        out = model(x, targets=y)
        total_loss += out["loss"].item()
        n_batches += 1
    model.train()
    if n_batches == 0:
        return float("inf")
    return total_loss / n_batches


# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────────────

def run_baselines(train_ids, val_ids, vocab_size):
    unigram = UnigramLM().fit(train_ids, vocab_size)
    bigram = BigramLM().fit(train_ids, vocab_size)
    unigram_ppl = unigram.perplexity(val_ids[:5000])
    bigram_ppl = bigram.perplexity(val_ids[:5000])
    print(f"Unigram perplexity (val): {unigram_ppl:.1f}")
    print(f"Bigram  perplexity (val): {bigram_ppl:.1f}")
    return {"unigram_ppl": unigram_ppl, "bigram_ppl": bigram_ppl}


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = select_device()

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
    ckpt_dir = ARTIFACT_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer}  vocab_size={tokenizer.vocab_size}")

    # Load data
    train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    val_text   = (DATA_DIR / "validation.txt").read_text(encoding="utf-8")
    train_ids  = text_to_ids(train_text, tokenizer)
    val_ids    = text_to_ids(val_text, tokenizer)
    print(f"Train tokens: {len(train_ids):,}  Val tokens: {len(val_ids):,}")

    fingerprint = corpus_fingerprint(train_text)

    # Baselines
    baseline_results = run_baselines(train_ids, val_ids, tokenizer.vocab_size)

    # Config
    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        embed_dim=args.embed_dim,
        ff_dim=args.embed_dim * 4,
        context_len=args.context_len,
        dropout=0.1,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=min(200, args.max_steps // 10),
        val_interval=args.val_interval,
        checkpoint_dir=str(ckpt_dir),
        seed=args.seed,
        tokenizer_type=args.tokenizer,
        run_id=run_id,
    )
    cfg.save(ckpt_dir / "config.json")

    # Model
    model = TinyTransformer(cfg).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}  Config: {cfg.param_summary()}")

    # Optionally resume
    start_step = 0
    if args.resume:
        model, payload = TinyTransformer.load_checkpoint(args.resume, device=str(device))
        model = model.to(device)
        start_step = payload.get("step", 0)
        print(f"Resumed from step {start_step}: {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    def lr_schedule(step: int) -> float:
        """Linear warmup then cosine decay."""
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    train_losses: list[dict] = []
    val_losses:   list[dict] = []
    best_val_loss = float("inf")
    best_ckpt = ""
    t0 = time.monotonic()
    tokens_processed = 0
    step = start_step

    # Infinite data loader cycling over shuffled chunks
    def data_stream():
        while True:
            batches = list(make_batches(train_ids, cfg.context_len, cfg.batch_size))
            if not batches:
                break
            import random
            random.shuffle(batches)
            yield from batches

    stream = data_stream()

    model.train()
    while step < cfg.max_steps:
        try:
            x, y = next(stream)
        except StopIteration:
            stream = data_stream()
            x, y = next(stream)

        x, y = x.to(device), y.to(device)
        out = model(x, targets=y)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        tokens_processed += x.numel()
        step += 1

        train_losses.append({"step": step, "loss": loss.item()})

        if step % cfg.val_interval == 0 or step == cfg.max_steps:
            val_loss = evaluate(model, val_ids, cfg, device)
            elapsed = time.monotonic() - t0
            lr_now = scheduler.get_last_lr()[0]
            val_losses.append({"step": step, "val_loss": val_loss})
            ppl = math.exp(min(val_loss, 20))
            print(
                f"step {step:>5}/{cfg.max_steps}  "
                f"train={loss.item():.4f}  val={val_loss:.4f}  "
                f"ppl={ppl:.1f}  lr={lr_now:.2e}  "
                f"tokens={tokens_processed:,}  elapsed={elapsed:.0f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = str(ckpt_dir / "best.pt")
                model.save_checkpoint(
                    best_ckpt,
                    extra={"step": step, "val_loss": val_loss, "run_id": run_id},
                )
                print(f"  ✓ New best checkpoint saved (val={val_loss:.4f})")

    # Save final checkpoint
    final_ckpt = str(ckpt_dir / "final.pt")
    model.save_checkpoint(
        final_ckpt,
        extra={"step": step, "val_loss": val_losses[-1]["val_loss"] if val_losses else None, "run_id": run_id},
    )

    elapsed_total = time.monotonic() - t0
    final_val = val_losses[-1]["val_loss"] if val_losses else float("inf")

    result = {
        "run_id": run_id,
        "git_commit": get_git_commit(),
        "tokenizer_type": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "config": cfg.__dict__,
        "n_params": n_params,
        "seed": args.seed,
        "dataset_fingerprint": fingerprint,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "baseline_unigram_ppl": baseline_results["unigram_ppl"],
        "baseline_bigram_ppl": baseline_results["bigram_ppl"],
        "final_train_loss": train_losses[-1]["loss"] if train_losses else None,
        "final_val_loss": final_val,
        "best_val_loss": best_val_loss,
        "best_checkpoint": best_ckpt,
        "final_checkpoint": final_ckpt,
        "tokens_processed": tokens_processed,
        "elapsed_seconds": round(elapsed_total, 1),
        "device": str(device),
        "steps_completed": step,
        "val_curve": val_losses,
        "status": "complete",
    }

    # Append to experiment log
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  Elapsed: {elapsed_total:.0f}s")
    print(f"Best checkpoint: {best_ckpt}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer",   default="bpe",  choices=["char", "bpe"])
    p.add_argument("--n-layers",    type=int, default=2)
    p.add_argument("--n-heads",     type=int, default=4)
    p.add_argument("--embed-dim",   type=int, default=128)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--max-steps",   type=int, default=3000)
    p.add_argument("--val-interval",type=int, default=300)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--resume",      type=str, default="")
    p.add_argument("--run-id",      type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
