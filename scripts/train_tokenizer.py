#!/usr/bin/env python3
"""
Train and evaluate both tokenizers on the Melbourne corpus.

Outputs:
    data/model/tokenizers/char_tokenizer.json
    data/model/tokenizers/bpe_tokenizer.json
    reports/tokenizer_report.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokenization import CharTokenizer, BPETokenizer

TRAIN_TEXT = REPO_ROOT / "data" / "model" / "train.txt"
TOKENIZER_DIR = REPO_ROOT / "data" / "model" / "tokenizers"
REPORT_PATH = REPO_ROOT / "reports" / "tokenizer_report.md"

EXAMPLE_PHRASES = [
    "Flinders Street Station was built in 1905.",
    "The Eastern Market in Melbourne was demolished in 1960.",
    "Located at 309 Bourke Street, the hotel opened in 1883.",
    "Carlton, Fitzroy and Collingwood are inner Melbourne suburbs.",
    "Heritage Victoria protects significant buildings across the state.",
]

NUM_BPE_MERGES = 1000


def stats_block(tok, text: str, name: str) -> str:
    t0 = time.monotonic()
    ids = tok.encode(text)
    t1 = time.monotonic()
    ratio = len(ids) / max(len(text), 1)
    tpw = tok.tokens_per_word(text)
    return (
        f"**{name}**\n"
        f"- Vocabulary size: {tok.vocab_size:,}\n"
        f"- Tokens for train.txt: {len(ids):,}\n"
        f"- Characters in train.txt: {len(text):,}\n"
        f"- Compression ratio (tokens/char): {ratio:.4f}\n"
        f"- Average tokens per word: {tpw:.2f}\n"
        f"- Encoding time: {(t1-t0)*1000:.1f} ms\n"
    )


def example_block(tok, phrase: str) -> str:
    ids = tok.encode(phrase)
    decoded = tok.decode(ids)
    rt_ok = decoded.replace(" ", "").lower() == phrase.replace(" ", "").lower()
    return (
        f"Input:   `{phrase}`\n"
        f"IDs:     {ids[:20]}{'...' if len(ids) > 20 else ''}\n"
        f"Decoded: `{decoded[:80]}`\n"
        f"Round-trip: {'✓' if rt_ok else '✗'}\n"
    )


def main() -> None:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    train_text = TRAIN_TEXT.read_text(encoding="utf-8")
    print(f"Training corpus: {len(train_text):,} chars, {len(train_text.split()):,} words")

    # ---- Character tokenizer ----
    print("Fitting CharTokenizer …")
    char_tok = CharTokenizer().fit(train_text)
    char_tok.save(TOKENIZER_DIR / "char_tokenizer.json")
    print(f"  vocab={char_tok.vocab_size}")

    # ---- BPE tokenizer ----
    print(f"Fitting BPETokenizer (num_merges={NUM_BPE_MERGES}) …")
    bpe_tok = BPETokenizer(num_merges=NUM_BPE_MERGES).fit(train_text)
    bpe_tok.save(TOKENIZER_DIR / "bpe_tokenizer.json")
    print(f"  vocab={bpe_tok.vocab_size}")

    # ---- Build report ----
    lines = [
        "# Tokenizer Report\n",
        "## Corpus\n",
        f"- File: `data/model/train.txt`\n",
        f"- Characters: {len(train_text):,}\n",
        f"- Words: {len(train_text.split()):,}\n",
        "\n## Statistics\n",
        stats_block(char_tok, train_text, "Character tokenizer"),
        "\n",
        stats_block(bpe_tok, train_text, f"BPE tokenizer (num_merges={NUM_BPE_MERGES})"),
        "\n## Round-trip examples\n",
        "### Character tokenizer\n",
    ]
    for phrase in EXAMPLE_PHRASES:
        lines.append(example_block(char_tok, phrase))
        lines.append("\n")
    lines.append("### BPE tokenizer\n")
    for phrase in EXAMPLE_PHRASES:
        lines.append(example_block(bpe_tok, phrase))
        lines.append("\n")

    lines.append("## Unknown-token behaviour\n")
    unk_test = "τελεία 日本語 émoji 🏛️"
    char_unk = char_tok.encode(unk_test)
    bpe_unk = bpe_tok.encode(unk_test)
    unk_rate_char = char_unk.count(1) / max(len(char_unk), 1)
    unk_rate_bpe = bpe_unk.count(1) / max(len(bpe_unk), 1)
    lines.append(
        f"Input: `{unk_test}`\n\n"
        f"| Tokenizer | UNK ids | UNK rate |\n"
        f"| --- | --- | --- |\n"
        f"| Char | {char_unk.count(1)} | {unk_rate_char:.0%} |\n"
        f"| BPE | {bpe_unk.count(1)} | {unk_rate_bpe:.0%} |\n\n"
    )

    lines.append("## Saved artifacts\n")
    lines.append(f"- `data/model/tokenizers/char_tokenizer.json` ({(TOKENIZER_DIR / 'char_tokenizer.json').stat().st_size:,} bytes)\n")
    lines.append(f"- `data/model/tokenizers/bpe_tokenizer.json` ({(TOKENIZER_DIR / 'bpe_tokenizer.json').stat().st_size:,} bytes)\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
