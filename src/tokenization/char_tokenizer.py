"""
Character-level tokenizer — simplest possible baseline.

Every unique character in the training text becomes one token.
Special tokens:
    <PAD>  id 0
    <UNK>  id 1
    <BOS>  id 2
    <EOS>  id 3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class CharTokenizer:
    """Character-level tokenizer built from a training corpus."""

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def fit(self, text: str) -> "CharTokenizer":
        """Build vocabulary from *text*. Returns self."""
        chars = sorted(set(text))
        vocab: dict[str, int] = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            vocab[tok] = i
        for ch in chars:
            if ch not in vocab:
                vocab[ch] = len(vocab)
        self._char_to_id = vocab
        self._id_to_char = {v: k for k, v in vocab.items()}
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        ids = [self._char_to_id.get(ch, UNK_ID) for ch in text]
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        special_ids = {PAD_ID, BOS_ID, EOS_ID}
        chars = []
        for i in ids:
            ch = self._id_to_char.get(i)
            if ch is None:
                continue
            if skip_special and i in special_ids:
                continue
            if skip_special and ch == "<UNK>":
                chars.append("?")
                continue
            if ch in SPECIAL_TOKENS and skip_special:
                continue
            chars.append(ch)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "char",
            "vocab": self._char_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["type"] == "char", "Wrong tokenizer type."
        tok = cls()
        tok._char_to_id = {k: int(v) for k, v in payload["vocab"].items()}
        tok._id_to_char = {int(v): k for k, v in payload["vocab"].items()}
        tok._fitted = True
        return tok

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def compression_ratio(self, text: str) -> float:
        """Tokens / characters — always 1.0 for char tokenizer."""
        return 1.0

    def tokens_per_word(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        total_tokens = sum(len(w) for w in words)
        return total_tokens / len(words)
