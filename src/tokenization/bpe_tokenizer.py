"""
Minimal Byte Pair Encoding (BPE) tokenizer built entirely on this corpus.

Algorithm (standard BPE):
1. Start with a character vocabulary where each word is represented as a
   sequence of characters with an end-of-word marker.
2. Count all adjacent symbol pairs across the corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat for `num_merges` steps.
5. To encode, apply merges greedily in the order they were learned.

Special tokens match CharTokenizer:
    <PAD>  0  <UNK>  1  <BOS>  2  <EOS>  3
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
EOW = "</w>"  # end-of-word marker


def _word_freqs(text: str) -> Counter:
    """Count space-separated words after lower-casing and basic punctuation split."""
    tokens = re.findall(r"[a-zA-Z']+|[^a-zA-Z'\s]", text)
    return Counter(tokens)


def _word_to_chars(word: str) -> tuple[str, ...]:
    return tuple(list(word[:-len(EOW)] if word.endswith(EOW) else word) + [EOW])


def _get_pairs(vocab: dict[tuple, int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_vocab(pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
    merged = pair[0] + pair[1]
    new_vocab: dict[tuple, int] = {}
    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


class BPETokenizer:
    """Byte Pair Encoding tokenizer built from scratch for the Melbourne corpus."""

    def __init__(self, num_merges: int = 1000) -> None:
        self.num_merges = num_merges
        self._merges: list[tuple[str, str]] = []
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, text: str) -> "BPETokenizer":
        word_freqs = _word_freqs(text)
        # Initial vocab: characters of each word + EOW marker
        vocab: dict[tuple, int] = {}
        for word, freq in word_freqs.items():
            chars = tuple(list(word) + [EOW])
            vocab[chars] = freq

        merges: list[tuple[str, str]] = []
        for _ in range(self.num_merges):
            pairs = _get_pairs(vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            vocab = _merge_vocab(best, vocab)
            merges.append(best)

        # Build token vocabulary from all subwords that appear
        all_tokens: set[str] = set()
        for symbols in vocab:
            all_tokens.update(symbols)

        token_to_id: dict[str, int] = {}
        for i, sp in enumerate(SPECIAL_TOKENS):
            token_to_id[sp] = i
        for tok in sorted(all_tokens):
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        self._merges = merges
        self._token_to_id = token_to_id
        self._id_to_token = {v: k for k, v in token_to_id.items()}
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned merges to a single word."""
        if not word:
            return []
        symbols = list(word) + [EOW]
        for pair in self._merges:
            merged = pair[0] + pair[1]
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Encode text to token ids.

        Whitespace is NOT encoded as separate tokens — word boundary
        information is carried by the EOW marker appended to each word's
        final subword token (e.g. ``the</w>``).  Skipping whitespace keeps
        the vocabulary clean and avoids large numbers of UNK tokens.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        tokens = re.findall(r"[a-zA-Z']+|[^a-zA-Z'\s]", text)
        ids: list[int] = []
        if add_special:
            ids.append(BOS_ID)
        for tok in tokens:
            for sub in self._tokenize_word(tok):
                ids.append(self._token_to_id.get(sub, UNK_ID))
        if add_special:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Decode token ids back to text.

        EOW markers are converted to spaces; unknown tokens become ``?``.
        """
        special_ids = {PAD_ID, BOS_ID, EOS_ID}
        parts: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i, "<UNK>")
            if skip_special and i in special_ids:
                continue
            if tok == "<UNK>" or tok not in self._token_to_id:
                parts.append("?")
                continue
            parts.append(tok)
        # EOW marks the end of a word → replace with space
        text = "".join(parts).replace(EOW, " ").strip()
        return text

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

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
            "type": "bpe",
            "num_merges": self.num_merges,
            "merges": self._merges,
            "vocab": self._token_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["type"] == "bpe", "Wrong tokenizer type."
        tok = cls(num_merges=payload["num_merges"])
        tok._merges = [tuple(m) for m in payload["merges"]]
        tok._token_to_id = {k: int(v) for k, v in payload["vocab"].items()}
        tok._id_to_token = {int(v): k for k, v in payload["vocab"].items()}
        tok._fitted = True
        return tok

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def compression_ratio(self, text: str) -> float:
        """Tokens produced per character — lower means better compression."""
        if not text:
            return 0.0
        return len(self.encode(text)) / max(len(text), 1)

    def tokens_per_word(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        total = sum(len(self._tokenize_word(w)) for w in words)
        return total / len(words)
