"""
Unigram and bigram language model baselines.

Used to establish a minimum bar before training the tiny Transformer.
Perplexity here is computed over the token sequence at character or BPE level.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence


class UnigramLM:
    """Smoothed unigram model: P(t) = (count(t) + alpha) / (N + alpha * V)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._counts: Counter = Counter()
        self._total: int = 0
        self._vocab_size: int = 0

    def fit(self, ids: Sequence[int], vocab_size: int) -> "UnigramLM":
        self._counts = Counter(ids)
        self._total = len(ids)
        self._vocab_size = vocab_size
        return self

    def log_prob(self, token_id: int) -> float:
        count = self._counts.get(token_id, 0)
        return math.log((count + self.alpha) / (self._total + self.alpha * self._vocab_size))

    def perplexity(self, ids: Sequence[int]) -> float:
        if not ids:
            return float("inf")
        log_sum = sum(self.log_prob(t) for t in ids)
        return math.exp(-log_sum / len(ids))


class BigramLM:
    """Smoothed bigram model: P(t | t-1) = (count(t-1, t) + alpha) / (count(t-1) + alpha * V)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._bigrams: Counter = Counter()
        self._unigrams: Counter = Counter()
        self._vocab_size: int = 0

    def fit(self, ids: Sequence[int], vocab_size: int) -> "BigramLM":
        ids = list(ids)
        self._unigrams = Counter(ids)
        self._bigrams = Counter(zip(ids[:-1], ids[1:]))
        self._vocab_size = vocab_size
        return self

    def log_prob(self, context_id: int, token_id: int) -> float:
        bigram_count = self._bigrams.get((context_id, token_id), 0)
        context_count = self._unigrams.get(context_id, 0)
        return math.log(
            (bigram_count + self.alpha) / (context_count + self.alpha * self._vocab_size)
        )

    def perplexity(self, ids: Sequence[int]) -> float:
        ids = list(ids)
        if len(ids) < 2:
            return float("inf")
        log_sum = sum(
            self.log_prob(ids[i], ids[i + 1]) for i in range(len(ids) - 1)
        )
        return math.exp(-log_sum / (len(ids) - 1))
