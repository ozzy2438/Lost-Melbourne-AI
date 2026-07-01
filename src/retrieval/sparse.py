"""Transparent BM25 and TF-IDF cosine retrieval baselines."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import RetrievalResult, SearchPassage


TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.I)


def tokenize(text: str) -> list[str]:
    return [token.casefold().replace("’", "'") for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, passages: list[SearchPassage], k1: float = 1.5, b: float = 0.75):
        if not passages:
            raise ValueError("cannot build BM25 over an empty corpus")
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(passage.search_text)) for passage in passages]
        self.lengths = np.asarray([sum(counts.values()) for counts in self.term_frequencies], dtype=np.float64)
        self.average_length = float(self.lengths.mean())
        document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            document_frequency.update(counts.keys())
        count = len(passages)
        self.idf = {term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5)) for term, frequency in document_frequency.items()}

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        scores = np.zeros(len(self.passages), dtype=np.float64)
        for term, query_frequency in Counter(tokenize(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += query_frequency * idf * frequency * (self.k1 + 1) / denominator
        return _rank(self.passages, scores, top_k, "bm25")


class TfidfIndex:
    def __init__(self, passages: list[SearchPassage]):
        if not passages:
            raise ValueError("cannot build TF-IDF over an empty corpus")
        self.passages = passages
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, norm="l2")
        self.matrix = self.vectorizer.fit_transform(passage.search_text for passage in passages)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        scores = self.score_array(query)
        return _rank(self.passages, scores, top_k, "tfidf")

    def score_array(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query])
        return (self.matrix @ query_vector.T).toarray().ravel()

    def score_by_passage_id(self, query: str) -> dict[str, float]:
        scores = self.score_array(query)
        return {
            passage.passage_id: float(scores[index])
            for index, passage in enumerate(self.passages)
        }


def _rank(passages: list[SearchPassage], scores: np.ndarray, top_k: int, component: str) -> list[RetrievalResult]:
    order = sorted(range(len(passages)), key=lambda index: (-float(scores[index]), passages[index].passage_id))
    results: list[RetrievalResult] = []
    seen: set[str] = set()
    for index in order:
        score = float(scores[index])
        if score <= 0:
            continue
        passage = passages[index]
        if passage.passage_id in seen:
            continue
        seen.add(passage.passage_id)
        results.append(RetrievalResult(
            passage_id=passage.passage_id,
            parent_passage_id=passage.parent_passage_id,
            score=score,
            rank=len(results) + 1,
            score_components={component: score},
            explanation=[f"{component} lexical score={score:.6f}"],
        ))
        if len(results) >= top_k:
            break
    return results
