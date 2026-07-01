"""Dense passage retrieval with explicit NumPy cosine similarity and deterministic caches."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import RetrievalResult, SearchPassage
from .sparse import tokenize


MODEL_CONFIGS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "dimension": 384,
        "licence": "Apache-2.0",
        "query_prefix": "",
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "dimension": 384,
        "licence": "MIT",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}


class Encoder(Protocol):
    dimension: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def memory_bytes(self) -> int: ...
    def storage_bytes(self) -> int: ...


class SentenceTransformerEncoder:
    def __init__(self, key: str):
        if key not in MODEL_CONFIGS:
            raise ValueError(f"unknown dense model key: {key}")
        from sentence_transformers import SentenceTransformer

        self.key = key
        self.config = MODEL_CONFIGS[key]
        self.model = SentenceTransformer(self.config["name"], revision=self.config["revision"])
        self.dimension = int(self.model.get_embedding_dimension())
        if self.dimension != self.config["dimension"]:
            raise ValueError(f"unexpected embedding dimension for {key}: {self.dimension}")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefix = self.config["query_prefix"]
        return np.asarray(self.model.encode([prefix + text for text in texts], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def memory_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())

    def storage_bytes(self) -> int:
        from huggingface_hub import scan_cache_dir

        return next((repo.size_on_disk for repo in scan_cache_dir().repos if repo.repo_id == self.config["name"]), 0)


class HashingEncoder:
    """Offline deterministic test encoder; never used for the production benchmark."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def _encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                matrix[row, index] += -1.0 if digest[4] & 1 else 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def memory_bytes(self) -> int:
        return 0

    def storage_bytes(self) -> int:
        return 0


@dataclass
class DenseBuildStats:
    dimension: int
    indexing_seconds: float
    memory_bytes: int
    model_storage_bytes: int
    index_bytes: int
    cache_hit: bool


class DenseIndex:
    def __init__(self, passages: list[SearchPassage], encoder: Encoder, vectors: np.ndarray):
        if vectors.shape != (len(passages), encoder.dimension):
            raise ValueError(f"invalid embedding matrix shape: {vectors.shape}")
        self.passages = passages
        self.encoder = encoder
        self.vectors = _normalise(vectors)

    @classmethod
    def build(
        cls,
        passages: list[SearchPassage],
        encoder: Encoder,
        cache_path: Path | None = None,
    ) -> tuple["DenseIndex", DenseBuildStats]:
        signature = _passage_signature(passages, encoder.dimension)
        start = time.perf_counter()
        cache_hit = False
        if cache_path and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["signature"].item()) == signature:
                vectors = cached["vectors"]
                cache_hit = True
            else:
                vectors = encoder.encode_documents([passage.search_text for passage in passages])
        else:
            vectors = encoder.encode_documents([passage.search_text for passage in passages])
        vectors = _normalise(np.asarray(vectors, dtype=np.float32))
        if cache_path and not cache_hit:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, vectors=vectors, signature=np.asarray(signature))
        elapsed = time.perf_counter() - start
        index_bytes = cache_path.stat().st_size if cache_path and cache_path.exists() else vectors.nbytes
        return cls(passages, encoder, vectors), DenseBuildStats(
            dimension=encoder.dimension,
            indexing_seconds=elapsed,
            memory_bytes=encoder.memory_bytes(),
            model_storage_bytes=encoder.storage_bytes(),
            index_bytes=index_bytes,
            cache_hit=cache_hit,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        query_vector = _normalise(self.encoder.encode_queries([query]))[0]
        scores = self.vectors @ query_vector
        order = sorted(range(len(self.passages)), key=lambda index: (-float(scores[index]), self.passages[index].passage_id))
        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for index in order:
            passage = self.passages[index]
            if passage.passage_id in seen:
                continue
            seen.add(passage.passage_id)
            score = float(scores[index])
            results.append(RetrievalResult(
                passage_id=passage.passage_id,
                parent_passage_id=passage.parent_passage_id,
                score=score,
                rank=len(results) + 1,
                score_components={"dense_cosine": score},
                explanation=[f"dense cosine similarity={score:.6f}"],
            ))
            if len(results) >= top_k:
                break
        return results


def write_model_metadata(path: Path, key: str, stats: DenseBuildStats) -> None:
    config = MODEL_CONFIGS[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_key": key,
        "model_name": config["name"],
        "revision": config["revision"],
        "dimension": stats.dimension,
        "licence": config["licence"],
        "memory_bytes": stats.memory_bytes,
        "model_storage_bytes": stats.model_storage_bytes,
        "index_bytes": stats.index_bytes,
        "indexing_seconds": stats.indexing_seconds,
        "cache_hit": stats.cache_hit,
        "similarity": "cosine over L2-normalised vectors",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def _passage_signature(passages: list[SearchPassage], dimension: int) -> str:
    payload = "\n".join(f"{passage.passage_id}\t{passage.search_text}" for passage in passages)
    return hashlib.sha256(f"{dimension}\n{payload}".encode()).hexdigest()
