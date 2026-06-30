#!/usr/bin/env python3
"""Build deterministic sparse and dense indexes for the Phase 3 laboratory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval import (  # noqa: E402
    BM25Index,
    DenseIndex,
    HistoricalCorpus,
    SentenceTransformerEncoder,
    TfidfIndex,
    build_passages,
)
from retrieval.dense import MODEL_CONFIGS, write_model_metadata  # noqa: E402


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "retrieval"


def build_indexes(processed_dir: Path, artifact_dir: Path, skip_dense: bool = False) -> dict:
    corpus = HistoricalCorpus.load(processed_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema_version": 1,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "sentence-transformers", "torch")
        },
        "corpus_counts": corpus.counts(),
        "representations": {},
        "models": {},
    }

    representations = {strategy: build_passages(corpus, strategy) for strategy in ("original", "small", "parent_child")}
    for strategy, passages in representations.items():
        strategy_dir = artifact_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        bm25 = BM25Index(passages)
        bm25_seconds = time.perf_counter() - start
        bm25_path = strategy_dir / "bm25.joblib"
        joblib.dump(bm25, bm25_path, compress=3)
        start = time.perf_counter()
        tfidf = TfidfIndex(passages)
        tfidf_seconds = time.perf_counter() - start
        tfidf_path = strategy_dir / "tfidf.joblib"
        joblib.dump(tfidf, tfidf_path, compress=3)
        child_lengths = [len(passage.metadata.get("child_text", passage.text).split()) for passage in passages]
        manifest["representations"][strategy] = {
            "passage_count": len(passages),
            "minimum_indexed_words": min(child_lengths),
            "maximum_indexed_words": max(child_lengths),
            "bm25_indexing_seconds": bm25_seconds,
            "bm25_index_bytes": bm25_path.stat().st_size,
            "tfidf_indexing_seconds": tfidf_seconds,
            "tfidf_index_bytes": tfidf_path.stat().st_size,
        }

    for model_key, config in ({} if skip_dense else MODEL_CONFIGS).items():
        encoder = SentenceTransformerEncoder(model_key)
        model_rows = {}
        for strategy in ("original", "small"):
            cache_path = artifact_dir / strategy / f"dense_{model_key}.npz"
            metadata_path = artifact_dir / strategy / f"dense_{model_key}.json"
            previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
            _, stats = DenseIndex.build(representations[strategy], encoder, cache_path)
            if stats.cache_hit and previous and previous.get("indexing_seconds"):
                stats.indexing_seconds = float(previous["indexing_seconds"])
            write_model_metadata(metadata_path, model_key, stats)
            model_rows[strategy] = {
                "dimension": stats.dimension,
                "indexing_seconds": stats.indexing_seconds,
                "memory_bytes": stats.memory_bytes,
                "model_storage_bytes": stats.model_storage_bytes,
                "index_bytes": stats.index_bytes,
                "cache_hit": stats.cache_hit,
            }
        manifest["models"][model_key] = {**config, "representations": model_rows}

    manifest_path = artifact_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--skip-dense", action="store_true", help="Build sparse indexes only (for offline smoke tests).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_indexes(args.processed_dir.resolve(), args.artifact_dir.resolve(), skip_dense=args.skip_dense)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "corpus_counts": manifest["corpus_counts"],
        "representations": manifest["representations"],
        "models": {key: {"name": row["name"], "revision": row["revision"]} for key, row in manifest["models"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
