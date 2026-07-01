#!/usr/bin/env python3
"""Run the retrieval-only answering pipeline for one question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval.answering import RetrievalAnswerer  # noqa: E402


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "retrieval"
QUERY_PATH = REPO_ROOT / "evaluation" / "retrieval_queries.jsonl"
REPORT_PATH = REPO_ROOT / "reports" / "retrieval_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to run against the retrieval-only answering pipeline.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--query-path", type=Path, default=QUERY_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--evidence-k", type=int, default=5, help="Number of evidence passages to return (clamped to 3-5).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        answerer = RetrievalAnswerer.from_artifacts(
            processed_dir=args.processed_dir.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            query_path=args.query_path.resolve(),
            report_path=args.report_path.resolve(),
        )
        result = answerer.answer(args.question, evidence_k=args.evidence_k)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
