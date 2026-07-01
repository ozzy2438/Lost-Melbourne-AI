"""Evaluation-set loading, ranking execution, trace creation, and failure labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .metrics import evaluate_rankings, tune_abstention_threshold
from .models import RetrievalResult


def load_queries(path: Path, valid_passage_ids: set[str], valid_entity_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "query_id", "question", "query_type", "expected_relevant_passage_ids",
                "expected_entity_ids", "expected_answerable", "difficulty", "annotation_notes",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"query {line_number} missing fields: {sorted(missing)}")
            if row["query_id"] in seen:
                raise ValueError(f"duplicate query_id: {row['query_id']}")
            seen.add(row["query_id"])
            if not set(row["expected_relevant_passage_ids"]) <= valid_passage_ids:
                raise ValueError(f"query has unknown gold passage: {row['query_id']}")
            if not set(row["expected_entity_ids"]) <= valid_entity_ids:
                raise ValueError(f"query has unknown gold entity: {row['query_id']}")
            if row["expected_answerable"] and not row["expected_relevant_passage_ids"]:
                raise ValueError(f"answerable query has no gold passage: {row['query_id']}")
            rows.append(row)
    if not 50 <= len(rows) <= 75:
        raise ValueError(f"evaluation set should contain 50-75 queries, found {len(rows)}")
    return rows


def run_method(
    method_name: str,
    queries: list[dict[str, Any]],
    search: Callable[[str, int], list[RetrievalResult]],
    transform: Callable[[str], Any] | None = None,
    structured: Callable[[str, list[RetrievalResult]], tuple[list[RetrievalResult], Any]] | None = None,
) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    for query in queries:
        transformed = transform(query["question"]) if transform else None
        search_query = transformed.transformed if transformed else query["question"]
        start = time.perf_counter()
        results = search(search_query, 10)
        structured_signal = None
        if structured:
            results, structured_signal = structured(query["question"], results)
        latency_ms = (time.perf_counter() - start) * 1000
        raw_rows.append({
            **query,
            "method": method_name,
            "original_query": query["question"],
            "transformed_query": search_query,
            "query_expansions": list(transformed.expansions) if transformed else [],
            "detected_entity_ids": list(transformed.detected_entity_ids) if transformed else [],
            "structured_filters": list(structured_signal.filters) if structured_signal else [],
            "results": [result.as_dict() for result in results],
            "ranked_passage_ids": [result.evaluation_id() for result in results],
            "top_score": results[0].score if results else 0.0,
            "latency_ms": latency_ms,
        })

    development = [row for row in raw_rows if row.get("evaluation_split", "development") == "development"]
    threshold = tune_abstention_threshold(development)
    for row in raw_rows:
        row["abstention_threshold"] = threshold
        row["abstained"] = row["top_score"] < threshold
        row["success"] = (
            (not row["expected_answerable"] and row["abstained"])
            or (
                row["expected_answerable"]
                and not row["abstained"]
                and bool(set(row["expected_relevant_passage_ids"]) & set(row["ranked_passage_ids"][:5]))
            )
        )
        row["failure_category"] = None if row["success"] else classify_failure(row)
    test_rows = [row for row in raw_rows if row.get("evaluation_split") == "test"]
    return {
        "method": method_name,
        "abstention_threshold": threshold,
        "metrics": evaluate_rankings(raw_rows),
        "development_metrics": evaluate_rankings(development),
        "test_metrics": evaluate_rankings(test_rows),
        "average_query_latency_ms": mean(row["latency_ms"] for row in raw_rows),
        "traces": raw_rows,
    }


def classify_failure(row: dict[str, Any]) -> str:
    if not row["expected_answerable"]:
        return "insufficient_corpus_coverage" if not row["abstained"] else "none"
    query_type = row.get("query_type", "")
    if "alias" in query_type:
        return "alias_mismatch"
    if "temporal" in query_type or "date" in query_type:
        return "date_filter_failure"
    if "geographic" in query_type or "location" in query_type:
        return "location_ambiguity"
    if row.get("detected_entity_ids"):
        return "semantic_confusion"
    return "lexical_mismatch"
