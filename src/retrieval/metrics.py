"""Information-retrieval metrics with explicit answerability accounting."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any


def evaluate_rankings(records: list[dict[str, Any]]) -> dict[str, float]:
    answerable = [record for record in records if record["expected_answerable"]]
    recalls = {k: [] for k in (1, 3, 5, 10)}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for record in answerable:
        gold = set(record["expected_relevant_passage_ids"])
        ranked = record["ranked_passage_ids"]
        for k in recalls:
            recalls[k].append(float(bool(gold & set(ranked[:k]))))
        first = next((index for index, passage_id in enumerate(ranked, 1) if passage_id in gold), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(1.0 / math.log2(index + 1) for index, passage_id in enumerate(ranked[:10], 1) if passage_id in gold)
        ideal_hits = min(len(gold), 10)
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)

    predicted_unanswerable = [record for record in records if record["abstained"]]
    true_unanswerable = [record for record in records if not record["expected_answerable"]]
    correct_abstentions = sum(not record["expected_answerable"] for record in predicted_unanswerable)
    return {
        **{f"recall_at_{k}": mean(values) if values else 0.0 for k, values in recalls.items()},
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_10": mean(ndcgs) if ndcgs else 0.0,
        "unanswerable_precision": correct_abstentions / len(predicted_unanswerable) if predicted_unanswerable else 0.0,
        "unanswerable_recall": correct_abstentions / len(true_unanswerable) if true_unanswerable else 0.0,
        "zero_result_rate": sum(record["abstained"] for record in records) / len(records) if records else 0.0,
        "query_count": float(len(records)),
    }


def tune_abstention_threshold(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    candidates = sorted({0.0, *[float(row["top_score"]) for row in rows]})
    best = (float("-inf"), 0.0)
    for threshold in candidates:
        answerable = [row for row in rows if row["expected_answerable"]]
        unanswerable = [row for row in rows if not row["expected_answerable"]]
        answerable_acceptance = sum(row["top_score"] >= threshold for row in answerable) / len(answerable) if answerable else 1.0
        unanswerable_rejection = sum(row["top_score"] < threshold for row in unanswerable) / len(unanswerable) if unanswerable else 1.0
        score = (answerable_acceptance + unanswerable_rejection) / 2
        if score > best[0] or (score == best[0] and threshold < best[1]):
            best = (score, threshold)
    return best[1]
