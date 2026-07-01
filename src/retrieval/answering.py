"""Retrieval-only answering pipeline with evidence, reranking, and abstention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import HistoricalCorpus
from .dense import DenseIndex, SentenceTransformerEncoder
from .evaluation import load_queries
from .hybrid import reciprocal_rank_fusion
from .metrics import tune_abstention_threshold
from .models import RetrievalResult, SearchPassage
from .query import QueryTransformer
from .sparse import BM25Index, TfidfIndex
from .structured import StructuredRetriever


FALLBACK_RESPONSE = "The current corpus does not contain enough reliable evidence to answer this question."


@dataclass(frozen=True)
class AnsweringEvidence:
    passage_id: str
    parent_passage_id: str
    title: str
    section_title: str
    text: str
    source_url: str
    licence: str
    score: float
    score_components: dict[str, float]
    explanation: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "parent_passage_id": self.parent_passage_id,
            "title": self.title,
            "section_title": self.section_title,
            "text": self.text,
            "source_url": self.source_url,
            "licence": self.licence,
            "score": self.score,
            "score_components": self.score_components,
            "explanation": list(self.explanation),
        }


class RetrievalAnswerer:
    def __init__(
        self,
        corpus: HistoricalCorpus,
        passages: list[SearchPassage],
        bm25: BM25Index,
        tfidf: TfidfIndex,
        dense: DenseIndex,
        transform_mode: str = "entity",
        candidate_top_k: int = 10,
        lexical_weight: float = 1.0,
        abstention_threshold: float | None = None,
    ):
        self.corpus = corpus
        self.passages = passages
        self.passages_by_id = {passage.passage_id: passage for passage in passages}
        self.bm25 = bm25
        self.tfidf = tfidf
        self.dense = dense
        self.transform_mode = transform_mode
        self.candidate_top_k = candidate_top_k
        self.lexical_weight = lexical_weight
        self.query_transformer = QueryTransformer(corpus)
        self.structured = StructuredRetriever(corpus)
        self.abstention_threshold = abstention_threshold if abstention_threshold is not None else 0.0

    @classmethod
    def from_artifacts(
        cls,
        processed_dir: Path,
        artifact_dir: Path,
        query_path: Path | None = None,
        report_path: Path | None = None,
    ) -> "RetrievalAnswerer":
        import joblib

        corpus = HistoricalCorpus.load(processed_dir)
        report = _load_report(report_path) if report_path else {}
        passage_strategy = report.get("passage_strategy_winner", "original")
        model_key = report.get("model_winner", "bge-small")
        transform_mode = report.get("query_transform_winner", "entity")
        if passage_strategy != "original":
            raise ValueError(f"answering pipeline requires original passages, found: {passage_strategy}")
        passages = _build_original_passages(corpus)
        bm25 = joblib.load(artifact_dir / "original" / "bm25.joblib")
        tfidf = joblib.load(artifact_dir / "original" / "tfidf.joblib")
        dense, _ = DenseIndex.build(
            passages,
            SentenceTransformerEncoder(model_key),
            artifact_dir / "original" / f"dense_{model_key}.npz",
        )
        answerer = cls(
            corpus=corpus,
            passages=passages,
            bm25=bm25,
            tfidf=tfidf,
            dense=dense,
            transform_mode=transform_mode,
            candidate_top_k=10,
            lexical_weight=1.0,
        )
        if query_path:
            answerer.abstention_threshold = answerer.calibrate_from_queries(query_path)
        return answerer

    def calibrate_from_queries(self, query_path: Path) -> float:
        queries = load_queries(query_path, set(self.corpus.passage_by_id), set(self.corpus.entity_by_id))
        return self.calibrate_from_rows(queries)

    def calibrate_from_rows(self, queries: list[dict[str, Any]]) -> float:
        development_rows: list[dict[str, Any]] = []
        for query in queries:
            if query.get("evaluation_split", "development") != "development":
                continue
            run = self._run(query["question"])
            development_rows.append({
                "top_score": run["top_score"],
                "expected_answerable": query["expected_answerable"],
            })
        self.abstention_threshold = tune_abstention_threshold(development_rows)
        return self.abstention_threshold

    def answer(self, query: str, evidence_k: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        run = self._run(query)
        evidence_limit = min(5, max(3, evidence_k))
        abstained = run["top_score"] < self.abstention_threshold
        evidence = [] if abstained else [row.as_dict() for row in run["evidence"][:evidence_limit]]
        return {
            "query": query,
            "transformed_query": run["transformed_query"],
            "query_expansions": list(run["query_expansions"]),
            "detected_entity_ids": list(run["detected_entity_ids"]),
            "structured_filters": list(run["structured_filters"]),
            "candidate_strategy": "hybrid_structured_top_10",
            "reranking_strategy": "tfidf_lexical_contribution",
            "abstention_threshold": self.abstention_threshold,
            "top_score": run["top_score"],
            "abstained": abstained,
            "fallback_response": FALLBACK_RESPONSE if abstained else None,
            "returned_evidence": evidence,
            "candidate_results": [self._result_to_dict(result) for result in run["candidate_results"]],
            "reranked_results": [self._result_to_dict(result) for result in run["reranked_results"]],
        }

    def _run(self, query: str) -> dict[str, Any]:
        transformed = self.query_transformer.transform(query, self.transform_mode)
        search_query = transformed.transformed
        hybrid = reciprocal_rank_fusion(search_query, self.passages, {
            "bm25": self.bm25.search(search_query, 50),
            "dense": self.dense.search(search_query, 50),
        }, top_k=self.candidate_top_k)
        structured_results, signal = self.structured.rerank(query, hybrid, self.passages, top_k=self.candidate_top_k)
        reranked = self._rerank_with_lexical(search_query, structured_results)
        evidence = [self._to_evidence(result) for result in reranked]
        return {
            "transformed_query": search_query,
            "query_expansions": transformed.expansions,
            "detected_entity_ids": transformed.detected_entity_ids,
            "structured_filters": signal.filters,
            "candidate_results": structured_results,
            "reranked_results": reranked,
            "evidence": evidence,
            "top_score": reranked[0].score if reranked else 0.0,
        }

    def _rerank_with_lexical(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        lexical_scores = self.tfidf.score_by_passage_id(query)
        reranked: list[RetrievalResult] = []
        for result in candidates:
            lexical = self.lexical_weight * lexical_scores.get(result.passage_id, 0.0)
            components = dict(result.score_components)
            components["tfidf_lexical"] = lexical
            reranked.append(RetrievalResult(
                passage_id=result.passage_id,
                parent_passage_id=result.parent_passage_id,
                score=result.score + lexical,
                score_components=components,
                explanation=[*result.explanation, f"tfidf lexical contribution={lexical:.6f}"],
            ))
        reranked.sort(key=lambda result: (-result.score, result.passage_id))
        for rank, result in enumerate(reranked, 1):
            result.rank = rank
        return reranked

    def _to_evidence(self, result: RetrievalResult) -> AnsweringEvidence:
        passage = self.passages_by_id[result.passage_id]
        return AnsweringEvidence(
            passage_id=result.passage_id,
            parent_passage_id=result.parent_passage_id,
            title=passage.title,
            section_title=passage.section_title,
            text=passage.text,
            source_url=str(passage.metadata.get("source_url", "")),
            licence=str(passage.metadata.get("licence", "")),
            score=result.score,
            score_components=dict(result.score_components),
            explanation=tuple(result.explanation),
        )

    @staticmethod
    def _result_to_dict(result: RetrievalResult) -> dict[str, Any]:
        return {
            "passage_id": result.passage_id,
            "parent_passage_id": result.parent_passage_id,
            "rank": result.rank,
            "score": result.score,
            "score_components": dict(result.score_components),
            "explanation": list(result.explanation),
        }


def _build_original_passages(corpus: HistoricalCorpus) -> list[SearchPassage]:
    from .chunking import build_passages

    return build_passages(corpus, "original")


def _load_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
