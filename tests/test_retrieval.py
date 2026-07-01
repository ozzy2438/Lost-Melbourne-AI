"""Offline tests for Phase 3 indexing, filtering, and evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval import (  # noqa: E402
    BM25Index,
    DenseIndex,
    FALLBACK_RESPONSE,
    HashingEncoder,
    HistoricalCorpus,
    QueryTransformer,
    RetrievalAnswerer,
    StructuredRetriever,
    TfidfIndex,
    build_passages,
    reciprocal_rank_fusion,
)
from retrieval.corpus import CorpusValidationError  # noqa: E402
from retrieval.metrics import evaluate_rankings, tune_abstention_threshold  # noqa: E402
from retrieval.models import SearchPassage  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    evidence = "Eastern Market was demolished in 1960."
    passage_texts = [
        "Eastern Market, also known as Paddys Market, was a public market in Melbourne. " + evidence + " It sold produce and hosted public gatherings. " * 35,
        "Hotel Windsor is located on Spring Street in Melbourne and was designed by Charles Webb. " * 35,
        "Port Melbourne is an inner suburb south-west of central Melbourne. " * 35,
    ]
    documents = [
        {"document_id": f"doc_{index}", "source_id": f"source_{index}", "title": title, "source_url": f"https://example.test/{index}", "licence": "CC-BY-4.0", "retrieved_at": "2026-01-01", "content_hash": str(index) * 64, "cleaned_text": text, "word_count": len(text.split()), "approximate_token_count": len(text.split()) * 2}
        for index, (title, text) in enumerate(zip(("Eastern Market", "Hotel Windsor", "Port Melbourne"), passage_texts), 1)
    ]
    passages = [
        {"passage_id": f"pass_{index}", "document_id": f"doc_{index}", "section_title": "Overview", "passage_index": 0, "text": text, "source_url": f"https://example.test/{index}", "licence": "CC-BY-4.0", "character_start": 0, "character_end": len(text)}
        for index, text in enumerate(passage_texts, 1)
    ]
    entities = [
        {"entity_id": "ent_market", "canonical_name": "Eastern Market", "entity_type": "market", "aliases": ["Paddys Market"], "description": None, "supporting_passage_ids": ["pass_1"], "confidence": 1.0, "extraction_method": "fixture"},
        {"entity_id": "ent_melbourne", "canonical_name": "Melbourne", "entity_type": "place", "aliases": [], "description": None, "supporting_passage_ids": ["pass_1", "pass_2", "pass_3"], "confidence": 1.0, "extraction_method": "fixture"},
        {"entity_id": "ent_hotel", "canonical_name": "Hotel Windsor", "entity_type": "hotel", "aliases": [], "description": None, "supporting_passage_ids": ["pass_2"], "confidence": 1.0, "extraction_method": "fixture"},
    ]
    span_start = passage_texts[0].index(evidence)
    events = [{"event_id": "evt_1", "event_type": "demolition", "involved_entity_ids": ["ent_market"], "date_original": "1960", "normalised_date": {"start_year": 1960, "end_year": 1960, "precision": "year"}, "location": "Melbourne", "supporting_passage_ids": ["pass_1"], "supporting_text": evidence, "supporting_span": {"start": span_start, "end": span_start + len(evidence)}, "confidence": 1.0, "uncertainty_notes": None, "extraction_method": "fixture"}]
    relations = [{"relation_id": "rel_1", "relation_type": "LOCATED_IN", "subject_entity_id": "ent_market", "object_entity_id": "ent_melbourne", "object_value": None, "supporting_passage_ids": ["pass_1"], "supporting_text": passage_texts[0][:80], "supporting_span": {"start": 0, "end": 80}, "confidence": 1.0, "extraction_method": "fixture"}]
    claims = [{"claim_id": "claim_1", "subject": {"entity_id": "ent_market", "name": "Eastern Market"}, "predicate": "was_demolished", "object_or_value": "1960", "supporting_text": evidence, "supporting_span": {"start": span_start, "end": span_start + len(evidence)}, "passage_id": "pass_1", "source_id": "source_1", "confidence": 1.0, "extraction_method": "fixture", "temporal_qualifier": "1960", "geographic_qualifier": "Melbourne", "conflict_group": None}]
    for name, rows in (("documents.jsonl", documents), ("passages.jsonl", passages), ("entities.jsonl", entities), ("events.jsonl", events), ("relations.jsonl", relations), ("claims.jsonl", claims)):
        write_jsonl(root / name, rows)
    (root / "places.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "id": "ent_market", "geometry": {"type": "Point", "coordinates": [144.97, -37.81]}, "properties": {"canonical_entity_id": "ent_market", "name": "Eastern Market", "entity_type": "market", "address": None, "suburb": "Melbourne", "coordinate_source": "fixture", "coordinate_confidence": 1.0, "location_precision": "point", "supporting_passage_ids": ["pass_1"]}}]}), encoding="utf-8")
    (root / "split_manifest.json").write_text(json.dumps({"assignments": [{"document_id": f"doc_{index}", "source_id": f"source_{index}", "split": split} for index, split in enumerate(("train", "validation", "test"), 1)]}), encoding="utf-8")
    return root


class RetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.processed = make_fixture(self.root / "processed")
        self.corpus = HistoricalCorpus.load(self.processed)
        self.passages = build_passages(self.corpus)

    def tearDown(self):
        self.temp.cleanup()

    def test_phase2_integrity_validation(self):
        self.assertEqual(self.corpus.counts()["passages"], 3)
        broken = json.loads((self.processed / "split_manifest.json").read_text())
        broken["assignments"][0]["document_id"] = "missing"
        (self.processed / "split_manifest.json").write_text(json.dumps(broken))
        with self.assertRaises(CorpusValidationError):
            HistoricalCorpus.load(self.processed)

    def test_deterministic_indexing_and_score_order(self):
        first = BM25Index(self.passages).search("When was Eastern Market demolished?", 3)
        second = BM25Index(self.passages).search("When was Eastern Market demolished?", 3)
        self.assertEqual([(row.passage_id, row.score) for row in first], [(row.passage_id, row.score) for row in second])
        self.assertEqual(first[0].passage_id, "pass_1")
        self.assertEqual([row.score for row in first], sorted((row.score for row in first), reverse=True))

    def test_tfidf_score_order(self):
        results = TfidfIndex(self.passages).search("Hotel Windsor architect", 3)
        self.assertEqual(results[0].passage_id, "pass_2")
        self.assertGreaterEqual(results[0].score, results[-1].score)

    def test_stable_small_passage_ids(self):
        first = build_passages(self.corpus, "small")
        second = build_passages(self.corpus, "small")
        self.assertEqual([row.passage_id for row in first], [row.passage_id for row in second])
        self.assertTrue(all(row.parent_passage_id.startswith("pass_") for row in first))

    def test_valid_embedding_dimensions_and_cosine_order(self):
        encoder = HashingEncoder(48)
        index, stats = DenseIndex.build(self.passages, encoder)
        self.assertEqual(index.vectors.shape, (3, 48))
        self.assertEqual(stats.dimension, 48)
        np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), np.ones(3), atol=1e-6)
        results = index.search("Eastern Market demolition", 3)
        self.assertEqual([row.score for row in results], sorted((row.score for row in results), reverse=True))

    def test_duplicate_result_prevention(self):
        bm25 = BM25Index(self.passages).search("Melbourne market", 3)
        dense, _ = DenseIndex.build(self.passages, HashingEncoder())
        fused = reciprocal_rank_fusion("Melbourne market", self.passages, {"a": bm25, "b": dense.search("Melbourne market", 3)}, 10)
        self.assertEqual(len({row.passage_id for row in fused}), len(fused))

    def test_structured_date_filter(self):
        result = StructuredRetriever(self.corpus).passages_for_date_range("demolition", 1950, 1980)
        self.assertEqual(result, {"pass_1"})

    def test_structured_geographic_filter(self):
        result = StructuredRetriever(self.corpus).passages_for_place("Melbourne")
        self.assertIn("pass_1", result)

    def test_alias_expansion(self):
        transform = QueryTransformer(self.corpus).transform("What happened to Paddys Market?", "alias")
        self.assertIn("Eastern Market", transform.expansions)
        self.assertIn("ent_market", transform.detected_entity_ids)

    def test_unanswerable_threshold_and_metrics(self):
        development = [
            {"top_score": 0.9, "expected_answerable": True},
            {"top_score": 0.8, "expected_answerable": True},
            {"top_score": 0.1, "expected_answerable": False},
            {"top_score": 0.2, "expected_answerable": False},
        ]
        threshold = tune_abstention_threshold(development)
        self.assertGreater(threshold, 0.2)
        rows = [{"expected_answerable": False, "abstained": True, "expected_relevant_passage_ids": [], "ranked_passage_ids": []}]
        self.assertEqual(evaluate_rankings(rows)["unanswerable_precision"], 1.0)

    def test_answering_pipeline_returns_evidence(self):
        dense, _ = DenseIndex.build(self.passages, HashingEncoder())
        answerer = RetrievalAnswerer(
            corpus=self.corpus,
            passages=self.passages,
            bm25=BM25Index(self.passages),
            tfidf=TfidfIndex(self.passages),
            dense=dense,
        )
        answerer.abstention_threshold = 0.0
        result = answerer.answer("When was Eastern Market demolished?", evidence_k=5)
        self.assertFalse(result["abstained"])
        self.assertEqual(result["candidate_strategy"], "hybrid_structured_top_10")
        self.assertEqual(result["reranking_strategy"], "tfidf_lexical_contribution")
        self.assertGreaterEqual(len(result["returned_evidence"]), 1)
        self.assertIn("tfidf_lexical", result["returned_evidence"][0]["score_components"])

    def test_answering_pipeline_uses_fallback_when_abstaining(self):
        dense, _ = DenseIndex.build(self.passages, HashingEncoder())
        answerer = RetrievalAnswerer(
            corpus=self.corpus,
            passages=self.passages,
            bm25=BM25Index(self.passages),
            tfidf=TfidfIndex(self.passages),
            dense=dense,
        )
        answerer.abstention_threshold = 999.0
        result = answerer.answer("Who curated the moon palace?", evidence_k=5)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["fallback_response"], FALLBACK_RESPONSE)
        self.assertEqual(result["returned_evidence"], [])

    def test_answering_pipeline_calibrates_threshold_from_development_split(self):
        dense, _ = DenseIndex.build(self.passages, HashingEncoder())
        answerer = RetrievalAnswerer(
            corpus=self.corpus,
            passages=self.passages,
            bm25=BM25Index(self.passages),
            tfidf=TfidfIndex(self.passages),
            dense=dense,
        )
        threshold = answerer.calibrate_from_rows([
            {
                "query_id": "q001",
                "question": "When was Eastern Market demolished?",
                "query_type": "temporal_fact",
                "expected_relevant_passage_ids": ["pass_1"],
                "expected_entity_ids": ["ent_market"],
                "expected_answerable": True,
                "difficulty": "easy",
                "annotation_notes": "fixture",
                "evaluation_split": "development",
            },
            {
                "query_id": "q002",
                "question": "Who designed Hotel Windsor?",
                "query_type": "architect_fact",
                "expected_relevant_passage_ids": ["pass_2"],
                "expected_entity_ids": ["ent_hotel"],
                "expected_answerable": True,
                "difficulty": "easy",
                "annotation_notes": "fixture",
                "evaluation_split": "development",
            },
            {
                "query_id": "q003",
                "question": "Who curated the moon palace?",
                "query_type": "unanswerable",
                "expected_relevant_passage_ids": [],
                "expected_entity_ids": [],
                "expected_answerable": False,
                "difficulty": "hard",
                "annotation_notes": "fixture",
                "evaluation_split": "development",
            },
            {
                "query_id": "q004",
                "question": "Was Eastern Market in Melbourne?",
                "query_type": "location_fact",
                "expected_relevant_passage_ids": ["pass_1"],
                "expected_entity_ids": ["ent_market", "ent_melbourne"],
                "expected_answerable": True,
                "difficulty": "easy",
                "annotation_notes": "fixture",
                "evaluation_split": "test",
            },
        ])
        self.assertGreater(threshold, 0.0)

    def test_sparse_build_from_different_working_directory_without_network(self):
        artifact = self.root / "artifacts"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_retrieval_indexes.py"),
            "--processed-dir", str(self.processed),
            "--artifact-dir", str(artifact),
            "--skip-dense",
        ]
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((artifact / "build_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
