"""Offline validation for the Phase 2 historical knowledge fabric."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_corpus as prepare  # noqa: E402


HISTORICAL_PARAGRAPH = (
    "Eastern Market was a public market in Melbourne, Victoria. The Eastern Market was "
    "constructed in 1847 and opened in the same year. It was designed by Charles Webb. "
    "The market stood near Bourke Street and served residents, traders, and visitors for "
    "many decades. Historical accounts describe produce stalls, public gatherings, and "
    "changes to the surrounding streets. In 1960 the Eastern Market was demolished after "
    "a long period of declining trade. The date is stated by the source and is not inferred. "
)


def make_source(raw: Path, source_id: str, licence: str = "CC-BY-SA-4.0", body: str | None = None) -> None:
    for name in ("html", "markdown", "metadata"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    html = f"<html><title>{source_id}</title><body>immutable {source_id}</body></html>".encode()
    digest = hashlib.sha256(html).hexdigest()
    timestamp = "2026-06-30T00:00:00+00:00"
    url = f"https://example.test/{source_id}"
    title = source_id.replace("_", " ").title()
    content = body if body is not None else "# Overview\n\n" + HISTORICAL_PARAGRAPH * 3
    header = (
        "<!-- PROVENANCE\n"
        f"source_id: {source_id}\n"
        f"source_url: {url}\n"
        f"page_title: {title}\n"
        f"licence: {licence}\n"
        "licence_url: https://creativecommons.org/licenses/by-sa/4.0/\n"
        f"retrieved: {timestamp}\n"
        f"content_hash_sha256: {digest}\n"
        "http_status: 200\n"
        "-->\n\n"
    )
    (raw / "html" / f"{source_id}.html").write_bytes(html)
    (raw / "markdown" / f"{source_id}.md").write_text(header + content, encoding="utf-8")
    metadata = {
        "source_id": source_id,
        "url": url,
        "status": "ok",
        "http_status": 200,
        "page_title": title,
        "licence": licence,
        "timestamp": timestamp,
        "content_hash_sha256": digest,
    }
    (raw / "metadata" / f"{source_id}.json").write_text(json.dumps(metadata), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestPhase2Pipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        make_source(
            self.raw,
            "eastern_market",
            body=(
                "# Eastern Market\n\nCoordinates: -37.8126; 144.9700\n\n"
                "| Location | Melbourne, Victoria |\n\n" + HISTORICAL_PARAGRAPH * 3
            ),
        )
        make_source(self.raw, "hotel_windsor", body="# Hotel Windsor\n\n" + HISTORICAL_PARAGRAPH * 3)
        make_source(
            self.raw,
            "restricted_source",
            licence="REVIEW-NEEDED",
            body="# Restricted history\n\n" + HISTORICAL_PARAGRAPH * 3,
        )
        self.processed = self.root / "processed"
        self.reports = self.root / "reports"
        self.fingerprint = self.reports / "corpus_fingerprint.json"
        self.before = prepare.snapshot_tree(self.raw)
        self.summary = prepare.run_pipeline(
            raw_dir=self.raw,
            processed_dir=self.processed,
            reports_dir=self.reports,
            config_path=self.root / "missing.yaml",
            fingerprint_path=self.fingerprint,
            expected_count=3,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_raw_files_remain_unchanged(self):
        self.assertEqual(self.before, prepare.snapshot_tree(self.raw))

    def test_every_document_has_provenance(self):
        documents = read_jsonl(self.processed / "documents.jsonl")
        self.assertEqual(len(documents), 3)
        for document in documents:
            for key in ("source_id", "source_url", "licence", "retrieved_at", "content_hash"):
                self.assertTrue(document[key])

    def test_passages_map_to_one_document_and_are_approximately_sized(self):
        document_ids = {row["document_id"] for row in read_jsonl(self.processed / "documents.jsonl")}
        passages = read_jsonl(self.processed / "passages.jsonl")
        self.assertTrue(passages)
        for passage in passages:
            self.assertIn(passage["document_id"], document_ids)
            self.assertLessEqual(prepare.word_count(passage["text"]), prepare.MAX_PASSAGE_WORDS)

    def test_all_graph_records_have_supporting_passages(self):
        passage_ids = {row["passage_id"] for row in read_jsonl(self.processed / "passages.jsonl")}
        for filename in ("entities.jsonl", "events.jsonl", "relations.jsonl"):
            for row in read_jsonl(self.processed / filename):
                self.assertTrue(row["supporting_passage_ids"])
                self.assertLessEqual(set(row["supporting_passage_ids"]), passage_ids)

    def test_every_claim_has_an_exact_valid_source_span(self):
        passages = {row["passage_id"]: row for row in read_jsonl(self.processed / "passages.jsonl")}
        claims = read_jsonl(self.processed / "claims.jsonl")
        self.assertTrue(claims)
        self.assertTrue(all(prepare.valid_supporting_span(claim, passages) for claim in claims))

    def test_unsupported_claim_is_rejected_by_validator(self):
        passage = {"passage_id": "pass_1", "text": "The market opened in 1847."}
        claim = {
            "passage_id": "pass_1",
            "supporting_text": "The market opened in 1900.",
            "supporting_span": {"start": 0, "end": 26},
        }
        self.assertFalse(prepare.valid_supporting_span(claim, {"pass_1": passage}))

    def test_document_splits_do_not_overlap(self):
        manifest = json.loads((self.processed / "split_manifest.json").read_text())
        ids = [row["document_id"] for row in manifest["assignments"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(manifest["counts"]), {"train", "validation", "test"})

    def test_incompatible_licence_is_excluded_from_training(self):
        balance = (self.reports / "source_balance.csv").read_text(encoding="utf-8")
        self.assertIn("restricted_source,REVIEW-NEEDED,False", balance)
        exclusions = read_jsonl(self.processed / "exclusions.jsonl")
        self.assertTrue(any(row["category"] == "training_licence_exclusion" for row in exclusions))

    def test_geojson_coordinates_are_valid(self):
        geojson = json.loads((self.processed / "places.geojson").read_text())
        self.assertTrue(geojson["features"])
        for feature in geojson["features"]:
            longitude, latitude = feature["geometry"]["coordinates"]
            self.assertTrue(-180 <= longitude <= 180)
            self.assertTrue(-90 <= latitude <= 90)

    def test_fingerprint_contains_identity_not_content(self):
        fingerprint = json.loads(self.fingerprint.read_text())
        self.assertEqual(fingerprint["corpus_file_count"], 3)
        self.assertNotIn("cleaned_text", json.dumps(fingerprint))

    def test_expected_outputs_and_reports_exist(self):
        outputs = {
            "documents.jsonl", "passages.jsonl", "entities.jsonl", "relations.jsonl",
            "events.jsonl", "claims.jsonl", "places.geojson", "exclusions.jsonl",
            "split_manifest.json", "training_corpus.txt",
        }
        self.assertEqual(outputs, {path.name for path in self.processed.iterdir()})
        for report in (
            "data_quality_report.md", "knowledge_fabric_report.md", "source_balance.csv",
            "licence_audit.csv", "conflict_report.md", "corpus_fingerprint.json",
        ):
            self.assertTrue((self.reports / report).exists())

    def test_repeated_execution_is_deterministic(self):
        second_processed = self.root / "processed-second"
        second_reports = self.root / "reports-second"
        prepare.run_pipeline(
            raw_dir=self.raw,
            processed_dir=second_processed,
            reports_dir=second_reports,
            config_path=self.root / "missing.yaml",
            fingerprint_path=second_reports / "corpus_fingerprint.json",
            expected_count=3,
        )
        for first in sorted(self.processed.iterdir()):
            self.assertEqual(first.read_bytes(), (second_processed / first.name).read_bytes())

    def test_command_works_from_different_current_directory(self):
        third_processed = self.root / "processed-third"
        third_reports = self.root / "reports-third"
        command = [
            sys.executable,
            str(prepare.REPO_ROOT / "scripts" / "prepare_corpus.py"),
            "--raw-dir", str(self.raw),
            "--processed-dir", str(third_processed),
            "--reports-dir", str(third_reports),
            "--fingerprint", str(third_reports / "corpus_fingerprint.json"),
            "--config", str(self.root / "missing.yaml"),
            "--expected-count", "3",
        ]
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestFailureModes(unittest.TestCase):
    def test_production_processing_fails_on_empty_input(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            (raw / "markdown").mkdir(parents=True)
            with self.assertRaisesRegex(prepare.PipelineError, "no raw Markdown"):
                prepare.run_pipeline(raw_dir=raw, expected_count=None)

    def test_processing_fails_when_all_bodies_are_unusable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            make_source(raw, "empty_page", body="Navigation\n\nMenu\n")
            with self.assertRaisesRegex(prepare.PipelineError, "all extracted bodies are empty"):
                prepare.run_pipeline(
                    raw_dir=raw,
                    processed_dir=root / "processed",
                    reports_dir=root / "reports",
                    fingerprint_path=root / "reports" / "fingerprint.json",
                    config_path=root / "missing.yaml",
                    expected_count=1,
                )

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            make_source(raw, "tampered")
            (raw / "html" / "tampered.html").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(prepare.PipelineError, "SHA-256"):
                prepare.load_and_validate_raw(raw, expected_count=1)

    def test_missing_provenance_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            make_source(raw, "missing_header")
            (raw / "markdown" / "missing_header.md").write_text("plain text", encoding="utf-8")
            with self.assertRaisesRegex(prepare.PipelineError, "provenance"):
                prepare.load_and_validate_raw(raw, expected_count=1)


class TestVersionedExtractionContract(unittest.TestCase):
    def test_schema_and_prompt_are_versioned(self):
        schema = json.loads((prepare.REPO_ROOT / "config" / "extraction_schema.json").read_text())
        prompt = (prepare.REPO_ROOT / "config" / "extraction_prompt.md").read_text()
        self.assertEqual(schema["version"], "1.0.0")
        self.assertIn("Prompt version: `1.0.0`", prompt)
        self.assertIn("Temperature: `0`", prompt)


if __name__ == "__main__":
    unittest.main()
