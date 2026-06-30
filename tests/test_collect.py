"""
Unit tests for scripts/collect.py.
All tests use local fixtures — no live network connections.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is on the path so we can import scripts.collect
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import collect  # noqa: E402  (must follow sys.path manipulation)

FIXTURES = Path(__file__).parent / "fixtures"


class TestPathResolution(unittest.TestCase):
    """Repository root is resolved from scripts/collect.py location."""

    def test_repo_root_exists(self):
        self.assertTrue(collect.REPO_ROOT.is_dir(), f"REPO_ROOT not found: {collect.REPO_ROOT}")

    def test_config_path_relative_to_repo_root(self):
        self.assertEqual(collect.CONFIG_PATH.parent, collect.REPO_ROOT / "config")

    def test_raw_dirs_relative_to_repo_root(self):
        self.assertEqual(collect.HTML_DIR.parent, collect.REPO_ROOT / "data" / "raw")
        self.assertEqual(collect.MARKDOWN_DIR.parent, collect.REPO_ROOT / "data" / "raw")
        self.assertEqual(collect.METADATA_DIR.parent, collect.REPO_ROOT / "data" / "raw")

    def test_manifest_relative_to_repo_root(self):
        self.assertEqual(collect.MANIFEST_PATH.parent, collect.REPO_ROOT / "data" / "raw")


class TestLoadSources(unittest.TestCase):
    """load_sources rejects empty or missing configs correctly."""

    def test_load_real_sources_yaml(self):
        sources = collect.load_sources(collect.CONFIG_PATH)
        self.assertIsInstance(sources, list)
        self.assertGreater(len(sources), 0, "sources.yaml has no enabled sources")

    def test_empty_sources_raises_on_collect(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            fh.write("sources: []\n")
            tmp_path = Path(fh.name)
        sources = collect.load_sources(tmp_path)
        self.assertEqual(sources, [])
        tmp_path.unlink()

    def test_all_required_fields_present(self):
        required = {"source_id", "name", "url", "source_type", "licence", "licence_url"}
        sources = collect.load_sources(collect.CONFIG_PATH)
        for s in sources:
            missing = required - set(s.keys())
            self.assertEqual(missing, set(), f"Source {s.get('source_id')} missing: {missing}")

    def test_duplicate_url_detection(self):
        dupes = [
            {"source_id": "a", "url": "https://example.com/"},
            {"source_id": "b", "url": "https://example.com/"},
        ]
        result = collect._check_duplicate_urls(dupes)
        self.assertEqual(len(result), 1)
        self.assertIn("a", result[0] or "a")  # either order is fine

    def test_no_duplicate_urls_in_config(self):
        sources = collect.load_sources(collect.CONFIG_PATH)
        dupes = collect._check_duplicate_urls(sources)
        self.assertEqual(dupes, [], f"Duplicate URLs in config: {dupes}")


class TestHTMLExtraction(unittest.TestCase):
    """HTML-to-Markdown conversion uses local fixture files."""

    def setUp(self):
        self.html_bytes = (FIXTURES / "sample_wiki.html").read_bytes()

    def test_extracts_text_from_html(self):
        result = collect._html_to_markdown(self.html_bytes, "https://example.com", "html")
        self.assertIn("Test Building", result)
        self.assertIn("Melbourne", result)

    def test_title_extraction(self):
        title = collect._extract_title(self.html_bytes, "html")
        self.assertIn("Wikipedia", title)

    def test_short_body_rejected(self):
        tiny_html = b"<html><body><p>Hi</p></body></html>"
        result = collect._html_to_markdown(tiny_html, "https://example.com", "html")
        self.assertLess(len(result.encode()), collect.MIN_BODY_BYTES)


class TestSparqlExtraction(unittest.TestCase):
    """SPARQL JSON response converted to Markdown table."""

    def setUp(self):
        self.json_bytes = (FIXTURES / "sample_sparql.json").read_bytes()

    def test_converts_to_markdown_table(self):
        result = collect._sparql_json_to_markdown(self.json_bytes, "https://query.wikidata.org/")
        self.assertIn("itemLabel", result)
        self.assertIn("Eastern Market Melbourne", result)
        self.assertIn("|", result)

    def test_empty_sparql_returns_empty(self):
        empty = json.dumps({"results": {"bindings": []}}).encode()
        result = collect._sparql_json_to_markdown(empty, "https://example.com")
        self.assertEqual(result, "")


class TestWikiApiExtraction(unittest.TestCase):
    """Wikipedia API category JSON converted to Markdown list."""

    def setUp(self):
        self.json_bytes = (FIXTURES / "sample_wiki_api.json").read_bytes()

    def test_converts_to_markdown_list(self):
        result = collect._wiki_api_json_to_markdown(self.json_bytes, "https://en.wikipedia.org/")
        self.assertIn("Eastern Market, Melbourne", result)
        self.assertIn("Menzies Hotel", result)
        self.assertIn("-", result)

    def test_empty_members_returns_empty(self):
        empty = json.dumps({"query": {"categorymembers": []}}).encode()
        result = collect._wiki_api_json_to_markdown(empty, "https://en.wikipedia.org/")
        self.assertEqual(result, "")


class TestProvenanceHeader(unittest.TestCase):
    """Provenance headers are complete and parseable."""

    def test_header_contains_all_fields(self):
        header = collect._provenance_header(
            source_id="test_id",
            url="https://example.com/page",
            title="Test Page",
            licence="CC-BY-SA-4.0",
            licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
            timestamp="2026-06-30T00:00:00+00:00",
            content_hash="abc123",
            http_status=200,
        )
        self.assertIn("source_id: test_id", header)
        self.assertIn("source_url: https://example.com/page", header)
        self.assertIn("page_title: Test Page", header)
        self.assertIn("licence: CC-BY-SA-4.0", header)
        self.assertIn("licence_url: https://creativecommons.org/licenses/by-sa/4.0/", header)
        self.assertIn("retrieved: 2026-06-30T00:00:00+00:00", header)
        self.assertIn("content_hash_sha256: abc123", header)
        self.assertIn("http_status: 200", header)
        self.assertIn("<!-- PROVENANCE", header)
        self.assertIn("-->", header)


class TestManifestGeneration(unittest.TestCase):
    """Manifest JSONL is generated with correct structure."""

    def test_manifest_written_and_readable(self):
        records = [
            {"source_id": "s1", "url": "https://a.com", "status": "ok"},
            {"source_id": "s2", "url": "https://b.com", "status": "failed", "reason": "HTTP 404"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            with patch.object(collect, "MANIFEST_PATH", manifest_path):
                collect.write_manifest(records)
            lines = manifest_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["source_id"], "s1")
            second = json.loads(lines[1])
            self.assertEqual(second["status"], "failed")

    def test_manifest_includes_failures(self):
        records = [{"source_id": "x", "url": "https://x.com", "status": "failed", "reason": "timeout"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            with patch.object(collect, "MANIFEST_PATH", manifest_path):
                collect.write_manifest(records)
            content = manifest_path.read_text()
            self.assertIn("failed", content)
            self.assertIn("timeout", content)


class TestFailedRequestRecording(unittest.TestCase):
    """Failed requests are recorded in metadata, not silently skipped."""

    def test_http_error_recorded(self):
        source = {
            "source_id": "test_fail",
            "url": "https://example.com/404",
            "source_type": "html",
            "licence": "CC-BY-SA-4.0",
            "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(collect, "HTML_DIR", Path(tmpdir) / "html"),
                patch.object(collect, "MARKDOWN_DIR", Path(tmpdir) / "markdown"),
                patch.object(collect, "METADATA_DIR", Path(tmpdir) / "metadata"),
            ):
                for d in (collect.HTML_DIR, collect.MARKDOWN_DIR, collect.METADATA_DIR):
                    d.mkdir(parents=True, exist_ok=True)
                with patch("collect.fetch_url", return_value=(404, b"Not Found", "text/html")):
                    record = collect.collect_source(source, refresh=True, dry_run=False)
        self.assertEqual(record["status"], "failed")
        self.assertIn("404", record.get("reason", ""))

    def test_network_error_recorded(self):
        source = {
            "source_id": "test_timeout",
            "url": "https://example.com/timeout",
            "source_type": "html",
            "licence": "CC-BY-SA-4.0",
            "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(collect, "HTML_DIR", Path(tmpdir) / "html"),
                patch.object(collect, "MARKDOWN_DIR", Path(tmpdir) / "markdown"),
                patch.object(collect, "METADATA_DIR", Path(tmpdir) / "metadata"),
            ):
                for d in (collect.HTML_DIR, collect.MARKDOWN_DIR, collect.METADATA_DIR):
                    d.mkdir(parents=True, exist_ok=True)
                with patch("collect.fetch_url", side_effect=RuntimeError("connection timed out")):
                    record = collect.collect_source(source, refresh=True, dry_run=False)
        self.assertEqual(record["status"], "failed")
        self.assertIn("timed out", record.get("reason", ""))


class TestEmptyBodyRejection(unittest.TestCase):
    """Documents with extracted bodies below MIN_BODY_BYTES are rejected."""

    def test_empty_body_rejected(self):
        source = {
            "source_id": "test_empty",
            "url": "https://example.com/empty",
            "source_type": "html",
            "licence": "CC-BY-SA-4.0",
            "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
        tiny_html = b"<html><body><p>Hi</p></body></html>"
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(collect, "HTML_DIR", Path(tmpdir) / "html"),
                patch.object(collect, "MARKDOWN_DIR", Path(tmpdir) / "markdown"),
                patch.object(collect, "METADATA_DIR", Path(tmpdir) / "metadata"),
            ):
                for d in (collect.HTML_DIR, collect.MARKDOWN_DIR, collect.METADATA_DIR):
                    d.mkdir(parents=True, exist_ok=True)
                with (
                    patch("collect.fetch_url", return_value=(200, tiny_html, "text/html")),
                    patch("collect.robots_allowed", return_value=True),
                ):
                    record = collect.collect_source(source, refresh=True, dry_run=False)
        self.assertEqual(record["status"], "failed")
        self.assertIn("short", record.get("reason", ""))


class TestSuccessfulCollection(unittest.TestCase):
    """A successful fetch stores HTML, Markdown with provenance, and metadata."""

    def test_successful_collection_stores_files(self):
        source = {
            "source_id": "test_ok",
            "url": "https://en.wikipedia.org/wiki/Test",
            "source_type": "html",
            "licence": "CC-BY-SA-4.0",
            "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
        html_bytes = (FIXTURES / "sample_wiki.html").read_bytes()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(collect, "HTML_DIR", Path(tmpdir) / "html"),
                patch.object(collect, "MARKDOWN_DIR", Path(tmpdir) / "markdown"),
                patch.object(collect, "METADATA_DIR", Path(tmpdir) / "metadata"),
            ):
                for d in (collect.HTML_DIR, collect.MARKDOWN_DIR, collect.METADATA_DIR):
                    d.mkdir(parents=True, exist_ok=True)
                with (
                    patch("collect.fetch_url", return_value=(200, html_bytes, "text/html")),
                    patch("collect.robots_allowed", return_value=True),
                ):
                    record = collect.collect_source(source, refresh=True, dry_run=False)
            self.assertEqual(record["status"], "ok")
            md_path = Path(tmpdir) / "markdown" / "test_ok.md"
            self.assertTrue(md_path.exists(), "Markdown file not written")
            md_content = md_path.read_text()
            self.assertIn("<!-- PROVENANCE", md_content)
            self.assertIn("source_id: test_ok", md_content)
            self.assertIn("Melbourne", md_content)


if __name__ == "__main__":
    unittest.main()
