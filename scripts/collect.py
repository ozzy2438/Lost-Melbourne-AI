#!/usr/bin/env python3
"""
collect.py — Deterministic, polite data collector for Lost Melbourne AI.

Reads sources from config/sources.yaml (relative to repository root).
Saves HTML, Markdown and metadata under data/raw/.
Writes a JSONL manifest with provenance for every attempt.

Usage:
    python3 scripts/collect.py [--refresh] [--dry-run]

Options:
    --refresh   Overwrite already-collected files.
    --dry-run   Print what would be done without fetching anything.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Resolve paths from repository root (parent of scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "sources.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"
HTML_DIR = RAW_DIR / "html"
MARKDOWN_DIR = RAW_DIR / "markdown"
METADATA_DIR = RAW_DIR / "metadata"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"

USER_AGENT = (
    "LostMelbourneAI-Collector/1.0 "
    "(educational research; github.com/ozzy2438/Lost-Melbourne-AI; "
    "osmanorka@gmail.com)"
)
REQUEST_TIMEOUT = 20          # seconds per request
MAX_RETRIES = 3
RETRY_BACKOFF = 4             # seconds between retries
MIN_DELAY_SAME_HOST = 2.0     # seconds between requests to the same host
MIN_BODY_BYTES = 100          # reject extracted bodies shorter than this

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect")


# ---------------------------------------------------------------------------
# robots.txt helpers
# ---------------------------------------------------------------------------
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _get_robots(base_url: str) -> urllib.robotparser.RobotFileParser | None:
    if base_url in _robots_cache:
        return _robots_cache[base_url]
    robots_url = base_url.rstrip("/") + "/robots.txt"
    try:
        resp = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(resp.text.splitlines())
        _robots_cache[base_url] = rp
        return rp
    except Exception as exc:
        log.warning("Could not fetch robots.txt for %s: %s", base_url, exc)
        _robots_cache[base_url] = None
        return None


def robots_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _get_robots(base)
    if rp is None:
        return True   # assume allowed if robots.txt unavailable
    return rp.can_fetch(USER_AGENT, url)


# ---------------------------------------------------------------------------
# Per-host delay tracking
# ---------------------------------------------------------------------------
_last_fetch: dict[str, float] = {}


def _polite_wait(host: str) -> None:
    now = time.monotonic()
    last = _last_fetch.get(host, 0.0)
    gap = now - last
    if gap < MIN_DELAY_SAME_HOST:
        time.sleep(MIN_DELAY_SAME_HOST - gap)
    _last_fetch[host] = time.monotonic()


# ---------------------------------------------------------------------------
# HTTP fetch with retries
# ---------------------------------------------------------------------------
def fetch_url(url: str) -> tuple[int, bytes, str]:
    """Return (http_status, raw_bytes, content_type)."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        _polite_wait(host)
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            return resp.status_code, resp.content, resp.headers.get("content-type", "")
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {url}: {last_exc}")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _html_to_markdown(html_bytes: bytes, url: str, source_type: str) -> str:
    """Convert raw HTML (or JSON API response) to a plain Markdown string."""
    # Handle API JSON types specially
    if source_type in ("sparql_json",):
        return _sparql_json_to_markdown(html_bytes, url)
    if source_type == "wikipedia_api":
        return _wiki_api_json_to_markdown(html_bytes, url)

    # General HTML → Markdown via markdownify (best-effort)
    try:
        from markdownify import markdownify as md  # type: ignore
        text = html_bytes.decode("utf-8", errors="replace")
        # Strip <script> and <style> blocks first
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        result = md(text, heading_style="ATX", strip=["a", "img"])
        # Collapse excessive blank lines
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        return result
    except ImportError:
        pass

    # Fallback: strip tags manually
    text = html_bytes.decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _sparql_json_to_markdown(data: bytes, url: str) -> str:
    obj = json.loads(data.decode("utf-8", errors="replace"))
    bindings = obj.get("results", {}).get("bindings", [])
    if not bindings:
        return ""
    keys = list(bindings[0].keys())
    rows = [" | ".join(keys), " | ".join(["---"] * len(keys))]
    for row in bindings:
        rows.append(" | ".join(row.get(k, {}).get("value", "") for k in keys))
    return "\n".join(rows)


def _wiki_api_json_to_markdown(data: bytes, url: str) -> str:
    obj = json.loads(data.decode("utf-8", errors="replace"))
    members = obj.get("query", {}).get("categorymembers", [])
    if not members:
        return ""
    lines = ["# Wikipedia Category Members\n"]
    for m in members:
        title = m.get("title", "")
        lines.append(f"- {title}")
    return "\n".join(lines)


def _extract_title(html_bytes: bytes, source_type: str) -> str:
    if source_type in ("sparql_json", "wikipedia_api"):
        return "(API response)"
    text = html_bytes.decode("utf-8", errors="replace")
    m = re.search(r"<title[^>]*>([^<]+)</title>", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "(no title)"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(source_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", source_id)


# ---------------------------------------------------------------------------
# Provenance header
# ---------------------------------------------------------------------------

def _provenance_header(
    source_id: str,
    url: str,
    title: str,
    licence: str,
    licence_url: str,
    timestamp: str,
    content_hash: str,
    http_status: int,
) -> str:
    return (
        f"<!-- PROVENANCE\n"
        f"source_id: {source_id}\n"
        f"source_url: {url}\n"
        f"page_title: {title}\n"
        f"licence: {licence}\n"
        f"licence_url: {licence_url}\n"
        f"retrieved: {timestamp}\n"
        f"content_hash_sha256: {content_hash}\n"
        f"http_status: {http_status}\n"
        f"-->\n\n"
    )


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------

def load_sources(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    sources = cfg.get("sources", []) or []
    enabled = [s for s in sources if s.get("enabled", True)]
    return enabled


def _check_duplicate_urls(sources: list[dict]) -> list[str]:
    seen: dict[str, str] = {}
    dupes = []
    for s in sources:
        url = s["url"]
        sid = s["source_id"]
        if url in seen:
            dupes.append(f"{sid} duplicates {seen[url]} (url={url})")
        else:
            seen[url] = sid
    return dupes


# ---------------------------------------------------------------------------
# Collect one source
# ---------------------------------------------------------------------------

def collect_source(
    source: dict,
    refresh: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_id = source["source_id"]
    url = source["url"]
    source_type = source.get("source_type", "html")
    licence = source.get("licence", "UNKNOWN")
    licence_url = source.get("licence_url", "")

    fname = _safe_filename(source_id)
    html_path = HTML_DIR / f"{fname}.html"
    md_path = MARKDOWN_DIR / f"{fname}.md"
    meta_path = METADATA_DIR / f"{fname}.json"

    # Skip if already collected and refresh not requested
    if not refresh and md_path.exists() and html_path.exists():
        log.info("SKIP (already collected): %s", source_id)
        # Read existing metadata for manifest
        if meta_path.exists():
            with meta_path.open() as fh:
                return json.load(fh)
        return {"source_id": source_id, "status": "skipped", "url": url}

    timestamp = datetime.now(timezone.utc).isoformat()

    if dry_run:
        log.info("DRY-RUN: would fetch %s -> %s", url, source_id)
        return {"source_id": source_id, "url": url, "status": "dry_run", "timestamp": timestamp}

    # Check robots.txt
    if not robots_allowed(url):
        reason = "disallowed by robots.txt"
        log.warning("SKIP (%s): %s", reason, url)
        record = {
            "source_id": source_id, "url": url, "status": "failed",
            "reason": reason, "timestamp": timestamp,
        }
        _write_json(meta_path, record)
        return record

    # Fetch
    try:
        http_status, raw_bytes, _ct = fetch_url(url)
    except Exception as exc:
        reason = str(exc)
        log.error("FAIL: %s — %s", source_id, reason)
        record = {
            "source_id": source_id, "url": url, "status": "failed",
            "reason": reason, "timestamp": timestamp, "http_status": None,
        }
        _write_json(meta_path, record)
        return record

    if http_status >= 400:
        reason = f"HTTP {http_status}"
        log.warning("FAIL (%s): %s", reason, url)
        record = {
            "source_id": source_id, "url": url, "status": "failed",
            "reason": reason, "http_status": http_status, "timestamp": timestamp,
        }
        _write_json(meta_path, record)
        return record

    content_hash = _sha256(raw_bytes)
    title = _extract_title(raw_bytes, source_type)
    markdown_body = _html_to_markdown(raw_bytes, url, source_type)

    if len(markdown_body.encode()) < MIN_BODY_BYTES:
        reason = f"extracted body too short ({len(markdown_body.encode())} bytes)"
        log.warning("FAIL (%s): %s", reason, source_id)
        record = {
            "source_id": source_id, "url": url, "status": "failed",
            "reason": reason, "http_status": http_status, "timestamp": timestamp,
            "content_hash": content_hash,
        }
        _write_json(meta_path, record)
        return record

    header = _provenance_header(
        source_id=source_id, url=url, title=title,
        licence=licence, licence_url=licence_url,
        timestamp=timestamp, content_hash=content_hash,
        http_status=http_status,
    )
    full_markdown = header + markdown_body

    # Write files
    html_path.write_bytes(raw_bytes)
    md_path.write_text(full_markdown, encoding="utf-8")

    record: dict[str, Any] = {
        "source_id": source_id,
        "url": url,
        "status": "ok",
        "http_status": http_status,
        "page_title": title,
        "licence": licence,
        "licence_url": licence_url,
        "timestamp": timestamp,
        "content_hash_sha256": content_hash,
        "html_bytes": len(raw_bytes),
        "markdown_bytes": len(full_markdown.encode()),
        "html_path": str(html_path.relative_to(REPO_ROOT)) if html_path.is_relative_to(REPO_ROOT) else str(html_path),
        "markdown_path": str(md_path.relative_to(REPO_ROOT)) if md_path.is_relative_to(REPO_ROOT) else str(md_path),
    }
    _write_json(meta_path, record)
    log.info("OK  [%d] %s — %d bytes MD", http_status, source_id, record["markdown_bytes"])
    return record


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(records: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Manifest written: %s (%d records)", MANIFEST_PATH, len(records))


# ---------------------------------------------------------------------------
# Post-collection validation
# ---------------------------------------------------------------------------

def validate_corpus(records: list[dict], sources: list[dict]) -> None:
    total = len(sources)
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") == "failed"]
    skipped = [r for r in records if r.get("status") == "skipped"]

    md_files = list(MARKDOWN_DIR.glob("*.md"))
    html_files = list(HTML_DIR.glob("*.html"))
    total_md_bytes = sum(f.stat().st_size for f in md_files)

    hashes = [r.get("content_hash_sha256") for r in ok if r.get("content_hash_sha256")]
    duplicate_hashes = len(hashes) - len(set(hashes))

    empty_docs = [r for r in ok if r.get("markdown_bytes", 9999) < MIN_BODY_BYTES]

    unclear_licences = [r for r in ok if r.get("licence") in ("REVIEW-NEEDED", "UNKNOWN", None)]
    missing_provenance = []
    for r in ok:
        mp = r.get("markdown_path")
        if mp:
            p = REPO_ROOT / mp
            if p.exists():
                content = p.read_text(encoding="utf-8")
                if "<!-- PROVENANCE" not in content:
                    missing_provenance.append(r["source_id"])

    print("\n" + "=" * 60)
    print("COLLECTION VALIDATION REPORT")
    print("=" * 60)
    print(f"  Configured sources  : {total}")
    print(f"  Successful          : {len(ok)}")
    print(f"  Failed              : {len(failed)}")
    print(f"  Skipped (cached)    : {len(skipped)}")
    print(f"  Markdown files      : {len(md_files)}")
    print(f"  HTML files          : {len(html_files)}")
    print(f"  Metadata records    : {len(list(METADATA_DIR.glob('*.json')))}")
    print(f"  Total Markdown bytes: {total_md_bytes:,}")
    print(f"  Empty documents     : {len(empty_docs)}")
    print(f"  Duplicate hashes    : {duplicate_hashes}")
    print(f"  Unclear licences    : {len(unclear_licences)}")
    print(f"  Missing provenance  : {len(missing_provenance)}")
    if failed:
        print("\n  Failed sources:")
        for r in failed:
            print(f"    - {r['source_id']}: {r.get('reason','?')}")
    if unclear_licences:
        print("\n  Sources needing licence review:")
        for r in unclear_licences:
            print(f"    - {r['source_id']} ({r.get('licence','?')})")
    print("=" * 60)

    if len(ok) == 0:
        log.error("Validation failed: no sources succeeded.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true", help="Re-fetch already-collected sources.")
    p.add_argument("--dry-run", action="store_true", help="Print plan without fetching.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for d in (HTML_DIR, MARKDOWN_DIR, METADATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        log.error("Source config not found: %s", CONFIG_PATH)
        sys.exit(1)

    sources = load_sources(CONFIG_PATH)
    if not sources:
        log.error("No enabled sources found in %s", CONFIG_PATH)
        sys.exit(1)

    dupes = _check_duplicate_urls(sources)
    if dupes:
        for d in dupes:
            log.warning("Duplicate URL: %s", d)

    log.info("Loaded %d enabled source(s) from %s", len(sources), CONFIG_PATH)

    records: list[dict] = []
    for source in sources:
        rec = collect_source(source, refresh=args.refresh, dry_run=args.dry_run)
        records.append(rec)

    if not args.dry_run:
        write_manifest(records)
        validate_corpus(records, sources)

    ok_count = sum(1 for r in records if r.get("status") == "ok")
    fail_count = sum(1 for r in records if r.get("status") == "failed")

    if ok_count == 0 and not args.dry_run:
        log.error("All sources failed. Exiting with error.")
        sys.exit(1)

    log.info("Done. %d succeeded, %d failed.", ok_count, fail_count)


if __name__ == "__main__":
    main()
