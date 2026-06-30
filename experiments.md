# Experiments Log

## 2026-06-30 - Phase 1: Scaffold and plan

### Goal

Establish base project structure and a strict phase-by-phase implementation plan.

### Created

- Directory scaffold for data ingestion, cleaning, wiki generation, model code, retrieval, RAG, evaluation, and app UI.
- `README.md` with project scope.
- `PROJECT_PLAN.md` with phased execution rules and milestones.

### Validation

- Checked required directories and core files exist.
- Result: pass.

### Notes

- No model or data processing code implemented in this phase.

---

## 2026-06-30 — CORRECTION: Phase 1 data results were not reproducibly present

**The original Phase 1 and any previously cited Phase 2 data results were not verifiable.**
Neither `scripts/collect.py`, `config/sources.yaml`, Markdown files, nor a manifest existed
in the repository, in git history, in any stash, in any worktree, or anywhere on the local
machine. The claim of "27 Markdown files and ~520,912 bytes" in earlier reports is retracted.

---

## 2026-06-30 — Phase 1 Recovery: Rebuild reproducible corpus collection pipeline

### Goal

Re-establish a reproducible, polite, robots.txt-respecting data collection pipeline from
scratch and produce a verified raw corpus about Melbourne's lost and heritage places.

### Files created

- `config/sources.yaml` — source registry with 27 entries (25 enabled, 2 disabled due to
  robots.txt disallow rules). This is a **replacement corpus**, not a reconstruction of any
  earlier unverified run.
- `scripts/collect.py` — deterministic collector with safe restart, provenance headers,
  per-host polite delays (≥2s), robots.txt checking (via `requests`, not `urllib`),
  bounded retries, failure recording, and JSONL manifest.
- `tests/test_collect.py` — 23 unit tests covering path resolution, source loading,
  duplicate URL detection, HTML extraction, SPARQL/API conversion, provenance headers,
  manifest generation, failure recording, and empty-body rejection. No live network.
- `tests/fixtures/` — local HTML and JSON fixtures for offline tests.
- `reports/collection_report.md` — full collection validation report.
- `.gitignore` — excludes raw HTML, Markdown, metadata and manifest from Git.
- `README.md` updated with canonical repo location, run instructions, folder layout and
  backup instructions.

### Real Phase 1 Recovery results

| Metric | Value |
| --- | --- |
| Configured sources (enabled) | 25 |
| Successful | **25** |
| Failed | **0** |
| Markdown files | **25** |
| HTML files | **25** |
| Total Markdown bytes | **739,293** |
| Empty documents | 0 |
| Duplicate content hashes | 0 |
| Unclear licences | 0 |
| Files missing provenance | 0 |
| Unit tests | 23 / 23 passed |

### Licence breakdown

- CC-BY-SA-4.0: 23 sources (Wikipedia)
- OGL-AU: 1 source (Victorian Heritage Database)
- CC-BY-4.0: 1 source (Public Record Office Victoria)

### Validation

- All 23 unit tests pass (no live network).
- Safe restart from `/tmp` confirmed: skips already-collected files.
- Backup at `~/Documents/Lost-Melbourne-AI-backups/raw_corpus_20260630.tar.gz` (1.2 MB).
- Result: **pass**.
