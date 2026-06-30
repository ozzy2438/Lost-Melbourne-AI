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

---

## 2026-06-30 — Phase 2: Historical Knowledge Fabric

### Goal

Transform the immutable Phase 1 evidence into a deterministic, provenance-preserving
historical data layer without embeddings, model training, a final RAG pipeline, or UI work.

### Evidence verification

- Verified 25 raw Markdown files, 25 raw HTML files, and successful metadata for every source.
- Parsed complete provenance headers and matched every stored SHA-256 to the corresponding
  raw HTML bytes.
- Snapshotted all raw evidence before and after processing; no raw file changed.
- Committed a content-free identity record in `reports/corpus_fingerprint.json`.

### Pipeline and model

- Added `scripts/prepare_corpus.py`, rooted from its own repository location and safe to run
  from another current working directory.
- Added deterministic cleaning, 291–448-word document-only passages with modest overlap,
  stable canonical IDs, conservative alias handling, explicit date preservation, document-level
  splits, coordinate extraction without fabricated points, and exact claim-span validation.
- Added versioned optional second-stage contracts in `config/extraction_schema.json` and
  `config/extraction_prompt.md`. No LLM was called in this run.
- Added a 6,000-word per-source training cap. Only cleaned source text with compatible
  licences is included in the future educational Transformer corpus.

### Real Phase 2 results

| Metric | Value |
| --- | ---: |
| Raw documents verified | 25 |
| Cleaned usable documents | 24 |
| Excluded documents | 1 (`wiki_moomba`, 28-word disambiguation page) |
| Raw words including page chrome | 106,206 |
| Cleaned words | 66,558 |
| Approximate cleaned tokens | 88,523 |
| Passages | 182 |
| Entities | 410 |
| Historical events | 61 |
| Relations | 7 |
| Validated claims | 68 |
| Unsupported claims accepted | 0 |
| GeoJSON point features | 12 |
| Detected cross-source conflict groups | 0 |
| Training words after caps | 59,389 |
| Approximate training tokens | 78,988 |
| Train / validation / test documents | 19 / 2 / 3 |

### Source balance and licences

- `wiki_architecture_melbourne` fell from 19.60% of cleaned words to 10.10% of training
  words after the 6,000-word cap.
- `wiki_flinders_street_station` was also capped from 6,124 to 6,000 words.
- All retained sources are training-compatible in this local corpus: 23 CC-BY-SA-4.0,
  one CC-BY-4.0, and one OGL-AU raw source; the excluded disambiguation page was CC-BY-SA-4.0.
- No exact or normalised near-duplicate documents were found.

### Validation

- 41 / 41 offline tests passed (23 collection tests + 18 preparation tests).
- Tests cover empty input, unusable bodies, tampered hashes, missing provenance, raw
  immutability, graph support links, exact claim spans, unsupported-claim rejection,
  licence exclusions, valid GeoJSON, split isolation, deterministic reruns, and execution
  from `/tmp`.
- Result: **pass**.
