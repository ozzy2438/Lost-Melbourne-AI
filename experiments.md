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

---

## 2026-06-30 — Phase 3: Representation and Retrieval Laboratory

### Goal

Measure how reliably sparse, dense, hybrid and structured retrieval can recover the
correct provenance-bearing passage without training the tiny Transformer or building RAG/UI.

### Phase 2 verification and correction

- Loaded and cross-validated all eight required Phase 2 outputs.
- The initial audit started from 410 entities, 61 events and 7 relations.
- Removed co-mention-only location edges, a government organisation incorrectly selected
  as a demolished structure, and other planned/unsupported event assignments.
- Expanded only explicit `designed by` and `operated by` patterns, including architectural
  practice names. The corrected fabric contains 413 entities, 52 events, 8 relations and
  60 validated claims. Every retained edge still has an exact passage span.
- Audit conclusion: graph sparsity is real and also reflects deliberately conservative
  rules; relation counts were not inflated to improve retrieval.

### Evaluation set

- Added 60 manually templated and deterministically validated questions.
- Mix: 50 answerable and 10 unanswerable; 26 easy, 18 medium and 16 hard.
- 45 development questions are used for model/chunk/transformation/threshold decisions.
- 15 held-out test questions are evaluated only after those choices are fixed.

### Dense models

| Model | Revision | Dimension | Licence | Estimated memory | Cached model size |
| --- | --- | ---: | --- | ---: | ---: |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | 384 | Apache-2.0 | 90,852,864 B | 91,599,528 B |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405e` | 384 | MIT | 133,440,000 B | 134,505,940 B |

Both models were run locally over passage-level text with L2-normalised vectors and
transparent NumPy cosine similarity. Development metrics selected BGE-small as the dense
component.

### Held-out retrieval results

| Method | R@1 | R@3 | R@5 | MRR | nDCG@10 | Unanswerable precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.500 | 0.833 | 0.917 | 0.683 | 0.758 | 0.167 |
| TF-IDF | **0.750** | **0.917** | 0.917 | **0.845** | **0.875** | **0.250** |
| MiniLM dense | 0.417 | 0.750 | 0.833 | 0.595 | 0.665 | 0.000 |
| BGE-small dense | 0.500 | 0.833 | **1.000** | 0.672 | 0.701 | 0.000 |
| Hybrid RRF | 0.667 | 0.750 | 0.917 | 0.760 | 0.818 | 0.111 |
| Hybrid + structured | 0.667 | 0.833 | 0.917 | 0.781 | 0.849 | 0.100 |

### Decisions and findings

- Default retrieval: **TF-IDF**, selected with a development-only reliability score weighted
  50% MRR, 30% Recall@5 and 20% unanswerable precision. It remained the strongest held-out
  method with MRR 0.845 and is also the lowest-latency practical default.
- Hybrid + structured is retained for evidence exploration, but its held-out MRR fell to
  0.781. BGE-small alone reached test Recall@5 = 1.000, while dense abstention remained weak.
- Original 291–448-word passages beat 157–220-word small and parent-child variants on
  development MRR. Smaller chunks damaged multi-passage/context questions in this corpus.
- Entity-aware deterministic expansion won on development data, although test metrics tied
  the no-transform and alias-only variants.
- The preliminary evidence threshold can correctly reject some absent-answer questions but
  unanswerable precision remains low. This is a measured limitation, not a solved problem.
- LLM query rewriting was not run; original queries and deterministic expansions are logged.

### Validation

- 52 / 52 offline tests passed: 23 collection, 18 preparation and 11 retrieval tests.
- Retrieval tests cover corpus integrity, deterministic indexing, stable child IDs, embedding
  dimensions, ordering, deduplication, temporal/geographic filters, alias expansion,
  abstention, alternate working directories and sparse operation with no network.
- The representation notebook ran top-to-bottom with six executed code cells and no errors.
- Result: **pass**.
