# Lost Melbourne — The City That Remembers

End-to-end Generative AI project about Melbourne's lost places:

1. Collect legally reusable public historical sources.
2. Preserve raw sources unchanged.
3. Clean and structure text into a linked Markdown wiki.
4. Train a tiny decoder-only Transformer from scratch in PyTorch.
5. Compare retrieval quality of custom embeddings vs pretrained sentence embeddings.
6. Build a citation-first RAG pipeline and evaluate factual grounding.
7. Expose results in a Streamlit interface with clickable citations and diagnostics.

## Current status

Phases 1–3 are complete. The Streamlit demo is production-ready with source citations and a
diagnostics panel. A Dockerfile is provided for one-command deployment.

---

## Architecture summary

```
Raw HTML sources (25 docs)
        │
        ▼
Phase 2 — Historical Knowledge Fabric
  documents → passages → entities / events / relations / claims / places
        │
        ▼
Phase 3 — Retrieval Pipeline
  BM25 + TF-IDF (sparse) + BGE-small (dense) → Reciprocal Rank Fusion
  → structured reranking → TF-IDF lexical reranking → abstention calibration
        │
        ▼
Streamlit UI — citations, diagnostics, evidence passages
```

- **No free-form generation.** All answers are extracted evidence passages.
- **No Tiny Transformer in the demo.** It is trained separately (Phase 5–6) and not connected to the UI.
- **Abstention.** If the top retrieved score falls below a calibrated threshold the system
  returns a "not enough evidence" message rather than a hallucinated answer.

---

## Evaluation results (Phase 3, held-out test set)

Default method: **TF-IDF hybrid** · Dense encoder: **BGE-small** · Query transform: **entity**

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.750 |
| Recall@3 | 0.917 |
| Recall@5 | 0.917 |
| Recall@10 | 1.000 |
| MRR | 0.845 |
| nDCG@10 | 0.875 |
| Unanswerable precision | 0.250 |

**Answerable vs Unanswerable (test set)**

| Slice | Count | Success rate | Abstention rate |
| --- | ---: | ---: | ---: |
| Answerable | 12 | 0.667 | 0.250 |
| Unanswerable | 3 | 0.333 | 0.333 |

Full results: [`evaluation/final_results.json`](evaluation/final_results.json) ·
[`reports/final_evaluation_report.md`](reports/final_evaluation_report.md)

---

## Quick start — local run

```bash
# 1. Create a virtual environment (Python 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-retrieval.txt -r requirements-app.txt

# 3. Collect raw sources (skips already-collected files)
python scripts/collect.py

# 4. Build the Phase 2 knowledge fabric
python scripts/prepare_corpus.py

# 5. Build retrieval indexes (BM25, TF-IDF, dense embeddings)
python scripts/build_retrieval_indexes.py

# 6. Run the Streamlit demo
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501> in your browser.

---

## Docker run

```bash
# Build the image (from the repo root)
docker build -t lost-melbourne .

# Run — mount pre-built artifacts and processed corpus
docker run -p 8501:8501 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  -v "$(pwd)/data:/app/data:ro" \
  lost-melbourne
```

Open <http://localhost:8501>.

> **Note:** The Docker image does not bundle the corpus or indexes (they are excluded from
> Git). Run steps 3–5 above locally first, then mount the resulting `artifacts/` and
> `data/` directories into the container.

---

## Known limitations

- **Corpus size.** Only 24 source documents are indexed. Recall is bounded by source coverage.
- **Unanswerable precision.** The abstention threshold is conservative; some answerable
  questions are also abstained on (0.25 false-abstention rate on test).
- **Dense retrieval.** BGE-small underperforms TF-IDF on this small domain corpus; a larger
  domain-specific fine-tuned encoder would help.
- **No generation.** The demo returns extracted passages, not fluent prose answers.
- **No authentication or database.** The Streamlit app is a stateless demo.
- **Scraped content not committed.** Raw HTML and processed corpus are excluded from Git
  due to copyright. Run `scripts/collect.py` to reproduce them.

---

## Canonical repository location

Remote: <https://github.com/ozzy2438/Lost-Melbourne-AI>

---

## How to run data collection

```bash
# Install dependencies
pip install requests pyyaml markdownify

# Collect from all enabled sources (safe restart — skips already-collected)
python3 scripts/collect.py

# Force re-fetch everything
python3 scripts/collect.py --refresh

# Preview what would be fetched without downloading
python3 scripts/collect.py --dry-run
```

## How to run collection tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Folder layout

| Path | Contents | In Git? |
| --- | --- | --- |
| `config/sources.yaml` | Source registry | Yes |
| `scripts/collect.py` | Collector script | Yes |
| `tests/` | Automated tests + fixtures | Yes |
| `data/raw/html/` | Original fetched HTML | **No** (see .gitignore) |
| `data/raw/markdown/` | Extracted Markdown with provenance | **No** |
| `data/raw/metadata/` | Per-source JSON metadata | **No** |
| `data/raw/manifest.jsonl` | Collection manifest | **No** |
| `data/processed/` | Generated Phase 2 knowledge-fabric records | **No** |
| `models/` | Trained model checkpoints | **No** |
| `reports/` | Collection and validation reports | Yes |
| `experiments.md` | Phase-by-phase experiment log | Yes |

## Generated files excluded from Git

Raw HTML, Markdown and manifest files are excluded because they may contain
copyrighted content. Do not commit scraped page content to a public repository
without verifying the licence of each source.

A local backup of the raw corpus is maintained at:

```text
~/Documents/Lost-Melbourne-AI-backups/
```

To restore from backup:

```bash
tar -xzf ~/Documents/Lost-Melbourne-AI-backups/raw_corpus_latest.tar.gz \
    -C /Users/osmanorka/Lost-Melbourne-AI-1/
```

## How to build the Phase 2 knowledge fabric

```bash
# Uses only the local Phase 1 corpus; no live network or LLM call
python3 scripts/prepare_corpus.py

# Run all offline collection and preparation tests
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The command validates the 25 raw HTML/Markdown/metadata triplets and their SHA-256
hashes before processing. It writes generated documents, passages, entities, events,
relations, claims, places, exclusions, document-level splits, and the balanced future
training text under `data/processed/`. Generated content stays ignored by Git; tracked
aggregate reports and the sanitised corpus fingerprint live under `reports/`.

The deterministic extractor is the production default. The versioned schema and prompt
under `config/` define an optional future LLM-assisted candidate stage, but all candidates
must still pass exact source-span validation.

Phase 2 deliberately does **not** create embeddings, train a Transformer, build the final
RAG pipeline, or create a user interface.

## How to reproduce the Phase 3 retrieval benchmark

Phase 3 uses Python 3.12 because the local system Python may be newer than the available
PyTorch wheels.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-retrieval.txt

# Regenerate Phase 2 after the supported extraction-rule correction
.venv/bin/python scripts/prepare_corpus.py

# Build sparse indexes and cached passage embeddings
.venv/bin/python scripts/build_retrieval_indexes.py

# Evaluate all retrieval, chunk, query-transform, structured and abstention variants
.venv/bin/python scripts/evaluate_retrieval.py

# Run the retrieval-only answerer with hybrid+structured candidates, TF-IDF reranking,
# returned evidence passages, and development-calibrated abstention
.venv/bin/python scripts/answer_query.py "Who designed the Metropolitan Meat Market?"

# Run the complete offline test suite
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

# Re-run the educational representation notebook
.venv/bin/jupyter execute notebooks/representation_lab.ipynb --inplace \
  --timeout=180 --kernel_name=python3
```

Generated indexes and dense vectors are cached under `artifacts/retrieval/` and excluded
from Git. The human-readable evaluation labels, aggregate benchmark reports, failure
analysis and content-free result traces are tracked. The dense model revisions are pinned
in `src/retrieval/dense.py`.

The retrieval-only answerer returns evidence passages or an abstention fallback; it does
not generate free-form answers, train the tiny Transformer, build the final RAG
application, or create a production UI.

## How to run the Streamlit demo

Minimal preview UI over the existing retrieval-only answerer (`app/streamlit_app.py`).
It does not change retrieval logic, does not use the Tiny Transformer, and has no
authentication or database.

```bash
uv pip install --python .venv/bin/python -r requirements-app.txt
.venv/bin/streamlit run app/streamlit_app.py
```

Requires the Phase 3 indexes to already be built (`scripts/build_retrieval_indexes.py`).
