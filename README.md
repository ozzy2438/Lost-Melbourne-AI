# Lost Melbourne — The City That Remembers

Small end-to-end Generative AI project about Melbourne's lost places:

1. Collect legally reusable public historical sources.
2. Preserve raw sources unchanged.
3. Clean and structure text into a linked Markdown wiki.
4. Train a tiny decoder-only Transformer from scratch in PyTorch.
5. Compare retrieval quality of custom embeddings vs pretrained sentence embeddings.
6. Build a citation-first RAG pipeline and evaluate factual grounding.
7. Expose results in a simple Streamlit interface.

## Current status

Phase 1 (scaffold), Phase 1 Recovery (reproducible data collection), and Phase 2
(provenance-aware Historical Knowledge Fabric) are complete. Phase 3 (Representation
and Retrieval Laboratory) is also complete.
The original Phase 1 report cited 27 Markdown files that were never committed to the
repository. The collection pipeline has been rebuilt from scratch with a replacement corpus.

---

## Canonical repository location

```text
/Users/osmanorka/Lost-Melbourne-AI-1
```

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

Phase 3 does **not** train the tiny Transformer, generate answers, build the final RAG
application, or create a production UI.
