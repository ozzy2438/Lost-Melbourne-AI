# Lost Melbourne AI

Small end-to-end Generative AI project about Melbourne's lost places:

1. Collect legally reusable public historical sources.
2. Preserve raw sources unchanged.
3. Clean and structure text into a linked Markdown wiki.
4. Train a tiny decoder-only Transformer from scratch in PyTorch.
5. Compare retrieval quality of custom embeddings vs pretrained sentence embeddings.
6. Build a citation-first RAG pipeline and evaluate factual grounding.
7. Expose results in a simple Streamlit interface.

## Current status

Phase 1 (scaffold) and Phase 1 Recovery (reproducible data collection pipeline) complete.
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
| `data/processed/` | Cleaned corpus (Phase 3+) | **No** |
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

## Phase 2 (corpus processing)

Phase 2 processing instructions will be added here after the collection
pipeline has been verified.
