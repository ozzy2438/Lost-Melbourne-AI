# Lost Melbourne AI Project Plan

## Delivery rule

Work one phase at a time. After each phase:

1. Explain outcomes in simple language.
2. Explain outcomes technically.
3. Show relevant files and commands.
4. Run validation checks.
5. Append results to `experiments.md`.
6. Make a small Git commit.

## Phases

### Phase 1 - Project scaffold and plan
- Create folder structure for data, code, wiki, evaluation, app.
- Create `README.md`, `PROJECT_PLAN.md`, `experiments.md`.

### Phase 2 - Data collection (30-50 pages)
- Build scraper that:
  - reads per-site robots.txt;
  - respects crawl delay and disallow rules;
  - stores page HTML and metadata unchanged;
  - stores source URL, timestamp, license/terms notes.
- Target publicly accessible and legally reusable sources.

### Phase 3 - Cleaning and normalization
- Parse raw HTML into clean text.
- Remove boilerplate/navigation noise.
- Preserve provenance for each extracted passage.
- Save cleaned docs as Markdown or JSONL for downstream steps.

### Phase 4 - LLM Wiki generation
- Create interconnected Markdown pages in `wiki/pages/`.
- Link entities (buildings, suburbs, timelines, people, events).
- Keep citation links to source passages.

### Phase 5 - Tokenizer implementation
- Implement tokenizer training and encode/decode pipeline.
- Persist vocab and tokenizer config.
- Create train/validation tokenized datasets.

### Phase 6 - Tiny decoder-only Transformer (PyTorch)
- Implement:
  - token embeddings;
  - positional embeddings;
  - masked multi-head self-attention;
  - feed-forward block;
  - residual connections;
  - layer normalization;
  - next-token prediction with cross-entropy loss.
- Keep local-Mac-friendly size:
  - 2-4 layers, 4 heads, 128-256 embedding dim, ~256 context length.

### Phase 7 - Training loop
- Train model locally.
- Track and save:
  - train loss;
  - validation loss;
  - checkpoints and configs.

### Phase 8 - Embedding search experiments
- Derive passage embeddings from trained tiny model.
- Implement cosine-similarity retrieval baseline.
- Build pretrained sentence-transformer index in FAISS or ChromaDB.
- Compare retrieval precision between both systems.

### Phase 9 - RAG pipeline
- Build manual retrieval + context assembly + generation flow.
- Return factual answers with citations.
- Optional mode: clearly labelled creative vignette (non-factual embellishment flagged).
- Compare:
  - tiny Transformer alone;
  - vector-search RAG;
  - LLM Wiki + RAG.

### Phase 10 - Evaluation
- Measure:
  - training loss;
  - validation loss;
  - retrieval precision;
  - citation correctness;
  - unsupported-claim rate.
- Record outputs and observations in `experiments.md`.

### Phase 11 - Streamlit interface
- Add simple UI to:
  - run queries;
  - view retrieved passages;
  - inspect citations;
  - choose mode (Tiny LM / RAG / Wiki+RAG).

