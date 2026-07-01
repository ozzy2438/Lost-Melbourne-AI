# Lost Melbourne — The City That Remembers — Project Plan

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

### Phase 1 Recovery - Evidence collection
- Build scraper that:
  - reads per-site robots.txt;
  - respects crawl delay and disallow rules;
  - stores page HTML and metadata unchanged;
  - stores source URL, timestamp, license/terms notes.
- Target publicly accessible and legally reusable sources.

### Phase 2 - Historical Knowledge Fabric
- Verify immutable raw evidence and publish a sanitised corpus fingerprint.
- Clean and split documents without crossing source boundaries.
- Extract source-supported entities, events, relations, claims, dates, and places.
- Create document-level splits and a licence-filtered, source-balanced future training corpus.
- Do not create embeddings, train a model, build final RAG, or create the UI.

### Phase 3 - Representation and Retrieval Laboratory
- Audit Historical Knowledge Fabric connectivity without accepting unsupported edges.
- Benchmark BM25, TF-IDF, two local dense encoders, hybrid RRF, and structured reranking.
- Compare original, small, and parent-child passage representations.
- Calibrate preliminary abstention behaviour on answerable and unanswerable questions.
- Preserve complete retrieval score traces and held-out test evaluation.

### Phase 4 - Tokenizer implementation
- Implement tokenizer training and encode/decode pipeline.
- Persist vocab and tokenizer config.
- Create train/validation tokenized datasets.

### Phase 5 - Tiny decoder-only Transformer (PyTorch)
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

### Phase 6 - Training loop
- Train model locally.
- Track and save:
  - train loss;
  - validation loss;
  - checkpoints and configs.

### Phase 7 - Embedding search experiments
- Derive passage embeddings from trained tiny model.
- Implement cosine-similarity retrieval baseline.
- Build pretrained sentence-transformer index in FAISS or ChromaDB.
- Compare retrieval precision between both systems.

### Phase 8 - RAG pipeline
- Build manual retrieval + context assembly + generation flow.
- Return factual answers with citations.
- Optional mode: clearly labelled creative vignette (non-factual embellishment flagged).
- Compare:
  - tiny Transformer alone;
  - vector-search RAG;
  - LLM Wiki + RAG.

### Phase 9 - Evaluation
- Measure:
  - training loss;
  - validation loss;
  - retrieval precision;
  - citation correctness;
  - unsupported-claim rate.
- Record outputs and observations in `experiments.md`.

### Phase 10 - Streamlit interface
- Add simple UI to:
  - run queries;
  - view retrieved passages;
  - inspect citations;
  - choose mode (Tiny LM / RAG / Wiki+RAG).
