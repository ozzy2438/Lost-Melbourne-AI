# Phase 3 Retrieval Failure Analysis

## Failure categories

| Type | Count |
| --- | ---: |
| alias_mismatch | 1 |
| date_filter_failure | 3 |
| insufficient_corpus_coverage | 5 |
| lexical_mismatch | 2 |
| location_ambiguity | 5 |

The abstention threshold is tuned only on the 45-question development split, then applied unchanged to the 15 held-out test questions. Unanswerable prompts that still retrieve a similarly named entity are counted as insufficient-corpus failures rather than successful retrievals.

## Representative full traces

### q001: Who designed the Metropolitan Meat Market?

- Transformed: Who designed the Metropolitan Meat Market?
- Detected entities: ['ent_602749a5f51504e6']
- Structured filters: ['entity:Metropolitan Meat Market']
- Expected: ['pass_64b916a429fca5c9']
- Default method: tfidf
- Abstained: False
- Success: True
- BM25 top 3: ['pass_c6c02e8d2b683d6e', 'pass_c1865c6c8b710e5f', 'pass_64b916a429fca5c9']
- TF-IDF top 3: ['pass_64b916a429fca5c9', 'pass_c6c02e8d2b683d6e', 'pass_c1865c6c8b710e5f']
- MiniLM top 3: ['pass_bd5db69af5eb92c9', 'pass_229f6a2d7967637d', 'pass_f9d3e777eb3e3f3e']
- BGE-small top 3: ['pass_b0118f37cf01d7ad', 'pass_c6c02e8d2b683d6e', 'pass_c1865c6c8b710e5f']
- Hybrid/structured score breakdown:
  - rank 1: `pass_c6c02e8d2b683d6e` score=0.092522; components={'rrf_bm25': 0.01639344262295082, 'rrf_dense': 0.016129032258064516, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.0, 'structured_bonus': 0.05}
  - rank 2: `pass_64b916a429fca5c9` score=0.089031; components={'rrf_bm25': 0.015873015873015872, 'rrf_dense': 0.013157894736842105, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.0, 'structured_bonus': 0.05}
  - rank 3: `pass_c1865c6c8b710e5f` score=0.042002; components={'rrf_bm25': 0.016129032258064516, 'rrf_dense': 0.015873015873015872, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.0, 'structured_bonus': 0.0}
  - rank 4: `pass_b0118f37cf01d7ad` score=0.032018; components={'rrf_bm25': 0.015625, 'rrf_dense': 0.01639344262295082, 'title_bonus': 0.0, 'entity_bonus': 0.0, 'alias_bonus': 0.0, 'structured_bonus': 0.0}
  - rank 5: `pass_1fa007360bb99ecb` score=0.031010; components={'rrf_bm25': 0.015384615384615385, 'rrf_dense': 0.015625, 'title_bonus': 0.0, 'entity_bonus': 0.0, 'alias_bonus': 0.0, 'structured_bonus': 0.0}

### q051: What was the ticket price on Eastern Market's opening day?

- Transformed: What was the ticket price on Eastern Market's opening day? Paddys Market
- Detected entities: ['ent_48c5209c62a61529']
- Structured filters: ['entity:Eastern Market', 'event:opening']
- Expected: []
- Default method: tfidf
- Abstained: True
- Success: True
- BM25 top 3: ['pass_143d209c0f4c231e', 'pass_bd5db69af5eb92c9', 'pass_de073779e178fddd']
- TF-IDF top 3: ['pass_143d209c0f4c231e', 'pass_bd5db69af5eb92c9', 'pass_f9d3e777eb3e3f3e']
- MiniLM top 3: ['pass_bd5db69af5eb92c9', 'pass_143d209c0f4c231e', 'pass_de073779e178fddd']
- BGE-small top 3: ['pass_bd5db69af5eb92c9', 'pass_f9d3e777eb3e3f3e', 'pass_de073779e178fddd']
- Hybrid/structured score breakdown:
  - rank 1: `pass_c1865c6c8b710e5f` score=0.109199; components={'rrf_bm25': 0.014705882352941176, 'rrf_dense': 0.014492753623188406, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.02, 'structured_bonus': 0.05}
  - rank 2: `pass_c84a9420195e17c1` score=0.108790; components={'rrf_bm25': 0.014084507042253521, 'rrf_dense': 0.014705882352941176, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.02, 'structured_bonus': 0.05}
  - rank 3: `pass_f9d3e777eb3e3f3e` score=0.071545; components={'rrf_bm25': 0.015151515151515152, 'rrf_dense': 0.01639344262295082, 'title_bonus': 0.0, 'entity_bonus': 0.02, 'alias_bonus': 0.02, 'structured_bonus': 0.0}
  - rank 4: `pass_bd5db69af5eb92c9` score=0.062258; components={'rrf_bm25': 0.016129032258064516, 'rrf_dense': 0.016129032258064516, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.02, 'structured_bonus': 0.0}
  - rank 5: `pass_143d209c0f4c231e` score=0.062018; components={'rrf_bm25': 0.01639344262295082, 'rrf_dense': 0.015625, 'title_bonus': 0.0, 'entity_bonus': 0.01, 'alias_bonus': 0.02, 'structured_bonus': 0.0}


## Interpretation

- Alias mismatch: deterministic alias expansion failed to expose the canonical entity.
- Wrong chunk boundary: the indexed child omitted evidence available in its parent.
- Lexical mismatch: query and evidence shared too little vocabulary.
- Semantic confusion: dense retrieval preferred a related but unsupported passage.
- Insufficient corpus coverage: no gold passage exists and the method failed to abstain.
- Date-filter failure: a temporal signal did not promote the supported event passage.
- Location ambiguity: a place name matched the wrong geographic context.
- Entity-resolution failure: entity/alias detection selected the wrong canonical record.
