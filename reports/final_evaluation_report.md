# Lost Melbourne — Final Evaluation Report

## Overview

This report summarises the Phase 3 retrieval benchmark results used for the production portfolio demo.
All metrics are computed on the held-out **test split** (never used for threshold tuning).
Development metrics were used only to select the retrieval configuration.

---

## Selected Configuration

| Parameter | Value |
| --- | --- |
| Default method | `tfidf` |
| Dense model | `bge-small` |
| Passage strategy | `original` |
| Query transform | `entity` |
| Abstention threshold | `0.099174` |

---

## Query Set

| Split | Answerable | Unanswerable | Total |
| --- | ---: | ---: | ---: |
| Development | 38 | 7 | 45 |
| Test (held-out) | 12 | 3 | 15 |
| **Total** | **50** | **10** | **60** |

---

## Test-Set Metrics — Default Method (`tfidf`)

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.750 |
| Recall@3 | 0.917 |
| Recall@5 | 0.917 |
| Recall@10 | 1.000 |
| MRR | 0.845 |
| nDCG@10 | 0.875 |
| Unanswerable precision | 0.250 |
| Unanswerable recall | 0.333 |
| Zero-result rate | 0.267 |

---

## Answerable vs Unanswerable — Separate Evaluation

### Test split

| Slice | Count | Success rate | Abstention rate |
| --- | ---: | ---: | ---: |
| Answerable | 12 | 0.667 | 0.250 |
| Unanswerable | 3 | 0.333 | 0.333 |

### Development split

| Slice | Count | Success rate | Abstention rate |
| --- | ---: | ---: | ---: |
| Answerable | 38 | 0.816 | 0.132 |
| Unanswerable | 7 | 0.571 | 0.571 |

**Interpretation:** A success for an unanswerable query means the system correctly abstained (returned a "not enough evidence" fallback). A success for an answerable query means the gold passage appeared in the top-5 results and the system did not abstain.

---

## All Methods — Test-Set Comparison

| Method | R@1 | R@3 | R@5 | MRR | nDCG@10 | Unanswerable precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.500 | 0.833 | 0.917 | 0.683 | 0.758 | 0.167 |
| dense_bge_small | 0.500 | 0.833 | 1.000 | 0.672 | 0.701 | 0.000 |
| dense_minilm | 0.417 | 0.750 | 0.833 | 0.595 | 0.665 | 0.000 |
| hybrid | 0.667 | 0.750 | 0.917 | 0.760 | 0.818 | 0.111 |
| hybrid_structured | 0.667 | 0.833 | 0.917 | 0.781 | 0.849 | 0.100 |
| tfidf | 0.750 | 0.917 | 0.917 | 0.845 | 0.875 | 0.250 |

---

## Knowledge Fabric (Phase 2)

| Asset | Count |
| --- | ---: |
| Documents | 24 |
| Passages | 182 |
| Entities | 413 |
| Events | 52 |
| Relations | 8 |
| Claims | 60 |

---

## Known Limitations

- The corpus covers only 24 source documents; recall is bounded by source coverage.
- Unanswerable precision is low (0.25 on test) — the abstention threshold is conservative.
- Dense retrieval (BGE-small) performs weaker than TF-IDF on this small domain corpus.
- No free-form generation; all answers are direct evidence passage extractions.
- No LLM rewriting was applied to queries.

---

*Generated from `evaluation/final_results.json` — source data in `reports/retrieval_results.json`.*
