# Phase 3 Retrieval Benchmark

## Simple finding

The benchmark selected **tfidf** using a development-only reliability score (50% MRR, 30% Recall@5, 20% unanswerable precision), then reported the held-out test metrics without retuning. The winning dense encoder was **bge-small**, the winning passage strategy was **original**, and the winning deterministic query transformation was **entity**. Dense search uses passage vectors and explicit NumPy cosine similarity; hybrid search uses Reciprocal Rank Fusion with visible bonuses.

## Held-out test comparison

| Method | R@1 | R@3 | R@5 | MRR | nDCG@10 | Unanswerable precision | Index bytes | Index seconds | Avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.500 | 0.833 | 0.917 | 0.683 | 0.758 | 0.167 | 436,882 | 0.022 | 0.440 |
| TF-IDF | 0.750 | 0.917 | 0.917 | 0.845 | 0.875 | 0.250 | 1,226,457 | 0.069 | 0.371 |
| Dense MiniLM | 0.417 | 0.750 | 0.833 | 0.595 | 0.665 | 0.000 | 259,698 | 1.890 | 9.098 |
| Dense BGE-small | 0.500 | 0.833 | 1.000 | 0.672 | 0.701 | 0.000 | 260,254 | 4.988 | 11.301 |
| Hybrid | 0.667 | 0.750 | 0.917 | 0.760 | 0.818 | 0.111 | 697,136 | 5.010 | 10.240 |
| Hybrid + structured | 0.667 | 0.833 | 0.917 | 0.781 | 0.849 | 0.100 | 697,136 | 5.010 | 11.028 |

## Dense model facts measured locally

| Model | Revision | Dimension | Licence | Estimated model memory | Cached model bytes | Original index bytes |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| BAAI/bge-small-en-v1.5 | `5c38ec7c405e` | 384 | MIT | 133,440,000 | 134,505,940 | 260,254 |
| sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4` | 384 | Apache-2.0 | 90,852,864 | 91,599,528 | 259,698 |

## Five successful queries

- `q001` Who designed the Metropolitan Meat Market? — top evidence `pass_64b916a429fca5c9`
- `q003` Who designed the Great Hall stained glass ceiling? — top evidence `pass_d8b83d8852008979`
- `q004` Which architects designed Bolte Bridge? — top evidence `pass_6a9d3b12c5173a50`
- `q005` Who was the architect of Hotel Windsor? — top evidence `pass_bbd8e23a66d4664e`
- `q006` Who designed Sacred Heart Church in St Kilda? — top evidence `pass_6e5038f696078efa`

## Five failed queries

- `q002` Which organisation operates Richmond Town Hall? — lexical_mismatch
- `q016` In which period was Richmond Town Hall built? — date_filter_failure
- `q018` What kind of temporary structures were described at Eastern Market in 1854? — lexical_mismatch
- `q026` When did Princes Bridge station close? — date_filter_failure
- `q033` How far is Port Melbourne from central Melbourne? — location_ambiguity

## Required query slices

### Alias-heavy

| Query | Question | Top passage | Abstained | Success |
| --- | --- | --- | --- | --- |
| q039 | What was Paddys Market also called? | `pass_f9d3e777eb3e3f3e` | False | True |
| q040 | What kinds of activity took place at Paddys Market? | `pass_f9d3e777eb3e3f3e` | True | False |
| q041 | When were the buildings of Paddys Market demolished? | `pass_f9d3e777eb3e3f3e` | False | True |

### Temporal

| Query | Question | Top passage | Abstained | Success |
| --- | --- | --- | --- | --- |
| q009 | When did Queen Victoria Market officially open? | `pass_c84a9420195e17c1` | False | True |
| q016 | In which period was Richmond Town Hall built? | `pass_12b103446aa8a862` | True | False |
| q045 | Compare the opening dates of Queen Victoria Market and Princes Bridge Hotel. | `pass_c84a9420195e17c1` | False | True |

### Geographic

| Query | Question | Top passage | Abstained | Success |
| --- | --- | --- | --- | --- |
| q007 | Where is South Melbourne Market located? | `pass_3ef76e53e39458b1` | False | True |
| q032 | What is the street address of Hotel Windsor? | `pass_bbd8e23a66d4664e` | False | True |
| q050 | Compare the recorded locations of Port Melbourne and Federation Square. | `pass_584b704e5c9fb2cd` | False | True |

### Unanswerable

| Query | Question | Top passage | Abstained | Success |
| --- | --- | --- | --- | --- |
| q051 | What was the ticket price on Eastern Market's opening day? | `pass_143d209c0f4c231e` | True | True |
| q052 | Which company insured the Royal Exhibition Building in 1880? | `pass_5ce5606d70f1828a` | False | False |
| q060 | What was the official name of the car park that replaced Melbourne Fish Market? | `pass_c6c02e8d2b683d6e` | True | True |

Full per-query score breakdowns, expected passages, transformations, filters, latency, and success flags are in `reports/retrieval_results.json`.

## Chunk strategy comparison

| Strategy | Test R@1 | Test R@5 | Test MRR |
| --- | ---: | ---: | ---: |
| original | 0.667 | 0.917 | 0.760 |
| small | 0.667 | 0.750 | 0.729 |
| parent_child | 0.667 | 0.750 | 0.729 |

## Query transformation comparison

| Strategy | Test R@1 | Test R@5 | Test MRR |
| --- | ---: | ---: | ---: |
| none | 0.667 | 0.917 | 0.760 |
| alias | 0.667 | 0.917 | 0.760 |
| entity | 0.667 | 0.917 | 0.760 |

LLM rewriting was not run. The original query is always retained, and deterministic expansions are logged separately.
