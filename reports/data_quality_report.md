# Phase 2 Data Quality Report

Generated deterministically by `scripts/prepare_corpus.py`. Raw evidence was hash-checked before and after processing.

| Metric | Value |
| --- | ---: |
| Raw Markdown documents | 25 |
| Processed documents | 24 |
| Raw words (including captured page chrome) | 106,206 |
| Cleaned words | 66,558 |
| Approximate cleaned tokens | 88,523 |
| Passages | 182 |
| Exact duplicate documents | 0 |
| Normalised near-duplicate documents | 0 |
| Exclusion records | 26 |
| Rejected unsupported claims | 0 |

Largest source: `wiki_architecture_melbourne` (13,045 words).
Smallest source: `wiki_south_melbourne_market` (271 words).

## Exclusions and reasons

- `boilerplate_removed`: 25
- `insufficient_historical_content`: 1

`wiki_moomba` was excluded after cleaning because the collected page is a 28-word disambiguation page, below the 80-word historical-content threshold. Boilerplate exclusions record removed navigation/footer lines without copying them.

## Dataset splits

| Split | Documents |
| --- | ---: |
| test | 3 |
| train | 19 |
| validation | 2 |

## Training corpus

Final training corpus: **59,389 words** (approximately **78,988 tokens**). Only cleaned source text with a compatible licence is included; each source is capped at 6,000 words.
