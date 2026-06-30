# Collection Report — Phase 1 Recovery

**Date:** 2026-06-30  
**Run by:** `python3 scripts/collect.py --refresh`  
**Canonical repo:** `/Users/osmanorka/Lost-Melbourne-AI-1`

---

## Recovery findings (Part A)

The original Phase 1 report (commit `276b0b0`) claimed 27 Markdown files and ~520,912 bytes
collected. Neither `scripts/collect.py`, `config/sources.yaml`, any Markdown files, nor a
manifest were found anywhere on the local machine, in git history, in stashes, in worktrees,
or on the remote branch. The previous result is treated as unverified.

**Recovery verdict: FAILED — original files do not exist. Rebuilt from scratch.**

---

## Corpus description

This is a **replacement corpus**, not an exact reconstruction of the earlier unverified run.
Sources were newly selected from reputable public sources about demolished buildings, heritage
places and changing neighbourhoods in Melbourne. All sources were verified to return HTTP 200
before inclusion in the final config.

---

## Collection summary

| Metric | Value |
| --- | --- |
| Configured sources (enabled) | 25 |
| Successful | 25 |
| Failed | 0 |
| Skipped (already cached) | 0 |
| Markdown files | 25 |
| HTML files | 25 |
| Metadata records | 25 |
| Total Markdown bytes | 739,293 |
| Empty documents | 0 |
| Duplicate content hashes | 0 |
| Sources with unclear licences | 0 |
| Files missing provenance | 0 |

---

## Sources collected

| source\_id | Licence | MD bytes |
| --- | --- | --- |
| wiki\_architecture\_melbourne | CC-BY-SA-4.0 | 122,019 |
| wiki\_federation\_square | CC-BY-SA-4.0 | 39,554 |
| wiki\_victoria\_market\_heritage | CC-BY-SA-4.0 | 20,265 |
| wiki\_moomba | CC-BY-SA-4.0 | 2,720 |
| wiki\_swanston\_street | CC-BY-SA-4.0 | 15,645 |
| wiki\_flinders\_street\_station | CC-BY-SA-4.0 | 70,044 |
| wiki\_royal\_exhibition\_building | CC-BY-SA-4.0 | 38,633 |
| wiki\_princes\_bridge | CC-BY-SA-4.0 | 20,892 |
| wiki\_melbournes\_past | CC-BY-SA-4.0 | 54,915 |
| wiki\_eastern\_market\_melbourne | CC-BY-SA-4.0 | 26,707 |
| wiki\_south\_melbourne\_market | CC-BY-SA-4.0 | 5,955 |
| wiki\_young\_and\_jacksons | CC-BY-SA-4.0 | 14,561 |
| wiki\_hotel\_windsor | CC-BY-SA-4.0 | 21,616 |
| wiki\_bourke\_street | CC-BY-SA-4.0 | 15,810 |
| wiki\_fitzroy\_suburb | CC-BY-SA-4.0 | 51,280 |
| wiki\_collingwood\_suburb | CC-BY-SA-4.0 | 25,529 |
| wiki\_carlton\_suburb | CC-BY-SA-4.0 | 26,087 |
| wiki\_richmond\_suburb | CC-BY-SA-4.0 | 20,661 |
| wiki\_north\_melbourne\_suburb | CC-BY-SA-4.0 | 16,434 |
| wiki\_port\_melbourne | CC-BY-SA-4.0 | 28,344 |
| vhd\_introduction | OGL-AU | 19,888 |
| prov\_about | CC-BY-4.0 | 7,684 |
| wiki\_melbourne\_gaol | CC-BY-SA-4.0 | 46,095 |
| wiki\_collins\_street | CC-BY-SA-4.0 | 18,780 |
| wiki\_spencer\_street | CC-BY-SA-4.0 | 9,175 |

---

## Disabled sources

| source\_id | Reason |
| --- | --- |
| wikidata\_demolished\_buildings\_mel | `query.wikidata.org` disallows generic agents in robots.txt |
| wiki\_api\_lost\_melbourne\_category | `/w/api.php` is within `Disallow: /w/` for wildcard agents |

---

## Licence summary

| Licence | Sources | Notes |
| --- | --- | --- |
| CC-BY-SA-4.0 | 23 | Wikipedia articles — attribution + share-alike required |
| OGL-AU | 1 | Victorian Heritage Database — Australian Open Govt Licence |
| CC-BY-4.0 | 1 | Public Record Office Victoria |

No sources with unclear or review-needed licences were included in the final corpus.

---

## Corpus integrity

- All 25 Markdown files contain a `<!-- PROVENANCE ... -->` header.
- No duplicate content hashes detected.
- No empty documents.
- Safe restart verified: running collector from `/tmp` correctly skips cached files.
- `robots.txt` checked for every host before fetching (via `requests` — not `urllib`, which
  was found to produce incorrect results for `en.wikipedia.org`).

---

## Backup

Raw corpus backed up to:

```text
~/Documents/Lost-Melbourne-AI-backups/raw_corpus_20260630.tar.gz
```

Size: ~1.2 MB (compressed). Contains `data/raw/html/`, `data/raw/markdown/`,
`data/raw/metadata/`, and `data/raw/manifest.jsonl`.

---

## Test results

```
Ran 23 tests in 0.095s

OK
```

All 23 unit tests pass using local fixtures only (no live network).
