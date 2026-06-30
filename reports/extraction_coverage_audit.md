# Phase 3 Extraction Coverage Audit

## Simple finding

The knowledge fabric is useful for evidence retrieval but sparse as a graph. The original Phase 2 run had 410 entities, 61 events and 7 relations. A source-span review found three unsafe relations/events and removed them, while expanded explicit `designed by` and `operated by` rules added four supported relations. The regenerated corpus has 413 entities, 52 events and 8 relations. The net relation count changed only slightly because precision was preferred over count inflation.

## Coverage

| Audit item | Count |
| --- | ---: |
| Entities with no event, relation or claim | 367 |
| Entities supported by only one passage | 251 |
| Documents producing no claims | 10 |
| Documents producing no relations | 20 |
| Unresolved aliases (aliases without explicit evidence) | 0 |
| Conservative possible duplicate-name pairs, not merged | 4 |
| Geographic-type entities without sourced coordinates | 346 |

### Relations by predicate

| Type | Count |
| --- | ---: |
| DEMOLISHED_IN | 1 |
| DESIGNED_BY | 5 |
| LOCATED_IN | 1 |
| OPERATED_BY | 1 |

### Events by type

| Type | Count |
| --- | ---: |
| closure | 1 |
| construction | 29 |
| demolition | 1 |
| heritage_listing | 3 |
| opening | 10 |
| redevelopment | 2 |
| relocation | 4 |
| renaming | 1 |
| renovation | 1 |

### Explicit aliases

- Eastern Market: Paddys Market
- North Melbourne: Hotham

### Possible duplicate names retained separately

- La Trobe Street ↔ Latrobe Street
- Prince's Bridge ↔ Princes Bridge
- Public Record Office ↔ Public Record Office Victoria
- Scots Church ↔ Scots' Church

## Interpretation

The low relation count is partly an extraction limitation and partly a source-shape limitation: these pages are broad narrative histories, while Phase 2 only accepts explicit named subject-predicate-object statements. Phase 3 improved descriptor handling for architects and operators, removed co-mention-only location edges, and rejected organisation names incorrectly selected as demolished structures. No similarity-only alias merge was made. Retrieval therefore uses entity mentions and passage evidence in addition to graph edges; graph expansion is a bounded bonus, not a source of answers.
