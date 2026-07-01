# Phase 2 Historical Knowledge Fabric Report

## Simple result

The 25 collected pages are now a linked historical evidence layer. One 28-word disambiguation page was excluded as unusable, leaving 24 cleaned documents. Every retained document keeps its source and licence; passages stay inside their source document; extracted entities, events, relations and claims point back to supporting passages. No embeddings, Transformer training, final RAG system or UI were created.

## Technical summary

| Record type | Count |
| --- | ---: |
| Documents | 24 |
| Passages | 182 |
| Entities | 413 |
| Historical events | 52 |
| Relations | 8 |
| Validated claims | 60 |
| Unsupported claims accepted | 0 |
| GeoJSON point features | 12 |
| Events with temporal expressions | 44 |
| Conflict groups | 0 |

### Entities by type

| Type | Count |
| --- | ---: |
| building | 67 |
| church | 51 |
| government_institution | 30 |
| hotel | 41 |
| landmark | 32 |
| market | 20 |
| organisation | 21 |
| person | 4 |
| place | 2 |
| railway_station | 16 |
| street | 112 |
| suburb | 6 |
| theatre | 11 |

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

### Relations by type

| Type | Count |
| --- | ---: |
| DEMOLISHED_IN | 1 |
| DESIGNED_BY | 5 |
| LOCATED_IN | 1 |
| OPERATED_BY | 1 |

## Five example entities

- `ent_9b2c88fe142f03be` — Royal Exhibition Building (building); support: `pass_5ce5606d70f1828a`
- `ent_b3639ba43c8599b6` — Public Record Office Victoria (government_institution); support: `pass_27a6fa066134beb6`
- `ent_6f68cb2dee115781` — Victorian Heritage Database (government_institution); support: `pass_6442c33f33aa65f8`
- `ent_46943f4e817a8347` — Hotel Windsor (hotel); support: `pass_bbd8e23a66d4664e`
- `ent_da5bb3cff56aead5` — Young and Jackson Hotel (hotel); support: `pass_519fbbdebbb9c941`

## Five example historical events

- `evt_0d85cbcae5c8e9d4` — opening; date: 1878; support: `pass_c84a9420195e17c1`
- `evt_0dd9a6a765c675fd` — construction; date: 1860s; support: `pass_c84a9420195e17c1`
- `evt_108430e0bd3b22b8` — construction; date: 1920s; support: `pass_6a9d3b12c5173a50`
- `evt_14cc684277665d3c` — construction; date: 1867; support: `pass_c1865c6c8b710e5f`
- `evt_1a4d5ffafda996a2` — heritage_listing; date: 1927; support: `pass_f6588730ceab38b2`

## Five example relations

- `rel_3c666008f6946048` — DESIGNED_BY from `ent_602749a5f51504e6` to `ent_456b18992261f49c`; support: `pass_64b916a429fca5c9`
- `rel_4f78fa7a38393efe` — OPERATED_BY from `ent_1b6a0d36fa4b47b8` to `ent_e33b17923a029127`; support: `pass_12b103446aa8a862`
- `rel_573b123bd4522596` — DESIGNED_BY from `ent_746dcb89cb29c4d9` to `ent_e1fc545200c08100`; support: `pass_d8b83d8852008979`
- `rel_5da43955fb7e6b53` — DESIGNED_BY from `ent_980d60a718678b06` to `ent_4448190e143b71ef`; support: `pass_6a9d3b12c5173a50`
- `rel_735814ac58fc4936` — DESIGNED_BY from `ent_46943f4e817a8347` to `ent_73717bb39ba5fa10`; support: `pass_9cd8ef1c7830bdfd`

## Five example claims and exact supporting spans

- **Apollo Hall / was_constructed / 1872** — “The building, Eastern Arcade and Apollo Hall, built in 1872, was constructed on the site of the old Haymarket Theatre.” (`pass_0996524939f8da77`)
- **Black Eagle Hotel / was_constructed / 1850** — “Another building known as the Black Eagle Hotel was built in 1850 as two storey Georgian terraces in Little Lonsdale street may have operated as a hotel from the outset.” (`pass_2a6b13dcf8c57264`)
- **Hotel Windsor / designed_by / {'entity_id': 'ent_73717bb39ba5fa10', 'name': 'Charles Webb'}** — “The best known survivor is the Hotel Windsor (1884) designed by Charles Webb and extended in 1888 as the Grand Coffee Palace.” (`pass_9cd8ef1c7830bdfd`)
- **Melbourne Fish Market / demolished_in / 1959** — “One of the largest and most spectacular landmarks, the Melbourne Fish Market (1889) was demolished in 1959 to make way for a carpark and road flyover.” (`pass_c6c02e8d2b683d6e`)
- **St James Old Cathedral / was_relocated / 1914** — “St James Old Cathedral (1839-1847) (relocated 1914), the most prominent of the few remaining buildings from the colonial era.” (`pass_e60a05ecc03936c0`)

## Complete provenance trace

`wiki_victoria_market_heritage.md` (SHA-256 `f0b263d7923eed450b19185f7d825c598baba92dacd39f210d00c123bf6f9512`)<br>
→ `doc_a0ee5cb925716d6c` (Queen Victoria Market)<br>
→ `pass_c84a9420195e17c1` (Overview)<br>
→ `ent_a538bb272575844b` (Queen Victoria Market)<br>
→ `evt_0d85cbcae5c8e9d4` (opening)<br>
→ `clm_b2391cc43999e5f0` (opened)

## Geographic coverage and sample locations

12 exact point features were copied from coordinates stated by sources; no coordinates were invented.

- Port Melbourne: -37.82389, 144.91111 (https://en.wikipedia.org/wiki/Port_Melbourne,_Victoria)
- Federation Square: -37.817798, 144.968714 (https://en.wikipedia.org/wiki/Federation_Square)
- Hotel Windsor: -37.81194, 144.97278 (https://en.wikipedia.org/wiki/Hotel_Windsor,_Melbourne)

## Split counts

test: 3, train: 19, validation: 2. Splits are assigned at document level; passages and claims inherit document membership and cannot leak across splits.
