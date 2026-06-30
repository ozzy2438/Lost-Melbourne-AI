# Phase 2 Historical Knowledge Fabric Report

## Simple result

The 25 collected pages are now a linked historical evidence layer. One 28-word disambiguation page was excluded as unusable, leaving 24 cleaned documents. Every retained document keeps its source and licence; passages stay inside their source document; extracted entities, events, relations and claims point back to supporting passages. No embeddings, Transformer training, final RAG system or UI were created.

## Technical summary

| Record type | Count |
| --- | ---: |
| Documents | 24 |
| Passages | 182 |
| Entities | 410 |
| Historical events | 61 |
| Relations | 7 |
| Validated claims | 68 |
| Unsupported claims accepted | 0 |
| GeoJSON point features | 12 |
| Events with temporal expressions | 50 |
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
| organisation | 19 |
| person | 3 |
| place | 2 |
| railway_station | 16 |
| street | 112 |
| suburb | 6 |
| theatre | 11 |

### Events by type

| Type | Count |
| --- | ---: |
| closure | 1 |
| construction | 31 |
| demolition | 2 |
| heritage_listing | 4 |
| opening | 14 |
| redevelopment | 2 |
| relocation | 5 |
| renaming | 1 |
| renovation | 1 |

### Relations by type

| Type | Count |
| --- | ---: |
| DEMOLISHED_IN | 2 |
| DESIGNED_BY | 3 |
| LOCATED_IN | 2 |

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

- `rel_573b123bd4522596` — DESIGNED_BY from `ent_746dcb89cb29c4d9` to `ent_e1fc545200c08100`; support: `pass_d8b83d8852008979`
- `rel_57eeedc50209c9f0` — DESIGNED_BY from `ent_3f75b9f1cdfb14c9` to `ent_4e83c5c5b5f60f70`; support: `pass_6e5038f696078efa`
- `rel_735814ac58fc4936` — DESIGNED_BY from `ent_46943f4e817a8347` to `ent_73717bb39ba5fa10`; support: `pass_9cd8ef1c7830bdfd`
- `rel_76292d4a95138a37` — LOCATED_IN from `ent_11ebaa16511600d1` to `ent_7c9b48c690a949ec`; support: `pass_64b916a429fca5c9`
- `rel_86d7e8b021afd6fa` — LOCATED_IN from `ent_b25ac7536e8bf87b` to `ent_6c8ebb854a44ee52`; support: `pass_3ef76e53e39458b1`

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
