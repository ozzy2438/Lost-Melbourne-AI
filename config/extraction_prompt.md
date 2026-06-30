# Historical knowledge extraction prompt — version 1.0.0

This prompt is reserved for an optional second-stage extractor. Phase 2 runs the
deterministic rules first and does not require an LLM.

## Versioned settings

- Prompt version: `1.0.0`
- Schema: `config/extraction_schema.json` version `1.0.0`
- Temperature: `0`
- Top-p: `1`
- Seed: `20260630` where supported
- Response format: JSON conforming to the schema

## System instruction

You extract candidate historical entities, events, relations, and factual claims from
one supplied passage. Use only the passage. Copy every supporting span exactly. Do not
infer a date, address, identity, relationship, coordinate, or event that the passage does
not state. Preserve vague dates verbatim. Use `null` rather than guessing. Return an empty
array when evidence is insufficient.

Names that merely look similar are not aliases. Record an alias only when the passage
explicitly states a renaming, former name, shared address, or equivalent identity.

## Input

The caller supplies exactly one object containing `passage_id`, `source_id`,
`source_url`, `licence`, and `text`.

## Required validation after generation

The caller must reject any item whose passage ID differs from the input, whose supporting
text is not an exact substring of `text`, whose type is outside the schema enums, or whose
claimed subject/object cannot be tied to the supporting span. Rejections are written to
`exclusions.jsonl`; LLM output never modifies raw evidence.
