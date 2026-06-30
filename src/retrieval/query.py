"""Deterministic alias and entity-aware query expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import HistoricalCorpus


@dataclass(frozen=True)
class QueryTransform:
    original: str
    transformed: str
    detected_entity_ids: tuple[str, ...]
    expansions: tuple[str, ...]


class QueryTransformer:
    def __init__(self, corpus: HistoricalCorpus):
        self.entities = corpus.entities

    def transform(self, query: str, mode: str = "none") -> QueryTransform:
        if mode not in {"none", "alias", "entity"}:
            raise ValueError(f"unknown query transformation: {mode}")
        detected: list[str] = []
        expansions: list[str] = []
        lowered = query.casefold()
        for entity, matched_name in detect_entity_mentions(self.entities, query):
            canonical = entity["canonical_name"]
            aliases = entity.get("aliases", [])
            detected.append(entity["entity_id"])
            if mode in {"alias", "entity"}:
                for value in [canonical, *aliases]:
                    if value.casefold() not in lowered and value not in expansions:
                        expansions.append(value)
        transformed = query if not expansions else f"{query} {' '.join(expansions)}"
        return QueryTransform(query, transformed, tuple(sorted(set(detected))), tuple(expansions))


def detect_entity_mentions(entities: list[dict], query: str) -> list[tuple[dict, str]]:
    """Return longest non-overlapping canonical/alias matches."""
    lowered = query.casefold()
    matches: list[tuple[int, int, dict, str]] = []
    for entity in entities:
        for name in [entity["canonical_name"], *entity.get("aliases", [])]:
            for match in re.finditer(rf"\b{re.escape(name.casefold())}\b", lowered):
                matches.append((match.start(), match.end(), entity, name))
    matches.sort(key=lambda row: (-(row[1] - row[0]), row[0], row[2]["entity_id"]))
    selected: list[tuple[int, int, dict, str]] = []
    for candidate in matches:
        if any(not (candidate[1] <= existing[0] or candidate[0] >= existing[1]) for existing in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda row: row[0])
    return [(entity, name) for _, _, entity, name in selected]
