"""Shared data structures for transparent retrieval implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchPassage:
    passage_id: str
    parent_passage_id: str
    document_id: str
    text: str
    search_text: str
    title: str
    section_title: str
    entity_names: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    passage_id: str
    parent_passage_id: str
    score: float
    rank: int = 0
    score_components: dict[str, float] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)

    def evaluation_id(self) -> str:
        return self.parent_passage_id or self.passage_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "parent_passage_id": self.parent_passage_id,
            "score": self.score,
            "rank": self.rank,
            "score_components": self.score_components,
            "explanation": self.explanation,
        }
