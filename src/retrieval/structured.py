"""Source-grounded structured filters and graph-neighbour expansion."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .corpus import HistoricalCorpus
from .models import RetrievalResult, SearchPassage
from .query import detect_entity_mentions


YEAR_RANGE_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\s*(?:-|–|to|and)\s*(1[0-9]{3}|20[0-2][0-9])\b", re.I)
EVENT_TERMS = {
    "demolish": "demolition",
    "demolished": "demolition",
    "construction": "construction",
    "constructed": "construction",
    "built": "construction",
    "opened": "opening",
    "opening": "opening",
    "closed": "closure",
    "closure": "closure",
    "renamed": "renaming",
    "relocated": "relocation",
    "heritage": "heritage_listing",
}


@dataclass(frozen=True)
class StructuredSignal:
    passage_ids: frozenset[str]
    detected_entity_ids: tuple[str, ...]
    filters: tuple[str, ...]


class StructuredRetriever:
    def __init__(self, corpus: HistoricalCorpus):
        self.corpus = corpus
        self.entity_by_id = corpus.entity_by_id
        self.entity_passages = {entity["entity_id"]: set(entity["supporting_passage_ids"]) for entity in corpus.entities}
        self.graph: dict[str, set[str]] = defaultdict(set)
        for relation in corpus.relations:
            object_id = relation.get("object_entity_id")
            if object_id:
                self.graph[relation["subject_entity_id"]].add(object_id)
                self.graph[object_id].add(relation["subject_entity_id"])

    def detect(self, query: str) -> StructuredSignal:
        lowered = query.casefold()
        detected: list[str] = []
        evidence: set[str] = set()
        filters: list[str] = []
        for entity, _ in detect_entity_mentions(self.corpus.entities, query):
            detected.append(entity["entity_id"])
            evidence.update(entity["supporting_passage_ids"])
            for neighbour in self.graph.get(entity["entity_id"], set()):
                evidence.update(self.entity_passages.get(neighbour, set()))
            filters.append(f"entity:{entity['canonical_name']}")

        event_type = next((value for term, value in EVENT_TERMS.items() if re.search(rf"\b{term}\w*\b", lowered)), None)
        year_match = YEAR_RANGE_RE.search(query)
        start_year, end_year = (int(year_match.group(1)), int(year_match.group(2))) if year_match else (None, None)
        if start_year is not None:
            filters.append(f"date:{start_year}-{end_year}")
        if event_type:
            filters.append(f"event:{event_type}")
        if event_type or start_year is not None:
            matching_events = []
            for event in self.corpus.events:
                if event_type and event["event_type"] != event_type:
                    continue
                normalised = event.get("normalised_date") or {}
                event_start, event_end = normalised.get("start_year"), normalised.get("end_year")
                if start_year is not None and (event_start is None or event_end < start_year or event_start > end_year):
                    continue
                matching_events.append(event)
            event_evidence = {passage_id for event in matching_events for passage_id in event["supporting_passage_ids"]}
            evidence = (evidence & event_evidence) if detected and event_evidence else evidence | event_evidence

        for feature in self.corpus.places["features"]:
            properties = feature["properties"]
            place_terms = [properties.get("name"), properties.get("suburb"), properties.get("address")]
            if any(term and term.casefold() in lowered for term in place_terms):
                evidence.update(properties["supporting_passage_ids"])
                filters.append(f"geo:{properties['name']}")
        return StructuredSignal(frozenset(evidence), tuple(sorted(set(detected))), tuple(dict.fromkeys(filters)))

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        passages: list[SearchPassage],
        top_k: int = 10,
    ) -> tuple[list[RetrievalResult], StructuredSignal]:
        signal = self.detect(query)
        if not signal.passage_ids:
            return results[:top_k], signal
        copied: list[RetrievalResult] = []
        for result in results:
            bonus = 0.05 if result.evaluation_id() in signal.passage_ids or result.passage_id in signal.passage_ids else 0.0
            components = dict(result.score_components)
            components["structured_bonus"] = bonus
            explanations = [*result.explanation]
            if bonus:
                explanations.append(f"structured evidence match: {', '.join(signal.filters)}")
            copied.append(RetrievalResult(
                passage_id=result.passage_id,
                parent_passage_id=result.parent_passage_id,
                score=result.score + bonus,
                score_components=components,
                explanation=explanations,
            ))
        copied.sort(key=lambda result: (-result.score, result.passage_id))
        for rank, result in enumerate(copied, 1):
            result.rank = rank
        return copied[:top_k], signal

    def passages_for_date_range(self, event_type: str, start_year: int, end_year: int) -> set[str]:
        output: set[str] = set()
        for event in self.corpus.events:
            if event["event_type"] != event_type:
                continue
            normalised = event.get("normalised_date") or {}
            start, end = normalised.get("start_year"), normalised.get("end_year")
            if start is not None and end >= start_year and start <= end_year:
                output.update(event["supporting_passage_ids"])
        return output

    def passages_for_place(self, place_name: str) -> set[str]:
        lowered = place_name.casefold()
        return {
            passage_id
            for feature in self.corpus.places["features"]
            if any(
                value and lowered in value.casefold()
                for value in (feature["properties"].get("name"), feature["properties"].get("suburb"), feature["properties"].get("address"))
            )
            for passage_id in feature["properties"]["supporting_passage_ids"]
        }
