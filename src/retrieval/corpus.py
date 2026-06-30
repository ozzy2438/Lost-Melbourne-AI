"""Load and strictly validate the Phase 2 Historical Knowledge Fabric."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "documents.jsonl",
    "passages.jsonl",
    "entities.jsonl",
    "events.jsonl",
    "relations.jsonl",
    "claims.jsonl",
    "places.geojson",
    "split_manifest.json",
)


class CorpusValidationError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusValidationError(f"invalid JSON in {path.name}:{line_number}: {exc}") from exc
    if not rows:
        raise CorpusValidationError(f"required Phase 2 file is empty: {path}")
    return rows


@dataclass
class HistoricalCorpus:
    root: Path
    documents: list[dict[str, Any]]
    passages: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    events: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    places: dict[str, Any]
    split_manifest: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "HistoricalCorpus":
        root = root.resolve()
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise CorpusValidationError(f"required Phase 2 outputs missing: {', '.join(missing)}")
        places = json.loads((root / "places.geojson").read_text(encoding="utf-8"))
        split_manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
        corpus = cls(
            root=root,
            documents=read_jsonl(root / "documents.jsonl"),
            passages=read_jsonl(root / "passages.jsonl"),
            entities=read_jsonl(root / "entities.jsonl"),
            events=read_jsonl(root / "events.jsonl"),
            relations=read_jsonl(root / "relations.jsonl"),
            claims=read_jsonl(root / "claims.jsonl"),
            places=places,
            split_manifest=split_manifest,
        )
        corpus.validate()
        return corpus

    def validate(self) -> None:
        if self.places.get("type") != "FeatureCollection" or not isinstance(self.places.get("features"), list):
            raise CorpusValidationError("places.geojson must be a FeatureCollection")
        document_ids = _unique_ids(self.documents, "document_id")
        passage_ids = _unique_ids(self.passages, "passage_id")
        entity_ids = _unique_ids(self.entities, "entity_id")
        _unique_ids(self.events, "event_id")
        _unique_ids(self.relations, "relation_id")
        _unique_ids(self.claims, "claim_id")

        for passage in self.passages:
            _require(passage["document_id"] in document_ids, f"passage has unknown document: {passage['passage_id']}")
            _require(bool(passage.get("text", "").strip()), f"passage text is empty: {passage['passage_id']}")
        for entity in self.entities:
            supports = set(entity.get("supporting_passage_ids", []))
            _require(bool(supports) and supports <= passage_ids, f"entity support is invalid: {entity['entity_id']}")
        for event in self.events:
            _require(set(event.get("involved_entity_ids", [])) <= entity_ids, f"event entity is invalid: {event['event_id']}")
            _require(set(event.get("supporting_passage_ids", [])) <= passage_ids, f"event support is invalid: {event['event_id']}")
        for relation in self.relations:
            _require(relation.get("subject_entity_id") in entity_ids, f"relation subject is invalid: {relation['relation_id']}")
            object_id = relation.get("object_entity_id")
            _require(object_id is None or object_id in entity_ids, f"relation object is invalid: {relation['relation_id']}")
            _require(set(relation.get("supporting_passage_ids", [])) <= passage_ids, f"relation support is invalid: {relation['relation_id']}")

        passage_by_id = self.passage_by_id
        for claim in self.claims:
            passage = passage_by_id.get(claim.get("passage_id"))
            _require(passage is not None, f"claim passage is invalid: {claim['claim_id']}")
            subject_id = claim.get("subject", {}).get("entity_id")
            _require(subject_id in entity_ids, f"claim subject is invalid: {claim['claim_id']}")
            object_value = claim.get("object_or_value")
            if isinstance(object_value, dict) and object_value.get("entity_id"):
                _require(object_value["entity_id"] in entity_ids, f"claim object is invalid: {claim['claim_id']}")
            span = claim.get("supporting_span", {})
            start, end = span.get("start"), span.get("end")
            _require(isinstance(start, int) and isinstance(end, int) and 0 <= start < end, f"claim span is invalid: {claim['claim_id']}")
            _require(passage["text"][start:end] == claim.get("supporting_text"), f"claim span does not match: {claim['claim_id']}")

        for feature in self.places["features"]:
            entity_id = feature.get("properties", {}).get("canonical_entity_id")
            _require(entity_id in entity_ids, f"GeoJSON entity is invalid: {entity_id}")
            coordinates = feature.get("geometry", {}).get("coordinates", [])
            _require(len(coordinates) == 2 and -180 <= coordinates[0] <= 180 and -90 <= coordinates[1] <= 90, f"GeoJSON coordinate is invalid: {entity_id}")

        assignments = self.split_manifest.get("assignments", [])
        assigned = [row.get("document_id") for row in assignments]
        _require(len(assigned) == len(set(assigned)), "split manifest contains document overlap")
        _require(set(assigned) == document_ids, "split manifest omits or adds documents")

    @property
    def document_by_id(self) -> dict[str, dict[str, Any]]:
        return {row["document_id"]: row for row in self.documents}

    @property
    def passage_by_id(self) -> dict[str, dict[str, Any]]:
        return {row["passage_id"]: row for row in self.passages}

    @property
    def entity_by_id(self) -> dict[str, dict[str, Any]]:
        return {row["entity_id"]: row for row in self.entities}

    @property
    def entities_by_passage(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in self.entities:
            for passage_id in entity["supporting_passage_ids"]:
                result[passage_id].append(entity)
        return result

    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "passages": len(self.passages),
            "entities": len(self.entities),
            "events": len(self.events),
            "relations": len(self.relations),
            "claims": len(self.claims),
            "places": len(self.places["features"]),
            "splits": len(self.split_manifest["assignments"]),
        }


def _unique_ids(rows: list[dict[str, Any]], key: str) -> set[str]:
    values = [row.get(key) for row in rows]
    if any(not value for value in values):
        raise CorpusValidationError(f"missing {key}")
    if len(values) != len(set(values)):
        raise CorpusValidationError(f"duplicate {key}")
    return set(values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusValidationError(message)
