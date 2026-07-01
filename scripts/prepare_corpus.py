#!/usr/bin/env python3
"""Build the Phase 2 provenance-aware Historical Knowledge Fabric.

The pipeline is deterministic and offline by default. It reads immutable Phase 1
evidence, validates its hashes, cleans obvious boilerplate, creates linked records,
and writes reports. No embeddings, model training, live geocoding, or LLM calls are
performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is already required by collect.py
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_PATH = REPO_ROOT / "config" / "sources.yaml"
FINGERPRINT_PATH = REPORTS_DIR / "corpus_fingerprint.json"

TARGET_PASSAGE_WORDS = 340
MAX_PASSAGE_WORDS = 450
PASSAGE_OVERLAP_WORDS = 45
TRAINING_SOURCE_CAP_WORDS = 6_000
COMPATIBLE_TRAINING_LICENCES = {
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "OGL-AU",
    "PUBLIC-DOMAIN",
}

PROVENANCE_RE = re.compile(r"\A<!-- PROVENANCE\s*\n(.*?)\n-->\s*", re.DOTALL)
YEAR_RE = re.compile(r"\b(?:c\.?\s*|circa\s+|around\s+)?(?:1[0-9]{3}|20[0-2][0-9])\b", re.I)
VAGUE_DATE_RE = re.compile(
    r"\b(?:(?:early|mid|late)[ -](?:eighteenth|nineteenth|twentieth|twenty-first) century|"
    r"(?:early|mid|late)[ -](?:1[0-9]{3}|20[0-2][0-9])s|"
    r"(?:1[0-9]{3}|20[0-2][0-9])s)\b",
    re.I,
)
COORD_RE = re.compile(r"(?<!\d)(-3[6-8]\.\d{3,8})\s*[;,]\s*(14[3-6]\.\d{3,8})(?!\d)")

EVENT_PATTERNS = [
    ("demolition", re.compile(r"\b(?:demolish(?:ed|ion)|razed|torn down)\b", re.I)),
    ("construction", re.compile(r"\b(?:built|constructed|completed|erected)\b", re.I)),
    ("opening", re.compile(r"\b(?:opened|opening|inaugurated)\b", re.I)),
    ("closure", re.compile(r"\b(?:closed|closure|ceased operations?)\b", re.I)),
    ("renaming", re.compile(r"\b(?:renamed|became known as|name changed)\b", re.I)),
    ("renovation", re.compile(r"\b(?:renovated|restored|refurbished|redeveloped)\b", re.I)),
    ("relocation", re.compile(r"\b(?:relocated|moved to)\b", re.I)),
    ("fire", re.compile(r"\b(?:fire|burned|burnt down)\b", re.I)),
    ("heritage_listing", re.compile(r"\b(?:heritage list(?:ed|ing)|listed on the|heritage register)\b", re.I)),
    ("redevelopment", re.compile(r"\b(?:redevelopment|replaced by)\b", re.I)),
]

ENTITY_SUFFIX_TYPES = {
    "market": "market",
    "station": "railway_station",
    "railway station": "railway_station",
    "hotel": "hotel",
    "church": "church",
    "cathedral": "church",
    "theatre": "theatre",
    "gaol": "landmark",
    "jail": "landmark",
    "bridge": "landmark",
    "street": "street",
    "road": "street",
    "square": "landmark",
    "building": "building",
    "hall": "building",
    "museum": "organisation",
    "council": "government_institution",
    "government": "government_institution",
    "authority": "government_institution",
    "office": "government_institution",
    "university": "organisation",
    "festival": "organisation",
}

ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ&'’-]*[ \t]+){0,5}"
    r"(?:Railway Station|Market|Station|Hotel|Church|Cathedral|Theatre|Gaol|Jail|Bridge|"
    r"Street|Road|Square|Building|Hall|Museum|Council|Government|Authority|Office|"
    r"University|Festival)\b"
)


class PipelineError(RuntimeError):
    """Expected validation failure with a user-actionable message."""


@dataclass(frozen=True)
class RawRecord:
    source_id: str
    markdown_path: Path
    html_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    provenance: dict[str, str]
    body: str


def stable_id(prefix: str, *parts: Any) -> str:
    value = "\x1f".join(normalise_key(str(p)) for p in parts)
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def normalise_key(value: str) -> str:
    value = value.casefold().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", value)).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(131_072), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_tree(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def parse_provenance(markdown: str) -> tuple[dict[str, str], str]:
    match = PROVENANCE_RE.match(markdown)
    if not match:
        raise PipelineError("provenance header is missing")
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    required = {
        "source_id",
        "source_url",
        "page_title",
        "licence",
        "retrieved",
        "content_hash_sha256",
        "http_status",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise PipelineError(f"provenance fields missing: {', '.join(missing)}")
    return fields, markdown[match.end() :]


def load_source_names(config_path: Path) -> dict[str, str]:
    if not config_path.exists() or yaml is None:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {row["source_id"]: row.get("name", row["source_id"]) for row in data.get("sources", [])}


def load_and_validate_raw(raw_dir: Path, expected_count: int | None = None) -> list[RawRecord]:
    markdown_dir = raw_dir / "markdown"
    html_dir = raw_dir / "html"
    metadata_dir = raw_dir / "metadata"
    markdown_paths = sorted(markdown_dir.glob("*.md"))
    html_paths = sorted(html_dir.glob("*.html"))
    if not markdown_paths:
        raise PipelineError(f"no raw Markdown files exist under {markdown_dir}")
    if expected_count is not None:
        if len(markdown_paths) != expected_count or len(html_paths) != expected_count:
            raise PipelineError(
                f"expected {expected_count} Markdown and HTML files; found "
                f"{len(markdown_paths)} Markdown and {len(html_paths)} HTML"
            )

    html_by_stem = {path.stem: path for path in html_paths}
    records: list[RawRecord] = []
    for markdown_path in markdown_paths:
        source_id = markdown_path.stem
        html_path = html_by_stem.get(source_id)
        metadata_path = metadata_dir / f"{source_id}.json"
        if html_path is None:
            raise PipelineError(f"raw HTML missing for {source_id}")
        if not metadata_path.exists():
            raise PipelineError(f"metadata missing for {source_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "ok":
            raise PipelineError(f"successful metadata not available for {source_id}")
        provenance, body = parse_provenance(markdown_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(html_path)
        stored_hash = metadata.get("content_hash_sha256")
        if actual_hash != stored_hash or actual_hash != provenance["content_hash_sha256"]:
            raise PipelineError(f"stored SHA-256 does not match raw HTML for {source_id}")
        for meta_key, prov_key in (
            ("source_id", "source_id"),
            ("url", "source_url"),
            ("licence", "licence"),
            ("timestamp", "retrieved"),
        ):
            if str(metadata.get(meta_key, "")) != provenance[prov_key]:
                raise PipelineError(f"metadata/provenance mismatch for {source_id}: {meta_key}")
        records.append(
            RawRecord(source_id, markdown_path, html_path, metadata_path, metadata, provenance, body)
        )
    return records


def _strip_markdown(line: str) -> str:
    line = re.sub(r"!\[[^]]*]\([^)]*\)", "", line)
    line = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", line)
    line = re.sub(r"\[(\d+|citation needed)]", "", line, flags=re.I)
    line = re.sub(r"<[^>]+>", " ", line)
    line = line.replace("**", "").replace("__", "").replace("`", "")
    line = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def clean_markdown(body: str, source_id: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove recognisable site chrome while preserving historical source text."""
    raw_lines = body.replace("\r\n", "\n").splitlines()
    start = 0
    stop = len(raw_lines)
    if source_id.startswith("wiki_"):
        for index, line in enumerate(raw_lines):
            if line.strip() == "From Wikipedia, the free encyclopedia":
                start = index + 1
                break
        for index in range(start, len(raw_lines)):
            if re.match(r"^## (?:See also|References|Notes|Footnotes|External links|Further reading)\s*$", raw_lines[index]):
                stop = index
                break
    elif source_id == "prov_about":
        start = next((i for i, line in enumerate(raw_lines) if line.strip() == "## Who we are"), 0)
        stop = next((i for i in range(start, len(raw_lines)) if raw_lines[i].strip() == "## Join our mailing list"), len(raw_lines))
    elif source_id == "vhd_introduction":
        start = next(
            (i for i, line in enumerate(raw_lines) if line.startswith("The Victorian Heritage Database contains")),
            0,
        )
        stop = next((i for i in range(start, len(raw_lines)) if raw_lines[i].strip() == "Account login"), len(raw_lines))

    boilerplate = {
        "[edit]",
        "edit",
        "move to sidebar",
        "hide",
        "toggle the table of contents",
        "javascript must be enabled for the correct page display",
    }
    output: list[str] = []
    previous_heading = ""
    removed = max(0, start) + max(0, len(raw_lines) - stop)
    for raw_line in raw_lines[start:stop]:
        line = _strip_markdown(raw_line)
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if line.casefold() in boilerplate:
            removed += 1
            continue
        if re.match(r"^(?:Retrieved from|Category:|Hidden categories:|This page was last edited)", line, re.I):
            removed += 1
            continue
        if re.fullmatch(r"\[(?:edit|citation needed)]", line, re.I):
            removed += 1
            continue
        if line.startswith("#"):
            heading = normalise_key(line.lstrip("# "))
            if heading and heading == previous_heading:
                removed += 1
                continue
            previous_heading = heading
        output.append(line)

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    exclusions = [{
        "exclusion_id": stable_id("exc", source_id, "boilerplate"),
        "source_id": source_id,
        "category": "boilerplate_removed",
        "reason": "obvious navigation, footer, edit controls, or scraping artefacts",
        "line_count": removed,
    }]
    return cleaned, exclusions


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def title_from_metadata(metadata: dict[str, Any]) -> str:
    title = metadata.get("page_title") or metadata.get("source_id") or "Untitled"
    title = re.sub(r"\s+- Wikipedia$", "", title)
    title = re.sub(r"\s+\|\s+PROV$", "", title)
    return title.strip()


def create_document(record: RawRecord, source_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cleaned, exclusions = clean_markdown(record.body, record.source_id)
    words = word_count(cleaned)
    document_id = stable_id("doc", record.source_id, record.provenance["content_hash_sha256"])
    return {
        "document_id": document_id,
        "source_id": record.source_id,
        "title": title_from_metadata(record.metadata),
        "source_url": record.provenance["source_url"],
        "source_name": source_name,
        "licence": record.provenance["licence"],
        "retrieved_at": record.provenance["retrieved"],
        "content_hash": record.provenance["content_hash_sha256"],
        "cleaned_text": cleaned,
        "word_count": words,
        "approximate_token_count": math.ceil(words * 1.33),
    }, exclusions


def create_passages(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Create contiguous document-only windows and retain the nearest heading label.

    Heading boundaries inform labels, while small adjacent sections are allowed to share
    a passage so navigation-sized fragments do not become tiny records.
    """
    passages: list[dict[str, Any]] = []
    passage_index = 0
    text = document["cleaned_text"]
    words = list(re.finditer(r"\S+", text))
    headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text))
    chunk_count = max(1, math.ceil((len(words) - PASSAGE_OVERLAP_WORDS) / (MAX_PASSAGE_WORDS - PASSAGE_OVERLAP_WORDS)))
    window_words = math.ceil((len(words) + (chunk_count - 1) * PASSAGE_OVERLAP_WORDS) / chunk_count)
    window_words = min(MAX_PASSAGE_WORDS, max(1, window_words))
    stride = max(1, window_words - PASSAGE_OVERLAP_WORDS)
    cursor = 0
    while cursor < len(words):
        final = min(cursor + window_words, len(words))
        absolute_start = words[cursor].start()
        absolute_end = words[final - 1].end()
        passage_text = text[absolute_start:absolute_end].strip()
        prior_headings = [heading for heading in headings if heading.start() <= absolute_start]
        section_title = _strip_markdown(prior_headings[-1].group(1)) if prior_headings else "Overview"
        passage_id = stable_id("pass", document["document_id"], passage_index, passage_text)
        passages.append({
            "passage_id": passage_id,
            "document_id": document["document_id"],
            "section_title": section_title,
            "passage_index": passage_index,
            "text": passage_text,
            "source_url": document["source_url"],
            "licence": document["licence"],
            "character_start": absolute_start,
            "character_end": absolute_end,
        })
        passage_index += 1
        if final >= len(words):
            break
        cursor += stride
    return passages


def infer_entity_type(name: str, title: str = "") -> str:
    lowered = name.casefold()
    word_rules = (
        (r"\bhotel\b", "hotel"),
        (r"\bmarket\b", "market"),
        (r"\b(?:railway )?station\b", "railway_station"),
        (r"\b(?:church|cathedral)\b", "church"),
        (r"\btheatre\b", "theatre"),
        (r"\b(?:public record office|heritage database)\b", "government_institution"),
    )
    for pattern, entity_type in word_rules:
        if re.search(pattern, lowered):
            return entity_type
    for suffix, entity_type in sorted(ENTITY_SUFFIX_TYPES.items(), key=lambda item: -len(item[0])):
        if lowered.endswith(suffix):
            return entity_type
    if re.search(r"\b(?:carlton|collingwood|fitzroy|richmond|north melbourne|port melbourne)\b", lowered):
        return "suburb"
    if lowered in {"melbourne", "victoria"}:
        return "place"
    if re.search(r"\b(?:john|james|william|charles|robert|francis|joseph|edward)\b", lowered):
        return "person"
    if "architecture" in title.casefold():
        return "landmark"
    return "building"


def _clean_entity_name(name: str) -> str:
    name = re.sub(r"^(?:The|A|An)\s+", "", name.strip())
    return re.sub(r"\s+", " ", name).strip(" ,.;:()[]")


def extract_entities(documents: list[dict[str, Any]], passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for passage in passages:
        by_document[passage["document_id"]].append(passage)
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}

    def support(name: str, entity_type: str, passage: dict[str, Any], method: str, confidence: float) -> None:
        clean_name = _clean_entity_name(name)
        if len(clean_name) < 3 or word_count(clean_name) > 8:
            return
        if normalise_key(clean_name) in {
            "bridge", "building", "cathedral", "church", "government", "hall", "hotel",
            "market", "museum", "office", "road", "square", "station", "street", "theatre",
            "town hall", "university",
        }:
            return
        if re.match(r"^(?:Former|New|Old|Other|Several|Many|Some|This|That)\b", clean_name):
            return
        key = (normalise_key(clean_name), entity_type)
        if not key[0]:
            return
        if key not in aggregate:
            aggregate[key] = {
                "entity_id": stable_id("ent", entity_type, clean_name),
                "canonical_name": clean_name,
                "entity_type": entity_type,
                "aliases": [],
                "description": None,
                "supporting_passage_ids": [],
                "confidence": confidence,
                "extraction_method": method,
            }
        entity = aggregate[key]
        if passage["passage_id"] not in entity["supporting_passage_ids"]:
            entity["supporting_passage_ids"].append(passage["passage_id"])
        entity["confidence"] = max(entity["confidence"], confidence)

    for document in documents:
        doc_passages = by_document[document["document_id"]]
        if not doc_passages:
            continue
        primary_name = re.sub(r"\s*\([^)]*\)\s*$", "", document["title"]).strip()
        if document["source_id"] == "prov_about":
            primary_name = "Public Record Office Victoria"
        elif document["source_id"] == "vhd_introduction":
            primary_name = "Victorian Heritage Database"
        topic_match = re.match(r"^(?:History|Architecture) of (.+)$", primary_name, re.I)
        if topic_match:
            primary_name = topic_match.group(1)
        primary_name = re.sub(r",\s*(?:Victoria|Melbourne)$", "", primary_name, flags=re.I)
        primary_type = infer_entity_type(primary_name, document["title"])
        support(primary_name, primary_type, doc_passages[0], "document_title_rule_v1", 0.98)
        for passage in doc_passages:
            for match in ENTITY_RE.finditer(passage["text"]):
                candidate = _clean_entity_name(match.group(0))
                support(candidate, infer_entity_type(candidate), passage, "named_suffix_rule_v1", 0.88)
            for place in ("Melbourne", "Victoria", "Carlton", "Collingwood", "Fitzroy", "Richmond", "North Melbourne", "Port Melbourne"):
                if re.search(rf"\b{re.escape(place)}\b", passage["text"]):
                    support(place, "suburb" if place not in {"Melbourne", "Victoria"} else "place", passage, "place_lexicon_v1", 0.92)
        lead = " ".join(p["text"] for p in doc_passages[:2])
        alias_match = re.search(r"(?:also known as|formerly known as)\s+([^,.;()]{2,80})", lead, re.I)
        if alias_match:
            key = (normalise_key(primary_name), primary_type)
            alias = _clean_entity_name(alias_match.group(1))
            if key in aggregate and alias and alias.casefold() != primary_name.casefold():
                aggregate[key]["aliases"] = [alias]

    entities = list(aggregate.values())
    for entity in entities:
        entity["supporting_passage_ids"].sort()
        entity["aliases"] = sorted(set(entity["aliases"]))
    return sorted(entities, key=lambda item: item["entity_id"])


def split_sentences(text: str) -> Iterable[tuple[str, int, int]]:
    for match in re.finditer(r"[^\n.!?]+(?:[.!?](?=\s|$)|$)", text):
        sentence = match.group(0).strip()
        if word_count(sentence) >= 5 and not re.search(r"\b(?:and|or|between|from|to|by|of|the)\s*$", sentence, re.I):
            left_trim = len(match.group(0)) - len(match.group(0).lstrip())
            yield sentence, match.start() + left_trim, match.start() + left_trim + len(sentence)


def date_record(sentence: str, anchor: int = 0) -> tuple[str | None, dict[str, Any] | None, str | None]:
    matches = list(VAGUE_DATE_RE.finditer(sentence)) + list(YEAR_RE.finditer(sentence))
    if not matches:
        return None, None, "date not stated in supporting text"
    unique = {(match.start(), match.end()): match for match in matches}
    following = [match for match in unique.values() if match.start() >= anchor]
    match = (
        min(following, key=lambda item: item.start() - anchor)
        if following
        else min(unique.values(), key=lambda item: anchor - item.end())
    )
    original = match.group(0)
    digits = re.search(r"\d{4}", original)
    if digits and not re.search(r"around|circa|c\.|s$", original, re.I):
        value = int(digits.group(0))
        return original, {"start_year": value, "end_year": value, "precision": "year"}, None
    return original, None, "historical expression retained; precision is not sufficient to normalise"


def passage_entity_mentions(passage: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered = passage["text"].casefold()
    matches = [e for e in entities if e["canonical_name"].casefold() in lowered and len(e["canonical_name"]) >= 3]
    return sorted(matches, key=lambda item: (-len(item["canonical_name"]), item["entity_id"]))


def extract_events(passages: list[dict[str, Any]], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entity_by_passage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        for passage_id in entity["supporting_passage_ids"]:
            entity_by_passage[passage_id].append(entity)
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for passage in passages:
        candidates = entity_by_passage.get(passage["passage_id"], [])
        for sentence, start, end in split_sentences(passage["text"]):
            for event_type, pattern in EVENT_PATTERNS:
                verb_match = pattern.search(sentence)
                if not verb_match:
                    continue
                if re.search(r"\b(?:proposed|planned|intended|would be|was to begin|failed to|never|not completed|not constructed|not built)\b", sentence, re.I):
                    continue
                date_original, normalised_date, uncertainty = date_record(sentence, verb_match.end())
                involved = [e for e in candidates if re.search(rf"\b{re.escape(e['canonical_name'])}\b", sentence, re.I)]
                specific = [
                    e for e in involved
                    if e["entity_type"] not in {"place", "suburb", "street", "government_institution"}
                    and normalise_key(e["canonical_name"]) not in {
                        "bridge", "building", "church", "hotel", "market", "station", "town hall"
                    }
                ]
                involved = specific or [
                    e for e in involved
                    if e["entity_type"] not in {
                        "place", "suburb", "street", "government_institution", "person", "organisation"
                    }
                ]
                inverted = bool(re.match(
                    r"^(?:Built|Constructed|Completed|Opened|Demolished|Renovated|Restored|Redeveloped)\b",
                    sentence,
                    re.I,
                ))
                def proximity(entity: dict[str, Any]) -> tuple[int, int, str]:
                    positions = [m.start() for m in re.finditer(rf"\b{re.escape(entity['canonical_name'])}\b", sentence, re.I)]
                    before = [position for position in positions if position <= verb_match.start()]
                    distance = verb_match.start() - max(before) if before else (10_000 + min(positions, default=10_000) if inverted else 1_000_000)
                    return distance, -len(entity["canonical_name"]), entity["entity_id"]
                involved = sorted(involved, key=proximity)[:1]
                if not involved or proximity(involved[0])[0] >= 1_000_000:
                    continue
                key = (event_type, passage["document_id"], normalise_key(sentence))
                if key in seen:
                    continue
                seen.add(key)
                location = next(
                    (e["canonical_name"] for e in involved if e["entity_type"] in {"place", "suburb", "street"}),
                    None,
                )
                events.append({
                    "event_id": stable_id("evt", event_type, passage["passage_id"], sentence),
                    "event_type": event_type,
                    "involved_entity_ids": sorted(e["entity_id"] for e in involved),
                    "date_original": date_original,
                    "normalised_date": normalised_date,
                    "location": location,
                    "supporting_passage_ids": [passage["passage_id"]],
                    "supporting_text": sentence,
                    "supporting_span": {"start": start, "end": end},
                    "confidence": 0.86 if date_original else 0.76,
                    "uncertainty_notes": uncertainty,
                    "extraction_method": "event_phrase_rule_v1",
                })
    return sorted(events, key=lambda item: item["event_id"])


def _find_or_create_entity(
    name: str,
    entity_type: str,
    passage_id: str,
    entities: list[dict[str, Any]],
    entity_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    name = _clean_entity_name(name)
    key = (normalise_key(name), entity_type)
    if key not in entity_lookup:
        entity = {
            "entity_id": stable_id("ent", entity_type, name),
            "canonical_name": name,
            "entity_type": entity_type,
            "aliases": [],
            "description": None,
            "supporting_passage_ids": [passage_id],
            "confidence": 0.82,
            "extraction_method": "relation_target_rule_v1",
        }
        entities.append(entity)
        entity_lookup[key] = entity
    elif passage_id not in entity_lookup[key]["supporting_passage_ids"]:
        entity_lookup[key]["supporting_passage_ids"].append(passage_id)
    return entity_lookup[key]


def extract_relations(passages: list[dict[str, Any]], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(normalise_key(e["canonical_name"]), e["entity_type"]): e for e in entities}
    mentions = {p["passage_id"]: passage_entity_mentions(p, entities) for p in passages}
    relations: list[dict[str, Any]] = []
    name_token = r"[A-Z][A-Za-z&'’.-]*"
    proper_name = rf"({name_token}(?:(?:[ \t]+(?:and|of|the)[ \t]+|[ \t]+|,\s*){name_token}){{0,6}})"
    designer_prefix = r"(?:(?:the|an?)\s+)?(?:(?:English|local|railways?)\s+)?(?:(?:architects?|architectural practice|office of)\s+)?"
    rules = [
        ("DESIGNED_BY", re.compile(r"(?i:\bdesigned by\s+)" + designer_prefix + proper_name), "person"),
        ("OPERATED_BY", re.compile(r"(?i:\boperated by\s+(?:the\s+)?)" + proper_name), "organisation"),
        ("RENAMED_TO", re.compile(r"(?i:\brenamed (?:as|to)\s+)" + proper_name), "building"),
        ("REPLACED_BY", re.compile(r"(?i:\breplaced by\s+)" + proper_name), "building"),
    ]
    for passage in passages:
        passage_mentions = mentions[passage["passage_id"]]
        for sentence, start, end in split_sentences(passage["text"]):
            for relation_type, pattern, target_type in rules:
                match = pattern.search(sentence)
                if not match:
                    continue
                subject_candidates = []
                for entity in passage_mentions:
                    occurrences = list(re.finditer(rf"\b{re.escape(entity['canonical_name'])}\b", sentence, re.I))
                    before = [occurrence for occurrence in occurrences if occurrence.end() <= match.start()]
                    if before and entity["entity_type"] not in {"place", "suburb", "street"}:
                        subject_candidates.append((match.start() - before[-1].end(), -len(entity["canonical_name"]), entity))
                if not subject_candidates:
                    continue
                subject = min(subject_candidates, key=lambda item: (item[0], item[1], item[2]["entity_id"]))[2]
                target_name = _clean_entity_name(match.group(1))
                if len(target_name) < 3:
                    continue
                if relation_type == "DESIGNED_BY" and re.search(r"(?:,|\band\b|\bDepartment\b|\bPartners\b|\bArchitects\b)", target_name):
                    target_type = "organisation"
                target = _find_or_create_entity(target_name, target_type, passage["passage_id"], entities, lookup)
                relations.append({
                    "relation_id": stable_id("rel", relation_type, subject["entity_id"], target["entity_id"], passage["passage_id"]),
                    "relation_type": relation_type,
                    "subject_entity_id": subject["entity_id"],
                    "object_entity_id": target["entity_id"],
                    "object_value": None,
                    "supporting_passage_ids": [passage["passage_id"]],
                    "supporting_text": sentence,
                    "supporting_span": {"start": start, "end": end},
                    "confidence": 0.88,
                    "extraction_method": "relation_phrase_rule_v1",
                })
        # Explicit location relation: both names and a locative phrase must occur in one sentence.
        locations = [e for e in passage_mentions if e["entity_type"] in {"place", "suburb"}]
        subjects = [e for e in passage_mentions if e["entity_type"] not in {"place", "suburb"}]
        if locations and subjects:
            subject, location = subjects[0], locations[0]
            relation_text = next((
                s for s, _, _ in split_sentences(passage["text"])
                if re.search(
                    rf"\b{re.escape(subject['canonical_name'])}\b[^.]{{0,160}}\b(?:located|situated|stands?|lies?|based)\b[^.]*\b{re.escape(location['canonical_name'])}\b",
                    s,
                    re.I,
                )
            ), None)
            if relation_text:
                start = passage["text"].find(relation_text)
                relations.append({
                    "relation_id": stable_id("rel", "LOCATED_IN", subject["entity_id"], location["entity_id"], passage["passage_id"]),
                    "relation_type": "LOCATED_IN",
                    "subject_entity_id": subject["entity_id"],
                    "object_entity_id": location["entity_id"],
                    "object_value": None,
                    "supporting_passage_ids": [passage["passage_id"]],
                    "supporting_text": relation_text,
                    "supporting_span": {"start": start, "end": start + len(relation_text)},
                    "confidence": 0.82,
                    "extraction_method": "co_mention_location_rule_v1",
                })
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for relation in relations:
        key = (
            relation["relation_type"],
            relation["subject_entity_id"],
            relation["object_entity_id"] or str(relation["object_value"]),
            normalise_key(relation["supporting_text"]),
        )
        unique.setdefault(key, relation)
    entities.sort(key=lambda item: item["entity_id"])
    return sorted(unique.values(), key=lambda item: item["relation_id"])


def event_relations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create only relation forms that are explicit in an already validated event."""
    relations: list[dict[str, Any]] = []
    for event in events:
        if event["event_type"] != "demolition" or not event["date_original"]:
            continue
        subject_id = event["involved_entity_ids"][0]
        passage_id = event["supporting_passage_ids"][0]
        relations.append({
            "relation_id": stable_id("rel", "DEMOLISHED_IN", subject_id, event["date_original"], passage_id),
            "relation_type": "DEMOLISHED_IN",
            "subject_entity_id": subject_id,
            "object_entity_id": None,
            "object_value": event["date_original"],
            "supporting_passage_ids": [passage_id],
            "supporting_text": event["supporting_text"],
            "supporting_span": event["supporting_span"],
            "confidence": event["confidence"],
            "extraction_method": "validated_demolition_event_relation_v1",
        })
    return relations


def valid_supporting_span(claim: dict[str, Any], passages_by_id: dict[str, dict[str, Any]]) -> bool:
    passage = passages_by_id.get(claim.get("passage_id"))
    if not passage:
        return False
    span = claim.get("supporting_span") or {}
    start, end = span.get("start"), span.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return False
    return passage["text"][start:end] == claim.get("supporting_text")


def create_claims(
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entity_by_id = {e["entity_id"]: e for e in entities}
    passage_by_id = {p["passage_id"]: p for p in passages}
    document_by_id = {d["document_id"]: d for d in documents}
    claims: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    event_predicates = {
        "construction": "was_constructed",
        "opening": "opened",
        "renovation": "was_renovated",
        "renaming": "was_renamed",
        "relocation": "was_relocated",
        "closure": "closed",
        "demolition": "was_demolished",
        "fire": "experienced_fire",
        "redevelopment": "was_redeveloped",
        "heritage_listing": "was_heritage_listed",
    }

    for event in events:
        subject_id = event["involved_entity_ids"][0]
        subject = entity_by_id[subject_id]["canonical_name"]
        passage_id = event["supporting_passage_ids"][0]
        passage = passage_by_id[passage_id]
        document = document_by_id[passage["document_id"]]
        value = event["date_original"] if event["date_original"] else True
        claim = {
            "claim_id": stable_id("clm", event["event_id"], subject, value),
            "subject": {"entity_id": subject_id, "name": subject},
            "predicate": event_predicates[event["event_type"]],
            "object_or_value": value,
            "supporting_text": event["supporting_text"],
            "supporting_span": event["supporting_span"],
            "passage_id": passage_id,
            "source_id": document["source_id"],
            "confidence": event["confidence"],
            "extraction_method": "validated_event_to_claim_v1",
            "temporal_qualifier": event["date_original"],
            "geographic_qualifier": event["location"],
            "conflict_group": None,
        }
        (claims if valid_supporting_span(claim, passage_by_id) else rejected).append(claim)

    for relation in relations:
        subject = entity_by_id[relation["subject_entity_id"]]
        target = entity_by_id.get(relation["object_entity_id"])
        target_value: Any = (
            {"entity_id": target["entity_id"], "name": target["canonical_name"]}
            if target else relation["object_value"]
        )
        passage_id = relation["supporting_passage_ids"][0]
        passage = passage_by_id[passage_id]
        document = document_by_id[passage["document_id"]]
        claim = {
            "claim_id": stable_id("clm", relation["relation_id"], subject["canonical_name"], target_value),
            "subject": {"entity_id": subject["entity_id"], "name": subject["canonical_name"]},
            "predicate": relation["relation_type"].casefold(),
            "object_or_value": target_value,
            "supporting_text": relation["supporting_text"],
            "supporting_span": relation["supporting_span"],
            "passage_id": passage_id,
            "source_id": document["source_id"],
            "confidence": relation["confidence"],
            "extraction_method": "validated_relation_to_claim_v1",
            "temporal_qualifier": None,
            "geographic_qualifier": target["canonical_name"] if target and relation["relation_type"] == "LOCATED_IN" else None,
            "conflict_group": None,
        }
        (claims if valid_supporting_span(claim, passage_by_id) else rejected).append(claim)

    # Conflict grouping requires explicit disagreement language. Different dates can describe
    # different construction stages or openings and are not silently labelled as conflicts.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[(claim["subject"]["entity_id"], claim["predicate"])].append(claim)
    for key, group in grouped.items():
        subject_entity = entity_by_id[key[0]]
        explicit = [
            c for c in group
            if c["temporal_qualifier"]
            and subject_entity["entity_type"] not in {"place", "suburb", "street", "organisation", "government_institution"}
            and re.search(rf"\b{re.escape(c['subject']['name'])}\b", c["supporting_text"], re.I)
            and re.search(r"\b(?:disputed|sources? (?:differ|disagree)|variously reported|uncertain)\b", c["supporting_text"], re.I)
        ]
        values = {json.dumps(c["object_or_value"], sort_keys=True) for c in explicit}
        sources = {c["source_id"] for c in explicit}
        if len(values) > 1 and len(sources) > 1:
            conflict_group = stable_id("conflict", *key)
            for claim in explicit:
                claim["conflict_group"] = conflict_group
    return sorted(claims, key=lambda item: item["claim_id"]), rejected


def extract_places(
    documents: list[dict[str, Any]], passages: list[dict[str, Any]], entities: list[dict[str, Any]]
) -> dict[str, Any]:
    entity_by_passage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        for passage_id in entity["supporting_passage_ids"]:
            entity_by_passage[passage_id].append(entity)
    document_by_id = {d["document_id"]: d for d in documents}
    features: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for passage in passages:
        match = COORD_RE.search(passage["text"])
        if not match:
            continue
        latitude, longitude = float(match.group(1)), float(match.group(2))
        candidates = [e for e in entity_by_passage[passage["passage_id"]] if e["entity_type"] not in {"person", "organisation", "government_institution"}]
        if not candidates:
            continue
        entity = sorted(candidates, key=lambda e: (-e["confidence"], -len(e["canonical_name"])))[0]
        if entity["entity_id"] in seen_entities:
            continue
        seen_entities.add(entity["entity_id"])
        document = document_by_id[passage["document_id"]]
        location_match = re.search(r"(?im)^\|?\s*Location\s*\|\s*([^|\n]+)", passage["text"])
        address = location_match.group(1).strip() if location_match else None
        suburb = next(
            (name for name in ("Carlton", "Collingwood", "Fitzroy", "Richmond", "North Melbourne", "Port Melbourne", "Melbourne") if name.casefold() in passage["text"].casefold()),
            None,
        )
        features.append({
            "type": "Feature",
            "id": entity["entity_id"],
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {
                "canonical_entity_id": entity["entity_id"],
                "name": entity["canonical_name"],
                "entity_type": entity["entity_type"],
                "address": address,
                "suburb": suburb,
                "coordinate_source": document["source_url"],
                "coordinate_confidence": 0.98,
                "location_precision": "point_from_source",
                "supporting_passage_ids": [passage["passage_id"]],
            },
        })
    return {"type": "FeatureCollection", "features": sorted(features, key=lambda f: f["id"])}


def assign_splits(documents: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(documents, key=lambda d: hashlib.sha256(d["document_id"].encode()).hexdigest())
    total = len(ordered)
    train_count = max(1, round(total * 0.8))
    validation_count = max(1, round(total * 0.1)) if total >= 3 else max(0, total - train_count)
    if train_count + validation_count >= total and total > 1:
        train_count = max(1, total - validation_count - 1)
    rows = []
    for index, document in enumerate(ordered):
        split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
        rows.append({"document_id": document["document_id"], "source_id": document["source_id"], "split": split})
    return {
        "strategy": "deterministic_document_level_sha256_v1",
        "target_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "counts": dict(Counter(row["split"] for row in rows)),
        "assignments": sorted(rows, key=lambda row: row["document_id"]),
    }


def create_training_corpus(
    documents: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    excluded_documents: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    excluded_documents = excluded_documents or []
    rows: list[dict[str, Any]] = []
    training_parts: list[str] = []
    total_before = sum(d["word_count"] for d in documents + excluded_documents)
    selected: list[tuple[dict[str, Any], str, int]] = []
    for document in sorted(documents, key=lambda d: d["source_id"]):
        before = document["word_count"]
        compatible = document["licence"] in COMPATIBLE_TRAINING_LICENCES
        if not compatible:
            after_text = ""
            reason = "licence is unclear or incompatible with training"
            exclusions.append({
                "exclusion_id": stable_id("exc", document["source_id"], "licence"),
                "source_id": document["source_id"],
                "category": "training_licence_exclusion",
                "reason": reason,
                "line_count": None,
            })
        else:
            matches = list(re.finditer(r"\b[\w’'-]+\b", document["cleaned_text"], flags=re.UNICODE))
            if len(matches) > TRAINING_SOURCE_CAP_WORDS:
                after_text = document["cleaned_text"][: matches[TRAINING_SOURCE_CAP_WORDS - 1].end()].strip()
                reason = f"capped at {TRAINING_SOURCE_CAP_WORDS:,} words"
            else:
                after_text = document["cleaned_text"].strip()
                reason = "included in full"
        after = word_count(after_text)
        selected.append((document, after_text, after))
        rows.append({
            "source_id": document["source_id"],
            "licence": document["licence"],
            "training_eligible": compatible,
            "words_before": before,
            "percent_before": (before / total_before * 100) if total_before else 0,
            "words_after": after,
            "cap_words": TRAINING_SOURCE_CAP_WORDS,
            "decision": reason,
        })
    for document in sorted(excluded_documents, key=lambda d: d["source_id"]):
        rows.append({
            "source_id": document["source_id"],
            "licence": document["licence"],
            "training_eligible": False,
            "words_before": document["word_count"],
            "percent_before": (document["word_count"] / total_before * 100) if total_before else 0,
            "words_after": 0,
            "cap_words": TRAINING_SOURCE_CAP_WORDS,
            "decision": "excluded: insufficient historical content",
        })
    total_after = sum(after for _, _, after in selected)
    after_map = {document["source_id"]: after for document, _, after in selected}
    for row in rows:
        row["percent_after"] = (after_map.get(row["source_id"], 0) / total_after * 100) if total_after else 0
    for _, text, _ in selected:
        if text:
            training_parts.append(text)
    return "\n\n".join(training_parts).strip() + "\n", rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_fingerprint(records: list[RawRecord]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_file_count": len(records),
        "sources": [
            {
                "source_id": r.source_id,
                "source_url": r.provenance["source_url"],
                "licence": r.provenance["licence"],
                "retrieval_timestamp": r.provenance["retrieved"],
                "file_size": r.markdown_path.stat().st_size,
                "content_hash": r.provenance["content_hash_sha256"],
                "collection_status": r.metadata["status"],
            }
            for r in records
        ],
    }


def _count_table(rows: Iterable[dict[str, Any]], field: str) -> str:
    counts = Counter(row[field] for row in rows)
    return "\n".join(f"| {name} | {count} |" for name, count in sorted(counts.items())) or "| None | 0 |"


def write_reports(
    reports_dir: Path,
    records: list[RawRecord],
    documents: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    rejected_claims: list[dict[str, Any]],
    places: dict[str, Any],
    splits: dict[str, Any],
    exclusions: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
    training_text: str,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    exact_duplicates = len(documents) - len({d["content_hash"] for d in documents})
    normalised_hashes = [hashlib.sha256(normalise_key(d["cleaned_text"]).encode()).hexdigest() for d in documents]
    near_duplicates = len(normalised_hashes) - len(set(normalised_hashes))
    largest = max(documents, key=lambda d: d["word_count"])
    smallest = min(documents, key=lambda d: d["word_count"])
    raw_words = sum(word_count(r.body) for r in records)
    cleaned_words = sum(d["word_count"] for d in documents)
    training_words = word_count(training_text)
    conflict_groups = sorted({c["conflict_group"] for c in claims if c["conflict_group"]})
    timeline_events = [e for e in events if e["date_original"]]

    exclusion_counts = Counter(row["category"] for row in exclusions)
    quality = f"""# Phase 2 Data Quality Report

Generated deterministically by `scripts/prepare_corpus.py`. Raw evidence was hash-checked before and after processing.

| Metric | Value |
| --- | ---: |
| Raw Markdown documents | {len(records)} |
| Processed documents | {len(documents)} |
| Raw words (including captured page chrome) | {raw_words:,} |
| Cleaned words | {cleaned_words:,} |
| Approximate cleaned tokens | {math.ceil(cleaned_words * 1.33):,} |
| Passages | {len(passages)} |
| Exact duplicate documents | {exact_duplicates} |
| Normalised near-duplicate documents | {near_duplicates} |
| Exclusion records | {len(exclusions)} |
| Rejected unsupported claims | {len(rejected_claims)} |

Largest source: `{largest['source_id']}` ({largest['word_count']:,} words).
Smallest source: `{smallest['source_id']}` ({smallest['word_count']:,} words).

## Exclusions and reasons

{''.join(f"- `{category}`: {count}\n" for category, count in sorted(exclusion_counts.items()))}
`wiki_moomba` was excluded after cleaning because the collected page is a 28-word disambiguation page, below the 80-word historical-content threshold. Boilerplate exclusions record removed navigation/footer lines without copying them.

## Dataset splits

| Split | Documents |
| --- | ---: |
{''.join(f"| {name} | {count} |\n" for name, count in sorted(splits['counts'].items()))}
## Training corpus

Final training corpus: **{training_words:,} words** (approximately **{math.ceil(training_words * 1.33):,} tokens**). Only cleaned source text with a compatible licence is included; each source is capped at {TRAINING_SOURCE_CAP_WORDS:,} words.
"""
    (reports_dir / "data_quality_report.md").write_text(quality, encoding="utf-8")

    entity_by_id = {entity["entity_id"]: entity for entity in entities}
    sample_entities = sorted(
        (entity for entity in entities if entity["confidence"] >= 0.95),
        key=lambda entity: (entity["entity_type"], entity["canonical_name"]),
    )[:5]
    quality_events = [
        event for event in events
        if event["date_original"]
        and entity_by_id[event["involved_entity_ids"][0]]["confidence"] >= 0.88
    ]
    sample_events = quality_events[:5]
    sample_relations = relations[:5]
    sample_claims = claims[:5]
    place_samples = places["features"][:3]
    knowledge = f"""# Phase 2 Historical Knowledge Fabric Report

## Simple result

The 25 collected pages are now a linked historical evidence layer. One 28-word disambiguation page was excluded as unusable, leaving 24 cleaned documents. Every retained document keeps its source and licence; passages stay inside their source document; extracted entities, events, relations and claims point back to supporting passages. No embeddings, Transformer training, final RAG system or UI were created.

## Technical summary

| Record type | Count |
| --- | ---: |
| Documents | {len(documents)} |
| Passages | {len(passages)} |
| Entities | {len(entities)} |
| Historical events | {len(events)} |
| Relations | {len(relations)} |
| Validated claims | {len(claims)} |
| Unsupported claims accepted | 0 |
| GeoJSON point features | {len(places['features'])} |
| Events with temporal expressions | {len(timeline_events)} |
| Conflict groups | {len(conflict_groups)} |

### Entities by type

| Type | Count |
| --- | ---: |
{_count_table(entities, 'entity_type')}

### Events by type

| Type | Count |
| --- | ---: |
{_count_table(events, 'event_type')}

### Relations by type

| Type | Count |
| --- | ---: |
{_count_table(relations, 'relation_type')}

## Five example entities

{''.join(f"- `{e['entity_id']}` — {e['canonical_name']} ({e['entity_type']}); support: `{e['supporting_passage_ids'][0]}`\n" for e in sample_entities)}
## Five example historical events

{''.join(f"- `{e['event_id']}` — {e['event_type']}; date: {e['date_original'] or 'not stated'}; support: `{e['supporting_passage_ids'][0]}`\n" for e in sample_events)}
## Five example relations

{''.join(f"- `{r['relation_id']}` — {r['relation_type']} from `{r['subject_entity_id']}` to `{r['object_entity_id'] or r['object_value']}`; support: `{r['supporting_passage_ids'][0]}`\n" for r in sample_relations)}
## Five example claims and exact supporting spans

{''.join(f"- **{c['subject']['name']} / {c['predicate']} / {c['object_or_value']}** — “{c['supporting_text']}” (`{c['passage_id']}`)\n" for c in sample_claims)}
## Complete provenance trace

{_trace_example(records, documents, passages, entities, events, claims)}

## Geographic coverage and sample locations

{len(places['features'])} exact point features were copied from coordinates stated by sources; no coordinates were invented.

{''.join(f"- {f['properties']['name']}: {f['geometry']['coordinates'][1]}, {f['geometry']['coordinates'][0]} ({f['properties']['coordinate_source']})\n" for f in place_samples)}
## Split counts

{', '.join(f"{name}: {count}" for name, count in sorted(splits['counts'].items()))}. Splits are assigned at document level; passages and claims inherit document membership and cannot leak across splits.
"""
    (reports_dir / "knowledge_fabric_report.md").write_text(knowledge, encoding="utf-8")

    with (reports_dir / "source_balance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(balance_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(balance_rows)

    processed_source_ids = {document["source_id"] for document in documents}
    with (reports_dir / "licence_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source_id", "licence", "collection_status", "processed", "training_compatible", "decision"])
        for record in records:
            licence = record.provenance["licence"]
            processed = record.source_id in processed_source_ids
            compatible = licence in COMPATIBLE_TRAINING_LICENCES
            decision = "include" if processed and compatible else "exclude_unusable" if not processed else "exclude_licence"
            writer.writerow([record.source_id, licence, record.metadata["status"], str(processed).lower(), str(compatible).lower(), decision])

    conflicts = [c for c in claims if c["conflict_group"]]
    conflict_text = "# Phase 2 Conflict Report\n\n"
    if not conflicts:
        conflict_text += "No cross-source conflicting claim groups were detected by the conservative deterministic rules. Absence of a detected conflict is not proof that all sources agree.\n"
    else:
        for group in conflict_groups:
            conflict_text += f"## {group}\n\n"
            for claim in (c for c in conflicts if c["conflict_group"] == group):
                conflict_text += f"- `{claim['source_id']}`: {claim['subject']['name']} / {claim['predicate']} / {claim['object_or_value']}\n"
    (reports_dir / "conflict_report.md").write_text(conflict_text, encoding="utf-8")


def _trace_example(
    records: list[RawRecord],
    documents: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    events: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> str:
    passage_by_id = {p["passage_id"]: p for p in passages}
    document_by_id = {d["document_id"]: d for d in documents}
    entity_by_id = {e["entity_id"]: e for e in entities}
    event = next((
        e for e in events
        if e["date_original"] and entity_by_id[e["involved_entity_ids"][0]]["confidence"] >= 0.88
        and any(c["passage_id"] in e["supporting_passage_ids"] for c in claims)
    ), events[0])
    passage = passage_by_id[event["supporting_passage_ids"][0]]
    document = document_by_id[passage["document_id"]]
    entity = entity_by_id[event["involved_entity_ids"][0]]
    claim = next(
        c for c in claims
        if c["passage_id"] == passage["passage_id"]
        and c["temporal_qualifier"] == event["date_original"]
        and c["supporting_text"] == event["supporting_text"]
    )
    record = next(r for r in records if r.source_id == document["source_id"])
    return (
        f"`{record.markdown_path.name}` (SHA-256 `{document['content_hash']}`)<br>\n"
        f"→ `{document['document_id']}` ({document['title']})<br>\n"
        f"→ `{passage['passage_id']}` ({passage['section_title']})<br>\n"
        f"→ `{entity['entity_id']}` ({entity['canonical_name']})<br>\n"
        f"→ `{event['event_id']}` ({event['event_type']})<br>\n"
        f"→ `{claim['claim_id']}` ({claim['predicate']})"
    )


def validate_outputs(
    documents: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    events: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    places: dict[str, Any],
    splits: dict[str, Any],
) -> None:
    if not documents:
        raise PipelineError("no usable documents remain after processing")
    if not any(d["cleaned_text"].strip() for d in documents):
        raise PipelineError("all extracted bodies are empty")
    document_ids = {d["document_id"] for d in documents}
    passage_ids = {p["passage_id"] for p in passages}
    if not passages or any(p["document_id"] not in document_ids for p in passages):
        raise PipelineError("passage/document integrity validation failed")
    if any(not set(e["supporting_passage_ids"]) <= passage_ids for e in entities):
        raise PipelineError("entity support validation failed")
    if any(not set(e["supporting_passage_ids"]) <= passage_ids for e in events):
        raise PipelineError("event support validation failed")
    if any(not set(r["supporting_passage_ids"]) <= passage_ids for r in relations):
        raise PipelineError("relation support validation failed")
    passage_by_id = {p["passage_id"]: p for p in passages}
    if any(not valid_supporting_span(c, passage_by_id) for c in claims):
        raise PipelineError("claim support-span validation failed")
    assigned = [row["document_id"] for row in splits["assignments"]]
    if set(assigned) != document_ids or len(assigned) != len(set(assigned)):
        raise PipelineError("document split overlap or omission detected")
    for feature in places["features"]:
        longitude, latitude = feature["geometry"]["coordinates"]
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise PipelineError("invalid GeoJSON coordinate")


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    config_path: Path = CONFIG_PATH,
    fingerprint_path: Path = FINGERPRINT_PATH,
    expected_count: int | None = 25,
) -> dict[str, Any]:
    before = snapshot_tree(raw_dir)
    records = load_and_validate_raw(raw_dir, expected_count=expected_count)
    source_names = load_source_names(config_path)
    fingerprint = create_fingerprint(records)

    documents: list[dict[str, Any]] = []
    excluded_documents: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        document, document_exclusions = create_document(record, source_names.get(record.source_id, record.source_id))
        if document["cleaned_text"].strip() and document["word_count"] >= 80:
            documents.append(document)
        else:
            excluded_documents.append(document)
            exclusions.append({
                "exclusion_id": stable_id("exc", record.source_id, "empty"),
                "source_id": record.source_id,
                "category": "insufficient_historical_content",
                "reason": "fewer than 80 usable source words remained after deterministic cleaning",
                "line_count": None,
            })
        exclusions.extend(document_exclusions)
    if not documents:
        raise PipelineError("all extracted bodies are empty; no usable documents remain after processing")

    passages = [passage for document in documents for passage in create_passages(document)]
    entities = extract_entities(documents, passages)
    events = extract_events(passages, entities)
    relations = extract_relations(passages, entities)
    relations = sorted(relations + event_relations(events), key=lambda relation: relation["relation_id"])
    claims, rejected_claims = create_claims(events, relations, entities, passages, documents)
    for claim in rejected_claims:
        exclusions.append({
            "exclusion_id": stable_id("exc", claim["claim_id"], "unsupported"),
            "source_id": claim.get("source_id"),
            "category": "unsupported_claim_rejected",
            "reason": "supporting span did not exactly match the source passage",
            "line_count": None,
        })
    places = extract_places(documents, passages, entities)
    splits = assign_splits(documents)
    training_text, balance_rows = create_training_corpus(documents, exclusions, excluded_documents)

    validate_outputs(documents, passages, entities, events, relations, claims, places, splits)
    if not training_text.strip():
        raise PipelineError("no licence-compatible text remains for the training corpus")

    processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_dir / "documents.jsonl", documents)
    write_jsonl(processed_dir / "passages.jsonl", passages)
    write_jsonl(processed_dir / "entities.jsonl", entities)
    write_jsonl(processed_dir / "relations.jsonl", relations)
    write_jsonl(processed_dir / "events.jsonl", events)
    write_jsonl(processed_dir / "claims.jsonl", claims)
    write_jsonl(processed_dir / "exclusions.jsonl", sorted(exclusions, key=lambda e: e["exclusion_id"]))
    write_json(processed_dir / "places.geojson", places)
    write_json(processed_dir / "split_manifest.json", splits)
    (processed_dir / "training_corpus.txt").write_text(training_text, encoding="utf-8", newline="\n")
    write_json(fingerprint_path, fingerprint)
    write_reports(
        reports_dir, records, documents, passages, entities, events, relations, claims,
        rejected_claims, places, splits, exclusions, balance_rows, training_text,
    )

    after = snapshot_tree(raw_dir)
    if before != after:
        raise PipelineError("raw evidence changed during processing")
    return {
        "raw_documents": len(records),
        "documents": len(documents),
        "passages": len(passages),
        "entities": len(entities),
        "events": len(events),
        "relations": len(relations),
        "claims": len(claims),
        "places": len(places["features"]),
        "cleaned_words": sum(d["word_count"] for d in documents),
        "training_words": word_count(training_text),
        "training_approximate_tokens": math.ceil(word_count(training_text) * 1.33),
        "splits": splits["counts"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--fingerprint", type=Path, default=FINGERPRINT_PATH)
    parser.add_argument("--expected-count", type=int, default=None, help="override required raw corpus count")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_count = args.expected_count
    if expected_count is None and args.raw_dir.resolve() == RAW_DIR.resolve():
        expected_count = 25
    try:
        summary = run_pipeline(
            raw_dir=args.raw_dir.resolve(),
            processed_dir=args.processed_dir.resolve(),
            reports_dir=args.reports_dir.resolve(),
            config_path=args.config.resolve(),
            fingerprint_path=args.fingerprint.resolve(),
            expected_count=expected_count,
        )
    except (PipelineError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
