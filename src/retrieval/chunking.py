"""Deterministic passage representations used by the retrieval benchmark."""

from __future__ import annotations

import hashlib
import math
import re

from .corpus import HistoricalCorpus
from .models import SearchPassage


def build_passages(corpus: HistoricalCorpus, strategy: str = "original") -> list[SearchPassage]:
    if strategy not in {"original", "small", "parent_child"}:
        raise ValueError(f"unknown passage strategy: {strategy}")
    documents = corpus.document_by_id
    entities_by_passage = corpus.entities_by_passage
    original: list[SearchPassage] = []
    for passage in sorted(corpus.passages, key=lambda row: row["passage_id"]):
        document = documents[passage["document_id"]]
        entities = entities_by_passage.get(passage["passage_id"], [])
        names = tuple(sorted({entity["canonical_name"] for entity in entities}))
        aliases = tuple(sorted({alias for entity in entities for alias in entity.get("aliases", [])}))
        searchable = "\n".join(filter(None, [document["title"], passage["section_title"], " ".join(names), " ".join(aliases), passage["text"]]))
        original.append(SearchPassage(
            passage_id=passage["passage_id"],
            parent_passage_id=passage["passage_id"],
            document_id=passage["document_id"],
            text=passage["text"],
            search_text=searchable,
            title=document["title"],
            section_title=passage["section_title"],
            entity_names=names,
            aliases=aliases,
            metadata={"source_url": passage["source_url"], "licence": passage["licence"], "strategy": "original"},
        ))
    if strategy == "original":
        return original
    return [child for parent in original for child in _split_parent(parent, return_parent=strategy == "parent_child")]


def _split_parent(parent: SearchPassage, target_words: int = 170, overlap_words: int = 30, return_parent: bool = False) -> list[SearchPassage]:
    words = list(re.finditer(r"\S+", parent.text))
    if len(words) <= 220:
        return [SearchPassage(
            passage_id=_child_id(parent.passage_id, 0, parent.text),
            parent_passage_id=parent.passage_id,
            document_id=parent.document_id,
            text=parent.text if not return_parent else parent.text,
            search_text=parent.search_text,
            title=parent.title,
            section_title=parent.section_title,
            entity_names=parent.entity_names,
            aliases=parent.aliases,
            metadata={**parent.metadata, "strategy": "parent_child" if return_parent else "small", "child_text": parent.text},
        )]
    chunk_count = max(1, math.ceil((len(words) - overlap_words) / (220 - overlap_words)))
    window = math.ceil((len(words) + (chunk_count - 1) * overlap_words) / chunk_count)
    stride = window - overlap_words
    children: list[SearchPassage] = []
    for index, start_word in enumerate(range(0, len(words), stride)):
        end_word = min(start_word + window, len(words))
        child_text = parent.text[words[start_word].start() : words[end_word - 1].end()]
        child_id = _child_id(parent.passage_id, index, child_text)
        searchable = "\n".join(filter(None, [parent.title, parent.section_title, " ".join(parent.entity_names), " ".join(parent.aliases), child_text]))
        children.append(SearchPassage(
            passage_id=child_id,
            parent_passage_id=parent.passage_id,
            document_id=parent.document_id,
            text=parent.text if return_parent else child_text,
            search_text=searchable,
            title=parent.title,
            section_title=parent.section_title,
            entity_names=parent.entity_names,
            aliases=parent.aliases,
            metadata={**parent.metadata, "strategy": "parent_child" if return_parent else "small", "child_text": child_text},
        ))
        if end_word >= len(words):
            break
    return children


def _child_id(parent_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{parent_id}\x1f{index}\x1f{text}".encode()).hexdigest()[:16]
    return f"small_{digest}"
