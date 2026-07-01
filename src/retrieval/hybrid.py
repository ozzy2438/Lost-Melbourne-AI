"""Explainable Reciprocal Rank Fusion with exact title/entity/alias bonuses."""

from __future__ import annotations

import re
from collections import defaultdict

from .models import RetrievalResult, SearchPassage


def reciprocal_rank_fusion(
    query: str,
    passages: list[SearchPassage],
    ranked_lists: dict[str, list[RetrievalResult]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    passage_by_id = {passage.passage_id: passage for passage in passages}
    components: dict[str, dict[str, float]] = defaultdict(dict)
    explanations: dict[str, list[str]] = defaultdict(list)
    for method, results in ranked_lists.items():
        for rank, result in enumerate(results, 1):
            value = 1.0 / (rrf_k + rank)
            components[result.passage_id][f"rrf_{method}"] = value
            explanations[result.passage_id].append(f"{method} rank={rank} contributed {value:.6f}")

    lowered = query.casefold()
    for passage_id in list(components):
        passage = passage_by_id[passage_id]
        title_bonus = 0.03 if passage.title.casefold() in lowered else 0.0
        alias_matches = [alias for alias in passage.aliases if re.search(rf"\b{re.escape(alias.casefold())}\b", lowered)]
        entity_matches = [name for name in passage.entity_names if re.search(rf"\b{re.escape(name.casefold())}\b", lowered)]
        entity_bonus = min(0.04, 0.01 * len(entity_matches))
        alias_bonus = min(0.04, 0.02 * len(alias_matches))
        components[passage_id].update({"title_bonus": title_bonus, "entity_bonus": entity_bonus, "alias_bonus": alias_bonus})
        if title_bonus:
            explanations[passage_id].append("exact title match bonus")
        if entity_bonus:
            explanations[passage_id].append(f"entity matches: {', '.join(entity_matches)}")
        if alias_bonus:
            explanations[passage_id].append(f"alias matches: {', '.join(alias_matches)}")

    order = sorted(components, key=lambda passage_id: (-sum(components[passage_id].values()), passage_id))
    output: list[RetrievalResult] = []
    for passage_id in order[:top_k]:
        passage = passage_by_id[passage_id]
        output.append(RetrievalResult(
            passage_id=passage_id,
            parent_passage_id=passage.parent_passage_id,
            score=sum(components[passage_id].values()),
            rank=len(output) + 1,
            score_components=components[passage_id],
            explanation=explanations[passage_id],
        ))
    return output
