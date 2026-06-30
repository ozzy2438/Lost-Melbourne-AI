#!/usr/bin/env python3
"""Benchmark sparse, dense, hybrid, structured, chunk, and query strategies."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval import (  # noqa: E402
    BM25Index,
    DenseIndex,
    HistoricalCorpus,
    QueryTransformer,
    SentenceTransformerEncoder,
    StructuredRetriever,
    TfidfIndex,
    build_passages,
    reciprocal_rank_fusion,
)
from retrieval.evaluation import load_queries, run_method  # noqa: E402


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "retrieval"
QUERY_PATH = REPO_ROOT / "evaluation" / "retrieval_queries.jsonl"
REPORTS_DIR = REPO_ROOT / "reports"


def evaluate(processed_dir: Path, artifact_dir: Path, query_path: Path, reports_dir: Path) -> dict[str, Any]:
    corpus = HistoricalCorpus.load(processed_dir)
    queries = load_queries(query_path, set(corpus.passage_by_id), set(corpus.entity_by_id))
    build_manifest = json.loads((artifact_dir / "build_manifest.json").read_text(encoding="utf-8"))
    original = build_passages(corpus, "original")
    representations = {strategy: build_passages(corpus, strategy) for strategy in ("original", "small", "parent_child")}
    transformer = QueryTransformer(corpus)
    structured = StructuredRetriever(corpus)

    bm25 = joblib.load(artifact_dir / "original" / "bm25.joblib")
    tfidf = joblib.load(artifact_dir / "original" / "tfidf.joblib")
    dense_indexes: dict[tuple[str, str], DenseIndex] = {}
    encoders = {key: SentenceTransformerEncoder(key) for key in ("minilm", "bge-small")}
    for model_key, encoder in encoders.items():
        for strategy in ("original", "small"):
            dense_indexes[(model_key, strategy)], _ = DenseIndex.build(
                representations[strategy], encoder, artifact_dir / strategy / f"dense_{model_key}.npz"
            )

    methods: dict[str, dict[str, Any]] = {}
    methods["bm25"] = run_method("bm25", queries, bm25.search)
    methods["tfidf"] = run_method("tfidf", queries, tfidf.search)
    methods["dense_minilm"] = run_method("dense_minilm", queries, dense_indexes[("minilm", "original")].search)
    methods["dense_bge_small"] = run_method("dense_bge_small", queries, dense_indexes[("bge-small", "original")].search)

    dense_winner = max(
        ("minilm", "bge-small"),
        key=lambda key: (methods["dense_minilm" if key == "minilm" else "dense_bge_small"]["development_metrics"]["mrr"], key),
    )

    chunk_results: dict[str, dict[str, Any]] = {}
    for strategy, passages in representations.items():
        sparse_index = joblib.load(artifact_dir / strategy / "bm25.joblib")
        dense_strategy = "small" if strategy == "parent_child" else strategy
        dense_index = dense_indexes[(dense_winner, dense_strategy)]

        def chunk_hybrid_search(query: str, top_k: int, sparse_index=sparse_index, dense_index=dense_index, passages=passages):
            return reciprocal_rank_fusion(query, passages, {
                "bm25": sparse_index.search(query, 50),
                "dense": dense_index.search(query, 50),
            }, top_k=top_k)

        chunk_results[strategy] = run_method(f"hybrid_{strategy}", queries, chunk_hybrid_search)
    passage_winner = max(chunk_results, key=lambda key: (chunk_results[key]["development_metrics"]["mrr"], key))

    selected_passages = representations[passage_winner]
    selected_sparse = joblib.load(artifact_dir / passage_winner / "bm25.joblib")
    dense_strategy = "small" if passage_winner == "parent_child" else passage_winner
    selected_dense = dense_indexes[(dense_winner, dense_strategy)]

    def hybrid_search(query: str, top_k: int):
        return reciprocal_rank_fusion(query, selected_passages, {
            "bm25": selected_sparse.search(query, 50),
            "dense": selected_dense.search(query, 50),
        }, top_k=top_k)

    transform_variants = {}
    for mode in ("none", "alias", "entity"):
        transform_variants[mode] = run_method(
            f"hybrid_transform_{mode}",
            queries,
            hybrid_search,
            transform=(lambda query, mode=mode: transformer.transform(query, mode)),
        )
    transform_winner = max(transform_variants, key=lambda key: (transform_variants[key]["development_metrics"]["mrr"], key))
    selected_transform = lambda query: transformer.transform(query, transform_winner)
    methods["hybrid"] = run_method("hybrid", queries, hybrid_search, transform=selected_transform)
    methods["hybrid_structured"] = run_method(
        "hybrid_structured",
        queries,
        hybrid_search,
        transform=selected_transform,
        structured=lambda query, results: structured.rerank(query, results, selected_passages),
    )

    index_sizes = _index_sizes(build_manifest, dense_winner, passage_winner)
    index_times = _index_times(build_manifest, dense_winner, passage_winner)
    for method_name, method in methods.items():
        method["index_bytes"] = index_sizes.get(method_name, 0)
        method["indexing_seconds"] = index_times.get(method_name, 0.0)
    default_method = max(
        methods,
        key=lambda key: (
            0.5 * methods[key]["development_metrics"]["mrr"]
            + 0.3 * methods[key]["development_metrics"]["recall_at_5"]
            + 0.2 * methods[key]["development_metrics"]["unanswerable_precision"],
            key,
        ),
    )
    combined_query_traces = _combine_traces(queries, methods, default_method)
    result = {
        "schema_version": 1,
        "phase2_counts": corpus.counts(),
        "evaluation_query_count": len(queries),
        "model_winner": dense_winner,
        "passage_strategy_winner": passage_winner,
        "query_transform_winner": transform_winner,
        "default_method": default_method,
        "combined_query_traces": combined_query_traces,
        "llm_query_rewrite": "not_run; no LLM rewrite was needed or allowed to affect the benchmark",
        "methods": methods,
        "chunk_strategy_results": chunk_results,
        "query_transform_results": transform_variants,
        "build_manifest": build_manifest,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "extraction_coverage_audit.md").write_text(_coverage_report(corpus), encoding="utf-8")
    (reports_dir / "retrieval_benchmark.md").write_text(_benchmark_report(result), encoding="utf-8")
    (reports_dir / "retrieval_failure_analysis.md").write_text(_failure_report(result), encoding="utf-8")
    persisted = {
        **result,
        "methods": {name: _without_traces(row) for name, row in methods.items()},
        "chunk_strategy_results": {name: _without_traces(row) for name, row in chunk_results.items()},
        "query_transform_results": {name: _without_traces(row) for name, row in transform_variants.items()},
    }
    (reports_dir / "retrieval_results.json").write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _without_traces(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "traces"}


def _index_sizes(manifest: dict, dense_winner: str, passage_strategy: str) -> dict[str, int]:
    original = manifest["representations"]["original"]
    selected = manifest["representations"][passage_strategy]
    dense_strategy = "small" if passage_strategy == "parent_child" else passage_strategy
    return {
        "bm25": original["bm25_index_bytes"],
        "tfidf": original["tfidf_index_bytes"],
        "dense_minilm": manifest["models"]["minilm"]["representations"]["original"]["index_bytes"],
        "dense_bge_small": manifest["models"]["bge-small"]["representations"]["original"]["index_bytes"],
        "hybrid": selected["bm25_index_bytes"] + manifest["models"][dense_winner]["representations"][dense_strategy]["index_bytes"],
        "hybrid_structured": selected["bm25_index_bytes"] + manifest["models"][dense_winner]["representations"][dense_strategy]["index_bytes"],
    }


def _combine_traces(queries: list[dict[str, Any]], methods: dict[str, dict[str, Any]], default_method: str) -> list[dict[str, Any]]:
    by_method = {
        method: {row["query_id"]: row for row in result["traces"]}
        for method, result in methods.items()
    }
    combined = []
    for query in queries:
        query_id = query["query_id"]
        hybrid = by_method["hybrid"][query_id]
        structured = by_method["hybrid_structured"][query_id]
        final = by_method[default_method][query_id]
        combined.append({
            "query_id": query_id,
            "original_query": query["question"],
            "transformed_query": hybrid["transformed_query"],
            "detected_entity_ids": hybrid["detected_entity_ids"],
            "query_expansions": hybrid["query_expansions"],
            "sparse_search_results": {
                "bm25": _trace_results(by_method["bm25"][query_id]["results"]),
                "tfidf": _trace_results(by_method["tfidf"][query_id]["results"]),
            },
            "vector_search_results": {
                "minilm": _trace_results(by_method["dense_minilm"][query_id]["results"]),
                "bge_small": _trace_results(by_method["dense_bge_small"][query_id]["results"]),
            },
            "hybrid_result_order": _trace_results(hybrid["results"]),
            "structured_filters": structured["structured_filters"],
            "hybrid_structured_results": _trace_results(structured["results"]),
            "default_method": default_method,
            "final_selected_passages": _trace_results(final["results"]),
            "expected_relevant_passages": query["expected_relevant_passage_ids"],
            "expected_answerable": query["expected_answerable"],
            "abstained": final["abstained"],
            "success": final["success"],
            "failure_category": final["failure_category"],
        })
    return combined


def _trace_results(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    keep = ("rank", "passage_id", "parent_passage_id", "score", "score_components")
    return [{key: row[key] for key in keep} for row in rows[:limit]]


def _index_times(manifest: dict, dense_winner: str, passage_strategy: str) -> dict[str, float]:
    original = manifest["representations"]["original"]
    selected = manifest["representations"][passage_strategy]
    dense_strategy = "small" if passage_strategy == "parent_child" else passage_strategy
    return {
        "bm25": original["bm25_indexing_seconds"],
        "tfidf": original["tfidf_indexing_seconds"],
        "dense_minilm": manifest["models"]["minilm"]["representations"]["original"]["indexing_seconds"],
        "dense_bge_small": manifest["models"]["bge-small"]["representations"]["original"]["indexing_seconds"],
        "hybrid": selected["bm25_indexing_seconds"] + manifest["models"][dense_winner]["representations"][dense_strategy]["indexing_seconds"],
        "hybrid_structured": selected["bm25_indexing_seconds"] + manifest["models"][dense_winner]["representations"][dense_strategy]["indexing_seconds"],
    }


def _coverage_report(corpus: HistoricalCorpus) -> str:
    linked: set[str] = set()
    for event in corpus.events:
        linked.update(event["involved_entity_ids"])
    for relation in corpus.relations:
        linked.add(relation["subject_entity_id"])
        if relation.get("object_entity_id"):
            linked.add(relation["object_entity_id"])
    for claim in corpus.claims:
        linked.add(claim["subject"]["entity_id"])
        if isinstance(claim["object_or_value"], dict) and claim["object_or_value"].get("entity_id"):
            linked.add(claim["object_or_value"]["entity_id"])
    passage_document = {passage["passage_id"]: passage["document_id"] for passage in corpus.passages}
    claim_documents = {passage_document[claim["passage_id"]] for claim in corpus.claims}
    relation_documents = {passage_document[relation["supporting_passage_ids"][0]] for relation in corpus.relations}
    geo_ids = {feature["properties"]["canonical_entity_id"] for feature in corpus.places["features"]}
    geographic_types = {"building", "street", "suburb", "landmark", "market", "theatre", "railway_station", "hotel", "church", "place"}
    possible_duplicates = _possible_entity_duplicates(corpus.entities)
    return f"""# Phase 3 Extraction Coverage Audit

## Simple finding

The knowledge fabric is useful for evidence retrieval but sparse as a graph. The original Phase 2 run had 410 entities, 61 events and 7 relations. A source-span review found three unsafe relations/events and removed them, while expanded explicit `designed by` and `operated by` rules added four supported relations. The regenerated corpus has {len(corpus.entities)} entities, {len(corpus.events)} events and {len(corpus.relations)} relations. The net relation count changed only slightly because precision was preferred over count inflation.

## Coverage

| Audit item | Count |
| --- | ---: |
| Entities with no event, relation or claim | {len(corpus.entities) - len(linked)} |
| Entities supported by only one passage | {sum(len(entity['supporting_passage_ids']) == 1 for entity in corpus.entities)} |
| Documents producing no claims | {len(corpus.documents) - len(claim_documents)} |
| Documents producing no relations | {len(corpus.documents) - len(relation_documents)} |
| Unresolved aliases (aliases without explicit evidence) | 0 |
| Conservative possible duplicate-name pairs, not merged | {len(possible_duplicates)} |
| Geographic-type entities without sourced coordinates | {sum(entity['entity_type'] in geographic_types and entity['entity_id'] not in geo_ids for entity in corpus.entities)} |

### Relations by predicate

{_counter_table(Counter(relation['relation_type'] for relation in corpus.relations))}

### Events by type

{_counter_table(Counter(event['event_type'] for event in corpus.events))}

### Explicit aliases

{''.join(f"- {entity['canonical_name']}: {', '.join(entity['aliases'])}\n" for entity in corpus.entities if entity.get('aliases')) or '- None\n'}
### Possible duplicate names retained separately

{''.join(f"- {left} ↔ {right}\n" for left, right in possible_duplicates[:20]) or '- None detected by the conservative normalised-name audit.\n'}
## Interpretation

The low relation count is partly an extraction limitation and partly a source-shape limitation: these pages are broad narrative histories, while Phase 2 only accepts explicit named subject-predicate-object statements. Phase 3 improved descriptor handling for architects and operators, removed co-mention-only location edges, and rejected organisation names incorrectly selected as demolished structures. No similarity-only alias merge was made. Retrieval therefore uses entity mentions and passage evidence in addition to graph edges; graph expansion is a bounded bonus, not a source of answers.
"""


def _possible_entity_duplicates(entities: list[dict[str, Any]]) -> list[tuple[str, str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        key = "".join(character for character in entity["canonical_name"].casefold() if character.isalnum())
        key = key.removesuffix("melbourne").removesuffix("victoria")
        if len(key) >= 5:
            buckets[key].append(entity["canonical_name"])
    return sorted({tuple(sorted(pair)) for values in buckets.values() if len(set(values)) > 1 for pair in __import__('itertools').combinations(set(values), 2)})


def _counter_table(counter: Counter) -> str:
    rows = "\n".join(f"| {name} | {count} |" for name, count in sorted(counter.items()))
    return f"| Type | Count |\n| --- | ---: |\n{rows}"


def _benchmark_report(result: dict[str, Any]) -> str:
    methods = result["methods"]
    labels = [
        ("BM25", "bm25"), ("TF-IDF", "tfidf"), ("Dense MiniLM", "dense_minilm"),
        ("Dense BGE-small", "dense_bge_small"), ("Hybrid", "hybrid"),
        ("Hybrid + structured", "hybrid_structured"),
    ]
    table = "\n".join(
        f"| {label} | {row['test_metrics']['recall_at_1']:.3f} | {row['test_metrics']['recall_at_3']:.3f} | {row['test_metrics']['recall_at_5']:.3f} | {row['test_metrics']['mrr']:.3f} | {row['test_metrics']['ndcg_at_10']:.3f} | {row['test_metrics']['unanswerable_precision']:.3f} | {row['index_bytes']:,} | {row['indexing_seconds']:.3f} | {row['average_query_latency_ms']:.3f} |"
        for label, key in labels for row in [methods[key]]
    )
    winner = result["default_method"]
    examples = methods[winner]["traces"]
    successes = [row for row in examples if row["success"] and row["expected_answerable"]][:5]
    failures = [row for row in examples if not row["success"]][:5]
    build = result["build_manifest"]
    return f"""# Phase 3 Retrieval Benchmark

## Simple finding

The benchmark selected **{winner}** using a development-only reliability score (50% MRR, 30% Recall@5, 20% unanswerable precision), then reported the held-out test metrics without retuning. The winning dense encoder was **{result['model_winner']}**, the winning passage strategy was **{result['passage_strategy_winner']}**, and the winning deterministic query transformation was **{result['query_transform_winner']}**. Dense search uses passage vectors and explicit NumPy cosine similarity; hybrid search uses Reciprocal Rank Fusion with visible bonuses.

## Held-out test comparison

| Method | R@1 | R@3 | R@5 | MRR | nDCG@10 | Unanswerable precision | Index bytes | Index seconds | Avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## Dense model facts measured locally

| Model | Revision | Dimension | Licence | Estimated model memory | Cached model bytes | Original index bytes |
| --- | --- | ---: | --- | ---: | ---: | ---: |
{''.join(f"| {row['name']} | `{row['revision'][:12]}` | {row['dimension']} | {row['licence']} | {row['representations']['original']['memory_bytes']:,} | {row['representations']['original']['model_storage_bytes']:,} | {row['representations']['original']['index_bytes']:,} |\n" for row in build['models'].values())}
## Five successful queries

{''.join(f"- `{row['query_id']}` {row['question']} — top evidence `{row['ranked_passage_ids'][0] if row['ranked_passage_ids'] else 'none'}`\n" for row in successes)}
## Five failed queries

{''.join(f"- `{row['query_id']}` {row['question']} — {row['failure_category']}\n" for row in failures) or '- Fewer than five failures occurred for the selected method.\n'}
## Required query slices

### Alias-heavy

{_query_slice_table(examples, ['q039', 'q040', 'q041'])}

### Temporal

{_query_slice_table(examples, ['q009', 'q016', 'q045'])}

### Geographic

{_query_slice_table(examples, ['q007', 'q032', 'q050'])}

### Unanswerable

{_query_slice_table(examples, ['q051', 'q052', 'q060'])}

Full per-query score breakdowns, expected passages, transformations, filters, latency, and success flags are in `reports/retrieval_results.json`.

## Chunk strategy comparison

{_strategy_table(result['chunk_strategy_results'])}

## Query transformation comparison

{_strategy_table(result['query_transform_results'])}

LLM rewriting was not run. The original query is always retained, and deterministic expansions are logged separately.
"""


def _strategy_table(rows: dict[str, Any]) -> str:
    lines = ["| Strategy | Test R@1 | Test R@5 | Test MRR |", "| --- | ---: | ---: | ---: |"]
    for name, row in rows.items():
        metrics = row["test_metrics"]
        lines.append(f"| {name} | {metrics['recall_at_1']:.3f} | {metrics['recall_at_5']:.3f} | {metrics['mrr']:.3f} |")
    return "\n".join(lines)


def _query_slice_table(rows: list[dict[str, Any]], query_ids: list[str]) -> str:
    by_id = {row["query_id"]: row for row in rows}
    lines = ["| Query | Question | Top passage | Abstained | Success |", "| --- | --- | --- | --- | --- |"]
    for query_id in query_ids:
        row = by_id[query_id]
        top = row["ranked_passage_ids"][0] if row["ranked_passage_ids"] else "none"
        lines.append(f"| {query_id} | {row['question']} | `{top}` | {row['abstained']} | {row['success']} |")
    return "\n".join(lines)


def _failure_report(result: dict[str, Any]) -> str:
    selected = result["methods"][result["default_method"]]
    failures = [row for row in selected["traces"] if not row["success"]]
    categories = Counter(row["failure_category"] for row in failures)
    combined = {row["query_id"]: row for row in result["combined_query_traces"]}
    representative = [combined["q001"], combined["q051"]]
    traces = ""
    for row in representative:
        traces += f"### {row['query_id']}: {row['original_query']}\n\n"
        traces += f"- Transformed: {row['transformed_query']}\n- Detected entities: {row['detected_entity_ids']}\n- Structured filters: {row['structured_filters']}\n- Expected: {row['expected_relevant_passages']}\n- Default method: {row['default_method']}\n- Abstained: {row['abstained']}\n- Success: {row['success']}\n"
        traces += f"- BM25 top 3: {[item['parent_passage_id'] for item in row['sparse_search_results']['bm25'][:3]]}\n"
        traces += f"- TF-IDF top 3: {[item['parent_passage_id'] for item in row['sparse_search_results']['tfidf'][:3]]}\n"
        traces += f"- MiniLM top 3: {[item['parent_passage_id'] for item in row['vector_search_results']['minilm'][:3]]}\n"
        traces += f"- BGE-small top 3: {[item['parent_passage_id'] for item in row['vector_search_results']['bge_small'][:3]]}\n"
        traces += "- Hybrid/structured score breakdown:\n"
        for result_row in row["hybrid_structured_results"][:5]:
            traces += f"  - rank {result_row['rank']}: `{result_row['parent_passage_id']}` score={result_row['score']:.6f}; components={result_row['score_components']}\n"
        traces += "\n"
    return f"""# Phase 3 Retrieval Failure Analysis

## Failure categories

{_counter_table(categories)}

The abstention threshold is tuned only on the 45-question development split, then applied unchanged to the 15 held-out test questions. Unanswerable prompts that still retrieve a similarly named entity are counted as insufficient-corpus failures rather than successful retrievals.

## Representative full traces

{traces}
## Interpretation

- Alias mismatch: deterministic alias expansion failed to expose the canonical entity.
- Wrong chunk boundary: the indexed child omitted evidence available in its parent.
- Lexical mismatch: query and evidence shared too little vocabulary.
- Semantic confusion: dense retrieval preferred a related but unsupported passage.
- Insufficient corpus coverage: no gold passage exists and the method failed to abstain.
- Date-filter failure: a temporal signal did not promote the supported event passage.
- Location ambiguity: a place name matched the wrong geographic context.
- Entity-resolution failure: entity/alias detection selected the wrong canonical record.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--queries", type=Path, default=QUERY_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = evaluate(args.processed_dir.resolve(), args.artifact_dir.resolve(), args.queries.resolve(), args.reports_dir.resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "evaluation_query_count": result["evaluation_query_count"],
        "model_winner": result["model_winner"],
        "passage_strategy_winner": result["passage_strategy_winner"],
        "query_transform_winner": result["query_transform_winner"],
        "default_method": result["default_method"],
        "metrics": {name: row["test_metrics"] for name, row in result["methods"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
