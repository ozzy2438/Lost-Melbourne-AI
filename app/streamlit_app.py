"""Streamlit demo for the retrieval-only answering pipeline.

Retrieval-only: this UI calls the existing ``RetrievalAnswerer`` and displays
its evidence passages or abstention message. It does not modify retrieval
logic, does not generate free-form text, and does not connect the Tiny
Transformer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval.answering import RetrievalAnswerer  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "retrieval"
QUERY_PATH = REPO_ROOT / "evaluation" / "retrieval_queries.jsonl"
REPORT_PATH = REPO_ROOT / "reports" / "retrieval_results.json"


@st.cache_resource(show_spinner="Loading retrieval indexes...")
def load_answerer() -> RetrievalAnswerer:
    return RetrievalAnswerer.from_artifacts(
        processed_dir=PROCESSED_DIR,
        artifact_dir=ARTIFACT_DIR,
        query_path=QUERY_PATH,
        report_path=REPORT_PATH,
    )


st.set_page_config(page_title="Lost Melbourne — Retrieval Demo", page_icon="🏛️")
st.title("Lost Melbourne — Retrieval-Only Answerer")
st.caption(
    "Evidence-grounded, retrieval-only demo over the Melbourne historical corpus. "
    "No free-form generation and no Tiny Transformer involved."
)

question = st.text_input("Ask a question about Melbourne's lost places", "")
ask = st.button("Ask", type="primary")

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            answerer = load_answerer()
        except Exception as exc:  # missing artifacts, unbuilt indexes, etc.
            st.error(
                "Could not load retrieval indexes. Build them first with "
                "`python scripts/build_retrieval_indexes.py`.\n\n"
                f"Details: {exc}"
            )
        else:
            t0 = time.perf_counter()
            with st.spinner("Retrieving evidence..."):
                result = answerer.answer(question)
            latency_ms = (time.perf_counter() - t0) * 1000

            # ── Answer ─────────────────────────────────────────────────────
            st.subheader("Answer")
            evidence = result["returned_evidence"]
            if result["abstained"] or not evidence:
                st.warning(result["fallback_response"])
            else:
                st.write(evidence[0]["text"])
                # Inline citation for the top passage
                top = evidence[0]
                if top["source_url"]:
                    st.markdown(
                        f"📎 **Source:** [{top['title']}]({top['source_url']}) — "
                        f"*{top['section_title']}*"
                    )
                else:
                    st.markdown(
                        f"📎 **Source:** {top['title']} — *{top['section_title']}*"
                    )

            # ── Diagnostics ────────────────────────────────────────────────
            with st.expander("🔍 Diagnostics", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieval method", result["candidate_strategy"])
                col2.metric("Passages returned", len(evidence))
                col3.metric("Confidence score", f"{result['top_score']:.4f}")

                col4, col5 = st.columns(2)
                abstention_label = "Yes ✋" if result["abstained"] else "No ✅"
                col4.metric("Abstained", abstention_label)
                col5.metric("Query latency", f"{latency_ms:.1f} ms")

                st.caption(
                    f"Reranking: `{result['reranking_strategy']}` · "
                    f"Abstention threshold: `{result['abstention_threshold']:.4f}` · "
                    f"Query transform: `{result['transformed_query']}`"
                )

            # ── Evidence passages ──────────────────────────────────────────
            if evidence:
                st.subheader(f"Evidence passages ({len(evidence)})")
                for rank, item in enumerate(evidence, start=1):
                    score_str = f"{item['score']:.4f}"
                    header = f"{rank}. {item['title']} — {item['section_title']} (score {score_str})"
                    with st.expander(header):
                        st.write(item["text"])
                        st.markdown(f"**Passage ID:** `{item['passage_id']}`")
                        if item["source_url"]:
                            st.markdown(
                                f"**Source:** [{item['title']}]({item['source_url']})"
                            )
                        else:
                            st.markdown(f"**Source:** {item['title']}")
                        if item.get("licence"):
                            st.caption(f"Licence: {item['licence']}")
