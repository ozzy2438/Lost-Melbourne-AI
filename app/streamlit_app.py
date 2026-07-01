"""Minimal Streamlit demo for the retrieval-only answering pipeline.

Retrieval-only: this UI calls the existing ``RetrievalAnswerer`` and displays
its evidence passages or abstention message. It does not modify retrieval
logic, does not generate free-form text, and does not connect the Tiny
Transformer.
"""

from __future__ import annotations

import sys
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
            with st.spinner("Retrieving evidence..."):
                result = answerer.answer(question)

            st.subheader("Answer")
            evidence = result["returned_evidence"]
            if result["abstained"] or not evidence:
                st.warning(result["fallback_response"])
            else:
                st.write(evidence[0]["text"])

            st.metric("Top score (confidence)", f"{result['top_score']:.4f}")
            st.caption(f"Abstention threshold: {result['abstention_threshold']:.4f}")

            if evidence:
                st.subheader(f"Evidence passages ({len(evidence)})")
                for rank, item in enumerate(evidence, start=1):
                    header = f"{rank}. {item['title']} — {item['section_title']} (score {item['score']:.4f})"
                    with st.expander(header):
                        st.write(item["text"])
                        st.markdown(f"**Source ID:** `{item['passage_id']}`")
                        st.markdown(f"**Section title:** {item['section_title']}")
                        if item["source_url"]:
                            st.markdown(f"**Source URL:** {item['source_url']}")
