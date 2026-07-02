"""Streamlit demo for the retrieval-only answering pipeline.

Retrieval-only: this UI calls the existing ``RetrievalAnswerer`` and displays
its evidence passages or abstention message. It does not modify retrieval
logic, does not generate free-form text, and does not connect the Tiny
Transformer. This module only changes presentation.
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

EXAMPLE_QUESTIONS = [
    "Who designed the Metropolitan Meat Market?",
    "Which architects designed Bolte Bridge?",
    "Where is South Melbourne Market located?",
    "In what year was Melbourne Fish Market demolished?",
]


@st.cache_resource(show_spinner="Loading retrieval indexes...")
def load_answerer() -> RetrievalAnswerer:
    return RetrievalAnswerer.from_artifacts(
        processed_dir=PROCESSED_DIR,
        artifact_dir=ARTIFACT_DIR,
        query_path=QUERY_PATH,
        report_path=REPORT_PATH,
    )


def confidence_badge(score: float, threshold: float) -> tuple[str, str]:
    """Return (label, colour) for a qualitative confidence tier."""
    if score >= threshold * 2:
        return "High confidence", "#16a34a"
    if score >= threshold:
        return "Moderate confidence", "#d97706"
    return "Below threshold", "#dc2626"


st.set_page_config(
    page_title="Lost Melbourne — Retrieval Demo",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .hero {
        padding: 2rem 2.25rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
        color: #f8fafc;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.98rem;
    }
    .pill-row { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.18);
        color: #e2e8f0;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .evidence-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.6rem;
        height: 1.6rem;
        border-radius: 50%;
        background: #1e293b;
        color: white;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.55rem;
    }
    .evidence-title { font-weight: 600; color: #0f172a; font-size: 1rem; }
    .evidence-meta { color: #64748b; font-size: 0.82rem; margin: 0.15rem 0 0.5rem 0; }
    .source-tag {
        display: inline-block;
        margin-top: 0.5rem;
        margin-right: 0.4rem;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        background: #f1f5f9;
        color: #475569;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .answer-label {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
    }
    .abstain-label {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: #ffedd5;
        color: #9a3412;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏛️ Lost Melbourne — Retrieval-Only Answerer</h1>
        <p>Evidence-grounded question answering over Melbourne's historical corpus.
        Every answer is backed by cited passages — no free-form generation, no Tiny Transformer.</p>
        <div class="pill-row">
            <span class="pill">🔍 Hybrid retrieval</span>
            <span class="pill">📎 Citation-first</span>
            <span class="pill">🛑 Calibrated abstention</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About this demo")
    st.write(
        "This interface calls the existing retrieval-only answering pipeline "
        "(`RetrievalAnswerer`). It combines BM25, dense embeddings and structured "
        "entity signals, then reranks with TF-IDF lexical scoring. If confidence "
        "falls below the calibrated threshold, it abstains instead of guessing."
    )
    st.markdown("### Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"example-{q}"):
            st.session_state["question_input"] = q

question = st.text_input(
    "Ask a question about Melbourne's lost places",
    key="question_input",
    placeholder="e.g. Who designed the Metropolitan Meat Market?",
)
ask = st.button("🔎 Ask", type="primary")

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

            evidence = result["returned_evidence"]
            top_score = result["top_score"]
            threshold = result["abstention_threshold"]

            with st.container(border=True):
                if result["abstained"] or not evidence:
                    st.markdown('<span class="abstain-label">⚠️ No confident answer</span>', unsafe_allow_html=True)
                    st.write(result["fallback_response"])
                else:
                    top = evidence[0]
                    st.markdown('<span class="answer-label">✅ Answer</span>', unsafe_allow_html=True)
                    st.markdown(top["text"])
                    st.caption(f"Source: {top['title']} — {top['section_title']}")

            label, colour = confidence_badge(top_score, threshold)
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("Top score", f"{top_score:.4f}")
            with col2:
                st.metric("Abstention threshold", f"{threshold:.4f}")
            with col3:
                st.markdown(
                    f"""<div style="padding-top:1.6rem;">
                        <span class="confidence-badge" style="background:{colour};">{label}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.progress(min(1.0, top_score / max(threshold * 2, 1e-6)))

            if evidence:
                st.markdown(f"#### 📚 Evidence passages ({len(evidence)})")
                for rank, item in enumerate(evidence, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f'<span class="evidence-rank">{rank}</span>'
                            f'<span class="evidence-title">{item["title"]} — {item["section_title"]}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="evidence-meta">score {item["score"]:.4f}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(item["text"])
                        tags = f'<span class="source-tag">🆔 {item["passage_id"]}</span>'
                        if item["source_url"]:
                            tags += (
                                f'<span class="source-tag">🔗 <a href="{item["source_url"]}" '
                                f'target="_blank">source</a></span>'
                            )
                        st.markdown(tags, unsafe_allow_html=True)
