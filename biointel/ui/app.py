"""Minimal Streamlit chat UI for BioIntel Agent.

A thin client over the FastAPI backend: it never imports the agent or touches the
databases directly, so it stays lightweight and demonstrates the API contract.
Run with ``streamlit run biointel/ui/app.py`` (or ``make ui``). Configure the
backend via ``BIOINTEL_API_URL`` and ``BIOINTEL_API_KEY`` env vars.

Features:
* Chat with conversation history (kept in session state).
* Source filters (source type, doc type, date-from) mapped to API ``filters``.
* Renders the answer plus expandable Citations, Contradictions, and a
  verification badge so the audit-ready nature of the system is visible.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("BIOINTEL_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("BIOINTEL_API_KEY", "dev-local-key-change-me")

SOURCE_TYPES = [
    "pubmed",
    "pmc",
    "clinicaltrials",
    "chembl",
    "opentargets",
    "patentsview",
]
DOC_TYPES = ["paper", "trial", "compound", "target", "patent"]

st.set_page_config(page_title="BioIntel Agent", page_icon="🧬", layout="wide")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def check_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        return None
    return None


def ask(question: str, filters: dict, session_id: str | None, auto_ingest: bool | None) -> dict:
    payload = {
        "question": question,
        "filters": filters or None,
        "session_id": session_id,
        "auto_ingest": auto_ingest,
    }
    r = requests.post(f"{API_URL}/query", json=payload, headers=_headers(), timeout=300)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧬 BioIntel Agent")
    st.caption("Agentic RAG for drug discovery & patent intelligence")

    health = check_health()
    if health:
        st.success(f"API online · profile **{health['model_profile']}**")
        st.caption(f"LLM: `{health['llm_model']}`")
        ollama = health.get("components", {}).get("ollama", "unknown")
        st.caption(f"Ollama: {'🟢' if ollama == 'up' else '🔴'} {ollama}")
    else:
        st.error(f"API unreachable at {API_URL}")

    st.divider()
    auto_ingest = st.toggle(
        "Auto-fetch sources for each question",
        value=True,
        help="Live-ingest PubMed, patents, trials, etc. for the topic before answering.",
    )
    st.subheader("Retrieval filters")
    sel_sources = st.multiselect("Source type", SOURCE_TYPES, default=[])
    sel_doctypes = st.multiselect("Document type", DOC_TYPES, default=[])
    date_from = st.text_input("Published on/after (YYYY-MM-DD)", value="")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


def build_filters() -> dict:
    f: dict = {}
    if sel_sources:
        f["source"] = sel_sources if len(sel_sources) > 1 else sel_sources[0]
    if sel_doctypes:
        f["doc_type"] = sel_doctypes if len(sel_doctypes) > 1 else sel_doctypes[0]
    if date_from.strip():
        f["date_from"] = date_from.strip()
    return f


# ------------------------------------------------------------------ renderers
def _render_citations(citations: list[dict]) -> None:
    with st.expander(f"📚 Citations ({len(citations)})"):
        for c in citations:
            marker = c.get("marker", "")
            label = c.get("label", "")
            url = c.get("source_url", "")
            quote = c.get("quote", "")
            line = f"**{marker}** {label}"
            if url:
                line += f" — [source]({url})"
            st.markdown(line)
            if quote:
                st.caption(f"“{quote}”")


def _render_contradictions(contradictions: list[dict]) -> None:
    if not contradictions:
        return
    with st.expander(f"⚠️ Contradictions ({len(contradictions)})"):
        for c in contradictions:
            st.markdown(f"- **A** ({c.get('source_a', '')}): {c.get('statement_a', '')}")
            st.markdown(f"  **B** ({c.get('source_b', '')}): {c.get('statement_b', '')}")
            st.caption(c.get("explanation", ""))


# ------------------------------------------------------------------ chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.title("BioIntel Agent")
st.caption(
    "Ask about drug targets, clinical trials, compounds, and patents. "
    "Every claim is grounded in cited sources."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations(msg["citations"])


prompt = st.chat_input("Ask a biomedical question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reasoning, and verifying citations…"):
            try:
                data = ask(prompt, build_filters(), st.session_state.session_id, auto_ingest)
            except requests.HTTPError as exc:
                st.error(f"Request failed: {exc.response.status_code} — {exc.response.text}")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()

        answer = data["answer"]
        st.session_state.session_id = data.get("session_id")

        st.markdown(answer["answer"])

        # verification badge
        if answer.get("verified"):
            st.success("✅ All claims verified against cited sources")
        else:
            st.warning("⚠️ Some claims could not be fully verified — see warnings")
        for w in answer.get("warnings", []):
            st.caption(f"• {w}")

        _render_citations(answer.get("citations", []))
        _render_contradictions(answer.get("contradictions", []))

        with st.expander("🔎 Trace"):
            st.caption(
                f"trace_id: `{data.get('trace_id')}` · "
                f"model: `{answer.get('model')}` · "
                f"latency: {data.get('latency_ms')} ms · "
                f"cached: {data.get('cached')}"
            )
            if answer.get("sub_questions"):
                st.caption("Sub-questions: " + " | ".join(answer["sub_questions"]))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer["answer"],
                "citations": answer.get("citations", []),
            }
        )
