"""Streamlit interface: upload PDFs, manage documents, chat with citations."""
from __future__ import annotations

import os
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Engineering Docs Assistant", page_icon="📘", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the backend. Shows an error in the UI and returns None on failure."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API_URL}. Is the backend running? ({exc})")
        return None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"API error {resp.status_code}: {detail}")
        return None
    return resp.json()


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['document']}, page {s['page']}** · relevance {s['score']:.2f}")
            st.caption(s["snippet"] + ("…" if len(s["snippet"]) >= 300 else ""))


# ---------- sidebar: documents ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload engineering PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index uploads", disabled=not uploads, use_container_width=True):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                result = api("POST", "/documents", files={"file": (f.name, f.getvalue(), "application/pdf")})
            if result:
                st.success(f"{result['document']}: {result['pages']} pages, {result['chunks']} chunks")

    docs = api("GET", "/documents") or []
    st.subheader("Indexed")
    if not docs:
        st.info("No documents indexed yet.")
    for d in docs:
        col_info, col_del = st.columns([4, 1])
        col_info.markdown(f"**{d['name']}**  \n{d['pages']} pages · {d['chunks']} chunks")
        if col_del.button("🗑", key=f"del-{d['name']}", help="Remove from index"):
            api("DELETE", f"/documents/{quote(d['name'])}")
            st.rerun()

    selected = st.multiselect("Limit questions to", [d["name"] for d in docs])

# ---------- main: chat ----------
st.title("📘 AI Engineering Documentation Assistant")
st.caption("Answers are grounded in your uploaded documents. Always verify against the source before acting on them.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

prompt = st.chat_input("Ask a question about your documents")
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            result = api(
                "POST",
                "/query",
                json={
                    "question": prompt,
                    "doc_names": selected or None,
                    "history": history[-12:],
                },
            )
        if result:
            st.markdown(result["answer"])
            render_sources(result["sources"])
            st.caption(f"Answered in {result['latency_ms']} ms")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
