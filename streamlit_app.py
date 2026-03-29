"""
Streamlit UI for the RAG chatbot (calls the FastAPI backend).

Run from the repository root (with the API already running on port 8000):

    streamlit run streamlit_app.py

Environment:
    RAG_API_BASE — default http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Optional: pick up repo .env for RAG_API_BASE if set there
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

DEFAULT_API = "http://127.0.0.1:8000"


def get_api_base() -> str:
    return os.environ.get("RAG_API_BASE", DEFAULT_API).rstrip("/")


def post_chat(api_base: str, query: str, timeout: float = 120.0) -> dict[str, Any]:
    url = f"{api_base}/chat"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json={"query": query})
        r.raise_for_status()
        return r.json()


def fetch_health(api_base: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{api_base}/health")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


def main() -> None:
    st.set_page_config(
        page_title="Help RAG Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi — ask about Duolingo help topics (streaks, Super, account, etc.). "
                    "Answers use your local RAG API. This demo is **not** affiliated with Duolingo."
                ),
            }
        ]

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 720px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Help RAG chat")
    st.caption("Independent demo · powered by FastAPI + Groq · not affiliated with Duolingo")

    with st.sidebar:
        st.subheader("API")
        api_default = get_api_base()
        api_base = st.text_input("RAG API base URL", value=api_default, help="FastAPI root, e.g. http://127.0.0.1:8000")
        if st.button("Check health"):
            h = fetch_health(api_base.rstrip("/"))
            if h:
                st.json(h)
            else:
                st.error("Could not reach /health. Is uvicorn running?")

    api = api_base.rstrip("/")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about streaks, subscriptions, your account…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    data = post_chat(api, prompt)
                    answer = data.get("answer", "")
                    sources = data.get("sources") or []
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                title = s.get("title") or s.get("source_url", "")
                                url = s.get("source_url", "")
                                if url:
                                    st.markdown(f"- [{title}]({url})")
                                else:
                                    st.markdown(f"- {title}")
                        src_lines = ["**Sources:**"]
                        for s in sources:
                            t = s.get("title") or ""
                            u = s.get("source_url", "")
                            src_lines.append(f"- [{t}]({u})" if u else f"- {t}")
                        to_store = answer + "\n\n" + "\n".join(src_lines)
                    else:
                        to_store = answer
                    st.session_state.messages.append({"role": "assistant", "content": to_store})
                except httpx.HTTPStatusError as e:
                    detail = e.response.json().get("detail", str(e))
                    err = f"**API error** ({e.response.status_code}): {detail}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                except httpx.RequestError as e:
                    err = f"**Connection error**: {e!s}. Start the API: `cd backend && uvicorn app.main:app --reload --port 8000`"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
