"""
Streamlit UI for the RAG chatbot (calls the FastAPI backend).

Duolingo-inspired dark lesson layout (independent demo; not affiliated with Duolingo).

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

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

DEFAULT_API = "http://127.0.0.1:8000"

# Palette inspired by Duolingo-style lesson UI (educational homage only).
C_BG = "#131f24"
C_SURFACE = "#202f36"
C_GREEN = "#78c800"
C_GREEN_SHADOW = "#5fa000"
C_PURPLE = "#ce82ff"
C_GOLD = "#ffc800"
C_TEXT = "#f7f7f7"
C_TEXT_MUTED = "#afafaf"


def inject_theme_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {C_BG} !important;
            color: {C_TEXT};
        }}
        .main .block-container {{
            padding: 0.75rem 1.25rem 2rem;
            max-width: 520px;
            font-family: 'Nunito', 'Segoe UI', system-ui, sans-serif;
        }}
        header[data-testid="stHeader"] {{
            background: {C_BG} !important;
        }}
        #MainMenu {{ visibility: hidden; height: 0; }}
        footer {{ visibility: hidden; height: 0; }}

        /* Top lesson bar */
        .duo-top-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.35rem 0 0.75rem;
            color: {C_TEXT_MUTED};
            font-size: 1.35rem;
            font-weight: 800;
        }}
        .duo-close {{
            cursor: default;
            opacity: 0.5;
            width: 2rem;
            flex-shrink: 0;
        }}
        .duo-logo {{
            flex: 1;
            min-width: 0;
            text-align: center;
            letter-spacing: 0.01em;
            color: {C_TEXT};
            font-size: clamp(1.05rem, 3.8vw, 1.55rem);
            font-weight: 800;
            line-height: 1.2;
            padding: 0 0.35rem;
        }}
        .duo-streak {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.9rem;
            font-weight: 700;
            color: {C_GOLD};
            flex-shrink: 0;
        }}

        .duo-progress-track {{
            height: 14px;
            background: {C_SURFACE};
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 1rem;
            border: 1px solid #2b3e47;
        }}
        .duo-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {C_GOLD}, #ffdd66);
            border-radius: 999px;
            transition: width 0.3s ease;
        }}

        .duo-tag {{
            display: inline-block;
            color: {C_PURPLE};
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}
        .duo-headline {{
            color: {C_TEXT};
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0 0 1rem 0;
            line-height: 1.25;
        }}

        /* Chat messages */
        [data-testid="stChatMessage"] {{
            background: transparent !important;
        }}
        [data-testid="stChatMessage"] > div {{
            gap: 0.5rem !important;
        }}
        [data-testid="stChatMessage"] section[data-testid="stMarkdownContainer"] {{
            background: {C_SURFACE};
            border: 1px solid #2b3e47;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            color: {C_TEXT} !important;
        }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
            color: {C_TEXT} !important;
        }}
        [data-testid="stMarkdownContainer"] a {{
            color: {C_PURPLE} !important;
        }}
        [data-testid="stMarkdownContainer"] strong {{
            color: {C_TEXT};
        }}

        /* Chat input */
        [data-testid="stChatInput"] {{
            border-radius: 16px !important;
        }}
        [data-testid="stChatInput"] textarea {{
            background-color: {C_SURFACE} !important;
            color: {C_TEXT} !important;
            border-radius: 14px !important;
            border: 1px solid #2b3e47 !important;
            font-family: 'Nunito', sans-serif !important;
        }}
        [data-testid="stChatInputSubmitButton"] button {{
            background-color: {C_GREEN} !important;
            color: #1a2e00 !important;
            font-weight: 800 !important;
            border-radius: 12px !important;
            border: none !important;
            box-shadow: 0 3px 0 {C_GREEN_SHADOW} !important;
        }}
        [data-testid="stChatInputSubmitButton"] button:hover {{
            filter: brightness(1.05);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {C_SURFACE};
            border-right: 1px solid #2b3e47;
        }}
        section[data-testid="stSidebar"] * {{
            color: {C_TEXT} !important;
        }}
        section[data-testid="stSidebar"] .stTextInput input {{
            background: {C_BG} !important;
            border: 1px solid #2b3e47 !important;
            border-radius: 10px !important;
        }}
        section[data-testid="stSidebar"] button {{
            background: {C_GREEN} !important;
            color: #1a2e00 !important;
            font-weight: 800 !important;
            border-radius: 12px !important;
        }}

        /* Expanders (sources) */
        [data-testid="stExpander"] {{
            background: {C_SURFACE};
            border: 1px solid #2b3e47 !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpander"] summary {{
            color: {C_TEXT} !important;
            font-weight: 700 !important;
        }}

        /* Errors & spinner */
        .stAlert {{ border-radius: 12px; }}
        div[data-testid="stSpinner"] > div {{
            border-color: {C_GREEN} transparent transparent transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def lesson_progress_pct(message_count: int) -> int:
    """Fake lesson bar: grows slightly with conversation length."""
    return min(100, 18 + min(message_count, 12) * 7)


def render_lesson_header(message_count: int) -> None:
    pct = lesson_progress_pct(message_count)
    st.markdown(
        f"""
        <div class="duo-top-bar">
            <span class="duo-close">×</span>
            <span class="duo-logo">Duolingo Inspired RAG Chatbot</span>
            <span class="duo-streak">⚡ <span>RAG</span></span>
        </div>
        <div class="duo-progress-track">
            <div class="duo-progress-fill" style="width: {pct}%;"></div>
        </div>
        <div class="duo-tag">HELP CHAT</div>
        <h1 class="duo-headline">What would you like to know?</h1>
        """,
        unsafe_allow_html=True,
    )


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
        page_title="Duolingo Inspired RAG Chatbot",
        page_icon="🦉",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    inject_theme_css()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I’m here to explain **help topics** in simple terms — streaks, Super, your account, and more. "
                    "I summarize public help info; this app is **not** affiliated with Duolingo."
                ),
            }
        ]

    n_msgs = len(st.session_state.messages)
    render_lesson_header(n_msgs)

    with st.sidebar:
        st.markdown("### ⚙️ API")
        api_default = get_api_base()
        api_base = st.text_input(
            "RAG API base URL",
            value=api_default,
            help="FastAPI root, e.g. https://your-api.run.app",
        )
        if st.button("CHECK CONNECTION", use_container_width=True):
            h = fetch_health(api_base.rstrip("/"))
            if h:
                st.success("OK — index_loaded: " + str(h.get("index_loaded")))
                st.json(h)
            else:
                st.error("Could not reach /health.")

    api = api_base.rstrip("/")

    for msg in st.session_state.messages:
        av = "🦉" if msg["role"] == "assistant" else "🙂"
        with st.chat_message(msg["role"], avatar=av):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🙂"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🦉"):
            with st.spinner("Checking…"):
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
                    err = (
                        f"**Connection error**: {e!s}. "
                        "Start the API or set **RAG API base URL** in the sidebar."
                    )
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
