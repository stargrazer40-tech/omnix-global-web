#!/usr/bin/env python3
"""OmniX AI — Premium SaaS Interface (No Modes, Deep Research, Clean UI)

Run with:
    streamlit run omnix_saas.py
Set GROQ_API_KEY in .streamlit/secrets.toml
"""

import html
import json
import time
import uuid
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

from omnixcore import (
    HAS_STT,
    SHORT_CTX,
    build_sys,
    deep_research,
    init_state,
    load_proj,
    save_proj,
    speak,
    voice_listen,
)

# ───────────────────────── Configuration ─────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
APP_VERSION = "3.0.1"

# ───────────────────────── Custom CSS ─────────────────────────
CUSTOM_CSS = """
<style>
    /* ─── Base & Reset ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0f0f13;
        --bg-secondary: #16161e;
        --bg-tertiary: #1e1e2e;
        --bg-elevated: #252535;
        --border-subtle: rgba(255,255,255,0.06);
        --border-medium: rgba(255,255,255,0.1);
        --text-primary: #f0f0f5;
        --text-secondary: #a0a0b0;
        --text-muted: #6e6e80;
        --accent: #6366f1;
        --accent-hover: #818cf8;
        --accent-glow: rgba(99,102,241,0.3);
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 30px rgba(0,0,0,0.5);
    }

    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px var(--accent-glow) !important;
    }

    .stButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px var(--accent-glow) !important;
    }

    .stButton > button[kind="secondary"] {
        background: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        border: 1px solid var(--border-medium) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }

    /* ─── Inputs ─── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ─── Selectbox & Sliders ─── */
    .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }

    .stSlider > div > div > div {
        background: var(--accent) !important;
    }

    /* ─── Toggle ─── */
    .stToggle > div > div > div {
        background: var(--accent) !important;
    }

    /* ─── Chat Messages ─── */
    .chat-message {
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: var(--radius-lg);
        max-width: 85%;
        animation: fadeInUp 0.3s ease-out;
        position: relative;
    }

    .chat-message.user {
        background: var(--accent);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: var(--shadow-md);
    }

    .chat-message.assistant {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        margin-right: auto;
        border: 1px solid var(--border-subtle);
        border-bottom-left-radius: 4px;
        box-shadow: var(--shadow-sm);
    }

    .chat-message .timestamp {
        font-size: 0.7rem;
        opacity: 0.6;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .chat-message .message-actions {
        position: absolute;
        top: -10px;
        right: 10px;
        opacity: 0;
        transition: opacity 0.2s ease;
        display: flex;
        gap: 4px;
    }

    .chat-message:hover .message-actions {
        opacity: 1;
    }

    .message-action-btn {
        background: var(--bg-elevated);
        border: 1px solid var(--border-medium);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.75rem;
        cursor: pointer;
        color: var(--text-secondary);
        transition: all 0.15s ease;
    }

    .message-action-btn:hover {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
    }

    /* ─── Code Blocks ─── */
    pre {
        background: #1a1a2e !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-subtle) !important;
        padding: 1rem !important;
        overflow-x: auto !important;
    }

    code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .code-block-header {
        background: var(--bg-elevated);
        padding: 0.5rem 1rem;
        border-radius: var(--radius-md) var(--radius-md) 0 0;
        border: 1px solid var(--border-subtle);
        border-bottom: none;
        font-size: 0.8rem;
        color: var(--text-muted);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* ─── Status Indicators ─── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.35rem 0.75rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-badge.online {
        background: rgba(34,197,94,0.15);
        color: var(--success);
    }

    .status-badge.offline {
        background: rgba(239,68,68,0.15);
        color: var(--error);
    }

    .status-badge.warning {
        background: rgba(245,158,11,0.15);
        color: var(--warning);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s infinite;
    }

    /* ─── Animations ─── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes typing {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    /* ─── Typing Indicator ─── */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 1rem 1.5rem;
        background: var(--bg-tertiary);
        border-radius: var(--radius-lg);
        border-bottom-left-radius: 4px;
        border: 1px solid var(--border-subtle);
        width: fit-content;
        margin: 0.75rem 0;
    }

    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: var(--accent);
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out both;
    }

    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

    /* ─── Empty State ─── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: var(--text-muted);
    }

    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    .suggestion-item {
        padding: 0.75rem 1rem;
        background: var(--bg-tertiary);
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9rem;
    }

    .suggestion-item:hover {
        background: var(--bg-elevated);
    }

    /* ─── Toast Notifications ─── */
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: var(--radius-md);
        color: white;
        font-weight: 500;
        z-index: 9999;
        animation: slideIn 0.3s ease-out;
        box-shadow: var(--shadow-lg);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .toast.success { background: var(--success); }
    .toast.error { background: var(--error); }
    .toast.warning { background: var(--warning); }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 100px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* ─── Hide Streamlit Branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ─── Chat Input ─── */
    .stChatInput > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ─── Divider ─── */
    hr {
        border-color: var(--border-subtle) !important;
        margin: 1.5rem 0 !important;
    }

    /* ─── Project List ─── */
    .project-item {
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }

    .project-item:hover {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }

    .project-item.active {
        background: var(--accent);
        color: white;
        box-shadow: 0 2px 8px var(--accent-glow);
    }

    .project-item .project-icon {
        font-size: 1.1rem;
    }

    /* ─── Header ─── */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border-subtle);
        position: sticky;
        top: 0;
        z-index: 100;
        backdrop-filter: blur(10px);
    }

    .app-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .app-title .logo {
        font-size: 1.5rem;
    }

    /* ─── Settings Panel ─── */
    .settings-group {
        background: var(--bg-tertiary);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        margin: 1rem 0;
        border: 1px solid var(--border-subtle);
    }

    .settings-group-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    /* ─── Tooltip ─── */
    .tooltip {
        position: relative;
    }

    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem 0.75rem;
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: 0.75rem;
        border-radius: var(--radius-sm);
        white-space: nowrap;
        border: 1px solid var(--border-medium);
        box-shadow: var(--shadow-md);
        z-index: 1000;
    }
</style>
"""


# ───────────────────────── Exceptions ─────────────────────────
class GroqStreamError(Exception):
    pass


# ───────────────────────── Helpers ─────────────────────────
def get_groq_key() -> Optional[str]:
    try:
        return st.secrets.get("GROQ_API_KEY") or None
    except (FileNotFoundError, KeyError):
        return None


def groq_stream(messages, model, temperature=0.7, max_tokens=1024, system=""):
    key = get_groq_key()
    if not key:
        raise GroqStreamError("No GROQ_API_KEY found. Add it to Streamlit secrets (.streamlit/secrets.toml).")
    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": payload_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise GroqStreamError(f"Could not reach Groq: {exc}") from exc
    if not response.ok:
        try:
            error = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            error = response.text or f"HTTP {response.status_code}"
        raise GroqStreamError(f"Groq error ({response.status_code}): {error}")
    received_content = False
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        except (IndexError, TypeError, ValueError):
            continue
        if token:
            received_content = True
            yield token
    if not received_content:
        raise GroqStreamError(f"Model '{model}' returned no text. Check the model name and your Groq access.")


def format_timestamp(ts=None):
    return ts or datetime.now().strftime("%H:%M")


def truncate_text(text, max_len=60):
    return text[:max_len] + "..." if len(text) > max_len else text


def safe_html(text: str) -> str:
    """Escape user/model content before embedding in raw HTML markdown blocks."""
    return html.escape(text or "").replace("\n", "<br>")


# ───────────────────────── Session State ─────────────────────────
def init_app_state():
    defaults = {
        "omnix_initialized": True,
        "projects": load_proj(),
        "active_project": "Default",
        "messages": [],
        "sel_model": DEFAULT_GROQ_MODEL,
        "thinking_mode": False,
        "tools_enabled": True,
        "use_sem_mem": True,
        "deep_research_enabled": False,
        "tts_on": False,
        "show_settings": False,
        "sidebar_section": "chat",
        "pending_voice_input": None,
        "toast_message": None,
        "toast_type": "info",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.messages:
        st.session_state.messages = st.session_state.projects.get(
            st.session_state.active_project, {"messages": []}
        ).get("messages", [])


def persist_messages():
    st.session_state.projects[st.session_state.active_project] = {
        "messages": st.session_state.messages
    }
    save_proj(st.session_state.projects)


def show_toast(message, toast_type="info"):
    st.session_state.toast_message = message
    st.session_state.toast_type = toast_type


# ───────────────────────── UI Components ─────────────────────────
def render_sidebar():
    # Keep restored/partial Streamlit sessions usable even if an older session
    # does not contain this newer UI-state key yet.
    sidebar_section = st.session_state.get("sidebar_section", "chat")
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 1rem 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.75rem;">⚡</span>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #f0f0f5;">OmniX AI</div>
                        <div style="font-size: 0.75rem; color: #6e6e80;">v{APP_VERSION}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        has_key = get_groq_key() is not None
        status_class = "online" if has_key else "offline"
        status_text = "Connected" if has_key else "No API Key"
        status_icon = "🟢" if has_key else "🔴"
        st.markdown(
            f"""
            <div class="status-badge {status_class}" style="margin-bottom: 1.5rem;">
                <span class="pulse-dot"></span>
                {status_icon} {status_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='settings-group-title'>Navigation</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💬 Chat",
                use_container_width=True,
                type="primary" if sidebar_section == "chat" else "secondary",
            ):
                st.session_state.sidebar_section = "chat"
                st.rerun()
        with col2:
            if st.button(
                "⚙️ Settings",
                use_container_width=True,
                type="primary" if sidebar_section == "settings" else "secondary",
            ):
                st.session_state.sidebar_section = "settings"
                st.rerun()

        st.divider()

        if sidebar_section == "chat":
            render_chat_sidebar()
        else:
            render_settings_sidebar()


def render_chat_sidebar():
    if st.button("✨ New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        persist_messages()
        show_toast("New conversation started", "success")
        st.rerun()

    st.divider()

    st.markdown("<div class='settings-group-title'>Workspaces</div>", unsafe_allow_html=True)

    project_names = list(st.session_state.projects.keys()) or ["Default"]
    for proj_name in project_names:
        msg_count = len(st.session_state.projects.get(proj_name, {}).get("messages", []))
        is_active = proj_name == st.session_state.active_project
        if st.button(
            f"📁 {proj_name} ({msg_count})",
            key=f"proj_{proj_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_project = proj_name
            st.session_state.messages = st.session_state.projects.get(
                proj_name, {"messages": []}
            ).get("messages", [])
            st.rerun()

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    new_project_name = st.text_input(
        "New workspace name",
        key="new_proj_input",
        placeholder="Enter name...",
        label_visibility="collapsed",
    )
    if st.button("➕ Create Workspace", use_container_width=True) and new_project_name.strip():
        name = new_project_name.strip()
        if name not in st.session_state.projects:
            st.session_state.projects[name] = {"messages": []}
            save_proj(st.session_state.projects)
            show_toast(f"Workspace '{name}' created", "success")
        st.session_state.active_project = name
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("<div class='settings-group-title'>Actions</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            persist_messages()
            show_toast("Conversation cleared", "warning")
            st.rerun()
    with col2:
        if st.button("💾 Export", use_container_width=True, type="secondary"):
            export_data = json.dumps(
                st.session_state.projects.get(st.session_state.active_project, {"messages": []}),
                indent=2,
            )
            st.download_button(
                label="📥 Download",
                data=export_data,
                file_name=f"omnix_{st.session_state.active_project}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )


def render_settings_sidebar():
    st.markdown("<div class='settings-group-title'>Model</div>", unsafe_allow_html=True)

    st.session_state.sel_model = st.text_input(
        "Model ID",
        value=st.session_state.sel_model,
        help="Enter a Groq model available to your account (e.g., llama-3.3-70b-versatile)",
        placeholder="llama-3.3-70b-versatile",
    )

    st.divider()

    st.markdown("<div class='settings-group-title'>Features</div>", unsafe_allow_html=True)

    st.session_state.thinking_mode = st.toggle(
        "🧠 Deep Thinking",
        value=st.session_state.thinking_mode,
        help="Enable step-by-step reasoning for complex problems",
    )

    st.session_state.deep_research_enabled = st.toggle(
        "🔎 Deep Research",
        value=st.session_state.deep_research_enabled,
        help="Perform multi-source web research and synthesize findings before answering",
    )

    st.session_state.tools_enabled = st.toggle(
        "🔧 Tools",
        value=st.session_state.tools_enabled,
        help="Allow AI to use external tools and functions",
    )

    st.session_state.use_sem_mem = st.toggle(
        "💾 Semantic Memory",
        value=st.session_state.use_sem_mem,
        help="Enable long-term memory across conversations",
    )

    st.session_state.tts_on = st.toggle(
        "🔊 Text-to-Speech",
        value=st.session_state.tts_on,
        help="Speak AI responses aloud",
    )

    st.divider()
    st.markdown("<div class='settings-group-title'>About</div>", unsafe_allow_html=True)
    st.info(f"OmniX AI v{APP_VERSION}\nLocal + cloud hybrid assistant\nMotto: 'Let my actions speak.'")


def render_header():
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">
                <span class="logo">⚡</span>
                <div>
                    <div>OmniX AI</div>
                    <div style="font-size: 0.75rem; color: #6e6e80; font-weight: 400;">
                        {html.escape(st.session_state.active_project)} · {html.escape(st.session_state.sel_model)}
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 0.75rem; align-items: center;">
                <span style="font-size: 0.8rem; color: #6e6e80;">
                    {len(st.session_state.messages)} messages
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(msg, idx):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    ts = msg.get("ts", format_timestamp())
    is_user = role == "user"
    msg_class = "user" if is_user else "assistant"
    avatar = "👤" if is_user else "⚡"
    st.markdown(
        f"""
        <div class="chat-message {msg_class}">
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <div style="font-size: 1.25rem; flex-shrink: 0; margin-top: 2px;">{avatar}</div>
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 0.95rem; line-height: 1.6; word-wrap: break-word;">
                        {safe_html(content)}
                    </div>
                    <div class="timestamp">{html.escape(ts)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    suggestions = [
        "Explain quantum computing in simple terms",
        "Help me write a professional email",
        "Summarize the key points of machine learning",
        "Write a short sci-fi story",
        "Debug this error: IndexError list index out of range",
    ]
    items_html = "".join(
        f'<div class="suggestion-item">💡 {html.escape(s)}</div>' for s in suggestions
    )
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">⚡</div>
            <h3 style="color: #f0f0f5; margin-bottom: 0.5rem; font-weight: 600;">OmniX AI</h3>
            <p style="margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto;">
                Your intelligent assistant. Deep thinking, deep research, and tools — all in one.
            </p>
            <div style="display: flex; flex-direction: column; gap: 0.5rem; max-width: 500px; margin: 0 auto;">
                {items_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_toast():
    if st.session_state.get("toast_message"):
        toast_type = st.session_state.get("toast_type", "info")
        icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
        icon = icons.get(toast_type, "ℹ️")
        st.markdown(
            f"""
            <div class="toast {toast_type}">
                {icon} {html.escape(st.session_state.toast_message)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.toast_message = None


def render_assistant_bubble(placeholder, text, streaming=False, error=False):
    """Render (or update) the assistant chat bubble in-place."""
    cursor = "▌" if streaming else ""
    border_style = ' style="border-color: #ef4444;"' if error else ""
    text_style = "color: #ef4444;" if error else ""
    ts_html = "" if streaming else f'<div class="timestamp">{datetime.now().strftime("%H:%M")}</div>'
    placeholder.markdown(
        f"""
        <div class="chat-message assistant"{border_style}>
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <div style="font-size: 1.25rem; flex-shrink: 0; margin-top: 2px;">⚡</div>
                <div style="flex: 1;">
                    <div style="font-size: 0.95rem; line-height: 1.6; {text_style}">{safe_html(text)}{cursor}</div>
                    {ts_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────── Main App ─────────────────────────
def main():
    st.set_page_config(
        page_title="OmniX AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "omnix_initialized" not in st.session_state:
        init_state()
    # This function fills in any keys absent from restored or upgraded sessions.
    init_app_state()

    render_toast()
    render_sidebar()
    render_header()

    st.markdown("<div style='padding: 1rem 2rem; max-width: 900px; margin: 0 auto;'>", unsafe_allow_html=True)

    if not st.session_state.messages:
        render_empty_state()
    else:
        for idx, message in enumerate(st.session_state.messages):
            render_message(message, idx)

    # Voice input button
    _, mic_col = st.columns([10, 1])
    with mic_col:
        if HAS_STT and st.button("🎙️", help="Speak your message"):
            with st.spinner("Listening..."):
                result = voice_listen(timeout=5)
            if result.get("ok") and result.get("text"):
                st.session_state.pending_voice_input = result["text"]
                show_toast("Voice input captured", "success")
            else:
                show_toast(result.get("error", "No speech detected"), "warning")
            st.rerun()

    user_text = st.chat_input("Send a message...")
    if not user_text and st.session_state.get("pending_voice_input"):
        user_text = st.session_state.pop("pending_voice_input")

    if user_text:
        # Add user message
        st.session_state.messages.append(
            {
                "id": str(uuid.uuid4())[:8],
                "role": "user",
                "content": user_text,
                "ts": datetime.now().strftime("%H:%M"),
            }
        )
        persist_messages()

        # Build system prompt (no modes, unified)
        system = build_sys(
            st.session_state.thinking_mode,
            st.session_state.tools_enabled,
            st.session_state.active_project,
            user_text,
            st.session_state,
        )

        # Deep research: if enabled, do it before answering
        if st.session_state.deep_research_enabled:
            with st.spinner("🔎 Performing deep research..."):
                research_brief = deep_research(user_text, st.session_state.sel_model)
                if research_brief and not research_brief.startswith("⚠️"):
                    system += "\n\n---\n\n" + research_brief

        context_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-SHORT_CTX:]
        ]

        full_response = ""
        had_error = False

        with st.chat_message("assistant", avatar="⚡"):
            placeholder = st.empty()
            try:
                placeholder.markdown(
                    """
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                time.sleep(0.3)

                for token in groq_stream(
                    context_messages,
                    st.session_state.sel_model,
                    0.7,   # fixed temperature
                    1024,  # fixed max tokens
                    system,
                ):
                    full_response += token
                    render_assistant_bubble(placeholder, full_response, streaming=True)

                render_assistant_bubble(placeholder, full_response, streaming=False)

            except GroqStreamError as exc:
                had_error = True
                full_response = f"❌ {exc}"
                render_assistant_bubble(placeholder, full_response, error=True)
                show_toast(str(exc), "error")
            except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the user
                had_error = True
                full_response = f"❌ Error: {exc}"
                render_assistant_bubble(placeholder, full_response, error=True)
                show_toast(str(exc), "error")

        if full_response and not had_error:
            st.session_state.messages.append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "role": "assistant",
                    "content": full_response,
                    "ts": datetime.now().strftime("%H:%M"),
                }
            )
            persist_messages()
            if st.session_state.tts_on:
                speak(full_response, st.session_state)

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
