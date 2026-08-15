#!/usr/bin/env python3
"""OmniX AI Streamlit UI using Groq for cloud inference.

Run with:
    streamlit run omnix_streamlit_groq.py

Set ``GROQ_API_KEY`` in ``.streamlit/secrets.toml`` (never hard-code it).
"""
import json
from datetime import datetime

import requests
import streamlit as st

from omnixcore import (
    HAS_STT,
    OMNIX_MODES,
    SHORT_CTX,
    build_sys,
    init_state,
    load_proj,
    save_proj,
    speak,
    voice_listen,
)


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqStreamError(Exception):
    """A user-displayable error returned while calling Groq."""


def get_groq_key():
    """Return the configured Groq key, without exposing it in the UI."""
    try:
        return st.secrets.get("GROQ_API_KEY") or None
    except (FileNotFoundError, KeyError):
        return None


def groq_stream(messages, model, temperature=0.7, max_tokens=1024, system=""):
    """Yield response tokens from Groq's OpenAI-compatible streaming endpoint."""
    key = get_groq_key()
    if not key:
        raise GroqStreamError(
            "No GROQ_API_KEY found. Add it to Streamlit secrets "
            "(.streamlit/secrets.toml locally)."
        )

    payload_messages = ([{"role": "system", "content": system}] if system else [])
    payload_messages.extend(messages)
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
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
        raise GroqStreamError(
            f"Model '{model}' returned no text. Check the model name and your Groq access."
        )


st.set_page_config(page_title="OmniX AI", page_icon="⚡", layout="wide")

if "omnix_initialized" not in st.session_state:
    init_state()
    st.session_state.omnix_initialized = True
    st.session_state.projects = load_proj()
    st.session_state.active_project = "Default"
    st.session_state.messages = st.session_state.projects.get(
        "Default", {"messages": []}
    ).get("messages", [])
    st.session_state.sel_model = DEFAULT_GROQ_MODEL
    st.session_state.active_mode = list(OMNIX_MODES.keys())[0]
    st.session_state.thinking_mode = False
    st.session_state.tools_enabled = True
    st.session_state.use_sem_mem = True
    st.session_state.temperature = 0.7
    st.session_state.max_tokens = 1024
    st.session_state.tts_on = False


def persist_messages():
    st.session_state.projects[st.session_state.active_project] = {
        "messages": st.session_state.messages
    }
    save_proj(st.session_state.projects)


with st.sidebar:
    st.markdown("## ⚡ OmniX AI")
    st.markdown(
        "🟢 Groq API key found"
        if get_groq_key()
        else "🔴 No `GROQ_API_KEY` in Streamlit secrets"
    )

    if st.button("✏️ New Chat", use_container_width=True):
        st.session_state.messages = []
        persist_messages()
        st.rerun()

    st.divider()
    project_names = list(st.session_state.projects.keys()) or ["Default"]
    new_project = st.selectbox(
        "Project",
        project_names,
        index=(
            project_names.index(st.session_state.active_project)
            if st.session_state.active_project in project_names
            else 0
        ),
    )
    if new_project != st.session_state.active_project:
        st.session_state.active_project = new_project
        st.session_state.messages = st.session_state.projects.get(
            new_project, {"messages": []}
        ).get("messages", [])
        st.rerun()

    new_project_name = st.text_input("New project name", key="new_proj_input")
    if st.button("➕ Add project") and new_project_name.strip():
        name = new_project_name.strip()
        if name not in st.session_state.projects:
            st.session_state.projects[name] = {"messages": []}
            save_proj(st.session_state.projects)
        st.session_state.active_project = name
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Settings")
    st.session_state.active_mode = st.selectbox(
        "Mode",
        list(OMNIX_MODES.keys()),
        index=list(OMNIX_MODES.keys()).index(st.session_state.active_mode),
    )
    st.session_state.sel_model = st.text_input(
        "Model",
        value=st.session_state.sel_model,
        help="Enter a Groq model available to your account.",
    )
    st.session_state.thinking_mode = st.toggle(
        "🧠 Deep thinking mode", value=st.session_state.thinking_mode
    )
    st.session_state.tools_enabled = st.toggle(
        "🔧 Tools enabled", value=st.session_state.tools_enabled
    )
    st.session_state.use_sem_mem = st.toggle(
        "💾 Use semantic memory", value=st.session_state.use_sem_mem
    )
    st.session_state.tts_on = st.toggle(
        "🔊 Speak replies (TTS)", value=st.session_state.tts_on
    )
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.5, st.session_state.temperature, 0.05
    )
    st.session_state.max_tokens = st.slider(
        "Max tokens", 128, 4096, st.session_state.max_tokens, 64
    )

    st.divider()
    if st.button("🗑️ Purge memory (this project)", type="secondary"):
        st.session_state.messages = []
        persist_messages()
        st.rerun()


st.markdown(f"#### {st.session_state.active_mode} — *{st.session_state.active_project}*")
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role, avatar="👤" if role == "user" else "⚡"):
        st.markdown(message["content"])
        if message.get("ts"):
            st.caption(message["ts"])


_, microphone_column = st.columns([10, 1])
with microphone_column:
    if HAS_STT and st.button("🎙️", help="Speak your message"):
        with st.spinner("Listening..."):
            result = voice_listen(timeout=5)
        if result.get("ok") and result.get("text"):
            st.session_state.pending_voice_input = result["text"]
        else:
            st.warning(result.get("error", "No speech detected."))

user_text = st.chat_input("Send a message...")
if not user_text and st.session_state.get("pending_voice_input"):
    user_text = st.session_state.pop("pending_voice_input")

if user_text:
    st.session_state.messages.append(
        {"role": "user", "content": user_text, "ts": datetime.now().strftime("%H:%M")}
    )
    persist_messages()
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)

    system = build_sys(
        st.session_state.active_mode,
        st.session_state.thinking_mode,
        st.session_state.tools_enabled,
        st.session_state.active_project,
        user_text,
        st.session_state,
    )
    context_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-SHORT_CTX:]
    ]

    with st.chat_message("assistant", avatar="⚡"):
        placeholder = st.empty()
        full_response = ""
        try:
            for token in groq_stream(
                context_messages,
                st.session_state.sel_model,
                st.session_state.temperature,
                st.session_state.max_tokens,
                system,
            ):
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except GroqStreamError as exc:
            full_response = f"❌ {exc}"
            placeholder.markdown(full_response)
        except Exception as exc:
            full_response = f"❌ Error: {exc}"
            placeholder.markdown(full_response)

    if full_response and not full_response.startswith("❌"):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "ts": datetime.now().strftime("%H:%M"),
            }
        )
        persist_messages()
        if st.session_state.tts_on:
            speak(full_response, st.session_state)
