"""
app.py - Streamlit web interface for Orbit (Solar System and space missions chatbot).

Run locally:   streamlit run app.py

All texts shown to the user (title, capabilities, limitations, disclaimer, example prompts) are read from the
`ui` section of config.yaml, in Spanish and English, so they can be edited without touching this code.
The chatbot itself (RAG pipeline, memory, safety rules, logging) lives in orbit_core.py.
"""
import os
import json
import base64
import time
import datetime
import threading

import streamlit as st
from PIL import Image

from orbit_core import load_config, load_chatbot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")      # logo, avatars, favicon and style.css (the background is in static/)


def asset_path(name: str):
    """Full path of a file in assets/, or None if it does not exist (the app then falls back to plain styling)."""
    path = os.path.join(ASSETS_DIR, name)
    return path if os.path.exists(path) else None


@st.cache_resource
def load_image(name: str, fallback):
    """Load an illustration once; if the file is missing, use the fallback (an emoji)."""
    path = asset_path(name)
    return Image.open(path) if path else fallback


@st.cache_resource
def background_data_uri() -> str:
    """The background image as a data URI, so it does not depend on the server's static-file feature
    (some hosting platforms do not serve it). Computed once and cached."""
    path = os.path.join(BASE_DIR, "static", "bg.webp")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/webp;base64," + base64.b64encode(f.read()).decode("ascii")


def inject_css() -> None:
    """Apply the visual theme (assets/style.css) to the page."""
    path = asset_path("style.css")
    if path:
        with open(path, encoding="utf-8") as f:
            css = "\n".join(line for line in f.read().splitlines() if line.strip())
        uri = background_data_uri()
        if uri:
            css = css.replace('url("/app/static/bg.webp")', f'url("{uri}")')
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_config() -> dict:
    return load_config(CONFIG_PATH)


@st.cache_resource(show_spinner="Loading Orbit (models and knowledge base)... the first start can take a few minutes.")
def get_bot():
    cfg = get_config()
    bot = load_chatbot(cfg)
    lock = threading.Lock()           
    return bot, lock


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def save_feedback(index: int) -> None:
    """Callback of the thumbs widget: store the rating (anonymous) in logs/feedback.jsonl."""
    value = st.session_state.get(f"fb_{index}")
    if value is None:
        return
    bot, _ = get_bot()
    meta = st.session_state.messages[index].get("meta", {})
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "session_id": st.session_state.session.user_id,         # random id, not a person
        "message_index": index,
        "rating": "up" if value == 1 else "down",
        "outcome": meta.get("outcome"),
        "question": bot.logger.mask(meta.get("question", "")),   # e-mails / phone numbers are masked
    }
    path = os.path.join(get_config()["paths"]["logs_dir"], "feedback.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    ui_texts = get_config()["ui"]
    texts = ui_texts.get(st.session_state.get("lang", ui_texts.get("default_language", "es")), {})
    st.toast(texts.get("feedback_thanks", "Thanks for your feedback!"), icon="✨")     # visible confirmation


def render_message(index: int, msg: dict, T: dict, show_details: bool) -> None:
    """Draw one chat message. User and Orbit have different avatars and alignment (st.chat_message)."""
    avatar = load_image("avatar_user.png", "🧑") if msg["role"] == "user" else load_image("avatar_orbit.png", "🪐")
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            if show_details and meta:
                sim = meta.get("top_similarity")
                st.caption(f"outcome: {meta.get('outcome')} · mode: {meta.get('mode')} · "
                           f"similarity: {sim if sim is not None else '-'} · {meta.get('seconds', 0):.0f}s")
            if meta.get("outcome") == "answered" and hasattr(st, "feedback"):
                st.feedback("thumbs", key=f"fb_{index}", on_change=save_feedback, args=(index,))


def reset_conversation() -> None:
    st.session_state.pop("session", None)
    st.session_state.messages = []


# --------------------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------------------
cfg = get_config()
ui = cfg["ui"]

st.set_page_config(page_title="Orbit", page_icon=load_image("icon.png", ui["page_icon"]), layout="centered")
inject_css()
if asset_path("logo.png"):
    st.logo(asset_path("logo.png"), size="large", icon_image=asset_path("icon.png"))

# ---- sidebar: language, guidance (capabilities / limitations), example prompts, feedback ----
languages = [code for code in ("es", "en") if code in ui]
default_index = languages.index(ui.get("default_language", "es")) if ui.get("default_language", "es") in languages else 0
with st.sidebar:
    lang = st.radio("Language / Idioma", languages, index=default_index, key="lang",
                    format_func=lambda code: ui[code]["language_name"], horizontal=True)
T = ui[lang]

with st.sidebar:
    st.header(T["about_header"])
    with st.expander(T["capabilities_header"], expanded=False):
        st.markdown("\n".join(f"- {item}" for item in T["capabilities"]))
    with st.expander(T["limitations_header"], expanded=False):
        st.markdown("\n".join(f"- {item}" for item in T["limitations"]))
    with st.expander(T["privacy_header"], expanded=False):
        st.markdown(T["privacy"])

    st.subheader(T["examples_header"])
    clicked_example = None
    for i, example in enumerate(T["examples"]):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            clicked_example = example

    st.divider()
    st.button(T["new_chat"], on_click=reset_conversation, use_container_width=True)
    st.subheader(T["feedback_header"])
    st.caption(T["feedback_help"])
    form_url = (ui.get("feedback_form_url") or "").strip()
    if form_url:
        if hasattr(st, "link_button"):
            st.link_button(T["feedback_button"], form_url, use_container_width=True)
        else:
            st.markdown(f"[{T['feedback_button']}]({form_url})")
    show_details = st.checkbox(T["technical_toggle"], value=bool(ui.get("show_technical_details", False)))

# ---- main area ----
with st.container(key="hero"):
    st.title(T["title"])
    st.caption(T["tagline"])
st.info(T["disclaimer"])                      # visible warning about possible errors / hallucinations

bot, model_lock = get_bot()
if "session" not in st.session_state:
    st.session_state.session = bot.new_session()      # one session (and memory) per browser tab
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message (shown only while the conversation is empty; it is not part of the model's memory)
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=load_image("avatar_orbit.png", "🪐")):
        st.markdown(T["welcome"])

# Previous turns of the conversation
for i, message in enumerate(st.session_state.messages):
    render_message(i, message, T, show_details)

# New question: typed by the user or chosen from the examples
prompt = st.chat_input(T["input_placeholder"]) or clicked_example
if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(len(st.session_state.messages) - 1, user_msg, T, show_details)

    with st.chat_message("assistant", avatar=load_image("avatar_orbit.png", "🪐")):
        try:
            with st.spinner(T["thinking"]):
                started = time.time()
                with model_lock:
                    result = bot.ask(st.session_state.session, prompt)
                seconds = time.time() - started
            reply = {"role": "assistant", "content": result["answer"],
                     "meta": {"outcome": result["outcome"], "mode": result["mode"], "seconds": seconds,
                              "top_similarity": result["top_similarity"], "question": prompt}}
        except Exception as error:                        # never show a stack trace to the user
            print(f"[Orbit] error while answering: {error!r}")
            reply = {"role": "assistant", "content": T["error"], "meta": {"outcome": "error", "question": prompt}}
    st.session_state.messages.append(reply)
    st.rerun()                                            # redraw the history, including the feedback widget
