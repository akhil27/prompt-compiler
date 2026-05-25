import streamlit as st
import streamlit.components.v1 as components
import os
import json
from pathlib import Path
from dotenv import load_dotenv, set_key

import llm
from compiler import compile_prompt
from utils import save_output, parse_sections, extract_final_prompt
from prompts import QUICK_EXAMPLES
from scorer import score_prompt

load_dotenv()

_ENV_PATH = Path(".env")
_TEMPERATURE = 0.3

_MODEL_META = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "short": "Qwen 2.5 Coder 7B",
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "short": "Qwen 3 4B",
    },
    "google/gemma-3n-E4B-it": {
        "short": "Gemma 3n E4B",
    },
}


def _looks_like_auth_error(msg: str) -> bool:
    """True when an error message indicates an HF_TOKEN / auth problem."""
    low = msg.lower()
    return (
        "401" in msg
        or "403" in msg
        or "unauthorized" in low
        or "forbidden" in low
        or "invalid token" in low
        or "hf_token" in low
        or "token rejected" in low
    )





st.set_page_config(
    page_title="Prompt Compiler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #080c14 !important;
    color: #e5e7eb !important;
    color-scheme: dark !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

iframe {
    background-color: #1a1a1a !important;
    background: #1a1a1a !important;
    border: none !important;
    border-radius: 6px !important;
    color-scheme: dark !important;
}

div[data-testid="stHtml"] {
    background-color: transparent !important;
    background: transparent !important;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
.stDeployButton,
.stAppDeployButton,
.viewerBadge_container__1QSob,
.styles_viewerBadge__1yB5_,
.viewerBadge_link__1S137,
.viewerBadge_text__1JaDK { display: none !important; }

#MainMenu { visibility: hidden !important; display: none !important; }
footer    { visibility: hidden !important; display: none !important; }

header[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}

/* ── Typography ── */
.pc-title {
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 55%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0;
}

.pc-subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 6px;
    letter-spacing: 0.01em;
}

/* ── Divider ── */
.pc-divider {
    border: none;
    border-top: 1px solid rgba(55,65,81,0.6);
    margin: 16px 0 20px 0;
}

/* ── Section label ── */
.pc-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Textarea ── */
textarea {
    background-color: #0f1623 !important;
    color: #f3f4f6 !important;
    border: 1px solid rgba(55,65,81,0.8) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s ease !important;
    resize: vertical !important;
}
textarea:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.08) !important;
    outline: none !important;
}

/* ── Primary compile button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.45) !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Secondary buttons (chips, settings) ── */
[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(17,24,39,0.6) !important;
    color: #9ca3af !important;
    border: 1px solid rgba(55,65,81,0.7) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.18s ease !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.4) !important;
    color: #c4b5fd !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    transition: color 0.18s !important;
}
button[data-baseweb="tab"]:hover {
    color: #c4b5fd !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a855f7 !important;
}

/* ── Code blocks ── */
[data-testid="stCodeBlock"] {
    border-radius: 10px !important;
    border: 1px solid rgba(55,65,81,0.5) !important;
    margin-top: 0 !important;
}
[data-testid="stCodeBlock"] pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
}

/* ── Info / error / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* ── Model selector pill ── */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 5px 13px;
    border-radius: 20px;
    letter-spacing: 0.03em;
    cursor: pointer;
    border: 1px solid rgba(124,58,237,0.35);
    background: rgba(124,58,237,0.12);
    color: #a78bfa;
    transition: all 0.18s ease;
    user-select: none;
}
.model-pill:hover {
    background: rgba(124,58,237,0.22);
    border-color: rgba(139,92,246,0.6);
}
.model-pill-fallback {
    border-color: rgba(234,179,8,0.35);
    background: rgba(234,179,8,0.10);
    color: #fbbf24;
}
.model-pill-fallback:hover {
    background: rgba(234,179,8,0.20);
    border-color: rgba(234,179,8,0.55);
}

/* ── Popover panel (model dropdown) ── */
[data-testid="stPopover"] > div {
    background: #111827 !important;
    border: 1px solid rgba(55,65,81,0.8) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
    padding: 6px 0 !important;
    min-width: 280px !important;
}

/* radio inside popover */
[data-testid="stPopover"] [data-testid="stRadio"] label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    color: #d1d5db !important;
    cursor: pointer !important;
    padding: 8px 14px !important;
    border-radius: 7px !important;
    transition: background 0.15s !important;
}
[data-testid="stPopover"] [data-testid="stRadio"] label:hover {
    background: rgba(124,58,237,0.1) !important;
}
[data-testid="stPopover"] [data-testid="stRadio"] [aria-checked="true"] + div label {
    color: #a78bfa !important;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    text-align: center;
    padding: 32px 24px;
    background: rgba(15,22,35,0.5);
    border: 1px dashed rgba(55,65,81,0.5);
    border-radius: 12px;
    color: #4b5563;
}
.empty-state-icon {
    font-size: 2.8rem;
    margin-bottom: 14px;
    opacity: 0.5;
}
.empty-state-text {
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 280px;
}

/* ── Scrollable output content ── */
.scrollable-md {
    max-height: 460px;
    overflow-y: auto;
    padding: 16px 20px;
    border: 1px solid rgba(55,65,81,0.5);
    border-radius: 10px;
    background: #0f1623;
}
.scrollable-md::-webkit-scrollbar { width: 4px; }
.scrollable-md::-webkit-scrollbar-track { background: transparent; }
.scrollable-md::-webkit-scrollbar-thumb {
    background: rgba(124,58,237,0.3);
    border-radius: 4px;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #a855f7 !important;
}
</style>
""", unsafe_allow_html=True)


@st.dialog("API Configuration")
def open_settings_modal():
    st.markdown("Configure your **Hugging Face token** to enable free-tier inference.")
    st.caption(
        "Create a fine-grained token at "
        "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
        "with **Make calls to Inference Providers** enabled."
    )
    current_token = os.getenv("HF_TOKEN", "") or os.getenv("HF_API_KEY", "")
    token_input = st.text_input(
        "HF_TOKEN",
        value=current_token,
        type="password",
        placeholder="hf_...",
    )
    if st.button("Save Token", use_container_width=True, type="primary"):
        token_value = token_input.strip()
        if not token_value:
            st.warning("Token is empty — paste a token starting with `hf_` first.")
            return
        os.environ["HF_TOKEN"] = token_value
        llm._client = None
        try:
            _ENV_PATH.touch(exist_ok=True)
            set_key(str(_ENV_PATH), "HF_TOKEN", token_value, quote_mode="never")
            st.success("Saved! The new token will be used on the next compile.")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving: {e}")


if "compiled_prompt" not in st.session_state:
    st.session_state.compiled_prompt = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "_preset" not in st.session_state:
    st.session_state._preset = ""

if "_model_used" not in st.session_state:
    st.session_state._model_used = ""

if "_selected_model" not in st.session_state:
    st.session_state._selected_model = llm.DEFAULT_MODEL

if st.session_state._preset:
    st.session_state.input_text = st.session_state._preset
    st.session_state._preset = ""


col_title, col_settings = st.columns([6, 1])
with col_title:
    st.markdown(
        '<p class="pc-title">⚡ prompt-compiler</p>'
        '<p class="pc-subtitle">Turn vague ideas into production-ready LLM instructions — free, instant, no account required.</p>',
        unsafe_allow_html=True,
    )
with col_settings:
    st.write("")
    st.write("")
    if st.button("⚙️ Token", use_container_width=True):
        open_settings_modal()

st.markdown('<hr class="pc-divider">', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p class="pc-label">Your prompt</p>', unsafe_allow_html=True)

    raw_prompt = st.text_area(
        label="Input prompt",
        key="input_text",
        placeholder="e.g. make landing page",
        height=180,
        label_visibility="collapsed",
    )

    st.markdown('<p class="pc-label" style="margin-top:14px;">Quick examples</p>', unsafe_allow_html=True)

    cols = st.columns(len(QUICK_EXAMPLES))
    for col, (label, value) in zip(cols, QUICK_EXAMPLES):
        with col:
            if st.button(label, use_container_width=True, key=f"chip_{value[:10]}"):
                st.session_state._preset = value
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    compile_btn = st.button("Compile Prompt ⚡", type="primary", use_container_width=True)

    if compile_btn:
        if not raw_prompt.strip():
            st.error("Please enter a prompt before compiling.")
        else:
            with st.spinner("Compiling your prompt…"):
                try:
                    compiled = compile_prompt(
                        raw_prompt,
                        model=st.session_state._selected_model,
                        temperature=_TEMPERATURE,
                    )
                    st.session_state.compiled_prompt = compiled
                    st.session_state._model_used = llm.last_used_model
                    save_output(raw_prompt, compiled, "json")
                    save_output(raw_prompt, compiled, "md")
                    save_output(raw_prompt, compiled, "txt")
                    st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    if _looks_like_auth_error(err_msg):
                        st.error(
                            "🔑 **Token problem.** Click ⚙️ Token and paste a valid `HF_TOKEN` "
                            "(fine-grained, **Make calls to Inference Providers** enabled). "
                            "Get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).\n\n"
                            f"Details: `{err_msg}`"
                        )
                    elif "rate-limited or unavailable" in err_msg.lower() or "rate limit" in err_msg.lower():
                        st.warning(
                            "⏳ All free-tier models are currently busy. Wait 30 seconds and try again. "
                            "No action needed — the app will automatically retry the next available model."
                        )
                    else:
                        st.error(f"**Compilation failed:** {err_msg}")

    # ── Model selector pill + popover ─────────────────────────────────────────
    st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)

    active_id   = st.session_state._selected_model
    actual_used = st.session_state._model_used
    meta        = _MODEL_META.get(active_id, {"short": active_id.split("/")[-1]})

    is_fallback_active = bool(actual_used and actual_used != active_id)
    pill_class = "model-pill-fallback" if is_fallback_active else "model-pill"
    pill_icon  = "⚠️" if is_fallback_active else "⚡"
    pill_label = f"{pill_icon} {meta['short']}"
    if is_fallback_active:
        fallback_meta = _MODEL_META.get(actual_used, {"short": actual_used.split("/")[-1]})
        pill_label = f"⚠️ Fallback: {fallback_meta['short']}"

    radio_labels = [m["short"] for m in _MODEL_META.values()]
    model_ids = list(_MODEL_META.keys())
    current_index = model_ids.index(active_id) if active_id in model_ids else 0

    with st.popover(pill_label):
        st.markdown(
            "<p style='font-size:0.75rem;font-weight:600;color:#6b7280;"
            "letter-spacing:0.07em;text-transform:uppercase;margin:0 0 8px 0;"
            "padding:0 4px;'>Select model</p>",
            unsafe_allow_html=True,
        )
        chosen_label = st.radio(
            "model_radio",
            options=radio_labels,
            index=current_index,
            label_visibility="collapsed",
            key="model_radio_widget",
        )
        chosen_index = radio_labels.index(chosen_label)
        if model_ids[chosen_index] != st.session_state._selected_model:
            st.session_state._selected_model = model_ids[chosen_index]
            st.session_state._model_used = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


with col_right:
    st.markdown('<p class="pc-label">Compiled output</p>', unsafe_allow_html=True)

    if st.session_state.compiled_prompt:
        tab_structured, tab_agent, tab_json, tab_score = st.tabs(
            ["📝 Structured", "🤖 Agent Prompt", "{ } JSON", "📊 Score"]
        )

        compiled_text = st.session_state.compiled_prompt

        with tab_structured:
            st.markdown(
                f'<div class="scrollable-md">{compiled_text}</div>',
                unsafe_allow_html=True,
            )

        with tab_agent:
            final_prompt = extract_final_prompt(compiled_text)
            st.code(final_prompt, language="markdown")

        with tab_json:
            sections_dict = parse_sections(compiled_text)
            st.json(sections_dict)

        with tab_score:
            score_data = score_prompt(compiled_text)
            total = score_data["score"]

            if total >= 85:
                rating_text = "Optimized Prompt"
                rating_desc = "Excellent structure, clear boundaries, and solid output formatting. Ready to run."
                color = "#10b981"  # Emerald
            elif total >= 70:
                rating_text = "Strong Prompt"
                rating_desc = "Good specificity. Add negative constraints or a detailed persona to optimize further."
                color = "#3b82f6"  # Blue
            elif total >= 50:
                rating_text = "Needs Improvement"
                rating_desc = "Basic structure is present, but lacks formatting rules, role assignment, or detail."
                color = "#f59e0b"  # Amber
            else:
                rating_text = "Weak / Unstructured"
                rating_desc = "Lacks critical context, role definition, formatting, or sufficient detail."
                color = "#ef4444"  # Red

            # Circular/Card Score Display
            st.markdown(
                f'<div style="background: rgba(17, 24, 39, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">'
                f'  <div style="background: radial-gradient(circle, {color}1a 0%, rgba(0, 0, 0, 0) 70%); border: 2.5px solid {color}; border-radius: 50%; width: 72px; height: 72px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 0 12px {color}30;">'
                f'    <span style="font-size: 1.8rem; font-weight: 800; color: {color};">{total}</span>'
                f'  </div>'
                f'  <div>'
                f'    <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #ffffff;">{rating_text}</h4>'
                f'    <p style="margin: 4px 0 0 0; font-size: 0.82rem; color: #9ca3af; line-height: 1.4;">{rating_desc}</p>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Criteria Bars
            st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#d1d5db; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.05em;'>📊 Evaluation Criteria</p>", unsafe_allow_html=True)
            for dim, val in score_data["criteria_scores"].items():
                pct = int((val / 20) * 100)
                st.markdown(
                    f'<div style="margin-bottom: 12px;">'
                    f'  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
                    f'    <span style="font-size: 0.8rem; font-weight: 600; color: #9ca3af; text-transform: capitalize;">{dim}</span>'
                    f'    <span style="font-size: 0.8rem; font-weight: 700; color: #a78bfa;">{val}/20</span>'
                    f'  </div>'
                    f'  <div style="background: #111827; border: 1px solid rgba(255,255,255,0.05); border-radius: 9999px; height: 8px; overflow: hidden;">'
                    f'    <div style="background: linear-gradient(90deg, #7c3aed 0%, #c084fc 100%); width: {pct}%; height: 100%; border-radius: 9999px; box-shadow: 0 0 6px rgba(124, 58, 237, 0.3);"></div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Check weaknesses
            valid_weaknesses = [w for w in score_data.get("weaknesses", []) if w not in ("None. The prompt is highly optimized!", "Prompt is empty.")]
            if valid_weaknesses:
                st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#ef4444; margin-top:20px; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;'>⚠️ Areas to Improve</p>", unsafe_allow_html=True)
                for w in valid_weaknesses:
                    st.markdown(
                        f'<div style="background: rgba(239, 68, 68, 0.05); border-left: 3px solid #ef4444; border-radius: 0 6px 6px 0; padding: 10px 14px; margin-bottom: 8px; font-size: 0.85rem; color: #fecaca; line-height: 1.4;">{w}</div>',
                        unsafe_allow_html=True
                    )

            # Check suggestions
            valid_suggestions = [s for s in score_data.get("suggestions", []) if s not in ("Your prompt is structural and precise. Ready to run.", "Please enter a prompt to begin scoring.")]
            if valid_suggestions:
                st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#a855f7; margin-top:20px; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;'>💡 Recommendations</p>", unsafe_allow_html=True)
                for s in valid_suggestions:
                    st.markdown(
                        f'<div style="background: rgba(168, 85, 247, 0.05); border-left: 3px solid #a855f7; border-radius: 0 6px 6px 0; padding: 10px 14px; margin-bottom: 8px; font-size: 0.85rem; color: #e9d5ff; line-height: 1.4;">{s}</div>',
                        unsafe_allow_html=True
                    )

            # Check missing components
            valid_missing = [m for m in score_data.get("missing_components", []) if m not in ("None.", "Everything is missing.")]
            if valid_missing:
                st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#9ca3af; margin-top:20px; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;'>🔍 Missing Elements</p>", unsafe_allow_html=True)
                missing_badges = "".join([f'<span style="display:inline-block; background: rgba(55,65,81,0.5); border: 1px dashed rgba(156,163,175,0.4); color: #d1d5db; font-size: 0.78rem; font-weight: 500; border-radius: 6px; padding: 4px 10px; margin: 4px 6px 4px 0;">{m}</span>' for m in valid_missing])
                st.markdown(f'<div style="margin-bottom: 12px;">{missing_badges}</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">⚡</div>
                <div class="empty-state-text">
                    Your compiled prompt will appear here.<br><br>
                    Enter a prompt on the left — or click a quick example — then hit <strong>Compile Prompt</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
