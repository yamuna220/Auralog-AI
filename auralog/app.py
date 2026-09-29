"""
AURAlog AI — Streamlit entrypoint (Enhanced UI Design from ZIP).

Preserves backend logic but redesigns the layout to precisely match
the provided React/Tailwind styling specifications from the user screenshot.
"""

from __future__ import annotations

import logging
import traceback
import math

import streamlit as st

import config
from core.demo_data import DEMO_POSTS
from core.models import SocialMediaPost
from services.analysis_service import BeforeAfterAnalysis, analyze_before_after, run_full_analysis
from services.gemini_service import GeminiService
from state.session import get_audit_session, reset_audit_session

logger = logging.getLogger("auralog.app")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AURAlog AI — Privacy Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Alegreya:ital,wght@0,400;0,500;0,700;1,400;1,700&family=Alegreya+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Base Theme Variables ── */
:root {
  --color-bg: #0d0f14;
  --color-surface: #141720;
  --color-surface-2: #1c2030;
  --color-border: #252a38;
  --color-border-2: #2e3548;
  --color-border-subtle: #1a1e28;
  --color-text: #e8eaf0;
  --color-muted: #6b7280;
  --color-faint: #3d4459;
  --color-accent: #ff4d4d;
  --color-accent-dim: rgba(255, 77, 77, 0.12);
  --color-teal: #00d4b4;
  --color-teal-dim: rgba(0, 212, 180, 0.1);
  --color-amber: #ffa800;
  --color-amber-dim: rgba(255, 168, 0, 0.12);
  
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
  --font-sans: 'Alegreya Sans', ui-sans-serif, system-ui, sans-serif;
  --font-serif: 'Alegreya', Georgia, serif;
}

/* ── Streamlit Overrides ── */
.stApp {
    background-color: var(--color-bg) !important;
    color: var(--color-text) !important;
    font-family: var(--font-sans) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--color-surface) !important;
    border-right: 1px solid var(--color-border) !important;
    min-width: 250px !important;
    max-width: 250px !important;
}

.main .block-container {
    max-width: 1200px !important;
    padding-top: 0px !important;
    padding-bottom: 60px !important;
    margin: 0 auto !important;
}

h1, h2, h3, h4, h5, .serif {
    font-family: var(--font-serif) !important;
    color: var(--color-text);
}

.font-mono { font-family: var(--font-mono); }
.font-sans { font-family: var(--font-sans); }

/* ── Sidebar Elements ── */
.sb-divider {
    border-bottom: 1px solid var(--color-border);
    margin: 0 -1.5rem;
    padding: 20px 1.5rem;
}
.pulse-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 8px;
}
.pulse-anim { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

/* Sidebar Inputs & Buttons */
.sb-input-wrap {
    display: flex;
    align-items: center;
    background: transparent;
    border: 1px solid var(--color-border-2);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 24px;
}
.sb-input-wrap input {
    background: transparent;
    border: none;
    color: var(--color-text);
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 8px 10px;
    outline: none;
    flex: 1;
}
.sb-btn-ghost {
    width: 100%;
    background: transparent;
    border: 1px solid var(--color-border-2);
    color: var(--color-text);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
}
.sb-btn-ghost:hover {
    border-color: var(--color-faint);
    background: rgba(255,255,255,0.05);
}

/* ── Step Header ── */
.step-header {
    margin: 0 -32px 32px -32px;
    padding: 12px 32px;
    background: transparent;
    border-bottom: 1px solid var(--color-border);
    display: flex;
    align-items: center;
    gap: 12px;
}
.step-header span.num {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--color-faint);
    letter-spacing: 2px;
}
.step-header span.label {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--color-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Main Layout Elements ── */
.page-title {
    font-size: 28px;
    font-weight: 500;
    color: var(--color-text);
    margin: 0 0 8px 0;
    letter-spacing: -0.01em;
}
.page-desc {
    font-size: 13px;
    color: var(--color-muted);
    max-width: 600px;
    line-height: 1.6;
    margin-bottom: 32px;
}

/* ── Input Cards ── */
.card-header-post {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-bottom: none;
    border-radius: 12px 12px 0 0;
    margin-bottom: -15px; /* Pull textarea up into it */
    position: relative;
    z-index: 5;
}
.card-post-label {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    color: var(--color-faint);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.card-post-sub {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--color-faint);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Streamlit Textarea Overrides */
div[data-testid="stTextArea"] > div {
    border: none !important;
    background-color: transparent !important;
}
div[data-testid="stTextArea"] textarea {
    background-color: var(--color-bg) !important;
    border-left: 1px solid var(--color-border) !important;
    border-right: 1px solid var(--color-border) !important;
    border-top: none !important;
    border-bottom: none !important;
    border-radius: 0 !important;
    color: var(--color-text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    line-height: 1.6 !important;
    padding: 14px 16px !important;
    margin-top: 15px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--color-border-2) !important;
    box-shadow: inset 0 0 0 1px var(--color-border-2) !important;
}

.card-footer-post {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 4px 16px 14px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: -15px; /* Pull footer up into textarea */
    position: relative;
    z-index: 5;
}

/* ── Buttons ── */
button[kind="primary"] {
    background-color: var(--color-accent) !important;
    color: #fff !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
}
button[kind="primary"]:disabled {
    opacity: 0.4;
}

button[kind="secondary"] {
    background-color: transparent !important;
    color: var(--color-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
}
button[kind="secondary"]:hover {
    border-color: var(--color-border-2) !important;
    color: var(--color-text) !important;
}


/* ── Empty State ── */
.empty-box {
    border: 1px dashed var(--color-border);
    background: var(--color-surface);
    border-radius: 8px;
    padding: 48px;
    text-align: center;
}
.empty-icon {
    font-size: 24px;
    color: var(--color-amber);
    margin-bottom: 12px;
}
.empty-title {
    font-family: var(--font-serif);
    font-size: 20px;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 12px;
}
.empty-desc {
    font-size: 12px;
    color: var(--color-muted);
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Analytical Results ── */
.risk-summary-box {
    background: var(--color-surface);
    border: 1px solid var(--color-border-2);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    align-items: flex-start;
    gap: 20px;
    margin-bottom: 24px;
}
.risk-icon-wrap {
    width: 48px; height: 48px;
    border-radius: 8px;
    background: var(--color-accent-dim);
    display: flex;
    align-items: center; justify-content: center;
    font-size: 20px;
}
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
}
.badge-high { background: var(--color-accent-dim); color: var(--color-accent); }
.badge-medium { background: var(--color-amber-dim); color: var(--color-amber); }
.badge-low { background: var(--color-teal-dim); color: var(--color-teal); }

.risk-badge-dot { width: 6px; height: 6px; border-radius: 50%; }
.dot-high { background: var(--color-accent); }
.dot-medium { background: var(--color-amber); }
.dot-low { background: var(--color-teal); }

/* ── Signal Cards ── */
.signal-card {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
    transition: border-color 0.15s;
}
.signal-card:hover { border-color: var(--color-border-2); }
.signal-icon { font-size: 16px; margin-top: 2px; }
.signal-title {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 2px;
}
.signal-detail {
    font-size: 11px;
    color: var(--color-muted);
    line-height: 1.4;
}

/* ── Cross Post & Flow Notes ── */
.cross-post-box {
    background: var(--color-teal-dim);
    border: 1px solid rgba(0, 212, 180, 0.2);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 24px;
}
.cross-title {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--color-teal);
    margin-bottom: 2px;
}
.cross-detail {
    font-size: 11px;
    color: var(--color-muted);
    line-height: 1.4;
}

/* ── Chain box ── */
.chain-box {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Before / After grid */
.ba-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
}
.ba-col {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session & Dependencies
# ---------------------------------------------------------------------------

def _init_state() -> None:
    for k, v in {
        "posts_text": ["", "", ""],
        "analysis_result": None,
        "inference_graph": None,
        "before_after": None,
        "gemini_service": None,
        "api_key_override": "",
        "analyzed_posts": [],
        "error_message": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def _get_gemini_service() -> GeminiService:
    override = st.session_state.get("api_key_override", "")
    cached: GeminiService | None = st.session_state["gemini_service"]
    if cached is not None and not override:
        return cached
    key = override.strip() or config.GEMINI_API_KEY or None
    svc = GeminiService(api_key=key)
    st.session_state["gemini_service"] = svc
    return svc


def _risk_level_config(level: str):
    lvl = level.lower()
    if "high" in lvl or "critical" in lvl:
        return "high"
    if "low" in lvl:
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Visual Components
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    texts = st.session_state.get("posts_text", ["", "", ""])
    filled_count = sum(1 for t in texts if t.strip())

    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sb-divider" style="padding-top:0">
            <div style="font-family:var(--font-serif); font-size:16px; font-weight:700; letter-spacing:-0.03em; color:var(--color-text); margin-bottom:4px">
                AURA<span style="color:var(--color-accent)">log</span> AI
            </div>
            <div style="font-family:var(--font-mono); font-size:9px; color:var(--color-faint); letter-spacing:1px; text-transform:uppercase">
                Privacy Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Status
        svc = _get_gemini_service()
        if svc.demo_mode:
            st.markdown("""
            <div class="sb-divider">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px">
                    <span class="pulse-dot pulse-anim" style="background:var(--color-teal)"></span>
                    <span style="font-family:var(--font-mono); font-size:11px; font-weight:600; color:var(--color-teal)">Demo Mode</span>
                </div>
                <div style="font-family:var(--font-mono); font-size:10px; color:var(--color-faint)">Synthetic data only</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sb-divider">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px">
                    <span class="pulse-dot pulse-anim" style="background:var(--color-teal)"></span>
                    <span style="font-family:var(--font-mono); font-size:11px; font-weight:600; color:var(--color-teal)">Gemini Active</span>
                </div>
                <div style="font-family:var(--font-mono); font-size:10px; color:var(--color-faint)">Live API processing</div>
            </div>
            """, unsafe_allow_html=True)

        # API Key 
        st.markdown("""
        <div style="padding: 24px 0 12px 0; display:flex; justify-content:space-between; align-items:center">
            <span style="font-family:var(--font-mono); font-size:9px; font-weight:600; letter-spacing:1px; color:var(--color-faint); text-transform:uppercase">Gemini API Key</span>
            <span style="border:1px solid var(--color-border); border-radius:50%; width:14px; height:14px; display:inline-flex; align-items:center; justify-content:center; color:var(--color-faint); font-size:9px;">?</span>
        </div>
        """, unsafe_allow_html=True)
        
        # We simulate the exact Dark input container in Streamlit
        # But we need to use Streamlit's text input. Our custom CSS already styled the stTextInput elsewhere if we wish, or we just leave Streamlit styled.
        # But to be visually exact, we can override sidebar text input:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] div[data-testid="stTextInput"] > div > div {
            background-color: transparent !important;
            border: 1px solid var(--color-border-2) !important;
            border-radius: 4px;
        }
        [data-testid="stSidebar"] div[data-testid="stTextInput"] input {
            color: var(--color-text);
            font-family: var(--font-mono);
            font-size: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        key_input = st.text_input(
            "API Key",
            type="password",
            value="",
            placeholder="System ENV used",
            key="sidebar_api",
            label_visibility="collapsed"
        )
        if key_input and key_input != st.session_state.get("api_key_override", ""):
            st.session_state["api_key_override"] = key_input
            st.session_state["gemini_service"] = None
            st.rerun()

        st.markdown('<div class="sb-divider" style="padding-top:24px"></div>', unsafe_allow_html=True)

        # Posts Ready Indicator
        st.markdown(f"""
        <div style="padding: 0 0 24px 0">
            <div style="font-family:var(--font-mono); font-size:9px; font-weight:600; letter-spacing:1px; color:var(--color-faint); text-transform:uppercase; margin-bottom:12px">
                Posts Ready
            </div>
            <div style="display:flex; gap:6px; margin-bottom:10px">
                <div style="flex:1; height:3px; background:{'var(--color-faint)' if filled_count > 0 else 'var(--color-border-subtle)'}; border-radius:2px"></div>
                <div style="flex:1; height:3px; background:{'var(--color-faint)' if filled_count > 1 else 'var(--color-border-subtle)'}; border-radius:2px"></div>
                <div style="flex:1; height:3px; background:{'var(--color-faint)' if filled_count > 2 else 'var(--color-border-subtle)'}; border-radius:2px"></div>
            </div>
            <div style="font-family:var(--font-mono); font-size:9px; color:var(--color-faint); text-transform:uppercase">
                {filled_count}/3 added
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reset Action
        if st.button("↺ RESET ANALYSIS", use_container_width=True, type="secondary"):
            for k in ["analysis_result", "inference_graph", "before_after", "posts_text", "analyzed_posts", "error_message"]:
                st.session_state[k] = None
            st.session_state["posts_text"] = ["", "", ""]
            st.rerun()

        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:var(--font-mono); font-size:9px; color:var(--color-faint); line-height:1.6">
            Privacy awareness &<br>defensive cybersecurity.<br><br>
            v0.1.0 demo
        </div>
        """, unsafe_allow_html=True)

def _render_step_header(step_num: str, step_label: str) -> None:
    st.markdown(f"""
    <div class="step-header">
        <span class="num">{step_num} |</span>
        <span class="label">{step_label}</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Phase 1: Input Setup
# ---------------------------------------------------------------------------

def _render_input_section() -> list[SocialMediaPost] | None:
    _render_step_header("01", "Analyze")
    
    st.markdown("""
    <div class="page-title serif">Analyze Your Public Posts</div>
    <div class="page-desc">Add up to three public posts. AURAlog identifies privacy signals and shows how they can combine across posts.</div>
    """, unsafe_allow_html=True)
    
    texts = st.session_state.get("posts_text", ["", "", ""])
    while len(texts) < 3: texts.append("")
        
    cols = st.columns(3)
    new_texts = []
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="card-header-post">
                <span class="card-post-label">POST 0{i+1}</span>
                <span class="card-post-sub">Public</span>
            </div>
            """, unsafe_allow_html=True)
            
            t = col.text_area(
                f"Post {i}",
                value=texts[i],
                height=180,
                placeholder=f"Enter public post {i+1}...",
                key=f"post_{i}",
                label_visibility="collapsed"
            )
            new_texts.append(t)
            
            pct = min(len(t) / 500.0, 1.0)
            barColor = 'var(--color-border-2)' if pct == 0 else 'var(--color-accent)' if pct > 0.9 else 'var(--color-amber)' if pct > 0.7 else 'var(--color-teal)'
            
            st.markdown(f"""
            <div class="card-footer-post">
                <div style="flex:1; height:1px; background:var(--color-border); overflow:hidden; margin-right:12px; position:relative">
                    <div style="position:absolute; left:0; top:0; height:100%; width:{pct * 100}%; background:{barColor}; transition:width 0.3s"></div>
                </div>
                <div style="font-family:var(--font-mono); font-size:9px; font-weight:600; color:var(--color-faint); white-space:nowrap">{len(t)}/500</div>
            </div>
            """, unsafe_allow_html=True)

    st.session_state["posts_text"] = new_texts
    posts = [SocialMediaPost(text=t.strip(), order_index=idx) for idx, t in enumerate(new_texts) if t.strip()]

    # Action row
    st.markdown("<div style='padding: 24px 0'></div>", unsafe_allow_html=True)
    
    colA, colB, _ = st.columns([1.5, 2, 5])
    
    with colA:
        if st.button("LOAD DEMO", use_container_width=True, type="secondary", key="demo_btn"):
            st.session_state["posts_text"] = [p.text for p in DEMO_POSTS]
            st.session_state["analysis_result"] = None
            st.session_state["before_after"] = None
            st.rerun()
            
    with colB:
        is_disabled = len(posts) == 0
        if st.button("● ANALYZE EXPOSURE", type="primary", use_container_width=True, disabled=is_disabled):
            gemini = _get_gemini_service()
            with st.spinner("Analyzing exposure..."):
                try:
                    rep, graph = run_full_analysis(posts, gemini)
                    st.session_state["analysis_result"] = rep
                    st.session_state["inference_graph"] = graph
                    st.session_state["analyzed_posts"] = posts
                    st.session_state["before_after"] = None
                    st.session_state["error_message"] = None
                except Exception as e:
                    logger.error("Analysis failed: %s", e)
                    st.session_state["error_message"] = str(e)
            st.rerun()

    # Empty State Display
    if not st.session_state.get("analysis_result"):
        st.markdown("""
        <div class="empty-box" style="margin-top:32px">
            <div class="empty-icon">🔒</div>
            <div class="empty-title">Privacy Audit Ready</div>
            <div class="empty-desc">
                Add your public posts above to begin. The system will analyze location clues, routine patterns, identity information, time patterns, and cross-post relationships.
            </div>
        </div>
        """, unsafe_allow_html=True)

    return posts

# ---------------------------------------------------------------------------
# Phase 2: Analysis Results
# ---------------------------------------------------------------------------

def _render_results() -> None:
    report = st.session_state.get("analysis_result")
    if not report:
        return

    _render_step_header("02", "Explain")
    
    lvl_str = report.risk_level.value if hasattr(report.risk_level, "value") else str(report.risk_level)
    r_cfg = _risk_level_config(lvl_str)
    
    st.markdown(f"""
    <div class="risk-summary-box">
        <div class="risk-icon-wrap">⚠️</div>
        <div style="flex:1">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px">
                <div class="serif" style="font-size:18px; font-weight:700; color:var(--color-text)">Privacy Risk Assessment</div>
                <div class="risk-badge badge-{r_cfg}">
                    <span class="risk-badge-dot dot-{r_cfg}"></span>
                    {lvl_str.upper()} RISK
                </div>
                <div style="font-family:var(--font-mono); font-size:10px; color:var(--color-muted); margin-left:auto">SCORE: {report.total_score:.0f}</div>
            </div>
            <div style="font-size:12px; color:var(--color-muted); line-height:1.6; max-width:600px">
                Across {len(st.session_state["analyzed_posts"])} posts, AURAlog identified <strong style="color:var(--color-text)">{len(report.signals)} privacy signals</strong> and constructed <strong style="color:var(--color-text)">{len(report.inference_chains)} potential inference chains</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Signals Grid
    if report.signals:
        col1, col2 = st.columns(2)
        for i, sig in enumerate(report.signals):
            cat = sig.category.value if hasattr(sig.category, "value") else str(sig.category)
            
            icon = "📍" if "location" in cat.lower() else "🕐" if "time" in cat.lower() else "🧑‍🤝‍🧑" if "identity" in cat.lower() else "✈️"
            
            html = f"""
            <div class="signal-card">
                <span class="signal-icon">{icon}</span>
                <div>
                    <div class="signal-title">{sig.signal or cat.replace('_', ' ').title()}</div>
                    <div class="signal-detail">"{sig.evidence}"</div>
                </div>
            </div>
            """
            (col1 if i % 2 == 0 else col2).markdown(html, unsafe_allow_html=True)

    # Connections
    if report.correlations:
        st.markdown("<br>", unsafe_allow_html=True)
        for corr in report.correlations:
            rel = corr.relationship_type.value if hasattr(corr.relationship_type, "value") else str(corr.relationship_type)
            st.markdown(f"""
            <div class="cross-post-box">
                <span style="font-size:16px; margin-top:2px">🔗</span>
                <div>
                    <div class="cross-title">Cross-Post Link: {rel.replace('_', ' ').upper()}</div>
                    <div class="cross-detail">{corr.description}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Inferences
    if report.inference_chains:
        for ch in report.inference_chains:
            st.markdown(f"""
            <div class="chain-box">
                <div style="font-family:var(--font-mono); font-size:10px; font-weight:600; color:var(--color-faint); margin-bottom:12px; letter-spacing:1px; text-transform:uppercase">
                    Potential Inference Chain ({ch.confidence:.0f}% Conf)
                </div>
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:12px">
            """, unsafe_allow_html=True)
            
            for idx, step in enumerate(ch.steps):
                st.markdown(f"""<span style="background:var(--color-surface-2); border:1px solid var(--color-border-2); padding:4px 8px; border-radius:4px; font-size:11px; color:var(--color-text)">{step}</span>""", unsafe_allow_html=True)
                if idx < len(ch.steps) - 1:
                    st.markdown("""<span style="color:var(--color-faint); font-weight:700">→</span>""", unsafe_allow_html=True)

            st.markdown(f"""
                </div>
                <div style="font-size:15px; color:var(--color-text); font-family:var(--font-serif)">{ch.potential_outcome}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Phase 3: Protect
# ---------------------------------------------------------------------------

def _render_sanitization() -> None:
    report = st.session_state.get("analysis_result")
    if not report: return

    _render_step_header("03", "Protect")
    
    st.markdown("""
    <div style="margin-bottom:24px">
        <div class="page-title serif" style="font-size:24px">Sanitize Your Posts</div>
        <div class="page-desc">Reduce unnecessary identifying details while preserving the original meaning of your content. AURAlog will re-evaluate the sanitized versions safely.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🛡️ Run Privacy Sanitizer", type="primary"):
        gemini = _get_gemini_service()
        with st.spinner("Applying safety filters..."):
            try:
                result = analyze_before_after(st.session_state["analyzed_posts"], gemini)
                st.session_state["before_after"] = result
                st.session_state["error_message"] = None
            except Exception as e:
                st.session_state["error_message"] = str(e)


# ---------------------------------------------------------------------------
# Phase 4: Verify
# ---------------------------------------------------------------------------

def _render_verification() -> None:
    ba: BeforeAfterAnalysis | None = st.session_state.get("before_after")
    if not ba: return

    _render_step_header("04", "Verify")

    sb = ba.risk_before.total_score
    sa = ba.risk_after.total_score
    pct = ba.risk_reduction_percentage
    
    all_safe = all(sp.status != "unsafe" for sp in ba.sanitized_posts)
    
    bg_color = "var(--color-teal-dim)" if all_safe else "var(--color-accent-dim)"
    bd_color = "var(--color-teal)" if all_safe else "var(--color-accent)"
    t_color = "var(--color-teal)" if all_safe else "var(--color-accent)"
    icon = "✓" if all_safe else "⚠️"
    msg = f"PRIVACY RISK REDUCED BY {pct:.0f}%" if all_safe else "SANITIZATION INCOMPLETE"

    st.markdown(f"""
    <div style="background:{bg_color}; border:1px solid {bd_color}; border-radius:12px; padding:24px; display:flex; align-items:center; gap:16px; margin-bottom:32px">
        <div style="font-size:32px; color:{t_color}">{icon}</div>
        <div>
            <div style="font-family:var(--font-mono); font-size:12px; font-weight:700; letter-spacing:2px; color:{t_color}; margin-bottom:4px">{msg}</div>
            <div style="font-size:12px; color:var(--color-text)">
                Risk score dropped from <strong style="color:var(--color-text)">{sb:.0f}</strong> to <strong style="color:var(--color-text)">{sa:.0f}</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, sp in enumerate(ba.sanitized_posts, 1):
        st.markdown(f"""
        <div class="ba-grid">
            <div class="ba-col">
                <div style="font-family:var(--font-mono); font-size:10px; font-weight:600; color:var(--color-muted); letter-spacing:1px; margin-bottom:12px; text-transform:uppercase">
                    Post 0{i} (Original)
                </div>
                <div style="color:var(--color-faint); font-family:var(--font-mono); font-size:11px; line-height:1.6">{sp.original_text}</div>
            </div>
            <div class="ba-col" style="border-color:{bd_color}">
                <div style="font-family:var(--font-mono); font-size:10px; font-weight:600; color:{t_color}; letter-spacing:1px; margin-bottom:12px; text-transform:uppercase">
                    Sanitized
                </div>
                <div style="color:var(--color-text); font-family:var(--font-mono); font-size:11px; line-height:1.6">{sp.sanitized_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()
    
    if st.session_state.get("error_message"):
        st.error(st.session_state["error_message"])

    _render_input_section()
    
    if st.session_state.get("analysis_result"):
        _render_results()
        _render_sanitization()
        _render_verification()


if __name__ == "__main__":
    main()
