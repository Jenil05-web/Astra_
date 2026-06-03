"""
app.py — Astra UI.  Run with:  streamlit run app.py
Fixed version: proper Streamlit-native layout, working optimizer status,
fixed chart, proper spacing, live terminal updates.
"""
from streamlit.runtime.scriptrunner import add_script_run_ctx
import json
import math
import time
import threading
import tempfile
import pandas as pd
import streamlit as st

from core import (
    load_documents, build_rag_pipeline, build_graph,
    run_rag, PROMPT_VARIANTS, DEFAULT_EVAL_DATASET,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASTRA | Self-Optimizing RAG Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Streamlit-compatible, no broken custom layout wrappers
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Geist:wght@400;500;600;900&display=swap');

:root {
  --bg:            #0A0A0A;
  --surface:       #121414;
  --surf-low:      #1a1c1c;
  --surf-mid:      #1e2020;
  --surf-high:     #282a2b;
  --surf-highest:  #333535;
  --outline:       #2e3030;
  --outline-soft:  #8c909f;
  --on-surface:    #e2e2e2;
  --on-surf-var:   #c2c6d6;
  --green:         #4ae176;
  --green-dim:     #00b954;
  --primary:       #adc6ff;
  --primary-cont:  #4d8eff;
  --tertiary:      #ffb786;
  --error:         #ff4d6a;
  --mono:          'JetBrains Mono', monospace;
  --sans:          'Geist', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: var(--sans) !important;
  background: var(--bg) !important;
  color: var(--on-surface) !important;
  -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stDecoration"]  { display: none !important; }
div[data-testid="stToolbar"]     { display: none !important; }

/* ── Main container padding ── */
.block-container {
  padding: 1rem 2rem 2rem 2rem !important;
  max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

/* ── Vertical block gap reset ── */
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
div[data-testid="column"] { padding: 0 8px !important; }

/* ── Typography helpers ── */
.slabel {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .10em !important;
  color: var(--on-surf-var) !important;
  text-transform: uppercase !important;
  margin-bottom: 8px !important;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── Cards ── */
.obs-card {
  background: var(--surf-mid);
  border: 1px solid var(--outline);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 10px;
}
.obs-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .10em;
  color: var(--on-surf-var);
  text-transform: uppercase;
  margin-bottom: 6px;
}
.obs-val {
  font-size: 32px;
  font-weight: 700;
  font-family: var(--sans);
  color: var(--on-surface);
  line-height: 1.1;
}
.obs-val-unit {
  font-size: 14px;
  font-weight: 400;
  color: var(--on-surf-var);
  margin-left: 4px;
}
.obs-sub {
  font-size: 11px;
  color: var(--on-surf-var);
  margin-top: 2px;
}

/* ── Terminal ── */
.terminal-box {
  background: #050505;
  border: 1px solid var(--outline);
  border-radius: 10px;
  overflow: hidden;
  margin-top: 8px;
}
.terminal-header {
  background: var(--surf-mid);
  border-bottom: 1px solid var(--outline);
  padding: 8px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.terminal-title {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .10em;
  color: var(--on-surf-var);
  text-transform: uppercase;
}
.term-body {
  padding: 12px 16px;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.9;
  max-height: 220px;
  overflow-y: auto;
  color: var(--on-surf-var);
}
.trow { display: flex; gap: 10px; align-items: flex-start; }
.tts  { color: #3a3d3d; min-width: 60px; padding-top: 1px; }
.tlvl-ok   { color: #4ae176; min-width: 52px; font-weight: 700; }
.tlvl-sys  { color: #adc6ff; min-width: 52px; }
.tlvl-eval { color: #ffb786; min-width: 52px; }
.tlvl-info { color: #4ae176; min-width: 52px; }
.tlvl-err  { color: #ff4d6a; min-width: 52px; font-weight: 700; }
.tlvl-wait { color: #444; min-width: 52px; }
.tmsg { color: var(--on-surf-var); flex: 1; word-break: break-word; }
.tmsg.bright { color: var(--on-surface); }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.blinking { animation: blink 1.2s ease-in-out infinite; }

/* ── Progress / status banner ── */
.status-banner {
  background: rgba(173,198,255,.08);
  border: 1px solid rgba(173,198,255,.25);
  border-radius: 8px;
  padding: 12px 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.status-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 0 8px var(--primary);
  flex-shrink: 0;
}
.status-text {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: .06em;
  color: var(--primary);
}

/* ── Score bar ── */
.sbar-wrap { margin: 7px 0; }
.sbar-head { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px; }
.sbar-name { color: var(--on-surf-var); font-family: var(--sans); }
.sbar-val  { font-family: var(--mono); font-weight: 600; }
.sbar-bg   { height: 3px; background: var(--surf-highest); border-radius: 2px; }
.sbar-fill { height: 3px; border-radius: 2px; transition: width .6s ease; }

/* ── Trail ── */
.trail-wrap {
  background: var(--surf-mid);
  border: 1px solid var(--outline);
  border-radius: 10px;
  padding: 4px 14px;
  margin-top: 8px;
}
.trail-row {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 0;
  border-bottom: 1px solid var(--outline);
}
.trail-row:last-child { border-bottom: none; }
.trail-iter { font-family: var(--mono); font-size: 9px; color: var(--on-surf-var); min-width: 44px; }
.trail-pills { display: flex; gap: 4px; flex-wrap: wrap; flex: 1; }
.tpill {
  font-family: var(--mono); font-size: 9px;
  padding: 2px 7px; border-radius: 3px;
  background: var(--surf-highest);
  border: 1px solid var(--outline);
  color: var(--on-surf-var);
}
.trail-score { font-family: var(--mono); font-size: 12px; font-weight: 600; min-width: 72px; text-align: right; }

/* ── Chunk accordion body ── */
.chunk-body {
  font-size: 12px;
  line-height: 1.7;
  color: var(--on-surf-var);
  padding: 4px 0;
}

/* ── stButton overrides ── */
div[data-testid="stButton"] > button {
  background: rgba(173,198,255,.08) !important;
  border: 1px solid rgba(173,198,255,.3) !important;
  color: var(--primary) !important;
  border-radius: 7px !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .08em !important;
  padding: 9px 16px !important;
  width: 100% !important;
  transition: all .2s !important;
}
div[data-testid="stButton"] > button:hover {
  background: rgba(173,198,255,.18) !important;
  box-shadow: 0 0 12px rgba(173,198,255,.15) !important;
}
div[data-testid="stButton"] > button:disabled {
  opacity: .35 !important;
  cursor: not-allowed !important;
}

/* ── Launch button special ── */
div[data-testid="stButton"].launch-btn-wrap > button {
  background: rgba(74,225,118,.08) !important;
  border: 1px solid rgba(74,225,118,.35) !important;
  color: var(--green) !important;
  font-size: 11px !important;
  padding: 13px !important;
}
div[data-testid="stButton"].launch-btn-wrap > button:hover {
  background: rgba(74,225,118,.16) !important;
  box-shadow: 0 0 18px rgba(74,225,118,.15) !important;
}

/* ── Slider labels ── */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  letter-spacing: .08em !important;
  color: var(--on-surf-var) !important;
  text-transform: uppercase !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
  background: var(--surf-mid) !important;
  border: 1px solid var(--outline) !important;
  border-radius: 6px !important;
  color: var(--on-surface) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* ── Text area ── */
div[data-testid="stTextArea"] textarea {
  background: var(--surf-mid) !important;
  border: 1px solid var(--outline) !important;
  border-radius: 6px !important;
  color: var(--on-surface) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
  background: var(--surf-mid) !important;
  border: 1px solid var(--outline) !important;
  border-radius: 8px !important;
  color: var(--on-surface) !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 2px rgba(173,198,255,.15) !important;
}

/* ── Expander ── */
details { border: 1px solid var(--outline) !important; border-radius: 8px !important; margin-bottom: 10px !important; background: var(--surf-mid) !important; }
details > summary {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .08em !important;
  color: var(--on-surf-var) !important;
  padding: 12px 16px !important;
  text-transform: uppercase !important;
  cursor: pointer !important;
}
details[open] > summary { border-bottom: 1px solid var(--outline) !important; }

/* ── Progress bar ── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--primary), var(--green)) !important;
  border-radius: 2px !important;
}
.stProgress {
  margin: 0 !important;
  padding: 0 !important;
}

/* ── Line chart ── */
div[data-testid="stVegaLiteChart"] { border-radius: 8px !important; overflow: hidden; }

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
  background: var(--surf-mid) !important;
  border: 1px dashed var(--outline) !important;
  border-radius: 8px !important;
}
div[data-testid="stFileUploader"] label { display: none !important; }
div[data-testid="stFileUploader"] small { color: var(--on-surf-var) !important; font-family: var(--mono) !important; font-size: 9px !important; }

/* ── Metric ── */
div[data-testid="stMetric"] {
  background: var(--surf-mid) !important;
  border: 1px solid var(--outline) !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
}
div[data-testid="stMetric"] label {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  letter-spacing: .10em !important;
  color: var(--on-surf-var) !important;
  text-transform: uppercase !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: var(--sans) !important;
  font-weight: 700 !important;
  color: var(--on-surface) !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] p {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .06em !important;
  color: var(--on-surf-var) !important;
  text-transform: uppercase !important;
}

/* ── Divider ── */
hr { border-color: var(--outline) !important; margin: 16px 0 !important; opacity: .6; }

/* ── Alert / warning ── */
div[data-testid="stAlert"] {
  background: rgba(255,77,106,.1) !important;
  border: 1px solid rgba(255,77,106,.3) !important;
  border-radius: 8px !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* ── Info ── */
.info-box {
  background: rgba(173,198,255,.06);
  border: 1px solid rgba(173,198,255,.2);
  border-radius: 8px;
  padding: 10px 14px;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: .05em;
  color: var(--on-surf-var);
  text-align: center;
  margin-top: 6px;
}

/* ── Param mini-grid ── */
.param-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 10px;
  margin-top: 12px;
  padding: 14px;
  background: rgba(0,0,0,.3);
  border: 1px solid var(--outline);
  border-radius: 8px;
}
.param-item .plabel {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: .10em;
  color: var(--on-surf-var);
  text-transform: uppercase;
  margin-bottom: 2px;
}
.param-item .pval {
  font-family: var(--mono);
  font-size: 20px;
  font-weight: 700;
  line-height: 1.1;
}

/* ── Chat message ── */
.chat-user {
  margin: 10px 0;
}
.chat-user .who {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .10em;
  color: var(--on-surf-var);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.chat-user .bubble {
  padding: 10px 14px;
  background: var(--surf-mid);
  border: 1px solid var(--outline);
  border-radius: 8px 8px 2px 8px;
  font-size: 13px;
  line-height: 1.7;
  color: var(--on-surface);
}
.chat-astra .who {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .10em;
  color: var(--primary);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.chat-astra .bubble {
  padding: 10px 14px;
  background: var(--surf-low);
  border: 1px solid var(--outline);
  border-left: 2px solid var(--primary);
  border-radius: 2px 8px 8px 8px;
  font-size: 13px;
  line-height: 1.7;
  color: var(--on-surface);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        mode="A", docs=None, pdf_name=None,
        chat_history=[], run_history=[], terminal_logs=[],
        final_state=None, last_resp=None, chain=None,
        target=0.85, max_iter=3, chunk_size=500,
        top_k=4, prompt_v="v1",
        running=False,
        opt_status="", opt_iter=0,
        opt_start_time=None,
        opt_elapsed=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _score_color(v):
    if v >= 0.85: return "var(--green)"
    if v >= 0.65: return "var(--tertiary)"
    return "var(--error)"

def _sbar(name, val):
    c = _score_color(val)
    pct = int(val * 100)
    return f"""
    <div class="sbar-wrap">
      <div class="sbar-head">
        <span class="sbar-name">{name.replace('_',' ').title()}</span>
        <span class="sbar-val" style="color:{c}">{val:.3f}</span>
      </div>
      <div class="sbar-bg">
        <div class="sbar-fill" style="width:{pct}%;background:{c}"></div>
      </div>
    </div>"""

def _gauge_svg(value, target=0.85, size=150):
    r  = 58; cx = cy = size // 2
    C  = 2 * math.pi * r
    arc  = C * min(max(value, 0), 1.0)
    gap  = C - arc
    tarc = C * min(target, 1.0)
    tgap = C - tarc
    color = ("#4ae176" if value >= 0.85 else "#ffb786" if value >= 0.65 else "#ff4d6a")
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:8px 0">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e2020" stroke-width="9"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="rgba(173,198,255,0.25)" stroke-width="9" stroke-linecap="round"
          stroke-dasharray="{tarc:.2f} {tgap:.2f}"
          transform="rotate(-90 {cx} {cy})"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="{color}" stroke-width="9" stroke-linecap="round"
          stroke-dasharray="{arc:.2f} {gap:.2f}"
          transform="rotate(-90 {cx} {cy})"
          filter="drop-shadow(0 0 8px {color})"/>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle" fill="{color}"
          font-family="Geist,sans-serif" font-size="22" font-weight="700">{value:.2f}</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle" fill="#8c909f"
          font-family="JetBrains Mono,monospace" font-size="8" letter-spacing="2">AVG RAGAS</text>
      </svg>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.08em;color:#8c909f;margin-top:2px">
        TARGET {target:.2f}
      </div>
    </div>"""

def _ts():
    return time.strftime("%H:%M:%S")

def _level_class(line):
    l = line.lower()
    if "✅" in line or "[ok]" in l or "complete" in l or "saving" in l: return "tlvl-ok"
    if "❌" in line or "error" in l or "fail" in l:                     return "tlvl-err"
    if "tuning" in l or "switching" in l or "opt" in l:                 return "tlvl-eval"
    if "building" in l or "evaluating" in l or "loading" in l:          return "tlvl-sys"
    if "wait" in l or "listen" in l:                                    return "tlvl-wait"
    return "tlvl-info"

def _render_terminal(logs):
    if not logs:
        rows = '<div class="trow"><span class="tts">--:--:--</span><span class="tlvl-wait blinking">[WAIT]</span><span class="tmsg">LISTENING FOR NEXT EVENT_</span></div>'
    else:
        rows = ""
        for entry in logs[-60:]:
            ts   = entry.get("ts", "--:--:--")
            line = entry.get("msg", "")
            lvl  = _level_class(line)
            bright = "bright" if ("✅" in line or "[ok]" in line.lower()) else ""
            rows += f'<div class="trow"><span class="tts">{ts}</span><span class="{lvl}">[LOG]</span><span class="tmsg {bright}">{line}</span></div>'
    return f"""
    <div class="terminal-box">
      <div class="terminal-header">
        <span class="terminal-title">⬛ Live Terminal</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#3a3d3d">{len(logs)} events</span>
      </div>
      <div class="term-body">{rows}</div>
    </div>"""

def _add_log(msg: str):
    st.session_state.terminal_logs.append({"ts": _ts(), "msg": msg})

def _section_header(icon, title):
    st.markdown(f'<div class="slabel">{icon}&nbsp; {title}</div>', unsafe_allow_html=True)

def _spacer(h=12):
    st.markdown(f'<div style="height:{h}px"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV  (rendered as native Streamlit columns)
# ─────────────────────────────────────────────────────────────────────────────
nav_l, nav_m, nav_r = st.columns([3, 4, 3], gap="small")

with nav_l:
    st.markdown("""
    <div style="padding:10px 0 6px;display:flex;align-items:center;gap:14px">
      <span style="font-family:'Geist',sans-serif;font-size:22px;font-weight:900;
                   letter-spacing:-.04em;color:#e2e2e2">ASTRA</span>
      <div style="display:flex;align-items:center;gap:6px;padding:4px 12px;
                  background:rgba(74,225,118,.08);border:1px solid rgba(74,225,118,.18);
                  border-radius:999px">
        <div style="width:7px;height:7px;border-radius:50%;background:#4ae176;
                    box-shadow:0 0 7px rgba(74,225,118,.5)"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                     letter-spacing:.06em;color:#4ae176">LIVE</span>
      </div>
    </div>""", unsafe_allow_html=True)

with nav_r:
    _spacer(8)
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        if st.button("◈ AUTONOMOUS", key="btn_mode_a"):
            st.session_state.mode = "A"; st.rerun()
    with mc2:
        if st.button("▶ PRODUCTION", key="btn_mode_b"):
            st.session_state.mode = "B"; st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)
_spacer(4)

# Active mode badge
mode_label = "Autonomous Optimizer" if st.session_state.mode == "A" else "Production Chat"
mode_desc  = ("Prototype and refine RAG chains. Configure parameters, run the optimizer, and inspect results."
              if st.session_state.mode == "A"
              else "Chat with your document using a live-optimized FAISS pipeline.")
active_col = "#4ae176" if st.session_state.mode == "A" else "#adc6ff"
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:18px">
  <div style="width:3px;height:36px;background:{active_col};border-radius:2px;flex-shrink:0"></div>
  <div>
    <div style="font-family:'Geist',sans-serif;font-size:18px;font-weight:600;
                letter-spacing:-.02em;color:#e2e2e2;margin-bottom:2px">{mode_label}</div>
    <div style="font-size:12px;color:var(--on-surf-var)">{mode_desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
LEFT, RIGHT = st.columns([1.5, 0.7], gap="large")


# ═════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ═════════════════════════════════════════════════════════════════════════════
with LEFT:

    # ── PDF Upload ─────────────────────────────────────────────────────────
    _section_header("📄", "DOCUMENT SOURCE")
    up_col, status_col = st.columns([1, 2], gap="small")
    with up_col:
        uploaded = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
        if uploaded and uploaded.name != st.session_state.pdf_name:
            with st.spinner("LOADING DOCUMENT…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded.read()); tmp = f.name
                st.session_state.docs     = load_documents(tmp)
                st.session_state.pdf_name = uploaded.name
                st.session_state.chain    = None
                _add_log(f"Document loaded: {uploaded.name} ({len(st.session_state.docs)} pages)")
            st.rerun()

    with status_col:
        _spacer(8)
        if st.session_state.pdf_name:
            st.markdown(f"""
            <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding-top:4px">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                           padding:5px 12px;background:var(--surf-mid);
                           border:1px solid rgba(173,198,255,.25);border-radius:6px;color:var(--primary)">
                📄 {st.session_state.pdf_name}
              </span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                           padding:5px 10px;background:var(--surf-mid);
                           border:1px solid var(--outline);border-radius:6px;color:var(--on-surf-var)">
                {len(st.session_state.docs)} pages
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">UPLOAD A PDF TO BEGIN</div>', unsafe_allow_html=True)

    _spacer(16)

    # =========================================================================
    # MODE A — AUTONOMOUS OPTIMIZER
    # =========================================================================
    if st.session_state.mode == "A":

        # ── Config ────────────────────────────────────────────────────────────
        with st.expander("⚙  OPTIMIZER CONFIG", expanded=True):
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.session_state.target     = st.slider("TARGET RAGAS SCORE",   0.5, 1.0, st.session_state.target, 0.05)
                st.session_state.chunk_size = st.slider("STARTING CHUNK SIZE",  300, 1200, st.session_state.chunk_size, 100)
                st.session_state.prompt_v   = st.selectbox("PROMPT VARIANT",    ["v1","v2","v3"],
                                                index=["v1","v2","v3"].index(st.session_state.prompt_v))
            with c2:
                st.session_state.max_iter   = st.slider("MAX ITERATIONS",       2, 6, st.session_state.max_iter)
                st.session_state.top_k      = st.slider("STARTING TOP-K",       2, 8, st.session_state.top_k)

            _spacer(4)
            st.markdown(f"""
            <div class="param-grid">
              <div class="param-item">
                <div class="plabel">CHUNK</div>
                <div class="pval" style="color:var(--primary)">{st.session_state.chunk_size}</div>
              </div>
              <div class="param-item">
                <div class="plabel">TOP-K</div>
                <div class="pval" style="color:var(--green)">{st.session_state.top_k}</div>
              </div>
              <div class="param-item">
                <div class="plabel">TARGET</div>
                <div class="pval" style="color:var(--tertiary)">{st.session_state.target}</div>
              </div>
              <div class="param-item">
                <div class="plabel">PROMPT</div>
                <div class="pval" style="color:#a78bfa">{st.session_state.prompt_v.upper()}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Eval Dataset ──────────────────────────────────────────────────────
        with st.expander("📋  VALIDATION DATASET", expanded=False):
            raw_ds = st.text_area(
                "JSON",
                value=json.dumps(DEFAULT_EVAL_DATASET, indent=2),
                height=160,
                label_visibility="collapsed",
            )

        _spacer(8)

        # ── Launch Button ─────────────────────────────────────────────────────
        run_disabled = (st.session_state.docs is None) or st.session_state.running

        # Wrap with class for green override
        st.markdown('<div class="launch-btn-wrap">', unsafe_allow_html=True)
        launch = st.button(
            "◈  LAUNCH AUTONOMOUS OPTIMIZER",
            disabled=run_disabled,
            use_container_width=True,
            key="launch_btn",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.docs:
            st.markdown('<div class="info-box">⬆ UPLOAD A PDF TO ENABLE OPTIMIZER</div>',
                        unsafe_allow_html=True)

        # ── Active-run status placeholders ────────────────────────────────────
        status_ph   = st.empty()   # running banner / completion card
        progress_ph = st.empty()   # progress bar
        timer_ph    = st.empty()   # ← live elapsed timer (clears on finish)

        # ── Run Logic ─────────────────────────────────────────────────────────
        if launch:
            try:
                eval_ds = json.loads(raw_ds)
            except json.JSONDecodeError:
                st.error("Invalid JSON in dataset."); st.stop()

            st.session_state.terminal_logs  = []
            st.session_state.run_history    = []
            st.session_state.final_state    = None
            st.session_state.running        = True
            st.session_state.opt_start_time = time.time()
            st.session_state.opt_elapsed    = None

            logs    = st.session_state.terminal_logs
            history = st.session_state.run_history

            def _log_cb(msg: str):
                logs.append({"ts": _ts(), "msg": msg})

            astra = build_graph(
                st.session_state.docs, eval_ds,
                st.session_state.target, st.session_state.max_iter,
                log_callback=_log_cb, history=history,
            )

            _add_log("INITIALIZING ASTRA AUTONOMOUS OPTIMIZER…")
            _add_log(f"Config → chunk={st.session_state.chunk_size} | k={st.session_state.top_k} | "
                     f"prompt={st.session_state.prompt_v} | target={st.session_state.target}")

            # ── Show running banner + progress bar ────────────────────────────
            status_ph.markdown("""
            <div class="status-banner">
              <div class="status-dot"></div>
              <div class="status-text">OPTIMIZER RUNNING — processing iterations…</div>
            </div>""", unsafe_allow_html=True)
            progress_ph.progress(0, text="Initializing…")

            # ── Ticker thread — updates timer_ph every second ─────────────────
            _stop_ticker = threading.Event()

            def _ticker():
                while not _stop_ticker.is_set():
                    elapsed = time.time() - st.session_state.opt_start_time
                    mins    = int(elapsed) // 60
                    secs    = int(elapsed) % 60
                    timer_ph.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;
                                padding:8px 16px;
                                background:rgba(0,0,0,.3);
                                border:1px solid rgba(173,198,255,.15);
                                border-radius:6px;margin-top:2px;width:fit-content">
                      <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                   letter-spacing:.10em;color:var(--on-surf-var)">⏱ ELAPSED</span>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:20px;
                                   font-weight:700;color:var(--primary);letter-spacing:.04em">
                        {mins:02d}:{secs:02d}
                      </span>
                    </div>""", unsafe_allow_html=True)
                    _stop_ticker.wait(timeout=1.0)

            ticker_thread = threading.Thread(target=_ticker, daemon=True)
            add_script_run_ctx(ticker_thread)
            ticker_thread.start()

            # ── Invoke the graph (blocking) ───────────────────────────────────
            final = astra.invoke({
                "chunk_size":    st.session_state.chunk_size,
                "chunk_overlap": int(st.session_state.chunk_size * 0.2),
                "top_k":         st.session_state.top_k,
                "prompt_variant": st.session_state.prompt_v,
                "scores": {}, "iteration": 0, "done": False,
            })

            # ── Stop ticker, record elapsed ───────────────────────────────────
            _stop_ticker.set()
            ticker_thread.join(timeout=2)
            total_elapsed = time.time() - st.session_state.opt_start_time
            st.session_state.opt_elapsed    = total_elapsed
            st.session_state.opt_start_time = None   # clear so timer doesn't re-show

            # ── Store results ─────────────────────────────────────────────────
            st.session_state.final_state = final
            st.session_state.running     = False

            best_avg = final["scores"].get("avg", 0)
            _add_log(f"✅ OPTIMIZER COMPLETE — Best avg={best_avg:.4f} | "
                     f"chunk={final['chunk_size']} | k={final['top_k']} | "
                     f"prompt={final['prompt_variant']} | "
                     f"time={total_elapsed:.1f}s")

            # ── Clear timer, update progress, show completion card ────────────
            timer_ph.empty()          # ← timer disappears on completion
            progress_ph.progress(1.0, text="Complete!")

            passed = best_avg >= st.session_state.target
            icon   = "✅" if passed else "⚠️"
            color  = "#4ae176" if passed else "#ffb786"
            mins_t = int(total_elapsed) // 60
            secs_t = int(total_elapsed) % 60

            status_ph.markdown(f"""
            <div style="background:rgba(74,225,118,.06);border:1px solid rgba(74,225,118,.25);
                        border-radius:8px;padding:12px 18px;display:flex;align-items:center;
                        gap:14px;margin-bottom:10px">
              <span style="font-size:20px">{icon}</span>
              <div style="flex:1">
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                            letter-spacing:.06em;color:{color};margin-bottom:3px">OPTIMIZATION COMPLETE</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                            color:var(--on-surf-var)">
                  avg={best_avg:.4f} &nbsp;|&nbsp; target={st.session_state.target}
                </div>
              </div>
              <div style="text-align:right">
                <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                            letter-spacing:.10em;color:var(--on-surf-var);margin-bottom:2px">TOTAL TIME</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                            font-weight:700;color:var(--primary)">{mins_t:02d}:{secs_t:02d}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            time.sleep(0.6)
            st.rerun()

        # ── If still running from a prior rerun, show persisted elapsed ───────
        elif st.session_state.running and st.session_state.opt_start_time:
            elapsed = time.time() - st.session_state.opt_start_time
            mins    = int(elapsed) // 60
            secs    = int(elapsed) % 60
            status_ph.markdown("""
            <div class="status-banner">
              <div class="status-dot"></div>
              <div class="status-text">OPTIMIZER RUNNING — processing iterations…</div>
            </div>""", unsafe_allow_html=True)
            timer_ph.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        padding:8px 16px;
                        background:rgba(0,0,0,.3);
                        border:1px solid rgba(173,198,255,.15);
                        border-radius:6px;margin-top:2px;width:fit-content">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                           letter-spacing:.10em;color:var(--on-surf-var)">⏱ ELAPSED</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:20px;
                           font-weight:700;color:var(--primary);letter-spacing:.04em">
                {mins:02d}:{secs:02d}
              </span>
            </div>""", unsafe_allow_html=True)

        _spacer(12)

        # ── Terminal ──────────────────────────────────────────────────────────
        _section_header("⬛", "LIVE TERMINAL")
        st.markdown(_render_terminal(st.session_state.terminal_logs), unsafe_allow_html=True)

        # ── Score chart ───────────────────────────────────────────────────────
        if st.session_state.run_history:
            _spacer(16)
            _section_header("📈", "SCORE PROGRESSION")

            df = pd.DataFrame(st.session_state.run_history)

            # Build chart — use Altair so it renders as proper interactive chart
            metric_cols = [c for c in ["faithfulness", "context_recall", "context_precision", "avg"]
                           if c in df.columns]
            if metric_cols and "iteration" in df.columns:
                import altair as alt

                color_map = {
                    "faithfulness":      "#adc6ff",
                    "context_recall":    "#4ae176",
                    "context_precision": "#ffb786",
                    "avg":               "#a78bfa",
                    "target":            "#ff4d6a",
                }

                # Melt into long form for Altair
                chart_df = df[["iteration"] + metric_cols].copy()
                chart_df["target"] = st.session_state.target
                all_cols = metric_cols + ["target"]
                melted = chart_df.melt(
                    id_vars="iteration",
                    value_vars=all_cols,
                    var_name="metric",
                    value_name="score",
                )
                domain   = list(color_map.keys())
                range_   = [color_map.get(c, "#ffffff") for c in domain]

                chart = (
                    alt.Chart(melted)
                    .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
                    .encode(
                        x=alt.X("iteration:O", axis=alt.Axis(
                            labelColor="#8c909f", tickColor="#2e3030",
                            domainColor="#2e3030", gridColor="#1e2020",
                            labelFont="JetBrains Mono", title=None,
                        )),
                        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1.05]),
                            axis=alt.Axis(
                                labelColor="#8c909f", tickColor="#2e3030",
                                domainColor="#2e3030", gridColor="#1e2020",
                                labelFont="JetBrains Mono", format=".2f", title=None,
                            )),
                        color=alt.Color("metric:N",
                            scale=alt.Scale(domain=domain, range=range_),
                            legend=alt.Legend(
                                orient="bottom", direction="horizontal",
                                labelColor="#c2c6d6", labelFont="JetBrains Mono",
                                labelFontSize=9, symbolSize=60,
                                padding=8, titleColor="#8c909f",
                            ),
                        ),
                        strokeDash=alt.condition(
                            alt.datum.metric == "target",
                            alt.value([4, 4]),
                            alt.value([0]),
                        ),
                        tooltip=[
                            alt.Tooltip("iteration:O", title="Run"),
                            alt.Tooltip("metric:N",    title="Metric"),
                            alt.Tooltip("score:Q",     title="Score", format=".4f"),
                        ],
                    )
                    .properties(height=180, background="transparent")
                    .configure_view(strokeWidth=0, fill="transparent")
                    .configure_axis(gridOpacity=0.3)
                )
                st.altair_chart(chart, use_container_width=True)

            # ── Parameter Trail ────────────────────────────────────────────────
            _spacer(8)
            _section_header("🔁", "PARAMETER TRAIL")
            st.markdown('<div class="trail-wrap">', unsafe_allow_html=True)
            for row in st.session_state.run_history:
                row_avg   = row.get("avg", 0)
                row_color = ("#4ae176" if row_avg >= 0.85 else "#ffb786" if row_avg >= 0.65 else "#ff4d6a")
                row_icon  = "✅" if row_avg >= st.session_state.target else "❌"
                st.markdown(f"""
                <div class="trail-row">
                  <span class="trail-iter">RUN {row.get('iteration', '?')}</span>
                  <span class="trail-pills">
                    <span class="tpill">chunk {row.get('chunk_size','?')}</span>
                    <span class="tpill">k={row.get('top_k','?')}</span>
                    <span class="tpill">{row.get('prompt_variant','?')}</span>
                  </span>
                  <span class="trail-score" style="color:{row_color}">{row_icon} {row_avg:.3f}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # MODE B — PRODUCTION CHAT
    # =========================================================================
    else:
        with st.expander("⚙  PIPELINE CONFIG", expanded=False):
            pc1, pc2, pc3 = st.columns(3, gap="medium")
            with pc1: st.session_state.chunk_size = st.slider("CHUNK SIZE", 300, 1200, st.session_state.chunk_size, 100)
            with pc2: st.session_state.top_k      = st.slider("TOP-K", 2, 8, st.session_state.top_k)
            with pc3: st.session_state.prompt_v   = st.selectbox("PROMPT VARIANT", ["v1","v2","v3"],
                                                       index=["v1","v2","v3"].index(st.session_state.prompt_v))
            _spacer(6)
            rc1, rc2 = st.columns(2, gap="small")
            with rc1:
                if st.button("↺ REBUILD INDEX", key="rebuild_idx"):
                    st.session_state.chain = None
                    _add_log("FAISS index cleared — will rebuild on next query.")
                    st.rerun()
            with rc2:
                if st.button("⌫ CLEAR CHAT", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.session_state.last_resp    = None
                    st.rerun()

        _spacer(12)
        _section_header("💬", "CHAT HISTORY")

        if st.session_state.chat_history:
            chat_html = '<div style="margin-bottom:16px">'
            for m in st.session_state.chat_history:
                if m["role"] == "user":
                    chat_html += f'<div class="chat-user"><div class="who">You</div><div class="bubble">{m["content"]}</div></div>'
                else:
                    chat_html += f'<div class="chat-astra" style="margin:10px 0"><div class="who">◈ ASTRA</div><div class="bubble">{m["content"]}</div></div>'
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box" style="margin-bottom:14px">ASK ANYTHING ABOUT YOUR DOCUMENT BELOW</div>',
                        unsafe_allow_html=True)

        question = st.chat_input("Ask anything about your document…")
        if question:
            if not st.session_state.docs:
                st.warning("Upload a PDF first."); st.stop()
            if not st.session_state.chain:
                with st.spinner("BUILDING FAISS INDEX…"):
                    _add_log("Building FAISS index…")
                    chain, n = build_rag_pipeline(
                        st.session_state.docs,
                        st.session_state.chunk_size,
                        int(st.session_state.chunk_size * 0.2),
                        st.session_state.top_k,
                        st.session_state.prompt_v,
                    )
                    st.session_state.chain = chain
                    _add_log(f"✅ FAISS ready — {n} chunks indexed.")
            with st.spinner("RETRIEVING…"):
                _add_log(f"Query: {question[:60]}…")
                resp = run_rag(question, st.session_state.chain)
                _add_log(f"✅ QUERY COMPLETE | latency={resp['latency']}s | tokens={resp['total_tokens']}")
            st.session_state.last_resp = resp
            st.session_state.chat_history.append({"role":"user",  "content": question})
            st.session_state.chat_history.append({"role":"astra", "content": resp["answer"]})
            st.rerun()

        _spacer(12)
        _section_header("⬛", "LIVE TERMINAL")
        st.markdown(_render_terminal(st.session_state.terminal_logs), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — OBSERVABILITY
# ═════════════════════════════════════════════════════════════════════════════
with RIGHT:
    st.markdown("""
    <div style="background:var(--surf-mid);border:1px solid var(--outline);
                border-radius:10px;padding:14px 18px;margin-bottom:14px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.10em;
                  color:var(--on-surf-var);text-transform:uppercase;margin-bottom:2px">◈ Observability</div>
      <div style="font-size:11px;color:var(--on-surf-var);opacity:.6">Real-time pipeline metrics</div>
    </div>""", unsafe_allow_html=True)

    # ── MODE A — Optimizer Results ──────────────────────────────────────────
    if st.session_state.mode == "A":
        fs = st.session_state.final_state

        if fs:
            scores = fs.get("scores", {})
            avg    = scores.get("avg", 0.0)
            passed = avg >= st.session_state.target

            # Gauge
            st.markdown(f"""
            <div class="obs-card" style="text-align:center;padding-bottom:16px">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                          letter-spacing:.10em;color:var(--on-surf-var);
                          text-transform:uppercase;margin-bottom:4px">Quality Score</div>
              {_gauge_svg(avg, st.session_state.target)}
            </div>""", unsafe_allow_html=True)

            # Key metrics
            status_color = "var(--green)" if passed else "var(--error)"
            status_txt   = "TARGET MET ✅" if passed else "BELOW TARGET ❌"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
              <div class="obs-card">
                <div class="obs-label">CHUNK SIZE</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:var(--primary)">{fs['chunk_size']}</div>
              </div>
              <div class="obs-card">
                <div class="obs-label">TOP-K</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:var(--green)">{fs['top_k']}</div>
              </div>
              <div class="obs-card">
                <div class="obs-label">PROMPT</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#a78bfa">{fs['prompt_variant'].upper()}</div>
              </div>
              <div class="obs-card">
                <div class="obs-label">STATUS</div>
                <div style="font-size:12px;font-family:'JetBrains Mono',monospace;
                            font-weight:700;color:{status_color};margin-top:6px">{status_txt}</div>
                <div class="obs-sub">{fs.get('iteration', '?')} iterations</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Score bars — render each individually to avoid % escaping issues
            _section_header("📊", "FINAL RAGAS SCORES")
            st.markdown('<div class="obs-card" style="padding-bottom:4px">', unsafe_allow_html=True)
            for k, v in scores.items():
                if k != "avg":
                    c   = _score_color(v)
                    pct = int(v * 100)
                    st.markdown(f"""
                    <div style="margin:7px 0">
                      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                        <span style="color:var(--on-surf-var);font-family:var(--sans)">{k.replace('_',' ').title()}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{c}">{v:.3f}</span>
                      </div>
                      <div style="height:3px;background:var(--surf-highest);border-radius:2px">
                        <div style="height:3px;width:{pct}%;background:{c};border-radius:2px"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            # Average divider + bar
            avg_c   = _score_color(avg)
            avg_pct = int(avg * 100)
            st.markdown(f"""
            <div style="border-top:1px solid var(--outline);margin-top:10px;padding-top:10px;margin-bottom:4px">
              <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                <span style="color:var(--on-surf-var);font-family:var(--sans)">Average Score</span>
                <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{avg_c}">{avg:.3f}</span>
              </div>
              <div style="height:3px;background:var(--surf-highest);border-radius:2px">
                <div style="height:3px;width:{avg_pct}%;background:{avg_c};border-radius:2px"></div>
              </div>
            </div>
            </div>""", unsafe_allow_html=True)

        else:
            # Placeholder
            st.markdown(f"""
            <div class="obs-card" style="text-align:center;padding:36px 0;margin-bottom:12px">
              <div style="font-size:36px;opacity:.1;margin-bottom:10px">◈</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                          letter-spacing:.08em;color:var(--on-surf-var);opacity:.5;line-height:1.8">
                CONFIGURE AND LAUNCH<br>TO SEE RESULTS
              </div>
            </div>
            <div class="obs-card" style="text-align:center">
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                          letter-spacing:.10em;color:var(--on-surf-var);
                          text-transform:uppercase;margin-bottom:4px">Quality Score</div>
              {_gauge_svg(0.0, st.session_state.target)}
            </div>""", unsafe_allow_html=True)

            # Show live progress if running
            if st.session_state.running:
                st.markdown("""
                <div class="status-banner" style="margin-top:12px">
                  <div class="status-dot"></div>
                  <div class="status-text">OPTIMIZER ACTIVE…</div>
                </div>""", unsafe_allow_html=True)
                st.progress(0.5)

    # ── MODE B — Live Telemetry ─────────────────────────────────────────────
    else:
        resp = st.session_state.last_resp

        if resp:
            # Latency
            st.markdown(f"""
            <div class="obs-card">
              <div class="obs-label">Execution Latency</div>
              <div class="obs-val">{resp['latency']}<span class="obs-val-unit">s</span></div>
            </div>""", unsafe_allow_html=True)

            # Tokens
            in_tok  = resp.get('input_tokens', 0)
            out_tok = resp.get('output_tokens', 0)
            tot_tok = resp.get('total_tokens', max(in_tok + out_tok, 1))
            in_pct  = int(in_tok / max(tot_tok, 1) * 100)
            st.markdown(f"""
            <div class="obs-card">
              <div class="obs-label">Token Usage</div>
              <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px">
                <div class="obs-val">{tot_tok}</div>
                <span style="color:var(--on-surf-var);font-size:11px">total</span>
              </div>
              <div style="height:4px;border-radius:2px;overflow:hidden;
                          background:var(--surf-highest);display:flex">
                <div style="width:{in_pct}%;background:var(--primary);border-radius:2px 0 0 2px"></div>
                <div style="flex:1;background:var(--green);border-radius:0 2px 2px 0"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:5px;
                          font-family:'JetBrains Mono',monospace;font-size:9px">
                <span style="color:var(--primary)">IN: {in_tok}</span>
                <span style="color:var(--green)">OUT: {out_tok}</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Cost
            st.markdown(f"""
            <div class="obs-card">
              <div class="obs-label">Estimated Cost</div>
              <div style="font-size:22px;font-weight:600;font-family:'Geist',sans-serif;
                          color:var(--on-surface);margin-top:4px">
                ${resp['cost_usd']} <span style="font-size:12px;color:var(--on-surf-var)">USD</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Chunks
            n_ctx = len(resp.get('contexts', []))
            st.markdown(f"""
            <div class="obs-card">
              <div class="obs-label">FAISS Chunks Retrieved</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:28px;
                          font-weight:700;color:var(--green)">{n_ctx}</div>
            </div>""", unsafe_allow_html=True)

            # Sub-queries
            sub_qs = resp.get("sub_queries", [])
            if sub_qs:
                _section_header("🔍", "REWRITER SUB-QUERIES")
                sq_html = '<div class="obs-card">'
                for i, q in enumerate(sub_qs, 1):
                    sq_html += f"""
                    <div style="display:flex;gap:10px;padding:7px 0;
                                border-bottom:1px solid var(--outline);
                                font-size:11px;align-items:flex-start">
                      <span style="font-family:'JetBrains Mono',monospace;
                                   color:var(--green);font-size:9px;min-width:14px;padding-top:2px">{i}.</span>
                      <span style="color:var(--on-surf-var);line-height:1.6">{q}</span>
                    </div>"""
                sq_html += "</div>"
                st.markdown(sq_html, unsafe_allow_html=True)

            # Source contexts
            _section_header("📚", "SOURCE CONTEXTS (FAISS)")
            for i, ctx in enumerate(resp.get("contexts", []), 1):
                with st.expander(f"Chunk {i} — {ctx[:35].strip()}…"):
                    st.markdown(f'<div class="chunk-body">{ctx}</div>', unsafe_allow_html=True)

            # Raw payload
            with st.expander("{ }  RAW PAYLOAD"):
                payload = {
                    "question":  resp.get("question"),
                    "answer":    resp.get("answer"),
                    "telemetry": {
                        "latency_s":     resp.get("latency"),
                        "input_tokens":  in_tok,
                        "output_tokens": out_tok,
                        "total_tokens":  tot_tok,
                        "cost_usd":      resp.get("cost_usd"),
                    },
                    "sub_queries":  resp.get("sub_queries", []),
                    "num_contexts": n_ctx,
                }
                st.markdown(f"""
                <div style="background:#020408;border:1px solid var(--outline);
                            border-radius:6px;padding:12px 14px;
                            font-family:'JetBrains Mono',monospace;font-size:10px;
                            color:#79c0ff;white-space:pre-wrap;overflow-x:auto;
                            max-height:220px;overflow-y:auto;line-height:1.7">
{json.dumps(payload, indent=2)}</div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="padding:60px 0;text-align:center">
              <div style="font-size:42px;opacity:.08;margin-bottom:12px">◈</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                          letter-spacing:.10em;color:var(--on-surf-var);opacity:.4">
                ASK A QUESTION<br>TO SEE TELEMETRY
              </div>
            </div>""", unsafe_allow_html=True)