"""
app.py — Astra UI.  Run with:  streamlit run app.py
Matches the HTML design: dark surface, green accents, two-panel layout.
Left  → query input + response + live terminal
Right → observability panel (latency, tokens, cost, gauge, FAISS chunks)
"""

import json
import math
import time
import tempfile
import threading
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
# CSS — mirrors the HTML design token-for-token
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Geist:wght@400;500;600;900&display=swap');

/* ── tokens ── */
:root {
  --bg:            #0A0A0A;
  --surface:       #121414;
  --surf-low:      #1a1c1c;
  --surf-mid:      #1e2020;
  --surf-high:     #282a2b;
  --surf-highest:  #333535;
  --outline:       #424754;
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

/* ── reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
  font-family: var(--sans) !important;
  background: var(--bg) !important;
  color: var(--on-surface) !important;
  -webkit-font-smoothing: antialiased;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #262626; border-radius: 2px; }

/* ── top nav ── */
.topnav {
  position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 200;
  background: var(--surf-low);
  border-bottom: 1px solid var(--outline);
  display: flex; align-items: center;
  justify-content: space-between;
  padding: 0 32px;
}
.topnav-logo {
  font-family: var(--sans); font-size: 24px; font-weight: 900;
  letter-spacing: -0.04em; color: var(--on-surface);
}
.topnav-badge {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 12px;
  background: rgba(74,225,118,0.1);
  border: 1px solid rgba(74,225,118,0.2);
  border-radius: 999px;
}
.badge-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px rgba(74,225,118,0.4);
  animation: pulse 2s cubic-bezier(.4,0,.6,1) infinite;
}
.badge-text {
  font-family: var(--mono); font-size: 10px; letter-spacing: .05em;
  color: var(--green);
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* mode switcher */
.mode-switcher {
  display: flex; background: var(--surf-mid);
  border: 1px solid var(--outline); border-radius: 6px; padding: 4px;
}
.mode-btn {
  padding: 6px 16px; border-radius: 4px; border: none;
  font-family: var(--mono); font-size: 11px; letter-spacing: .05em;
  cursor: pointer; transition: all .2s;
}
.mode-btn.active {
  background: var(--surf-highest); color: var(--on-surface);
  box-shadow: 0 1px 3px rgba(0,0,0,.4);
}
.mode-btn.inactive {
  background: transparent; color: var(--on-surf-var);
}
.mode-btn.inactive:hover { color: var(--on-surface); }

/* ── sidebar icons ── */
.sidebar {
  position: fixed; left: 0; top: 64px;
  width: 64px; height: calc(100vh - 64px); z-index: 100;
  background: var(--surf-low);
  border-right: 1px solid var(--outline);
  display: flex; flex-direction: column;
  align-items: center; padding: 24px 0; gap: 24px;
}
.sb-icon { font-size: 20px; cursor: pointer; opacity: .5; transition: opacity .2s; }
.sb-icon:hover { opacity: 1; }
.sb-icon.active { opacity: 1; color: var(--primary); }

/* ── main layout ── */
.main-wrap {
  margin-left: 64px;
  margin-top: 64px;
  height: calc(100vh - 64px);
  display: flex;
  overflow: hidden;
}

/* ── panels ── */
.left-panel {
  flex: 1; display: flex; flex-direction: column;
  border-right: 1px solid var(--outline);
  padding: 32px; background: var(--bg);
  overflow: hidden; gap: 20px;
}
.right-panel {
  width: 400px; overflow-y: auto;
  background: var(--surf-low);
  padding: 24px; display: flex;
  flex-direction: column; gap: 24px;
}

/* ── section label ── */
.slabel {
  font-family: var(--mono); font-size: 10px; letter-spacing: .08em;
  color: var(--on-surf-var); text-transform: uppercase;
  display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}

/* ── input row ── */
.input-row {
  display: flex; gap: 10px; align-items: stretch;
}
.query-input {
  flex: 1; background: var(--bg);
  border: 1px solid var(--outline);
  border-radius: 8px;
  padding: 14px 16px;
  font-family: var(--sans); font-size: 14px; color: var(--on-surface);
  outline: none; transition: border-color .2s;
}
.query-input:focus { border-color: var(--primary); }
.run-btn {
  background: var(--primary); color: #001a42;
  border: none; border-radius: 6px;
  padding: 0 20px; cursor: pointer;
  font-family: var(--mono); font-size: 11px; letter-spacing: .05em;
  font-weight: 600; transition: filter .2s, transform .1s;
  white-space: nowrap;
}
.run-btn:hover { filter: brightness(1.1); }
.run-btn:active { transform: scale(.97); }

/* ── response pane ── */
.pane-header {
  background: var(--surf-mid);
  border-bottom: 1px solid var(--outline);
  padding: 8px 16px;
  display: flex; justify-content: space-between; align-items: center;
}
.pane-label {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .08em; color: var(--on-surf-var); text-transform: uppercase;
}
.response-pane {
  background: var(--surf-low); border: 1px solid var(--outline);
  border-radius: 8px; overflow: hidden;
  display: flex; flex-direction: column;
  flex: 3; min-height: 150px;
}
.response-body {
  padding: 20px 24px; overflow-y: auto;
  font-size: 14px; line-height: 1.7;
  color: var(--on-surface); flex: 1;
}
.response-body code {
  font-family: var(--mono); font-size: 12px;
  background: rgba(0,0,0,.4); color: var(--green);
  padding: 1px 5px; border-radius: 3px;
}
.response-empty {
  display: flex; align-items: center; justify-content: center;
  height: 100%; color: var(--on-surf-var);
  font-size: 13px; font-family: var(--mono); letter-spacing: .05em;
  opacity: .5;
}

/* ── terminal ── */
.terminal-pane {
  background: #050505; border: 1px solid var(--outline);
  border-radius: 8px; overflow: hidden;
  display: flex; flex-direction: column;
  flex: 2; min-height: 120px;
}
.term-body {
  padding: 14px 16px; overflow-y: auto;
  font-family: var(--mono); font-size: 11px;
  line-height: 2; flex: 1; color: var(--on-surf-var);
}
.tlog-row { display: flex; gap: 12px; }
.tlog-time { color: var(--outline); min-width: 56px; }
.tlog-level-ok    { color: var(--green); font-weight: 700; min-width: 48px; }
.tlog-level-sys   { color: var(--primary); min-width: 48px; }
.tlog-level-eval  { color: var(--tertiary); min-width: 48px; }
.tlog-level-info  { color: var(--green); min-width: 48px; }
.tlog-level-err   { color: var(--error); min-width: 48px; }
.tlog-level-wait  { color: var(--outline); min-width: 48px; animation: pulse 2s infinite; }
.tlog-msg { color: var(--on-surf-var); }
.tlog-msg.bright { color: var(--on-surface); }

/* ── right panel cards ── */
.obs-card {
  background: var(--surf-mid); border: 1px solid var(--outline);
  border-radius: 8px; padding: 16px 18px;
}
.obs-label {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .08em; color: var(--on-surf-var);
  margin-bottom: 4px; text-transform: uppercase;
}
.obs-val {
  font-size: 32px; font-weight: 700; font-family: var(--sans);
  color: var(--on-surface); line-height: 1.1;
}
.obs-val-unit {
  font-size: 16px; font-weight: 400; color: var(--on-surf-var); margin-left: 4px;
}
.obs-sub { font-size: 12px; color: var(--on-surf-var); margin-top: 2px; }

/* ── gauge ── */
.gauge-wrap {
  text-align: center; padding: 12px 0;
}
.gauge-title {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: .08em; color: var(--on-surf-var);
  text-transform: uppercase; margin-bottom: 20px;
}

/* ── chunk accordion ── */
.chunk-btn {
  width: 100%;
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 12px;
  background: var(--surf-high); border: 1px solid var(--outline);
  border-radius: 4px; cursor: pointer;
  font-family: var(--mono); font-size: 11px; color: var(--on-surface);
  letter-spacing: .05em; transition: background .15s;
  text-align: left;
}
.chunk-btn:hover { background: var(--surf-highest); }
.chunk-body {
  padding: 12px;
  border-left: 1px solid var(--outline);
  border-right: 1px solid var(--outline);
  border-bottom: 1px solid var(--outline);
  border-radius: 0 0 4px 4px;
  background: rgba(0,0,0,.2);
  font-size: 13px; line-height: 1.6; color: var(--on-surf-var);
  margin-bottom: 8px;
}
.chunk-tag {
  background: rgba(74,225,118,.15); color: var(--green);
  padding: 1px 5px; border-radius: 3px; font-size: 12px;
}

/* ── optimizer config card (mode A) ── */
.opt-card {
  background: var(--surf-mid); border: 1px solid var(--outline);
  border-radius: 8px; padding: 16px 18px; margin-bottom: 12px;
}
.param-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;
}
.param-item .param-label {
  font-family: var(--mono); font-size: 9px; letter-spacing: .08em;
  color: var(--on-surf-var); text-transform: uppercase; margin-bottom: 2px;
}
.param-item .param-val {
  font-family: var(--mono); font-size: 22px; font-weight: 700;
}
.pv-cyan   { color: var(--primary); }
.pv-green  { color: var(--green); }
.pv-amber  { color: var(--tertiary); }
.pv-purple { color: #a78bfa; }

/* score bar */
.sbar-wrap { margin: 6px 0; }
.sbar-header { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px; }
.sbar-name { color: var(--on-surf-var); font-family: var(--sans); }
.sbar-val  { font-family: var(--mono); font-weight: 500; }
.sbar-bg   { height: 3px; background: var(--surf-highest); border-radius: 2px; }
.sbar-fill { height: 3px; border-radius: 2px; }

/* trail */
.trail-row {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 0; border-bottom: 1px solid var(--outline);
}
.trail-iter { font-family: var(--mono); font-size: 10px; color: var(--on-surf-var); min-width: 40px; }
.trail-pills { display: flex; gap: 4px; flex-wrap: wrap; flex: 1; }
.tpill {
  font-family: var(--mono); font-size: 9px;
  padding: 2px 6px; border-radius: 3px;
  background: var(--surf-highest); border: 1px solid var(--outline);
  color: var(--on-surf-var);
}
.trail-score { font-family: var(--mono); font-size: 13px; font-weight: 600; min-width: 68px; text-align: right; }

/* launch btn */
.launch-btn {
  width: 100%; padding: 12px;
  background: rgba(173,198,255,.1);
  border: 1px solid var(--primary);
  border-radius: 6px; color: var(--primary);
  font-family: var(--mono); font-size: 11px; letter-spacing: .08em;
  cursor: pointer; transition: all .2s; margin-top: 6px;
}
.launch-btn:hover { background: rgba(173,198,255,.2); box-shadow: 0 0 16px rgba(173,198,255,.15); }
.launch-btn:disabled { opacity: .4; cursor: not-allowed; }

/* upload btn */
.upload-btn {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 12px;
  background: var(--surf-mid); border: 1px solid var(--outline);
  border-radius: 6px; color: var(--on-surf-var);
  font-family: var(--mono); font-size: 10px; letter-spacing: .05em;
  cursor: pointer; transition: all .15s; white-space: nowrap;
}
.upload-btn:hover { background: var(--surf-highest); color: var(--on-surface); }

/* stButton overrides */
div[data-testid="stButton"] > button {
  background: rgba(173,198,255,.08) !important;
  border: 1px solid var(--primary) !important;
  color: var(--primary) !important;
  border-radius: 6px !important;
  font-family: var(--mono) !important;
  font-size: 10px !important; letter-spacing: .08em !important;
  padding: 8px 16px !important;
  width: 100% !important; transition: all .2s !important;
}
div[data-testid="stButton"] > button:hover {
  background: rgba(173,198,255,.18) !important;
  box-shadow: 0 0 12px rgba(173,198,255,.2) !important;
}
div[data-testid="stSlider"]  label { font-family: var(--mono) !important; font-size: 9px !important; letter-spacing: .08em !important; color: var(--on-surf-var) !important; }
div[data-testid="stSelectbox"] label { font-family: var(--mono) !important; font-size: 9px !important; letter-spacing: .08em !important; color: var(--on-surf-var) !important; }
div[data-testid="stTextArea"]  label { font-family: var(--mono) !important; font-size: 9px !important; letter-spacing: .08em !important; color: var(--on-surf-var) !important; }
div[data-testid="stFileUploader"] label { display: none !important; }

/* progress bar */
.stProgress > div > div { background: linear-gradient(90deg, var(--primary), var(--green)) !important; border-radius: 2px !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* chat input */
div[data-testid="stChatInput"] textarea {
  background: var(--bg) !important;
  border: 1px solid var(--outline) !important;
  border-radius: 8px !important;
  font-family: var(--sans) !important;
  color: var(--on-surface) !important;
  font-size: 14px !important;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: var(--primary) !important;
}

/* expander */
details summary { font-family: var(--mono) !important; font-size: 10px !important; letter-spacing: .06em !important; color: var(--on-surf-var) !important; }
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
        query_text="", answer_text="",
        running=False,
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
      <div class="sbar-header">
        <span class="sbar-name">{name.replace('_',' ').title()}</span>
        <span class="sbar-val" style="color:{c}">{val:.3f}</span>
      </div>
      <div class="sbar-bg">
        <div class="sbar-fill" style="width:{pct}%;background:{c}"></div>
      </div>
    </div>"""

def _gauge_svg(value, size=140):
    r = 60; cx = cy = size // 2
    circumference = 2 * math.pi * r
    arc = circumference * min(max(value, 0), 1.0)
    gap = circumference - arc
    color = ("#4ae176" if value >= 0.85 else "#ffb786" if value >= 0.65 else "#ff4d6a")
    target_color = "rgba(173,198,255,0.3)"
    target_arc = circumference * st.session_state.target
    target_gap = circumference - target_arc
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="#262626" stroke-width="8"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="{target_color}" stroke-width="8" stroke-linecap="round"
          stroke-dasharray="{target_arc:.2f} {target_gap:.2f}"
          transform="rotate(-90 {cx} {cy})"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="{color}" stroke-width="8" stroke-linecap="round"
          stroke-dasharray="{arc:.2f} {gap:.2f}"
          transform="rotate(-90 {cx} {cy})"
          filter="drop-shadow(0 0 6px {color})"/>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle" fill="{color}"
          font-family="Geist" font-size="20" font-weight="700">{value:.2f}</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle" fill="#8c909f"
          font-family="JetBrains Mono" font-size="9" letter-spacing="2">AVG RAGAS</text>
      </svg>
      <div style="font-family:'JetBrains Mono';font-size:9px;letter-spacing:.08em;color:#8c909f;margin-top:4px">
        TARGET: {st.session_state.target:.2f}
      </div>
    </div>"""

def _ts():
    return time.strftime("%H:%M:%S")

def _level_class(line):
    l = line.lower()
    if "✅" in line or "complete" in l or "saving" in l or "[ok]" in l: return "tlog-level-ok"
    if "❌" in line or "error" in l or "fail" in l:   return "tlog-level-err"
    if "tuning" in l or "switching" in l:              return "tlog-level-eval"
    if "building" in l or "evaluating" in l:          return "tlog-level-sys"
    if "wait" in l or "listening" in l:               return "tlog-level-wait"
    return "tlog-level-info"

def _render_terminal(logs):
    if not logs:
        rows = '<div class="tlog-row"><span class="tlog-time">--:--:--</span><span class="tlog-level-wait">[WAIT]</span><span class="tlog-msg">LISTENING FOR NEXT EVENT_</span></div>'
    else:
        rows = ""
        for entry in logs[-80:]:          # cap at 80 lines
            ts   = entry.get("ts", "--:--:--")
            line = entry.get("msg", "")
            lvl  = _level_class(line)
            bright = "bright" if ("✅" in line or "[OK]" in line) else ""
            rows += f'<div class="tlog-row"><span class="tlog-time">{ts}</span><span class="{lvl}">[LOG]</span><span class="tlog-msg {bright}">{line}</span></div>'
    return f"""
    <div class="terminal-pane">
      <div class="pane-header">
        <span class="pane-label">⬛ Live Terminal Logs</span>
      </div>
      <div class="term-body" id="term-scroll" style="max-height:220px;overflow-y:auto">
        {rows}
      </div>
    </div>
    <script>
      const t = document.getElementById('term-scroll');
      if(t) t.scrollTop = t.scrollHeight;
    </script>"""

def _add_log(msg: str):
    st.session_state.terminal_logs.append({"ts": _ts(), "msg": msg})


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────────────────────────────────────────
nav_left, nav_mid, nav_right = st.columns([3, 4, 3])

with nav_left:
    st.markdown("""
    <div style="padding:14px 0 0 0;display:flex;align-items:center;gap:16px">
      <span style="font-family:'Geist',sans-serif;font-size:22px;font-weight:900;letter-spacing:-.04em">ASTRA</span>
      <div style="display:flex;align-items:center;gap:6px;padding:4px 12px;
                  background:rgba(74,225,118,.1);border:1px solid rgba(74,225,118,.2);
                  border-radius:999px">
        <div style="width:8px;height:8px;border-radius:50%;background:#4ae176;
                    box-shadow:0 0 8px rgba(74,225,118,.4);
                    animation:pulse 2s cubic-bezier(.4,0,.6,1) infinite"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                     letter-spacing:.05em;color:#4ae176">Live-Tracking</span>
      </div>
    </div>""", unsafe_allow_html=True)

with nav_right:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("AUTONOMOUS", key="btn_mode_a",
                     type="primary" if st.session_state.mode == "A" else "secondary"):
            st.session_state.mode = "A"; st.rerun()
    with c2:
        if st.button("PRODUCTION", key="btn_mode_b",
                     type="primary" if st.session_state.mode == "B" else "secondary"):
            st.session_state.mode = "B"; st.rerun()

st.markdown("<hr style='border-color:var(--outline);margin:0;opacity:.6'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
LEFT, DIVIDER, RIGHT = st.columns([1.2, 0.02, 0.6])

with DIVIDER:
    st.markdown(
        "<div style='background:var(--outline);width:1px;min-height:90vh;margin:0 auto'></div>",
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ═════════════════════════════════════════════════════════════════════════════
with LEFT:
    st.markdown("<div style='padding:24px 28px 0'>", unsafe_allow_html=True)

    # ── Panel header ──────────────────────────────────────────────────────────
    mode_label = "Autonomous mode" if st.session_state.mode == "A" else "Production mode"
    mode_desc  = ("Prototype and refine RAG chains in real-time."
                  if st.session_state.mode == "A"
                  else "Chat with your document using a live-optimized pipeline.")
    st.markdown(f"""
    <div style="margin-bottom:20px">
      <h1 style="font-family:'Geist',sans-serif;font-size:22px;font-weight:600;
                 letter-spacing:-.02em;margin:0 0 4px">{mode_label}</h1>
      <p style="color:var(--on-surf-var);font-size:13px;margin:0">{mode_desc}</p>
    </div>""", unsafe_allow_html=True)

    # ── PDF upload row ────────────────────────────────────────────────────────
    up_col, status_col = st.columns([2, 3])
    with up_col:
        uploaded = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
        if uploaded and uploaded.name != st.session_state.pdf_name:
            with st.spinner("Loading document…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded.read()); tmp = f.name
                st.session_state.docs     = load_documents(tmp)
                st.session_state.pdf_name = uploaded.name
                st.session_state.chain    = None
                _add_log(f"Document loaded: {uploaded.name} ({len(st.session_state.docs)} pages)")
            st.rerun()

    with status_col:
        if st.session_state.pdf_name:
            st.markdown(f"""
            <div style="padding-top:8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                           padding:4px 10px;background:var(--surf-mid);
                           border:1px solid var(--outline);border-radius:4px;color:var(--primary)">
                📄 {st.session_state.pdf_name}
              </span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                           padding:4px 10px;background:var(--surf-mid);
                           border:1px solid var(--outline);border-radius:4px;color:var(--on-surf-var)">
                {len(st.session_state.docs)} pages
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding-top:10px;font-family:'JetBrains Mono',monospace;
                        font-size:9px;letter-spacing:.05em;color:var(--on-surf-var);opacity:.6">
              UPLOAD A PDF TO BEGIN
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE A — AUTONOMOUS OPTIMIZER
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.mode == "A":

        # Config sliders
        with st.expander("⚙  OPTIMIZER CONFIG", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.target     = st.slider("TARGET RAGAS SCORE", 0.5, 1.0, st.session_state.target, 0.05)
                st.session_state.chunk_size = st.slider("STARTING CHUNK SIZE", 300, 1200, st.session_state.chunk_size, 100)
                st.session_state.prompt_v   = st.selectbox("PROMPT VARIANT", ["v1","v2","v3"],
                                                index=["v1","v2","v3"].index(st.session_state.prompt_v))
            with c2:
                st.session_state.max_iter   = st.slider("MAX ITERATIONS", 2, 6, st.session_state.max_iter)
                st.session_state.top_k      = st.slider("STARTING TOP-K", 2, 8, st.session_state.top_k)

            st.markdown(f"""
            <div class="param-grid" style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;
                         margin-top:12px;padding:14px;background:var(--surf-mid);
                         border:1px solid var(--outline);border-radius:6px">
              <div><div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.08em;color:var(--on-surf-var)">CHUNK</div>
                   <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:var(--primary)">{st.session_state.chunk_size}</div></div>
              <div><div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.08em;color:var(--on-surf-var)">TOP-K</div>
                   <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:var(--green)">{st.session_state.top_k}</div></div>
              <div><div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.08em;color:var(--on-surf-var)">TARGET</div>
                   <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:var(--tertiary)">{st.session_state.target}</div></div>
              <div><div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.08em;color:var(--on-surf-var)">PROMPT</div>
                   <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#a78bfa">{st.session_state.prompt_v.upper()}</div></div>
            </div>""", unsafe_allow_html=True)

        # Eval dataset
        with st.expander("📋  VALIDATION DATASET", expanded=False):
            raw_ds = st.text_area("JSON", value=json.dumps(DEFAULT_EVAL_DATASET, indent=2),
                                  height=160, label_visibility="collapsed")

        # Launch button
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        run_disabled = st.session_state.docs is None or st.session_state.running
        launch = st.button(
            "◈  LAUNCH AUTONOMOUS OPTIMIZER",
            disabled=run_disabled,
            use_container_width=True,
        )
        if not st.session_state.docs:
            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        'color:var(--on-surf-var);text-align:center;margin-top:4px;opacity:.5">'
                        'UPLOAD PDF TO ENABLE</div>', unsafe_allow_html=True)

        if launch:
            try:
                eval_ds = json.loads(raw_ds)
            except json.JSONDecodeError:
                st.error("Invalid JSON in dataset."); st.stop()

            st.session_state.terminal_logs = []
            st.session_state.run_history   = []
            st.session_state.final_state   = None
            st.session_state.running       = True

            logs    = st.session_state.terminal_logs
            history = st.session_state.run_history

            def _log_cb(msg):
                logs.append({"ts": _ts(), "msg": msg})

            astra = build_graph(
                st.session_state.docs, eval_ds,
                st.session_state.target, st.session_state.max_iter,
                log_callback=_log_cb, history=history,
            )

            prog_placeholder = st.empty()
            prog_placeholder.progress(0, text="Initializing optimizer…")

            _log_cb("INITIALIZING ASTRA AUTONOMOUS OPTIMIZER...")
            _log_cb(f"Config: chunk={st.session_state.chunk_size}, k={st.session_state.top_k}, "
                    f"prompt={st.session_state.prompt_v}, target={st.session_state.target}")

            # Run — Streamlit reruns after completion
            final = astra.invoke({
                "chunk_size":    st.session_state.chunk_size,
                "chunk_overlap": int(st.session_state.chunk_size * 0.2),
                "top_k":         st.session_state.top_k,
                "prompt_variant": st.session_state.prompt_v,
                "scores": {}, "iteration": 0, "done": False,
            })

            st.session_state.final_state = final
            st.session_state.running     = False
            prog_placeholder.empty()
            _log_cb(f"OPTIMIZER COMPLETE. Best avg={final['scores'].get('avg',0):.4f}")
            st.rerun()

        # ── terminal + score progression in Mode A ────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_render_terminal(st.session_state.terminal_logs), unsafe_allow_html=True)

        if st.session_state.run_history:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="slabel">SCORE PROGRESSION</div>', unsafe_allow_html=True)
            df = pd.DataFrame(st.session_state.run_history)
            chart_df = df.set_index("iteration")[
                [c for c in ["avg","faithfulness","context_recall","context_precision"] if c in df.columns]
            ].copy()
            chart_df["target"] = st.session_state.target
            st.line_chart(chart_df, use_container_width=True, height=160,
                          color=["#adc6ff","#4ae176","#ffb786","#a78bfa","#ff4d6a"])

            # Parameter trail
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="slabel">PARAMETER TRAIL</div>', unsafe_allow_html=True)
            trail_html = '<div style="background:var(--surf-mid);border:1px solid var(--outline);border-radius:8px;padding:4px 14px">'
            for row in st.session_state.run_history:
                avg  = row["avg"]
                c    = _score_color(avg)
                icon = "✅" if avg >= st.session_state.target else "❌"
                trail_html += f"""
                <div class="trail-row">
                  <span class="trail-iter">RUN {row['iteration']}</span>
                  <span class="trail-pills">
                    <span class="tpill">chunk {row['chunk_size']}</span>
                    <span class="tpill">k={row['top_k']}</span>
                    <span class="tpill">{row['prompt_variant']}</span>
                  </span>
                  <span class="trail-score" style="color:{c}">{icon} {avg:.3f}</span>
                </div>"""
            trail_html += "</div>"
            st.markdown(trail_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE B — PRODUCTION CHAT
    # ─────────────────────────────────────────────────────────────────────────
    else:
        # Config
        with st.expander("⚙  PIPELINE CONFIG", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1: st.session_state.chunk_size = st.slider("CHUNK SIZE", 300, 1200, st.session_state.chunk_size, 100)
            with c2: st.session_state.top_k      = st.slider("TOP-K", 2, 8, st.session_state.top_k)
            with c3: st.session_state.prompt_v   = st.selectbox("PROMPT VARIANT", ["v1","v2","v3"],
                                                       index=["v1","v2","v3"].index(st.session_state.prompt_v))
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("↺ REBUILD INDEX"):
                    st.session_state.chain = None
                    _add_log("FAISS index cleared. Will rebuild on next query.")
                    st.success("Index cleared.")
            with rc2:
                if st.button("⌫ CLEAR CHAT"):
                    st.session_state.chat_history = []
                    st.session_state.last_resp    = None
                    st.rerun()

        # Query input
        st.markdown('<div class="slabel" style="margin-top:12px">TEST YOUR RAG PIPELINE</div>',
                    unsafe_allow_html=True)

        # Chat history display
        if st.session_state.chat_history:
            chat_html = '<div style="max-height:280px;overflow-y:auto;padding-right:4px">'
            for m in st.session_state.chat_history:
                if m["role"] == "user":
                    chat_html += f'''
                    <div style="margin:10px 0">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                  letter-spacing:.08em;color:var(--on-surf-var);margin-bottom:4px">YOU</div>
                      <div style="padding:10px 14px;background:var(--surf-mid);
                                  border:1px solid var(--outline);border-radius:8px 8px 2px 8px;
                                  font-size:13px;line-height:1.7">{m["content"]}</div>
                    </div>'''
                else:
                    chat_html += f'''
                    <div style="margin:10px 0">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                  letter-spacing:.08em;color:var(--primary);margin-bottom:4px">◈ ASTRA</div>
                      <div style="padding:10px 14px;background:var(--surf-low);
                                  border:1px solid var(--outline);border-left:2px solid var(--primary);
                                  border-radius:2px 8px 8px 8px;font-size:13px;line-height:1.7">{m["content"]}</div>
                    </div>'''
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

        question = st.chat_input("Ask anything about your document…")
        if question:
            if not st.session_state.docs:
                st.warning("Upload a PDF first."); st.stop()
            if not st.session_state.chain:
                with st.spinner("Building FAISS index…"):
                    _add_log("Building FAISS index...")
                    chain, n = build_rag_pipeline(
                        st.session_state.docs,
                        st.session_state.chunk_size, int(st.session_state.chunk_size * 0.2),
                        st.session_state.top_k, st.session_state.prompt_v,
                    )
                    st.session_state.chain = chain
                    _add_log(f"FAISS index ready. {n} chunks indexed.")
            with st.spinner("Retrieving…"):
                _add_log(f"Query: {question[:60]}...")
                resp = run_rag(question, st.session_state.chain)
                _add_log(f"[OK] QUERY COMPLETE. LATENCY: {resp['latency']}s | TOKENS: {resp['total_tokens']}")
            st.session_state.last_resp = resp
            st.session_state.chat_history.append({"role":"user",   "content": question})
            st.session_state.chat_history.append({"role":"astra",  "content": resp["answer"]})
            st.rerun()

        # Terminal below chat
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_render_terminal(st.session_state.terminal_logs), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — OBSERVABILITY
# ═════════════════════════════════════════════════════════════════════════════
with RIGHT:
    st.markdown("""
    <div style="padding:20px 20px 0">
      <div class="slabel" style="font-size:11px;margin-bottom:16px">
        ◈ OBSERVABILITY
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 20px'>", unsafe_allow_html=True)

    # ── MODE A — Final scores after optimizer run ─────────────────────────────
    if st.session_state.mode == "A":
        fs = st.session_state.final_state

        if fs:
            scores = fs["scores"]
            avg    = scores.get("avg", 0)
            passed = avg >= st.session_state.target

            # Gauge
            st.markdown(f"""
            <div class="obs-card" style="text-align:center;margin-bottom:12px">
              <div class="gauge-title">QUALITY SCORE</div>
              {_gauge_svg(avg)}
            </div>""", unsafe_allow_html=True)

            # Key metric cards
            status_color = "var(--green)" if passed else "var(--error)"
            status_txt   = "TARGET MET" if passed else "BELOW TARGET"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
              <div class="obs-card">
                <div class="obs-label">CHUNK SIZE</div>
                <div class="obs-val" style="font-size:26px;color:var(--primary)">{fs['chunk_size']}</div>
              </div>
              <div class="obs-card">
                <div class="obs-label">TOP-K</div>
                <div class="obs-val" style="font-size:26px;color:var(--green)">{fs['top_k']}</div>
              </div>
              <div class="obs-card">
                <div class="obs-label">PROMPT</div>
                <div class="obs-val" style="font-size:26px;color:#a78bfa">{fs['prompt_variant'].upper()}</div>
              </div>
              <div class="obs-card">
                <div class="obs-label">STATUS</div>
                <div class="obs-val" style="font-size:15px;color:{status_color};margin-top:4px">{status_txt}</div>
                <div class="obs-sub">{fs['iteration']} iterations</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Score bars
            st.markdown('<div class="slabel" style="margin-bottom:8px">FINAL RAGAS SCORES</div>',
                        unsafe_allow_html=True)
            bars_html = '<div class="obs-card">'
            for k, v in scores.items():
                if k != "avg":
                    bars_html += _sbar(k, v)
            bars_html += f"""<div style="border-top:1px solid var(--outline);margin-top:8px;padding-top:8px">
                {_sbar("Average Score", avg)}</div></div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

        else:
            # Placeholder when no run yet
            st.markdown(f"""
            <div class="obs-card" style="text-align:center;padding:36px 0;margin-bottom:12px">
              <div style="font-size:32px;opacity:.15;margin-bottom:10px">◈</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                          letter-spacing:.08em;color:var(--on-surf-var);opacity:.6">
                CONFIGURE AND LAUNCH<br>TO SEE RESULTS
              </div>
            </div>""", unsafe_allow_html=True)

            # Static placeholder gauge
            st.markdown(f"""
            <div class="obs-card" style="text-align:center">
              <div class="gauge-title">QUALITY SCORE</div>
              {_gauge_svg(0.0)}
            </div>""", unsafe_allow_html=True)

    # ── MODE B — Live telemetry ───────────────────────────────────────────────
    else:
        resp = st.session_state.last_resp

        if resp:
            # Latency
            st.markdown(f"""
            <div class="obs-card" style="margin-bottom:10px">
              <div class="obs-label">Execution Latency</div>
              <div class="obs-val">{resp['latency']}<span class="obs-val-unit">s</span></div>
            </div>""", unsafe_allow_html=True)

            # Tokens
            st.markdown(f"""
            <div class="obs-card" style="margin-bottom:10px">
              <div class="obs-label">Total Token Usage</div>
              <div style="display:flex;align-items:baseline;gap:10px">
                <div class="obs-val">{resp['total_tokens']}</div>
                <span style="color:var(--on-surf-var);font-size:12px">
                  Prompt: {resp['input_tokens']} / Comp: {resp['output_tokens']}
                </span>
              </div>
              <!-- token split bar -->
              <div style="height:4px;border-radius:2px;overflow:hidden;
                          background:var(--surf-highest);display:flex;margin-top:10px">
                <div style="width:{int(resp['input_tokens']/max(resp['total_tokens'],1)*100)}%;
                            background:var(--primary)"></div>
                <div style="flex:1;background:var(--green)"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:5px;
                          font-family:'JetBrains Mono',monospace;font-size:9px">
                <span style="color:var(--primary)">INPUT</span>
                <span style="color:var(--green)">OUTPUT</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Cost
            st.markdown(f"""
            <div class="obs-card" style="margin-bottom:10px">
              <div class="obs-label">Estimated Cost</div>
              <div style="font-size:20px;font-weight:600;color:var(--on-surf-var);margin-top:4px">
                ${resp['cost_usd']} <span style="font-size:12px">USD</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Chunks count
            st.markdown(f"""
            <div class="obs-card" style="margin-bottom:16px">
              <div class="obs-label">FAISS Chunks Retrieved</div>
              <div class="obs-val" style="font-size:26px;color:var(--green)">{len(resp['contexts'])}</div>
            </div>""", unsafe_allow_html=True)

            # Sub-queries
            st.markdown('<div class="slabel">REWRITER SUB-QUERIES</div>', unsafe_allow_html=True)
            sq_html = '<div class="obs-card" style="margin-bottom:12px">'
            for i, q in enumerate(resp.get("sub_queries", []), 1):
                sq_html += f"""
                <div style="display:flex;gap:10px;padding:8px 0;
                            border-bottom:1px solid var(--outline);
                            font-size:12px;align-items:flex-start">
                  <span style="font-family:'JetBrains Mono',monospace;
                               color:var(--green);font-size:10px;min-width:16px;padding-top:2px">{i}.</span>
                  <span style="color:var(--on-surf-var);line-height:1.6">{q}</span>
                </div>"""
            sq_html += "</div>"
            st.markdown(sq_html, unsafe_allow_html=True)

            # Source contexts accordion
            st.markdown('<div class="slabel">SOURCE CONTEXTS (FAISS)</div>', unsafe_allow_html=True)
            for i, ctx in enumerate(resp["contexts"], 1):
                with st.expander(f"Chunk {i}: {ctx[:40].strip()}…"):
                    st.markdown(f"""
                    <div style="font-size:12px;line-height:1.7;
                                color:var(--on-surf-var);padding:4px 0">{ctx}</div>
                    """, unsafe_allow_html=True)

            # Raw payload
            with st.expander("{ }  RAW BACKEND PAYLOAD"):
                payload = {
                    "question":  resp["question"],
                    "answer":    resp["answer"],
                    "telemetry": {
                        "latency_s":     resp["latency"],
                        "input_tokens":  resp["input_tokens"],
                        "output_tokens": resp["output_tokens"],
                        "total_tokens":  resp["total_tokens"],
                        "cost_usd":      resp["cost_usd"],
                    },
                    "sub_queries":  resp["sub_queries"],
                    "num_contexts": len(resp["contexts"]),
                }
                st.markdown(f"""
                <div style="background:#020408;border:1px solid var(--outline);
                            border-radius:6px;padding:12px 14px;
                            font-family:'JetBrains Mono',monospace;font-size:11px;
                            color:#79c0ff;white-space:pre;overflow-x:auto;max-height:220px;overflow-y:auto">
                {json.dumps(payload, indent=2)}</div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="padding:60px 0;text-align:center">
              <div style="font-size:36px;opacity:.12;margin-bottom:12px">◈</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                          letter-spacing:.08em;color:var(--on-surf-var);opacity:.5">
                ASK A QUESTION TO SEE TELEMETRY
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)