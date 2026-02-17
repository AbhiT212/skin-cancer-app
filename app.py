import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import io
import math
from inference import run_inference
from config import LOC_COLS

def get_meta(class_name):
    """Case-insensitive lookup into CLASS_META."""
    return CLASS_META.get(class_name.lower(), {"color": "#64748b", "glow": "rgba(100,116,139,0.2)", "risk": "Unknown", "icon": "○"})

def get_desc(class_name):
    return CLASS_DESC.get(class_name.lower(), "Consult a dermatologist.")

def get_risk_style(risk):
    return RISK_STYLE.get(risk, {"bg": "rgba(100,116,139,0.1)", "border": "rgba(100,116,139,0.3)", "color": "#64748b"})
# ─────────────────────────────────────────────────────────────
#  App init
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap",
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"},
    ],
)
server = app.server
app.title = "DermAI — Diagnostic Intelligence"

# ─────────────────────────────────────────────────────────────
#  Data constants
# ─────────────────────────────────────────────────────────────
loc_options = [
    loc.replace("localization_", "").replace("_", " ").title() for loc in LOC_COLS
]

CLASS_META = {
    "melanoma":             {"color": "#ff4d6d", "glow": "rgba(255,77,109,0.35)", "risk": "High",          "icon": "⬟"},
    "melanocytic nevi":     {"color": "#00d9a3", "glow": "rgba(0,217,163,0.3)",   "risk": "Low",           "icon": "◉"},
    "basal cell carcinoma": {"color": "#ff8c42", "glow": "rgba(255,140,66,0.32)", "risk": "Moderate-High", "icon": "◈"},
    "actinic keratoses":    {"color": "#ffd166", "glow": "rgba(255,209,102,0.3)", "risk": "Moderate",      "icon": "◇"},
    "benign keratosis":     {"color": "#DADAD6", "glow": "rgba(56,189,248,0.28)", "risk": "Low",           "icon": "○"},
    "dermatofibroma":       {"color": "#4cc9f0", "glow": "rgba(76,201,240,0.3)",  "risk": "Low",           "icon": "◎"},
    "vascular lesions":     {"color": "#b185db", "glow": "rgba(177,133,219,0.3)", "risk": "Low-Moderate",  "icon": "◐"},
}

CLASS_DESC = {
    "melanoma":             "Malignant melanocytic tumour. Urgent specialist review required.",
    "melanocytic nevi":     "Common benign mole. Monitor for ABCDE changes.",
    "basal cell carcinoma": "Most common skin cancer. Requires clinical treatment.",
    "actinic keratoses":    "Pre-cancerous UV-induced lesion. Proactive treatment advised.",
    "benign keratosis":     "Non-cancerous growth. Usually harmless.",
    "dermatofibroma":       "Benign fibrous nodule. Generally harmless.",
    "vascular lesions":     "Includes haemangiomas and angiomas. Usually benign.",
}

RISK_STYLE = {
    "High":          {"bg": "rgba(255,77,109,0.15)",  "border": "rgba(255,77,109,0.5)",  "color": "#ff4d6d"},
    "Moderate-High": {"bg": "rgba(255,140,66,0.15)",  "border": "rgba(255,140,66,0.5)",  "color": "#ff8c42"},
    "Moderate":      {"bg": "rgba(255,209,102,0.15)", "border": "rgba(255,209,102,0.5)", "color": "#ffd166"},
    "Low-Moderate":  {"bg": "rgba(76,201,240,0.12)",  "border": "rgba(76,201,240,0.4)",  "color": "#4cc9f0"},
    "Low":           {"bg": "rgba(0,217,163,0.12)",   "border": "rgba(0,217,163,0.4)",   "color": "#00d9a3"},
}


# ─────────────────────────────────────────────────────────────
#  CSS (loaded via assets/custom.css OR app.index_string)
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════════
   DERM AI — FUTURISTIC DARK UI
═══════════════════════════════════════════════════════ */
:root {
  --bg:         #050a14;
  --bg-1:       #080e1c;
  --bg-2:       #0c1526;
  --bg-3:       #111e35;
  --border:     rgba(255,255,255,0.07);
  --border-hi:  rgba(76,201,240,0.25);
  --cyan:       #4cc9f0;
  --violet:     #7c3aed;
  --pink:       #ff4d6d;
  --green:      #00d9a3;
  --amber:      #ffd166;
  --text:       #e2eaf8;
  --muted:      #4a6080;
  --font:       'Inter', sans-serif;
  --mono:       'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

body {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(76,201,240,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.05) 0%, transparent 50%) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: 14px;
  line-height: 1.55;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

/* ══════════════════════════════════════════════════
   HEADER
══════════════════════════════════════════════════ */
.ai-header {
  position: sticky; top: 0; z-index: 1000;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 32px;
  height: 62px;
  background: rgba(5,10,20,0.85);
  backdrop-filter: blur(20px) saturate(180%);
  border-bottom: 1px solid var(--border);
  gap: 16px;
}

.ai-header::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
  opacity: 0.5;
}

.hdr-brand {
  display: flex; align-items: center; gap: 14px;
}

.hdr-logo-wrap {
  position: relative; width: 40px; height: 40px;
}

.hdr-logo {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, rgba(76,201,240,0.2), rgba(124,58,237,0.3));
  border: 1px solid rgba(76,201,240,0.3);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 19px;
  box-shadow: 0 0 24px rgba(76,201,240,0.2), inset 0 1px 0 rgba(255,255,255,0.08);
}

.hdr-text-block { display: flex; flex-direction: column; }
.hdr-title { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.3px; color: var(--text); }
.hdr-sub   { font-size: 0.63rem; color: var(--muted); letter-spacing: 1.2px; text-transform: uppercase; margin-top: 1px; }

.hdr-right { display: flex; align-items: center; gap: 10px; }

.status-dot {
  display: flex; align-items: center; gap: 6px;
  background: rgba(0,217,163,0.08);
  border: 1px solid rgba(0,217,163,0.25);
  border-radius: 99px; padding: 4px 12px;
  font-size: 0.65rem; font-weight: 600; color: var(--green);
  letter-spacing: 0.5px; text-transform: uppercase;
}

.status-dot::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--green);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--green);
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

.model-tag {
  background: rgba(76,201,240,0.07);
  border: 1px solid rgba(76,201,240,0.2);
  border-radius: 99px; padding: 4px 12px;
  font-size: 0.63rem; font-weight: 500; color: var(--cyan);
  letter-spacing: 0.4px; text-transform: uppercase;
  font-family: var(--mono);
}

/* ══════════════════════════════════════════════════
   PAGE WRAP
══════════════════════════════════════════════════ */
.page-wrap {
  max-width: 1420px;
  margin: 0 auto;
  padding: 28px 20px 48px;
}

/* ══════════════════════════════════════════════════
   GLASS PANELS
══════════════════════════════════════════════════ */
.glass-panel {
  background: rgba(8,14,28,0.7);
  border: 1px solid var(--border);
  border-radius: 16px;
  backdrop-filter: blur(12px);
  overflow: hidden;
  position: relative;
  transition: border-color 0.3s;
}

.glass-panel::before {
  content: '';
  position: absolute;
  inset: 0; border-radius: 16px;
  background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
  pointer-events: none;
}

.panel-hdr {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}

.panel-hdr-icon {
  width: 28px; height: 28px;
  background: rgba(76,201,240,0.1);
  border: 1px solid rgba(76,201,240,0.22);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; flex-shrink: 0;
  box-shadow: 0 0 12px rgba(76,201,240,0.12);
}

.panel-hdr-title {
  font-size: 0.7rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--muted);
}

.panel-body { padding: 20px; }

/* ══════════════════════════════════════════════════
   FORM ELEMENTS
══════════════════════════════════════════════════ */
.field-label {
  display: block;
  font-size: 0.63rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 6px;
}

input[type="number"] {
  width: 100% !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 9px !important;
  padding: 9px 13px !important;
  font-size: 0.85rem !important;
  font-family: var(--mono) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  margin-bottom: 16px;
  outline: none !important;
}

input[type="number"]:focus {
  border-color: rgba(76,201,240,0.5) !important;
  box-shadow: 0 0 0 3px rgba(76,201,240,0.08), 0 0 16px rgba(76,201,240,0.08) !important;
}

/* ── Dropdown (react-select via Dash) ── */
/* ── Dropdown (react-select via Dash) ── */
.Select-control {
  background: rgba(12, 21, 38, 0.95) !important;
  border: 1px solid var(--border) !important;
  border-radius: 9px !important;
  min-height: 40px !important;
  cursor: pointer !important;
  transition: border-color 0.2s !important;
}
.Select-control:hover { border-color: rgba(76,201,240,0.3) !important; }

.Select-menu-outer {
  background: #0c1526 !important;
  border: 1px solid rgba(76,201,240,0.2) !important;
  border-radius: 10px !important;
  box-shadow: 0 16px 48px rgba(0,0,0,0.8) !important;
  z-index: 9999 !important;
  overflow: hidden !important;
}

.Select-menu { background: #0c1526 !important; }

.Select-option {
  background: #0c1526 !important;
  color: #8aa0be !important;
  font-size: 0.83rem !important;
  padding: 9px 14px !important;
  transition: background 0.15s, color 0.15s !important;
}

.Select-option.is-focused, .Select-option:hover {
  background: rgba(76,201,240,0.1) !important;
  color: var(--cyan) !important;
}

.Select-option.is-selected {
  background: rgba(76,201,240,0.15) !important;
  color: var(--text) !important;
}

.Select-value          { background: transparent !important; }
.Select-value-label    { color: var(--text) !important; font-size: 0.85rem !important; }
.Select-placeholder    { color: var(--muted) !important; font-size: 0.85rem !important; }
.Select-arrow          { border-top-color: var(--muted) !important; }
.Select-input          { background: transparent !important; }
.Select-input input    { color: var(--text) !important; font-family: var(--font) !important; background: transparent !important; }
.VirtualizedSelectFocusedOption { background: rgba(76,201,240,0.1) !important; color: var(--cyan) !important; }
.VirtualizedSelectOption { background: #0c1526 !important; color: #8aa0be !important; }


/* ── Native select (dbc.Select) ── */
select {
  background: #0c1526 !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  color: #e2eaf8 !important;
  border-radius: 9px !important;
  padding: 9px 13px !important;
  font-size: 0.85rem !important;
  font-family: 'JetBrains Mono', monospace !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  outline: none !important;
  -webkit-appearance: auto !important;
}

select:focus {
  border-color: rgba(76,201,240,0.5) !important;
  box-shadow: 0 0 0 3px rgba(76,201,240,0.08) !important;
}

select option {
  background: #0c1526 !important;
  color: #e2eaf8 !important;
}
/* ── Upload drop zone ── */
.drop-zone {
  border: 2px dashed rgba(76,201,240,0.2) !important;
  border-radius: 12px !important;
  background: rgba(76,201,240,0.03) !important;
  padding: 28px 16px !important;
  text-align: center !important;
  cursor: pointer !important;
  transition: all 0.25s !important;
  position: relative;
  overflow: hidden;
}

.drop-zone::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at center, rgba(76,201,240,0.05), transparent 70%);
  opacity: 0; transition: opacity 0.25s;
  pointer-events: none;
}

.drop-zone:hover {
  border-color: rgba(76,201,240,0.5) !important;
  background: rgba(76,201,240,0.06) !important;
  box-shadow: 0 0 24px rgba(76,201,240,0.06) !important;
}
.drop-zone:hover::before { opacity: 1; }

.dz-icon   { font-size: 28px; color: var(--cyan); margin-bottom: 8px; opacity: 0.7; }
.dz-title  { font-size: 0.82rem; color: var(--muted); }
.dz-link   { color: var(--cyan); font-weight: 600; }
.dz-hint   { font-size: 0.63rem; color: var(--muted); margin-top: 4px; opacity: 0.6; }

/* ── Run button ── */
.run-btn {
  width: 100%;
  position: relative;
  background: linear-gradient(135deg, rgba(76,201,240,0.15), rgba(124,58,237,0.2)) !important;
  border: 1px solid rgba(76,201,240,0.3) !important;
  border-radius: 10px !important;
  padding: 12px !important;
  font-size: 0.83rem !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  color: var(--cyan) !important;
  cursor: pointer !important;
  overflow: hidden;
  transition: all 0.25s !important;
  box-shadow: 0 0 20px rgba(76,201,240,0.08), inset 0 1px 0 rgba(255,255,255,0.06) !important;
  margin-top: 8px;
}

.run-btn::before {
  content: '';
  position: absolute; top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(76,201,240,0.12), transparent);
  transition: left 0.5s;
}

.run-btn:hover {
  border-color: rgba(76,201,240,0.6) !important;
  box-shadow: 0 0 32px rgba(76,201,240,0.18), inset 0 1px 0 rgba(255,255,255,0.1) !important;
  color: #fff !important;
}
.run-btn:hover::before { left: 100%; }

/* ── Spec chips ── */
.spec-row  { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 8px; }
.spec-chip {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 8px; text-align: center;
  transition: border-color 0.2s;
}
.spec-chip:hover { border-color: var(--border-hi); }
.spec-val { font-size: 0.95rem; font-weight: 800; color: var(--text); line-height: 1.1; font-family: var(--mono); }
.spec-lbl { font-size: 0.58rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 3px; }

/* ══════════════════════════════════════════════════
   RESULTS — PREDICTION BANNER
══════════════════════════════════════════════════ */
.pred-card {
  border-radius: 14px;
  padding: 20px 22px;
  display: flex; align-items: center;
  justify-content: space-between;
  flex-wrap: wrap; gap: 14px;
  margin-bottom: 20px;
  position: relative; overflow: hidden;
}

.pred-card::after {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 160px; height: 160px;
  border-radius: 50%;
  filter: blur(50px);
  opacity: 0.25;
  pointer-events: none;
}

.pred-eyebrow {
  font-size: 0.62rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  opacity: 0.6; margin-bottom: 4px;
}
.pred-name  { font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.15; }
.pred-desc  { font-size: 0.76rem; opacity: 0.65; margin-top: 5px; max-width: 400px; line-height: 1.5; }

.risk-tag {
  display: inline-flex; align-items: center; gap: 5px;
  border-radius: 6px; padding: 3px 10px;
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.5px; text-transform: uppercase;
  margin-top: 8px;
}

.conf-ring {
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  flex-shrink: 0;
}
.conf-num { font-size: 2.2rem; font-weight: 800; line-height: 1; font-family: var(--mono); }
.conf-lbl { font-size: 0.58rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

/* ══════════════════════════════════════════════════
   UNCERTAINTY NOTICE
══════════════════════════════════════════════════ */
.unc-bar {
  background: rgba(255,209,102,0.06);
  border: 1px solid rgba(255,209,102,0.25);
  border-left: 3px solid var(--amber);
  border-radius: 0 9px 9px 0;
  padding: 10px 14px;
  font-size: 0.75rem;
  color: var(--amber);
  display: flex; align-items: flex-start; gap: 9px;
  margin-bottom: 16px; line-height: 1.45;
}

.unc-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }

/* ══════════════════════════════════════════════════
   IMAGE PANELS
══════════════════════════════════════════════════ */
.img-card {
  border-radius: 12px; overflow: hidden;
  border: 1px solid var(--border);
  background: var(--bg-2);
  transition: border-color 0.25s;
}
.img-card:hover { border-color: var(--border-hi); }

.img-card-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 9px 14px;
  background: rgba(255,255,255,0.025);
  border-bottom: 1px solid var(--border);
}

.img-card-lbl {
  font-size: 0.63rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1px;
  color: var(--muted);
}

.img-card-badge {
  font-size: 0.58rem; font-weight: 600;
  padding: 2px 7px; border-radius: 4px;
  background: rgba(76,201,240,0.1);
  border: 1px solid rgba(76,201,240,0.25);
  color: var(--cyan); letter-spacing: 0.4px;
}

.img-card img {
  width: 100%;
  height: auto;
  max-height: 340px;
  object-fit: contain;
  display: block;
  background: #080e1c;
  padding: 8px;
}

.cam-badge {
  background: rgba(255,77,109,0.12) !important;
  border-color: rgba(255,77,109,0.3) !important;
  color: #ff4d6d !important;
}

/* ══════════════════════════════════════════════════
   SECTION / BOX
══════════════════════════════════════════════════ */
.sec-lbl {
  font-size: 0.62rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.1px;
  color: var(--muted); margin-bottom: 10px;
}

.data-box {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  height: 100%;
  transition: border-color 0.25s;
}
.data-box:hover { border-color: var(--border-hi); }

/* ── Horizontal divider ── */
.hdiv {
  border: 0;
  border-top: 1px solid var(--border);
  margin: 14px 0;
}

/* ══════════════════════════════════════════════════
   PROBABILITY LIST
══════════════════════════════════════════════════ */
.prob-item {
  display: flex; align-items: center; gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.prob-item:last-child { border-bottom: none; }

.prob-dot   { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.prob-label { font-size: 0.76rem; color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob-track { flex: 2; height: 3px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 99px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1); }
.prob-pct   { font-size: 0.68rem; font-weight: 600; color: var(--muted); width: 38px; text-align: right; font-family: var(--mono); }

/* ══════════════════════════════════════════════════
   ENTROPY / PATIENT SUMMARY
══════════════════════════════════════════════════ */
.ent-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.ent-label { font-size: 0.72rem; color: var(--muted); }
.ent-val   { font-size: 0.78rem; font-weight: 700; font-family: var(--mono); }

.ent-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 99px; height: 4px; overflow: hidden; margin-bottom: 10px; }
.ent-bar-fill { height: 100%; border-radius: 99px; transition: width 0.7s ease; }

.patient-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.pt-chip {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 9px 10px; text-align: center;
}
.pt-chip.full { grid-column: span 2; }
.pt-val { font-size: 0.95rem; font-weight: 700; color: var(--text); font-family: var(--mono); }
.pt-lbl { font-size: 0.58rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.7px; margin-top: 2px; }

/* ══════════════════════════════════════════════════
   PLACEHOLDER
══════════════════════════════════════════════════ */
.empty-state {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: 380px; gap: 14px; color: var(--muted);
}
.empty-icon { font-size: 42px; opacity: 0.12; }
.empty-title { font-size: 0.88rem; font-weight: 600; opacity: 0.35; }
.empty-sub   { font-size: 0.73rem; opacity: 0.25; text-align: center; max-width: 220px; line-height: 1.5; }

/* ══════════════════════════════════════════════════
   DISCLAIMER
══════════════════════════════════════════════════ */
.disclaimer {
  background: rgba(255,77,109,0.04);
  border: 1px solid rgba(255,77,109,0.15);
  border-left: 3px solid rgba(255,77,109,0.6);
  border-radius: 0 10px 10px 0;
  padding: 14px 18px;
  margin-top: 28px;
}
.dis-hdr { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.dis-title { font-size: 0.65rem; font-weight: 700; color: rgba(255,77,109,0.8); text-transform: uppercase; letter-spacing: 1px; }
.dis-body  { font-size: 0.74rem; color: var(--muted); line-height: 1.6; }

/* ══════════════════════════════════════════════════
   PLOTLY OVERRIDES
══════════════════════════════════════════════════ */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
.js-plotly-plot { background: transparent !important; }
.modebar { display: none !important; }

/* ══════════════════════════════════════════════════
   DASH LOADING
══════════════════════════════════════════════════ */
._dash-loading { background: transparent !important; }
.dash-spinner circle { stroke: var(--cyan) !important; }

/* ══════════════════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════════════════ */
@media (max-width: 767px) {
  .ai-header { padding: 0 16px; }
  .hdr-right { display: none; }
  .page-wrap { padding: 16px 12px 36px; }
  .pred-card { flex-direction: column; }
  .img-card img { height: 170px; }
}
@media (max-width: 480px) {
  .pred-name { font-size: 1.1rem; }
  .conf-num  { font-size: 1.7rem; }
  .spec-row  { gap: 5px; }
}
"""

# Inject CSS via index_string
app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CUSTOM_CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""


# ─────────────────────────────────────────────────────────────
#  Placeholder
# ─────────────────────────────────────────────────────────────
def _empty_state():
    return html.Div(className="empty-state", children=[
        html.Div("◎", className="empty-icon"),
        html.P("No analysis yet", className="empty-title"),
        html.P("Upload a dermoscopic image and complete patient metadata, then run analysis.", className="empty-sub"),
    ])


# ─────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"minHeight": "100vh"},
    children=[

        # ── Header ──────────────────────────────────────────
        html.Header(className="ai-header", children=[
            html.Div(className="hdr-brand", children=[
                html.Div(className="hdr-logo-wrap", children=[
                    html.Div("◈", className="hdr-logo"),
                ]),
                html.Div(className="hdr-text-block", children=[
                    html.Span("DermAI", className="hdr-title"),
                    html.Span("Skin Lesion Intelligence Platform", className="hdr-sub"),
                ]),
            ]),
            html.Div(className="hdr-right", children=[
                html.Div("Model Online", className="status-dot"),
                html.Div("EfficientNet · TTA · GradCAM XAI", className="model-tag"),
            ]),
        ]),

        # ── Page ────────────────────────────────────────────
        html.Div(className="page-wrap", children=[
            dbc.Row(className="g-3", children=[

                # ── LEFT — Input panel ───────────────────────
                dbc.Col(md=4, lg=3, children=[
                    html.Div(className="glass-panel", children=[
                        html.Div(className="panel-hdr", children=[
                            html.Div("⊕", className="panel-hdr-icon"),
                            html.P("Patient Input", className="panel-hdr-title"),
                        ]),
                        html.Div(className="panel-body", children=[

                            html.Label("Patient Age", className="field-label"),
                            dbc.Input(id="input-age", type="number", value=45, min=0, max=120,
                                      style={"background": "rgba(255,255,255,0.04)",
                                             "border": "1px solid rgba(255,255,255,0.07)",
                                             "color": "#e2eaf8", "borderRadius": "9px",
                                             "marginBottom": "16px", "fontFamily": "JetBrains Mono, monospace"}),
                            html.Label("Biological Sex", className="field-label"),
                            dbc.Select(
                                id="input-sex",
                                options=[{"label": "Male", "value": "male"},
                                        {"label": "Female", "value": "female"}],
                                value="male",
                                style={
                                    "background": "#0c1526",
                                    "border": "1px solid rgba(255,255,255,0.07)",
                                    "color": "#e2eaf8",
                                    "borderRadius": "9px",
                                    "padding": "9px 13px",
                                    "fontSize": "0.85rem",
                                    "fontFamily": "JetBrains Mono, monospace",
                                    "marginBottom": "16px",
                                    "cursor": "pointer",
                                    "appearance": "auto",
                                    "width": "100%",
                                },
                            ),

                            html.Label("Lesion Location", className="field-label"),
                            dbc.Select(
                                id="input-loc",
                                options=[{"label": l, "value": l.lower()} for l in loc_options],
                                value="back",
                                style={
                                    "background": "#0c1526",
                                    "border": "1px solid rgba(255,255,255,0.07)",
                                    "color": "#e2eaf8",
                                    "borderRadius": "9px",
                                    "padding": "9px 13px",
                                    "fontSize": "0.85rem",
                                    "fontFamily": "JetBrains Mono, monospace",
                                    "marginBottom": "16px",
                                    "cursor": "pointer",
                                    "appearance": "auto",
                                    "width": "100%",
                                },
                            ),

                            html.Label("Dermoscopic Image", className="field-label"),
                            dcc.Upload(
                                id="upload-image",
                                children=html.Div(className="drop-zone", children=[
                                    html.Div("⊡", className="dz-icon"),
                                    html.P(["Drag & drop or ", html.Span("select file", className="dz-link")], className="dz-title"),
                                    html.P("PNG · JPG · JPEG — max 10 MB", className="dz-hint"),
                                ]),
                                multiple=False,
                                style={"marginBottom": "6px"},
                            ),
                            html.Div(id="upload-status",
                                     style={"fontSize": "0.7rem", "color": "#00d9a3",
                                            "minHeight": "18px", "marginBottom": "10px",
                                            "fontFamily": "JetBrains Mono, monospace"}),

                            html.Button("⟶  Run Analysis", id="submit-button",
                                        className="run-btn", n_clicks=0),

                            html.Hr(className="hdiv", style={"margin": "20px 0 14px"}),

                            html.P("Architecture", className="field-label"),
                            html.Div(className="spec-row", children=[
                                html.Div(className="spec-chip", children=[
                                    html.Div("7", className="spec-val"),
                                    html.Div("Classes", className="spec-lbl"),
                                ]),
                                html.Div(className="spec-chip", children=[
                                    html.Div("×4", className="spec-val"),
                                    html.Div("TTA Views", className="spec-lbl"),
                                ]),
                                html.Div(className="spec-chip", children=[
                                    html.Div("CAM", className="spec-val"),
                                    html.Div("XAI", className="spec-lbl"),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),

                # ── RIGHT — Results panel ────────────────────
                dbc.Col(md=8, lg=9, children=[
                    html.Div(className="glass-panel", style={"height": "100%"}, children=[
                        html.Div(className="panel-hdr", children=[
                            html.Div("◉", className="panel-hdr-icon"),
                            html.P("Diagnostic Output", className="panel-hdr-title"),
                        ]),
                        html.Div(className="panel-body", children=[
                            dcc.Loading(
                                type="circle",
                                color="#4cc9f0",
                                children=html.Div(id="results-container", children=_empty_state()),
                            ),
                        ]),
                    ]),
                ]),
            ]),

            # ── Disclaimer ───────────────────────────────────
            html.Div(className="disclaimer", children=[
                html.Div(className="dis-hdr", children=[
                    html.Span("⚑"),
                    html.Span("Medical Disclaimer", className="dis-title"),
                ]),
                html.P(
                    "This platform is intended solely for research and educational purposes. It does not constitute "
                    "medical advice, diagnosis, or treatment. Artificial intelligence models can produce inaccurate "
                    "or misleading outputs. Always consult a qualified dermatologist or licensed healthcare "
                    "professional for any skin concern. Do not delay or disregard professional medical advice "
                    "based on the output of this system. If you are concerned about a skin lesion, seek an "
                    "in-person clinical evaluation promptly.",
                    className="dis-body",
                ),
            ]),
        ]),
    ],
)


# ─────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("upload-status", "children"),
    Input("upload-image", "filename"),
)
def cb_upload_status(filename):
    return f"✓  {filename}" if filename else ""


@app.callback(
    Output("results-container", "children"),
    Input("submit-button", "n_clicks"),
    State("upload-image", "contents"),
    State("input-age", "value"),
    State("input-sex", "value"),
    State("input-loc", "value"),
    prevent_initial_call=True,
)
def cb_run_inference(n_clicks, image_contents, age, sex, loc):
    if image_contents is None:
        return html.Div(className="unc-bar", children=[
            html.Span("⚠", className="unc-icon"),
            html.Span("No image uploaded. Please select a dermoscopic image before running analysis."),
        ])

    try:
        _, content_string = image_contents.split(",")
        image_bytes = io.BytesIO(base64.b64decode(content_string))
        results = run_inference(image_bytes, age, sex, loc)

        top     = results["top_prediction"]
        conf    = results["top_confidence"]
        margin  = results["margin"]
        is_unc  = results["is_uncertain"]
        classes = results["classes"]
        probs   = results["probabilities"]

        meta  = get_meta(top)
        desc  = get_desc(top)
        risk  = meta["risk"]
        color = meta["color"]
        glow  = meta["glow"]
        rs    = get_risk_style(risk)

        # ── Sorted pairs ──────────────────────────────────────
        pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        prob_items = [
            html.Div(className="prob-item", children=[
                html.Div(className="prob-dot",
                        style={"background": get_meta(c)["color"],
                                "boxShadow": f"0 0 6px {get_meta(c)['glow']}"}),
                html.Span(c, className="prob-label"),
                html.Div(className="prob-track", children=[
                    html.Div(className="prob-fill", style={
                        "width": f"{p * 100:.1f}%",
                        "background": get_meta(c)["color"],
                    }),
                ]),
                html.Span(f"{p * 100:.1f}%", className="prob-pct"),
            ])
            for c, p in pairs
]

        # ── Horizontal bar chart ───────────────────────────────
        b_colors = [get_meta(c)["color"] for c in classes]
        bar_fig = go.Figure(go.Bar(
            x=probs, y=classes, orientation="h",
            marker=dict(color=b_colors, opacity=0.78, line=dict(width=0)),
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(size=9, color="#4a6080", family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>%{x:.2%}<extra></extra>",
        ))
        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 1.14], tickformat=".0%", showgrid=True,
                       gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                       tickfont=dict(size=8, color="#4a6080", family="JetBrains Mono"),
                       showline=False, tickcolor="rgba(0,0,0,0)"),
            yaxis=dict(categoryorder="total ascending",
                       tickfont=dict(size=8, color="#6a8aaa"), showgrid=False,
                       tickcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=10, t=6, b=6),
            height=220, showlegend=False,
            font=dict(family="Inter"),
        )

        # ── Confidence donut ───────────────────────────────────
        top3   = [p for _, p in pairs[:3]]
        rest   = max(0.0, 1.0 - sum(top3))
        d_clrs = [
            get_meta(pairs[0][0])["color"],
            get_meta(pairs[1][0])["color"],
            get_meta(pairs[2][0])["color"],
            "rgba(255,255,255,0.04)",
        ]
        donut_fig = go.Figure(go.Pie(
            labels=[pairs[0][0], pairs[1][0], pairs[2][0], "Other"],
            values=top3 + [rest],
            hole=0.65,
            marker=dict(colors=d_clrs, line=dict(color="rgba(5,10,20,1)", width=2)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:.2%}<extra></extra>",
        ))
        donut_fig.add_annotation(
            text=f"<b>{conf * 100:.0f}%</b><br>"
                 f"<span style='font-size:8px;color:#4a6080;font-family:Inter'>TOP-1</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=color, family="JetBrains Mono"),
            align="center",
        )
        donut_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), height=220,
            showlegend=False, font=dict(family="Inter"),
        )

        # ── Decision margin gauge ──────────────────────────────
        # FIX: 'transparent' replaced with 'rgba(0,0,0,0)' for all tickcolor fields
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(margin * 100, 1),
            number={
                "suffix": "%",
                "font": {"size": 20, "color": "#e2eaf8", "family": "JetBrains Mono"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": "rgba(0,0,0,0)",          # ← FIXED
                    "tickfont": {"color": "#4a6080", "size": 8, "family": "Inter"},
                    "showticklabels": False,
                },
                "bar":      {"color": color, "thickness": 0.16},
                "bgcolor":  "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  15], "color": "rgba(255,77,109,0.1)"},
                    {"range": [15, 40], "color": "rgba(255,209,102,0.07)"},
                    {"range": [40,100], "color": "rgba(0,217,163,0.06)"},
                ],
                "threshold": {
                    "line": {"color": "#4a6080", "width": 1},
                    "thickness": 0.5, "value": 15,
                },
            },
            title={
                "text": "Decision<br>Margin",
                "font": {"size": 9, "color": "#4a6080", "family": "Inter"},
            },
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=36, b=6),
            height=200,
            font=dict(family="Inter"),
        )

        # ── Entropy ────────────────────────────────────────────
        entropy  = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)
        max_ent  = math.log(len(probs))
        norm_ent = entropy / max_ent if max_ent > 0 else 0
        ent_pct  = f"{norm_ent * 100:.0f}%"
        ent_clr  = "#ffd166" if norm_ent > 0.5 else "#00d9a3"

        # ── ASSEMBLE ───────────────────────────────────────────
        return html.Div([

            # Uncertain warning bar
            html.Div(
                className="unc-bar",
                style={"display": "flex" if is_unc else "none"},
                children=[
                    html.Span("⚠", className="unc-icon"),
                    html.Span(
                        "Low decision margin — the model is uncertain between multiple classes. "
                        "These results should be interpreted with additional caution."
                    ),
                ],
            ),

            # Prediction card
            html.Div(
                className="pred-card",
                style={
                    "background": f"linear-gradient(135deg, {rs['bg']}, rgba(8,14,28,0.6))",
                    "border": f"1px solid {rs['border']}",
                    "--glow-color": glow,
                },
                children=[
                    html.Div(children=[
                        html.P("Primary Diagnosis", className="pred-eyebrow",
                               style={"color": color}),
                        html.P(top, className="pred-name", style={"color": color}),
                        html.P(desc, className="pred-desc"),
                        html.Span(
                            f"{meta['icon']}  Risk Level: {risk}",
                            className="risk-tag",
                            style={"background": rs["bg"], "border": f"1px solid {rs['border']}", "color": rs["color"]},
                        ),
                    ]),
                    html.Div(className="conf-ring", children=[
                        html.Div(f"{conf * 100:.1f}%", className="conf-num", style={"color": color}),
                        html.Div("Confidence", className="conf-lbl"),
                    ]),
                ],
            ),

            # Images row
            dbc.Row(className="g-2 mb-3", children=[
                dbc.Col(xs=12, sm=6, children=[
                    html.Div(className="img-card", children=[
                        html.Div(className="img-card-hdr", children=[
                            html.Span("Original Upload", className="img-card-lbl"),
                            html.Span("INPUT", className="img-card-badge"),
                        ]),
                        html.Img(src=image_contents, style={
                            "width": "100%", "height": "auto","maxHeight": "340px",
                            "objectFit": "contain","background": "#080e1c", "display": "block","padding": "8px",
                        }),
                    ]),
                ]),
                dbc.Col(xs=12, sm=6, children=[
                    html.Div(className="img-card", children=[
                        html.Div(className="img-card-hdr", children=[
                            html.Span("Grad-CAM Attention", className="img-card-lbl"),
                            html.Span("XAI", className="img-card-badge cam-badge"),
                        ]),
                        html.Img(src=results["gradcam_base64"], style={
                            "width": "100%", "height": "auto","maxHeight": "340px",
                            "objectFit": "contain", "display": "block","background": "#080e1c","padding": "8px",
                        }),
                    ]),
                ]),
            ]),

            # Charts row
            dbc.Row(className="g-2 mb-3", children=[
                dbc.Col(xs=12, md=5, children=[
                    html.Div(className="data-box", children=[
                        html.P("Class Probabilities", className="sec-lbl"),
                        dcc.Graph(figure=bar_fig, config={"displayModeBar": False},
                                  style={"height": "220px"}),
                    ]),
                ]),
                dbc.Col(xs=12, sm=6, md=4, children=[
                    html.Div(className="data-box", children=[
                        html.P("Top-3 Confidence Split", className="sec-lbl"),
                        dcc.Graph(figure=donut_fig, config={"displayModeBar": False},
                                  style={"height": "220px"}),
                    ]),
                ]),
                dbc.Col(xs=12, sm=6, md=3, children=[
                    html.Div(className="data-box", children=[
                        html.P("Decision Margin", className="sec-lbl"),
                        dcc.Graph(figure=gauge_fig, config={"displayModeBar": False},
                                  style={"height": "200px"}),
                        html.Div(style={
                            "display": "flex", "justifyContent": "space-between", "marginTop": "2px",
                        }, children=[
                            html.Span("Uncertain", style={"fontSize": "0.6rem", "color": "#ff4d6d", "fontFamily": "JetBrains Mono, monospace"}),
                            html.Span("15%", style={"fontSize": "0.6rem", "color": "#4a6080", "fontFamily": "JetBrains Mono, monospace"}),
                            html.Span("Confident", style={"fontSize": "0.6rem", "color": "#00d9a3", "fontFamily": "JetBrains Mono, monospace"}),
                        ]),
                    ]),
                ]),
            ]),

            # Entropy + prob list + patient summary
            dbc.Row(className="g-2", children=[

                dbc.Col(xs=12, md=4, children=[
                    html.Div(className="data-box", style={"height": "100%"}, children=[

                        html.P("Prediction Entropy", className="sec-lbl"),
                        html.Div(className="ent-row", children=[
                            html.Span("Distribution spread", className="ent-label"),
                            html.Span(ent_pct, className="ent-val", style={"color": ent_clr}),
                        ]),
                        html.Div(className="ent-bar-wrap", children=[
                            html.Div(className="ent-bar-fill", style={
                                "width": ent_pct,
                                "background": f"linear-gradient(90deg, {ent_clr}, {ent_clr}88)",
                            }),
                        ]),
                        html.P(
                            "Higher entropy = probability mass spread across classes = greater diagnostic uncertainty.",
                            style={"fontSize": "0.67rem", "color": "#4a6080",
                                   "lineHeight": "1.5", "marginBottom": "14px"},
                        ),

                        html.Hr(className="hdiv"),

                        html.P("Patient Summary", className="sec-lbl"),
                        html.Div(className="patient-grid", children=[
                            html.Div(className="pt-chip", children=[
                                html.Div(str(age) if age else "—", className="pt-val"),
                                html.Div("Age (yrs)", className="pt-lbl"),
                            ]),
                            html.Div(className="pt-chip", children=[
                                html.Div(str(sex).capitalize() if sex else "—", className="pt-val"),
                                html.Div("Sex", className="pt-lbl"),
                            ]),
                            html.Div(className="pt-chip full", children=[
                                html.Div(
                                    str(loc).replace("_", " ").title() if loc else "—",
                                    className="pt-val",
                                    style={"fontSize": "0.82rem"},
                                ),
                                html.Div("Lesion Location", className="pt-lbl"),
                            ]),
                        ]),
                    ]),
                ]),

                dbc.Col(xs=12, md=8, children=[
                    html.Div(className="data-box", children=[
                        html.P("Full Probability Breakdown", className="sec-lbl"),
                        html.Div(prob_items),
                    ]),
                ]),
            ]),
        ])

    except Exception as exc:
        return html.Div(
            style={
                "background": "rgba(255,77,109,0.06)",
                "border": "1px solid rgba(255,77,109,0.25)",
                "borderRadius": "10px", "padding": "16px",
            },
            children=[
                html.P("Analysis Error", style={
                    "color": "#ff4d6d", "fontWeight": "700",
                    "fontSize": "0.75rem", "marginBottom": "5px",
                    "textTransform": "uppercase", "letterSpacing": "0.8px",
                }),
                html.P(str(exc), style={
                    "color": "#4a6080", "fontSize": "0.78rem",
                    "fontFamily": "JetBrains Mono, monospace",
                }),
            ],
        )


if __name__ == "__main__":
    app.run(debug=True)