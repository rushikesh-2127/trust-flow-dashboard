"""
dashboard/app.py

TrustFL // Intelligent IoT Intrusion Detection System
Byzantine-Resilient Federated Learning Dashboard
"""

import base64
import datetime
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auth import MAX_PASSWORD_BYTES, hash_password, load_users, save_users, verify_password  # noqa: E402
from data_audit import normalize_and_align_features, trace_file_bugs_and_anomalies  # noqa: E402
from scoring import score_device  # noqa: E402
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SHIELD_IMG = ASSETS_DIR / "soc_shield.jpg"
MESH_IMG = ASSETS_DIR / "threat_mesh.jpg"
BG_IMG = ASSETS_DIR / "sample_frame.jpg"



st.set_page_config(
    page_title="TrustFL // Federated IoT Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# USER DATABASE & PERSISTENCE (Sign Up / Sign In)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DESIGN SYSTEM: PLUS JAKARTA SANS & MODERN MINIMALIST SAAS
# ---------------------------------------------------------------------------
COLOR_BG = "#0F131D"          # Stitch surface/bg
COLOR_CARD = "#1C1F2A"        # Stitch surface-container
COLOR_CARD_LOW = "#171B26"    # Stitch surface-container-low
COLOR_CARD_HIGH = "#262A35"   # Stitch surface-container-high
COLOR_BORDER = "rgba(60, 73, 76, 0.50)"  # Stitch outline-variant
COLOR_PRIMARY = "#22D3EE"     # Stitch primary electric cyan
COLOR_SECONDARY = "#ADC6FF"   # Stitch secondary ice blue
COLOR_SUCCESS = "#68F5B8"     # Stitch tertiary neon mint
COLOR_WARN = "#F59E0B"        # Amber
COLOR_DANGER = "#FFB4AB"      # Stitch error crimson
COLOR_MUTED = "#859397"       # Stitch outline
COLOR_TEXT = "#DFE2F1"        # Stitch on-surface
COLOR_TEXT_MUTED = "#BBC9CD"  # Stitch on-surface-variant

def load_bg_img_b64() -> str:
    """Load and base64-encode sample_frame.jpg for CSS background.

    :returns: Base64-encoded JPEG string, or empty string if file is missing.
    """
    sample_file = ASSETS_DIR / "sample_frame.jpg"
    target = sample_file if sample_file.exists() else BG_IMG
    if target.exists():
        try:
            with open(target, "rb") as fh:
                return base64.b64encode(fh.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


_bg_img_b64 = load_bg_img_b64()
_bg_img_layer = f"url('data:image/jpeg;base64,{_bg_img_b64}')" if _bg_img_b64 else "none"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* =========================================================
   KEYFRAME ANIMATIONS
   ========================================================= */
@keyframes gridPan {{
    0%   {{ background-position: 0 0, 0 0; }}
    100% {{ background-position: 60px 60px, 60px 60px; }}
}}
@keyframes radialPulse {{
    0%, 100% {{ opacity: 0.18; transform: scale(1);   }}
    50%        {{ opacity: 0.30; transform: scale(1.08); }}
}}
@keyframes naturalScanner {{
    0%   {{ top: -4px;  opacity: 0; }}
    2%   {{ opacity: 0.95; }}
    22%  {{ opacity: 1.0; }}
    48%  {{ opacity: 0.90; }}
    70%  {{ top: 100vh; opacity: 1.0; }}
    72%  {{ top: 102vh; opacity: 0; }}
    100% {{ top: 102vh; opacity: 0; }}
}}
@keyframes naturalScannerTrail {{
    0%   {{ top: -26px; opacity: 0; }}
    2%   {{ opacity: 0.85; }}
    48%  {{ opacity: 0.75; }}
    70%  {{ top: calc(100vh - 22px); opacity: 0.85; }}
    72%  {{ top: 100vh; opacity: 0; }}
    100% {{ top: 100vh; opacity: 0; }}
}}
@keyframes nodeBlink {{
    0%, 80%, 100% {{ opacity: 0.0; }}
    40%           {{ opacity: 0.9; }}
}}
@keyframes hexFloat {{
    0%   {{ transform: translateY(0px)   rotate(0deg);   opacity: 0.04; }}
    50%  {{ transform: translateY(-22px) rotate(180deg); opacity: 0.09; }}
    100% {{ transform: translateY(0px)   rotate(360deg); opacity: 0.04; }}
}}

/* =========================================================
   GLOBAL FONT & BASE THEME
   ========================================================= */
html, body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #06080E !important;
    color: #DFE2F1 !important;
    min-height: 100vh !important;
    background-image:
        linear-gradient(rgba(6, 8, 14, 0.78) 0%, rgba(6, 8, 14, 0.90) 100%),
        {_bg_img_layer} !important;
    background-size: cover !important;
    background-position: center top !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}}

/* =========================================================
   MAIN BACKGROUND — sample_frame.jpg Cyber Command Center
   ========================================================= */
[data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #DFE2F1 !important;
    min-height: 100vh !important;
    background-color: transparent !important;
    background-image:
        linear-gradient(rgba(6, 8, 14, 0.78) 0%, rgba(6, 8, 14, 0.90) 100%),
        {_bg_img_layer} !important;
    background-size: cover !important;
    background-position: center top !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}}

.main, section.main, [data-testid="stMain"], [data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"], .block-container, [data-testid="stVerticalBlock"] {{
    background: transparent !important;
    background-color: transparent !important;
}}

.block-container {{
    position: relative;
    z-index: 2;
}}

/* =========================================================
   NATURAL OPTICAL SCANNER — Realistic 1.5px laser sweep with 1.8s pause cycle
   ========================================================= */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    top: -4px;
    left: 0;
    right: 0;
    height: 1.5px;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(34, 211, 238, 0.15) 6%,
        rgba(34, 211, 238, 0.85) 24%,
        #FFFFFF 50%,
        rgba(34, 211, 238, 0.85) 76%,
        rgba(34, 211, 238, 0.15) 94%,
        transparent 100%
    );
    box-shadow:
        0 0 4px rgba(255, 255, 255, 0.95),
        0 0 10px rgba(34, 211, 238, 0.80),
        0 0 22px rgba(34, 211, 238, 0.40);
    animation: naturalScanner 6.2s cubic-bezier(0.25, 1, 0.5, 1) infinite;
    z-index: 9999;
    pointer-events: none;
}}

/* Subtle optical phosphor trailing wash */
[data-testid="stAppViewContainer"]::after {{
    content: '';
    position: fixed;
    top: -26px;
    left: 0;
    right: 0;
    height: 24px;
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(34, 211, 238, 0.03) 40%,
        rgba(34, 211, 238, 0.18) 100%
    );
    animation: naturalScannerTrail 6.2s cubic-bezier(0.25, 1, 0.5, 1) infinite;
    z-index: 9998;
    pointer-events: none;
}}

/* Transparent header */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* =========================================================
   FLOATING HEX DECO ELEMENTS (pure CSS, no JS)
   Applied to ::before / ::after of sidebar
   ========================================================= */
[data-testid="stSidebar"]::before {{
    content: '⬡';
    position: absolute;
    font-size: 9rem;
    color: rgba(0, 200, 255, 0.06);
    top: 12%;
    left: -10px;
    animation: hexFloat 14s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}}
[data-testid="stSidebar"]::after {{
    content: '⬡';
    position: absolute;
    font-size: 5.5rem;
    color: rgba(99, 102, 241, 0.07);
    bottom: 18%;
    right: -6px;
    animation: hexFloat 18s ease-in-out infinite reverse;
    pointer-events: none;
    z-index: 0;
}}

/* =========================================================
   MONOSPACE ACCENT
   ========================================================= */
.mono, .stMetric, code, .stDataFrame, [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* =========================================================
   STITCH HUD CARD & SURFACE CONTAINERS
   ========================================================= */
.tf-card {{
    background: #1C1F2A !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(60, 73, 76, 0.45) !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.6rem !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.45),
                inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
}}

.tf-card-title {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #DFE2F1 !important;
    margin-bottom: 0.4rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.015em !important;
}}

.tf-card-sub {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.86rem !important;
    color: #BBC9CD !important;
    line-height: 1.6 !important;
    margin-bottom: 0.8rem !important;
}}

/* =========================================================
   STITCH CONSOLE NAVBAR
   ========================================================= */
.tf-navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.4rem;
    background: rgba(15, 19, 29, 0.90) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(60, 73, 76, 0.50) !important;
    border-radius: 12px;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 8px rgba(0, 0, 0, 0.4) !important;
}}

.tf-brand {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #DFE2F1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.tf-brand span {{
    color: #22D3EE;
    text-shadow: 0 0 14px rgba(34, 211, 238, 0.6);
}}

/* =========================================================
   STITCH BADGES & TELEMETRY PILLS
   ========================================================= */
.tf-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.02em;
}}
.tf-badge-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
}}

.badge-trust {{
    background: rgba(104, 245, 184, 0.12) !important;
    color: #68F5B8 !important;
    border: 1px solid rgba(104, 245, 184, 0.35) !important;
}}
.badge-trust .tf-badge-dot {{ background: #68F5B8; box-shadow: 0 0 6px #68F5B8; }}

.badge-warn {{
    background: rgba(245, 158, 11, 0.12) !important;
    color: #F59E0B !important;
    border: 1px solid rgba(245, 158, 11, 0.35) !important;
}}
.badge-warn .tf-badge-dot {{ background: #F59E0B; box-shadow: 0 0 6px #F59E0B; }}

.badge-alert {{
    background: rgba(255, 180, 171, 0.12) !important;
    color: #FFB4AB !important;
    border: 1px solid rgba(255, 180, 171, 0.35) !important;
}}
.badge-alert .tf-badge-dot {{ background: #FFB4AB; box-shadow: 0 0 6px #FFB4AB; }}

/* =========================================================
   STITCH SEGMENTED OPERATIONS TABS
   ========================================================= */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: #171B26 !important;
    border: 1px solid rgba(60, 73, 76, 0.45) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    border: none !important;
    color: #BBC9CD !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.1rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #DFE2F1 !important;
    background-color: #262A35 !important;
}}
.stTabs [aria-selected="true"] {{
    color: #22D3EE !important;
    background-color: #262A35 !important;
    border-bottom: 2px solid #22D3EE !important;
    box-shadow: 0 0 12px rgba(34, 211, 238, 0.20) !important;
    text-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
}}

/* =========================================================
   FORM INPUTS, SELECTBOXES & BUTTONS (STITCH SYSTEM)
   ========================================================= */
.stTextInput input, .stSelectbox [data-baseweb="select"] > div {{
    background-color: #171B26 !important;
    border: 1px solid rgba(60, 73, 76, 0.6) !important;
    border-radius: 8px !important;
    color: #DFE2F1 !important;
    font-size: 0.88rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
.stTextInput input:focus, .stSelectbox [data-baseweb="select"]:focus-within > div {{
    border-color: #22D3EE !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.20) !important;
}}
.stButton > button {{
    background: #262A35 !important;
    color: #DFE2F1 !important;
    border: 1px solid rgba(60, 73, 76, 0.6) !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: #313540 !important;
    color: #22D3EE !important;
    border-color: rgba(34, 211, 238, 0.5) !important;
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.25) !important;
}}
.stButton > button[kind="primary"] {{
    background: #22D3EE !important;
    color: #00363E !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(34, 211, 238, 0.4) !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #8AEBFF !important;
    color: #001F25 !important;
    box-shadow: 0 0 18px rgba(34, 211, 238, 0.75) !important;
}}

/* =========================================================
   STREAMLIT METRIC HUD TILES (STITCH SYSTEM)
   ========================================================= */
[data-testid="stMetric"] {{
    background: #1C1F2A !important;
    border: 1px solid rgba(60, 73, 76, 0.45) !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.1rem !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #859397 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #22D3EE !important;
}}

/* =========================================================
   SIDEBAR (STITCH SYSTEM)
   ========================================================= */
[data-testid="stSidebar"] {{
    background: rgba(10, 14, 24, 0.92) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(60, 73, 76, 0.50) !important;
}}

/* =========================================================
   PAGE FRAME — neon corner border around entire viewport
   ========================================================= */
@keyframes frameGlow {{
    0%,100% {{ opacity: 0.35; }}
    50%      {{ opacity: 0.70; }}
}}
body::before {{
    content: '';
    position: fixed;
    inset: 6px;
    border: 1px solid rgba(0, 200, 255, 0.22);
    border-radius: 6px;
    pointer-events: none;
    z-index: 99990;
    animation: frameGlow 4s ease-in-out infinite;
    box-shadow:
        0  0  8px  2px rgba(0, 200, 255, 0.10),
        inset 0 0 12px 2px rgba(0, 200, 255, 0.04);
}}

/* Corner accent marks */
body::after {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 99991;
    background:
        /* top-left */
        linear-gradient(90deg,  rgba(34,211,238,0.7) 0%, transparent 40px) 6px 6px / 40px 2px no-repeat,
        linear-gradient(180deg, rgba(34,211,238,0.7) 0%, transparent 40px) 6px 6px / 2px 40px no-repeat,
        /* top-right */
        linear-gradient(270deg, rgba(34,211,238,0.7) 0%, transparent 40px) calc(100% - 6px) 6px / 40px 2px no-repeat,
        linear-gradient(180deg, rgba(34,211,238,0.7) 0%, transparent 40px) calc(100% - 7px) 6px / 2px 40px no-repeat,
        /* bottom-left */
        linear-gradient(90deg,  rgba(34,211,238,0.7) 0%, transparent 40px) 6px calc(100% - 7px) / 40px 2px no-repeat,
        linear-gradient(0deg,   rgba(34,211,238,0.7) 0%, transparent 40px) 6px calc(100% - 6px) / 2px 40px no-repeat,
        /* bottom-right */
        linear-gradient(270deg, rgba(34,211,238,0.7) 0%, transparent 40px) calc(100% - 6px) calc(100% - 7px) / 40px 2px no-repeat,
        linear-gradient(0deg,   rgba(34,211,238,0.7) 0%, transparent 40px) calc(100% - 7px) calc(100% - 6px) / 2px 40px no-repeat;
}}

/* =========================================================
   IMAGE BORDERS — glowing neon frame on all images
   ========================================================= */
@keyframes imgBorderPulse {{
    0%,100% {{ box-shadow: 0 0  6px 2px rgba(0,200,255,0.20), 0 0 18px 4px rgba(0,180,255,0.10); border-color: rgba(0,200,255,0.25); }}
    50%      {{ box-shadow: 0 0 14px 4px rgba(0,200,255,0.40), 0 0 32px 8px rgba(0,180,255,0.18); border-color: rgba(34,211,238,0.50); }}
}}
.stImage img, [data-testid="stImage"] img {{
    border: 1.5px solid rgba(0, 200, 255, 0.28);
    border-radius: 10px;
    animation: imgBorderPulse 4s ease-in-out infinite;
    transition: transform 0.3s ease;
}}
.stImage img:hover, [data-testid="stImage"] img:hover {{
    transform: scale(1.015);
}}

/* =========================================================
   CARD HOVER LIFT
   ========================================================= */
.tf-card {{
    transition: transform 0.25s cubic-bezier(0.45,0,0.55,1),
                box-shadow 0.25s ease, border-color 0.25s ease;
}}
.tf-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(0,200,255,0.28) !important;
    box-shadow: 0 8px 32px -4px rgba(0,0,0,0.65),
                0 0 20px 2px rgba(0,200,255,0.10),
                inset 0 1px 0 rgba(0,200,255,0.10) !important;
}}

/* =========================================================
   LIVE FEED — animated ring + row flash
   ========================================================= */
@keyframes radarRing {{
    0%   {{ transform: scale(0.6); opacity: 0.8; }}
    100% {{ transform: scale(1.8); opacity: 0; }}
}}
@keyframes attackFlash {{
    0%,100% {{ background: rgba(239,68,68,0.00); }}
    20%      {{ background: rgba(239,68,68,0.18); }}
}}
@keyframes normalSlide {{
    from {{ opacity: 0; transform: translateX(-12px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
.live-radar-ring {{
    position: absolute;
    width: 100%; height: 100%;
    border-radius: 50%;
    border: 2px solid rgba(34,211,238,0.55);
    animation: radarRing 2s ease-out infinite;
}}
.live-feed-attack-row {{
    animation: attackFlash 0.8s ease-out;
}}
.live-feed-normal-row {{
    animation: normalSlide 0.4s ease-out;
}}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Readability layer: frosted content panel + solid tab bar so text never
# competes with the background photo.
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 1280px !important;
    margin: 0.6rem auto 1.5rem auto !important;
    padding: 1.4rem 2rem 2rem 2rem !important;
    background: rgba(10, 14, 22, 0.80) !important;
    backdrop-filter: blur(18px) saturate(1.2);
    -webkit-backdrop-filter: blur(18px) saturate(1.2);
    border: 1px solid rgba(60, 73, 76, 0.45);
    border-radius: 18px;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.45);
}
[data-baseweb="tab-list"] {
    background: rgba(23, 27, 38, 0.95) !important;
    border: 1px solid rgba(60, 73, 76, 0.5);
    border-radius: 12px;
    padding: 0.3rem !important;
    gap: 0.25rem !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    color: #BBC9CD !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(34, 211, 238, 0.14) !important;
    color: #22D3EE !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stMetric"] {
    background: #1C1F2A;
    border: 1px solid rgba(60, 73, 76, 0.5);
    border-radius: 12px;
    padding: 0.9rem 1rem;
}
.tf-statrow { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.4rem 0 0.9rem 0; }
.tf-stat {
    flex: 1 1 140px; background: #1C1F2A; border: 1px solid rgba(60,73,76,0.5);
    border-radius: 12px; padding: 0.7rem 0.9rem;
}
.tf-stat .k { font-size: 0.68rem; color: #859397; text-transform: uppercase; letter-spacing: 0.06em; }
.tf-stat .v { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: #DFE2F1; }
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem !important; border-radius: 0; margin: 0 !important; }
}
</style>
""", unsafe_allow_html=True)

if _bg_img_b64:
    st.markdown(f"""
<div id="tf-command-bg" style="
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -1;
    pointer-events: none;
    background-image: linear-gradient(rgba(6, 8, 14, 0.78) 0%, rgba(6, 8, 14, 0.90) 100%), url('data:image/jpeg;base64,{_bg_img_b64}');
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    background-attachment: fixed;
"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# AUTHENTICATION: SIGN IN & SIGN UP SCREENS
# ---------------------------------------------------------------------------
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

users_db = load_users()


def render_auth_portal():
    """Render the user authentication portal containing Sign In and Sign Up tabs."""
    col_vis, col_auth = st.columns([1.15, 1], gap="large")

    with col_vis:
        st.markdown("<br>", unsafe_allow_html=True)
        if SHIELD_IMG.exists():
            st.image(str(SHIELD_IMG), use_container_width=True)
        st.markdown("""<div style="padding: 0.8rem 0.2rem 1rem 0.2rem; text-align: center; display: flex; flex-direction: column; align-items: center;">
<div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:600; color:#22D3EE; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:0.6rem; display:flex; align-items:center; justify-content:center; gap:0.5rem; width:100%;">
<span style="flex:1; max-width:80px; height:1px; background:linear-gradient(90deg,transparent,#22D3EE44);"></span>
🛡️ &nbsp;Trust no one. Verify everything. Defend everywhere.
<span style="flex:1; max-width:80px; height:1px; background:linear-gradient(90deg,#22D3EE44,transparent);"></span>
</div>
<div style="display:flex; align-items:center; justify-content:center; gap:0.8rem; margin-bottom:0.5rem;">
<div style="width:42px; height:42px; display:flex; align-items:center; justify-content:center; background:rgba(34,211,238,0.08); border:1px solid rgba(34,211,238,0.35); border-radius:10px; box-shadow:0 0 16px rgba(34,211,238,0.25);">
<svg width="28" height="28" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter:drop-shadow(0 0 6px rgba(34,211,238,0.7));">
<path d="M50 8 L86 24 V52 C86 73 50 93 50 93 C50 93 14 73 14 52 V24 L50 8 Z" stroke="#22D3EE" stroke-width="4.5" fill="rgba(15,23,42,0.85)"/>
<path d="M50 20 L76 32 V50 C76 64 50 78 50 78 C50 78 24 64 24 50 V32 L50 20 Z" stroke="rgba(34,211,238,0.6)" stroke-width="2.5" fill="rgba(34,211,238,0.12)"/>
<circle cx="50" cy="48" r="8" fill="#68F5B8" style="filter:drop-shadow(0 0 6px #68F5B8);"/>
<line x1="50" y1="28" x2="50" y2="40" stroke="#22D3EE" stroke-width="2.5" stroke-dasharray="2 2"/>
<line x1="36" y1="56" x2="44" y2="52" stroke="#22D3EE" stroke-width="2.5"/>
<line x1="64" y1="56" x2="56" y2="52" stroke="#22D3EE" stroke-width="2.5"/>
<circle cx="50" cy="28" r="3.5" fill="#22D3EE"/>
<circle cx="34" cy="58" r="3.5" fill="#22D3EE"/>
<circle cx="66" cy="58" r="3.5" fill="#22D3EE"/>
</svg>
</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:3rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.04em; line-height:1;">
Trust<span style="color:#22D3EE; text-shadow:0 0 28px rgba(34,211,238,0.75);">FL</span>
</div>
</div>
<div style="font-size:0.95rem; color:#94A3B8; margin-bottom:1rem; line-height:1.55; text-align:center;">
Byzantine-Resilient Federated IoT Intrusion Detection
</div>
<div style="display:flex; gap:0.6rem; flex-wrap:wrap; justify-content:center;">
<span class="tf-badge badge-trust"><span class="tf-badge-dot"></span> Zero-Trust</span>
<span class="tf-badge badge-warn"><span class="tf-badge-dot"></span> Byzantine Resilient</span>
<span class="tf-badge badge-alert"><span class="tf-badge-dot"></span> Attack Detection</span>
</div>
</div>""", unsafe_allow_html=True)

    with col_auth:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="tf-card" style="border-top: 3px solid #22D3EE; padding: 2.2rem 2rem;">
<div style="text-align: center; margin-bottom: 1.5rem;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #22D3EE; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.55rem; opacity: 0.85;">
🔒 &nbsp;Zero-Trust &bull; Federated &bull; Byzantine-Resilient
</div>
<div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.04em; color: #FFFFFF; margin-bottom: 0.2rem; line-height: 1;">
Trust<span style="color: #22D3EE; text-shadow: 0 0 22px rgba(34,211,238,0.75);">FL</span>
</div>
<div style="font-size: 0.82rem; color: #64748B; font-style: italic; margin-bottom: 0.25rem;">
&ldquo;Where every gradient is earned, not assumed.&rdquo;
</div>
<div style="font-size: 0.78rem; color: #475569; margin-top: 0.5rem;">
Sign in to access the Security Console
</div>
</div>""", unsafe_allow_html=True)

        tab_signin, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_signin:
            st.write("")
            with st.form("signin_form"):
                email = st.text_input("Email address", placeholder="name@company.com").strip().lower()
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In to Console", use_container_width=True, type="primary")

                if submitted:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    elif email not in users_db:
                        st.error("No account found with this email address.")
                    elif not verify_password(password, users_db[email]["password_hash"]):
                        st.error("Incorrect password. Please try again.")
                    else:
                        st.session_state.auth_user = {
                            "email": email,
                            "name": users_db[email].get("name", email.split("@")[0].capitalize()),
                        }
                        st.rerun()

            st.markdown("""
            <div style="margin-top: 1.25rem; padding: 0.75rem 1rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; font-size: 0.8rem; color: #94A3B8;">
                <span style="font-weight: 600; color: #FFF;">Quick Demo Login:</span><br>
                Email: <code>demo@trustfl.org</code> &nbsp;|&nbsp; Pass: <code>TrustFL@2026</code>
            </div>
            """, unsafe_allow_html=True)

        with tab_signup:
            st.write("")
            with st.form("signup_form"):
                new_name = st.text_input("Full Name", placeholder="e.g. Alex Morgan").strip()
                new_email = st.text_input("Email address", placeholder="name@company.com").strip().lower()
                new_password = st.text_input("Create Password", type="password", placeholder="Minimum 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                created = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if created:
                    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
                    if not re.match(email_pattern, new_email):
                        st.error("Please provide a valid email address.")
                    elif new_email in users_db:
                        st.error("An account with this email already exists. Please Sign In.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif len(new_password.encode("utf-8")) > MAX_PASSWORD_BYTES:
                        st.error(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        users_db[new_email] = {
                            "password_hash": hash_password(new_password),
                            "name": new_name or new_email.split("@")[0].capitalize(),
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        save_users(users_db)
                        st.session_state.auth_user = {
                            "email": new_email,
                            "name": users_db[new_email]["name"],
                        }
                        st.success("Account created successfully!")
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.auth_user:
    render_auth_portal()
    st.stop()

# ---------------------------------------------------------------------------
# AUTHENTICATED USER SESSION & TOPBAR
# ---------------------------------------------------------------------------
user = st.session_state.auth_user or {"name": "Guest", "email": "guest@trustfl.org"}

# ---------------------------------------------------------------------------
# LOAD PERSISTENT RESULTS DATA
# ---------------------------------------------------------------------------
@st.cache_data
def _load_json_cached(path: Path, mtime: float):
    """Parse a JSON file; *mtime* is part of the cache key so edits invalidate it."""
    with open(path) as f:
        return json.load(f)


def load_json(path: Path):
    """Safely load and parse a JSON file from disk.

    :param path: Path to the target JSON file.
    :returns: Parsed dictionary/list structure, or None if the file does not exist.
    """
    if not path.exists():
        return None
    return _load_json_cached(path, path.stat().st_mtime)


def md_html(html: str):
    """Render an HTML snippet via st.markdown without Markdown mangling it.

    Markdown ends an HTML block at a blank line and turns 4+ space indents
    into code blocks, so dedent and drop blank lines first.
    """
    cleaned = " ".join(ln.strip() for ln in html.splitlines() if ln.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


baseline = load_json(RESULTS_DIR / "baseline_results.json")
comparison = load_json(RESULTS_DIR / "comparison.json")


def summarize_defense(comparison: dict | None) -> dict:
    """Derive real sidebar/header status values from the saved experiment results.

    :param comparison: Parsed ``comparison.json`` or None.
    :returns: Dict with ``n_nodes``, ``attacker`` (node id or None),
        ``attacker_trust``/``honest_trust`` (final-round means) and ``acc_gain``.
    """
    out = {"n_nodes": None, "attacker": None, "attacker_trust": None, "honest_trust": None, "acc_gain": None}
    if not comparison:
        return out
    run = comparison.get("trust_under_attack") or comparison.get("clean_trust")
    if run:
        out["n_nodes"] = run.get("n_nodes")
        pn = run.get("poison_node")
        out["attacker"] = None if pn is None else str(pn)
        log = run.get("trust_log") or []
        if log:
            last = max(e["round"] for e in log)
            final = {e["node_id"]: e["trust"] for e in log if e["round"] == last}
            if out["attacker"] in final:
                out["attacker_trust"] = final[out["attacker"]]
            honest = [v for k, v in final.items() if k != out["attacker"]]
            if honest:
                out["honest_trust"] = sum(honest) / len(honest)
    try:
        out["acc_gain"] = (comparison["trust_under_attack"]["final_metrics"]["accuracy"]
                           - comparison["fedavg_under_attack"]["final_metrics"]["accuracy"])
    except KeyError:
        pass
    return out


defense = summarize_defense(comparison)
_nodes_label = f"{defense['n_nodes']} Nodes" if defense["n_nodes"] else "Nodes: n/a"

_tested_ratio = f"1 of {defense['n_nodes']}" if defense["n_nodes"] else "n/a"
_fa_txt = _ta_txt = "run the suite"
if comparison:
    try:
        _fc = comparison["clean_fedavg"]["final_metrics"]["accuracy"]
        _fa = comparison["fedavg_under_attack"]["final_metrics"]["accuracy"]
        _ta = comparison["trust_under_attack"]["final_metrics"]["accuracy"]
        _fa_txt = f"{_fa:.1%} ({_fa - _fc:+.1%})"
        _ta_txt = f"{_ta:.1%} ({_ta - _fc:+.1%})"
    except KeyError:
        pass

# Slim header
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; padding:0.2rem 0 0.9rem 0;">
  <div style="display:flex; align-items:center; gap:0.8rem;">
    <div style="width:40px; height:40px; display:flex; align-items:center; justify-content:center; background:rgba(34,211,238,0.08); border:1px solid rgba(34,211,238,0.35); border-radius:10px;">
      <svg width="26" height="26" viewBox="0 0 100 100" fill="none"><path d="M50 8 L86 24 V52 C86 73 50 93 50 93 C50 93 14 73 14 52 V24 L50 8 Z" stroke="#22D3EE" stroke-width="5" fill="rgba(15,23,42,0.85)"/><circle cx="50" cy="48" r="9" fill="#68F5B8"/></svg>
    </div>
    <div>
      <div style="font-size:1.6rem; font-weight:800; letter-spacing:-0.03em; color:#FFF; line-height:1.1;">Trust<span style="color:#22D3EE;">FL</span>
        <span style="font-size:0.8rem; font-weight:600; color:{COLOR_MUTED}; margin-left:0.5rem; font-family:'JetBrains Mono',monospace;">// SECURITY CONSOLE</span></div>
      <div style="font-size:0.8rem; color:#94A3B8;">Trust-weighted federated learning for IoT intrusion detection</div>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
    <span class="tf-badge badge-trust"><span class="tf-badge-dot"></span> {_nodes_label}</span>
    <span class="tf-badge" style="background:rgba(34,211,238,0.12); color:#8aebff; border:1px solid rgba(34,211,238,0.25);">Trust floor 0.05</span>
    <span class="tf-badge" style="background:rgba(104,245,184,0.10); color:#68f5b8; border:1px solid rgba(104,245,184,0.22);">Cosine + EMA</span>
  </div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    if SHIELD_IMG.exists():
        st.image(str(SHIELD_IMG), use_container_width=True)
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid {COLOR_BORDER}; margin-bottom: 1rem;">
        <div style="font-size: 0.76rem; color: {COLOR_MUTED}; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Logged In As</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-top: 0.2rem;">{html.escape(user['name'])}</div>
        <div style="font-size: 0.8rem; color: {COLOR_MUTED};">{html.escape(user['email'])}</div>
    </div>
    """, unsafe_allow_html=True)

    if defense["honest_trust"] is not None:
        atk = defense["attacker_trust"]
        atk_txt = f"{atk:.2f}" if atk is not None else "n/a"
        gain_txt = f"+{defense['acc_gain']:.1%}" if defense["acc_gain"] is not None else "n/a"
        st.markdown(f"""
        <div style="padding: 0.8rem; border-radius: 10px; background: rgba(23, 27, 38, 0.75); border: 1px solid rgba(255,255,255,0.07); font-family: 'JetBrains Mono', monospace; margin-bottom: 1.25rem; font-size: 0.72rem; color: #bbc9cd; line-height: 2;">
            <div style="color:#859397;">LATEST EXPERIMENT</div>
            <div style="display:flex; justify-content:space-between;"><span>Honest trust</span><span style="color:#68f5b8;">{defense['honest_trust']:.2f}</span></div>
            <div style="display:flex; justify-content:space-between;"><span>Attacker trust</span><span style="color:#FFB4AB;">{atk_txt}</span></div>
            <div style="display:flex; justify-content:space-between;"><span>Acc. vs FedAvg</span><span style="color:#8aebff;">{gain_txt}</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("No experiment results yet. Run `python src/run_experiment.py --suite`.")

    if st.button("Sign Out", use_container_width=True):
        st.session_state.auth_user = None
        st.rerun()


# ---------------------------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------------------------
tab_device, tab_live, tab_metrics, tab_about = st.tabs([
    "Device Analyzer",
    "Live Traffic Feed",
    "Model Performance",
    "How It Works",
])


# ===========================================================================
PRESET_SCENARIOS = {
    "trusted": {
        "title": "🟢 Clean IoT Edge Node (Trusted)",
        "file": "sample_device_trusted.csv",
        "badge": "Expected: Trusted",
        "color": COLOR_SUCCESS,
        "desc": "Healthy IoT sensor telemetry exhibiting normal traffic flow and strong gradient consensus alignment.",
    },
    "recon": {
        "title": "🟡 Stealth Recon & Port Scan (Suspicious)",
        "file": "sample_device_recon_probe.csv",
        "badge": "Expected: Suspicious",
        "color": COLOR_WARN,
        "desc": "Low-and-slow network reconnaissance with anomalous connection sweeps and subtle label divergence.",
    },
    "ddos": {
        "title": "🟠 Volumetric DDoS Surge (Heavy Threat)",
        "file": "sample_device_ddos_surge.csv",
        "badge": "Expected: Suspicious",
        "color": COLOR_WARN,
        "desc": "Severe denial-of-service volumetric packet burst with 53%+ attack proportion.",
    },
    "adversary": {
        "title": "🔴 Byzantine Model Poisoning (Adversary)",
        "file": "sample_device_adversary.csv",
        "badge": "Expected: Adversary",
        "color": COLOR_DANGER,
        "desc": "Coordinated label-inversion poisoning attack designed to corrupt the global federated model.",
    },
    "unlabeled": {
        "title": "🟣 Unlabeled Wild Capture (No Labels)",
        "file": "sample_device_unlabeled.csv",
        "badge": "Unlabeled",
        "color": COLOR_PRIMARY,
        "desc": "Raw sensor traffic without ground-truth labels for unsupervised real-time classification.",
    },
    "corrupted": {
        "title": "⚠️ Corrupted Telemetry (Bugs & Glitches)",
        "file": "sample_device_corrupted_telemetry.csv",
        "badge": "Buggy Data",
        "color": COLOR_DANGER,
        "desc": "Sensor data containing realistic bugs: clock skews (negative duration), counter underflow, NaNs, and duplicates.",
    },
    "clean_bench": {
        "title": "🛡️ Benchmark Clean Slice",
        "file": "sample_device_clean.csv",
        "badge": "Baseline Clean",
        "color": COLOR_MUTED,
        "desc": "Standard project benchmark clean node slice.",
    },
    "poisoned_bench": {
        "title": "⚡ Benchmark Poisoned Slice",
        "file": "sample_device_poisoned.csv",
        "badge": "Baseline Poisoned",
        "color": COLOR_DANGER,
        "desc": "Standard project benchmark poisoned node slice.",
    },
}


# ===========================================================================
# TAB 1: DEVICE ANALYZER & BUG TRACER
# ===========================================================================
with tab_device:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
            <span style="color: #22D3EE; text-transform: uppercase;">SPEC: ENCLAVE-DEFENSE-V4.2</span>
            <span style="color: #859397;">/</span>
            <span style="color: #68F5B8; text-transform: uppercase;">DEVICE TELEMETRY AUDIT</span>
            <span style="color: #859397;">/</span>
            <span style="color: #BBC9CD; text-transform: uppercase;">FAULT & ANOMALY SCANNER</span>
        </div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #68F5B8; background: rgba(104,245,184,0.1); padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid rgba(104,245,184,0.3);">
            DIAGNOSTIC ENGINE: ARMED
        </span>
    </div>
    <div class="tf-card">
        <div class="tf-card-title">
            <span style="color:#22D3EE;">⬡</span> Device Telemetry Analyzer & Anomaly Bug Tracer
        </div>
        <div class="tf-card-sub">
            Upload custom IoT network traffic or select adversarial benchmark profiles. The zero-trust diagnostic engine audits schema integrity, detects hardware clock/counter faults, and computes the real-time <b>Byzantine Trust Score (0.00 to 1.00)</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)


    if "active_scenario_key" not in st.session_state:
        st.session_state.active_scenario_key = "trusted"

    scen_col, up_col = st.columns([1.1, 1], gap="medium")

    with scen_col:
        st.markdown("<b style='font-size:0.95rem; color:#FFF;'>Preset Device Scenarios</b>", unsafe_allow_html=True)
        scenario_options = list(PRESET_SCENARIOS.keys())
        selected_scenario = st.selectbox(
            "Select Scenario",
            scenario_options,
            index=scenario_options.index(st.session_state.active_scenario_key) if st.session_state.active_scenario_key in scenario_options else 0,
            format_func=lambda k: PRESET_SCENARIOS[k]["title"],
            label_visibility="collapsed",
        )
        scen_info = PRESET_SCENARIOS[selected_scenario]
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.5rem 0 0.8rem 0; font-size: 0.84rem; color: {COLOR_MUTED};">
            <span style="color:{scen_info['color']}; font-weight: 600;">● {scen_info['badge']}</span>: {scen_info['desc']}
        </div>
        """, unsafe_allow_html=True)

        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            if st.button("⚡ Load Scenario", use_container_width=True, type="primary"):
                st.session_state.active_scenario_key = selected_scenario
                if "custom_csv_df" in st.session_state:
                    del st.session_state["custom_csv_df"]
                st.rerun()

        with b_col2:
            scen_file_path = DATA_DIR / scen_info["file"]
            if scen_file_path.exists():
                with open(scen_file_path, "rb") as f:
                    st.download_button(
                        "📥 Download CSV",
                        f,
                        file_name=scen_info["file"],
                        use_container_width=True,
                    )

    with up_col:
        st.markdown("<b style='font-size:0.95rem; color:#FFF;'>Or Upload Any Custom CSV File</b>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload any CSV file",
            type=["csv", "txt"],
            label_visibility="collapsed",
            help="Accepts any network flow CSV. Auto-maps column aliases and imputes missing features.",
        )
        if uploaded_file is not None:
            try:
                st.session_state.custom_csv_df = pd.read_csv(uploaded_file)
                st.session_state.custom_csv_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Determine which DataFrame to inspect
    target_df = None
    target_name = ""
    if "custom_csv_df" in st.session_state and st.session_state.custom_csv_df is not None:
        target_df = st.session_state.custom_csv_df
        target_name = f"Uploaded File: {html.escape(str(st.session_state.get('custom_csv_name', 'custom.csv')))}"
    else:
        active_key = st.session_state.active_scenario_key
        active_path = DATA_DIR / PRESET_SCENARIOS[active_key]["file"]
        if active_path.exists():
            target_df = pd.read_csv(active_path)
            target_name = f"Preset: {PRESET_SCENARIOS[active_key]['title']}"
        elif (DATA_DIR / "sample_device_clean.csv").exists():
            target_df = pd.read_csv(DATA_DIR / "sample_device_clean.csv")
            target_name = "Default: sample_device_clean.csv"

    if target_df is not None and not target_df.empty:
        try:
            import joblib
            # pyrefly: ignore [missing-import]
            import model_utils as mu
            # pyrefly: ignore [missing-import]
            from common import FEATURE_COLUMNS

            # 1. Normalize and Align schema
            aligned_df, imputed_cols = normalize_and_align_features(target_df)

            # 2. Trace bugs & anomalies
            audit = trace_file_bugs_and_anomalies(aligned_df, imputed_cols)
            findings = audit["findings"]
            health_score = audit["health_score"]
            sanitized_df = audit["sanitized_df"]

            # Display Bug Tracer Report
            st.markdown(f"<div style='margin-top:1.2rem; margin-bottom:0.4rem; font-size:1.1rem; font-weight:700; color:#FFF;'>🔍 Bug & Data Integrity Trace Audit <span style='font-size:0.82rem; font-weight:400; color:{COLOR_MUTED};'>({target_name})</span></div>", unsafe_allow_html=True)

            score_color = COLOR_SUCCESS if health_score >= 80 else (COLOR_WARN if health_score >= 50 else COLOR_DANGER)
            health_label = "Pristine Telemetry" if health_score >= 90 else ("Minor Warnings" if health_score >= 70 else ("Degraded Sensor Data" if health_score >= 40 else "Severe Corruptions Detected"))

            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            with stat_c1:
                st.markdown(f"""
                <div class="tf-card" style="padding: 1rem; text-align:center;">
                    <div style="font-size:0.75rem; color:{COLOR_MUTED};">Integrity Health</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{score_color};">{health_score} <span style="font-size:0.9rem;">/ 100</span></div>
                    <div style="font-size:0.72rem; color:{score_color}; font-weight:600;">{health_label}</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_c2:
                st.markdown(f"""
                <div class="tf-card" style="padding: 1rem; text-align:center;">
                    <div style="font-size:0.75rem; color:{COLOR_MUTED};">Total Flows</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#FFF;">{len(target_df):,}</div>
                    <div style="font-size:0.72rem; color:{COLOR_MUTED};">{len(sanitized_df):,} sanitized</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_c3:
                n_crit = sum(1 for f in findings if f["severity"] == "Critical")
                st.markdown(f"""
                <div class="tf-card" style="padding: 1rem; text-align:center;">
                    <div style="font-size:0.75rem; color:{COLOR_MUTED};">Critical Bugs</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{COLOR_DANGER if n_crit > 0 else COLOR_SUCCESS};">{n_crit}</div>
                    <div style="font-size:0.72rem; color:{COLOR_MUTED};">{len(findings)} total findings</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_c4:
                st.markdown(f"""
                <div class="tf-card" style="padding: 1rem; text-align:center;">
                    <div style="font-size:0.75rem; color:{COLOR_MUTED};">Auto-Remediation</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{COLOR_SUCCESS};">ACTIVE</div>
                    <div style="font-size:0.72rem; color:{COLOR_SUCCESS};">Cleaned for Neural Scoring</div>
                </div>
                """, unsafe_allow_html=True)

            if findings:
                with st.expander(f"📋 View Detailed Bug Trace Findings ({len(findings)} events detected)", expanded=True):
                    for f in findings:
                        f_color = COLOR_DANGER if f["severity"] == "Critical" else (COLOR_WARN if f["severity"] == "Warning" else COLOR_PRIMARY)
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.02); border-left: 3px solid {f_color}; padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; border-radius: 4px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b style="color:#FFF; font-size:0.9rem;">{html.escape(f['bug'])}</b>
                                <span style="font-size:0.75rem; color:{f_color}; font-weight:700; background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:10px;">{f['severity']} // {f['category']}</span>
                            </div>
                            <div style="font-size:0.82rem; color:#CBD5E1; margin: 0.3rem 0;">{html.escape(f['details'])}</div>
                            <div style="font-size:0.75rem; color:{COLOR_MUTED};"><b>Remediation:</b> {html.escape(f['action'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid {COLOR_SUCCESS}; border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #E2E8F0;">
                    <b style="color:{COLOR_SUCCESS};">✅ Clean Telemetry Verification:</b> All network flow records and feature columns passed integrity checks with 0 bugs or corrupted counters.
                </div>
                """, unsafe_allow_html=True)

            # 3. Model Classification & Trust Consensus Scoring
            scaler = joblib.load(RESULTS_DIR / "scaler.joblib")
            npz = np.load(RESULTS_DIR / "global_model_weights.npz")
            global_weights = [npz[k] for k in npz.files]
            reference_consensus = np.load(RESULTS_DIR / "reference_consensus.npy")

            scored = score_device(sanitized_df, scaler, global_weights, reference_consensus)
            preds, probs = scored["preds"], scored["probs"]
            attack_rate, trust_score = scored["attack_rate"], scored["trust_score"]

            if trust_score is None:
                verdict, badge_class, gauge_color = "LABELS MISSING", "badge-warn", COLOR_WARN
                explanation = "No ground-truth labels were included in this CSV. Individual flows were classified by the global neural net, but trust consensus requires ground truth."
            elif trust_score > 0.60:
                verdict, badge_class, gauge_color = "TRUSTED DEVICE", "badge-trust", COLOR_SUCCESS
                explanation = "This device's traffic patterns closely align with healthy network nodes. Full trust and normal aggregation weighting are applied."
            elif trust_score > 0.20:
                verdict, badge_class, gauge_color = "SUSPICIOUS ACTIVITY", "badge-warn", COLOR_WARN
                explanation = "This device exhibits noticeable anomalies and gradient divergence. Its aggregation weight is partially down-weighted."
            else:
                verdict, badge_class, gauge_color = "UNTRUSTED / ADVERSARY", "badge-alert", COLOR_DANGER
                explanation = "Severe anomaly or poisoning signature detected. This node is clamped to the minimum trust floor (0.05) to prevent corrupting the shared model."

            st.markdown("<div style='margin-top:1.5rem; margin-bottom:0.4rem; font-size:1.1rem; font-weight:700; color:#FFF;'>🛡️ Consensus Trust & Threat Classification</div>", unsafe_allow_html=True)

            out_col1, out_col2 = st.columns([1, 1.3])

            with out_col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=trust_score if trust_score is not None else 0.0,
                    number={"valueformat": ".2f", "font": {"family": "Plus Jakarta Sans", "size": 48, "color": gauge_color}},
                    gauge={
                        "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": COLOR_MUTED},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 0.20], "color": "rgba(239, 68, 68, 0.12)"},
                            {"range": [0.20, 0.60], "color": "rgba(245, 158, 11, 0.12)"},
                            {"range": [0.60, 1.00], "color": "rgba(16, 185, 129, 0.12)"},
                        ],
                    },
                ))
                fig.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f'<div style="text-align:center; margin-top:-0.5rem;"><span class="tf-badge {badge_class}"><span class="tf-badge-dot"></span> {verdict}</span></div>',
                    unsafe_allow_html=True,
                )

            with out_col2:
                st.markdown(f"""
                <div class="tf-card" style="margin-top: 0.5rem;">
                    <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
                        <div>
                            <div style="font-size: 0.78rem; color: {COLOR_MUTED};">Total Flows Analyzed</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #FFF;">{len(sanitized_df):,}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; color: {COLOR_MUTED};">Flagged as Attack</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: {COLOR_DANGER if attack_rate > 0.3 else COLOR_SUCCESS};">{attack_rate:.1%}</div>
                        </div>
                    </div>
                    <div style="border-top: 1px solid {COLOR_BORDER}; padding-top: 0.8rem; font-size: 0.88rem; color: #E2E8F0; line-height: 1.5;">
                        {explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("<b>Per-Flow Classification Details</b> (first 50 flows)", unsafe_allow_html=True)
            preview = sanitized_df[FEATURE_COLUMNS].copy()
            preview.insert(0, "Prediction", np.where(preds == 1, "ATTACK", "NORMAL"))
            preview.insert(1, "Confidence", probs.max(axis=1).round(3))
            st.dataframe(preview.head(50), use_container_width=True, height=280)

        except FileNotFoundError:
            st.warning("Model artifacts not found. Run `python src/run_experiment.py --suite` to build reference weights.")
        except Exception as exc:
            st.error(f"Analysis error: {exc}")
    else:
        st.info("Please select a scenario or upload a CSV file to begin analysis.")


# ===========================================================================
# TAB 2: LIVE TRAFFIC FEED
# ===========================================================================
with tab_live:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
            <span style="color: #22D3EE; text-transform: uppercase;">SPEC: ENCLAVE-DEFENSE-V4.2</span>
            <span style="color: #859397;">/</span>
            <span style="color: #68F5B8; text-transform: uppercase;">LIVE TRAFFIC STREAM</span>
            <span style="color: #859397;">/</span>
            <span style="color: #ADC6FF; text-transform: uppercase;">REAL-TIME THREAT RADAR</span>
        </div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #22D3EE; background: rgba(34,211,238,0.1); padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid rgba(34,211,238,0.3);">
            STREAM: ONLINE
        </span>
    </div>
    <div class="tf-card" style="
        border: 1px solid rgba(60, 73, 76, 0.5);
        border-top: 3px solid #22D3EE;
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute; top:24px; right:28px; width:48px; height:48px;">
            <div class="live-radar-ring" style="animation-delay:0s;"></div>
            <div class="live-radar-ring" style="animation-delay:0.7s;"></div>
            <div class="live-radar-ring" style="animation-delay:1.4s;"></div>
            <div style="
                position:absolute; inset:0;
                display:flex; align-items:center; justify-content:center;
                font-size:1.4rem;
            ">📡</div>
        </div>
        <div class="tf-card-title" style="font-size:1.05rem; color:#22D3EE; text-shadow:0 0 10px rgba(34,211,238,0.4);">
            ⚡ Live Device Traffic Stream
        </div>
        <div class="tf-card-sub">
            Streams flows from the held-out test set through the <b>trained federated global model</b>.
            Inject an <b style="color:#F59E0B;">attack burst</b> to watch the model flag threats in real time.
        </div>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.4rem;">
            <span class="tf-badge badge-trust"><span class="tf-badge-dot"></span> Federated Global Model (MLP)</span>
            <span class="tf-badge badge-warn" ><span class="tf-badge-dot"></span> 2s Refresh Rate</span>
            <span class="tf-badge badge-alert"><span class="tf-badge-dot"></span> Attack Detection Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        # pyrefly: ignore [missing-import]
        from common import FEATURE_COLUMNS, get_global_test_split, to_xy
        import joblib
        import model_utils as mu

        @st.cache_resource
        def get_demo_model():
            """Load the federated global model trained by ``run_experiment.py --suite``.

            :returns: Tuple of (model, fitted scaler, test DataFrame).
            """
            _, test_df = get_global_test_split()
            scaler = joblib.load(RESULTS_DIR / "scaler.joblib")
            npz = np.load(RESULTS_DIR / "global_model_weights.npz")
            weights = [npz[k] for k in npz.files]
            clf = mu.build_model()
            X0, _ = to_xy(test_df.head(8), scaler)
            mu.init_architecture(clf, X0[:4], np.array([0, 1, 0, 1]))
            mu.set_weights(clf, weights)
            return clf, scaler, test_df

        clf, scaler, test_df = get_demo_model()
        normal_rows = test_df[test_df["label"] == 0]
        attack_rows = test_df[test_df["label"] == 1]

        if "live_feed_log" not in st.session_state:
            st.session_state.live_feed_log = []
        if "live_feed_running" not in st.session_state:
            st.session_state.live_feed_running = False
        if "force_attack_ticks" not in st.session_state:
            st.session_state.force_attack_ticks = 0

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1.2, 2])
        with ctrl_col1:
            btn_txt = "Pause Feed" if st.session_state.live_feed_running else "Start Live Feed"
            if st.button(btn_txt, use_container_width=True, type="primary" if not st.session_state.live_feed_running else "secondary"):
                st.session_state.live_feed_running = not st.session_state.live_feed_running
                st.rerun()

        with ctrl_col2:
            if st.button("Simulate Attack Burst", use_container_width=True):
                st.session_state.force_attack_ticks = 5
                st.session_state.live_feed_running = True
                st.rerun()

        with ctrl_col3:
            badge_html = '<span class="tf-badge badge-trust"><span class="tf-badge-dot"></span> Live Streaming</span>' if st.session_state.live_feed_running else '<span class="tf-badge" style="background:#1E293B;color:#94A3B8;">Paused</span>'
            st.markdown(f'<div style="padding-top:0.4rem;">{badge_html}</div>', unsafe_allow_html=True)

        @st.fragment(run_every=2 if st.session_state.live_feed_running else None)
        def render_live_feed():
            """Render real-time streaming traffic packet inspection and intrusion alerts."""
            if st.session_state.live_feed_running:
                if st.session_state.force_attack_ticks > 0 and len(attack_rows) > 0:
                    row = attack_rows.sample(1)
                    st.session_state.force_attack_ticks -= 1
                else:
                    row = normal_rows.sample(1) if len(normal_rows) > 0 else test_df.sample(1)

                X_row, y_row = to_xy(row, scaler)
                pred = clf.predict(X_row)[0]
                proba = clf.predict_proba(X_row)[0]

                st.session_state.live_feed_log.insert(0, {
                    "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Status": "ATTACK" if pred == 1 else "NORMAL",
                    "Confidence": f"{max(proba):.1%}",
                    "Attack Type": row["attack_type"].iloc[0],
                    "Action": "Blocked" if pred == 1 else "Permitted",
                })
                st.session_state.live_feed_log = st.session_state.live_feed_log[:25]

            if st.session_state.live_feed_log:
                feed_df = pd.DataFrame(st.session_state.live_feed_log)
                n_total = len(feed_df)
                n_atk = int((feed_df["Status"] == "ATTACK").sum())
                st.markdown(f"""<div class="tf-statrow">
                <div class="tf-stat"><div class="k">Recent flows</div><div class="v">{n_total}</div></div>
                <div class="tf-stat"><div class="k">Attacks blocked</div><div class="v" style="color:#FFB4AB;">{n_atk}</div></div>
                <div class="tf-stat"><div class="k">Attack rate</div><div class="v">{n_atk / n_total:.0%}</div></div>
                </div>""", unsafe_allow_html=True)
                st.dataframe(feed_df, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("Click 'Start Live Feed' to begin streaming real-time network traffic.")

        render_live_feed()

    except FileNotFoundError:
        st.warning("Global model not found. Run `python src/run_experiment.py --suite` to train it.")
    except Exception as exc:
        st.error(f"Live feed error: {exc}")


# ===========================================================================
# TAB 3: MODEL PERFORMANCE
# ===========================================================================
with tab_metrics:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
            <span style="color: #22D3EE; text-transform: uppercase;">SPEC: ENCLAVE-DEFENSE-V4.2</span>
            <span style="color: #859397;">/</span>
            <span style="color: #68F5B8; text-transform: uppercase;">MODEL PERFORMANCE</span>
            <span style="color: #859397;">/</span>
            <span style="color: #ADC6FF; text-transform: uppercase;">BYZANTINE BENCHMARKS</span>
        </div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #68F5B8; background: rgba(104,245,184,0.1); padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid rgba(104,245,184,0.3);">
            TESTED: 1 ATTACKER IN {defense['n_nodes'] or '?'} NODES
        </span>
    </div>
    <div class="tf-card">
        <div class="tf-card-title">
            <span style="color:#22D3EE;">⬡</span> Model Performance &amp; Byzantine Resiliency Benchmarks
        </div>
        <div class="tf-card-sub">
            Evaluation comparing standard Federated Averaging (FedAvg) against our <b>Trust-Weighted FL Enclave Consensus</b> under clean and adversarial poisoning conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)


    if baseline is None and comparison is None:
        st.warning("No experiment results found. Run `python src/run_experiment.py --suite` first.")
    else:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        if baseline:
            if "mlp" in baseline:
                m_col1.metric("Centralized (same MLP)", f"{baseline['mlp']['accuracy']:.1%}",
                              help=f"Random Forest reference: {baseline['accuracy']:.1%}")
            else:
                m_col1.metric("Centralized (Random Forest)", f"{baseline['accuracy']:.1%}")

        if comparison and "clean_fedavg" in comparison:
            m_col2.metric("FedAvg (Clean)", f"{comparison['clean_fedavg']['final_metrics']['accuracy']:.1%}")

        if comparison and "fedavg_under_attack" in comparison and "clean_fedavg" in comparison:
            f_clean = comparison["clean_fedavg"]["final_metrics"]["accuracy"]
            f_atk = comparison["fedavg_under_attack"]["final_metrics"]["accuracy"]
            m_col3.metric("FedAvg Under Attack", f"{f_atk:.1%}", delta=f"{(f_atk - f_clean):.1%}", delta_color="inverse")

        if comparison and "trust_under_attack" in comparison and "fedavg_under_attack" in comparison:
            t_atk = comparison["trust_under_attack"]["final_metrics"]["accuracy"]
            f_atk = comparison["fedavg_under_attack"]["final_metrics"]["accuracy"]
            m_col4.metric("Trust-FL Under Attack", f"{t_atk:.1%}", delta=f"+{(t_atk - f_atk):.1%} vs FedAvg")

        st.write("")

        if comparison:
            p_col1, p_col2 = st.columns([1.2, 1])

            with p_col1:
                st.markdown("<b>Model Comparison (Accuracy & F1-Score)</b>", unsafe_allow_html=True)
                rows = []
                for tag, res in comparison.items():
                    fm = res["final_metrics"]
                    rows.append({
                        "Experiment": tag.replace("_", " ").title(),
                        "Accuracy": fm.get("accuracy"),
                        "F1-Score": fm.get("f1"),
                    })
                bar_df = pd.DataFrame(rows)
                fig_bar = px.bar(
                    bar_df,
                    x="Experiment",
                    y=["Accuracy", "F1-Score"],
                    barmode="group",
                    color_discrete_sequence=[COLOR_PRIMARY, COLOR_SUCCESS],
                    labels={"value": "Score", "variable": "Metric"},
                )
                fig_bar.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Plus Jakarta Sans", "color": COLOR_MUTED},
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with p_col2:
                st.markdown("<b>Rounds Convergence</b>", unsafe_allow_html=True)
                chosen_run = st.selectbox("Select Experiment Run", list(comparison.keys()), label_visibility="collapsed")
                round_metrics = comparison[chosen_run].get("metrics_by_round", {})

                if "accuracy" in round_metrics:
                    line_df = pd.DataFrame({
                        "Round": list(range(0, len(round_metrics["accuracy"]))),  # 0 = untrained initial model
                        "Accuracy": round_metrics["accuracy"],
                    })
                    fig_line = px.line(line_df, x="Round", y="Accuracy", markers=True)
                    fig_line.update_traces(line_color=COLOR_SUCCESS)
                    fig_line.update_layout(
                        height=270,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Plus Jakarta Sans", "color": COLOR_MUTED},
                        yaxis=dict(range=[0.4, 1.0], gridcolor="rgba(255,255,255,0.06)"),
                        xaxis=dict(dtick=1, gridcolor="rgba(255,255,255,0.04)"),
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

            # Node Trust Trajectory
            trust_run = None
            for tag in ["trust_under_attack", "clean_trust"]:
                if tag in comparison and "trust_log" in comparison[tag]:
                    trust_run = tag
                    break

            if trust_run:
                st.write("")
                st.markdown("<b>Per-Node Trust Scores Over Rounds (Adversary Down-Weighting)</b>", unsafe_allow_html=True)
                trust_df = pd.DataFrame(comparison[trust_run]["trust_log"])
                if trust_df.duplicated(subset=["round", "node_id"]).any():
                    st.warning("Results file has duplicate node entries (old run); re-run the suite for exact values.")
                    trust_df = trust_df.drop_duplicates(subset=["round", "node_id"], keep="last")
                if defense["attacker"] is not None and defense["attacker"] not in set(trust_df["node_id"]):
                    st.warning("The attacker node is missing from the trust log; re-run the experiment suite.")
                pivot = trust_df.pivot(index="round", columns="node_id", values="trust").reset_index()

                fig_traj = go.Figure()
                for col in pivot.columns:
                    if col != "round":
                        is_adv = defense["attacker"] is not None and col == defense["attacker"] and trust_run == "trust_under_attack"
                        fig_traj.add_trace(go.Scatter(
                            x=pivot["round"],
                            y=pivot[col],
                            mode="lines+markers",
                            name=f"Node {col} (Adversary)" if is_adv else f"Node {col} (Honest)",
                            line=dict(
                                color=COLOR_DANGER if is_adv else COLOR_SUCCESS,
                                width=3 if is_adv else 1.5,
                            ),
                        ))

                fig_traj.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Plus Jakarta Sans", "color": COLOR_MUTED},
                    xaxis=dict(title="Round", dtick=1, gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="Trust Score", range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_traj, use_container_width=True)


# ===========================================================================
# TAB 4: HOW IT WORKS
# ===========================================================================
with tab_about:
    md_html("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
            <span style="color: #22D3EE; text-transform: uppercase;">SPEC: ENCLAVE-DEFENSE-V4.2</span>
            <span style="color: #859397;">/</span>
            <span style="color: #68F5B8; text-transform: uppercase;">BYZANTINE-FAULT-TOLERANT FEDERATED AGGREGATION</span>
        </div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #68F5B8; background: rgba(104,245,184,0.1); padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid rgba(104,245,184,0.3);">
            ZERO-TRUST CONSENSUS
        </span>
    </div>
    """)

    # 4 Key Invariants (Stitch HUD tiles)
    inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)
    with inv_col1:
        md_html(f"""
        <div class="tf-card" style="padding: 1rem; margin-bottom: 0.8rem;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#859397; text-transform:uppercase;">Attackers Tested</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#68F5B8;">{_tested_ratio}</div>
            <div style="font-size:0.75rem; color:#BBC9CD;">Poisoned nodes in experiment</div>
        </div>
        """)
    with inv_col2:
        md_html("""
        <div class="tf-card" style="padding: 1rem; margin-bottom: 0.8rem;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#859397; text-transform:uppercase;">Cosine Floor (τ)</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#22D3EE;">0.05</div>
            <div style="font-size:0.75rem; color:#BBC9CD;">Adversary weight bound</div>
        </div>
        """)
    with inv_col3:
        md_html("""
        <div class="tf-card" style="padding: 1rem; margin-bottom: 0.8rem;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#859397; text-transform:uppercase;">EMA Factor (α)</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#ADC6FF;">0.50</div>
            <div style="font-size:0.75rem; color:#BBC9CD;">Weight on the newest round</div>
        </div>
        """)
    with inv_col4:
        md_html("""
        <div class="tf-card" style="padding: 1rem; margin-bottom: 0.8rem;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#859397; text-transform:uppercase;">Raw Traffic Shared</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#68F5B8;">None</div>
            <div style="font-size:0.75rem; color:#BBC9CD;">Only model weights are sent</div>
        </div>
        """)

    # Core Architecture & Topography
    col_about_text, col_about_img = st.columns([1.25, 1], gap="large")

    with col_about_text:
        md_html(f"""
        <div class="tf-card">
            <div class="tf-card-title">
                <span style="color:#22D3EE;">⬡</span> How Trust-Weighted Federated Learning Works
            </div>
            <div class="tf-card-sub">
                A clear breakdown of the privacy-preserving principles and Byzantine defense mechanisms.
            </div>

            <h4 style="color:#DFE2F1; margin-top: 1rem; font-size: 0.95rem;">1. Privacy-Preserving Collaborative Training</h4>
            <p style="font-size: 0.86rem; color: #BBC9CD; line-height: 1.6;">
                Multiple distributed edge IoT networks train intrusion detection models locally on raw network flows. Rather than centralizing sensitive packet captures, edge nodes transmit only mathematical model weight deltas (gradients).
            </p>

            <h4 style="color:#DFE2F1; margin-top: 1rem; font-size: 0.95rem;">2. The Vulnerability: Poisoning & Model Replacement</h4>
            <p style="font-size: 0.86rem; color: #BBC9CD; line-height: 1.6;">
                Standard Federated Averaging (FedAvg) blindly averages client weights. A single compromised IoT gateway transmitting inverted labels or scaled weights corrupts the shared global model in as few as 1–2 rounds.
            </p>

            <h4 style="color:#DFE2F1; margin-top: 1rem; font-size: 0.95rem;">3. The Defense: Cosine-EMA Consensus & Excision</h4>
            <p style="font-size: 0.86rem; color: #BBC9CD; line-height: 1.6;">
                TrustFL computes high-dimensional vector alignments against the consensus direction (mean of unit-normalized updates):
            </p>
            <ul style="font-size: 0.84rem; color: #BBC9CD; line-height: 1.7;">
                <li><b>Directional Similarity:</b> Quantifies cosine angle between local updates and the consensus direction.</li>
                <li><b>Historical Memory:</b> Compounded over rounds via Exponential Moving Average (<code style="color:#22D3EE;">α = 0.5</code>).</li>
                <li><b>Dynamic Excision:</b> Divergent or malicious updates get their aggregation weight clamped to a floor (<code style="color:#22D3EE;">τ = 0.05</code>).</li>
            </ul>
        </div>

        <div class="tf-card">
            <div class="tf-card-title">
                <span style="color:#ADC6FF;">⚖</span> Mathematical Resilience: FedAvg vs. TrustFL
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 0.8rem;">
                <div style="background: #171B26; padding: 0.8rem; border-radius: 8px; border: 1px solid rgba(255,180,171,0.25);">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#FFB4AB; font-weight:700;">STANDARD FEDAVG</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#859397; margin: 0.3rem 0;">w_{{t+1}} = Σ (n_k/n · w_k)</div>
                    <div style="font-size:0.78rem; color:#BBC9CD;">Post-Attack Accuracy:</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; color:#FFB4AB; font-weight:700;">{_fa_txt}</div>
                </div>
                <div style="background: #171B26; padding: 0.8rem; border-radius: 8px; border: 1px solid rgba(104,245,184,0.25);">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#68F5B8; font-weight:700;">TRUSTFL ENCLAVE</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#859397; margin: 0.3rem 0;">w_{{t+1}} = Σ (Trust_k·n_k·w_k) / Σ (Trust_k·n_k)</div>
                    <div style="font-size:0.78rem; color:#BBC9CD;">Post-Attack Accuracy:</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; color:#68F5B8; font-weight:700;">{_ta_txt}</div>
                </div>
            </div>
        </div>
        """)

    with col_about_img:
        if MESH_IMG.exists():
            st.image(str(MESH_IMG), caption="Decentralized Edge IoT Mesh // Federated Shield Nodes", use_container_width=True)
        elif SHIELD_IMG.exists():
            st.image(str(SHIELD_IMG), caption="TrustFL Shield", use_container_width=True)

        md_html("""
        <div class="tf-card" style="margin-top: 1rem;">
            <div class="tf-card-title">Enclave Topography Specifications</div>
            <div style="font-size: 0.82rem; color: #BBC9CD; line-height: 1.8; font-family: 'JetBrains Mono', monospace;">
                <div><span style="color:#859397;">CONSENSUS:</span> <span style="color:#22D3EE;">Cosine Directional EMA</span></div>
                <div><span style="color:#859397;">ENCLAVE PROTO:</span> <span style="color:#68F5B8;">Weight-Update Sharing (no raw traffic)</span></div>
                <div><span style="color:#859397;">TOLERANCE:</span> <span style="color:#68F5B8;">Byzantine Label-Flipping & Amplification</span></div>
                <div><span style="color:#859397;">TARGET:</span> <span style="color:#DFE2F1;">Industrial IoT & Decentralized Sensors</span></div>
            </div>
        </div>
        """)

    # 4-Stage Neutralization Pipeline
    st.write("")
    st.markdown("### 🛡️ 4-Stage Byzantine Neutralization Pipeline")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        md_html("""
        <div class="tf-card" style="padding: 1rem; height: 100%;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#22D3EE; font-weight:700;">STAGE 01</div>
            <div style="font-weight:700; color:#DFE2F1; font-size:0.88rem; margin:0.3rem 0;">Decentralized Edge Telemetry</div>
            <div style="font-size:0.78rem; color:#BBC9CD; line-height:1.5;">
                Raw packet flows never leave edge nodes. Edge nodes perform local training on private telemetry.
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#22D3EE; margin-top:0.6rem; background:#171B26; padding:0.25rem 0.5rem; border-radius:4px;">
                Δw_i = w_local - w_global
            </div>
        </div>
        """)
    with pcol2:
        md_html("""
        <div class="tf-card" style="padding: 1rem; height: 100%;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#ADC6FF; font-weight:700;">STAGE 02</div>
            <div style="font-weight:700; color:#DFE2F1; font-size:0.88rem; margin:0.3rem 0;">Weight-Update Passing</div>
            <div style="font-size:0.78rem; color:#BBC9CD; line-height:1.5;">
                Only model weights exit edge nodes. (Authentication/signing is out of scope for this simulation.)
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#ADC6FF; margin-top:0.6rem; background:#171B26; padding:0.25rem 0.5rem; border-radius:4px;">
                Simulated via Flower
            </div>
        </div>
        """)
    with pcol3:
        md_html("""
        <div class="tf-card" style="padding: 1rem; height: 100%;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#68F5B8; font-weight:700;">STAGE 03</div>
            <div style="font-weight:700; color:#DFE2F1; font-size:0.88rem; margin:0.3rem 0;">Cosine EMA Consensus</div>
            <div style="font-size:0.78rem; color:#BBC9CD; line-height:1.5;">
                Server measures angular similarity against the consensus direction to update node reputation.
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#68F5B8; margin-top:0.6rem; background:#171B26; padding:0.25rem 0.5rem; border-radius:4px;">
                R_t = 0.5·s_t + 0.5·R_{t-1}
            </div>
        </div>
        """)
    with pcol4:
        md_html("""
        <div class="tf-card" style="padding: 1rem; height: 100%;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#FFB4AB; font-weight:700;">STAGE 04</div>
            <div style="font-weight:700; color:#DFE2F1; font-size:0.88rem; margin:0.3rem 0;">Dynamic Byzantine Excision</div>
            <div style="font-size:0.78rem; color:#BBC9CD; line-height:1.5;">
                Poisoned gradients with low or negative cosine similarity are down-weighted to τ = 0.05.
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#FFB4AB; margin-top:0.6rem; background:#171B26; padding:0.25rem 0.5rem; border-radius:4px;">
                w_{t+1} = Σ (Trust_i·n_i·w_i) / Σ (Trust_i·n_i)
            </div>
        </div>
        """)

    # Video Keynotes
    st.write("")
    md_html("""
    <div class="tf-card">
        <div class="tf-card-title">
            <span style="color:#22D3EE;">▶</span> Further Reading
        </div>
        <div class="tf-card-sub">
            Framework documentation and the original federated averaging paper.
        </div>
    </div>
    """)

    st.markdown(
        "- [Flower: federated learning framework docs](https://flower.ai/docs/)\n"
        "- [FedAvg paper: Communication-Efficient Learning from Decentralized Data](https://arxiv.org/abs/1602.05629)"
    )
