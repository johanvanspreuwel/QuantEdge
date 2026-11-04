"""
QuantEdge – Quantitative Trading Dashboard
===========================================
Direct uitvoerbaar: streamlit run quantedge_dashboard.py
Vereisten: pip install streamlit yfinance pandas numpy plotly requests scipy
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import re
import json
import os
from datetime import datetime, timedelta
import warnings
import math
import time
import logging
from html import escape as _esc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

# yfinance logt voor elke gedelisteerde/rate-limited ticker meerdere regels naar
# de Streamlit-logs. De app vangt die gevallen zelf netjes af, dus dempen we ruis.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:                                   # oudere yfinance-versies
    class YFRateLimitError(Exception):
        pass

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:                                   # pragma: no cover
    add_script_run_ctx = get_script_run_ctx = None

# ─────────────────────────────────────────────────────────────────────────────
# CENTRALE CONSTANTEN  –  alle aanpasbare parameters op één plek
# ─────────────────────────────────────────────────────────────────────────────

# ── Technische indicatoren ────────────────────────────────────────────────────
RSI_PERIOD            = 14    # Perioden voor RSI berekening
RSI_OVERSOLD          = 30    # RSI onder dit = oversold zone (grafiek lijn)
RSI_OVERBOUGHT        = 70    # RSI boven dit = overbought zone (grafiek lijn)
BB_PERIOD             = 20    # Bollinger Band SMA-periode
BB_STD_NORMAL         = 2.0   # BB standaard deviatie (normaal kanaal)
BB_STD_WIDE           = 2.5   # BB standaard deviatie (mean reversion signaal)
SUPPORT_WINDOW        = 30    # Rolling window (dagen) voor support/resistance
VOL_MA_PERIOD         = 20    # Volume moving average periode
MTF_WEEKLY_MA         = 10    # Weken MA voor wekelijkse macro trendcheck (1W)
MTF_HOURLY_AVG_BARS   = 8     # Uurstaven rolling gemiddelde voor volume check
MTF_VOLUME_SPIKE      = 1.10  # Volume moet ≥ dit × gemiddelde zijn = RISING

# ── Fase drempelwaarden (determine_phase) ─────────────────────────────────────
PHASE_BODEM_DEV_MAX   = 2.0   # Max afwijking% van support voor Bodemfase
PHASE_BODEM_RSI_MAX   = 43    # Max RSI voor Bodemfase
PHASE_HERSTEL_RSI_MIN = 42    # Min RSI voor Vroeg Herstel signaal (= SCAN_MOM_RSI_MIN, bewust gelijk)
PHASE_UPTREND_DEV_MIN = 5.0   # Min afwijking% voor Sterke Uptrend
PHASE_UPTREND_RSI_MIN = 60    # Min RSI voor Sterke Uptrend

# ── Actie drempelwaarden (determine_action + tabelkleuring) ──────────────────
ACTION_VOORZICHTIG_RSI = 75   # RSI boven dit = VOORZICHTIG
ACTION_VOORZICHTIG_DEV = 15.0 # Afwijking% boven dit = VOORZICHTIG
ACTION_AANHOUDEN_RSI   = 70   # RSI onder dit bij uptrend = AANHOUDEN
TABLE_RSI_GREEN        = 35   # RSI onder dit → groene RSI-cel in tabel
TABLE_RSI_RED          = 70   # RSI boven dit → rode RSI-cel in tabel
TABLE_DEV_GREEN        = 2.0  # Afwijking% onder dit → groene cel in tabel
TABLE_DEV_GOLD         = 10.0 # Afwijking% boven dit → gouden cel in tabel

# ── Strijdplan / risicobeheer ─────────────────────────────────────────────────
TRADE_RISK_PCT      = 0.02    # SL = support × (1 − RISK_PCT); 2% buffer onder support
TRADE_REWARD_RATIO  = 2.5     # Doel Risk:Reward verhouding voor TP2
TRADE_TP1_RATIO     = 0.60    # TP1 = 60% van de volledige R:R afstand (partieel exit)
TRADE_TP3_BUFFER    = 0.02    # TP3 = resistance × (1 − BUFFER); 2% onder resistance
TRADE_FALLBACK_SL   = 0.05    # Fallback SL als support NaN: koers × (1 − 5%)
TRADE_FALLBACK_RES  = 0.05    # Fallback resistance als data NaN: koers × (1 + 5%)

# ── Scanner strategie parameters ──────────────────────────────────────────────
SCAN_MOM_RSI_MIN      = 42    # RSI ondergrens Momentum Trein
SCAN_MOM_RSI_MAX      = 62    # RSI bovengrens Momentum Trein
SCAN_MOM_RUIMTE_MIN   = 2.0   # Min ruimte% tot weerstand
SCAN_MOM_RUIMTE_MAX   = 10.0  # Max ruimte% tot weerstand
SCAN_MEAN_RSI_OS      = 25    # RSI oversold grens Mean Reversion
SCAN_MEAN_RSI_OB      = 75    # RSI overbought grens Mean Reversion
SCAN_SQZ_WINDOW       = 120   # Dagen voor Volatiliteit Squeeze venster
SCAN_SQZ_THRESH       = 0.02  # BB Width within 2% of 120D minimum
SCAN_SR_DIST          = 1.5   # Max afstand% tot S/R voor Bounce signaal
SCAN_SENT_VOL_SPIKE   = 2.0   # Volume spike factor Event Sentiment (hard veto: vol_today ≥ 2× vol_20d_avg)
SCAN_SENT_NEWS_MIN    = 5     # Min nieuws items Event Sentiment (alleen als ook volume-veto is gehaald)
SCAN_SENT_SCORE_BULL  = 0.85  # Extreme bullish sentiment score drempel (>  0.85 = doorlaten)
SCAN_SENT_SCORE_BEAR  =-0.85  # Extreme bearish sentiment score drempel (< -0.85 = doorlaten)
SCAN_SENT_MAX_RESULTS = 5     # Max goedgekeurde Event Sentiment hits in eindtabel (alleen APPROVED)

# Alpha Scanner (Bodemfase / Waarde)
SCAN_ALPHA_RSI_MAX    = 45    # RSI ≤ dit (verhoogd van 42 → 45 voor meer ademruimte)
SCAN_ALPHA_DEV_MAX    = 5.0   # Koers mag max dit % boven support liggen (dicht bij bodem)
SCAN_ALPHA_DEV_MIN    = 0.0   # Koers mag NIET onder support liggen
SCAN_ALPHA_VOL_SPIKE  = 0.8   # Volume ≥ 0.8× 20D gemiddelde (verlaagd: bodem = laag volume)

# S/R Bounce Scanner (Pullback Springplank)
SCAN_SR_BOUNCE_MAX    = 1.5   # Max afstand% boven support voor bounce-signaal (= SCAN_SR_DIST)
SCAN_SR_BOUNCE_VOL    = 1.2   # Volume vandaag ≥ 1.2× gemiddelde (licht herstel volume)

# Warrior Trading Momentum Scanner (Low Float Momentum)
SCAN_WT_PRICE_MIN     = 2.0   # Min koers in USD
SCAN_WT_PRICE_MAX     = 20.0  # Max koers in USD
SCAN_WT_CHANGE_MIN    = 10.0  # Min dagelijkse stijging % (10%)
SCAN_WT_RVOL_MIN      = 5.0   # Min Relative Volume vs 20D gemiddelde (5×)
SCAN_WT_FLOAT_MAX     = 10_000_000   # Max float (10 miljoen vrij verhandelbare aandelen)

# ── BB Momentum Scanner ────────────────────────────────────────────────────────
SCAN_BB_PERIOD        = 20    # SMA-periode voor Bollinger Bands
SCAN_BB_STD           = 2.0   # Standaarddeviatie Bollinger Bands
SCAN_BB_MIN_DAYS      = 2     # Aantal aaneengesloten dagen 'positief' (instelbaar in UI)
SCAN_BB_UB_TOLERANCE  = 0.005 # Optie B: sluit binnen 0.5% van Upper Band = telt mee als 'aan de band'

# ── Candlestick patroon verhoudingen ──────────────────────────────────────────
CANDLE_DOJI_MAX_BODY    = 0.05  # Lichaam ≤ 5% van totale range = Doji
CANDLE_HAMMER_SHADOW    = 2     # Onderste schaduw ≥ 2× lichaam = Hammer
CANDLE_HAMMER_TOP_MAX   = 0.10  # Max bovenste schaduw als % van range = Hammer
CANDLE_HAMMER_CLOSE_MIN = 0.75  # Sluiting ≥ 75% van range-bodem = Hammer
CANDLE_STAR_SHADOW      = 2     # Bovenste schaduw ≥ 2× lichaam = Shooting Star
CANDLE_STAR_LOW_MAX     = 0.10  # Max onderste schaduw als % van range = Shooting Star
CANDLE_STAR_CLOSE_MAX   = 0.35  # Sluiting ≤ 35% van range-bodem = Shooting Star

# ── Data & caching ────────────────────────────────────────────────────────────
CACHE_PRICE_TTL       = 300   # Seconden cache koersdata (5 min)
CACHE_POOL_TTL        = 3600  # Seconden cache tickerpool (1 uur)
DATA_MAIN_PERIOD      = "3mo" # Periode hoofdtabel
DATA_SCAN_PERIOD      = "6mo" # Periode scanner (terug naar 6mo — geen bulk-download meer)
DATA_WL_PERIOD        = "1mo" # Periode watchlist
DATA_MTF_WEEKLY       = "1y"  # Periode wekelijkse MTF check
DATA_MTF_HOURLY       = "5d"  # Periode uurlijkse MTF check
DATA_MIN_ROWS_MAIN    = 30    # Min rijen voor hoofdtabel
DATA_MIN_ROWS_SCAN    = 50    # Min rijen voor scanner (oude, robuustere waarde)
DATA_MIN_ROWS_FETCH   = 5     # Min rijen voor geldige datafetch

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantEdge · QT Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS – Dark Terminal Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

  :root {
    color-scheme: dark;
    --bg-primary:   #0B0E11;
    --bg-secondary: #13171C;
    --bg-card:      #1A1F27;
    --bg-hover:     #222831;
    --gold:         #F0B90B;
    --gold-dim:     #A07C08;
    --green:        #00C853;
    --green-dim:    #00843A;
    --red:          #F6465D;
    --red-dim:      #A32040;
    --blue:         #2196F3;
    --text-primary: #E8ECEF;
    --text-muted:   #848E9C;
    --border:       #2B3139;
    --radius:       12px;
  }

  /* ── Basis ─────────────────────────────────────────────────────────────── */
  html { -webkit-text-size-adjust: 100%; }
  html, body, [data-testid="stApp"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  * { -webkit-tap-highlight-color: transparent; }
  h1, h2, h3, h4, h5 { color: var(--gold) !important; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.2px; }
  h3 { font-size: 1.25rem !important; margin-bottom: .2rem !important; }

  [data-testid="stHeader"] { background: rgba(11,14,17,.88) !important; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
  [data-testid="stToolbarActions"], [data-testid="stDecoration"], footer { display: none !important; }
  [data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }

  .block-container { padding: 3.5rem 1.2rem calc(4rem + env(safe-area-inset-bottom)) !important; max-width: 1500px; }

  /* ── Invoer ────────────────────────────────────────────────────────────── */
  input, textarea, select { background-color: var(--bg-card) !important; color: var(--text-primary) !important;
                            border: 1px solid var(--border) !important; border-radius: 10px !important; }
  .stTextInput > div > div, .stSelectbox > div > div, .stTextArea > div > div { background-color: var(--bg-card) !important; border-radius: 10px !important; }
  .stSelectbox > div > div { color: var(--text-primary) !important; min-height: 44px; }
  input:focus, textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 1px var(--gold) !important; }

  /* ── Knoppen ───────────────────────────────────────────────────────────── */
  .stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-hover));
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 10px;
    min-height: 44px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: .3px;
    transition: transform .12s ease, box-shadow .2s ease, background .2s ease;
  }
  .stButton > button:active { transform: scale(.97); background: var(--gold-dim) !important; color: #000 !important; }
  @media (hover: hover) {
    .stButton > button:hover { background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
                               color: #000 !important; box-shadow: 0 4px 15px rgba(240,185,11,.3); }
  }

  /* ── Tabs: horizontaal scrollbaar, grote tikvlakken ────────────────────── */
  .stTabs [data-baseweb="tab-list"] { background-color: var(--bg-secondary); border-bottom: 2px solid var(--gold);
      overflow-x: auto; flex-wrap: nowrap; scrollbar-width: none; gap: 2px; border-radius: 10px 10px 0 0; }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
  .stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace;
      white-space: nowrap; min-height: 46px; padding: 0 16px; font-size: .85rem; }
  .stTabs [aria-selected="true"] { color: var(--gold) !important; }
  .stTabs [data-baseweb="tab-highlight"] { background-color: var(--gold) !important; }

  .stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px; overflow: hidden; }

  [data-testid="stMetric"] { background-color: var(--bg-card); padding: 12px 16px; border-radius: var(--radius); border-left: 3px solid var(--gold); }
  [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: .75rem !important; }
  [data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'JetBrains Mono', monospace !important; }

  [data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; background: var(--bg-secondary); }
  [data-testid="stExpander"] summary { min-height: 46px; }
  hr { border-color: var(--border) !important; margin: .8rem 0 !important; }
  .stAlert { background-color: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
  .stRadio > div { background-color: var(--bg-card); padding: 8px 12px; border-radius: 10px; }
  [data-testid="stProgress"] div[role="progressbar"] > div { background-color: var(--gold) !important; }

  .section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 20px;
  }

  /* ── Header ────────────────────────────────────────────────────────────── */
  .header-bar {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    background: linear-gradient(90deg, var(--bg-secondary), var(--bg-card));
    border: 1px solid var(--border); border-left: 3px solid var(--gold);
    border-radius: var(--radius);
    padding: 10px 14px; margin-bottom: 14px;
    font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: var(--text-muted);
  }
  .header-bar .hb-brand { color: var(--gold); font-weight: 700; letter-spacing: 1px; font-size: .95rem; }
  .header-bar .hb-time  { color: var(--green); }
  .header-bar .hb-dot   { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green);
                          margin-right:6px; animation: qe-pulse 2s ease-in-out infinite; }
  @keyframes qe-pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

  /* ── KPI-tegels ────────────────────────────────────────────────────────── */
  .qe-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 6px 0 12px; }
  .qe-kpis.n6 { grid-template-columns: repeat(6, 1fr); }
  .qe-kpi  { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius);
             padding: 10px 12px; border-left: 3px solid var(--gold); }
  .qe-kpi .k-l { font-size: .7rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .qe-kpi .k-v { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--gold); line-height: 1.2; }
  .qe-kpi.g { border-left-color: var(--green); } .qe-kpi.g .k-v { color: var(--green); }
  .qe-kpi.r { border-left-color: var(--red); }   .qe-kpi.r .k-v { color: var(--red); }

  /* ── Ticker-kaarten (tik = uitklappen, geen herlaadactie nodig) ────────── */
  .qe-list { display: flex; flex-direction: column; gap: 8px; animation: qe-fade .25s ease-out; }
  @keyframes qe-fade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
  .qe-card { background: var(--bg-secondary); border: 1px solid var(--border); border-left: 4px solid var(--border);
             border-radius: var(--radius); overflow: hidden; }
  .qe-card.t-g { border-left-color: var(--green); background: linear-gradient(90deg, rgba(0,200,83,.07), var(--bg-secondary) 40%); }
  .qe-card.t-y { border-left-color: var(--gold);  background: linear-gradient(90deg, rgba(240,185,11,.07), var(--bg-secondary) 40%); }
  .qe-card.t-r { border-left-color: var(--red); }
  .qe-card.t-x { border-left-color: var(--border); opacity: .6; }
  .qe-card > summary { list-style: none; cursor: pointer; padding: 10px 12px; min-height: 48px; }
  .qe-card > summary::-webkit-details-marker { display: none; }
  .qe-card > summary:active { background: var(--bg-hover); }
  .qe-r1, .qe-r2 { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .qe-r2 { margin-top: 6px; }
  .qe-tk  { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem; color: var(--gold); }
  .qe-px  { font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: var(--text-primary); text-align: right; }
  .qe-ext { display: block; font-size: .68rem; color: var(--text-muted); }
  .qe-kpis.n6 .k-v { font-size: 1.1rem; }
  .qe-chg { font-family: 'JetBrains Mono', monospace; font-size: .8rem; font-weight: 600; color: var(--text-muted); }
  .qe-chg.up { color: var(--green); } .qe-chg.dn { color: var(--red); }
  .qe-chips { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .qe-chip { font-family: 'JetBrains Mono', monospace; font-size: .7rem; padding: 2px 7px; border-radius: 20px;
             background: var(--bg-card); border: 1px solid var(--border); color: var(--text-muted); white-space: nowrap; }
  .qe-chip b { color: var(--text-primary); font-weight: 600; }
  .qe-chip.g { background: rgba(0,132,58,.35); border-color: var(--green-dim); color: #CFF5DD; }
  .qe-chip.g b { color: #fff; }
  .qe-chip.r { background: rgba(163,32,64,.4); border-color: var(--red-dim); color: #FFD3DA; }
  .qe-chip.r b { color: #fff; }
  .qe-chip.y { background: rgba(160,124,8,.25); border-color: var(--gold-dim); color: var(--gold); }
  .qe-act { font-size: .72rem; font-weight: 700; padding: 2px 9px; border-radius: 20px; white-space: nowrap;
            background: var(--bg-card); color: var(--text-muted); border: 1px solid var(--border); }
  .qe-act.g { background: var(--green-dim); color: #fff; border-color: var(--green); }
  .qe-act.y { background: rgba(160,124,8,.25); color: var(--gold); border-color: var(--gold-dim); }
  .qe-act.r { background: var(--red-dim); color: #fff; border-color: var(--red); }
  .qe-body { padding: 4px 12px 12px; border-top: 1px dashed var(--border); display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
  .qe-body .f-l { font-size: .66rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; }
  .qe-body .f-v { font-family: 'JetBrains Mono', monospace; font-size: .85rem; color: var(--text-primary); word-break: break-word; }
  .qe-body .wide { grid-column: 1 / -1; }
  .qe-empty { padding: 18px; text-align: center; color: var(--text-muted); border: 1px dashed var(--border); border-radius: var(--radius); }

  /* Rijen met key="qe_row_*" blijven ook op een telefoon naast elkaar staan */
  [class*="st-key-qe_row"] [data-testid="stHorizontalBlock"] { flex-direction: row !important; flex-wrap: nowrap !important; gap: .5rem !important; }
  [class*="st-key-qe_row"] [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
  .st-key-qe_row_top [data-testid="stColumn"]:last-child { flex: 0 0 64px !important; }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-secondary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Telefoon ──────────────────────────────────────────────────────────── */
  @media (max-width: 640px) {
    .block-container { padding: 3.2rem .6rem calc(5rem + env(safe-area-inset-bottom)) !important; }
    /* 16px voorkomt dat iOS Safari inzoomt bij het tikken op een invoerveld */
    input, textarea, select, .stSelectbox div[data-baseweb="select"] * { font-size: 16px !important; }
    .section-card { padding: 12px 10px; border-radius: 10px; margin-bottom: 14px; }
    h3 { font-size: 1.1rem !important; }
    .qe-kpis, .qe-kpis.n6 { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .qe-kpi .k-v { font-size: 1.15rem; word-break: break-word; }
    .qe-kpi .k-v { font-size: 1.35rem; }
    .header-bar .hb-long { display: none; }
    .stTabs [data-baseweb="tab"] { padding: 0 12px; font-size: .8rem; }
    [data-testid="stMetric"] { padding: 8px 10px; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    .stButton > button { width: 100%; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
_DAGEN   = ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"]
_MAANDEN = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


def _now_nl() -> datetime:
    """Huidige tijd in Nederland (Streamlit Cloud draait in UTC)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Amsterdam"))
    except Exception:
        return datetime.now()


_n = _now_nl()
now_str = f"{_DAGEN[_n.weekday()][:2]} {_n.day} {_MAANDEN[_n.month - 1]} · {_n:%H:%M}"
st.markdown(f"""
<div class="header-bar">
  <span class="hb-brand">⬛ QUANTEDGE</span>
  <span class="hb-long">QUANTITATIVE TRADING DASHBOARD · yfinance · MULTI-TF</span>
  <span class="hb-time"><span class="hb-dot"></span>{now_str}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OPSLAG  —  Streamlit Cloud compatibel
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    # Tech, Semis & Software
    "NVDA","ASML.AS","MSFT","AMD","TSLA","ARM","SAP","CSCO","MU",
    "ADYEN.AS","TRMB","SITE","CMCSA","FMC","COIN","INTU",
    "KR","HPQ","CRWD","PANW","CRM","MA","PYPL","AMZN","NFLX","DIS",
    # Healthcare & Consumer
    "ZTS","BAYN.DE","JNJ","ABBV","OR.PA","PG","PEP","AD.AS","KO",
    "MCD","UNA.AS","PFE","MRNA","ILMN","NKE","SBUX",
    # Finance & Real Estate
    "JPM","ADP","V","O","ESRT",
    # Industrials & Materials
    "CAT","WM","LOW","COST","LMT","RTX","FDX","UPS","EME",
    "ALB","NEM","GOLD","FCX",
    # Commodities & Precious Metals ETFs
    "GLD","SLV","USO",
    # European Tickers
    "MAERSK-B.CO","SHELL.AS","RHM.DE","MC.PA","WMT.DE",
    # Traditional Energy — Oil & Gas
    "XOM","CVX","COP","EOG","OXY","FANG","DVN","APA",
    "EQT","SHEL","BP","TTE",
    # Traditional Energy — Refining & Infrastructure
    "MPC","VLO","PSX","KMI","WMB","MPLX",
    # Traditional Energy — Equipment & Services
    "SLB","HAL","BKR",
    # Power Generation & Utilities
    "CEG","VST","NEE","DUK","SO","D","AEP","ED","PCG","SRE",
    "EIX","ETR","FE","AES","XEL",
    # Clean Tech & Alternative Energy
    "GEV","ENPH","SEDG","BEP","RUN","PLUG",
    # Energy ETFs
    "XLE","VDE"]

DEFAULT_WATCHLIST = {
    "MCD":  "MCD",
    "AMZN": "alles",
    "RTX":  "defensie",
    "TRMB": "defensie",
    "ALB":  "ALB",
    "CEG":  "CEG",
    "DVN":  "DVN",
    "EME":  "EME",
    "EQT":  "EQT",
    "ESRT": "ESRT",
}

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENTE OPSLAG via GitHub API
# Werkt op Streamlit Cloud én lokaal.
# Vereist in Streamlit Secrets:
#   GITHUB_TOKEN = "ghp_..."
#   GITHUB_REPO  = "gebruikersnaam/QuantEdge"
#   GITHUB_FILE  = "quantedge_userdata.json"
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_storage():
    """Lokale JSON fallback voor thuis gebruik."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantedge_userdata.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data.get('tickers'), list) and isinstance(data.get('watchlist'), dict):
                return data
    except Exception:
        pass
    return None

def _save_local_storage() -> None:
    """Sla op naar lokaal JSON-bestand (thuis op pc)."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantedge_userdata.json")
        data = {'tickers': list(st.session_state.main_market_tickers),
                'watchlist': dict(st.session_state.custom_watchlist)}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _get_github_config():
    """Haal GitHub configuratie op uit Streamlit Secrets of omgevingsvariabelen."""
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo  = st.secrets.get("GITHUB_REPO",  "")
        file  = st.secrets.get("GITHUB_FILE",  "quantedge_userdata.json")
        if token and repo:
            return token, repo, file
    except Exception:
        pass
    return None, None, "quantedge_userdata.json"

def _load_from_github():
    """Laad tickers en watchlist uit GitHub. Retourneert None als niet beschikbaar."""
    token, repo, file = _get_github_config()
    if not token:
        return None
    try:
        url  = f"https://api.github.com/repos/{repo}/contents/{file}"
        hdrs = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        resp = requests.get(url, headers=hdrs, timeout=10)
        if resp.status_code == 200:
            import base64
            content = base64.b64decode(resp.json()["content"]).decode("utf-8")
            data    = json.loads(content)
            if isinstance(data.get("tickers"), list) and isinstance(data.get("watchlist"), dict):
                return data
        elif resp.status_code == 404:
            return None  # Bestand bestaat nog niet — eerste keer
    except Exception:
        pass
    return None

def _save_to_github():
    """Sla tickers en watchlist op in GitHub."""
    token, repo, file = _get_github_config()
    if not token:
        return False, "Geen GitHub token"
    try:
        import base64
        data = {
            "tickers":   list(st.session_state.main_market_tickers),
            "watchlist": dict(st.session_state.custom_watchlist),
        }
        content_b64 = base64.b64encode(
            json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        ).decode("utf-8")
        url  = f"https://api.github.com/repos/{repo}/contents/{file}"
        hdrs = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        sha = None
        resp = requests.get(url, headers=hdrs, timeout=15)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
        elif resp.status_code == 401:
            return False, "GitHub token ongeldig (401)"
        payload = {"message": "QuantEdge: tickers bijgewerkt", "content": content_b64}
        if sha:
            payload["sha"] = sha
        resp = requests.put(url, headers=hdrs, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            return True, "OK"
        return False, f"GitHub fout: {resp.status_code}"
    except Exception as e:
        return False, str(e)


def _save_userdata() -> None:
    """Sla op via GitHub (cloud) en lokaal (thuis)."""
    ok, msg = _save_to_github()
    if not ok:
        st.session_state["_last_save_error"] = msg
    else:
        st.session_state["_last_save_error"] = None
    _save_local_storage()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# Laadvolgorde:
#   1. GitHub (Streamlit Cloud) — persistent over alle sessies
#   2. Lokaal JSON-bestand (thuis op pc)
#   3. DEFAULT_TICKERS (ingebakken in code = jouw 72 tickers)
# ─────────────────────────────────────────────────────────────────────────────

if 'main_market_tickers' not in st.session_state:
    # Probeer GitHub eerst
    gh_data = _load_from_github()
    if gh_data:
        st.session_state.main_market_tickers = gh_data['tickers']
        st.session_state.custom_watchlist    = gh_data['watchlist']
    else:
        # Lokaal bestand als fallback
        local = _get_local_storage()
        if local:
            st.session_state.main_market_tickers = local['tickers']
            st.session_state.custom_watchlist    = local['watchlist']
        else:
            st.session_state.main_market_tickers = list(DEFAULT_TICKERS)
            st.session_state.custom_watchlist    = dict(DEFAULT_WATCHLIST)

if 'custom_watchlist' not in st.session_state:
    st.session_state.custom_watchlist = dict(DEFAULT_WATCHLIST)

# Live Monitor — vastgepinde tickers voor real-time tracking
if 'pinned_tickers' not in st.session_state:
    st.session_state['pinned_tickers'] = []
if 'live_monitor_active' not in st.session_state:
    st.session_state['live_monitor_active'] = False
if 'live_refresh_interval' not in st.session_state:
    st.session_state['live_refresh_interval'] = 30   # seconden

if 'current_alpha' not in st.session_state:
    st.session_state.current_alpha = 'NVDA'

if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = pd.DataFrame()

if 'active_strategy' not in st.session_state:
    st.session_state.active_strategy = None

if 'bb_min_days' not in st.session_state:
    st.session_state['bb_min_days'] = SCAN_BB_MIN_DAYS  # Default: 2 aaneengesloten dagen

if 'scan_pool_size' not in st.session_state:
    st.session_state['scan_pool_size'] = 0

# Pre-initialiseer pool-tellers met hardcoded minimums zodat de sidebar
# nooit '–' of '0' toont vóór de eerste scan. De werkelijke aantallen
# worden bijgewerkt zodra get_large_ticker_pool() wordt aangeroepen.
if 'pool_sp500' not in st.session_state:
    st.session_state['pool_sp500']  = '488 (hardcoded)'
if 'pool_midcap' not in st.session_state:
    st.session_state['pool_midcap'] = '350 (hardcoded)'
if 'pool_europe' not in st.session_state:
    st.session_state['pool_europe'] = '~80'
if 'pool_extras' not in st.session_state:
    st.session_state['pool_extras'] = '~270'
if 'total_ticker_count' not in st.session_state:
    st.session_state['total_ticker_count'] = '~1000+ (laden...)'
if 'pool_scrape_status' not in st.session_state:
    st.session_state['pool_scrape_status'] = (
        'Pool nog niet geladen — klik op Scanner tab om te initialiseren'
    )

# ── Sidebar pool-diagnostics (live tellers zodra pool geladen is) ─────────────
with st.sidebar:
    st.markdown("### 🔬 Ticker Pool Status")
    _sp   = st.session_state.get('pool_sp500',  '–')
    _mid  = st.session_state.get('pool_midcap', '–')
    _eu   = st.session_state.get('pool_europe', '–')
    _ex   = st.session_state.get('pool_extras', '–')
    _tot  = st.session_state.get('total_ticker_count', '–')
    st.markdown(f"""
    <div style="font-family:monospace; font-size:0.8rem; color:#E8ECEF;
                background:#13171C; padding:10px 12px; border-radius:6px;
                border:1px solid #2B3139;">
      📈 S&P 500 &nbsp;&nbsp;&nbsp;: <b style="color:#F0B90B;">{_sp}</b><br>
      🏢 MidCap 400 : <b style="color:#F0B90B;">{_mid}</b><br>
      🇪🇺 Europa Top : <b style="color:#F0B90B;">{_eu}</b><br>
      ➕ Extra's &nbsp;&nbsp;&nbsp;: <b style="color:#F0B90B;">{_ex}</b><br>
      <hr style="border-color:#2B3139; margin:6px 0;">
      📊 <b>Totaal</b> &nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#00C853; font-size:1rem;">{_tot}</b>
    </div>
    """, unsafe_allow_html=True)
    scrape_info = st.session_state.get('pool_scrape_status', 'Pool nog niet geladen')
    st.caption(scrape_info)
    if st.button("🔄 Pool herladen", key="btn_reload_pool"):
        st.cache_data.clear()
        # Reset pool-tellers zodat de sidebar 'laden...' toont
        st.session_state['total_ticker_count'] = '~1000+ (laden...)'
        st.session_state['pool_scrape_status'] = 'Pool wordt herladen...'
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """Bereken RSI voor een prijsserie. Retourneert geheel getal."""
    if len(series) < period + 1:
        return float('nan')
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)))  # Geen decimalen


def compute_rsi_series(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Bereken een volledige RSI-serie."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bollinger_series(series: pd.Series, period: int = BB_PERIOD, std_mult: float = BB_STD_NORMAL):
    """Retourneert upper, mid, lower als volledige Series."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def compute_support_resistance(df: pd.DataFrame, window: int = SUPPORT_WINDOW, current_price: float = None):
    """
    Robuuste support/resistance berekening met drielaagse fallback procedure.
    Voorkomt $nan output bij korte tijdshorizons of dunne data.
    """

    support = resistance = float('nan')

    if df is not None and not df.empty and len(df) > 0:
        # Laag 1: rolling min/max met min_periods=1 zodat ook korte series werken
        effective_window = min(window, len(df))
        calc_support    = df['Low'].rolling(window=effective_window, min_periods=1).min().iloc[-1]
        calc_resistance = df['High'].rolling(window=effective_window, min_periods=1).max().iloc[-1]

        # Laag 2: als het nóg NaN is, pak de absolute min/max van de volledige reeks
        if math.isnan(float(calc_support)) or float(calc_support) == 0:
            calc_support = df['Low'].dropna().min()
        if math.isnan(float(calc_resistance)) or float(calc_resistance) == 0:
            calc_resistance = df['High'].dropna().max()

        support    = float(calc_support)
        resistance = float(calc_resistance)

    # Laag 3: ultieme failsafe op basis van huidige koers
    ref_price = current_price if (current_price and current_price > 0) else 1.0
    if math.isnan(support)    or support == 0:
        support    = ref_price * (1 - TRADE_FALLBACK_SL)
    if math.isnan(resistance) or resistance == 0:
        resistance = ref_price * (1 + TRADE_FALLBACK_RES)

    # Sanity check: support mag nooit groter zijn dan resistance
    if support > resistance:
        support, resistance = resistance, support

    return round(support, 4), round(resistance, 4)


def detect_candlestick_pattern(df: pd.DataFrame) -> str:
    """
    Detecteer candlestick-patronen op de laatste 3 kaarsen. MultiIndex-safe.

    Patronen (bullish):
      Doji, Bullish Hammer, Inverted Hammer, Bullish Engulfing,
      Morning Star, Bullish Harami, Piercing Line, Three White Soldiers,
      Bullish Marubozu, Tweezer Bottom

    Patronen (bearish):
      Bearish Engulfing, Shooting Star, Hanging Man, Evening Star,
      Bearish Harami, Dark Cloud Cover, Three Black Crows,
      Bearish Marubozu, Tweezer Top
    """
    try:
        if df is None or len(df) < 3:
            return "Geen data"

        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] for col in df.columns]

        def _sc(val):
            if hasattr(val, 'iloc'): val = val.iloc[0]
            return float(val)

    
        c3 = df.iloc[-3]   # 3 kaarsen geleden
        c2 = df.iloc[-2]   # gisteren
        c1 = df.iloc[-1]   # vandaag

        # Extraheer OHLC voor alle drie
        o1,h1,l1,c1v = _sc(c1['Open']),_sc(c1['High']),_sc(c1['Low']),_sc(c1['Close'])
        o2,h2,l2,c2v = _sc(c2['Open']),_sc(c2['High']),_sc(c2['Low']),_sc(c2['Close'])
        o3,h3,l3,c3v = _sc(c3['Open']),_sc(c3['High']),_sc(c3['Low']),_sc(c3['Close'])

        if any(math.isnan(v) for v in [o1,h1,l1,c1v,o2,h2,l2,c2v,o3,h3,l3,c3v]):
            return "Geen data"

        # Hulpvariabelen vandaag
        body1        = abs(c1v - o1)
        range1       = h1 - l1 if h1-l1 > 0 else 0.0001
        lower_sh1    = min(o1,c1v) - l1
        upper_sh1    = h1 - max(o1,c1v)
        is_bull1     = c1v > o1
        is_bear1     = c1v < o1

        # Hulpvariabelen gisteren
        body2        = abs(c2v - o2)
        range2       = h2 - l2 if h2-l2 > 0 else 0.0001
        lower_sh2    = min(o2,c2v) - l2
        upper_sh2    = h2 - max(o2,c2v)
        is_bull2     = c2v > o2
        is_bear2     = c2v < o2

        # Hulpvariabelen eergisteren
        body3        = abs(c3v - o3)
        is_bull3     = c3v > o3
        is_bear3     = c3v < o3

        avg_body = (body1 + body2 + body3) / 3 if (body1+body2+body3) > 0 else 0.0001

        # ── DOJI ─────────────────────────────────────────────────────────────
        if body1 / range1 <= CANDLE_DOJI_MAX_BODY:
            if upper_sh1 > 2 * lower_sh1 and upper_sh1 > body1 * 2:
                return "Gravestone Doji"       # lange bovenste schaduw
            if lower_sh1 > 2 * upper_sh1 and lower_sh1 > body1 * 2:
                return "Dragonfly Doji"        # lange onderste schaduw
            return "Doji"

        # ── BULLISH HAMMER ────────────────────────────────────────────────────
        if (lower_sh1 >= CANDLE_HAMMER_SHADOW * body1 and
                upper_sh1 <= CANDLE_HAMMER_TOP_MAX * range1 and
                c1v >= l1 + CANDLE_HAMMER_CLOSE_MIN * range1):
            return "Bullish Hammer"

        # ── INVERTED HAMMER (bullish na neerwaartse trend) ────────────────────
        if (upper_sh1 >= CANDLE_HAMMER_SHADOW * body1 and
                lower_sh1 <= CANDLE_HAMMER_TOP_MAX * range1 and
                is_bull2 == False):           # vorige kaars was bearish
            return "Inverted Hammer"

        # ── HANGING MAN (bearish na opwaartse trend) ──────────────────────────
        if (lower_sh1 >= CANDLE_HAMMER_SHADOW * body1 and
                upper_sh1 <= CANDLE_HAMMER_TOP_MAX * range1 and
                is_bull2):                    # vorige kaars was bullish
            return "Hanging Man"

        # ── SHOOTING STAR ─────────────────────────────────────────────────────
        if (upper_sh1 >= CANDLE_STAR_SHADOW * body1 and
                lower_sh1 <= CANDLE_STAR_LOW_MAX * range1 and
                c1v <= l1 + CANDLE_STAR_CLOSE_MAX * range1):
            return "Shooting Star"

        # ── BULLISH ENGULFING ─────────────────────────────────────────────────
        if is_bull1 and is_bear2 and c1v > o2 and o1 < c2v:
            return "Bullish Engulfing"

        # ── BEARISH ENGULFING ─────────────────────────────────────────────────
        if is_bear1 and is_bull2 and c1v < o2 and o1 > c2v:
            return "Bearish Engulfing"

        # ── BULLISH HARAMI (klein groen lichaam in groot rood lichaam) ─────────
        if (is_bull1 and is_bear2 and
                o1 > c2v and c1v < o2 and
                body1 < body2 * 0.5):
            return "Bullish Harami"

        # ── BEARISH HARAMI ────────────────────────────────────────────────────
        if (is_bear1 and is_bull2 and
                o1 < c2v and c1v > o2 and
                body1 < body2 * 0.5):
            return "Bearish Harami"

        # ── PIERCING LINE (bullish) ────────────────────────────────────────────
        if (is_bear2 and is_bull1 and
                o1 < l2 and                            # opent onder vorig dieptepunt
                c1v > (o2 + c2v) / 2 and              # sluit boven midden vorige kaars
                c1v < o2):                             # maar nog onder vorige open
            return "Piercing Line"

        # ── DARK CLOUD COVER (bearish) ────────────────────────────────────────
        if (is_bull2 and is_bear1 and
                o1 > h2 and                            # opent boven vorig hoogtepunt
                c1v < (o2 + c2v) / 2 and              # sluit onder midden vorige kaars
                c1v > o2):                             # maar nog boven vorige open
            return "Dark Cloud Cover"

        # ── MORNING STAR (bullish, 3-kaars patroon) ───────────────────────────
        if (is_bear3 and                               # kaars 3 geleden: bearish
                body2 < avg_body * 0.3 and            # gisteren: kleine kaars (ster)
                is_bull1 and                           # vandaag: bullish
                c1v > (o3 + c3v) / 2):                # sluit boven midden eerste kaars
            return "Morning Star"

        # ── EVENING STAR (bearish, 3-kaars patroon) ───────────────────────────
        if (is_bull3 and
                body2 < avg_body * 0.3 and
                is_bear1 and
                c1v < (o3 + c3v) / 2):
            return "Evening Star"

        # ── THREE WHITE SOLDIERS (bullish) ────────────────────────────────────
        if (is_bull1 and is_bull2 and is_bull3 and
                c1v > c2v > c3v and                    # elke sluit hoger
                o1 > o2 > o3 and                       # elke open hoger
                body1 > avg_body * 0.6 and             # stevige lichamen
                body2 > avg_body * 0.6 and
                body3 > avg_body * 0.6):
            return "Three White Soldiers"

        # ── THREE BLACK CROWS (bearish) ───────────────────────────────────────
        if (is_bear1 and is_bear2 and is_bear3 and
                c1v < c2v < c3v and
                o1 < o2 < o3 and
                body1 > avg_body * 0.6 and
                body2 > avg_body * 0.6 and
                body3 > avg_body * 0.6):
            return "Three Black Crows"

        # ── BULLISH MARUBOZU (geen schaduwen, sterke koop) ────────────────────
        if (is_bull1 and
                upper_sh1 <= range1 * 0.02 and
                lower_sh1 <= range1 * 0.02 and
                body1 > avg_body * 1.5):
            return "Bullish Marubozu"

        # ── BEARISH MARUBOZU ──────────────────────────────────────────────────
        if (is_bear1 and
                upper_sh1 <= range1 * 0.02 and
                lower_sh1 <= range1 * 0.02 and
                body1 > avg_body * 1.5):
            return "Bearish Marubozu"

        # ── TWEEZER BOTTOM (bullish) ──────────────────────────────────────────
        if (is_bear2 and is_bull1 and
                abs(l1 - l2) / (range1 + range2 + 0.0001) < 0.02):  # gelijke dieptepunten
            return "Tweezer Bottom"

        # ── TWEEZER TOP (bearish) ─────────────────────────────────────────────
        if (is_bull2 and is_bear1 and
                abs(h1 - h2) / (range1 + range2 + 0.0001) < 0.02):  # gelijke hoogtepunten
            return "Tweezer Top"

        return "Geen Patroon"

    except Exception:
        return "Geen Patroon"


def determine_phase(rsi: float, deviation_pct: float, pattern: str) -> str:
    """Bepaal de marktfase/status op basis van RSI, afwijking en patroon."""
    if 0 <= deviation_pct <= PHASE_BODEM_DEV_MAX and rsi < PHASE_BODEM_RSI_MAX:
        return "⚡ Bodemfase (Steun Test)"
    if "BULLISH" in pattern.upper() and rsi > PHASE_HERSTEL_RSI_MIN:
        return "🌱 Vroeg Herstel (Trend Ommekeer)"
    if deviation_pct > PHASE_UPTREND_DEV_MIN and rsi > PHASE_UPTREND_RSI_MIN:
        return "🚀 Sterke Uptrend (Uitbraak Bevestigd)"
    return "〰 Consolidatie"


def determine_action(rsi: float, deviation_pct: float, pattern: str, phase: str) -> str:
    """Bepaal de actie-aanbeveling."""
    if 'Bodemfase' in phase or 'Vroeg Herstel' in phase:
        return "🟢 KOOPWAARDIG"
    if 'Uptrend' in phase and rsi < ACTION_AANHOUDEN_RSI:
        return "🟡 AANHOUDEN"
    if rsi > ACTION_VOORZICHTIG_RSI or deviation_pct > ACTION_VOORZICHTIG_DEV:
        return "🔴 VOORZICHTIG"
    if 'Bullish' in pattern:
        return "🟢 KOOPWAARDIG"
    return "⚪ NEUTRAAL"


@st.cache_data(ttl=60)   # 1 minuut cache — live prijs verandert snel
def fetch_live_price(ticker: str) -> dict:
    """
    Haal de meest actuele prijs op inclusief after-hours / pre-market.

    yfinance geeft via t.info de volgende velden terug:
      - currentPrice       : reguliere slotkoers (of intraday)
      - postMarketPrice    : after-hours prijs (na 16:00 ET)
      - preMarketPrice     : pre-market prijs (voor 09:30 ET)
      - regularMarketPrice : reguliere marktprijs

    Retourneert een dict met:
      'price'       : meest actuele prijs (aftermarket > premarket > regular)
      'regular'     : reguliere slotkoers
      'extended'    : after/pre-market prijs (None als niet beschikbaar)
      'market_phase': 'AFTER-HOURS' | 'PRE-MARKET' | 'REGULAR' | 'CLOSED'
      'change_ext'  : % verandering extended t.o.v. regular (None als n.v.t.)
    """
    result = {
        'price': None, 'regular': None, 'extended': None,
        'market_phase': 'CLOSED', 'change_ext': None,
    }
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        regular  = info.get('regularMarketPrice') or info.get('currentPrice')
        postmkt  = info.get('postMarketPrice')
        premkt   = info.get('preMarketPrice')

        result['regular'] = regular

        if postmkt and postmkt != regular:
            result['extended']     = postmkt
            result['price']        = postmkt
            result['market_phase'] = 'AFTER-HOURS'
        elif premkt and premkt != regular:
            result['extended']     = premkt
            result['price']        = premkt
            result['market_phase'] = 'PRE-MARKET'
        else:
            result['price']        = regular
            result['market_phase'] = 'REGULAR'

        # Bereken % verandering extended t.o.v. regular
        if result['extended'] and regular and regular != 0:
            result['change_ext'] = round(
                ((result['extended'] - regular) / regular) * 100, 2
            )

    except Exception:
        pass

    return result


@st.cache_data(ttl=CACHE_PRICE_TTL)
def fetch_ticker_data_fast(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """
    Snelle data-fetch ZONDER t.info — gebruikt door de Scanner, die alleen
    OHLCV-data nodig heeft en geen bedrijfsmetadata (currency, sector, etc.).
    t.info is de traagste en meest rate-limit-gevoelige yfinance-call;
    die overslaan scheelt bij 1000+ tickers een aanzienlijke hoeveelheid tijd
    en verkleint de kans op een Yahoo 429 (Too Many Requests) flink.
    """
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty or len(df) < DATA_MIN_ROWS_SCAN:
            # Fallback voor Europese/exotische tickers waarbij .history() leeg is
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < DATA_MIN_ROWS_SCAN:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] for c in df.columns]
        required = {'Open','High','Low','Close','Volume'}
        if not required.issubset(set(df.columns)):
            return None
        df = df.dropna(subset=['Close'])
        return df if len(df) >= DATA_MIN_ROWS_SCAN else None
    except Exception:
        return None


@st.cache_data(ttl=CACHE_PRICE_TTL)
def fetch_ticker_data(ticker: str, period: str = "3mo"):
    """
    Haal OHLCV-data op via yfinance met caching.
    Gebruikt prepost=True zodat extended-hours kaarsen beschikbaar zijn.
    Fallback naar yf.download() voor Europese/exotische tickers waarbij
    .history() soms leeg teruggeeft ondanks geldige ticker.
    """
    def _clean_df(df):
        """Normaliseer MultiIndex, valideer kolommen, verwijder NaN-close rijen."""
        if df is None or df.empty or len(df) < DATA_MIN_ROWS_FETCH:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] for col in df.columns]
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(set(df.columns)):
            return None
        df = df.dropna(subset=['Close'])
        if len(df) < DATA_MIN_ROWS_FETCH:
            return None
        for col in list(df.columns):
            if hasattr(df[col], 'squeeze'):
                df[col] = df[col].squeeze()
        return df

    try:
        t  = yf.Ticker(ticker)
        df = t.history(period=period, prepost=True)
        df = _clean_df(df)

        # Fallback: yf.download werkt soms beter voor Europese/speciale tickers
        if df is None:
            df_dl = yf.download(ticker, period=period, interval="1d",
                                progress=False, auto_adjust=True)
            df = _clean_df(df_dl)

        if df is None:
            return None, None

        info = {}
        try:
            info = t.info
        except Exception:
            pass

        return df, info

    except Exception:
        return None, None


@st.cache_data(ttl=60)   # 1 minuut cache — sluit aan op 5M kaars-interval
def fetch_intraday_patterns(ticker: str) -> dict:
    """
    Haal de laatste 2 kaarsen op voor 15M en 5M tijdschalen.
    Retourneert patroon + richting (Bullish/Bearish) voor elke kaars.
    Cache TTL = 60s zodat nieuwe 5M kaarsen snel worden opgepikt.
    """
    result = {
        '15M[-1]': '–', '15M[0]': '–',
        '5M[-1]':  '–', '5M[0]':  '–',
    }
    try:
        def _get_pattern_and_dir(df_tf, label):
            """Geeft patroon-string terug, of Bullish/Bearish als geen patroon."""
            if df_tf is None or len(df_tf) < 3:
                return '–'
            if isinstance(df_tf.columns, pd.MultiIndex):
                df_tf = df_tf.copy()
                df_tf.columns = [c[0] for c in df_tf.columns]
            df_tf = df_tf.dropna(subset=['Close'])
            if len(df_tf) < 3:
                return '–'
            pat = detect_candlestick_pattern(df_tf)
            # Als geen specifiek patroon, toon richting van de kaars
            if pat == 'Geen Patroon':
                try:
                    c = float(df_tf['Close'].iloc[-1])
                    o = float(df_tf['Open'].iloc[-1])
                    return '🟢 Bullish' if c >= o else '🔴 Bearish'
                except Exception:
                    return 'Geen Patroon'
            return pat

        def _get_prev_pattern_and_dir(df_tf):
            """Geeft patroon voor de kaars vóór de laatste ([-1])."""
            if df_tf is None or len(df_tf) < 4:
                return '–'
            if isinstance(df_tf.columns, pd.MultiIndex):
                df_tf = df_tf.copy()
                df_tf.columns = [c[0] for c in df_tf.columns]
            df_prev = df_tf.iloc[:-1]   # Verwijder laatste kaars → [-1] wordt de huidige
            if len(df_prev) < 3:
                return '–'
            pat = detect_candlestick_pattern(df_prev)
            if pat == 'Geen Patroon':
                try:
                    c = float(df_prev['Close'].iloc[-1])
                    o = float(df_prev['Open'].iloc[-1])
                    return '🟢 Bullish' if c >= o else '🔴 Bearish'
                except Exception:
                    return 'Geen Patroon'
            return pat

        # 15M data ophalen (laatste 2 dagen = genoeg voor patroonherkenning)
        # Ticker.history is thread-safe (yf.download deelt globale state)
        df_15m = yf.Ticker(ticker).history(period='2d', interval='15m')
        if df_15m is not None and len(df_15m) >= 4:
            result['15M[0]']  = _get_pattern_and_dir(df_15m, '15M[0]')
            result['15M[-1]'] = _get_prev_pattern_and_dir(df_15m)

        # 5M data ophalen (laatste 1 dag)
        df_5m = yf.Ticker(ticker).history(period='1d', interval='5m')
        if df_5m is not None and len(df_5m) >= 4:
            result['5M[0]']  = _get_pattern_and_dir(df_5m, '5M[0]')
            result['5M[-1]'] = _get_prev_pattern_and_dir(df_5m)

    except Exception:
        pass

    return result


@st.cache_data(ttl=60)
def compute_15m_mean_reversion(ticker: str) -> dict:
    """
    15M Mean Reversion Scanner — detecteert vroegtijdige trendomkeer signalen.

    Drie scenario's:
      1. BULLISH REVERSION — Failed Breakdown:
         Koers test/breekt Opening Range Low (ORL) of 15M support,
         gevolgd door Morning Star / Hammer / Bullish Engulfing op hoog volume.

      2. BEARISH REVERSION — Failed Breakout:
         Koers test/breekt Opening Range High (ORH) of 15M resistance,
         gevolgd door Evening Star / Shooting Star / Bearish Engulfing op hoog volume.

      3. Volumefilter (verplicht):
         Volume trigger-kaars >= VOL_MULTI x gemiddelde volume afgelopen 5 kaarsen.
    """
    VOL_MULTI      = 1.5    # Volume multiplier voor validatie
    ORL_BUFFER_PCT = 0.003  # 0.3% buffer rond ORL/ORH voor "test"

    result = {
        'status':    'Geen Signaal',
        'detail':    '',
        'direction': None,   # 'BULLISH' | 'BEARISH' | None
        'vol_ok':    False,
        'vol_ratio': 0.0,
        'pattern':   '',
        'orl':       None,
        'orh':       None,
    }

    try:
        # ── Data ophalen ─────────────────────────────────────────────────────
        df = yf.Ticker(ticker).history(period='2d', interval='15m')
        if df is None or len(df) < 6:
            result['status'] = '– Geen data'
            return result

        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy(); df.columns = [c[0] for c in df.columns]
        df = df.dropna(subset=['Close'])
        if len(df) < 6:
            result['status'] = '– Te weinig kaarsen'
            return result

        # ── Opening Range berekenen (eerste 2 kaarsen van vandaag) ───────────
        # Haal kaarsen van vandaag op
        today = pd.Timestamp.now().normalize()
        try:
            df_today = df[df.index.normalize() >= today]
        except Exception:
            df_today = df.iloc[-16:]   # Fallback: laatste 16 kaarsen (~4 uur)

        if len(df_today) >= 2:
            orl = float(df_today['Low'].iloc[:2].min())    # Opening Range Low
            orh = float(df_today['High'].iloc[:2].max())   # Opening Range High
        else:
            # Fallback op dagelijkse min/max van beschikbare data
            orl = float(df['Low'].iloc[-16:].min())
            orh = float(df['High'].iloc[-16:].max())

        result['orl'] = round(orl, 4)
        result['orh'] = round(orh, 4)

        # ── Huidige en vorige kaarsen ─────────────────────────────────────────
        last  = df.iloc[-1]    # Trigger-kaars (laatste gesloten 15M kaars)
        prev  = df.iloc[-2]    # Kaars daarvoor
        prev2 = df.iloc[-3]    # Twee kaarsen terug

        def _sc(v):
            return float(v.iloc[0]) if hasattr(v, 'iloc') else float(v)

        c_last  = _sc(last['Close']);  o_last  = _sc(last['Open'])
        h_last  = _sc(last['High']);   l_last  = _sc(last['Low'])
        vol_last= _sc(last['Volume'])

        c_prev  = _sc(prev['Close']);  o_prev  = _sc(prev['Open'])
        h_prev  = _sc(prev['High']);   l_prev  = _sc(prev['Low'])

        c_prev2 = _sc(prev2['Close']); o_prev2 = _sc(prev2['Open'])

        # ── Volume check (trigger-kaars vs gemiddelde afgelopen 5) ────────────
        vol_5avg = float(df['Volume'].iloc[-6:-1].mean())
        vol_ratio = round(vol_last / vol_5avg, 2) if vol_5avg > 0 else 0.0
        vol_ok = vol_ratio >= VOL_MULTI
        result['vol_ok']    = vol_ok
        result['vol_ratio'] = vol_ratio

        # ── Hulpvariabelen trigger-kaars ──────────────────────────────────────
        body_last  = abs(c_last - o_last)
        range_last = h_last - l_last if h_last - l_last > 0 else 0.0001
        bull_last  = c_last > o_last
        bear_last  = c_last < o_last
        bull_prev  = c_prev > o_prev
        bear_prev  = c_prev < o_prev
        bull_prev2 = c_prev2 > o_prev2
        bear_prev2 = c_prev2 < o_prev2

        lower_sh = min(o_last, c_last) - l_last
        upper_sh = h_last - max(o_last, c_last)

        # ── BULLISH PATROON DETECTIE ──────────────────────────────────────────
        bullish_pat = None

        # Morning Star: bearish → kleine ster → bullish boven midden kaars 1
        if (bear_prev2 and
                abs(c_prev - o_prev) < abs(c_prev2 - o_prev2) * 0.4 and
                bull_last and
                c_last > (o_prev2 + c_prev2) / 2):
            bullish_pat = 'Morning Star'

        # Hammer: lange onderste schaduw, kleine bovenschaduw, sluit hoog
        elif (lower_sh >= 2 * body_last and
              upper_sh <= 0.1 * range_last and
              c_last >= l_last + 0.75 * range_last):
            bullish_pat = 'Hammer'

        # Bullish Engulfing: vorige bearish, huidige bullish en omsluit
        elif (bear_prev and bull_last and
              c_last > o_prev and o_last < c_prev):
            bullish_pat = 'Bullish Engulfing'

        # Bullish Harami: kleine bullish kaars binnen grote bearish kaars
        elif (bear_prev and bull_last and
              o_last > c_prev and c_last < o_prev and
              body_last < abs(c_prev - o_prev) * 0.5):
            bullish_pat = 'Bullish Harami'

        # Inverted Hammer (na daling)
        elif (upper_sh >= 2 * body_last and
              lower_sh <= 0.1 * range_last and
              bear_prev):
            bullish_pat = 'Inverted Hammer'

        # ── BEARISH PATROON DETECTIE ──────────────────────────────────────────
        bearish_pat = None

        # Evening Star: bullish → kleine ster → bearish onder midden kaars 1
        if (bull_prev2 and
                abs(c_prev - o_prev) < abs(c_prev2 - o_prev2) * 0.4 and
                bear_last and
                c_last < (o_prev2 + c_prev2) / 2):
            bearish_pat = 'Evening Star'

        # Shooting Star: lange bovenschaduw, sluit laag
        elif (upper_sh >= 2 * body_last and
              lower_sh <= 0.1 * range_last and
              c_last <= l_last + 0.35 * range_last):
            bearish_pat = 'Shooting Star'

        # Bearish Engulfing
        elif (bull_prev and bear_last and
              c_last < o_prev and o_last > c_prev):
            bearish_pat = 'Bearish Engulfing'

        # Bearish Harami
        elif (bull_prev and bear_last and
              o_last < c_prev and c_last > o_prev and
              body_last < abs(c_prev - o_prev) * 0.5):
            bearish_pat = 'Bearish Harami'

        # Hanging Man (na stijging)
        elif (lower_sh >= 2 * body_last and
              upper_sh <= 0.1 * range_last and
              bull_prev):
            bearish_pat = 'Hanging Man'

        # ── REVERSION SIGNAAL BEPALEN ─────────────────────────────────────────
        # Bullish reversion: bullish patroon NA test van ORL
        if bullish_pat:
            # Test van ORL: laagste punt van vorige/huidige kaars <= ORL + buffer
            tested_orl = (l_prev <= orl * (1 + ORL_BUFFER_PCT) or
                          l_last  <= orl * (1 + ORL_BUFFER_PCT))
            if tested_orl:
                status = f'🟢 BULLISH REVERSION [{bullish_pat}]'
                if vol_ok:
                    status += f' ✅ Vol {vol_ratio}x'
                else:
                    status += f' ⚠ Vol {vol_ratio}x (laag)'
                result['status']    = status
                result['direction'] = 'BULLISH'
                result['pattern']   = bullish_pat
                result['detail']    = f'ORL={orl:.2f} getest · {bullish_pat} · Vol {vol_ratio}x'
                return result
            else:
                # Bullish patroon maar geen ORL-test → zwakker signaal
                if vol_ok:
                    result['status']    = f'🟡 BULLISH SETUP [{bullish_pat}] · Vol {vol_ratio}x'
                    result['direction'] = 'BULLISH'
                    result['pattern']   = bullish_pat
                    result['detail']    = f'Geen ORL-test · {bullish_pat} · Vol {vol_ratio}x'

        # Bearish reversion: bearish patroon NA test van ORH
        if bearish_pat:
            tested_orh = (h_prev >= orh * (1 - ORL_BUFFER_PCT) or
                          h_last  >= orh * (1 - ORL_BUFFER_PCT))
            if tested_orh:
                status = f'🔴 BEARISH REVERSION [{bearish_pat}]'
                if vol_ok:
                    status += f' ✅ Vol {vol_ratio}x'
                else:
                    status += f' ⚠ Vol {vol_ratio}x (laag)'
                result['status']    = status
                result['direction'] = 'BEARISH'
                result['pattern']   = bearish_pat
                result['detail']    = f'ORH={orh:.2f} getest · {bearish_pat} · Vol {vol_ratio}x'
                return result
            else:
                if vol_ok and result['direction'] != 'BULLISH':
                    result['status']    = f'🟡 BEARISH SETUP [{bearish_pat}] · Vol {vol_ratio}x'
                    result['direction'] = 'BEARISH'
                    result['pattern']   = bearish_pat
                    result['detail']    = f'Geen ORH-test · {bearish_pat} · Vol {vol_ratio}x'

        # Geen patroon gevonden
        if result['direction'] is None:
            result['status'] = '⚪ Geen Signaal'

    except Exception as e:
        result['status'] = f'– Fout: {str(e)[:30]}'

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HOOFDTABEL — parallel, per ticker gecachet, mislukkingen worden NIET gecachet
#
# Waarom zo:
#   Eerder liep één grote gecachete functie sequentieel over alle tickers en deed
#   per ticker ~6 Yahoo-aanroepen (history, 2× .info, 15M, 5M, 15M-MR). Zodra
#   Yahoo ging throttlen werden de mislukte tickers 5 minuten lang als "N/A"
#   gecachet. Nu:
#     • dagdata + indicatoren per ticker in een eigen cache (exception = niet gecachet)
#     • 6 tickers tegelijk, met backoff bij rate-limits
#     • geen .info meer voor de valuta (afgeleid van het ticker-suffix)
#     • 15M/5M-kolommen apart en alleen voor tickers die dagdata hebben
# ─────────────────────────────────────────────────────────────────────────────
MAIN_TABLE_WORKERS    = 6     # Gelijktijdige Yahoo-aanroepen (hoger = sneller, maar sneller rate-limit)
MAIN_TABLE_REUSE_SECS = 300   # Hergebruik de samengestelde tabel zo lang bij widget-interacties
_EUR_SUFFIXES = {'AS', 'DE', 'PA', 'MI', 'MC', 'BR', 'HE', 'LS', 'VI', 'IR', 'F'}
_INTRA_COLS   = ['15M [-1]', '15M [0]', '5M [-1]', '5M [0]', '15M MR Status']


def _currency_symbol(ticker: str) -> str:
    """Valutateken zonder netwerkaanroep (t.info was de traagste yfinance-call)."""
    t = str(ticker).upper()
    if '.' not in t:
        return '$'
    return '€' if t.rsplit('.', 1)[1] in _EUR_SUFFIXES else ''


def _explain_error(exc_name: str) -> str:
    if 'RateLimit' in exc_name:
        return 'Yahoo rate-limit — vernieuw zo'
    if 'Missing' in exc_name or 'Delisted' in exc_name or exc_name == 'ValueError':
        return 'Geen koersdata (delisted?)'
    return 'Tijdelijk niet beschikbaar'


def _parallel_map(func, items, workers: int = MAIN_TABLE_WORKERS, on_progress=None) -> dict:
    """
    Voer func(item) voor alle items parallel uit. Retourneert {item: resultaat}.
    on_progress(done, total) wordt vanuit de hoofd-thread aangeroepen, dus het
    is veilig om daar een st.progress bij te werken.
    """
    items = list(items)
    results = {}
    if not items:
        return results
    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def _init():
        # Geeft de worker-threads toegang tot st.cache_data zonder waarschuwingen
        if ctx is not None and add_script_run_ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)

    with ThreadPoolExecutor(max_workers=workers, initializer=_init) as ex:
        futs = {ex.submit(func, it): it for it in items}
        done = 0
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                results[it] = fut.result()
            except Exception as e:                      # worker mag nooit de app slopen
                results[it] = e
            done += 1
            if on_progress:
                on_progress(done, len(items))
    return results


@st.cache_data(ttl=CACHE_PRICE_TTL, show_spinner=False)
def _fetch_history(ticker: str, period: str, min_rows: int) -> pd.DataFrame:
    """
    Dagkoersen voor één ticker, zonder .info. Bij geen data of een fout wordt een
    exception gegooid: st.cache_data cachet exceptions NIET, dus een tijdelijke
    Yahoo-storing blijft niet 5 minuten als 'N/A' hangen.
    """
    df, last_exc = None, None
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, raise_errors=True)
            last_exc = None
            break
        except YFRateLimitError as e:                  # backoff en opnieuw proberen
            last_exc = e
            time.sleep(1.2 * (attempt + 1))
        except Exception as e:                         # delisted e.d.: niet opnieuw proberen
            last_exc = e
            break
    if last_exc is not None:
        raise last_exc

    if df is None or df.empty:
        raise ValueError("Geen koersdata")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    if not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(set(df.columns)):
        raise ValueError("Onvolledige data")
    df = df.dropna(subset=['Close'])
    if len(df) < min_rows:
        raise ValueError("Te weinig data")
    return df


@st.cache_data(ttl=CACHE_PRICE_TTL, show_spinner=False)
def _fetch_daily_core(ticker: str, period: str) -> dict:
    """Dagdata + indicatoren voor één ticker (exception bij mislukken → niet gecachet)."""
    df = _fetch_history(ticker, period, DATA_MIN_ROWS_MAIN)

    close      = df['Close'].squeeze()
    last_price = round(float(close.iloc[-1]), 2)
    rsi_val    = compute_rsi(close, RSI_PERIOD)
    support_30, resistance_30 = compute_support_resistance(df, SUPPORT_WINDOW, current_price=last_price)
    deviation  = round(((last_price - support_30) / support_30) * 100, 2) if support_30 > 0 else 0.0
    pattern    = detect_candlestick_pattern(df)
    phase      = determine_phase(rsi_val, deviation, pattern)
    action     = determine_action(rsi_val, deviation, pattern, phase)
    prev       = float(close.iloc[-2]) if len(close) >= 2 else 0.0
    day_chg    = round((last_price / prev - 1) * 100, 2) if prev > 0 else 0.0

    return {
        'last_price': last_price, 'rsi': rsi_val,
        'support': support_30, 'resistance': resistance_30,
        'deviation': deviation, 'pattern': pattern,
        'phase': phase, 'action': action, 'day_chg': day_chg,
    }


def _daily_worker(ticker: str) -> dict:
    try:
        core = _fetch_daily_core(ticker, DATA_MAIN_PERIOD)
    except Exception as e:
        return {'ok': False, 'ticker': ticker, 'err': _explain_error(type(e).__name__)}
    live = fetch_live_price(ticker)          # 60s cache; mislukking → val terug op slotkoers
    return {'ok': True, 'ticker': ticker, 'core': core, 'live': live}


def _row_from_result(res: dict) -> dict:
    ticker = res['ticker']
    if not res.get('ok'):
        return {
            'Ticker': ticker, 'Koers': 'N/A', 'Dag %': 'N/A', 'RSI (14D)': 'N/A',
            'Support (30D)': 'N/A', 'Weerstand (30D)': 'N/A', 'Afwijking %': 'N/A',
            '_dev_float': np.nan, '_rsi_float': np.nan, '_ext_chg': np.nan, '_day_chg': np.nan,
            'Patroon (1D)': 'N/A',
            '15M [-1]': '–', '15M [0]': '–', '5M [-1]': '–', '5M [0]': '–',
            '15M MR Status': '–', '_mr15_dir': '',
            'Koers Status / Fase': '⚠ Geen Data', 'Actie': '⚠ Geen Data',
            '_ok': False, '_err': res.get('err', ''),
        }

    core, live = res['core'], (res.get('live') or {})
    symbol        = _currency_symbol(ticker)
    last_price    = core['last_price']
    display_price = live.get('price') or last_price
    ext_price     = live.get('extended')
    market_phase  = live.get('market_phase')
    change_ext    = live.get('change_ext')

    if ext_price and market_phase in ('AFTER-HOURS', 'PRE-MARKET'):
        chg_str = ''
        if change_ext is not None:
            arrow   = '▲' if change_ext >= 0 else '▼'
            chg_str = f" {arrow}{abs(change_ext):.2f}%"
        badge     = '🌙' if market_phase == 'AFTER-HOURS' else '🌅'
        koers_str = f"{symbol}{display_price:,.2f} {badge}{chg_str}"
    else:
        koers_str = f"{symbol}{display_price:,.2f}"

    rsi_val        = core['rsi']
    safe_deviation = core['deviation'] if not math.isnan(core['deviation']) else 0.0
    rsi_is_nan     = rsi_val is None or (isinstance(rsi_val, float) and np.isnan(rsi_val))

    return {
        'Ticker': ticker,
        'Koers': koers_str,
        'Dag %': f"{'▲' if core['day_chg'] >= 0 else '▼'} {abs(core['day_chg']):.2f}%",
        'RSI (14D)': 'N/A' if rsi_is_nan else str(int(rsi_val)),
        'Support (30D)': f"{symbol}{core['support']:,.2f}",
        'Weerstand (30D)': f"{symbol}{core['resistance']:,.2f}",
        'Afwijking %': f"{safe_deviation:.2f}",
        '_dev_float': safe_deviation,
        '_rsi_float': 50.0 if rsi_is_nan else float(rsi_val),
        '_ext_chg': change_ext if (ext_price and market_phase in ('AFTER-HOURS', 'PRE-MARKET')) else None,
        '_day_chg': core['day_chg'],
        'Patroon (1D)': core['pattern'],
        '15M [-1]': '–', '15M [0]': '–', '5M [-1]': '–', '5M [0]': '–',
        '15M MR Status': '–', '_mr15_dir': '',
        'Koers Status / Fase': core['phase'],
        'Actie': core['action'],
        '_ok': True, '_err': '',
    }


def _finalize_main_df(rows: list) -> pd.DataFrame:
    df_result = pd.DataFrame(rows)
    # Forceer display-kolommen naar string zodat Arrow nooit crasht op gemengde types
    for _col in ['RSI (14D)', 'Afwijking %', 'Koers', 'Dag %', 'Support (30D)', 'Weerstand (30D)',
                 'Patroon (1D)', '15M [-1]', '15M [0]', '5M [-1]', '5M [0]',
                 '15M MR Status', 'Koers Status / Fase', 'Actie']:
        if _col in df_result.columns:
            df_result[_col] = df_result[_col].astype(str)
    for _col in ('_ext_chg', '_day_chg', '_dev_float', '_rsi_float'):
        if _col in df_result.columns:
            df_result[_col] = pd.to_numeric(df_result[_col], errors='coerce')
    return df_result


def build_main_table(tickers: tuple, progress=None) -> pd.DataFrame:
    """Fase 1: dagdata + live prijs, parallel. Geeft altijd één rij per ticker terug."""
    uniq    = list(dict.fromkeys(tickers))
    results = _parallel_map(_daily_worker, uniq, on_progress=progress)
    rows = []
    for t in uniq:
        r = results.get(t)
        if not isinstance(r, dict):                     # worker crashte onverwacht
            r = {'ok': False, 'ticker': t, 'err': 'Tijdelijk niet beschikbaar'}
        rows.append(_row_from_result(r))
    return _finalize_main_df(rows)


def _intraday_worker(ticker: str) -> dict:
    try:
        intra = fetch_intraday_patterns(ticker)
    except Exception:
        intra = {'15M[-1]': '–', '15M[0]': '–', '5M[-1]': '–', '5M[0]': '–'}
    try:
        mr15 = compute_15m_mean_reversion(ticker)
    except Exception:
        mr15 = {'status': '–', 'direction': None}
    return {
        '15M [-1]': intra.get('15M[-1]', '–'), '15M [0]': intra.get('15M[0]', '–'),
        '5M [-1]':  intra.get('5M[-1]', '–'),  '5M [0]':  intra.get('5M[0]', '–'),
        '15M MR Status': mr15.get('status', '–'),
        '_mr15_dir': mr15.get('direction') or '',
    }


def add_intraday_columns(df: pd.DataFrame, progress=None) -> pd.DataFrame:
    """Fase 2: 15M/5M-patronen en 15M Mean Reversion — alleen voor tickers met dagdata."""
    if df is None or df.empty or '_ok' not in df.columns:
        return df
    ok_tickers = df.loc[df['_ok'] == True, 'Ticker'].tolist()   # noqa: E712
    results = _parallel_map(_intraday_worker, ok_tickers, on_progress=progress)
    out = df.copy()
    for col in _INTRA_COLS + ['_mr15_dir']:
        out[col] = [
            (results.get(t) or {}).get(col, '–' if col != '_mr15_dir' else '')
            if isinstance(results.get(t), dict) else ('–' if col != '_mr15_dir' else '')
            for t in out['Ticker']
        ]
        if col != '_mr15_dir':
            out[col] = out[col].astype(str)
    return out


def refresh_market_data() -> None:
    """Wis alléén de koersdata-caches (niet de dure ticker-pool) en de sessie-cache."""
    for fn in (_fetch_history, _fetch_daily_core, fetch_live_price, fetch_intraday_patterns,
               compute_15m_mean_reversion, fetch_ticker_data, fetch_ticker_data_fast):
        try:
            fn.clear()
        except Exception:
            pass
    st.session_state.pop('_main_cache', None)



# ─────────────────────────────────────────────────────────────────────────────
# UI-HELPERS  (iPhone-first)
# ─────────────────────────────────────────────────────────────────────────────
def _is_mobile_ua() -> bool:
    """Telefoon-herkenning via de User-Agent van het verzoek (Streamlit ≥ 1.37)."""
    try:
        ua = str(st.context.headers.get("User-Agent", "")).lower()
    except Exception:
        return False
    return any(k in ua for k in ("iphone", "ipod", "android", "mobile"))


def load_main_df(tickers: tuple, want_intra: bool):
    """
    Geeft (df, laadtijd). De samengestelde tabel wordt MAIN_TABLE_REUSE_SECS in de
    sessie bewaard, zodat tikken op filters/zoekvelden niets opnieuw hoeft op te halen.
    Intraday-kolommen (15M/5M) worden alleen berekend als de tabelweergave actief is.
    """
    if not tickers:
        return pd.DataFrame(), time.time()

    cache = st.session_state.get('_main_cache')
    now   = time.time()
    fresh = bool(cache) and cache['key'] == tickers and (now - cache['ts']) < MAIN_TABLE_REUSE_SECS

    if not fresh:
        bar = st.progress(0.0, text=f"Koersen laden… 0/{len(tickers)}")

        def _cb(done, total):
            bar.progress(done / total, text=f"Koersen laden… {done}/{total}")

        try:
            df = build_main_table(tickers, progress=_cb)
        finally:
            bar.empty()
        cache = {'key': tickers, 'ts': now, 'df': df, 'intra': False}
        st.session_state['_main_cache'] = cache

    if want_intra and not cache['intra']:
        bar = st.progress(0.0, text="Intraday-signalen laden…")

        def _cb2(done, total):
            bar.progress(done / total, text=f"Intraday-signalen laden… {done}/{total}")

        try:
            cache['df'] = add_intraday_columns(cache['df'], progress=_cb2)
            cache['intra'] = True
        finally:
            bar.empty()

    return cache['df'], cache['ts']


def render_kpis(items: list) -> str:
    """items = [(label, waarde, kleurklasse '' | 'g' | 'r'), ...] → HTML-grid (2×2 op telefoon)."""
    tiles = ''.join(
        f'<div class="qe-kpi {cls}"><div class="k-l">{_esc(str(lab))}</div><div class="k-v">{_esc(str(val))}</div></div>'
        for lab, val, cls in items
    )
    return f'<div class="qe-kpis">{tiles}</div>'


MAIN_FILTERS = ["Alles", "🟢 Koopwaardig", "⚡ Bodem", "🚀 Uptrend", "🌱 Herstel", "⚠ Geen data"]
MAIN_SORTS   = ["Standaard", "Dag % ↓ (stijgers)", "Dag % ↑ (dalers)", "RSI laag → hoog", "Dichtst bij support", "A → Z"]


def filter_sort_main(df: pd.DataFrame, flt: str, query: str, sort: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    phase, action = out['Koers Status / Fase'], out['Actie']
    if flt == "🟢 Koopwaardig":
        out = out[action.str.contains('KOOPWAARDIG', na=False)]
    elif flt == "⚡ Bodem":
        out = out[phase.str.contains('Bodemfase', na=False)]
    elif flt == "🚀 Uptrend":
        out = out[phase.str.contains('Uptrend', na=False)]
    elif flt == "🌱 Herstel":
        out = out[phase.str.contains('Herstel', na=False)]
    elif flt == "⚠ Geen data":
        out = out[out['_ok'] == False]                  # noqa: E712

    terms = [t for t in re.split(r'[\s,;]+', (query or '').upper()) if t]
    if terms:
        out = out[out['Ticker'].str.upper().apply(lambda tk: any(t in tk for t in terms))]

    if sort == "Dag % ↓ (stijgers)":
        out = out.sort_values('_day_chg', ascending=False, na_position='last')
    elif sort == "Dag % ↑ (dalers)":
        out = out.sort_values('_day_chg', ascending=True, na_position='last')
    elif sort == "RSI laag → hoog":
        out = out.sort_values('_rsi_float', ascending=True, na_position='last')
    elif sort == "Dichtst bij support":
        out = out.sort_values('_dev_float', ascending=True, na_position='last')
    elif sort == "A → Z":
        out = out.sort_values('Ticker')
    return out


def _card_tone(phase: str, action: str, ok: bool) -> str:
    if not ok:
        return 't-x'
    if 'KOOPWAARDIG' in action or 'Bodemfase' in phase or 'Vroeg Herstel' in phase:
        return 't-g'
    if 'Uptrend' in phase:
        return 't-y'
    if 'VOORZICHTIG' in action:
        return 't-r'
    return ''


def _field(label: str, value: str, wide: bool = False) -> str:
    return (f'<div class="{"wide" if wide else ""}"><div class="f-l">{_esc(label)}</div>'
            f'<div class="f-v">{_esc(str(value))}</div></div>')


def _px_html(koers) -> str:
    """Koers + (optioneel) after-/pre-market-regel apart, zodat die niet voor dagverandering doorgaat."""
    txt = str(koers)
    m = re.match(r'^(.*?)\s+([🌙🌅].*)$', txt)
    if not m:
        return _esc(txt)
    return f'{_esc(m.group(1))}<small class="qe-ext">{_esc(m.group(2))}</small>'


def render_ticker_cards(df: pd.DataFrame) -> str:
    """Compacte kaartenlijst. <details> klapt lokaal uit, dus zonder server-rerun."""
    if df is None or df.empty:
        return '<div class="qe-empty">Geen tickers voor dit filter.</div>'
    parts = ['<div class="qe-list">']
    for _, r in df.iterrows():
        ok     = bool(r.get('_ok', True))
        phase  = str(r.get('Koers Status / Fase', ''))
        action = str(r.get('Actie', ''))
        tone   = _card_tone(phase, action, ok)
        tk     = _esc(str(r['Ticker']))

        if not ok:
            parts.append(
                f'<details class="qe-card {tone}"><summary>'
                f'<div class="qe-r1"><span class="qe-tk">{tk}</span><span class="qe-act">⚠ Geen data</span></div>'
                f'</summary><div class="qe-body">{_field("Reden", r.get("_err") or "Onbekend", True)}</div></details>'
            )
            continue

        rsi_f, dev_f, dc = r.get('_rsi_float'), r.get('_dev_float'), r.get('_day_chg')
        rsi_cls = 'g' if (pd.notna(rsi_f) and rsi_f < TABLE_RSI_GREEN) else 'r' if (pd.notna(rsi_f) and rsi_f > TABLE_RSI_RED) else ''
        dev_cls = 'g' if (pd.notna(dev_f) and dev_f < TABLE_DEV_GREEN) else 'y' if (pd.notna(dev_f) and dev_f > TABLE_DEV_GOLD) else ''
        act_cls = 'g' if 'KOOPWAARDIG' in action else 'r' if 'VOORZICHTIG' in action else 'y' if 'AANHOUDEN' in action else ''
        chg_cls = 'up' if (pd.notna(dc) and dc >= 0) else 'dn' if pd.notna(dc) else ''

        body = (
            _field('Fase', phase, True)
            + _field('Support (30D)', r.get('Support (30D)', '–'))
            + _field('Weerstand (30D)', r.get('Weerstand (30D)', '–'))
            + _field('Patroon (1D)', r.get('Patroon (1D)', '–'), True)
        )
        mr = str(r.get('15M MR Status', '–'))
        if mr not in ('–', '', 'nan'):
            body += _field('15M Mean Reversion', mr, True)
        for _ic in ('15M [0]', '5M [0]'):
            _iv = str(r.get(_ic, '–'))
            if _iv not in ('–', '', 'nan'):
                body += _field(_ic, _iv)

        parts.append(
            f'<details class="qe-card {tone}"><summary>'
            f'<div class="qe-r1"><span class="qe-tk">{tk}</span><span class="qe-px">{_px_html(r.get("Koers", ""))}</span></div>'
            f'<div class="qe-r2"><div class="qe-chips">'
            f'<span class="qe-act {act_cls}">{_esc(action)}</span>'
            f'<span class="qe-chip {rsi_cls}">RSI <b>{_esc(str(r.get("RSI (14D)", "–")))}</b></span>'
            f'<span class="qe-chip {dev_cls}">Afw <b>{_esc(str(r.get("Afwijking %", "–")))}%</b></span>'
            f'</div><span class="qe-chg {chg_cls}">{_esc(str(r.get("Dag %", "")))}</span></div>'
            f'</summary><div class="qe-body">{body}</div></details>'
        )
    parts.append('</div>')
    return ''.join(parts)


def _wl_row(item) -> dict:
    """Eén watchlist-rij (draait in een worker-thread; geen .info-aanroep)."""
    ticker, label = item
    try:
        df_wl = _fetch_history(ticker, DATA_WL_PERIOD, DATA_MIN_ROWS_FETCH)
    except Exception as e:
        return {'Ticker': ticker, 'Label': label, 'Koers': 'N/A', 'Dag %': 'N/A', 'RSI': 'N/A',
                'Support': 'N/A', 'Weerstand': 'N/A', 'Afw. Support %': 'N/A',
                'Gedetecteerd Patroon': f'⚠ {_explain_error(type(e).__name__)}'}
    cl  = df_wl['Close'].squeeze()
    lp  = round(float(cl.iloc[-1]), 2)
    sym = _currency_symbol(ticker)
    rsi_wl = compute_rsi(cl, RSI_PERIOD)
    pat_wl = detect_candlestick_pattern(df_wl)
    sup_wl, res_wl = compute_support_resistance(df_wl, SUPPORT_WINDOW, current_price=lp)
    dev_wl = round(((lp - sup_wl) / sup_wl) * 100, 2) if sup_wl > 0 else 0.0
    day_chg = round((cl.iloc[-1] / cl.iloc[-2] - 1) * 100, 2) if len(cl) >= 2 and cl.iloc[-2] else 0.0
    return {
        'Ticker': ticker, 'Label': label,
        'Koers': f"{sym}{lp:,.2f}",
        'Dag %': f"{'▲' if day_chg >= 0 else '▼'} {abs(day_chg):.2f}%",
        'RSI': 'N/A' if (rsi_wl is None or (isinstance(rsi_wl, float) and np.isnan(rsi_wl))) else str(int(rsi_wl)),
        'Support': f"{sym}{sup_wl:,.2f}", 'Weerstand': f"{sym}{res_wl:,.2f}",
        'Afw. Support %': f"{dev_wl:.2f}",
        'Gedetecteerd Patroon': pat_wl,
    }


def style_main_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Pas dark terminal-stijl toe op de hoofdtabel."""
    # Verberg interne float-kolommen
    display_cols = [c for c in df.columns if not c.startswith('_')]
    display_df = df[display_cols].copy()

    def row_style(row):
        base = 'background-color: #13171C; color: #E8ECEF;'
        styles = [base] * len(row)

        # Interne float-kolommen ophalen via het rij-label (O(1) i.p.v. filteren per rij)
        orig_row = df.loc[row.name]
        rsi_v    = float(orig_row['_rsi_float'])
        dev_v    = float(orig_row['_dev_float'])
        ext_raw  = orig_row['_ext_chg']
        ext_chg_f = float(ext_raw) if pd.notna(ext_raw) else None
        day_raw  = orig_row['_day_chg'] if '_day_chg' in df.columns else np.nan

        phase  = str(row.get('Koers Status / Fase', ''))
        action = str(row.get('Actie', ''))

        # Rij-achtergrond op basis van fase
        if 'Bodemfase' in phase or 'Vroeg Herstel' in phase:
            styles = ['background-color: #0D2818; color: #00C853;'] * len(row)
        elif 'Uptrend' in phase:
            styles = ['background-color: #1A1500; color: #F0B90B;'] * len(row)
        else:
            styles = [base] * len(row)

        col_list = list(row.index)

        # ── Koers kolom: tekstkleur o.b.v. extended-change, achtergrond ONGEWIJZIGD ──
        if 'Koers' in col_list and ext_chg_f is not None:
            idx = col_list.index('Koers')
            # Pak de huidige achtergrond uit de rij-stijl
            current_bg = styles[idx].split(';')[0]  # 'background-color: ...'
            if ext_chg_f > 0:
                txt_color = '#00C853'   # groen — positief extended
            elif ext_chg_f < 0:
                txt_color = '#F6465D'   # rood — negatief extended
            else:
                txt_color = '#E8ECEF'   # neutraal
            styles[idx] = f'{current_bg}; color: {txt_color}; font-weight: 600;'

        # Dag % kolom: groen/rood tekst
        if 'Dag %' in col_list and pd.notna(day_raw):
            idx = col_list.index('Dag %')
            current_bg = styles[idx].split(';')[0]
            styles[idx] = f"{current_bg}; color: {'#00C853' if day_raw >= 0 else '#F6465D'}; font-weight: 600;"

        # RSI kolom
        if 'RSI (14D)' in col_list:
            idx = col_list.index('RSI (14D)')
            if rsi_v < TABLE_RSI_GREEN:
                styles[idx] = 'background-color: #00843A; color: #FFFFFF; font-weight:600;'
            elif rsi_v > TABLE_RSI_RED:
                styles[idx] = 'background-color: #A32040; color: #FFFFFF; font-weight:600;'

        # Afwijking kolom
        if 'Afwijking %' in col_list:
            idx = col_list.index('Afwijking %')
            if dev_v < TABLE_DEV_GREEN:
                styles[idx] = 'background-color: #00843A; color: #FFFFFF; font-weight:600;'
            elif dev_v > TABLE_DEV_GOLD:
                styles[idx] = 'background-color: #1A1500; color: #F0B90B;'

        # Patroon (1D) kolom
        for pat_col in ['Patroon', 'Patroon (1D)']:
            if pat_col in col_list:
                idx = col_list.index(pat_col)
                pat = str(row.get(pat_col, ''))
                if any(p in pat for p in ['Bullish','Hammer','Morning','Soldiers','Piercing','Tweezer Bottom','Dragonfly']) and 'Bearish' not in pat:
                    styles[idx] = 'background-color: #00843A; color: #FFFFFF; font-weight:600;'
                elif any(p in pat for p in ['Bearish','Shooting','Evening','Crows','Dark Cloud','Hanging','Gravestone','Tweezer Top']):
                    styles[idx] = 'background-color: #A32040; color: #FFFFFF; font-weight:600;'
                elif 'Doji' in pat:
                    styles[idx] = 'background-color: #1A1500; color: #F0B90B;'

        # Intraday kolommen (15M en 5M)
        for intra_col in ['15M [-1]', '15M [0]', '5M [-1]', '5M [0]']:
            if intra_col in col_list:
                idx = col_list.index(intra_col)
                val_s = str(row.get(intra_col, ''))
                if any(p in val_s for p in ['Bullish','Hammer','Morning','Soldiers','Piercing','Tweezer Bottom','Dragonfly']) and 'Bearish' not in val_s:
                    styles[idx] = 'background-color: #0D2818; color: #00C853; font-weight:600;'
                elif any(p in val_s for p in ['Bearish','Shooting','Evening','Crows','Dark Cloud','Hanging','Gravestone','Tweezer Top']):
                    styles[idx] = 'background-color: #1A0008; color: #F6465D; font-weight:600;'
                elif '🟢' in val_s:
                    styles[idx] = 'background-color: #0D2818; color: #00C853;'
                elif '🔴' in val_s:
                    styles[idx] = 'background-color: #1A0008; color: #F6465D;'
                elif 'Doji' in val_s:
                    styles[idx] = 'background-color: #1A1500; color: #F0B90B;'

        # 15M Mean Reversion Status kolom
        if '15M MR Status' in col_list:
            idx = col_list.index('15M MR Status')
            mr_val = str(row.get('15M MR Status', ''))
            if 'BULLISH REVERSION' in mr_val and '✅' in mr_val:
                # Sterk bevestigd bullish signaal
                styles[idx] = ('background-color: #00843A; color: #FFFFFF; '
                                'font-weight: 700; font-size: 0.78rem;')
            elif 'BULLISH REVERSION' in mr_val:
                # Bullish maar volume laag
                styles[idx] = ('background-color: #0D2818; color: #00C853; '
                                'font-weight: 600; font-size: 0.78rem;')
            elif 'BEARISH REVERSION' in mr_val and '✅' in mr_val:
                # Sterk bevestigd bearish signaal
                styles[idx] = ('background-color: #A32040; color: #FFFFFF; '
                                'font-weight: 700; font-size: 0.78rem;')
            elif 'BEARISH REVERSION' in mr_val:
                # Bearish maar volume laag
                styles[idx] = ('background-color: #1A0008; color: #F6465D; '
                                'font-weight: 600; font-size: 0.78rem;')
            elif 'SETUP' in mr_val:
                # Opbouwend signaal (patroon aanwezig, geen ORL/ORH-test)
                styles[idx] = ('background-color: #1A1500; color: #F0B90B; '
                                'font-weight: 600; font-size: 0.78rem;')

        # Actie kolom
        if 'Actie' in col_list:
            idx = col_list.index('Actie')
            if 'KOOPWAARDIG' in action:
                styles[idx] = 'background-color: #00843A; color: #FFFFFF; font-weight:700;'
            elif 'VOORZICHTIG' in action:
                styles[idx] = 'background-color: #A32040; color: #FFFFFF; font-weight:600;'
            elif 'AANHOUDEN' in action:
                styles[idx] = 'background-color: #1A1500; color: #F0B90B; font-weight:600;'

        return styles

    styled = display_df.style.apply(row_style, axis=1)
    styled.set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#0B0E11'),
            ('color', '#F0B90B'),
            ('font-family', 'JetBrains Mono, monospace'),
            ('font-size', '0.78rem'),
            ('border-bottom', '2px solid #F0B90B'),
            ('padding', '8px 12px')]},
        {'selector': 'tbody td', 'props': [
            ('font-family', 'JetBrains Mono, monospace'),
            ('font-size', '0.8rem'),
            ('padding', '6px 12px'),
            ('border-bottom', '1px solid #2B3139')]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#222831 !important')]}])
    return styled


@st.cache_data(ttl=CACHE_POOL_TTL)
def get_large_ticker_pool() -> list:
    """
    Bouwt de volledige tickerpool van 900+ tickers.

    Drie lagen — nooit minder dan de ingebakken hardcoded pools:
      Layer 1 : Live Wikipedia scrape (S&P 500 + MidCap 400)
      Layer 2 : Hardcoded fallback S&P 500 (~488) + MidCap 400 (~400)
      Layer 3 : Altijd actief — Europese top + extra's
    """

    # ── LAYER 3 — altijd actief ───────────────────────────────────────────────
    EUROPE: list = [
        # Nederland (AEX)
        'ASML.AS','AD.AS','ADYEN.AS','UNA.AS','HEIA.AS','PHIA.AS','ABN.AS',
        'INGA.AS','RAND.AS','WKL.AS','IMCD.AS','BESI.AS','AKZA.AS',
        # Duitsland (DAX) — EON.DE → EOAN.DE (correct yfinance symbool)
        'SAP.DE','SIE.DE','DTE.DE','VOW3.DE','BMW.DE','ADS.DE',
        'MUV2.DE','ALV.DE','BAS.DE','BAYN.DE','DBK.DE','HEN3.DE','LIN.DE',
        'MRK.DE','RWE.DE','EOAN.DE','FRE.DE','ZAL.DE','IFX.DE','PAH3.DE',
        'RHM.DE',
        # Frankrijk (CAC 40)
        'OR.PA','MC.PA','TTE.PA','SAN.PA','BNP.PA','ACA.PA','SGO.PA',
        'KER.PA','RMS.PA','DSY.PA','AIR.PA','DG.PA','ORA.PA','VIE.PA',
        # Zwitserland — ROG.SW verwijderd (via Roche delisting op yfinance)
        'NESN.SW','NOVN.SW','UBSG.SW','ABBN.SW','ZURN.SW',
        # VK (FTSE 100)
        'AZN.L','SHEL.L','HSBA.L','BP.L','GSK.L','RIO.L','ULVR.L',
        'LLOY.L','BATS.L','REL.L','DGE.L','EXPN.L','LSEG.L',
        # Scandinavië — ERICB.ST → ERIC-B.ST (correct yfinance suffix)
        'NOVO-B.CO','ERIC-B.ST','VOLV-B.ST','SAND.ST','ATCO-A.ST','SEB-A.ST',
        # Italië — STM.MI verwijderd (gebruik STM op NYSE ipv Milaan-notering)
        'RACE.MI','ENI.MI','ENEL.MI','ISP.MI','UCG.MI','PRY.MI',
        # Spanje (IBEX 35)
        'TEF.MC','IBE.MC','ITX.MC','BBVA.MC','SAN.MC','REP.MC']

    EXTRAS: list = [
        'NVDA','MSFT','AAPL','GOOGL','GOOG','AMZN','META','TSLA','AMD','ARM',
        'INTC','QCOM','TXN','AVGO','MU','MRVL','AMAT','KLAC','LRCX','ASML',
        'MCHP','ADI','NXPI','ON','SWKS','MPWR','ENPH','FSLR','SEDG',
        'JPM','BAC','GS','MS','WFC','C','BLK','AXP','V','MA','PYPL','COF',
        'USB','PNC','TFC','SCHW','ICE','CME','SPGI','MCO','FIS','FISV','GPN',
        'JNJ','ABBV','PFE','MRK','BMY','UNH','CVS','AMGN','GILD','REGN',
        'LLY','TMO','DHR','ISRG','SYK','BSX','MDT','ABT','ZTS','VRTX',
        'BIIB','MRNA','BNTX','INCY','ALNY','BEAM','NTLA',
        'XOM','CVX','COP','SLB','EOG','HAL','MPC','PSX','VLO',
        'OXY','DVN','APA','FANG','BKR',
        'GLD','SLV','USO','FCX','NEM','GOLD','WPM','CAT','DE','EMR',
        'NUE','STLD','CLF','AA','MP','ALB','SQM','LAC',
        'WMT','COST','TGT','HD','LOW','MCD','SBUX','NKE','KO','PEP',
        'PG','CL','KMB','GIS','MDLZ','PM','MO','WM','RSG',
        'SYY','CAG','CPB','HSY','MKC','CHD','CLX','EL','DIS','NFLX','CMCSA','T','VZ','TMUS','CHTR','WBD','FOX',
        'TTWO','EA','RBLX','U',
        'O','PLD','SPG','WELL','AMT','CCI','EQIX','DLR','PSA','EXR',
        'AVB','EQR','ESS','UDR','MAA','CPT','NNN','VICI','GLPI',
        'ADP','PAYX','INTU','ORCL','CRM','NOW','SNOW','DDOG','CRWD',
        'ZS','OKTA','PANW','FTNT','NET','MDB','GTLB',
        'SHOP','HUBS','BILL','PCTY','PAYC','WEX','FOUR',
        'GE','HON','MMM','ITW','ETN','PH','ROK','AME','FTV','GNRC',
        'UPS','FDX','DAL','UAL','AAL','LUV','ALK',
        'NSC','CSX','UNP','CP','CNI','WAB','TT','CARR','OTIS',
        'NTRA','PACB','ILMN','BIO','TECH','IDXX','PODD','DXCM',
        'BKNG','ABNB','LYFT','UBER','DASH','MTN',
        'GM','F','RIVN','LCID','NIO','LI','XPEV',
        'COIN','MSTR','RIOT','MARA','HUT','CLSK',
        'SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLI','XLY',
        'XLP','XLU','XLB','XLRE','XLC','GDX','GDXJ',
        'TLT','HYG','LQD','EMB','VNQ','ARKK','ARKG','ARKF']

    SP500_HARDCODED: list = [
        'A','AAL','AAP','AAPL','ABBV','ABT','ACN','ADBE',
        'ADI','ADM','ADP','ADSK','AEE','AEP','AES','AFL','AIG','AIZ',
        'AJG','AKAM','ALB','ALGN','ALK','ALL','ALLE','AMAT','AMCR','AMD',
        'AME','AMGN','AMP','AMT','AMZN','ANET','AON','AOS','APA',
        'APD','APH','APTV','ARE','ATO','AVB','AVGO','AVY','AWK',
        'AXP','AZO','BA','BAC','BALL','BAX','BBY','BDX','BEN','BF-B',
        'BIIB','BIO','BK','BKNG','BKR','BLK','BMY','BR','BRK-B',
        'BSX','BWA','BXP','C','CAG','CAH','CARR','CAT','CB','CBOE',
        'CBRE','CCI','CCL','CDNS','CDW','CE','CEG','CF','CFG','CHD',
        'CHRW','CHTR','CI','CINF','CL','CLX','CMCSA','CME','CMG',
        'CMI','CMS','CNC','CNP','COF','COO','COP','COST','CPB','CPRT',
        'CRL','CRM','CSCO','CSX','CTAS','CTSH','CTVA',
        'CVS','CVX','CZR','D','DAL','DD','DE','DG','DGX','DHI',
        'DHR','DIS','DLR','DLTR','DOV','DOW','DPZ','DRI',
        'DTE','DUK','DVA','DVN','DXC','DXCM','EA','EBAY','ECL','ED',
        'EFX','EIX','EL','EMN','EMR','ENPH','EOG','EPAM','EQIX','EQR',
        'EQT','ES','ESS','ETN','ETR','ETSY','EVRG','EW','EXC','EXPD',
        'EXPE','EXR','F','FANG','FAST','FCX','FDS','FDX','FE','FFIV',
        'FIS','FISV','FITB','FMC','FOX','FOXA','FRT','FSLR',
        'FTNT','FTV','GD','GE','GILD','GIS','GL','GLW','GM','GNRC',
        'GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN',
        'HCA','HD','HIG','HII','HLT','HON','HPE','HPQ',
        'HRL','HSIC','HST','HSY','HUM','HWM','IBM','ICE','IDXX','IEX',
        'IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR',
        'IRM','ISRG','IT','ITW','IVZ','J','JBHT','JCI','JKHY','JNJ',
        'JPM','KEY','KEYS','KHC','KIM','KLAC','KMB','KMI',
        'KMX','KO','KR','L','LDOS','LEN','LH','LHX','LIN','LKQ','LLY',
        'LMT','LNC','LNT','LOW','LRCX','LUV','LVS','LW','LYB','LYV',
        'MA','MAA','MAR','MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT',
        'MET','META','MGM','MHK','MKC','MKTX','MLM','MMC','MMM','MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MS','MSCI',
        'MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ','NEM','NEE',
        'NI','NKE','NOC','NOW','NRG','NSC','NTAP','NTRS','NUE','NVDA',
        'NVR','NWL','NWS','NWSA','NXPI','O','OGN','OMC','ON',
        'ORCL','ORLY','OXY','PAYC','PAYX','PCAR','PCG',
        'PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG',
        'PLD','PM','PNC','PNR','PNW','POOL','PPG','PPL','PRU','PSA',
        'PSX','PTC','PWR','PYPL','QCOM','RCL','EG','REG',
        'REGN','RF','RHI','RJF','RL','RMD','ROK','ROL','ROP','ROST',
        'RSG','RTX','SBAC','SBUX','SCHW','SHW','SJM','SLB',
        'SNA','SNPS','SO','SPG','SPGI','SRE','STE','STT','STX','STZ',
        'SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG','TDY','TECH',
        'TEL','TER','TFC','TFX','TGT','TJX','TMO','TMUS','TPR','TRMB',
        'TROW','TRV','TSCO','TSLA','TSN','TT','TTWO','TXN','TXT','TYL',
        'UA','UAA','UAL','UDR','UHS','ULTA','UNH','UNP','UPS','URI',
        'USB','V','VFC','VICI','VLO','VMC','VNO','VRSK','VRSN','VRTX',
        'VZ','WAB','WAT','WBD','WEC','WELL','WFC','WHR','WM',
        'WMB','WMT','WST','WTW','WY','WYNN','XEL','XOM',
        'XRAY','XYL','YUM','ZBH','ZBRA','ZION','ZTS']

    # ── LAYER 2 — hardcoded MidCap 400 fallback (~400 tickers) ──────────────
    MIDCAP_HARDCODED: list = [
        # Industrials & Business Services
        'ABCB','ABR','ACHC','ACI','ACM','ADNT','AEO','AFG','AGCO',
        'AIN','AKR','ALG','ALKS','AM','AMG','AMKR','APLE','APO',
        'ARCO','AROC','ASH','ATR','AVAV','AVT','AX','AYI','AZZ',
        'BCO','BDC','BFH','BGS','BHF','BJ','BKH','BLKB','BMI','BOH',
        'BRC','BRX','BSY','BTU','BWXT','CABO','CACI','CAKE','CALX',
        'CARS','CASY','CC','CCOI','CFR','CHCO','CHE','CHH','CIEN',
        'CLB','CNA','CNK','CNO','COHU','COLB','COLM','CPF','CPRX','CRC',
        'CROX','CSL','CW','CWEN','CWK','CWST','CWT',
        # D-G
        'DAN','DFIN','DLX','DOCS','DRH','DY',
        'EAT','EFC','EGP','EHC','EME','ENVA','EPC','ESE','ESRT','EVH',
        'EXP','EXPI','FAF','FCN','FG','FHN','FIBK','FIX',
        'FLO','FLS','FNB','FOR','FORM','FR','FRME','FSS','FUL',
        'GBCI','GEF','GFF','GHC','GKOS','GMED','GPI','GPRE',
        'GRBK','GRC','GTN','GTLS','GVA',
        # H-L
        'HAE','HALO','HCC','HCI','HGV','HLIT','HLX',
        'HMN','HNI','HOPE','HRI','HTBK','HUBG','HWC',
        'ICLR','IDCC','IIPR','INDB','INGR','IPGP','ITT',
        'JACK','JBL','JOE','KAI','KBH','KBR','KFY',
        'KMT','KMPR','KN','KNX','KNSL','KSS','KTOS',
        'LAUR','LE','LEA','LGND','LII','LIVN','LKQ',
        'LMAT','LNTH','LPLA','LRN','LSTR','LUMN',
        # M-N
        'MGEE','MHO','MIR','MKSI','MLCO','MLI','MMSI','MNRO',
        'MP','MRCY','MTH','MTN','MUR','NATL','NBR',
        'NEO','NGL','NHC','NHI','NJR','NNN','NSP','NUS','NVT',
        # O-R
        'OFG','OGE','OGS','OI','OII','OLN','OMF','ONTO','ORA',
        'OWL','OXM','PBH','PCRX','PEB','PFSI','PJT','PLMR',
        'PLUS','PNM','POR','POWL','PRDO','PRG','PRIM','PRVA','PTEN',
        'R','RBCAA','RDNT','REX','RH','RIG','RLJ','RMBS',
        'RNR','ROCK','RPM','RRC','RRX','RUN','RXO',
        # S
        'SAH','SANM','SBCF','SBRA','SCI','SEIC','SF','SFBS','SIGI',
        'SITC','SITE','SKT','SLM','SM','SMTC','SNV',
        'STE','STEP','SWN','SXT','SYBT','SXI',
        # T-Z
        'TCBK','TCBI','TEX','TGI','TILE','TISI','TNET','TPH','TREX',
        'TRNO','TRN','TRMK','TTC','TWI','UBSI','UGI','UNF',
        'UVV','VBTX','VG','VIRT','VLY','VRTS','VSCO','VSH',
        'WABC','WAFD','WD','WEN','WERN','WFRD','WK','WLK',
        'WMS','WOR','WRLD','WSM','WSO','WTS','WWD','WWW',
        'XPO','YELP','ZWS',
        # Extra bekende MidCap namen (gevraagd in bugfix)
        'DECK','TOLL','FLEX','EG','ANET','SITE','TREX','MEDP',
        'HTGC','LADR','SMCI','BILL','CELH','HIMS','DUOL','GLBE',
        'EXEL','PRCT','QLYS','HALO','CRVL','IIPR','UFPT',
        'PRGS','FCFS','MGNI','GKOS','IRDM','LCII','SAIA','EXPO',
        'MGEE','HWKN','POWI','IPAR','MSEX','SPSC','CSGS','NVCR',
        'AEIS','ICFI','HURN','NTCT','PSMT','EPAC','VRRM','ACLS','FORM','TMDX','PCVX','XPEL','DXPE','AVNS','LKFN','SKYW','PRAA',
        'AMSF','HSTM','PFIS','RCUS','CCOI','NKTR']

    # ── LAYER 1: Probeer Wikipedia live te scrapen ────────────────────────────
    scraped_sp500:  list = []
    scraped_mid400: list = []
    scrape_status = {'sp500': False, 'mid400': False}

    def _clean(t: str) -> str:
        """Converteer ticker naar yfinance-formaat: punt → koppelteken."""
        return str(t).strip().replace('.', '-').upper()

    # S&P 500
    try:
        df_sp = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            attrs={'id': 'constituents'}
        )[0]
        col = 'Symbol' if 'Symbol' in df_sp.columns else df_sp.columns[0]
        scraped_sp500 = [_clean(t) for t in df_sp[col].dropna().tolist() if t]
        scrape_status['sp500'] = len(scraped_sp500) > 400
        print(f"DEBUG SCRAPE S&P500: {len(scraped_sp500)} tickers (kolom='{col}')")
    except Exception as e:
        print(f"DEBUG SCRAPE S&P500 MISLUKT: {e}")

    # MidCap 400 — meerdere URL-varianten proberen
    for _mid_url, _id in [
        ('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', None),
        ('https://en.wikipedia.org/wiki/S%26P_400',                   None)]:
        if scrape_status['mid400']:
            break
        try:
            kwargs = {'attrs': {'id': _id}} if _id else {}
            tables = pd.read_html(_mid_url, **kwargs)
            for df_mid in tables:
                # Zoek kolom met ticker-achtige waarden (2-5 hoofdletters)
                for col in df_mid.columns:
                    sample = df_mid[col].dropna().astype(str).head(10).tolist()
                    looks_like_ticker = sum(
                        1 for s in sample
                        if 2 <= len(s.strip()) <= 6
                        and s.strip().replace('.','').replace('-','').isupper()
                    )
                    if looks_like_ticker >= 6:
                        candidates = [
                            _clean(t) for t in df_mid[col].dropna().tolist() if t
                        ]
                        if len(candidates) > 100:
                            scraped_mid400 = candidates
                            scrape_status['mid400'] = True
                            print(f"DEBUG SCRAPE MidCap: {len(scraped_mid400)} tickers "
                                  f"(url={_mid_url}, kolom='{col}')")
                            break
                if scrape_status['mid400']:
                    break
        except Exception as e:
            print(f"DEBUG SCRAPE MidCap MISLUKT ({_mid_url}): {e}")

    # ── Kies live of hardcoded per lijst ────────────────────────────────────
    sp500_tickers  = scraped_sp500  if scrape_status['sp500']  else SP500_HARDCODED
    midcap_tickers = scraped_mid400 if scrape_status['mid400'] else MIDCAP_HARDCODED
    europe_tickers = EUROPE   # altijd hardcoded (geen publieke Wikipedia-lijst)
    extra_tickers  = EXTRAS

    # Extra schoonmaak: punten → koppeltekens, lege waarden eruit
    sp500_tickers  = [_clean(t) for t in sp500_tickers  if t]
    midcap_tickers = [_clean(t) for t in midcap_tickers if t]
    europe_tickers = [t for t in europe_tickers if t and isinstance(t, str)]
    extra_tickers  = [t for t in extra_tickers  if t and isinstance(t, str)]

    # ── Harde concatenatie — expliciet, geen verborgen defaults ─────────────
    volledige_pool: list = list(set(
        sp500_tickers +
        midcap_tickers +
        europe_tickers +
        extra_tickers
    ))

    # Verwijder lege strings en ongeldige entries
    volledige_pool = sorted([
        t for t in volledige_pool
        if t and isinstance(t, str) and len(t.strip()) >= 1
    ])

    _sp_label  = f"{'✅ live' if scrape_status['sp500']  else '⚠ hardcoded'} ({len(sp500_tickers)})"
    _mid_label = f"{'✅ live' if scrape_status['mid400'] else '⚠ hardcoded'} ({len(midcap_tickers)})"
    _eu_label  = f"altijd actief ({len(europe_tickers)})"
    _ex_label  = f"altijd actief ({len(extra_tickers)})"

    print(f"DEBUG POOL TOTAAL: {len(volledige_pool)} | "
          f"S&P500={len(sp500_tickers)} | MidCap={len(midcap_tickers)} | "
          f"Europe={len(europe_tickers)} | Extras={len(extra_tickers)}")

    # Retourneer pool + metadata als tuple zodat de aanroeper session_state
    # kan updaten (st.session_state mag NIET binnen @st.cache_data worden gebruikt)
    meta = {
        'total':        len(volledige_pool),
        'sp500':        len(sp500_tickers),
        'midcap':       len(midcap_tickers),
        'europe':       len(europe_tickers),
        'extras':       len(extra_tickers),
        'status_str':   (
            f"S&P500: {_sp_label} · MidCap400: {_mid_label} · "
            f"Europa: {_eu_label} · Extra's: {_ex_label}"
        ),
    }
    return volledige_pool, meta


def load_ticker_pool(use_large_pool: bool = False) -> list:
    """
    Geef de tickerlijst terug voor de scanner.
    Standaard: alleen de eigen tickers van de gebruiker (snel, geen rate-limit).
    Optioneel: grote pool van 1000+ tickers (langzaam, kans op rate-limit).
    """
    if not use_large_pool:
        # Gebruik alleen eigen tickers — snel en betrouwbaar
        pool = list(st.session_state.get('main_market_tickers', DEFAULT_TICKERS))
        total = len(pool)
        st.session_state['total_ticker_count'] = total
        st.session_state['scan_pool_size']     = total
        st.session_state['pool_scrape_status'] = '📋 Eigen tickerlijst'
        return pool
    else:
        # Grote pool — langzaam, kans op Yahoo rate-limit
        pool, meta = get_large_ticker_pool()
        st.session_state['total_ticker_count']  = meta['total']
        st.session_state['pool_sp500']          = meta['sp500']
        st.session_state['pool_midcap']         = meta['midcap']
        st.session_state['pool_europe']         = meta['europe']
        st.session_state['pool_extras']         = meta['extras']
        st.session_state['pool_scrape_status']  = meta['status_str']
        st.session_state['scan_pool_size']      = meta['total']
        return pool



def apply_rr_veto(mtf_result: dict,
                  current_price: float,
                  resistance_target: float,
                  actual_rr: float) -> dict:
    """
    Overschrijft de MTF-status met REJECTED als de Risk:Reward onvoldoende is.

    Veto-regels (worden ACHTER de MTF-check uitgevoerd):
      1. Koers >= resistance_target  → geen opwaarts potentieel meer
      2. actual_rr <= 0              → reward is nul of negatief
      3. actual_rr < TRADE_REWARD_RATIO → R:R te laag voor een kwalitatieve trade

    Retourneert het (mogelijk gewijzigde) mtf_result dict met extra sleutels:
      'rr_veto'       : True/False — of het veto is getriggerd
      'rr_veto_reason': leesbare reden voor de override
    """

    result = dict(mtf_result)  # kopie, nooit het origineel muteren
    result['rr_veto']        = False
    result['rr_veto_reason'] = ''

    # Guard: ongeldige inputs behandelen als veto
    if any(math.isnan(v) if isinstance(v, float) else False
           for v in [current_price, resistance_target, actual_rr]):
        result['status']         = 'REJECTED'
        result['rr_veto']        = True
        result['rr_veto_reason'] = 'N/A data (NaN in berekening)'
        return result

    # Veto 1 — koers staat al op of boven het target
    if current_price >= resistance_target:
        result['status']         = 'REJECTED'
        result['rr_veto']        = True
        result['rr_veto_reason'] = (
            f'Geen opwaarts potentieel (koers {current_price:.2f} ≥ '
            f'resistance {resistance_target:.2f})'
        )
        return result

    # Veto 2 — reward is nul of negatief
    if actual_rr <= 0:
        result['status']         = 'REJECTED'
        result['rr_veto']        = True
        result['rr_veto_reason'] = f'Negatieve of nul reward (R:R = {actual_rr}:1)'
        return result

    # Veto 3 — R:R positief maar onder de centrale drempel
    if actual_rr < TRADE_REWARD_RATIO:
        result['status']         = 'REJECTED'
        result['rr_veto']        = True
        result['rr_veto_reason'] = (
            f'R:R te laag ({actual_rr:.2f}:1 < drempel {TRADE_REWARD_RATIO}:1)'
        )
        return result

    # Geen veto getriggerd — status ongewijzigd
    return result


def compute_multi_timeframe_check(ticker: str) -> dict:
    """Multi-timeframe validatie: 1W macro + 1H volume micro."""
    result = {'weekly_trend': 'UNKNOWN', 'hourly_volume': 'UNKNOWN', 'status': 'WATCH'}

    try:
        df_w = yf.download(ticker, period=DATA_MTF_WEEKLY, interval="1wk", progress=False, auto_adjust=True)
        if df_w is not None and len(df_w) >= MTF_WEEKLY_MA:
            weekly_close = df_w['Close'].squeeze()
            ma10 = weekly_close.rolling(MTF_WEEKLY_MA).mean().iloc[-1]
            last_w = float(weekly_close.iloc[-1])
            result['weekly_trend'] = 'BULLISH' if last_w > ma10 else 'BEARISH'
        else:
            result['weekly_trend'] = 'N/A'
    except Exception:
        result['weekly_trend'] = 'N/A'

    try:
        df_h = yf.download(ticker, period=DATA_MTF_HOURLY, interval="1h", progress=False, auto_adjust=True)
        if df_h is not None and len(df_h) >= MTF_HOURLY_AVG_BARS:
            vol = df_h['Volume'].squeeze()
            avg_vol = vol.rolling(MTF_HOURLY_AVG_BARS).mean().iloc[-3:-1].mean()
            last_vol = float(vol.iloc[-1])
            result['hourly_volume'] = 'RISING' if last_vol > avg_vol * MTF_VOLUME_SPIKE else 'FLAT'
        else:
            result['hourly_volume'] = 'N/A'
    except Exception:
        result['hourly_volume'] = 'N/A'

    if result['weekly_trend'] == 'BULLISH' and result['hourly_volume'] in ('RISING', 'N/A'):
        result['status'] = 'APPROVED'
    elif result['weekly_trend'] == 'BEARISH':
        result['status'] = 'REJECTED'
    else:
        result['status'] = 'WATCH'

    return result


def run_scanner(strategy: str, pool: list, max_results: int = 9999) -> pd.DataFrame:
    """
    Scan de tickerpool op basis van de geselecteerde strategie.

    TERUGGEZET naar het oudere, bewezen patroon: per ticker individueel
    ophalen via fetch_ticker_data() (zelf al @st.cache_data-gecached),
    met een live progress bar. Dit voorkomt:
      - MultiIndex key-mismatches bij bulk-download (bv. ASML.AS)
      - Yahoo rate-limits (429) door 1 megacall met 1000+ tickers
      - Een "hangende" UI zonder enig voortgangssignaal tijdens de scan
    """
    print(f"DEBUG: Totaal aantal tickers aangeboden aan loop: {len(pool)}")

    rows        = []
    failed      = []
    errored     = []
    total       = len(pool)

    # ── Voortgangsbalk — altijd gebaseerd op de volledige pool ────────────────
    progress_bar = st.progress(0.0, text=f"⏳ Initialiseren scan van {total} tickers...")
    status_box   = st.empty()

    for idx, ticker in enumerate(pool):

        pct = (idx + 1) / total
        if idx % 10 == 0 or idx == total - 1:
            progress_bar.progress(
                min(pct, 1.0),
                text=f"⏳ Scanning {idx + 1}/{total} · {len(rows)} hits · {len(failed)} geen data · {len(errored)} fouten"
            )
            status_box.markdown(
                f"<small style='color:#848E9C; font-family:monospace;'>"
                f"Huidig: <b style='color:#F0B90B;'>{ticker}</b> &nbsp;|&nbsp; "
                f"Hits: <b style='color:#00C853;'>{len(rows)}</b> &nbsp;|&nbsp; "
                f"Geen data: {len(failed)} &nbsp;|&nbsp; Fouten: {len(errored)}"
                f"</small>",
                unsafe_allow_html=True,
            )

        try:
            # ── Data ophalen via lichte, gecachte functie (geen .info!) ───────
            # De scanner-strategieën gebruiken alleen OHLCV — t.info (bedrijfs-
            # metadata) overslaan voorkomt onnodige, trage en rate-limit-
            # gevoelige calls over 1000+ tickers.
            df = fetch_ticker_data_fast(ticker, period=DATA_SCAN_PERIOD)
            if df is None or len(df) < DATA_MIN_ROWS_SCAN:
                failed.append(ticker)
                continue

            close  = df['Close'].squeeze()
            high   = df['High'].squeeze()
            low    = df['Low'].squeeze()
            open_s = df['Open'].squeeze()
            volume = df['Volume'].squeeze()
            last_price = float(close.iloc[-1])
            rsi_val    = compute_rsi(close, RSI_PERIOD)
            n = len(close)

            # Bollinger bands alleen berekenen voor strategieën die ze nodig
            # hebben — scheelt rekenwerk over 1000+ tickers voor o.a.
            # alpha_scanner, sr_bounce, momentum_low_float.
            if strategy in ("mean_reversion", "vol_squeeze"):
                ub, mb, lb = compute_bollinger_series(close, BB_PERIOD, BB_STD_NORMAL)
                ub25, mb25, lb25 = compute_bollinger_series(close, BB_PERIOD, BB_STD_WIDE)

            match      = False
            extra_info = {}

            # ── Strategie logica ──────────────────────────────────────────────
            if strategy == "momentum_train":
                x = np.arange(n)
                poly     = np.polyfit(x, close.values, 1)
                reg_line = np.polyval(poly, x)
                central  = reg_line[-1]
                std_dev  = float(np.std(close.values - reg_line))
                resistance      = central + 2 * std_dev
                deviation_to_res = ((resistance - last_price) / last_price) * 100
                last_two_green  = (close.iloc[-1] > open_s.iloc[-1]) and \
                                  (close.iloc[-2] > open_s.iloc[-2])
                if (last_price > central and
                        SCAN_MOM_RSI_MIN <= rsi_val <= SCAN_MOM_RSI_MAX and
                        SCAN_MOM_RUIMTE_MIN <= deviation_to_res <= SCAN_MOM_RUIMTE_MAX and
                        last_two_green):
                    match      = True
                    extra_info = {'Ruimte tot Weerstand %': round(deviation_to_res, 2)}

            elif strategy == "mean_reversion":
                last_ub25 = float(ub25.iloc[-1])
                last_lb25 = float(lb25.iloc[-1])
                if ((rsi_val < SCAN_MEAN_RSI_OS or rsi_val > SCAN_MEAN_RSI_OB) and
                        (last_price < last_lb25 or last_price > last_ub25)):
                    match     = True
                    direction = "OVERSOLD" if rsi_val < SCAN_MEAN_RSI_OS else "OVERBOUGHT"
                    extra_info = {'Signaal': direction}

            elif strategy == "vol_squeeze":
                bb_width = (ub - lb) / mb
                if len(bb_width.dropna()) >= SCAN_SQZ_WINDOW:
                    min_width = bb_width.dropna().rolling(SCAN_SQZ_WINDOW).min().iloc[-1]
                    cur_width = bb_width.iloc[-1]
                    if abs(cur_width - min_width) / (min_width + 1e-9) < SCAN_SQZ_THRESH:
                        match      = True
                        extra_info = {'BB Width': round(float(cur_width), 4)}

            elif strategy == "support_bounce":
                # Bestaande S/R — nadert support OF resistance (bidirectioneel)
                sup30, res30 = compute_support_resistance(
                    df, SUPPORT_WINDOW, current_price=last_price)
                dist_sup = abs(last_price - sup30)  / last_price * 100
                dist_res = abs(last_price - res30)  / last_price * 100
                if dist_sup <= SCAN_SR_DIST or dist_res <= SCAN_SR_DIST:
                    match   = True
                    nearest = "SUPPORT" if dist_sup < dist_res else "RESISTANCE"
                    extra_info = {
                        'Nearest': nearest,
                        'Dist %':  round(min(dist_sup, dist_res), 2),
                    }

            elif strategy == "alpha_scanner":
                # ── Alpha Scanner: Bodemfase / Waarde ────────────────────────
                # Zoekt aandelen die stabiliseren op of vlak boven hun support
                # met een oversold RSI. Volume mag juist LAAG zijn in een bodem
                # (accumulatie = stille kopers) — eis is daarom 0.8× ipv 1.3×.
                #
                # Voorwaarden:
                #   1. RSI ≤ SCAN_ALPHA_RSI_MAX (oversold / vroege herstelzone)
                #   2. Koers BOVEN support — niet eronder (geen valende messen)
                #   3. Afwijking van support tussen 0% en SCAN_ALPHA_DEV_MAX
                #      (exact op bodem of maximaal 5% erboven)
                #   4. Volume ≥ SCAN_ALPHA_VOL_SPIKE × 20D (minimumcheck: 0.8×)
                sup_a, res_a = compute_support_resistance(
                    df, SUPPORT_WINDOW, current_price=last_price)

                # Expliciete float-cast + veilige deling — nooit NaN of negatief
                sup_a_f = float(sup_a) if sup_a and sup_a > 0 else last_price * 0.95
                dev_from_sup = float(
                    ((last_price - sup_a_f) / sup_a_f) * 100
                ) if sup_a_f > 0 else 999.0

                vol_avg_a   = float(volume.rolling(VOL_MA_PERIOD).mean().iloc[-1])
                vol_now_a   = float(volume.iloc[-1])
                vol_ratio_a = round(vol_now_a / vol_avg_a, 2) if vol_avg_a > 0 else 0.0

                # Alle drie condities expliciet met float-vergelijking
                cond_rsi = float(rsi_val) <= float(SCAN_ALPHA_RSI_MAX)
                cond_dev = float(SCAN_ALPHA_DEV_MIN) <= dev_from_sup <= float(SCAN_ALPHA_DEV_MAX)
                cond_vol = vol_ratio_a >= float(SCAN_ALPHA_VOL_SPIKE)

                if cond_rsi and cond_dev and cond_vol:
                    match = True
                    extra_info = {
                        'RSI':           int(rsi_val),
                        'Dev Support %': round(dev_from_sup, 2),
                        'Vol Ratio':     vol_ratio_a,
                        'Support':       round(sup_a_f, 2),
                    }

            elif strategy == "sr_bounce":
                # ── S/R Bounce Scanner: Pullback Springplank ─────────────────
                # Zoekt aandelen die net van hun support omhoog stuiteren na
                # een pullback. Verschil met alpha_scanner: RSI hoeft NIET
                # laag te zijn; de close moet hoger zijn dan de open (herstel
                # kaars), en volume trekt licht aan.
                #
                # Voorwaarden:
                #   1. Koers staat 0-SCAN_SR_BOUNCE_MAX% BOVEN support
                #      (extreem dicht erbij, maar niet eronder)
                #   2. Huidige kaars is groen (close > open) = technisch herstel
                #   3. Volume ≥ SCAN_SR_BOUNCE_VOL × 20D gemiddelde
                #   4. Koers is NIET onder support gedoken (last_price >= sup_b)
                sup_b, res_b = compute_support_resistance(
                    df, SUPPORT_WINDOW, current_price=last_price)
                dev_sup_b    = ((last_price - sup_b) / sup_b * 100) if sup_b > 0 else 999.0
                dev_res_b    = ((res_b - last_price) / last_price * 100) if last_price > 0 else 0.0
                vol_avg_b    = float(volume.rolling(VOL_MA_PERIOD).mean().iloc[-1])
                vol_now_b    = float(volume.iloc[-1])
                vol_ratio_b  = round(vol_now_b / vol_avg_b, 2) if vol_avg_b > 0 else 0.0
                green_kaars  = float(close.iloc[-1]) > float(open_s.iloc[-1])

                if (0 <= dev_sup_b <= SCAN_SR_BOUNCE_MAX and
                        green_kaars and
                        vol_ratio_b >= SCAN_SR_BOUNCE_VOL and
                        last_price >= sup_b):
                    match = True
                    extra_info = {
                        'Dev Support %':   round(dev_sup_b, 2),
                        'Ruimte Res %':    round(dev_res_b, 2),
                        'Vol Ratio':       vol_ratio_b,
                        'Support':         round(sup_b, 2),
                        'Resistance':      round(res_b, 2),
                        'Kaars':           '🟢 Groen herstel',
                    }

            elif strategy == "momentum_low_float":
                # ── Warrior Trading Momentum Scanner ─────────────────────────
                # 4 harde poorten — alle 4 moeten waar zijn:
                #
                #   1. PRICE RANGE    : $2.00 ≤ koers ≤ $20.00 USD
                #   2. DAILY CHANGE   : vandaag ≥ +10% t.o.v. vorige slotkoers
                #   3. REL. VOLUME    : RVOL ≥ 5× 20D gemiddelde (institutioneel)
                #   4. FLOAT          : vrij verhandelbare aandelen < 10M
                #
                # Alleen USD-genoteerde aandelen (geen Europese tickers).

                # Poort 1 — prijsrange $2-$20
                if not (SCAN_WT_PRICE_MIN <= last_price <= SCAN_WT_PRICE_MAX):
                    continue

                # Poort 2 — dagelijkse stijging ≥ 10%
                if len(close) < 2:
                    continue
                prev_close   = float(close.iloc[-2])
                daily_change = ((last_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
                if daily_change < SCAN_WT_CHANGE_MIN:
                    continue

                # Poort 3 — Relative Volume ≥ 5×
                vol_20d_avg_wt = float(volume.rolling(20).mean().iloc[-1])
                vol_today_wt   = float(volume.iloc[-1])
                rvol = round(vol_today_wt / vol_20d_avg_wt, 2) if vol_20d_avg_wt > 0 else 0.0
                if rvol < SCAN_WT_RVOL_MIN:
                    continue

                # Poort 4 — Float < 10 miljoen (via yfinance info)
                try:
                    t_info   = yf.Ticker(ticker).info
                    float_sh = t_info.get('floatShares', None)
                    if float_sh is None or float_sh <= 0:
                        # Float niet beschikbaar — gebruik shares outstanding als proxy
                        float_sh = t_info.get('sharesOutstanding', None)
                    if float_sh is None:
                        continue    # Geen data → skip
                    if float_sh >= SCAN_WT_FLOAT_MAX:
                        continue    # Te grote float
                except Exception:
                    continue

                # ✅ Alle 4 poorten gehaald → MOMENTUM KANDIDAAT
                match = True
                float_m = round(float_sh / 1_000_000, 2)
                extra_info = {
                    'Dagstijging %': f"+{daily_change:.1f}%",
                    'RVOL':          f"{rvol}×",
                    'Float':         f"{float_m}M aandelen",
                    'Vol Vandaag':   f"{int(vol_today_wt):,}",
                    'Label':         '🔥 MOMENTUM KANDIDAAT',
                }

            elif strategy == "event_sentiment":
                # ── Event Sentiment Scanner ───────────────────────────────────
                # Poort 1 — Hard volume veto (institutioneel geld verplicht)
                vol_20d_avg = float(volume.rolling(VOL_MA_PERIOD).mean().iloc[-1])
                vol_today   = float(volume.iloc[-1])
                vol_spike   = round(vol_today / vol_20d_avg, 2) if vol_20d_avg > 0 else 0.0

                if vol_today < (SCAN_SENT_VOL_SPIKE * vol_20d_avg):
                    # Hard veto — geen institutioneel volume
                    continue

                # Poort 2 & 3 — nieuwsanalyse + extreme sentiment score
                try:
                    t_obj      = yf.Ticker(ticker)
                    news_items = t_obj.news if t_obj.news else []
                    news_count = len(news_items)

                    pos_words = ['surge','gain','rise','rally','beat','upgrade','record',
                                 'profit','growth','strong','boost','jump','buy','bullish',
                                 'breakout','milestone','raised','guidance','exceed']
                    neg_words = ['fall','drop','loss','decline','miss','downgrade','cut',
                                 'risk','warn','crash','sink','plunge','sell','bearish',
                                 'lawsuit','fraud','recall','investigation','downside']

                    pos_hits = neg_hits = 0
                    for item in news_items:
                        tl = item.get('title', '').lower()
                        pos_hits += sum(1 for w in pos_words if w in tl)
                        neg_hits += sum(1 for w in neg_words if w in tl)

                    total_hits     = pos_hits + neg_hits
                    sentiment_score = round(
                        (pos_hits - neg_hits) / total_hits, 2
                    ) if total_hits > 0 else 0.0

                    is_extreme_bull = sentiment_score >  SCAN_SENT_SCORE_BULL
                    is_extreme_bear = sentiment_score <  SCAN_SENT_SCORE_BEAR
                    has_enough_news = news_count >= SCAN_SENT_NEWS_MIN

                    if (is_extreme_bull or is_extreme_bear) and has_enough_news:
                        match     = True
                        direction = "🟢 BULLISH CATALYST" if is_extreme_bull \
                                    else "🔴 BEARISH CATALYST"
                        extra_info = {
                            'Signaal':         direction,
                            'Vol Spike':       vol_spike,
                            'Sentiment Score': sentiment_score,
                            'Nieuws Items':    news_count,
                        }
                except Exception:
                    pass  # Geen nieuws → match blijft False, loop gaat door

            elif strategy == "bb_momentum":
                # ── BB Momentum Scanner ───────────────────────────────────────
                #
                # Zoekt aandelen die al 'min_days' aaneengesloten handelsdagen
                # aan de POSITIEVE kant van de Bollinger Bands sluiten.
                #
                # OPTIE A — BB MOMENTUM: slotkoers boven de 20-daagse SMA
                #   (middellijn) voor N achtereenvolgende dagen.
                #
                # OPTIE B — UPPER BAND WALK: slotkoers boven (of binnen 0.5%
                #   van) de Upper Band voor N achtereenvolgende dagen.
                #   Dit signaleert een echte "band-walk" — zeldzaam en sterk
                #   bullish momentum.
                #
                # min_days wordt per scan ingesteld via de UI-slider en
                # doorgegeven als extra kwarg via st.session_state.

                # Haal instelbare parameter op (default = SCAN_BB_MIN_DAYS)
                min_days = st.session_state.get('bb_min_days', SCAN_BB_MIN_DAYS)

                # ── Bereken BB voor de gevraagde periode ─────────────────────
                if n < SCAN_BB_PERIOD + min_days:
                    continue  # Niet genoeg data voor betrouwbare BB

                sma   = close.rolling(SCAN_BB_PERIOD).mean()
                std   = close.rolling(SCAN_BB_PERIOD).std()
                upper = sma + SCAN_BB_STD * std
                # Pak de laatste min_days + 1 waarden (meest recent = index [-1])
                # We checken indices [-1] t/m [-min_days]: dag 0 t/m dag N-1
                close_window = [float(close.iloc[-(i+1)]) for i in range(min_days)]
                sma_window   = [float(sma.iloc[-(i+1)])   for i in range(min_days)]
                upper_window = [float(upper.iloc[-(i+1)]) for i in range(min_days)]

                if any(np.isnan(v) for row in [close_window, sma_window, upper_window] for v in row):
                    continue  # NaN in BB → niet genoeg data

                # ── Optie A: alle days boven SMA (middellijn) ────────────────
                optie_a = all(c > m for c, m in zip(close_window, sma_window))

                # ── Optie B: alle days aan/boven Upper Band (binnen tolerance)
                # "Binnen 0.5%" = sluit op minder dan 0.5% ONDER de upper band
                # of erboven → telt mee als "aan de band"
                optie_b = all(
                    c >= ub * (1 - SCAN_BB_UB_TOLERANCE)
                    for c, ub in zip(close_window, upper_window)
                )

                if not (optie_a or optie_b):
                    continue

                # ── Bereken nuttige outputwaarden ─────────────────────────────
                cur_close = close_window[0]   # Vandaag
                cur_upper = upper_window[0]
                cur_sma   = sma_window[0]

                # Afstand (%) van slotkoers t.o.v. Upper Band
                dist_to_upper = round((cur_close - cur_upper) / cur_upper * 100, 2)
                # Afstand (%) t.o.v. SMA (middellijn)
                dist_to_sma   = round((cur_close - cur_sma) / cur_sma * 100, 2)

                # Bepaal het signaaltype
                if optie_b:
                    signaal = f"🚀 UPPER BAND WALK ({min_days}D+)"
                else:
                    signaal = f"📈 BB MOMENTUM ({min_days}D+)"

                match = True
                extra_info = {
                    'Signaal':           signaal,
                    'Afstand UB %':      f"{dist_to_upper:+.2f}%",
                    'Afstand SMA %':     f"{dist_to_sma:+.2f}%",
                    'Upper Band':        f"{cur_upper:.2f}",
                    'SMA 20':            f"{cur_sma:.2f}",
                    'Optie A (>SMA)':   '✅' if optie_a else '—',
                    'Optie B (UB Walk)': '✅' if optie_b else '—',
                }

            # ── Match gevonden → R:R berekening (geen zware MTF call) ──────────
            if match:
                # R:R berekening puur op dagdata (al gecached)
                _safe_sup = float(df['Low'].rolling(
                    min(SUPPORT_WINDOW, len(df)), min_periods=1).min().iloc[-1])
                _safe_res = float(df['High'].rolling(
                    min(SUPPORT_WINDOW, len(df)), min_periods=1).max().iloc[-1])
                if _safe_sup <= 0 or _safe_sup > last_price:
                    _safe_sup = last_price * (1 - TRADE_FALLBACK_SL)
                if _safe_res <= last_price:
                    _safe_res = last_price * (1 + TRADE_FALLBACK_RES)

                _sl_scan   = _safe_sup * (1 - TRADE_RISK_PCT)
                _risk_scan = last_price - _sl_scan
                _tp3_scan  = _safe_res * (1 - TRADE_TP3_BUFFER)
                _rew_scan  = _tp3_scan - last_price
                _rr_scan   = round(_rew_scan / _risk_scan, 2) \
                             if _risk_scan > 0 and _rew_scan > 0 else 0.0

                # ── Lichte MTF: alleen weektrend via dagdata (geen extra download) ─
                # Gebruik de al gecachte dagdata ipv nieuwe 1W download
                try:
                    _wk_sma = close.rolling(MTF_WEEKLY_MA * 5).mean()  # ~10 weken
                    _wk_trend = 'BULLISH' if last_price > float(_wk_sma.iloc[-1]) else 'BEARISH'
                except Exception:
                    _wk_trend = 'UNKNOWN'

                # Snelle R:R check zonder volledige MTF
                if _rr_scan < TRADE_REWARD_RATIO:
                    _status = 'REJECTED'
                    _rr_note = f'R:R {_rr_scan}:1 < {TRADE_REWARD_RATIO}:1'
                elif last_price >= _tp3_scan:
                    _status = 'REJECTED'
                    _rr_note = 'Koers >= resistance'
                elif _wk_trend == 'BULLISH':
                    _status = 'APPROVED'
                    _rr_note = f'R:R {_rr_scan}:1'
                else:
                    _status = 'WATCH'
                    _rr_note = f'R:R {_rr_scan}:1'

                # Alpha override: bearish trend normaal aan bodem
                if (strategy == "alpha_scanner" and
                        _status in ('REJECTED', 'WATCH') and
                        _rr_scan >= TRADE_REWARD_RATIO and
                        rsi_val <= SCAN_ALPHA_RSI_MAX and
                        _wk_trend == 'BEARISH'):
                    _status = 'APPROVED'
                    _rr_note = f'R:R {_rr_scan}:1 ⚡ Bodem-override'

                override_note = ''

                row = {
                    'Ticker':    ticker,
                    'Koers':     str(round(last_price, 2)),
                    'RSI (14D)': str(int(rsi_val)),
                    'R:R':       str(_rr_scan),
                    'R:R Status':_rr_note,
                    'MTF Status':_status,
                    '1W Trend':  _wk_trend,
                    '1H Volume': '–',
                }
                if override_note:
                    row['⚡ Bodem Override'] = override_note
                row.update(extra_info)
                rows.append(row)
                print(f"DEBUG HIT [{len(rows):03d}]: {ticker} | RSI={int(rsi_val)} "
                      f"| Status={_status} | R:R={_rr_scan}")

        except Exception as exc:
            errored.append(ticker)
            print(f"DEBUG FOUT: {ticker} → {str(exc)[:80]}")
            continue  # Loop gaat altijd door, ook bij onverwachte fouten

    # ── Loop klaar — opruimen UI ──────────────────────────────────────────────
    progress_bar.progress(1.0, text=f"✅ Scan voltooid: {total} tickers · {len(rows)} hits · {len(failed)} geen data · {len(errored)} fouten")
    status_box.empty()

    print(f"DEBUG: Scan klaar. Pool={total} | Hits={len(rows)} | Geen data={len(failed)} | Fouten={len(errored)}")

    if failed:
        print(f"DEBUG GEEN DATA: {', '.join(failed[:30])}{'...' if len(failed)>30 else ''}")

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # Normaliseer alle kolommen naar string zodat Arrow nooit crasht
    # op gemengde types (int vs str, float vs str, etc.)
    for _col in result_df.columns:
        if _col not in ('_order',):   # interne sorteerkolom mag float blijven
            result_df[_col] = result_df[_col].astype(str).replace('nan', 'N/A')

    # ── Achteraf filteren op max_results (niet tijdens de loop!) ─────────────
    # Sorteer: APPROVED eerst, dan WATCH, dan REJECTED
    status_order = {'APPROVED': 0, 'WATCH': 1, 'REJECTED': 2}
    result_df['_order'] = result_df['MTF Status'].map(status_order).fillna(3)
    result_df = result_df.sort_values('_order').drop(columns=['_order'])

    # ── Post-scan filter voor Event Sentiment ─────────────────────────────────
    # Event Sentiment is het enige dat een eigen limiet heeft (max APPROVED hits)
    # omdat het signaal anders volledig vervuilt door ruis.
    # Alle andere strategieën: GEEN limiet — volledige resultatenlijst.
    if strategy == "event_sentiment" and not result_df.empty:
        result_df = result_df[result_df['MTF Status'] != 'REJECTED'].copy()
        if 'Sentiment Score' in result_df.columns:
            result_df['_abs_score'] = result_df['Sentiment Score'].abs()
            result_df = result_df.sort_values('_abs_score', ascending=False) \
                                  .drop(columns=['_abs_score'])
        approved  = result_df[result_df['MTF Status'] == 'APPROVED'].head(SCAN_SENT_MAX_RESULTS)
        watch     = result_df[result_df['MTF Status'] == 'WATCH']
        result_df = pd.concat([approved, watch], ignore_index=True)
    # Alle andere strategieën: geen .head() — volledige lijst teruggeven

    return result_df


def style_scanner_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Stijl de scanner-resultaten tabel."""
    def row_style(row):
        base = 'background-color: #13171C; color: #E8ECEF;'
        mtf = str(row.get('MTF Status', ''))
        if mtf == 'APPROVED':
            return ['background-color: #0D2818; color: #00C853; font-weight:600;'] * len(row)
        elif mtf == 'REJECTED':
            return ['background-color: #1A0008; color: #F6465D;'] * len(row)
        elif mtf == 'WATCH':
            return ['background-color: #1A1500; color: #F0B90B;'] * len(row)
        return [base] * len(row)

    styled = df.style.apply(row_style, axis=1)
    styled.set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#0B0E11'), ('color', '#F0B90B'),
            ('font-family', 'JetBrains Mono, monospace'), ('font-size', '0.78rem'),
            ('border-bottom', '2px solid #F0B90B'), ('padding', '8px 12px')]},
        {'selector': 'tbody td', 'props': [
            ('font-family', 'JetBrains Mono, monospace'), ('font-size', '0.8rem'),
            ('padding', '6px 12px'), ('border-bottom', '1px solid #2B3139')]}])
    return styled


def _compute_atr_rma(high, low, close, period=14):
    """Bereken ATR met RMA (Running/Wilder Moving Average) smoothing."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    # RMA = EWM met alpha = 1/period (Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr


def build_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Bloomberg-stijl grafiek met:
    1. SMA 200 (oranje, solide, met as-label)
    2. Session background shading (pre-market / regulier / after-hours)
    3. Support/Resistance consolidation box (paars-blauw)
    4. ATR(14) sub-panel met RMA smoothing (donkerrood)
    5. Gekleurde volume bars (groen/rood)
    """

    # Normaliseer MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]

    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low_s = df['Low'].squeeze()
    open_ = df['Open'].squeeze()
    vol_s = df['Volume'].squeeze()

    n = len(close)
    x = np.arange(n)
    dates = df.index

    # Indicatoren
    poly     = np.polyfit(x, close.values.astype(float), 1)
    reg_line = np.polyval(poly, x)
    std_dev  = np.std(close.values.astype(float) - reg_line)

    bb_upper, bb_mid, bb_lower = compute_bollinger_series(close, BB_PERIOD, BB_STD_NORMAL)
    rsi_s  = compute_rsi_series(close, RSI_PERIOD)
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14  = _compute_atr_rma(high, low_s, close, period=14)

    # Volume kleuren
    vol_colors = ['#00C853' if c >= o else '#F6465D'
                  for c, o in zip(close.values, open_.values)]

    # ── 4 rijen: Prijs | Volume | RSI | ATR ──────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.018,
        subplot_titles=[
            f'{ticker}  ·  SMA20/50/200  ·  BB  ·  Regressiekanaal',
            'Volume',
            'RSI (14D)',
            'ATR (14 · RMA)'
        ]
    )

    # ── 2. SESSION BACKGROUND SHADING ────────────────────────────────────────
    # Markeer afwisselend periodes (per 5 handelsdagen = week) met subtiele bands
    # zodat sessieblokken zichtbaar zijn op daggrafieken.
    # Op intraday (1H/4H) zou dit pre/regulier/after-hours zijn.
    # Op daggrafiek gebruiken we alternatief lichte/donkere weekbanden.
    is_daily = True
    try:
        if len(dates) > 1:
            delta = dates[1] - dates[0]
            is_daily = getattr(delta, 'days', 1) >= 1
    except Exception:
        is_daily = True

    if is_daily and len(dates) >= 2:
        # Weekbanden: even weken lichtblauw, oneven weken lichtbruin
        week_starts = []
        current_week = None
        for i, d in enumerate(dates):
            try:
                wk = d.isocalendar()[1]
                if wk != current_week:
                    week_starts.append((i, wk))
                    current_week = wk
            except Exception:
                pass

        for idx_w, (start_i, week_num) in enumerate(week_starts):
            end_i = week_starts[idx_w+1][0] if idx_w+1 < len(week_starts) else n
            if start_i >= len(dates) or end_i > len(dates): continue
            x0 = dates[start_i]
            x1 = dates[min(end_i, len(dates)-1)]
            # Even week = donkerblauw, oneven = donkerbruin
            if week_num % 2 == 0:
                fill_col = 'rgba(33,58,100,0.12)'   # donkerblauw
            else:
                fill_col = 'rgba(90,60,20,0.10)'    # donkerbruin/amber

            # Voeg toe aan alle 4 panels
            for row_n in [1, 2, 3, 4]:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=fill_col,
                    line_width=0,
                    row=row_n, col=1
                )

    # ── CANDLESTICKS ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates, open=open_, high=high,
        low=low_s, close=close,
        increasing_line_color='#00C853', decreasing_line_color='#F6465D',
        increasing_fillcolor='#00C853', decreasing_fillcolor='#F6465D',
        name='OHLC', line=dict(width=1),
    ), row=1, col=1)

    # ── REGRESSIEKANAAL ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=reg_line,
        line=dict(color='#F0B90B', width=1.5, dash='dash'),
        name='Regressie', hovertemplate='%{y:.2f}'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=reg_line + 2*std_dev,
        line=dict(color='#F0B90B', width=1, dash='dot'),
        name='+2σ', hovertemplate='%{y:.2f}'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=reg_line - 2*std_dev,
        line=dict(color='#2196F3', width=1, dash='dot'),
        name='-2σ', hovertemplate='%{y:.2f}',
    ), row=1, col=1)

    # ── BOLLINGER BANDS ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=bb_upper,
        line=dict(color='rgba(240,185,11,0.25)', width=1),
        name='BB Upper', showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bb_lower,
        line=dict(color='rgba(33,150,243,0.25)', width=1),
        name='BB Lower', fill='tonexty',
        fillcolor='rgba(240,185,11,0.03)', showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bb_mid,
        line=dict(color='rgba(255,255,255,0.25)', width=1),
        name='SMA 20', showlegend=False
    ), row=1, col=1)

    # ── SMA 50 ────────────────────────────────────────────────────────────────
    if sma50.notna().sum() >= 10:
        fig.add_trace(go.Scatter(
            x=dates, y=sma50,
            line=dict(color='#29B6F6', width=1.5),
            name='SMA 50', hovertemplate='SMA50: %{y:.2f}'
        ), row=1, col=1)

    # ── 1. SMA 200 — Oranje, solide, met as-label ─────────────────────────────
    last_sma200 = None
    if sma200.notna().sum() >= 10:
        last_sma200 = float(sma200.dropna().iloc[-1])
        fig.add_trace(go.Scatter(
            x=dates, y=sma200,
            line=dict(color='#FF6D00', width=2.5),   # Solid Orange
            name='SMA 200',
            hovertemplate='SMA200: %{y:.2f}',
        ), row=1, col=1)

        # As-label rechts: oranje badge met SMA200 waarde
        fig.add_annotation(
            x=1.0, y=last_sma200,
            xref='paper', yref='y',
            text=f"<b>SMA200  {last_sma200:.2f}</b>",
            showarrow=False,
            font=dict(size=10, color='#FF6D00'),
            bgcolor='rgba(11,14,17,0.85)',
            bordercolor='#FF6D00', borderwidth=1,
            xanchor='left', yanchor='middle',
        )

        # Golden / Death Cross badge
        last_sma50 = float(sma50.dropna().iloc[-1]) if sma50.notna().any() else None
        if last_sma50:
            if last_sma50 > last_sma200:
                cx_txt, cx_col = '✅ Golden Cross', '#00C853'
            else:
                cx_txt, cx_col = '⚠ Death Cross',  '#F6465D'
            fig.add_annotation(
                x=dates[-1], y=last_sma200,
                text=cx_txt, showarrow=False,
                font=dict(size=10, color=cx_col),
                bgcolor='rgba(11,14,17,0.85)',
                bordercolor=cx_col, borderwidth=1,
                xanchor='right', yanchor='bottom',
            )

    # ── 3. SUPPORT/RESISTANCE CONSOLIDATION BOX ───────────────────────────────
    _last_price = float(close.iloc[-1])
    support_val, resistance_val = compute_support_resistance(
        df, SUPPORT_WINDOW, current_price=_last_price)

    # Bovenlijn: lichtpaars   Onderlijn: lichtblauw   Vulling: semi-transparant paars
    fig.add_hrect(
        y0=support_val, y1=resistance_val,
        fillcolor='rgba(120,80,200,0.06)',
        line_width=0,
        row=1, col=1
    )
    # Support lijn — lichtblauw
    fig.add_hline(
        y=support_val,
        line_color='#64B5F6', line_width=1.5, line_dash='dot',
        row=1, col=1,
        annotation_text=f"Support  {support_val:.2f}",
        annotation_font_color='#64B5F6',
        annotation_position='bottom right'
    )
    # Resistance lijn — lichtpaars
    fig.add_hline(
        y=resistance_val,
        line_color='#CE93D8', line_width=1.5, line_dash='dot',
        row=1, col=1,
        annotation_text=f"Resistance  {resistance_val:.2f}",
        annotation_font_color='#CE93D8',
        annotation_position='top right'
    )

    # ── 5. VOLUME BARS (groen/rood gekleurd) ──────────────────────────────────
    fig.add_trace(go.Bar(
        x=dates, y=vol_s,
        marker_color=vol_colors,
        marker_line_width=0,
        name='Volume', showlegend=False,
        opacity=0.85,
    ), row=2, col=1)

    vol_ma = vol_s.rolling(VOL_MA_PERIOD).mean()
    fig.add_trace(go.Scatter(
        x=dates, y=vol_ma,
        line=dict(color='#F0B90B', width=1.5),
        name='Vol MA20', showlegend=False
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=rsi_s,
        line=dict(color='#9C27B0', width=1.5),
        name='RSI', fill='tozeroy', fillcolor='rgba(156,39,176,0.07)'
    ), row=3, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_color='#F6465D', line_width=1, line_dash='dot', row=3, col=1)
    fig.add_hline(y=RSI_OVERSOLD,   line_color='#00C853', line_width=1, line_dash='dot', row=3, col=1)
    fig.add_hline(y=50, line_color='rgba(255,255,255,0.15)', line_width=1, row=3, col=1)
    fig.add_hrect(y0=RSI_OVERSOLD, y1=RSI_OVERBOUGHT,
                  fillcolor='rgba(156,39,176,0.04)', line_width=0, row=3, col=1)

    # ── 4. ATR(14) SUB-PANEL met RMA ─────────────────────────────────────────
    last_atr = float(atr14.dropna().iloc[-1]) if atr14.notna().any() else 0.0
    fig.add_trace(go.Scatter(
        x=dates, y=atr14,
        line=dict(color='#C62828', width=2),    # Solid Dark Red
        name='ATR 14',
        fill='tozeroy',
        fillcolor='rgba(198,40,40,0.07)',
        hovertemplate='ATR14: %{y:.4f}',
    ), row=4, col=1)

    # ATR as-label rechts: rood badge met huidige waarde
    fig.add_annotation(
        x=1.0, y=last_atr,
        xref='paper', yref='y4',
        text=f"<b>ATR14  {last_atr:.3f}</b>",
        showarrow=False,
        font=dict(size=10, color='#C62828'),
        bgcolor='rgba(11,14,17,0.85)',
        bordercolor='#C62828', borderwidth=1,
        xanchor='left', yanchor='middle',
    )

    # ── LAYOUT ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor='#0B0E11',
        plot_bgcolor='#0B0E11',
        font=dict(family='JetBrains Mono, monospace', size=11, color='#E8ECEF'),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor='rgba(19,23,28,0.8)',
            bordercolor='#2B3139', borderwidth=1,
            font=dict(size=10, color='#848E9C'),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        ),
        margin=dict(l=10, r=80, t=40, b=10),
        height=860,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#13171C',
            font_color='#E8ECEF',
            bordercolor='#F0B90B',
            font=dict(family='JetBrains Mono, monospace', size=11),
            namelength=-1,          # Volledige naam tonen in hover
        ),
        dragmode='pan',             # Standaard: pan (zoals TradingView)
        selectdirection='h',
        newshape=dict(
            line=dict(color='#F0B90B', width=1.5),
            fillcolor='rgba(240,185,11,0.1)',
        ),
        modebar=dict(
            bgcolor='rgba(19,23,28,0.85)',
            color='#848E9C',
            activecolor='#F0B90B',
        ),
    )

    # ── ASSEN — crosshair + zoom op alle panels ───────────────────────────────
    for r in [1, 2, 3, 4]:
        fig.update_xaxes(
            row=r, col=1,
            gridcolor='#2B3139',
            zeroline=False,
            showgrid=True,
            tickfont=dict(color='#848E9C'),
            linecolor='#2B3139',
            # Crosshair spike
            showspikes=True,
            spikemode='across+toaxis',
            spikesnap='cursor',
            spikecolor='#848E9C',
            spikethickness=1,
            spikedash='dot',
            fixedrange=False,       # X-as zoombaar
        )
        fig.update_yaxes(
            row=r, col=1,
            gridcolor='#2B3139',
            zeroline=False,
            showgrid=True,
            tickfont=dict(color='#848E9C'),
            linecolor='#2B3139',
            # Crosshair spike
            showspikes=True,
            spikemode='across+toaxis',
            spikesnap='cursor',
            spikecolor='#848E9C',
            spikethickness=1,
            spikedash='dot',
            fixedrange=False,       # Y-as zoombaar
            # Prijs-label rechts op cursor
            showticklabels=True,
        )

    fig.update_yaxes(title_text="Prijs",   row=1, col=1, title_font=dict(color='#F0B90B'))
    fig.update_yaxes(title_text="Volume",  row=2, col=1, title_font=dict(color='#F0B90B'))
    fig.update_yaxes(title_text="RSI",     row=3, col=1, title_font=dict(color='#F0B90B'), range=[0,100])
    fig.update_yaxes(title_text="ATR 14",  row=4, col=1, title_font=dict(color='#C62828'))

    return fig


def fetch_yahoo_rss(ticker: str) -> list:
    """Haal Yahoo Finance RSS-nieuws op voor een ticker."""
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        items = []
        titles = re.findall(r'<title>(.*?)</title>', r.text)
        links = re.findall(r'<link>(.*?)</link>', r.text)
        dates = re.findall(r'<pubDate>(.*?)</pubDate>', r.text)
        for i in range(min(10, len(titles))):
            if i == 0:
                continue
            title = titles[i] if i < len(titles) else ''
            link = links[i] if i < len(links) else ''
            date = dates[i - 1] if (i - 1) < len(dates) else ''
            if title and link:
                items.append({'title': title, 'link': link, 'date': date})
        return items
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA TRACKER — de beste Alpha-scan-resultaten over langere tijd volgen
#   • elke "snapshot" = dezelfde Alpha Scan (run_scanner) + score + opslag
#   • per ticker: hoe vaak in de top, rendement sinds eerste keer, en of hij uit
#     de bodem komt ("op weg omhoog")
#   • opslag: GitHub-bestand (zoals je tickerlijst) + lokaal bestand als terugval
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_HISTORY_FILE = "quantedge_alpha_history.json"
ALPHA_UP_MIN_RET   = 3.0     # % boven de koers van de laatste keer in de top → "op weg omhoog"
ALPHA_FIRST_MIN_RET = 1.0    # % → "eerste herstel"
ALPHA_BROKEN_RET   = -4.0    # % onder de koers van de laatste keer → "support gebroken"

ALPHA_STATUS = {             # sleutel: (label, kaarttint, volgorde)
    'up':      ("🚀 Op weg omhoog",    't-g', 0),
    'first':   ("↗ Eerste herstel",    't-g', 1),
    'bottom':  ("🔎 In bodem",         't-y', 2),
    'wait':    ("⏳ Afwachten",         '',    3),
    'broken':  ("🔴 Support gebroken", 't-r', 4),
    'nodata':  ("⚠ Geen data",         't-x', 5),
}


def _alpha_local_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ALPHA_HISTORY_FILE)


def _alpha_gh():
    """(token, repo, bestandsnaam) voor de historie; hergebruikt je bestaande GitHub-secrets."""
    token, repo, _ = _get_github_config()
    fname = ALPHA_HISTORY_FILE
    try:
        fname = st.secrets.get("GITHUB_ALPHA_FILE", ALPHA_HISTORY_FILE)
    except Exception:
        pass
    return token, repo, fname


def alpha_load_history() -> dict:
    empty = {'version': 1, 'snapshots': []}
    token, repo, fname = _alpha_gh()
    if token and repo:
        try:
            url  = f"https://api.github.com/repos/{repo}/contents/{fname}"
            hdrs = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.raw+json",
                    "X-GitHub-Api-Version": "2022-11-28"}
            resp = requests.get(url, headers=hdrs, timeout=15)
            if resp.status_code == 200:
                data = json.loads(resp.text)
                if isinstance(data.get('snapshots'), list):
                    return data
            elif resp.status_code == 404:
                pass                                     # nog geen historie op GitHub
        except Exception:
            pass
    try:
        with open(_alpha_local_path(), encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data.get('snapshots'), list):
            return data
    except Exception:
        pass
    return empty


def alpha_save_history(hist: dict, message: str = "QuantEdge: alpha-snapshot") -> tuple:
    """Slaat op naar GitHub én lokaal. Retourneert (ok, melding)."""
    payload_txt = json.dumps(hist, separators=(',', ':'), ensure_ascii=False)
    try:
        with open(_alpha_local_path(), 'w', encoding='utf-8') as f:
            f.write(payload_txt)
    except Exception:
        pass
    token, repo, fname = _alpha_gh()
    if not (token and repo):
        return False, "Geen GitHub-token ingesteld — historie alleen lokaal bewaard (verdwijnt bij herstart van de app)"
    try:
        import base64
        url  = f"https://api.github.com/repos/{repo}/contents/{fname}"
        hdrs = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}
        sha  = None
        resp = requests.get(url, headers=hdrs, timeout=15)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
        elif resp.status_code == 401:
            return False, "GitHub-token ongeldig (401)"
        body = {"message": message, "content": base64.b64encode(payload_txt.encode("utf-8")).decode("utf-8")}
        if sha:
            body["sha"] = sha
        resp = requests.put(url, headers=hdrs, json=body, timeout=20)
        if resp.status_code in (200, 201):
            return True, "OK"
        return False, f"GitHub-fout {resp.status_code}"
    except Exception as e:
        return False, str(e)


def _num(v, default=np.nan) -> float:
    try:
        return float(str(v).replace(',', '.'))
    except (TypeError, ValueError):
        return default


def alpha_score(rsi: float, dev: float, vol: float, rr: float, mtf: str) -> float:
    """
    Kwaliteitsscore 0–100 van één Alpha-hit (hoger = sterker bodem-setup):
      30  dicht bij support            (0% boven support = 30, 5% = 0)
      25  diep oversold                (RSI 20 = 25, RSI 45 = 0)
      25  weekly-trend/MTF-oordeel     (APPROVED 25, WATCH 10)
      10  Risk:Reward                  (≥ 2× doel-R:R = 10)
      10  volume t.o.v. 20D            (≥ 1.5× = 10)
    """
    s = 0.0
    if not np.isnan(dev):
        s += 30 * max(0.0, 1 - dev / max(SCAN_ALPHA_DEV_MAX, 0.1))
    if not np.isnan(rsi):
        s += 25 * min(1.0, max(0.0, (SCAN_ALPHA_RSI_MAX - rsi) / max(SCAN_ALPHA_RSI_MAX - 20, 1)))
    s += {'APPROVED': 25, 'WATCH': 10}.get(str(mtf), 0)
    if not np.isnan(rr):
        s += 10 * min(1.0, rr / (2 * TRADE_REWARD_RATIO))
    if not np.isnan(vol):
        s += 10 * min(1.0, vol / 1.5)
    return round(s, 1)


def alpha_make_snapshot(result_df: pd.DataFrame, pool_size: int, keep_top: int) -> dict:
    items = []
    if result_df is not None and not result_df.empty:
        for _, r in result_df.iterrows():
            rsi = _num(r.get('RSI (14D)'));  dev = _num(r.get('Dev Support %'))
            vol = _num(r.get('Vol Ratio'));  rr  = _num(r.get('R:R'))
            px  = _num(r.get('Koers'));      mtf = str(r.get('MTF Status', ''))
            if np.isnan(px):
                continue
            items.append({
                't': str(r['Ticker']), 'px': round(px, 2),
                'rsi': None if np.isnan(rsi) else round(rsi),
                'dev': None if np.isnan(dev) else round(dev, 2),
                'vol': None if np.isnan(vol) else round(vol, 2),
                'rr':  None if np.isnan(rr) else round(rr, 2),
                'mtf': mtf, 'score': alpha_score(rsi, dev, vol, rr, mtf),
            })
    items.sort(key=lambda d: d['score'], reverse=True)
    return {'ts': _now_nl().isoformat(timespec='minutes'), 'pool': int(pool_size),
            'hits': len(items), 'top': items[:keep_top], 'states': {}}


def alpha_status(core, last_px: float, last_rsi) -> str:
    """Bepaal de status van een gevolgde ticker t.o.v. de laatste keer dat hij in de top stond."""
    if not core:
        return 'nodata'
    px, rsi, dev = core['last_price'], core['rsi'], core['deviation']
    rsi_ok = rsi is not None and not (isinstance(rsi, float) and np.isnan(rsi))
    ret_last = (px / last_px - 1) * 100 if last_px else 0.0
    left_zone = (rsi_ok and rsi > SCAN_ALPHA_RSI_MAX) or dev > SCAN_ALPHA_DEV_MAX
    rising    = rsi_ok and last_rsi is not None and rsi > last_rsi
    if dev < 0 or ret_last <= ALPHA_BROKEN_RET:
        return 'broken'
    if left_zone and ret_last >= ALPHA_UP_MIN_RET and rising:
        return 'up'
    if ret_last >= ALPHA_FIRST_MIN_RET and rising:
        return 'first'
    if not left_zone:
        return 'bottom'
    return 'wait'


def alpha_fetch_current(tickers: list) -> dict:
    """Actuele dagdata van alle gevolgde tickers, parallel (5 min gecachet)."""
    def _w(t):
        try:
            return _fetch_daily_core(t, DATA_MAIN_PERIOD)
        except Exception:
            return None
    res = _parallel_map(_w, tickers)
    return {t: (r if isinstance(r, dict) else None) for t, r in res.items()}


def alpha_evaluate(hist: dict, cur: dict) -> pd.DataFrame:
    """Eén rij per ooit-gevolgde ticker, met statistieken en actuele status."""
    snaps = hist.get('snapshots', [])
    seen = {}
    for i, sn in enumerate(snaps):
        for it in sn.get('top', []):
            d = seen.setdefault(it['t'], {'recs': []})
            d['recs'].append((i, sn['ts'], it))
    rows = []
    n_snaps = len(snaps)
    for t, d in seen.items():
        recs = d['recs']
        first_i, first_ts, first_it = recs[0]
        last_i,  last_ts,  last_it  = recs[-1]
        scores = [r[2]['score'] for r in recs]
        since_first = n_snaps - first_i
        core = cur.get(t)
        px_now = core['last_price'] if core else np.nan
        rsi_now = core['rsi'] if core else np.nan
        key = alpha_status(core, last_it['px'], last_it.get('rsi'))
        rows.append({
            'Ticker': t,
            'Status': ALPHA_STATUS[key][0], '_status': key,
            'Keer in top': len(recs), 'Van snapshots': since_first,
            'Consistentie %': round(100 * len(recs) / since_first),
            'Gem. score': round(float(np.mean(scores)), 1), 'Beste score': max(scores),
            'Eerste keer': first_ts[:10], 'Instapkoers': first_it['px'],
            'Laatst in top': last_ts[:10], 'Koers laatst': last_it['px'],
            'RSI laatst': last_it.get('rsi'),
            'Koers nu': px_now, 'RSI nu': rsi_now,
            'Afw. support nu %': core['deviation'] if core else np.nan,
            'Rend. sinds 1e keer %': round((px_now / first_it['px'] - 1) * 100, 1) if core and first_it['px'] else np.nan,
            'Rend. sinds laatst %':  round((px_now / last_it['px'] - 1) * 100, 1) if core and last_it['px'] else np.nan,
            '_symbol': _currency_symbol(t),
            '_in_latest': last_i == n_snaps - 1,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ('RSI nu', 'RSI laatst'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


ALPHA_SORTS = {
    "Meest consistent":         (['Keer in top', 'Gem. score'], [False, False]),
    "Hoogste gem. score":       (['Gem. score', 'Keer in top'], [False, False]),
    "Beste rendement":          (['Rend. sinds 1e keer %'],     [False]),
    "Status (omhoog eerst)":    (['_ord', 'Keer in top'],       [True, False]),
}


def alpha_sorted(df: pd.DataFrame, sort_key: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out['_ord'] = out['_status'].map(lambda k: ALPHA_STATUS[k][2])
    cols, asc = ALPHA_SORTS.get(sort_key, ALPHA_SORTS["Meest consistent"])
    return out.sort_values(cols, ascending=asc, na_position='last')


def _fmt_pct(v) -> str:
    return '–' if pd.isna(v) else f"{v:+.1f}%"


def render_alpha_cards(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '<div class="qe-empty">Geen tickers voor dit filter.</div>'
    parts = ['<div class="qe-list">']
    for _, r in df.iterrows():
        key  = r['_status']
        tone = ALPHA_STATUS[key][1]
        sym  = r['_symbol']
        px   = '–' if pd.isna(r['Koers nu']) else f"{sym}{r['Koers nu']:,.2f}"
        ret1 = r['Rend. sinds 1e keer %']
        ret_cls = 'g' if (pd.notna(ret1) and ret1 > 0) else 'r' if (pd.notna(ret1) and ret1 < 0) else ''
        act_cls = 'g' if key in ('up', 'first') else 'r' if key == 'broken' else 'y' if key == 'bottom' else ''
        rsi_txt = f"{r['RSI laatst']:.0f}→{r['RSI nu']:.0f}" if pd.notna(r['RSI nu']) and pd.notna(r['RSI laatst']) else '–'
        body = (
            _field('Eerste keer', f"{r['Eerste keer']} @ {sym}{r['Instapkoers']:,.2f}")
            + _field('Laatst in top', f"{r['Laatst in top']} @ {sym}{r['Koers laatst']:,.2f}")
            + _field('Rend. sinds 1e keer', _fmt_pct(ret1))
            + _field('Rend. sinds laatst', _fmt_pct(r['Rend. sinds laatst %']))
            + _field('RSI (toen → nu)', rsi_txt)
            + _field('Afw. support nu', '–' if pd.isna(r['Afw. support nu %']) else f"{r['Afw. support nu %']:.1f}%")
            + _field('Gem. / beste score', f"{r['Gem. score']:.0f} / {r['Beste score']:.0f}")
            + _field('Consistentie', f"{int(r['Keer in top'])} van {int(r['Van snapshots'])} scans")
        )
        parts.append(
            f'<details class="qe-card {tone}"><summary>'
            f'<div class="qe-r1"><span class="qe-tk">{_esc(str(r["Ticker"]))}</span><span class="qe-px">{_esc(px)}</span></div>'
            f'<div class="qe-r2"><div class="qe-chips">'
            f'<span class="qe-act {act_cls}">{_esc(ALPHA_STATUS[key][0])}</span>'
            f'<span class="qe-chip">🏆 <b>{int(r["Keer in top"])}×</b></span>'
            f'<span class="qe-chip {ret_cls}">Sinds 1e <b>{_esc(_fmt_pct(ret1))}</b></span>'
            f'</div></div></summary><div class="qe-body">{body}</div></details>'
        )
    parts.append('</div>')
    return ''.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NAVIGATION  (Watchlist-tab verwijderd – functionaliteit in Tab 1)
# ─────────────────────────────────────────────────────────────────────────────
main_tabs = st.tabs([
    "📊 Market Tracker",
    "🔍 Multi-Strategie Scanner",
    "📈 Deep Dive & Grafiek",
    "📡 Live Monitor",
    "🏆 Alpha Tracker"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CORE MARKET TRACKER  (+ ingebouwde Watchlist)
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[0]:
    # Op een telefoon standaard de kaartweergave (eenmalig; daarna bepaalt de gebruiker het zelf)
    if 'mobile_view' not in st.session_state:
        st.session_state['mobile_view'] = _is_mobile_ua()

    # ── Titel + refresh ───────────────────────────────────────────────────────
    with st.container(key="qe_row_top"):
        _t1, _t2 = st.columns([5, 1])
        with _t1:
            st.markdown("### 📊 Market Tracker")
        with _t2:
            if st.button("🔄", key="btn_refresh_main", help="Koersen opnieuw ophalen"):
                refresh_market_data()

    # ── Beheer (ingeklapt, zodat de koersen direct bovenaan staan) ─────────────
    with st.expander(f"⚙️ Tickers beheren · {len(st.session_state.main_market_tickers)} in lijst", expanded=False):
        tab_add, tab_imp, tab_del, tab_exp = st.tabs(["➕ Toevoegen", "📋 Import", "🗑 Verwijderen", "📤 Export"])

        with tab_add:
            new_ticker = st.text_input(
                "Ticker toevoegen",
                placeholder="bijv. AAPL, ADYEN.AS, BESI.AS...",
                key="add_ticker_input",
            )
            if st.button("Toevoegen", key="btn_add_ticker"):
                t = new_ticker.strip().upper()
                if t and t not in st.session_state.main_market_tickers:
                    st.session_state.main_market_tickers.append(t)
                    _save_userdata()
                    err = st.session_state.get("_last_save_error")
                    if err:
                        st.warning(f"✅ {t} toegevoegd (lokaal) maar GitHub opslag mislukt: {err}")
                    else:
                        st.success(f"✅ {t} toegevoegd en opgeslagen.")
                    st.rerun()
                elif t in st.session_state.main_market_tickers:
                    st.warning(f"⚠ {t} staat al in de lijst.")

        with tab_imp:
            import_txt = st.text_area(
                "Plak tickers (komma- of regelgescheiden)",
                placeholder="AAPL, MSFT, GOOGL\nTSLA\nNVDA",
                height=100, key="import_tickers_txt"
            )
            if st.button("✅ Importeer lijst", key="btn_import_tickers"):
                raw = re.split(r'[\n,;]+', import_txt)
                added = []
                for t in raw:
                    t = t.strip().upper()
                    if t and t not in st.session_state.main_market_tickers:
                        st.session_state.main_market_tickers.append(t)
                        added.append(t)
                if added:
                    _save_userdata()
                    st.success(f"✅ {len(added)} tickers toegevoegd: {', '.join(added)}")
                    st.rerun()
                else:
                    st.info("Geen nieuwe tickers gevonden.")

        with tab_del:
            ticker_to_remove = st.selectbox(
                "Ticker verwijderen",
                options=["— Selecteer —"] + st.session_state.main_market_tickers,
                key="remove_ticker_select"
            )
            if st.button("Verwijderen", key="btn_remove_ticker"):
                if ticker_to_remove != "— Selecteer —" and ticker_to_remove in st.session_state.main_market_tickers:
                    st.session_state.main_market_tickers.remove(ticker_to_remove)
                    _save_userdata()
                    st.success(f"🗑 {ticker_to_remove} verwijderd.")
                    st.rerun()

        with tab_exp:
            st.text_area(
                "Kopieer deze lijst en bewaar hem",
                value=", ".join(st.session_state.main_market_tickers),
                height=120,
                key="export_tickers_txt"
            )
            st.caption("💡 Plak dit de volgende keer bij 'Import' om je lijst te herstellen.")

    # ── Data ophalen (parallel, met voortgangsbalk) ───────────────────────────
    want_intra = not st.session_state.get('mobile_view', False)
    try:
        main_df, loaded_ts = load_main_df(tuple(st.session_state.main_market_tickers), want_intra)
    except Exception as _e:
        st.error(f"❌ Fout bij laden van de hoofdtabel: {type(_e).__name__}: {_e}")
        main_df, loaded_ts = pd.DataFrame(), time.time()

    if main_df.empty:
        st.warning("Geen tickers in je lijst of geen data beschikbaar. Voeg tickers toe via ⚙️ Tickers beheren.")
    else:
        ok_mask = main_df['_ok'] == True                                    # noqa: E712
        bad_df  = main_df[~ok_mask]
        age_s   = int(time.time() - loaded_ts)
        st.caption(f"{int(ok_mask.sum())} van {len(main_df)} tickers geladen · "
                   f"{'zojuist' if age_s < 10 else f'{age_s}s geleden'} bijgewerkt")

        # ── Tickers zonder data: laat zien waarom, verwijder NIETS automatisch ──
        if not bad_df.empty:
            n_rl = int(bad_df['_err'].str.contains('rate-limit', na=False).sum())
            with st.expander(f"⚠ {len(bad_df)} tickers zonder data" + (f" · {n_rl} door Yahoo rate-limit" if n_rl else ""),
                             expanded=False):
                if n_rl:
                    st.info("Yahoo remde even af. Dit is tijdelijk: tik op 'Opnieuw proberen' — "
                            "alleen de mislukte tickers worden dan opnieuw opgehaald.")
                st.markdown('<br>'.join(
                    f"<span style='font-family:monospace;'><b style='color:#F0B90B;'>{_esc(str(tk))}</b>"
                    f" <span style='color:#848E9C;'>— {_esc(str(er))}</span></span>"
                    for tk, er in zip(bad_df['Ticker'], bad_df['_err'])), unsafe_allow_html=True)
                if st.button("🔁 Opnieuw proberen", key="btn_retry_failed"):
                    st.session_state.pop('_main_cache', None)   # mislukte calls zijn niet gecachet
                    st.rerun()
                to_drop = st.multiselect("Uit mijn lijst verwijderen (alleen als je zeker weet dat ze niet meer bestaan)",
                                         options=bad_df['Ticker'].tolist(), key="drop_bad_tickers")
                if to_drop and st.button(f"🗑 Verwijder {len(to_drop)} ticker(s)", key="btn_drop_bad"):
                    st.session_state.main_market_tickers = [
                        t for t in st.session_state.main_market_tickers if t not in to_drop]
                    _save_userdata()
                    st.session_state.pop('_main_cache', None)
                    st.rerun()

        # ── KPI-tegels (2×2 op telefoon, 4 naast elkaar op desktop) ───────────
        n_bodem   = int(main_df['Koers Status / Fase'].str.contains('Bodemfase', na=False).sum())
        n_uptrend = int(main_df['Koers Status / Fase'].str.contains('Uptrend', na=False).sum())
        n_herstel = int(main_df['Koers Status / Fase'].str.contains('Herstel', na=False).sum())
        n_koop    = int(main_df['Actie'].str.contains('KOOPWAARDIG', na=False).sum())
        st.markdown(render_kpis([
            ("🟢 Koopwaardig", n_koop, 'g'), ("⚡ Bodemfase", n_bodem, ''),
            ("🚀 Uptrend", n_uptrend, ''),   ("🌱 Vroeg herstel", n_herstel, ''),
        ]), unsafe_allow_html=True)

        # ── Filter · zoek · sorteer ───────────────────────────────────────────
        try:
            flt = st.pills("Filter", MAIN_FILTERS, default="Alles", key="main_filter",
                           label_visibility="collapsed")
        except Exception:                                  # oudere Streamlit zonder st.pills
            flt = st.radio("Filter", MAIN_FILTERS, horizontal=True, key="main_filter_r",
                           label_visibility="collapsed")
        flt = flt or "Alles"

        with st.container(key="qe_row_filter"):
            _f1, _f2 = st.columns(2)
            with _f1:
                q = st.text_input("Zoek", placeholder="🔎 Zoek ticker", key="main_search",
                                  label_visibility="collapsed")
            with _f2:
                sort_by = st.selectbox("Sorteer", MAIN_SORTS, key="main_sort", label_visibility="collapsed")

        use_cards = st.toggle("📱 Kaartweergave", key="mobile_view",
                              help="Aan: compacte kaarten (tik om uit te klappen). "
                                   "Uit: volledige tabel met intraday-kolommen (15M/5M).")

        view_df = filter_sort_main(main_df, flt, q, sort_by)
        if len(view_df) != len(main_df):
            st.caption(f"{len(view_df)} van {len(main_df)} tickers getoond")

        if use_cards:
            st.markdown(render_ticker_cards(view_df), unsafe_allow_html=True)
        else:
            if view_df.empty:
                st.info("Geen tickers voor dit filter.")
            else:
                st.dataframe(style_main_table(view_df), width='stretch', hide_index=True,
                             height=min(640, 60 + 36 * len(view_df)))

        # ── Deep Dive selectie ────────────────────────────────────────────────
        st.markdown("---")
        dive_opts = main_df.loc[ok_mask, 'Ticker'].tolist() or main_df['Ticker'].tolist()
        with st.container(key="qe_row_dive"):
            sel_c1, sel_c2 = st.columns([3, 2])
            with sel_c1:
                selected_for_dive = st.selectbox(
                    "🔭 Ticker voor Deep Dive",
                    options=dive_opts,
                    index=0,
                    key="main_deep_dive_select"
                )
            with sel_c2:
                st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
                if st.button("📈 Open Deep Dive", key="btn_open_deep_dive_main"):
                    st.session_state.current_alpha = selected_for_dive
                    st.success(f"✅ {selected_for_dive} geladen → ga naar '📈 Deep Dive & Grafiek'")

    # ── WATCHLIST (ingebouwd in Tab 1 als expander) ───────────────────────────
    st.markdown("---")
    with st.expander("👁 Watchlist & Kaarsherkenner", expanded=False):
        st.markdown("##### Handmatige Watchlist — Bullish Hammer · Engulfing · Doji detectie")

        wl_c1, wl_c2, wl_c3 = st.columns([2, 2, 2])

        with wl_c1:
            wl_new_ticker = st.text_input("➕ Ticker", placeholder="bijv. BESI.AS", key="wl_add_input")
            wl_new_label  = st.text_input("📝 Label", placeholder="bijv. Chip leader", key="wl_add_label")
            if st.button("Toevoegen", key="btn_wl_add"):
                t = wl_new_ticker.strip().upper()
                if t:
                    st.session_state.custom_watchlist[t] = wl_new_label or t
                    _save_userdata()
                    st.success(f"✅ {t} toegevoegd.")
                    st.rerun()

        with wl_c2:
            if st.session_state.custom_watchlist:
                wl_del = st.selectbox(
                    "🗑 Verwijderen",
                    options=["— Selecteer —"] + list(st.session_state.custom_watchlist.keys()),
                    key="wl_del_select"
                )
                if st.button("Verwijderen", key="btn_wl_del"):
                    if wl_del != "— Selecteer —" and wl_del in st.session_state.custom_watchlist:
                        del st.session_state.custom_watchlist[wl_del]
                        _save_userdata()
                        st.success(f"🗑 {wl_del} verwijderd.")
                        st.rerun()

        with wl_c3:
            st.markdown("**Actieve Watchlist**")
            for t, lbl in st.session_state.custom_watchlist.items():
                st.markdown(f"- `{t}` <span style='color:#848E9C;'>— {lbl}</span>", unsafe_allow_html=True)

        if st.session_state.custom_watchlist:
            st.markdown("---")
            _wl_items = list(st.session_state.custom_watchlist.items())
            _wl_res   = _parallel_map(_wl_row, _wl_items)
            wl_rows = []
            for _it in _wl_items:
                _r = _wl_res.get(_it)
                if not isinstance(_r, dict):
                    _r = {'Ticker': _it[0], 'Label': _it[1], 'Koers': 'ERR', 'Dag %': 'ERR', 'RSI': 'ERR',
                          'Support': 'ERR', 'Weerstand': 'ERR', 'Afw. Support %': 'ERR',
                          'Gedetecteerd Patroon': '⚠ Tijdelijk niet beschikbaar'}
                wl_rows.append(_r)

            if wl_rows:
                wl_df = pd.DataFrame(wl_rows)

                def style_wl(df):
                    def rs(row):
                        base = 'background-color: #13171C; color: #E8ECEF;'
                        styles = [base] * len(row)
                        col_list = list(row.index)

                        # Patroon kleur
                        pat = str(row.get('Gedetecteerd Patroon', ''))
                        if 'Gedetecteerd Patroon' in col_list:
                            idx = col_list.index('Gedetecteerd Patroon')
                            if 'Bullish' in pat:
                                styles[idx] = 'background-color: #00843A; color: #FFFFFF; font-weight:700;'
                            elif 'Doji' in pat:
                                styles[idx] = 'background-color: #1A1500; color: #F0B90B;'
                            elif 'Bearish' in pat or 'Shooting' in pat:
                                styles[idx] = 'background-color: #A32040; color: #FFFFFF; font-weight:600;'

                        # Afwijking Support % — groen als dicht bij support
                        if 'Afw. Support %' in col_list:
                            idx = col_list.index('Afw. Support %')
                            try:
                                dev_f = float(str(row.get('Afw. Support %', '999')))
                                if dev_f <= 2.0:
                                    styles[idx] = 'background-color: #00843A; color: #FFFFFF; font-weight:600;'
                                elif dev_f > 10.0:
                                    styles[idx] = 'background-color: #1A1500; color: #F0B90B;'
                            except (ValueError, TypeError):
                                pass

                        # Support cel — subtiel groen
                        if 'Support' in col_list:
                            idx = col_list.index('Support')
                            styles[idx] = 'color: #00C853;'

                        # Weerstand cel — subtiel rood
                        if 'Weerstand' in col_list:
                            idx = col_list.index('Weerstand')
                            styles[idx] = 'color: #F6465D;'

                        return styles

                    s = df.style.apply(rs, axis=1)
                    s.set_table_styles([
                        {'selector': 'thead th', 'props': [
                            ('background-color', '#0B0E11'), ('color', '#F0B90B'),
                            ('font-family', 'JetBrains Mono, monospace'), ('font-size', '0.78rem'),
                            ('border-bottom', '2px solid #F0B90B'), ('padding', '8px 12px')]},
                        {'selector': 'tbody td', 'props': [
                            ('font-family', 'JetBrains Mono, monospace'), ('font-size', '0.8rem'),
                            ('padding', '6px 12px'), ('border-bottom', '1px solid #2B3139')]}])
                    return s

                wl_mobile = st.session_state.get('mobile_view', False)
                if wl_mobile:
                    wl_mobile_cols = ['Ticker', 'Koers', 'Dag %', 'RSI', 'Afw. Support %', 'Gedetecteerd Patroon']
                    wl_mobile_cols = [c for c in wl_mobile_cols if c in wl_df.columns]
                    st.dataframe(style_wl(wl_df[wl_mobile_cols]), width='stretch', height=300, hide_index=True)
                else:
                    st.dataframe(style_wl(wl_df), width='stretch', height=300, hide_index=True)

                bullish_n = sum(1 for r in wl_rows if 'Bullish' in str(r.get('Gedetecteerd Patroon', '')))
                if bullish_n > 0:
                    st.success(f"🟢 **{bullish_n} Bullish patroon(en)** gedetecteerd in je watchlist!")
                else:
                    st.info("Geen bullish patronen op dit moment.")



# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: MULTI-STRATEGIE SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.markdown("### 🔍 Multi-Strategie Scanner — 600+ Tickers")
    st.markdown('<small style="color:#848E9C;">S&P 500 · S&P MidCap 400 · Europese Top · Multi-Timeframe Validatie</small>', unsafe_allow_html=True)

    # Toon pool-grootte zonder data te laden
    _own_count = len(st.session_state.get('main_market_tickers', DEFAULT_TICKERS))
    if 'total_ticker_count' not in st.session_state:
        st.session_state['total_ticker_count'] = _own_count
        st.session_state['scan_pool_size']     = _own_count
        st.session_state['pool_scrape_status'] = '📋 Eigen tickerlijst'

    st.markdown("---")

    with st.expander("ℹ Over de Scanner — Strategieën & Tickerpool", expanded=False):
        st.markdown(f"""
        Gecombineerde pool: **S&P 500** · **S&P MidCap 400** · **Europese top-tickers** · Extra ETFs & grondstoffen — volledig, geen limieten.

        Alle resultaten worden gevalideerd via de **Multi-Timeframe Engine** (1W macro + 1H volume) én het **R:R Veto** (minimum {TRADE_REWARD_RATIO}:1).

        | Strategie | Logica |
        |---|---|
        | 🚂 **Momentum Trein** | RSI {SCAN_MOM_RSI_MIN}-{SCAN_MOM_RSI_MAX} · koers boven regressie · {SCAN_MOM_RUIMTE_MIN}-{SCAN_MOM_RUIMTE_MAX}% ruimte tot weerstand · 2 groene kaarsen |
        | 🎯 **Mean Reversion** | RSI <{SCAN_MEAN_RSI_OS} of >{SCAN_MEAN_RSI_OB} · koers buiten Bollinger {BB_STD_WIDE}σ |
        | ⚡ **Volatiliteit Squeeze** | BB Width op {SCAN_SQZ_WINDOW}-daags laagste punt · explosieve move verwacht |
        | 📰 **Event Sentiment** | Hard volume veto {SCAN_SENT_VOL_SPIKE}× + sentiment score > {SCAN_SENT_SCORE_BULL}/< {SCAN_SENT_SCORE_BEAR} + ≥{SCAN_SENT_NEWS_MIN} nieuws |
        | 🔎 **Alpha Scanner** | RSI ≤ {SCAN_ALPHA_RSI_MAX} (oversold) · koers {SCAN_ALPHA_DEV_MIN}-{SCAN_ALPHA_DEV_MAX}% boven support · volume ≥ {SCAN_ALPHA_VOL_SPIKE}× (laag ok = accumulatie) |
        | 🧱 **S/R Bounce Scanner** | Koers 0-{SCAN_SR_BOUNCE_MAX}% boven support · groene herstelkaars · volume ≥ {SCAN_SR_BOUNCE_VOL}× |
        | 🔥 **Low Float Momentum** | Warrior Trading: koers ${SCAN_WT_PRICE_MIN}-${SCAN_WT_PRICE_MAX} · dagstijging ≥{SCAN_WT_CHANGE_MIN}% · RVOL ≥{SCAN_WT_RVOL_MIN}× · float <{SCAN_WT_FLOAT_MAX//1_000_000}M |

        **Alpha vs S/R Bounce verschil:**
        - *Alpha Scanner*: diep oversold (RSI < {SCAN_ALPHA_RSI_MAX}), vroeg instapmoment aan de bodem
        - *S/R Bounce*: RSI onbeperkt, pure price-action bounce van support met herstel-kaars
        """)


    st.markdown("**Kies een Strategie:**")

    strategy_map = {
        "momentum_train":  (
            "🚂 Momentum Trein",
            f"RSI {SCAN_MOM_RSI_MIN}-{SCAN_MOM_RSI_MAX} · koers boven regressie · {SCAN_MOM_RUIMTE_MIN}-{SCAN_MOM_RUIMTE_MAX}% ruimte tot weerstand"
        ),
        "mean_reversion":  (
            "🎯 Mean Reversion",
            f"RSI <{SCAN_MEAN_RSI_OS} of >{SCAN_MEAN_RSI_OB} · buiten Bollinger {BB_STD_WIDE}σ"
        ),
        "vol_squeeze":     (
            "⚡ Volatiliteit Squeeze",
            f"BB Width op {SCAN_SQZ_WINDOW}-daags minimum · explosieve move verwacht"
        ),
        "event_sentiment": (
            "📰 Event Sentiment",
            f"Hard volume veto {SCAN_SENT_VOL_SPIKE}× 20D avg · sentiment > {SCAN_SENT_SCORE_BULL} of < {SCAN_SENT_SCORE_BEAR} · max {SCAN_SENT_MAX_RESULTS} APPROVED"
        ),
        "alpha_scanner":   (
            "🔎 Alpha Scanner",
            f"Bodemfase / Waarde: RSI ≤ {SCAN_ALPHA_RSI_MAX} · koers {SCAN_ALPHA_DEV_MIN}-{SCAN_ALPHA_DEV_MAX}% boven support · volume ≥ {SCAN_ALPHA_VOL_SPIKE}× (laag volume = accumulatie)"
        ),
        "sr_bounce":       (
            "🧱 S/R Bounce Scanner",
            f"Pullback Springplank: koers 0-{SCAN_SR_BOUNCE_MAX}% boven support · groene herstelkaars · volume ≥ {SCAN_SR_BOUNCE_VOL}×"
        ),
        "momentum_low_float": (
            "🔥 Low Float Momentum",
            f"Warrior Trading: koers ${SCAN_WT_PRICE_MIN}-${SCAN_WT_PRICE_MAX} · dagstijging ≥{SCAN_WT_CHANGE_MIN}% · RVOL ≥{SCAN_WT_RVOL_MIN}× · float <{SCAN_WT_FLOAT_MAX//1_000_000}M"
        ),
        "bb_momentum": (
            "📊 BB Momentum",
            f"BB({SCAN_BB_PERIOD},{SCAN_BB_STD}) · N achtereenvolgende dagen boven SMA of Upper Band · instelbare min_days parameter"
        ),
    }

    # Twee rijen: 4 + 4
    row1_keys = list(strategy_map.keys())[:4]
    row2_keys = list(strategy_map.keys())[4:]

    btn_row1 = st.columns(4)
    for i, sk in enumerate(row1_keys):
        label, tooltip = strategy_map[sk]
        with btn_row1[i]:
            if st.button(label, key=f"scan_btn_{sk}", help=tooltip, width='stretch'):
                st.session_state.active_strategy = sk
                st.session_state.scanner_results = pd.DataFrame()
                st.session_state['scan_running'] = False
                st.session_state['scan_rows']    = []
                st.session_state['scan_offset']  = 0

    btn_row2 = st.columns(4)
    for i, sk in enumerate(row2_keys):
        label, tooltip = strategy_map[sk]
        with btn_row2[i]:
            if st.button(label, key=f"scan_btn_{sk}", help=tooltip, width='stretch'):
                st.session_state.active_strategy = sk
                st.session_state.scanner_results = pd.DataFrame()
                st.session_state['scan_running'] = False
                st.session_state['scan_rows']    = []
                st.session_state['scan_offset']  = 0

    if st.session_state.active_strategy:
        _active = st.session_state.active_strategy
        if _active in strategy_map:
            strategy_label, strategy_desc = strategy_map[_active]
        else:
            strategy_label, strategy_desc = _active, ""
        st.markdown(f"**Actieve Strategie:** {strategy_label} — <span style='color:#848E9C;'>{strategy_desc}</span>", unsafe_allow_html=True)

        # ── BB Momentum: instelbare min_days slider ───────────────────────────
        if _active == "bb_momentum":
            _col_sl, _col_exp = st.columns([1, 3])
            with _col_sl:
                bb_min_days_val = st.slider(
                    "📅 Min. aaneengesloten dagen",
                    min_value=1, max_value=10,
                    value=st.session_state.get('bb_min_days', SCAN_BB_MIN_DAYS),
                    step=1,
                    key="bb_min_days_slider",
                    help="Aantal handelsdagen dat de koers aaneengesloten boven de SMA (Optie A) "
                         "of Upper Band (Optie B) moet sluiten om als hit te worden gemarkeerd."
                )
                st.session_state['bb_min_days'] = bb_min_days_val
            with _col_exp:
                st.markdown(
                    f"<div style='background:#13171C;border:1px solid #2B3139;border-radius:6px;"
                    f"padding:10px 14px;font-size:0.78rem;font-family:monospace;'>"
                    f"<span style='color:#848E9C;'>Optie A</span> "
                    f"<b style='color:#F0B90B;'>BB MOMENTUM</b>: slotkoers ≥ {bb_min_days_val}× boven "
                    f"<span style='color:#2196F3;'>SMA {SCAN_BB_PERIOD}</span> (middellijn)<br>"
                    f"<span style='color:#848E9C;'>Optie B</span> "
                    f"<b style='color:#00C853;'>UPPER BAND WALK</b>: slotkoers ≥ {bb_min_days_val}× binnen "
                    f"<span style='color:#00C853;'>0.5% van Upper Band</span> "
                    f"(sterk momentum, zeldzaam signaal)"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        sc1, sc2 = st.columns([1, 3])
        with sc1:
            use_large = st.toggle(
                "🌐 Grote pool",
                value=False,
                key="use_large_pool",
                help="Uit = eigen tickers (snel). Aan = 1000+ tickers (langzaam)."
            )

            if st.button("▶ Start Scan", key="btn_start_scan"):
                st.session_state['scan_running'] = True
                try:
                    st.session_state['lm_active'] = False     # auto-refresh uit tijdens de scan
                except Exception:
                    pass
                pool = load_ticker_pool(use_large_pool=use_large)
                st.info(f"🔍 Scan gestart: **{len(pool)} tickers** worden doorlopen...")
                results = run_scanner(st.session_state.active_strategy, pool)
                st.session_state.scanner_results = results
                st.session_state['scan_running'] = False

        with sc2:
            pool_n = st.session_state.get('total_ticker_count', '?')
            st.markdown(
                f"<div style='background:#13171C;border:1px solid #2B3139;border-radius:6px;"
                f"padding:10px 14px;font-family:monospace;font-size:0.78rem;'>"
                f"<span style='color:#848E9C;'>Pool: </span>"
                f"<b style='color:#F0B90B;font-size:1rem;'>{pool_n}</b>"
                f"<span style='color:#848E9C;'> tickers</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        if not st.session_state.scanner_results.empty:
            df_scan = st.session_state.scanner_results

            approved = len(df_scan[df_scan['MTF Status'] == 'APPROVED']) if 'MTF Status' in df_scan.columns else 0
            watch    = len(df_scan[df_scan['MTF Status'] == 'WATCH'])    if 'MTF Status' in df_scan.columns else 0
            rejected = len(df_scan[df_scan['MTF Status'] == 'REJECTED']) if 'MTF Status' in df_scan.columns else 0
            pool_n   = st.session_state.get('scan_pool_size', '?')

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("🔍 Pool Gescand", pool_n)
            sm2.metric("🟢 APPROVED",     approved)
            sm3.metric("🟡 WATCH",        watch)
            sm4.metric("🔴 REJECTED",     rejected)
            sm5.metric("📊 Totaal Hits",  len(df_scan))

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"🎯 Gevonden Opportuniteiten ({len(df_scan)} tickers goedgekeurd)")

            scan_mobile = st.session_state.get('mobile_view', False)
            if scan_mobile:
                scan_mobile_cols = ['Ticker', 'Koers', 'RSI (14D)', 'MTF Status', 'R:R']
                scan_mobile_cols = [c for c in scan_mobile_cols if c in df_scan.columns]
                st.dataframe(style_scanner_df(df_scan[scan_mobile_cols]), width='stretch')
            else:
                st.dataframe(style_scanner_df(df_scan), width='stretch')

            st.markdown("---")
            all_tickers_scan = df_scan['Ticker'].tolist()
            if all_tickers_scan:
                dd_sc1, dd_sc2, dd_sc3 = st.columns([3, 1, 1])
                with dd_sc1:
                    selected_scan = st.selectbox(
                        "🔭 Selecteer voor Deep Dive / Live Monitor →",
                        options=all_tickers_scan,
                        key="scan_deep_dive_select"
                    )
                with dd_sc2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("📈 Open Deep Dive", key="btn_scan_to_deep_dive"):
                        st.session_state.current_alpha = selected_scan
                        st.success(f"✅ {selected_scan} geladen → ga naar '📈 Deep Dive & Grafiek'")
                with dd_sc3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("📌 Pin naar Live Monitor", key="btn_pin_from_scan"):
                        if selected_scan not in st.session_state['pinned_tickers']:
                            st.session_state['pinned_tickers'].append(selected_scan)
                            st.success(f"📌 {selected_scan} vastgezet in Live Monitor!")
                        else:
                            st.info(f"{selected_scan} staat al in de Live Monitor.")

            # Snel alle hits pinnen
            if all_tickers_scan:
                with st.expander("📌 Pin meerdere tickers naar Live Monitor", expanded=False):
                    pin_options = [t for t in all_tickers_scan
                                   if t not in st.session_state['pinned_tickers']]
                    if pin_options:
                        selected_pins = st.multiselect(
                            "Selecteer tickers om vast te zetten:",
                            options=pin_options,
                            key="multipin_select"
                        )
                        if st.button("📌 Pin geselecteerde tickers", key="btn_multipin"):
                            for t in selected_pins:
                                if t not in st.session_state['pinned_tickers']:
                                    st.session_state['pinned_tickers'].append(t)
                            if selected_pins:
                                st.success(f"✅ {len(selected_pins)} tickers vastgezet!")
                                st.rerun()
                    else:
                        st.info("Alle hits zijn al vastgezet in de Live Monitor.")
        else:
            st.info("Klik op '▶ Start Scan' om te beginnen.")
    else:
        st.info("👆 Selecteer een strategie hierboven om te beginnen met scannen.")



# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: LIVE MONITOR — Real-time tracking van vastgepinde tickers
# ═════════════════════════════════════════════════════════════════════════════
def refresh_live_data() -> None:
    """Wis alleen de snelle caches (live prijs + intraday); de hoofdtabel blijft ongemoeid."""
    for fn in (fetch_live_price, fetch_intraday_patterns, compute_15m_mean_reversion):
        try:
            fn.clear()
        except Exception:
            pass


def _live_view() -> None:
    """
    Live Monitor-weergave. Draait als st.fragment: bij auto-refresh wordt ALLEEN dit blok
    opnieuw uitgevoerd — de rest van de app (en de rest van de pagina) blijft staan.
    """
    pinned = tuple(st.session_state.get('pinned_tickers', []))
    if not pinned:
        st.info("📌 Geen tickers vastgezet. Pin tickers via de Scanner of voeg ze hierboven toe.")
        return

    lm_df = add_intraday_columns(build_main_table(pinned))
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:.75rem;color:#848E9C;margin:4px 0 10px;'>"
        f"Bijgewerkt <b style='color:#F0B90B;'>{_now_nl():%H:%M:%S}</b> · {len(pinned)} tickers</div>",
        unsafe_allow_html=True)

    if st.session_state.get('mobile_view', False):
        st.markdown(render_ticker_cards(lm_df), unsafe_allow_html=True)
    else:
        cols = ['Ticker', 'Koers', 'Dag %', 'RSI (14D)', 'Support (30D)', 'Weerstand (30D)', 'Afwijking %',
                '15M MR Status', '15M [0]', '5M [0]', 'Koers Status / Fase', 'Actie',
                '_rsi_float', '_dev_float', '_ext_chg', '_day_chg']
        st.dataframe(style_main_table(lm_df[[c for c in cols if c in lm_df.columns]]),
                     width='stretch', hide_index=True, height=min(600, 60 + len(lm_df) * 38))


with main_tabs[3]:
    # ── Titel ─────────────────────────────────────────────────────────────────
    st.markdown("### 📡 Live Monitor")
    st.caption("Vastgepinde tickers · 15M mean reversion · after-hours · "
               "auto-refresh ververst alleen dit blok")

    # ── Besturing ─────────────────────────────────────────────────────────────
    with st.container(key="qe_row_lm1"):
        _l1, _l2 = st.columns(2)
        with _l1:
            refresh_interval = st.selectbox(
                "Interval",
                options=[30, 60, 120, 300],
                format_func=lambda x: f"Elke {x}s" if x < 60 else f"Elke {x//60} min",
                index=1,
                key="lm_refresh_interval",
            )
        with _l2:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            monitor_active = st.toggle(
                "▶ Auto-refresh", value=False, key="lm_active",
                help="Ververst alleen de Live Monitor. Zet dit UIT voordat je een Scanner-scan "
                     "start: een ververs-actie tijdens een lopende scan kan die onderbreken.")

    _scan_in_progress = st.session_state.get('scan_running', False)
    if _scan_in_progress:
        monitor_active = False
        st.warning("⛔ Auto-refresh pauzeert zolang er een scan loopt.")

    if st.button("🔄 Nu vernieuwen", key="btn_lm_manual_refresh"):
        refresh_live_data()

    # ── Ticker beheer ─────────────────────────────────────────────────────────
    with st.expander("📌 Tickers vastzetten / losmaken", expanded=not st.session_state['pinned_tickers']):
        with st.container(key="qe_row_lm2"):
            _a1, _a2 = st.columns([3, 2])
            with _a1:
                manual_pin = st.text_input("Ticker", placeholder="bijv. NVDA, AMD, TSLA…",
                                           key="lm_manual_pin_input", label_visibility="collapsed")
            with _a2:
                if st.button("📌 Vastzetten", key="btn_lm_add"):
                    t = manual_pin.strip().upper()
                    if t and t not in st.session_state['pinned_tickers']:
                        st.session_state['pinned_tickers'].append(t)
                        st.rerun()
        if st.session_state['pinned_tickers']:
            to_remove = st.selectbox("Ticker losmaken",
                                     options=["— Selecteer —"] + list(st.session_state['pinned_tickers']),
                                     key="lm_remove_select")
            _b1, _b2 = st.columns(2)
            with _b1:
                if st.button("Losmaken", key="btn_lm_remove"):
                    if to_remove != "— Selecteer —":
                        st.session_state['pinned_tickers'].remove(to_remove)
                        st.rerun()
            with _b2:
                if st.button("🗑 Alles wissen", key="btn_lm_clear"):
                    st.session_state['pinned_tickers'] = []
                    st.rerun()

    # ── Live blok (fragment) ──────────────────────────────────────────────────
    _every = refresh_interval if monitor_active else None
    if hasattr(st, "fragment"):
        st.fragment(run_every=_every)(_live_view)()
    else:                                    # zeer oude Streamlit: eenmalig renderen
        _live_view()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DEEP DIVE & INTERACTIEVE GRAFIEK
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[2]:

    active_ticker = st.session_state.current_alpha
    st.markdown(f"### 📈 Deep Dive: <span style='color:#F0B90B;'>{active_ticker}</span>", unsafe_allow_html=True)

    dd_c1, dd_c2 = st.columns([3, 1])
    with dd_c1:
        new_active = st.text_input(
            "🔍 Andere ticker laden",
            value=active_ticker,
            placeholder="bijv. NVDA, ASML.AS...",
            key="dd_ticker_input"
        )
    with dd_c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Laden", key="btn_dd_load"):
            if new_active.strip():
                st.session_state.current_alpha = new_active.strip().upper()
                st.rerun()

    active_ticker = st.session_state.current_alpha

    period_map = {"1 Maand": "1mo", "3 Maanden": "3mo", "6 Maanden": "6mo", "1 Jaar": "1y", "2 Jaar": "2y"}
    period_choice = st.radio(
        "📅 Tijdshorizon",
        options=list(period_map.keys()),
        index=3,
        horizontal=True,
        key="dd_period_radio"
    )
    period_str = period_map[period_choice]
    if period_choice in ("1 Maand", "3 Maanden", "6 Maanden"):
        st.caption("💡 SMA 200 vereist minimaal 1 jaar data — kies '1 Jaar' of '2 Jaar' voor de volledige lijn.")

    st.markdown("---")

    _dd_ph = st.empty()
    df_dd, info_dd = fetch_ticker_data(active_ticker, period=period_str)

    if df_dd is None or df_dd.empty:
        st.error(f"❌ Geen data beschikbaar voor **{active_ticker}**. Controleer de ticker-naam.")
    else:

        close_dd = df_dd['Close'].squeeze()

        # Robuuste scalar extractie — beschermt tegen MultiIndex-resten
        _raw_last_dd = close_dd.iloc[-1]
        last_price_dd = float(_raw_last_dd.iloc[0]) if hasattr(_raw_last_dd, 'iloc') else float(_raw_last_dd)
        if math.isnan(last_price_dd):
            _non_nan_dd = close_dd.dropna()
            last_price_dd = float(_non_nan_dd.iloc[-1]) if not _non_nan_dd.empty else 0.0

        rsi_dd = compute_rsi(close_dd, RSI_PERIOD)
        # Robuuste berekening: geef huidige koers mee als fallback-anker
        support_dd, resistance_dd = compute_support_resistance(df_dd, SUPPORT_WINDOW, current_price=last_price_dd)
        pattern_dd = detect_candlestick_pattern(df_dd)
        deviation_dd = round(((last_price_dd - support_dd) / support_dd) * 100, 2) if support_dd > 0 else 0.0

        # Valutasymbool voor Europese tickers
        _currency_dd = info_dd.get('currency', 'USD') if info_dd else 'USD'
        _sym_dd = '€' if _currency_dd == 'EUR' else '£' if _currency_dd == 'GBP' else '$'

        # ── Live prijs incl. aftermarket / pre-market ─────────────────────────
        _live_dd     = fetch_live_price(active_ticker)
        _disp_dd     = _live_dd['price'] if _live_dd['price'] else last_price_dd
        _phase_dd    = _live_dd['market_phase']
        _ext_dd      = _live_dd['extended']
        _chgext_dd   = _live_dd['change_ext']

        # Koers-label voor metric
        if _ext_dd and _phase_dd in ('AFTER-HOURS', 'PRE-MARKET'):
            _badge_dd  = '🌙 After-Hours' if _phase_dd == 'AFTER-HOURS' else '🌅 Pre-Market'
            _delta_dd  = f"{'+' if (_chgext_dd or 0) >= 0 else ''}{_chgext_dd:.2f}% vs slot" if _chgext_dd else None
            _koers_lbl = f"{_sym_dd}{_disp_dd:,.2f}"
        else:
            _badge_dd  = '🟡 Regulier'
            _delta_dd  = None
            _koers_lbl = f"{_sym_dd}{_disp_dd:,.2f}"

        _fmt = lambda v, f: f.format(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else "N/A"
        _kpi_html = render_kpis([
            (f"💰 Koers ({_badge_dd})", _koers_lbl + (f"  ({_delta_dd})" if _delta_dd else ''), ''),
            ("📊 RSI (14D)", f"{int(rsi_dd)}" if not np.isnan(rsi_dd) else "N/A", ''),
            ("🟢 Support 30D", _sym_dd + _fmt(support_dd, "{:,.2f}") if not math.isnan(support_dd) else "N/A", 'g'),
            ("🔴 Weerstand 30D", _sym_dd + _fmt(resistance_dd, "{:,.2f}") if not math.isnan(resistance_dd) else "N/A", 'r'),
            ("📐 Afwijking", f"{deviation_dd:.2f}%" if not math.isnan(deviation_dd) else "N/A", ''),
            ("📡 Marktfase", _phase_dd, ''),
        ]).replace('class="qe-kpis"', 'class="qe-kpis n6"')
        st.markdown(_kpi_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig = build_candlestick_chart(df_dd, active_ticker)

        # ── TradingView-stijl interactie ──────────────────────────────────────
        # Scrollwiel = inzoomen, dubbelklik = reset, crosshair cursor
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode='across+toaxis',
                spikesnap='cursor',
                spikecolor='#848E9C',
                spikethickness=1,
                spikedash='dot',
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across+toaxis',
                spikesnap='cursor',
                spikecolor='#848E9C',
                spikethickness=1,
                spikedash='dot',
                fixedrange=False,   # Y-as ook zoombaar
            ),
            hovermode='x unified',
            dragmode='pan',         # Standaard: slepen = pannen (zoals TradingView)
        )

        # Zelfde crosshair op alle subpanels
        for ax in ['xaxis2','xaxis3','xaxis4','yaxis2','yaxis3','yaxis4']:
            fig.update_layout(**{ax: dict(
                showspikes=True,
                spikemode='across+toaxis',
                spikesnap='cursor',
                spikecolor='#848E9C',
                spikethickness=1,
                spikedash='dot',
                fixedrange=False,
            )})

        _mob_chart = st.session_state.get('mobile_view', False)
        if _mob_chart:
            # Telefoon: één vinger scrolt de pagina i.p.v. de grafiek te verslepen
            fig.update_layout(dragmode=False, height=560, margin=dict(l=6, r=48, t=36, b=6),
                              legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width='stretch', config={
            'scrollZoom':           not _mob_chart,       # Scrollwiel = zoom (zoals TradingView)
            'displayModeBar':       not _mob_chart,
            'modeBarButtonsToRemove': ['toImage', 'lasso2d', 'select2d'],
            'modeBarButtonsToAdd':  [
                'drawline',         # Trendlijn tekenen
                'drawopenpath',     # Vrije lijn
                'drawrect',         # Rechthoek (box selectie)
                'eraseshape',       # Verwijder getekende lijnen
            ],
            'displaylogo':          False,
            'doubleClick':          'reset',    # Dubbelklik = reset zoom (zoals TradingView)
            'showTips':             True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'QuantEdge_{active_ticker}',
                'scale': 2,
            },
        })

        st.markdown("---")
        st.markdown("#### 🔬 Multi-Timeframe Validatie")
        _mtf_ph = st.empty()
        mtf_result = compute_multi_timeframe_check(active_ticker)

        # ── Pre-bereken R:R zodat veto de MTF-badge kan overschrijven ─────────
        _safe_sup_dd  = support_dd    if (not math.isnan(support_dd)    and support_dd > 0)    else last_price_dd * (1 - TRADE_FALLBACK_SL)
        _safe_res_dd  = resistance_dd if (not math.isnan(resistance_dd) and resistance_dd > 0) else last_price_dd * (1 + TRADE_FALLBACK_RES)
        _sl_dd        = _safe_sup_dd * (1 - TRADE_RISK_PCT)
        _risk_dd      = last_price_dd - _sl_dd
        _raw_tp3_dd   = _safe_res_dd * (1 - TRADE_TP3_BUFFER)

        # Als TP3 onder de koers ligt → gebruik 52W high als hogere target
        if _raw_tp3_dd <= last_price_dd:
            _52w_dd   = info_dd.get('fiftyTwoWeekHigh', 0) if info_dd else 0
            _raw_tp3_dd = round(_52w_dd * (1 - TRADE_TP3_BUFFER), 2) \
                          if _52w_dd and _52w_dd > last_price_dd else last_price_dd * (1 + TRADE_FALLBACK_RES)

        _reward_dd  = _raw_tp3_dd - last_price_dd
        _prerr_dd   = round(_reward_dd / _risk_dd, 2) if (_risk_dd > 0 and _reward_dd > 0) else 0.0

        # Pas veto toe — overschrijft status als R:R onvoldoende is
        mtf_result = apply_rr_veto(
            mtf_result        = mtf_result,
            current_price     = last_price_dd,
            resistance_target = _raw_tp3_dd,
            actual_rr         = _prerr_dd,
        )

        status_color = {'APPROVED': '#00C853', 'WATCH': '#F0B90B', 'REJECTED': '#F6465D'}.get(mtf_result['status'], '#848E9C')
        trend_color  = '#00C853' if mtf_result['weekly_trend'] == 'BULLISH' else '#F6465D' if mtf_result['weekly_trend'] == 'BEARISH' else '#848E9C'
        vol_color    = '#00C853' if mtf_result['hourly_volume'] == 'RISING' else '#848E9C'

        mtf_cols = st.columns(3)

        # Badge 1: Overall status — met veto-reden als die getriggerd is
        _veto_sub = (f"<div style='color:#F6465D;font-size:0.65rem;margin-top:4px;"
                     f"line-height:1.3;'>{mtf_result['rr_veto_reason']}</div>"
                     if mtf_result.get('rr_veto') else "")
        mtf_cols[0].markdown(f"""
        <div style="background:#13171C;border:1px solid #2B3139;border-top:3px solid {status_color};padding:12px 16px;border-radius:6px;text-align:center;">
          <div style="color:#848E9C;font-size:0.75rem;font-family:monospace;">OVERALL STATUS</div>
          <div style="color:{status_color};font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono';">{mtf_result['status']}</div>
          {_veto_sub}
        </div>""", unsafe_allow_html=True)

        mtf_cols[1].markdown(f"""
        <div style="background:#13171C;border:1px solid #2B3139;border-top:3px solid {trend_color};padding:12px 16px;border-radius:6px;text-align:center;">
          <div style="color:#848E9C;font-size:0.75rem;font-family:monospace;">1W MACRO TREND</div>
          <div style="color:{trend_color};font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono';">{mtf_result['weekly_trend']}</div>
        </div>""", unsafe_allow_html=True)

        mtf_cols[2].markdown(f"""
        <div style="background:#13171C;border:1px solid #2B3139;border-top:3px solid {vol_color};padding:12px 16px;border-radius:6px;text-align:center;">
          <div style="color:#848E9C;font-size:0.75rem;font-family:monospace;">1H INTRADAY VOLUME</div>
          <div style="color:{vol_color};font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono';">{mtf_result['hourly_volume']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analysis_tabs = st.tabs(["🎯 Strijdplan", "🧱 Fundamentele Waardering", "📰 Sentiment & Nieuws"])

        # ── STRIJDPLAN ────────────────────────────────────────────────────────
        with analysis_tabs[0]:
            st.markdown("#### 🎯 Trading Strijdplan")


            risk_pct     = TRADE_RISK_PCT
            reward_ratio = TRADE_REWARD_RATIO

            # NaN-safe support/resistance — valt terug op koers-gebaseerde waardes
            safe_support    = support_dd    if (not math.isnan(support_dd)    and support_dd > 0) \
                              else last_price_dd * (1 - TRADE_FALLBACK_SL)
            safe_resistance = resistance_dd if (not math.isnan(resistance_dd) and resistance_dd > 0) \
                              else last_price_dd * (1 + TRADE_FALLBACK_RES)

            instap    = round(last_price_dd, 2)
            stop_loss = round(safe_support * (1 - risk_pct), 2)

            # ── Expliciete afstanden (nooit negatief gebruiken) ───────────────
            risk_distance   = instap - stop_loss          # altijd > 0 als SL < instap
            if risk_distance <= 0:
                # SL ligt boven of gelijk aan instap — gebruik fallback buffer
                stop_loss     = round(instap * (1 - TRADE_FALLBACK_SL), 2)
                risk_distance = instap - stop_loss

            sl_distance = round((risk_distance / instap) * 100, 2) if instap > 0 else 0.0

            # TP1 en TP2 op basis van doel R:R
            take_profit_1 = round(instap + risk_distance * reward_ratio * TRADE_TP1_RATIO, 2)
            take_profit_2 = round(instap + risk_distance * reward_ratio, 2)

            # TP3 = resistance min buffer — maar NOOIT onder instap
            _raw_tp3      = round(safe_resistance * (1 - TRADE_TP3_BUFFER), 2)
            if _raw_tp3 <= instap:
                # Aandeel staat al boven/op resistance → gebruik TP2 als TP3
                # en zoek een hogere resistance (52W high als proxy)
                _52w_high = info_dd.get('fiftyTwoWeekHigh', 0) if info_dd else 0
                if _52w_high and _52w_high > instap:
                    _raw_tp3 = round(_52w_high * (1 - TRADE_TP3_BUFFER), 2)
                else:
                    _raw_tp3 = take_profit_2  # ultieme fallback = doel TP2
            take_profit_3 = _raw_tp3

            # ── Werkelijke R:R op basis van TP3 (markt-target) ───────────────
            reward_distance = take_profit_3 - instap      # altijd > 0 na bovenstaande guard
            if risk_distance <= 0:
                actual_rr = 0.0
            elif reward_distance <= 0:
                actual_rr = 0.0
            else:
                actual_rr = round(reward_distance / risk_distance, 2)

            # R:R kleur: groen ≥ 2:1, geel 1-2:1, rood < 1:1
            rr_color = '#00C853' if actual_rr >= 2.0 else '#F0B90B' if actual_rr >= 1.0 else '#F6465D'

            sp_c1, sp_c2 = st.columns(2)

            with sp_c1:
                st.markdown(f"""
                <div style="background:#0D2818;border:1px solid #00843A;border-radius:8px;padding:20px;margin-bottom:12px;">
                  <div style="color:#848E9C;font-size:0.75rem;margin-bottom:4px;">🟢 INSTAPZONE</div>
                  <div style="color:#00C853;font-size:1.6rem;font-weight:700;font-family:monospace;">{_sym_dd}{instap:,.2f}</div>
                  <div style="color:#848E9C;font-size:0.8rem;margin-top:8px;">Huidig niveau / Marktprijs</div>
                </div>
                <div style="background:#1A0008;border:1px solid #A32040;border-radius:8px;padding:20px;">
                  <div style="color:#848E9C;font-size:0.75rem;margin-bottom:4px;">🔴 STOP-LOSS</div>
                  <div style="color:#F6465D;font-size:1.6rem;font-weight:700;font-family:monospace;">{_sym_dd}{stop_loss:,.2f}</div>
                  <div style="color:#848E9C;font-size:0.8rem;margin-top:8px;">
                    Risico: {sl_distance:.2f}% · {_sym_dd}{risk_distance:,.2f} per aandeel
                  </div>
                </div>""", unsafe_allow_html=True)

            with sp_c2:
                for tp_label, tp_val, tp_note in [
                    ("🎯 TAKE-PROFIT 1 (Partieel 40%)", take_profit_1,
                     f"Doel R:R {round(reward_ratio * TRADE_TP1_RATIO, 2)}:1"),
                    ("🎯 TAKE-PROFIT 2 (Partieel 40%)", take_profit_2,
                     f"Doel R:R {reward_ratio}:1"),
                    ("🎯 TAKE-PROFIT 3 (Resistance Target)", take_profit_3,
                     f"Werkelijke R:R <b style='color:{rr_color};'>{actual_rr}:1</b>")]:
                    pct_gain = round(((tp_val - instap) / instap) * 100, 2)
                    gain_pts = round(tp_val - instap, 2)
                    st.markdown(f"""
                    <div style="background:#1A1500;border:1px solid #A07C08;border-radius:8px;padding:14px;margin-bottom:8px;">
                      <div style="color:#848E9C;font-size:0.75rem;">{tp_label}</div>
                      <div style="color:#F0B90B;font-size:1.3rem;font-weight:700;font-family:monospace;">{_sym_dd}{tp_val:,.2f}</div>
                      <div style="color:#848E9C;font-size:0.75rem;">
                        +{pct_gain:.1f}% &nbsp;·&nbsp; +{_sym_dd}{gain_pts:,.2f} &nbsp;·&nbsp; {tp_note}
                      </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            _rsi_display = int(rsi_dd) if not math.isnan(rsi_dd) else 'N/A'
            _dev_display = f"{deviation_dd:.2f}%" if not math.isnan(deviation_dd) else 'N/A'
            _rr_label    = f"{actual_rr}:1" if actual_rr > 0 else "N/A (koers boven resistance)"

            st.markdown(f"""
            <div style="background:#13171C;border:1px solid #2B3139;border-radius:8px;padding:16px;">
              <div style="color:#F0B90B;font-size:0.9rem;font-weight:600;margin-bottom:12px;">📋 Samenvatting</div>
              <table style="width:100%;font-family:monospace;font-size:0.85rem;color:#E8ECEF;">
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Ticker</td>
                    <td><b>{active_ticker}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Koers</td>
                    <td><b>{_sym_dd}{instap:,.2f}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">RSI (14D)</td>
                    <td><b>{_rsi_display}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Patroon</td>
                    <td><b>{pattern_dd}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Fase</td>
                    <td><b>{determine_phase(rsi_dd, deviation_dd if not math.isnan(deviation_dd) else 0.0, pattern_dd)}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Afwijking Support</td>
                    <td><b>{_dev_display}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Stop-Loss</td>
                    <td><b style="color:#F6465D;">{_sym_dd}{stop_loss:,.2f} (−{sl_distance:.2f}%)</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Risico per aandeel</td>
                    <td><b style="color:#F6465D;">{_sym_dd}{risk_distance:,.2f}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Reward per aandeel (TP3)</td>
                    <td><b style="color:#00C853;">{_sym_dd}{round(reward_distance,2):,.2f}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">MTF Status</td>
                    <td><b style="color:{status_color};">{mtf_result['status']}</b></td></tr>
                <tr><td style="color:#848E9C;padding:4px 12px 4px 0;">Reward Ratio (vs Resistance)</td>
                    <td><b style="color:{rr_color};">{_rr_label}</b></td></tr>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.warning("⚠️ Dit strijdplan is louter informatief en geen financieel advies. Alle handelsbeslissingen zijn op eigen risico.")

        # ── FUNDAMENTALS ──────────────────────────────────────────────────────
        with analysis_tabs[1]:
            st.markdown("#### 🧱 Fundamentele Waardering")

            if info_dd:
                company_name = info_dd.get('longName', active_ticker)
                sector       = info_dd.get('sector', 'N/A')
                industry     = info_dd.get('industry', 'N/A')
                country      = info_dd.get('country', 'N/A')
                summary      = info_dd.get('longBusinessSummary', '')

                st.markdown(f"""
                <div style="background:#13171C;border:1px solid #2B3139;border-radius:8px;padding:16px;margin-bottom:16px;">
                  <div style="color:#F0B90B;font-size:1.1rem;font-weight:700;">{company_name}</div>
                  <div style="color:#848E9C;font-size:0.8rem;margin-top:4px;">{sector} · {industry} · {country}</div>
                  <div style="color:#E8ECEF;font-size:0.8rem;margin-top:8px;line-height:1.5;">{summary[:400] + '...' if len(summary) > 400 else summary}</div>
                </div>""", unsafe_allow_html=True)

                fundamentals = {
                    "P/E Ratio (TTM)":         info_dd.get('trailingPE', 'N/A'),
                    "Forward P/E":             info_dd.get('forwardPE', 'N/A'),
                    "P/S Ratio":               info_dd.get('priceToSalesTrailing12Months', 'N/A'),
                    "P/B Ratio":               info_dd.get('priceToBook', 'N/A'),
                    "EV/EBITDA":               info_dd.get('enterpriseToEbitda', 'N/A'),
                    "Market Cap":              info_dd.get('marketCap', 'N/A'),
                    "52W High":                info_dd.get('fiftyTwoWeekHigh', 'N/A'),
                    "52W Low":                 info_dd.get('fiftyTwoWeekLow', 'N/A'),
                    "Omzet (TTM)":             info_dd.get('totalRevenue', 'N/A'),
                    "Brutomarge %":            info_dd.get('grossMargins', 'N/A'),
                    "Operationele Marge %":    info_dd.get('operatingMargins', 'N/A'),
                    "Nettomarge %":            info_dd.get('profitMargins', 'N/A'),
                    "ROE":                     info_dd.get('returnOnEquity', 'N/A'),
                    "ROA":                     info_dd.get('returnOnAssets', 'N/A'),
                    "Schuld/Eigen Vermogen":   info_dd.get('debtToEquity', 'N/A'),
                    "YoY Omzetgroei":          info_dd.get('revenueGrowth', 'N/A'),
                    "YoY EPS Groei":           info_dd.get('earningsGrowth', 'N/A'),
                    "Dividend Yield":          info_dd.get('dividendYield', 'N/A'),
                    "Payout Ratio":            info_dd.get('payoutRatio', 'N/A'),
                    "Beta":                    info_dd.get('beta', 'N/A'),
                    "Float Shares":            info_dd.get('floatShares', 'N/A'),
                    "Short Float %":           info_dd.get('shortPercentOfFloat', 'N/A'),
                }

                def fmt_val(v, key):
                    if v == 'N/A' or v is None:
                        return '<span style="color:#848E9C;">N/A</span>'
                    if isinstance(v, float):
                        k = key.lower()
                        if any(w in k for w in ['marge','groei','yield','payout','roe','roa','float %']):
                            pct = round(v * 100, 1)
                            color = '#00C853' if pct > 0 else '#F6465D'
                            return f'<span style="color:{color};">{pct}%</span>'
                        if any(w in k for w in ['cap','omzet','shares']):
                            if v > 1e12:   return f'${v/1e12:.2f}T'
                            elif v > 1e9:  return f'${v/1e9:.2f}B'
                            elif v > 1e6:  return f'${v/1e6:.2f}M'
                        return f'{round(v, 2)}'
                    return str(v)

                fund_items = list(fundamentals.items())
                for i in range(0, len(fund_items), 3):
                    row_cols = st.columns(3)
                    for j, (key, val) in enumerate(fund_items[i:i+3]):
                        with row_cols[j]:
                            st.markdown(f"""
                            <div style="background:#1A1F27;border:1px solid #2B3139;border-radius:6px;padding:12px;margin-bottom:8px;">
                              <div style="color:#848E9C;font-size:0.7rem;font-family:monospace;">{key.upper()}</div>
                              <div style="font-size:1rem;font-weight:600;font-family:monospace;margin-top:4px;">{fmt_val(val, key)}</div>
                            </div>""", unsafe_allow_html=True)
            else:
                st.warning(f"⚠ Geen fundamentele data beschikbaar voor **{active_ticker}**.")

        # ── SENTIMENT & NIEUWS ────────────────────────────────────────────────
        with analysis_tabs[2]:
            st.markdown("#### 📰 Sentiment & Live Nieuws")
            st.markdown(f"<small style='color:#848E9C;'>Yahoo Finance RSS feed voor {active_ticker}</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            pos_words = ['surge','gain','rise','rally','growth','beat','upgrade','buy','strong','profit','record','jump','boost']
            neg_words = ['fall','drop','loss','decline','miss','downgrade','sell','cut','risk','warn','crash','sink','plunge']

            _news_ph = st.empty()
            news_items = fetch_yahoo_rss(active_ticker)

            if news_items:
                for item in news_items:
                    title = item.get('title', '')
                    link  = item.get('link', '')
                    date  = item.get('date', '')
                    tl    = title.lower()
                    if any(w in tl for w in pos_words):
                        sentiment, sent_color = "🟢 Positief", "#00C853"
                    elif any(w in tl for w in neg_words):
                        sentiment, sent_color = "🔴 Negatief", "#F6465D"
                    else:
                        sentiment, sent_color = "🟡 Neutraal", "#848E9C"

                    st.markdown(f"""
                    <div style="background:#13171C;border:1px solid #2B3139;border-radius:6px;padding:12px 16px;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px;">
                      <div style="min-width:90px;font-size:0.7rem;font-family:monospace;color:{sent_color};padding-top:2px;">{sentiment}</div>
                      <div style="flex:1;">
                        <a href="{link}" target="_blank" style="color:#E8ECEF;text-decoration:none;font-size:0.875rem;line-height:1.4;">{title}</a>
                        <div style="color:#848E9C;font-size:0.7rem;margin-top:4px;">{date}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                pos_count = sum(1 for i in news_items if any(w in i.get('title','').lower() for w in pos_words))
                neg_count = sum(1 for i in news_items if any(w in i.get('title','').lower() for w in neg_words))
                neu_count = len(news_items) - pos_count - neg_count
                st.markdown("---")
                st.markdown("**Sentiment Samenvatting**")
                scols = st.columns(3)
                scols[0].metric("🟢 Positief", pos_count)
                scols[1].metric("🟡 Neutraal", neu_count)
                scols[2].metric("🔴 Negatief", neg_count)
            else:
                try:
                    t_obj  = yf.Ticker(active_ticker)
                    yf_news = t_obj.news
                    if yf_news:
                        st.markdown("**Nieuws via yfinance:**")
                        for item in yf_news[:10]:
                            title = item.get('title', '')
                            link  = item.get('link', '')
                            pub   = item.get('providerPublishTime', 0)
                            pub_s = datetime.fromtimestamp(pub).strftime('%d %b %Y %H:%M') if pub else ''
                            st.markdown(f"""
                            <div style="background:#13171C;border:1px solid #2B3139;border-radius:6px;padding:10px 14px;margin-bottom:6px;">
                              <a href="{link}" target="_blank" style="color:#E8ECEF;text-decoration:none;font-size:0.85rem;">{title}</a>
                              <div style="color:#848E9C;font-size:0.7rem;margin-top:4px;">{pub_s}</div>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info(f"📭 Geen nieuws gevonden voor {active_ticker}.")
                except Exception:
                    st.info(f"📭 Geen nieuws beschikbaar voor {active_ticker}.")



# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: ALPHA TRACKER
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[4]:
    st.markdown("### 🏆 Alpha Tracker")
    st.caption("Neem regelmatig dezelfde Alpha Scan op en volg de beste uitkomsten. "
               "Zo zie je welke aandelen steeds terugkomen — en welke uit de bodem omhoog komen.")

    if 'alpha_history' not in st.session_state:
        st.session_state['alpha_history'] = alpha_load_history()
    hist  = st.session_state['alpha_history']
    snaps = hist.get('snapshots', [])

    # ── Snapshot nemen ────────────────────────────────────────────────────────
    with st.container(key="qe_row_alpha1"):
        _al1, _al2 = st.columns(2)
        with _al1:
            alpha_keep = st.selectbox("Bewaar top", [5, 10, 15, 25], index=1, key="alpha_keep_top",
                                      format_func=lambda n: f"Top {n} per scan")
        with _al2:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            alpha_large = st.toggle("🌐 Grote pool", value=False, key="alpha_use_large",
                                    help="Uit = jouw eigen tickers (snel). Aan = 1000+ tickers (langzaam).")

    if snaps:
        st.caption(f"{len(snaps)} snapshot(s) · laatste: {snaps[-1]['ts'].replace('T', ' ')} "
                   f"({snaps[-1].get('hits', '?')} hits van {snaps[-1].get('pool', '?')} tickers)")

    if st.button("📸 Alpha Scan draaien & snapshot opslaan", key="btn_alpha_snapshot"):
        st.session_state['scan_running'] = True
        try:
            st.session_state['lm_active'] = False
        except Exception:
            pass
        try:
            _pool = load_ticker_pool(use_large_pool=alpha_large)
            st.info(f"🔍 Alpha Scan gestart op **{len(_pool)} tickers**…")
            _res  = run_scanner("alpha_scanner", _pool)
            _snap = alpha_make_snapshot(_res, len(_pool), alpha_keep)
            if not _snap['top']:
                st.warning("Geen Alpha-hits gevonden; er is geen snapshot opgeslagen.")
            else:
                hist['snapshots'].append(_snap)
                _tracked = sorted({it['t'] for sn in hist['snapshots'] for it in sn['top']})
                _cur = alpha_fetch_current(_tracked)
                _df0 = alpha_evaluate(hist, _cur)
                _snap['states'] = dict(zip(_df0['Ticker'], _df0['_status']))
                st.session_state.pop('_alpha_eval', None)
                _ok, _msg = alpha_save_history(hist)
                if _ok:
                    st.success(f"✅ Snapshot opgeslagen ({len(_snap['top'])} tickers in de top).")
                else:
                    st.warning(f"Snapshot staat in deze sessie, maar opslaan op GitHub mislukte: {_msg}")
        finally:
            st.session_state['scan_running'] = False
        snaps = hist.get('snapshots', [])

    # ── Overzicht van gevolgde tickers ────────────────────────────────────────
    if not snaps:
        st.info("Nog geen snapshots. Tik hierboven op **Alpha Scan draaien** om te beginnen. "
                "Herhaal dit bijvoorbeeld elke dag of week.")
    else:
        _tracked = sorted({it['t'] for sn in snaps for it in sn['top']})
        _ck = (len(snaps), tuple(_tracked))
        _cache = st.session_state.get('_alpha_eval')
        if not _cache or _cache['key'] != _ck or time.time() - _cache['ts'] > MAIN_TABLE_REUSE_SECS:
            with st.spinner("Actuele koersen van gevolgde tickers ophalen…"):
                _df_all = alpha_evaluate(hist, alpha_fetch_current(_tracked))
            st.session_state['_alpha_eval'] = {'key': _ck, 'ts': time.time(), 'df': _df_all}
        df_all = st.session_state['_alpha_eval']['df']

        n_up     = int((df_all['_status'] == 'up').sum())
        n_first  = int((df_all['_status'] == 'first').sum())
        n_bottom = int((df_all['_status'] == 'bottom').sum())
        st.markdown(render_kpis([
            ("🚀 Op weg omhoog", n_up, 'g'), ("↗ Eerste herstel", n_first, 'g'),
            ("🔎 Nog in bodem", n_bottom, ''), ("📸 Scans / tickers", f"{len(snaps)} / {len(df_all)}", ''),
        ]), unsafe_allow_html=True)

        # Nieuw sinds de laatste snapshot: wie is er net omhoog gekomen?
        prev_states = snaps[-1].get('states') or {}
        moved = [(t, prev_states.get(t)) for t, k in zip(df_all['Ticker'], df_all['_status'])
                 if k in ('up', 'first') and prev_states.get(t) not in (None, k)]
        if moved:
            st.success("🚀 **Net omhoog sinds de laatste scan:** " + ", ".join(
                f"{t} (was {ALPHA_STATUS.get(p, ('?',))[0]})" for t, p in moved))

        ALPHA_FILTERS = ["Alles", "🚀 Omhoog", "🔎 In bodem", "🔴 Gebroken", "🏆 Meerdere keren"]
        try:
            a_flt = st.pills("Filter", ALPHA_FILTERS, default="Alles", key="alpha_filter",
                             label_visibility="collapsed")
        except Exception:
            a_flt = st.radio("Filter", ALPHA_FILTERS, horizontal=True, key="alpha_filter_r",
                             label_visibility="collapsed")
        a_flt = a_flt or "Alles"
        a_sort = st.selectbox("Sorteer", list(ALPHA_SORTS.keys()), key="alpha_sort",
                              label_visibility="collapsed")

        view = df_all
        if a_flt == "🚀 Omhoog":
            view = view[view['_status'].isin(['up', 'first'])]
        elif a_flt == "🔎 In bodem":
            view = view[view['_status'] == 'bottom']
        elif a_flt == "🔴 Gebroken":
            view = view[view['_status'] == 'broken']
        elif a_flt == "🏆 Meerdere keren":
            view = view[view['Keer in top'] >= 2]
        view = alpha_sorted(view, a_sort)

        if st.session_state.get('mobile_view', False):
            st.markdown(render_alpha_cards(view), unsafe_allow_html=True)
        elif view.empty:
            st.info("Geen tickers voor dit filter.")
        else:
            show_cols = ['Ticker', 'Status', 'Keer in top', 'Van snapshots', 'Gem. score', 'Beste score',
                         'Eerste keer', 'Instapkoers', 'Koers nu', 'Rend. sinds 1e keer %',
                         'Rend. sinds laatst %', 'RSI laatst', 'RSI nu', 'Afw. support nu %']
            st.dataframe(view[show_cols], width='stretch', hide_index=True,
                         height=min(640, 60 + 36 * len(view)),
                         column_config={
                             'Rend. sinds 1e keer %': st.column_config.NumberColumn(format="%+.1f%%"),
                             'Rend. sinds laatst %':  st.column_config.NumberColumn(format="%+.1f%%"),
                             'Afw. support nu %':     st.column_config.NumberColumn(format="%.1f%%"),
                             'Instapkoers':           st.column_config.NumberColumn(format="%.2f"),
                             'Koers nu':              st.column_config.NumberColumn(format="%.2f"),
                             'RSI laatst':            st.column_config.NumberColumn(format="%.0f"),
                             'RSI nu':                st.column_config.NumberColumn(format="%.0f"),
                         })

        # ── Acties op een gevolgde ticker ─────────────────────────────────────
        st.markdown("---")
        a_pick = st.selectbox("🔭 Ticker", df_all['Ticker'].tolist(), key="alpha_pick")
        with st.container(key="qe_row_alpha2"):
            _p1, _p2 = st.columns(2)
            with _p1:
                if st.button("📈 Deep Dive", key="btn_alpha_dive"):
                    st.session_state.current_alpha = a_pick
                    st.success(f"✅ {a_pick} geladen → ga naar '📈 Deep Dive & Grafiek'")
            with _p2:
                if st.button("📌 Live Monitor", key="btn_alpha_pin"):
                    if a_pick not in st.session_state['pinned_tickers']:
                        st.session_state['pinned_tickers'].append(a_pick)
                    st.success(f"📌 {a_pick} vastgezet in Live Monitor.")

        # ── Historie & uitleg ─────────────────────────────────────────────────
        with st.expander("🗂 Snapshot-historie", expanded=False):
            for sn in reversed(snaps[-15:]):
                tops = ", ".join(f"{it['t']} ({it['score']:.0f})" for it in sn['top'][:6])
                st.markdown(f"**{sn['ts'].replace('T', ' ')}** · {sn.get('hits', '?')} hits  \n"
                            f"<span style='color:#848E9C;font-size:.8rem;'>{_esc(tops)}</span>",
                            unsafe_allow_html=True)
            if st.button("↩ Laatste snapshot verwijderen", key="btn_alpha_undo"):
                hist['snapshots'].pop()
                st.session_state.pop('_alpha_eval', None)
                _ok, _msg = alpha_save_history(hist, "QuantEdge: alpha-snapshot verwijderd")
                if not _ok:
                    st.warning(f"Verwijderd in deze sessie; opslaan mislukte: {_msg}")
                st.rerun()

        with st.expander("ℹ️ Hoe werkt dit?", expanded=False):
            st.markdown(
                f"""
**Score (0–100)** per Alpha-hit: 30 punten voor dichtbij support · 25 voor diep oversold RSI · 25 voor het weekly-trend/MTF-oordeel
(APPROVED 25, WATCH 10) · 10 voor Risk:Reward · 10 voor volume.

**Status** t.o.v. de laatste keer dat de ticker in de top stond:
- 🚀 **Op weg omhoog** — uit de bodemzone (RSI > {SCAN_ALPHA_RSI_MAX} of > {SCAN_ALPHA_DEV_MAX}% boven support), koers ≥ {ALPHA_UP_MIN_RET:.0f}% hoger en RSI stijgend
- ↗ **Eerste herstel** — koers ≥ {ALPHA_FIRST_MIN_RET:.0f}% hoger en RSI stijgend
- 🔎 **In bodem** — voldoet nog aan de Alpha-zone
- 🔴 **Support gebroken** — onder support of ≥ {abs(ALPHA_BROKEN_RET):.0f}% lager
- ⏳ **Afwachten** — niets van het bovenstaande

Een snapshot draait alleen als je de app opent en op de knop tikt; de app draait niet op de achtergrond.
Geen financieel advies.
""")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2B3139;font-family:monospace;font-size:0.72rem;padding:8px 0;">
  QuantEdge Dashboard · Data: Yahoo Finance (yfinance) · Geen financieel advies · Uitsluitend voor educatieve doeleinden
  · <span style="color:#F0B90B;">BUILD 3.0</span>
</div>
""", unsafe_allow_html=True)
