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
warnings.filterwarnings("ignore")

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
DATA_SCAN_PERIOD      = "6mo" # Periode scanner
DATA_WL_PERIOD        = "1mo" # Periode watchlist
DATA_MTF_WEEKLY       = "1y"  # Periode wekelijkse MTF check
DATA_MTF_HOURLY       = "5d"  # Periode uurlijkse MTF check
DATA_MIN_ROWS_MAIN    = 30    # Min rijen voor hoofdtabel
DATA_MIN_ROWS_SCAN    = 50    # Min rijen voor scanner
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
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  :root {
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
  }

  html, body, [data-testid="stApp"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  h1, h2, h3, h4, h5 { color: var(--gold) !important; font-family: 'JetBrains Mono', monospace; }

  [data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }

  input, textarea, select { background-color: var(--bg-card) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; }
  .stTextInput > div > div { background-color: var(--bg-card) !important; }
  .stSelectbox > div > div { background-color: var(--bg-card) !important; color: var(--text-primary) !important; }

  .stButton > button {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-hover));
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
    color: #000 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(240,185,11,0.3);
  }

  .stTabs [data-baseweb="tab-list"] { background-color: var(--bg-secondary); border-bottom: 2px solid var(--gold); }
  .stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace; }
  .stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important; }

  .stDataFrame { border: 1px solid var(--border) !important; }

  [data-testid="stMetric"] { background-color: var(--bg-card); padding: 12px 16px; border-radius: 6px; border-left: 3px solid var(--gold); }
  [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
  [data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'JetBrains Mono', monospace !important; }

  .streamlit-expanderHeader { background-color: var(--bg-card) !important; color: var(--gold) !important; border: 1px solid var(--border) !important; }
  .streamlit-expanderContent { background-color: var(--bg-secondary) !important; border: 1px solid var(--border) !important; }

  hr { border-color: var(--border) !important; }

  .stAlert { background-color: var(--bg-card) !important; border: 1px solid var(--border) !important; }
  .stRadio > div { background-color: var(--bg-card); padding: 8px 12px; border-radius: 6px; }

  .section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 24px;
  }

  .header-bar {
    background: linear-gradient(90deg, var(--bg-secondary), var(--bg-card));
    border-bottom: 2px solid var(--gold);
    padding: 10px 20px;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
  }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-secondary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%A %d %b %Y  |  %H:%M:%S CET")
st.markdown(f"""
<div class="header-bar">
  ⬛ <span style="color:#F0B90B; font-weight:700;">QUANTEDGE</span> &nbsp;|&nbsp;
  QUANTITATIVE TRADING DASHBOARD &nbsp;|&nbsp;
  <span style="color:#00C853;">{now_str}</span> &nbsp;|&nbsp;
  <span style="color:#848E9C;">LIVE DATA · yfinance · MULTI-TF VALIDATED</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OPSLAG  —  Streamlit Cloud compatibel
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    # Tech, Semis & Software
    "NVDA","ASML.AS","MSFT","AMD","TSLA","ARM","SAP","CSCO","MU",
    "ADYEN.AS","TRMB","COTY","SITE","CMCSA","FMC","COIN","INTU",
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
    "XOM","CVX","COP","EOG","OXY","FANG","DVN","HES","MRO","APA",
    "EQT","CTRA","SHEL","BP","TTE",
    # Traditional Energy — Refining & Infrastructure
    "MPC","VLO","PSX","OKE","KMI","WMB","MPLX",
    # Traditional Energy — Equipment & Services
    "SLB","HAL","BKR",
    # Power Generation & Utilities
    "CEG","VST","NEE","DUK","SO","D","AEP","ED","PCG","SRE",
    "EIX","ETR","FE","AES","XEL",
    # Clean Tech & Alternative Energy
    "GEV","ENPH","SEDG","BEP","RUN","PLUG",
    # Energy ETFs
    "XLE","VDE",
]

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
        return False
    try:
        import base64
        data    = {
            "tickers":   list(st.session_state.main_market_tickers),
            "watchlist": dict(st.session_state.custom_watchlist),
        }
        content = base64.b64encode(
            json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        ).decode("utf-8")
        url  = f"https://api.github.com/repos/{repo}/contents/{file}"
        hdrs = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        # Haal huidige SHA op (nodig voor update)
        sha  = None
        resp = requests.get(url, headers=hdrs, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
        payload = {"message": "QuantEdge: tickers bijgewerkt", "content": content}
        if sha:
            payload["sha"] = sha
        resp = requests.put(url, headers=hdrs, json=payload, timeout=10)
        return resp.status_code in (200, 201)
    except Exception:
        return False

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

def _save_userdata() -> None:
    """Sla op via GitHub (cloud) én lokaal (thuis). Beide tegelijk."""
    _save_to_github()
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
def fetch_ticker_data(ticker: str, period: str = "3mo"):
    """
    Haal OHLCV-data op via yfinance met caching.
    Gebruikt prepost=True zodat extended-hours kaarsen beschikbaar zijn.
    """
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period=period, prepost=True)  # ← aftermarket/premarket inbegrepen

        if df is None or df.empty or len(df) < DATA_MIN_ROWS_FETCH:
            return None, None

        # ── MultiIndex flatten ──────────────────────────────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Zorg dat de verwachte OHLCV-kolommen aanwezig zijn
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(set(df.columns)):
            return None, None

        # Verwijder rijen waar Close NaN is
        df = df.dropna(subset=['Close'])
        if len(df) < DATA_MIN_ROWS_FETCH:
            return None, None

        # Zorg dat elke kolom een 1D-serie is
        for col in list(df.columns):
            if hasattr(df[col], 'squeeze'):
                df[col] = df[col].squeeze()

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
        df_15m = yf.download(ticker, period='2d', interval='15m',
                             progress=False, auto_adjust=True)
        if df_15m is not None and len(df_15m) >= 4:
            result['15M[0]']  = _get_pattern_and_dir(df_15m, '15M[0]')
            result['15M[-1]'] = _get_prev_pattern_and_dir(df_15m)

        # 5M data ophalen (laatste 1 dag)
        df_5m = yf.download(ticker, period='1d', interval='5m',
                            progress=False, auto_adjust=True)
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
        df = yf.download(ticker, period='2d', interval='15m',
                         progress=False, auto_adjust=True)
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


@st.cache_data(ttl=CACHE_PRICE_TTL)
def build_main_table(tickers: tuple) -> pd.DataFrame:
    """Bouw de hoofdmarkttabel op voor een lijst van tickers."""
    import traceback as _tb
    rows = []
    for ticker in tickers:
        try:
            df, info = fetch_ticker_data(ticker, period=DATA_MAIN_PERIOD)
            if df is None or len(df) < DATA_MIN_ROWS_MAIN:
                rows.append({
                    'Ticker': ticker, 'Koers': 'N/A', 'RSI (14D)': 'N/A',
                    'Support (30D)': 'N/A', 'Weerstand (30D)': 'N/A', 'Afwijking %': 'N/A',
                    'Patroon (1D)': 'N/A', '15M [-1]': '–', '15M [0]': '–',
                    '5M [-1]': '–', '5M [0]': '–', '15M MR Status': '–', '_mr15_dir': '',
                    'Koers Status / Fase': '⚠ Geen Data', 'Actie': '⚠ Geen Data'
                })
                continue

            close = df['Close'].squeeze()   # Garandeert 1D Series

            # Scalar extractie — pakt altijd één getal, ook als squeeze een rij retourneert
            raw_last = close.iloc[-1]
            last_price = round(float(raw_last.iloc[0]) if hasattr(raw_last, 'iloc') else float(raw_last), 2)

            if math.isnan(last_price):
                # Probeer de laatste niet-NaN waarde
                non_nan = close.dropna()
                if non_nan.empty:
                    raise ValueError(f"Alle Close-waarden zijn NaN voor {ticker}")
                last_price = round(float(non_nan.iloc[-1]), 2)
            rsi_val = compute_rsi(close, RSI_PERIOD)
            support_30, resistance_30 = compute_support_resistance(df, SUPPORT_WINDOW, current_price=last_price)
            deviation = round(((last_price - support_30) / support_30) * 100, 2) if support_30 > 0 else 0.0
            pattern = detect_candlestick_pattern(df)
            phase = determine_phase(rsi_val, deviation, pattern)
            action = determine_action(rsi_val, deviation, pattern, phase)

            currency = info.get('currency', '')
            symbol = '€' if currency == 'EUR' else '$' if currency == 'USD' else ''

            # ── Live prijs ophalen incl. aftermarket / pre-market ─────────────
            live = fetch_live_price(ticker)
            display_price = live['price'] if live['price'] else last_price
            ext_price     = live['extended']
            market_phase  = live['market_phase']
            change_ext    = live['change_ext']

            # Koers-label: toon extended prijs + badge als beschikbaar
            if ext_price and market_phase in ('AFTER-HOURS', 'PRE-MARKET'):
                chg_str = ''
                if change_ext is not None:
                    arrow = '▲' if change_ext >= 0 else '▼'
                    chg_str = f" {arrow}{abs(change_ext):.2f}%"
                badge = '🌙' if market_phase == 'AFTER-HOURS' else '🌅'
                koers_str = f"{symbol}{display_price:,.2f} {badge}{chg_str}"
            else:
                koers_str = f"{symbol}{display_price:,.2f}"

            safe_deviation = deviation if not math.isnan(deviation) else 0.0
            safe_rsi_float = float(rsi_val) if not np.isnan(rsi_val) else 50.0
            rsi_display = str(int(rsi_val)) if not np.isnan(rsi_val) else 'N/A'

            # ── Intraday patronen (15M en 5M) ────────────────────────────────
            try:
                intra = fetch_intraday_patterns(ticker)
            except Exception:
                intra = {'15M[-1]': '–', '15M[0]': '–', '5M[-1]': '–', '5M[0]': '–'}
            try:
                mr15 = compute_15m_mean_reversion(ticker)
            except Exception:
                mr15 = {'status': '–', 'direction': None}

            rows.append({
                'Ticker': ticker,
                'Koers': koers_str,
                'RSI (14D)': rsi_display,
                'Support (30D)': f"{symbol}{support_30:,.2f}",
                'Weerstand (30D)': f"{symbol}{resistance_30:,.2f}",
                'Afwijking %': f"{safe_deviation:.2f}",
                '_dev_float': safe_deviation,
                '_rsi_float': safe_rsi_float,
                '_ext_chg': change_ext if (ext_price and market_phase in ('AFTER-HOURS', 'PRE-MARKET')) else None,
                'Patroon (1D)': pattern,
                '15M [-1]': intra['15M[-1]'],
                '15M [0]':  intra['15M[0]'],
                '5M [-1]':  intra['5M[-1]'],
                '5M [0]':   intra['5M[0]'],
                '15M MR Status': mr15['status'],
                '_mr15_dir': mr15['direction'] or '',
                'Koers Status / Fase': phase,
                'Actie': action,
            })
        except Exception as e:
            err_detail = str(e)
            rows.append({
                'Ticker': ticker, 'Koers': 'ERR', 'RSI (14D)': 'ERR',
                'Support (30D)': 'ERR', 'Weerstand (30D)': 'ERR', 'Afwijking %': 'ERR',
                '_dev_float': 999.0, '_rsi_float': 50.0, '_ext_chg': None,
                'Patroon (1D)': 'ERR', '15M [-1]': '–', '15M [0]': '–',
                '5M [-1]': '–', '5M [0]': '–', '15M MR Status': '–', '_mr15_dir': '',
                'Koers Status / Fase': f'⚠ {err_detail[:40]}', 'Actie': '⚠ Fout'
            })
    # Zorg dat de volledige DataFrame geen gemengde kolomtypes heeft
    df_result = pd.DataFrame(rows)
    # Forceer alle display-kolommen naar string zodat Arrow nooit crasht
    for _col in ['RSI (14D)', 'Afwijking %', 'Koers', 'Support (30D)', 'Weerstand (30D)',
                 'Patroon (1D)', '15M [-1]', '15M [0]', '5M [-1]', '5M [0]',
                 '15M MR Status', 'Koers Status / Fase', 'Actie']:
        if _col in df_result.columns:
            df_result[_col] = df_result[_col].astype(str)
    # _ext_chg mag None bevatten — convert naar float waarbij None → NaN
    if '_ext_chg' in df_result.columns:
        df_result['_ext_chg'] = pd.to_numeric(df_result['_ext_chg'], errors='coerce')
    return df_result


def style_main_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Pas dark terminal-stijl toe op de hoofdtabel."""
    # Verberg interne float-kolommen
    display_cols = [c for c in df.columns if not c.startswith('_')]
    display_df = df[display_cols].copy()

    def row_style(row):
        base = 'background-color: #13171C; color: #E8ECEF;'
        styles = [base] * len(row)

        # Haal intern opgeslagen floats op via de originele df
        ticker_val = row.get('Ticker', '')
        orig_row = df[df['Ticker'] == ticker_val]
        rsi_v    = float(orig_row['_rsi_float'].iloc[0]) if not orig_row.empty else 50.0
        dev_v    = float(orig_row['_dev_float'].iloc[0]) if not orig_row.empty else 999.0
        ext_chg  = orig_row['_ext_chg'].iloc[0] if not orig_row.empty else None
        # ext_chg kan 'None' als string zijn na astype(str) — zet terug naar None
        try:
            ext_chg_f = float(ext_chg) if ext_chg not in (None, 'None', '', 'nan') else None
        except (ValueError, TypeError):
            ext_chg_f = None

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
            ('padding', '8px 12px'),
        ]},
        {'selector': 'tbody td', 'props': [
            ('font-family', 'JetBrains Mono, monospace'),
            ('font-size', '0.8rem'),
            ('padding', '6px 12px'),
            ('border-bottom', '1px solid #2B3139'),
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#222831 !important'),
        ]},
    ])
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
        'ASML.AS','AD.AS','ADYEN.AS','UNA.AS','HEIA.AS','PHIA.AS','ABN.AS',
        'ING.AS','RAND.AS','WKL.AS','IMCD.AS','BESI.AS','OCI.AS','AKZA.AS',
        'TKWY.AS','SAP.DE','SIE.DE','DTE.DE','VOW3.DE','BMW.DE','ADS.DE',
        'MUV2.DE','ALV.DE','BAS.DE','BAYN.DE','DBK.DE','HEN3.DE','LIN.DE',
        'MRK.DE','RWE.DE','EON.DE','FRE.DE','ZAL.DE','IFX.DE','PAH3.DE',
        'OR.PA','MC.PA','TTE.PA','SAN.PA','BNP.PA','ACA.PA','SGO.PA',
        'KER.PA','RMS.PA','DSY.PA','AIR.PA','DG.PA','ORA.PA','VIE.PA',
        'NESN.SW','ROG.SW','NOVN.SW','UBSG.SW','CSGN.SW','ABBN.SW','ZURN.SW',
        'AZN.L','SHEL.L','HSBA.L','BP.L','GSK.L','RIO.L','ULVR.L',
        'LLOY.L','BATS.L','REL.L','DGE.L','EXPN.L','LSEG.L','CPG.L',
        'NOVO-B.CO','ERICB.ST','VOLV-B.ST','SAND.ST','ATCO-A.ST','SEB-A.ST',
        'RACE.MI','ENI.MI','ENEL.MI','ISP.MI','UCG.MI','STM.MI','PRY.MI',
        'TEF.MC','IBE.MC','ITX.MC','BBVA.MC','SAN.MC','REP.MC',
    ]

    EXTRAS: list = [
        'NVDA','MSFT','AAPL','GOOGL','GOOG','AMZN','META','TSLA','AMD','ARM',
        'INTC','QCOM','TXN','AVGO','MU','MRVL','AMAT','KLAC','LRCX','ASML',
        'MCHP','ADI','NXPI','ON','SWKS','QRVO','MPWR','ENPH','FSLR','SEDG',
        'JPM','BAC','GS','MS','WFC','C','BLK','AXP','V','MA','PYPL','COF',
        'USB','PNC','TFC','SCHW','ICE','CME','SPGI','MCO','FIS','FISV','GPN',
        'JNJ','ABBV','PFE','MRK','BMY','UNH','CVS','AMGN','GILD','REGN',
        'LLY','TMO','DHR','ISRG','SYK','BSX','MDT','ABT','ZTS','VRTX',
        'BIIB','MRNA','BNTX','INCY','ALNY','SGEN','BEAM','NTLA',
        'XOM','CVX','COP','SLB','EOG','PXD','HAL','MPC','PSX','VLO',
        'OXY','DVN','HES','MRO','APA','FANG','BKR','NOV','WHD',
        'GLD','SLV','USO','FCX','NEM','GOLD','WPM','CAT','DE','EMR',
        'NUE','STLD','CLF','AA','CENX','MP','ALB','SQM','LTHM','LAC',
        'WMT','COST','TGT','HD','LOW','MCD','SBUX','NKE','KO','PEP',
        'PG','CL','KMB','GIS','K','MDLZ','PM','MO','WM','RSG',
        'SYY','CAG','CPB','HSY','MKC','CHD','CLX','EL','COTY',
        'DIS','NFLX','CMCSA','T','VZ','TMUS','CHTR','PARA','WBD','FOX',
        'TTWO','EA','ATVI','RBLX','U',
        'O','PLD','SPG','WELL','AMT','CCI','EQIX','DLR','PSA','EXR',
        'AVB','EQR','ESS','UDR','MAA','CPT','NNN','VICI','GLPI',
        'ADP','PAYX','INTU','ORCL','CRM','NOW','SNOW','DDOG','CRWD',
        'ZS','OKTA','PANW','FTNT','NET','CFLT','MDB','ESTC','GTLB',
        'SHOP','HUBS','BILL','PCTY','PAYC','WEX','FOUR',
        'GE','HON','MMM','ITW','ETN','PH','ROK','AME','FTV','GNRC',
        'UPS','FDX','DAL','UAL','AAL','LUV','JBLU','ALK',
        'NSC','CSX','UNP','CP','CNI','WAB','TT','CARR','OTIS',
        'EXAS','NTRA','PACB','ILMN','BIO','TECH','IDXX','PODD','DXCM',
        'BKNG','ABNB','LYFT','UBER','DASH','MTN',
        'GM','F','RIVN','LCID','NIO','LI','XPEV',
        'COIN','MSTR','RIOT','MARA','HUT','CLSK',
        'SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLI','XLY',
        'XLP','XLU','XLB','XLRE','XLC','GDX','GDXJ',
        'TLT','HYG','LQD','EMB','VNQ','ARKK','ARKG','ARKF',
    ]

    # ── LAYER 2 — hardcoded S&P 500 fallback ──────────────────────────────────
    SP500_HARDCODED: list = [
        'A','AAL','AAP','AAPL','ABBV','ABC','ABMD','ABT','ACN','ADBE',
        'ADI','ADM','ADP','ADSK','AEE','AEP','AES','AFL','AIG','AIZ',
        'AJG','AKAM','ALB','ALGN','ALK','ALL','ALLE','AMAT','AMCR','AMD',
        'AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA',
        'APD','APH','APTV','ARE','ATO','ATVI','AVB','AVGO','AVY','AWK',
        'AXP','AZO','BA','BAC','BALL','BAX','BBY','BDX','BEN','BF-B',
        'BIIB','BIO','BK','BKNG','BKR','BLK','BLL','BMY','BR','BRK-B',
        'BSX','BWA','BXP','C','CAG','CAH','CARR','CAT','CB','CBOE',
        'CBRE','CCI','CCL','CDNS','CDW','CE','CEG','CF','CFG','CHD',
        'CHRW','CHTR','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMG',
        'CMI','CMS','CNC','CNP','COF','COO','COP','COST','CPB','CPRT',
        'CRL','CRM','CSCO','CSX','CTAS','CTLT','CTRA','CTSH','CTVA',
        'CVS','CVX','CZR','D','DAL','DD','DE','DFS','DG','DGX','DHI',
        'DHR','DIS','DISH','DLR','DLTR','DOV','DOW','DPZ','DRE','DRI',
        'DTE','DUK','DVA','DVN','DXC','DXCM','EA','EBAY','ECL','ED',
        'EFX','EIX','EL','EMN','EMR','ENPH','EOG','EPAM','EQIX','EQR',
        'EQT','ES','ESS','ETN','ETR','ETSY','EVRG','EW','EXC','EXPD',
        'EXPE','EXR','F','FANG','FAST','FCX','FDS','FDX','FE','FFIV',
        'FIS','FISV','FITB','FLT','FMC','FOX','FOXA','FRC','FRT','FSLR',
        'FTNT','FTV','GD','GE','GILD','GIS','GL','GLW','GM','GNRC',
        'GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN',
        'HCA','HD','HES','HIG','HII','HLT','HOLX','HON','HPE','HPQ',
        'HRL','HSIC','HST','HSY','HUM','HWM','IBM','ICE','IDXX','IEX',
        'IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR',
        'IRM','ISRG','IT','ITW','IVZ','J','JBHT','JCI','JKHY','JNJ',
        'JNPR','JPM','K','KEY','KEYS','KHC','KIM','KLAC','KMB','KMI',
        'KMX','KO','KR','L','LDOS','LEN','LH','LHX','LIN','LKQ','LLY',
        'LMT','LNC','LNT','LOW','LRCX','LUV','LVS','LW','LYB','LYV',
        'MA','MAA','MAR','MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT',
        'MET','META','MGM','MHK','MKC','MKTX','MLM','MMC','MMM','MNST',
        'MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MRO','MS','MSCI',
        'MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ','NEM','NEE',
        'NI','NKE','NOC','NOW','NRG','NSC','NTAP','NTRS','NUE','NVDA',
        'NVR','NWL','NWS','NWSA','NXPI','O','OGN','OKE','OMC','ON',
        'ORCL','ORLY','OXY','PARA','PAYC','PAYX','PCAR','PCG','PEAK',
        'PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PKI',
        'PLD','PM','PNC','PNR','PNW','POOL','PPG','PPL','PRU','PSA',
        'PSX','PTC','PWR','PXD','PYPL','QCOM','QRVO','RCL','RE','REG',
        'REGN','RF','RHI','RJF','RL','RMD','ROK','ROL','ROP','ROST',
        'RSG','RTX','SBAC','SBUX','SCHW','SEE','SHW','SIVB','SJM','SLB',
        'SNA','SNPS','SO','SPG','SPGI','SRE','STE','STT','STX','STZ',
        'SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG','TDY','TECH',
        'TEL','TER','TFC','TFX','TGT','TJX','TMO','TMUS','TPR','TRMB',
        'TROW','TRV','TSCO','TSLA','TSN','TT','TTWO','TXN','TXT','TYL',
        'UA','UAA','UAL','UDR','UHS','ULTA','UNH','UNP','UPS','URI',
        'USB','V','VFC','VICI','VLO','VMC','VNO','VRSK','VRSN','VRTX',
        'VZ','WAB','WAT','WBA','WBD','WEC','WELL','WFC','WHR','WM',
        'WMB','WMT','WRB','WRK','WST','WTW','WY','WYNN','XEL','XOM',
        'XRAY','XYL','YUM','ZBH','ZBRA','ZION','ZTS',
    ]

    # ── LAYER 2 — hardcoded MidCap 400 fallback (~400 tickers) ──────────────
    MIDCAP_HARDCODED: list = [
        # Industrials & Business Services
        'AAN','ABCB','ABR','ACHC','ACI','ACM','ADNT','AEO','AFG','AGCO',
        'AIN','AKR','AL','ALG','ALKS','AM','AMG','AMKR','APLE','APO',
        'ARCO','AROC','ASH','ATR','AVAV','AVT','AX','AYI','AZZ',
        'BCO','BDC','BFH','BGS','BHF','BJ','BKH','BLKB','BMI','BOH',
        'BRC','BRKL','BRX','BSY','BTU','BWXT','CABO','CACI','CAKE','CALX',
        'CARS','CASY','CC','CCOI','CFR','CHCO','CHE','CHFC','CHH','CIEN',
        'CLB','CNA','CNK','CNO','COHU','COLB','COLM','CPF','CPRX','CRC',
        'CROX','CSL','CW','CWEN','CWK','CWST','CWT',
        # D-G
        'DAN','DFIN','DLX','DM','DNB','DOCS','DOOR','DRH','DRQ','DY',
        'EAT','EFC','EGP','EHC','EME','ENVA','EPC','ESE','ESRT','EVH',
        'EXP','EXPI','FAF','FARO','FCN','FG','FHN','FIBK','FIX','FL',
        'FLO','FLS','FNB','FOR','FORM','FR','FRME','FSS','FUL',
        'GBCI','GEF','GFF','GHC','GKOS','GLDD','GMED','GPI','GPRE',
        'GRBK','GRC','GTN','GTLS','GVA',
        # H-L
        'HAE','HALO','HBI','HCC','HCI','HGV','HIBB','HLIT','HLX',
        'HMN','HNI','HOPE','HRI','HSC','HTBK','HTZ','HUBG','HWC',
        'ICLR','IDCC','IIPR','INDB','INGR','IPGP','ITT',
        'JACK','JBL','JOE','KAI','KAMN','KAR','KBH','KBR','KFY',
        'KMT','KMPR','KN','KNX','KNSL','KSS','KTOS',
        'LAUR','LCI','LE','LEA','LGND','LII','LIVN','LKQ',
        'LMAT','LNTH','LPLA','LRN','LSI','LSTR','LUMN',
        # M-N
        'MGEE','MHO','MIR','MKSI','MLCO','MLI','MMSI','MNRO','MODG',
        'MP','MRC','MRCY','MTH','MTN','MUR','NATL','NBR',
        'NEO','NGL','NHC','NHI','NJR','NNN','NSP','NUS','NVT','NYCB',
        # O-R
        'ODP','OFG','OGE','OGS','OI','OII','OLN','OMF','ONTO','ORA',
        'OWL','OXM','PACW','PBH','PCRX','PEB','PFSI','PJT','PLMR',
        'PLUS','PNM','POR','POWL','PRDO','PRG','PRIM','PRVA','PTEN',
        'R','RBCAA','RCII','RDNT','REX','RH','RIG','RLJ','RMBS',
        'RNR','ROCK','RPM','RRC','RRX','RUN','RXO',
        # S
        'SAH','SANM','SBCF','SBRA','SCI','SEIC','SF','SFBS','SIGI',
        'SITC','SITE','SJW','SKT','SLM','SM','SMTC','SNV','SPTN',
        'STE','STEP','STRA','SUM','SWN','SXT','SYBT','SXI',
        # T-Z
        'TCBK','TCBI','TEX','TGI','TILE','TISI','TNET','TPH','TREX',
        'TRNO','TRN','TRMK','TTC','TWI','UBSI','UCBI','UGI','UNF',
        'UNVR','UVV','VBTX','VG','VIRT','VLY','VRTS','VSCO','VSH',
        'WABC','WAFD','WD','WEN','WERN','WFRD','WHD','WK','WLK',
        'WMS','WOR','WRLD','WSM','WSO','WTS','WWD','WWW',
        'XPO','YELP','ZWS',
        # Extra bekende MidCap namen (gevraagd in bugfix)
        'DECK','TOLL','FLEX','RE','ANET','SITE','TREX','MEDP',
        'HTGC','LADR','SMCI','BILL','CELH','HIMS','DUOL','GLBE',
        'EXEL','ITCI','PRCT','QLYS','HALO','CRVL','IIPR','UFPT',
        'PRGS','FCFS','MGNI','GKOS','IRDM','LCII','SAIA','EXPO',
        'MGEE','HWKN','POWI','IPAR','MSEX','SPSC','CSGS','NVCR',
        'AEIS','ICFI','AFTS','HURN','NTCT','PSMT','EPAC','CSWI',
        'VRRM','ACLS','FORM','PTGX','TMDX','PCVX','RXST','PGNY',
        'FTDR','XPEL','UFPT','DXPE','AVNS','LKFN','SKYW','PRAA',
        'AMSF','HSTM','PFIS','RCUS','CCOI','SANA','NKTR','INVA',
    ]

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
        ('https://en.wikipedia.org/wiki/S%26P_400',                   None),
    ]:
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


def load_ticker_pool() -> list:
    """
    Wrapper om get_large_ticker_pool() aan te roepen en session_state bij te werken.
    Gebruik deze functie in de UI — nooit get_large_ticker_pool() direct aanroepen
    vanuit code die session_state nodig heeft (mag niet binnen @st.cache_data).
    """
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
    Scan de volledige tickerpool op basis van de geselecteerde strategie.

    - GEEN vroege break of head()-limiet — alle hits worden geretourneerd.
    - max_results wordt genegeerd voor alle strategieën behalve event_sentiment
      (die heeft een eigen SCAN_SENT_MAX_RESULTS constante voor ruis-controle).
    - Voortgangsbalk loopt altijd over len(pool), ongeacht hits of fouten.
    - Gefaalde tickers worden gelogd maar onderbreken de loop niet.
    """
    print(f"DEBUG: Totaal aantal tickers aangeboden aan loop: {len(pool)}")

    rows        = []
    failed      = []           # tickers waarvoor geen data beschikbaar was
    errored     = []           # tickers met een onverwachte fout
    total       = len(pool)

    # ── Voortgangsbalk — altijd gebaseerd op de volledige pool ────────────────
    progress_bar = st.progress(0.0, text=f"⏳ Initialiseren scan van {total} tickers...")
    status_box   = st.empty()  # Tweede regel voor live statistieken

    for idx, ticker in enumerate(pool):

        # Voortgang: altijd op basis van idx / total, nooit op hits
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

        # ── Verborgen break verwijderd ────────────────────────────────────────
        # NIET: if len(rows) >= max_results: break
        # De volledige pool wordt altijd doorlopen; filtering gebeurt achteraf.

        try:
            # ── Data ophalen via gecachte functie (per ticker, niet bulk) ─────
            df, info = fetch_ticker_data(ticker, period=DATA_SCAN_PERIOD)
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
            ub, mb, lb = compute_bollinger_series(close, BB_PERIOD, BB_STD_NORMAL)
            ub25, mb25, lb25 = compute_bollinger_series(close, BB_PERIOD, BB_STD_WIDE)
            n = len(close)

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
                    market_cap = t_info.get('marketCap', 0) or 0
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
                # Poort 1 — Hard volume veto (institutioneel geld verplicht)
                vol_20d_avg = float(volume.rolling(VOL_MA_PERIOD).mean().iloc[-1])
                vol_today   = float(volume.iloc[-1])
                vol_spike   = round(vol_today / vol_20d_avg, 2) if vol_20d_avg > 0 else 0.0

                if vol_today < (SCAN_SENT_VOL_SPIKE * vol_20d_avg):
                    # Hard veto — geen institutioneel volume, maar loop gaat door
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

            # ── Match gevonden → MTF validatie + R:R veto + rij toevoegen ─────
            if match:
                mtf = compute_multi_timeframe_check(ticker)

                # Bereken R:R voor dit aandeel (zelfde logica als Deep Dive)
                _safe_sup = float(df['Low'].rolling(
                    min(SUPPORT_WINDOW, len(df)), min_periods=1).min().iloc[-1])
                _safe_res = float(df['High'].rolling(
                    min(SUPPORT_WINDOW, len(df)), min_periods=1).max().iloc[-1])
                if _safe_sup <= 0 or _safe_sup > last_price:
                    _safe_sup = last_price * (1 - TRADE_FALLBACK_SL)
                if _safe_res <= last_price:
                    _safe_res = last_price * (1 + TRADE_FALLBACK_RES)

                _sl_scan      = _safe_sup * (1 - TRADE_RISK_PCT)
                _risk_scan    = last_price - _sl_scan
                _tp3_scan     = _safe_res * (1 - TRADE_TP3_BUFFER)
                _reward_scan  = _tp3_scan - last_price
                _rr_scan      = round(_reward_scan / _risk_scan, 2) \
                                 if _risk_scan > 0 and _reward_scan > 0 else 0.0

                # Pas R:R veto toe — kan status naar REJECTED zetten
                mtf = apply_rr_veto(
                    mtf_result       = mtf,
                    current_price    = last_price,
                    resistance_target= _tp3_scan,
                    actual_rr        = _rr_scan,
                )

                # ── Alpha Scanner: BEARISH weektrend is normaal aan de bodem ──
                # Een bodem-signaal heeft per definitie een bearish achtergrond.
                # Als de R:R excellent is EN RSI oversold, override REJECTED → APPROVED.
                if (strategy == "alpha_scanner" and
                        mtf['status'] == 'REJECTED' and
                        not mtf.get('rr_veto', False) and   # R:R veto mag NIET actief zijn
                        _rr_scan >= TRADE_REWARD_RATIO and  # R:R is voldoende
                        rsi_val <= SCAN_ALPHA_RSI_MAX and   # RSI is oversold
                        mtf.get('weekly_trend') == 'BEARISH'):
                    # Bearish weektrend is de reden — dit is verwacht aan een bodem
                    mtf = dict(mtf)  # kopie
                    mtf['status']         = 'APPROVED'
                    mtf['rr_veto']        = False
                    mtf['rr_veto_reason'] = ''
                    mtf['alpha_override'] = True
                    mtf['alpha_override_note'] = (
                        f'Bodem-override: bearish 1W normaal bij RSI {int(rsi_val)} '
                        f'(R:R {_rr_scan}:1 ≥ {TRADE_REWARD_RATIO}:1)'
                    )
                    print(f"DEBUG ALPHA OVERRIDE: {ticker} → APPROVED "
                          f"(RSI={int(rsi_val)}, R:R={_rr_scan})")

                rr_note = (f"⚠ {mtf['rr_veto_reason']}"
                           if mtf.get('rr_veto') else f"R:R {_rr_scan}:1")

                # Voeg override-notitie toe als Alpha bodem-override actief is
                override_note = mtf.get('alpha_override_note', '')

                row = {
                    'Ticker':    ticker,
                    'Koers':     str(round(last_price, 2)),   # str → geen Arrow int/float mix
                    'RSI (14D)': str(int(rsi_val)),           # str → geen Arrow int/'N/A' mix
                    'R:R':       str(_rr_scan),
                    'R:R Status':rr_note,
                    'Patroon':   detect_candlestick_pattern(df),
                    'MTF Status':mtf['status'],
                    '1W Trend':  mtf['weekly_trend'],
                    '1H Volume': mtf['hourly_volume'],
                }
                if override_note:
                    row['⚡ Bodem Override'] = override_note
                row.update(extra_info)
                rows.append(row)
                print(f"DEBUG HIT [{len(rows):03d}]: {ticker} | RSI={int(rsi_val)} "
                      f"| MTF={mtf['status']} | R:R={_rr_scan} "
                      f"| VETO={'JA: '+mtf['rr_veto_reason'] if mtf['rr_veto'] else 'nee'}")

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


def style_scanner_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
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
            ('border-bottom', '2px solid #F0B90B'), ('padding', '8px 12px'),
        ]},
        {'selector': 'tbody td', 'props': [
            ('font-family', 'JetBrains Mono, monospace'), ('font-size', '0.8rem'),
            ('padding', '6px 12px'), ('border-bottom', '1px solid #2B3139'),
        ]},
    ])
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
# MAIN NAVIGATION  (Watchlist-tab verwijderd – functionaliteit in Tab 1)
# ─────────────────────────────────────────────────────────────────────────────
main_tabs = st.tabs([
    "📊 Market Tracker",
    "🔍 Multi-Strategie Scanner",
    "📈 Deep Dive & Grafiek",
    "📡 Live Monitor",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CORE MARKET TRACKER  (+ ingebouwde Watchlist)
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Core Market Tracker — Live Overzicht")
    st.markdown('<small style="color:#848E9C;">Live koersen · RSI 14D · Support 30D · Patroon Detectie · Fase Analyse · Actie Signaal</small>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Beheer Panel ──────────────────────────────────────────────────────────
    col_add, col_del, col_act = st.columns([2, 2, 1])

    with col_add:
        new_ticker = st.text_input(
            "➕ Ticker Toevoegen",
            placeholder="bijv. AAPL, ADYEN.AS, BESI.AS...",
            key="add_ticker_input",
        )
        if st.button("Toevoegen", key="btn_add_ticker"):
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.main_market_tickers:
                st.session_state.main_market_tickers.append(t)
                _save_userdata()
                st.success(f"✅ {t} toegevoegd.")
                st.rerun()
            elif t in st.session_state.main_market_tickers:
                st.warning(f"⚠ {t} staat al in de lijst.")

        # Import: plak een kommalijst om tickers toe te voegen
        with st.expander("📋 Import — meerdere tickers tegelijk", expanded=False):
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

    with col_del:
        ticker_to_remove = st.selectbox(
            "🗑 Ticker Verwijderen",
            options=["— Selecteer —"] + st.session_state.main_market_tickers,
            key="remove_ticker_select"
        )
        if st.button("Verwijderen", key="btn_remove_ticker"):
            if ticker_to_remove != "— Selecteer —" and ticker_to_remove in st.session_state.main_market_tickers:
                st.session_state.main_market_tickers.remove(ticker_to_remove)
                _save_userdata()
                st.success(f"🗑 {ticker_to_remove} verwijderd.")
                st.rerun()

        # Export: toon huidige lijst om te kopiëren
        with st.expander("📤 Export — kopieer mijn lijst", expanded=False):
            export_str = ", ".join(st.session_state.main_market_tickers)
            st.text_area(
                "Kopieer deze lijst en bewaar hem",
                value=export_str,
                height=120,
                key="export_tickers_txt"
            )
            st.caption("💡 Plak dit de volgende keer bij 'Import' om je lijst te herstellen.")

    with col_act:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", key="btn_refresh_main"):
            st.cache_data.clear()
            st.rerun()
        st.markdown(f"<small style='color:#848E9C;'>{len(st.session_state.main_market_tickers)} tickers</small>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Data Ophalen & Tabel ──────────────────────────────────────────────────
    try:
        with st.spinner("⏳ Live marktdata ophalen..."):
            main_df = build_main_table(tuple(st.session_state.main_market_tickers))
    except Exception as _e:
        st.error(f"❌ Fout in build_main_table: {type(_e).__name__}: {str(_e)}")
        import traceback as _tb
        st.code(_tb.format_exc())
        main_df = pd.DataFrame()

    if not main_df.empty:
        m1, m2, m3, m4 = st.columns(4)
        try:
            n_bodem   = len(main_df[main_df['Koers Status / Fase'].str.contains('Bodemfase', na=False)])
            n_uptrend = len(main_df[main_df['Koers Status / Fase'].str.contains('Uptrend', na=False)])
            n_herstel = len(main_df[main_df['Koers Status / Fase'].str.contains('Herstel', na=False)])
            n_koop    = len(main_df[main_df['Actie'].str.contains('KOOPWAARDIG', na=False)])
            m1.metric("⚡ Bodemfase", n_bodem)
            m2.metric("🚀 Uptrend", n_uptrend)
            m3.metric("🌱 Vroeg Herstel", n_herstel)
            m4.metric("🟢 Koopwaardig", n_koop)
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Mobiel-vriendelijke tabelweergave ─────────────────────────────────
        # Detecteer schermgrootte via URL-parameter of toon toggle
        is_mobile = st.checkbox("📱 Compacte weergave (mobiel)", value=False, key="mobile_view")

        if is_mobile:
            # Compacte versie: alleen de meest essentiële kolommen
            display_cols = [c for c in main_df.columns if not c.startswith('_')]
            mobile_cols  = ['Ticker', 'Koers', 'RSI (14D)', 'Afwijking %', 'Actie']
            mobile_cols  = [c for c in mobile_cols if c in display_cols]
            mobile_df    = main_df[mobile_cols].copy()

            def style_mobile(df):
                def row_style(row):
                    # Haal rij-data op uit originele df
                    ticker_val = row.get('Ticker','')
                    orig = main_df[main_df['Ticker'] == ticker_val]
                    rsi_v = float(orig['_rsi_float'].iloc[0]) if not orig.empty else 50.0
                    dev_v = float(orig['_dev_float'].iloc[0]) if not orig.empty else 999.0
                    action = str(row.get('Actie',''))
                    base   = 'background-color:#13171C;color:#E8ECEF;'
                    styles = [base]*len(row)
                    col_list = list(row.index)
                    # RSI kleur
                    if 'RSI (14D)' in col_list:
                        idx = col_list.index('RSI (14D)')
                        if rsi_v < TABLE_RSI_GREEN:   styles[idx]='background-color:#00843A;color:#FFF;font-weight:600;'
                        elif rsi_v > TABLE_RSI_RED:   styles[idx]='background-color:#A32040;color:#FFF;font-weight:600;'
                    # Afwijking kleur
                    if 'Afwijking %' in col_list:
                        idx = col_list.index('Afwijking %')
                        if dev_v < TABLE_DEV_GREEN:   styles[idx]='background-color:#00843A;color:#FFF;font-weight:600;'
                    # Actie kleur
                    if 'Actie' in col_list:
                        idx = col_list.index('Actie')
                        if 'KOOPWAARDIG' in action:   styles[idx]='background-color:#00843A;color:#FFF;font-weight:700;'
                        elif 'VOORZICHTIG' in action: styles[idx]='background-color:#A32040;color:#FFF;font-weight:600;'
                        elif 'AANHOUDEN' in action:   styles[idx]='background-color:#1A1500;color:#F0B90B;font-weight:600;'
                    return styles
                s = df.style.apply(row_style, axis=1)
                s.set_table_styles([
                    {'selector':'thead th','props':[
                        ('background-color','#0B0E11'),('color','#F0B90B'),
                        ('font-size','0.75rem'),('padding','6px 8px'),
                        ('border-bottom','2px solid #F0B90B')]},
                    {'selector':'tbody td','props':[
                        ('font-size','0.8rem'),('padding','5px 8px'),
                        ('border-bottom','1px solid #2B3139')]},
                ])
                return s

            st.dataframe(style_mobile(mobile_df), width='stretch', height=600)
            st.caption("Toon alle kolommen → vink 'Compacte weergave' uit")
        else:
            st.dataframe(style_main_table(main_df), width='stretch', height=620)

        # Deep Dive selectie
        st.markdown("---")
        sel_c1, sel_c2 = st.columns([3, 1])
        with sel_c1:
            selected_for_dive = st.selectbox(
                "🔭 Selecteer ticker voor Deep Dive →",
                options=main_df['Ticker'].tolist(),
                index=0,
                key="main_deep_dive_select"
            )
        with sel_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📈 Open Deep Dive", key="btn_open_deep_dive_main"):
                st.session_state.current_alpha = selected_for_dive
                st.success(f"✅ {selected_for_dive} geladen → ga naar '📈 Deep Dive & Grafiek'")
    else:
        st.warning("Geen data beschikbaar. Controleer je internetverbinding.")

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
            wl_rows = []
            with st.spinner("⏳ Watchlist analyseren..."):
                for ticker, label in st.session_state.custom_watchlist.items():
                    try:
                        df_wl, info_wl = fetch_ticker_data(ticker, period=DATA_WL_PERIOD)
                        if df_wl is None or len(df_wl) < DATA_MIN_ROWS_FETCH:
                            wl_rows.append({'Ticker': ticker, 'Label': label, 'Koers': 'N/A',
                                            'Dag %': 'N/A', 'RSI': 'N/A',
                                            'Support': 'N/A', 'Weerstand': 'N/A',
                                            'Afw. Support %': 'N/A',
                                            'Gedetecteerd Patroon': '⚠ Geen Data'})
                            continue

                        cl  = df_wl['Close'].squeeze()
                        lp  = round(float(cl.iloc[-1]), 2)

                        currency_wl = info_wl.get('currency', '') if info_wl else ''
                        sym_wl = '€' if currency_wl == 'EUR' else '$' if currency_wl == 'USD' else ''

                        rsi_wl  = int(compute_rsi(cl, RSI_PERIOD))
                        pat_wl  = detect_candlestick_pattern(df_wl)
                        sup_wl, res_wl = compute_support_resistance(df_wl, SUPPORT_WINDOW, current_price=lp)
                        dev_wl  = round(((lp - sup_wl) / sup_wl) * 100, 2) if sup_wl > 0 else 0.0

                        day_chg = round(((cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2]) * 100, 2) if len(cl) >= 2 else 0.0
                        trend_str = f"{'▲' if day_chg >= 0 else '▼'} {abs(day_chg):.2f}%"

                        wl_rows.append({
                            'Ticker':             ticker,
                            'Label':              label,
                            'Koers':              f"{sym_wl}{lp:,.2f}",
                            'Dag %':              trend_str,
                            'RSI':                str(rsi_wl),
                            'Support':            f"{sym_wl}{sup_wl:,.2f}",
                            'Weerstand':          f"{sym_wl}{res_wl:,.2f}",
                            'Afw. Support %':     f"{dev_wl:.2f}",
                            'Gedetecteerd Patroon': pat_wl,
                        })
                    except Exception as e:
                        wl_rows.append({'Ticker': ticker, 'Label': label, 'Koers': 'ERR',
                                        'Dag %': 'ERR', 'RSI': 'ERR',
                                        'Support': 'ERR', 'Weerstand': 'ERR',
                                        'Afw. Support %': 'ERR',
                                        'Gedetecteerd Patroon': f'⚠ {str(e)[:40]}'})

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
                            ('border-bottom', '2px solid #F0B90B'), ('padding', '8px 12px'),
                        ]},
                        {'selector': 'tbody td', 'props': [
                            ('font-family', 'JetBrains Mono, monospace'), ('font-size', '0.8rem'),
                            ('padding', '6px 12px'), ('border-bottom', '1px solid #2B3139'),
                        ]},
                    ])
                    return s

                wl_mobile = st.session_state.get('mobile_view', False)
                if wl_mobile:
                    wl_mobile_cols = ['Ticker', 'Koers', 'RSI', 'Afw%', 'Gedetecteerd Patroon']
                    wl_mobile_cols = [c for c in wl_mobile_cols if c in wl_df.columns]
                    st.dataframe(style_wl(wl_df[wl_mobile_cols]), width='stretch', height=300)
                else:
                    st.dataframe(style_wl(wl_df), width='stretch', height=300)

                bullish_n = sum(1 for r in wl_rows if 'Bullish' in str(r.get('Gedetecteerd Patroon', '')))
                if bullish_n > 0:
                    st.success(f"🟢 **{bullish_n} Bullish patroon(en)** gedetecteerd in je watchlist!")
                else:
                    st.info("Geen bullish patronen op dit moment.")

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: MULTI-STRATEGIE SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Multi-Strategie Scanner — 600+ Tickers")
    st.markdown('<small style="color:#848E9C;">S&P 500 · S&P MidCap 400 · Europese Top · Multi-Timeframe Validatie</small>', unsafe_allow_html=True)

    # Pool alvast laden bij openen tab — wrapper update ook session_state voor sidebar
    _preload_pool = load_ticker_pool()

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
    }

    # Twee rijen: 3 + 4
    row1_keys = list(strategy_map.keys())[:3]
    row2_keys = list(strategy_map.keys())[3:]

    btn_row1 = st.columns(3)
    for i, sk in enumerate(row1_keys):
        label, tooltip = strategy_map[sk]
        with btn_row1[i]:
            if st.button(label, key=f"scan_btn_{sk}", help=tooltip, use_container_width=True):
                st.session_state.active_strategy = sk
                st.session_state.scanner_results = pd.DataFrame()

    btn_row2 = st.columns(4)
    for i, sk in enumerate(row2_keys):
        label, tooltip = strategy_map[sk]
        with btn_row2[i]:
            if st.button(label, key=f"scan_btn_{sk}", help=tooltip, use_container_width=True):
                st.session_state.active_strategy = sk
                st.session_state.scanner_results = pd.DataFrame()

    if st.session_state.active_strategy:
        _active = st.session_state.active_strategy
        if _active in strategy_map:
            strategy_label, strategy_desc = strategy_map[_active]
        else:
            strategy_label, strategy_desc = _active, ""
        st.markdown(f"**Actieve Strategie:** {strategy_label} — <span style='color:#848E9C;'>{strategy_desc}</span>", unsafe_allow_html=True)
        st.markdown("---")

        sc1, sc2 = st.columns([1, 3])
        with sc1:
            if st.button(f"▶ Start Volledige Scan", key="btn_start_scan"):
                pool = load_ticker_pool()
                st.info(f"🔍 Scan gestart: **{len(pool)} tickers** worden doorlopen...")
                results = run_scanner(
                    st.session_state.active_strategy,
                    pool,
                )
                st.session_state.scanner_results = results

        with sc2:
            # Toon pool-grootte en scrape-status
            pool_n = st.session_state.get('total_ticker_count', '?')
            scrape_info = st.session_state.get('pool_scrape_status', '')
            st.markdown(
                f"<div style='background:#13171C;border:1px solid #2B3139;border-radius:6px;"
                f"padding:10px 14px;font-family:monospace;font-size:0.78rem;'>"
                f"<span style='color:#848E9C;'>Pool: </span>"
                f"<b style='color:#F0B90B;font-size:1rem;'>{pool_n}</b>"
                f"<span style='color:#848E9C;'> tickers</span><br>"
                f"<span style='color:#848E9C;font-size:0.72rem;'>{scrape_info}</span>"
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

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: LIVE MONITOR — Real-time tracking van vastgepinde tickers
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[3]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    lm_h1, lm_h2 = st.columns([3, 1])
    with lm_h1:
        st.markdown("### 📡 Live Monitor — Real-Time Prijsactie")
        st.markdown('<small style="color:#848E9C;">Vastgepinde tickers · Auto-refresh · 15M MR Status · After-hours</small>', unsafe_allow_html=True)
    with lm_h2:
        n_pinned = len(st.session_state['pinned_tickers'])
        st.metric("📌 Vastgezet", n_pinned)

    st.markdown("---")

    # ── Auto-refresh configuratie ─────────────────────────────────────────────
    try:
        from streamlit_autorefresh import st_autorefresh
        autorefresh_available = True
    except ImportError:
        autorefresh_available = False

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        refresh_interval = st.selectbox(
            "🔄 Refresh interval",
            options=[15, 30, 60, 120, 300],
            format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m",
            index=1,
            key="lm_refresh_interval"
        )
    with ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        monitor_active = st.toggle("▶ Auto-refresh AAN", value=True, key="lm_active")
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Nu vernieuwen", key="btn_lm_manual_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Auto-refresh via streamlit-autorefresh indien beschikbaar
    if autorefresh_available and monitor_active:
        count = st_autorefresh(
            interval=refresh_interval * 1000,
            limit=None,
            key="live_monitor_refresh"
        )
    else:
        if monitor_active and not autorefresh_available:
            st.caption("⚠ streamlit-autorefresh niet beschikbaar — gebruik 'Nu vernieuwen'")

    # ── Ticker beheer ──────────────────────────────────────────────────────────
    lm_add1, lm_add2, lm_add3 = st.columns([2, 1, 1])
    with lm_add1:
        manual_pin = st.text_input(
            "➕ Ticker handmatig toevoegen",
            placeholder="bijv. NVDA, AMD, TSLA...",
            key="lm_manual_pin_input"
        )
    with lm_add2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📌 Vastzetten", key="btn_lm_add"):
            t = manual_pin.strip().upper()
            if t and t not in st.session_state['pinned_tickers']:
                st.session_state['pinned_tickers'].append(t)
                st.rerun()
    with lm_add3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Alles verwijderen", key="btn_lm_clear"):
            st.session_state['pinned_tickers'] = []
            st.rerun()

    # ── Live tabel ────────────────────────────────────────────────────────────
    if not st.session_state['pinned_tickers']:
        st.info("📌 Geen tickers vastgezet. Pin tickers via de Scanner of voeg ze hierboven toe.")
    else:
        pinned = st.session_state['pinned_tickers']

        # Tijdstempel laatste update
        now_str = pd.Timestamp.now().strftime("%H:%M:%S")
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.75rem;color:#848E9C;"
            f"margin-bottom:8px;'>Laatste update: <b style='color:#F0B90B;'>{now_str}</b>"
            f" · {len(pinned)} tickers · interval: {refresh_interval}s</div>",
            unsafe_allow_html=True
        )

        # ── Data ophalen per ticker ───────────────────────────────────────────
        @st.cache_data(ttl=refresh_interval)
        def fetch_live_monitor_row(ticker: str) -> dict:
            """Haal alle live data op voor één ticker — gecached op refresh_interval."""
            row = {
                'Ticker': ticker, 'Koers': '–', 'Δ Dag%': '–', 'RSI': '–',
                'Support': '–', 'Weerstand': '–', 'Afw%': '–',
                '15M MR': '–', '15M [0]': '–', '5M [0]': '–',
                'Fase': '–', 'Actie': '–',
                '_rsi': 50.0, '_dev': 999.0, '_ext_chg': None,
                '_action': '', '_mr_dir': '',
            }
            try:
                df, info = fetch_ticker_data(ticker, period='5d')
                if df is None or len(df) < 3:
                    return row

                close = df['Close'].squeeze()
                lp    = float(close.iloc[-1])
                prev  = float(close.iloc[-2]) if len(close) >= 2 else lp
                dc    = round((lp - prev) / prev * 100, 2) if prev > 0 else 0.0
                rsi_v = compute_rsi(close)
                sup, res = compute_support_resistance(df, SUPPORT_WINDOW, current_price=lp)
                dev   = round((lp - sup) / sup * 100, 2) if sup > 0 else 0.0
                pat   = detect_candlestick_pattern(df)
                phase = determine_phase(rsi_v, dev, pat)
                act   = determine_action(rsi_v, dev, pat, phase)

                cur = info.get('currency', '') if info else ''
                sym = '€' if cur == 'EUR' else '$' if cur == 'USD' else ''

                # Live prijs incl. extended hours
                live = fetch_live_price(ticker)
                dp   = live['price'] if live['price'] else lp
                mp   = live['market_phase']
                ce   = live['change_ext']

                badge = ''
                if mp == 'AFTER-HOURS': badge = ' 🌙'
                elif mp == 'PRE-MARKET': badge = ' 🌅'

                koers_str = f"{sym}{dp:,.2f}{badge}"
                dc_str    = f"{'▲' if dc >= 0 else '▼'}{abs(dc):.2f}%"

                # Intraday
                intra = fetch_intraday_patterns(ticker)
                mr15  = compute_15m_mean_reversion(ticker)

                row.update({
                    'Ticker':   ticker,
                    'Koers':    koers_str,
                    'Δ Dag%':   dc_str,
                    'RSI':      str(int(rsi_v)) if not math.isnan(rsi_v) else 'N/A',
                    'Support':  f"{sym}{sup:,.2f}",
                    'Weerstand':f"{sym}{res:,.2f}",
                    'Afw%':     f"{dev:.2f}",
                    '15M MR':   mr15['status'],
                    '15M [0]':  intra['15M[0]'],
                    '5M [0]':   intra['5M[0]'],
                    'Fase':     phase,
                    'Actie':    act,
                    '_rsi':     float(rsi_v) if not math.isnan(rsi_v) else 50.0,
                    '_dev':     dev,
                    '_ext_chg': ce,
                    '_action':  act,
                    '_mr_dir':  mr15.get('direction') or '',
                })
            except Exception:
                pass
            return row

        # Haal alle rijen op
        lm_rows = [fetch_live_monitor_row(t) for t in pinned]

        # ── Render als kaartjes (compacter dan tabel op mobiel) ───────────────
        use_cards = st.session_state.get('mobile_view', False)

        if use_cards:
            # Mobiele kaartjes weergave
            for r in lm_rows:
                action = r['_action']
                mr_dir = r['_mr_dir']
                if 'KOOPWAARDIG' in action:   card_border = '#00C853'
                elif 'VOORZICHTIG' in action:  card_border = '#F6465D'
                elif 'BULLISH' in mr_dir:      card_border = '#00C853'
                elif 'BEARISH' in mr_dir:      card_border = '#F6465D'
                else:                          card_border = '#2B3139'

                mr_color = '#00C853' if 'BULLISH' in r['15M MR'] else '#F6465D' if 'BEARISH' in r['15M MR'] else '#848E9C'

                st.markdown(f"""
                <div style="background:#13171C;border:1px solid {card_border};border-left:4px solid {card_border};
                            border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <b style="color:#F0B90B;font-size:1.1rem;">{r['Ticker']}</b>
                    <span style="color:#E8ECEF;font-size:1.1rem;">{r['Koers']}</span>
                    <span style="color:{'#00C853' if '▲' in r['Δ Dag%'] else '#F6465D'};">{r['Δ Dag%']}</span>
                  </div>
                  <div style="display:flex;gap:12px;margin-top:6px;font-size:0.78rem;">
                    <span style="color:#848E9C;">RSI: <b style="color:#E8ECEF;">{r['RSI']}</b></span>
                    <span style="color:#848E9C;">Afw: <b style="color:#E8ECEF;">{r['Afw%']}%</b></span>
                    <span style="color:#848E9C;">Actie: <b style="color:{'#00C853' if 'KOOPWAARDIG' in r['Actie'] else '#F6465D' if 'VOORZICHTIG' in r['Actie'] else '#F0B90B'};">{r['Actie']}</b></span>
                  </div>
                  <div style="margin-top:4px;font-size:0.72rem;color:{mr_color};">{r['15M MR']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            # Desktop tabel-weergave
            lm_df = pd.DataFrame(lm_rows)
            display_cols = ['Ticker','Koers','Δ Dag%','RSI','Support','Weerstand',
                            'Afw%','15M MR','15M [0]','5M [0]','Fase','Actie']
            display_cols = [c for c in display_cols if c in lm_df.columns]
            lm_display = lm_df[display_cols].copy()

            def style_lm(df):
                def rs(row):
                    base  = f'background-color:#13171C;color:#E8ECEF;'
                    styles= [base]*len(row)
                    cl    = list(row.index)

                    action= str(row.get('Actie',''))
                    mr    = str(row.get('15M MR',''))
                    dc    = str(row.get('Δ Dag%',''))
                    orig  = lm_df[lm_df['Ticker']==row.get('Ticker','')]
                    rsi_f = float(orig['_rsi'].iloc[0]) if not orig.empty else 50.0
                    dev_f = float(orig['_dev'].iloc[0]) if not orig.empty else 999.0

                    # Dag% kleur
                    if 'Δ Dag%' in cl:
                        idx=cl.index('Δ Dag%')
                        styles[idx]= f'background-color:#13171C;color:{"#00C853" if "▲" in dc else "#F6465D"};font-weight:600;'

                    # RSI
                    if 'RSI' in cl:
                        idx=cl.index('RSI')
                        if rsi_f < TABLE_RSI_GREEN:   styles[idx]='background-color:#00843A;color:#FFF;font-weight:600;'
                        elif rsi_f > TABLE_RSI_RED:   styles[idx]='background-color:#A32040;color:#FFF;font-weight:600;'

                    # Afw%
                    if 'Afw%' in cl:
                        idx=cl.index('Afw%')
                        if dev_f < TABLE_DEV_GREEN:   styles[idx]='background-color:#00843A;color:#FFF;font-weight:600;'

                    # Support/Weerstand
                    if 'Support'   in cl: styles[cl.index('Support')]   = 'color:#64B5F6;'
                    if 'Weerstand' in cl: styles[cl.index('Weerstand')] = 'color:#CE93D8;'

                    # 15M MR
                    if '15M MR' in cl:
                        idx=cl.index('15M MR')
                        if 'BULLISH REVERSION' in mr and '✅' in mr:
                            styles[idx]='background-color:#00843A;color:#FFF;font-weight:700;font-size:0.72rem;'
                        elif 'BULLISH REVERSION' in mr:
                            styles[idx]='background-color:#0D2818;color:#00C853;font-weight:600;font-size:0.72rem;'
                        elif 'BEARISH REVERSION' in mr and '✅' in mr:
                            styles[idx]='background-color:#A32040;color:#FFF;font-weight:700;font-size:0.72rem;'
                        elif 'BEARISH REVERSION' in mr:
                            styles[idx]='background-color:#1A0008;color:#F6465D;font-weight:600;font-size:0.72rem;'
                        elif 'SETUP' in mr:
                            styles[idx]='background-color:#1A1500;color:#F0B90B;font-size:0.72rem;'

                    # Actie
                    if 'Actie' in cl:
                        idx=cl.index('Actie')
                        if 'KOOPWAARDIG' in action:   styles[idx]='background-color:#00843A;color:#FFF;font-weight:700;'
                        elif 'VOORZICHTIG' in action: styles[idx]='background-color:#A32040;color:#FFF;font-weight:600;'
                        elif 'AANHOUDEN' in action:   styles[idx]='background-color:#1A1500;color:#F0B90B;font-weight:600;'
                    return styles

                return df.style.apply(rs, axis=1)

            st.dataframe(style_lm(lm_display), width='stretch', height=min(600, 60+len(lm_rows)*38))

        # ── Ticker verwijderen ─────────────────────────────────────────────────
        st.markdown("---")
        rm1, rm2 = st.columns([3, 1])
        with rm1:
            to_remove = st.selectbox(
                "🗑 Ticker losmaken",
                options=["— Selecteer —"] + list(st.session_state['pinned_tickers']),
                key="lm_remove_select"
            )
        with rm2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Verwijderen", key="btn_lm_remove"):
                if to_remove != "— Selecteer —":
                    st.session_state['pinned_tickers'].remove(to_remove)
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DEEP DIVE & INTERACTIEVE GRAFIEK
# ═════════════════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

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

    with st.spinner(f"⏳ Data ophalen voor {active_ticker}..."):
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

        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric(f"💰 Koers ({_badge_dd})", _koers_lbl, delta=_delta_dd)
        mc2.metric("📊 RSI (14D)",       f"{int(rsi_dd)}"                                       if not np.isnan(rsi_dd)      else "N/A")
        mc3.metric("🟢 Support 30D",     f"{_sym_dd}{support_dd:,.2f}"    if not math.isnan(support_dd)    else "N/A")
        mc4.metric("🔴 Resistance 30D",  f"{_sym_dd}{resistance_dd:,.2f}" if not math.isnan(resistance_dd) else "N/A")
        mc5.metric("📐 Afwijking",       f"{deviation_dd:.2f}%"            if not math.isnan(deviation_dd)  else "N/A")
        mc6.metric("📡 Marktfase",       _phase_dd)

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

        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom':           True,       # Scrollwiel = zoom (zoals TradingView)
            'displayModeBar':       True,
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
        with st.spinner("Validatie uitvoeren op 1W en 1H..."):
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
                     f"Werkelijke R:R <b style='color:{rr_color};'>{actual_rr}:1</b>"),
                ]:
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

            with st.spinner("📡 Nieuws ophalen..."):
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

    st.markdown('</div>', unsafe_allow_html=True)


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
