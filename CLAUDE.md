# QuantEdge — projectcontext voor Claude Code

Persoonlijk trading-dashboard in **Streamlit** (Nederlandse UI, koersdata via **yfinance**). Eén bestand: `quantedge_dashboard.py` (~6200 regels). Eigenaar: Johan; kijkt er vooral naar op een **iPhone** (Safari, Streamlit Community Cloud). Communiceer in het **Nederlands**, kort en concreet.

## Draaien en deployen
- Lokaal: `pip install -r requirements.txt && streamlit run quantedge_dashboard.py` (Python 3.14, Streamlit ≥1.58, pandas 3, yfinance 1.4.x).
- Deploy: push naar `main` van `johanvanspreuwel/QuantEdge` → Streamlit Community Cloud herbouwt automatisch. Na een deploy moet de app op de telefoon **helemaal gesloten en heropend** worden.
- Bestanden: `quantedge_dashboard.py`, `quantedge_userdata.json` (eigen ticker-lijst ~117 tickers + watchlist), `requirements.txt`, `.streamlit/config.toml` (donker thema, goud `#F0B90B`).
- Werkwijze eerder: feature-branch + PR die Johan zelf mergt. Nu in Claude Code mag je rechtstreeks werken; vraag niet om toestemming voor elke kleine stap, maar commit niet naar `main` zonder dat Johan dat wil.
- Test altijd echt (`streamlit run`, ook in mobiele breedte ~390px). Eerdere sessies draaiden in een sandbox zonder internet naar Yahoo/PyPI en testten alleen met een nagebouwde Streamlit/yfinance — echte Yahoo-gedrag en echte Plotly-rendering zijn dus **nooit visueel geverifieerd**.

## Opbouw van `quantedge_dashboard.py` (volgorde in het bestand)
1. **Constanten** bovenaan (`TABLE_*`, `PHASE_*`, `SCAN_*`, `TRADE_*`, `MAIN_TABLE_WORKERS`, cache-TTL's, `DATA_*`). Pas drempels hier aan.
2. **CSS** (Inter + JetBrains Mono, klassen `qe-*`), header, opslag (lokaal bestand + GitHub API), session-state defaults.
3. **Helpers**: `compute_rsi`, `compute_support_resistance`, `detect_candlestick_pattern`, `determine_phase`, `determine_action`, `fetch_live_price`, `get_large_ticker_pool` (S&P 500 + MidCap 400 + Europa + extra's, ~970 tickers; Wikipedia-scrape met hardcoded fallback), `run_scanner` (strategieën: `momentum_train`, `mean_reversion`, `vol_squeeze`, `support_bounce`, `alpha_scanner`, `sr_bounce`, `momentum_low_float`, `event_sentiment`, `bb_momentum`).
4. **Hoofdtabel**: `_fetch_history` → `_fetch_daily_core` → `_row_from_result` → `build_main_table` / `load_main_df`; parallel via `_parallel_map` (ThreadPoolExecutor + `add_script_run_ctx`). Mislukkingen worden bewust **niet** gecachet (exceptions in `st.cache_data`).
5. **UI-helpers** (iPhone-first): `render_ticker_cards` (`<details>`-kaarten met SVG-sparkline, RSI-meter, range-balk), `render_kpis`, `render_breadth`, `style_main_table`.
6. **Alpha Tracker**: scant periodiek de Alpha-scan en volgt de beste resultaten (`alpha_*`); geschiedenis in `quantedge_alpha_history.json` op een aparte GitHub-branch `quantedge-data` (auto aangemaakt), lokaal bestand als fallback.
7. **Combi-Scan**: resultaten van meerdere scanners samenvoegen (`combo_merge`, `COMBO_PRESETS`, `render_combo_*`).
8. **Sectoren**: `SECTOR_SEED` = `_S_EXTRA` (volledige S&P 500 met GICS-sector) + `_S` (eigen indeling, wint) → `sector_map`, `with_sector`, `render_sector_summary`, `render_grouped_cards`. Wikipedia en Yahoo-`.info` zijn alleen nog reserve (Yahoo gaf op Streamlit Cloud te veel fouten).
9. **Sector-trend**: `_phase_codes` (scanner-fases per dag, vectorised; Vroeg Herstel is een benadering), `build_trend_panels`, `trends_from_panels`, `sector_members` (gelijkloop/rendement/Trend-score), `_trend_svg` (inline SVG), `sector_trend_section` (bron: eigen lijst of grote pool op verzoek; tot 5 jaar historie; periodes 3M–5J).
10. **Hoofdnavigatie**: `st.tabs` → 📊 Markt · 🔍 Scanner · 📈 Grafiek (Deep Dive, Plotly) · 📡 Live (fragment met `run_every`) · 🏆 Alpha · 🎯 Combi. Op mobiel staat de tab-balk onderaan (CSS, `@media (max-width:640px)`).

## Streamlit-/iPhone-valkuilen die we al tegenkwamen
- HTML in `st.markdown(..., unsafe_allow_html=True)` moet **één regel zonder lege regels/inspringing** zijn, anders ziet markdown het als codeblok.
- Streamlit 1.58 heeft geen `.stTabs`-klassen; gebruik `[data-baseweb="tab-list"]` / `[data-testid=...]`. Streamlit-CSS laadt later → eigen CSS vaak `!important` + `div[...]`-specificiteit nodig. `st.container(key="x")` geeft class `st-key-x`.
- Widgets: `st.pills`, `st.toggle`, `st.progress(text=)`; `st.context.headers` voor mobiele user-agent-detectie. Altijd fallback als een widget ontbreekt (`try/except`).
- `Ticker.history(raise_errors=True)` is thread-safe; `yf.download` niet. `.info` is traag/rate-limited: vermijd het in loops over veel tickers.
- `st.cache_data` cachet geen exceptions — gebruik dat voor Yahoo-fouten.
- Widgetwaarden die je via `st.session_state` zet moet je **vóór** het aanmaken van het widget zetten.

## Openstaand
- Bevestigen dat de **onderste tab-balk** op de echte iPhone-app werkt (CSS-fixes zijn gemerged; Johan heeft het nog niet bevestigd). Zo niet: DOM inspecteren met een screenshot of de app-URL.
- **Alpha-geschiedenis opslaan** werkt pas als in Streamlit *Secrets* staan: `GITHUB_TOKEN` (fine-grained, Contents: read & write op deze repo) en `GITHUB_REPO = "johanvanspreuwel/QuantEdge"`; optioneel `GITHUB_ALPHA_FILE`, `GITHUB_ALPHA_BRANCH`. Tokens horen alleen daar — nooit in de chat of in de code.
- Sector-trend voor de **grote pool** is alleen met nagebootste data getest; echte laadtijd (~970 tickers × 5 jaar) en Yahoo-rate-limits zijn onbekend.
- Sectorindeling van MidCap/Europese tickers is handmatig; Johan meldt afwijkingen, pas `_S` aan.
- Ideeën die we besproken hebben: dagelijkse Alpha-snapshot via GitHub Actions; beste Combi-Scan-resultaten naar de Alpha Tracker; sector-trend ook voor scanner-resultaten.
