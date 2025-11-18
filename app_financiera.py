from datetime import datetime, timedelta, date
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from yahooquery import Ticker as YQTicker

APP_VERSION = "FinanzasUP v9.2 — Gemini + UI pulida (fix IDs + tz)"

pio.templates.default = "plotly_dark"

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

TRADING_DAYS = {"1d": 252, "1wk": 52, "1mo": 12}


def configure_page() -> None:
    st.set_page_config(
        page_title="FinanzasUP — Análisis de Acciones/ETFs",
        page_icon="📈",
        layout="wide",
    )
    apply_dark_theme()


def apply_dark_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #f5f5f5;
        }
        [data-testid="stSidebar"] {
            background-color: #111827;
        }
        [data-testid="stHeader"] {
            background-color: #0e1117;
        }
        .stMarkdown, .stText, .stMetric {
            color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_{uuid4()}")


@st.cache_resource(show_spinner=False)
def init_gemini(api_key: str | None):
    if not GEMINI_AVAILABLE or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("models/gemini-1.5-flash")
    except Exception:
        return None


def translate_with_gemini(model, text: str, target_lang: str = "es") -> str:
    if not model or not text:
        return text or "No hay descripción disponible"
    prompt = (
        "El siguiente texto está en inglés. Traduce TODO al español neutro, claro y conciso. "
        "No agregues comentarios, notas ni explicaciones, solo la traducción. "
        "Máximo ~1200 caracteres.\n\n"
        f"Texto original:\n{text}"
    )
    try:
        resp = model.generate_content(prompt)
        txt = getattr(resp, "text", "") or ""
        txt = txt.strip()
        return txt if txt else text
    except Exception:
        return text


def _to_1d_numeric(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.to_numeric(getattr(x, "squeeze", lambda: x)(), errors="coerce")


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols_final = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols_final)

    df = df.reset_index().rename(columns=str.title)
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    if "Close" in df.columns:
        for c in ["Open", "High", "Low"]:
            if c not in df.columns:
                df[c] = df["Close"]

    for c in cols_final:
        if c not in df.columns:
            df[c] = np.nan

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ✅ FIX: forzar todas las fechas a ser "sin zona horaria"
    # Esto evita el error "Cannot join tz-naive with tz-aware DatetimeIndex"
    df["Date"] = df["Date"].dt.tz_localize(None)

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[c] = _to_1d_numeric(df[c])

    df = df.dropna(subset=["Date", "Close"])
    return df[cols_final].sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fetch_ohlc(symbol: str, start_dt: datetime, end_dt: datetime, interval: str, version: int = 6) -> pd.DataFrame:
    try:
        yq = YQTicker(symbol)
        yq_interval = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}[interval]
        yq_hist = yq.history(start=start_dt.date(), end=end_dt.date(), interval=yq_interval)
        if isinstance(yq_hist, pd.DataFrame) and not yq_hist.empty:
            if "symbol" in yq_hist.columns:
                yq_hist = yq_hist.drop(columns=["symbol"])
            yq_hist = yq_hist.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "adjclose": "Adj Close",
                    "adj_close": "Adj Close",
                    "volume": "Volume",
                }
            )
            if isinstance(yq_hist.index, pd.MultiIndex):
                yq_hist.index = yq_hist.index.get_level_values(-1)
            yq_hist = yq_hist.reset_index().rename(columns={"index": "Date"})
            out = _normalize_ohlc(yq_hist)
            if not out.empty:
                return out
    except Exception:
        pass

    try:
        tk = yf.Ticker(symbol)
        yf_hist = tk.history(
            start=start_dt,
            end=end_dt,
            interval=interval,
            actions=False,
            auto_adjust=False,
        )
        out = _normalize_ohlc(yf_hist)
        if not out.empty:
            return out
    except Exception:
        pass

    try:
        dl = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(dl.columns, pd.MultiIndex):
            dl.columns = ["_".join([str(x) for x in t if x]) for t in dl.columns]
        rename_map = {}
        for c in list(dl.columns):
            uc = c.upper()
            if "OPEN" in uc and "Open" not in rename_map.values():
                rename_map[c] = "Open"
            if "HIGH" in uc and "High" not in rename_map.values():
                rename_map[c] = "High"
            if "LOW" in uc and "Low" not in rename_map.values():
                rename_map[c] = "Low"
            if uc.endswith("CLOSE") and "Close" not in rename_map.values():
                rename_map[c] = "Close"
            if "ADJ" in uc and "CLOSE" in uc and "Adj Close" not in rename_map.values():
                rename_map[c] = "Adj Close"
            if "VOLUME" in uc and "Volume" not in rename_map.values():
                rename_map[c] = "Volume"
        dl = dl.rename(columns=rename_map)
        return _normalize_ohlc(dl)
    except Exception:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])


def pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def annualized_vol(returns: pd.Series, interval: str) -> float:
    periods = TRADING_DAYS.get(interval, 252)
    return float(returns.std(ddof=0) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, interval: str, rf_annual: float = 0.0) -> float:
    periods = TRADING_DAYS.get(interval, 252)
    rf_periodic = (1 + rf_annual) ** (1 / periods) - 1
    excess = returns - rf_periodic
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, interval: str, rf_annual: float = 0.0) -> float:
    periods = TRADING_DAYS.get(interval, 252)
    rf_periodic = (1 + rf_annual) ** (1 / periods) - 1
    excess = returns - rf_periodic
    downside = excess[excess < 0]
    dvol = downside.std(ddof=0)
    if dvol == 0 or np.isnan(dvol):
        return np.nan
    return float(excess.mean() * np.sqrt(periods) / dvol)


def cagr(returns: pd.Series, interval: str) -> float:
    if returns.empty:
        return np.nan
    periods = TRADING_DAYS.get(interval, 252)
    equity = (1 + returns).cumprod()
    total_ret = float(equity.iloc[-1] - 1)
    n_periods = returns.shape[0]
    years = n_periods / periods
    if years <= 0:
        return np.nan
    return float((1 + total_ret) ** (1 / years) - 1)


def calmar_ratio(cagr_value: float, mdd: float) -> float:
    if np.isnan(cagr_value) or np.isnan(mdd) or mdd >= 0:
        return np.nan
    return float(cagr_value / abs(mdd))


def var_es(returns: pd.Series, level: float = 0.95) -> tuple[float, float]:
    if returns.empty:
        return np.nan, np.nan
    q = np.quantile(returns.dropna(), 1 - level)
    tail = returns[returns <= q]
    es = tail.mean() if not tail.empty else np.nan
    return float(q), float(es)


def beta_vs_benchmark(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    joined = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
    if joined.shape[0] < 3:
        return np.nan
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]
    var = np.var(joined.iloc[:, 1])
    if var == 0:
        return np.nan
    return float(cov / var)


def rolling_beta_series(asset_ret: pd.Series, bench_ret: pd.Series, window: int = 63) -> pd.Series:
    joined = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    cov = joined.iloc[:, 0].rolling(window).cov(joined.iloc[:, 1])
    var = joined.iloc[:, 1].rolling(window).var()
    beta_series = cov / var.replace(0, np.nan)
    return beta_series.dropna()


def build_candles(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df["Date"], y=ma20, mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=ma50, mode="lines", name="MA50"))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Precio", hovermode="x unified")
    return fig


def build_volume(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volumen"))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Volumen", hovermode="x unified")
    return fig


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_rsi(df: pd.DataFrame) -> go.Figure:
    r = rsi(df["Close"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=r, mode="lines", name="RSI"))
    fig.add_hrect(y0=70, y1=70, line_width=1, line_dash="dot")
    fig.add_hrect(y0=30, y1=30, line_width=1, line_dash="dot")
    fig.update_layout(title="RSI (14)", xaxis_title="Fecha", yaxis_title="RSI", hovermode="x unified")
    return fig


def risk_summary_spa(metrics: dict) -> list[str]:
    vol = metrics.get("vol")
    mdd = metrics.get("mdd")
    sr = metrics.get("sharpe")
    beta = metrics.get("beta")
    var_ = metrics.get("var")
    es_ = metrics.get("es")
    cagr_value = metrics.get("cagr")
    sortino_value = metrics.get("sortino")
    calmar_value = metrics.get("calmar")
    corr_value = metrics.get("corr")

    def ok(x):
        return x is not None and not np.isnan(x)

    lines = []
    if ok(cagr_value):
        lines.append(f"Rentabilidad anualizada (CAGR): {cagr_value:.2%}. Crecimiento medio compuesto en el periodo.")
    if ok(vol):
        lines.append(f"Volatilidad anualizada: {vol:.2%}. Mayor volatilidad implica variaciones de precio más amplias.")
    if ok(mdd):
        lines.append(f"Drawdown máximo: {mdd:.2%}. Peor caída pico-a-valle observada.")
    if ok(sr):
        lines.append(f"Sharpe: {sr:.2f}. Retorno ajustado por riesgo usando volatilidad total.")
    if ok(sortino_value):
        lines.append(f"Sortino: {sortino_value:.2f}. Retorno ajustado penalizando solo la volatilidad bajista.")
    if ok(calmar_value):
        lines.append(f"Calmar: {calmar_value:.2f}. Rentabilidad anualizada frente a la peor caída histórica.")
    if ok(beta):
        postura = "más sensible que el mercado" if beta > 1 else "menos sensible que el mercado"
        lines.append(f"Beta vs benchmark: {beta:.2f}, {postura}. Mide la exposición sistemática.")
    if ok(corr_value):
        lines.append(f"Correlación con el benchmark: {corr_value:.2f}. Qué tan alineados se mueven los retornos.")
    if ok(var_):
        lines.append(f"VaR 95% diario: {var_:.2%}. Pérdida mínima esperada en el peor 5% de los días.")
    if ok(es_):
        lines.append(f"ES 95% diario: {es_:.2%}. Pérdida promedio condicionada a ese 5% peor.")
    return lines


@st.cache_data(show_spinner=False)
def fetch_info(symbol: str) -> dict:
    try:
        tk = yf.Ticker(symbol)
        return tk.info or {}
    except Exception:
        return {}


def company_explainer(info: dict, translated: str) -> dict:
    name = info.get("longName") or info.get("shortName") or "Nombre no disponible"
    sector = info.get("sector") or "Sector no disponible"
    industry = info.get("industry") or "Industria no disponible"
    country = info.get("country") or "País no disponible"
    website = info.get("website") or "Sitio no disponible"
    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "country": country,
        "website": website,
        "description_es": translated,
    }


def clamp_dates(d_start: date, d_end: date) -> tuple[date, date]:
    today = datetime.today().date()
    if d_end > today:
        d_end = today
    if d_start >= d_end:
        d_start = d_end - timedelta(days=365)
    return d_start, d_end


def resolve_quick_period(end_date: date, quick: str) -> tuple[date, date]:
    today = datetime.today().date()
    if end_date > today:
        end_date = today
    if quick == "YTD":
        start_date = date(end_date.year, 1, 1)
    elif quick == "1 año":
        start_date = end_date - timedelta(days=365)
    elif quick == "3 años":
        start_date = end_date - timedelta(days=3 * 365)
    elif quick == "5 años":
        start_date = end_date - timedelta(days=5 * 365)
    else:
        start_date = end_date - timedelta(days=365)
    return clamp_dates(start_date, end_date)


def basic_metrics(returns: pd.Series, interval: str, rf: float) -> dict:
    if returns.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan}
    return {
        "CAGR": cagr(returns, interval),
        "Vol": annualized_vol(returns, interval),
        "Sharpe": sharpe_ratio(returns, interval, rf),
    }


def main() -> None:
    configure_page()
    st.caption(APP_VERSION)

    st.title("📈 FinanzasUP — Análisis de Acciones, ETFs y Portafolios")
    st.markdown(
        """
        <div style="font-size:0.9rem; color:#9ca3af; margin-bottom:0.75rem;">
        1) Configura símbolo, benchmark y periodo en el menú lateral.<br>
        2) Navega por las pestañas: resumen, riesgos, comparación y portafolio.<br>
        3) Usa umbrales y escenario para evaluar tu tolerancia al riesgo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 1️⃣ Activos principales")
        ticker = st.text_input(
            "Símbolo a analizar",
            value="SPY",
            help="Ejemplos: AAPL, MSFT, TSLA, SPY...",
        ).strip().upper()
        benchmark = st.text_input(
            "Benchmark de referencia",
            value="^GSPC",
            help="Ejemplos: ^GSPC (S&P 500), ^NDX (Nasdaq 100), ^STOXX50E...",
        ).strip().upper()

        st.markdown("---")
        st.markdown("### 2️⃣ Periodo de análisis")

        period_mode = st.radio(
            "Modo de periodo",
            options=["Rápido", "Personalizado"],
            index=0,
            horizontal=True,
        )

        today = datetime.today().date()
        default_start = today - timedelta(days=5 * 365)

        if period_mode == "Rápido":
            end_input = st.date_input("Fecha de fin", value=today, max_value=today)
            quick = st.selectbox(
                "Periodo rápido",
                options=["1 año", "3 años", "5 años", "YTD"],
                index=2,
                help="Rango típico a partir de la fecha de fin.",
            )
            start, end = resolve_quick_period(end_input, quick)
            period_label = quick
        else:
            start_input = st.date_input("Inicio", value=default_start, max_value=today)
            end_input = st.date_input("Fin", value=today, max_value=today)
            start, end = clamp_dates(start_input, end_input)
            period_label = "Personalizado"

        st.markdown(
            f"**Periodo efectivo:** {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
        )

        st.markdown("---")
        st.markdown("### 3️⃣ Parámetros de cálculo")

        interval = st.selectbox(
            "Intervalo de datos",
            options=["1d", "1wk", "1mo"],
            index=0,
            help="Frecuencia de los datos históricos.",
        )
        rf = st.number_input(
            "Tasa libre de riesgo anual (%)",
            value=4.0,
            step=0.25,
            help="Se usa para Sharpe y Sortino.",
        ) / 100.0
        show_adj = st.toggle(
            "Usar 'Adj Close' para retornos",
            value=True,
            help="Si está activo, usa precios ajustados por dividendos y splits.",
        )

        st.markdown("---")
        with st.expander("4️⃣ Comparación y portafolio", expanded=False):
            extra_tickers_str = st.text_input(
                "Tickers adicionales (separados por coma)",
                value="",
                help="Ejemplo: QQQ, IWM, EFA",
            )
            extra_tickers = [t.strip().upper() for t in extra_tickers_str.split(",") if t.strip()]

            include_main_in_port = st.checkbox(
                "Incluir símbolo principal en portafolio",
                value=True,
            )

            weights_str = st.text_input(
                "Pesos del portafolio (mismo orden, separados por coma)",
                value="",
                help="Ejemplo: 0.4, 0.3, 0.3. Si se deja vacío o no coincide, se usan pesos iguales.",
            )

        st.markdown("---")
        with st.expander("5️⃣ Umbrales y escenario de estrés", expanded=False):
            mdd_limit_pct = st.number_input(
                "Drawdown máximo tolerable (%)",
                value=-30.0,
                step=1.0,
                help="Si el drawdown histórico es más negativo que este valor, se mostrará alerta.",
            )
            var_limit_pct = st.number_input(
                "VaR 95% límite (%)",
                value=-3.0,
                step=0.5,
                help="Si el VaR histórico es más negativo que este valor, se mostrará alerta.",
            )
            shock_ret_pct = st.number_input(
                "Shock al retorno medio (p.p.)",
                value=-2.0,
                step=0.5,
                help="Escenario: se resta este valor al retorno medio periódico.",
            )
            vol_factor = st.number_input(
                "Factor de volatilidad en escenario",
                value=1.5,
                step=0.1,
                help="Escenario: la volatilidad de los retornos se multiplica por este factor.",
            )

        st.markdown("---")
        st.markdown("### 6️⃣ Avanzado y Gemini")

        debug = st.toggle("Modo debug", value=False)
        api_key = st.text_input("GEMINI_API_KEY (opcional)", type="password")
        model = init_gemini(api_key)

        status_color = "#22c55e" if model else "#ef4444"
        status_label = "ACTIVO" if model else "INACTIVO"
        st.markdown(
            f"<div style='font-size:0.85rem; margin-top:0.25rem;'>"
            f"Gemini: <span style='color:{status_color}; font-weight:600;'>{status_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "📌 **Guía rápida:**\n"
            "- Ajusta símbolo, benchmark y periodo.\n"
            "- Usa *Comparación y portafolio* para varios activos.\n"
            "- Define *Umbrales* para ver alertas de riesgo.\n"
            "- Revisa el tab *Portafolio* para métricas agregadas.\n"
            "- Si Gemini está ACTIVO, la descripción se traduce al español."
        )

    # Datos del activo principal
    df = fetch_ohlc(
        ticker,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
        interval,
        version=6,
    )
    if df.empty:
        st.warning("Sin datos para el símbolo/fechas/intervalo seleccionados.")
        st.stop()

    if debug:
        st.code(f"Columnas {ticker}: {list(df.columns)}")
        st.write(df.head())
        st.write("GEMINI_AVAILABLE:", GEMINI_AVAILABLE)
        st.write("Gemini model cargado:", bool(model))

    price_col = "Adj Close" if show_adj and "Adj Close" in df.columns else "Close"
    price_series = df.set_index("Date")[price_col]
    ret = pct_change(price_series)

    # Benchmark
    bench_df = fetch_ohlc(
        benchmark,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
        interval,
        version=6,
    )
    if debug and not bench_df.empty:
        st.code(f"Columnas {benchmark}: {list(bench_df.columns)}")

    bench_ret = pd.Series(dtype=float)
    if not bench_df.empty:
        bench_price_col = "Adj Close" if show_adj and "Adj Close" in bench_df.columns else "Close"
        bench_price_series = bench_df.set_index("Date")[bench_price_col]
        bench_ret = pct_change(bench_price_series)

    # Métricas del activo principal
    vol = annualized_vol(ret, interval) if not ret.empty else np.nan
    sr = sharpe_ratio(ret, interval, rf) if not ret.empty else np.nan
    eq = (1 + ret).cumprod()
    mdd = max_drawdown(eq)
    var95, es95 = var_es(ret, 0.95)
    beta = beta_vs_benchmark(ret, bench_ret) if not bench_ret.empty else np.nan
    cagr_value = cagr(ret, interval)
    sortino_value = sortino_ratio(ret, interval, rf)
    calmar_value = calmar_ratio(cagr_value, mdd)
    corr_value = ret.corr(bench_ret) if not bench_ret.empty else np.nan
    rolling_beta = rolling_beta_series(ret, bench_ret, window=63) if not bench_ret.empty else pd.Series(dtype=float)

    # Escenario simulado
    mdd_limit = mdd_limit_pct / 100.0
    var_limit = var_limit_pct / 100.0
    shock_mean = shock_ret_pct / 100.0

    if not ret.empty:
        scenario_ret = (ret - shock_mean) * vol_factor
        scenario_eq = (1 + scenario_ret).cumprod()
        scenario_mdd = max_drawdown(scenario_eq)
        scenario_vol = annualized_vol(scenario_ret, interval)
        scenario_sr = sharpe_ratio(scenario_ret, interval, rf)
        scenario_var95, scenario_es95 = var_es(scenario_ret, 0.95)
    else:
        scenario_ret = pd.Series(dtype=float)
        scenario_mdd = scenario_vol = scenario_sr = scenario_var95 = scenario_es95 = np.nan

    # Info de empresa + traducción
    info = fetch_info(ticker)
    original_desc = info.get("longBusinessSummary", "") or ""
    translated_desc = translate_with_gemini(model, original_desc, "es")
    details = company_explainer(info, translated_desc)

    # Construcción de estructuras para comparación y portafolio
    comp_prices: dict[str, pd.Series] = {}
    comp_returns: dict[str, pd.Series] = {}

    comp_prices[ticker] = price_series
    comp_returns[ticker] = ret

    for sym in extra_tickers:
        if sym == ticker:
            continue
        sym_df = fetch_ohlc(
            sym,
            datetime.combine(start, datetime.min.time()),
            datetime.combine(end, datetime.min.time()),
            interval,
            version=6,
        )
        if sym_df.empty:
            continue
        sym_price_col = "Adj Close" if show_adj and "Adj Close" in sym_df.columns else "Close"
        sym_price_series = sym_df.set_index("Date")[sym_price_col]
        sym_ret = pct_change(sym_price_series)
        if sym_ret.empty:
            continue
        comp_prices[sym] = sym_price_series
        comp_returns[sym] = sym_ret

    # Portafolio
    portfolio_tickers: list[str] = []
    if include_main_in_port:
        portfolio_tickers.append(ticker)
    portfolio_tickers.extend(extra_tickers)
    portfolio_tickers = [t for t in portfolio_tickers if t in comp_returns]

    weights = None
    port_ret = pd.Series(dtype=float)

    if portfolio_tickers:
        usable = [s for s in portfolio_tickers if not comp_returns[s].empty]
        portfolio_tickers = usable
        if portfolio_tickers:
            if weights_str:
                try:
                    raw_w = [float(x) for x in weights_str.split(",")]
                    if len(raw_w) == len(portfolio_tickers):
                        w = np.array(raw_w, dtype=float)
                        if np.all(np.isfinite(w)) and np.any(w != 0):
                            w = w / w.sum()
                            weights = w
                except Exception:
                    weights = None
            if weights is None:
                weights = np.ones(len(portfolio_tickers), dtype=float) / len(portfolio_tickers)

            rets_df = pd.concat([comp_returns[s] for s in portfolio_tickers], axis=1, join="inner")
            rets_df.columns = portfolio_tickers
            if not rets_df.empty:
                port_ret = rets_df @ pd.Series(weights, index=portfolio_tickers)

    # Tabs
    tabs = st.tabs(
        [
            "📊 Resumen",
            "🕯️ Gráficas",
            "⚠️ Riesgos",
            "📈 Distribución",
            "📐 Métricas",
            "📚 Comparación",
            "💼 Portafolio",
            "📄 Datos",
        ]
    )
    t1, t2, t3, t4, t5, t6, t7, t8 = tabs

    # TAB 1: Resumen
    with t1:
        last_price = float(price_series.iloc[-1]) if not price_series.empty else np.nan
        total_return = float((1 + ret).prod() - 1) if not ret.empty else np.nan

        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
        c1.metric("Símbolo", ticker)
        c2.metric("Precio actual", f"{last_price:,.2f}" if not np.isnan(last_price) else "—")
        c3.metric("CAGR", f"{cagr_value:.2%}" if not np.isnan(cagr_value) else "—")
        c4.metric("Volatilidad anual", f"{vol:.2%}" if not np.isnan(vol) else "—")
        c5.metric("Rendimiento total", f"{total_return:.2%}" if not np.isnan(total_return) else "—")

        st.markdown(
            f"Analizando **{ticker}** frente a **{benchmark}** "
            f"desde **{start.strftime('%Y-%m-%d')}** hasta **{end.strftime('%Y-%m-%d')}**, "
            f"con datos **{interval}** y periodo **{period_label}**."
        )

        st.subheader(details["name"])
        colA, colB = st.columns(2)
        with colA:
            st.write(f"**Sector:** {details['sector']}")
            st.write(f"**Industria:** {details['industry']}")
            st.write(f"**País:** {details['country']}")
            st.write(f"**Sitio:** {details['website']}")
        with colB:
            st.write(f"**Benchmark:** {benchmark}")
            st.write(f"**Precio usado:** {price_col}")
            st.write(f"**Observaciones:** {ret.shape[0]}")

        st.markdown("**Descripción (inglés — original)**")
        st.write(original_desc or "No hay descripción disponible")

        if model:
            st.markdown("**Descripción traducida (Gemini)**")
        else:
            st.markdown("**Descripción (sin traducir — Gemini inactivo)**")
        st.write(details["description_es"] or "No hay descripción disponible")

        st.markdown("**Rendimiento acumulado**")
        cum_fig = go.Figure()
        cum_asset = cumulative_returns(ret)
        cum_fig.add_trace(go.Scatter(x=cum_asset.index, y=cum_asset, mode="lines", name=ticker))
        if not bench_ret.empty:
            cum_bench = cumulative_returns(bench_ret)
            cum_fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, mode="lines", name=benchmark))
        cum_fig.update_layout(xaxis_title="Fecha", yaxis_title="Rendimiento acumulado", hovermode="x unified")
        show_plot(cum_fig, "resumen_cum")

    # TAB 2: Gráficas
    with t2:
        st.subheader("Velas japonesas + Medias móviles")
        show_plot(build_candles(df, f"{ticker} — OHLC"), "graf_ohlc")

        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Volumen")
            show_plot(build_volume(df, f"{ticker} — Volumen"), "graf_vol")
        with c_right:
            st.subheader("RSI (14)")
            show_plot(build_rsi(df), "graf_rsi")

    # TAB 3: Riesgos
    with t3:
        st.subheader("Mapa de riesgos del activo principal")
        bullets = risk_summary_spa(
            {
                "vol": vol,
                "mdd": mdd,
                "sharpe": sr,
                "beta": beta,
                "var": var95,
                "es": es95,
                "cagr": cagr_value,
                "sortino": sortino_value,
                "calmar": calmar_value,
                "corr": corr_value,
            }
        )
        if bullets:
            for b in bullets:
                st.write(f"- {b}")
        else:
            st.write("No fue posible calcular métricas de riesgo con los datos actuales.")

        st.markdown("### Umbrales de riesgo")
        if not np.isnan(mdd):
            if mdd <= mdd_limit:
                st.error(f"⚠️ Drawdown máximo {mdd:.2%} PEOR que tu umbral de {mdd_limit:.2%}.")
            else:
                st.success(f"✅ Drawdown máximo {mdd:.2%} dentro de tu umbral de {mdd_limit:.2%}.")
        if not np.isnan(var95):
            if var95 <= var_limit:
                st.error(f"⚠️ VaR 95% {var95:.2%} PEOR que tu límite de {var_limit:.2%}.")
            else:
                st.success(f"✅ VaR 95% {var95:.2%} dentro de tu límite de {var_limit:.2%}.")

        st.markdown("### Escenario simulado (estrés)")
        if scenario_ret.empty:
            st.write("No fue posible construir un escenario simulado por falta de datos.")
        else:
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Vol. anual (escenario)", f"{scenario_vol:.2%}" if not np.isnan(scenario_vol) else "—")
            c_s2.metric("Drawdown máx. (escenario)", f"{scenario_mdd:.2%}" if not np.isnan(scenario_mdd) else "—")
            c_s3.metric("VaR 95% (escenario)", f"{scenario_var95:.2%}" if not np.isnan(scenario_var95) else "—")

            scen_eq = (1 + scenario_ret).cumprod()
            scen_dd = scen_eq / scen_eq.cummax() - 1
            scen_fig = go.Figure()
            scen_fig.add_trace(
                go.Scatter(x=scen_dd.index, y=scen_dd, mode="lines", name="Drawdown escenario")
            )
            scen_fig.update_layout(xaxis_title="Fecha", yaxis_title="Drawdown", hovermode="x unified")
            show_plot(scen_fig, "riesgos_escenario_dd")

            st.caption(
                "El escenario aplica un shock al retorno medio y un factor a la volatilidad. "
                "Es solo una simulación didáctica, no una proyección real."
            )

    # TAB 4: Distribución
    with t4:
        st.subheader("Distribución de retornos")
        if ret.empty:
            st.write("No hay suficientes datos de retornos para mostrar la distribución.")
        else:
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=ret, nbinsx=50, name="Retornos"))
            hist_fig.update_layout(
                xaxis_title="Retorno periódico",
                yaxis_title="Frecuencia",
                hovermode="x",
            )
            show_plot(hist_fig, "dist_hist")

            if not np.isnan(var95):
                st.write(f"VaR 95% histórico aproximado: {var95:.2%}")

            st.markdown("### Dispersión vs benchmark")
            if not bench_ret.empty:
                scatter_df = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
                if not scatter_df.empty:
                    scatter_fig = go.Figure()
                    scatter_fig.add_trace(
                        go.Scatter(
                            x=scatter_df.iloc[:, 1],
                            y=scatter_df.iloc[:, 0],
                            mode="markers",
                            name="Retornos",
                        )
                    )
                    scatter_fig.update_layout(
                        xaxis_title=f"Retornos {benchmark}",
                        yaxis_title=f"Retornos {ticker}",
                        hovermode="closest",
                    )
                    show_plot(scatter_fig, "dist_scatter")
                else:
                    st.write("No hay datos suficientes para la dispersión con el benchmark.")
            else:
                st.write("No se pudo calcular la dispersión porque no hay datos del benchmark.")

    # TAB 5: Métricas detalladas del activo principal
    with t5:
        st.subheader("Métricas detalladas (activo principal)")

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "—")
        r1c2.metric("Correlación", f"{corr_value:.2f}" if not np.isnan(corr_value) else "—")
        r1c3.metric("Sortino", f"{sortino_value:.2f}" if not np.isnan(sortino_value) else "—")
        r1c4.metric("Calmar", f"{calmar_value:.2f}" if not np.isnan(calmar_value) else "—")

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("VaR 95% (diario)", f"{var95:.2%}" if not np.isnan(var95) else "—")
        r2c2.metric("ES 95% (diario)", f"{es95:.2%}" if not np.isnan(es95) else "—")

        st.markdown("### Serie de drawdown")
        dd = eq / eq.cummax() - 1
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown"))
        dd_fig.update_layout(xaxis_title="Fecha", yaxis_title="Drawdown", hovermode="x unified")
        show_plot(dd_fig, "metric_dd")

        st.markdown("### Volatilidad móvil (21)")
        rolling_vol = ret.rolling(21).std() * np.sqrt(TRADING_DAYS.get(interval, 252))
        rv_fig = go.Figure()
        rv_fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode="lines", name="Vol móvil 21"))
        rv_fig.update_layout(xaxis_title="Fecha", yaxis_title="Vol anualizada", hovermode="x unified")
        show_plot(rv_fig, "metric_rv")

        if not rolling_beta.empty:
            st.markdown("### Beta móvil (63 periodos)")
            rb_fig = go.Figure()
            rb_fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode="lines", name="Beta 63"))
            rb_fig.update_layout(xaxis_title="Fecha", yaxis_title="Beta", hovermode="x unified")
            show_plot(rb_fig, "metric_rb")

        if not ret.empty:
            st.markdown("### Estadísticos de retornos")
            stats = pd.Series(
                {
                    "Media periódica": ret.mean(),
                    "Mediana": ret.median(),
                    "Desviación estándar": ret.std(),
                    "Mínimo": ret.min(),
                    "Máximo": ret.max(),
                    "Sesgo (skew)": ret.skew(),
                    "Curtosis": ret.kurt(),
                }
            )
            stats_df = stats.to_frame(name="Valor")
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

    # TAB 6: Comparación múltiple
    with t6:
        st.subheader("Comparación de activos")
        if len(comp_returns) <= 1 and bench_ret.empty:
            st.write("Agrega tickers adicionales en el panel lateral para comparar varios activos.")
        else:
            comp_fig = go.Figure()
            for name, series in comp_returns.items():
                if series.empty:
                    continue
                cum = cumulative_returns(series)
                comp_fig.add_trace(
                    go.Scatter(
                        x=cum.index,
                        y=cum,
                        mode="lines",
                        name=name,
                    )
                )
            if not bench_ret.empty:
                cum_bench = cumulative_returns(bench_ret)
                comp_fig.add_trace(
                    go.Scatter(
                        x=cum_bench.index,
                        y=cum_bench,
                        mode="lines",
                        name=benchmark,
                    )
                )
            comp_fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Rendimiento acumulado",
                hovermode="x unified",
            )
            show_plot(comp_fig, "comparacion_activos")

            rows = []
            for name, series in comp_returns.items():
                m = basic_metrics(series, interval, rf)
                rows.append(
                    {
                        "Activo": name,
                        "CAGR": m["CAGR"],
                        "Vol anual": m["Vol"],
                        "Sharpe": m["Sharpe"],
                    }
                )
            if not bench_ret.empty:
                m_bench = basic_metrics(bench_ret, interval, rf)
                rows.append(
                    {
                        "Activo": benchmark,
                        "CAGR": m_bench["CAGR"],
                        "Vol anual": m_bench["Vol"],
                        "Sharpe": m_bench["Sharpe"],
                    }
                )
            comp_df = pd.DataFrame(rows)
            if not comp_df.empty:
                st.markdown("### Cuadro comparativo de métricas")
                st.dataframe(
                    comp_df.style.format(
                        {
                            "CAGR": "{:.2%}",
                            "Vol anual": "{:.2%}",
                            "Sharpe": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

            if len(comp_returns) > 1:
                ret_df = pd.concat(comp_returns.values(), axis=1, join="inner")
                ret_df.columns = list(comp_returns.keys())
                corr_mat = ret_df.corr()
                st.markdown("### Matriz de correlaciones entre activos")
                st.dataframe(corr_mat.style.format("{:.2f}"), use_container_width=True)

    # TAB 7: Portafolio
    with t7:
        st.subheader("Análisis de portafolio")
        if portfolio_tickers and not port_ret.empty and weights is not None:
            st.write(
                f"Portafolio con activos: {', '.join(portfolio_tickers)}.\n\n"
                f"Pesos usados: {', '.join(f'{w:.1%}' for w in weights)}"
            )
            port_eq = (1 + port_ret).cumprod()
            port_cagr = cagr(port_ret, interval)
            port_vol = annualized_vol(port_ret, interval)
            port_sr = sharpe_ratio(port_ret, interval, rf)
            port_mdd = max_drawdown(port_eq)
            port_var95, port_es95 = var_es(port_ret, 0.95)

            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            c_p1.metric("CAGR portafolio", f"{port_cagr:.2%}" if not np.isnan(port_cagr) else "—")
            c_p2.metric("Vol anual", f"{port_vol:.2%}" if not np.isnan(port_vol) else "—")
            c_p3.metric("Sharpe", f"{port_sr:.2f}" if not np.isnan(port_sr) else "—")
            c_p4.metric("Drawdown máx.", f"{port_mdd:.2%}" if not np.isnan(port_mdd) else "—")

            port_fig = go.Figure()
            port_fig.add_trace(go.Scatter(x=port_eq.index, y=port_eq - 1, mode="lines", name="Portafolio"))
            if not ret.empty:
                port_fig.add_trace(go.Scatter(x=cum_asset.index, y=cum_asset, mode="lines", name=ticker))
            if not bench_ret.empty:
                cum_bench = cumulative_returns(bench_ret)
                port_fig.add_trace(
                    go.Scatter(x=cum_bench.index, y=cum_bench, mode="lines", name=benchmark)
                )
            port_fig.update_layout(xaxis_title="Fecha", yaxis_title="Rendimiento acumulado", hovermode="x unified")
            show_plot(port_fig, "portfolio")

            if len(portfolio_tickers) > 1:
                port_rets_df = pd.concat([comp_returns[s] for s in portfolio_tickers], axis=1, join="inner")
                port_rets_df.columns = portfolio_tickers
                corr_mat = port_rets_df.corr()
                st.markdown("### Correlaciones dentro del portafolio")
                st.dataframe(corr_mat.style.format("{:.2f}"), use_container_width=True)
        else:
            st.write(
                "Configura tickers y pesos válidos en el panel lateral para ver el análisis de portafolio.\n\n"
                "Recuerda: los pesos deben tener el mismo número de elementos que los activos del portafolio."
            )

    # TAB 8: Datos crudos
    with t8:
        st.subheader("Datos crudos del activo principal")
        st.download_button(
            label="Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{start}_{end}_{interval}.csv",
            mime="text/csv",
        )
        st.dataframe(df, use_container_width=True, height=500)

    st.divider()
    st.markdown("**© 2025 FinanzasUP. Desarrollado por Carlos Santana — FinanzasUP.**")


if __name__ == "__main__":
    main()

