"""
========================================================================================
HOSE V4 - QUANTITATIVE TOP N STOCKS SELECTION & PORTFOLIO OPTIMIZATION WEB APP
Nền tảng kiểm định chiến lược định lượng cổ phiếu Việt Nam (Nested Walk-Forward Framework)
Tuân thủ nghiêm ngặt nguyên tắc No Look-Ahead Bias & Causal Event-Driven Backtesting
========================================================================================
"""

import os
import io
import re
import math
import glob
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT & GIAO DIỆN
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="HOSE V4 | Định Lượng TOP 5 & Tối Ưu Danh Mục",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách giao diện tài chính chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
    }
    .champion-box {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# 2. HẰNG SỐ & ĐỊNH NGHĨA ALIASES
# --------------------------------------------------------------------------------------
BENCHMARK_ALIASES = {"VNINDEX", "VN-INDEX", "VNI", "^VNINDEX", "VNINDEXINDEX"}
TRADING_DAYS = 252

FA_ALIASES = {
    "ROE": ["roe", "return_on_equity", "returnonequity"],
    "ROA": ["roa", "return_on_assets", "returnonassets"],
    "RevenueGrowth": ["revenue_growth", "sales_growth", "revenuegrowth", "tangtruongdoanhthu"],
    "EPSGrowth": ["eps_growth", "epsgrowth", "tangtruongeps"],
    "GrossMargin": ["gross_margin", "grossmargin", "bienloinhuangop"],
    "OperatingMargin": ["operating_margin", "operatingmargin"],
    "NetMargin": ["net_margin", "netmargin", "bienloinhuanrong"],
    "CurrentRatio": ["current_ratio", "currentratio"],
    "DebtToEquity": ["debt_to_equity", "de", "d_e", "debttoequity"],
    "PE": ["pe", "p_e", "price_to_earnings", "pricetoearnings"],
    "PB": ["pb", "p_b", "price_to_book", "pricetobook"],
    "EVEBITDA": ["ev_ebitda", "evebitda"],
}

ACTIVE_MODEL_SPECS = {
    "DYN_EW_REGIME":      {"family": "EW",        "regime": True, "mask": True, "name": "Equal Weight (EW) + Regime"},
    "DYN_MPT_LW_REGIME":  {"family": "MPT_LW",    "regime": True, "mask": True, "name": "Mean-Variance Ledoit-Wolf (MPT-LW) + Regime"},
    "DYN_MINVAR_REGIME":  {"family": "MINVAR_LW", "regime": True, "mask": True, "name": "Minimum Variance (MinVar-LW) + Regime"},
    "DYN_HRP_REGIME":     {"family": "HRP",       "regime": True, "mask": True, "name": "Hierarchical Risk Parity (HRP) + Regime"},
    "DYN_BLEND50_REGIME": {"family": "BLEND50",   "regime": True, "mask": True, "name": "Blend 50/50 (MPT-LW + EW) + Regime"},
}


# --------------------------------------------------------------------------------------
# 3. CÁC HÀM XỬ LÝ DỮ LIỆU & CHUẨN HÓA (NO FUTURE LEAKAGE)
# --------------------------------------------------------------------------------------
def _canon(s):
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

def find_col(df, candidates):
    lookup = {_canon(c): c for c in df.columns}
    for c in candidates:
        if _canon(c) in lookup:
            return lookup[_canon(c)]
    for c in candidates:
        can_c = _canon(c)
        for k, orig in lookup.items():
            if can_c in k or k in can_c:
                return orig
    return None

def parse_dates_safely(s):
    s = pd.Series(s)
    x = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return x

def read_file_flexibly(file_input):
    """Hỗ trợ đọc cả file upload BytesIO hoặc đường dẫn file CSV/Excel với đa dạng encoding."""
    if hasattr(file_input, "name"):
        fname = file_input.name
    else:
        fname = str(file_input)

    if fname.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_input)
    else:
        # Thử các bảng mã phổ biến của tiếng Việt và Excel
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "cp1258"]
        for enc in encodings:
            try:
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                return pd.read_csv(file_input, encoding=enc)
            except Exception:
                continue
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        return pd.read_csv(file_input, encoding="utf-8", errors="replace")

def standardize_ohlcv(df_raw, use_adjusted=True):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ticker_col = find_col(df, ["ticker", "symbol", "code", "stock_code", "mack", "ma", "m"])
    date_col = find_col(df, ["date", "datetime", "trading_date", "time", "ngay", "ngy"])
    volume_col = find_col(df, ["volume", "vol", "trading_volume", "khoiluong", "khoplenh", "khilngkhplnh"])

    raw_cols = {
        "Open": find_col(df, ["open", "open_price", "giamocua", "gimca", "gia_mo_cua"]),
        "High": find_col(df, ["high", "high_price", "giacaonhat", "gicaonht", "gia_cao_nhat"]),
        "Low": find_col(df, ["low", "low_price", "giathapnhat", "githpnht", "gia_thap_nhat"]),
        "Close": find_col(df, ["close", "close_price", "giadongcua", "gingca", "gia_dong_cua"]),
    }
    adj_cols = {
        "Open": find_col(df, ["adj_open", "adjusted_open"]),
        "High": find_col(df, ["adj_high", "adjusted_high"]),
        "Low": find_col(df, ["adj_low", "adjusted_low"]),
        "Close": find_col(df, ["adj_close", "adjusted_close", "adjclose", "giadieuchinh", "giiuchnh", "gia_dieu_chinh"]),
    }

    if ticker_col is None or date_col is None:
        raise ValueError(f"Không nhận diện được cột Mã CK (Ticker) hoặc Ngày (Date). Cột hiện có: {list(df.columns)}")

    use_adj = bool(use_adjusted and all(v is not None for v in adj_cols.values()))
    pcols = adj_cols if use_adj else raw_cols
    if not all(v is not None for v in pcols.values()):
        # Fallback nếu thiếu một số cột raw nhưng có các biến thể tên cột
        for k in ["Open", "High", "Low", "Close"]:
            if pcols[k] is None:
                raise ValueError(f"Thiếu cột giá: {k}. Hãy kiểm tra cấu trúc file dữ liệu.")

    out = pd.DataFrame({
        "Ticker": df[ticker_col].astype(str).str.strip().str.upper(),
        "Date": parse_dates_safely(df[date_col]),
        "Open": pd.to_numeric(df[pcols["Open"]].astype(str).str.replace(",", ""), errors="coerce"),
        "High": pd.to_numeric(df[pcols["High"]].astype(str).str.replace(",", ""), errors="coerce"),
        "Low": pd.to_numeric(df[pcols["Low"]].astype(str).str.replace(",", ""), errors="coerce"),
        "Close": pd.to_numeric(df[pcols["Close"]].astype(str).str.replace(",", ""), errors="coerce"),
        "Volume": pd.to_numeric(df[volume_col].astype(str).str.replace(",", ""), errors="coerce") if volume_col else 0.0,
    })
    out = out.dropna(subset=["Ticker", "Date", "Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)

    valid_ohlc = (
        (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)) &
        (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))
    )
    out = out.loc[valid_ohlc]
    out = out.drop_duplicates(["Ticker", "Date"], keep="last")
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out, "ADJUSTED" if use_adj else "RAW"

def data_quality_table(data):
    rows = []
    for t, g in data.groupby("Ticker"):
        rows.append({
            "Ticker": t,
            "Start": g["Date"].min().strftime("%Y-%m-%d"),
            "End": g["Date"].max().strftime("%Y-%m-%d"),
            "Sessions": g["Date"].nunique(),
            "MissingVolumePct": 100 * g["Volume"].isna().mean(),
            "MedianTradedValue": (g["Close"] * g["Volume"]).median(),
        })
    return pd.DataFrame(rows).sort_values("Sessions", ascending=False).reset_index(drop=True)

def standardize_fa(df, allow_undated=False):
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    ticker_col = find_col(d, ["ticker", "symbol", "code", "stock_code", "mack", "ma", "m"])
    date_col = find_col(d, ["effective_date", "publish_date", "announcement_date", "report_date", "date", "ngay"])
    if ticker_col is None:
        return None

    out = pd.DataFrame({"Ticker": d[ticker_col].astype(str).str.strip().str.upper()})
    if date_col is not None:
        out["Date"] = parse_dates_safely(d[date_col])
    elif allow_undated:
        out["Date"] = pd.Timestamp("1900-01-01")
    else:
        return None

    found = 0
    for canon_name, aliases in FA_ALIASES.items():
        c = find_col(d, aliases)
        if c is not None:
            out[canon_name] = pd.to_numeric(d[c], errors="coerce")
            found += 1
    if found == 0:
        return None
    return out.dropna(subset=["Ticker", "Date"]).sort_values(["Ticker", "Date"])

def load_sector_map(df_raw):
    if df_raw is None or len(df_raw) == 0:
        return {}
    src = df_raw.copy()
    ticker_col = find_col(src, ["ticker", "symbol", "code", "stock_code", "mack", "ma", "m"])
    sector_col = find_col(src, ["sector", "industry", "industry_group", "nganh", "phan_nganh"])
    if ticker_col is None or sector_col is None:
        return {}
    m = src[[ticker_col, sector_col]].dropna().drop_duplicates(ticker_col, keep="last")
    m[ticker_col] = m[ticker_col].astype(str).str.upper().str.strip()
    return dict(zip(m[ticker_col], m[sector_col].astype(str)))

def fa_snapshot_score(fa, asof_date, tickers):
    if fa is None or fa.empty:
        return pd.DataFrame(index=tickers)
    x = fa[(fa["Date"] <= pd.Timestamp(asof_date)) & (fa["Ticker"].isin(tickers))].copy()
    if x.empty:
        return pd.DataFrame(index=tickers)
    x = x.sort_values("Date").groupby("Ticker", as_index=False).tail(1).set_index("Ticker")
    metrics = [c for c in FA_ALIASES if c in x.columns]
    if not metrics:
        return pd.DataFrame(index=tickers)

    lower_better = {"DebtToEquity", "PE", "PB", "EVEBITDA"}
    score_parts = []
    for c in metrics:
        s = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if c in {"PE", "PB", "EVEBITDA"}:
            s = s.where(s > 0)
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            s = s.clip(lo, hi)
        pct = s.rank(pct=True)
        if c in lower_better:
            pct = 1 - pct + (1 / max(len(pct.dropna()), 1))
        score_parts.append(pct.rename(c))

    z = pd.concat(score_parts, axis=1)
    out = pd.DataFrame(index=x.index)
    out["FA_MetricCount"] = z.notna().sum(axis=1)
    out["FA_Score"] = z.mean(axis=1) * 100
    return out.reindex(tickers)


# --------------------------------------------------------------------------------------
# 4. CHIA TẬP DỮ LIỆU & PRICE DICTIONARY / BENCHMARK PROXY
# --------------------------------------------------------------------------------------
def resolve_auto_splits(data, split_mode="auto", train_prop=0.60, val_prop=0.20, test_prop=0.20,
                         min_total=420, min_train=252, min_val=63, min_test=100,
                         prefer_last_full_year=True, last_year_min_sessions=180):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["Date"].unique())))
    n = len(dates)
    if n < min_total:
        raise ValueError(f"Dữ liệu có {n} phiên. Cần tối thiểu {min_total} phiên để chia Core Train/Validation/Final Test.")

    if split_mode == "proportions":
        i_train = max(min_train, int(n * train_prop))
        i_val_end = max(i_train + min_val, int(n * (train_prop + val_prop)))
        i_val_end = min(i_val_end, n - min_test)
        if i_val_end <= i_train:
            raise ValueError("Không đủ dữ liệu để chia tỷ lệ hợp lệ.")
        core_end = dates[i_train - 1]
        val_start, val_end = dates[i_train], dates[i_val_end - 1]
        test_start, test_end = dates[i_val_end], dates[-1]
        return dates, core_end, val_start, val_end, test_start, test_end

    # Auto mode: Ưu tiên năm dương lịch cuối cùng làm Final Test nếu đủ số phiên
    last_year = dates[-1].year
    last_year_dates = dates[dates.year == last_year]
    pre_last_year_count = int((dates < last_year_dates[0]).sum()) if len(last_year_dates) else 0

    if (
        prefer_last_full_year and
        len(last_year_dates) >= last_year_min_sessions and
        pre_last_year_count >= min_train + min_val
    ):
        test_start, test_end = last_year_dates[0], last_year_dates[-1]
        pretest = dates[dates < test_start]
    else:
        test_n = max(min_test, min(252, int(round(test_prop * n))))
        test_start, test_end = dates[-test_n], dates[-1]
        pretest = dates[dates < test_start]

    val_n = max(min_val, min(252, int(round(0.25 * len(pretest)))))
    val_n = min(val_n, len(pretest) - min_train)
    if val_n < min_val:
        raise ValueError("Không đủ lịch sử trước Test để tách riêng tập Validation.")

    val_start, val_end = pretest[-val_n], pretest[-1]
    core_end = pretest[-val_n - 1]
    return dates, core_end, val_start, val_end, test_start, test_end

def build_price_dict(data):
    return {
        t: g[["Date", "Open", "High", "Low", "Close", "Volume"]]
             .drop_duplicates("Date").sort_values("Date").set_index("Date")
        for t, g in data.groupby("Ticker")
    }

def build_synthetic_market_proxy(price_dict, exclude=None):
    exclude = set(exclude or [])
    closes = [px["Close"].rename(t) for t, px in price_dict.items() if t not in exclude]
    m = pd.concat(closes, axis=1).sort_index()
    ret = m.pct_change(fill_method=None)
    ewret = ret.mean(axis=1, skipna=True).fillna(0)
    close = 1000.0 * (1.0 + ewret).cumprod()
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0}, index=close.index)


# --------------------------------------------------------------------------------------
# 5. CHỈ BÁO KỸ THUẬT & TA SCORE (0-10)
# --------------------------------------------------------------------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi_wilder(close, window=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    al = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi.loc[(al == 0) & (ag > 0)] = 100.0
    rsi.loc[(ag == 0) & (al > 0)] = 0.0
    return rsi

def atr_wilder(df, window=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def adx_wilder(df, window=14):
    high, low = df["High"], df["Low"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = atr_wilder(df, window)
    plus_di = 100.0 * plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return adx, plus_di, minus_di

def calculate_ta_features(df):
    out = df.copy().sort_index()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)

    ema12, ema26 = ema(out["Close"], 12), ema(out["Close"], 26)
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = ema(out["MACD"], 9)

    out["ADX"], out["Plus_DI"], out["Minus_DI"] = adx_wilder(out, 14)
    out["ATR"] = atr_wilder(out, 14)
    out["ATR_pct"] = 100.0 * out["ATR"] / out["Close"].replace(0, np.nan)
    out["ATR_pct_Q90"] = out["ATR_pct"].rolling(126, min_periods=40).quantile(0.90)

    out["Volume_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)
    out["OBV"] = (np.sign(out["Close"].diff()).fillna(0) * out["Volume"]).cumsum()
    out["OBV_EMA20"] = ema(out["OBV"], 20)

    out["RSI"] = rsi_wilder(out["Close"], 14)
    out["ROC10"] = out["Close"].pct_change(10) * 100.0
    out["High_52W"] = out["High"].rolling(252, min_periods=150).max()
    out["Close_to_52W_High"] = out["Close"] / out["High_52W"].replace(0, np.nan)
    out["Avg_Traded_Value_60"] = (out["Close"] * out["Volume"]).rolling(60, min_periods=20).mean()

    out["Momentum63"] = out["Close"].pct_change(63)
    out["Momentum126"] = out["Close"].pct_change(126)
    out["Momentum252"] = out["Close"].pct_change(252)
    out["Vol63"] = out["Close"].pct_change().rolling(63, min_periods=40).std() * np.sqrt(TRADING_DAYS)
    rollmax = out["Close"].rolling(126, min_periods=60).max()
    out["DD126"] = out["Close"] / rollmax - 1.0

    # TA Score 0-10 chuẩn nhóm 1
    out["Trend_Score"] = (
        (out["Close"] > out["EMA50"]).astype(int) * 2 +
        (out["EMA50"] > out["EMA200"]).astype(int) * 2 +
        (out["MACD"] > out["MACD_Signal"]).astype(int)
    ).clip(0, 5)

    momentum_core = ((out["RSI"] >= 45) & (out["RSI"] <= 75)) | (out["ROC10"] > 0)
    out["Momentum_Score"] = (
        momentum_core.astype(int) * 2 +
        (out["Close_to_52W_High"] >= 0.85).astype(int)
    ).clip(0, 3)

    out["Volume_Score"] = ((out["Volume_Ratio"] >= 1.0) | (out["OBV"] > out["OBV_EMA20"])).astype(int)
    out["Volatility_Score"] = (
        out["ATR_pct"].notna() & out["ATR_pct_Q90"].notna() &
        (out["ATR_pct"] <= out["ATR_pct_Q90"])
    ).astype(int)

    out["TA_Score"] = (
        out["Trend_Score"] + out["Momentum_Score"] +
        out["Volume_Score"] + out["Volatility_Score"]
    ).clip(0, 10).astype(int)

    return out

def stateful_active_from_score(score, buy_threshold, exit_threshold):
    state = 0
    vals = []
    for v in score.fillna(-np.inf):
        if v >= buy_threshold:
            state = 1
        elif v <= exit_threshold:
            state = 0
        vals.append(state)
    return pd.Series(vals, index=score.index, dtype=int)


# --------------------------------------------------------------------------------------
# 6. PERFORMANCE METRICS & MÔ PHỎNG 1 MÃ (EVENT-DRIVEN OPEN T+1)
# --------------------------------------------------------------------------------------
def annualized_sharpe(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    return np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=1) if len(r) > 1 and r.std(ddof=1) > 1e-12 else np.nan

def annualized_sortino(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    neg = r[r < 0]
    return np.sqrt(TRADING_DAYS) * r.mean() / neg.std(ddof=1) if len(neg) > 1 and neg.std(ddof=1) > 1e-12 else np.nan

def metrics_from_equity(eq):
    eq = pd.Series(eq).dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    days_cnt = (eq.index[-1] - eq.index[0]).days
    years = max(days_cnt / 365.25, 1.0 / 365.25)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    dd = eq / eq.cummax() - 1.0
    mdd = dd.min()
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan
    return {
        "Return [%]": total * 100.0,
        "CAGR [%]": cagr * 100.0,
        "Sharpe": annualized_sharpe(r),
        "Sortino": annualized_sortino(r),
        "Volatility [%]": vol * 100.0 if pd.notna(vol) else np.nan,
        "MaxDD [%]": mdd * 100.0 if pd.notna(mdd) else np.nan,
        "Calmar": cagr / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan,
    }

def simulate_single_stock(features, start, end, buy_threshold, exit_threshold,
                          initial_cash=1_000_000.0, commission=0.0015, slippage=0.0005):
    active_close = stateful_active_from_score(features["TA_Score"], buy_threshold, exit_threshold)
    active_open = active_close.shift(1).fillna(0).astype(int)
    x = features.loc[(features.index >= pd.Timestamp(start)) & (features.index <= pd.Timestamp(end))].copy()
    active_open = active_open.reindex(x.index).fillna(0).astype(int)
    if x.empty:
        return {}

    cash, shares, state, trades = float(initial_cash), 0.0, 0, 0
    eq = []
    for dt, row in x.iterrows():
        d = int(active_open.loc[dt])
        op, cl = float(row["Open"]), float(row["Close"])
        if d != state:
            if d == 1:
                px = op * (1.0 + slippage)
                qty = math.floor(cash / (px * (1.0 + commission)))
                if qty > 0:
                    cash -= qty * px * (1.0 + commission)
                    shares = qty
                    state = 1
                    trades += 1
            else:
                px = op * (1.0 - slippage)
                cash += shares * px * (1.0 - commission)
                shares = 0.0
                state = 0
                trades += 1
        eq.append((dt, cash + shares * cl))

    m = metrics_from_equity(pd.Series(dict(eq)).sort_index())
    m["Trades"] = trades
    return m

def make_equal_folds(dates, start, end, n_folds=3):
    d = pd.DatetimeIndex(dates)
    d = d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    if len(d) < n_folds * 20:
        n_folds = max(1, len(d) // 20)
    chunks = np.array_split(d, n_folds)
    folds = []
    for i, c in enumerate(chunks, 1):
        if len(c) >= 20:
            folds.append({"Fold": f"F{i}", "start": pd.Timestamp(c[0]), "end": pd.Timestamp(c[-1])})
    return folds

def build_universe(price_dict, asof, benchmark_ticker=None, use_liq_filter=True,
                   liq_lookback=60, liq_min_pct=0.20, min_hist_floor=180, min_hist_cap=400):
    all_dates = set()
    for px in price_dict.values():
        all_dates.update(px.loc[:asof].index)
    pre_sessions = len(all_dates)
    min_hist = min(min_hist_cap, max(min_hist_floor, int(0.55 * pre_sessions)))

    rows = []
    for t, px in price_dict.items():
        if t == benchmark_ticker:
            continue
        h = px.loc[:asof]
        if h.empty:
            continue
        avg_value = (h["Close"] * h["Volume"]).tail(liq_lookback).mean()
        rows.append({
            "Ticker": t,
            "HistorySessions": len(h),
            "AvgTradedValue": avg_value,
            "EligibleHistory": len(h) >= min_hist,
        })

    u = pd.DataFrame(rows)
    if u.empty:
        return u
    ok = u["EligibleHistory"] & u["AvgTradedValue"].notna()
    u.loc[ok, "LiquidityPercentile"] = u.loc[ok, "AvgTradedValue"].rank(pct=True)
    u["PassLiquidity"] = (
        u["LiquidityPercentile"].fillna(0) >= liq_min_pct
        if use_liq_filter else True
    )
    u["Candidate"] = u["EligibleHistory"] & u["PassLiquidity"]
    return u.sort_values("AvgTradedValue", ascending=False).reset_index(drop=True)

def choose_global_ta_thresholds(feature_store, candidates, folds,
                                buy_choices=[5, 6, 7], exit_choices=[2, 3, 4],
                                stability_penalty=0.20):
    trials = []
    for b, e in product(buy_choices, exit_choices):
        if e >= b:
            continue
        stock_scores = []
        for t in candidates:
            ms = []
            for f in folds:
                m = simulate_single_stock(feature_store[t], f["start"], f["end"], b, e)
                if m:
                    sh = m.get("Sharpe", np.nan)
                    sh = -3.0 if not np.isfinite(sh) else float(sh)
                    ms.append((sh, float(m.get("Return [%]", 0)), float(m.get("MaxDD [%]", -100))))
            if ms:
                a = np.array(ms, float)
                stock_scores.append([np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), np.nanmin(a[:, 2])])
        if not stock_scores:
            continue

        z = np.array(stock_scores, float)
        med_sh = float(np.nanmedian(z[:, 0]))
        sd_sh = float(np.nanstd(z[:, 0]))
        med_ret = float(np.nanmedian(z[:, 1]))
        med_dd = float(np.nanmedian(z[:, 2]))
        score = med_sh - stability_penalty * sd_sh + 0.005 * med_ret + 0.003 * med_dd
        trials.append({
            "BuyThreshold": b,
            "ExitThreshold": e,
            "GlobalCVScore": score,
            "MedianSharpe": med_sh,
            "SharpeStd": sd_sh,
            "MedianReturnPct": med_ret,
            "MedianMaxDDPct": med_dd,
            "NStocks": len(stock_scores),
        })

    tr = pd.DataFrame(trials).sort_values("GlobalCVScore", ascending=False).reset_index(drop=True)
    if tr.empty:
        return 6, 3, tr
    best = tr.iloc[0]
    return int(best["BuyThreshold"]), int(best["ExitThreshold"]), tr


# --------------------------------------------------------------------------------------
# 7. XẾP HẠNG & LỰA CHỌN TOP N CỔ PHIẾU (POINT-IN-TIME AS-OF)
# --------------------------------------------------------------------------------------
def trailing_stock_quality(features, asof, b, e, lookback=126):
    h = features.loc[:asof]
    if len(h) < 40:
        return {"TrailingSharpe": -3.0, "TrailingReturnPct": -100.0, "TrailingMaxDDPct": -100.0}
    sub_idx = h.index[-min(lookback, len(h)):]
    m = simulate_single_stock(features, sub_idx[0], sub_idx[-1], b, e)
    sh = m.get("Sharpe", np.nan) if m else np.nan
    return {
        "TrailingSharpe": -3.0 if not np.isfinite(sh) else float(sh),
        "TrailingReturnPct": float(m.get("Return [%]", -100.0)) if m else -100.0,
        "TrailingMaxDDPct": float(m.get("MaxDD [%]", -100.0)) if m else -100.0,
    }

def rank_and_select_asof(feature_store, price_dict, benchmark_features, benchmark_ticker,
                         asof, b, e, fa=None, sector_map=None, top_n=5,
                         min_ta_score=4, max_vol_pct=0.75, min_dd_pct=0.15, max_rsi=80.0,
                         max_pairwise_corr=0.80, corr_lookback=126,
                         weights_dict=None, filter_mode="auto",
                         fa_weight=0.30, ta_weight=0.70, max_stocks_per_sector=2,
                         min_fa_metrics=2, min_fa_coverage=0.50):
    sector_map = sector_map or {}
    weights_dict = weights_dict or {
        "trailing": 0.25, "ta": 0.20, "rel_strength": 0.25,
        "low_vol": 0.10, "liquidity": 0.05, "momentum": 0.15
    }

    universe = build_universe(price_dict, asof, benchmark_ticker)
    if universe.empty:
        raise ValueError(f"Universe rỗng tại {asof}")
    candidates = universe.loc[universe["Candidate"], "Ticker"].tolist()
    bench = benchmark_features.loc[:asof]
    bench_m126 = bench["Close"].pct_change(126).iloc[-1] if len(bench) > 126 else np.nan
    uidx = universe.set_index("Ticker")

    rows = []
    for t in candidates:
        feat = feature_store[t].loc[:asof]
        if feat.empty:
            continue
        snap = feat.iloc[-1]
        tq = trailing_stock_quality(feature_store[t], asof, b, e)
        rows.append({
            "Ticker": t,
            **tq,
            "TA_Score": float(snap.get("TA_Score", np.nan)),
            "Momentum63": float(snap.get("Momentum63", np.nan)),
            "Momentum126": float(snap.get("Momentum126", np.nan)),
            "Momentum252": float(snap.get("Momentum252", np.nan)),
            "RelativeStrength126": (
                float(snap.get("Momentum126", np.nan) - bench_m126)
                if pd.notna(bench_m126) else float(snap.get("Momentum126", np.nan))
            ),
            "Vol63": float(snap.get("Vol63", np.nan)),
            "RSI": float(snap.get("RSI", np.nan)),
            "LiquidityPercentile": float(uidx.loc[t, "LiquidityPercentile"]),
            "Sector": sector_map.get(t, "UNKNOWN"),
        })

    r = pd.DataFrame(rows)
    if r.empty:
        raise ValueError(f"Không có cổ phiếu đủ điều kiện xếp hạng tại {asof}")

    # Gate 1: Điểm TA Score tối thiểu
    r["PassTA"] = r["TA_Score"].fillna(0) >= min_ta_score
    if int(r["PassTA"].sum()) >= top_n:
        r = r[r["PassTA"]].copy()

    # Gate 2: Giới hạn rủi ro Volatility / Drawdown / RSI
    vol_cap = r["Vol63"].quantile(max_vol_pct)
    dd_floor = r["TrailingMaxDDPct"].quantile(min_dd_pct)
    r["PassRiskGate"] = (
        (r["Vol63"].isna() | (r["Vol63"] <= vol_cap)) &
        (r["TrailingMaxDDPct"].isna() | (r["TrailingMaxDDPct"] >= dd_floor)) &
        (r["RSI"].isna() | (r["RSI"] <= max_rsi))
    )
    if int(r["PassRiskGate"].sum()) >= top_n:
        r = r[r["PassRiskGate"]].copy()

    # Cross-sectional percentile ranking
    r["P_Trailing"] = r["TrailingSharpe"].rank(pct=True)
    r["P_TA"] = r["TA_Score"].rank(pct=True)
    r["P_RS"] = r["RelativeStrength126"].rank(pct=True)
    r["P_LowVol"] = (-r["Vol63"]).rank(pct=True)
    r["P_Liq"] = r["LiquidityPercentile"].rank(pct=True)
    mommix = 0.6 * r["Momentum126"].fillna(r["Momentum63"]) + 0.4 * r["Momentum63"].fillna(0)
    r["P_Momentum"] = mommix.rank(pct=True)

    r["TA_Composite"] = (
        weights_dict["trailing"] * r["P_Trailing"] +
        weights_dict["ta"] * r["P_TA"] +
        weights_dict["rel_strength"] * r["P_RS"] +
        weights_dict["low_vol"] * r["P_LowVol"] +
        weights_dict["liquidity"] * r["P_Liq"] +
        weights_dict["momentum"] * r["P_Momentum"]
    ) * 100.0

    # Phân tích cơ bản FA Point-in-time
    fas = fa_snapshot_score(fa, asof, r["Ticker"].tolist())
    if not fas.empty:
        r = r.merge(fas.reset_index().rename(columns={"index": "Ticker"}), on="Ticker", how="left")
    else:
        r["FA_MetricCount"] = 0
        r["FA_Score"] = np.nan

    fa_eligible = r["FA_MetricCount"].fillna(0) >= min_fa_metrics
    fa_coverage = float(fa_eligible.mean()) if len(r) else 0.0
    use_fa = (
        filter_mode == "FA_TA" or
        (filter_mode == "auto" and fa is not None and fa_coverage >= min_fa_coverage)
    )
    if filter_mode == "TA":
        use_fa = False

    if use_fa:
        neutral = r["FA_Score"].median() if r["FA_Score"].notna().any() else 50.0
        r["FinalScore"] = ta_weight * r["TA_Composite"] + fa_weight * r["FA_Score"].fillna(neutral)
        mode_used = "FA+TA"
    else:
        r["FinalScore"] = r["TA_Composite"]
        mode_used = "TA"

    r = r.sort_values(["FinalScore", "TrailingSharpe"], ascending=False).reset_index(drop=True)

    # Đa dạng hóa: Sector Cap & Pairwise Correlation Cap
    corr_tickers = r["Ticker"].tolist()
    close_list = [feature_store[t].loc[:asof, "Close"].rename(t) for t in corr_tickers]
    corr_px = pd.concat(close_list, axis=1, join="inner").tail(corr_lookback + 1) if close_list else pd.DataFrame()
    corr = corr_px.pct_change(fill_method=None).dropna().corr() if len(corr_px) > 20 else pd.DataFrame()

    selected, sector_count = [], {}
    for _, row in r.iterrows():
        t, sec = row["Ticker"], row["Sector"]
        if sector_map and sec != "UNKNOWN" and sector_count.get(sec, 0) >= max_stocks_per_sector:
            continue
        if selected and not corr.empty:
            too_close = any(
                t in corr.index and x in corr.columns and
                pd.notna(corr.loc[t, x]) and abs(corr.loc[t, x]) > max_pairwise_corr
                for x in selected
            )
            if too_close:
                continue
        selected.append(t)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for t in r["Ticker"]:
            if t not in selected:
                selected.append(t)
            if len(selected) >= top_n:
                break

    locked = r[r["Ticker"].isin(selected)].copy()
    locked["SelectedOrder"] = locked["Ticker"].map({t: i + 1 for i, t in enumerate(selected)})
    locked["FilterModeUsed"] = mode_used
    locked["FA_Coverage"] = fa_coverage
    locked["AsOfDate"] = pd.Timestamp(asof)
    locked = locked.sort_values("SelectedOrder")
    return r, locked, selected, mode_used, fa_coverage

def first_session_each_period(calendar, start, end, freq="Q"):
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    key = idx.to_period(freq)
    firsts = s.groupby(key).first().tolist()
    if idx[0] not in firsts:
        firsts = [idx[0]] + firsts
    return sorted(set(pd.Timestamp(x) for x in firsts))

def previous_session(calendar, dt):
    idx = pd.DatetimeIndex(calendar)
    prev = idx[idx < pd.Timestamp(dt)]
    return pd.Timestamp(prev[-1]) if len(prev) else None

def build_selection_schedule(feature_store, price_dict, benchmark_ticker, benchmark_features,
                             calendar, start, end, b, e, fa=None, sector_map=None,
                             selection_freq="Q", top_n=5, ranking_kwargs=None):
    ranking_kwargs = ranking_kwargs or {}
    effective_dates = first_session_each_period(calendar, start, end, selection_freq)
    rows = []
    details = {}

    for eff in effective_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        ranking, locked, selected, mode, fa_cov = rank_and_select_asof(
            feature_store, price_dict, benchmark_features, benchmark_ticker,
            asof, b, e, fa, sector_map, top_n=top_n, **ranking_kwargs
        )
        details[pd.Timestamp(eff)] = {"ranking": ranking, "locked": locked}
        row = {
            "EffectiveDate": pd.Timestamp(eff),
            "AsOfDate": pd.Timestamp(asof),
            "FilterMode": mode,
            "FACoverage": fa_cov,
        }
        for i in range(top_n):
            row[f"Top{i+1}"] = selected[i] if i < len(selected) else None
        rows.append(row)

    schedule = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    if schedule.empty:
        raise ValueError("Không tạo được selection schedule")
    return schedule, details

def selected_tickers_asof(selection_schedule, effective_date, top_n=5):
    x = selection_schedule[selection_schedule["EffectiveDate"] <= pd.Timestamp(effective_date)]
    if x.empty:
        return []
    row = x.iloc[-1]
    return [row[f"Top{i+1}"] for i in range(top_n) if pd.notna(row.get(f"Top{i+1}"))]


# --------------------------------------------------------------------------------------
# 8. CÁC MÔ HÌNH PHÂN BỔ TỶ TRỌNG (EW, MPT-LW, MINVAR-LW, HRP, BLEND50)
# --------------------------------------------------------------------------------------
def close_matrix(price_dict, tickers, end=None, start=None):
    series = [price_dict[t]["Close"].rename(t) for t in tickers if t in price_dict]
    if not series:
        return pd.DataFrame()
    m = pd.concat(series, axis=1, join="inner").sort_index().dropna(how="any")
    if start is not None:
        m = m.loc[m.index >= pd.Timestamp(start)]
    if end is not None:
        m = m.loc[m.index <= pd.Timestamp(end)]
    return m

def equal_weights(tickers):
    tickers = list(tickers)
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float) if tickers else pd.Series(dtype=float)

def robust_mu(prices, lookback=252, mu_shrinkage=0.50):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if r.empty:
        return pd.Series(0.0, index=prices.columns)
    span = max(20, min(lookback, len(r)))
    raw = r.ewm(span=span, adjust=False).mean().iloc[-1] * TRADING_DAYS
    grand = float(raw.mean())
    return mu_shrinkage * raw + (1.0 - mu_shrinkage) * grand

def lw_cov(prices, lookback=252):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if len(r) < 30:
        return r.cov() * TRADING_DAYS
    est = LedoitWolf().fit(r.values)
    return pd.DataFrame(est.covariance_ * TRADING_DAYS, index=r.columns, columns=r.columns)

def max_sharpe_weights(mu, cov, max_weight=0.50, l2_gamma=0.03, risk_free_rate=0.0):
    tickers = list(mu.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1.0 / n)
    muv = mu.values.astype(float)
    cv = cov.reindex(index=tickers, columns=tickers).fillna(0).values.astype(float) + np.eye(n) * 1e-8

    def obj(w):
        ret = float(w @ muv) - risk_free_rate
        vol = math.sqrt(max(float(w @ cv @ w), 1e-12))
        return -(ret / vol) + l2_gamma * float(np.sum(w * w))

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success or np.any(~np.isfinite(res.x)):
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=tickers) if w.sum() > 0 else equal_weights(tickers)

def min_variance_weights(cov, max_weight=0.50):
    tickers = list(cov.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1.0 / n)
    cv = cov.values.astype(float) + np.eye(n) * 1e-8

    def obj(w):
        return float(w @ cv @ w)

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=tickers)

def _cluster_var(cov, items):
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)

def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def hrp_weights(prices, method="single"):
    r = prices.pct_change(fill_method=None).dropna()
    if len(r) < 30 or prices.shape[1] < 2:
        return equal_weights(prices.columns)
    cov = r.cov() * TRADING_DAYS
    corr = r.corr().clip(-1, 1)
    dist = np.sqrt((1.0 - corr) / 2.0)
    link = linkage(squareform(dist.values, checks=False), method=method)
    ordered = corr.index[_quasi_diag(link)].tolist()

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            s = len(c) // 2
            c0, c1 = c[:s], c[s:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            a = 1.0 - v0 / (v0 + v1) if v0 + v1 > 0 else 0.5
            w[c0] *= a
            w[c1] *= (1.0 - a)
            nxt += [c0, c1]
        clusters = nxt
    return w.reindex(prices.columns).fillna(0) / w.sum()

def compute_weights(model_family, prices, weight_lookback=252, mpt_max_weight=0.50,
                    l2_gamma=0.03, mu_shrinkage=0.50, blend_weight=0.50, rf_rate=0.0):
    tickers = list(prices.columns)
    ew = equal_weights(tickers)
    if len(tickers) == 0:
        return ew
    if model_family == "EW":
        return ew

    cov = lw_cov(prices, weight_lookback)
    if model_family == "MINVAR_LW":
        return min_variance_weights(cov, max_weight=mpt_max_weight)
    if model_family == "HRP":
        return hrp_weights(prices.tail(weight_lookback))

    mu = robust_mu(prices, weight_lookback, mu_shrinkage=mu_shrinkage)
    mpt = max_sharpe_weights(mu, cov, max_weight=mpt_max_weight, l2_gamma=l2_gamma, risk_free_rate=rf_rate)
    if model_family == "MPT_LW":
        return mpt
    if model_family == "BLEND50":
        w = blend_weight * mpt + (1.0 - blend_weight) * ew
        return w / w.sum()
    raise ValueError("Model family không hỗ trợ: " + str(model_family))


# --------------------------------------------------------------------------------------
# 9. MARKET REGIME, STOCK ACTIVE MASK & DYNAMIC PORTFOLIO ENGINE
# --------------------------------------------------------------------------------------
def build_market_regime(benchmark_features, ema_span=200, slope_lookback=20, use_slope_confirm=True):
    x = benchmark_features.copy()
    ema_m = x["Close"].ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    slope = ema_m - ema_m.shift(slope_lookback)
    risk_close = x["Close"] > ema_m
    if use_slope_confirm:
        risk_close = risk_close & (slope > 0)
    return pd.DataFrame({
        "EMA_Market": ema_m,
        "EMA_Slope": slope,
        "RiskOn_Close": risk_close.fillna(False),
        "RiskOn_Open": risk_close.shift(1).fillna(False),
    }, index=x.index)

def build_stock_open_masks(feature_store, tickers, b, e):
    masks = {}
    for t in tickers:
        active_close = stateful_active_from_score(feature_store[t]["TA_Score"], b, e)
        masks[t] = active_close.shift(1).fillna(0).astype(bool)
    return pd.DataFrame(masks).sort_index()

def build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end,
                                  model_family, allocation_freq="M", weight_lookback=252,
                                  top_n=5, opt_kwargs=None):
    opt_kwargs = opt_kwargs or {}
    rebalance_dates = first_session_each_period(calendar, start, end, allocation_freq)
    rows = []
    all_tickers = set()

    for eff in rebalance_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        top = selected_tickers_asof(selection_schedule, eff, top_n=top_n)
        if not top:
            continue
        all_tickers.update(top)
        prices = close_matrix(price_dict, top, end=asof)
        if prices.empty or len(prices) < 30:
            w = equal_weights(top)
        else:
            w = compute_weights(model_family, prices.tail(weight_lookback + 1), weight_lookback=weight_lookback, **opt_kwargs)
        row = {"EffectiveDate": pd.Timestamp(eff), "AsOfDate": pd.Timestamp(asof)}
        row.update({t: float(w.get(t, 0.0)) for t in top})
        rows.append(row)

    if not rows:
        raise ValueError("Không tạo được weight schedule")

    meta = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    tickers = sorted(all_tickers)
    weights = meta.set_index("EffectiveDate").reindex(columns=tickers).fillna(0.0)
    return weights, meta[["EffectiveDate", "AsOfDate"]]

def build_static_buyhold_schedule(price_dict, selection_schedule, calendar, start, end, top_n=5):
    start = pd.Timestamp(start)
    top = selected_tickers_asof(selection_schedule, start, top_n=top_n)
    if not top:
        raise ValueError("Không có Top N ở đầu kỳ cho Buy & Hold")
    return pd.DataFrame([equal_weights(top)], index=[start])

def run_portfolio_engine_dynamic(price_dict, weight_schedule, calendar, start, end,
                                 market_regime_open=None, stock_active_open=None,
                                 use_market_regime=True, use_stock_mask=True,
                                 individual_trailing_stop=False, individual_stop_pct=0.12,
                                 portfolio_trailing_stop=False, portfolio_stop_pct=0.15,
                                 initial_cash=1_000_000_000.0, commission=0.0015,
                                 slippage=0.0005, fractional=False):
    tickers = list(weight_schedule.columns)
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        raise ValueError("Không có ngày giao dịch trong kỳ kiểm định.")

    opens = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    closes = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        if t in price_dict:
            opens[t] = price_dict[t]["Open"].reindex(idx)
            closes[t] = price_dict[t]["Close"].reindex(idx)

    ws = weight_schedule.sort_index().reindex(columns=tickers)
    ws = ws.reindex(ws.index.union(idx)).sort_index().ffill().reindex(idx).fillna(0.0)

    regime = (
        pd.Series(True, index=idx)
        if market_regime_open is None
        else pd.Series(market_regime_open).reindex(idx).ffill().fillna(False).astype(bool)
    )
    masks = (
        pd.DataFrame(True, index=idx, columns=tickers)
        if stock_active_open is None
        else stock_active_open.reindex(index=idx, columns=tickers).fillna(False).astype(bool)
    )

    cash = float(initial_cash)
    shares = pd.Series(0.0, index=tickers)
    last_close = pd.Series(np.nan, index=tickers)
    last_target = pd.Series(np.nan, index=tickers)
    entry_peak = pd.Series(np.nan, index=tickers)
    stop_blocked = pd.Series(False, index=tickers)
    portfolio_peak = initial_cash
    portfolio_blocked = False

    orders, eq_rows, exposure_rows = [], [], []
    turnover, rebals = 0.0, 0

    for dt in idx:
        op = opens.loc[dt].astype(float)
        cl = closes.loc[dt].astype(float)

        valid_close = cl.notna()
        last_close.loc[valid_close] = cl.loc[valid_close]
        mark_open = op.where(op.notna(), last_close)
        mark_close = cl.where(cl.notna(), last_close)

        base = ws.loc[dt].clip(lower=0)
        if base.sum() > 1.0 + 1e-8:
            base = base / base.sum()

        risk_on = bool(regime.loc[dt]) if use_market_regime else True
        active = masks.loc[dt].astype(bool) if use_stock_mask else pd.Series(True, index=tickers)
        for t in tickers:
            if not active[t]:
                stop_blocked[t] = False

        target = base.copy()
        if use_stock_mask:
            target *= active.astype(float)
        if individual_trailing_stop:
            target[stop_blocked] = 0.0
        if use_market_regime and not risk_on:
            target[:] = 0.0
        if portfolio_trailing_stop and portfolio_blocked:
            target[:] = 0.0

        # NGUYÊN TẮC: KHÔNG normalize lại sau mask/regime -> phần vốn dư tự động là Tiền mặt (Cash)
        target = target.clip(lower=0)
        if target.sum() > 1.0 + 1e-8:
            target /= target.sum()

        changed = last_target.isna().any() or (target - last_target.fillna(0)).abs().max() > 1e-10
        if changed:
            equity_open = cash + float((shares * mark_open.fillna(0)).sum())
            desired = shares.copy()

            for t in tickers:
                if pd.isna(op[t]) or op[t] <= 0:
                    continue
                if target[t] <= 0:
                    desired[t] = 0.0
                else:
                    px = op[t] * (1.0 + slippage)
                    raw_qty = equity_open * target[t] / (px * (1.0 + commission))
                    desired[t] = raw_qty if fractional else math.floor(raw_qty)

            # Bán trước
            for t in tickers:
                d = desired[t] - shares[t]
                if d < -1e-12 and pd.notna(op[t]) and op[t] > 0:
                    qty = -d
                    px = op[t] * (1.0 - slippage)
                    notional = qty * px
                    fee = notional * commission
                    cash += notional - fee
                    shares[t] -= qty
                    turnover += notional
                    orders.append({
                        "Date": dt, "Ticker": t, "Side": "SELL", "Shares": qty,
                        "ExecPrice": px, "Fee": fee, "Reason": "TARGET"
                    })
                    if shares[t] <= 1e-12:
                        entry_peak[t] = np.nan

            # Mua sau
            for t in tickers:
                d = desired[t] - shares[t]
                if d > 1e-12 and pd.notna(op[t]) and op[t] > 0:
                    px = op[t] * (1.0 + slippage)
                    unit = px * (1.0 + commission)
                    qty = min(d, cash / unit) if fractional else min(d, math.floor(cash / unit))
                    if qty > 0:
                        notional = qty * px
                        fee = notional * commission
                        was_zero = shares[t] <= 1e-12
                        cash -= notional + fee
                        shares[t] += qty
                        turnover += notional
                        orders.append({
                            "Date": dt, "Ticker": t, "Side": "BUY", "Shares": qty,
                            "ExecPrice": px, "Fee": fee, "Reason": "TARGET"
                        })
                        if was_zero and pd.notna(mark_close[t]):
                            entry_peak[t] = mark_close[t]

            last_target = target.copy()
            rebals += 1

        if individual_trailing_stop:
            for t in tickers:
                if shares[t] > 0 and pd.notna(mark_close[t]):
                    entry_peak[t] = max(entry_peak[t] if pd.notna(entry_peak[t]) else mark_close[t], mark_close[t])
                    if mark_close[t] <= entry_peak[t] * (1.0 - individual_stop_pct):
                        stop_blocked[t] = True

        equity = float(cash + (shares * mark_close.fillna(0)).sum())
        portfolio_peak = max(portfolio_peak, equity)
        if portfolio_trailing_stop and equity <= portfolio_peak * (1.0 - portfolio_stop_pct):
            portfolio_blocked = True

        invested = float((shares * mark_close.fillna(0)).sum())
        exposure = invested / equity if equity > 0 else 0.0
        exposure_rows.append(exposure)
        eq_rows.append((dt, equity))

    eq = pd.Series(dict(eq_rows)).sort_index()
    st_res = metrics_from_equity(eq)
    st_res.update({
        "#Orders": len(orders),
        "#Rebalances": rebals,
        "Turnover [%]": 100.0 * turnover / initial_cash,
        "AvgExposure [%]": 100.0 * np.mean(exposure_rows),
        "AvgCash [%]": 100.0 * (1.0 - np.mean(exposure_rows)),
    })
    return st_res, eq, pd.DataFrame(orders)

def benchmark_nav(benchmark_px, start, end, initial_cash=1_000_000_000.0):
    x = benchmark_px.loc[(benchmark_px.index >= pd.Timestamp(start)) & (benchmark_px.index <= pd.Timestamp(end)), "Close"].dropna()
    if x.empty:
        return pd.Series(dtype=float)
    return initial_cash * x / x.iloc[0]

def model_validation_score(stats, benchmark_stats):
    sh = stats.get("Sharpe", np.nan)
    sh = -3.0 if not np.isfinite(sh) else sh
    so = stats.get("Sortino", np.nan)
    so = -3.0 if not np.isfinite(so) else so
    ret = stats.get("Return [%]", -100.0)
    dd = stats.get("MaxDD [%]", -100.0)
    turnover = stats.get("Turnover [%]", 0.0)
    bret = benchmark_stats.get("Return [%]", 0.0)
    return 0.45 * sh + 0.15 * so + 0.012 * (ret - bret) + 0.004 * dd - 0.0005 * turnover


# --------------------------------------------------------------------------------------
# 10. EVALUATION PIPELINES (VALIDATION & FINAL TEST)
# --------------------------------------------------------------------------------------
def evaluate_dynamic_models(price_dict, feature_store, benchmark_ticker, benchmark_features,
                            selection_schedule, calendar, start, end, b, e,
                            model_specs=ACTIVE_MODEL_SPECS, top_n=5,
                            use_market_regime=True, use_stock_mask=True,
                            regime_kwargs=None, engine_kwargs=None, opt_kwargs=None):
    regime_kwargs = regime_kwargs or {}
    engine_kwargs = engine_kwargs or {}
    opt_kwargs = opt_kwargs or {}

    regime = build_market_regime(benchmark_features, **regime_kwargs)
    union_tickers = sorted({
        selection_schedule.loc[i, f"Top{j+1}"]
        for i in selection_schedule.index for j in range(top_n)
        if pd.notna(selection_schedule.loc[i, f"Top{j+1}"])
    })
    masks = build_stock_open_masks(feature_store, union_tickers, b, e)

    bnav = benchmark_nav(price_dict[benchmark_ticker], start, end, initial_cash=engine_kwargs.get("initial_cash", 1_000_000_000.0))
    bstats = metrics_from_equity(bnav)

    rows, curves, orders, schedules = [], {"BENCHMARK": bnav}, {}, {}
    for name, spec in model_specs.items():
        sched, _ = build_dynamic_weight_schedule(
            price_dict, selection_schedule, calendar, start, end, spec["family"],
            top_n=top_n, opt_kwargs=opt_kwargs
        )
        st_res, eq, od = run_portfolio_engine_dynamic(
            price_dict, sched, calendar, start, end,
            market_regime_open=regime["RiskOn_Open"],
            stock_active_open=masks,
            use_market_regime=(use_market_regime and spec["regime"]),
            use_stock_mask=(use_stock_mask and spec["mask"]),
            **engine_kwargs
        )
        st_res["Model"] = name
        st_res["ValidationScore"] = model_validation_score(st_res, bstats)
        rows.append(st_res)
        curves[name] = eq
        orders[name] = od
        schedules[name] = sched

    bstats.update({"Model": "BENCHMARK", "ValidationScore": np.nan})
    rows.append(bstats)
    res = pd.DataFrame(rows).set_index("Model")
    champion = res.drop(index="BENCHMARK")["ValidationScore"].idxmax()
    return res, champion, curves, orders, schedules

def evaluate_final_test(price_dict, feature_store, benchmark_ticker, benchmark_features,
                        selection_schedule, calendar, start, end, b, e, champion,
                        model_specs=ACTIVE_MODEL_SPECS, top_n=5,
                        use_market_regime=True, use_stock_mask=True,
                        regime_kwargs=None, engine_kwargs=None, opt_kwargs=None):
    regime_kwargs = regime_kwargs or {}
    engine_kwargs = engine_kwargs or {}
    opt_kwargs = opt_kwargs or {}

    regime = build_market_regime(benchmark_features, **regime_kwargs)
    union_tickers = sorted({
        selection_schedule.loc[i, f"Top{j+1}"]
        for i in selection_schedule.index for j in range(top_n)
        if pd.notna(selection_schedule.loc[i, f"Top{j+1}"])
    })
    masks = build_stock_open_masks(feature_store, union_tickers, b, e)

    init_cash = engine_kwargs.get("initial_cash", 1_000_000_000.0)
    bnav = benchmark_nav(price_dict[benchmark_ticker], start, end, initial_cash=init_cash)
    rows = []
    curves = {"VNINDEX_or_MarketBenchmark": bnav}
    order_map = {}

    # 1. VNINDEX / Benchmark
    bstats = metrics_from_equity(bnav)
    bstats["Model"] = "VNINDEX_or_MarketBenchmark"
    rows.append(bstats)

    # 2. Static Buy & Hold EW
    bh_sched = build_static_buyhold_schedule(price_dict, selection_schedule, calendar, start, end, top_n=top_n)
    bh_stats, bh_eq, bh_od = run_portfolio_engine_dynamic(
        price_dict, bh_sched, calendar, start, end,
        market_regime_open=None, stock_active_open=None,
        use_market_regime=False, use_stock_mask=False,
        **engine_kwargs
    )
    bh_stats["Model"] = "STATIC_EW_BH"
    rows.append(bh_stats)
    curves["STATIC_EW_BH"] = bh_eq
    order_map["STATIC_EW_BH"] = bh_od

    # 3. Dynamic EW không Regime
    ew_sched, _ = build_dynamic_weight_schedule(
        price_dict, selection_schedule, calendar, start, end, "EW",
        top_n=top_n, opt_kwargs=opt_kwargs
    )
    ew_stats, ew_eq, ew_od = run_portfolio_engine_dynamic(
        price_dict, ew_sched, calendar, start, end,
        market_regime_open=None, stock_active_open=None,
        use_market_regime=False, use_stock_mask=False,
        **engine_kwargs
    )
    ew_stats["Model"] = "DYN_EW_NO_REGIME"
    rows.append(ew_stats)
    curves["DYN_EW_NO_REGIME"] = ew_eq
    order_map["DYN_EW_NO_REGIME"] = ew_od

    # 4. Locked Champion & Diagnostic models
    for name, spec in model_specs.items():
        sched, _ = build_dynamic_weight_schedule(
            price_dict, selection_schedule, calendar, start, end, spec["family"],
            top_n=top_n, opt_kwargs=opt_kwargs
        )
        st_res, eq, od = run_portfolio_engine_dynamic(
            price_dict, sched, calendar, start, end,
            market_regime_open=regime["RiskOn_Open"],
            stock_active_open=masks,
            use_market_regime=(use_market_regime and spec["regime"]),
            use_stock_mask=(use_stock_mask and spec["mask"]),
            **engine_kwargs
        )
        label = "LOCKED_CHAMPION" if name == champion else "DIAG_" + name
        st_res["Model"] = label
        st_res["OriginalModel"] = name
        rows.append(st_res)
        curves[label] = eq
        order_map[label] = od

    res = pd.DataFrame(rows).set_index("Model")
    if "STATIC_EW_BH" in res.index:
        res["ExcessCAGR_vs_StaticEW [pp]"] = res["CAGR [%]"] - res.loc["STATIC_EW_BH", "CAGR [%]"]
        res["ExcessSharpe_vs_StaticEW"] = res["Sharpe"] - res.loc["STATIC_EW_BH", "Sharpe"]
    bname = "VNINDEX_or_MarketBenchmark"
    res["ExcessCAGR_vs_Benchmark [pp]"] = res["CAGR [%]"] - res.loc[bname, "CAGR [%]"]
    res["DD_Improvement_vs_Benchmark [pp]"] = res["MaxDD [%]"] - res.loc[bname, "MaxDD [%]"]
    return res, curves, order_map


# --------------------------------------------------------------------------------------
# 11. SIDEBAR CẤU HÌNH NGHIÊN CỨU & THAM SỐ ĐỊNH LƯỢNG
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Tham Số Nghiên Cứu")

with st.sidebar.expander("📁 1. Dữ Liệu Đầu Vào", expanded=True):
    uploaded_price = st.file_uploader("File OHLCV (CSV/Excel)", type=["csv", "xlsx", "xls"], key="upload_price")
    
    # Tự động quét file có sẵn trong thư mục
    local_csvs = [f for f in glob.glob("*.csv") if not f.startswith("0") and "test" not in f.lower()]
    local_excels = glob.glob("*.xlsx")
    available_locals = local_csvs + local_excels

    use_local_default = False
    if uploaded_price is None and available_locals:
        use_local_default = st.checkbox(f"Dùng file mẫu sẵn có: `{os.path.basename(available_locals[0])}`", value=True)

    uploaded_fa = st.file_uploader("File Phân Tích Cơ Bản - FA (Tùy chọn)", type=["csv", "xlsx", "xls"], key="upload_fa")
    uploaded_sector = st.file_uploader("File Phân Ngành - Sector (Tùy chọn)", type=["csv", "xlsx", "xls"], key="upload_sector")
    use_adj_price = st.checkbox("Ưu tiên Giá Điều Chỉnh (Adjusted)", value=True)

with st.sidebar.expander("📐 2. Phân Chia Thời Gian (Splits)", expanded=False):
    split_mode = st.selectbox("Phương thức chia tập", ["auto", "proportions"], index=0,
                              help="Auto: Dùng năm dương lịch cuối cùng làm Final Test. Proportions: Chia theo tỷ lệ cố định.")
    train_prop = st.slider("Tỷ lệ Core Train", 0.40, 0.80, 0.60, 0.05)
    val_prop = st.slider("Tỷ lệ Validation", 0.10, 0.40, 0.20, 0.05)
    test_prop = st.slider("Tỷ lệ Final Test", 0.10, 0.40, 0.20, 0.05)
    top_n = st.number_input("Số lượng cổ phiếu TOP N", min_value=1, max_value=20, value=5, step=1)

with st.sidebar.expander("🎯 3. Tối Ưu Ngưỡng TA (Core Train)", expanded=False):
    buy_choices = st.multiselect("Ngưỡng BUY ứng viên", [4, 5, 6, 7, 8], default=[5, 6, 7])
    exit_choices = st.multiselect("Ngưỡng EXIT ứng viên", [1, 2, 3, 4, 5], default=[2, 3, 4])
    n_folds = st.number_input("Số fold Cross-Validation", min_value=2, max_value=5, value=3)
    cv_stability_penalty = st.slider("Hình phạt độ biến thiên Sharpe (Penalty)", 0.0, 0.50, 0.20, 0.05)

with st.sidebar.expander("⚖️ 4. Tiêu Chí Xếp Hạng Cổ Phiếu", expanded=False):
    st.caption("Tổng trọng số chuẩn hóa về 1.0")
    w_trailing = st.slider("W Trailing Quality", 0.0, 0.5, 0.25, 0.05)
    w_ta = st.slider("W TA Score", 0.0, 0.5, 0.20, 0.05)
    w_rel = st.slider("W Relative Strength (126d)", 0.0, 0.5, 0.25, 0.05)
    w_lowvol = st.slider("W Low Volatility", 0.0, 0.3, 0.10, 0.05)
    w_mom = st.slider("W Momentum", 0.0, 0.3, 0.15, 0.05)
    w_liq = st.slider("W Liquidity", 0.0, 0.2, 0.05, 0.01)

    max_vol_p = st.slider("Max Volatility Percentile Gate", 0.50, 1.0, 0.75, 0.05)
    max_corr_cap = st.slider("Max Pairwise Correlation Cap", 0.50, 0.95, 0.80, 0.05)

with st.sidebar.expander("🔄 5. Tần Suất & Quản Trị Rủi Ro", expanded=False):
    stock_freq = st.selectbox("Tái chọn TOP N", ["Q (Hàng quý)", "M (Hàng tháng)"], index=0)
    alloc_freq = st.selectbox("Cập nhật tỷ trọng", ["M (Hàng tháng)", "Q (Hàng quý)"], index=0)
    stock_freq_code = "Q" if "Q" in stock_freq else "M"
    alloc_freq_code = "M" if "M" in alloc_freq else "Q"

    use_regime = st.checkbox("Bật Bộ Lọc Market Regime", value=True)
    ema_regime_span = st.number_input("Regime EMA Span", 50, 300, 200, 10)
    use_slope = st.checkbox("Yêu cầu độ dốc Slope EMA > 0", value=True)
    use_mask = st.checkbox("Bật Stock Active Mask (Chỉ giữ mã TA Active)", value=True)

with st.sidebar.expander("💼 6. Danh Mục & Chi Phí Giao Dịch", expanded=False):
    init_cash = st.number_input("Vốn ban đầu (VND)", min_value=10_000_000.0, value=1_000_000_000.0, step=50_000_000.0, format="%.0f")
    comm_pct = st.slider("Phí giao dịch (Commission)", 0.0, 0.005, 0.0015, 0.0005, format="%.4f")
    slip_pct = st.slider("Trượt giá (Slippage)", 0.0, 0.003, 0.0005, 0.0001, format="%.4f")
    max_weight_cap = st.slider("Tỷ trọng tối đa cho 1 cổ phiếu (Max Weight Cap)", 0.20, 1.0, 0.50, 0.05)
    mpt_gamma = st.slider("L2 Regularization Gamma (MPT)", 0.0, 0.10, 0.03, 0.01)
    weight_lookback_days = st.number_input("Lookback tính ma trận hiệp phương sai (phiên)", 60, 500, 252, 20)

run_button = st.sidebar.button("🚀 Chạy Kiểm Định Mô Hình (Run Pipeline)", type="primary", use_container_width=True)


# --------------------------------------------------------------------------------------
# 12. XỬ LÝ SỰ KIỆN CHẠY KIỂM ĐỊNH (PIPELINE EXECUTION)
# --------------------------------------------------------------------------------------
st.markdown('<div class="main-header">📈 HOSE V4: Kiểm Định Chiến Lược TOP 5 & Tối Ưu Danh Mục</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mô hình định lượng Nested Walk-Forward | TA Score 0–10 | Ledoit-Wolf Shrinkage | Quản trị rủi ro Regime & Cash | Tuyệt đối không rò rỉ dữ liệu tương lai</div>', unsafe_allow_html=True)

if run_button:
    # 1. Xác định file dữ liệu
    target_price_file = None
    if uploaded_price is not None:
        target_price_file = uploaded_price
    elif use_local_default and available_locals:
        target_price_file = available_locals[0]

    if target_price_file is None:
        st.error("⚠️ Vui lòng tải lên file dữ liệu giá OHLCV (CSV hoặc Excel) ở thanh bên trái!")
        st.stop()

    prog_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Bước 1: Chuẩn hóa dữ liệu OHLCV
        status_text.info("⏳ Bước 1/6: Đang đọc và chuẩn hóa dữ liệu OHLCV...")
        prog_bar.progress(10)
        raw_df = read_file_flexibly(target_price_file)
        data, price_source = standardize_ohlcv(raw_df, use_adjusted=use_adj_price)
        quality = data_quality_table(data)

        # Xử lý FA và Sector map nếu có
        fa_data = None
        if uploaded_fa is not None:
            fa_raw = read_file_flexibly(uploaded_fa)
            fa_data = standardize_fa(fa_raw)

        sector_mapping = {}
        if uploaded_sector is not None:
            sec_raw = read_file_flexibly(uploaded_sector)
            sector_mapping = load_sector_map(sec_raw)
        else:
            sector_mapping = load_sector_map(raw_df)

        # Bước 2: Chia tập dữ liệu
        status_text.info("⏳ Bước 2/6: Đang phân chia Core Train / Validation / Final Test...")
        prog_bar.progress(25)
        dates, core_end, val_start, val_end, test_start, test_end = resolve_auto_splits(
            data, split_mode=split_mode, train_prop=train_prop, val_prop=val_prop, test_prop=test_prop
        )

        split_table = pd.DataFrame([
            {"Set": "Core Train", "Start": dates[0].strftime("%Y-%m-%d"), "End": core_end.strftime("%Y-%m-%d"), "Sessions": int((dates <= core_end).sum())},
            {"Set": "Validation", "Start": val_start.strftime("%Y-%m-%d"), "End": val_end.strftime("%Y-%m-%d"), "Sessions": int(((dates >= val_start) & (dates <= val_end)).sum())},
            {"Set": "Final Test", "Start": test_start.strftime("%Y-%m-%d"), "End": test_end.strftime("%Y-%m-%d"), "Sessions": int((dates >= test_start).sum())},
        ])

        # Bước 3: Tính toán chỉ báo kỹ thuật & TA Score
        status_text.info("⏳ Bước 3/6: Đang tính toán các chỉ báo kỹ thuật & TA Score (0–10)...")
        prog_bar.progress(40)
        price_dict = build_price_dict(data)

        bench_ticker = next((t for t in price_dict if t in BENCHMARK_ALIASES), None)
        if bench_ticker is None:
            bench_ticker = "__MARKET_PROXY__"
            price_dict[bench_ticker] = build_synthetic_market_proxy(price_dict)

        feature_store = {t: calculate_ta_features(px) for t, px in price_dict.items()}

        # Bước 4: Tối ưu ngưỡng BUY/EXIT chung trên Core Train
        status_text.info("⏳ Bước 4/6: Đang tối ưu ngưỡng BUY/EXIT bằng Cross-Validation trên Core Train...")
        prog_bar.progress(55)
        core_universe = build_universe(price_dict, core_end, bench_ticker)
        core_candidates = core_universe.loc[core_universe["Candidate"], "Ticker"].tolist()
        core_dates = dates[dates <= core_end]
        inner_n = max(63, min(126, int(0.30 * len(core_dates))))
        inner_start = core_dates[-inner_n]
        folds = make_equal_folds(core_dates, inner_start, core_end, n_folds=n_folds)

        best_buy, best_exit, threshold_trials = choose_global_ta_thresholds(
            feature_store, core_candidates, folds,
            buy_choices=buy_choices, exit_choices=exit_choices,
            stability_penalty=cv_stability_penalty
        )

        # Cấu hình Ranking
        sum_w = w_trailing + w_ta + w_rel + w_lowvol + w_liq + w_mom
        weights_dict = {
            "trailing": w_trailing / sum_w, "ta": w_ta / sum_w,
            "rel_strength": w_rel / sum_w, "low_vol": w_lowvol / sum_w,
            "liquidity": w_liq / sum_w, "momentum": w_mom / sum_w
        }
        ranking_kwargs = {
            "max_vol_pct": max_vol_p,
            "max_pairwise_corr": max_corr_cap,
            "weights_dict": weights_dict,
        }
        engine_kwargs = {
            "initial_cash": init_cash,
            "commission": comm_pct,
            "slippage": slip_pct,
        }
        opt_kwargs = {
            "mpt_max_weight": max_weight_cap,
            "l2_gamma": mpt_gamma,
        }
        regime_kwargs = {
            "ema_span": int(ema_regime_span),
            "use_slope_confirm": use_slope,
        }

        # Bước 5: Lập lịch Top N & Chạy Validation chọn Champion
        status_text.info("⏳ Bước 5/6: Đang lập lịch Top N và đánh giá các mô hình trên Validation...")
        prog_bar.progress(75)
        val_selection, val_selection_details = build_selection_schedule(
            feature_store=feature_store, price_dict=price_dict,
            benchmark_ticker=bench_ticker, benchmark_features=feature_store[bench_ticker],
            calendar=dates, start=val_start, end=val_end,
            b=best_buy, e=best_exit, fa=fa_data, sector_map=sector_mapping,
            selection_freq=stock_freq_code, top_n=top_n, ranking_kwargs=ranking_kwargs
        )

        val_results, champion, val_curves, val_orders, val_schedules = evaluate_dynamic_models(
            price_dict=price_dict, feature_store=feature_store,
            benchmark_ticker=bench_ticker, benchmark_features=feature_store[bench_ticker],
            selection_schedule=val_selection, calendar=dates,
            start=val_start, end=val_end, b=best_buy, e=best_exit,
            top_n=top_n, use_market_regime=use_regime, use_stock_mask=use_mask,
            regime_kwargs=regime_kwargs, engine_kwargs=engine_kwargs, opt_kwargs=opt_kwargs
        )

        # Bước 6: Khóa Champion & Chạy Out-of-Sample Final Test
        status_text.info(f"⏳ Bước 6/6: Khóa Champion ({champion}) và kiểm định ngoài mẫu trên Final Test...")
        prog_bar.progress(90)
        test_selection, test_selection_details = build_selection_schedule(
            feature_store=feature_store, price_dict=price_dict,
            benchmark_ticker=bench_ticker, benchmark_features=feature_store[bench_ticker],
            calendar=dates, start=test_start, end=test_end,
            b=best_buy, e=best_exit, fa=fa_data, sector_map=sector_mapping,
            selection_freq=stock_freq_code, top_n=top_n, ranking_kwargs=ranking_kwargs
        )

        test_results, test_curves, test_orders = evaluate_final_test(
            price_dict=price_dict, feature_store=feature_store,
            benchmark_ticker=bench_ticker, benchmark_features=feature_store[bench_ticker],
            selection_schedule=test_selection, calendar=dates,
            start=test_start, end=test_end, b=best_buy, e=best_exit,
            champion=champion, top_n=top_n, use_market_regime=use_regime, use_stock_mask=use_mask,
            regime_kwargs=regime_kwargs, engine_kwargs=engine_kwargs, opt_kwargs=opt_kwargs
        )

        # Bảng Leakage Audit
        leakage = pd.DataFrame([
            ["TA BUY/EXIT", core_end.strftime("%Y-%m-%d"), False, "Chỉ Core Train"],
            ["Model family", val_end.strftime("%Y-%m-%d"), False, "Chọn trên Validation bằng nested selection"],
            ["Quarterly Top N in Test", "t-1", False, "Mỗi kỳ chọn lại bằng dữ liệu <= phiên trước"],
            ["Monthly portfolio weights in Test", "t-1", False, "MPT/MinVar/HRP chỉ dùng prices <= phiên trước"],
            ["FA snapshot", "t-1", False, "Chỉ FA có effective_date <= as-of" if fa_data is not None else "Không có FA -> TA-only"],
            ["Market regime", "t-1 close", False, "Close(t-1) -> Open(t)"],
            ["Final Test metrics", test_end.strftime("%Y-%m-%d"), True, "Chỉ đánh giá; không retune"],
        ], columns=["Giai đoạn (Stage)", "Ngày thông tin tối đa (Max Info Date)", "Dùng Final Test để tinh chỉnh?", "Quy tắc kiểm định (Rule)"])

        prog_bar.progress(100)
        status_text.success("🎉 Quá trình kiểm định hoàn tất thành công 100%!")

        # Lưu toàn bộ vào session_state
        st.session_state["pipeline_results"] = {
            "data": data,
            "price_source": price_source,
            "quality": quality,
            "split_table": split_table,
            "bench_ticker": bench_ticker,
            "feature_store": feature_store,
            "best_buy": best_buy,
            "best_exit": best_exit,
            "threshold_trials": threshold_trials,
            "val_selection": val_selection,
            "val_selection_details": val_selection_details,
            "val_results": val_results,
            "champion": champion,
            "val_curves": val_curves,
            "val_orders": val_orders,
            "test_selection": test_selection,
            "test_selection_details": test_selection_details,
            "test_results": test_results,
            "test_curves": test_curves,
            "test_orders": test_orders,
            "leakage": leakage,
        }

    except Exception as ex:
        status_text.empty()
        st.error(f"❌ Đã xảy ra lỗi trong quá trình thực thi: {str(ex)}")
        st.exception(ex)


# --------------------------------------------------------------------------------------
# 13. HIỂN THỊ KẾT QUẢ THEO CÁC TABS CHUYÊN NGHIỆP
# --------------------------------------------------------------------------------------
if "pipeline_results" in st.session_state:
    res = st.session_state["pipeline_results"]

    tabs = st.tabs([
        "🚀 Final Test (OOS)",
        "🏆 Validation & Champion",
        "📋 Lịch Trình TOP N",
        "🎯 Tối Ưu Ngưỡng BUY/EXIT",
        "📈 Phân Tích Kỹ Thuật & TA",
        "📊 Tổng Quan & Dữ Liệu",
        "🛡️ Kiểm Toán No-Leakage",
        "💾 Xuất Báo Cáo & File",
    ])

    # ----------------------------------------------------------------------------------
    # TAB 1: FINAL TEST (OUT-OF-SAMPLE)
    # ----------------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("🚀 Kết Quả Kiểm Định Ngoài Mẫu (Out-of-Sample Final Test)")
        st.caption("Giai đoạn kiểm định hoàn toàn độc lập, mô hình Champion đã được khóa từ Validation.")

        champ_name = res["champion"]
        test_res_df = res["test_results"]

        # KPI Metrics Cards
        kpi_cols = st.columns(4)
        c_stats = test_res_df.loc["LOCKED_CHAMPION"] if "LOCKED_CHAMPION" in test_res_df.index else None
        b_stats = test_res_df.loc["VNINDEX_or_MarketBenchmark"] if "VNINDEX_or_MarketBenchmark" in test_res_df.index else None
        bh_stats = test_res_df.loc["STATIC_EW_BH"] if "STATIC_EW_BH" in test_res_df.index else None

        with kpi_cols[0]:
            ret_val = c_stats.get("Return [%]", 0.0) if c_stats is not None else 0.0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Lợi Nhuận Locked Champion</div>
                <div class="kpi-val" style="color: {'#10B981' if ret_val >= 0 else '#EF4444'};">{ret_val:.2f}%</div>
                <small>Mô hình: {champ_name}</small>
            </div>
            """, unsafe_allow_html=True)

        with kpi_cols[1]:
            cagr_val = c_stats.get("CAGR [%]", 0.0) if c_stats is not None else 0.0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">CAGR Champion</div>
                <div class="kpi-val" style="color: {'#10B981' if cagr_val >= 0 else '#EF4444'};">{cagr_val:.2f}%</div>
                <small>Sharpe: {c_stats.get('Sharpe', 0.0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)

        with kpi_cols[2]:
            b_ret = b_stats.get("Return [%]", 0.0) if b_stats is not None else 0.0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">VN-Index / Benchmark</div>
                <div class="kpi-val">{b_ret:.2f}%</div>
                <small>MaxDD: {b_stats.get('MaxDD [%]', 0.0):.2f}%</small>
            </div>
            """, unsafe_allow_html=True)

        with kpi_cols[3]:
            bh_ret = bh_stats.get("Return [%]", 0.0) if bh_stats is not None else 0.0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Static Buy & Hold (EW)</div>
                <div class="kpi-val">{bh_ret:.2f}%</div>
                <small>Sharpe: {bh_stats.get('Sharpe', 0.0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Biểu đồ NAV Tăng Trưởng (Plotly Interactive Chart)
        st.markdown("### 📈 Biểu Đồ Tăng Trưởng NAV (Normalized NAV Curves)")
        
        curves = res["test_curves"]
        fig_nav = go.Figure()

        # Benchmarks
        if "VNINDEX_or_MarketBenchmark" in curves and len(curves["VNINDEX_or_MarketBenchmark"]):
            c = curves["VNINDEX_or_MarketBenchmark"]
            fig_nav.add_trace(go.Scatter(
                x=c.index, y=c / c.iloc[0], name="📊 VN-Index / Market Benchmark",
                line=dict(color="#1F2937", width=2.5, dash="solid")
            ))

        if "STATIC_EW_BH" in curves and len(curves["STATIC_EW_BH"]):
            c = curves["STATIC_EW_BH"]
            fig_nav.add_trace(go.Scatter(
                x=c.index, y=c / c.iloc[0], name="📌 STATIC_EW_BH (Equal Weight - Buy & Hold)",
                line=dict(color="#2563EB", width=2, dash="dash")
            ))

        # Dynamic EW No Regime
        if "DYN_EW_NO_REGIME" in curves and len(curves["DYN_EW_NO_REGIME"]):
            c = curves["DYN_EW_NO_REGIME"]
            fig_nav.add_trace(go.Scatter(
                x=c.index, y=c / c.iloc[0], name="⚡ DYN_EW_NO_REGIME (Dynamic Top 5 không Regime)",
                line=dict(color="#F59E0B", width=2)
            ))

        # Locked Champion
        if "LOCKED_CHAMPION" in curves and len(curves["LOCKED_CHAMPION"]):
            c = curves["LOCKED_CHAMPION"]
            fig_nav.add_trace(go.Scatter(
                x=c.index, y=c / c.iloc[0], name=f"🏆 LOCKED_CHAMPION ({champ_name})",
                line=dict(color="#10B981", width=3.5)
            ))

        # Các mô hình Diagnostic khác
        for k in curves.keys():
            if k.startswith("DIAG_"):
                c = curves[k]
                clean_lbl = k.replace("DIAG_", "")
                fig_nav.add_trace(go.Scatter(
                    x=c.index, y=c / c.iloc[0], name=f"✨ {clean_lbl}",
                    line=dict(width=1.5), visible="legendonly"
                ))

        fig_nav.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6)
        fig_nav.update_layout(
            height=500, hovermode="x unified",
            xaxis_title="Thời Gian (Phiên Giao Dịch)",
            yaxis_title="Giá Trị Tài Sản Chuẩn Hóa (Normalized NAV)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white"
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        # Biểu đồ Drawdown (Underwater Chart)
        st.markdown("### 🌊 Biểu Đồ Sụt Giảm Tài Sản (Underwater Drawdown Chart)")
        fig_dd = go.Figure()
        for k, lbl, col in [
            ("LOCKED_CHAMPION", f"Champion ({champ_name})", "#10B981"),
            ("STATIC_EW_BH", "Static EW Buy & Hold", "#2563EB"),
            ("VNINDEX_or_MarketBenchmark", "VN-Index", "#1F2937")
        ]:
            if k in curves and len(curves[k]):
                eq = curves[k]
                dd_series = (eq / eq.cummax() - 1.0) * 100.0
                fig_dd.add_trace(go.Scatter(
                    x=dd_series.index, y=dd_series, name=lbl,
                    line=dict(color=col, width=1.8), fill="tozeroy" if k == "LOCKED_CHAMPION" else None
                ))

        fig_dd.update_layout(
            height=300, hovermode="x unified",
            xaxis_title="Thời Gian", yaxis_title="Drawdown (%)",
            yaxis=dict(ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Bảng Số Liệu Chi Tiết Final Test
        st.markdown("### 📋 Bảng So Sánh Hiệu Quả Tất Cả Các Chiến Lược (Final Test)")
        cols_display = [
            "Return [%]", "CAGR [%]", "Sharpe", "Sortino", "MaxDD [%]", "Calmar",
            "Turnover [%]", "AvgExposure [%]", "AvgCash [%]",
            "ExcessCAGR_vs_Benchmark [pp]", "DD_Improvement_vs_Benchmark [pp]"
        ]
        test_table = test_res_df[[c for c in cols_display if c in test_res_df.columns]].copy()
        st.dataframe(
            test_table.style.format({
                "Return [%]": "{:.2f}%", "CAGR [%]": "{:.2f}%", "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}", "MaxDD [%]": "{:.2f}%", "Calmar": "{:.2f}",
                "Turnover [%]": "{:.1f}%", "AvgExposure [%]": "{:.1f}%", "AvgCash [%]": "{:.1f}%",
                "ExcessCAGR_vs_Benchmark [pp]": "{:+.2f} pp", "DD_Improvement_vs_Benchmark [pp]": "{:+.2f} pp",
            }, na_rep="N/A"),
            use_container_width=True
        )

        # Nhật ký lệnh giao dịch (Trade Orders Log)
        if "LOCKED_CHAMPION" in res["test_orders"] and not res["test_orders"]["LOCKED_CHAMPION"].empty:
            st.markdown("### 🧾 Nhật Ký Lệnh Giao Dịch Mô Hình Champion (Final Test)")
            od_df = res["test_orders"]["LOCKED_CHAMPION"].copy()
            st.dataframe(
                od_df.style.format({
                    "Shares": "{:,.0f}", "ExecPrice": "{:,.2f}", "Fee": "{:,.2f}"
                }),
                use_container_width=True, height=280
            )

    # ----------------------------------------------------------------------------------
    # TAB 2: VALIDATION & CHAMPION SELECTION
    # ----------------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("🏆 Đánh Giá & Khóa Mô Hình Champion Trên Validation")
        st.markdown(f"""
        <div class="champion-box">
            <h3 style="margin:0; font-size:1.4rem;">🏅 MÔ HÌNH CHAMPION ĐÃ KHÓA: <b>{res['champion']}</b></h3>
            <p style="margin:5px 0 0 0; opacity:0.95;">
                Được lựa chọn dựa trên điểm chuẩn Validation Score kết hợp Risk-Adjusted Return (Sharpe 45%, Sortino 15%), Excess Return (1.2%), Drawdown (0.4%) và trừ phạt Turnover (0.05%).
            </p>
        </div>
        """, unsafe_allow_html=True)

        val_df = res["val_results"].copy()
        st.markdown("### 📊 Bảng Xếp Hạng & Chấm Điểm Các Mô Hình Trên Validation")
        v_cols = ["Return [%]", "Sharpe", "Sortino", "MaxDD [%]", "Turnover [%]", "ValidationScore"]
        st.dataframe(
            val_df[[c for c in v_cols if c in val_df.columns]].sort_values("ValidationScore", ascending=False).style.format({
                "Return [%]": "{:.2f}%", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                "MaxDD [%]": "{:.2f}%", "Turnover [%]": "{:.1f}%", "ValidationScore": "{:.4f}",
            }, na_rep="N/A"),
            use_container_width=True
        )

        st.markdown("### 📈 Biểu Đồ NAV Các Mô Hình Cạnh Tranh Trên Tập Validation")
        v_curves = res["val_curves"]
        fig_val = go.Figure()
        for m_name, c in v_curves.items():
            if len(c):
                is_champ = (m_name == res["champion"])
                is_bench = (m_name == "BENCHMARK")
                fig_val.add_trace(go.Scatter(
                    x=c.index, y=c / c.iloc[0], name=m_name + (" (CHAMPION)" if is_champ else ""),
                    line=dict(
                        width=3.5 if is_champ else (2.0 if is_bench else 1.5),
                        dash="dash" if is_bench else "solid"
                    )
                ))
        fig_val.update_layout(
            height=450, hovermode="x unified",
            xaxis_title="Thời Gian Validation",
            yaxis_title="Normalized NAV",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template="plotly_white"
        )
        st.plotly_chart(fig_val, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # TAB 3: LỊCH TRÌNH LỰA CHỌN TOP N
    # ----------------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("📋 Lịch Trình Tái Lựa Chọn TOP N (Nested Walk-Forward)")
        st.markdown("""
        > **Nguyên tắc Causal:** Tại mỗi `EffectiveDate`, danh sách Top N được tính toán độc quyền bằng dữ liệu đóng cửa phiên liền trước `AsOfDate (t-1)`. Không có bất kỳ dữ liệu tương lai nào được sử dụng.
        """)

        sub_top_tabs = st.tabs(["Final Test Top N", "Validation Top N"])
        with sub_top_tabs[0]:
            st.markdown("#### Danh Sách TOP N Tái Lựa Chọn Trong Final Test")
            t_sched = res["test_selection"].copy()
            st.dataframe(t_sched, use_container_width=True)

            eff_selected = st.selectbox(
                "Xem bảng xếp hạng chi tiết tại kỳ EffectiveDate:",
                options=t_sched["EffectiveDate"].tolist(),
                key="test_eff_select"
            )
            if eff_selected in res["test_selection_details"]:
                st.markdown(f"**Bảng cổ phiếu được khóa tại {pd.Timestamp(eff_selected).strftime('%Y-%m-%d')}:**")
                locked_df = res["test_selection_details"][eff_selected]["locked"]
                disp_cols = ["Ticker", "SelectedOrder", "FinalScore", "TA_Composite", "FA_Score",
                             "TrailingSharpe", "RelativeStrength126", "Vol63", "Sector"]
                st.dataframe(
                    locked_df[[c for c in disp_cols if c in locked_df.columns]].style.format({
                        "FinalScore": "{:.2f}", "TA_Composite": "{:.2f}", "FA_Score": "{:.1f}",
                        "TrailingSharpe": "{:.2f}", "RelativeStrength126": "{:.2f}", "Vol63": "{:.2%}"
                    }, na_rep="N/A"),
                    use_container_width=True
                )

        with sub_top_tabs[1]:
            st.markdown("#### Danh Sách TOP N Tái Lựa Chọn Trong Validation")
            v_sched = res["val_selection"].copy()
            st.dataframe(v_sched, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # TAB 4: TỐI ƯU NGƯỠNG BUY/EXIT
    # ----------------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("🎯 Tối Ưu Hóa Ngưỡng BUY / EXIT Trên Core Train")
        st.markdown(f"""
        <div class="alert-box">
            <b>Kết quả tối ưu hóa trên Core Train:</b> Ngưỡng BUY đã khóa = <b>{res['best_buy']}</b> | Ngưỡng EXIT đã khóa = <b>{res['best_exit']}</b><br>
            Cặp tham số này được tối ưu duy nhất trên Core Train bằng phương pháp Cross-Validation ({n_folds} folds) và được cố định cho toàn bộ quá trình Validation và Final Test.
        </div>
        """, unsafe_allow_html=True)

        th_df = res["threshold_trials"].copy()
        st.markdown("### 📊 Kết Quả Đánh Giá Các Cặp Ngưỡng BUY / EXIT Ứng Viên")
        st.dataframe(
            th_df.style.format({
                "GlobalCVScore": "{:.4f}", "MedianSharpe": "{:.2f}", "SharpeStd": "{:.2f}",
                "MedianReturnPct": "{:.2f}%", "MedianMaxDDPct": "{:.2f}%"
            }).highlight_max(subset=["GlobalCVScore"], color="#D1FAE5"),
            use_container_width=True
        )

    # ----------------------------------------------------------------------------------
    # TAB 5: PHÂN TÍCH KỸ THUẬT & TA SCORE
    # ----------------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("📈 Phân Tích Kỹ Thuật Chi Tiết & Điểm TA Score (0–10)")
        fstore = res["feature_store"]
        tickers_list = [t for t in fstore.keys() if t != res["bench_ticker"]]

        selected_ticker = st.selectbox("Chọn mã cổ phiếu để xem biểu đồ kỹ thuật:", sorted(tickers_list))
        if selected_ticker:
            df_tick = fstore[selected_ticker].copy()
            
            st.markdown(f"#### 🔍 Thông Số Kỹ Thuật Gần Nhất Của Mã **{selected_ticker}**")
            recent = df_tick[["Close", "EMA20", "EMA50", "EMA200", "RSI", "MACD", "MACD_Signal",
                              "Trend_Score", "Momentum_Score", "Volume_Score", "Volatility_Score", "TA_Score"]].tail(5)
            st.dataframe(recent.style.format({
                "Close": "{:,.2f}", "EMA20": "{:,.2f}", "EMA50": "{:,.2f}", "EMA200": "{:,.2f}",
                "RSI": "{:.1f}", "MACD": "{:.2f}", "MACD_Signal": "{:.2f}"
            }), use_container_width=True)

            # Interactive Plotly Subplots for Stock Technicals
            fig_ta = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.55, 0.25, 0.20],
                subplot_titles=(f"Giá {selected_ticker} & Đường Trung Bình EMA", "RSI (14)", "TA Score (0-10)")
            )

            # Price & EMAs
            fig_ta.add_trace(go.Scatter(x=df_tick.index, y=df_tick["Close"], name="Close", line=dict(color="#1E293B", width=1.5)), row=1, col=1)
            fig_ta.add_trace(go.Scatter(x=df_tick.index, y=df_tick["EMA20"], name="EMA 20", line=dict(color="#3B82F6", width=1.2)), row=1, col=1)
            fig_ta.add_trace(go.Scatter(x=df_tick.index, y=df_tick["EMA50"], name="EMA 50", line=dict(color="#F59E0B", width=1.2)), row=1, col=1)
            fig_ta.add_trace(go.Scatter(x=df_tick.index, y=df_tick["EMA200"], name="EMA 200", line=dict(color="#EF4444", width=1.5)), row=1, col=1)

            # RSI
            fig_ta.add_trace(go.Scatter(x=df_tick.index, y=df_tick["RSI"], name="RSI", line=dict(color="#8B5CF6", width=1.5)), row=2, col=1)
            fig_ta.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig_ta.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

            # TA Score
            fig_ta.add_trace(go.Bar(x=df_tick.index, y=df_tick["TA_Score"], name="TA Score", marker_color="#10B981"), row=3, col=1)
            fig_ta.add_hline(y=res["best_buy"], line_dash="dash", line_color="#059669", annotation_text=f"BUY ({res['best_buy']})", row=3, col=1)
            fig_ta.add_hline(y=res["best_exit"], line_dash="dash", line_color="#DC2626", annotation_text=f"EXIT ({res['best_exit']})", row=3, col=1)

            fig_ta.update_layout(height=650, hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_ta, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # TAB 6: TỔNG QUAN & DỮ LIỆU
    # ----------------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("📊 Tổng Quan Tập Dữ Liệu & Kiểm Tra Chất Lượng")

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("#### 🕒 Bảng Phân Chia Thời Gian (Time Splits)")
            st.dataframe(res["split_table"], use_container_width=True)

        with col_q2:
            st.markdown("#### 📌 Thông Tin Thị Trường & Nguồn Giá")
            st.info(f"""
            - **Nguồn giá:** `{res['price_source']}`
            - **Benchmark sử dụng:** `{res['bench_ticker']}`
            - **Tổng số mã:** `{len(res['data']['Ticker'].unique())}`
            - **Tổng số phiên:** `{res['data']['Date'].nunique()}`
            """)

        st.markdown("#### 📋 Kiểm Tra Độ Đầy Đủ & Thanh Khoản Từng Mã Cổ Phiếu")
        st.dataframe(
            res["quality"].style.format({
                "MissingVolumePct": "{:.1f}%",
                "MedianTradedValue": "{:,.0f}"
            }), use_container_width=True, height=350
        )

    # ----------------------------------------------------------------------------------
    # TAB 7: KIỂM TOÁN CHỐNG RÒ RỈ DỮ LIỆU (LEAKAGE AUDIT)
    # ----------------------------------------------------------------------------------
    with tabs[6]:
        st.subheader("🛡️ Báo Cáo Kiểm Toán Chống Rò Rỉ Dữ Liệu (No-Leakage Audit)")
        st.markdown("""
        Hệ thống được thiết kế theo chuẩn định lượng cao nhất nhằm triệt tiêu hoàn toàn **Look-Ahead Bias** (sử dụng dữ liệu tương lai) và **Survivorship Bias**:
        """)
        st.dataframe(res["leakage"], use_container_width=True)

        st.markdown("""
        ### 🔍 Các nguyên tắc cốt lõi:
        1. **Khóa tham số:** Ngưỡng BUY/EXIT chỉ tối ưu trên `Core Train`. Không bao giờ điều chỉnh lại ngưỡng sau khi nhìn thấy kết quả Test.
        2. **Khóa mô hình:** Mô hình Champion được lựa chọn độc quyền trên `Validation`. Tập `Final Test` hoàn toàn là kiểm định ngoài mẫu thực sự.
        3. **Causal Timing:** Tín hiệu sinh ra từ giá đóng cửa phiên `Close(t-1)` chỉ được khớp lệnh từ giá mở cửa phiên kế tiếp `Open(t)`.
        4. **Point-in-time Fundamental:** Dữ liệu BCTC phân tích cơ bản chỉ được tính từ ngày công bố chính thức (`Effective/Publish Date`).
        """)

    # ----------------------------------------------------------------------------------
    # TAB 8: XUẤT BÁO CÁO & DỮ LIỆU
    # ----------------------------------------------------------------------------------
    with tabs[7]:
        st.subheader("💾 Xuất Báo Cáo Kết Quả Nghiên Cứu (Excel & CSV)")
        st.markdown("Tải về toàn bộ số liệu và kết quả nghiên cứu định lượng dưới định dạng file Excel 8 sheets hoàn chỉnh hoặc các file CSV riêng lẻ.")

        # Tạo file Excel đa trang trong bộ nhớ
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            res["quality"].to_excel(writer, sheet_name="DataQuality", index=False)
            res["split_table"].to_excel(writer, sheet_name="TimeSplit", index=False)
            res["threshold_trials"].to_excel(writer, sheet_name="CoreTAThresholds", index=False)
            res["val_selection"].to_excel(writer, sheet_name="ValSelectionHistory", index=False)
            res["val_results"].reset_index().to_excel(writer, sheet_name="ValidationModels", index=False)
            res["test_selection"].to_excel(writer, sheet_name="TestSelectionHistory", index=False)
            res["test_results"].reset_index().to_excel(writer, sheet_name="FinalTest", index=False)
            res["leakage"].to_excel(writer, sheet_name="LeakageAudit", index=False)

        excel_data = excel_buffer.getvalue()

        st.download_button(
            label="📥 Tải Báo Cáo Đầy Đủ (Excel .xlsx - 8 Sheets)",
            data=excel_data,
            file_name="HOSE_V4_QUANTITATIVE_RESULTS.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

        st.markdown("#### Hoặc tải các file CSV riêng rẽ:")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                "Tải CSV: Kết Quả Final Test",
                data=res["test_results"].reset_index().to_csv(index=False, encoding="utf-8-sig"),
                file_name="07_final_test_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                "Tải CSV: Lịch Trình Top N (Final Test)",
                data=res["test_selection"].to_csv(index=False, encoding="utf-8-sig"),
                file_name="06_test_selection_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d_col2:
            st.download_button(
                "Tải CSV: Kết Quả Validation Models",
                data=res["val_results"].reset_index().to_csv(index=False, encoding="utf-8-sig"),
                file_name="05_validation_models.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                "Tải CSV: Tối Ưu Ngưỡng BUY/EXIT",
                data=res["threshold_trials"].to_csv(index=False, encoding="utf-8-sig"),
                file_name="03_core_threshold_trials.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    # Màn hình chờ khi chưa nhấn chạy
    st.info("👈 **Hướng dẫn:** Vui lòng kiểm tra các tham số ở thanh bên trái (Sidebar) và nhấn nút **'🚀 Chạy Kiểm Định Mô Hình'** để bắt đầu quy trình phân tích và tối ưu danh mục đầu tư!")

    # Giới thiệu kiến trúc HOSE V4
    st.markdown("""
    ### 🔬 Quy Trình Nghiên Cứu Định Lượng HOSE V4
    ```
    Dữ liệu OHLCV 
      └── Chuẩn hóa dữ liệu & Kiểm tra chất lượng
            └── Tính toán TA Score (0–10) & FA Score (Point-in-time)
                  └── Chia Core Train / Validation / Final Test
                        └── Tối ưu ngưỡng BUY/EXIT chung trên Core Train (Cross-Validation)
                              └── Nested Walk-Forward: Tái chọn TOP N theo Quý (Dữ liệu t-1)
                                    └── Tối ưu tỷ trọng danh mục theo Tháng (EW / MPT-LW / MinVar-LW / HRP / Blend)
                                          └── Đánh giá & Khóa mô hình Champion trên Validation
                                                └── Kiểm định Ngoài Mẫu (Out-of-Sample) trên Final Test
                                                      └── Đối chiếu Benchmark (VN-Index & Static Buy & Hold)
    ```
    """)
