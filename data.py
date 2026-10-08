"""抓資料：股價、基本面（yfinance），台股月營收（FinMind）。全部免費。"""
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

import config


def load_watchlist():
    df = pd.read_excel(config.WATCHLIST, sheet_name="觀察清單", dtype=str)
    df = df[df["保留(Y/N)"].str.upper().str.strip() == "Y"].copy()
    df["代號"] = df["代號"].str.strip()
    return df.reset_index(drop=True)


# ---------- 代號轉換 ----------
_tw_suffix_cache = {}


def yf_symbol(market, code):
    """台股要加 .TW（上市）或 .TWO（上櫃）。先試 .TW，抓不到再試 .TWO。"""
    if market != "台股":
        return code
    if code in _tw_suffix_cache:
        return _tw_suffix_cache[code]
    import yfinance as yf
    for suf in (".TW", ".TWO"):
        h = yf.Ticker(code + suf).history(period="5d")
        if not h.empty:
            _tw_suffix_cache[code] = code + suf
            return code + suf
    _tw_suffix_cache[code] = code + ".TW"
    return code + ".TW"


# ---------- 股價 ----------
def fetch_prices(symbols, period="2y"):
    """回傳 {symbol: DataFrame(Close, Volume)}"""
    import yfinance as yf
    out = {}
    raw = yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    for s in symbols:
        try:
            d = raw[s] if len(symbols) > 1 else raw
            d = d[["Close", "Volume"]].dropna()
            if len(d) > 60:
                out[s] = d
        except KeyError:
            pass
    return out


# ---------- 基本面 ----------
def _cache_path(name):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{name}.json"


def _cached(name):
    p = _cache_path(name)
    if p.exists():
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        if age < timedelta(days=config.FUNDAMENTAL_CACHE_DAYS):
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _row(df, *names):
    """從財報表裡找某一列（不同公司欄位名稱略有不同）"""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if len(s):
                return s.sort_index(ascending=False)  # 新 → 舊
    return None


def fetch_fundamentals(symbol):
    hit = _cached("f_" + symbol.replace(".", "_"))
    if hit is not None:
        return hit
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    f = {k: info.get(k) for k in (
        "marketCap", "freeCashflow", "priceToBook", "ebitdaMargins", "returnOnAssets",
        "returnOnEquity", "grossMargins", "revenueGrowth", "earningsGrowth",
        "totalCash", "operatingCashflow", "currency")}

    # 季營收 YoY 與加速度
    try:
        q = t.quarterly_income_stmt
        rev = _row(q, "Total Revenue", "Operating Revenue")
        if rev is not None and len(rev) >= 5:
            f["rev_yoy_q0"] = float(rev.iloc[0] / rev.iloc[4] - 1) if rev.iloc[4] > 0 else None
            if len(rev) >= 6 and rev.iloc[5] > 0:
                f["rev_yoy_q1"] = float(rev.iloc[1] / rev.iloc[5] - 1)
    except Exception:
        pass

    # 年度：資產成長 vs EBITDA 成長、營收 3 年 CAGR
    try:
        inc, bs = t.income_stmt, t.balance_sheet
        e = _row(inc, "EBITDA", "Normalized EBITDA")
        a = _row(bs, "Total Assets")
        if e is not None and a is not None and len(e) >= 2 and len(a) >= 2 and e.iloc[1] > 0:
            f["asset_growth"] = float(a.iloc[0] / a.iloc[1] - 1)
            f["ebitda_growth"] = float(e.iloc[0] / e.iloc[1] - 1)
        r = _row(inc, "Total Revenue", "Operating Revenue")
        if r is not None and len(r) >= 4 and r.iloc[3] > 0:
            f["rev_cagr3"] = float((r.iloc[0] / r.iloc[3]) ** (1 / 3) - 1)
    except Exception:
        pass

    _cache_path("f_" + symbol.replace(".", "_")).write_text(json.dumps(f), encoding="utf-8")
    time.sleep(0.3)
    return f


# ---------- 台股月營收（FinMind）----------
def fetch_tw_monthly_revenue(code):
    hit = _cached("mrev_" + code)
    if hit is not None:
        return hit
    start = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
    headers = {}
    tok = os.environ.get(config.FINMIND_TOKEN_ENV)
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    out = {}
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", headers=headers, timeout=20,
                         params={"dataset": "TaiwanStockMonthRevenue", "data_id": code, "start_date": start})
        rows = r.json().get("data", [])
        df = pd.DataFrame(rows)
        if not df.empty:
            df["ym"] = df["revenue_year"].astype(int) * 100 + df["revenue_month"].astype(int)
            s = df.groupby("ym")["revenue"].sum().sort_index()
            yoy = {}
            for ym, v in s.items():
                prev = s.get(ym - 100)
                if prev and prev > 0:
                    yoy[ym] = v / prev - 1
            y = pd.Series(yoy).sort_index()
            if len(y) >= 6:
                out["mrev_yoy3"] = float(y.iloc[-3:].mean())      # 近 3 個月平均 YoY
                out["mrev_yoy3_prev"] = float(y.iloc[-6:-3].mean())  # 前 3 個月平均 YoY
                out["mrev_latest_month"] = int(y.index[-1])
    except Exception:
        pass
    _cache_path("mrev_" + code).write_text(json.dumps(out), encoding="utf-8")
    time.sleep(0.5)
    return out
