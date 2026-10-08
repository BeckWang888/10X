"""美股歷史財報（SEC EDGAR XBRL frames API，免費）。

frames API 一次回傳「所有上市公司」某個財報項目在某一期的數字，所以幾百次請求就能拿到 2009 年至今的資料。
「什麼時候算已知」：期末後 90 天（10-K 申報期限），避免回測偷看未來。
整理好的資料存在 data/backtest/us_fundamentals.csv.gz（放進 repo，每日更新直接讀，不用每天連 SEC）。

SEC 規定程式存取要附聯絡 email（User-Agent）。
"""
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config

UA = {"User-Agent": "10X-tracker beckwang888@users.noreply.github.com", "Accept-Encoding": "gzip, deflate"}
CACHE = config.CACHE_DIR / "sec"
STORE = config.ROOT / "data" / "backtest" / "us_fundamentals.csv.gz"
LAG_DAYS = 90
START_YEAR = 2009

# 年度（流量）項目：同一個欄位有好幾種 XBRL 名稱，依序取第一個有值的
FLOW = {
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "opinc": ["OperatingIncomeLoss"],
    "ni": ["NetIncomeLoss"],
}
# 時點（存量）項目：每季
INSTANT = {
    "assets": [("us-gaap", "Assets", "USD")],
    "equity": [("us-gaap", "StockholdersEquity", "USD"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding", "shares"),
               ("us-gaap", "CommonStockSharesOutstanding", "shares")],
}

_last = [0.0]


def _get(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (url.split("/api/xbrl/frames/")[-1].replace("/", "_") if "frames" in url else url.split("/")[-1])
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    for attempt in range(4):
        wait = 0.15 - (time.time() - _last[0])          # SEC 上限每秒 10 次，保守一點
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            p.write_text(r.text, encoding="utf-8")
            return r.json()
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"SEC 連線失敗 {r.status_code}：{url}")


CIK_STORE = config.ROOT / "data" / "backtest" / "us_cik.csv"


def ticker_cik():
    d = _get("https://www.sec.gov/files/company_tickers.json")
    m = {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}
    pd.Series(m, name="cik").rename_axis("ticker").to_csv(CIK_STORE)
    return m


def load_cik():
    if not CIK_STORE.exists():
        return ticker_cik()
    return pd.read_csv(CIK_STORE, index_col=0)["cik"].to_dict()


def _frame(tax, concept, unit, period):
    d = _get(f"https://data.sec.gov/api/xbrl/frames/{tax}/{concept}/{unit}/{period}.json")
    if not d or not d.get("data"):
        return pd.DataFrame(columns=["cik", "end", "val"])
    return pd.DataFrame(d["data"])[["cik", "end", "val"]]


def build(years=None):
    """抓 SEC 資料並整理成兩張表：annual（每年流量）與 instant（每季存量），合併成一個長表存檔。"""
    now = datetime.now()
    years = years or list(range(START_YEAR, now.year + 1))
    ticker_cik()
    rows = []
    n_req = 0
    for y in years:
        # 年度流量
        for col, names in FLOW.items():
            parts = []
            for i, nm in enumerate(names):
                f = _frame("us-gaap", nm, "USD", f"CY{y}")
                n_req += 1
                if len(f):
                    parts.append(f.assign(pri=i))
            if parts:
                f = pd.concat(parts).sort_values("pri").drop_duplicates("cik")
                rows.append(f.assign(kind="A", col=col, period=f"{y}")[["cik", "end", "val", "kind", "col", "period"]])
        # 每季存量
        for q in range(1, 5):
            if datetime(y, q * 3, 28) > now:
                continue
            for col, alts in INSTANT.items():
                parts = []
                for i, (tax, nm, unit) in enumerate(alts):
                    f = _frame(tax, nm, unit, f"CY{y}Q{q}I")
                    n_req += 1
                    if len(f):
                        parts.append(f.assign(pri=i))
                if parts:
                    f = pd.concat(parts).sort_values("pri").drop_duplicates("cik")
                    rows.append(f.assign(kind="Q", col=col, period=f"{y}Q{q}")[["cik", "end", "val", "kind", "col", "period"]])
        print(f"\r  SEC 財報 {y} 年完成（累計 {n_req} 次請求）", end="", flush=True)
    print()
    long = pd.concat(rows, ignore_index=True)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(STORE, index=False, compression="gzip")
    return long


def load():
    return pd.read_csv(STORE, parse_dates=["end"]) if STORE.exists() else None


def panels(long):
    """回傳 (annual, instant)：每家公司每期一列，avail＝可被知道的日期"""
    a = long[long["kind"] == "A"].pivot_table(index=["cik", "period"], columns="col", values="val", aggfunc="last")
    a_end = long[long["kind"] == "A"].groupby(["cik", "period"])["end"].max()
    a = a.join(a_end).reset_index().sort_values(["cik", "period"])
    g = a.groupby("cik")
    for c in ("rev", "opinc"):
        a[c + "_prev"] = g[c].shift(1)
    a["avail"] = pd.to_datetime(a["end"]) + pd.Timedelta(days=LAG_DAYS)
    q = long[long["kind"] == "Q"].pivot_table(index=["cik", "period"], columns="col", values="val", aggfunc="last")
    q_end = long[long["kind"] == "Q"].groupby(["cik", "period"])["end"].max()
    q = q.join(q_end).reset_index().sort_values(["cik", "period"])
    q["assets_prev"] = q.groupby("cik")["assets"].shift(4)     # 一年前的總資產
    q["avail"] = pd.to_datetime(q["end"]) + pd.Timedelta(days=LAG_DAYS)
    return a, q


# 基本面特徵（Yartseva 2025 十倍股研究的重點）
FUND_FEATURES = {
    "fcf_yield": "自由現金流殖利率",
    "bm": "帳面市值比（越高越便宜）",
    "log_mcap": "市值（log10 美元）",
    "roa": "資產報酬率",
    "op_margin": "營業利益率",
    "rev_g": "營收年增",
    "asset_vs_op": "資產成長－獲利成長",
}


def features(df, a, q, close_col="close"):
    """df 需要 cik、date、close。用 merge_asof 接上「當時已知」最新的財報，算出特徵。"""
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for t in (a, q):
        t["avail"] = pd.to_datetime(t["avail"]).astype("datetime64[ns]")
        t["cik"] = t["cik"].astype("int64")
    df["cik"] = df["cik"].astype("int64")
    df = pd.merge_asof(df, a.drop(columns=["end", "period"]).sort_values("avail").rename(columns={"avail": "a_avail"}),
                       left_on="date", right_on="a_avail", by="cik", direction="backward", tolerance=pd.Timedelta(days=550))
    df = pd.merge_asof(df, q.drop(columns=["end", "period"]).sort_values("avail").rename(columns={"avail": "q_avail"}),
                       left_on="date", right_on="q_avail", by="cik", direction="backward", tolerance=pd.Timedelta(days=200))
    mcap = df[close_col] * df["shares"]
    mcap = mcap.where(mcap > 1e6)
    fcf = df["ocf"] - df["capex"].fillna(0)
    df["mcap"] = mcap
    df["fcf_yield"] = (fcf / mcap).clip(-1, 1)
    df["bm"] = (df["equity"] / mcap).clip(-2, 5)
    df["log_mcap"] = np.log10(mcap)
    df["roa"] = (df["ni"] / df["assets"]).clip(-1, 1)
    df["op_margin"] = (df["opinc"] / df["rev"].where(df["rev"] > 0)).clip(-2, 1)
    df["rev_g"] = (df["rev"] / df["rev_prev"].where(df["rev_prev"] > 0) - 1).clip(-1, 5)
    ag = df["assets"] / df["assets_prev"].where(df["assets_prev"] > 0) - 1
    og = (df["opinc"] - df["opinc_prev"]) / df["assets_prev"].where(df["assets_prev"] > 0)   # 獲利變化／資產（避免負數分母）
    df["asset_vs_op"] = (ag - og).clip(-2, 2)
    return df


if __name__ == "__main__":
    long = build()
    print(f"已存 {STORE}（{len(long):,} 筆）")
