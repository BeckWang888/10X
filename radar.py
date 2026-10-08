"""暴衝雷達：每天掃描全市場（美股＋台股全部上市櫃），找「突然暴衝」的股票。

基本面突然改變（財報大好、接到大單、新產品、併購…）時，最先反應的是股價和成交量：
  近 5 日漲幅 ≥ 15%，且近 5 日平均量 ≥ 季均量 2.5 倍（和 backtest.py 的 SURGE 定義相同，歷史表現可回測）。
掃到之後再交給 AI（news.py）讀新聞，判斷是不是真的有新商機。
"""
from io import StringIO

import numpy as np
import pandas as pd
import requests

import config
import stages

RET5, VOL5 = 0.15, 2.5
LIQ = {"美股": (5e6, 3.0), "台股": (2e7, 10.0)}   # 20 日均成交金額、最低股價（美股比回測嚴格，排除炒作微型股）
TOP_N = 12


def us_list():
    out = {}
    for url, col in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
                     ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")):
        df = pd.read_csv(StringIO(requests.get(url, timeout=30).text), sep="|", dtype=str).dropna(subset=[col])
        df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
        bad = df["Security Name"].str.contains(
            "Warrant|Unit|Right|Preferred|Depositary|Notes|Debenture|%|Trust|Fund|Acquisition", case=False, na=True)
        df = df[~bad & df[col].str.fullmatch(r"[A-Z]{1,5}")]
        for s, n in zip(df[col], df["Security Name"]):
            out[s] = (s, n.split(" - ")[0].replace(" Common Stock", "").replace(", Inc.", "").strip()[:40])
    return out


def tw_list():
    r = requests.get("https://api.finmindtrade.com/api/v4/data", params={"dataset": "TaiwanStockInfo"}, timeout=30)
    df = pd.DataFrame(r.json()["data"])
    df = df[df["stock_id"].str.fullmatch(r"[1-9]\d{3}") & df["type"].isin(["twse", "tpex"])]
    df = df.sort_values("date").drop_duplicates("stock_id", keep="last")       # 同一檔有舊紀錄（改上市別、下市），取最新
    df = df[pd.to_datetime(df["date"]) >= pd.Timestamp.now() - pd.Timedelta(days=30)]   # 只留還在交易的
    return {s + (".TW" if t == "twse" else ".TWO"): (s, n) for s, t, n in zip(df["stock_id"], df["type"], df["stock_name"])}


def _download(symbols, batch=200):
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            raw = yf.download(chunk, period="1y", interval="1d", auto_adjust=True,
                              group_by="ticker", threads=True, progress=False, timeout=30)
        except Exception:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for s in chunk:
            try:
                d = (raw[s] if multi else raw)[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(d) > 70:
                out[s] = d
        print(f"\r  雷達下載 {min(i + batch, len(symbols))}/{len(symbols)}", end="", flush=True)
    print()
    return out


def scan(exclude=()):
    """回傳 (DataFrame 暴衝清單, {代號: 股價 DataFrame})"""
    rows, keep = [], {}
    for market, lst in (("美股", us_list()), ("台股", tw_list())):
        px = _download(list(lst))
        min_dollar, min_px = LIQ[market]
        for sym, d in px.items():
            c, v = d["Close"], d["Volume"].astype(float)
            if len(c) < 70:
                continue
            ret5 = float(c.iloc[-1] / c.iloc[-6] - 1)
            vol5 = float(v.iloc[-5:].mean() / max(v.iloc[-65:-5].mean(), 1))
            dollar = float((c * v).iloc[-65:-5].mean())        # 暴衝「之前」的成交金額：排除平常沒人交易、突然被炒的股票
            if ret5 < RET5 or vol5 < VOL5 or dollar < min_dollar or c.iloc[-1] < min_px:
                continue
            code, name = lst[sym]
            s = stages.compute(d)
            last = s.iloc[-1]
            st, sub, pos = stages.describe(last) if last["stage"] >= 0 else ("資料不足", "", "")
            rows.append({"市場": market, "代號": code, "名稱": name, "股價": float(c.iloc[-1]),
                         "日漲跌": float(c.iloc[-1] / c.iloc[-2] - 1), "資料日期": c.index[-1].strftime("%Y-%m-%d"),
                         "5日漲幅": ret5, "量能倍數": vol5, "1年漲幅": float(c.iloc[-1] / c.iloc[0] - 1),
                         "離一年高點": float(last["from_high"]), "階段": st, "細分": sub, "週期位置": pos,
                         "暴衝強度": ret5 * min(np.log2(vol5), 4.0), "在觀察清單": code in exclude})
            keep[code] = (d, s)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    # 美股：接上 SEC 財報。回測顯示暴衝時自由現金流為正的，之後表現好很多 → 排序加權
    import picks
    fund = picks.us_fundamentals(df)
    df["自由現金流殖利率"] = df["代號"].map(lambda c: (fund.get(c) or {}).get("fcf_yield"))
    df["暴衝強度"] = df["暴衝強度"] * np.where(df["自由現金流殖利率"].fillna(-1) > 0, 1.5, 1.0)
    df = df.sort_values("暴衝強度", ascending=False)
    df = pd.concat([g.head(TOP_N) for _, g in df.groupby("市場")]).reset_index(drop=True)
    return df, {k: keep[k] for k in df["代號"]}
