"""急跌警報：觀察清單（＋今日暴衝股）裡，突然大跌或剛跌破出場線的股票。

  🚨 急跌：近 5 日跌 ≥15% 且量 ≥ 季均量 2 倍；或單日跌 ≥8% 且當日量 ≥ 季均量 2.5 倍（台股接近跌停）
  ⛔ 跌破出場線：原本在多頭（發動／主升／過熱），近 3 個交易日內收盤跌破 150 日線 → 規則上的出場訊號
抓到之後交給 AI（news.py）查原因：是短期情緒，還是基本面轉壞。
"""
import numpy as np
import pandas as pd

import stages

DROP5, VOL5 = -0.15, 2.0
DROP1, VOL1 = -0.08, 2.5
ALIVE = {stages.CODE[stages.IGNITE], stages.CODE[stages.RUN], stages.CODE[stages.HOT]}


def check(code, px, series):
    """px：股價（Close、Volume）；series：stages.compute 的結果。回傳警報 dict 或 None"""
    c, v = px["Close"].astype(float), px["Volume"].astype(float)
    if len(c) < 70:
        return None
    base_v = max(v.iloc[-65:-5].mean(), 1)
    ret1, ret5 = float(c.iloc[-1] / c.iloc[-2] - 1), float(c.iloc[-1] / c.iloc[-6] - 1)
    v1, v5 = float(v.iloc[-1] / base_v), float(v.iloc[-5:].mean() / base_v)
    kinds = []
    if (ret5 <= DROP5 and v5 >= VOL5) or (ret1 <= DROP1 and v1 >= VOL1):
        kinds.append("🚨 急跌")
    st = series["stage"].to_numpy()
    below = (series["close"] < series["ma150"]).to_numpy()
    # 近 3 天內剛跌破 150 日線，而且跌破前 10 天內還在多頭
    if len(st) > 15 and below[-1] and not below[-4:-1].all():
        if any(x in ALIVE for x in st[-14:-3]):
            kinds.append("⛔ 跌破出場線")
    if not kinds:
        return None
    return {"代號": code, "警報": "・".join(kinds), "嚴重": bool(ret5 <= -0.25 or ret1 <= -0.095),
            "1日漲跌": ret1, "5日漲跌": ret5, "量能倍數": max(v1, v5),
            "股價": float(c.iloc[-1]), "150日線": float(series["ma150"].iloc[-1]),
            "資料日期": c.index[-1].strftime("%Y-%m-%d")}


def scan(wl, techs, prices, sym_of, surge=None, surge_px=None):
    """觀察清單＋暴衝股，回傳 DataFrame"""
    rows = []
    names = dict(zip(wl["代號"], wl["名稱"]))
    markets = dict(zip(wl["代號"], wl["市場"]))
    for code, t in techs.items():
        px = prices.get(sym_of.get(code))
        if px is None:
            continue
        a = check(code, px, t["series"])
        if a:
            rows.append({**a, "名稱": names.get(code, code), "市場": markets.get(code, ""), "來源": "觀察清單"})
    if surge is not None and len(surge):
        names.update(dict(zip(surge["代號"], surge["名稱"])))
        markets.update(dict(zip(surge["代號"], surge["市場"])))
    for code, (px, s) in (surge_px or {}).items():
        if code in techs:
            continue
        a = check(code, px, s)
        if a:
            rows.append({**a, "名稱": names.get(code, code), "市場": markets.get(code, ""), "來源": "暴衝股"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["嚴重", "5日漲跌"], ascending=[False, True]).reset_index(drop=True)
