"""四大支柱評分＋階段判斷＋雙標籤。

四大支柱（各 25 分，共 100）：
  ① 底子 DNA   ：小市值、自由現金流殖利率、便宜、獲利能力、沒有亂擴張（Yartseva 2025）
  ② 商機催化劑 ：營收成長與加速、盈餘成長、所屬產業的指標股是否已發動
  ③ 資金與技術 ：相對強度、均線多頭排列、帶量、近期突破（O'Neil / Minervini）
  ④ 進場位置   ：離均線乖離、離高點距離（避免追在過熱處）

雙標籤：⚡爆發力（1–2 年）、🏔️長跑力（3–5 年以上）
"""
import numpy as np
import pandas as pd

import config
import stages


def lin(x, lo, hi):
    """把 x 線性對應到 0–1，超出範圍就夾住；x 是 None 時回傳 None"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def pts(x, lo, hi, weight, missing=0.4):
    """缺資料時給 40% 的分數（不獎勵也不重罰）"""
    v = lin(x, lo, hi)
    return weight * (missing if v is None else v)


# ---------- 技術指標 ----------
def technicals(px, bench):
    """最新一天的技術面。階段判斷交給 stages.py（和回測同一套規則）。"""
    s = stages.compute(px)
    last = s.iloc[-1]
    c, v = px["Close"], px["Volume"]
    st, sub, pos = stages.describe(last)
    feat = stages.features(s, px["Volume"]).iloc[-1].to_dict()
    t = {
        "series": s, "feat": feat,
        "close": float(last["close"]), "date": c.index[-1].strftime("%Y-%m-%d"),
        "chg1d": float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else None,
        "ret_6m": float(c.iloc[-1] / c.iloc[-127] - 1) if len(c) > 127 else np.nan,
        "ext50": float(last["ext50"]), "from_high": float(last["from_high"]), "rsi": float(last["rsi"]),
        "ma50": float(last["ma50"]), "ma150": float(last["ma150"]), "low120": float(last["low120"]),
        "trend_ok": st in (stages.IGNITE, stages.RUN, stages.HOT),
        "vol_surge": float(v.iloc[-20:].mean() / max(v.iloc[-120:].mean(), 1)),
        "stage": st, "phase": sub, "pos": pos,
        "start_date": last["start_date"].strftime("%Y-%m-%d") if pd.notna(last["start_date"]) else None,
        "start_px": float(last["start_px"]) if pd.notna(last["start_px"]) else None,
    }
    # 近 20 個交易日內是否創 52 週新高（突破）
    prior_hi = float(c.iloc[-252:-20].max()) if len(c) > 272 else float(c.iloc[:-20].max())
    t["breakout"] = bool(c.iloc[-20:].max() >= prior_hi)
    # 相對大盤的 6 個月超額報酬
    if bench is not None and len(bench) > 127:
        b = bench["Close"]
        t["rs_6m"] = t["ret_6m"] - float(b.iloc[-1] / b.iloc[-127] - 1)
    else:
        t["rs_6m"] = t["ret_6m"]
    return t


# ---------- 進出場參考 ----------
ACTION = {
    stages.SEED: "觀察；營收加速才小量試單",
    stages.IGNITE: "進場區：可分批買進",
    stages.RUN: "續抱；新進場等回檔近 50 日線",
    stages.HOT: "不追；持有者分批獲利、跌破 50 日線出清",
    stages.WEAK: "減碼／出場",
    stages.DOWN: "避開，不攤平",
    stages.RANGE: "觀望",
}


RUN_ACTION = {
    "前段": "續抱；回檔近 50 日線可加碼",
    "中段": "續抱，不追高；守 50 日線",
    "後段": "不加碼；跌破 50 日線先賣一半",
}


def action(t):
    """建議動作。主升段依前/中/後段分開（回測：後段之後一年中位數為負、腰斬機率高）"""
    if t["stage"] == stages.RUN and t["phase"] in RUN_ACTION:
        return RUN_ACTION[t["phase"]]
    return ACTION.get(t["stage"], "")


def exit_lines(t):
    """回傳（減碼線, 出場線）。多頭中：跌破 50 日線減碼、跌破 150 日線出場；潛伏期：跌破半年低點停損。"""
    if t["stage"] in (stages.IGNITE, stages.RUN, stages.HOT):
        return t["ma50"], t["ma150"]
    if t["stage"] == stages.SEED:
        return None, t["low120"] * 0.97
    return None, None


def rev_tier(yoy3, accel):
    """和回測同一個定義：近 3 月年增 >20% 且加速＝加速；年增 <0＝衰退"""
    if yoy3 is None:
        return ""
    if yoy3 > 0.2 and accel is not None and accel > 0:
        return "加速"
    return "衰退" if yoy3 < 0 else "一般"


# ---------- 主評分 ----------
def score_one(row, f, t, mrev, theme_hot):
    market, model = row["市場"], row["評分模型"]
    mcap = f.get("marketCap")
    if mcap and market == "台股":
        mcap = mcap / config.TWD_PER_USD
    mcap_b = mcap / 1e9 if mcap else None

    # 營收成長：台股優先用月營收（比財報早）
    if mrev.get("mrev_yoy3") is not None:
        rev_yoy, rev_prev = mrev["mrev_yoy3"], mrev.get("mrev_yoy3_prev")
    else:
        rev_yoy = f.get("rev_yoy_q0", f.get("revenueGrowth"))
        rev_prev = f.get("rev_yoy_q1")
    accel_v = (rev_yoy - rev_prev) if (rev_yoy is not None and rev_prev is not None) else None

    # 市值越小分越高
    if mcap_b is None:
        size = 0.4
    elif mcap_b < 2:
        size = 1.0
    elif mcap_b < 10:
        size = 0.65
    elif mcap_b < 50:
        size = 0.35
    else:
        size = 0.1

    # ① 底子
    flags = []
    if model == "生技":
        cash, ocf = f.get("totalCash"), f.get("operatingCashflow")
        if cash and ocf is not None and ocf < 0:
            runway = cash / -ocf
        elif ocf is not None and ocf >= 0:
            runway = 5.0  # 已經自給自足
        else:
            runway = None
        if runway is not None and runway < 1.5:
            flags.append(f"現金只夠燒 {runway:.1f} 年")
        p1 = 6 * size + pts(runway, 0.5, 3, 14) + pts(f.get("revenueGrowth"), 0, 0.5, 5)
    else:
        fcf_y = (f["freeCashflow"] / f["marketCap"]) if f.get("freeCashflow") and f.get("marketCap") else None
        b2m = (1 / f["priceToBook"]) if f.get("priceToBook") and f["priceToBook"] > 0 else None
        ag, eg = f.get("asset_growth"), f.get("ebitda_growth")
        if ag is not None and eg is not None:
            sane = 4.0 if ag <= eg else 0.0
            if ag > eg + 0.15:
                flags.append("資產擴張遠快於獲利")
        else:
            sane = 1.6
        p1 = (6 * size + pts(fcf_y, 0, 0.08, 6) + pts(b2m, 0.05, 0.6, 4)
              + pts(f.get("ebitdaMargins"), 0, 0.30, 5) + sane)

    # ② 商機催化劑
    p2 = (pts(rev_yoy, 0, 0.5, 8) + pts(accel_v, -0.05, 0.20, 7)
          + pts(f.get("earningsGrowth"), 0, 0.8, 5) + (5 if theme_hot else 0))

    # ③ 資金與技術（RS 的百分位在外面算好放進 t["rs_pct"]）
    p3 = (8 * t.get("rs_pct", 0.5) + (7 if t["trend_ok"] else 0)
          + pts(t["vol_surge"], 1.0, 2.0, 4, missing=0) + (6 if t["breakout"] else 0))

    # ④ 進場位置：乖離小、離高點不遠最好；潛伏期底子好也給分
    ext = t["ext50"]
    p4_ext = 15 if ext <= 0.10 else (15 * (1 - lin(ext, 0.10, 0.40)))
    p4_hi = 5 * (1 - lin(-t["from_high"], 0.15, 0.50))
    p4_base = 5 if (t["stage"] == stages.SEED and p1 >= 15) else 0
    p4 = min(25, p4_ext + p4_hi + p4_base)
    # 回測：主升段後段之後一年中位數為負、腰斬機率高 → 進場位置打折
    p4 *= {"後段": 0.5, "中段": 0.85}.get(t["phase"], 1.0)

    total = p1 + p2 + p3 + p4
    burst = (p2 + p3) * 2
    longrun = (30 * (lin(f.get("grossMargins"), 0.2, 0.6) or 0.3)
               + 30 * (lin(f.get("returnOnEquity"), 0, 0.30) or 0.3)
               + 25 * (lin(f.get("rev_cagr3"), 0, 0.30) or 0.3)
               + 15 * size)

    cut, stop = exit_lines(t)
    return {
        "股價": t["close"], "日漲跌": t["chg1d"], "資料日期": t["date"],
        "階段": t["stage"], "細分": t["phase"], "週期位置": t["pos"], "起漲日": t["start_date"],
        "操作": action(t), "減碼線": cut, "出場線": stop,
        "營收狀態": rev_tier(mrev.get("mrev_yoy3"), accel_v) if market == "台股" else "",
        "相對強度": t.get("rs_tier", ""), "RSI": t["rsi"],
        "市值(億美元)": round(mcap_b * 10, 1) if mcap_b else None,
        "營收YoY": rev_yoy, "營收加速": accel_v,
        "①底子": round(p1, 1), "②催化劑": round(p2, 1), "③資金技術": round(p3, 1), "④位置": round(p4, 1),
        "總分": round(total, 1), "⚡爆發力": round(burst), "🏔️長跑力": round(longrun),
        "紅旗": "、".join(flags),
        "6月超額": t["rs_6m"], "6月漲幅": t["ret_6m"], "離高點": t["from_high"], "50MA乖離": ext,
    }


def score_all(wl, prices, funds, mrevs, sym_of, rs_cut=None):
    """rs_cut: {市場: (後20%門檻, 前20%門檻)}，來自回測時的全市場 6 個月漲幅分布"""
    # 先算技術面
    techs = {}
    for _, r in wl.iterrows():
        s = sym_of[r["代號"]]
        if s in prices:
            bench = prices.get(config.BENCHMARK[r["市場"]])
            techs[r["代號"]] = technicals(prices[s], bench)
    # RS 百分位（在同市場的觀察清單裡排名，給評分用）；強/中/弱 分級用全市場門檻（和回測一致）
    for mkt in ("美股", "台股"):
        codes = [c for c in wl[wl["市場"] == mkt]["代號"] if c in techs]
        vals = pd.Series({c: techs[c]["rs_6m"] for c in codes}, dtype=float).rank(pct=True)
        for c, p in vals.items():
            techs[c]["rs_pct"] = float(p) if not np.isnan(p) else 0.5
        lo, hi = (rs_cut or {}).get(mkt) or (None, None)
        for c in codes:
            r6 = techs[c]["ret_6m"]
            techs[c]["rs_tier"] = "" if (hi is None or np.isnan(r6)) else ("強" if r6 >= hi else "弱" if r6 <= lo else "中")

    # 指標股是否發動 → 主題/子題熱度
    hot_sub, hot_theme = set(), set()
    ind_rows = []
    for _, r in wl[wl["用途"] == "指標"].iterrows():
        t = techs.get(r["代號"])
        if not t:
            continue
        ind_rows.append({"市場": r["市場"], "代號": r["代號"], "名稱": r["名稱"], "主題": r["主題"],
                         "子題": r["子題"], "階段": t["stage"], "細分": t["phase"], "股價": t["close"],
                         "日漲跌": t["chg1d"], "6月超額": t["rs_6m"], "離高點": t["from_high"]})
        if t["trend_ok"] and t["rs_6m"] > 0:
            hot_sub.add(r["子題"])
            hot_theme.add(r["主題"])

    rows = []
    for _, r in wl[wl["用途"] == "候選"].iterrows():
        t = techs.get(r["代號"])
        if not t:
            continue
        hot = r["子題"] in hot_sub
        res = score_one(r, funds.get(r["代號"], {}), t, mrevs.get(r["代號"], {}), hot)
        rows.append({"市場": r["市場"], "代號": r["代號"], "名稱": r["名稱"], "主題": r["主題"],
                     "子題": r["子題"], "角色": r["角色"], "產業已發動": "是" if hot else "", **res})
    cand = pd.DataFrame(rows).sort_values("總分", ascending=False).reset_index(drop=True)
    return cand, pd.DataFrame(ind_rows), sorted(hot_sub), techs
