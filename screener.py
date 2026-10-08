"""每月自動篩選：從全市場找「有機會成為十倍股」的新候選股，給使用者確認後才加入觀察清單。

  python screener.py            執行篩選＋AI 初評，結果存 data/screen/latest.json
  python screener.py --no-ai    不做 AI 初評

篩選條件（和 backtest.py 共用 route_masks，網頁上的回測數字就是這套條件的歷史表現）：
  美股 A 價值低檔型（Yartseva）：市值 1–20 億美元、自由現金流殖利率前 30%、帳面市值比前 30%（便宜）、資產沒有擴張得比獲利快
  美股 B 成長突破型（O'Neil）  ：市值 1–20 億美元、營收年增 ≥25%、相對強度前 20%、離一年高點 15% 以內
  台股 A 營收拐點低檔型        ：近 3 月營收年增由負轉正、離一年高點跌 30% 以上
  台股 B 營收加速發動型        ：近 3 月營收年增 ≥20% 且加速、階段＝發動期
  共同：流動性門檻（美股日均成交 ≥200 萬美元、股價 ≥2 美元；台股 ≥2,000 萬台幣、股價 ≥10 元）
  台股另外用 FinMind 季報確認：本業有獲利（近 4 季營業利益率 >0）
使用者確認要加入的股票寫在 data/screen/approved.csv，data.load_watchlist() 會自動併入。
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
import stages

OUT_DIR = config.ROOT / "data" / "screen"
APPROVED = OUT_DIR / "approved.csv"
MCAP_LO, MCAP_HI = 1e8, 2e9
LIQ = {"美股": (2e6, 2.0), "台股": (2e7, 10.0)}
TOP_N = 15          # 每條路線最多列出幾檔（AI 判斷符合主題的）
POOL_N = 40         # 每條路線先取幾檔量化最好的給 AI 初評

ROUTES = {
    "美股": {"A": "價值低檔型（Yartseva）", "B": "成長突破型（O'Neil）"},
    "台股": {"A": "營收拐點低檔型", "B": "營收加速發動型"},
}
ROUTE_RULES = {
    "美股A": "市值 1–20 億美元、自由現金流殖利率前 30%、便宜（帳面市值比）前 30%、資產沒有擴張得比獲利快",
    "美股B": "市值 1–20 億美元、營收年增 ≥25%、相對強度前 20%、離一年高點 15% 以內",
    "台股A": "近 3 個月營收年增由負轉正、股價離一年高點跌 30% 以上、近 4 季本業有獲利",
    "台股B": "近 3 個月營收年增 ≥20% 且比前 3 個月加速、階段＝🔥 發動期、近 4 季本業有獲利",
}


def route_masks(df, market):
    """df 需要的欄位：
       美股：log_mcap、fcf_r、bm_r（同一天全市場百分位）、asset_vs_op、rev_g、rs_pct、from_high
       台股：yoy3、accel、from_high、stage
    回傳 {"A": 布林 Series, "B": 布林 Series}（回測與每月篩選共用）"""
    if market == "美股":
        small = (df["log_mcap"] >= np.log10(MCAP_LO)) & (df["log_mcap"] < np.log10(MCAP_HI))
        a = small & (df["fcf_r"] >= 0.7) & (df["bm_r"] >= 0.7) & (df["asset_vs_op"] <= 0.05)
        b = small & (df["rev_g"] >= 0.25) & (df["rs_pct"] >= 0.8) & (df["from_high"] >= -0.15)
    else:
        prev = df["yoy3"] - df["accel"]
        a = (df["yoy3"] > 0) & (prev < 0) & (df["from_high"] <= -0.30)
        b = (df["yoy3"] >= 0.2) & (df["accel"] > 0) & (df["stage"] == stages.CODE[stages.IGNITE])
    return {"A": a.fillna(False), "B": b.fillna(False)}


# ---------- 全市場現況 ----------
def _snapshot(market, lst):
    import radar
    px = radar._download(list(lst))
    min_dollar, min_px = LIQ[market]
    rows, keep = [], {}
    for sym, d in px.items():
        c, v = d["Close"].astype(float), d["Volume"].astype(float)
        if len(c) < 180:
            continue
        dollar = float((c * v).iloc[-60:].mean())
        if dollar < min_dollar or c.iloc[-1] < min_px:
            continue
        s = stages.compute(d)
        last = s.iloc[-1]
        code, name = lst[sym]
        rows.append({"市場": market, "代號": code, "名稱": name, "yf": sym, "close": float(c.iloc[-1]),
                     "from_high": float(last["from_high"]), "past6": float(c.iloc[-1] / c.iloc[-127] - 1),
                     "stage": int(last["stage"]), "日均成交": dollar})
        keep[code] = d
    df = pd.DataFrame(rows)
    df["rs_pct"] = df["past6"].rank(pct=True)
    return df, keep


def screen_us():
    import radar
    import sec
    df, _ = _snapshot("美股", radar.us_list())
    long = sec.load()
    cik = sec.load_cik()
    df["cik"] = df["代號"].map(cik)
    df = df.dropna(subset=["cik"])
    a, q = sec.panels(long)
    f = sec.features(df.assign(date=pd.Timestamp.now().normalize()), a, q)
    f = f[f["fcf_yield"].notna() & f["bm"].notna()].copy()
    f["fcf_r"] = f["fcf_yield"].rank(pct=True)
    f["bm_r"] = f["bm"].rank(pct=True)
    m = route_masks(f, "美股")
    out = []
    for r, mask in m.items():
        g = f[mask].copy()
        # 排序：A 看便宜＋現金流＋離低點近；B 看營收成長＋相對強度
        g["排序分"] = (g["fcf_r"] + g["bm_r"] - g["from_high"].rank(pct=True)) if r == "A" else \
            (g["rev_g"].rank(pct=True) + g["rs_pct"])
        g = g.sort_values("排序分", ascending=False).head(POOL_N)
        for _, x in g.iterrows():
            out.append({"市場": "美股", "路線": r, "代號": x["代號"], "名稱": x["名稱"], "yf": x["yf"],
                        "股價": x["close"], "市值(億美元)": round(x["mcap"] / 1e8, 1),
                        "自由現金流殖利率": x["fcf_yield"], "帳面市值比": x["bm"], "營收年增": x["rev_g"],
                        "資產報酬率": x["roa"], "離一年高點": x["from_high"], "相對強度": x["rs_pct"],
                        "階段": stages.ORDER[x["stage"]] if x["stage"] >= 0 else ""})
    return out, len(f)


def screen_tw():
    import backtest
    import radar
    df, _ = _snapshot("台股", radar.tw_list())
    rv = backtest.tw_revenue(datetime.now().year - 2)
    rv = rv[rv["avail"] <= pd.Timestamp.now()].sort_values("avail").groupby("code").tail(1)
    df = df.merge(rv[["code", "yoy3", "accel"]], left_on="代號", right_on="code", how="left")
    m = route_masks(df, "台股")
    out = []
    for r, mask in m.items():
        g = df[mask].copy()
        g = g.sort_values("yoy3", ascending=False).head(POOL_N * 2)
        for _, x in g.iterrows():
            opm = _tw_profit(x["代號"])
            if opm is not None and opm <= 0:
                continue                                   # 確定本業虧損的不要；查不到的保留並標示
            info = {}
            out.append({"市場": "台股", "路線": r, "代號": x["代號"], "名稱": x["名稱"], "yf": x["yf"],
                        "股價": x["close"],
                        "營收年增": x["yoy3"], "營收加速": x["accel"], "營業利益率": opm,
                        "股價淨值比": info.get("priceToBook"), "離一年高點": x["from_high"],
                        "未確認獲利": opm is None,
                        "相對強度": x["rs_pct"], "階段": stages.ORDER[x["stage"]] if x["stage"] >= 0 else "",
                        "_summary": info.get("longBusinessSummary", "")})
            if sum(1 for o in out if o["路線"] == r) >= POOL_N:
                break
    return out, int(df["yoy3"].notna().sum())


def _tw_profit(code):
    """台股近 4 季營業利益率（FinMind 季報，快取一週）。查不到回傳 None"""
    import os
    import time
    import requests
    p = config.CACHE_DIR / "twprofit" / f"{code}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < 7 * 86400:
        return json.loads(p.read_text(encoding="utf-8")).get("opm")
    headers = {}
    if os.environ.get(config.FINMIND_TOKEN_ENV):
        headers["Authorization"] = f"Bearer {os.environ[config.FINMIND_TOKEN_ENV]}"
    opm = None
    try:
        start = (datetime.now() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        r = requests.get("https://api.finmindtrade.com/api/v4/data", headers=headers, timeout=30,
                         params={"dataset": "TaiwanStockFinancialStatements", "data_id": code, "start_date": start}).json()
        df = pd.DataFrame(r.get("data", []))
        if len(df):
            piv = df[df["type"].isin(["Revenue", "OperatingIncome"])].pivot_table(index="date", columns="type", values="value")
            piv = piv.dropna().sort_index().tail(4)
            if len(piv) and piv["Revenue"].sum() > 0:
                opm = float(piv["OperatingIncome"].sum() / piv["Revenue"].sum())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"opm": opm}), encoding="utf-8")
        time.sleep(0.5)
    except Exception:
        pass
    return opm


def _yf_info(sym):
    """Yahoo 公司資料（逐一查、有間隔、失敗重試，結果快取一週；Yahoo 查太快會被擋）"""
    import time
    import yfinance as yf
    p = config.CACHE_DIR / "yfinfo" / f"{sym.replace('.', '_')}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < 7 * 86400:
        return json.loads(p.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            time.sleep(0.8)
            info = yf.Ticker(sym).info or {}
            if info:
                keep = {k: info.get(k) for k in ("marketCap", "operatingMargins", "priceToBook", "longBusinessSummary",
                                                 "sector", "industry")}
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
                return keep
        except Exception:
            time.sleep(10 * (attempt + 1))
    return {}


# ---------- 建議移除 ----------
def removals():
    """現有候選股裡，基本面明顯轉壞的（只列出，給使用者決定）"""
    import data
    import sec
    wl = data.load_watchlist()
    cand = wl[wl["用途"] == "候選"]
    out = []
    long = sec.load()
    if long is not None:
        us = cand[cand["市場"] == "美股"]
        cik = sec.load_cik()
        a, q = sec.panels(long)
        hist = sorted(config.HISTORY_DIR.glob("scores_*.csv"))
        price = {}
        if hist:
            h = pd.read_csv(hist[-1], dtype={"代號": str})
            price = dict(zip(h["代號"], h["股價"]))
        d = pd.DataFrame({"代號": us["代號"], "cik": us["代號"].map(cik), "close": us["代號"].map(price),
                          "date": pd.Timestamp.now().normalize()}).dropna()
        if len(d):
            f = sec.features(d, a, q)
            for _, x in f.iterrows():
                why = []
                if x["fcf_yield"] < 0 and x["asset_vs_op"] > 0.15:
                    why.append("自由現金流為負且資產擴張遠快於獲利")
                op_roa = x["opinc"] / x["assets"] if x["assets"] and x["assets"] > 0 else np.nan
                if op_roa < -0.15 and x["fcf_yield"] < 0:       # 用營業利益判斷，避免一次性會計損失誤判
                    why.append(f"本業虧損（營業利益／資產 {op_roa:.0%}）且自由現金流為負")
                if why:
                    out.append({"市場": "美股", "代號": x["代號"], "原因": "、".join(why)})
    try:
        import backtest
        rv = backtest.tw_revenue(datetime.now().year - 1)
        rv = rv[rv["avail"] <= pd.Timestamp.now()].sort_values("avail").groupby("code").tail(1).set_index("code")
        for c in cand[cand["市場"] == "台股"]["代號"]:
            if c in rv.index and rv.loc[c, "yoy3"] < -0.2:
                out.append({"市場": "台股", "代號": c, "原因": f"近 3 個月營收年減 {rv.loc[c, 'yoy3']:.0%}"})
    except Exception as e:
        print(f"  台股營收檢查失敗：{e}")
    names = dict(zip(wl["代號"], wl["名稱"]))
    for o in out:
        o["名稱"] = names.get(o["代號"], o["代號"])
    return out


# ---------- AI 初評 ----------
EVAL_PROMPT = """你是謹慎的成長股研究員，請用繁體中文回答。
我在找「有機會在 1–5 年成為十倍股」的公司。使用者關注的主題：半導體、AI 與其延伸、AI 基建、電力、機器人、生技新藥（量子、太空當雷達）。
公司：{name}（{code}，{market}）
量化篩選路線：{route}
關鍵數字：{facts}
公司簡介：{summary}
最近新聞標題：
{news}

請只輸出 JSON：
{{"theme": "最符合的主題（上面清單之一），都不符合就寫「不屬於」",
  "subtheme": "子題，例如 散熱、先進封裝、重電",
  "business": "一句話說明公司做什麼（25 字內）",
  "moat": "競爭優勢或護城河（30 字內，沒有就說沒有）",
  "runway": "成長跑道與市場規模（30 字內）",
  "risks": ["主要風險 1", "主要風險 2"],
  "verdict": "建議加入" 或 "觀察" 或 "不建議",
  "reason": "結論理由（40 字內）"}}
不確定的事不要編造；資訊不足就寫「資訊不足」。"""


def ai_review(items):
    import news
    if not news._key():
        print("  （沒有 GEMINI_API_KEY，跳過 AI 初評）")
        return
    model = news.pick_model()
    from concurrent.futures import ThreadPoolExecutor

    def one(x):
        heads = news.headlines(x["名稱"], x["代號"], x["市場"], days=30, limit=10)
        summary = x.get("_summary") or ""        # 美股不另查 Yahoo（避免被擋），AI 靠公司名稱與新聞判斷
        facts = "、".join(f"{k} {v:.0%}" if isinstance(v, float) and abs(v) < 20 else f"{k} {v}"
                         for k, v in x.items() if k in ("市值(億美元)", "自由現金流殖利率", "帳面市值比", "營收年增",
                                                         "營收加速", "營業利益率", "離一年高點") and v is not None)
        prompt = EVAL_PROMPT.format(name=x["名稱"], code=x["代號"], market=x["市場"],
                                    route=ROUTES[x["市場"]][x["路線"]], facts=facts,
                                    summary=(summary or "（無）")[:1500],
                                    news="\n".join(f"- {n['date']} {n['title']}" for n in heads) or "（沒有新聞）")
        return news.ask_json(prompt, model)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for x, res in zip(items, ex.map(one, items)):
            x["ai"] = res
    print(f"  AI 初評：{sum(1 for x in items if x.get('ai'))}/{len(items)} 檔")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not APPROVED.exists():
        pd.DataFrame(columns=["市場", "代號", "名稱", "主題", "子題", "角色", "用途", "評分模型", "理由", "保留(Y/N)"]) \
            .to_csv(APPROVED, index=False, encoding="utf-8-sig")
    import data
    wl_codes = set(data.load_watchlist()["代號"])
    print("每月篩選：美股…")
    us, n_us = screen_us()
    print(f"  美股：{n_us} 檔有財報資料，入選 {len(us)} 檔")
    print("每月篩選：台股…")
    tw, n_tw = screen_tw()
    print(f"  台股：{n_tw} 檔有月營收資料，入選 {len(tw)} 檔")
    items = us + tw
    for x in items:
        x["已在清單"] = x["代號"] in wl_codes
    new = [x for x in items if not x["已在清單"]]
    if "--no-ai" not in sys.argv:
        ai_review(new)
    # 排序與顯示：AI 判斷符合主題的排前面（建議加入 > 觀察 > 不建議），每條路線最多顯示 TOP_N 檔
    rank = {"建議加入": 0, "觀察": 1, "不建議": 2}
    for x in items:
        x.pop("_summary", None)
        ai = x.get("ai") or {}
        x["主題符合"] = (ai.get("theme") not in (None, "", "不屬於")) if ai else None
    shown = []
    for mk, routes in ROUTES.items():
        for r in routes:
            g = [x for x in items if x["市場"] == mk and x["路線"] == r]
            g.sort(key=lambda x: (x["主題符合"] is False, rank.get((x.get("ai") or {}).get("verdict"), 3)))
            for i, x in enumerate(g):
                x["顯示"] = (x["主題符合"] is not False) and i < TOP_N
            shown += g
    items = shown
    rm = removals()
    print(f"  建議檢視移除：{len(rm)} 檔")
    res = {"date": datetime.now().strftime("%Y-%m-%d"), "rules": ROUTE_RULES, "routes": ROUTES,
           "universe": {"美股": n_us, "台股": n_tw}, "items": items, "removals": rm}
    txt = json.dumps(res, ensure_ascii=False, default=float, indent=1)
    (OUT_DIR / "latest.json").write_text(txt, encoding="utf-8")
    (OUT_DIR / f"screen_{datetime.now():%Y%m}.json").write_text(txt, encoding="utf-8")
    print(f"已存 {OUT_DIR / 'latest.json'}")


if __name__ == "__main__":
    main()
