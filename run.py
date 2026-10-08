"""十倍股追蹤器 v1 主程式

  python run.py          抓真實資料、評分、產生 dashboard.html
  python run.py --demo   用模擬資料跑一次（不用網路，確認環境 OK）
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
import dashboard
import picks
import scoring
from data import load_watchlist


def demo_data(wl):
    """隨機產生股價與基本面，只為了測試流程。"""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=datetime.now(), periods=900)
    prices, funds, mrevs, sym_of = {}, {}, {}, {}
    for _, r in wl.iterrows():
        s = r["代號"]
        sym_of[s] = s
        drift = rng.normal(0.0006, 0.0012)
        kink = rng.integers(250, 480)
        rets = rng.normal(drift, 0.025, len(idx))
        rets[kink:] += rng.normal(0.002, 0.003)
        close = 50 * np.exp(np.cumsum(rets))
        opn = close * np.exp(rng.normal(0, 0.01, len(idx)))
        prices[s] = pd.DataFrame({"Open": opn, "High": np.maximum(opn, close) * 1.01,
                                  "Low": np.minimum(opn, close) * 0.99, "Close": close,
                                  "Volume": rng.lognormal(13, 0.4, len(idx))}, index=idx)
        mc = float(rng.lognormal(23, 1.4)) * (32 if r["市場"] == "台股" else 1)
        funds[s] = {"marketCap": mc, "freeCashflow": mc * rng.normal(0.03, 0.04),
                    "priceToBook": abs(rng.normal(4, 3)) + 0.3, "ebitdaMargins": rng.normal(0.18, 0.12),
                    "returnOnEquity": rng.normal(0.15, 0.12), "grossMargins": rng.uniform(0.15, 0.7),
                    "revenueGrowth": rng.normal(0.2, 0.3), "earningsGrowth": rng.normal(0.3, 0.6),
                    "rev_yoy_q0": rng.normal(0.25, 0.3), "rev_yoy_q1": rng.normal(0.18, 0.25),
                    "asset_growth": rng.normal(0.12, 0.1), "ebitda_growth": rng.normal(0.15, 0.25),
                    "rev_cagr3": rng.normal(0.12, 0.12), "totalCash": mc * 0.1,
                    "operatingCashflow": mc * rng.normal(0.01, 0.05)}
    for b in config.BENCHMARK.values():
        sym_of[b] = b
        bc = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx))))
        prices[b] = pd.DataFrame({"Open": bc, "High": bc, "Low": bc, "Close": bc, "Volume": np.ones(len(idx))}, index=idx)
    return prices, funds, mrevs, sym_of


def real_data(wl):
    from data import fetch_fundamentals, fetch_prices, fetch_tw_monthly_revenue, yf_symbol
    print(f"觀察清單 {len(wl)} 檔，開始抓資料…")
    sym_of = {r["代號"]: yf_symbol(r["市場"], r["代號"]) for _, r in wl.iterrows()}
    syms = list(sym_of.values()) + list(config.BENCHMARK.values())
    prices = fetch_prices(syms)
    for b in config.BENCHMARK.values():
        sym_of[b] = b
    print(f"  股價：{len(prices)}/{len(syms)} 檔")
    funds, mrevs = {}, {}
    cand = wl[wl["用途"] == "候選"]
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        funds[r["代號"]] = fetch_fundamentals(sym_of[r["代號"]])
        if r["市場"] == "台股":
            mrevs[r["代號"]] = fetch_tw_monthly_revenue(r["代號"])
        print(f"\r  基本面：{i}/{len(cand)}", end="", flush=True)
    print()
    missing = [c for c, s in sym_of.items() if s not in prices]
    if missing:
        print("  ⚠ 抓不到股價：", ", ".join(missing))
    return prices, funds, mrevs, sym_of


def load_backtest():
    p = config.ROOT / "data" / "backtest" / "stage_stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    demo = "--demo" in sys.argv
    wl = load_watchlist()
    prices, funds, mrevs, sym_of = (demo_data if demo else real_data)(wl)
    bt = load_backtest()
    rs_cut = {m: v.get("rs_cut") for m, v in bt.get("markets", {}).items()}
    cand, ind, hot_sub, techs = scoring.score_all(wl, prices, funds, mrevs, sym_of, rs_cut)
    us_fund = {} if demo else picks.us_fundamentals(cand)
    cand = picks.apply(cand, techs, mrevs, bt, us_fund)
    sel = picks.select(cand)

    # 暴衝雷達：全市場掃描（--no-radar 可跳過，本機測試比較快）
    surge, surge_px = pd.DataFrame(), {}
    if not demo and "--no-radar" not in sys.argv:
        import radar
        print("暴衝雷達：掃描全市場…")
        try:
            surge, surge_px = radar.scan(exclude=set(wl["代號"]))
            print(f"  找到 {len(surge)} 檔暴衝股")
        except Exception as e:
            print(f"  雷達失敗：{e}")
    sel["⚡ 突然暴衝"] = {mk: surge[surge["市場"] == mk]["代號"].tolist()[:8] if len(surge) else [] for mk in ("台股", "美股")}

    # 急跌警報：觀察清單＋暴衝股
    import alerts
    alert = alerts.scan(wl, techs, prices, sym_of, surge, surge_px) if not demo else pd.DataFrame()
    if len(alert):
        fcf = {c: (f or {}).get("fcf_yield") for c, f in us_fund.items()}
        if len(surge) and "自由現金流殖利率" in surge:
            fcf.update(dict(zip(surge["代號"], surge["自由現金流殖利率"])))
        alert["自由現金流殖利率"] = alert["代號"].map(fcf)
        print(f"  急跌警報：{len(alert)} 檔")

    # 霸榜：和過去每天的精選比，連續上榜幾天
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    streak = picks.streaks(config.HISTORY_DIR, sel) if not demo else {}

    # AI 新聞分析：只分析今日精選（需要 GEMINI_API_KEY）
    ai = {}
    if not demo:
        import news
        rows = []
        for cat, by_mk in sel.items():
            for mk, codes in by_mk.items():
                src = surge if cat == "⚡ 突然暴衝" else cand
                for c in codes:
                    r = src[src["代號"] == c].iloc[0].to_dict()
                    if cat == "⚡ 突然暴衝":
                        r["_extra"] = f"近 5 日成交量是季均量的 {r['量能倍數']:.1f} 倍，請特別說明暴衝原因與是否為新商機。"
                    rows.append(r)
        for _, a in alert.iterrows():
            src = cand if a["代號"] in set(cand["代號"]) else surge
            base = src[src["代號"] == a["代號"]].iloc[0].to_dict() if len(src) and a["代號"] in set(src["代號"]) else {}
            rows.insert(0, {**base, **a.to_dict(),
                            "_extra": f"近 5 日下跌 {a['5日漲跌']:.0%}（{a['警報']}），請說明下跌原因，並判斷是短期情緒／錯殺，還是基本面轉壞。"})
        seen = set()
        rows = [r for r in rows if not (r["代號"] in seen or seen.add(r["代號"]))]
        ai = news.analyze_many(rows)

    # 存歷史紀錄（之後回測校準機率要用）
    if not demo:
        cand.to_csv(config.HISTORY_DIR / f"scores_{datetime.now():%Y%m%d}.csv", index=False, encoding="utf-8-sig")
        (config.HISTORY_DIR / f"picks_{datetime.now():%Y%m%d}.json").write_text(
            json.dumps(sel, ensure_ascii=False), encoding="utf-8")

    # 當日／五日分時（5 分鐘線），給個股 K 線的「當日」「五日」用
    intraday = {}
    if not demo:
        from data import fetch_intraday
        mk_of = dict(zip(wl["代號"], wl["市場"]))
        by_mk = {"台股": [], "美股": []}
        for c in techs:
            if mk_of.get(c) in by_mk and sym_of.get(c):
                by_mk[mk_of[c]].append(sym_of[c])
        if len(surge):
            for m, y in zip(surge["市場"], surge["yf"]):
                by_mk[m].append(y)
        intraday = fetch_intraday(by_mk)
        print(f"  分時資料：{len(intraday)} 檔")

    out = dashboard.build(cand, ind, hot_sub, techs, prices, sym_of, bt, demo=demo, intraday=intraday,
                          sel=sel, streak=streak, surge=surge, surge_px=surge_px, ai=ai, alert=alert)
    print(f"\n完成，共評分 {len(cand)} 檔。前 10 名：")
    print(cand[["代號", "名稱", "股價", "階段", "細分", "總分"]].head(10).to_string(index=False))
    print(f"\n儀表板：{out}")


if __name__ == "__main__":
    main()
