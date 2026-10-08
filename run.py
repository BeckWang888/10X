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

    # 存歷史紀錄（之後回測校準機率要用）
    if not demo:
        config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        cand.to_csv(config.HISTORY_DIR / f"scores_{datetime.now():%Y%m%d}.csv", index=False, encoding="utf-8-sig")

    out = dashboard.build(cand, ind, hot_sub, techs, prices, sym_of, bt, demo=demo)
    print(f"\n完成，共評分 {len(cand)} 檔。前 10 名：")
    print(cand[["代號", "名稱", "股價", "階段", "細分", "總分"]].head(10).to_string(index=False))
    print(f"\n儀表板：{out}")


if __name__ == "__main__":
    main()
