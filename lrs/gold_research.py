"""黃金長抱：1 倍 vs 2 倍（正2），以及放進股票組合的效果。

用法：python lrs/gold_research.py → lrs/results/gold_research.md
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
px = pd.read_csv(os.path.join(HERE, "results", "prices_more.csv.gz"), index_col=0, parse_dates=True)
OUT = []
say = OUT.extend


def clean_ret(s, lim=0.5):
    r = s.dropna().pct_change().dropna()
    return r.where(r.abs() < lim, 0.0)


usd_rf = (px["^IRX"].ffill() / 100 / 252).fillna(0)
tw_rate = pd.Series(np.select([px.index < "2001-01-01", px.index < "2009-01-01"], [0.045, 0.02], 0.012), px.index)
tw_rf = tw_rate / 252

gold = clean_ret(px["GLD"]) + 0.004 / 252                     # 金價（GLD 加回費用）
gold = gold.reindex(clean_ret(px["GC=F"]).index.union(gold.index))
gold = gold.where(gold.notna(), clean_ret(px["GC=F"])).dropna()   # 2004 前用期貨
fx = px["TWD=X"].dropna(); fx = fx[(fx > 25) & (fx < 40)]; fxr = fx.pct_change().dropna()
taiex = clean_ret(px["^TWII"]) + 0.032 / 252


def lev(r, k, rf, fee, spread=0.0):
    f = rf.reindex(r.index).fillna(0)
    return k * r - (k - 1) * (f + spread / 252) - (fee / 252 if k != 1 else 0)


def in_twd(r):
    f = fxr.reindex(r.index); m = f.notna()
    return (1 + r[m]) * (1 + f[m]) - 1


def cagr(r):
    r = r.dropna(); return (1 + r).prod() ** (252 / len(r)) - 1


def mdd(r):
    e = (1 + r.dropna()).cumprod(); return (e / e.cummax() - 1).min()


def w12(r):
    e = (1 + r.dropna()).cumprod(); return e.pct_change(252).min()


def per(r, a, b):
    x = r[a:b]; return (1 + x).prod() - 1 if len(x) > 5 else np.nan


def P(x, d=0):
    return "—" if x != x else f"{x*100:.{d}f}%"


G2_FEE, G2_SPREAD = 0.0095, 0.003
g1 = gold
g2 = lev(gold, 2, usd_rf, G2_FEE, G2_SPREAD)


def tw_futures_gold(k, fee):
    """台股黃金期貨 ETF（00635U、00708L）：本金放台幣賺台幣利率，加上 k 倍「金價−美元利率」的期貨損益，幾乎不受匯率影響"""
    ur = usd_rf.reindex(gold.index).fillna(0); tr_ = tw_rf.reindex(gold.index).fillna(0)
    return tr_ + k * (gold - ur) - fee / 252


tw_g1 = tw_futures_gold(1, 0.0115)   # 00635U 型
tw_g2 = tw_futures_gold(2, 0.0130)   # 00708L 型

say(["# 黃金長抱：1 倍 vs 2 倍（自動產生）", "",
     f"- 2 倍黃金模擬：2×金價日報酬 −（國庫券＋{G2_SPREAD*100:.1f}%）− 費用 {G2_FEE*100:.2f}%（黃金沒有利息，借錢成本全部要自己付）", "",
     "## 0. 模擬校驗", "", "| 真實 | 期間 | 真實年化 | 模擬年化 | 日相關 |", "|---|---|---|---|---|"])
for real, sim, twd in (("UGL", g2, False), ("00708L.TW", tw_g2, False), ("00635U.TW", tw_g1, False), ("GLD", g1, False)):
    rr = clean_ret(px[real]); ss = in_twd(sim) if twd else sim
    c = rr.index.intersection(ss.index)
    # 台股與美股收盤時差 → 用週報酬算相關
    wk = lambda s: (1 + s).resample("W").prod() - 1
    say([f"| {real} | {c[0].date()}~{c[-1].date()} | {P(cagr(rr[c]),1)} | {P(cagr(ss[c]),1)} | "
         f"{np.corrcoef(wk(rr[c]), wk(ss[c]))[0,1]:.3f}（週） |"])
say([""])

say(["## 1. 單筆長抱：黃金 1 倍 vs 2 倍", ""])
for cur, f in (("美元", lambda r: r), ("台幣", in_twd)):
    say([f"### {cur}計價", "", "| | 期間 | 年化 | 最大跌幅 | 最差12個月 | 2008 | 2011/9~2015/12 黃金熊市 | 2022 |", "|---|---|---|---|---|---|---|---|"])
    rows = [("黃金 1 倍（GLD 型）", g1), ("黃金 2 倍（UGL 型）", g2), ("黃金 1.5 倍", lev(gold, 1.5, usd_rf, G2_FEE, G2_SPREAD))]
    if cur == "台幣":
        rows += [("台股黃金 1 倍（00635U 型，期貨、無匯率）", None), ("台股黃金正2（00708L 型，期貨、無匯率）", None)]
    for lab, r in rows:
        if r is None:
            x = (tw_g1 if "1 倍" in lab else tw_g2)["2004-04-01":]
        else:
            x = f(r)["2000-09-01":] if cur == "美元" else f(r)["2004-04-01":]
        say([f"| {lab} | {x.index[0].year}~ | {P(cagr(x),1)} | {P(mdd(x))} | {P(w12(x))} | {P(per(x,'2008-01','2008-12'))} | "
             f"{P(per(x,'2011-09-06','2015-12-17'))} | {P(per(x,'2022-01','2022-12'))} |"])
    say([""])

say(["### 幾倍黃金年化最高？（美元，每日再平衡）", "", "| 期間 | 1x | 1.5x | 2x | 2.5x | 3x | 最佳 |", "|---|---|---|---|---|---|---|"])
for lab, a, b in (("2000–2026", "2000-09-01", None), ("2000–2011 黃金大多頭", "2000-09-01", "2011-09-06"),
                  ("2011–2026", "2011-09-06", None), ("2011–2018 黃金熊市＋盤整", "2011-09-06", "2018-12-31")):
    x = gold[a:b] if b else gold[a:]
    cs = {k: cagr(lev(x, k, usd_rf, G2_FEE if k > 1 else 0, G2_SPREAD)) for k in np.arange(0.5, 3.01, 0.1)}
    best = max(cs, key=cs.get)
    say([f"| {lab} | " + " | ".join(P(cagr(lev(x, k, usd_rf, G2_FEE if k > 1 else 0, G2_SPREAD)), 1) for k in (1, 1.5, 2, 2.5, 3)) + f" | {best:.1f}x |"])
say([""])


def mix(parts):
    df = pd.concat([p for _, p in parts], axis=1, sort=True).dropna()
    w0 = np.array([w for w, _ in parts]); h = w0.copy(); out = []
    ye = set(df.groupby(df.index.year).tail(1).index)
    for d, row in df.iterrows():
        h = h * (1 + row.to_numpy()); t = h.sum(); out.append(t)
        if d in ye:
            h = w0 * t
    e = pd.Series(out, df.index)
    return e.pct_change().fillna(e.iloc[0] - 1)


tw2 = lev(taiex, 2, tw_rf, 0.0104)
g1t, g2t = in_twd(g1), in_twd(g2)
say(["## 2. 當保險：配上台股正2（台幣、2004 起、每年再平衡）", "",
     "「黃金曝險」＝黃金部位 × 倍數。用 2 倍黃金，可以用一半的錢買到同樣的保險，多出來的錢放股票。",
     "沒標台股的黃金＝美股 GLD／UGL 換算台幣（含匯率）。", "",
     "| 組合 | 黃金曝險 | 年化 | 最大跌幅 | 最差12個月 | 2008 | 2011~2015 | 2022 |", "|---|---|---|---|---|---|---|---|"])
for lab, expo, parts in (("100% 正2", "0%", [(1, tw2)]),
                         ("70% 正2＋30% 黃金 1 倍", "30%", [(0.7, tw2), (0.3, g1t)]),
                         ("85% 正2＋15% 黃金 2 倍", "30%", [(0.85, tw2), (0.15, g2t)]),
                         ("70% 正2＋30% 黃金 2 倍", "60%", [(0.7, tw2), (0.3, g2t)]),
                         ("80% 正2＋20% 黃金 1 倍", "20%", [(0.8, tw2), (0.2, g1t)]),
                         ("90% 正2＋10% 黃金 2 倍", "20%", [(0.9, tw2), (0.1, g2t)]),
                         ("50% 正2＋50% 黃金 1 倍", "50%", [(0.5, tw2), (0.5, g1t)]),
                         ("70% 正2＋30% 台股黃金 1 倍（00635U 型）", "30%", [(0.7, tw2), (0.3, tw_g1)]),
                         ("85% 正2＋15% 台股黃金正2（00708L 型）", "30%", [(0.85, tw2), (0.15, tw_g2)])):
    r = mix(parts)["2004-04-01":]
    say([f"| {lab} | {expo} | {P(cagr(r),1)} | {P(mdd(r))} | {P(w12(r))} | {P(per(r,'2008-01','2008-12'))} | "
         f"{P(per(r,'2011-09-06','2015-12-17'))} | {P(per(r,'2022-01','2022-12'))} |"])
say([""])

with open(os.path.join(HERE, "results", "gold_research.md"), "w") as fh:
    fh.write("\n".join(OUT))
print("\n".join(OUT))
