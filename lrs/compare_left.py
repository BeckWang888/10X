"""每月定額投入下，比較：定期定額長抱、左側分批長抱（cheap-dashboard 規則）、LRS。

用法：python lrs/compare_left.py <cheap-dashboard 路徑>
- 價格：lrs/results/prices.csv.gz（backtest_lrs.py 在 GitHub Actions 上抓的 Yahoo 資料）
- 便宜度分數與「跨上 20/50/80」事件：直接 import cheap-dashboard 的 cheapdash.model / cheapdash.backtest
- 3 倍 ETF：用指數報酬模擬（和 backtest_lrs.py 同一套，已用真實 TQQQ/SOXL 校驗）
- 閒置現金一律以國庫券利率計息（對左側分批與 LRS 公平）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from cheapdash.backtest import TRANCHES, cross_events  # noqa: E402
from cheapdash.model import compute  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
px = pd.read_csv(os.path.join(HERE, "results", "prices.csv.gz"), index_col=0, parse_dates=True)
rf = (px["^IRX"].ffill() / 100 / 252).fillna(0)
FEE_3X, SPREAD, COST = 0.009, 0.005, 0.001


def total_return(idx, etf, etf_fee, div_guess):
    r = px[idx].pct_change() + div_guess / 252
    re = px[etf].pct_change() + etf_fee / 252
    ok = re.notna()
    r[ok] = re[ok]
    return r.dropna()


UND = {
    "TQQQ": total_return("^NDX", "QQQ", 0.0020, 0.006),
    "SOXL": total_return("^SOX", "SOXX", 0.0035, 0.008),
}


def lev(r, k=3):
    f = rf.reindex(r.index).fillna(0)
    return k * r - (k - 1) * (f + SPREAD / 252) - FEE_3X / 252


def irr_monthly(dates, flows, final_value, end):
    """每月存入的資金加權年化報酬（二分法解 IRR）。"""
    t = np.array([(end - d).days / 365.25 for d in dates])
    f = np.array(flows)

    def npv(r):
        return (f * (1 + r) ** t).sum() - final_value

    lo, hi = -0.99, 5.0
    if npv(lo) > 0:
        return lo
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def simulate(r_risky, buy_rule, start, end=None):
    """buy_rule(d, holding_flag) → 回傳 (要買進的現金比例, 是否全部賣出)。每月第一個交易日存 1。"""
    idx = r_risky.index[(r_risky.index >= start) & ((r_risky.index <= end) if end else True)]
    f = rf.reindex(idx).fillna(0)
    rr = r_risky.reindex(idx).fillna(0)
    first_of_month = set(idx.to_series().groupby(idx.to_period("M")).first())
    risky = cash = dep = 0.0
    vals, deps, flows_d = [], [], []
    for d in idx:
        risky *= 1 + rr[d]
        cash *= 1 + f[d]
        if d in first_of_month:
            cash += 1; dep += 1; flows_d.append(d)
        frac, sell_all = buy_rule(d)
        if sell_all and risky > 0:
            cash += risky * (1 - COST); risky = 0.0
        if frac > 0 and cash > 0:
            amt = cash * frac
            risky += amt * (1 - (COST if buy_rule.__name__ == "lrs" else 0))
            cash -= amt
        vals.append(risky + cash); deps.append(dep)
    v = pd.Series(vals, idx); dp = pd.Series(deps, idx)
    final = v.iloc[-1]
    irr = irr_monthly(flows_d, [1.0] * len(flows_d), final, idx[-1])
    # 帳戶「相對已投入本金」的最大虧損，與帳戶淨值從高點的最大跌幅（扣掉新存入的影響：用每單位淨值）
    units = pd.Series(np.nan, idx)
    nav = 1.0; u = 0.0; prev_v = None
    navs = []
    for d, val, ddp in zip(idx, vals, deps):
        if prev_v is None:
            u = val / nav if val else 0
        else:
            new_money = ddp - prev_dep
            if u > 0:
                nav = (val - new_money) / u
            u += new_money / nav
        navs.append(nav); prev_v = val; prev_dep = ddp
    nav_s = pd.Series(navs, idx)
    return dict(final_multiple=round(final / dep, 2), irr=round(irr * 100, 1), deposited_months=int(dep),
                max_dd=round((nav_s / nav_s.cummax() - 1).min() * 100, 1),
                worst_vs_principal=round((v / dp - 1).min() * 100, 1),
                end_cash_pct=round(cash / final * 100, 0) if final else None), v


rows = []
detail = {}
for name, r1 in UND.items():
    level = (1 + r1).cumprod()
    m = compute(level)                         # 你的便宜度分數（全部歷史百分位）
    score = m["score"]
    sma200 = level.rolling(200).mean()
    above = (level > sma200)
    r3 = lev(r1)

    buy_frac = {}
    left_share = 1.0
    for th, share in TRANCHES:
        frac, left_share = share / left_share, left_share - share
        for d in cross_events(score.dropna(), th):
            buy_frac[d] = max(buy_frac.get(d, 0), frac)

    def dca(d):
        return 1.0, False

    def left(d):
        return buy_frac.get(d, 0.0), False

    def lrs(d):
        a = bool(above.get(d, False))
        return (1.0 if a else 0.0), (not a)

    def left_lrs(d):  # 混合：左側分批買進，但跌破 200 日線時整包出場、站回時再全部買回
        return lrs(d)

    first = score.first_valid_index()
    periods = [("分數可用起（全期間）", first, None), ("2000 起", pd.Timestamp("2000-01-01"), None),
               ("2010 起", pd.Timestamp("2010-03-01"), None), ("2015 起", pd.Timestamp("2015-01-01"), None),
               ("2020 起", pd.Timestamp("2020-01-01"), None),
               ("2000–2009", pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")),
               ("2010–2019", pd.Timestamp("2010-03-01"), pd.Timestamp("2019-12-31"))]
    for pname, a, b in periods:
        a = max(a, first)
        for sname, rule, series in (("定期定額 1x（大盤）", dca, r1), ("定期定額 3x 長抱", dca, r3),
                                    ("左側分批 3x 長抱", left, r3), ("LRS 3x", lrs, r3),
                                    ("左側分批 1x 長抱", left, r1)):
            s, v = simulate(series, rule, a, b)
            rows.append(dict(target=name, period=pname, strategy=sname, **s))
            if pname.startswith("分數可用"):
                detail[(name, sname)] = v
    n_ev = {th: len(cross_events(score.dropna(), th)) for th, _ in TRANCHES}
    yrs = score.notna().sum() / 252
    print(name, "分數起點", first.date(), "跨上事件", n_ev, f"（{yrs:.0f} 年）")

df = pd.DataFrame(rows)
out = os.path.join(HERE, "results", "compare_left.md")
lines = ["# 每月定額：左側分批長抱 vs LRS vs 定期定額（自動產生）", "",
         "- 每月第一個交易日存入 1 份；閒置現金以國庫券計息；LRS 每次換倉成本 0.1%",
         "- 左側分批＝cheap-dashboard 規則：分數跨上 20／50／80 投入當時現金 20%／30%／50%，買了不賣",
         "- 資產倍數＝期末資產÷累計投入；年化＝資金加權報酬（IRR）；最大跌幅＝扣除新存入後的淨值跌幅；"
         "最慘時帳面＝相對已投入本金的最大虧損", ""]
for (t, p), g in df.groupby(["target", "period"], sort=False):
    lines += [f"## {t}｜{p}", "", "| 策略 | 投入月數 | 資產倍數 | 年化(IRR) | 最大跌幅 | 最慘時帳面 | 期末現金% |", "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r.strategy} | {r.deposited_months} | {r.final_multiple}x | {r.irr}% | {r.max_dd}% | {r.worst_vs_principal}% | {r.end_cash_pct:.0f} |")
    lines.append("")
open(out, "w").write("\n".join(lines))
print("\n".join(lines))
