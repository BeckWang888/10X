"""散戶長期投資研究：定期定額間隔、最佳標的與槓桿倍數、防大跌策略。

用法：python lrs/research.py → lrs/results/research.md
資料：lrs/results/prices_more.csv.gz（GitHub Actions 抓的 Yahoo 還原價）
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
px = pd.read_csv(os.path.join(HERE, "results", "prices_more.csv.gz"), index_col=0, parse_dates=True)
rf = (px["^IRX"].ffill() / 100 / 252).fillna(0)
FEE_LEV, SPREAD = 0.009, 0.005
OUT = []


def say(*lines):
    OUT.extend(lines)


def tr(primary, fee, fallback=None, fb_div=0.0):
    """還原股息的日報酬；primary（基金/ETF）上市前用 fallback 指數＋估計股息。加回管理費 → 接近指數總報酬。"""
    r = px[primary].pct_change() + fee / 252
    if fallback:
        rb = px[fallback].pct_change() + fb_div / 252
        r = r.where(r.notna(), rb)
    return r.dropna()


U = {
    "S&P500(VOO)": tr("VFINX", 0.0014),
    "那斯達克100(QQQ)": tr("QQQ", 0.0020, "^NDX", 0.006),
    "費半(SOXX)": tr("SOXX", 0.0035, "^SOX", 0.008),
    "SMH": tr("SMH", 0.0035),
}
BOND = tr("VUSTX", 0.002)          # 長天期美國公債
GOLD = px["GC=F"].pct_change().dropna()


def lev(r, k):
    f = rf.reindex(r.index).fillna(0)
    return k * r - (k - 1) * (f + SPREAD / 252) - (FEE_LEV / 252 if k > 1 else 0)


def cagr(r):
    r = r.dropna()
    return (1 + r).prod() ** (252 / len(r)) - 1


def maxdd(r):
    eq = (1 + r.dropna()).cumprod()
    return (eq / eq.cummax() - 1).min()


def worst12(r):
    eq = (1 + r.dropna()).cumprod()
    return eq.pct_change(252).min()


def sharpe(r):
    r = r.dropna(); ex = r - rf.reindex(r.index).fillna(0)
    return ex.mean() / ex.std() * np.sqrt(252)


def yr(r, y):
    x = r[str(y)]
    return (1 + x).prod() - 1 if len(x) else np.nan


def pct(x, d=1):
    return "—" if x is None or x != x else f"{x*100:.{d}f}%"


# ---------------------------------------------------------------- 0. 模擬校驗
say("# 散戶長期投資研究（自動產生）", "",
    "- 槓桿 ETF 用「k×指數日報酬 −(k−1)×(國庫券＋0.5%) − 管理費 0.9%」模擬；下表為模擬和真實 ETF 的比對", "",
    "| 真實 ETF | 模擬 | 期間 | 真實年化 | 模擬年化 | 日報酬相關 |", "|---|---|---|---|---|---|")
for etf, base, k in (("SSO", "S&P500(VOO)", 2), ("UPRO", "S&P500(VOO)", 3), ("QLD", "那斯達克100(QQQ)", 2),
                     ("TQQQ", "那斯達克100(QQQ)", 3), ("USD", "費半(SOXX)", 2), ("SOXL", "費半(SOXX)", 3)):
    real = px[etf].pct_change().dropna()
    sim = lev(U[base], k)
    c = real.index.intersection(sim.index)
    say(f"| {etf} | {base}×{k} | {c[0].date()}~{c[-1].date()} | {pct(cagr(real[c]))} | {pct(cagr(sim[c]))} | {np.corrcoef(real[c], sim[c])[0,1]:.3f} |")
say("", "（USD 追蹤的半導體指數和費半不同，所以相關性較低）", "")


# ---------------------------------------------------------------- 1. 定期定額的間隔與日期
def dca_wealth(r, start, end, schedule):
    """每月 1 份錢在當月第一個交易日入帳（先放國庫券），依 schedule 在當月哪些交易日投入多少比例。
    schedule(n_days_in_month) → list[(第幾個交易日 index, 比例)]。回傳 (期末資產, 投入月數)"""
    idx = r.index[(r.index >= start) & (r.index < end)]
    rr = r.reindex(idx).to_numpy(); f = rf.reindex(idx).fillna(0).to_numpy()
    per = idx.to_period("M")
    pos_in_month = pd.Series(1, idx).groupby(per).cumcount().to_numpy()
    n_in_month = pd.Series(1, idx).groupby(per).transform("size").to_numpy()
    risky = cash = 0.0; months = 0
    cache = {}
    for i in range(len(idx)):
        risky *= 1 + rr[i]; cash *= 1 + f[i]
        p, n = pos_in_month[i], n_in_month[i]
        if p == 0:
            cash += 1; months += 1; owed = 1.0
        if n not in cache:
            cache[n] = dict()
            for d, w in schedule(n):
                cache[n][d] = cache[n].get(d, 0) + w
        w = cache[n].get(p, 0)
        if w:
            amt = min(cash, w); risky += amt; cash -= amt
    return risky + cash, months


SCHED = {
    "每月第1個交易日": lambda n: [(0, 1.0)],
    "每月第6個交易日（約6號）": lambda n: [(5, 1.0)],
    "每月第11個交易日（約15號）": lambda n: [(10, 1.0)],
    "每月最後第3個交易日": lambda n: [(n - 3, 1.0)],
    "每月最後1個交易日": lambda n: [(n - 1, 1.0)],
    "每月2次（1號＋15號）": lambda n: [(0, 0.5), (10, 0.5)],
    "每週1次": lambda n: [(d, 1 / len(range(0, n, 5))) for d in range(0, n, 5)],
    "每天": lambda n: [(d, 1 / n) for d in range(n)],
}

say("## 1. 定期定額：哪一天、多久投一次比較好？", "",
    "- 每月薪水在月初入帳，差別只在「這個月的錢哪一天投進去」；還沒投的錢放國庫券",
    "- 比較方式：所有「10 年滾動區間」（每個月起算一次）期末資產相對「每月第1個交易日」的差距", "")
for name in ("S&P500(VOO)", "那斯達克100(QQQ)"):
    r = U[name]
    starts = pd.date_range(r.index[0] + pd.offsets.MonthBegin(1), r.index[-1] - pd.DateOffset(years=10), freq="6MS")
    res = {k: [] for k in SCHED}
    for s in starts:
        e = s + pd.DateOffset(years=10)
        base, _ = dca_wealth(r, s, e, SCHED["每月第1個交易日"])
        for k, sc in SCHED.items():
            w, _ = dca_wealth(r, s, e, sc)
            res[k].append(w / base - 1)
    say(f"### {name}（{len(starts)} 個 10 年區間，{starts[0].year}~{starts[-1].year} 起算）", "",
        "| 投入時點 | 期末資產平均差距 | 贏過月初的比例 | 最好 | 最差 |", "|---|---|---|---|---|")
    for k, v in res.items():
        v = np.array(v)
        say(f"| {k} | {v.mean()*100:+.2f}% | {(v > 1e-9).mean()*100:.0f}% | {v.max()*100:+.2f}% | {v.min()*100:+.2f}% |")
    say("")

# 單筆 vs 定期定額（同一筆錢 12 個月分批）
say("### 一筆錢：立刻全部投入 vs 分 12 個月投入（之後都持有到滿 10 年）", "",
    "| 標的 | 區間數 | 單筆勝出比例 | 單筆平均多賺 |", "|---|---|---|---|")
for name in ("S&P500(VOO)", "那斯達克100(QQQ)", "費半(SOXX)"):
    r = U[name]; eq = (1 + r).cumprod(); f = rf.reindex(r.index).fillna(0); cq = (1 + f).cumprod()
    m_idx = r.index.to_series().groupby(r.index.to_period("M")).first()
    wins, diffs = [], []
    for i in range(0, len(m_idx) - 121):
        d0, dN = m_idx.iloc[i], m_idx.iloc[i + 120]
        ls = eq[dN] / eq[d0]
        dca = 0
        for j in range(12):
            dj = m_idx.iloc[i + j]
            dca += (1 / 12) * (cq[dj] / cq[d0]) * (eq[dN] / eq[dj])
        wins.append(ls > dca); diffs.append(ls / dca - 1)
    say(f"| {name} | {len(wins)} | {np.mean(wins)*100:.0f}% | {np.mean(diffs)*100:+.1f}% |")
say("")


# ---------------------------------------------------------------- 2. 定期定額最佳標的與槓桿倍數
def dca_series(r, start, end):
    """每月第1個交易日投入 1，回傳 (IRR, 期末倍數, 相對本金最慘)"""
    idx = r.index[(r.index >= start) & (r.index < end)]
    rr = r.reindex(idx).to_numpy()
    first = pd.Series(1, idx).groupby(idx.to_period("M")).cumcount().to_numpy() == 0
    v = 0.0; dep = 0.0; worst = 0.0; t_dep = []
    for i in range(len(idx)):
        v *= 1 + rr[i]
        if first[i]:
            v += 1; dep += 1; t_dep.append(idx[i])
        worst = min(worst, v / dep - 1)
    T = np.array([(idx[-1] - d).days / 365.25 for d in t_dep])
    lo, hi = -0.99, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 + mid) ** T @ np.ones(len(T)) > v:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2, v / dep, worst


say("## 2. 定期定額最佳標的：1 倍、2 倍、3 倍", "",
    "- 每月第1個交易日投入；看所有 10 年與 20 年滾動區間（每季起算一次）",
    "- IRR＝資金加權年化報酬；虧本機率＝期末資產 < 投入本金；最慘帳面＝期間中相對已投入本金最大虧損", "")
for years in (10, 20):
    say(f"### {years} 年定期定額", "",
        "| 標的 | 區間數 | IRR 中位數 | IRR 最差10% | IRR 最差 | 虧本機率 | 最慘帳面中位數 | 最慘帳面最差 |",
        "|---|---|---|---|---|---|---|---|")
    for name in ("S&P500(VOO)", "那斯達克100(QQQ)", "費半(SOXX)", "SMH"):
        for k in (1, 2, 3):
            r = lev(U[name], k)
            starts = pd.date_range(r.index[0] + pd.offsets.MonthBegin(1), r.index[-1] - pd.DateOffset(years=years), freq="QS")
            if len(starts) < 4:
                continue
            irr, mult, worst = zip(*[dca_series(r, s, s + pd.DateOffset(years=years)) for s in starts])
            irr, mult, worst = map(np.array, (irr, mult, worst))
            say(f"| {name} ×{k} | {len(starts)}（{starts[0].year}~{starts[-1].year}） | {pct(np.median(irr))} | "
                f"{pct(np.quantile(irr, .1))} | {pct(irr.min())} | {(mult < 1).mean()*100:.0f}% | "
                f"{pct(np.median(worst),0)} | {pct(worst.min(),0)} |")
    say("")

# 同一段期間（1995 起，四個標的都有資料）比較
say("### 同一起點比較（2000-07 起每月定額至今）", "", "| 標的 | ×1 IRR | ×2 IRR | ×3 IRR | ×3 最慘帳面 |", "|---|---|---|---|---|")
for name in ("S&P500(VOO)", "那斯達克100(QQQ)", "費半(SOXX)", "SMH"):
    vals = [dca_series(lev(U[name], k), pd.Timestamp("2000-07-01"), pd.Timestamp("2100-01-01")) for k in (1, 2, 3)]
    say(f"| {name} | {pct(vals[0][0])} | {pct(vals[1][0])} | {pct(vals[2][0])} | {pct(vals[2][2],0)} |")
say("")

# ---------------------------------------------------------------- 3. 最佳槓桿倍數（單筆持有）
say("## 3. 單筆長抱：幾倍槓桿最好？（每日再平衡，含成本）", "")
for name in ("S&P500(VOO)", "那斯達克100(QQQ)", "費半(SOXX)"):
    r = U[name]
    say(f"### {name}", "", "| 期間 | " + " | ".join(f"{k}x" for k in (1, 1.5, 2, 2.5, 3, 4)) + " | 年化最高的倍數 |",
        "|---|" + "---|" * 7)
    for lab, a, b in (("全期間", r.index[0], None), ("2000 起", "2000-01-01", None), ("2010 起", "2010-01-01", None),
                      ("2000–2012", "2000-01-01", "2012-12-31")):
        rr = r[a:b] if b else r[a:]
        cs = {k: cagr(lev(rr, k)) for k in np.arange(0.5, 4.01, 0.1)}
        best = max(cs, key=cs.get)
        say(f"| {lab}（{rr.index[0].year}~） | " + " | ".join(pct(cagr(lev(rr, k)), 1) for k in (1, 1.5, 2, 2.5, 3, 4)) +
            f" | {best:.1f}x |")
    say("")


# ---------------------------------------------------------------- 4. 防大跌策略（單筆）
def strat_returns(r1, kind, k=3):
    f = rf.reindex(r1.index).fillna(0)
    lvl = (1 + r1).cumprod()
    above = (lvl > lvl.rolling(200).mean()).shift(1).fillna(False).astype(float)
    if kind == "hold":
        return lev(r1, k)
    if kind == "lrs":
        rk = lev(r1, k)
        sw = above.diff().abs().fillna(0)
        return above * rk + (1 - above) * f - sw * 0.001
    if kind == "voltarget":
        vol = r1.rolling(20).std() * np.sqrt(252)
        # 目標波動＝「到前一天為止」的長期平均波動 × k/2（不偷看未來）
        target = r1.expanding(252).std() * np.sqrt(252) * k / 2
        w = (target / vol).clip(0, k).shift(1).fillna(0)
        turn = w.diff().abs().fillna(0)
        return (w * r1 - (w - 1).clip(lower=0) * (f + SPREAD / 252) + (1 - w).clip(lower=0) * f
                - turn * 0.0005 - (w > 1) * FEE_LEV / 252)
    raise ValueError(kind)


def mix(parts, freq="QE"):
    """parts: list[(weight, returns)]；依 freq 再平衡"""
    df = pd.concat([p[1] for p in parts], axis=1).dropna()
    w0 = np.array([p[0] for p in parts])
    val = []; hold = w0.copy()
    reb = set(df.groupby(df.index.to_period(freq[0])).tail(1).index)
    out = []
    for d, row in df.iterrows():
        hold = hold * (1 + row.to_numpy())
        tot = hold.sum(); out.append(tot)
        if d in reb:
            hold = w0 * tot
    eq = pd.Series(out, df.index)
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def row(lab, r, y0):
    r = r[y0:].dropna()
    return (f"| {lab} | {pct(cagr(r))} | {pct(maxdd(r),0)} | {pct(worst12(r),0)} | {sharpe(r):.2f} | "
            f"{pct(yr(r, 2008),0)} | {pct((1 + r['2000-03':'2002-10']).prod() - 1, 0) if r.index[0] < pd.Timestamp('2000-03-01') else '—'} | "
            f"{pct(yr(r, 2022),0)} |")


say("## 4. 怎麼防史詩大跌？（單筆投入，從可比較的最早年份到現在）", "",
    "- 網路泡沫＝2000/3~2002/10；再平衡為每季；200日線、波動控制都用前一天的資料決定", "")
for name, y0 in (("S&P500(VOO)", "1987-01-01"), ("那斯達克100(QQQ)", "1987-01-01"), ("費半(SOXX)", "1996-01-01")):
    r1 = U[name]
    cash = rf.reindex(r1.index).fillna(0)
    b = BOND.reindex(r1.index)
    say(f"### {name}", "", "| 策略 | 年化 | 最大跌幅 | 最差12個月 | Sharpe | 2008 | 網路泡沫 | 2022 |", "|---|---|---|---|---|---|---|---|")
    rows = [("1x 長抱", r1), ("2x 長抱", strat_returns(r1, "hold", 2)), ("3x 長抱", strat_returns(r1, "hold", 3)),
            ("2x＋200日線（LRS）", strat_returns(r1, "lrs", 2)), ("3x＋200日線（LRS）", strat_returns(r1, "lrs", 3)),
            ("波動控制（0~2x）", strat_returns(r1, "voltarget", 2)), ("波動控制（0~3x）", strat_returns(r1, "voltarget", 3)),
            ("1/3 放 3x＋2/3 放現金（≈1x，每季再平衡）", mix([(1/3, lev(r1, 3)), (2/3, cash)])),
            ("1/2 放 3x＋1/2 放現金（≈1.5x）", mix([(0.5, lev(r1, 3)), (0.5, cash)])),
            ("核心衛星：80% 1x＋20% 3x（≈1.4x）", mix([(0.8, r1), (0.2, lev(r1, 3))])),
            ("60% 股＋40% 長天期美債", mix([(0.6, r1), (0.4, b)])),
            ("2x 股＋…：60% 2x 股＋40% 美債（≈1.2x 股）", mix([(0.6, lev(r1, 2)), (0.4, b)])),
            ("HFEA 型：55% 3x 股＋45% 3x 長債", mix([(0.55, lev(r1, 3)), (0.45, lev(b.dropna(), 3))]))]
    if name == "S&P500(VOO)":
        g = GOLD.reindex(r1.index)
        rows.append(("全天候型：40% 股＋40% 長債＋20% 黃金（2001 起）", mix([(0.4, r1), (0.4, b), (0.2, g)])))
    for lab, r in rows:
        say(row(lab, r, y0))
    say("")

# ---------------------------------------------------------------- 5. 因子與其他市場（贏 S&P 的嘗試）
say("## 5. 其他「想贏大盤」的常見方法（同期間對比 S&P 500）", "", "| 標的 | 期間 | 年化 | S&P 同期年化 | 最大跌幅 | S&P 同期最大跌幅 |", "|---|---|---|---|---|---|")
sp = U["S&P500(VOO)"]
for t, lab, fee in (("DFSVX", "美國小型價值（DFA）", 0.0), ("VISVX", "美國小型價值（Vanguard）", 0.0),
                    ("MTUM", "動能因子 MTUM", 0.0), ("QUAL", "品質因子 QUAL", 0.0), ("VGTSX", "美國以外國際股", 0.0),
                    ("VTSMX", "美國全市場", 0.0)):
    r = px[t].pct_change().dropna()
    c = r.index.intersection(sp.index)
    say(f"| {lab} | {c[0].year}~ | {pct(cagr(r[c]))} | {pct(cagr(sp[c]))} | {pct(maxdd(r[c]),0)} | {pct(maxdd(sp[c]),0)} |")
say("")

with open(os.path.join(HERE, "results", "research.md"), "w") as fh:
    fh.write("\n".join(OUT))
print("\n".join(OUT))
