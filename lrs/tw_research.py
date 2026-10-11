"""台股正2、0050、VT／WLDU、黃金、單筆 vs 選點。

用法：python lrs/tw_research.py → lrs/results/tw_research.md
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
    return r.where(r.abs() < lim, 0.0)          # Yahoo 偶發錯誤跳空（如 0050 2014-01-02）當作 0


# ---------------- 利率：美元用國庫券；台幣用近似值（無免費日資料）
usd_rf = (px["^IRX"].ffill() / 100 / 252).fillna(0)
idx_all = px.index
tw_rate = pd.Series(np.select([idx_all < "2001-01-01", idx_all < "2009-01-01"], [0.045, 0.02], 0.012), idx_all)
tw_rf = tw_rate / 252

TW_DIV = 0.032   # 加權指數平均現金殖利率（估計，用真實正2 ETF 校驗）


def lev(r, k, rf, fee, spread=0.0):
    f = rf.reindex(r.index).fillna(0)
    return k * r - (k - 1) * (f + spread / 252) - fee / 252


taiex = clean_ret(px["^TWII"]) + TW_DIV / 252                                   # 加權報酬指數（估）
tw50 = clean_ret(px["0050.TW"]) + 0.0043 / 252                                  # 0050 還原（加回費用）
tw50 = tw50.reindex(taiex.index).where(lambda x: x.notna(), taiex).dropna()     # 2009 前用加權代替
vt = clean_ret(px["VT"]) + 0.0006 / 252
vt_proxy = (0.58 * clean_ret(px["VTSMX"]) + 0.42 * clean_ret(px["VGTSX"])).dropna()
vt = vt.reindex(vt_proxy.index.union(vt.index)).where(lambda x: x.notna(), vt_proxy).dropna()
sp = clean_ret(px["VFINX"]) + 0.0014 / 252
gold = clean_ret(px["GC=F"])
fx = px["TWD=X"].dropna()
fx = fx[(fx > 25) & (fx < 40)]
fxr = fx.pct_change().dropna()


def in_twd(r):
    f = fxr.reindex(r.index)
    m = f.notna()
    return ((1 + r[m]) * (1 + f[m]) - 1)


TW_FEE = 0.0104
S = {
    "加權指數 1x（含息）": taiex,
    "0050 1x": tw50,
    "加權正2（00675L 型）": lev(taiex, 2, tw_rf, TW_FEE),
    "0050正2（00631L 型）": lev(tw50, 2, tw_rf, TW_FEE),
    "VT 1x（美元）": vt,
    "WLDU 2x VT（美元）": lev(vt, 2, usd_rf, 0.0075, 0.005),
    "S&P500 1x（美元）": sp,
    "黃金（美元）": gold,
}


def cagr(r):
    r = r.dropna(); return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 50 else np.nan


def mdd(r):
    e = (1 + r.dropna()).cumprod(); return (e / e.cummax() - 1).min()


def w12(r):
    e = (1 + r.dropna()).cumprod(); return e.pct_change(252).min()


def vol(r):
    return r.std() * np.sqrt(252)


def per(r, a, b):
    x = r[a:b]; return (1 + x).prod() - 1 if len(x) > 5 else np.nan


def P(x, d=0):
    return "—" if x is None or x != x else f"{x*100:.{d}f}%"


# ---------------- 0. 校驗
say(["# 台股正2、VT／WLDU、黃金研究（自動產生）", "",
     f"- 加權報酬指數＝加權指數＋估計殖利率 {TW_DIV*100:.1f}%；台幣借貸成本 2009 後估 1.2%；正2 費用 1.04%",
     "- 模擬校驗（真實 ETF 上市後）：", "", "| 真實 | 模擬 | 期間 | 真實年化 | 模擬年化 | 日相關 |", "|---|---|---|---|---|---|"])
for real, sim in (("00631L.TW", "0050正2（00631L 型）"), ("00675L.TW", "加權正2（00675L 型）"),
                  ("00663L.TW", "加權正2（00675L 型）"), ("00685L.TW", "加權正2（00675L 型）"),
                  ("006208.TW", "0050 1x"), ("VT", "VT 1x（美元）"), ("WLDU", "WLDU 2x VT（美元）")):
    rr = clean_ret(px[real]); ss = S[sim]
    c = rr.index.intersection(ss.index)
    c = c[c >= pd.Timestamp("2015-02-01")] if real == "00631L.TW" else c
    say([f"| {real} | {sim} | {c[0].date()}~{c[-1].date()} | {P(cagr(rr[c]),1)} | {P(cagr(ss[c]),1)} | {np.corrcoef(rr[c], ss[c])[0,1]:.3f} |"])
say([""])

# ---------------- 1. 標的比較（單筆長抱）
say(["## 1. 單筆長抱：各標的比較", ""])
for lab, a in (("1997 起（含網路泡沫、2008）", "1997-08-01"), ("2009 起", "2009-01-01"), ("2016 起（真實正2 上市後）", "2016-10-01")):
    say([f"### {lab}", "", "| 標的 | 年化 | 最大跌幅 | 最差12個月 | 波動 | 網路泡沫 | 2008 | 2022 |", "|---|---|---|---|---|---|---|---|"])
    for k, r in S.items():
        x = r[a:]
        if len(x) < 252 or x.index[0] > pd.Timestamp(a) + pd.Timedelta(days=400):
            continue
        say([f"| {k} | {P(cagr(x),1)} | {P(mdd(x))} | {P(w12(x))} | {P(vol(x))} | {P(per(x,'2000-03','2002-10'))} | "
             f"{P(per(x,'2008-01','2008-12'))} | {P(per(x,'2022-01','2022-12'))} |"])
    say([""])

# ---------------- 2. 單筆 vs 選點（買點等待）
def outcomes(r, rf, start_dates, years, rule):
    res = []
    lvl = (1 + r).cumprod()
    hi52 = lvl.rolling(252, min_periods=60).max()
    ma200 = lvl.rolling(200).mean()
    f = (1 + rf.reindex(r.index).fillna(0)).cumprod()
    for d0 in start_dates:
        d1 = d0 + pd.DateOffset(years=years)
        seg = r.index[(r.index >= d0) & (r.index < d1)]
        if len(seg) < 200 or seg[-1] < d1 - pd.Timedelta(days=10):
            continue
        end = seg[-1]
        if rule == "單筆":
            v = lvl[end] / lvl[seg[0]]
        elif rule.startswith("分"):
            n = int(rule[1:3])
            months = pd.date_range(seg[0], periods=n, freq="MS")
            v = 0
            for m in months:
                dd = r.index[r.index >= m][0]
                v += (1 / n) * (f[dd] / f[seg[0]]) * (lvl[end] / lvl[dd])
        else:  # 等回檔
            if rule == "等跌10%":
                cond = (lvl[seg] / hi52[seg] - 1) <= -0.10
            elif rule == "等跌20%":
                cond = (lvl[seg] / hi52[seg] - 1) <= -0.20
            else:  # 等跌破200日線
                cond = lvl[seg] < ma200[seg]
            hit = cond[cond].index
            if len(hit):
                dd = hit[0]
                v = (f[dd] / f[seg[0]]) * (lvl[end] / lvl[dd])
            else:
                v = f[end] / f[seg[0]]
        res.append(v)
    return np.array(res)


say(["## 2. 有一大筆錢：一次投入 vs 分批 vs 等回檔再買（「選點」）", "",
     "- 每個月都當作起點，看 5 年與 10 年後的資產倍數；等待中的錢放定存／國庫券",
     "- 「贏時平均多」＝一次投入贏的那些起點，平均多賺幾 %；「輸時平均少」＝一次投入輸的那些起點，平均少賺幾 %", ""])
for name, r, rf in (("加權指數 1x", taiex, tw_rf), ("加權正2", S["加權正2（00675L 型）"], tw_rf),
                    ("VT 1x", vt, usd_rf), ("WLDU 2x VT", S["WLDU 2x VT（美元）"], usd_rf)):
    starts = pd.date_range(r.index[0] + pd.DateOffset(years=1), r.index[-1], freq="MS")
    for years in (5, 10):
        base = outcomes(r, rf, starts, years, "單筆")
        say([f"### {name}｜持有 {years} 年（{len(base)} 個起點）", "",
             "| 方法 | 資產倍數中位數 | 最差 | 虧本機率 | 一次投入贏 | 這個方法贏 | 一樣（起點就符合條件） | 贏時平均多 | 輸時平均少 |",
             "|---|---|---|---|---|---|---|---|---|",
             f"| 一次投入 | {np.median(base):.2f}x | {base.min():.2f}x | {(base<1).mean()*100:.0f}% | — | — | — | — | — |"])
        for rule in ("分12個月", "分24個月", "等跌10%", "等跌20%", "等跌破200日線"):
            o = outcomes(r, rf, starts, years, rule)
            d = base / o - 1
            win, lose = d > 1e-6, d < -1e-6
            tie = ~(win | lose)
            say([f"| {rule} | {np.median(o):.2f}x | {o.min():.2f}x | {(o<1).mean()*100:.0f}% | {win.mean()*100:.0f}% | "
                 f"{lose.mean()*100:.0f}% | {tie.mean()*100:.0f}% | {P(d[win].mean() if win.any() else np.nan)} | "
                 f"{P(-d[lose].mean() if lose.any() else np.nan)} |"])
        say([""])

# ---------------- 3. 台灣投資人的角度（台幣計價）
say(["## 3. 用台幣計算（2004 起，匯率資料起點）", "", "| 標的 | 年化（台幣） | 最大跌幅 | 與加權正2 的相關係數 |", "|---|---|---|---|"])
tw2 = S["加權正2（00675L 型）"]
for k in ("加權指數 1x（含息）", "0050 1x", "加權正2（00675L 型）", "0050正2（00631L 型）", "VT 1x（美元）", "WLDU 2x VT（美元）", "S&P500 1x（美元）", "黃金（美元）"):
    r = S[k]
    r = in_twd(r) if "美元" in k else r
    x = r["2004-04-01":]
    c = x.index.intersection(tw2.index)
    # 週報酬相關（避開台美時差）
    wk = lambda s: (1 + s).resample("W").prod() - 1
    corr = np.corrcoef(wk(x[c]), wk(tw2[c]))[0, 1]
    say([f"| {k.replace('（美元）','')} | {P(cagr(x),1)} | {P(mdd(x))} | {corr:.2f} |"])
say([""])

# ---------------- 4. 黃金當保險：加進組合的效果（台幣、每年再平衡）
def mix(parts):
    df = pd.concat([p for _, p in parts], axis=1).dropna()
    w0 = np.array([w for w, _ in parts]); h = w0.copy(); out = []
    ye = set(df.groupby(df.index.year).tail(1).index)
    for d, row in df.iterrows():
        h = h * (1 + row.to_numpy()); t = h.sum(); out.append(t)
        if d in ye:
            h = w0 * t
    e = pd.Series(out, df.index)
    return e.pct_change().fillna(e.iloc[0] - 1)


g_twd = in_twd(gold)
vt_twd = in_twd(vt)
wl_twd = in_twd(S["WLDU 2x VT（美元）"])
say(["## 4. 黃金當保險：放進組合（台幣計價、每年再平衡、2004 起）", "",
     "| 組合 | 年化 | 最大跌幅 | 最差12個月 | 2008 | 2022 |", "|---|---|---|---|---|---|"])
for lab, parts in (("100% 加權正2", [(1, tw2)]), ("80% 加權正2＋20% 黃金", [(0.8, tw2), (0.2, g_twd)]),
                   ("70% 加權正2＋30% 黃金", [(0.7, tw2), (0.3, g_twd)]),
                   ("50% 加權正2＋50% 現金（≈1x）", [(0.5, tw2), (0.5, tw_rf)]),
                   ("100% 加權 1x", [(1, taiex)]), ("100% VT（台幣）", [(1, vt_twd)]),
                   ("50% 加權正2＋50% WLDU", [(0.5, tw2), (0.5, wl_twd)]),
                   ("40% 加權正2＋40% WLDU＋20% 黃金", [(0.4, tw2), (0.4, wl_twd), (0.2, g_twd)]),
                   ("100% 黃金", [(1, g_twd)])):
    r = mix(parts)["2004-04-01":]
    say([f"| {lab} | {P(cagr(r),1)} | {P(mdd(r))} | {P(w12(r))} | {P(per(r,'2008-01','2008-12'))} | {P(per(r,'2022-01','2022-12'))} |"])
say([""])
# 黃金在股市大跌時
say(["### 股市大跌期間，黃金表現（美元）", "", "| 期間 | S&P500 | 加權（含息） | 黃金 |", "|---|---|---|---|"])
for lab, a, b in (("2000/9~2002/10 網路泡沫後段", "2000-09-01", "2002-10-09"), ("2007/10~2009/3 金融海嘯", "2007-10-09", "2009-03-09"),
                  ("2020/2~3 疫情", "2020-02-19", "2020-03-23"), ("2022 升息", "2022-01-03", "2022-10-12"),
                  ("2011/9~2015/12 黃金自己的熊市", "2011-09-06", "2015-12-17")):
    say([f"| {lab} | {P(per(sp,a,b))} | {P(per(taiex,a,b))} | {P(per(gold,a,b))} |"])
say([""])

with open(os.path.join(HERE, "results", "tw_research.md"), "w") as fh:
    fh.write("\n".join(OUT))
print("\n".join(OUT))

# ---------------- 5. 幾倍槓桿最好？
say(["## 5. 單筆長抱：幾倍槓桿年化最高？（每日再平衡，含成本）", "",
     "| 標的 | 期間 | 1x | 1.5x | 2x | 2.5x | 3x | 最佳倍數 |", "|---|---|---|---|---|---|---|---|"])
for name, r, rf, fee in (("加權（含息）", taiex, tw_rf, TW_FEE), ("VT", vt, usd_rf, 0.0075)):
    for lab, a, b in (("全期間", None, None), ("2000–2012", "2000-01-01", "2012-12-31"), ("2009 起", "2009-01-01", None)):
        x = r[a:b] if a else r
        cs = {k: cagr(lev(x, k, rf, fee if k > 1 else 0)) for k in np.arange(0.5, 3.51, 0.1)}
        best = max(cs, key=cs.get)
        say([f"| {name} | {lab}（{x.index[0].year}~{x.index[-1].year}） | " +
             " | ".join(P(cagr(lev(x, k, rf, fee if k > 1 else 0)), 1) for k in (1, 1.5, 2, 2.5, 3)) + f" | {best:.1f}x |"])
say([""])
with open(os.path.join(HERE, "results", "tw_research.md"), "w") as fh:
    fh.write("\n".join(OUT))
