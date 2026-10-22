"""資產配置回測：各種配置比較、黃金比例、防守／進攻部位換標的、台股正2 拆分、各國股市比較。

整理 2026-10-10～12 在對話中討論 cheap-dashboard「目標配置」時做的分析，用修正後的算法：
- 年化用實際經過的日曆時間（台股、美股一年交易日數不同，用 252 天換算會高估）
- 台美資產混在一起時，日期取聯集、休市那邊沿用前一天，不會整天丟掉另一邊的報酬
- 美元資產換成台幣用「淨值×匯率」

用法：python lrs/portfolio_research.py → lrs/results/portfolio_research.md
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
    return r.where(r.abs() < lim, 0.0)          # Yahoo 偶發錯誤跳空當作 0


def lev(r, k, rf, fee, spread=0.0):
    f = rf.reindex(r.index).fillna(0)
    return k * r - (k - 1) * (f + spread / 252) - fee / 252


def cagr(r):
    r = r.dropna()
    if len(r) < 50:
        return np.nan
    return (1 + r).prod() ** (365.25 / (r.index[-1] - r.index[0]).days) - 1


def mdd(r):
    e = (1 + r.dropna()).cumprod(); return (e / e.cummax() - 1).min()


def w12(r):
    e = (1 + r.dropna()).cumprod().asfreq("D").ffill(); return e.pct_change(365).min()


def per(r, a, b):
    x = r[a:b]; return (1 + x).prod() - 1 if len(x) > 5 else np.nan


def P(x, d=0):
    return "—" if x != x else f"{x*100:.{d}f}%"


# ---------------- 利率、匯率
usd_rf = (px["^IRX"].ffill() / 100 / 252).fillna(0)
tw_rate = pd.Series(np.select([px.index < "2001-01-01", px.index < "2009-01-01"], [0.045, 0.02], 0.012), px.index)
tw_rf = tw_rate / 252
fx = px["TWD=X"].dropna(); fx = fx[(fx > 25) & (fx < 40)]


def in_twd(r):
    lv = (1 + r.dropna()).cumprod()
    f = fx.reindex(lv.index.union(fx.index)).ffill()
    v = (lv.reindex(f.index).ffill() * f).dropna()
    return v[v.index >= r.dropna().index[0]].pct_change().dropna()


def mix(parts):
    """parts: list[(weight, returns)]，每年最後一個交易日再平衡"""
    lv = pd.concat([(1 + p.dropna()).cumprod() for _, p in parts], axis=1, sort=True).ffill()
    lv = lv[lv.notna().all(axis=1)]
    df = lv.pct_change().fillna(0)
    w0 = np.array([w for w, _ in parts]); h = w0.copy(); out = []
    ye = set(df.groupby(df.index.year).tail(1).index)
    for d, row in df.iterrows():
        h = h * (1 + row.to_numpy()); t = h.sum(); out.append(t)
        if d in ye:
            h = w0 * t
    e = pd.Series(out, df.index) / w0.sum()
    return e.pct_change().fillna(e.iloc[0] - 1)


def splice(primary, fee, fallback, fb_div):
    """ETF 上市前用指數＋估計股息接起來（加回管理費 ≈ 指數總報酬）"""
    a = clean_ret(px[primary]) + fee / 252
    b = clean_ret(px[fallback]) + fb_div / 252
    a = a.reindex(a.index.union(b.index))
    return a.where(a.notna(), b.reindex(a.index)).dropna()


# ---------------- 各資產日報酬
taiex = clean_ret(px["^TWII"]) + 0.032 / 252                      # 加權報酬指數（估 3.2% 股息）
tw2 = lev(taiex, 2, tw_rf, 0.0104)                                 # 加權正2（00675L／00631L 型）
sp = clean_ret(px["VFINX"]) + 0.0014 / 252
vt = clean_ret(px["VT"]) + 0.0006 / 252
vt_proxy = (0.58 * clean_ret(px["VTSMX"]) + 0.42 * clean_ret(px["VGTSX"])).dropna()
vt = vt.reindex(vt_proxy.index.union(vt.index)).where(lambda x: x.notna(), vt_proxy).dropna()
ndx = splice("QQQ", 0.002, "^NDX", 0.008)
sox = splice("SOXX", 0.0035, "^SOX", 0.008)
fxi = clean_ret(px["FXI"]) + 0.0074 / 252

T = {   # 台幣計價
    "tw2": tw2, "taiex": taiex, "vt": in_twd(vt), "sp": in_twd(sp), "fxi": in_twd(fxi),
    "semi": in_twd(sox), "semi3": in_twd(lev(sox, 3, usd_rf, 0.009, 0.005)),
    "gold": in_twd(clean_ret(px["GC=F"])), "cash": tw_rf,
    "sp2": in_twd(lev(sp, 2, usd_rf, 0.009, 0.003)), "ndx2": in_twd(lev(ndx, 2, usd_rf, 0.009, 0.003)),
    "tqqq": in_twd(lev(ndx, 3, usd_rf, 0.009, 0.005)), "ndx": in_twd(ndx),
}
START = "2005-01-01"   # 台幣匯率 2004/3、FXI 2004/10 才有資料；統一從 2005 起，各列才能比較

say(["# 資產配置回測（自動產生）", "",
     "- 台幣計價、每年最後一個交易日再平衡；起點 2005 年（台幣匯率與 FXI 資料的共同起點），所以**沒有包含 2000 年網路泡沫**",
     "- 部位代表：台股正2＝加權指數 2 倍（每日再平衡、扣費用與台幣利息）；主題 ETF＝費半 1 倍；高風險衛星＝費半 3 倍；"
     "全球 ETF＝VT；1 倍大盤＝S&P 500；黃金＝國際金價；現金＝台幣定存利率（估）",
     "- 年化用實際經過的日曆時間；台美休市日不同時，休市那邊沿用前一天，不會漏掉另一邊的報酬",
     "- 2026-10-12 修正：先前對話與舊版報告的組合年化約高估 2 個百分點、最大跌幅約少算 3～4 個百分點，本檔為修正後數字", ""])


def line(lab, parts, start=START, extra=()):
    r = mix(parts)[start:]
    cells = [P(cagr(r), 1), P(mdd(r)), P(w12(r)), P(per(r, "2008-01", "2008-12")), P(per(r, "2022-01", "2022-12")),
             P(cagr(r["2019":]), 1)] + list(extra)
    say([f"| {lab} | " + " | ".join(cells) + " |"])


HEAD = "| 年化 | 最大跌幅 | 最差12個月 | 2008 | 2022 | 2019 起年化 |"
SEP = "|---|---|---|---|---|---|---|"

# ---------------- 1. 配置比較
FINAL = [(.28, "tw2"), (.15, "semi"), (.07, "semi3"), (.18, "vt"), (.05, "sp"), (.12, "gold"), (.15, "cash")]
CONF = {
    "2026-10-10 持倉（近似）": [(.07, "taiex"), (.03, "tw2"), (.40, "semi"), (.10, "semi3"), (.12, "vt"), (.02, "gold"), (.26, "cash")],
    "積極成長": [(.30, "tw2"), (.15, "vt"), (.20, "semi"), (.10, "semi3"), (.15, "gold"), (.10, "cash")],
    "初版建議（黃金 18%）": [(.25, "tw2"), (.15, "semi"), (.07, "semi3"), (.15, "vt"), (.05, "sp"), (.18, "gold"), (.15, "cash")],
    "**定案 2026-10-11**（28／15／7／18／5／12／15）": FINAL,
    "偏防守": [(.20, "tw2"), (.10, "semi"), (.05, "semi3"), (.25, "vt"), (.20, "gold"), (.20, "cash")],
    "對照：100% 台股 1 倍": [(1, "taiex")],
    "對照：100% VT": [(1, "vt")],
}
say(["## 1. 配置比較", "", "| 配置 " + HEAD, SEP])
for k, v in CONF.items():
    line(k, [(w, T[a]) for w, a in v])
say(["", "定案配置的部位：台股正2 28%、主題 ETF 15%、高風險衛星 7%、全球 ETF 18%、1 倍大盤 5%、黃金 12%、現金 15%。", ""])

# ---------------- 2. 黃金比例
say(["## 2. 黃金放多少？（定案配置，只改黃金，其他部位等比例縮放）", "",
     "2011/9 是金價高點、之後跌了 4 年，可以看成「對黃金不利的期間」。", "",
     "| 黃金 | 2005 起年化 | 2005 起最大跌幅 | 2011/9 起年化 | 2011/9 起最大跌幅 | 2008 | 2022 |", "|---|---|---|---|---|---|---|"])
others = [(w, a) for w, a in FINAL if a != "gold"]; ow = sum(w for w, _ in others)
for g in (0, .10, .12, .15, .18, .25):
    parts = [(w / ow * (1 - g), T[a]) for w, a in others] + ([(g, T["gold"])] if g else [])
    r = mix(parts)[START:]; r2 = r["2011-09-06":]
    say([f"| {g*100:.0f}% | {P(cagr(r),1)} | {P(mdd(r))} | {P(cagr(r2),1)} | {P(mdd(r2))} | "
         f"{P(per(r,'2008-01','2008-12'))} | {P(per(r,'2022-01','2022-12'))} |"])
say([""])

# ---------------- 3. 防守部位（全球 18%＋1 倍大盤 5%）換標的
say(["## 3. 防守的 23%（全球＋1 倍大盤）用什麼？", "", "| 防守 23% " + HEAD, SEP])
core = [(w, T[a]) for w, a in FINAL if a not in ("vt", "sp")]
for lab, parts in (("全部 VT／VWRA", [(.23, T["vt"])]), ("18% VT＋5% S&P（定案）", [(.18, T["vt"]), (.05, T["sp"])]),
                   ("13% VT＋10% S&P", [(.13, T["vt"]), (.10, T["sp"])]), ("全部 S&P 500", [(.23, T["sp"])]),
                   ("全部台股 1 倍（0050）", [(.23, T["taiex"])]), ("18% S&P＋5% 中國 FXI", [(.18, T["sp"]), (.05, T["fxi"])]),
                   ("全部中國 FXI", [(.23, T["fxi"])])):
    line(lab, core + parts)
say([""])

# ---------------- 4. 進攻主力 28% 換標的
say(["## 4. 進攻主力的 28% 用什麼？", "", "其他部位不變，只換台股正2 那 28%。", "", "| 進攻 28% " + HEAD, SEP])
rest = [(w, T[a]) for w, a in FINAL if a != "tw2"]
for lab, key in (("台股正2（定案）", "tw2"), ("那斯達克 2 倍（00670L／QLD 型）", "ndx2"), ("S&P 2 倍（00647L／SSO 型）", "sp2"),
                 ("TQQQ 3 倍", "tqqq"), ("SOXL 3 倍", "semi3"), ("QQQ 1 倍", "ndx"), ("SOXX／SMH 1 倍", "semi"), ("台股 1 倍（0050）", "taiex")):
    line(lab, [(.28, T[key])] + rest)
say(["", "### 從 2000/3 網路泡沫高點單獨買進（當地貨幣，到今天）", "",
     "| 標的 | 年化 | 最大跌幅 | 回到原點 | 現在是幾倍 |", "|---|---|---|---|---|"])
L = {"台股正2": tw2, "台股 1 倍": taiex, "S&P 2 倍": lev(sp, 2, usd_rf, 0.009, 0.003), "那斯達克 2 倍": lev(ndx, 2, usd_rf, 0.009, 0.003),
     "TQQQ 3 倍": lev(ndx, 3, usd_rf, 0.009, 0.005), "SOXL 3 倍": lev(sox, 3, usd_rf, 0.009, 0.005),
     "QQQ 1 倍": ndx, "SOXX 1 倍": sox, "S&P 1 倍": sp}
for k, r in L.items():
    x = r["2000-03-10":]; e = (1 + x).cumprod(); back = e[(e.index > "2001-01-01") & (e >= 1)]
    say([f"| {k} | {P(cagr(x),1)} | {P(mdd(x))} | {back.index[0].year if len(back) else '還沒回到'} | {e.iloc[-1]:.1f} 倍 |"])
say([""])

# ---------------- 5. 台股正2 拆一部分到那斯達克 2 倍
say(["## 5. 台股正2 要不要拆一部分到那斯達克 2 倍（00670L）？", "",
     "| 台股正2／那斯達克 2 倍 " + HEAD + " 進攻部位單獨 2000/3 起最大跌幅 |", SEP + "---|"])
ndx2_local = lev(ndx, 2, usd_rf, 0.009, 0.003)
for a, b in ((28, 0), (24, 4), (20, 8), (18, 10), (14, 14), (0, 28)):
    solo = mix([(a / 28, tw2)] + ([(b / 28, ndx2_local)] if b else []))["2000-03-10":]
    line(f"{a}%／{b}%", [(a / 100, T["tw2"])] + ([(b / 100, T["ndx2"])] if b else []) + rest, extra=[P(mdd(solo))])
say(["", "→ 怎麼拆報酬都差不多；拆越多，2022 年和 2000 年科技泡沫跌越深。2026-10-12 使用者決定不拆。", ""])

# ---------------- 6. 各國股市
say(["## 6. 各國股市每 10 年的年化報酬（美元或當地貨幣、含估計股息）", "",
     "| 市場 | 1990 年代 | 2000 年代 | 2010 年代 | 2020～ | 最大跌幅 | 任一天買進抱 10 年：年化中位數 | 10 年後仍虧損的機率 |",
     "|---|---|---|---|---|---|---|---|"])
M = {"美國 S&P 500": sp, "全球 VT": vt, "台股加權": taiex, "中國上證": clean_ret(px["000001.SS"]) + 0.02 / 252,
     "香港恆生": clean_ret(px["^HSI"]) + 0.035 / 252, "中國大型股 FXI": fxi, "日本日經": clean_ret(px["^N225"]) + 0.015 / 252,
     "新興市場 VEIEX": clean_ret(px["VEIEX"]) + 0.0015 / 252}
for k, r in M.items():
    dec = []
    for a, b in (("1990", "1999"), ("2000", "2009"), ("2010", "2019"), ("2020", "2030")):
        x = r[a:b]
        if len(x) < 200: dec.append("—"); continue
        y0 = x.index[0].year   # 資料晚於年代開頭時標出起始年
        dec.append(P(cagr(x), 1) + (f"（{y0} 起）" if y0 > int(a) else ""))
    e = (1 + r).cumprod().asfreq("D").ffill(); ten = ((e.shift(-3652) / e) ** 0.1 - 1).dropna()
    say([f"| {k} | " + " | ".join(dec) + f" | {P(mdd(r))} | {P(ten.median(),1)} | {P((ten < 0).mean())} |"])
say(["", "以上都是歷史回測，不代表未來，也不是投資建議。"])

with open(os.path.join(HERE, "results", "portfolio_research.md"), "w") as fh:
    fh.write("\n".join(OUT) + "\n")
print("\n".join(OUT))
