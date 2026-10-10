"""LRS（槓桿輪動）回測：TQQQ / SOXL

規則：標的指數收盤 > 200 日均線 → 持有 3 倍 ETF；跌破 → 換成短期國庫券（現金）。
比較：大盤（1 倍）長抱、3 倍 ETF 長抱、LRS。

資料
- 指數：^NDX（1985~）、^SOX（1994~），用 QQQ / SOXX 還原股息後的報酬補上股息
- 3 倍 ETF 上市前：用「3 × 指數日報酬 − 2 × 借錢成本 − 管理費」模擬，並用真實 TQQQ / SOXL 校驗
- 現金報酬：^IRX（13 週美國國庫券殖利率）

輸出：lrs/results/results.md、results.json、圖表 png
"""
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

FEE_3X = 0.0090        # 3 倍 ETF 管理費＋雜支（年）
SPREAD = 0.0050        # 借錢成本高於國庫券的部分（年，乘上 2 倍槓桿部位）
COST = 0.0010          # 每次換倉成本（手續費＋滑價）
SMA_N = 200
BAND = 0.02            # 緩衝版：漲破均線 2% 才進、跌破 2% 才出


def dl(tickers):
    d = yf.download(tickers, start="1980-01-01", auto_adjust=True, progress=False)["Close"]
    return d


px = dl(["^NDX", "^SOX", "QQQ", "SOXX", "TQQQ", "SOXL", "^IRX"])
px.index = pd.to_datetime(px.index).tz_localize(None)
px.to_csv(os.path.join(OUT, "prices.csv.gz"))
print(px.apply(lambda s: s.first_valid_index()))

rf = (px["^IRX"].ffill() / 100 / 252).fillna(0)


def total_return(idx, etf, etf_fee, div_guess):
    """指數價格報酬；ETF 上市後改用 ETF 還原報酬（加回 ETF 管理費）"""
    r_idx = px[idx].pct_change() + div_guess / 252
    r_etf = px[etf].pct_change() + etf_fee / 252
    r = r_idx.copy()
    ok = r_etf.notna()
    r[ok] = r_etf[ok]
    return r.dropna()


UNDER = {
    "NDX": dict(idx="^NDX", r=total_return("^NDX", "QQQ", 0.0020, 0.006), etf="TQQQ"),
    "SOX": dict(idx="^SOX", r=total_return("^SOX", "SOXX", 0.0035, 0.008), etf="SOXL"),
}


def lev_returns(r, lev=3):
    f = rf.reindex(r.index).fillna(0)
    return lev * r - (lev - 1) * (f + SPREAD / 252) - FEE_3X / 252


def lrs_position(price, band=0.0):
    sma = price.rolling(SMA_N).mean()
    pos = pd.Series(np.nan, index=price.index)
    if band == 0:
        pos[:] = (price > sma).astype(float)
    else:
        state = 0.0
        vals = []
        for p, s in zip(price.values, sma.values):
            if np.isnan(s):
                vals.append(np.nan); continue
            if state == 0 and p > s * (1 + band):
                state = 1.0
            elif state == 1 and p < s * (1 - band):
                state = 0.0
            vals.append(state)
        pos[:] = vals
    pos[sma.isna()] = np.nan
    return pos


def run_strategy(r_risky, pos, lag=1):
    """pos 在第 t 天收盤決定，lag=1 表示用第 t 天收盤成交（收盤前下單），lag=2 表示隔天收盤才成交"""
    p = pos.shift(lag).reindex(r_risky.index)
    f = rf.reindex(r_risky.index).fillna(0)
    sw = p.diff().abs().fillna(0)
    ret = p * r_risky + (1 - p) * f - sw * COST
    return ret.dropna(), p


def stats(ret, pos=None):
    ret = ret.dropna()
    if len(ret) < 50:
        return None
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    f = rf.reindex(ret.index).fillna(0)
    ex = ret - f
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan
    roll = eq.pct_change(252)
    out = dict(start=str(ret.index[0].date()), end=str(ret.index[-1].date()), years=round(yrs, 1),
               cagr=round(cagr * 100, 1), maxdd=round(dd.min() * 100, 1),
               vol=round(ret.std() * np.sqrt(252) * 100, 1), sharpe=round(sharpe, 2),
               multiple=round(eq.iloc[-1], 1), worst12m=round(roll.min() * 100, 1),
               calmar=round(cagr / abs(dd.min()), 2) if dd.min() < 0 else np.nan)
    if pos is not None:
        pp = pos.reindex(ret.index)
        out["time_in"] = round(pp.mean() * 100, 0)
        out["switches_per_year"] = round(pp.diff().abs().sum() / yrs, 1)
    return out


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v}%"


results = {}
curves = {}
md = ["# LRS 回測結果（自動產生）", "",
      f"- 規則：指數收盤 > {SMA_N} 日均線 → 持有 3 倍；跌破 → 國庫券。換倉成本 {COST*100:.1f}%／次",
      f"- 3 倍模擬：3×指數報酬 − 2×(國庫券＋{SPREAD*100:.1f}%) − 管理費 {FEE_3X*100:.1f}%／年",
      "- CAGR＝年化報酬；MaxDD＝最大跌幅；Sharpe＝風險調整後報酬（越高越好）", ""]

for name, u in UNDER.items():
    r1 = u["r"]
    r3 = lev_returns(r1)
    price = px[u["idx"]].dropna()
    pos = lrs_position(price)
    posb = lrs_position(price, BAND)
    real = px[u["etf"]].pct_change().dropna()

    # 校驗：模擬 3 倍 vs 真實 ETF
    common = real.index.intersection(r3.index)
    sim_eq = (1 + r3[common]).cumprod().iloc[-1]
    real_eq = (1 + real[common]).cumprod().iloc[-1]
    yrs = len(common) / 252
    calib = dict(period=f"{common[0].date()}~{common[-1].date()}",
                 sim_cagr=round((sim_eq ** (1 / yrs) - 1) * 100, 1),
                 real_cagr=round((real_eq ** (1 / yrs) - 1) * 100, 1),
                 daily_corr=round(np.corrcoef(r3[common], real[common])[0, 1], 4))
    md += [f"## {name}（{u['etf']}）", "", f"**模擬校驗**（{calib['period']}）：模擬 3 倍年化 {calib['sim_cagr']}%、"
           f"真實 {u['etf']} 年化 {calib['real_cagr']}%、日報酬相關 {calib['daily_corr']}", ""]

    strat = {}
    strat["1x長抱"] = (r1, None)
    strat["3x長抱"] = (r3, None)
    lrs3, p3 = run_strategy(r3, pos); strat["LRS 3x"] = (lrs3, p3)
    lrs3b, p3b = run_strategy(r3, posb); strat[f"LRS 3x 緩衝{int(BAND*100)}%"] = (lrs3b, p3b)
    lrs3l2, p3l2 = run_strategy(r3, pos, lag=2); strat["LRS 3x 隔日成交"] = (lrs3l2, p3l2)
    lrs1, p1 = run_strategy(r1, pos); strat["LRS 1x"] = (lrs1, p1)
    # 真實 ETF（上市後）
    strat[f"{u['etf']}真實長抱"] = (real, None)
    lrsr, pr = run_strategy(real, pos); strat[f"LRS 真實{u['etf']}"] = (lrsr, pr)

    first = pos.first_valid_index() + pd.Timedelta(days=5)
    periods = {
        "全期間": (first, None),
        "2000起（含網路泡沫）": (pd.Timestamp("2000-01-01"), None),
        "2010起（ETF 上市後）": (pd.Timestamp("2010-03-01"), None),
        "2015起": (pd.Timestamp("2015-01-01"), None),
        "2020起": (pd.Timestamp("2020-01-01"), None),
        "2000–2009（熊市十年）": (pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")),
        "2010–2019（大多頭）": (pd.Timestamp("2010-03-01"), pd.Timestamp("2019-12-31")),
        "2022（升息熊市）": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    }
    results[name] = dict(calibration=calib, periods={})
    for pname, (a, b) in periods.items():
        a = max(a, first)
        rows = []
        md += [f"### {pname}", "", "| 策略 | 期間 | 年化 | 最大跌幅 | 最差12個月 | 波動 | Sharpe | 資產倍數 | 在場% | 換倉/年 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for sname, (ret, p) in strat.items():
            rr = ret[(ret.index >= a) & ((ret.index <= b) if b is not None else True)]
            if "真實" in sname and a < pd.Timestamp("2010-03-01"):
                continue
            s = stats(rr, p)
            if s is None:
                continue
            rows.append(dict(strategy=sname, **s))
            md.append(f"| {sname} | {s['start']}~{s['end']} | {s['cagr']}% | {s['maxdd']}% | {fmt(s['worst12m'])} | {s['vol']}% | "
                      f"{s['sharpe']} | {s['multiple']}x | {s.get('time_in', 100):.0f} | {s.get('switches_per_year', 0)} |")
        md.append("")
        results[name]["periods"][pname] = rows

    # 滾動 5 年勝率
    win = {}
    eqs = {k: (1 + v[0][v[0].index >= first]).cumprod() for k, v in strat.items() if "真實" not in k}
    w = 252 * 5
    base = eqs["LRS 3x"]
    for k in ("1x長抱", "3x長抱"):
        other = eqs[k].reindex(base.index)
        a_ = base.pct_change(w); b_ = other.pct_change(w)
        m = a_.notna() & b_.notna()
        win[k] = round((a_[m] > b_[m]).mean() * 100, 1)
    results[name]["rolling5y_lrs_beats"] = win
    md += [f"**任意 5 年區間，LRS 3x 贏過**：1x 長抱 {win['1x長抱']}% 的時間、3x 長抱 {win['3x長抱']}% 的時間", ""]

    # 逐年報酬
    yr = pd.DataFrame({k: (1 + strat[k][0][strat[k][0].index >= first]).groupby(lambda d: d.year).prod() - 1
                       for k in ("1x長抱", "3x長抱", "LRS 3x")}) * 100
    md += ["### 逐年報酬 %", "", "| 年 | 1x長抱 | 3x長抱 | LRS 3x |", "|---|---|---|---|"]
    for y_, row in yr.round(0).iterrows():
        md.append(f"| {y_} | {row['1x長抱']:.0f} | {row['3x長抱']:.0f} | {row['LRS 3x']:.0f} |")
    md.append("")
    results[name]["yearly"] = yr.round(1).to_dict()
    curves[name] = {k: eqs[k] for k in ("1x長抱", "3x長抱", "LRS 3x", f"LRS 3x 緩衝{int(BAND*100)}%")}

with open(os.path.join(OUT, "results.md"), "w") as f:
    f.write("\n".join(md))
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=1, default=str)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, cs in curves.items():
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
        en = {"1x長抱": "Index buy&hold (1x)", "3x長抱": "3x buy&hold", "LRS 3x": "LRS 3x",
              f"LRS 3x 緩衝{int(BAND*100)}%": f"LRS 3x, {int(BAND*100)}% band"}
        for k, eq in cs.items():
            ax[0].plot(eq.index, eq.values, label=en.get(k, k), lw=1.2)
            ax[1].plot(eq.index, (eq / eq.cummax() - 1).values * 100, lw=0.9)
        ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title(f"{name}: equity (log)")
        ax[1].set_ylabel("drawdown %")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
except Exception as e:  # 圖表失敗不影響數字
    print("chart error", e)

print("\n".join(md))
