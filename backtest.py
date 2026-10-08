"""回測：過去 10 年，股票處在各種狀態時，之後的實際報酬（＝儀表板上的「歷史勝率」）。

  python backtest.py              美股抽樣 2500 檔＋台股全部上市櫃（約 30–60 分鐘）
  python backtest.py --quick      每個市場各抽 300 檔，快速試跑

測的東西（全部只用「當時已知」的資料，避免偷看未來）：
  1. 股價階段（stages.py，和儀表板同一套規則）
  2. 相對強度：過去 6 個月漲幅在全市場的排名（前 20%＝強勢股）
  3. 台股月營收：近 3 個月平均年增率、是否比前 3 個月加速（公開資訊觀測站，次月 10 日後才算已知）

為了避免「後見之明」，用的是全市場，不是觀察清單（觀察清單是事後挑過的，會高估）。
仍有的偏差：已下市的股票抓不到（存活者偏差），所以「絕對報酬」會偏高；
看的時候以「比全市場基準好多少」為主。結果存到 data/backtest/stage_stats.json。
"""
import json
import random
import sys
import time
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests

import config
import stages

OUT = config.ROOT / "data" / "backtest" / "stage_stats.json"
SAMPLE_PKL = config.CACHE_DIR / "backtest_samples.pkl"
MOPS_DIR = config.CACHE_DIR / "mops"
REV_STORE = config.ROOT / "data" / "backtest" / "tw_revenue.csv.gz"
STEP = 5                 # 每 5 個交易日取樣一次（約每週）
H6, H12, H24 = 126, 252, 504
LIQ = {"美股": (2e6, 2.0), "台股": (2e7, 10.0)}   # 20 日平均成交金額、最低股價，太冷門的不算
RS_TOP, RS_LOW = 0.8, 0.2


# ---------- 股票清單 ----------
def us_universe():
    rows = []
    for url, col in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
                     ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")):
        df = pd.read_csv(StringIO(requests.get(url, timeout=30).text), sep="|", dtype=str).dropna(subset=[col])
        df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
        bad = df["Security Name"].str.contains(
            "Warrant|Unit|Right|Preferred|Depositary|Notes|Debenture|%|Trust|Fund|Acquisition", case=False, na=True)
        rows += df.loc[~bad & df[col].str.fullmatch(r"[A-Z]{1,5}"), col].tolist()
    return sorted(set(rows))


def tw_universe():
    r = requests.get("https://api.finmindtrade.com/api/v4/data", params={"dataset": "TaiwanStockInfo"}, timeout=30)
    df = pd.DataFrame(r.json()["data"])
    df = df[df["stock_id"].str.fullmatch(r"[1-9]\d{3}") & df["type"].isin(["twse", "tpex"])].drop_duplicates("stock_id")
    return [s + (".TW" if t == "twse" else ".TWO") for s, t in zip(df["stock_id"], df["type"])]


# ---------- 股價 ----------
def download(symbols, batch=150):
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        raw = None
        for _ in range(3):
            try:
                raw = yf.download(chunk, period="10y", interval="1d", auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"\n  下載失敗，30 秒後重試（{e}）")
                time.sleep(30)
        if raw is None:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for s in chunk:
            try:
                d = (raw[s] if multi else raw)[["Close", "Volume"]].dropna()
            except KeyError:
                continue
            if len(d) > 400:
                out[s] = d
        print(f"\r  已下載 {min(i + batch, len(symbols))}/{len(symbols)}，可用 {len(out)}", end="", flush=True)
        time.sleep(2)
    print()
    return out


# ---------- 台股月營收（公開資訊觀測站彙總表，一個月一頁）----------
def mops_month(market, y, m):
    MOPS_DIR.mkdir(parents=True, exist_ok=True)
    p = MOPS_DIR / f"{market}_{y}_{m:02d}.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"code": str})
    url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{y - 1911}_{m}_0.html"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        tables = pd.read_html(StringIO(r.content.decode("cp950", errors="ignore")))
    except Exception:
        return pd.DataFrame(columns=["code", "rev", "rev_ly"])
    rows = []
    for t in tables:
        if t.shape[1] < 10:
            continue
        t = t.iloc[:, [0, 2, 4]]
        t.columns = ["code", "rev", "rev_ly"]
        rows.append(t)
    df = pd.concat(rows) if rows else pd.DataFrame(columns=["code", "rev", "rev_ly"])
    df = df[df["code"].astype(str).str.fullmatch(r"\d{4}")]
    df["rev"] = pd.to_numeric(df["rev"], errors="coerce")
    df["rev_ly"] = pd.to_numeric(df["rev_ly"], errors="coerce")
    df.to_csv(p, index=False)
    time.sleep(1)
    return df


def tw_revenue(start_year):
    """回傳 DataFrame(code, avail, yoy3, accel)：avail＝這筆資料可被知道的日期（次月 11 日）。
    抓過的月份存在 data/backtest/tw_revenue.csv.gz（放進 repo，GitHub 上就不用每次重抓、也不怕連不到觀測站）"""
    now = datetime.now()
    if REV_STORE.exists():
        store = pd.read_csv(REV_STORE, dtype={"code": str}, parse_dates=["ym"])
    else:
        store = pd.DataFrame(columns=["code", "rev", "rev_ly", "ym"])
    have = set(store["ym"].dt.strftime("%Y-%m")) if len(store) else set()
    recent = {(now - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, 4)}   # 最近 3 個月重抓（有人晚公布）
    months = [(y, m) for y in range(start_year, now.year + 1) for m in range(1, 13)
              if datetime(y, m, 1) < datetime(now.year, now.month, 1)]
    parts = [store]
    for i, (y, m) in enumerate(months, 1):
        ym = f"{y}-{m:02d}"
        if ym in have and ym not in recent:
            continue
        for mk in ("sii", "otc"):
            d = mops_month(mk, y, m)
            if len(d):
                parts.append(d.assign(ym=pd.Timestamp(y, m, 1)))
        print(f"\r  月營收 {i}/{len(months)} 個月", end="", flush=True)
    print()
    raw = pd.concat(parts).drop_duplicates(["code", "ym"], keep="last").sort_values(["code", "ym"])
    raw.to_csv(REV_STORE, index=False, compression="gzip")
    df = raw.copy()
    rev, rev_ly = (pd.to_numeric(df[c], errors="coerce").astype(float) for c in ("rev", "rev_ly"))
    df["yoy"] = (rev / rev_ly.where(rev_ly > 0)) - 1
    g = df.groupby("code")["yoy"]
    df["yoy3"] = g.transform(lambda s: s.rolling(3).mean())
    df["accel"] = df["yoy3"] - g.transform(lambda s: s.rolling(3).mean().shift(3))
    df["avail"] = df["ym"] + pd.DateOffset(months=1, days=10)
    return df[["code", "avail", "yoy3", "accel"]].dropna(subset=["yoy3"])


# ---------- 取樣 ----------
def samples(sym, px, bench_close, market, dates):
    s = stages.compute(px)
    c = s["close"]
    fwd = {h: c.shift(-h) / c - 1 for h in (H6, H12)}
    rev = pd.Series(c.to_numpy()[::-1])
    max24 = pd.Series(rev.rolling(H24, min_periods=H24).max().to_numpy()[::-1], index=c.index).shift(-1) / c
    min6 = pd.Series(rev.rolling(H6, min_periods=H6).min().to_numpy()[::-1], index=c.index).shift(-1) / c - 1
    b = bench_close.reindex(c.index).ffill()
    dollar = (px["Close"] * px["Volume"]).rolling(20).mean()
    min_dollar, min_px = LIQ[market]
    df = pd.DataFrame({
        "stage": s["stage"], "gain_base": s["gain_base"], "past6": c / c.shift(H6) - 1,
        "r6": fwd[H6], "r12": fwd[H12], "x12": fwd[H12] - (b.shift(-H12) / b - 1),
        "max24": max24, "dd6": min6,
    })
    min24 = pd.Series(rev.rolling(H24, min_periods=H24).min().to_numpy()[::-1], index=c.index).shift(-1) / c
    df["half24"] = (min24 <= 0.5).where(min24.notna())          # 2 年內曾腰斬（跌掉一半）
    ok = (s["stage"] >= 0) & (dollar >= min_dollar) & (c >= min_px)
    out = df[ok & df.index.isin(dates)]
    out.index.name = "date"
    return out.reset_index().assign(sym=sym), trades(sym, s, b, ok)


def trades(sym, s, bench, ok):
    """照規則操作：進入多頭（發動）那天收盤買進，多頭結束（跌破出場線、轉弱）那天收盤賣出。"""
    st = s["stage"].to_numpy()
    alive = np.isin(st, [stages.CODE[stages.IGNITE], stages.CODE[stages.RUN], stages.CODE[stages.HOT]])
    c, b, idx, okv = s["close"].to_numpy(), bench.to_numpy(), s.index, ok.to_numpy()
    rows, i, n = [], 1, len(st)
    while i < n:
        if alive[i] and not alive[i - 1] and okv[i]:
            j = i + 1
            while j < n and alive[j]:
                j += 1
            closed = j < n
            j = min(j, n - 1)
            rows.append({"date": idx[i], "sym": sym, "ret": c[j] / c[i] - 1,
                         "bret": (b[j] / b[i] - 1) if b[i] > 0 else np.nan,
                         "days": j - i, "peak": c[i:j + 1].max() / c[i] - 1, "closed": closed})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def build(market, syms):
    bench_sym = config.BENCHMARK[market]
    print(f"{market}：{len(syms)} 檔")
    px = download(syms + [bench_sym])
    bench = px.pop(bench_sym)["Close"]
    dates = set(bench.index[::STEP])
    parts, tparts = [], []
    for i, (s, d) in enumerate(px.items(), 1):
        try:
            a, t = samples(s, d, bench, market, dates)
            parts.append(a)
            tparts.append(t)
        except Exception as e:
            print(f"\n  {s} 失敗：{e}")
        if i % 100 == 0:
            print(f"\r  計算中 {i}/{len(px)}", end="", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    # 相對強度：同一天、同市場裡，過去 6 個月漲幅的百分位
    df["rs_pct"] = df.groupby("date")["past6"].rank(pct=True)
    if market == "台股":
        rv = tw_revenue(bench.index[0].year - 1)
        df["code"] = df["sym"].str.split(".").str[0].astype(str)
        rv["code"] = rv["code"].astype(str)
        rv["avail"] = pd.to_datetime(rv["avail"]).astype("datetime64[ns]")
        df = df.sort_values("date")
        df = pd.merge_asof(df, rv.sort_values("avail"), left_on="date", right_on="avail", by="code",
                           direction="backward", tolerance=pd.Timedelta(days=70))
    # 每筆交易的進場狀態（相對強度、月營收）取進場前最近一次取樣
    tr = pd.concat([t for t in tparts if len(t)], ignore_index=True)
    tr["date"] = pd.to_datetime(tr["date"]).astype("datetime64[ns]")
    cols = ["date", "sym", "rs_pct"] + (["yoy3", "accel"] if "yoy3" in df else [])
    tr = pd.merge_asof(tr.sort_values("date"), df[cols].sort_values("date"), on="date", by="sym",
                       direction="backward", tolerance=pd.Timedelta(days=10))
    df["market"] = market
    print(f"\n  樣本 {len(df):,} 筆、{df['sym'].nunique()} 檔；規則交易 {len(tr):,} 筆")
    return df, tr, bench


# ---------- 統計 ----------
def stat(g):
    r12, m24, x12 = g["r12"].dropna(), g["max24"].dropna(), g["x12"].dropna()
    f = lambda v: float(v) if pd.notna(v) else None
    return {
        "n": int(len(g)), "n_stocks": int(g["sym"].nunique()),
        "win12": f((r12 > 0).mean()) if len(r12) else None,     # 12 個月後是上漲的比例
        "beat12": f((x12 > 0).mean()) if len(x12) else None,    # 12 個月後贏過大盤的比例
        "med6": f(g["r6"].median()), "med12": f(r12.median()),
        "avg12": f(r12.clip(upper=5).mean()),
        "p2x": f((m24 >= 2).mean()) if len(m24) else None,      # 2 年內曾漲到 2 倍
        "p3x": f((m24 >= 3).mean()) if len(m24) else None,      # 2 年內曾漲到 3 倍
        "p5x": f((m24 >= 5).mean()) if len(m24) else None,
        "dd6": f(g["dd6"].median()),                            # 之後 6 個月內最大跌幅（中位數）
        "half24": f(g["half24"].dropna().mean()) if g["half24"].notna().any() else None,  # 2 年內曾腰斬
    }


def trade_stat(g):
    """規則交易的統計：勝率、平均賺賠、期望值"""
    if not len(g):
        return None
    r = g["ret"]
    win, loss = r[r > 0], r[r <= 0]
    f = lambda v: float(v) if pd.notna(v) else None
    return {
        "n": int(len(g)), "n_stocks": int(g["sym"].nunique()),
        "win": f((r > 0).mean()),                         # 交易勝率
        "avg_win": f(win.mean()) if len(win) else None,   # 賺的時候平均賺
        "avg_loss": f(loss.mean()) if len(loss) else None,
        "exp": f(r.clip(upper=10).mean()),                # 每筆期望值（單筆最多算 +1000%，避免極端值）
        "med": f(r.median()),
        "beat": f((r - g["bret"] > 0).mean()),            # 這段持有期間贏過大盤的比例
        "bexp": f(g["bret"].mean()),                      # 同一段持有期間，買大盤平均賺多少
        "xexp": f((r.clip(upper=10) - g["bret"]).mean()), # 平均每筆比大盤多賺（超額報酬）
        "p100": f((r >= 1).mean()), "p200": f((r >= 2).mean()),   # 單筆賺 1 倍、2 倍以上
        "weeks": f(g["days"].median() / 5),
    }


def stage_key(df):
    names = np.array(stages.ORDER, dtype=object)[df["stage"].to_numpy()]
    phase = pd.cut(df["gain_base"], [-np.inf] + [h for h, _ in stages.RUN_PHASES],
                   labels=[p for _, p in stages.RUN_PHASES]).astype(str)
    return pd.Series([stages.key(a, p if a == stages.RUN else "") for a, p in zip(names, phase)], index=df.index)


def rs_tier(df):
    return np.select([df["rs_pct"] >= RS_TOP, df["rs_pct"] <= RS_LOW], ["強", "弱"], "中")


def rev_tier(df):
    """月營收：成長加速（近3月年增 >20% 且比前3月加速）／衰退（近3月年增 <0）／其他"""
    if "yoy3" not in df:
        return np.full(len(df), "")
    return np.select([(df["yoy3"] > 0.2) & (df["accel"] > 0), df["yoy3"] < 0, df["yoy3"].notna()],
                     ["加速", "衰退", "一般"], "")


def summarize(df):
    df = df.copy()
    df["sk"] = stage_key(df)
    df["st"] = np.array(stages.ORDER, dtype=object)[df["stage"].to_numpy()]
    df["rs"] = rs_tier(df)
    df["rv"] = rev_tier(df)
    res = {"ALL": stat(df)}
    groups = [("sk",), ("st",), ("rs",), ("rv",), ("st", "rs"), ("sk", "rs"), ("st", "rv"), ("rs", "rv"), ("st", "rs", "rv")]
    for cols in groups:
        for k, g in df.groupby(list(cols)):
            k = k if isinstance(k, tuple) else (k,)
            if any(v == "" for v in k) or len(g) < 150:   # 樣本太少不算
                continue
            label = "|".join(f"{c}={v}" for c, v in zip(cols, k))
            res[label] = stat(g)
    base = res["ALL"]
    for v in res.values():
        for m in ("p2x", "p3x", "win12", "beat12"):
            v[m + "_lift"] = (v[m] / base[m]) if v.get(m) and base.get(m) else None
    return res


def summarize_trades(tr):
    tr = tr.copy()
    tr["rs"] = rs_tier(tr)
    tr["rv"] = rev_tier(tr)
    res = {"ALL": trade_stat(tr)}
    for cols in (("rs",), ("rv",), ("rs", "rv")):
        for k, g in tr.groupby(list(cols)):
            k = k if isinstance(k, tuple) else (k,)
            if any(v == "" for v in k) or len(g) < 80:
                continue
            res["|".join(f"{c}={v}" for c, v in zip(cols, k))] = trade_stat(g)
    return res


def main():
    quick = "--quick" in sys.argv
    reuse = "--reuse" in sys.argv        # 用上次下載的樣本重算統計（調整分組時用）
    random.seed(42)
    if reuse and SAMPLE_PKL.exists():
        frames = pd.read_pickle(SAMPLE_PKL)
    else:
        us, tw = us_universe(), tw_universe()
        if quick:
            us, tw = random.sample(us, 300), random.sample(tw, 300)
        elif len(us) > 2500:
            us = random.sample(us, 2500)
        frames = {}
        for market, syms in (("美股", us), ("台股", tw)):
            df, tr, bench = build(market, syms)
            frames[market] = {"df": df, "trades": tr, "period": [str(bench.index[0].date()), str(bench.index[-1].date())]}
        SAMPLE_PKL.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(frames, SAMPLE_PKL)

    result = {"generated": datetime.now().strftime("%Y-%m-%d"), "step_days": STEP,
              "rs_top": RS_TOP, "rs_low": RS_LOW, "markets": {}}
    for market, fr in frames.items():
        df = fr["df"]
        stats = summarize(df)
        # 最近一次取樣日，全市場 6 個月漲幅的前 20%／後 20% 門檻（儀表板判斷「強／弱」用）
        last = df[df["date"] == df["date"].max()]["past6"].dropna()
        result["markets"][market] = {"n_stocks": int(df["sym"].nunique()), "n": int(len(df)),
                                     "period": fr["period"], "stats": stats,
                                     "rs_cut": [float(last.quantile(RS_LOW)), float(last.quantile(RS_TOP))],
                                     "trades": summarize_trades(fr["trades"])}
        print(f"\n===== {market}（{df['sym'].nunique()} 檔，{len(df):,} 筆）")
        for k, v in stats.items():
            print(f"  {k:<28} n={v['n']:>7}  12月勝率 {v['win12'] or 0:.0%}  贏大盤 {v['beat12'] or 0:.0%}  "
                  f"12月中位 {v['med12'] or 0:+.0%}  2年3倍 {v['p3x'] or 0:.1%}（{v['p3x_lift'] or 0:.1f}x）  "
                  f"6月最大跌 {v['dd6']:.0%}  2年腰斬 {v['half24'] or 0:.0%}")
        print("  --- 規則交易（發動買進、跌破出場線賣出）")
        for k, v in result["markets"][market]["trades"].items():
            print(f"  {k:<14} n={v['n']:>6}  勝率 {v['win']:.0%}  平均賺 {v['avg_win'] or 0:+.0%}  平均賠 {v['avg_loss'] or 0:+.0%}  "
                  f"期望值 {v['exp']:+.1%}  同期大盤 {v['bexp']:+.1%}  超額 {v['xexp']:+.1%}  贏大盤 {v['beat']:.0%}  賺1倍以上 {v['p100']:.1%}  持有 {v['weeks']:.0f} 週")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {OUT}")


if __name__ == "__main__":
    main()
