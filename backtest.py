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
import picks
import sec
import stages

OUT = config.ROOT / "data" / "backtest" / "stage_stats.json"
SAMPLE_PKL = config.CACHE_DIR / "backtest_samples.pkl"
MOPS_DIR = config.CACHE_DIR / "mops"
REV_STORE = config.ROOT / "data" / "backtest" / "tw_revenue.csv.gz"
STEP = 5                 # 每 5 個交易日取樣一次（約每週）
H6, H12, H24, H48 = 126, 252, 504, 1008
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
    df = df[df["stock_id"].str.fullmatch(r"[1-9]\d{3}") & df["type"].isin(["twse", "tpex"])]
    df = df.sort_values("date").drop_duplicates("stock_id", keep="last")       # 同一檔有舊紀錄時取最新的上市別
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
    feat = stages.features(s, px["Volume"])
    df = pd.DataFrame({
        "stage": s["stage"], "gain_base": s["gain_base"],
        "r6": fwd[H6], "r12": fwd[H12], "x12": fwd[H12] - (b.shift(-H12) / b - 1),
        "max24": max24, "dd6": min6,
    })
    min24 = pd.Series(rev.rolling(H24, min_periods=H24).min().to_numpy()[::-1], index=c.index).shift(-1) / c
    df["half24"] = (min24 <= 0.5).where(min24.notna())          # 2 年內曾腰斬（跌掉一半）
    df["max48"] = pd.Series(rev.rolling(H48, min_periods=H48).max().to_numpy()[::-1], index=c.index).shift(-1) / c
    df["close"] = c
    vol = px["Volume"].reindex(c.index).astype(float)
    df["ret5"] = c / c.shift(5) - 1                                          # 近 5 日漲幅
    df["vol5"] = vol.rolling(5).mean() / vol.rolling(60).mean().replace(0, np.nan)   # 近 5 日量／季均量
    df = df.join(feat)
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


def add_us_fundamentals(df):
    """接上 SEC 歷史財報（只用當時已公布的）。抓不到 SEC 時用 repo 裡存的資料。"""
    long = sec.load()          # 由 sec.yml 每月更新（python sec.py）
    if long is None:
        try:
            long = sec.build()
        except Exception as e:
            print(f"\n  SEC 連線失敗（{e}）")
    if long is None:
        print("  沒有 SEC 財報資料，跳過基本面")
        return df
    cik = sec.load_cik()
    df = df.assign(cik=df["sym"].map(cik))
    has = df["cik"].notna()
    a, q = sec.panels(long)
    f = sec.features(df[has].copy(), a, q)
    out = pd.concat([f, df[~has]], ignore_index=True)
    print(f"\n  基本面：{f['fcf_yield'].notna().sum():,} 筆樣本有自由現金流資料（{f.loc[f['fcf_yield'].notna(), 'sym'].nunique()} 檔）")
    return out


def build(market, syms, reuse_px=False):
    bench_sym = config.BENCHMARK[market]
    print(f"{market}：{len(syms)} 檔")
    cache = config.CACHE_DIR / f"bt_prices_{'us' if market == '美股' else 'tw'}.pkl"
    if reuse_px and cache.exists():
        px = pd.read_pickle(cache)
    else:
        px = download(syms + [bench_sym])
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(px, cache)
    px = dict(px)
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
    if market == "美股":
        df = add_us_fundamentals(df)
    # 每筆交易的進場狀態（相對強度、月營收、基本面）取進場前最近一次取樣
    tr = pd.concat([t for t in tparts if len(t)], ignore_index=True)
    tr["date"] = pd.to_datetime(tr["date"]).astype("datetime64[ns]")
    cols = ["date", "sym", "rs_pct"] + (["yoy3", "accel"] if "yoy3" in df else []) +         [c for c in sec.FUND_FEATURES if c in df]
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
        "p5x48": f((g["max48"].dropna() >= 5).mean()) if "max48" in g and g["max48"].notna().any() else None,   # 4 年內曾漲到 5 倍
        "p10x48": f((g["max48"].dropna() >= 10).mean()) if "max48" in g and g["max48"].notna().any() else None, # 4 年內曾漲到 10 倍
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


def summarize_fund(df):
    """美股基本面：每個指標分 5 組，看各組後來的表現（含 4 年內漲 10 倍的比例）"""
    out = {}
    for fcol, label in sec.FUND_FEATURES.items():
        if fcol not in df or df[fcol].notna().sum() < 5000:
            continue
        sub = df[df[fcol].notna()]
        bins = pd.qcut(sub[fcol], 5, duplicates="drop")
        rows = []
        for b, g in sub.groupby(bins, observed=True):
            rows.append({"lo": float(b.left), "hi": float(b.right), **stat(g)})
        out[fcol] = {"label": label, "bins": rows}
    return out


SURGE = {"暴衝": (0.15, 2.5), "強烈暴衝": (0.30, 3.0)}   # 近 5 日漲幅 ≥、近 5 日量是季均量的幾倍 ≥


def is_surge(ret5, vol5, level="暴衝"):
    r, v = SURGE[level]
    return (ret5 >= r) & (vol5 >= v)


def summarize_surge(df):
    """突然暴衝（短期急漲＋爆量）之後的表現；也分階段、分基本面看"""
    if "ret5" not in df:
        return {}
    out = {"ALL": stat(df)}
    for lv in SURGE:
        m = is_surge(df["ret5"], df["vol5"], lv)
        if m.sum() >= 100:
            out[lv] = stat(df[m])
    m = is_surge(df["ret5"], df["vol5"])
    ev = df[m]
    names = np.array(stages.ORDER, dtype=object)[ev["stage"].to_numpy()]
    for st in stages.ORDER:
        g = ev[names == st]
        if len(g) >= 100:
            out["暴衝|" + st] = stat(g)
    if "fcf_yield" in ev:
        for lab, mm in (("自由現金流為正", ev["fcf_yield"] > 0), ("自由現金流為負", ev["fcf_yield"] <= 0)):
            if mm.sum() >= 100:
                out["暴衝|" + lab] = stat(ev[mm])
    if "accel" in ev:
        for lab, mm in (("月營收加速", (ev["yoy3"] > 0.2) & (ev["accel"] > 0)), ("月營收未加速", ~((ev["yoy3"] > 0.2) & (ev["accel"] > 0)))):
            if mm.sum() >= 100:
                out["暴衝|" + lab] = stat(ev[mm])
    return out


def summarize_crash(df):
    """急跌（近 5 日跌 ≥15% 且量 ≥ 季均量 2 倍）之後的表現：是繼續跌，還是反彈？"""
    if "ret5" not in df:
        return {}
    m = (df["ret5"] <= -0.15) & (df["vol5"] >= 2.0)
    ev = df[m]
    if len(ev) < 100:
        return {}
    out = {"ALL": stat(df), "急跌": stat(ev)}
    names = np.array(stages.ORDER, dtype=object)[ev["stage"].to_numpy()]
    for st in stages.ORDER:
        g = ev[names == st]
        if len(g) >= 100:
            out["急跌|" + st] = stat(g)
    if "fcf_yield" in ev:
        for lab, mm in (("自由現金流為正", ev["fcf_yield"] > 0), ("自由現金流為負", ev["fcf_yield"] <= 0)):
            if mm.sum() >= 100:
                out["急跌|" + lab] = stat(ev[mm])
    return out


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
            df, tr, bench = build(market, syms, reuse_px="--reuse-prices" in sys.argv)
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
                                     "trades": summarize_trades(fr["trades"]),
                                     "past6_q": [float(x) for x in last.quantile(np.linspace(0, 1, 21))],
                                     "model": picks.train(df, market)}
        if market == "美股" and "fcf_yield" in df:
            fund = summarize_fund(df)
            result["markets"][market]["fund"] = fund
            base = stat(df[df["fcf_yield"].notna()])
            print(f"  --- 美股基本面分組（有財報的樣本：12月勝率 {base['win12']:.0%}、2年3倍 {base['p3x']:.1%}、4年10倍 {base['p10x48'] or 0:.2%}）")
            for fcol, v in fund.items():
                print(f"  {v['label']}")
                for b in v["bins"]:
                    print(f"    {b['lo']:>8.3f}～{b['hi']:<8.3f} n={b['n']:>6}  12月勝率 {b['win12'] or 0:.0%}  贏大盤 {b['beat12'] or 0:.0%}  "
                          f"中位 {b['med12'] or 0:+.0%}  2年3倍 {b['p3x'] or 0:.1%}  4年5倍 {b['p5x48'] or 0:.1%}  "
                          f"4年10倍 {b['p10x48'] or 0:.2%}  腰斬 {b['half24'] or 0:.0%}")
        # 每月篩選的兩條路線（和 screener.py 同一套條件）的歷史表現
        import screener
        sdf = df
        if market == "美股" and "fcf_yield" in df:
            sdf = df[df["fcf_yield"].notna() & df["bm"].notna()].copy()
            sdf["fcf_r"] = sdf.groupby("date")["fcf_yield"].rank(pct=True)
            sdf["bm_r"] = sdf.groupby("date")["bm"].rank(pct=True)
        if market == "台股" or "fcf_yield" in df:
            masks = screener.route_masks(sdf, market)
            result["markets"][market]["screens"] = {"ALL": stat(sdf), **{k: stat(sdf[v]) for k, v in masks.items() if v.sum() >= 100}}
            print("  --- 每月篩選路線的歷史表現")
            for k, v in result["markets"][market]["screens"].items():
                print(f"  {k:<4} n={v['n']:>7} 檔{v['n_stocks']:>5}  12月勝率 {v['win12'] or 0:.0%}  中位 {v['med12'] or 0:+.0%}  "
                      f"2年3倍 {v['p3x'] or 0:.1%}  4年5倍 {v['p5x48'] or 0:.1%}  4年10倍 {v['p10x48'] or 0:.2%}  腰斬 {v['half24'] or 0:.0%}")
        cr = summarize_crash(df)
        result["markets"][market]["crash"] = cr
        print("  --- 急跌（近 5 日跌 ≥15% 且量 ≥ 季均量 2 倍）之後")
        for k, v in cr.items():
            print(f"  {k:<16} n={v['n']:>7}  6月中位 {v['med6'] or 0:+.0%}  12月勝率 {v['win12'] or 0:.0%}  "
                  f"12月中位 {v['med12'] or 0:+.0%}  6月內再跌 {v['dd6'] or 0:.0%}  腰斬 {v['half24'] or 0:.0%}")
        sg = summarize_surge(df)
        result["markets"][market]["surge"] = sg
        print("  --- 突然暴衝（近 5 日漲 ≥15% 且量 ≥ 季均量 2.5 倍）之後")
        for k, v in sg.items():
            print(f"  {k:<16} n={v['n']:>7}  6月中位 {v['med6'] or 0:+.0%}  12月勝率 {v['win12'] or 0:.0%}  贏大盤 {v['beat12'] or 0:.0%}  "
                  f"12月中位 {v['med12'] or 0:+.0%}  2年3倍 {v['p3x'] or 0:.1%}  4年10倍 {v['p10x48'] or 0:.2%}  腰斬 {v['half24'] or 0:.0%}")
        mv = result["markets"][market]["model"]
        print(f"  --- 精選模型驗證（2021 前建模、2022 後檢驗；w={mv['w']}）")
        for st, v in mv["val"].items():
            if "w12" in v:
                print(f"  {st:<8} 預估最高 20% 實際12月勝率 {v['w12']['top']:.0%}、最低 20% {v['w12']['bottom']:.0%}、"
                      f"全體 {v['w12']['all']:.0%}；腰斬 高預估組 {v.get('h24', {}).get('top', 0):.0%}")
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
