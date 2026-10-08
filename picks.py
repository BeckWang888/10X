"""精選模型：同一個階段裡，哪些股票歷史上勝率比較高、腰斬風險比較低？

做法（簡單、可解釋）：
  每個市場 × 每個階段，把每個特徵（相對強度、乖離、離出場線距離、起漲天數、量能、月營收…）
  依歷史分成 5 組，算每組「12 個月後上漲」「6 個月後上漲」「2 年內腰斬」的比例。
  預估一檔股票時，從階段平均出發，加上它每個特徵所在組別比平均好（或差）多少，再乘上縮小係數。
驗證：只用 2021 年以前的資料建模，拿 2022 年以後的資料檢查——預估最高的 20% 實際勝率有沒有比最低的 20% 高。
"""
import numpy as np
import pandas as pd

import sec
import stages

BASE_FEATS = list(stages.FEATURES) + ["rs_pct"]
TW_FEATS = ["yoy3", "accel"]
TARGETS = {"w12": ("r12", lambda r: r > 0), "w6": ("r6", lambda r: r > 0), "h24": ("half24", lambda r: r > 0.5),
           "x3": ("max24", lambda r: r >= 3)}                     # x3＝2 年內曾漲到 3 倍（爆發力）
SPLIT = pd.Timestamp("2022-01-01")
PRIOR = 300          # 每組樣本少時往平均拉（同一檔股票每週取樣彼此相關，有效樣本比看起來少）
MIN_N = 500


def _targets(df):
    out = {}
    for k, (col, fn) in TARGETS.items():
        v = df[col]
        out[k] = fn(v.astype(float)).astype(float).where(v.notna())
    return pd.DataFrame(out, index=df.index)


def _fit_stage(sub, feats):
    y = _targets(sub)
    base = {k: float(y[k].mean()) for k in TARGETS}
    model = {"base": base, "n": int(len(sub)), "feats": {}}
    for f in feats:
        if f not in sub or sub[f].notna().sum() < MIN_N:
            continue
        x = sub[f]
        edges = np.unique(np.nanquantile(x, [0.2, 0.4, 0.6, 0.8]))
        b = np.searchsorted(edges, x, side="right")
        rec = {"edges": [float(e) for e in edges], "rate": {}, "n": []}
        for k in TARGETS:
            rates = []
            for i in range(len(edges) + 1):
                m = (b == i) & x.notna() & y[k].notna()
                n, hit = int(m.sum()), float(y[k][m].sum())
                rates.append((hit + base[k] * PRIOR) / (n + PRIOR))
            rec["rate"][k] = rates
        rec["n"] = [int(((b == i) & x.notna()).sum()) for i in range(len(edges) + 1)]
        model["feats"][f] = rec
    return model


def predict(model, x, w):
    """x：{特徵: 數值}。回傳 ({目標: 預估機率}, [(特徵, 對 w12 的影響)])"""
    est, contrib = {}, []
    for k, base in model["base"].items():
        tot = 0.0
        for f, rec in model["feats"].items():
            v = x.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            i = int(np.searchsorted(rec["edges"], v, side="right"))
            d = rec["rate"][k][i] - base
            tot += d
            if k == "w12":
                contrib.append((f, w * d))
        est[k] = float(min(0.99, max(0.01, base + w * tot)))
    return est, contrib


def _predict_frame(model, sub, w):
    res = {k: np.full(len(sub), model["base"][k]) for k in TARGETS}
    for f, rec in model["feats"].items():
        x = sub[f].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        i = np.searchsorted(rec["edges"], np.nan_to_num(x), side="right")
        for k in TARGETS:
            d = np.array(rec["rate"][k])[i] - model["base"][k]
            res[k] = res[k] + w * np.where(ok, d, 0.0)
    return {k: np.clip(v, 0.01, 0.99) for k, v in res.items()}


def _logloss(p, y):
    m = ~np.isnan(y)
    p, y = p[m], y[m]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else np.nan


STRONG_W = {"x3": 0.4, "w12": 0.3, "h24": 0.3}   # 最強分：爆發力 40%＋勝率 30%＋低腰斬 30%（2022 後驗證選出）


def strong_score(g):
    """g：同一市場、同一天的股票（要有 x3、w12、h24 預估）→ 0–100 的最強分"""
    return (STRONG_W["x3"] * g["x3"].rank(pct=True) + STRONG_W["w12"] * g["w12"].rank(pct=True)
            + STRONG_W["h24"] * g["h24"].rank(pct=True, ascending=False)) * 100


def simulate_top3(test_df, fitted, w, market):
    """用 2021 前建的模型，在 2022 後每週照規則挑最強 3 檔，和候選池平均比"""
    import backtest
    OK = {stages.CODE[stages.IGNITE], stages.CODE[stages.RUN]}
    rows = []
    for st in (stages.IGNITE, stages.RUN):
        if st not in fitted:
            continue
        sub = test_df[(test_df["stage"] == stages.CODE[st]) & (test_df["gain_base"] < 2.5)]
        if market == "美股" and "fcf_yield" in sub:
            sub = sub[sub["fcf_yield"] >= -0.01]
        p = _predict_frame(fitted[st], sub, w)
        rows.append(sub.assign(**{k: p[k] for k in ("x3", "w12", "h24")}))
    if not rows:
        return {}
    pool = pd.concat(rows)
    pool = pool[pool["max24"].notna()]
    pool["s"] = pool.groupby("date", group_keys=False).apply(strong_score)
    top = pool.sort_values("s", ascending=False).groupby("date").head(3)
    return {"pool": backtest.stat(pool), "top": backtest.stat(top)}


def train(df, market):
    feats = BASE_FEATS + (TW_FEATS if market == "台股" else [f for f in sec.FUND_FEATURES if f in df])
    names = np.array(stages.ORDER, dtype=object)[df["stage"].to_numpy()]
    out = {"w": None, "stages": {}, "val": {}}
    # 1) 用 2021 年以前建模、2022 年以後驗證，順便選縮小係數 w
    train_df, test_df = df[df["date"] < SPLIT], df[df["date"] >= SPLIT]
    tn = np.array(stages.ORDER, dtype=object)[train_df["stage"].to_numpy()]
    vn = np.array(stages.ORDER, dtype=object)[test_df["stage"].to_numpy()]
    fitted = {st: _fit_stage(train_df[tn == st], feats) for st in stages.ORDER if (tn == st).sum() >= MIN_N}
    best_w, best_ll = 0.5, np.inf
    for w in (0.2, 0.35, 0.5, 0.75, 1.0):
        ll = []
        for st, mdl in fitted.items():
            sub = test_df[vn == st]
            if len(sub) < MIN_N:
                continue
            p = _predict_frame(mdl, sub, w)
            y = _targets(sub)
            ll.append(_logloss(p["w12"], y["w12"].to_numpy()) * len(sub))
        if ll and sum(ll) < best_ll:
            best_w, best_ll = w, sum(ll)
    out["w"] = best_w
    for st, mdl in fitted.items():
        sub = test_df[vn == st]
        if len(sub) < MIN_N:
            continue
        p = _predict_frame(mdl, sub, best_w)
        y = _targets(sub)
        v = {}
        for k in ("w12", "w6", "h24", "x3"):
            yy, pp = y[k].to_numpy(), p[k]
            m = ~np.isnan(yy)
            if m.sum() < 200:
                continue
            q20, q80 = np.quantile(pp[m], [0.2, 0.8])
            v[k] = {"top": float(yy[m][pp[m] >= q80].mean()), "bottom": float(yy[m][pp[m] <= q20].mean()),
                    "all": float(yy[m].mean()), "n": int(m.sum())}
        out["val"][st] = v
    # 1b) 最強名單規則的事後驗證：2022 年後每週挑最強 3 檔（發動期＋主升段前中段、沒有疑慮）
    out["top3"] = simulate_top3(test_df, fitted, best_w, market)
    # 2) 正式模型：用全部資料
    for st in stages.ORDER:
        if (names == st).sum() >= MIN_N:
            out["stages"][st] = _fit_stage(df[names == st], feats)
    return out


# ---------- 即時：替觀察清單算預估值與精選 ----------
LABELS = {**stages.FEATURES, **sec.FUND_FEATURES, "rs_pct": "相對強度（全市場）", "yoy3": "月營收年增（近3月）", "accel": "月營收加速"}


def _fmt(f, v):
    if f == "rs_pct":
        return f"前 {max(1, round((1 - v) * 100))}%"
    if f == "days_in":
        return f"第 {int(v // 5) + 1} 週"
    if f == "rsi":
        return f"{v:.0f}"
    if f == "vol_ratio":
        return f"{v:.1f} 倍"
    if f == "bm":
        return f"{v:.2f}"
    if f == "log_mcap":
        return f"{10 ** v / 1e8:,.0f} 億美元"
    return f"{v:+.0%}"


def us_fundamentals(cand):
    """美股候選股的即時基本面特徵（SEC 財報＋今天股價），和回測用同一套算法"""
    long = sec.load()
    us = cand[cand["市場"] == "美股"]
    if long is None or us.empty:
        return {}
    cik = sec.load_cik()
    df = pd.DataFrame({"code": us["代號"], "cik": us["代號"].map(cik), "close": us["股價"],
                       "date": pd.Timestamp.now().normalize()}).dropna(subset=["cik"])
    if df.empty:
        return {}
    a, q = sec.panels(long)
    f = sec.features(df, a, q)
    return {r["code"]: {k: (None if pd.isna(r[k]) else float(r[k])) for k in sec.FUND_FEATURES}
            for _, r in f.iterrows()}


def live_features(t, mrev, past6_q, fund=None):
    x = {k: (None if pd.isna(v) else float(v)) for k, v in t["feat"].items()}
    x.update({k: v for k, v in (fund or {}).items() if v is not None})
    p6 = x.get("past6")
    if p6 is not None and past6_q:
        x["rs_pct"] = float(np.interp(p6, past6_q, np.linspace(0, 1, len(past6_q))))
    y3, prev = mrev.get("mrev_yoy3"), mrev.get("mrev_yoy3_prev")
    if y3 is not None:
        x["yoy3"] = y3
        if prev is not None:
            x["accel"] = y3 - prev
    return x


MIN_EDGE = 0.05   # 驗證時，預估最高 20% 要比最低 20% 的實際勝率高至少 5 個百分點才算有效


def valid(bt, market, stage, key):
    """這個市場×階段×目標的模型，在 2022 年後的資料上有沒有通過驗證"""
    v = (((bt.get("markets", {}).get(market) or {}).get("model") or {}).get("val") or {}).get(stage, {}).get(key)
    if not v:
        return False
    return v["top"] - v["bottom"] >= MIN_EDGE   # 腰斬（h24）也是：預估最高組實際腰斬要明顯比最低組多


CATEGORIES = [  # （分類名稱, 階段, 排序用的勝率）
    ("🔥 發動期最佳", stages.IGNITE, "w12"),
    ("🚀 主升段最佳", stages.RUN, "w12"),
    ("⚠️ 過熱但可能續漲", stages.HOT, "w6"),
    ("🌱 潛伏期最佳", stages.SEED, "w12"),
]


def apply(cand, techs, mrevs, bt, us_fund=None):
    """在 cand 加上：預估勝率（12/6 個月）、預估腰斬、階段平均、理由；並算精選分。"""
    cols = {k: [] for k in ("預估勝率", "預估6月勝率", "預估腰斬", "階段勝率", "階段6月勝率", "階段腰斬", "加分理由", "扣分理由",
                            "勝率模型有效", "6月模型有效", "預估3倍", "階段3倍")}
    for _, r in cand.iterrows():
        m = bt.get("markets", {}).get(r["市場"], {})
        mdl = (m.get("model") or {}).get("stages", {}).get(r["階段"])
        if not mdl:
            for k in cols:
                cols[k].append(None)
            continue
        ok12, ok6 = valid(bt, r["市場"], r["階段"], "w12"), valid(bt, r["市場"], r["階段"], "w6")
        x = live_features(techs[r["代號"]], mrevs.get(r["代號"], {}), m.get("past6_q"), (us_fund or {}).get(r["代號"]))
        est, contrib = predict(mdl, x, m["model"]["w"])
        contrib = [(f, d) for f, d in contrib if abs(d) >= 0.004]
        pos = sorted([c for c in contrib if c[1] > 0], key=lambda c: -c[1])[:3]
        neg = sorted([c for c in contrib if c[1] < 0], key=lambda c: c[1])[:2]
        why = lambda cs: "、".join(f"{LABELS.get(f, f)} {_fmt(f, x[f])}（{d * 100:+.1f}）" for f, d in cs)
        # 沒通過驗證的勝率模型不拿來用：顯示階段平均、不列理由
        for k, v in (("預估勝率", est["w12"] if ok12 else mdl["base"]["w12"]),
                     ("預估6月勝率", est["w6"] if ok6 else mdl["base"]["w6"]), ("預估腰斬", est["h24"]),
                     ("階段勝率", mdl["base"]["w12"]), ("階段6月勝率", mdl["base"]["w6"]), ("階段腰斬", mdl["base"]["h24"]),
                     ("加分理由", why(pos) if ok12 else ""), ("扣分理由", why(neg) if ok12 else ""),
                     ("勝率模型有效", ok12), ("6月模型有效", ok6),
                     ("預估3倍", est.get("x3")), ("階段3倍", mdl["base"].get("x3"))):
            cols[k].append(v)
    for k, v in cols.items():
        cand[k] = v
    # 精選分（各自在同市場候選股裡排百分位）：
    #   勝率模型有通過驗證：預估勝率 50%＋低腰斬風險 25%＋基本面總分 25%
    #   沒通過：低腰斬風險 40%＋基本面總分 60%（腰斬模型在各階段都有通過驗證）
    cand["精選分"] = np.nan
    cand["精選分類"] = ""
    for name, st, key in CATEGORIES:
        win_col = "預估6月勝率" if key == "w6" else "預估勝率"
        for mk in cand["市場"].unique():
            pool = cand[(cand["市場"] == mk) & cand[win_col].notna()]
            if pool.empty:
                continue
            idx = pool.index[(pool["階段"] == st)]
            if not len(idx):
                continue
            ok = valid(bt, mk, st, key)
            w_win, w_half, w_score = (0.5, 0.25, 0.25) if ok else (0.0, 0.4, 0.6)
            sc = (w_win * pool[win_col].rank(pct=True) + w_half * (1 - pool["預估腰斬"]).rank(pct=True)
                  + w_score * pool["總分"].rank(pct=True)) * 100
            cand.loc[idx, "精選分"] = sc[idx].round(0)
            cand.loc[idx, "精選分類"] = name
    return cand


def select(cand, n=5):
    """每個分類 × 市場的前 n 名（有紅旗的不列入）→ {分類: {市場: [代號...]}}"""
    out = {}
    for name, _, _ in CATEGORIES:
        out[name] = {}
        for mk in ("台股", "美股"):
            g = cand[(cand["精選分類"] == name) & (cand["市場"] == mk) & (cand["紅旗"].fillna("") == "")
                     & cand["精選分"].notna()].sort_values("精選分", ascending=False)
            out[name][mk] = g["代號"].head(n).tolist()
    return out


def streaks(history_dir, today_sel):
    """連續上榜天數：從最近一天往回數，每個分類裡連續出現幾次（每次執行＝一個交易日）。
    回傳 {代號: {分類: 天數}}"""
    import json
    from datetime import datetime
    today = f"picks_{datetime.now():%Y%m%d}.json"
    files = sorted(p for p in history_dir.glob("picks_*.json") if p.name != today)   # 今天另外算 1 天
    days = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    out = {}
    for cat, by_mk in today_sel.items():
        for codes in by_mk.values():
            for c in codes:
                k = 1
                for d in reversed(days):
                    if any(c in v for v in d.get(cat, {}).values()):
                        k += 1
                    else:
                        break
                out.setdefault(c, {})[cat] = k
    return out


def strongest(cand, alert=None, ai=None, n=3):
    """🏆 最強名單：全部候選裡，條件最強的前 n 名（每個市場）。全部用條件，不靠感覺：
      必要條件（有任何疑慮就排除）：
        - 階段＝🔥 發動期，或 🚀 主升段前段／中段（後段、過熱、轉弱、下跌都不要）
        - 沒有紅旗（美股自由現金流為負、生技現金不夠燒等）
        - 今天沒有急跌警報、AI 新聞判斷不是「偏空」
        - 模型要有 2 年 3 倍、勝率、腰斬的預估
      排序：最強分＝爆發力（2 年漲 3 倍機率）40%＋12 個月勝率 30%＋低腰斬 30%（同市場排百分位）
      → 2022 年後每週照這規則挑 3 檔的實際表現見 model.top3"""
    bad = set(alert["代號"]) if alert is not None and len(alert) else set()
    ai = ai or {}
    ok = (cand["階段"].isin([stages.IGNITE, stages.RUN]) & (cand["細分"].fillna("") != "後段")
          & (cand["紅旗"].fillna("") == "") & ~cand["代號"].isin(bad)
          & ~cand["代號"].map(lambda c: (ai.get(c) or {}).get("sentiment") == "偏空")
          & cand["預估3倍"].notna() & cand["預估勝率"].notna() & cand["預估腰斬"].notna())
    out = {}
    cand["最強分"] = np.nan
    for mk in ("台股", "美股"):
        g = cand[ok & (cand["市場"] == mk)].rename(columns={"預估3倍": "x3", "預估勝率": "w12", "預估腰斬": "h24"})
        if g.empty:
            out[mk] = []
            continue
        sc = strong_score(g).round(0)
        cand.loc[sc.index, "最強分"] = sc
        out[mk] = sc.sort_values(ascending=False).head(n).index.map(lambda i: cand.at[i, "代號"]).tolist()
    return out
