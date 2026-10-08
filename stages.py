"""股價循環階段：每一天都判斷這檔股票在循環的哪個位置。

即時儀表板和回測（backtest.py）共用這一套，所以回測出來的勝率就是儀表板上那個階段的歷史表現。

規則依據 Weinstein 四階段（打底 → 上漲 → 做頭 → 下跌）與 O'Neil / Minervini 的多頭排列：
  多頭條件 = 收盤 > 150 日線、50 日線 > 150 日線、150 日線往上（比 20 天前高）
  多頭條件中斷不超過 10 個交易日，算回檔、還在同一段多頭；超過就算這段多頭結束。

  🌱 潛伏期  不在多頭、150 日線走平、股價貼著 150 日線、離一年高點跌了 25% 以上（低檔打底）
  🔥 發動期  剛進入多頭：起漲後 40 個交易日（約 2 個月）內，且從起漲點漲不到 30%
  🚀 主升段  多頭持續中；再依「從底部算起漲了多少」分前段／中段／後段
  ⚠️ 過熱    多頭中但離 50 日線太遠（乖離 > 35%，或 RSI > 80 且乖離 > 20%）
  🟠 轉弱    多頭剛結束（120 個交易日內），還沒進入下跌段
  🔻 下跌段  收盤 < 150 日線，且 150 日線往下
  · 盤整     以上皆非
"""
import numpy as np
import pandas as pd

SEED, IGNITE, RUN, HOT, WEAK, DOWN, RANGE = (
    "🌱 潛伏期", "🔥 發動期", "🚀 主升段", "⚠️ 過熱", "🟠 轉弱", "🔻 下跌段", "· 盤整")
ORDER = [SEED, IGNITE, RUN, HOT, WEAK, DOWN, RANGE]          # 循環順序
CODE = {s: i for i, s in enumerate(ORDER)}

GAP = 10            # 多頭條件中斷幾天內算回檔
EARLY_DAYS = 40     # 發動期：起漲後幾個交易日內
EARLY_GAIN = 0.30   # 發動期：從起漲點最多漲多少
WEAK_DAYS = 120     # 多頭結束後多久內算「轉弱」
RUN_PHASES = [(1.0, "前段"), (2.5, "中段"), (np.inf, "後段")]  # 主升段：從底部漲幅 <100%、100–250%、>250%


def rsi(c, n=14):
    """Wilder RSI（和一般看盤軟體相同算法）"""
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run_phase(gain_from_base):
    for hi, name in RUN_PHASES:
        if gain_from_base < hi:
            return name
    return RUN_PHASES[-1][1]


def compute(px):
    """px: DataFrame(Close[, Volume])，日期由舊到新。回傳每天的指標與階段（stage 欄是 ORDER 的編號）。"""
    c = px["Close"].astype(float)
    ma50, ma150 = c.rolling(50).mean(), c.rolling(150).mean()
    slope150 = ma150 / ma150.shift(20) - 1
    ext50 = c / ma50 - 1
    r = rsi(c)
    hi252 = c.rolling(252, min_periods=120).max()

    up = (c > ma150) & (ma50 > ma150) & (slope150 > 0)
    alive = up.astype(float).rolling(GAP, min_periods=1).max().astype(bool)
    start = alive & ~alive.shift(1, fill_value=False)
    seg = start.cumsum()
    days_in = alive.groupby(seg).cumsum().where(alive)              # 起漲第幾個交易日
    start_px = c.where(start).ffill().where(alive)                  # 起漲點收盤價
    start_date = pd.Series(c.index, index=c.index).where(start).ffill().where(alive)
    base_low = c.rolling(120, min_periods=20).min().where(start).ffill().where(alive)  # 起漲前 120 天的最低點
    gain_start = c / start_px - 1
    gain_base = c / base_low - 1
    # 多頭結束後經過幾天
    last_up = pd.Series(np.where(up, np.arange(len(c)), np.nan), index=c.index).ffill()
    since_up = np.arange(len(c)) - last_up

    hot = (ext50 > 0.35) | ((r > 80) & (ext50 > 0.20))
    early = (days_in <= EARLY_DAYS) & (gain_start <= EARLY_GAIN)
    down = (c < ma150) & (slope150 < -0.01)
    weak = ~alive & (since_up <= WEAK_DAYS)
    seed = (slope150.abs() <= 0.02) & ((c / ma150 - 1).abs() <= 0.12) & (c <= 0.75 * hi252)

    st = np.select(
        [alive & hot, alive & early, alive, down, weak, seed],
        [CODE[HOT], CODE[IGNITE], CODE[RUN], CODE[DOWN], CODE[WEAK], CODE[SEED]],
        default=CODE[RANGE])
    out = pd.DataFrame({
        "close": c, "ma50": ma50, "ma150": ma150, "ext50": ext50, "rsi": r,
        "from_high": c / hi252 - 1, "days_in": days_in, "start_px": start_px, "start_date": start_date,
        "base_low": base_low, "gain_start": gain_start, "gain_base": gain_base,
        "low120": c.rolling(120, min_periods=20).min(), "stage": st,
    }, index=c.index)
    out.loc[ma150.isna() | slope150.isna(), "stage"] = -1   # 資料不足
    return out


# 精選模型用的個股特徵（即時與回測共用）。rs_pct（全市場相對強度百分位）和台股月營收另外加。
FEATURES = {
    "above_low": "離半年低點漲幅",
    "ext50": "50 日線乖離",
    "dist150": "離 150 日線（出場線）",
    "from_high": "離一年高點",
    "days_in": "起漲天數",
    "vol_ratio": "近 20 日量／半年均量",
    "rsi": "RSI",
    "slope150": "150 日線斜率",
}


def features(s, volume):
    """s：compute() 的結果；volume：成交量。回傳每天的特徵 DataFrame。"""
    v = volume.reindex(s.index).astype(float)
    return pd.DataFrame({
        "above_low": s["close"] / s["low120"] - 1,
        "ext50": s["ext50"],
        "dist150": s["close"] / s["ma150"] - 1,
        "from_high": s["from_high"],
        "days_in": s["days_in"],
        "vol_ratio": v.rolling(20).mean() / v.rolling(120).mean().replace(0, np.nan),
        "rsi": s["rsi"],
        "slope150": s["ma150"] / s["ma150"].shift(20) - 1,
        "past6": s["close"] / s["close"].shift(126) - 1,
    }, index=s.index)


def describe(row):
    """把某一天的計算結果轉成白話：階段、細分、週期位置說明。"""
    st = ORDER[int(row["stage"])] if row["stage"] >= 0 else "資料不足"
    sub, pos = "", ""
    if st in (IGNITE, RUN, HOT):
        wk = int(row["days_in"] // 5) + 1
        pos = f"起漲第 {wk} 週・起漲以來 {row['gain_start']:+.0%}・從底部 {row['gain_base']:+.0%}"
        if st == RUN:
            sub = run_phase(row["gain_base"])
    elif st == SEED:
        pos = f"離一年高點 {row['from_high']:.0%}・貼近 150 日線打底"
    elif st == WEAK:
        pos = "多頭剛結束，跌破 150 日線或均線轉弱"
    elif st == DOWN:
        pos = f"離一年高點 {row['from_high']:.0%}・150 日線往下"
    return st, sub, pos


def key(stage, sub=""):
    """回測統計的分組鍵（主升段再分前/中/後段）"""
    return f"{stage}・{sub}" if sub else stage
