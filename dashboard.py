"""把評分結果輸出成儀表板：site/index.html（表格＋說明）＋ site/charts/*.json（每檔的 K 線資料，點開才載入）。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import picks
import scoring
import stages

MEANING = {
    stages.SEED: "跌深後在低檔打底，150 日線走平。還沒開始漲，可能要等很久",
    stages.IGNITE: "剛突破進入多頭：起漲 2 個月內、離起漲點漲不到 30%，是上漲循環的最早段",
    stages.RUN: "多頭持續中。從底部算起漲不到 1 倍＝前段（最穩）、1–2.5 倍＝中段、>2.5 倍＝後段（風險高）",
    stages.HOT: "漲太快、離 50 日線太遠，短線容易拉回",
    stages.WEAK: "多頭結束：跌破 150 日線或均線轉弱，這段上漲可能告一段落",
    stages.DOWN: "股價在 150 日線下、而且 150 日線往下，下跌循環中",
    stages.RANGE: "沒有明確方向",
}
COLOR = {stages.SEED: "#0d9488", stages.IGNITE: "#ea580c", stages.RUN: "#dc2626", stages.HOT: "#9333ea",
         stages.WEAK: "#ca8a04", stages.DOWN: "#475569", stages.RANGE: "#a1a1aa"}


def _clean(df):
    recs = df.replace({np.nan: None}).to_dict(orient="records")
    return recs


def pick_stats(bt, market, stage, phase, rs, rv):
    """找最貼近這檔股票目前狀態、且樣本夠多的回測分組"""
    m = bt.get("markets", {}).get(market)
    if not m:
        return None
    S = m["stats"]
    sk = stages.key(stage, phase)
    tries = []
    if rs and rv:
        tries.append((f"st={stage}|rs={rs}|rv={rv}", f"{stage}＋相對強度{rs}＋營收{rv}"))
    if rs:
        tries += [(f"sk={sk}|rs={rs}", f"{sk}＋相對強度{rs}"), (f"st={stage}|rs={rs}", f"{stage}＋相對強度{rs}")]
    if rv:
        tries.append((f"st={stage}|rv={rv}", f"{stage}＋營收{rv}"))
    tries += [(f"sk={sk}", sk), (f"st={stage}", stage)]
    for k, label in tries:
        if k in S:
            return {"key": label, **S[k]}
    return None


def pick_trade(bt, market, rs, rv):
    """照規則操作（發動買進、跌破出場線賣出）的歷史成績，找和這檔進場條件最像的分組"""
    m = bt.get("markets", {}).get(market)
    T = (m or {}).get("trades") or {}
    tries = []
    if rs and rv:
        tries.append((f"rs={rs}|rv={rv}", f"相對強度{rs}＋營收{rv}"))
    if rs:
        tries.append((f"rs={rs}", f"相對強度{rs}"))
    if rv:
        tries.append((f"rv={rv}", f"營收{rv}"))
    tries.append(("ALL", "全部訊號"))
    for k, label in tries:
        if T.get(k):
            return {"key": label, **T[k]}
    return None


def _ohlc(px, rule, n):
    """日線換算成週 K（W-FRI）或月 K（ME），取最後 n 根"""
    g = px.resample(rule)
    d = pd.DataFrame({"o": g["Open"].first(), "h": g["High"].max(), "l": g["Low"].min(),
                      "c": g["Close"].last(), "v": g["Volume"].sum()}).dropna(subset=["c"]).iloc[-n:]
    d["o"] = d["o"].fillna(d["c"])
    dec = 2 if float(d["c"].iloc[-1]) >= 10 else 3
    r = lambda a: [round(float(x), dec) for x in a]
    # 週／月 K 的時間用該期第一個交易日
    first = px["Close"].resample(rule).apply(lambda s: s.index[0] if len(s) else pd.NaT).reindex(d.index)
    return {"t": [x.strftime("%Y-%m-%d") for x in first], "o": r(d["o"]), "h": r(np.maximum(d["h"], d[["o", "c"]].max(axis=1))),
            "l": r(np.minimum(d["l"], d[["o", "c"]].min(axis=1))), "c": r(d["c"]), "v": [int(x) for x in d["v"].fillna(0)]}


def chart_json(t, px):
    s = t["series"].iloc[-520:]
    p = px.reindex(s.index)
    c = p["Close"]
    o = p["Open"].fillna(c)
    h = np.maximum(p["High"].fillna(c), np.maximum(o, c))
    l = np.minimum(p["Low"].fillna(c), np.minimum(o, c))
    full = px["Close"]
    dif = full.ewm(span=12, adjust=False).mean() - full.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    dec = 2 if float(c.iloc[-1]) >= 10 else 3
    r = lambda a, d=dec: [None if pd.isna(x) else round(float(x), d) for x in a]
    return {
        "t": [d.strftime("%Y-%m-%d") for d in s.index],
        "o": r(o), "h": r(h), "l": r(l), "c": r(c), "v": [0 if pd.isna(x) else int(x) for x in p["Volume"]],
        "ma50": r(s["ma50"]), "ma150": r(s["ma150"]), "rsi": r(s["rsi"], 1),
        "dif": r(dif.reindex(s.index), dec + 1), "dea": r(dea.reindex(s.index), dec + 1),
        "st": [int(x) for x in s["stage"]],
        "w": _ohlc(px, "W-FRI", 260), "mo": _ohlc(px, "ME", 120),
    }


def build(cand, ind, hot_sub, techs, prices, sym_of, bt, demo=False, sel=None, streak=None,
          surge=None, surge_px=None, ai=None, alert=None, intraday=None):
    now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M") + "（台灣時間）"
    rows = _clean(cand)
    for r in rows:
        r["bt"] = pick_stats(bt, r["市場"], r["階段"], r["細分"], r["相對強度"], r["營收狀態"])
        r["tr"] = pick_trade(bt, r["市場"], r["相對強度"], r["營收狀態"])
    # 各階段的歷史表現（給上方說明卡用）
    guide = []
    for st in stages.ORDER:
        g = {"stage": st, "meaning": MEANING[st], "action": scoring.ACTION[st], "color": COLOR[st], "bt": {}}
        for mk, m in bt.get("markets", {}).items():
            v = m["stats"].get(f"st={st}")
            if v:
                g["bt"][mk] = v
        guide.append(g)
    base = {mk: {**m["stats"]["ALL"], "n_stocks": m["n_stocks"], "period": m["period"]}
            for mk, m in bt.get("markets", {}).items()}
    dates = cand.groupby("市場")["資料日期"].max().to_dict() if len(cand) else {}

    # 每檔的 K 線資料另存一個檔，點開才載入
    chart_dir = config.OUTPUT_HTML.parent / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for code, t in techs.items():
        px = prices.get(sym_of.get(code))
        if px is not None:
            cj = chart_json(t, px)
            cj["i"] = (intraday or {}).get(sym_of.get(code))
            (chart_dir / f"{code}.json").write_text(json.dumps(cj, separators=(",", ":")), encoding="utf-8")

    for code, (px, s) in (surge_px or {}).items():           # 暴衝雷達的股票（可能不在觀察清單）
        if code not in techs:
            cj = chart_json({"series": s}, px)
            ys = dict(zip(surge["代號"], surge["yf"])).get(code) if surge is not None and "yf" in surge else None
            cj["i"] = (intraday or {}).get(ys)
            (chart_dir / f"{code}.json").write_text(json.dumps(cj, separators=(",", ":")), encoding="utf-8")
    surge_stats = {mk: m.get("surge", {}) for mk, m in bt.get("markets", {}).items()}
    trades = {mk: m.get("trades") for mk, m in bt.get("markets", {}).items() if m.get("trades")}
    val = {mk: (m.get("model") or {}).get("val", {}) for mk, m in bt.get("markets", {}).items()}
    cats = [{"name": n, "stage": st, "key": k} for n, st, k in picks.CATEGORIES]
    cats.append({"name": "⚡ 突然暴衝", "stage": "surge", "key": "surge"})
    # 公司資料連結要用的代號：台股的 .TW／.TWO、美股的 SEC CIK
    yf = {c: s_ for c, s_ in sym_of.items() if s_.endswith((".TW", ".TWO"))}
    if surge is not None and len(surge) and "yf" in surge:
        yf.update({c: s_ for c, s_ in zip(surge["代號"], surge["yf"]) if str(s_).endswith((".TW", ".TWO"))})
    try:
        import sec
        cikmap = sec.load_cik()
    except Exception:
        cikmap = {}
    codes = set(cand["代號"]) | set(ind["代號"] if len(ind) else []) | set(surge["代號"] if surge is not None and len(surge) else [])
    cik = {c: int(cikmap[c]) for c in codes if c in cikmap}
    crash = {mk: m.get("crash", {}) for mk, m in bt.get("markets", {}).items()}
    data = {"alerts": _clean(alert) if alert is not None and len(alert) else [], "crashStats": crash,
            "yf": yf, "cik": cik, "sel": sel or {}, "streak": streak or {}, "ai": ai or {}, "surgeStats": surge_stats,
            "surge": _clean(surge) if surge is not None and len(surge) else [],
            "trades": trades, "val": val, "cats": cats, "cand": rows, "ind": _clean(ind), "hot": hot_sub, "guide": guide, "base": base,
            "dates": dates, "time": now + ("（模擬資料，僅供預覽）" if demo else ""),
            "btDate": bt.get("generated"), "order": stages.ORDER,
            "colors": [COLOR[s] for s in stages.ORDER]}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/"))
    config.OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_HTML.write_text(html, encoding="utf-8")
    return config.OUTPUT_HTML


TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>十倍股追蹤器</title>
<script>try{if(localStorage.getItem("theme")==="light")document.documentElement.dataset.theme="light"}catch(e){}</script>
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#141416;--card:#1d1d20;--ink:#ececee;--mute:#9a9aa2;--line:#2c2c31;--acc:#c2410c;
--up:#f87171;--dn:#4ade80;--chip:#26262b;--good:#3b2f12;color-scheme:dark}
:root[data-theme="light"]{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e2dc;--acc:#c2410c;
--up:#dc2626;--dn:#16a34a;--chip:#f0eee8;--good:#fef3c7}

*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,"Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0}.sub{color:var(--mute);font-size:12px;margin-top:2px}
h2{font-size:16px;margin:26px 0 8px}
details.guide{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:16px 0}
details.guide summary{cursor:pointer;font-weight:600}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin:10px 0 4px}
.step{background:var(--chip);border-radius:8px;padding:8px 10px;font-size:13px}.step b{color:var(--acc)}
.cycle{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:8px;margin:14px 0}
.st{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--c);border-radius:10px;padding:9px 11px;cursor:pointer;font-size:12px;position:relative}
.st.on{outline:2px solid var(--c)}
.st .h{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;font-weight:700}
.st .h span{font-size:20px;color:var(--c)}
.st .m{color:var(--mute);margin:3px 0 5px;min-height:3.1em}
.st .a{font-weight:600;margin-bottom:4px}
.st .b{border-top:1px dashed var(--line);padding-top:4px;color:var(--mute)}
.rule{background:var(--chip);border-radius:10px;padding:10px 12px;margin:12px 0 4px;font-size:13px}
.rule h4{margin:0 0 4px;font-size:14px}.rule table{min-width:0;width:auto;margin-top:6px}
.rule td,.rule th{padding:3px 10px;border-bottom:1px solid var(--line);position:static;background:none}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
select,input{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font:inherit}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:auto}
table{border-collapse:collapse;width:100%;min-width:1060px}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;vertical-align:top}
th{position:sticky;top:0;background:var(--card);font-size:12px;color:var(--mute);cursor:pointer;user-select:none;z-index:1}
td.l,th.l{text-align:left}
#t td:first-child,#t th:first-child{position:sticky;left:0;background:var(--card);z-index:2}
#t th:first-child{z-index:3}
td.wrapc{white-space:normal;min-width:120px;max-width:170px}
.seg{display:inline-flex;background:var(--chip);border-radius:10px;padding:3px;gap:3px;margin:14px 0 4px}
.seg button{border:0;background:none;color:var(--ink);font:inherit;font-weight:600;padding:6px 18px;border-radius:8px;cursor:pointer}
.seg button.on{background:var(--acc);color:#fff}
.picks{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px;align-items:start}
@media (min-width:1240px){.picks{grid-template-columns:repeat(5,1fr)}}
.pk{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--c);border-radius:12px;padding:10px 12px;position:relative}
.pk .body{height:330px;overflow:hidden;position:relative}
.pk.open .body{height:auto}
.pk:not(.open) .body::after{content:"";position:absolute;left:0;right:0;bottom:0;height:56px;background:linear-gradient(transparent,var(--card))}
.pk .more{display:block;width:100%;margin-top:6px;border:0;border-radius:8px;background:var(--chip);color:var(--ink);font:inherit;font-size:12px;padding:5px;cursor:pointer}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:0 6px;border-radius:99px;margin-left:4px;vertical-align:1px;background:var(--chip);color:var(--mute)}
.badge.b5{background:#fed7aa;color:#9a3412}.badge.b10{background:linear-gradient(90deg,#fde68a,#f59e0b);color:#78350f}
.sent{display:inline-block;font-size:11px;padding:0 6px;border-radius:99px;color:#fff;margin-right:4px}
.sent.偏多{background:var(--up)}.sent.偏空{background:var(--dn)}.sent.中性{background:#71717a}
.newop{color:var(--acc);font-weight:700;font-size:11px}
.alerts{background:var(--card);border:1px solid var(--line);border-left:5px solid #dc2626;border-radius:12px;padding:10px 12px;margin:14px 0 4px}
.alerts h3{margin:0 0 4px;font-size:15px}.okbar{color:var(--mute);font-size:13px;margin:14px 0 4px}
.al{display:grid;grid-template-columns:1fr auto;gap:2px 10px;padding:7px 0;border-top:1px solid var(--line);cursor:pointer}
.al:hover{background:var(--chip)}.al .tg{display:inline-block;font-size:11px;font-weight:700;padding:0 6px;border-radius:99px;background:#fee2e2;color:#991b1b;margin-right:4px}
.al .hint{grid-column:1/3;font-size:12px;color:var(--mute)}
.links{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.links a{font-size:12px;padding:3px 10px;border-radius:99px;background:var(--chip);color:var(--ink);text-decoration:none;border:1px solid var(--line)}
.links a:hover{border-color:var(--acc);color:var(--acc)}
.ai ul{margin:4px 0 6px 18px;padding:0}.ai li{margin:2px 0}
.pk h3{margin:0;font-size:15px;cursor:pointer;user-select:none;display:flex;justify-content:space-between}
.pk h3 .tog{color:var(--mute);font-size:12px;font-weight:400}.pk .vn{color:var(--mute);font-size:11px;margin:2px 0 6px}
.pi{display:grid;grid-template-columns:22px 1fr auto;gap:2px 8px;padding:7px 0;border-top:1px solid var(--line);cursor:pointer}
.pi:hover{background:var(--chip)}.pi .rk{font-weight:700;color:var(--c);font-size:16px}
.pi .sc{font-size:20px;font-weight:700;text-align:right}.pi .why{grid-column:2/4;font-size:12px;color:var(--mute)}
.empty{color:var(--mute);font-size:12px;padding:8px 0}.empty a{color:var(--acc)}tbody tr{cursor:pointer}tbody tr:hover td{background:var(--chip)}
.name{font-weight:600}.code{color:var(--mute);font-size:12px;margin-left:4px}.small{color:var(--mute);font-size:12px}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;color:#fff;background:var(--c)}
.sbar{display:inline-block;width:56px;height:7px;background:var(--chip);border-radius:9px;vertical-align:middle;margin-right:5px;overflow:hidden}
.sbar i{display:block;height:100%;background:var(--acc)}
.up{color:var(--up)}.dn{color:var(--dn)}.flag{color:var(--up);font-size:12px}
.hot{color:var(--acc);font-weight:600}.good{background:var(--good);border-radius:5px;padding:0 4px}
.stale{color:var(--up);font-weight:600}
.note{color:var(--mute);font-size:12px;margin-top:14px;line-height:1.8}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);font-size:12px}
/* 個股視窗 */
#dlg{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;z-index:10;overflow:auto;padding:24px 12px}
#dlg.open{display:block}
.panel{background:var(--bg);max-width:1120px;margin:0 auto;border-radius:14px;padding:16px;position:relative}
.x{position:absolute;right:12px;top:10px;font-size:26px;cursor:pointer;color:var(--mute);background:none;border:0}
.ph{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}.ph .px{font-size:26px;font-weight:700}
.boxes{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin:12px 0}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.box h3{font-size:13px;margin:0 0 6px;color:var(--mute)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:13px}.kv b{text-align:right}
.chartbox{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.legend{font-size:12px;color:var(--mute);min-height:18px;padding:2px 4px}
.rng button{background:var(--chip);border:0;border-radius:6px;padding:3px 9px;margin-right:4px;cursor:pointer;color:var(--ink);font:inherit;font-size:12px}
.rng button.on{background:var(--acc);color:#fff}
.ribbon{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:11px;color:var(--mute);margin:4px}
.ribbon i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px;vertical-align:-1px}
</style></head><body><div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:center;gap:10px"><h1>十倍股追蹤器</h1>
<button id="theme" style="border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:99px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer"></button></div>
<div class="sub" id="sub"></div>
<div class="seg" id="mk"><button data-m="" class="on">全部</button><button data-m="台股">台股</button><button data-m="美股">美股</button></div>

<div id="alerts"></div>

<h2>今日精選（依回測勝率＋基本面）</h2>
<div class="small" id="pknote"></div>
<div class="picks" id="picks" style="margin-top:8px"></div>

<details class="guide"><summary>怎麼看這張表：階段說明、進出場規則、回測（點開）</summary>
<div class="steps">
 <div class="step"><b>① 看階段</b>：這檔股票現在在漲跌循環的哪個位置（下方 7 張卡，依循環順序排）。</div>
 <div class="step"><b>② 看歷史勝率</b>：過去 10 年全市場股票在「同樣狀態」時，12 個月後上漲的比例。比全市場平均高才有優勢。</div>
 <div class="step"><b>③ 看出場線</b>：持有中的股票跌破出場線（150 日線）、10 個交易日內沒站回，就是規則上的出場訊號；跌破減碼線（50 日線）先減碼。</div>
 <div class="step"><b>④ 分數用來排序</b>：同一個階段裡，總分高的基本面＋動能比較好，優先研究。點任一列看 K 線、MACD、RSI。</div>
</div>
<div class="rule" id="rule"></div>
<div class="cycle" id="cycle"></div>
<div class="small" id="btnote"></div>
</details>

<h2>全部候選股</h2>
<div class="bar">
 <select id="ft"><option value="">全部主題</option></select>
 <input id="q" placeholder="搜尋代號或名稱">
 <label><input type="checkbox" id="fh"> 只看產業已發動</label>
</div>
<div class="card"><table id="t"><thead></thead><tbody></tbody></table></div>

<h2>產業啟動訊號（指標股）</h2>
<div class="small">大型龍頭不太可能十倍，用來判斷子產業有沒有發動：龍頭進入多頭且強過大盤 → 同子題的小型股加分。</div>
<div class="chips" id="hot" style="margin-top:8px"></div>
<div class="card" style="margin-top:10px"><table id="ti" style="min-width:700px"><thead></thead><tbody></tbody></table></div>

<div class="note">
勝率是用全市場過去 10 年的歷史回測算的「同樣狀態後來的表現」，不是對個股的預測；已下市的股票抓不到，絕對報酬會偏高，請以「比全市場平均好多少」來看。<br>
分數是相對排名的參考。顏色採台股習慣：<span class="up">紅＝漲</span>、<span class="dn">綠＝跌</span>。<b>這不是買賣建議。</b>
</div>
</div>

<div id="dlg"><div class="panel">
 <button class="x" onclick="closeDlg()">×</button>
 <div id="dhead"></div>
 <div class="boxes" id="dboxes"></div>
 <div class="chartbox">
  <div class="rng" id="rng"><button data-tf="d1">當日</button><button data-tf="d5">五日</button><button data-tf="day">日K</button><button data-tf="week" class="on">週K</button><button data-tf="month">月K</button></div>
  <div class="ribbon" id="ribbon"></div>
  <div class="legend" id="lg1"></div><div id="c1" style="height:340px"></div>
  <div class="legend" id="lg2"></div><div id="c2" style="height:130px"></div>
  <div class="legend" id="lg3"></div><div id="c3" style="height:120px"></div>
 </div>
 <div class="boxes" id="dboxes2"></div>
</div></div>

<script>
const D=__DATA__, C=D.cand, I=D.ind;
const colorOf=s=>D.colors[D.order.indexOf(s)]||"#a1a1aa";
let fStage="", sortK="總分", sortD=-1;
const pc=(v,d=0)=>v==null?"–":`<span class="${v>0?'up':v<0?'dn':''}">${v>0?"+":""}${(v*100).toFixed(d)}%</span>`;
const pp=(v,d=0)=>v==null?"–":(v*100).toFixed(d)+"%";
const num=v=>v==null?"–":(v>=1000?v.toLocaleString(undefined,{maximumFractionDigits:0}):v>=10?v.toFixed(2):v.toFixed(3));
const sb=(v,m)=>`<span class="sbar"><i style="width:${Math.max(0,Math.min(100,v/m*100))}%"></i></span>${v??"–"}`;
const tag=s=>`<span class="tag" style="--c:${colorOf(s)}">${s}</span>`;
const latest=D.dates;

document.getElementById("sub").innerHTML=`更新時間：${D.time}　｜　最新股價日期：`+
 Object.entries(latest).map(([m,d])=>`${m} ${d}`).join("、")+(D.btDate?`　｜　回測更新：${D.btDate}`:"");

function rule(){
 const T=D.trades, rows=[];
 const lab={"ALL":"全部訊號","rs=強":"＋相對強度強","rv=加速":"＋月營收加速","rs=強|rv=加速":"＋強勢＋營收加速"};
 Object.entries(T).forEach(([m,t])=>Object.entries(lab).forEach(([k,l])=>{const v=t[k];if(v)rows.push(`<tr><td class="l">${m}</td><td class="l">${l}</td><td>${v.n.toLocaleString()}</td><td><b>${pp(v.win)}</b></td><td class="up">${pc(v.avg_win)}</td><td>${pc(v.avg_loss)}</td><td><b>${pc(v.exp,1)}</b></td><td>${pc(v.bexp,1)}</td><td>${pp(v.p100,1)}</td><td>${v.weeks.toFixed(0)} 週</td></tr>`)}));
 document.getElementById("rule").innerHTML=rows.length?`<h4>進出場規則（10 年全市場回測）</h4>
  <b>🔥 發動（進入多頭）那天買進 → 跌破出場線（150 日線）後 10 個交易日內沒站回（＝轉弱）就賣出</b>。持有中跌破 50 日線可先減碼。<br>
  <span class="small">勝率不到一半是正常的：這類規則靠「小賠大賺」——買錯時照出場線小賠出場，抓對時讓它一路漲。重點是<b>期望值</b>（平均每筆賺多少）為正，而且<b>一定要守出場線</b>。<br>
  <b>誠實提醒：</b>平均每筆的報酬和「同一段時間買大盤」差不多，代表光靠這個技術規則不會穩定贏大盤；它的價值在<b>控制虧損</b>、<b>避開過熱與主升後段</b>（之後一年中位數為負、腰斬機率最高）。要找到真正的十倍股，還是要靠基本面（總分）先選對股票，再用規則決定進出場時機。</span>
  <div style="overflow:auto"><table><tr><th class="l">市場</th><th class="l">進場條件</th><th>交易數</th><th>勝率</th><th>賺時平均</th><th>賠時平均</th><th>每筆期望值</th><th>同期間買大盤</th><th>單筆賺 1 倍以上</th><th>平均持有</th></tr>${rows.join("")}</table></div>`:"";
}
function guide(){
 const b=D.base;
 document.getElementById("cycle").innerHTML=D.guide.map((g,i)=>{
  const n=C.filter(r=>r.階段==g.stage).length;
  const bt=Object.entries(g.bt).map(([m,v])=>{const base=b[m];
    const w=v.win12!=null&&base.win12!=null?((v.win12-base.win12)*100):null;
    return `${m}：12月勝率 <b>${pp(v.win12)}</b>${w!=null?`（${w>=0?"+":""}${w.toFixed(0)}）`:""}<br>　2年漲3倍 ${pp(v.p3x)}・腰斬 ${pp(v.half24)}`}).join("<br>");
  return `<div class="st ${fStage==g.stage?'on':''}" style="--c:${g.color}" data-s="${g.stage}">
   <div class="h">${i+1}. ${g.stage}<span>${n}</span></div><div class="m">${g.meaning}</div>
   <div class="a">→ ${g.action}</div><div class="b">${bt||"尚無回測"}</div></div>`}).join("");
 document.querySelectorAll(".st").forEach(e=>e.onclick=()=>{fStage=fStage==e.dataset.s?"":e.dataset.s;guide();draw()});
 const parts=Object.entries(b).map(([m,v])=>`${m} ${v.n_stocks} 檔（${v.period[0]}～${v.period[1]}）全市場平均：12月勝率 ${pp(v.win12)}、2年內漲3倍 ${pp(v.p3x,1)}、腰斬 ${pp(v.half24)}`);
 document.getElementById("btnote").innerHTML=parts.length?`卡片下方括號＝比全市場平均高／低幾個百分點；3倍率＝2 年內曾漲到 3 倍的機率是平均的幾倍。回測樣本：${parts.join("；")}。「腰斬」＝2 年內曾跌掉一半的機率，和「漲 3 倍」一起看才知道風險。點卡片可篩選。`:"尚未執行回測（python backtest.py）";
}

/* ---------- 市場切換 ---------- */
let mk="";
document.querySelectorAll("#mk button").forEach(b=>b.onclick=()=>{mk=b.dataset.m;
 document.querySelectorAll("#mk button").forEach(x=>x.classList.toggle("on",x==b));alertsView();picks();draw()});

/* ---------- 急跌警報 ---------- */
const AL={};D.alerts.forEach(a=>AL[a.代號]=a);
function crashHint(a){
 if(a.警報.includes("跌破出場線")&&!a.警報.includes("急跌"))return "依進出場規則（回測過的）：跌破出場線（150 日線）後，10 個交易日內沒站回就出場；先減碼也可以。";
 const cs=D.crashStats[a.市場]||{};
 if(a.市場=="美股"&&a.自由現金流殖利率!=null){const pos=a.自由現金流殖利率>0, v=cs[pos?"急跌|自由現金流為正":"急跌|自由現金流為負"];
  if(v)return `歷史：美股急跌時自由現金流${pos?"為正":"為負"}的，1 年後中位數 ${pc(v.med12)}、腰斬 ${pp(v.half24)}——${pos?"常反彈，可能是錯殺，先查原因":"常續跌，偏向出場"}。`}
 const v=cs["急跌"];return v?`歷史：${a.市場}急跌後 1 年中位數 ${pc(v.med12)}、6 個月內常再跌 ${pc(v.dd6)}。先看 AI 判斷的原因。`:"";}
function alertsView(){
 const rows=D.alerts.filter(a=>!mk||a.市場==mk);
 const el=document.getElementById("alerts");
 if(!rows.length){el.innerHTML=`<div class="okbar">✅ 今天${mk||""}觀察清單沒有急跌或跌破出場線的股票</div>`;return}
 el.innerHTML=`<div class="alerts"><h3>🚨 急跌警報（${rows.length}）</h3><div class="small">觀察清單＋今日暴衝股：🚨 急跌＝5 日跌 ≥15% 或單日跌 ≥8% 且爆量；⛔ 跌破出場線＝多頭中剛跌破 150 日線。</div>
  ${rows.map(a=>`<div class="al" data-c="${a.代號}"><div>${a.警報.split("・").map(t=>`<span class="tg">${t}</span>`).join("")}<span class="name">${a.名稱}</span><span class="code">${a.代號}・${a.市場}${a.來源=="暴衝股"?"・暴衝股":""}</span>　<b>${num(a.股價)}</b>　今日 ${pc(a["1日漲跌"],1)}・5 日 ${pc(a["5日漲跌"])}・量 ${a.量能倍數.toFixed(1)} 倍${a.嚴重?' <b class="flag">嚴重</b>':''}</div>
   <div></div><div class="hint">${aiLine(a.代號)?aiLine(a.代號)+"<br>":""}${crashHint(a)}</div></div>`).join("")}</div>`;
 el.querySelectorAll(".al").forEach(e=>e.onclick=()=>openDlg(e.dataset.c));
}
function links(code,m){
 if(m=="台股"){const ys=D.yf[code]||code+".TW";
  return [["Yahoo 股市・公司基本資料",`https://tw.stock.yahoo.com/quote/${ys}/profile`],["Goodinfo・財報／股利",`https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID=${code}`],["鉅亨網・個股",`https://www.cnyes.com/twstock/${code}`]]}
 const cik=D.cik[code];
 return [["Yahoo Finance・公司簡介",`https://finance.yahoo.com/quote/${code}/profile/`],["StockAnalysis・完整財報",`https://stockanalysis.com/stocks/${code.toLowerCase()}/financials/`],
  ["SEC EDGAR・官方申報",cik?`https://www.sec.gov/edgar/browse/?CIK=${cik}`:`https://www.sec.gov/edgar/search/#/q=${code}`]]}
const linkHtml=(code,m)=>`<div class="links">🔗 ${links(code,m).map(([t,u])=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`).join("")}</div>`;

/* ---------- 今日精選 ---------- */
const openCards=new Set();
function badge(code,cat){const n=((D.streak[code]||{})[cat])||0;if(n<2)return "";
 const t=n>=10?`👑 ${Math.floor(n/5)}週`:n>=5?`🔥 ${n}天`:`連${n}天`;
 return `<span class="badge ${n>=10?'b10':n>=5?'b5':''}" title="連續 ${n} 個交易日上榜">${t}</span>`}
function aiLine(code){const a=D.ai[code];if(!a)return "";
 return `<span class="sent ${a.sentiment}">${a.sentiment}</span>${a.new_opportunity?'<span class="newop">★ 新商機 </span>':''}${a.summary||""}`}
function pickItem(r,i,c){
 const isHot=c.key=="w6", wk=isHot?"預估6月勝率":"預估勝率", sk=isHot?"階段6月勝率":"階段勝率", okKey=isHot?"6月模型有效":"勝率模型有效";
 const better=r[wk]-r[sk];
 return `<div class="pi" data-c="${r.代號}"><span class="rk">${i+1}</span>
  <div><span class="name">${r.名稱}</span>${badge(r.代號,c.name)}<span class="code">${r.代號}・${r.市場}</span>　<b>${num(r.股價)}</b> ${pc(r.日漲跌,1)}<br>
  <span class="small">${r.細分?r.細分+"・":""}${r[okKey]?`${isHot?"6 個月":"12 個月"}勝率 <b class="${better>0?'up':''}">${pp(r[wk])}</b>（階段 ${pp(r[sk])}）`:`勝率 階段平均 ${pp(r[sk])}`}・腰斬 <b>${pp(r.預估腰斬)}</b>・總分 ${r.總分}</span></div>
  <div class="sc">${r.精選分}<div class="small" style="font-weight:400">精選分</div></div>
  <div class="why">${aiLine(r.代號)?aiLine(r.代號)+"<br>":""}${r.加分理由?"✔ "+r.加分理由:""}${r.扣分理由?"<br>✘ "+r.扣分理由:""}<br>出場線 ${r.出場線?num(r.出場線)+"（距離 "+pp(r.股價/r.出場線-1)+"）":"–"}</div></div>`}
function surgeItem(r,i){
 const st=((D.surgeStats[r.市場]||{})["暴衝|"+r.階段])||{};
 return `<div class="pi" data-c="${r.代號}"><span class="rk">${i+1}</span>
  <div><span class="name">${r.名稱}</span>${badge(r.代號,"⚡ 突然暴衝")}<span class="code">${r.代號}・${r.市場}${r.在觀察清單?"・觀察清單":""}</span>　<b>${num(r.股價)}</b> ${pc(r.日漲跌,1)}<br>
  <span class="small">5 日 <b class="up">${pc(r["5日漲幅"])}</b>・量 <b>${r.量能倍數.toFixed(1)} 倍</b>・${r.階段}${r.細分?" "+r.細分:""}${r.市場=="美股"?(r.自由現金流殖利率!=null?`・FCF 殖利率 <b class="${r.自由現金流殖利率>0?'up':'dn'}">${pp(r.自由現金流殖利率,1)}</b>`:"・無財報"):""}</span></div>
  <div class="sc" style="font-size:16px">${pc(r["5日漲幅"])}<div class="small" style="font-weight:400">5 日</div></div>
  <div class="why">${aiLine(r.代號)||"（尚無 AI 分析）"}${st.n?`<br>歷史上「${r.階段}時暴衝」：12 月勝率 ${pp(st.win12)}、腰斬 ${pp(st.half24)}、4 年 10 倍 ${pp(st.p10x48,1)}`:""}</div></div>`}
function picks(){
 const out=D.cats.map(c=>{
  const isSurge=c.key=="surge", isHot=c.key=="w6";
  const mks=mk?[mk]:["台股","美股"];
  const codes=mks.flatMap(m=>((D.sel[c.name]||{})[m]||[]));
  let items, vn="";
  if(isSurge){
   const rows=codes.map(x=>D.surge.find(r=>r.代號==x)).filter(x=>x).sort((a,b)=>b.暴衝強度-a.暴衝強度);
   items=rows.length?rows.map((r,i)=>surgeItem(r,i)).join(""):`<div class="empty">今天全市場沒有${mk||""}股票符合暴衝條件</div>`;
   const ss=mks.map(m=>{const v=(D.surgeStats[m]||{})["暴衝"],b=(D.surgeStats[m]||{}).ALL;return v&&b?`${m}暴衝後 12 月中位 ${pc(v.med12)}、腰斬 ${pp(v.half24)}（平均 ${pp(b.half24)}）、4 年 10 倍 ${pp(v.p10x48,1)}（平均 ${pp(b.p10x48,1)}）`:""}).filter(x=>x).join("；");
   vn=`全市場掃描：近 5 日漲 ≥15% 且量 ≥ 季均量 2.5 倍。<b>這是研究名單不是買進名單</b>——回測：${ss}。大商機藏在裡面但多數會回吐，請看 AI 的原因分析，並優先挑自由現金流為正的。`;
  }else{
   const rows=codes.map(x=>C.find(r=>r.代號==x)).filter(x=>x).sort((a,b)=>b.精選分-a.精選分);
   items=rows.length?rows.map((r,i)=>pickItem(r,i,c)).join(""):`<div class="empty">目前沒有${mk||""}候選股在這個階段</div>`;
   vn=mks.map(m=>{const v=(D.val[m]||{})[c.stage];const t=v&&v[c.key];
    return t?`${m} ${pp(t.top)} vs ${pp(t.bottom)}${t.top-t.bottom<0.05?' <span class="flag">未通過→改用腰斬＋總分</span>':' ✅'}`:""}).filter(x=>x).join("；");
   vn=`${isHot?"以「6 個月後仍上漲」排序。":""}驗證（2022 後，預估最高 vs 最低 20% 的實際勝率）：${vn||"樣本不足"}`;
   const flagged=C.filter(r=>r.精選分類==c.name&&r.紅旗&&(!mk||r.市場==mk));
   if(flagged.length)items+=`<div class="empty">另有 ${flagged.length} 檔因紅旗未列入：${flagged.map(r=>`<a href="#${r.代號}" onclick="openDlg('${r.代號}');return false">${r.名稱}</a>`).join("、")}</div>`;
  }
  const open=openCards.has(c.name);
  return `<div class="pk ${open?'open':''}" style="--c:${isSurge?'#e11d48':colorOf(c.stage)}" data-n="${c.name}"><h3 title="點一下展開／收合">${c.name}<span class="tog">${open?"收合 ▴":"展開 ▾"}</span></h3>
   <div class="body"><div class="vn">${vn}</div>${items}</div><button class="more">${open?"收合 ▴":"展開全部 ▾"}</button></div>`}).join("");
 document.getElementById("picks").innerHTML=out;
 document.querySelectorAll(".pi").forEach(e=>e.onclick=()=>openDlg(e.dataset.c));
 document.querySelectorAll(".pk .more, .pk h3").forEach(b=>b.onclick=()=>{const n=b.closest(".pk").dataset.n;openCards.has(n)?openCards.delete(n):openCards.add(n);picks()});
 document.getElementById("pknote").innerHTML=`<b>精選分</b>＝回測預估勝率 50%＋低腰斬風險 25%＋基本面總分 25%（勝率模型沒通過驗證的改成腰斬 40%＋總分 60%）。徽章＝連續上榜：連N天／🔥 5 天以上／👑 2 週以上。<span class="sent 偏多">偏多</span><span class="sent 中性">中性</span><span class="sent 偏空">偏空</span>＝AI 讀近兩週新聞的判斷，點個股看利多利空。`;
}

const cols=[["名稱","l"],["股價"],["階段","l"],["操作","l"],["出場線"],["預估勝率"],["腰斬風險"],["強度／營收"],["總分"],["精選分"]];
const val=(r,k)=>k=="出場線"?(r.出場線?r.股價/r.出場線-1:null):k=="名稱"?r.代號:k=="強度／營收"?r["6月漲幅"]:k=="腰斬風險"?r.預估腰斬:r[k];
function draw(){
 const m=mk,t=ft.value,q=document.getElementById("q").value.trim().toLowerCase(),h=fh.checked;
 let rows=C.filter(r=>(!m||r.市場==m)&&(!t||r.主題==t)&&(!fStage||r.階段==fStage)&&(!h||r.產業已發動)
   &&(!q||(r.代號+r.名稱).toLowerCase().includes(q)));
 rows.sort((a,b)=>{const x=val(a,sortK),y=val(b,sortK);return (x==null)-(y==null)||(x>y?1:x<y?-1:0)*sortD});
 document.querySelector("#t thead").innerHTML="<tr>"+cols.map(([c,a])=>`<th class="${a||''}" data-k="${c}">${c}${sortK==c?(sortD<0?" ▼":" ▲"):""}</th>`).join("")+"</tr>";
 document.querySelectorAll("#t th").forEach(e=>e.onclick=()=>{const k=e.dataset.k;sortD=sortK==k?-sortD:-1;sortK=k;draw()});
 document.querySelector("#t tbody").innerHTML=rows.map(r=>{
  const stale=latest[r.市場]&&r.資料日期<latest[r.市場];
  const dist=r.出場線?r.股價/r.出場線-1:null;
  const wd=r.預估勝率!=null&&r.階段勝率!=null?r.預估勝率-r.階段勝率:null;
  const hd=r.預估腰斬!=null&&r.階段腰斬!=null?r.預估腰斬-r.階段腰斬:null;
  return `<tr data-c="${r.代號}">
 <td class="l">${AL[r.代號]?'<span title="'+AL[r.代號].警報+'">🚨</span>':''}<span class="name">${r.名稱}</span><span class="code">${r.代號}</span><br><span class="small">${r.市場}・${r.子題}</span>${r.產業已發動?' <span class="hot small">●發動</span>':''}${r.紅旗?`<br><span class="flag">⚑ ${r.紅旗}</span>`:""}</td>
 <td><b>${num(r.股價)}</b><br>${pc(r.日漲跌,1)} <span class="small ${stale?'stale':''}">${r.資料日期.slice(5)}</span></td>
 <td class="l">${tag(r.階段)}${r.細分?` <b>${r.細分}</b>`:""}<br><span class="small">${r.週期位置||""}</span></td>
 <td class="l wrapc">${r.操作}</td>
 <td>${r.出場線?num(r.出場線):"–"}<br><span class="small">${dist!=null?"距離 "+pp(dist):""}</span></td>
 <td><span class="${r.勝率模型有效&&wd>0.03?'good':''}">${pp(r.預估勝率)}</span><br><span class="small">${r.勝率模型有效?"階段 "+pp(r.階段勝率):"未驗證＝階段平均"}</span></td>
 <td><span class="${hd!=null&&hd<-0.03?'good':''}">${pp(r.預估腰斬)}</span><br><span class="small">階段 ${pp(r.階段腰斬)}</span></td>
 <td>${r.相對強度||"–"}・${pc(r["6月漲幅"])}<br><span class="small">營收 ${pc(r.營收YoY)} ${r.營收狀態||(r.營收加速!=null?(r.營收加速>0?"加速":"減速"):"")}</span></td>
 <td><b>${sb(r.總分,100)}</b><br><span class="small">⚡${r["⚡爆發力"]} 🏔️${r["🏔️長跑力"]}</span></td>
 <td><b>${r.精選分??"–"}</b></td></tr>`}).join("");
 document.querySelectorAll("#t tbody tr").forEach(e=>e.onclick=()=>openDlg(e.dataset.c));
}
function ind(){
 document.getElementById("hot").innerHTML=D.hot.length?D.hot.map(s=>`<span class="chip hot">● ${s}</span>`).join(""):'<span class="small">目前沒有子產業被指標股點火</span>';
 document.querySelector("#ti thead").innerHTML="<tr><th class='l'>指標股</th><th>股價</th><th class='l'>子題</th><th class='l'>階段</th><th>6月超額</th><th>離高點</th></tr>";
 document.querySelector("#ti tbody").innerHTML=[...I].sort((a,b)=>b["6月超額"]-a["6月超額"]).map(r=>`<tr data-c="${r.代號}"><td class="l"><span class="name">${r.名稱}</span><span class="code">${r.代號}</span></td>
 <td>${num(r.股價)} ${pc(r.日漲跌,1)}</td><td class="l">${r.子題}</td><td class="l">${tag(r.階段)} ${r.細分||""}</td><td>${pc(r["6月超額"])}</td><td>${pc(r.離高點)}</td></tr>`).join("");
 document.querySelectorAll("#ti tbody tr").forEach(e=>e.onclick=()=>openDlg(e.dataset.c));
}

/* ---------- 個股視窗 ---------- */
let charts=[], cdata=null, curR=null, tf="week";   // 預設週 K（比較宏觀）
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function closeDlg(){document.getElementById("dlg").classList.remove("open");charts.forEach(c=>c.remove());charts=[];if(location.hash)history.replaceState(null,"",location.pathname)}
document.addEventListener("keydown",e=>{if(e.key=="Escape")closeDlg()});
document.getElementById("dlg").onclick=e=>{if(e.target.id=="dlg")closeDlg()};
document.querySelectorAll("#rng button").forEach(b=>b.onclick=()=>{tf=b.dataset.tf;document.querySelectorAll("#rng button").forEach(x=>x.classList.toggle("on",x==b));if(cdata)drawCharts(cdata,curR)});

/* 指標計算（週 K、月 K、分時用；日 K 用 Python 算好的） */
function sma(a,n){const o=[];let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(i>=n)s-=a[i-n];o.push(i>=n-1?s/n:null)}return o}
function ema(a,n){const o=[],k=2/(n+1);let p=null;for(const x of a){p=p==null?x:x*k+p*(1-k);o.push(p)}return o}
function rsiW(a,n=14){const o=[null];let up=0,dn=0;for(let i=1;i<a.length;i++){const d=a[i]-a[i-1],u=Math.max(d,0),w=Math.max(-d,0);
 if(i<=n){up+=u/n;dn+=w/n}else{up=(up*(n-1)+u)/n;dn=(dn*(n-1)+w)/n}o.push(i>=n?(dn==0?100:100-100/(1+up/dn)):null)}return o}
function macdOf(c){const f=ema(c,12),s=ema(c,26),dif=c.map((_,i)=>f[i]-s[i]),dea=ema(dif,9);return {dif,dea}}
function vwapDaily(T,h,l,c,v){const o=[];let pv=0,vv=0,day=null;for(let i=0;i<T.length;i++){const dd=Math.floor(T[i]/86400);if(dd!==day){day=dd;pv=0;vv=0}
 pv+=(h[i]+l[i]+c[i])/3*v[i];vv+=v[i];o.push(vv?pv/vv:c[i])}return o}

/* 依週期整理出要畫的資料 */
function frame(d,tf){
 if(tf=="day")return {T:d.t,o:d.o,h:d.h,l:d.l,c:d.c,v:d.v,st:d.st,dif:d.dif,dea:d.dea,rsi:d.rsi,
  ma:[["50日線","#f59e0b",d.ma50],["150日線","#3b82f6",d.ma150]],show:252,intra:false};
 let b;
 if(tf=="week"||tf=="month"){b=tf=="week"?d.w:d.mo;if(!b)return null;
  const ma=tf=="week"?[["10週線","#f59e0b",sma(b.c,10)],["30週線","#3b82f6",sma(b.c,30)]]:[["6月線","#f59e0b",sma(b.c,6)],["12月線","#3b82f6",sma(b.c,12)]];
  return {T:b.t,o:b.o,h:b.h,l:b.l,c:b.c,v:b.v,...macdOf(b.c),rsi:rsiW(b.c),ma,show:tf=="week"?156:120,intra:false}}
 b=d.i;if(!b||!b.t.length)return null;
 let s=0;if(tf=="d1"){const last=Math.floor(b.t[b.t.length-1]/86400);s=b.t.findIndex(x=>Math.floor(x/86400)==last)}
 const sl=a=>a.slice(s), T=sl(b.t), o=sl(b.o), h=sl(b.h), l=sl(b.l), c=sl(b.c), v=sl(b.v);
 return {T,o,h,l,c,v,...macdOf(c),rsi:rsiW(c),ma:[["均價","#f59e0b",vwapDaily(T,h,l,c,v)]],show:T.length,intra:true};
}

async function openDlg(code){
 const r=C.find(x=>x.代號==code), ir=I.find(x=>x.代號==code), sr=D.surge.find(x=>x.代號==code), o=r||ir||sr;
 if(!o)return;
 history.replaceState(null,"","#"+code);
 document.getElementById("dlg").classList.add("open");
 const base=D.base[o.市場]||{}, bt=r&&r.bt;
 document.getElementById("dhead").innerHTML=`<div class="ph"><span style="font-size:20px;font-weight:700">${o.名稱}</span><span class="code">${o.代號}・${o.市場}${r?"・"+r.角色:ir?"・指標股":"・暴衝雷達"}</span>
  <span class="px">${num(o.股價)}</span><span>${pc(o.日漲跌,2)}</span><span class="small">資料日期 ${o.資料日期||""}</span></div>
  ${linkHtml(o.代號,o.市場)}
  <div class="small" style="margin-top:6px">${o.主題||""}${o.子題?"・"+o.子題:""}${sr?`⚡ 近 5 日 ${pc(sr["5日漲幅"])}、量 ${sr.量能倍數.toFixed(1)} 倍・${sr.階段} ${sr.細分||""}　${sr.週期位置||""}`:""}${r&&r.產業已發動?' <span class="hot">● 產業發動</span>':''}</div>`;
 let h="";
 if(r){
  const g=D.guide.find(x=>x.stage==r.階段)||{};
  h+=`<div class="box" style="border-left:5px solid ${colorOf(r.階段)}"><h3>現在在循環的哪裡</h3>
   <div style="font-size:18px;font-weight:700">${r.階段} ${r.細分||""}</div>
   <div class="small" style="margin:2px 0 6px">${g.meaning||""}</div>
   <div class="kv"><span>週期位置</span><b>${r.週期位置||"–"}</b>
   <span>起漲日</span><b>${r.起漲日||"–"}</b>
   <span>建議動作</span><b>${r.操作}</b>
   <span>減碼線（50 日線）</span><b>${r.減碼線?num(r.減碼線)+"（距離 "+pp(r.股價/r.減碼線-1)+"）":"–"}</b>
   <span>出場線</span><b>${r.出場線?num(r.出場線)+"（距離 "+pp(r.股價/r.出場線-1)+"）":"–"}</b>
   <span>RSI(14)</span><b>${r.RSI?.toFixed(0)??"–"}</b></div></div>`;
  if(r.預估勝率!=null){const v=((D.val[o.市場]||{})[r.階段]||{}).w12;
   h+=`<div class="box"><h3>回測模型：這檔的預估（和同階段股票比）</h3>
   <div class="kv"><span>12 個月後上漲機率</span><b>${r.勝率模型有效?pp(r.預估勝率)+'　<span class="small">階段平均 '+pp(r.階段勝率)+'</span>':'<span class="small">模型未通過驗證，僅供參考：階段平均</span> '+pp(r.階段勝率)}</b>
   <span>6 個月後上漲機率</span><b>${r["6月模型有效"]?pp(r.預估6月勝率)+'　<span class="small">階段平均 '+pp(r.階段6月勝率)+'</span>':'<span class="small">未通過驗證：階段平均</span> '+pp(r.階段6月勝率)}</b>
   <span>2 年內腰斬機率</span><b>${pp(r.預估腰斬)}　<span class="small">階段平均 ${pp(r.階段腰斬)}</span></b>
   <span>精選分</span><b>${r.精選分??"–"}</b></div>
   <div class="small" style="margin-top:6px">${r.加分理由?"✔ 加分："+r.加分理由+"<br>":""}${r.扣分理由?"✘ 扣分："+r.扣分理由+"<br>":""}括號＝這個條件讓 12 個月勝率比同階段平均高／低幾個百分點。
   ${v?`<br>模型驗證：2022 年後，預估最高 20% 實際勝率 ${pp(v.top)}、最低 20% ${pp(v.bottom)}。`:""}</div></div>`;}
  h+=`<div class="box"><h3>歷史上同樣狀態的股票，後來怎麼樣</h3>${bt?`
   <div class="small" style="margin-bottom:6px">依據：<b>${bt.key}</b>（${o.市場}全市場 ${bt.n_stocks} 檔、${bt.n.toLocaleString()} 次樣本）</div>
   <div class="kv"><span>12 個月後上漲機率</span><b>${pp(bt.win12)}　<span class="small">平均 ${pp(base.win12)}</span></b>
   <span>12 個月後贏過大盤</span><b>${pp(bt.beat12)}　<span class="small">平均 ${pp(base.beat12)}</span></b>
   <span>12 個月報酬中位數</span><b>${pc(bt.med12)}　<span class="small">平均 ${pc(base.med12)}</span></b>
   <span>2 年內曾漲到 2 倍</span><b>${pp(bt.p2x,1)}　<span class="small">平均 ${pp(base.p2x,1)}</span></b>
   <span>2 年內曾漲到 3 倍</span><b>${pp(bt.p3x,1)}（${bt.p3x_lift?.toFixed(1)}x）　<span class="small">平均 ${pp(base.p3x,1)}</span></b>
   <span>2 年內曾腰斬</span><b>${pp(bt.half24)}　<span class="small">平均 ${pp(base.half24)}</span></b>
   <span>之後 6 個月常見最大跌幅</span><b>${pc(bt.dd6)}</b></div>
   <div class="small" style="margin-top:6px">最大跌幅＝停損要預留的空間：正常波動就會跌這麼多，停損設太近容易被洗出場。</div>`:"尚無回測資料"}</div>`;
  const tr=r.tr;
  if(tr)h+=`<div class="box"><h3>如果照規則操作（發動買進、跌破出場線賣出）</h3>
   <div class="small" style="margin-bottom:6px">${o.市場}過去 10 年、進場條件「${tr.key}」的 ${tr.n.toLocaleString()} 筆交易</div>
   <div class="kv"><span>勝率</span><b>${pp(tr.win)}</b><span>賺的時候平均</span><b>${pc(tr.avg_win)}</b>
   <span>賠的時候平均</span><b>${pc(tr.avg_loss)}</b><span>每筆期望值</span><b>${pc(tr.exp,1)}</b>
   <span>同期間買大盤</span><b>${pc(tr.bexp,1)}</b>
   <span>單筆賺 1 倍以上</span><b>${pp(tr.p100,1)}</b><span>平均持有</span><b>${tr.weeks.toFixed(0)} 週</b></div>
   <div class="small" style="margin-top:6px">${r.階段==D.order[1]?"<b>這檔現在就在發動期＝規則上的進場點。</b>":[D.order[2],D.order[3]].includes(r.階段)?"這檔已經在多頭中，規則上的進場點已過；持有者守住出場線即可。":"這檔現在不在進場點，等它下次進入發動期。"}</div></div>`;
 }
 const ai=D.ai[code], al=AL[code];
 if(al)h=`<div class="box" style="grid-column:1/-1;border-left:5px solid #dc2626"><h3>🚨 ${al.警報}</h3>今日 ${pc(al["1日漲跌"],1)}・5 日 ${pc(al["5日漲跌"])}・量 ${al.量能倍數.toFixed(1)} 倍・150 日線 ${num(al["150日線"])}<div class="small" style="margin-top:4px">${crashHint(al)}</div></div>`+h;
 if(ai)h=`<div class="box ai" style="grid-column:1/-1"><h3>AI 新聞分析（近兩週・${ai.model||"Gemini"}・${ai.n_news} 則新聞）</h3>
  <div><span class="sent ${ai.sentiment}">${ai.sentiment}</span>${ai.new_opportunity?'<span class="newop">★ 可能出現新商機 </span>':''}<b>${ai.summary||""}</b></div>
  ${ai.why_move?`<div style="margin-top:4px">📈 <b>近期漲勢原因：</b>${ai.why_move}</div>`:""}
  <div class="boxes" style="margin:6px 0 0"><div><b class="up">利多</b><ul>${(ai.bull||[]).map(x=>`<li>${x}</li>`).join("")}</ul></div>
  <div><b class="dn">利空／風險</b><ul>${(ai.bear||[]).map(x=>`<li>${x}</li>`).join("")}</ul></div></div>
  <div class="small">${(ai.news||[]).map(n=>`<a href="${n.link}" target="_blank" rel="noopener" style="color:var(--mute)">${n.date}・${n.source}：${n.title}</a>`).join("<br>")}</div>
  <div class="small" style="margin-top:4px">AI 整理僅供參考，可能有錯，重要消息請點新聞原文確認。</div></div>`+h;
 document.getElementById("dboxes").innerHTML=h;
 document.getElementById("dboxes2").innerHTML=r?`<div class="box"><h3>四大支柱分數</h3><div class="kv">
   <span>總分</span><b>${r.總分}</b><span>① 底子 DNA</span><b>${r["①底子"]} / 25</b><span>② 商機催化劑</span><b>${r["②催化劑"]} / 25</b>
   <span>③ 資金與技術</span><b>${r["③資金技術"]} / 25</b><span>④ 進場位置</span><b>${r["④位置"]} / 25</b>
   <span>⚡ 爆發力（1–2 年）</span><b>${r["⚡爆發力"]}</b><span>🏔️ 長跑力（3–5 年）</span><b>${r["🏔️長跑力"]}</b></div></div>
   <div class="box"><h3>基本面與動能</h3><div class="kv">
   <span>營收年增（${o.市場=="台股"?"近 3 個月月營收":"最近一季"}）</span><b>${pc(r.營收YoY)}</b>
   <span>營收加速（比前期）</span><b>${pc(r.營收加速)}</b>
   <span>相對強度（全市場）</span><b>${r.相對強度||"–"}</b><span>6 個月漲幅</span><b>${pc(r["6月漲幅"])}</b>
   <span>6 個月贏大盤</span><b>${pc(r["6月超額"])}</b><span>離一年高點</span><b>${pc(r.離高點)}</b>
   <span>市值</span><b>${r["市值(億美元)"]??"–"} 億美元</b><span>紅旗</span><b class="flag">${r.紅旗||"無"}</b></div></div>`:"";
 charts.forEach(c=>c.remove());charts=[];cdata=null;
 document.getElementById("lg1").textContent="載入中…";
 try{cdata=await (await fetch(`charts/${encodeURIComponent(code)}.json`)).json()}catch(e){document.getElementById("lg1").textContent="K 線資料載入失敗";return}
 drawCharts(cdata,r);
}

function drawCharts(d,r){
 curR=r;charts.forEach(c=>c.remove());charts=[];
 const F=frame(d,tf);
 if(!F){document.getElementById("lg1").textContent=tf.startsWith("d")&&tf!="day"?"這檔沒有分時資料（每天收盤後更新一次）":"沒有資料";document.getElementById("lg2").textContent="";document.getElementById("lg3").textContent="";document.getElementById("ribbon").textContent="";return}
 const LW=LightweightCharts, up=css("--up"), dn=css("--dn"), mute=css("--mute"), line=css("--line");
 const opt=(time,logo)=>({autoSize:true,layout:{background:{color:"transparent"},textColor:mute,fontSize:11,attributionLogo:!!logo},
  grid:{vertLines:{color:line},horzLines:{color:line}},rightPriceScale:{borderColor:line,minimumWidth:72},
  timeScale:{borderColor:line,visible:time,rightOffset:3,timeVisible:F.intra,secondsVisible:false},crosshair:{mode:0},localization:{dateFormat:"yyyy-MM-dd"}});
 const T=F.T, pts=a=>a.map((v,i)=>v==null||isNaN(v)?{time:T[i]}:{time:T[i],value:v});
 // 主圖：（日 K）階段底色 + K 線 + 均線 + 成交量
 const c1=LW.createChart(document.getElementById("c1"),opt(false,true));
 if(F.st){const bg=c1.addHistogramSeries({priceScaleId:"bg",priceLineVisible:false,lastValueVisible:false});
  c1.priceScale("bg").applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
  bg.setData(T.map((t,i)=>({time:t,value:1,color:(F.st[i]>=0?D.colors[F.st[i]]:"#888")+"33"})));}
 const vol=c1.addHistogramSeries({priceScaleId:"vol",priceFormat:{type:"volume"},priceLineVisible:false,lastValueVisible:false});
 c1.priceScale("vol").applyOptions({scaleMargins:{top:0.82,bottom:0},visible:false});
 vol.setData(T.map((t,i)=>({time:t,value:F.v[i],color:(F.c[i]>=F.o[i]?up:dn)+"66"})));
 const k=c1.addCandlestickSeries({upColor:up,downColor:dn,borderUpColor:up,borderDownColor:dn,wickUpColor:up,wickDownColor:dn});
 k.setData(T.map((t,i)=>({time:t,open:F.o[i],high:F.h[i],low:F.l[i],close:F.c[i]})));
 F.ma.forEach(([n,col,vals],j)=>{const s=c1.addLineSeries({color:col,lineWidth:j?2:1.5,priceLineVisible:false,lastValueVisible:false});s.setData(pts(vals))});
 if(tf=="day"&&r&&r.出場線&&r.階段==D.order[0])k.createPriceLine({price:r.出場線,color:dn,lineStyle:2,title:"停損"});
 if(tf=="day"&&r&&r.起漲日&&T.includes(r.起漲日))k.setMarkers([{time:r.起漲日,position:"belowBar",color:colorOf(D.order[1]),shape:"arrowUp",text:"起漲"}]);
 // MACD
 const c2=LW.createChart(document.getElementById("c2"),opt(false));
 const osc=c2.addHistogramSeries({priceLineVisible:false,lastValueVisible:false});
 osc.setData(T.map((t,i)=>{const v=F.dif[i]-F.dea[i];return {time:t,value:+v.toFixed(4),color:v>=0?up:dn}}));
 const dif=c2.addLineSeries({color:"#f59e0b",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});dif.setData(pts(F.dif));
 const dea=c2.addLineSeries({color:"#3b82f6",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});dea.setData(pts(F.dea));
 // RSI
 const c3=LW.createChart(document.getElementById("c3"),opt(true));
 const rsi=c3.addLineSeries({color:"#9333ea",lineWidth:1.5,priceLineVisible:false});rsi.setData(pts(F.rsi));
 rsi.createPriceLine({price:70,color:up,lineStyle:2,axisLabelVisible:false});rsi.createPriceLine({price:30,color:dn,lineStyle:2,axisLabelVisible:false});
 rsi.applyOptions({autoscaleInfoProvider:()=>({priceRange:{minValue:0,maxValue:100}})});
 charts=[c1,c2,c3];
 let lock=false;
 charts.forEach(c=>c.timeScale().subscribeVisibleLogicalRangeChange(rg=>{if(lock||!rg)return;lock=true;charts.forEach(o=>o!==c&&o.timeScale().setVisibleLogicalRange(rg));lock=false}));
 const f=v=>v==null||isNaN(v)?"–":num(v);
 const tlabel=i=>F.intra?new Date(T[i]*1000).toISOString().slice(5,16).replace("T"," "):T[i];
 const legend=i=>{if(i==null||i<0||i>=T.length)i=T.length-1;
  const chg=i>0?F.c[i]/F.c[i-1]-1:null;
  document.getElementById("lg1").innerHTML=`${tlabel(i)}　開 ${f(F.o[i])} 高 ${f(F.h[i])} 低 ${f(F.l[i])} 收 <b>${f(F.c[i])}</b> ${pc(chg,2)}　量 ${F.v[i].toLocaleString()}　`+
   F.ma.map(([n,col,vals])=>`<span style="color:${col}">${n} ${f(vals[i])}</span>`).join("　")+(F.st&&F.st[i]>=0?"　"+tag(D.order[F.st[i]]):"");
  document.getElementById("lg2").innerHTML=`MACD(12,26,9)　<span style="color:#f59e0b">DIF ${F.dif[i]?.toFixed(3)}</span>　<span style="color:#3b82f6">MACD ${F.dea[i]?.toFixed(3)}</span>　柱 ${(F.dif[i]-F.dea[i]).toFixed(3)}`;
  document.getElementById("lg3").innerHTML=`<span style="color:#9333ea">RSI(14) ${F.rsi[i]?.toFixed(1)??"–"}</span>　（70 以上偏熱、30 以下偏冷）`;};
 charts.forEach(c=>c.subscribeCrosshairMove(p=>legend(p&&p.logical!=null?Math.round(p.logical):null)));
 legend(null);
 document.getElementById("ribbon").innerHTML=tf=="day"?"K 線底色＝當時的階段："+D.order.map((s,i)=>`<span><i style="background:${D.colors[i]}"></i>${s}</span>`).join(""):
  F.intra?"5 分鐘 K 線（每天收盤後更新一次，不是即時報價）；黃線＝當日均價":"由日線換算；階段底色只在日 K 顯示";
 const n=T.length;charts.forEach(c=>c.timeScale().setVisibleLogicalRange({from:Math.max(0,n-F.show),to:n+3}));
}

[...new Set(C.map(r=>r.主題))].forEach(t=>ft.add(new Option(t,t)));
[ft,fh].forEach(e=>e.onchange=draw);document.getElementById("q").oninput=draw;
rule();guide();alertsView();picks();draw();ind();
/* ---------- 深色／淺色 ---------- */
const themeBtn=document.getElementById("theme");
const setThemeLabel=()=>themeBtn.textContent=document.documentElement.dataset.theme==="light"?"🌙 深色":"☀️ 淺色";
setThemeLabel();
themeBtn.onclick=()=>{const light=document.documentElement.dataset.theme!=="light";
 if(light)document.documentElement.dataset.theme="light";else delete document.documentElement.dataset.theme;
 try{localStorage.setItem("theme",light?"light":"dark")}catch(e){}
 setThemeLabel();if(cdata&&document.getElementById("dlg").classList.contains("open"))drawCharts(cdata,curR)};
if(location.hash.length>1)openDlg(decodeURIComponent(location.hash.slice(1)));
</script></body></html>"""
