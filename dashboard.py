"""把評分結果輸出成儀表板：site/index.html（表格＋說明）＋ site/charts/*.json（每檔的 K 線資料，點開才載入）。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
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
    }


def build(cand, ind, hot_sub, techs, prices, sym_of, bt, demo=False):
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
            (chart_dir / f"{code}.json").write_text(json.dumps(chart_json(t, px), separators=(",", ":")),
                                                    encoding="utf-8")

    trades = {mk: m.get("trades") for mk, m in bt.get("markets", {}).items() if m.get("trades")}
    data = {"trades": trades, "cand": rows, "ind": _clean(ind), "hot": hot_sub, "guide": guide, "base": base,
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
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e2dc;--acc:#c2410c;
--up:#dc2626;--dn:#16a34a;--chip:#f0eee8;--good:#fef3c7}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d20;--ink:#ececee;--mute:#9a9aa2;
--line:#2c2c31;--chip:#26262b;--up:#f87171;--dn:#4ade80;--good:#3b2f12}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,"Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1320px;margin:0 auto;padding:20px 16px 60px}
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
table{border-collapse:collapse;width:100%;min-width:1180px}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;vertical-align:top}
th{position:sticky;top:0;background:var(--card);font-size:12px;color:var(--mute);cursor:pointer;user-select:none;z-index:1}
td.l,th.l{text-align:left}tbody tr{cursor:pointer}tbody tr:hover td{background:var(--chip)}
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
<h1>十倍股追蹤器</h1>
<div class="sub" id="sub"></div>

<details class="guide" open><summary>怎麼看這張表（第一次請看）</summary>
<div class="steps">
 <div class="step"><b>① 看階段</b>：這檔股票現在在漲跌循環的哪個位置（下方 7 張卡，依循環順序排）。</div>
 <div class="step"><b>② 看歷史勝率</b>：過去 10 年全市場股票在「同樣狀態」時，12 個月後上漲的比例。比全市場平均高才有優勢。</div>
 <div class="step"><b>③ 看出場線</b>：持有中的股票，收盤跌破出場線（150 日線）就是規則上的出場訊號；跌破減碼線（50 日線）先減碼。</div>
 <div class="step"><b>④ 分數用來排序</b>：同一個階段裡，總分高的基本面＋動能比較好，優先研究。點任一列看 K 線、MACD、RSI。</div>
</div>
<div class="rule" id="rule"></div>
<div class="cycle" id="cycle"></div>
<div class="small" id="btnote"></div>
</details>

<div class="bar">
 <select id="fm"><option value="">全部市場</option><option>美股</option><option>台股</option></select>
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
  <div class="rng" id="rng"><button data-n="126">6 個月</button><button data-n="252" class="on">1 年</button><button data-n="520">2 年</button></div>
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
  <b>🔥 發動（進入多頭）那天買進 → 收盤跌破出場線（150 日線，轉弱）就賣出</b>。持有中跌破 50 日線可先減碼。<br>
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

const cols=[["名稱","l"],["股價"],["階段","l"],["操作","l"],["出場線"],["歷史勝率"],["3倍／腰斬"],["相對強度"],["營收YoY"],["總分"],["⚡爆發力"],["🏔️長跑力"],["RSI"],["市值(億美元)"],["紅旗","l"]];
const val=(r,k)=>k=="歷史勝率"?(r.bt?r.bt.win12:null):k=="3倍／腰斬"?(r.bt?r.bt.p3x:null):k=="出場線"?(r.出場線?r.股價/r.出場線-1:null):k=="名稱"?r.代號:r[k];
function draw(){
 const m=fm.value,t=ft.value,q=document.getElementById("q").value.trim().toLowerCase(),h=fh.checked;
 let rows=C.filter(r=>(!m||r.市場==m)&&(!t||r.主題==t)&&(!fStage||r.階段==fStage)&&(!h||r.產業已發動)
   &&(!q||(r.代號+r.名稱).toLowerCase().includes(q)));
 rows.sort((a,b)=>{const x=val(a,sortK),y=val(b,sortK);return (x==null)-(y==null)||(x>y?1:x<y?-1:0)*sortD});
 document.querySelector("#t thead").innerHTML="<tr>"+cols.map(([c,a])=>`<th class="${a||''}" data-k="${c}">${c}${sortK==c?(sortD<0?" ▼":" ▲"):""}</th>`).join("")+"</tr>";
 document.querySelectorAll("#t th").forEach(e=>e.onclick=()=>{const k=e.dataset.k;sortD=sortK==k?-sortD:-1;sortK=k;draw()});
 document.querySelector("#t tbody").innerHTML=rows.map(r=>{
  const base=D.base[r.市場]||{}, bt=r.bt;
  const good=bt&&base.win12!=null&&bt.win12-base.win12>=0.05;
  const stale=latest[r.市場]&&r.資料日期<latest[r.市場];
  const dist=r.出場線?r.股價/r.出場線-1:null;
  return `<tr data-c="${r.代號}">
 <td class="l"><span class="name">${r.名稱}</span><span class="code">${r.代號}</span><br><span class="small">${r.市場}・${r.主題}・${r.子題}</span>${r.產業已發動?' <span class="hot small">● 產業發動</span>':''}</td>
 <td><b>${num(r.股價)}</b><br>${pc(r.日漲跌,1)} <span class="small ${stale?'stale':''}">${r.資料日期.slice(5)}</span></td>
 <td class="l">${tag(r.階段)}${r.細分?` <b>${r.細分}</b>`:""}<br><span class="small">${r.週期位置||""}</span></td>
 <td class="l">${r.操作}</td>
 <td>${r.出場線?num(r.出場線):"–"}<br><span class="small">${dist!=null?"距離 "+pp(dist):""}</span></td>
 <td><span class="${good?'good':''}">${bt?pp(bt.win12):"–"}</span><br><span class="small">平均 ${pp(base.win12)}</span></td>
 <td>${bt?`<span class="up">${pp(bt.p3x)}</span> / <span class="dn">${pp(bt.half24)}</span>`:"–"}<br><span class="small">平均 ${pp(base.p3x)} / ${pp(base.half24)}</span></td>
 <td>${r.相對強度||"–"}<br><span class="small">6月 ${pc(r["6月漲幅"])}</span></td>
 <td>${pc(r.營收YoY)}<br><span class="small">${r.營收狀態||(r.營收加速!=null?(r.營收加速>0?"加速":"減速"):"")}</span></td>
 <td><b>${sb(r.總分,100)}</b></td><td>${sb(r["⚡爆發力"],100)}</td><td>${sb(r["🏔️長跑力"],100)}</td>
 <td>${r.RSI!=null?r.RSI.toFixed(0):"–"}</td><td>${r["市值(億美元)"]??"–"}</td><td class="l flag">${r.紅旗||""}</td></tr>`}).join("");
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
let charts=[], cdata=null, curN=252;
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function closeDlg(){document.getElementById("dlg").classList.remove("open");charts.forEach(c=>c.remove());charts=[];if(location.hash)history.replaceState(null,"",location.pathname)}
document.addEventListener("keydown",e=>{if(e.key=="Escape")closeDlg()});
document.getElementById("dlg").onclick=e=>{if(e.target.id=="dlg")closeDlg()};
document.querySelectorAll("#rng button").forEach(b=>b.onclick=()=>{curN=+b.dataset.n;document.querySelectorAll("#rng button").forEach(x=>x.classList.toggle("on",x==b));setRange()});
function setRange(){if(!cdata)return;const n=cdata.t.length;charts.forEach(c=>c.timeScale().setVisibleLogicalRange({from:Math.max(0,n-curN),to:n+3}))}

async function openDlg(code){
 const r=C.find(x=>x.代號==code), ir=I.find(x=>x.代號==code), o=r||ir;
 if(!o)return;
 history.replaceState(null,"","#"+code);
 document.getElementById("dlg").classList.add("open");
 const base=D.base[o.市場]||{}, bt=r&&r.bt;
 document.getElementById("dhead").innerHTML=`<div class="ph"><span style="font-size:20px;font-weight:700">${o.名稱}</span><span class="code">${o.代號}・${o.市場}${r?"・"+r.角色:"・指標股"}</span>
  <span class="px">${num(o.股價)}</span><span>${pc(o.日漲跌,2)}</span><span class="small">資料日期 ${r?r.資料日期:""}</span></div>
  <div class="small">${o.主題}・${o.子題}${r&&r.產業已發動?' <span class="hot">● 產業發動</span>':''}</div>`;
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
 const LW=LightweightCharts, up=css("--up"), dn=css("--dn"), mute=css("--mute"), line=css("--line");
 const opt=(time,logo)=>({autoSize:true,layout:{background:{color:"transparent"},textColor:mute,fontSize:11,attributionLogo:!!logo},
  grid:{vertLines:{color:line},horzLines:{color:line}},rightPriceScale:{borderColor:line,minimumWidth:72},
  timeScale:{borderColor:line,visible:time,rightOffset:3},crosshair:{mode:0},localization:{dateFormat:"yyyy-MM-dd"}});
 const T=d.t, pts=a=>a.map((v,i)=>v==null?{time:T[i]}:{time:T[i],value:v});
 // 主圖：階段底色 + K 線 + 均線 + 成交量
 const c1=LW.createChart(document.getElementById("c1"),opt(false,true));
 const bg=c1.addHistogramSeries({priceScaleId:"bg",priceLineVisible:false,lastValueVisible:false});
 c1.priceScale("bg").applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
 bg.setData(T.map((t,i)=>({time:t,value:1,color:(d.st[i]>=0?D.colors[d.st[i]]:"#888")+"33"})));
 const vol=c1.addHistogramSeries({priceScaleId:"vol",priceFormat:{type:"volume"},priceLineVisible:false,lastValueVisible:false});
 c1.priceScale("vol").applyOptions({scaleMargins:{top:0.82,bottom:0},visible:false});
 vol.setData(T.map((t,i)=>({time:t,value:d.v[i],color:(d.c[i]>=d.o[i]?up:dn)+"66"})));
 const k=c1.addCandlestickSeries({upColor:up,downColor:dn,borderUpColor:up,borderDownColor:dn,wickUpColor:up,wickDownColor:dn});
 k.setData(T.map((t,i)=>({time:t,open:d.o[i],high:d.h[i],low:d.l[i],close:d.c[i]})));
 const m50=c1.addLineSeries({color:"#f59e0b",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});m50.setData(pts(d.ma50));
 const m150=c1.addLineSeries({color:"#3b82f6",lineWidth:2,priceLineVisible:false,lastValueVisible:false});m150.setData(pts(d.ma150));
 if(r&&r.出場線&&r.階段==D.order[0])k.createPriceLine({price:r.出場線,color:dn,lineStyle:2,title:"停損"});
 if(r&&r.起漲日&&T.includes(r.起漲日))k.setMarkers([{time:r.起漲日,position:"belowBar",color:colorOf(D.order[1]),shape:"arrowUp",text:"起漲"}]);
 // MACD
 const c2=LW.createChart(document.getElementById("c2"),opt(false));
 const osc=c2.addHistogramSeries({priceLineVisible:false,lastValueVisible:false});
 osc.setData(T.map((t,i)=>{const v=d.dif[i]-d.dea[i];return {time:t,value:+v.toFixed(4),color:v>=0?up:dn}}));
 const dif=c2.addLineSeries({color:"#f59e0b",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});dif.setData(pts(d.dif));
 const dea=c2.addLineSeries({color:"#3b82f6",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});dea.setData(pts(d.dea));
 // RSI
 const c3=LW.createChart(document.getElementById("c3"),opt(true));
 const rsi=c3.addLineSeries({color:"#9333ea",lineWidth:1.5,priceLineVisible:false});rsi.setData(pts(d.rsi));
 rsi.createPriceLine({price:70,color:up,lineStyle:2,axisLabelVisible:false});rsi.createPriceLine({price:30,color:dn,lineStyle:2,axisLabelVisible:false});
 c3.priceScale("right").applyOptions({autoScale:false});
 rsi.applyOptions({autoscaleInfoProvider:()=>({priceRange:{minValue:0,maxValue:100}})});
 charts=[c1,c2,c3];
 let lock=false;
 charts.forEach(c=>c.timeScale().subscribeVisibleLogicalRangeChange(rg=>{if(lock||!rg)return;lock=true;charts.forEach(o=>o!==c&&o.timeScale().setVisibleLogicalRange(rg));lock=false}));
 const f=v=>v==null?"–":num(v);
 const legend=i=>{if(i==null||i<0||i>=T.length)i=T.length-1;
  const chg=i>0?d.c[i]/d.c[i-1]-1:null;
  document.getElementById("lg1").innerHTML=`${T[i]}　開 ${f(d.o[i])} 高 ${f(d.h[i])} 低 ${f(d.l[i])} 收 <b>${f(d.c[i])}</b> ${pc(chg,2)}　量 ${d.v[i].toLocaleString()}　<span style="color:#f59e0b">50日線 ${f(d.ma50[i])}</span>　<span style="color:#3b82f6">150日線 ${f(d.ma150[i])}</span>　${d.st[i]>=0?tag(D.order[d.st[i]]):""}`;
  document.getElementById("lg2").innerHTML=`MACD(12,26,9)　<span style="color:#f59e0b">DIF ${d.dif[i]?.toFixed(3)}</span>　<span style="color:#3b82f6">MACD ${d.dea[i]?.toFixed(3)}</span>　柱 ${(d.dif[i]-d.dea[i]).toFixed(3)}`;
  document.getElementById("lg3").innerHTML=`<span style="color:#9333ea">RSI(14) ${d.rsi[i]?.toFixed(1)??"–"}</span>　（70 以上偏熱、30 以下偏冷）`;};
 charts.forEach(c=>c.subscribeCrosshairMove(p=>legend(p&&p.logical!=null?Math.round(p.logical):null)));
 legend(null);
 document.getElementById("ribbon").innerHTML="K 線底色＝當時的階段："+D.order.map((s,i)=>`<span><i style="background:${D.colors[i]}"></i>${s}</span>`).join("");
 setRange();
}

[...new Set(C.map(r=>r.主題))].forEach(t=>ft.add(new Option(t,t)));
[fm,ft,fh].forEach(e=>e.onchange=draw);document.getElementById("q").oninput=draw;
rule();guide();draw();ind();
if(location.hash.length>1)openDlg(decodeURIComponent(location.hash.slice(1)));
</script></body></html>"""
