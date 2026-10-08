"""把評分結果輸出成一個單檔 HTML 儀表板（瀏覽器直接打開）。"""
import json
from datetime import datetime

import numpy as np

import config


def _clean(df):
    recs = df.replace({np.nan: None}).to_dict(orient="records")
    return json.dumps(recs, ensure_ascii=False, default=float)


def build(cand, ind, hot_sub, demo=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = TEMPLATE.replace("__CAND__", _clean(cand)).replace("__IND__", _clean(ind)) \
        .replace("__HOT__", json.dumps(hot_sub, ensure_ascii=False)) \
        .replace("__TIME__", now + ("（模擬資料，僅供預覽）" if demo else ""))
    config.OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_HTML.write_text(html, encoding="utf-8")
    return config.OUTPUT_HTML


TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>十倍股追蹤器</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e2dc;--acc:#c2410c;
--hot:#dc2626;--go:#ea580c;--up:#16a34a;--seed:#0d9488;--warn:#a16207;--chip:#f0eee8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d20;--ink:#ececee;--mute:#9a9aa2;
--line:#2c2c31;--chip:#26262b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,"Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0}.sub{color:var(--mute);font-size:12px;margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mute);font-size:12px}
.kpi.on{outline:2px solid var(--acc)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
select,input{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font:inherit}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:auto}
table{border-collapse:collapse;width:100%;min-width:980px}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);font-size:12px;color:var(--mute);cursor:pointer;user-select:none}
td.l,th.l{text-align:left}tr:hover td{background:var(--chip)}
.name{font-weight:600}.code{color:var(--mute);font-size:12px;margin-left:4px}
.tag{display:inline-block;padding:1px 7px;border-radius:99px;background:var(--chip);font-size:12px}
.sbar{display:inline-block;width:70px;height:7px;background:var(--chip);border-radius:9px;vertical-align:middle;margin-right:6px;overflow:hidden}
.sbar i{display:block;height:100%;background:var(--acc)}
.pos{color:var(--up)}.neg{color:var(--hot)}.flag{color:var(--hot);font-size:12px}
.hot{color:var(--go);font-weight:600}
h2{font-size:16px;margin:28px 0 8px}
.note{color:var(--mute);font-size:12px;margin-top:14px;line-height:1.7}
.chips{display:flex;flex-wrap:wrap;gap:6px}
</style></head><body><div class="wrap">
<h1>十倍股追蹤器</h1><div class="sub">更新時間：__TIME__</div>

<div class="grid" id="kpis"></div>
<div class="bar">
 <select id="fm"><option value="">全部市場</option><option>美股</option><option>台股</option></select>
 <select id="ft"><option value="">全部主題</option></select>
 <input id="q" placeholder="搜尋代號或名稱">
 <label><input type="checkbox" id="fh"> 只看產業已發動</label>
</div>
<div class="card"><table id="t"><thead></thead><tbody></tbody></table></div>

<h2>產業啟動訊號（指標股）</h2>
<div class="chips" id="hot"></div>
<div class="card" style="margin-top:10px"><table id="ti" style="min-width:640px"><thead></thead><tbody></tbody></table></div>

<div class="note">
階段：🌱潛伏期＝低檔＋營收開始加速（可小量布局）　🔥發動期＝多頭排列＋突破或帶量、離高點近（主要加碼區）
🚀主升段＝已大漲（續抱、回檔再加）　⚠️過熱＝乖離過大（不追、考慮分批出場）　📈趨勢中＝多頭但沒有新突破<br>
分數是相對排名的參考，不是成功機率。真正的機率要等累積歷史紀錄後回測校準（v3）。這不是買賣建議。
</div>
</div>
<script>
const C=__CAND__, I=__IND__, HOT=__HOT__;
const STAGES=["🔥 發動期","🌱 潛伏期","🚀 主升段","📈 趨勢中","⚠️ 過熱","· 無訊號"];
let fStage="", sortK="總分", sortD=-1;
const pct=v=>v==null?"–":`<span class="${v>=0?'pos':'neg'}">${(v*100).toFixed(0)}%</span>`;
const sb=(v,m)=>`<span class="sbar"><i style="width:${Math.max(0,Math.min(100,v/m*100))}%"></i></span>${v??"–"}`;
function kpis(){const k=document.getElementById("kpis");
 k.innerHTML=`<div class="kpi ${fStage==""?"on":""}" data-s=""><b>${C.length}</b><span>全部候選</span></div>`+
 STAGES.map(s=>`<div class="kpi ${fStage==s?"on":""}" data-s="${s}"><b>${C.filter(r=>r.階段==s).length}</b><span>${s}</span></div>`).join("");
 k.querySelectorAll(".kpi").forEach(e=>e.onclick=()=>{fStage=e.dataset.s;kpis();draw()});}
const cols=[["名稱","l"],["主題","l"],["階段","l"],["總分"],["⚡爆發力"],["🏔️長跑力"],["①底子"],["②催化劑"],["③資金技術"],["④位置"],
 ["營收YoY"],["6月超額"],["離高點"],["50MA乖離"],["市值(億美元)"],["紅旗","l"]];
function draw(){
 const m=fm.value,t=ft.value,q=document.getElementById("q").value.trim().toLowerCase(),h=fh.checked;
 let rows=C.filter(r=>(!m||r.市場==m)&&(!t||r.主題==t)&&(!fStage||r.階段==fStage)&&(!h||r.產業已發動)
   &&(!q||(r.代號+r.名稱).toLowerCase().includes(q)));
 rows.sort((a,b)=>{const x=a[sortK],y=b[sortK];return (x==null)-(y==null)||(x>y?1:x<y?-1:0)*sortD});
 document.querySelector("#t thead").innerHTML="<tr>"+cols.map(([c,a])=>`<th class="${a||''}" data-k="${c}">${c}${sortK==c?(sortD<0?" ▼":" ▲"):""}</th>`).join("")+"</tr>";
 document.querySelectorAll("#t th").forEach(e=>e.onclick=()=>{const k=e.dataset.k;sortD=sortK==k?-sortD:-1;sortK=k;draw()});
 document.querySelector("#t tbody").innerHTML=rows.map(r=>`<tr>
 <td class="l"><span class="name">${r.名稱}</span><span class="code">${r.代號}</span><br><span class="code">${r.市場}・${r.角色}</span></td>
 <td class="l">${r.主題}<br><span class="code">${r.子題}${r.產業已發動?' <span class="hot">● 產業發動</span>':''}</span></td>
 <td class="l"><span class="tag">${r.階段}</span></td>
 <td><b>${sb(r.總分,100)}</b></td><td>${sb(r["⚡爆發力"],100)}</td><td>${sb(r["🏔️長跑力"],100)}</td>
 <td>${r["①底子"]}</td><td>${r["②催化劑"]}</td><td>${r["③資金技術"]}</td><td>${r["④位置"]}</td>
 <td>${pct(r.營收YoY)}</td><td>${pct(r["6月超額"])}</td><td>${pct(r.離高點)}</td><td>${pct(r["50MA乖離"])}</td>
 <td>${r["市值(億美元)"]??"–"}</td><td class="l flag">${r.紅旗||""}</td></tr>`).join("");}
function ind(){
 document.getElementById("hot").innerHTML=HOT.length?HOT.map(s=>`<span class="tag hot">● ${s}</span>`).join(""):'<span class="code">目前沒有子產業被指標股點火</span>';
 document.querySelector("#ti thead").innerHTML="<tr><th class='l'>指標股</th><th class='l'>子題</th><th class='l'>階段</th><th>6月超額</th><th>離高點</th></tr>";
 document.querySelector("#ti tbody").innerHTML=[...I].sort((a,b)=>b["6月超額"]-a["6月超額"]).map(r=>`<tr><td class="l"><span class="name">${r.名稱}</span><span class="code">${r.代號}</span></td>
 <td class="l">${r.子題}</td><td class="l"><span class="tag">${r.階段}</span></td><td>${pct(r["6月超額"])}</td><td>${pct(r.離高點)}</td></tr>`).join("");}
[...new Set(C.map(r=>r.主題))].forEach(t=>ft.add(new Option(t,t)));
[fm,ft,fh].forEach(e=>e.onchange=draw);document.getElementById("q").oninput=draw;
kpis();draw();ind();
</script></body></html>"""
