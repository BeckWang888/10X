"""AI 新聞分析（只分析今日精選的股票）：Google 新聞 RSS 抓近兩週標題 → Gemini 整理利多／利空。

需要環境變數 GEMINI_API_KEY（GitHub 上是 repo secret）。沒設定就跳過，不影響其他功能。
模型：環境變數 GEMINI_MODEL 可指定；沒指定就自動挑目前最新的 Flash-Lite（整理新聞夠用、約 1 秒一檔）。
搜尋：GEMINI_SEARCH=1 才讓 Gemini 自己上網搜尋（免費額度常不含，遇到額度不足會自動改成不搜尋）。
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote

import requests

import config

API = "https://generativelanguage.googleapis.com/v1beta"
CACHE = config.CACHE_DIR / "news"


def _key():
    return os.environ.get("GEMINI_API_KEY")


def _headers():
    return {"x-goog-api-key": _key(), "Content-Type": "application/json"}


def pick_model():
    if os.environ.get("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    r = requests.get(f"{API}/models", headers=_headers(), params={"pageSize": 200}, timeout=30)
    r.raise_for_status()
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    flash = [n for n in names if re.fullmatch(r"gemini-\d+(\.\d+)?-flash-lite", n)]   # 正式版 Flash-Lite
    if not flash:
        flash = [n for n in names if re.fullmatch(r"gemini-\d+(\.\d+)?-flash", n)] or names
    ver = lambda n: [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0]
    return sorted(flash, key=ver)[-1]


def headlines(name, code, market, days=14, limit=15):
    if market == "台股":
        q, loc = f"{name} {code}", "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    else:
        q, loc = f"{code} {name} stock", "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={quote(q)}+when:{days}d&{loc}"
    try:
        root = ET.fromstring(requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).content)
    except Exception:
        return []
    out = []
    for it in root.iter("item"):
        out.append({"title": (it.findtext("title") or "").strip(), "date": (it.findtext("pubDate") or "")[:16],
                    "source": (it.findtext("source") or "").strip(), "link": (it.findtext("link") or "").strip()})
        if len(out) >= limit:
            break
    return out


PROMPT = """你是謹慎的股票研究助理，請用繁體中文回答。
股票：{name}（{code}，{market}），主題：{theme}
目前狀態：{stage}；股價 {price}；近 5 日漲幅 {ret5}；{extra}
以下是最近兩週的新聞標題（可能不完整，可再用搜尋補充，但只根據可查證的事實）：
{news}

請只輸出 JSON，格式：
{{"summary": "一句話結論（30 字內）",
  "bull": ["利多 1（具體事實，20–40 字）", "..."],
  "bear": ["可能的利空或風險 1", "..."],
  "why_move": "如果近期大漲或暴衝，最可能的原因（沒有就留空字串）",
  "new_opportunity": true 或 false（是否出現可能讓公司營收獲利長期大幅成長的新商機，例如拿到大客戶／大訂單、新產品開始放量、打進新市場；被收購、短線題材炒作、單純營收創新高都不算）,
  "sentiment": "偏多" 或 "中性" 或 "偏空"}}
利多、利空各 1–4 點；沒有可靠資訊就寫「近期無重大消息」，不要編造數字。"""


def analyze(row, model, extra=""):
    """row 需要 名稱、代號、市場、主題、階段、股價、5日漲幅（可無）"""
    if not _key():
        return None
    day = datetime.now().strftime("%Y%m%d")
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "_alert" if "下跌" in extra else ""          # 急跌警報的分析（問下跌原因）和精選的分開存
    p = CACHE / f"{row['代號']}_{day}{tag}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    news = headlines(row["名稱"], row["代號"], row["市場"])
    ret5 = row.get("5日漲幅")
    text = PROMPT.format(name=row["名稱"], code=row["代號"], market=row["市場"], theme=row.get("主題") or "—",
                         stage=row.get("階段") or "—", price=row.get("股價"),
                         ret5=f"{ret5:+.0%}" if isinstance(ret5, float) else "—", extra=extra,
                         news="\n".join(f"- {n['date']} {n['source']}：{n['title']}" for n in news) or "（沒有抓到新聞）")
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.2, "thinkingConfig": {"thinkingLevel": "low"}}}
    if os.environ.get("GEMINI_SEARCH") == "1":
        body["tools"] = [{"google_search": {}}]
    res = None
    for attempt in range(4):
        try:
            r = requests.post(f"{API}/models/{model}:generateContent", headers=_headers(), json=body, timeout=60)
            if r.status_code in (400, 429) and "tools" in body:   # 不支援搜尋或搜尋額度用完 → 改成不搜尋
                body.pop("tools")
                continue
            if r.status_code == 400 and "thinkingConfig" in body["generationConfig"]:
                body["generationConfig"].pop("thinkingConfig")
                continue
            if r.status_code in (429, 500, 503):
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            raw = "".join(x.get("text", "") for x in parts)
            m = re.search(r"\{.*\}", raw, re.S)
            res = json.loads(m.group(0)) if m else None
            break
        except Exception as e:
            print(f"\n  AI 分析 {row['代號']} 失敗：{e}")
            time.sleep(5)
    if res:
        res["n_news"] = len(news)
        res["news"] = news[:5]
        res["model"] = model
        p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def analyze_many(rows):
    """rows：list of dict。回傳 {代號: 分析結果}"""
    if not _key():
        print("  （沒有設定 GEMINI_API_KEY，跳過 AI 新聞分析）")
        return {}
    try:
        model = pick_model()
    except Exception as e:
        print(f"  Gemini 連線失敗：{e}")
        return {}
    print(f"  AI 新聞分析：{len(rows)} 檔（模型 {model}）")
    from concurrent.futures import ThreadPoolExecutor
    out = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r, a in zip(rows, ex.map(lambda r: analyze(r, model, r.get("_extra", "")), rows)):
            if a:
                out[r["代號"]] = a
    print(f"  AI 分析完成 {len(out)}/{len(rows)} 檔")
    return out
