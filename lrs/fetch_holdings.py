"""抓持倉體檢用的最新收盤價（台股先試 .TW 再試 .TWO）。存到 lrs/results/holdings_prices.json。"""
import json
import os

import yfinance as yf

TW = ["0050", "0052", "00631L", "00662", "00663L", "00685L", "00708L", "00757", "00830", "00892", "00941",
      "00947", "009800", "009801", "009816", "00990A", "00995A", "6940"]
US = ["AMDC", "DRAM", "GGLL", "GOOG", "IBB", "INTC", "MAGX", "MU", "MUU", "PLTR", "QQQ", "SKUU", "SNXX", "SOXL",
      "SOXL.L", "SOXQ", "SOXX", "TSLA", "USD", "VT", "GC=F", "TWD=X", "GBPUSD=X", "^TWII", "^GSPC"]


def last(sym):
    try:
        h = yf.Ticker(sym).history(period="10d", auto_adjust=False)
        h = h["Close"].dropna()
        if len(h):
            return float(h.iloc[-1]), str(h.index[-1].date())
    except Exception as e:  # noqa: BLE001
        print(sym, e)
    return None


out = {}
for s in TW:
    for suf in (".TW", ".TWO"):
        r = last(s + suf)
        if r:
            out[s] = {"px": r[0], "date": r[1], "yahoo": s + suf}
            break
for s in US:
    r = last(s)
    if r:
        out[s] = {"px": r[0], "date": r[1], "yahoo": s}
p = os.path.join(os.path.dirname(__file__), "results", "holdings_prices.json")
with open(p, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
