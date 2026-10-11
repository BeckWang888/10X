"""抓長期研究用的價格（還原股息）。在 GitHub Actions 上執行，存到 lrs/results/prices_more.csv.gz。"""
import os

import pandas as pd
import yfinance as yf

TICKERS = [
    # 大盤與類股（含還原股息的長歷史基金）
    "VFINX", "^SP500TR", "^GSPC", "SPY", "VOO", "VTI", "VTSMX",
    "^NDX", "QQQ", "^SOX", "SOXX", "SMH",
    # 槓桿 ETF
    "SSO", "UPRO", "QLD", "TQQQ", "USD", "SOXL",
    # 債券、黃金、現金
    "VUSTX", "VFITX", "TLT", "IEF", "SHY", "GLD", "GC=F", "^IRX",
    # 因子（小型價值、動能、品質）與全球
    "DFSVX", "VISVX", "MTUM", "QUAL", "VEU", "VGTSX",
]

d = yf.download(TICKERS, start="1970-01-01", auto_adjust=True, progress=False)["Close"]
d.index = pd.to_datetime(d.index).tz_localize(None)
out = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out, exist_ok=True)
d.to_csv(os.path.join(out, "prices_more.csv.gz"))
print(d.apply(lambda s: s.first_valid_index()))
print(d.tail(3).T)
