"""追蹤器設定。數字都可以自己調。"""
from pathlib import Path

ROOT = Path(__file__).parent
WATCHLIST = ROOT / "watchlist.xlsx"
CACHE_DIR = ROOT / "data" / "cache"
HISTORY_DIR = ROOT / "data" / "history"
OUTPUT_HTML = ROOT / "site" / "index.html"   # GitHub Pages 發佈這個資料夾

# 比較基準（算相對強度用）
BENCHMARK = {"美股": "SPY", "台股": "0050.TW"}

# 台幣換美元（只用來把台股市值換成美元比大小，粗估即可）
TWD_PER_USD = 32.0

# 基本面快取幾天（財報一季才更新一次，不用每天抓）
FUNDAMENTAL_CACHE_DAYS = 7

# FinMind token（台股月營收）。不填也能用，只是速度限制比較嚴。
# 到 https://finmindtrade.com 免費註冊後，把 token 設成環境變數 FINMIND_TOKEN
FINMIND_TOKEN_ENV = "FINMIND_TOKEN"
