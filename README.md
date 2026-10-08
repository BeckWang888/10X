# 十倍股追蹤器

每個交易日由 GitHub Actions 自動更新，結果發佈在 GitHub Pages。

## 第一次設定
1. Settings → Pages → Source 選 **GitHub Actions**
2. （選用）Settings → Secrets and variables → Actions → New secret：`FINMIND_TOKEN`（finmindtrade.com 免費註冊取得，台股月營收抓得更快）
3. Actions → 「更新十倍股追蹤器」→ Run workflow，跑完就有網址

## 微調
- `watchlist.xlsx`：保留=N 不追蹤；用途 候選＝評分、指標＝產業發動訊號
- `scoring.py`：評分權重與階段判斷
- `config.py`：基準指數、快取天數等

## 本機測試
`pip install -r requirements.txt` → `python run.py --demo` → 打開 `site/index.html`

## 檔案
- `data/history/`：每次的分數紀錄（之後回測校準機率用，Actions 會自動存）
- 資料來源：Yahoo Finance（yfinance）、FinMind，全部免費

分數是相對排名，不是成功機率，也不是買賣建議。
