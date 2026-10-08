# 十倍股追蹤器

每個交易日由 GitHub Actions 自動更新，結果發佈在 GitHub Pages。

## 怎麼看
1. **階段**：股票現在在漲跌循環的哪個位置（🌱潛伏 → 🔥發動 → 🚀主升 前/中/後段 → ⚠️過熱 → 🟠轉弱 → 🔻下跌）
2. **歷史勝率**：過去 10 年全市場在「同樣狀態」時，12 個月後上漲的比例、2 年內漲 3 倍／腰斬的比例
3. **進出場規則**：🔥發動買進、收盤跌破出場線（150 日線）賣出；跌破 50 日線先減碼
4. **分數**：同階段內排序用。點任一檔看 K 線（底色＝當時階段）、MACD、RSI

## 第一次設定
1. Settings → Pages → Source 選 **GitHub Actions**
2. （選用）Settings → Secrets and variables → Actions → New secret：`FINMIND_TOKEN`（finmindtrade.com 免費註冊取得，台股月營收抓得更快）
3. Secret `SEC_EMAIL`：SEC EDGAR 要求的聯絡 email（美股歷史財報，`sec.yml` 每月更新）
4. Secret `GEMINI_API_KEY`：AI 新聞分析（今日精選與暴衝股的利多／利空）；沒設定就跳過
5. Actions → 「更新十倍股追蹤器」→ Run workflow，跑完就有網址

## 微調
- `watchlist.xlsx`：保留=N 不追蹤；用途 候選＝評分、指標＝產業發動訊號
- `stages.py`：階段規則（儀表板與回測共用）
- `scoring.py`：評分權重、建議動作
- `config.py`：基準指數、快取天數等

## 本機測試
`pip install -r requirements.txt` → `python run.py --demo` → 打開 `site/index.html`
回測：`python backtest.py --quick`（快速）或 `python backtest.py`（完整，約 30–60 分鐘；每月 2 日 Actions 也會自動跑）

## 檔案
- `data/history/`：每次的分數紀錄
- `data/backtest/stage_stats.json`：回測結果（儀表板讀這個）；`tw_revenue.csv.gz`：台股歷史月營收
- `data/history/picks_*.json`：每天的今日精選名單（算連續上榜天數）
- `data/backtest/us_fundamentals.csv.gz`：SEC 美股歷史財報
- 資料來源：Yahoo Finance（yfinance）、FinMind、公開資訊觀測站、Nasdaq Trader 股票清單、SEC EDGAR、Google 新聞，全部免費；AI 分析用 Gemini（自己的 API key）

勝率是歷史統計，不是對個股的預測，也不是買賣建議。
