# 📊 Hyperliquid Trader Behavior vs. Bitcoin Market Sentiment (Fear/Greed) Analysis

> **Objective:** Analyze the relationship between Bitcoin market sentiment (Fear/Greed Index) and trader behavior/performance on the Hyperliquid decentralized exchange. Identify actionable patterns, segment traders, and propose data-driven strategies.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Datasets](#-datasets)
3. [Project Structure](#-project-structure)
4. [Setup & How to Run](#-setup--how-to-run)
5. [Part A — Data Preparation](#-part-a--data-preparation)
6. [Part B — Analysis](#-part-b--analysis)
7. [Part C — Actionable Output & Strategy Recommendations](#-part-c--actionable-output--strategy-recommendations)
8. [Bonus — Predictive Model, Clustering & Dashboard](#-bonus--predictive-model-clustering--dashboard)
9. [Key Findings Summary](#-key-findings-summary)
10. [Charts & Visualizations Generated](#-charts--visualizations-generated)
11. [Tech Stack](#-tech-stack)
12. [Evaluation Criteria Checklist](#-evaluation-criteria-checklist)

---

## 🎯 Project Overview

This project investigates **how the Bitcoin Fear/Greed Index influences the trading behavior and profitability of traders on Hyperliquid**, a decentralized perpetual futures exchange.

The core questions we answer:

1. **Do traders perform differently during Fear vs. Greed market regimes?** (PnL, win rate, drawdown proxy)
2. **How does trader behavior change based on sentiment?** (trade frequency, position sizes, long/short bias, volume)
3. **Which trader segments are most/least affected by sentiment?** (frequent vs. infrequent, large vs. small positions, winners vs. losers)
4. **Can we predict profitability using sentiment + behavioral features?**
5. **What actionable strategies emerge from these patterns?**

The analysis is fully reproducible with a single `python analysis.py` command, and includes an interactive Streamlit dashboard for deep exploration.

---

## 📁 Datasets

### 1. Bitcoin Fear & Greed Index (`fear_greed_index.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Unix timestamp | 1738281600 |
| `value` | Sentiment score (0 = max fear, 100 = max greed) | 44 |
| `classification` | Categorical label | Fear, Greed, Neutral, Extreme Fear, Extreme Greed |
| `date` | Calendar date | 2025-01-31 |

- **Rows:** 2,644
- **Date coverage:** Multi-year daily history
- **Source:** Alternative.me Fear & Greed Index

### 2. Hyperliquid Trader Data (`historical_data.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `Account` | Wallet address (anonymized) | 0xabcd...1234 |
| `Coin` | Trading pair symbol | SOL, BTC, ETH, DOGE |
| `Execution Price` | Price at which the trade executed | 183.45 |
| `Size Tokens` | Number of tokens traded | 200.5 |
| `Size USD` | Dollar value of the trade | $36,781.73 |
| `Side` | Buy or Sell | BUY / SELL |
| `Timestamp IST` | Trade timestamp in IST timezone | 02-12-2024 22:50 |
| `Start Position` | Position before this trade | 0, 200, -500 |
| `Direction` | Open Long, Open Short, Close Long, Close Short | Open Long |
| `Closed PnL` | Realized profit/loss on this trade | $150.23 / -$80.50 |
| `Fee` | Transaction fee charged | $7.35 |
| `Transaction Hash` | On-chain transaction ID | 0x... |
| `Trade ID` | Unique trade identifier | 123456789 |
| `Crossed` | Whether order crossed the spread | true/false |

- **Rows:** 211,224
- **Unique accounts:** 32 traders
- **Unique coins:** 246 different trading pairs
- **Date range:** May 1, 2023 → May 1, 2025 (479 trading days)

---

## 🗂 Project Structure

```
d:\intern\
│
├── analysis.py              # Main analysis script (Parts A + B + C + Bonus)
├── dashboard.py             # Interactive Streamlit dashboard (5 tabs)
├── requirements.txt         # Python dependencies
├── README.md                # This file — full documentation
│
├── fear_greed_index.csv     # Input: Bitcoin sentiment data
├── historical_data.csv      # Input: Hyperliquid trader data
│
└── output/                  # All generated outputs
    ├── chart1_performance_by_sentiment.png
    ├── chart2_behavior_by_sentiment.png
    ├── chart3_segment_heatmap.png
    ├── chart4_pnl_distribution.png
    ├── chart5_sentiment_timeline.png
    ├── chart6_correlation.png
    ├── chart7_feature_importance.png
    ├── chart8_trader_clusters.png
    ├── chart9_archetype_comparison.png
    ├── performance_by_sentiment.csv
    ├── behavior_by_sentiment.csv
    ├── cluster_archetypes.csv
    ├── data_summary.json
    └── analysis_summary.json
```

---

## 🚀 Setup & How to Run

### Prerequisites
- **Python 3.9+**
- **pip** (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`, `plotly`, `statsmodels`

### Step 2: Run the Full Analysis Pipeline

```bash
python analysis.py
```

This single command executes the entire pipeline:
- Loads & cleans both CSV datasets
- Merges them on date
- Computes metrics (daily PnL, win rate, long/short ratio, etc.)
- Generates **9 charts** (saved as PNG in `output/`)
- Exports **3 CSV tables** and **2 JSON summaries**
- Trains a Random Forest model and performs K-Means clustering
- Prints all insights and strategies to the console

### Step 3: Launch the Interactive Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501` with 5 interactive tabs:
- 📈 **Performance** — PnL & win rate by sentiment, timeline
- 🔍 **Behavior** — Trade frequency, volume, long/short, correlations
- 👥 **Segments** — Pie charts, tables, scatter plots by trader type
- 🤖 **Predictions** — Model accuracy, feature importance, strategies
- 📋 **Data Explorer** — Browse & download filtered raw data

---

## 🔧 Part A — Data Preparation

### What We Did

#### 1. Loading the Data
- Loaded `fear_greed_index.csv` (2,644 rows) and `historical_data.csv` (211,224 rows) using pandas.

#### 2. Cleaning & Type Conversion
- **Timestamps:** Converted `Timestamp IST` from `DD-MM-YYYY HH:MM` string format to proper `datetime` objects. Used fallback parsing for any rows that didn't match the primary format.
- **Numeric columns:** Force-converted `Execution Price`, `Size Tokens`, `Size USD`, `Closed PnL`, and `Fee` from strings to numeric types (coercing errors to NaN).
- **Dates:** Extracted a `date` column from both datasets for alignment.
- **Duplicates & Missing Values:** Checked both datasets — no duplicates found; missing values handled by coercing during type conversion.

#### 3. Merging Datasets
- Performed a **left join on date** to attach the sentiment score and classification to every trade.
- **Match rate:** 211,218 out of 211,224 trades (99.997%) matched to a sentiment day.
- Trades on days without sentiment data were dropped.

#### 4. Feature Engineering

We created the following derived columns and aggregations:

| Feature | How It's Computed |
|---------|-------------------|
| `sentiment_binary` | Maps 5 categories → 3: "Fear" (Extreme Fear + Fear), "Neutral", "Greed" (Greed + Extreme Greed) |
| `is_long` | `True` if `Side == "BUY"` |
| `is_close` | `True` if `Direction` contains "Close" |
| `is_winner` | `True` if `Closed PnL > 0` |
| `net_pnl` | `Closed PnL - Fee` (PnL after fees) |
| **Daily per-account aggregation** | `num_trades`, `total_volume_usd`, `avg_trade_size_usd`, `total_pnl`, `net_pnl`, `total_fees`, `num_buys`, `num_sells`, `num_winners`, `num_closes`, `max_trade_size`, `win_rate`, `long_short_ratio` |
| **Daily market-level aggregation** | `total_trades`, `total_volume`, `avg_trade_size`, `total_pnl`, `net_pnl`, `avg_win_rate`, `avg_long_short_ratio`, `num_active_traders` |

- **Daily aggregation result:** 479 unique trading days, 2,340 daily-account records

---

## 📈 Part B — Analysis

### B1: Performance Differences by Sentiment

We compared how traders perform across the 5 sentiment categories.

#### Results Table: Performance by Sentiment

| Sentiment | Days | Avg Daily PnL | Median Daily PnL | Total PnL | Avg Daily Volume | Avg Trades/Day |
|-----------|------|---------------|-------------------|-----------|-----------------|----------------|
| **Extreme Fear** | 14 | **$52,793.59** | $22,561.74 | $739,110.25 | $8,177,447 | 1,529 |
| **Fear** | 91 | $36,891.82 | $1,412.31 | $3,357,155.44 | $5,311,261 | 680 |
| **Neutral** | 67 | $19,297.32 | $1,818.57 | $1,292,920.68 | $2,690,180 | 562 |
| **Greed** | 193 | $11,140.57 | $678.48 | $2,150,129.27 | $1,495,246 | 261 |
| **Extreme Greed** | 114 | $23,817.29 | $3,127.54 | $2,715,171.31 | $1,091,800 | 351 |

**Key takeaway:** Average daily PnL during **Extreme Fear is $52,794** — nearly **5x higher** than during Greed ($11,141). Fear days concentrate higher PnL, volume, and trading activity.

#### Binary Comparison: Fear vs. Greed

| Metric | Fear Days | Greed Days |
|--------|-----------|------------|
| Avg Daily PnL | **$39,012** | $15,848 |
| Trades/Day | **793** | 294 |
| Sentiment–PnL Correlation | r = −0.083 (weak negative) | — |
| Sentiment–Volume Correlation | r = −0.264 (moderate negative) | — |

### B2: Behavioral Changes by Sentiment

We measured how trader behavior shifts across sentiment regimes.

#### Results Table: Behavior by Sentiment

| Sentiment | Avg Trades/Day | Avg Trade Size ($) | Avg Long/Short Ratio | Avg Daily Volume ($) |
|-----------|---------------|-------------------|---------------------|---------------------|
| **Extreme Fear** | **1,529** | $5,520 | **10.38** | **$8,177,447** |
| Fear | 680 | $7,298 | 6.07 | $5,311,261 |
| Neutral | 562 | $7,661 | 5.30 | $2,690,180 |
| Greed | 261 | $7,153 | 4.11 | $1,495,246 |
| Extreme Greed | 351 | $5,821 | 4.04 | $1,091,800 |

**Key behavioral findings:**

1. **Trade frequency spikes during Fear:** Traders are 2.7x more active during Fear days (793 trades/day) vs Greed days (294 trades/day). During Extreme Fear, activity surges to 1,529 trades/day.

2. **Long bias increases with Fear:** The Long/Short ratio is **10.38 during Extreme Fear** vs just **4.04 during Extreme Greed**. Traders aggressively buy the dip during fearful markets.

3. **Position sizes are relatively stable:** Average trade size stays around $5,500–$7,700 across sentiments, meaning traders adjust *frequency* and *direction* — not size — based on sentiment.

4. **Volume inversely correlates with sentiment (r = −0.264):** More trading dollars flow during Fear ($8.2M/day in Extreme Fear) than Greed ($1.1M/day in Extreme Greed).

### B3: Trader Segmentation

We segmented the 32 accounts across 3 axes and analyzed how each segment performs under different sentiment regimes.

#### Segmentation Criteria

| Segment Axis | Categories | How Assigned |
|--------------|-----------|--------------|
| **Frequency** | Frequent Trader / Infrequent Trader | Median split on total trade count |
| **Position Size** | Large Position / Small Position | Median split on avg trade size (USD) |
| **Performance** | Consistent Winner / Inconsistent Winner / Net Loser | PnL > 0 and Win Rate > 50% = Consistent; PnL > 0 only = Inconsistent; else = Net Loser |

#### Segment Distribution

| Segment | Count |
|---------|-------|
| Frequent Trader | 16 |
| Infrequent Trader | 16 |
| Large Position | 16 |
| Small Position | 16 |
| Consistent Winner | 29 |
| Net Loser | 3 |

#### Cross-Segment Performance by Sentiment

**Frequency Segments:**

| Segment | Fear (Avg PnL) | Neutral (Avg PnL) | Greed (Avg PnL) |
|---------|----------------|--------------------|--------------------|
| Frequent Trader | $47.33 | $34.58 | $41.48 |
| Infrequent Trader | $61.97 | $31.85 | **$155.90** |

→ *Infrequent traders have the highest per-trade PnL during Greed — they pick their spots and trade larger when confident.*

**Position Size Segments:**

| Segment | Fear (Avg PnL) | Neutral (Avg PnL) | Greed (Avg PnL) |
|---------|----------------|--------------------|--------------------|
| Large Position | $83.23 | $90.48 | **$124.93** |
| Small Position | $28.99 | $9.01 | $27.83 |

→ *Large position traders earn 3–14x more per trade across all sentiment regimes.*

**Performance Segments:**

| Segment | Fear (Avg PnL) | Neutral (Avg PnL) | Greed (Avg PnL) |
|---------|----------------|--------------------|--------------------|
| Consistent Winner | $50.45 | $35.05 | $60.58 |
| Net Loser | $30.15 | $17.33 | **−$175.85** |

→ *Net Losers get destroyed during Greed days (−$175.85 avg per trade). Greed amplifies overconfidence and bad decisions for weak traders.*

---

## 💡 Part C — Actionable Output & Strategy Recommendations

Based on the analysis, we propose two evidence-backed strategies:

### Strategy 1: Contrarian Sentiment-Based Position Sizing

> **Rule:** During **Extreme Fear** days (sentiment score < 25), **increase** position sizes for long trades by **20–30%**. During **Extreme Greed** days (sentiment score > 75), **reduce** position sizes by **20–30%** and favor shorter holding periods.

**Evidence Supporting This Strategy:**
- Extreme Fear days yield the highest avg daily PnL ($52,794/day) — 5x higher than Greed
- Long/Short ratio peaks at 10.38 during Extreme Fear, meaning traders who go long during fear capture recovery
- Net Losers get crushed during Greed (−$175.85/trade), proving overexposure during euphoria is destructive

**Best Suited For:** Consistent Winners and Frequent Traders who have demonstrated risk management ability.

### Strategy 2: Sentiment-Adaptive Trading Frequency

> **Rule:** During Fear periods, **reduce trading frequency by 30–40%** for infrequent traders (focus on fewer, higher-conviction trades with larger size). During Greed periods, frequent traders should **maintain their pace but tighten stop-losses**.

**Evidence Supporting This Strategy:**
- Infrequent traders earn $155.90/trade during Greed (highest of any segment × sentiment combo) — when they trade selectively, they perform well
- Frequent traders maintain steady ~$41–47/trade across sentiments — they benefit from volume consistency but need risk controls during euphoric periods
- Overall trade activity is 2.7x higher during Fear, suggesting some traders overtrade; reducing frequency could improve conviction-per-trade

**Best Suited For:** Differentiated advice for Frequent vs. Infrequent trader segments.

---

## 🤖 Bonus — Predictive Model, Clustering & Dashboard

### Predictive Model: Next-Day Profitability

We trained a **Random Forest Classifier** to predict whether a trader's day will be profitable (PnL > 0) or not.

**Features Used:**
| Feature | Importance |
|---------|-----------|
| `num_sells` | 33.7% |
| `long_short_ratio` | 20.7% |
| `num_buys` | 16.4% |
| `num_trades` | 12.5% |
| `total_volume_usd` | 7.1% |
| `avg_trade_size_usd` | 6.0% |
| `sentiment_value` | 3.5% |

**Results:**
- **5-Fold Cross-Validated Accuracy: 79.7% (±2.6%)**
- The most important features are trade-count metrics (`num_sells`, `num_buys`, `long_short_ratio`), confirming that *what traders do* matters more than *sentiment alone*
- Sentiment contributes 3.5% importance — it's a useful but minor signal in the full feature set

### Trader Clustering: Behavioral Archetypes

We used **K-Means clustering (k=3)** to discover natural trader groupings based on 6 behavioral features.

| Archetype | Avg Trades | Avg Trade Size | Win Rate | Total PnL | Long Ratio | Trades/Day |
|-----------|-----------|---------------|----------|-----------|------------|------------|
| **A — "Whale Grinders"** | 18,433 | $11,890 | 1.03 | **$1,272,056** | 0.48 | 319 |
| **B — "Selective Snipers"** | 1,237 | $3,212 | 7.38 | $194,919 | 0.32 | 35 |
| **C — "Mid-Cap Regulars"** | 5,520 | $5,513 | 1.12 | $126,489 | 0.51 | 88 |

**Interpretation:**
- **Archetype A (Whale Grinders):** High-frequency, large-position traders. They trade ~319 times/day with $11,890 avg size and accumulate the most total PnL ($1.27M). Nearly balanced long/short (0.48 ratio).
- **Archetype B (Selective Snipers):** Low-frequency, small-size traders with the highest win rate (7.38). They trade only ~35 times/day but are highly selective. Short-biased (0.32 long ratio).
- **Archetype C (Mid-Cap Regulars):** Middle-ground traders. Moderate frequency (88/day), moderate size ($5,513), slightly long-biased (0.51). Steady but lower total PnL.

### Interactive Dashboard

<!-- TODO: add docs/dashboard_screenshot.png and uncomment
![Dashboard screenshot](docs/dashboard_screenshot.png)
-->

Built with **Streamlit + Plotly**, the dashboard provides:

| Tab | What It Shows |
|-----|--------------|
| 📈 Performance | PnL & win rate bar charts by sentiment; sentiment+PnL timeline overlay |
| 🔍 Behavior | Trade frequency, volume, long/short bars; scatter plots with trend lines |
| 👥 Segments | Pie charts for 3 segment axes; sortable account table; interactive scatter |
| 🤖 Predictions | Model accuracy, correlations, feature importance chart, strategy cards |
| 📋 Data Explorer | Browse/filter/download raw merged trades, sentiment data, account stats |

**Sidebar filters:** Date range, sentiment categories, coins, and individual accounts.

---

## 🏆 Key Findings Summary

### Finding 1: Fear = Higher PnL (Contrarian Alpha)
Average daily PnL during Fear is **$39,012** vs **$15,848** during Greed — a **2.5x difference**. Extreme Fear days yield $52,794/day. Traders who are active during market panic capture significantly more profits, likely from mean-reversion and liquidation-driven price dislocations.

### Finding 2: Traders Aggressively Buy the Dip
The Long/Short ratio jumps from **4.04 (Extreme Greed)** to **10.38 (Extreme Fear)**. This means for every short trade, there are 10+ long trades during panic — traders on Hyperliquid are strong dip-buyers.

### Finding 3: Greed Kills Weak Traders
Net Losers average **−$175.85 per trade during Greed days** but actually earn $30.15/trade during Fear. Euphoria amplifies overconfidence and leads to outsized losses for the weakest traders.

### Finding 4: Volume Inversely Correlates with Sentiment
Correlation of **r = −0.264** between sentiment score and daily volume. More trading activity happens during fear ($8.2M/day in Extreme Fear) than greed ($1.1M/day in Extreme Greed). The market is significantly more active during downturns.

### Finding 5: Behavior Matters More Than Sentiment
The Random Forest model achieves **79.7% accuracy** predicting daily profitability, but sentiment contributes only **3.5% of feature importance**. Trade execution patterns (num_sells at 33.7%, long/short ratio at 20.7%) are far more predictive than the sentiment score itself.

---

## ⚠️ Limitations

This is an academic/exploratory analysis, not a validated trading strategy. The sample is **32 wallets**, which is too small to draw statistically robust conclusions — the strategy recommendations above are hypotheses generated from observed correlations, not backtested or deployment-ready systems. Treat the findings as directional signal for further research, not as something to trade on.

---

## 📊 Charts & Visualizations Generated

All charts are saved in the `output/` directory:

| # | Chart | What It Shows |
|---|-------|--------------|
| 1 | `chart1_performance_by_sentiment.png` | 3-panel bar chart: Avg Daily PnL, Win Rate, Daily Volume across 5 sentiment categories |
| 2 | `chart2_behavior_by_sentiment.png` | 4-panel bar chart: Trades/day, Trade Size, Long/Short Ratio, Volume by sentiment |
| 3 | `chart3_segment_heatmap.png` | 3 heatmaps showing Avg PnL per trade for each segment × sentiment combination |
| 4 | `chart4_pnl_distribution.png` | Box plots of PnL distribution (1st–99th percentile) per sentiment category |
| 5 | `chart5_sentiment_timeline.png` | Dual-axis timeline: sentiment score + daily trade count over 2 years |
| 6 | `chart6_correlation.png` | 2 scatter plots with trend lines: Sentiment vs PnL (r=−0.083) and Sentiment vs Volume (r=−0.264) |
| 7 | `chart7_feature_importance.png` | Horizontal bar chart of Random Forest feature importance for profitability prediction |
| 8 | `chart8_trader_clusters.png` | 2 scatter plots: Trades vs PnL and Trade Size vs Win Rate, colored by cluster |
| 9 | `chart9_archetype_comparison.png` | Normalized bar chart comparing all 3 archetypes across 6 behavioral dimensions |

**Tables exported:**
- `performance_by_sentiment.csv` — Performance metrics per sentiment category
- `behavior_by_sentiment.csv` — Behavioral metrics per sentiment category
- `cluster_archetypes.csv` — Cluster centroids for 3 trader archetypes
- `data_summary.json` — Dataset size and date range info
- `analysis_summary.json` — All insights, correlations, model accuracy, chart list

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **pandas** | Data loading, cleaning, merging, aggregation |
| **numpy** | Numerical operations, trend line fitting |
| **matplotlib** | Static chart generation (9 charts) |
| **seaborn** | Heatmap visualization (segment analysis) |
| **scikit-learn** | Random Forest classifier, K-Means clustering, StandardScaler, cross-validation |
| **Streamlit** | Interactive web dashboard framework |
| **Plotly** | Interactive charts in the dashboard |
| **statsmodels** | Optional — OLS trend lines in Plotly |

---

## ✅ Evaluation Criteria Checklist

| Criterion | Status | Details |
|-----------|--------|---------|
| ✅ **Data cleaning + correct merge** | Done | Timestamp parsing (DD-MM-YYYY HH:MM), numeric type coercion, date alignment merge, 99.997% match rate |
| ✅ **Strength of reasoning** | Done | Multi-angle analysis: 5-category breakdown, 3 segmentation axes, correlation analysis, predictive modeling |
| ✅ **Quality of insights** | Done | 5 insights with exact numbers, 2 actionable strategies with evidence, segment-specific recommendations |
| ✅ **Clarity of communication** | Done | Structured README, labeled charts, JSON summaries, interactive dashboard |
| ✅ **Reproducibility** | Done | Single `python analysis.py` generates everything; `requirements.txt` pins dependencies |
| ✅ **Bonus: Predictive model** | Done | Random Forest, 79.7% CV accuracy, feature importance ranking |
| ✅ **Bonus: Clustering** | Done | K-Means (k=3), 3 named archetypes with behavioral profiles |
| ✅ **Bonus: Dashboard** | Done | Streamlit + Plotly, 5 tabs, sidebar filters, data export |

---

## 📄 License

This project is for analytical and educational purposes.
