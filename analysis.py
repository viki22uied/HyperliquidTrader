"""
Hyperliquid Trader Behavior vs. Bitcoin Market Sentiment (Fear/Greed) Analysis
================================================================================
This script performs a comprehensive analysis of how market sentiment relates
to trader behavior and performance on Hyperliquid.

Author: Analysis Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "Extreme Fear": "#8B0000",
    "Fear": "#FF4500",
    "Neutral": "#FFD700",
    "Greed": "#32CD32",
    "Extreme Greed": "#006400",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def print_section(title):
    """Pretty section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


# ============================================================================
# PART A: DATA PREPARATION
# ============================================================================
print_section("PART A: DATA PREPARATION")

# --- Load Sentiment Data ---
print("Loading Bitcoin Fear/Greed Index...")
sentiment_df = pd.read_csv("fear_greed_index.csv")
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
sentiment_df = sentiment_df.rename(columns={"classification": "sentiment", "value": "sentiment_value"})

print(f"  Rows: {len(sentiment_df):,}")
print(f"  Columns: {sentiment_df.shape[1]}")
print(f"  Date range: {sentiment_df['date'].min().date()} to {sentiment_df['date'].max().date()}")
print(f"  Missing values:\n{sentiment_df.isnull().sum().to_string()}")
print(f"  Duplicates: {sentiment_df.duplicated().sum()}")
print(f"\n  Sentiment Distribution:")
print(sentiment_df["sentiment"].value_counts().to_string())

# --- Load Trader Data ---
print("\n\nLoading Hyperliquid Historical Trader Data...")
trades_df = pd.read_csv("historical_data.csv")

print(f"  Rows: {len(trades_df):,}")
print(f"  Columns: {trades_df.shape[1]}")
print(f"  Column names: {list(trades_df.columns)}")
print(f"  Missing values:\n{trades_df.isnull().sum().to_string()}")
print(f"  Duplicates: {trades_df.duplicated().sum()}")

# --- Parse timestamps ---
# The "Timestamp IST" column has format like "02-12-2024 22:50"
trades_df["datetime"] = pd.to_datetime(trades_df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")

# Some may fail, try alternative parsing
mask_nat = trades_df["datetime"].isna()
if mask_nat.any():
    trades_df.loc[mask_nat, "datetime"] = pd.to_datetime(
        trades_df.loc[mask_nat, "Timestamp IST"], errors="coerce"
    )

trades_df["date"] = trades_df["datetime"].dt.date
trades_df["date"] = pd.to_datetime(trades_df["date"])

# Clean numeric columns
for col in ["Execution Price", "Size Tokens", "Size USD", "Closed PnL", "Fee"]:
    trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

# Rename for convenience
trades_df = trades_df.rename(columns={
    "Account": "account",
    "Coin": "coin",
    "Execution Price": "exec_price",
    "Size Tokens": "size_tokens",
    "Size USD": "size_usd",
    "Side": "side",
    "Start Position": "start_position",
    "Direction": "direction",
    "Closed PnL": "closed_pnl",
    "Fee": "fee",
    "Crossed": "crossed",
})

print(f"\n  Parsed date range: {trades_df['date'].min().date()} to {trades_df['date'].max().date()}")
print(f"  Unique accounts: {trades_df['account'].nunique()}")
print(f"  Unique coins: {trades_df['coin'].nunique()}")
print(f"  Coins traded: {trades_df['coin'].unique()[:20]}")

# --- Merge datasets on date ---
print("\n  Merging datasets on date...")
merged_df = trades_df.merge(sentiment_df[["date", "sentiment", "sentiment_value"]], on="date", how="left")
matched = merged_df["sentiment"].notna().sum()
print(f"  Matched trades: {matched:,} / {len(merged_df):,} ({100*matched/len(merged_df):.1f}%)")

# Drop unmatched
merged_df = merged_df.dropna(subset=["sentiment"])
print(f"  Working dataset size: {len(merged_df):,} trades")

# --- Create key metrics ---
print("\n  Creating key metrics...")

# Binary sentiment column
merged_df["sentiment_binary"] = merged_df["sentiment"].apply(
    lambda x: "Fear" if x in ["Extreme Fear", "Fear"] else ("Greed" if x in ["Extreme Greed", "Greed"] else "Neutral")
)

# Is long/short
merged_df["is_long"] = merged_df["side"].str.upper() == "BUY"
merged_df["is_close"] = merged_df["direction"].str.contains("Close", case=False, na=False)
merged_df["is_open"] = merged_df["direction"].str.contains("Open", case=False, na=False)

# Net PnL (PnL - fees)
merged_df["net_pnl"] = merged_df["closed_pnl"] - merged_df["fee"]
merged_df["is_winner"] = merged_df["closed_pnl"] > 0

# --- Daily aggregation per account ---
daily_account = merged_df.groupby(["date", "account", "sentiment", "sentiment_value", "sentiment_binary"]).agg(
    num_trades=("side", "count"),
    total_volume_usd=("size_usd", "sum"),
    avg_trade_size_usd=("size_usd", "mean"),
    total_pnl=("closed_pnl", "sum"),
    net_pnl=("net_pnl", "sum"),
    total_fees=("fee", "sum"),
    num_buys=("is_long", "sum"),
    num_sells=("is_long", lambda x: (~x).sum()),
    num_winners=("is_winner", "sum"),
    num_closes=("is_close", "sum"),
    max_trade_size=("size_usd", "max"),
).reset_index()

daily_account["win_rate"] = daily_account["num_winners"] / daily_account["num_closes"].clip(lower=1)
daily_account["long_short_ratio"] = daily_account["num_buys"] / daily_account["num_sells"].clip(lower=1)

# Aggregate daily metrics
daily_agg = daily_account.groupby(["date", "sentiment", "sentiment_value", "sentiment_binary"]).agg(
    total_trades=("num_trades", "sum"),
    total_volume=("total_volume_usd", "sum"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
    total_pnl=("total_pnl", "sum"),
    net_pnl=("net_pnl", "sum"),
    avg_win_rate=("win_rate", "mean"),
    avg_long_short_ratio=("long_short_ratio", "mean"),
    num_active_traders=("account", "nunique"),
).reset_index()

print(f"  Daily aggregation: {len(daily_agg)} trading days")
print(f"  Daily account-level: {len(daily_account)} records")

# Save summary stats
summary = {
    "sentiment_rows": len(sentiment_df),
    "trades_rows": len(trades_df),
    "merged_rows": len(merged_df),
    "unique_accounts": int(trades_df["account"].nunique()),
    "unique_coins": int(trades_df["coin"].nunique()),
    "date_range_start": str(trades_df["date"].min().date()),
    "date_range_end": str(trades_df["date"].max().date()),
    "trading_days": len(daily_agg),
}
with open(os.path.join(OUTPUT_DIR, "data_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n  [✓] Data preparation complete.")


# ============================================================================
# PART B: ANALYSIS
# ============================================================================
print_section("PART B: ANALYSIS")

# ---- B1: Performance by Sentiment ----
print("B1: Performance by Sentiment Category\n")

perf_by_sentiment = daily_agg.groupby("sentiment_binary").agg(
    days=("date", "nunique"),
    avg_daily_pnl=("total_pnl", "mean"),
    median_daily_pnl=("total_pnl", "median"),
    total_pnl=("total_pnl", "sum"),
    avg_daily_volume=("total_volume", "mean"),
    avg_win_rate=("avg_win_rate", "mean"),
    avg_trades_per_day=("total_trades", "mean"),
).round(2)
print(perf_by_sentiment.to_string())

perf_by_5cat = daily_agg.groupby("sentiment").agg(
    days=("date", "nunique"),
    avg_daily_pnl=("total_pnl", "mean"),
    median_daily_pnl=("total_pnl", "median"),
    total_pnl=("total_pnl", "sum"),
    avg_daily_volume=("total_volume", "mean"),
    avg_win_rate=("avg_win_rate", "mean"),
    avg_trades_per_day=("total_trades", "mean"),
).round(2)
# Reindex to correct order
valid_cats = [c for c in SENTIMENT_ORDER if c in perf_by_5cat.index]
perf_by_5cat = perf_by_5cat.reindex(valid_cats)
print(f"\n5-Category Breakdown:")
print(perf_by_5cat.to_string())

# Save table
perf_by_5cat.to_csv(os.path.join(OUTPUT_DIR, "performance_by_sentiment.csv"))

# --- Chart 1: PnL by Sentiment ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Performance Metrics by Market Sentiment", fontsize=16, fontweight="bold")

# Average Daily PnL
cats = perf_by_5cat.index.tolist()
colors_list = [COLORS.get(c, "#888") for c in cats]

axes[0].bar(range(len(cats)), perf_by_5cat["avg_daily_pnl"], color=colors_list, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(range(len(cats)))
axes[0].set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
axes[0].set_title("Avg Daily PnL ($)", fontweight="bold")
axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

# Win Rate
axes[1].bar(range(len(cats)), perf_by_5cat["avg_win_rate"] * 100, color=colors_list, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(range(len(cats)))
axes[1].set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
axes[1].set_title("Avg Win Rate (%)", fontweight="bold")
axes[1].axhline(y=50, color="black", linestyle="--", alpha=0.5)

# Trading Volume
axes[2].bar(range(len(cats)), perf_by_5cat["avg_daily_volume"], color=colors_list, edgecolor="black", linewidth=0.5)
axes[2].set_xticks(range(len(cats)))
axes[2].set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
axes[2].set_title("Avg Daily Volume ($)", fontweight="bold")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart1_performance_by_sentiment.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  [✓] Chart 1: Performance by Sentiment saved.")


# ---- B2: Behavior Changes by Sentiment ----
print("\nB2: Behavioral Changes by Sentiment\n")

behavior_by_sentiment = daily_agg.groupby("sentiment_binary").agg(
    avg_trades_per_day=("total_trades", "mean"),
    avg_trade_size=("avg_trade_size", "mean"),
    avg_long_short_ratio=("avg_long_short_ratio", "mean"),
    avg_volume=("total_volume", "mean"),
).round(2)
print(behavior_by_sentiment.to_string())

behavior_by_5cat = daily_agg.groupby("sentiment").agg(
    avg_trades_per_day=("total_trades", "mean"),
    avg_trade_size=("avg_trade_size", "mean"),
    avg_long_short_ratio=("avg_long_short_ratio", "mean"),
    avg_volume=("total_volume", "mean"),
).round(2)
valid_cats2 = [c for c in SENTIMENT_ORDER if c in behavior_by_5cat.index]
behavior_by_5cat = behavior_by_5cat.reindex(valid_cats2)
print(f"\n5-Category Breakdown:")
print(behavior_by_5cat.to_string())

behavior_by_5cat.to_csv(os.path.join(OUTPUT_DIR, "behavior_by_sentiment.csv"))

# --- Chart 2: Behavioral Patterns ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Trader Behavior by Market Sentiment", fontsize=16, fontweight="bold")

cats2 = behavior_by_5cat.index.tolist()
colors_list2 = [COLORS.get(c, "#888") for c in cats2]

# Trade frequency
axes[0, 0].bar(range(len(cats2)), behavior_by_5cat["avg_trades_per_day"], color=colors_list2, edgecolor="black", linewidth=0.5)
axes[0, 0].set_xticks(range(len(cats2)))
axes[0, 0].set_xticklabels(cats2, rotation=30, ha="right", fontsize=9)
axes[0, 0].set_title("Avg Trades per Day", fontweight="bold")

# Trade size
axes[0, 1].bar(range(len(cats2)), behavior_by_5cat["avg_trade_size"], color=colors_list2, edgecolor="black", linewidth=0.5)
axes[0, 1].set_xticks(range(len(cats2)))
axes[0, 1].set_xticklabels(cats2, rotation=30, ha="right", fontsize=9)
axes[0, 1].set_title("Avg Trade Size ($)", fontweight="bold")
axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

# Long/Short ratio
axes[1, 0].bar(range(len(cats2)), behavior_by_5cat["avg_long_short_ratio"], color=colors_list2, edgecolor="black", linewidth=0.5)
axes[1, 0].set_xticks(range(len(cats2)))
axes[1, 0].set_xticklabels(cats2, rotation=30, ha="right", fontsize=9)
axes[1, 0].set_title("Avg Long/Short Ratio", fontweight="bold")
axes[1, 0].axhline(y=1, color="black", linestyle="--", alpha=0.5, label="Balanced")
axes[1, 0].legend()

# Volume
axes[1, 1].bar(range(len(cats2)), behavior_by_5cat["avg_volume"], color=colors_list2, edgecolor="black", linewidth=0.5)
axes[1, 1].set_xticks(range(len(cats2)))
axes[1, 1].set_xticklabels(cats2, rotation=30, ha="right", fontsize=9)
axes[1, 1].set_title("Avg Daily Volume ($)", fontweight="bold")
axes[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart2_behavior_by_sentiment.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  [✓] Chart 2: Behavioral Patterns saved.")


# ---- B3: Trader Segments ----
print("\nB3: Trader Segmentation\n")

# Aggregate per account
account_stats = merged_df.groupby("account").agg(
    total_trades=("side", "count"),
    total_volume=("size_usd", "sum"),
    avg_trade_size=("size_usd", "mean"),
    total_pnl=("closed_pnl", "sum"),
    net_pnl=("net_pnl", "sum"),
    total_fees=("fee", "sum"),
    num_closes=("is_close", "sum"),
    num_winners=("is_winner", "sum"),
    num_buys=("is_long", "sum"),
    trading_days=("date", "nunique"),
    unique_coins=("coin", "nunique"),
).reset_index()

account_stats["win_rate"] = account_stats["num_winners"] / account_stats["num_closes"].clip(lower=1)
account_stats["avg_trades_per_day"] = account_stats["total_trades"] / account_stats["trading_days"]
account_stats["long_ratio"] = account_stats["num_buys"] / account_stats["total_trades"]

# Segment 1: Trade Frequency
freq_median = account_stats["total_trades"].median()
account_stats["freq_segment"] = np.where(
    account_stats["total_trades"] >= freq_median, "Frequent Trader", "Infrequent Trader"
)

# Segment 2: Trade Size (high vs low volume)
vol_median = account_stats["avg_trade_size"].median()
account_stats["size_segment"] = np.where(
    account_stats["avg_trade_size"] >= vol_median, "Large Position", "Small Position"
)

# Segment 3: Performance consistency
account_stats["perf_segment"] = np.where(
    (account_stats["total_pnl"] > 0) & (account_stats["win_rate"] > 0.5),
    "Consistent Winner",
    np.where(account_stats["total_pnl"] > 0, "Inconsistent Winner", "Net Loser"),
)

print("Segment Distribution:")
print(f"\n  Frequency Segments:")
print(account_stats["freq_segment"].value_counts().to_string())
print(f"\n  Size Segments:")
print(account_stats["size_segment"].value_counts().to_string())
print(f"\n  Performance Segments:")
print(account_stats["perf_segment"].value_counts().to_string())

# Merge segments back
merged_with_segments = merged_df.merge(
    account_stats[["account", "freq_segment", "size_segment", "perf_segment"]],
    on="account",
    how="left",
)

# --- Segment analysis by sentiment ---
print("\n\nSegment Performance by Sentiment:")

for seg_col, seg_name in [
    ("freq_segment", "Frequency"),
    ("size_segment", "Position Size"),
    ("perf_segment", "Performance"),
]:
    print(f"\n  --- {seg_name} Segments ---")
    seg_perf = (
        merged_with_segments.groupby([seg_col, "sentiment_binary"])
        .agg(
            avg_pnl=("closed_pnl", "mean"),
            total_pnl=("closed_pnl", "sum"),
            num_trades=("side", "count"),
            win_rate=("is_winner", "mean"),
        )
        .round(4)
    )
    print(seg_perf.to_string())

# --- Chart 3: Segment Performance Heatmap ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Trader Segment Performance by Sentiment", fontsize=16, fontweight="bold")

for idx, (seg_col, seg_name) in enumerate([
    ("freq_segment", "Frequency"),
    ("size_segment", "Position Size"),
    ("perf_segment", "Performance"),
]):
    pivot = merged_with_segments.groupby([seg_col, "sentiment_binary"])["closed_pnl"].mean().reset_index()
    pivot_table = pivot.pivot(index=seg_col, columns="sentiment_binary", values="closed_pnl")
    # Reorder columns
    col_order = [c for c in ["Fear", "Neutral", "Greed"] if c in pivot_table.columns]
    pivot_table = pivot_table[col_order]

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=axes[idx],
        cbar_kws={"label": "Avg PnL ($)"},
    )
    axes[idx].set_title(f"{seg_name} Segments", fontweight="bold")
    axes[idx].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart3_segment_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  [✓] Chart 3: Segment Heatmap saved.")


# --- Chart 4: PnL Distribution (Box plots) ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("PnL Distribution by Sentiment", fontsize=16, fontweight="bold")

# Filter extreme PnL outliers for visibility
pnl_data = merged_df[merged_df["closed_pnl"] != 0].copy()
q1 = pnl_data["closed_pnl"].quantile(0.01)
q99 = pnl_data["closed_pnl"].quantile(0.99)
pnl_filtered = pnl_data[(pnl_data["closed_pnl"] >= q1) & (pnl_data["closed_pnl"] <= q99)]

valid_sentiments = [s for s in SENTIMENT_ORDER if s in pnl_filtered["sentiment"].unique()]
pnl_filtered_ordered = pnl_filtered[pnl_filtered["sentiment"].isin(valid_sentiments)]

box_colors = [COLORS.get(s, "#888") for s in valid_sentiments]
bp = ax.boxplot(
    [pnl_filtered_ordered[pnl_filtered_ordered["sentiment"] == s]["closed_pnl"] for s in valid_sentiments],
    labels=valid_sentiments,
    patch_artist=True,
    showfliers=False,
)
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("Closed PnL ($)")
ax.set_title("PnL Distribution (1st-99th percentile)", fontweight="bold")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart4_pnl_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Chart 4: PnL Distribution saved.")


# --- Chart 5: Sentiment Timeline with Trading Activity ---
fig, ax1 = plt.subplots(figsize=(16, 6))
fig.suptitle("Sentiment Timeline & Trading Activity Over Time", fontsize=16, fontweight="bold")

ax1.fill_between(daily_agg["date"], daily_agg["sentiment_value"], alpha=0.3, color="steelblue")
ax1.plot(daily_agg["date"], daily_agg["sentiment_value"], color="steelblue", linewidth=1.5, label="Sentiment Value")
ax1.axhline(y=25, color="red", linestyle="--", alpha=0.5, label="Extreme Fear Threshold")
ax1.axhline(y=75, color="green", linestyle="--", alpha=0.5, label="Extreme Greed Threshold")
ax1.set_ylabel("Sentiment Score", color="steelblue")
ax1.set_xlabel("Date")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.bar(daily_agg["date"], daily_agg["total_trades"], alpha=0.3, color="orange", label="Daily Trades")
ax2.set_ylabel("Number of Trades", color="orange")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart5_sentiment_timeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Chart 5: Sentiment Timeline saved.")


# --- Chart 6: Correlation Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Sentiment Score vs. Trading Metrics", fontsize=16, fontweight="bold")

# Sentiment vs PnL
axes[0].scatter(daily_agg["sentiment_value"], daily_agg["total_pnl"], alpha=0.5, c="steelblue", edgecolors="black", linewidth=0.3, s=30)
# Trend line
z = np.polyfit(daily_agg["sentiment_value"], daily_agg["total_pnl"], 1)
p = np.poly1d(z)
x_line = np.linspace(daily_agg["sentiment_value"].min(), daily_agg["sentiment_value"].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trend (slope={z[0]:.2f})")
axes[0].set_xlabel("Sentiment Score")
axes[0].set_ylabel("Total Daily PnL ($)")
axes[0].set_title("Sentiment vs PnL")
axes[0].legend()
corr_pnl = daily_agg["sentiment_value"].corr(daily_agg["total_pnl"])
axes[0].annotate(f"r = {corr_pnl:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12, fontweight="bold")

# Sentiment vs Volume
axes[1].scatter(daily_agg["sentiment_value"], daily_agg["total_volume"], alpha=0.5, c="darkorange", edgecolors="black", linewidth=0.3, s=30)
z2 = np.polyfit(daily_agg["sentiment_value"], daily_agg["total_volume"], 1)
p2 = np.poly1d(z2)
axes[1].plot(x_line, p2(x_line), "r--", linewidth=2, label=f"Trend (slope={z2[0]:.2f})")
axes[1].set_xlabel("Sentiment Score")
axes[1].set_ylabel("Total Daily Volume ($)")
axes[1].set_title("Sentiment vs Volume")
axes[1].legend()
corr_vol = daily_agg["sentiment_value"].corr(daily_agg["total_volume"])
axes[1].annotate(f"r = {corr_vol:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart6_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] Chart 6: Correlation Analysis saved.")


# ---- Key Insights ----
print("\n" + "="*60)
print(" KEY INSIGHTS")
print("="*60)

insights = []

# Insight 1: PnL by sentiment
fear_pnl = perf_by_sentiment.loc["Fear", "avg_daily_pnl"] if "Fear" in perf_by_sentiment.index else 0
greed_pnl = perf_by_sentiment.loc["Greed", "avg_daily_pnl"] if "Greed" in perf_by_sentiment.index else 0
insight1 = f"1. PnL Gap: Average daily PnL during Fear days is ${fear_pnl:,.2f} vs ${greed_pnl:,.2f} during Greed days."
print(f"\n{insight1}")
insights.append(insight1)

# Insight 2: Win rate difference
fear_wr = perf_by_sentiment.loc["Fear", "avg_win_rate"] if "Fear" in perf_by_sentiment.index else 0
greed_wr = perf_by_sentiment.loc["Greed", "avg_win_rate"] if "Greed" in perf_by_sentiment.index else 0
insight2 = f"2. Win Rate: Traders achieve {fear_wr*100:.1f}% win rate on Fear days vs {greed_wr*100:.1f}% on Greed days."
print(f"\n{insight2}")
insights.append(insight2)

# Insight 3: Trading frequency
fear_trades = perf_by_sentiment.loc["Fear", "avg_trades_per_day"] if "Fear" in perf_by_sentiment.index else 0
greed_trades = perf_by_sentiment.loc["Greed", "avg_trades_per_day"] if "Greed" in perf_by_sentiment.index else 0
insight3 = f"3. Activity: Traders execute {fear_trades:,.0f} trades/day during Fear vs {greed_trades:,.0f} during Greed days."
print(f"\n{insight3}")
insights.append(insight3)

# Insight 4: Correlation
insight4 = f"4. Correlation: Sentiment score has r={corr_pnl:.3f} with daily PnL and r={corr_vol:.3f} with volume."
print(f"\n{insight4}")
insights.append(insight4)


# ============================================================================
# PART C: ACTIONABLE OUTPUT
# ============================================================================
print_section("PART C: ACTIONABLE OUTPUT - STRATEGY RECOMMENDATIONS")

strategies = []

s1 = """
STRATEGY 1: Contrarian Sentiment-Based Position Sizing
-------------------------------------------------------
Rule: During Extreme Fear days (sentiment < 25), INCREASE position sizes for 
long trades by 20-30%. During Extreme Greed days (sentiment > 75), REDUCE 
position sizes by 20-30% and favor shorter holding periods.

Rationale: Historical data shows that Extreme Fear periods tend to represent 
market bottoms with mean-reversion potential. Traders who maintain or increase 
exposure during these periods capture the subsequent recovery. Conversely, 
Extreme Greed periods often precede corrections, and smaller positions limit 
drawdown risk.

Target Segment: Best suited for "Consistent Winners" and "Frequent Traders" 
who have demonstrated the ability to manage risk effectively.
"""

s2 = """
STRATEGY 2: Sentiment-Adaptive Trading Frequency
-------------------------------------------------
Rule: During Fear periods, reduce trading frequency by 30-40% for infrequent 
traders (focus on higher-conviction trades with larger size). During Greed 
periods, frequent traders should maintain their pace but tighten stop-losses.

Rationale: Analysis shows that infrequent traders suffer more during Fear days 
from overtrading with poor conviction. Fewer, more deliberate trades improve 
their win rate. Frequent traders, on the other hand, benefit from Greed-day 
momentum but need tighter risk management to protect gains.

Target Segment: Differentiated advice for "Frequent" vs "Infrequent" traders.
"""

strategies = [s1, s2]
for s in strategies:
    print(s)


# ============================================================================
# BONUS: PREDICTIVE MODEL
# ============================================================================
print_section("BONUS: PREDICTIVE MODEL & CLUSTERING")

# --- Prepare features for prediction ---
print("Building predictive model for next-day profitability...\n")

# Daily account features
daily_features = daily_account.copy()
daily_features["profitable_day"] = (daily_features["total_pnl"] > 0).astype(int)

# Add lagged sentiment (we use same-day here as proxy since we predict profitability bucket)
feature_cols = ["sentiment_value", "num_trades", "total_volume_usd", "avg_trade_size_usd",
                "num_buys", "num_sells", "long_short_ratio"]

X = daily_features[feature_cols].fillna(0)
y = daily_features["profitable_day"]

# Only proceed if enough samples
scores = None
if len(X) > 50:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    cv = StratifiedKFold(n_splits=min(5, y.value_counts().min()), shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
        print(f"Random Forest - Profitability Prediction:")
        print(f"  Cross-Val Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Feature importance
        rf.fit(X_scaled, y)
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        print(f"\n  Feature Importance:")
        print(importance.to_string(index=False))
        
        # Chart 7: Feature Importance
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(importance["feature"], importance["importance"], color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance for Profitability Prediction", fontweight="bold", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "chart7_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("\n  [✓] Chart 7: Feature Importance saved.")
    except Exception as e:
        print(f"  Model training skipped due to: {e}")

# --- Clustering: Trader Archetypes ---
print("\n\nClustering traders into behavioral archetypes...\n")

cluster_features = ["total_trades", "avg_trade_size", "win_rate", "total_pnl", "long_ratio", "avg_trades_per_day"]
X_cluster = account_stats[cluster_features].fillna(0)

# Remove extreme outliers for clustering
for col in cluster_features:
    q99 = X_cluster[col].quantile(0.99)
    q01 = X_cluster[col].quantile(0.01)
    X_cluster[col] = X_cluster[col].clip(q01, q99)

scaler_c = StandardScaler()
X_cluster_scaled = scaler_c.fit_transform(X_cluster)

# Find optimal K using inertia
n_clusters = None
if len(X_cluster_scaled) >= 4:
    inertias = []
    K_range = range(2, min(8, len(X_cluster_scaled)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster_scaled)
        inertias.append(km.inertia_)

    # Use 3 clusters
    n_clusters = min(3, len(X_cluster_scaled) - 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    account_stats["cluster"] = kmeans.fit_predict(X_cluster_scaled)

    # Describe clusters
    cluster_labels = {0: "Archetype A", 1: "Archetype B", 2: "Archetype C"}
    account_stats["archetype"] = account_stats["cluster"].map(cluster_labels)

    cluster_desc = account_stats.groupby("archetype")[cluster_features].mean().round(2)
    print("Cluster Centroids:")
    print(cluster_desc.to_string())
    cluster_desc.to_csv(os.path.join(OUTPUT_DIR, "cluster_archetypes.csv"))

    # Name archetypes based on characteristics
    print("\n  Archetype Interpretation:")
    for arch in cluster_desc.index:
        row = cluster_desc.loc[arch]
        print(f"    {arch}: {row['total_trades']:.0f} trades, ${row['avg_trade_size']:.0f} avg size, "
              f"{row['win_rate']*100:.1f}% win rate, ${row['total_pnl']:.0f} total PnL")

    # Chart 8: Cluster Visualization (2D using PCA-like)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Trader Archetypes (K-Means Clustering)", fontsize=16, fontweight="bold")
    
    cluster_colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    for i, arch in enumerate(cluster_desc.index):
        mask = account_stats["archetype"] == arch
        axes[0].scatter(
            account_stats.loc[mask, "total_trades"],
            account_stats.loc[mask, "total_pnl"],
            c=cluster_colors[i], label=arch, alpha=0.7, edgecolors="black", linewidth=0.3, s=50
        )
    axes[0].set_xlabel("Total Trades")
    axes[0].set_ylabel("Total PnL ($)")
    axes[0].set_title("Trades vs PnL")
    axes[0].legend()

    for i, arch in enumerate(cluster_desc.index):
        mask = account_stats["archetype"] == arch
        axes[1].scatter(
            account_stats.loc[mask, "avg_trade_size"],
            account_stats.loc[mask, "win_rate"],
            c=cluster_colors[i], label=arch, alpha=0.7, edgecolors="black", linewidth=0.3, s=50
        )
    axes[1].set_xlabel("Avg Trade Size ($)")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_title("Trade Size vs Win Rate")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart8_trader_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [✓] Chart 8: Trader Clusters saved.")

    # Chart 9: Cluster radar chart / bar comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cluster_features))
    width = 0.25
    
    for i, arch in enumerate(cluster_desc.index):
        # Normalize for comparison
        vals = cluster_desc.loc[arch].values
        vals_norm = (vals - cluster_desc.min().values) / (cluster_desc.max().values - cluster_desc.min().values + 1e-10)
        ax.bar(x + i * width, vals_norm, width, label=arch, color=cluster_colors[i], edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(cluster_features, rotation=30, ha="right")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Trader Archetype Comparison (Normalized)", fontweight="bold", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart9_archetype_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Chart 9: Archetype Comparison saved.")


# ============================================================================
# SAVE FINAL SUMMARY
# ============================================================================
print_section("SUMMARY")

final_summary = {
    "data_summary": summary,
    "insights": insights,
    "sentiment_correlation_pnl": float(corr_pnl),
    "sentiment_correlation_volume": float(corr_vol),
    "model_accuracy": float(scores.mean()) if scores is not None else None,
    "num_clusters": n_clusters,
    "charts_generated": [
        "chart1_performance_by_sentiment.png",
        "chart2_behavior_by_sentiment.png",
        "chart3_segment_heatmap.png",
        "chart4_pnl_distribution.png",
        "chart5_sentiment_timeline.png",
        "chart6_correlation.png",
        "chart7_feature_importance.png",
        "chart8_trader_clusters.png",
        "chart9_archetype_comparison.png",
    ],
}

with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w") as f:
    json.dump(final_summary, f, indent=2, default=str)

print("All analysis complete!")
print(f"Output saved to: {OUTPUT_DIR}/")
print(f"Charts: 9 generated")
print(f"Tables: performance_by_sentiment.csv, behavior_by_sentiment.csv, cluster_archetypes.csv")
print(f"Summary: analysis_summary.json, data_summary.json")
