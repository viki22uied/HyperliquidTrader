"""
Streamlit Dashboard: Hyperliquid Trader Behavior vs. Market Sentiment
=====================================================================
Interactive explorer for the analysis results.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Hyperliquid Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .stMetric {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load and prepare all datasets."""
    sentiment_df = pd.read_csv("fear_greed_index.csv")
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    sentiment_df = sentiment_df.rename(columns={"classification": "sentiment", "value": "sentiment_value"})

    trades_df = pd.read_csv("historical_data.csv")
    trades_df["datetime"] = pd.to_datetime(trades_df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    mask_nat = trades_df["datetime"].isna()
    if mask_nat.any():
        trades_df.loc[mask_nat, "datetime"] = pd.to_datetime(trades_df.loc[mask_nat, "Timestamp IST"], errors="coerce")
    trades_df["date"] = pd.to_datetime(trades_df["datetime"].dt.date)

    for col in ["Execution Price", "Size Tokens", "Size USD", "Closed PnL", "Fee"]:
        trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

    trades_df = trades_df.rename(columns={
        "Account": "account", "Coin": "coin", "Execution Price": "exec_price",
        "Size Tokens": "size_tokens", "Size USD": "size_usd", "Side": "side",
        "Start Position": "start_position", "Direction": "direction",
        "Closed PnL": "closed_pnl", "Fee": "fee",
    })

    merged_df = trades_df.merge(sentiment_df[["date", "sentiment", "sentiment_value"]], on="date", how="left")
    merged_df = merged_df.dropna(subset=["sentiment"])
    merged_df["sentiment_binary"] = merged_df["sentiment"].apply(
        lambda x: "Fear" if x in ["Extreme Fear", "Fear"] else ("Greed" if x in ["Extreme Greed", "Greed"] else "Neutral")
    )
    merged_df["is_long"] = merged_df["side"].str.upper() == "BUY"
    merged_df["is_close"] = merged_df["direction"].str.contains("Close", case=False, na=False)
    merged_df["is_winner"] = merged_df["closed_pnl"] > 0
    merged_df["net_pnl"] = merged_df["closed_pnl"] - merged_df["fee"]

    return sentiment_df, trades_df, merged_df


try:
    sentiment_df, trades_df, merged_df = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading data: {e}")

if not data_loaded:
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🎛️ Filters")

    # Date range
    min_date = merged_df["date"].min().date()
    max_date = merged_df["date"].max().date()
    date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Sentiment filter
    all_sentiments = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    available_sentiments = [s for s in all_sentiments if s in merged_df["sentiment"].unique()]
    selected_sentiments = st.multiselect("Sentiment Categories", available_sentiments, default=available_sentiments)

    # Coin filter
    all_coins = sorted(merged_df["coin"].unique())
    selected_coins = st.multiselect("Coins", all_coins, default=all_coins)

    # Account filter
    all_accounts = sorted(merged_df["account"].unique())
    selected_accounts = st.multiselect(
        "Accounts", all_accounts, default=all_accounts,
        help="Select specific accounts to analyze"
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    st.metric("Total Trades", f"{len(merged_df):,}")
    st.metric("Unique Accounts", f"{merged_df['account'].nunique()}")
    st.metric("Trading Days", f"{merged_df['date'].nunique()}")

# Apply filters
if len(date_range) == 2:
    mask = (
        (merged_df["date"].dt.date >= date_range[0])
        & (merged_df["date"].dt.date <= date_range[1])
        & (merged_df["sentiment"].isin(selected_sentiments))
        & (merged_df["coin"].isin(selected_coins))
        & (merged_df["account"].isin(selected_accounts))
    )
    filtered_df = merged_df[mask].copy()
else:
    filtered_df = merged_df.copy()


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown('<p class="main-header">📊 Hyperliquid × Sentiment Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">How Bitcoin Fear/Greed Index Affects Trader Behavior on Hyperliquid</p>', unsafe_allow_html=True)

# --- Top Metrics ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🔄 Trades", f"{len(filtered_df):,}")
with col2:
    total_pnl = filtered_df["closed_pnl"].sum()
    st.metric("💰 Total PnL", f"${total_pnl:,.2f}", delta=f"{'Profit' if total_pnl > 0 else 'Loss'}")
with col3:
    total_vol = filtered_df["size_usd"].sum()
    st.metric("📊 Volume", f"${total_vol:,.0f}")
with col4:
    closes = filtered_df[filtered_df["is_close"]]
    wr = closes["is_winner"].mean() * 100 if len(closes) > 0 else 0
    st.metric("🎯 Win Rate", f"{wr:.1f}%")
with col5:
    avg_sent = filtered_df["sentiment_value"].mean()
    st.metric("😰/😊 Avg Sentiment", f"{avg_sent:.0f}")

st.markdown("---")

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Performance", "🔍 Behavior", "👥 Segments", "🤖 Predictions", "📋 Data Explorer"]
)

# ============================================================================
# TAB 1: PERFORMANCE
# ============================================================================
with tab1:
    st.header("Performance by Market Sentiment")

    # Aggregations
    daily_agg = filtered_df.groupby(["date", "sentiment", "sentiment_value", "sentiment_binary"]).agg(
        total_trades=("side", "count"),
        total_volume=("size_usd", "sum"),
        total_pnl=("closed_pnl", "sum"),
        avg_win_rate=("is_winner", "mean"),
    ).reset_index()

    col_a, col_b = st.columns(2)

    with col_a:
        # PnL by sentiment
        perf = daily_agg.groupby("sentiment").agg(
            avg_daily_pnl=("total_pnl", "mean"),
            total_pnl=("total_pnl", "sum"),
        ).reset_index()

        sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        perf["sentiment"] = pd.Categorical(perf["sentiment"], categories=sentiment_order, ordered=True)
        perf = perf.sort_values("sentiment")

        color_map = {
            "Extreme Fear": "#8B0000", "Fear": "#FF4500",
            "Neutral": "#FFD700", "Greed": "#32CD32", "Extreme Greed": "#006400"
        }

        fig = px.bar(
            perf, x="sentiment", y="avg_daily_pnl",
            color="sentiment", color_discrete_map=color_map,
            title="Average Daily PnL by Sentiment",
            labels={"avg_daily_pnl": "Avg Daily PnL ($)", "sentiment": "Sentiment"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            template="plotly_dark", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
        )
        st.plotly_chart(fig, width="stretch")

    with col_b:
        # Win rate by sentiment
        wr_by_sent = daily_agg.groupby("sentiment")["avg_win_rate"].mean().reset_index()
        wr_by_sent["sentiment"] = pd.Categorical(wr_by_sent["sentiment"], categories=sentiment_order, ordered=True)
        wr_by_sent = wr_by_sent.sort_values("sentiment")
        wr_by_sent["avg_win_rate"] = wr_by_sent["avg_win_rate"] * 100

        fig2 = px.bar(
            wr_by_sent, x="sentiment", y="avg_win_rate",
            color="sentiment", color_discrete_map=color_map,
            title="Average Win Rate by Sentiment",
            labels={"avg_win_rate": "Win Rate (%)", "sentiment": "Sentiment"},
        )
        fig2.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5)
        fig2.update_layout(
            template="plotly_dark", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
        )
        st.plotly_chart(fig2, width="stretch")

    # Timeline chart
    st.subheader("📅 Sentiment & PnL Timeline")
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(
        go.Scatter(
            x=daily_agg["date"], y=daily_agg["sentiment_value"],
            name="Sentiment", fill="tozeroy",
            line=dict(color="steelblue", width=1.5),
            fillcolor="rgba(70,130,180,0.2)",
        ), secondary_y=False,
    )
    fig3.add_trace(
        go.Bar(
            x=daily_agg["date"], y=daily_agg["total_pnl"],
            name="Daily PnL", opacity=0.6,
            marker_color=np.where(daily_agg["total_pnl"] > 0, "#2ECC71", "#E74C3C"),
        ), secondary_y=True,
    )
    fig3.update_layout(
        template="plotly_dark", title="Sentiment Score & Daily PnL Over Time",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
        height=400,
    )
    fig3.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig3.update_yaxes(title_text="Daily PnL ($)", secondary_y=True)
    st.plotly_chart(fig3, width="stretch")


# ============================================================================
# TAB 2: BEHAVIOR
# ============================================================================
with tab2:
    st.header("Trader Behavior by Sentiment")

    col_c, col_d = st.columns(2)

    with col_c:
        # Trade frequency
        freq_data = daily_agg.groupby("sentiment")["total_trades"].mean().reset_index()
        freq_data["sentiment"] = pd.Categorical(freq_data["sentiment"], categories=sentiment_order, ordered=True)
        freq_data = freq_data.sort_values("sentiment")
        
        fig4 = px.bar(
            freq_data, x="sentiment", y="total_trades",
            color="sentiment", color_discrete_map=color_map,
            title="Average Trades per Day by Sentiment",
        )
        fig4.update_layout(template="plotly_dark", showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)")
        st.plotly_chart(fig4, width="stretch")

    with col_d:
        # Volume
        vol_data = daily_agg.groupby("sentiment")["total_volume"].mean().reset_index()
        vol_data["sentiment"] = pd.Categorical(vol_data["sentiment"], categories=sentiment_order, ordered=True)
        vol_data = vol_data.sort_values("sentiment")
        
        fig5 = px.bar(
            vol_data, x="sentiment", y="total_volume",
            color="sentiment", color_discrete_map=color_map,
            title="Average Daily Volume by Sentiment",
        )
        fig5.update_layout(template="plotly_dark", showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)")
        st.plotly_chart(fig5, width="stretch")

    # Long/Short analysis
    st.subheader("📊 Long/Short Ratio by Sentiment")
    ls_by_sent = filtered_df.groupby("sentiment_binary").agg(
        num_buys=("is_long", "sum"),
        num_sells=("is_long", lambda x: (~x).sum()),
    ).reset_index()
    ls_by_sent["ratio"] = ls_by_sent["num_buys"] / ls_by_sent["num_sells"].clip(lower=1)

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name="Longs", x=ls_by_sent["sentiment_binary"], y=ls_by_sent["num_buys"], marker_color="#2ECC71"))
    fig6.add_trace(go.Bar(name="Shorts", x=ls_by_sent["sentiment_binary"], y=ls_by_sent["num_sells"], marker_color="#E74C3C"))
    fig6.update_layout(
        barmode="group", template="plotly_dark",
        title="Long vs Short Trades by Sentiment",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig6, width="stretch")

    # Correlation scatter
    st.subheader("🔗 Sentiment Score Correlations")
    col_e, col_f = st.columns(2)
    
    with col_e:
        fig7 = px.scatter(
            daily_agg, x="sentiment_value", y="total_pnl",
            color="sentiment_binary",
            color_discrete_map={"Fear": "#FF4500", "Neutral": "#FFD700", "Greed": "#32CD32"},
            title="Sentiment vs Daily PnL",
            labels={"sentiment_value": "Sentiment Score", "total_pnl": "Daily PnL ($)"},
        )
        # Manual numpy trend line
        x_vals = daily_agg["sentiment_value"].values
        y_vals = daily_agg["total_pnl"].values
        mask_finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask_finite.sum() > 2:
            z = np.polyfit(x_vals[mask_finite], y_vals[mask_finite], 1)
            x_line = np.linspace(x_vals[mask_finite].min(), x_vals[mask_finite].max(), 100)
            fig7.add_trace(go.Scatter(x=x_line, y=np.polyval(z, x_line), mode="lines",
                                      name=f"Trend (slope={z[0]:.1f})", line=dict(color="white", dash="dash", width=2)))
        fig7.update_layout(template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig7, width="stretch")
    
    with col_f:
        fig8 = px.scatter(
            daily_agg, x="sentiment_value", y="total_volume",
            color="sentiment_binary",
            color_discrete_map={"Fear": "#FF4500", "Neutral": "#FFD700", "Greed": "#32CD32"},
            title="Sentiment vs Daily Volume",
            labels={"sentiment_value": "Sentiment Score", "total_volume": "Daily Volume ($)"},
        )
        # Manual numpy trend line
        y_vals2 = daily_agg["total_volume"].values
        mask_finite2 = np.isfinite(x_vals) & np.isfinite(y_vals2)
        if mask_finite2.sum() > 2:
            z2 = np.polyfit(x_vals[mask_finite2], y_vals2[mask_finite2], 1)
            fig8.add_trace(go.Scatter(x=x_line, y=np.polyval(z2, x_line), mode="lines",
                                      name=f"Trend (slope={z2[0]:.1f})", line=dict(color="white", dash="dash", width=2)))
        fig8.update_layout(template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig8, width="stretch")


# ============================================================================
# TAB 3: SEGMENTS
# ============================================================================
with tab3:
    st.header("Trader Segmentation")

    # Build account stats
    account_stats = filtered_df.groupby("account").agg(
        total_trades=("side", "count"),
        total_volume=("size_usd", "sum"),
        avg_trade_size=("size_usd", "mean"),
        total_pnl=("closed_pnl", "sum"),
        num_closes=("is_close", "sum"),
        num_winners=("is_winner", "sum"),
        num_buys=("is_long", "sum"),
        trading_days=("date", "nunique"),
    ).reset_index()

    account_stats["win_rate"] = account_stats["num_winners"] / account_stats["num_closes"].clip(lower=1)
    account_stats["long_ratio"] = account_stats["num_buys"] / account_stats["total_trades"]

    freq_med = account_stats["total_trades"].median()
    account_stats["freq_segment"] = np.where(account_stats["total_trades"] >= freq_med, "Frequent", "Infrequent")
    
    vol_med = account_stats["avg_trade_size"].median()
    account_stats["size_segment"] = np.where(account_stats["avg_trade_size"] >= vol_med, "Large Position", "Small Position")
    
    account_stats["perf_segment"] = np.where(
        (account_stats["total_pnl"] > 0) & (account_stats["win_rate"] > 0.5),
        "Consistent Winner",
        np.where(account_stats["total_pnl"] > 0, "Inconsistent Winner", "Net Loser"),
    )

    col_g, col_h, col_i = st.columns(3)
    with col_g:
        fig_seg1 = px.pie(
            account_stats, names="freq_segment",
            title="Frequency Segments",
            color_discrete_sequence=["#3498DB", "#E74C3C"]
        )
        fig_seg1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seg1, width="stretch")
    
    with col_h:
        fig_seg2 = px.pie(
            account_stats, names="size_segment",
            title="Size Segments",
            color_discrete_sequence=["#2ECC71", "#F39C12"]
        )
        fig_seg2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seg2, width="stretch")
    
    with col_i:
        fig_seg3 = px.pie(
            account_stats, names="perf_segment",
            title="Performance Segments",
            color_discrete_sequence=["#2ECC71", "#F39C12", "#E74C3C"]
        )
        fig_seg3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seg3, width="stretch")

    # Segment performance table
    st.subheader("📊 Account Performance Table")
    display_cols = ["account", "total_trades", "total_volume", "avg_trade_size", "total_pnl",
                    "win_rate", "freq_segment", "size_segment", "perf_segment"]
    st.dataframe(
        account_stats[display_cols].sort_values("total_pnl", ascending=False).reset_index(drop=True),
        width="stretch",
        height=400,
    )

    # Segment scatter
    st.subheader("🔍 Segment Visualization")
    segment_choice = st.selectbox("Color by:", ["freq_segment", "size_segment", "perf_segment"])
    
    fig_scatter = px.scatter(
        account_stats, x="total_trades", y="total_pnl",
        color=segment_choice, size="total_volume",
        hover_data=["account", "win_rate", "avg_trade_size"],
        title="Traders: Trades vs PnL",
        labels={"total_trades": "Total Trades", "total_pnl": "Total PnL ($)"},
    )
    fig_scatter.update_layout(template="plotly_dark",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_scatter, width="stretch")


# ============================================================================
# TAB 4: PREDICTIONS
# ============================================================================
with tab4:
    st.header("🤖 Predictive Insights")

    # Load pre-computed results if available
    summary_path = os.path.join("output", "analysis_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results = json.load(f)
        
        col_j, col_k = st.columns(2)
        with col_j:
            if results.get("model_accuracy"):
                st.metric("Model Accuracy", f"{results['model_accuracy']*100:.1f}%")
            st.metric("Sentiment-PnL Correlation", f"{results.get('sentiment_correlation_pnl', 0):.3f}")
        with col_k:
            if results.get("num_clusters"):
                st.metric("Trader Archetypes", results["num_clusters"])
            st.metric("Sentiment-Volume Correlation", f"{results.get('sentiment_correlation_volume', 0):.3f}")

        st.markdown("---")
        
        st.subheader("📊 Key Insights")
        for insight in results.get("insights", []):
            st.info(insight)

    # Show charts from output
    st.subheader("📈 Generated Charts")
    chart_files = [
        ("chart7_feature_importance.png", "Feature Importance for Profitability Prediction"),
        ("chart8_trader_clusters.png", "Trader Archetypes (K-Means Clustering)"),
        ("chart9_archetype_comparison.png", "Archetype Comparison"),
    ]
    
    for fname, title in chart_files:
        fpath = os.path.join("output", fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=title, width="stretch")

    # Strategy Recommendations
    st.subheader("💡 Strategy Recommendations")
    
    with st.expander("Strategy 1: Contrarian Sentiment-Based Position Sizing", expanded=True):
        st.markdown("""
        **Rule:** During **Extreme Fear** days (sentiment < 25), increase position sizes for long trades 
        by 20-30%. During **Extreme Greed** days (sentiment > 75), reduce position sizes by 20-30%.
        
        **Rationale:** Extreme Fear periods tend to represent market bottoms with mean-reversion 
        potential. Traders who increase exposure capture the recovery. Extreme Greed often precedes 
        corrections.
        
        **Target:** Consistent Winners and Frequent Traders with proven risk management.
        """)
    
    with st.expander("Strategy 2: Sentiment-Adaptive Trading Frequency", expanded=True):
        st.markdown("""
        **Rule:** During Fear periods, reduce trading frequency by 30-40% for infrequent traders. 
        During Greed periods, frequent traders should maintain pace but tighten stop-losses.
        
        **Rationale:** Infrequent traders overtrade during Fear with poor conviction. Fewer, 
        deliberate trades improve win rate. Frequent traders benefit from Greed momentum but 
        need tighter risk management.
        
        **Target:** Differentiated for Frequent vs Infrequent trader segments.
        """)


# ============================================================================
# TAB 5: DATA EXPLORER
# ============================================================================
with tab5:
    st.header("📋 Raw Data Explorer")
    
    data_choice = st.selectbox("Select Dataset", ["Merged Trades", "Sentiment Data", "Account Statistics"])
    
    if data_choice == "Merged Trades":
        st.dataframe(filtered_df.head(500), width="stretch", height=600)
        st.caption(f"Showing first 500 of {len(filtered_df):,} rows")
    elif data_choice == "Sentiment Data":
        st.dataframe(sentiment_df, width="stretch", height=600)
    else:
        st.dataframe(account_stats, width="stretch", height=600)

    # Download buttons
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button("📥 Download Filtered Trades", csv_data, "filtered_trades.csv", "text/csv")
    with col_dl2:
        if "account_stats" in dir():
            csv_acc = account_stats.to_csv(index=False)
            st.download_button("📥 Download Account Stats", csv_acc, "account_stats.csv", "text/csv")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Built with Streamlit • Data: Hyperliquid + Bitcoin Fear/Greed Index"
    "</div>",
    unsafe_allow_html=True,
)
