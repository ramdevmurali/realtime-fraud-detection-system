import streamlit as st
import sqlite3
import pandas as pd
import time
import sys
import os
import altair as alt

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config

# Page Config
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide"
)

config = load_config()
DB_PATH = config['data'].get('db_path', 'data/database.db')

def load_data():
    """Fetch the last 1000 transactions from the DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1000"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return pd.DataFrame()

# Title
st.title("🛡️ Real-Time Fraud Detection Monitor")

# Sidebar settings
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 2)
auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=True)

# Main Loop
placeholder = st.empty()

while True:
    df = load_data()

    if not df.empty:
        # Calculate Metrics
        total_tx = len(df)
        fraud_tx = df[df['is_fraud'] == 1].shape[0]
        fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0
        avg_latency = df['latency_ms'].mean()

        with placeholder.container():
            # KPI Metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Transactions (Window)", total_tx)
            kpi2.metric("Fraud Cases Detected", fraud_tx, delta_color="inverse")
            kpi3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            kpi4.metric("Avg Latency", f"{avg_latency:.1f} ms")

            # Layout: Charts and Tables
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("⚠️ Recent Fraud Alerts")
                frauds = df[df['is_fraud'] == 1][['timestamp', 'prediction_score', 'latency_ms']].head(10)
                if not frauds.empty:
                    st.dataframe(frauds, use_container_width=True)
                else:
                    st.info("No fraud detected in current window.")

            with col2:
                st.subheader("Prediction Distribution")
                # Histogram of scores
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('prediction_score', bin=alt.Bin(maxbins=20), title='Fraud Score'),
                    y='count()',
                    color=alt.condition(
                        alt.datum.prediction_score > 0.5,
                        alt.value('red'),  # Fraud color
                        alt.value('green')   # Normal color
                    )
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # Raw Data Stream
            st.subheader("Live Transaction Feed")
            st.dataframe(df[['timestamp', 'prediction_score', 'is_fraud', 'latency_ms']].head(10), use_container_width=True)

    else:
        st.warning("Waiting for data... Ensure the simulator is running.")

    if not auto_refresh:
        break
    
    time.sleep(refresh_rate)