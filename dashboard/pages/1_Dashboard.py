"""
Dashboard Page - Real-time Monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Dashboard - AnomalyGuard", page_icon="📊", layout="wide")

st.title("📊 Real-time Monitoring Dashboard")

# Get managers
database_manager = st.session_state.get('database_manager')
model_manager = st.session_state.get('model_manager')
airflow_manager = st.session_state.get('airflow_manager')
docker_manager = st.session_state.get('docker_manager')

# Auto-refresh
refresh_interval = st.sidebar.selectbox("Auto-refresh", ["Off", "5s", "30s", "60s"], index=0)

if refresh_interval != "Off":
    import time
    interval_map = {"5s": 5, "30s": 30, "60s": 60}
    time.sleep(interval_map[refresh_interval])
    st.rerun()

# Key Metrics
st.header("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

if database_manager:
    # Total transactions
    total_tx = database_manager.get_transaction_count()
    col1.metric("Total Transactions", f"{total_tx:,}")

    # Anomalies today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    anomalies_today = database_manager.get_anomaly_stats(start_date=today_start)
    total_anomalies_today = sum(anomalies_today.values())
    col2.metric("Anomalies Today", f"{total_anomalies_today:,}")

    # Critical alerts
    critical_count = anomalies_today.get('critical', 0)
    col3.metric("Critical Alerts", critical_count, delta=critical_count, delta_color="inverse")

    # Active models
    if model_manager:
        model_summary = model_manager.get_models_summary()
        col4.metric("Active Models", model_summary.get('active_models', 0))

st.markdown("---")

# System Health
st.header("🔍 System Health")

col1, col2, col3 = st.columns(3)

with col1:
    if database_manager:
        ch_ok, _ = database_manager.test_clickhouse_connection()
        if ch_ok:
            st.success("✅ ClickHouse Online")
        else:
            st.error("❌ ClickHouse Offline")
    else:
        st.warning("⚠️ ClickHouse Status Unknown")

with col2:
    if airflow_manager:
        af_ok, _ = airflow_manager.health_check()
        if af_ok:
            st.success("✅ Airflow Online")
        else:
            st.error("❌ Airflow Offline")
    else:
        st.warning("⚠️ Airflow Status Unknown")

with col3:
    if docker_manager and docker_manager.is_docker_available():
        health = docker_manager.get_service_health_summary()
        healthy_pct = health.get('health_percentage', 0)
        if healthy_pct >= 80:
            st.success(f"✅ Services {healthy_pct:.0f}% Healthy")
        else:
            st.warning(f"⚠️ Services {healthy_pct:.0f}% Healthy")
    else:
        st.warning("⚠️ Docker Status Unknown")

st.markdown("---")

# Recent Anomalies
st.header("🔍 Recent Anomalies (Last 24 Hours)")

if database_manager:
    yesterday = datetime.now() - timedelta(days=1)
    recent_anomalies = database_manager.get_anomalies(limit=100, start_date=yesterday)

    if not recent_anomalies.empty:
        # Risk level distribution
        col1, col2 = st.columns([1, 2])

        with col1:
            risk_counts = recent_anomalies['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    'low': '#28a745',
                    'medium': '#ffc107',
                    'high': '#fd7e14',
                    'critical': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Timeline
            if 'detected_at' in recent_anomalies.columns:
                recent_anomalies['detected_at'] = pd.to_datetime(recent_anomalies['detected_at'])
                hourly_counts = recent_anomalies.groupby([
                    recent_anomalies['detected_at'].dt.floor('H'),
                    'risk_level'
                ]).size().reset_index(name='count')

                fig = px.line(
                    hourly_counts,
                    x='detected_at',
                    y='count',
                    color='risk_level',
                    title="Anomalies Over Time",
                    color_discrete_map={
                        'low': '#28a745',
                        'medium': '#ffc107',
                        'high': '#fd7e14',
                        'critical': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

        # Top anomalies table
        st.subheader("Highest Risk Anomalies")
        top_anomalies = recent_anomalies.nlargest(10, 'ensemble_score')

        display_cols = ['transaction_id', 'user_id', 'risk_level', 'ensemble_score', 'detected_at']
        available_cols = [col for col in display_cols if col in top_anomalies.columns]

        st.dataframe(
            top_anomalies[available_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No anomalies detected in the last 24 hours")
else:
    st.warning("Database connection not available")

st.markdown("---")

# Transaction Statistics
st.header("📊 Transaction Statistics")

if database_manager:
    hourly_stats = database_manager.get_hourly_stats(hours=24)

    if not hourly_stats.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Transaction volume
            if 'hour' in hourly_stats.columns and 'transaction_count' in hourly_stats.columns:
                fig = px.bar(
                    hourly_stats,
                    x='hour',
                    y='transaction_count',
                    title="Hourly Transaction Volume",
                    labels={'hour': 'Hour', 'transaction_count': 'Transactions'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Transaction amounts
            if 'hour' in hourly_stats.columns and 'total_amount' in hourly_stats.columns:
                fig = px.line(
                    hourly_stats,
                    x='hour',
                    y='total_amount',
                    title="Hourly Transaction Amounts",
                    labels={'hour': 'Hour', 'total_amount': 'Total Amount'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transaction statistics available")

st.markdown("---")

# Model Performance
st.header("🤖 Model Performance")

if model_manager and database_manager:
    performance = database_manager.get_model_performance(days=7)

    if not performance.empty:
        # Average scores by model type
        avg_by_type = performance.groupby('model_type').agg({
            'avg_score': 'mean',
            'avg_processing_time_ms': 'mean'
        }).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            if not avg_by_type.empty:
                fig = px.bar(
                    avg_by_type,
                    x='model_type',
                    y='avg_score',
                    title="Average Anomaly Score by Model",
                    labels={'model_type': 'Model Type', 'avg_score': 'Avg Score'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not avg_by_type.empty:
                fig = px.bar(
                    avg_by_type,
                    x='model_type',
                    y='avg_processing_time_ms',
                    title="Average Processing Time by Model",
                    labels={'model_type': 'Model Type', 'avg_processing_time_ms': 'Avg Time (ms)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model performance data available")

# Refresh button
if st.button("🔄 Refresh Dashboard", use_container_width=True):
    st.rerun()
