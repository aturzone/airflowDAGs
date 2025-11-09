"""
Analytics Page - Advanced Analytics and Insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Analytics - AnomalyGuard", page_icon="📈", layout="wide")

st.title("📈 Advanced Analytics")

database_manager = st.session_state.get('database_manager')
model_manager = st.session_state.get('model_manager')

if not database_manager:
    st.error("Database manager not initialized")
    st.stop()

# Time range selector
days_back = st.selectbox("Analysis Period", [7, 30, 90, 365], index=1)
start_date = datetime.now() - timedelta(days=days_back)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Detection Trends",
    "🤖 Model Performance",
    "💰 Transaction Patterns"
])

# Tab 1: Detection Trends
with tab1:
    st.header("Anomaly Detection Trends")

    anomalies = database_manager.get_anomalies(
        limit=10000,
        start_date=start_date
    )

    if not anomalies.empty:
        # Daily trend
        anomalies['detected_at'] = pd.to_datetime(anomalies['detected_at'])
        daily_counts = anomalies.groupby([
            anomalies['detected_at'].dt.date,
            'risk_level'
        ]).size().reset_index(name='count')

        fig = px.area(
            daily_counts,
            x='detected_at',
            y='count',
            color='risk_level',
            title=f"Daily Anomaly Detections (Last {days_back} Days)",
            color_discrete_map={
                'low': '#28a745',
                'medium': '#ffc107',
                'high': '#fd7e14',
                'critical': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk level breakdown
        col1, col2 = st.columns(2)

        with col1:
            risk_stats = anomalies['risk_level'].value_counts()
            fig = px.bar(
                x=risk_stats.index,
                y=risk_stats.values,
                title="Anomalies by Risk Level",
                labels={'x': 'Risk Level', 'y': 'Count'},
                color=risk_stats.index,
                color_discrete_map={
                    'low': '#28a745',
                    'medium': '#ffc107',
                    'high': '#fd7e14',
                    'critical': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Score distribution
            fig = px.histogram(
                anomalies,
                x='ensemble_score',
                nbins=30,
                title="Ensemble Score Distribution",
                labels={'ensemble_score': 'Score', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Hourly patterns
        st.subheader("Temporal Patterns")

        anomalies['hour'] = anomalies['detected_at'].dt.hour
        hourly_counts = anomalies.groupby('hour').size().reset_index(name='count')

        fig = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            title="Anomalies by Hour of Day",
            labels={'hour': 'Hour', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No anomaly data available for analysis")

# Tab 2: Model Performance
with tab2:
    st.header("Model Performance Analytics")

    if model_manager:
        # Get performance data
        performance = database_manager.get_model_performance(days=days_back)

        if not performance.empty:
            # Performance over time
            performance['date'] = pd.to_datetime(performance['date'])

            fig = px.line(
                performance,
                x='date',
                y='avg_score',
                color='model_type',
                title=f"Model Performance Trend (Last {days_back} Days)",
                labels={'date': 'Date', 'avg_score': 'Average Score', 'model_type': 'Model'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Processing time comparison
            col1, col2 = st.columns(2)

            with col1:
                avg_times = performance.groupby('model_type')['avg_processing_time_ms'].mean()
                fig = px.bar(
                    x=avg_times.index,
                    y=avg_times.values,
                    title="Average Processing Time by Model",
                    labels={'x': 'Model Type', 'y': 'Time (ms)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Detection counts by model
                detections = performance.groupby('model_type')[
                    ['low_count', 'medium_count', 'high_count', 'critical_count']
                ].sum()

                fig = go.Figure(data=[
                    go.Bar(name='Low', x=detections.index, y=detections['low_count']),
                    go.Bar(name='Medium', x=detections.index, y=detections['medium_count']),
                    go.Bar(name='High', x=detections.index, y=detections['high_count']),
                    go.Bar(name='Critical', x=detections.index, y=detections['critical_count'])
                ])
                fig.update_layout(
                    barmode='stack',
                    title="Detections by Model and Risk Level"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No model performance data available")

    else:
        st.warning("Model manager not available")

# Tab 3: Transaction Patterns
with tab3:
    st.header("Transaction Pattern Analysis")

    # Get hourly stats
    hourly_stats = database_manager.get_hourly_stats(hours=days_back * 24)

    if not hourly_stats.empty:
        hourly_stats['hour'] = pd.to_datetime(hourly_stats['hour'])

        # Transaction volume trend
        fig = px.line(
            hourly_stats,
            x='hour',
            y='transaction_count',
            title=f"Transaction Volume Trend (Last {days_back} Days)",
            labels={'hour': 'Date/Time', 'transaction_count': 'Transaction Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Amount trends
        col1, col2 = st.columns(2)

        with col1:
            if 'total_amount' in hourly_stats.columns:
                fig = px.line(
                    hourly_stats,
                    x='hour',
                    y='total_amount',
                    title="Transaction Amount Trend",
                    labels={'hour': 'Date/Time', 'total_amount': 'Total Amount'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'unique_users' in hourly_stats.columns:
                fig = px.line(
                    hourly_stats,
                    x='hour',
                    y='unique_users',
                    title="Unique Users Trend",
                    labels={'hour': 'Date/Time', 'unique_users': 'Unique Users'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Transaction type breakdown
        transactions = database_manager.get_transactions(
            limit=10000,
            start_date=start_date
        )

        if not transactions.empty:
            col1, col2 = st.columns(2)

            with col1:
                type_counts = transactions['transaction_type'].value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Transaction Types"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                currency_counts = transactions['currency'].value_counts()
                fig = px.pie(
                    values=currency_counts.values,
                    names=currency_counts.index,
                    title="Currencies"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No transaction statistics available")

# Summary Statistics
st.markdown("---")
st.header("📊 Summary Statistics")

col1, col2, col3 = st.columns(3)

# Anomalies summary
anomalies = database_manager.get_anomalies(limit=10000, start_date=start_date)
if not anomalies.empty:
    col1.metric("Total Anomalies", len(anomalies))
    col1.metric("Avg Ensemble Score", f"{anomalies['ensemble_score'].mean():.2f}")

# Transactions summary
transactions = database_manager.get_transactions(limit=10000, start_date=start_date)
if not transactions.empty:
    col2.metric("Total Transactions", len(transactions))
    col2.metric("Avg Transaction Amount", f"${transactions['amount'].mean():.2f}")

# Model summary
if model_manager:
    summary = model_manager.get_models_summary()
    col3.metric("Active Models", summary.get('active_models', 0))
    col3.metric("Total Model Size", f"{summary.get('total_size_mb', 0):.2f} MB")
