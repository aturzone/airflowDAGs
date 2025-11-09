"""
Anomalies Page - Anomaly Detection Results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Anomalies - AnomalyGuard", page_icon="🔍", layout="wide")

st.title("🔍 Anomaly Detection Results")

database_manager = st.session_state.get('database_manager')

if not database_manager:
    st.error("Database manager not initialized")
    st.stop()

# Filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    days_back = st.selectbox("Time Range", [1, 7, 30, 90], index=1)
    start_date = datetime.now() - timedelta(days=days_back)

with col2:
    risk_filter = st.selectbox("Risk Level", ["All", "low", "medium", "high", "critical"])

with col3:
    min_score = st.number_input("Min Score", 0, 100, 0)

with col4:
    limit = st.number_input("Max Results", 10, 1000, 100)

# Get anomalies
risk_level = None if risk_filter == "All" else risk_filter

anomalies = database_manager.get_anomalies(
    limit=limit,
    start_date=start_date,
    risk_level=risk_level,
    min_score=min_score if min_score > 0 else None
)

# Stats
st.subheader("📊 Summary")

if not anomalies.empty:
    col1, col2, col3, col4 = st.columns(4)

    total = len(anomalies)
    low_count = len(anomalies[anomalies['risk_level'] == 'low'])
    medium_count = len(anomalies[anomalies['risk_level'] == 'medium'])
    high_count = len(anomalies[anomalies['risk_level'] == 'high'])
    critical_count = len(anomalies[anomalies['risk_level'] == 'critical'])

    col1.metric("Total", total)
    col2.metric("🟢 Low", low_count)
    col3.metric("🟡 Medium/High", medium_count + high_count)
    col4.metric("🔴 Critical", critical_count)

    st.markdown("---")

    # Risk distribution chart
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = anomalies['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
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
        # Score distribution
        fig = px.histogram(
            anomalies,
            x='ensemble_score',
            nbins=20,
            title="Anomaly Score Distribution",
            labels={'ensemble_score': 'Ensemble Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Anomalies table
    st.subheader("🔍 Detected Anomalies")

    display_cols = [
        'transaction_id', 'user_id', 'risk_level', 'ensemble_score',
        'statistical_score', 'isolation_forest_score', 'autoencoder_score',
        'detected_at'
    ]

    available_cols = [col for col in display_cols if col in anomalies.columns]

    st.dataframe(
        anomalies[available_cols].sort_values('detected_at', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Export
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export to CSV", use_container_width=True):
            csv = anomalies.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

else:
    st.info("No anomalies found with the selected filters")
