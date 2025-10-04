"""
Streamlit Dashboard for Anomaly Detection
==========================================
Real-time visualization of detected anomalies.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
from typing import Dict, List

# Page config
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# Database Connection
# ================================================================
@st.cache_resource
def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host='postgres',
        port=5432,
        database='airflow',
        user='airflow',
        password='EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S'
    )

@st.cache_data(ttl=60)
def fetch_anomalies(hours: int = 24) -> pd.DataFrame:
    """Fetch recent anomalies"""
    conn = get_db_connection()
    
    query = f"""
        SELECT 
            id,
            detected_at,
            dag_run_id,
            anomaly_score,
            anomaly_type,
            source_table,
            source_id,
            feature_values,
            created_at
        FROM anomaly_detections
        WHERE detected_at >= NOW() - INTERVAL '{hours} hours'
        ORDER BY detected_at DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

@st.cache_data(ttl=60)
def fetch_statistics() -> Dict:
    """Fetch anomaly statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Total anomalies
    cursor.execute("SELECT COUNT(*) FROM anomaly_detections")
    stats['total'] = cursor.fetchone()[0]
    
    # Last 24h
    cursor.execute("""
        SELECT COUNT(*) FROM anomaly_detections 
        WHERE detected_at >= NOW() - INTERVAL '24 hours'
    """)
    stats['last_24h'] = cursor.fetchone()[0]
    
    # Critical (score > 0.9)
    cursor.execute("""
        SELECT COUNT(*) FROM anomaly_detections 
        WHERE anomaly_score > 0.9 
        AND detected_at >= NOW() - INTERVAL '24 hours'
    """)
    stats['critical_24h'] = cursor.fetchone()[0]
    
    # Average score
    cursor.execute("""
        SELECT AVG(anomaly_score) FROM anomaly_detections 
        WHERE detected_at >= NOW() - INTERVAL '24 hours'
    """)
    result = cursor.fetchone()[0]
    stats['avg_score'] = float(result) if result else 0.0
    
    cursor.close()
    conn.close()
    
    return stats

# ================================================================
# Sidebar
# ================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/apache/airflow/main/docs/apache-airflow/img/logos/wordmark_1.png", width=200)
    
    st.title("🎯 Anomaly Detection")
    st.divider()
    
    # Time range selector
    time_range = st.selectbox(
        "📅 Time Range",
        options=[1, 6, 12, 24, 48, 72, 168],  # hours
        index=3,  # Default 24h
        format_func=lambda x: f"Last {x} hours" if x < 24 else f"Last {x//24} days"
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 120, 30)
    
    st.divider()
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    stats = fetch_statistics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Anomalies", f"{stats['total']:,}")
    with col2:
        st.metric("Last 24h", stats['last_24h'])
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Critical", stats['critical_24h'], 
                 delta="High" if stats['critical_24h'] > 0 else "None")
    with col4:
        st.metric("Avg Score", f"{stats['avg_score']:.3f}")

# ================================================================
# Main Dashboard
# ================================================================
st.title("🎯 Universal Anomaly Detection Dashboard")
st.markdown("### Real-time monitoring of ML-detected anomalies")

# Fetch data
anomalies_df = fetch_anomalies(hours=time_range)

if len(anomalies_df) == 0:
    st.info(f"ℹ️ No anomalies detected in the last {time_range} hours")
    st.balloons()
else:
    # ================================================================
    # Overview Metrics
    # ================================================================
    st.divider()
    st.subheader("📈 Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Anomalies",
            len(anomalies_df),
            delta=f"+{len(anomalies_df[anomalies_df['detected_at'] >= datetime.now() - timedelta(hours=1)])}" 
        )
    
    with col2:
        critical_count = len(anomalies_df[anomalies_df['anomaly_score'] > 0.9])
        st.metric(
            "🔴 Critical",
            critical_count,
            delta="Urgent" if critical_count > 0 else "Clear"
        )
    
    with col3:
        high_count = len(anomalies_df[(anomalies_df['anomaly_score'] > 0.8) & 
                                      (anomalies_df['anomaly_score'] <= 0.9)])
        st.metric("🟠 High", high_count)
    
    with col4:
        avg_score = anomalies_df['anomaly_score'].mean()
        st.metric("Average Score", f"{avg_score:.3f}")
    
    with col5:
        unique_sources = anomalies_df['source_table'].nunique()
        st.metric("Affected Tables", unique_sources)
    
    # ================================================================
    # Time Series Chart
    # ================================================================
    st.divider()
    st.subheader("📊 Anomaly Timeline")
    
    # Prepare time series data
    anomalies_df['hour'] = pd.to_datetime(anomalies_df['detected_at']).dt.floor('H')
    hourly_counts = anomalies_df.groupby('hour').size().reset_index(name='count')
    
    # Create chart
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=hourly_counts['hour'],
        y=hourly_counts['count'],
        mode='lines+markers',
        name='Anomalies per Hour',
        line=dict(color='#FF4B4B', width=2),
        marker=dict(size=8)
    ))
    
    fig_timeline.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Time",
        yaxis_title="Number of Anomalies",
        hovermode='x unified',
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # ================================================================
    # Distribution Charts
    # ================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Anomaly Score Distribution")
        
        fig_score = px.histogram(
            anomalies_df,
            x='anomaly_score',
            nbins=20,
            color_discrete_sequence=['#FF4B4B'],
            title="Score Distribution"
        )
        
        fig_score.update_layout(
            xaxis_title="Anomaly Score",
            yaxis_title="Count",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        st.subheader("📋 Anomaly Types")
        
        type_counts = anomalies_df['anomaly_type'].value_counts()
        
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Distribution by Type",
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        
        fig_types.update_layout(
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig_types, use_container_width=True)
    
    # ================================================================
    # Source Table Analysis
    # ================================================================
    st.divider()
    st.subheader("📊 Anomalies by Source Table")
    
    source_analysis = anomalies_df.groupby('source_table').agg({
        'id': 'count',
        'anomaly_score': ['mean', 'max']
    }).round(3)
    
    source_analysis.columns = ['Count', 'Avg Score', 'Max Score']
    source_analysis = source_analysis.sort_values('Count', ascending=False)
    
    fig_sources = px.bar(
        source_analysis.reset_index(),
        x='source_table',
        y='Count',
        color='Avg Score',
        color_continuous_scale='Reds',
        title="Anomalies per Source Table"
    )
    
    fig_sources.update_layout(
        xaxis_title="Source Table",
        yaxis_title="Number of Anomalies",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig_sources, use_container_width=True)
    
    # ================================================================
    # Recent Critical Anomalies
    # ================================================================
    st.divider()
    st.subheader("🔴 Recent Critical Anomalies (Score > 0.9)")
    
    critical_anomalies = anomalies_df[anomalies_df['anomaly_score'] > 0.9].head(10)
    
    if len(critical_anomalies) > 0:
        for idx, row in critical_anomalies.iterrows():
            with st.expander(
                f"🚨 Score: {row['anomaly_score']:.3f} | {row['anomaly_type']} | {row['source_table']} | {row['detected_at']}"
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    **Details:**
                    - **Score**: {row['anomaly_score']:.4f}
                    - **Type**: {row['anomaly_type']}
                    - **Source**: {row['source_table']}
                    - **Detected**: {row['detected_at']}
                    - **DAG Run**: {row['dag_run_id']}
                    """)
                
                with col2:
                    st.markdown("**Feature Values:**")
                    try:
                        import json
                        features = json.loads(row['feature_values'])
                        st.json(features)
                    except:
                        st.text(row['feature_values'])
    else:
        st.success("✅ No critical anomalies detected!")
    
    # ================================================================
    # All Anomalies Table
    # ================================================================
    st.divider()
    st.subheader("📋 All Detected Anomalies")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)
    
    with col2:
        selected_types = st.multiselect(
            "Anomaly Types",
            options=anomalies_df['anomaly_type'].unique().tolist(),
            default=anomalies_df['anomaly_type'].unique().tolist()
        )
    
    with col3:
        selected_tables = st.multiselect(
            "Source Tables",
            options=anomalies_df['source_table'].unique().tolist(),
            default=anomalies_df['source_table'].unique().tolist()
        )
    
    # Filter data
    filtered_df = anomalies_df[
        (anomalies_df['anomaly_score'] >= min_score) &
        (anomalies_df['anomaly_type'].isin(selected_types)) &
        (anomalies_df['source_table'].isin(selected_tables))
    ]
    
    # Display table
    display_columns = ['detected_at', 'anomaly_score', 'anomaly_type', 
                      'source_table', 'dag_run_id']
    
    st.dataframe(
        filtered_df[display_columns].sort_values('anomaly_score', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "detected_at": st.column_config.DatetimeColumn("Detected At", format="DD/MM/YY HH:mm"),
            "anomaly_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
        }
    )
    
    # Export button
    if st.button("📥 Export to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ================================================================
# Footer
# ================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎯 <strong>Universal Anomaly Detection</strong> | Powered by Apache Airflow + ML</p>
    <p><small>Last updated: {}</small></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# ================================================================
# Auto Refresh
# ================================================================
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()