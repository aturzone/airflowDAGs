"""
🚀 Airflow Crypto Price Monitor - Streamlit Dashboard
Real-time monitoring dashboard for crypto price tracking DAG
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from requests.auth import HTTPBasicAuth
import psycopg2
from typing import Dict, List, Optional
import logging

# ================================================================
# CONFIGURATION
# ================================================================
AIRFLOW_BASE_URL = "http://airflow-webserver:8080"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "QfUe9Bz3kZt2NsRb7hC8jXv9Lp6YwKdG"  # از .env بخون
DAG_ID = "crypto_price_monitor_fixed"

# Database config
DB_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'EKQH9jQX7gAfV7pLwVmsbLbF3XfY6n4S'
}

# Page config
st.set_page_config(
    page_title="Crypto Monitor Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# HELPER FUNCTIONS
# ================================================================

@st.cache_data(ttl=30)
def get_airflow_api(endpoint: str) -> Optional[Dict]:
    """Make API call to Airflow"""
    try:
        url = f"{AIRFLOW_BASE_URL}/api/v1/{endpoint}"
        response = requests.get(
            url,
            auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_dag_runs(limit: int = 10) -> Optional[List[Dict]]:
    """Get recent DAG runs"""
    data = get_airflow_api(f"dags/{DAG_ID}/dagRuns?limit={limit}")
    if data and 'dag_runs' in data:
        return data['dag_runs']
    return None

def get_task_instances(dag_run_id: str) -> Optional[List[Dict]]:
    """Get task instances for a specific DAG run"""
    data = get_airflow_api(f"dags/{DAG_ID}/dagRuns/{dag_run_id}/taskInstances")
    if data and 'task_instances' in data:
        return data['task_instances']
    return None

def get_xcom_value(dag_run_id: str, task_id: str, key: str = "return_value") -> Optional[Dict]:
    """Get XCom value from a specific task"""
    try:
        data = get_airflow_api(
            f"dags/{DAG_ID}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries/{key}"
        )
        if data and 'value' in data:
            return json.loads(data['value']) if isinstance(data['value'], str) else data['value']
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def get_crypto_prices_from_db(limit: int = 100) -> pd.DataFrame:
    """Get crypto prices from PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = f"""
        SELECT 
            coin_id,
            symbol,
            price_usd,
            price_eur,
            change_24h,
            volume_24h,
            created_at
        FROM crypto_prices
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_alerts_from_db(limit: int = 50) -> pd.DataFrame:
    """Get price alerts from database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = f"""
        SELECT 
            coin_id,
            symbol,
            alert_type,
            threshold_value,
            current_value,
            message,
            created_at
        FROM price_alerts
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

def get_state_color(state: str) -> str:
    """Get color for task state"""
    colors = {
        'success': '🟢',
        'running': '🔵',
        'failed': '🔴',
        'upstream_failed': '🟠',
        'skipped': '⚪',
        'up_for_retry': '🟡',
        'queued': '⚫'
    }
    return colors.get(state, '⚪')

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/apache/airflow/main/docs/apache-airflow/img/logos/wordmark_1.png", width=200)
    st.title("⚙️ Dashboard Settings")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 120, 30)
    
    st.divider()
    
    # DAG Controls
    st.subheader("🎮 DAG Controls")
    
    if st.button("▶️ Trigger DAG", use_container_width=True):
        try:
            response = requests.post(
                f"{AIRFLOW_BASE_URL}/api/v1/dags/{DAG_ID}/dagRuns",
                auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                json={"conf": {}}
            )
            if response.status_code == 200:
                st.success("✅ DAG triggered successfully!")
            else:
                st.error(f"❌ Failed to trigger DAG: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    dag_runs = get_dag_runs(limit=100)
    if dag_runs:
        total_runs = len(dag_runs)
        success_runs = len([r for r in dag_runs if r.get('state') == 'success'])
        failed_runs = len([r for r in dag_runs if r.get('state') == 'failed'])
        
        st.metric("Total Runs", total_runs)
        st.metric("Success Rate", f"{(success_runs/total_runs*100):.1f}%" if total_runs > 0 else "0%")
        st.metric("Failed Runs", failed_runs)

# ================================================================
# MAIN DASHBOARD
# ================================================================
st.title("💰 Crypto Price Monitor Dashboard")
st.markdown("### Real-time monitoring of cryptocurrency prices via Apache Airflow")

# Status Bar
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌐 Airflow Status",
        value="Online" if get_airflow_api("health") else "Offline",
        delta="Healthy" if get_airflow_api("health") else "Error"
    )

with col2:
    dag_runs = get_dag_runs(limit=1)
    if dag_runs:
        last_state = dag_runs[0].get('state', 'unknown')
        st.metric(
            label="📊 Last Run",
            value=last_state.title(),
            delta=get_state_color(last_state)
        )

with col3:
    df_prices = get_crypto_prices_from_db(limit=1)
    if not df_prices.empty:
        last_update = df_prices['created_at'].iloc[0]
        time_diff = datetime.now() - pd.to_datetime(last_update)
        st.metric(
            label="⏰ Last Update",
            value=f"{time_diff.seconds // 60}m ago",
            delta=last_update.strftime("%H:%M:%S")
        )

with col4:
    df_alerts = get_alerts_from_db(limit=100)
    new_alerts = len(df_alerts[df_alerts['created_at'] > datetime.now() - timedelta(hours=24)])
    st.metric(
        label="🚨 Alerts (24h)",
        value=new_alerts,
        delta="New" if new_alerts > 0 else "None"
    )

st.divider()

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Prices", 
    "🔄 DAG Runs", 
    "📋 Latest Report", 
    "🚨 Alerts",
    "📊 Analytics"
])

# ----------------------------------------------------------------
# TAB 1: LIVE PRICES
# ----------------------------------------------------------------
with tab1:
    st.header("💹 Current Cryptocurrency Prices")
    
    df_prices = get_crypto_prices_from_db(limit=8)
    
    if not df_prices.empty:
        # Latest prices for each coin
        latest_prices = df_prices.sort_values('created_at', ascending=False).groupby('symbol').first().reset_index()
        
        # Display as cards
        cols = st.columns(4)
        for idx, row in latest_prices.iterrows():
            with cols[idx % 4]:
                change_color = "🟢" if row['change_24h'] > 0 else "🔴"
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 10px 0;">
                    <h3 style="margin:0; color: white;">{row['symbol']}</h3>
                    <h2 style="margin:5px 0; color: white;">${row['price_usd']:,.2f}</h2>
                    <p style="margin:0; color: white;">{change_color} {row['change_24h']:.2f}% (24h)</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Price history chart
        st.subheader("📊 Price History")
        
        # Get more historical data
        df_history = get_crypto_prices_from_db(limit=200)
        
        if not df_history.empty:
            # Create interactive chart
            fig = go.Figure()
            
            for symbol in df_history['symbol'].unique():
                df_coin = df_history[df_history['symbol'] == symbol].sort_values('created_at')
                fig.add_trace(go.Scatter(
                    x=df_coin['created_at'],
                    y=df_coin['price_usd'],
                    mode='lines+markers',
                    name=symbol,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title="Cryptocurrency Price Trends",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No price data available yet. Wait for the first DAG run to complete.")

# ----------------------------------------------------------------
# TAB 2: DAG RUNS
# ----------------------------------------------------------------
with tab2:
    st.header("🔄 Recent DAG Executions")
    
    dag_runs = get_dag_runs(limit=20)
    
    if dag_runs:
        for run in dag_runs:
            run_id = run['dag_run_id']
            state = run.get('state', 'unknown')
            start_date = run.get('start_date', 'N/A')
            end_date = run.get('end_date', 'N/A')
            
            # Expandable section for each run
            with st.expander(f"{get_state_color(state)} {run_id} - {state.upper()}", expanded=(run == dag_runs[0])):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Run ID:** `{run_id}`  
                    **State:** {state.upper()}  
                    **Started:** {start_date}  
                    **Ended:** {end_date if end_date != 'N/A' else 'Running...'}
                    """)
                
                with col2:
                    if state == 'success':
                        st.success("✅ Completed Successfully")
                    elif state == 'failed':
                        st.error("❌ Failed")
                    elif state == 'running':
                        st.info("🔵 Running...")
                    else:
                        st.warning(f"⚠️ {state.title()}")
                
                # Task instances
                tasks = get_task_instances(run_id)
                if tasks:
                    st.markdown("#### Tasks:")
                    
                    task_df = pd.DataFrame([{
                        'Task': t['task_id'],
                        'State': f"{get_state_color(t.get('state', 'unknown'))} {t.get('state', 'unknown').upper()}",
                        'Start': t.get('start_date', 'N/A'),
                        'Duration': t.get('duration', 'N/A')
                    } for t in tasks])
                    
                    st.dataframe(task_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No DAG runs found. Trigger the DAG to start monitoring.")

# ----------------------------------------------------------------
# TAB 3: LATEST REPORT
# ----------------------------------------------------------------
with tab3:
    st.header("📋 Latest Execution Report")
    
    dag_runs = get_dag_runs(limit=1)
    
    if dag_runs and dag_runs[0].get('state') == 'success':
        run_id = dag_runs[0]['dag_run_id']
        
        # Get final_report XCom
        report = get_xcom_value(run_id, 'final_report')
        
        if report:
            # Display report in nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Execution Summary")
                st.json(report)
            
            with col2:
                st.subheader("📈 Metrics")
                
                # Status indicators
                if report.get('overall_status') == 'completed':
                    st.success("✅ All tasks completed successfully")
                else:
                    st.warning("⚠️ Completed with errors")
                
                # Metrics
                st.metric("Test Connection", report.get('test_connection', 'unknown').upper())
                st.metric("Price Fetch", f"{report.get('price_fetch_count', 0)} coins")
                st.metric("Records Saved", report.get('records_saved', 0))
                
                # Timeline
                st.markdown("#### ⏱️ Execution Timeline")
                st.text(f"Date: {report.get('execution_date', 'N/A')}")
                st.text(f"Time: {report.get('execution_time', 'N/A')}")
        else:
            st.info("ℹ️ No report data available for this run.")
    else:
        st.warning("⚠️ No successful DAG runs found yet.")

# ----------------------------------------------------------------
# TAB 4: ALERTS
# ----------------------------------------------------------------
with tab4:
    st.header("🚨 Price Alerts")
    
    df_alerts = get_alerts_from_db(limit=100)
    
    if not df_alerts.empty:
        # Alert statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_alerts = len(df_alerts)
            st.metric("Total Alerts", total_alerts)
        
        with col2:
            alerts_24h = len(df_alerts[df_alerts['created_at'] > datetime.now() - timedelta(hours=24)])
            st.metric("Last 24h", alerts_24h)
        
        with col3:
            unique_coins = df_alerts['symbol'].nunique()
            st.metric("Affected Coins", unique_coins)
        
        st.divider()
        
        # Recent alerts
        st.subheader("📋 Recent Alerts")
        
        for idx, alert in df_alerts.head(10).iterrows():
            alert_type = "🚨 CRITICAL" if alert['current_value'] > 10 else "⚠️ WARNING"
            
            st.markdown(f"""
            <div style="padding: 15px; border-left: 4px solid {'#ff4444' if alert['current_value'] > 10 else '#ffaa00'}; background: #f0f0f0; margin: 10px 0; border-radius: 5px;">
                <strong>{alert_type} {alert['symbol']}</strong><br>
                {alert['message']}<br>
                <small style="color: #666;">📅 {alert['created_at']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No alerts triggered yet. Alerts will appear when price changes exceed the threshold.")

# ----------------------------------------------------------------
# TAB 5: ANALYTICS
# ----------------------------------------------------------------
with tab5:
    st.header("📊 Analytics & Insights")
    
    df_prices = get_crypto_prices_from_db(limit=500)
    
    if not df_prices.empty:
        # Volume analysis
        st.subheader("💰 24h Trading Volume")
        
        volume_by_coin = df_prices.groupby('symbol')['volume_24h'].mean().sort_values(ascending=False)
        
        fig_volume = px.bar(
            x=volume_by_coin.index,
            y=volume_by_coin.values,
            labels={'x': 'Cryptocurrency', 'y': 'Average Volume (USD)'},
            title="Average 24h Trading Volume",
            color=volume_by_coin.values,
            color_continuous_scale='Viridis'
        )
        fig_volume.update_layout(showlegend=False, template="plotly_dark", height=400)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Price change distribution
        st.subheader("📈 24h Price Change Distribution")
        
        latest_changes = df_prices.sort_values('created_at', ascending=False).groupby('symbol')['change_24h'].first()
        
        fig_change = px.bar(
            x=latest_changes.index,
            y=latest_changes.values,
            labels={'x': 'Cryptocurrency', 'y': '24h Change (%)'},
            title="Latest 24h Price Changes",
            color=latest_changes.values,
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig_change.update_layout(showlegend=False, template="plotly_dark", height=400)
        st.plotly_chart(fig_change, use_container_width=True)
        
        # Data table
        st.subheader("📋 Detailed Data Table")
        latest_all = df_prices.sort_values('created_at', ascending=False).groupby('symbol').first().reset_index()
        st.dataframe(
            latest_all[['symbol', 'price_usd', 'price_eur', 'change_24h', 'volume_24h', 'created_at']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ Not enough data for analytics. Run the DAG multiple times to see trends.")

# ================================================================
# FOOTER
# ================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>💰 <strong>Crypto Price Monitor Dashboard</strong> | Powered by Apache Airflow + Streamlit</p>
    <p><small>Last updated: {}</small></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# ================================================================
# AUTO REFRESH
# ================================================================
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()