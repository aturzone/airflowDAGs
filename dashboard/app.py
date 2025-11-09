"""
AnomalyGuard Control Center
Main Streamlit Dashboard Application
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dashboard.utils.config_manager import ConfigManager
from dashboard.utils.database_manager import DatabaseManager
from dashboard.utils.airflow_manager import AirflowManager
from dashboard.utils.docker_manager import DockerManager
from dashboard.utils.model_manager import ModelManager


# Page configuration
st.set_page_config(
    page_title="AnomalyGuard Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.875rem;
    }
    .status-success {
        background-color: #28a745;
        color: white;
    }
    .status-warning {
        background-color: #ffc107;
        color: black;
    }
    .status-danger {
        background-color: #dc3545;
        color: white;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'config_manager' not in st.session_state:
        try:
            st.session_state.config_manager = ConfigManager()
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            st.stop()

    if 'database_manager' not in st.session_state:
        try:
            st.session_state.database_manager = DatabaseManager(st.session_state.config_manager)
        except Exception as e:
            st.warning(f"Database connection not available: {e}")
            st.session_state.database_manager = None

    if 'airflow_manager' not in st.session_state:
        try:
            st.session_state.airflow_manager = AirflowManager(st.session_state.config_manager)
        except Exception as e:
            st.warning(f"Airflow connection not available: {e}")
            st.session_state.airflow_manager = None

    if 'docker_manager' not in st.session_state:
        try:
            st.session_state.docker_manager = DockerManager(st.session_state.config_manager)
        except Exception as e:
            st.warning(f"Docker connection not available: {e}")
            st.session_state.docker_manager = None

    if 'model_manager' not in st.session_state:
        try:
            st.session_state.model_manager = ModelManager(
                st.session_state.config_manager,
                st.session_state.database_manager
            )
        except Exception as e:
            st.warning(f"Model manager not available: {e}")
            st.session_state.model_manager = None


# Initialize
init_session_state()


# Sidebar
with st.sidebar:
    st.markdown("# 🛡️ AnomalyGuard")
    st.markdown("### Control Center")

    st.markdown("---")

    # System Status Overview
    st.markdown("### 🔍 System Status")

    # Check ClickHouse
    if st.session_state.database_manager:
        ch_status, ch_msg = st.session_state.database_manager.test_clickhouse_connection()
        if ch_status:
            st.success("✅ ClickHouse")
        else:
            st.error("❌ ClickHouse")
    else:
        st.warning("⚠️ ClickHouse")

    # Check Airflow
    if st.session_state.airflow_manager:
        af_status, af_data = st.session_state.airflow_manager.health_check()
        if af_status:
            st.success("✅ Airflow")
        else:
            st.error("❌ Airflow")
    else:
        st.warning("⚠️ Airflow")

    # Check Docker
    if st.session_state.docker_manager:
        if st.session_state.docker_manager.is_docker_available():
            st.success("✅ Docker")
        else:
            st.error("❌ Docker")
    else:
        st.warning("⚠️ Docker")

    st.markdown("---")

    # Navigation
    st.markdown("### 📍 Navigation")

    pages = {
        "📊 Dashboard": "pages/1_Dashboard.py",
        "🤖 Models": "pages/2_Models.py",
        "⚙️ DAGs": "pages/3_DAGs.py",
        "💾 Data": "pages/4_Data.py",
        "🔍 Anomalies": "pages/5_Anomalies.py",
        "🎛️ Configuration": "pages/6_Configuration.py",
        "🐳 Services": "pages/7_Services.py",
        "📈 Analytics": "pages/8_Analytics.py"
    }

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()

    with col2:
        if st.button("⚙️ Settings"):
            st.switch_page("pages/6_Configuration.py")

    st.markdown("---")

    # Info
    st.markdown("### ℹ️ Info")
    st.caption(f"Version: 1.0.0")
    st.caption(f"Powered by Streamlit")


# Main content
st.markdown('<h1 class="main-header">🛡️ AnomalyGuard Control Center</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome to the **AnomalyGuard Control Center** - your centralized dashboard for managing
cryptocurrency transaction anomaly detection system.
""")

st.markdown("---")

# Overview Metrics
st.markdown("## 📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

# Get statistics
try:
    if st.session_state.database_manager:
        # Transaction count
        tx_count = st.session_state.database_manager.get_transaction_count()
        col1.metric("Total Transactions", f"{tx_count:,}")

        # Anomaly stats
        anomaly_stats = st.session_state.database_manager.get_anomaly_stats()
        total_anomalies = sum(anomaly_stats.values())
        col2.metric("Total Anomalies", f"{total_anomalies:,}")

        # Model count
        if st.session_state.model_manager:
            model_summary = st.session_state.model_manager.get_models_summary()
            col3.metric("Active Models", model_summary.get('active_models', 0))
        else:
            col3.metric("Active Models", "N/A")

        # DAG status
        if st.session_state.airflow_manager:
            dag_stats = st.session_state.airflow_manager.get_dag_statistics()
            col4.metric("Active DAGs", dag_stats.get('active_dags', 0))
        else:
            col4.metric("Active DAGs", "N/A")
    else:
        col1.metric("Total Transactions", "N/A")
        col2.metric("Total Anomalies", "N/A")
        col3.metric("Active Models", "N/A")
        col4.metric("Active DAGs", "N/A")

except Exception as e:
    st.error(f"Error loading system overview: {e}")

st.markdown("---")

# Quick Links
st.markdown("## 🚀 Quick Links")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 📊 Dashboard")
    st.markdown("View real-time monitoring and metrics")
    if st.button("Go to Dashboard →"):
        st.switch_page("pages/1_Dashboard.py")

with col2:
    st.markdown("### 🤖 Models")
    st.markdown("Manage ML models and versions")
    if st.button("Go to Models →"):
        st.switch_page("pages/2_Models.py")

with col3:
    st.markdown("### ⚙️ DAGs")
    st.markdown("Control Airflow DAGs")
    if st.button("Go to DAGs →"):
        st.switch_page("pages/3_DAGs.py")

with col4:
    st.markdown("### 💾 Data")
    st.markdown("Browse and manage data")
    if st.button("Go to Data →"):
        st.switch_page("pages/4_Data.py")

st.markdown("---")

# Recent Activity
st.markdown("## 📋 Recent Activity")

try:
    if st.session_state.database_manager:
        # Recent anomalies
        recent_anomalies = st.session_state.database_manager.get_anomalies(limit=5)

        if not recent_anomalies.empty:
            st.markdown("### 🔍 Latest Anomalies")

            for _, anomaly in recent_anomalies.iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

                with col1:
                    st.write(f"**Transaction:** {anomaly.get('transaction_id', 'N/A')}")

                with col2:
                    st.write(f"**User:** {anomaly.get('user_id', 'N/A')}")

                with col3:
                    risk = anomaly.get('risk_level', 'unknown')
                    risk_colors = {
                        'low': '🟢',
                        'medium': '🟡',
                        'high': '🟠',
                        'critical': '🔴'
                    }
                    st.write(f"{risk_colors.get(risk, '⚪')} {risk.upper()}")

                with col4:
                    score = anomaly.get('ensemble_score', 0)
                    st.write(f"**Score:** {score:.1f}")

            if st.button("View All Anomalies →"):
                st.switch_page("pages/5_Anomalies.py")
        else:
            st.info("No anomalies detected yet")
    else:
        st.warning("Database connection not available")

except Exception as e:
    st.error(f"Error loading recent activity: {e}")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>AnomalyGuard v1.0.0 | Ensemble Anomaly Detection System</p>
    <p>Powered by Apache Airflow, ClickHouse, and Machine Learning</p>
</div>
""", unsafe_allow_html=True)
