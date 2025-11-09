"""
DAGs Page - Airflow DAG Management
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="DAGs - AnomalyGuard", page_icon="⚙️", layout="wide")

st.title("⚙️ DAG Management")

airflow_manager = st.session_state.get('airflow_manager')
config_manager = st.session_state.get('config_manager')

if not airflow_manager:
    st.error("Airflow manager not initialized")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚂 Training DAG",
    "🔍 Detection DAG",
    "⚙️ Configuration"
])

# Tab 1: Overview
with tab1:
    st.header("DAG Overview")

    # Get DAG statistics
    dag_stats = airflow_manager.get_dag_statistics(['train_ensemble_models', 'ensemble_anomaly_detection'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total DAGs", dag_stats.get('total_dags', 0))
    col2.metric("Active DAGs", dag_stats.get('active_dags', 0))
    col3.metric("Running", dag_stats.get('running_dag_runs', 0))
    col4.metric("Success Rate", f"{dag_stats.get('success_rate', 0):.1f}%")

    st.markdown("---")

    # Get all DAGs
    success, dags = airflow_manager.get_dags()

    if success and dags:
        st.subheader("All DAGs")

        dag_data = []
        for dag in dags:
            if dag['dag_id'] in ['train_ensemble_models', 'ensemble_anomaly_detection']:
                dag_data.append({
                    'DAG ID': dag.get('dag_id', 'N/A'),
                    'Status': '✅ Active' if not dag.get('is_paused', True) else '⏸️ Paused',
                    'Schedule': dag.get('schedule_interval', 'N/A'),
                    'Last Run': dag.get('last_parsed_time', 'N/A'),
                    'Next Run': dag.get('next_dagrun', 'N/A')
                })

        if dag_data:
            st.dataframe(pd.DataFrame(dag_data), use_container_width=True, hide_index=True)
    else:
        st.error("Failed to fetch DAGs")

# Tab 2: Training DAG
with tab2:
    st.header("Training DAG Control")

    training_status = airflow_manager.get_training_dag_status()

    if 'error' not in training_status:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Status")
            is_paused = training_status.get('is_paused', True)
            st.write(f"**Status:** {'⏸️ Paused' if is_paused else '✅ Active'}")
            st.write(f"**Next Run:** {training_status.get('next_dagrun', 'N/A')}")

        with col2:
            st.subheader("Controls")

            if is_paused:
                if st.button("▶️ Unpause DAG", use_container_width=True):
                    success, msg = airflow_manager.unpause_dag('train_ensemble_models')
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                if st.button("⏸️ Pause DAG", use_container_width=True):
                    success, msg = airflow_manager.pause_dag('train_ensemble_models')
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("---")

        # Trigger with parameters
        st.subheader("Trigger Training")

        col1, col2 = st.columns(2)

        with col1:
            training_days = st.number_input(
                "Training Window (days)",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of days of historical data to use for training"
            )

        with col2:
            min_samples = st.number_input(
                "Minimum Samples",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Minimum number of samples required for training"
            )

        if st.button("🚀 Trigger Training DAG", use_container_width=True, type="primary"):
            success, msg = airflow_manager.trigger_training_dag(
                training_days=training_days,
                min_samples=min_samples
            )

            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown("---")

        # Recent runs
        st.subheader("Recent Runs")

        recent_runs = training_status.get('recent_runs', [])

        if recent_runs:
            run_data = []
            for run in recent_runs[:10]:
                run_data.append({
                    'Run ID': run.get('dag_run_id', 'N/A'),
                    'State': run.get('state', 'N/A'),
                    'Started': run.get('start_date', 'N/A'),
                    'Ended': run.get('end_date', 'N/A'),
                })

            st.dataframe(pd.DataFrame(run_data), use_container_width=True, hide_index=True)
        else:
            st.info("No recent runs")
    else:
        st.error(training_status.get('error'))

# Tab 3: Detection DAG
with tab3:
    st.header("Detection DAG Control")

    detection_status = airflow_manager.get_detection_dag_status()

    if 'error' not in detection_status:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Status")
            is_paused = detection_status.get('is_paused', True)
            st.write(f"**Status:** {'⏸️ Paused' if is_paused else '✅ Active'}")
            st.write(f"**Next Run:** {detection_status.get('next_dagrun', 'N/A')}")

        with col2:
            st.subheader("Controls")

            if is_paused:
                if st.button("▶️ Unpause DAG", key="unpause_detection", use_container_width=True):
                    success, msg = airflow_manager.unpause_dag('ensemble_anomaly_detection')
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                if st.button("⏸️ Pause DAG", key="pause_detection", use_container_width=True):
                    success, msg = airflow_manager.pause_dag('ensemble_anomaly_detection')
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("---")

        # Trigger with parameters
        st.subheader("Trigger Detection")

        lookback_hours = st.number_input(
            "Lookback Hours",
            min_value=1,
            max_value=168,
            value=1,
            help="Number of hours to look back for new transactions"
        )

        if st.button("🚀 Trigger Detection DAG", use_container_width=True, type="primary"):
            success, msg = airflow_manager.trigger_detection_dag(
                lookback_hours=lookback_hours
            )

            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown("---")

        # Recent runs
        st.subheader("Recent Runs")

        recent_runs = detection_status.get('recent_runs', [])

        if recent_runs:
            run_data = []
            for run in recent_runs[:10]:
                run_data.append({
                    'Run ID': run.get('dag_run_id', 'N/A'),
                    'State': run.get('state', 'N/A'),
                    'Started': run.get('start_date', 'N/A'),
                    'Ended': run.get('end_date', 'N/A'),
                })

            st.dataframe(pd.DataFrame(run_data), use_container_width=True, hide_index=True)
        else:
            st.info("No recent runs")
    else:
        st.error(detection_status.get('error'))

# Tab 4: Configuration
with tab4:
    st.header("DAG Configuration")

    # Training DAG config
    st.subheader("🚂 Training DAG")

    training_config = config_manager.get('airflow.dags.training', {})

    col1, col2 = st.columns(2)

    with col1:
        train_schedule = st.text_input(
            "Schedule (cron)",
            value=training_config.get('schedule', '0 2 * * 0'),
            help="Cron expression for training schedule"
        )

    with col2:
        train_enabled = st.checkbox(
            "Enabled",
            value=training_config.get('enabled', True)
        )

    if st.button("💾 Update Training Config", use_container_width=True):
        config_manager.set('airflow.dags.training.schedule', train_schedule)
        config_manager.set('airflow.dags.training.enabled', train_enabled)

        if config_manager.save_config():
            st.success("Training DAG configuration updated")
        else:
            st.error("Failed to update configuration")

    st.markdown("---")

    # Detection DAG config
    st.subheader("🔍 Detection DAG")

    detection_config = config_manager.get('airflow.dags.detection', {})

    col1, col2 = st.columns(2)

    with col1:
        detect_schedule = st.text_input(
            "Schedule (cron)",
            value=detection_config.get('schedule', '0 * * * *'),
            help="Cron expression for detection schedule"
        )

    with col2:
        detect_enabled = st.checkbox(
            "Enabled",
            value=detection_config.get('enabled', True),
            key="detect_enabled"
        )

    if st.button("💾 Update Detection Config", use_container_width=True):
        config_manager.set('airflow.dags.detection.schedule', detect_schedule)
        config_manager.set('airflow.dags.detection.enabled', detect_enabled)

        if config_manager.save_config():
            st.success("Detection DAG configuration updated")
        else:
            st.error("Failed to update configuration")
