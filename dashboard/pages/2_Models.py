"""
Models Page - ML Model Management
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Models - AnomalyGuard", page_icon="🤖", layout="wide")

st.title("🤖 Model Management")

# Get managers from session state
model_manager = st.session_state.get('model_manager')
database_manager = st.session_state.get('database_manager')
config_manager = st.session_state.get('config_manager')

if not model_manager or not database_manager:
    st.error("Model or Database manager not initialized")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎯 Active Models",
    "📚 Model Versions",
    "⚖️ Model Comparison",
    "⚙️ Settings"
])

# Tab 1: Overview
with tab1:
    st.header("Model Overview")

    # Get model summary
    summary = model_manager.get_models_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Models", summary.get('total_registered', 0))
    col2.metric("Active Models", summary.get('active_models', 0))
    col3.metric("Total Files", summary.get('total_files', 0))
    col4.metric("Total Size", f"{summary.get('total_size_mb', 0):.2f} MB")

    st.markdown("---")

    # Model breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Types")
        type_data = {
            'Type': ['Isolation Forest', 'Autoencoder'],
            'Count': [
                summary.get('isolation_forest_files', 0),
                summary.get('autoencoder_files', 0)
            ]
        }
        fig = px.pie(
            type_data,
            values='Count',
            names='Type',
            title='Models by Type',
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Status")
        status_data = {
            'Status': ['Active', 'Inactive'],
            'Count': [
                summary.get('active_models', 0),
                summary.get('inactive_models', 0)
            ]
        }
        fig = px.pie(
            status_data,
            values='Count',
            names='Status',
            title='Models by Status',
            color_discrete_sequence=['#28a745', '#6c757d']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # File list
    st.subheader("📁 Model Files")

    model_files = model_manager.list_model_files()

    if model_files:
        df_files = pd.DataFrame(model_files)
        df_files['modified'] = pd.to_datetime(df_files['modified'])

        st.dataframe(
            df_files[['filename', 'type', 'size_mb', 'modified']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No model files found")

# Tab 2: Active Models
with tab2:
    st.header("Active Models")

    active_models = model_manager.get_latest_active_models()

    if active_models:
        for model_type, model_data in active_models.items():
            st.subheader(f"{model_type.replace('_', ' ').title()}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model ID", model_data.get('model_id', 'N/A')[:12] + '...')

            with col2:
                st.metric("Version", model_data.get('version', 'N/A'))

            with col3:
                trained_at = model_data.get('trained_at')
                if trained_at:
                    if isinstance(trained_at, str):
                        trained_at = pd.to_datetime(trained_at)
                    st.metric("Trained", trained_at.strftime('%Y-%m-%d'))
                else:
                    st.metric("Trained", "N/A")

            with col4:
                metrics = model_data.get('metrics', {})
                if isinstance(metrics, str):
                    import json
                    try:
                        metrics = json.loads(metrics)
                    except:
                        metrics = {}
                f1_score = metrics.get('f1_score', 0) if metrics else 0
                st.metric("F1 Score", f"{f1_score:.3f}")

            # Show metrics details
            if metrics:
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Precision:** {metrics.get('precision', 0):.3f}")
                col2.write(f"**Recall:** {metrics.get('recall', 0):.3f}")
                col3.write(f"**Threshold:** {model_data.get('threshold', 0):.3f}")

            st.markdown("---")
    else:
        st.info("No active models found")

    # Quick actions
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("🚀 Trigger Training DAG", use_container_width=True):
            airflow_manager = st.session_state.get('airflow_manager')
            if airflow_manager:
                success, msg = airflow_manager.trigger_training_dag()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.error("Airflow manager not available")

# Tab 3: Model Versions
with tab3:
    st.header("Model Versions")

    model_type_filter = st.selectbox(
        "Select Model Type",
        ["isolation_forest", "autoencoder", "ensemble"]
    )

    versions = model_manager.get_model_versions(model_type_filter)

    if versions:
        df_versions = pd.DataFrame(versions)

        # Display as table
        st.dataframe(
            df_versions[[
                'model_id', 'version', 'trained_at', 'status', 'threshold'
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Version selection
        st.subheader("Manage Version")

        selected_model_id = st.selectbox(
            "Select a model version",
            df_versions['model_id'].tolist(),
            format_func=lambda x: f"{x[:12]}... ({df_versions[df_versions['model_id']==x].iloc[0]['status']})"
        )

        if selected_model_id:
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("✅ Activate", use_container_width=True):
                    success, msg = model_manager.activate_model_version(selected_model_id)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            with col2:
                if st.button("📦 Archive", use_container_width=True):
                    success = database_manager.update_model_status(selected_model_id, 'archived')
                    if success:
                        st.success("Model archived successfully")
                        st.rerun()
                    else:
                        st.error("Failed to archive model")

            with col3:
                if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                    # Show confirmation
                    st.warning("Delete functionality requires additional confirmation")
    else:
        st.info(f"No {model_type_filter} models found")

# Tab 4: Model Comparison
with tab4:
    st.header("Compare Models")

    all_models = database_manager.get_models()

    if not all_models.empty:
        # Select models to compare
        model_ids = all_models['model_id'].tolist()

        selected_models = st.multiselect(
            "Select models to compare (up to 5)",
            model_ids,
            max_selections=5,
            format_func=lambda x: f"{x[:12]}... ({all_models[all_models['model_id']==x].iloc[0]['model_type']})"
        )

        if selected_models and len(selected_models) >= 2:
            comparison_df = model_manager.compare_models(selected_models)

            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Performance comparison chart
                st.subheader("Performance Comparison")

                # Extract metrics for chart
                chart_data = []
                for _, row in comparison_df.iterrows():
                    metrics = row.get('metrics', {})
                    if isinstance(metrics, str):
                        import json
                        try:
                            metrics = json.loads(metrics)
                        except:
                            metrics = {}

                    chart_data.append({
                        'Model': row['model_id'][:12],
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1 Score': metrics.get('f1_score', 0)
                    })

                if chart_data:
                    df_chart = pd.DataFrame(chart_data)
                    df_melted = df_chart.melt(
                        id_vars=['Model'],
                        var_name='Metric',
                        value_name='Score'
                    )

                    fig = px.bar(
                        df_melted,
                        x='Model',
                        y='Score',
                        color='Metric',
                        barmode='group',
                        title='Model Performance Metrics',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 models to compare")
    else:
        st.info("No models available for comparison")

# Tab 5: Settings
with tab5:
    st.header("Model Settings")

    # Ensemble Weights
    st.subheader("⚖️ Ensemble Weights")

    current_weights = model_manager.get_ensemble_weights()

    col1, col2, col3 = st.columns(3)

    with col1:
        stat_weight = st.slider(
            "Statistical Layer",
            0.0, 1.0,
            current_weights.get('statistical', 0.2),
            0.01
        )

    with col2:
        if_weight = st.slider(
            "Isolation Forest",
            0.0, 1.0,
            current_weights.get('isolation_forest', 0.4),
            0.01
        )

    with col3:
        ae_weight = st.slider(
            "Autoencoder",
            0.0, 1.0,
            current_weights.get('autoencoder', 0.4),
            0.01
        )

    total_weight = stat_weight + if_weight + ae_weight

    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
    else:
        st.success(f"✅ Weights sum to {total_weight:.2f}")

    if st.button("💾 Update Ensemble Weights", use_container_width=True):
        new_weights = {
            'statistical': stat_weight,
            'isolation_forest': if_weight,
            'autoencoder': ae_weight
        }

        if abs(sum(new_weights.values()) - 1.0) <= 0.01:
            success = model_manager.update_ensemble_weights(new_weights)
            if success:
                st.success("Ensemble weights updated successfully")
                st.rerun()
            else:
                st.error("Failed to update ensemble weights")
        else:
            st.error("Weights must sum to 1.0")

    st.markdown("---")

    # Risk Thresholds
    st.subheader("🎯 Risk Thresholds")

    current_thresholds = model_manager.get_risk_thresholds()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        low_threshold = st.number_input(
            "Low (0-X)",
            0, 100,
            current_thresholds.get('low', 30),
            1
        )

    with col2:
        medium_threshold = st.number_input(
            "Medium (X-Y)",
            0, 100,
            current_thresholds.get('medium', 60),
            1
        )

    with col3:
        high_threshold = st.number_input(
            "High (Y-Z)",
            0, 100,
            current_thresholds.get('high', 80),
            1
        )

    with col4:
        critical_threshold = st.number_input(
            "Critical (Z-100)",
            0, 100,
            current_thresholds.get('critical', 90),
            1
        )

    if not (low_threshold < medium_threshold < high_threshold < critical_threshold):
        st.warning("⚠️ Thresholds must be in ascending order: Low < Medium < High < Critical")
    else:
        st.success("✅ Thresholds are valid")

    if st.button("💾 Update Risk Thresholds", use_container_width=True):
        new_thresholds = {
            'low': low_threshold,
            'medium': medium_threshold,
            'high': high_threshold,
            'critical': critical_threshold
        }

        if low_threshold < medium_threshold < high_threshold < critical_threshold:
            success = model_manager.update_risk_thresholds(new_thresholds)
            if success:
                st.success("Risk thresholds updated successfully")
                st.rerun()
            else:
                st.error("Failed to update risk thresholds")
        else:
            st.error("Thresholds must be in ascending order")
