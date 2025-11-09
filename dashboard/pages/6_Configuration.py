"""
Configuration Page - System Configuration Management
"""

import streamlit as st
import yaml
from datetime import datetime

st.set_page_config(page_title="Configuration - AnomalyGuard", page_icon="🎛️", layout="wide")

st.title("🎛️ System Configuration")

config_manager = st.session_state.get('config_manager')

if not config_manager:
    st.error("Configuration manager not initialized")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔌 Databases",
    "⚙️ Airflow",
    "📦 Export/Import",
    "✅ Validation"
])

# Tab 1: Database Configuration
with tab1:
    st.header("Database Configuration")

    # ClickHouse
    st.subheader("ClickHouse")

    ch_config = config_manager.get_clickhouse_config()

    col1, col2, col3 = st.columns(3)

    with col1:
        ch_host = st.text_input("Host", value=ch_config.get('host', 'clickhouse'))
        ch_port = st.number_input("Port", value=ch_config.get('port', 8123))

    with col2:
        ch_database = st.text_input("Database", value=ch_config.get('database', 'analytics'))
        ch_user = st.text_input("User", value=ch_config.get('user', 'airflow'))

    with col3:
        ch_password = st.text_input("Password", value=ch_config.get('password', ''), type="password")
        ch_enabled = st.checkbox("Enabled", value=ch_config.get('enabled', True))

    if st.button("💾 Update ClickHouse Config", use_container_width=True):
        new_config = {
            'host': ch_host,
            'port': ch_port,
            'database': ch_database,
            'user': ch_user,
            'password': ch_password,
            'enabled': ch_enabled
        }

        success = config_manager.update_database_connection('clickhouse', new_config)
        if success:
            st.success("ClickHouse configuration updated successfully")
        else:
            st.error("Failed to update configuration")

    if st.button("🧪 Test ClickHouse Connection"):
        db_manager = st.session_state.get('database_manager')
        if db_manager:
            success, msg = db_manager.test_clickhouse_connection()
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Database manager not available")

    st.markdown("---")

    # PostgreSQL
    st.subheader("PostgreSQL (Airflow Metadata)")

    pg_config = config_manager.get_postgres_config()

    col1, col2, col3 = st.columns(3)

    with col1:
        pg_host = st.text_input("Host", value=pg_config.get('host', 'postgres'), key="pg_host")
        pg_port = st.number_input("Port", value=pg_config.get('port', 5432), key="pg_port")

    with col2:
        pg_database = st.text_input("Database", value=pg_config.get('database', 'airflow'), key="pg_db")
        pg_user = st.text_input("User", value=pg_config.get('user', 'airflow'), key="pg_user")

    with col3:
        pg_password = st.text_input("Password", value=pg_config.get('password', ''), type="password", key="pg_pass")
        pg_enabled = st.checkbox("Enabled", value=pg_config.get('enabled', True), key="pg_enabled")

    if st.button("💾 Update PostgreSQL Config", use_container_width=True):
        new_config = {
            'host': pg_host,
            'port': pg_port,
            'database': pg_database,
            'user': pg_user,
            'password': pg_password,
            'enabled': pg_enabled
        }

        success = config_manager.update_database_connection('postgres', new_config)
        if success:
            st.success("PostgreSQL configuration updated successfully")
        else:
            st.error("Failed to update configuration")

# Tab 2: Airflow Configuration
with tab2:
    st.header("Airflow Configuration")

    airflow_config = config_manager.get_airflow_config()

    col1, col2 = st.columns(2)

    with col1:
        af_url = st.text_input("Webserver URL", value=airflow_config.get('webserver_url', 'http://localhost:8080'))
        af_username = st.text_input("Username", value=airflow_config.get('username', 'admin'))

    with col2:
        af_password = st.text_input("Password", value=airflow_config.get('password', ''), type="password")

    if st.button("💾 Update Airflow Config", use_container_width=True):
        config_manager.set('airflow.webserver_url', af_url)
        config_manager.set('airflow.username', af_username)
        config_manager.set('airflow.password', af_password)

        if config_manager.save_config():
            st.success("Airflow configuration updated successfully")
            st.info("Please refresh the page to apply changes")
        else:
            st.error("Failed to update configuration")

# Tab 3: Export/Import
with tab3:
    st.header("Configuration Export/Import")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Export Configuration")

        if st.button("Export Current Config", use_container_width=True):
            config_yaml = yaml.dump(config_manager.config, default_flow_style=False, sort_keys=False)

            st.download_button(
                "Download YAML",
                config_yaml,
                f"anomalyguard-config-{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                "text/yaml"
            )

        # Show current config
        with st.expander("View Current Configuration"):
            st.code(
                yaml.dump(config_manager.config, default_flow_style=False, sort_keys=False),
                language="yaml"
            )

    with col2:
        st.subheader("📥 Import Configuration")

        uploaded_file = st.file_uploader("Upload Configuration YAML", type=['yaml', 'yml'])

        if uploaded_file is not None:
            try:
                imported_config = yaml.safe_load(uploaded_file)

                st.write("**Preview:**")
                st.code(yaml.dump(imported_config, default_flow_style=False), language="yaml")

                merge_config = st.checkbox("Merge with existing (instead of replace)")

                if st.button("Import Configuration", use_container_width=True, type="primary"):
                    if merge_config:
                        config_manager._deep_merge(config_manager.config, imported_config)
                    else:
                        config_manager.config = imported_config

                    if config_manager.save_config():
                        st.success("Configuration imported successfully")
                        st.info("Please refresh the page to apply changes")
                    else:
                        st.error("Failed to save imported configuration")

            except Exception as e:
                st.error(f"Error reading configuration file: {e}")

# Tab 4: Validation
with tab4:
    st.header("Configuration Validation")

    if st.button("🔍 Validate Configuration", use_container_width=True):
        valid, errors = config_manager.validate_config()

        if valid:
            st.success("✅ Configuration is valid")

            # Show key settings
            st.subheader("Key Settings")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Ensemble Weights:**")
                weights = config_manager.get('ensemble.weights', {})
                for key, value in weights.items():
                    st.write(f"  - {key}: {value}")

                st.write("**Risk Thresholds:**")
                thresholds = config_manager.get('ensemble.thresholds', {})
                for key, value in thresholds.items():
                    st.write(f"  - {key}: {value}")

            with col2:
                st.write("**Training Parameters:**")
                train_params = config_manager.get('airflow.dags.training.params', {})
                for key, value in train_params.items():
                    st.write(f"  - {key}: {value}")

                st.write("**Detection Parameters:**")
                detect_params = config_manager.get('airflow.dags.detection.params', {})
                for key, value in detect_params.items():
                    st.write(f"  - {key}: {value}")

        else:
            st.error("❌ Configuration has errors:")
            for error in errors:
                st.write(f"  - {error}")
