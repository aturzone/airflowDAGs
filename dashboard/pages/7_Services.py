"""
Services Page - Docker Service Management
"""

import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Services - AnomalyGuard", page_icon="🐳", layout="wide")

st.title("🐳 Service Management")

docker_manager = st.session_state.get('docker_manager')

if not docker_manager:
    st.error("Docker manager not initialized")
    st.stop()

# Check Docker availability
if not docker_manager.is_docker_available():
    st.error("Docker is not available or not running")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🐳 Containers", "⚙️ System"])

# Tab 1: Overview
with tab1:
    st.header("Service Overview")

    # Get service health summary
    health_summary = docker_manager.get_service_health_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Services", health_summary.get('total_services', 0))
    col2.metric("Healthy", health_summary.get('healthy_services', 0), delta_color="normal")
    col3.metric("Unhealthy", health_summary.get('unhealthy_services', 0), delta_color="inverse")
    col4.metric("Health %", f"{health_summary.get('health_percentage', 0):.1f}%")

    st.markdown("---")

    # Service status cards
    st.subheader("Service Status")

    services = health_summary.get('services', [])

    for service in services:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

            with col1:
                st.write(f"**{service.get('service_name', 'Unknown')}**")
                st.caption(f"Container: {service.get('name', 'N/A')}")

            with col2:
                status = service.get('status', 'unknown')
                st.write(f"Status: {status}")

            with col3:
                if service.get('is_running', False):
                    st.success("✅ Running")
                else:
                    st.error("❌ Stopped")

            with col4:
                container_name = service.get('name')
                if container_name:
                    if st.button("🔄", key=f"restart_{container_name}"):
                        success, msg = docker_manager.restart_container(container_name)
                        if success:
                            st.success(msg)
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(msg)

        st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

# Tab 2: Containers
with tab2:
    st.header("Container Management")

    # Get all services
    services = docker_manager.get_all_services_status()

    if services:
        for service in services:
            container_name = service.get('name')

            with st.expander(f"🐳 {service.get('service_name', 'Unknown')} ({container_name})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Container Information**")
                    st.write(f"ID: {service.get('id', 'N/A')[:12]}")
                    st.write(f"Image: {service.get('image', 'N/A')}")
                    st.write(f"State: {service.get('state', 'N/A')}")
                    st.write(f"Status: {service.get('status', 'N/A')}")

                with col2:
                    st.write("**Actions**")

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if st.button("▶️ Start", key=f"start_{container_name}"):
                            success, msg = docker_manager.start_container(container_name)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

                    with col_b:
                        if st.button("⏸️ Stop", key=f"stop_{container_name}"):
                            success, msg = docker_manager.stop_container(container_name)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

                    with col_c:
                        if st.button("🔄 Restart", key=f"restart_tab2_{container_name}"):
                            success, msg = docker_manager.restart_container(container_name)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

                # Resource stats
                stats = docker_manager.get_container_stats(container_name)

                if 'error' not in stats:
                    st.write("**Resource Usage**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("CPU", stats.get('cpu_percent', 'N/A'))
                    col2.metric("Memory", stats.get('memory_usage', 'N/A'))
                    col3.metric("Memory %", stats.get('memory_percent', 'N/A'))

                # Logs
                if st.checkbox("Show Logs", key=f"logs_{container_name}"):
                    success, logs = docker_manager.get_container_logs(container_name, tail=50)

                    if success:
                        st.code(logs, language="log")
                    else:
                        st.error(f"Failed to get logs: {logs}")

# Tab 3: System
with tab3:
    st.header("Docker System Information")

    # Docker info
    docker_info = docker_manager.get_docker_info()

    if 'error' not in docker_info:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Containers", docker_info.get('containers', 0))
        col2.metric("Running", docker_info.get('containers_running', 0))
        col3.metric("Stopped", docker_info.get('containers_stopped', 0))
        col4.metric("Images", docker_info.get('images', 0))

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        col1.write(f"**Driver:** {docker_info.get('driver', 'N/A')}")
        col2.write(f"**CPUs:** {docker_info.get('cpus', 'N/A')}")
        col3.write(f"**Version:** {docker_info.get('server_version', 'N/A')}")
    else:
        st.error(docker_info.get('error'))

    st.markdown("---")

    # Docker Compose
    st.subheader("Docker Compose")

    compose_status = docker_manager.get_compose_status()

    if 'error' not in compose_status:
        col1, col2 = st.columns(2)

        col1.metric("Total Services", compose_status.get('total', 0))
        col2.metric("Running", compose_status.get('running', 0))

        services_list = compose_status.get('services', [])

        if services_list:
            st.dataframe(
                pd.DataFrame(services_list),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")

        # Compose controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Start All", use_container_width=True):
                success, msg = docker_manager.compose_up()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        with col2:
            if st.button("🔄 Restart All", use_container_width=True):
                success, msg = docker_manager.compose_restart()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        with col3:
            if st.button("⏹️ Stop All", use_container_width=True, type="secondary"):
                success, msg = docker_manager.compose_down()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    else:
        st.error(compose_status.get('error'))

    st.markdown("---")

    # Volumes and Networks
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Volumes")
        volumes = docker_manager.get_volumes()
        if volumes:
            st.dataframe(pd.DataFrame(volumes), use_container_width=True, hide_index=True)
        else:
            st.info("No volumes found")

    with col2:
        st.subheader("Networks")
        networks = docker_manager.get_networks()
        if networks:
            st.dataframe(pd.DataFrame(networks), use_container_width=True, hide_index=True)
        else:
            st.info("No networks found")
