"""
Data Page - Transaction Data Management
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Data - AnomalyGuard", page_icon="💾", layout="wide")

st.title("💾 Data Management")

database_manager = st.session_state.get('database_manager')

if not database_manager:
    st.error("Database manager not initialized")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Browse Data", "📥 Import Data", "ℹ️ Database Info"])

# Tab 1: Browse Data
with tab1:
    st.header("Browse Transaction Data")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        days_back = st.selectbox("Time Range", [1, 7, 30, 90, 365], index=2)
        start_date = datetime.now() - timedelta(days=days_back)

    with col2:
        user_id_filter = st.text_input("User ID (optional)")

    with col3:
        limit = st.number_input("Max Results", 10, 10000, 1000)

    # Get transactions
    transactions = database_manager.get_transactions(
        limit=limit,
        start_date=start_date,
        user_id=user_id_filter if user_id_filter else None
    )

    if not transactions.empty:
        st.subheader(f"📊 Results ({len(transactions)} transactions)")

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Amount", f"${transactions['amount'].sum():,.2f}")
        col2.metric("Avg Amount", f"${transactions['amount'].mean():,.2f}")
        col3.metric("Unique Users", transactions['user_id'].nunique())
        col4.metric("Currencies", transactions['currency'].nunique())

        st.markdown("---")

        # Data table
        st.dataframe(
            transactions,
            use_container_width=True,
            hide_index=True
        )

        # Export
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export to CSV", use_container_width=True):
                csv = transactions.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

        with col2:
            if st.button("📥 Export to JSON", use_container_width=True):
                json_data = transactions.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
    else:
        st.info("No transactions found with the selected filters")

# Tab 2: Import Data
with tab2:
    st.header("Import Transaction Data")

    st.info("Upload a CSV file with transaction data to import into the system")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read and preview
            df = pd.read_csv(uploaded_file)

            st.subheader("Preview")
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")

            st.dataframe(df.head(10), use_container_width=True)

            # Required columns
            required_cols = [
                'transaction_id', 'timestamp', 'user_id',
                'transaction_type', 'currency', 'amount', 'fee', 'status'
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns present")

                batch_size = st.number_input("Batch Size", 100, 5000, 500)

                if st.button("📤 Import Data", use_container_width=True, type="primary"):
                    with st.spinner("Importing data..."):
                        inserted, errors = database_manager.import_transactions(df, batch_size)

                        if errors:
                            st.warning(f"Imported {inserted} rows with {len(errors)} errors")
                            with st.expander("View Errors"):
                                for error in errors:
                                    st.write(f"- {error}")
                        else:
                            st.success(f"✅ Successfully imported {inserted} rows")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# Tab 3: Database Info
with tab3:
    st.header("Database Information")

    # Database size
    st.subheader("📊 Table Sizes")

    db_size = database_manager.get_database_size()

    if db_size:
        df_size = pd.DataFrame(db_size)
        st.dataframe(df_size, use_container_width=True, hide_index=True)
    else:
        st.info("No size information available")

    st.markdown("---")

    # Table schemas
    st.subheader("📋 Table Schemas")

    table_name = st.selectbox(
        "Select Table",
        [
            "crypto_transactions",
            "model_registry",
            "detected_anomalies_ensemble",
            "daily_model_performance",
            "mv_hourly_stats"
        ]
    )

    if table_name:
        table_info = database_manager.get_table_info(table_name)

        if table_info and table_info['columns']:
            st.write(f"**Columns:** {table_info['total_columns']}")

            df_schema = pd.DataFrame(table_info['columns'])
            st.dataframe(df_schema, use_container_width=True, hide_index=True)
        else:
            st.info("No schema information available")

    st.markdown("---")

    # Connection test
    st.subheader("🔌 Connection Test")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Test ClickHouse", use_container_width=True):
            success, msg = database_manager.test_clickhouse_connection()
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    with col2:
        if st.button("Test PostgreSQL", use_container_width=True):
            success, msg = database_manager.test_postgres_connection()
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
