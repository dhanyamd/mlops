"""Streamlit dashboard — reads forecasts from ClickHouse warehouse."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from shared.clients import ClickHouseClient

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Demand Forecasting Dashboard")
st.caption("Forecasts served from ClickHouse OLAP warehouse")

ch = ClickHouseClient()

try:
    forecasts = ch.query_df(
        """
        SELECT product_id, forecast_date, predicted_demand,
               lower_bound, upper_bound, model_version
        FROM demand.forecasts
        ORDER BY scored_at DESC
        LIMIT 10000
        """
    )
except Exception as exc:
    st.error(f"ClickHouse unavailable: {exc}. Run: make bootstrap")
    st.stop()

if forecasts.empty:
    st.warning("No forecasts in ClickHouse. Run: make demand")
    st.stop()

forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
latest_version = forecasts["model_version"].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Products Scored", forecasts["product_id"].nunique())
col2.metric("Forecast Rows", len(forecasts))
col3.metric("Model", latest_version)

product = st.selectbox("Product", sorted(forecasts["product_id"].unique()))
subset = forecasts[forecasts["product_id"] == product]

fig = px.line(
    subset,
    x="forecast_date",
    y="predicted_demand",
    title=f"Demand forecast — {product}",
)
st.plotly_chart(fig, use_container_width=True)

agg = forecasts.groupby("forecast_date")["predicted_demand"].sum().reset_index()
st.plotly_chart(
    px.bar(agg, x="forecast_date", y="predicted_demand", title="Total forecast across products"),
    use_container_width=True,
)
