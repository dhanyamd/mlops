-- ClickHouse OLAP warehouse (Snowflake / BigQuery class for analytics)
CREATE DATABASE IF NOT EXISTS demand;
CREATE DATABASE IF NOT EXISTS fraud;

-- Demand forecasting: historical sales (913k rows from Kaggle Store Item Demand)
CREATE TABLE IF NOT EXISTS demand.sales (
    sale_date       Date,
    store_id        UInt8,
    item_id         UInt8,
    product_id      String,
    quantity_sold   Float64,
    revenue         Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(sale_date)
ORDER BY (product_id, sale_date);

-- Demand forecasting: batch inference output
CREATE TABLE IF NOT EXISTS demand.forecasts (
    product_id        String,
    forecast_date     Date,
    predicted_demand  Float64,
    lower_bound       Float64,
    upper_bound       Float64,
    model_version     String,
    scored_at         DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (product_id, forecast_date);

-- Fraud: historical transactions (ULB Credit Card Fraud dataset)
CREATE TABLE IF NOT EXISTS fraud.transactions (
    transaction_id  String,
    time_seconds    Float64,
    amount          Float64,
    is_fraud        UInt8,
    v1 Float64, v2 Float64, v3 Float64, v4 Float64, v5 Float64,
    v6 Float64, v7 Float64, v8 Float64, v9 Float64, v10 Float64,
    v11 Float64, v12 Float64, v13 Float64, v14 Float64, v15 Float64,
    v16 Float64, v17 Float64, v18 Float64, v19 Float64, v20 Float64,
    v21 Float64, v22 Float64, v23 Float64, v24 Float64, v25 Float64,
    v26 Float64, v27 Float64, v28 Float64
) ENGINE = MergeTree()
ORDER BY (time_seconds, transaction_id);

-- Fraud: real-time scored transactions (audit log)
CREATE TABLE IF NOT EXISTS fraud.scored_transactions (
    transaction_id  String,
    amount          Float64,
    fraud_score     Float64,
    vector_score    Float64,
    fraud_label     UInt8,
    model_version   String,
    scored_at       DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY scored_at;

-- PostgreSQL mapping tables for DBT Staging Models to read catalog data from PostgreSQL
CREATE TABLE IF NOT EXISTS demand.products (
    product_id   String,
    store_id     Int32,
    item_id      Int32,
    product_name String,
    category     String,
    unit_price   Float64,
    cost         Float64
) ENGINE = PostgreSQL('postgres:5432', 'mlops', 'products', 'mlops', 'mlops', 'demand_forecasting');

CREATE TABLE IF NOT EXISTS demand.promotions (
    promotion_id String,
    product_id   String,
    discount_pct Float64,
    start_date   Date,
    end_date     Date,
    channel      String
) ENGINE = PostgreSQL('postgres:5432', 'mlops', 'promotions', 'mlops', 'mlops', 'demand_forecasting');

CREATE TABLE IF NOT EXISTS demand.external_daily (
    sale_date        Date,
    is_holiday       UInt8,
    temperature_c    Nullable(Float64),
    precipitation_mm Nullable(Float64)
) ENGINE = PostgreSQL('postgres:5432', 'mlops', 'external_daily', 'mlops', 'mlops', 'demand_forecasting');

