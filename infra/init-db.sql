-- Simulates multi-source data layer from ML Academy architecture diagrams.
-- PostgreSQL: product catalog (demand forecasting) + user profiles (fraud detection)

CREATE SCHEMA IF NOT EXISTS demand_forecasting;
CREATE SCHEMA IF NOT EXISTS fraud_detection;

-- Demand forecasting: product info (replaces Salesforce product catalog slice)
CREATE TABLE IF NOT EXISTS demand_forecasting.products (
    product_id VARCHAR(32) PRIMARY KEY,
    product_name VARCHAR(256) NOT NULL,
    category VARCHAR(64) NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    cost DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Demand forecasting: promotions (Salesforce CRM stand-in)
CREATE TABLE IF NOT EXISTS demand_forecasting.promotions (
    promotion_id VARCHAR(32) PRIMARY KEY,
    product_id VARCHAR(32) REFERENCES demand_forecasting.products(product_id),
    discount_pct DECIMAL(5, 2) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    channel VARCHAR(64) NOT NULL
);

-- Demand forecasting: forecast output table (Snowflake/ClickHouse stand-in)
CREATE TABLE IF NOT EXISTS demand_forecasting.forecasts (
    forecast_id SERIAL PRIMARY KEY,
    product_id VARCHAR(32) NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_demand DECIMAL(12, 2) NOT NULL,
    lower_bound DECIMAL(12, 2),
    upper_bound DECIMAL(12, 2),
    model_version VARCHAR(64) NOT NULL,
    scored_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (product_id, forecast_date, model_version)
);

-- Fraud detection: user profiles
CREATE TABLE IF NOT EXISTS fraud_detection.user_profiles (
    user_id VARCHAR(32) PRIMARY KEY,
    account_age_days INTEGER NOT NULL,
    avg_transaction_amount DECIMAL(12, 2) NOT NULL,
    country VARCHAR(8) NOT NULL,
    risk_tier VARCHAR(16) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Fraud detection: transaction labels (historical ground truth)
CREATE TABLE IF NOT EXISTS fraud_detection.transaction_labels (
    transaction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(32) NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    labeled_at TIMESTAMP DEFAULT NOW()
);

-- Fraud detection: scored transactions (actioning output)
CREATE TABLE IF NOT EXISTS fraud_detection.scored_transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(32) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    fraud_score DECIMAL(8, 6) NOT NULL,
    fraud_label BOOLEAN NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    scored_at TIMESTAMP DEFAULT NOW()
);
