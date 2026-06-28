-- PostgreSQL OLTP: catalog, promotions, metadata (Supabase/Neon class)

CREATE SCHEMA IF NOT EXISTS demand_forecasting;
CREATE SCHEMA IF NOT EXISTS fraud_detection;

-- Product catalog (enriched from store sales dataset)
CREATE TABLE IF NOT EXISTS demand_forecasting.products (
    product_id   VARCHAR(32) PRIMARY KEY,
    store_id     SMALLINT NOT NULL,
    item_id      SMALLINT NOT NULL,
    product_name VARCHAR(256) NOT NULL,
    category     VARCHAR(64) NOT NULL,
    unit_price   DECIMAL(10, 2) NOT NULL,
    cost         DECIMAL(10, 2) NOT NULL
);

-- Promotions (Salesforce CRM stand-in)
CREATE TABLE IF NOT EXISTS demand_forecasting.promotions (
    promotion_id  VARCHAR(32) PRIMARY KEY,
    product_id    VARCHAR(32) NOT NULL REFERENCES demand_forecasting.products(product_id),
    discount_pct  DECIMAL(5, 2) NOT NULL,
    start_date    DATE NOT NULL,
    end_date      DATE NOT NULL,
    channel       VARCHAR(64) NOT NULL
);

-- External calendar features
CREATE TABLE IF NOT EXISTS demand_forecasting.external_daily (
    sale_date         DATE PRIMARY KEY,
    is_holiday        BOOLEAN NOT NULL DEFAULT FALSE,
    temperature_c     DECIMAL(5, 1),
    precipitation_mm  DECIMAL(5, 1)
);

-- Fraud: model metadata / alert routing config
CREATE TABLE IF NOT EXISTS fraud_detection.alert_rules (
    rule_id     SERIAL PRIMARY KEY,
    name        VARCHAR(128) NOT NULL,
    threshold   DECIMAL(5, 4) NOT NULL DEFAULT 0.5,
    action      VARCHAR(32) NOT NULL DEFAULT 'block',
    enabled     BOOLEAN NOT NULL DEFAULT TRUE
);

INSERT INTO fraud_detection.alert_rules (name, threshold, action)
VALUES ('default_fraud_block', 0.5, 'block')
ON CONFLICT DO NOTHING;
