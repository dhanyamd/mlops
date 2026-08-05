{{ config(materialized='table') }}

with source as (
    select * from {{ source('bronze', 'company_facts') }}
),

deduped as (
    select
        ticker,
        cik,
        metric,
        fiscal_year,
        value,
        unit,
        filed_at,
        loaded_at
    from source
    -- Keep the latest landing per point-in-time key (ingestion reruns upsert).
    qualify row_number() over (
        partition by ticker, metric, fiscal_year, filed_at
        order by loaded_at desc
    ) = 1
)

select * from deduped
