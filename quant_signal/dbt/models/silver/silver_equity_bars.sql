{{ config(
    materialized='incremental',
    unique_key=['symbol', 'timeframe', 'ts'],
    incremental_strategy='merge',
    on_schema_change='fail'
) }}

-- Same reasoning as silver_crypto_bars: bars accumulate, so rebuild cost should
-- track arrivals rather than total history. Merge on the natural grain so a
-- re-sent bar corrects the existing row instead of duplicating it.

with source as (
    select * from {{ source('bronze', 'equity_bars') }}

    {% if is_incremental() %}
    where loaded_at > (
        select dateadd('minute', -30, coalesce(max(loaded_at), '1970-01-01'::timestamp_ntz))
        from {{ this }}
    )
    {% endif %}
),

deduped as (
    select
        symbol,
        ts,
        timeframe,
        open,
        high,
        low,
        close,
        volume,
        provider,
        loaded_at
    from source
    -- Keep the latest landing for (symbol, timeframe, ts) in case of late/re-sent data.
    qualify row_number() over (
        partition by symbol, timeframe, ts
        order by loaded_at desc
    ) = 1
)

select * from deduped
