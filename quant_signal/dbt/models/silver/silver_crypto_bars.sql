{{ config(materialized='table') }}

with source as (
    select * from {{ source('bronze', 'crypto_bars') }}
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
