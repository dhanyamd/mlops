{{ config(
    materialized='incremental',
    unique_key=['symbol', 'ts'],
    incremental_strategy='merge',
    on_schema_change='fail'
) }}

-- 8-hourly perpetual funding. The SRP return construction subtracts
-- ``w_t . funding_{t+1}`` explicitly, so a gap here does not degrade the
-- backtest gracefully -- it silently makes a carry-exposed book look free.
-- Deduplicated on (symbol, ts) because funding history is re-fetched whole
-- rather than incrementally by the backfill.

with source as (
    select * from {{ source('bronze', 'crypto_funding') }}

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
        rate,
        provider,
        loaded_at
    from source
    qualify row_number() over (
        partition by symbol, ts
        order by loaded_at desc
    ) = 1
)

select * from deduped
