{{ config(
    materialized='incremental',
    unique_key=['symbol', 'ts'],
    incremental_strategy='merge',
    on_schema_change='fail'
) }}

-- The research panel: daily perpetual bars for the SRP universe. Same
-- incremental discipline as silver_crypto_bars -- a backfill re-sends history,
-- and rebuilding seven years of 202 symbols on every run would burn credits to
-- recompute rows that cannot have changed.
--
-- The grain here is (symbol, ts) with no timeframe column, because this table is
-- daily by construction. Carrying a timeframe that only ever holds one value
-- would invite someone to load a second resolution into it and silently break
-- every weekly resample downstream.

with source as (
    select * from {{ source('bronze', 'crypto_panel_bars') }}

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
        close,
        volume,
        provider,
        loaded_at
    from source
    -- A re-backfill lands the same bar again; keep the most recent landing.
    qualify row_number() over (
        partition by symbol, ts
        order by loaded_at desc
    ) = 1
)

select * from deduped
-- Non-positive prices are not tradable and would corrupt a log return or a
-- cross-sectional rank. The research loader screens for this too; enforcing it
-- here means the warehouse never serves a row the research would have to reject.
where close > 0
