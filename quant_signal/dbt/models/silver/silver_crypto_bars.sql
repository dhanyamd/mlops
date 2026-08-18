{{ config(
    materialized='incremental',
    unique_key=['symbol', 'timeframe', 'ts'],
    incremental_strategy='merge',
    on_schema_change='fail'
) }}

-- Minute bars grow without bound, so a full rebuild on every run costs warehouse
-- credits proportional to total history rather than to what actually arrived.
-- Incremental + merge processes only new landings and still corrects rows that
-- are re-sent late, because the merge key is the natural grain of the table.
--
-- ``on_schema_change='fail'`` is deliberate: the model carries an enforced
-- contract, and a silent column addition would let the contract and the table
-- drift apart. Failing loudly is the point of having the contract at all.

with source as (
    select * from {{ source('bronze', 'crypto_bars') }}

    {% if is_incremental() %}
    -- Only rows landed since the last successful run. A small overlap window
    -- guards against a landing whose ``loaded_at`` is marginally behind the
    -- watermark at the moment the previous run read it; the merge de-duplicates
    -- anything that overlaps, so re-reading is safe but missing rows is not.
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
