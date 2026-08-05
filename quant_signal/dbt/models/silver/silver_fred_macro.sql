{{ config(materialized='table') }}

with source as (
    select * from {{ source('bronze', 'fred_macro') }}
),

deduped as (
    select
        series_id,
        date,
        value,
        loaded_at
    from source
    -- Macro series are revised upstream; keep the latest landing per (series, date).
    qualify row_number() over (
        partition by series_id, date
        order by loaded_at desc
    ) = 1
)

select * from deduped
