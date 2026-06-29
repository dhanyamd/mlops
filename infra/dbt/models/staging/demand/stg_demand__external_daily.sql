with source as (
    select * from {{ source('clickhouse_warehouse_demand', 'external_daily') }}
),

renamed as (
    select
        sale_date as date,
        cast(is_holiday, 'UInt8') as is_holiday,
        cast(temperature_c, 'Nullable(Float64)') as temperature_c,
        cast(precipitation_mm, 'Nullable(Float64)') as precipitation_mm
    from source
)

select * from renamed
