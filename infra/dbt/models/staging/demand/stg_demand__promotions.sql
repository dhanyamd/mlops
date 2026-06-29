with source as (
    select * from {{ source('clickhouse_warehouse_demand', 'promotions') }}
),

renamed as (
    select
        promotion_id,
        product_id,
        cast(discount_pct, 'Float64') as discount_pct,
        start_date,
        end_date,
        channel
    from source
)

select * from renamed
