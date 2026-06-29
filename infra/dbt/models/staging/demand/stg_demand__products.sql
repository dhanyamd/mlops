with source as (
    select * from {{ source('clickhouse_warehouse_demand', 'products') }}
),

renamed as (
    select
        product_id,
        store_id,
        item_id,
        product_name,
        category,
        cast(unit_price, 'Float64') as unit_price,
        cast(cost, 'Float64') as cost
    from source
)

select * from renamed
