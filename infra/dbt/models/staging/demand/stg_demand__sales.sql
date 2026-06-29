with source as (
    select * from {{ source('clickhouse_warehouse_demand', 'sales') }}
),

renamed as (
    select
        sale_date as date,
        store_id,
        item_id,
        product_id,
        cast(quantity_sold, 'Float64') as quantity_sold,
        cast(revenue, 'Float64') as revenue
    from source
)

select * from renamed
