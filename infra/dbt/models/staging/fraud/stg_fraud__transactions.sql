-- Staging model for raw fraud transactions

with source as (
    select * from {{ source('clickhouse_warehouse', 'transactions') }}
),

renamed as (
    select
        transaction_id,
        cast(time_seconds, 'Float64') as time_seconds,
        cast(amount, 'Float64') as amount,
        cast(is_fraud, 'Int8') as is_fraud,
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
        v21, v22, v23, v24, v25, v26, v27, v28
    from source
)

select * from renamed
