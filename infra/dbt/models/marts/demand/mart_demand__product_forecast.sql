with forecasts as (
    select * from {{ source('clickhouse_warehouse_demand', 'forecasts') }}
),

sales as (
    select * from {{ ref('stg_demand__sales') }}
),

joined as (
    select
        f.product_id,
        f.forecast_date as date,
        f.predicted_demand,
        f.lower_bound,
        f.upper_bound,
        f.model_version,
        f.scored_at,
        s.quantity_sold as actual_demand,
        s.revenue as actual_revenue,
        -- Calculate absolute error and squared error for accuracy reporting
        abs(f.predicted_demand - s.quantity_sold) as absolute_error,
        pow(f.predicted_demand - s.quantity_sold, 2) as squared_error
    from forecasts f
    left join sales s on f.product_id = s.product_id and f.forecast_date = s.date
)

select * from joined
