{{ config(materialized='table') }}

with bars as (
    select * from {{ ref('silver_equity_bars') }}
)

select
    symbol,
    timeframe,
    ts::date as trade_date,
    min_by(open, ts) as day_open,
    max(high) as day_high,
    min(low) as day_low,
    max_by(close, ts) as day_close,
    sum(volume) as volume,
    count(*) as n_bars
from bars
group by symbol, timeframe, ts::date
