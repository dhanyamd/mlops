{{ config(materialized='table') }}

with source as (
    select * from {{ source('bronze', 'pipeline_metrics') }}
),

typed as (
    select
        run_id,
        flow,
        stage,
        started_at,
        elapsed_ms,
        n_rows,
        loaded_at
    from source
    -- Telemetry sanity: wall-clock durations are non-negative.
    where elapsed_ms >= 0
)

select * from typed
