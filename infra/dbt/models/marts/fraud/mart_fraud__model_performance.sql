with predictions as (
    select * from {{ source('clickhouse_warehouse', 'scored_transactions') }}
),

ground_truth as (
    select transaction_id, is_fraud from {{ ref('stg_fraud__transactions') }}
),

joined as (
    select
        p.transaction_id,
        p.amount,
        p.fraud_score,
        p.vector_score,
        p.fraud_label,
        p.model_version,
        p.scored_at,
        g.is_fraud as actual_is_fraud,
        -- Calculate TP, FP, FN, TN assuming threshold is 0.5
        case when p.fraud_score >= 0.5 and g.is_fraud = 1 then 1 else 0 end as is_tp,
        case when p.fraud_score >= 0.5 and g.is_fraud = 0 then 1 else 0 end as is_fp,
        case when p.fraud_score < 0.5 and g.is_fraud = 1 then 1 else 0 end as is_fn,
        case when p.fraud_score < 0.5 and g.is_fraud = 0 then 1 else 0 end as is_tn
    from predictions p
    inner join ground_truth g on p.transaction_id = g.transaction_id
)

select * from joined
