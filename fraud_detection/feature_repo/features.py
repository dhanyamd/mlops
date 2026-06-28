"""Feast feature definitions for fraud detection.

Feature Store architecture (ML Academy Day 4):
- Offline Store: historical features for training
- Online Store: low-latency features for real-time inference
- Feature Views: unified interface preventing training-serving skew
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int64

user = Entity(name="user_id", value_type=ValueType.STRING, description="Transaction user")

transaction_batch_source = FileSource(
    name="transaction_features_batch",
    path="../../data/fraud_detection/offline/transaction_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

user_profile_source = FileSource(
    name="user_profile_features",
    path="../../data/fraud_detection/offline/user_profile_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Historical/batch features — written to Offline Store
transaction_features = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="txn_count_1h", dtype=Int64),
        Field(name="txn_count_24h", dtype=Int64),
        Field(name="amount_sum_24h", dtype=Float64),
        Field(name="amount_mean_24h", dtype=Float64),
        Field(name="amount_std_24h", dtype=Float64),
        Field(name="velocity_ratio", dtype=Float32),
        Field(name="amount_deviation", dtype=Float32),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="is_night", dtype=Int64),
    ],
    online=True,
    source=transaction_batch_source,
    tags={"team": "fraud", "type": "velocity"},
)

user_profile_features = FeatureView(
    name="user_profile_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="account_age_days", dtype=Int64),
        Field(name="avg_transaction_amount", dtype=Float64),
        Field(name="risk_score", dtype=Float32),
    ],
    online=True,
    source=user_profile_source,
    tags={"team": "fraud", "type": "profile"},
)
