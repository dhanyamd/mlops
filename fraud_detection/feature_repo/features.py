"""Feast feature definitions for fraud detection.

Feature Store architecture:
  - Entity: card_id (corresponds to card_id in streaming and online lookup)
  - FileSource: Parquet files as offline store
  - FeatureViews: Unified schema for training and serving.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int64

# Define the Card Entity (join key is card_id)
card = Entity(
    name="card_id",
    value_type=ValueType.STRING,
    description="Credit card identifier",
)

# S3 Data Lake is the source of truth for the offline store (Pillar 16/15)
transaction_batch_source = FileSource(
    name="transaction_features_batch",
    path="s3://ml-data-lake/fraud_detection/offline/transaction_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

user_profile_source = FileSource(
    name="user_profile_features",
    path="s3://ml-data-lake/fraud_detection/offline/user_profile_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Velocity features Feature View
transaction_features = FeatureView(
    name="transaction_features",
    entities=[card],
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

# User profile features Feature View
user_profile_features = FeatureView(
    name="user_profile_features",
    entities=[card],
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
