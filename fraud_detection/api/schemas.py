"""Pydantic schemas for the Fraud Detection FastAPI microservice.

Why Pydantic:
  - Input validation happens at the boundary (API layer), not deep in model code.
  - If the client sends amount=-999 or a missing field, you get a 422 immediately
    with a clear error message, not a silent NaN propagating into XGBoost.
  - Output schema documents exactly what callers receive — acts as a contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Input schema for a single transaction to be scored.

    Fields match what the Kafka producer emits and what the streaming
    inference service receives — keeping the API and streaming paths identical.
    """

    transaction_id: str = Field(..., description="Unique transaction identifier.")
    card_id: str = Field(..., description="Card or user identifier for feature lookup.")
    amount: float = Field(..., gt=0, description="Transaction amount in USD. Must be positive.")
    merchant_category: str = Field(
        default="unknown",
        description="Merchant category code (retail, travel, food, digital, atm).",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 transaction timestamp. Defaults to now if not provided.",
    )

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite(cls, v: float) -> float:
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError("amount must be a finite number")
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "TXN_RT_000001",
                    "card_id": "CARD_0042",
                    "amount": 1250.00,
                    "merchant_category": "retail",
                    "timestamp": "2024-06-01T14:32:00Z",
                }
            ]
        }
    }


class FraudScoreResponse(BaseModel):
    """Output schema for a scored transaction."""

    transaction_id: str
    card_id: str
    amount: float
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Combined fraud probability (0-1).")
    model_score: float = Field(..., description="XGBoost model probability.")
    vector_score: float = Field(..., description="Qdrant pattern similarity score.")
    fraud_label: bool = Field(..., description="True if fraud_score >= threshold.")
    model_version: str
    scored_at: datetime
    cache_hit: bool = Field(default=False, description="True if result was served from Redis cache.")
    explanation: Optional[dict] = Field(
        default=None,
        description="SHAP feature contributions (only returned when explain=true).",
    )


class BatchTransactionRequest(BaseModel):
    """Batch scoring — up to 100 transactions in one request."""

    transactions: list[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of transactions to score.",
    )


class BatchScoreResponse(BaseModel):
    results: list[FraudScoreResponse]
    total: int
    fraud_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    redis_connected: bool
    qdrant_connected: bool
    uptime_seconds: float
