"""SQLAlchemy ORM models — no raw SQL needed in application code."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    __table_args__ = {"schema": "demand_forecasting"}

    product_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    store_id: Mapped[int] = mapped_column(Integer, nullable=False)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)
    product_name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    cost: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)


class Promotion(Base):
    __tablename__ = "promotions"
    __table_args__ = {"schema": "demand_forecasting"}

    promotion_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    discount_pct: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    channel: Mapped[str] = mapped_column(String(64), nullable=False)


class Sale(Base):
    """Historical sales — column-store / warehouse table (Snowflake/ClickHouse stand-in)."""

    __tablename__ = "sales"
    __table_args__ = (
        UniqueConstraint("sale_date", "product_id", name="uq_sales_date_product"),
        {"schema": "demand_forecasting"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sale_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    store_id: Mapped[int] = mapped_column(Integer, nullable=False)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    quantity_sold: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    revenue: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)


class ExternalDaily(Base):
    __tablename__ = "external_daily"
    __table_args__ = {"schema": "demand_forecasting"}

    sale_date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_holiday: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(Float, nullable=True)


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("product_id", "forecast_date", "model_version"),
        {"schema": "demand_forecasting"},
    )

    forecast_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    predicted_demand: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    lower_bound: Mapped[float | None] = mapped_column(Numeric(12, 2))
    upper_bound: Mapped[float | None] = mapped_column(Numeric(12, 2))
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FraudTransaction(Base):
    """ULB credit card fraud dataset rows."""

    __tablename__ = "transactions"
    __table_args__ = {"schema": "fraud_detection"}

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    time_seconds: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    v1: Mapped[float] = mapped_column(Float)
    v2: Mapped[float] = mapped_column(Float)
    v3: Mapped[float] = mapped_column(Float)
    v4: Mapped[float] = mapped_column(Float)
    v5: Mapped[float] = mapped_column(Float)
    v6: Mapped[float] = mapped_column(Float)
    v7: Mapped[float] = mapped_column(Float)
    v8: Mapped[float] = mapped_column(Float)
    v9: Mapped[float] = mapped_column(Float)
    v10: Mapped[float] = mapped_column(Float)
    v11: Mapped[float] = mapped_column(Float)
    v12: Mapped[float] = mapped_column(Float)
    v13: Mapped[float] = mapped_column(Float)
    v14: Mapped[float] = mapped_column(Float)
    v15: Mapped[float] = mapped_column(Float)
    v16: Mapped[float] = mapped_column(Float)
    v17: Mapped[float] = mapped_column(Float)
    v18: Mapped[float] = mapped_column(Float)
    v19: Mapped[float] = mapped_column(Float)
    v20: Mapped[float] = mapped_column(Float)
    v21: Mapped[float] = mapped_column(Float)
    v22: Mapped[float] = mapped_column(Float)
    v23: Mapped[float] = mapped_column(Float)
    v24: Mapped[float] = mapped_column(Float)
    v25: Mapped[float] = mapped_column(Float)
    v26: Mapped[float] = mapped_column(Float)
    v27: Mapped[float] = mapped_column(Float)
    v28: Mapped[float] = mapped_column(Float)


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"
    __table_args__ = {"schema": "fraud_detection"}

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    fraud_score: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    fraud_label: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
