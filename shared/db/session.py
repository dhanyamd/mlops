"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from shared.config import POSTGRES
from shared.db.models import Base


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(POSTGRES.url, pool_pre_ping=True)


def get_session() -> Session:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)()


def init_database() -> None:
    """Create all ORM tables (schemas must exist — see infra/init-db.sql)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
