"""
Database connection and session factory for PostgreSQL / SQLite.
Supports DATABASE_URL environment configuration with automatic engine creation and table initialization.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_SQLITE_PATH = os.path.join(PROJECT_ROOT, "data", "digital_twin.db")
DEFAULT_DB_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")


def get_db_url() -> str:
    """Retrieve database URL from environment or fallback to project SQLite database."""
    url = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)
    # SQLAlchemy 2.0 requires postgresql:// instead of postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def get_engine(db_url: str = None, echo: bool = False):
    """Create SQLAlchemy engine with appropriate dialect arguments."""
    url = db_url or get_db_url()
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure parent directory exists for SQLite db file
        db_file = url.replace("sqlite:///", "")
        if db_file and not db_file.startswith(":memory:"):
            os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)

    return create_engine(url, connect_args=connect_args, echo=echo)


def init_db(engine=None):
    """Initialize database tables according to declarative metadata."""
    eng = engine or get_engine()
    Base.metadata.create_all(eng)
    return eng


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
