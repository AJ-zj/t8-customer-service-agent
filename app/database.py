"""Shared SQLAlchemy engine and session factory for the T8 project."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_database_url = os.getenv("DATABASE_URL")
if not _database_url:
    raise RuntimeError(
        "项目根目录 .env 未配置 DATABASE_URL，请补充数据库连接配置。"
    )

_url = make_url(_database_url)
if _url.get_backend_name() != "postgresql":
    raise RuntimeError("DATABASE_URL 必须指向 PostgreSQL。")

engine = create_engine(_database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def configured_database_name() -> str | None:
    """Return only the configured database name, never connection credentials."""
    return _url.database


def current_database_name() -> str:
    """Ask PostgreSQL which database this engine is connected to."""
    with engine.connect() as connection:
        return connection.execute(text("SELECT current_database()")).scalar_one()
