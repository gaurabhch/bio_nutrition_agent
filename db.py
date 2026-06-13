# pipeline_1_nutrition/db.py

import os
import sys
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

_DATABASE_URL = os.getenv("DATABASE_URL_DIRECT")

if not _DATABASE_URL:
    print("❌  DATABASE_URL is not set in .env", file=sys.stderr)
    sys.exit(1)

# connect_args: increase statement timeout for large batch inserts on Neon
_engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,          # auto-reconnect on dropped Neon connections
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=120000",  # 120s for large ingestions
    },
)

_SessionFactory = sessionmaker(bind=_engine)


@contextmanager
def get_session(table: str | None = None):
    """
    Yields a SQLAlchemy Session as a context manager.
    Commits on clean exit, rolls back on exception, always closes.

    Args:
        table: If provided, verifies the table exists in Neon before yielding.
               Raises RuntimeError if not found — prevents silent wrong-table inserts.
    """
    session: Session = _SessionFactory()
    try:
        if table:
            _verify_table_exists(session, table)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _verify_table_exists(session: Session, table: str) -> None:
    """
    Checks pg_tables to confirm the target table exists in Neon.
    Raises RuntimeError with a clear message if it doesn't.
    """
    result = session.execute(
        text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename = :table"
        ),
        {"table": table},
    ).fetchone()

    if result is None:
        raise RuntimeError(
            f"Table '{table}' does not exist in Neon. "
            f"Run the CREATE TABLE SQL first before ingesting."
        )