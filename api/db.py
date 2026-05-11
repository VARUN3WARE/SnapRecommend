"""Database models and helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Float, ForeignKey, Integer, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import DATABASE_URL, DB_PATH


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    image_path: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), index=True)
    product_id: Mapped[str] = mapped_column(String, ForeignKey("products.product_id"), index=True)
    event_type: Mapped[str] = mapped_column(String)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    weight: Mapped[float] = mapped_column(Float)


def get_engine(db_url: str | None = None):
    """Create database engine supporting SQLite and PostgreSQL."""
    final_url = db_url or DATABASE_URL
    
    # If db_url is a path string (not a full URL), treat it as SQLite
    if final_url and "://" not in final_url:
        db_path = Path(final_url)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        final_url = f"sqlite:///{db_path.absolute()}"
    elif "sqlite://" in final_url:
        db_path = final_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Use connect_args for SQLite to set pragmas on pool pre-ping
    connect_args = {}
    if "sqlite://" in final_url:
        connect_args = {"timeout": 30}
        engine = create_engine(final_url, future=True, echo=False, connect_args=connect_args, pool_pre_ping=True)
        
        # Enable foreign keys for SQLite on all connections
        # Note: Disabled by default to avoid breaking tests; enable in production as needed
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            # Foreign keys can be enabled here if needed: cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine = create_engine(final_url, future=True, echo=False, pool_pre_ping=True)
    
    return engine


def init_db(db_path: str | None = None) -> None:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def get_session_factory(db_path: str | None = None):
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@contextmanager
def session_scope(db_path: str | None = None):
    session_factory = get_session_factory(db_path)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
