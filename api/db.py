"""Database models and helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import DB_PATH


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


def get_engine(db_path: str | None = None):
    final_path = Path(db_path) if db_path else DB_PATH
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{final_path}", future=True)


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
