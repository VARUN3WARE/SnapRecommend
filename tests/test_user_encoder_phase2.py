from pathlib import Path

import numpy as np
import pytest

from api.db import Interaction, Product, User, get_session_factory, init_db
from models.user_encoder import TransformerSequenceUserEncoder, encode_user


def _seed_user_history_db(db_path: Path) -> tuple[dict[str, int], np.ndarray]:
    init_db(str(db_path))
    session_factory = get_session_factory(str(db_path))
    session = session_factory()
    try:
        item_ids = []
        for i in range(5):
            pid = f"p{i:06d}"
            item_ids.append(pid)
            session.add(
                Product(
                    product_id=pid,
                    title=f"P{i}",
                    category="cat",
                    price=1.0,
                    image_path="img.png",
                    description="demo",
                )
            )

        session.add(User(user_id="u1", created_at=1))
        session.add(Interaction(user_id="u1", product_id="p000000", event_type="view", timestamp=1, weight=0.1))
        session.add(Interaction(user_id="u1", product_id="p000001", event_type="click", timestamp=2, weight=0.3))
        session.add(Interaction(user_id="u1", product_id="p000002", event_type="purchase", timestamp=3, weight=1.0))
        session.commit()
    finally:
        session.close()

    item_to_idx = {pid: i for i, pid in enumerate(item_ids)}
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(len(item_ids), 512)).astype(np.float32)
    return item_to_idx, embeddings


def test_encode_user_legacy_mode_returns_normalized_vector(tmp_path: Path):
    db_path = tmp_path / "user.db"
    item_to_idx, embeddings = _seed_user_history_db(db_path)

    session_factory = get_session_factory(str(db_path))
    session = session_factory()
    try:
        vec = encode_user("u1", session, item_to_idx, embeddings, mode="legacy")
    finally:
        session.close()

    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_encode_user_transformer_falls_back_without_encoder(tmp_path: Path):
    db_path = tmp_path / "user.db"
    item_to_idx, embeddings = _seed_user_history_db(db_path)

    session_factory = get_session_factory(str(db_path))
    session = session_factory()
    try:
        vec = encode_user("u1", session, item_to_idx, embeddings, mode="transformer", sequence_encoder=None)
    finally:
        session.close()

    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_transformer_encoder_path_if_torch_available(tmp_path: Path):
    pytest.importorskip("torch")

    db_path = tmp_path / "user.db"
    item_to_idx, embeddings = _seed_user_history_db(db_path)

    session_factory = get_session_factory(str(db_path))
    session = session_factory()
    try:
        encoder = TransformerSequenceUserEncoder(embedding_dim=512)
        vec = encode_user(
            "u1",
            session,
            item_to_idx,
            embeddings,
            mode="transformer",
            sequence_encoder=encoder,
            device="cpu",
        )
    finally:
        session.close()

    assert vec.shape == (512,)
    assert np.isfinite(vec).all()
