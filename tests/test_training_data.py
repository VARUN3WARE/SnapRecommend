from pathlib import Path

from api.db import Interaction, Product, User, get_session_factory, init_db
from pipeline.training_data import build_training_pairs


def _seed_minimal_dataset(db_path: Path) -> None:
    init_db(str(db_path))
    session_factory = get_session_factory(str(db_path))
    session = session_factory()
    try:
        for i in range(6):
            session.add(
                Product(
                    product_id=f"p{i:06d}",
                    title=f"P{i}",
                    category="cat",
                    price=10.0,
                    image_path=f"data/images/p{i:06d}.png",
                    description="demo",
                )
            )

        session.add(User(user_id="u00000", created_at=1))
        session.add(User(user_id="u00001", created_at=1))

        # u00000 has 3 interactions; latest should go to val.
        session.add(Interaction(user_id="u00000", product_id="p000000", event_type="view", timestamp=10, weight=0.1))
        session.add(Interaction(user_id="u00000", product_id="p000001", event_type="click", timestamp=20, weight=0.3))
        session.add(Interaction(user_id="u00000", product_id="p000002", event_type="purchase", timestamp=30, weight=1.0))

        # u00001 has 2 interactions; latest should go to val.
        session.add(Interaction(user_id="u00001", product_id="p000003", event_type="view", timestamp=40, weight=0.1))
        session.add(Interaction(user_id="u00001", product_id="p000004", event_type="purchase", timestamp=50, weight=1.0))

        session.commit()
    finally:
        session.close()


def test_build_training_pairs_holdout_latest_per_user(tmp_path: Path):
    db_path = tmp_path / "train.db"
    _seed_minimal_dataset(db_path)

    session_factory = get_session_factory(str(db_path))
    session = session_factory()
    try:
        train_rows, val_rows = build_training_pairs(session, num_negatives=2, seed=42)
    finally:
        session.close()

    assert len(train_rows) == 3
    assert len(val_rows) == 2

    val_pos_ids = {row["pos_item_id"] for row in val_rows}
    assert val_pos_ids == {"p000002", "p000004"}

    for row in train_rows + val_rows:
        assert len(row["neg_item_ids"]) > 0
        assert row["pos_item_id"] not in row["neg_item_ids"]
