import numpy as np
import pytest

from models.two_tower import TwoTowerModel, TwoTowerTrainer, bpr_loss, build_dataloader, build_user_index


torch = pytest.importorskip("torch")


def _sample_rows():
    return [
        {"user_id": "u1", "pos_item_id": "p1", "neg_item_ids": ["p2", "p3"]},
        {"user_id": "u2", "pos_item_id": "p2", "neg_item_ids": ["p1", "p3"]},
        {"user_id": "u1", "pos_item_id": "p3", "neg_item_ids": ["p2", "p4"]},
    ]


def _item_embeddings():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(4, 512)).astype(np.float32)
    item_to_idx = {"p1": 0, "p2": 1, "p3": 2, "p4": 3}
    return vectors, item_to_idx


def test_two_tower_forward_shapes():
    rows = _sample_rows()
    item_embeddings, item_to_idx = _item_embeddings()
    user_to_idx = build_user_index(rows)

    loader = build_dataloader(
        rows=rows,
        item_embeddings=item_embeddings,
        item_to_idx=item_to_idx,
        user_to_idx=user_to_idx,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    model = TwoTowerModel(num_users=len(user_to_idx))
    user_emb, pos_emb, neg_emb = model(*batch)

    assert user_emb.shape[0] == 2
    assert user_emb.shape[1] == 512
    assert pos_emb.shape == (2, 512)
    assert neg_emb.shape[0] == 2
    assert neg_emb.shape[2] == 512

    loss = bpr_loss(user_emb, pos_emb, neg_emb)
    assert float(loss.item()) > 0.0


def test_two_tower_checkpoint_roundtrip(tmp_path):
    rows = _sample_rows()
    item_embeddings, item_to_idx = _item_embeddings()
    user_to_idx = build_user_index(rows)

    loader = build_dataloader(
        rows=rows,
        item_embeddings=item_embeddings,
        item_to_idx=item_to_idx,
        user_to_idx=user_to_idx,
        batch_size=2,
        shuffle=True,
    )

    model = TwoTowerModel(num_users=len(user_to_idx))
    trainer = TwoTowerTrainer(model=model, device="cpu", learning_rate=1e-3)
    stats = trainer.train_epoch(loader)
    assert stats.batches > 0

    ckpt = tmp_path / "two_tower.pt"
    trainer.save_checkpoint(ckpt, metadata={"ok": True})
    assert ckpt.exists()
    assert ckpt.with_suffix(".json").exists()

    fresh = TwoTowerModel(num_users=len(user_to_idx))
    fresh_trainer = TwoTowerTrainer(model=fresh, device="cpu", learning_rate=1e-3)
    fresh_trainer.load_checkpoint(ckpt)
