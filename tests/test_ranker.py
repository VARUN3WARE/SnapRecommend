import numpy as np
import pytest

from models.ranker import Ranker, RankerTrainer, build_dataloader, build_ranking_features


torch = pytest.importorskip("torch")


def _sample_rows():
    return [
        {"user_id": "u1", "pos_item_id": "p1", "neg_item_ids": ["p2", "p3"]},
        {"user_id": "u2", "pos_item_id": "p2", "neg_item_ids": ["p1", "p3"]},
    ]


def _item_embeddings():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(3, 512)).astype(np.float32)
    item_to_idx = {"p1": 0, "p2": 1, "p3": 2}
    return vectors, item_to_idx


def test_build_ranking_features_shape():
    rng = np.random.default_rng(2)
    user_vec = rng.normal(size=(512,)).astype(np.float32)
    item_vec = rng.normal(size=(512,)).astype(np.float32)
    features = build_ranking_features(user_vec, item_vec)
    assert features.shape == (1538,)


def test_ranker_forward_and_checkpoint(tmp_path):
    rows = _sample_rows()
    item_embeddings, item_to_idx = _item_embeddings()
    loader = build_dataloader(rows, item_embeddings, item_to_idx, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    model = Ranker()
    scores = model(batch[0])
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()

    trainer = RankerTrainer(model=model, device="cpu", learning_rate=1e-3)
    loss = trainer.train_epoch(loader)
    assert loss > 0

    ckpt = tmp_path / "ranker.pt"
    trainer.save_checkpoint(ckpt, metadata={"ok": True})
    assert ckpt.exists()
    assert ckpt.with_suffix(".json").exists()

    fresh = Ranker()
    fresh_trainer = RankerTrainer(model=fresh, device="cpu", learning_rate=1e-3)
    fresh_trainer.load_checkpoint(ckpt)