"""Phase 2 ranking model and trainer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import EMBEDDING_DIM, RANKER_DROPOUT, RANKER_FEATURE_DIM, RANKER_HIDDEN_DIMS


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = object
    Dataset = object


TORCH_AVAILABLE = torch is not None and nn is not None


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _hash_vector(value: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return _l2_normalize(rng.normal(size=(dim,)).astype(np.float32))


def build_ranking_features(
    user_vec: np.ndarray,
    item_vec: np.ndarray,
    delta_vec: np.ndarray | None = None,
    price: float = 0.0,
    category_id: float = 0.0,
) -> np.ndarray:
    if delta_vec is None:
        delta_vec = item_vec - user_vec
    features = np.concatenate(
        [
            user_vec.astype(np.float32),
            item_vec.astype(np.float32),
            delta_vec.astype(np.float32),
            np.array([price, category_id], dtype=np.float32),
        ]
    )
    if features.shape[0] != RANKER_FEATURE_DIM:
        raise ValueError(f"Expected feature dim {RANKER_FEATURE_DIM}, got {features.shape[0]}")
    return features


def _pairwise_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


if TORCH_AVAILABLE:
    class Ranker(nn.Module):
        def __init__(self, input_dim: int = RANKER_FEATURE_DIM):
            super().__init__()
            layers = []
            current_dim = input_dim
            for hidden_dim in RANKER_HIDDEN_DIMS:
                layers.extend(
                    [
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(RANKER_DROPOUT),
                    ]
                )
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            logits = self.network(features)
            return torch.sigmoid(logits).squeeze(-1)

        def score(self, features: torch.Tensor) -> torch.Tensor:
            return self.forward(features)


    class RankingDataset(Dataset):
        def __init__(self, rows: list[dict], item_embeddings: np.ndarray, item_to_idx: dict[str, int]):
            self.rows = rows
            self.item_embeddings = item_embeddings.astype(np.float32)
            self.item_to_idx = item_to_idx

        def __len__(self) -> int:
            return len(self.rows)

        def _user_vec(self, user_id: str) -> np.ndarray:
            return _hash_vector(user_id)

        def _build_feature(self, user_id: str, item_id: str) -> np.ndarray:
            user_vec = self._user_vec(user_id)
            item_vec = self.item_embeddings[self.item_to_idx[item_id]]
            return build_ranking_features(user_vec=user_vec, item_vec=item_vec)

        def __getitem__(self, index: int):
            row = self.rows[index]
            pos_feature = self._build_feature(row["user_id"], row["pos_item_id"])
            neg_features = [self._build_feature(row["user_id"], neg_id) for neg_id in row["neg_item_ids"] if neg_id in self.item_to_idx]
            if not neg_features:
                neg_features = [pos_feature]
            return pos_feature, np.stack(neg_features, axis=0)


    class RankerTrainer:
        def __init__(self, model: Ranker, device: str = "cpu", learning_rate: float = 1e-4):
            self.model = model.to(device)
            self.device = device
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        def _to_device(self, batch):
            pos_features, neg_features = batch
            return pos_features.to(self.device), neg_features.to(self.device)

        def train_epoch(self, dataloader) -> float:
            self.model.train()
            total = 0.0
            batches = 0
            for batch in dataloader:
                pos_features, neg_features = self._to_device(batch)
                self.optimizer.zero_grad()
                pos_score = self.model(pos_features)
                bsz, n_neg, dim = neg_features.shape
                neg_scores = self.model(neg_features.view(bsz * n_neg, dim)).view(bsz, n_neg)
                loss = _pairwise_loss(pos_score.unsqueeze(1).expand_as(neg_scores), neg_scores)
                loss.backward()
                self.optimizer.step()
                total += float(loss.detach().cpu().item())
                batches += 1
            return total / max(1, batches)

        def evaluate_epoch(self, dataloader) -> float:
            self.model.eval()
            total = 0.0
            batches = 0
            with torch.no_grad():
                for batch in dataloader:
                    pos_features, neg_features = self._to_device(batch)
                    pos_score = self.model(pos_features)
                    bsz, n_neg, dim = neg_features.shape
                    neg_scores = self.model(neg_features.view(bsz * n_neg, dim)).view(bsz, n_neg)
                    loss = _pairwise_loss(pos_score.unsqueeze(1).expand_as(neg_scores), neg_scores)
                    total += float(loss.detach().cpu().item())
                    batches += 1
            return total / max(1, batches)

        def save_checkpoint(self, path: Path, metadata: dict | None = None) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": self.model.state_dict()}, path)
            if metadata is not None:
                path.with_suffix(".json").write_text(
                    __import__("json").dumps(metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            return path

        def load_checkpoint(self, path: Path) -> None:
            payload = torch.load(path, map_location=self.device)
            self.model.load_state_dict(payload["model_state_dict"])


    def build_dataloader(rows: list[dict], item_embeddings: np.ndarray, item_to_idx: dict[str, int], batch_size: int, shuffle: bool):
        dataset = RankingDataset(rows=rows, item_embeddings=item_embeddings, item_to_idx=item_to_idx)

        def _collate(batch):
            pos = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            max_negs = max(b[1].shape[0] for b in batch)
            neg_stack = []
            for _, neg in batch:
                neg_arr = np.asarray(neg, dtype=np.float32)
                if neg_arr.shape[0] < max_negs:
                    pad = np.repeat(neg_arr[-1][None, :], repeats=max_negs - neg_arr.shape[0], axis=0)
                    neg_arr = np.concatenate([neg_arr, pad], axis=0)
                neg_stack.append(neg_arr)
            neg = torch.tensor(np.stack(neg_stack, axis=0), dtype=torch.float32)
            return pos, neg

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)
else:
    class Ranker:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for Ranker")

        def score(self, _features):
            raise RuntimeError("PyTorch is required for Ranker")


    class RankerTrainer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for RankerTrainer")


    def build_dataloader(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("PyTorch is required for build_dataloader")
