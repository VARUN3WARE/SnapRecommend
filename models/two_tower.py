"""Two-tower model and trainer for Phase 2 retrieval learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import EMBEDDING_DIM, TWO_TOWER_DROPOUT, TWO_TOWER_HIDDEN_DIM


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = object
    Dataset = object

TORCH_AVAILABLE = torch is not None and nn is not None and F is not None


@dataclass
class TrainStats:
    loss: float
    batches: int


if TORCH_AVAILABLE:
    class PairwiseDataset(Dataset):
        def __init__(self, rows: list[dict], user_to_idx: dict[str, int], item_embeddings: np.ndarray, item_to_idx: dict[str, int]):
            self.rows = rows
            self.user_to_idx = user_to_idx
            self.item_embeddings = item_embeddings.astype(np.float32)
            self.item_to_idx = item_to_idx

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            user_idx = self.user_to_idx[row["user_id"]]
            pos_idx = self.item_to_idx[row["pos_item_id"]]
            neg_indices = [self.item_to_idx[nid] for nid in row["neg_item_ids"] if nid in self.item_to_idx]
            if not neg_indices:
                neg_indices = [pos_idx]

            pos_vec = self.item_embeddings[pos_idx]
            neg_vecs = self.item_embeddings[neg_indices]
            return user_idx, pos_vec, neg_vecs


    class UserTower(nn.Module):
        def __init__(self, num_users: int, hidden_dim: int, out_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=hidden_dim)
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, user_idx):
            x = self.embedding(user_idx)
            x = self.proj(x)
            return F.normalize(x, p=2, dim=-1)


    class ItemTower(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, item_vecs):
            x = self.net(item_vecs)
            return F.normalize(x, p=2, dim=-1)


    class TwoTowerModel(nn.Module):
        def __init__(self, num_users: int, in_dim: int = EMBEDDING_DIM, hidden_dim: int = TWO_TOWER_HIDDEN_DIM):
            super().__init__()
            self.user_tower = UserTower(num_users=num_users, hidden_dim=hidden_dim, out_dim=in_dim)
            self.item_tower = ItemTower(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=in_dim, dropout=TWO_TOWER_DROPOUT)

        def forward(self, user_idx, pos_vecs, neg_vecs):
            user_emb = self.user_tower(user_idx)
            pos_emb = self.item_tower(pos_vecs)

            bsz, n_neg, dim = neg_vecs.shape
            flat_negs = neg_vecs.view(bsz * n_neg, dim)
            neg_emb = self.item_tower(flat_negs).view(bsz, n_neg, dim)
            return user_emb, pos_emb, neg_emb


    def bpr_loss(user_emb, pos_emb, neg_emb):
        pos_score = torch.sum(user_emb * pos_emb, dim=-1, keepdim=True)
        neg_score = torch.sum(user_emb.unsqueeze(1) * neg_emb, dim=-1)
        margin = pos_score - neg_score
        return -torch.log(torch.sigmoid(margin) + 1e-8).mean()
else:
    class PairwiseDataset:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PairwiseDataset")


    class TwoTowerModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for TwoTowerModel")


    def bpr_loss(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("PyTorch is required for bpr_loss")


def build_user_index(rows: list[dict]) -> dict[str, int]:
    users = sorted({row["user_id"] for row in rows})
    return {uid: i for i, uid in enumerate(users)}


class TwoTowerTrainer:
    def __init__(self, model: TwoTowerModel, device: str = "cpu", learning_rate: float = 1e-4):
        if torch is None:
            raise RuntimeError("PyTorch is required for TwoTowerTrainer")
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _to_device_batch(self, batch):
        user_idx, pos_vec, neg_vecs = batch
        user_idx = user_idx.to(self.device)
        pos_vec = pos_vec.to(self.device)
        neg_vecs = neg_vecs.to(self.device)
        return user_idx, pos_vec, neg_vecs

    def train_epoch(self, dataloader) -> TrainStats:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for batch in dataloader:
            user_idx, pos_vec, neg_vecs = self._to_device_batch(batch)
            self.optimizer.zero_grad()
            user_emb, pos_emb, neg_emb = self.model(user_idx, pos_vec, neg_vecs)
            loss = bpr_loss(user_emb, pos_emb, neg_emb)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        avg_loss = total_loss / max(1, batches)
        return TrainStats(loss=avg_loss, batches=batches)

    def evaluate_epoch(self, dataloader) -> TrainStats:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in dataloader:
                user_idx, pos_vec, neg_vecs = self._to_device_batch(batch)
                user_emb, pos_emb, neg_emb = self.model(user_idx, pos_vec, neg_vecs)
                loss = bpr_loss(user_emb, pos_emb, neg_emb)
                total_loss += float(loss.detach().cpu().item())
                batches += 1
        avg_loss = total_loss / max(1, batches)
        return TrainStats(loss=avg_loss, batches=batches)

    def save_checkpoint(self, path: Path, metadata: dict | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict()}, path)
        if metadata is not None:
            metadata_path = path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])


def build_dataloader(
    rows: list[dict],
    item_embeddings: np.ndarray,
    item_to_idx: dict[str, int],
    user_to_idx: dict[str, int],
    batch_size: int,
    shuffle: bool,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for build_dataloader")

    dataset = PairwiseDataset(
        rows=rows,
        user_to_idx=user_to_idx,
        item_embeddings=item_embeddings,
        item_to_idx=item_to_idx,
    )

    def _collate(batch):
        user_idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
        pos_vec = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        max_negs = max(len(b[2]) for b in batch)
        neg_tensors = []
        for _, _, neg in batch:
            neg_array = np.asarray(neg, dtype=np.float32)
            if len(neg_array) < max_negs:
                pad = np.repeat(neg_array[-1][None, :], repeats=max_negs - len(neg_array), axis=0)
                neg_array = np.concatenate([neg_array, pad], axis=0)
            neg_tensors.append(neg_array)
        neg_vecs = torch.tensor(np.stack(neg_tensors, axis=0), dtype=torch.float32)
        return user_idx, pos_vec, neg_vecs

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)
