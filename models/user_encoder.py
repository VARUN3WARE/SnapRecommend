"""User encoder based on weighted average of interaction history embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from api.db import Interaction
from config import EMBEDDING_DIM, MAX_HISTORY_LEN, TRANSFORMER_HEADS, TRANSFORMER_LAYERS, USER_ENCODER_MODE


try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _fetch_user_rows(user_id: str, db_session: Session) -> list[Interaction]:
    stmt = (
        select(Interaction)
        .where(Interaction.user_id == user_id)
        .order_by(desc(Interaction.timestamp))
        .limit(MAX_HISTORY_LEN)
    )
    return db_session.execute(stmt).scalars().all()


def _weighted_average_from_rows(
    rows: list[Interaction],
    item_id_to_index: Dict[str, int],
    item_embeddings: np.ndarray,
) -> np.ndarray:
    vectors = []
    weights = []
    for row in rows:
        idx = item_id_to_index.get(row.product_id)
        if idx is None:
            continue
        vectors.append(item_embeddings[idx])
        weights.append(float(row.weight))

    if not vectors:
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    v = np.stack(vectors, axis=0).astype(np.float32)
    w = np.array(weights, dtype=np.float32)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return _l2_normalize(np.mean(v, axis=0))
    return _l2_normalize(np.sum(v * w[:, None], axis=0) / w_sum)


if nn is not None and torch is not None:
    class TransformerSequenceUserEncoder(nn.Module):
        def __init__(self, embedding_dim: int = EMBEDDING_DIM):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=TRANSFORMER_HEADS,
                dim_feedforward=embedding_dim * 2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
            batch_size = sequence_embeddings.shape[0]
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, sequence_embeddings], dim=1)
            out = self.encoder(x)
            cls_out = out[:, 0, :]
            return torch.nn.functional.normalize(cls_out, p=2, dim=-1)

        def encode_numpy(self, history_vectors: np.ndarray, device: str = "cpu") -> np.ndarray:
            if history_vectors.size == 0:
                return np.zeros((EMBEDDING_DIM,), dtype=np.float32)
            self.eval()
            with torch.no_grad():
                seq = torch.tensor(history_vectors[None, :, :], dtype=torch.float32, device=device)
                vec = self.forward(seq)[0].detach().cpu().numpy().astype(np.float32)
                return _l2_normalize(vec)
else:
    class TransformerSequenceUserEncoder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for TransformerSequenceUserEncoder")


def encode_user(
    user_id: str,
    db_session: Session,
    item_id_to_index: Dict[str, int],
    item_embeddings: np.ndarray,
    mode: str = USER_ENCODER_MODE,
    sequence_encoder: TransformerSequenceUserEncoder | None = None,
    device: str = "cpu",
) -> np.ndarray:
    rows = _fetch_user_rows(user_id=user_id, db_session=db_session)

    if mode == "transformer" and sequence_encoder is not None:
        history_vectors = []
        for row in rows:
            idx = item_id_to_index.get(row.product_id)
            if idx is None:
                continue
            history_vectors.append(item_embeddings[idx])
        if history_vectors:
            v = np.stack(history_vectors, axis=0).astype(np.float32)
            return sequence_encoder.encode_numpy(v, device=device)

    return _weighted_average_from_rows(
        rows=rows,
        item_id_to_index=item_id_to_index,
        item_embeddings=item_embeddings,
    )
