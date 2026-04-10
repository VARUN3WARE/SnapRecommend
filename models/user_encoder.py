"""User encoder based on weighted average of interaction history embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from api.db import Interaction
from config import EMBEDDING_DIM, MAX_HISTORY_LEN


def encode_user(
    user_id: str,
    db_session: Session,
    item_id_to_index: Dict[str, int],
    item_embeddings: np.ndarray,
) -> np.ndarray:
    stmt = (
        select(Interaction)
        .where(Interaction.user_id == user_id)
        .order_by(desc(Interaction.timestamp))
        .limit(MAX_HISTORY_LEN)
    )
    rows = db_session.execute(stmt).scalars().all()

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
        return np.mean(v, axis=0)
    user_vec = np.sum(v * w[:, None], axis=0) / w_sum
    norm = np.linalg.norm(user_vec)
    if norm == 0:
        return user_vec.astype(np.float32)
    return (user_vec / norm).astype(np.float32)
