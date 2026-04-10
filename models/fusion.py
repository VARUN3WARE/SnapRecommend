"""Fusion layer for combining user and query vectors."""

from __future__ import annotations

import numpy as np

from config import IMAGE_WEIGHT, USER_WEIGHT


def fuse(user_vec: np.ndarray, image_vec: np.ndarray, text_vec: np.ndarray | None = None) -> np.ndarray:
    if text_vec is not None:
        query_vec = (image_vec + text_vec) / 2.0
    else:
        query_vec = image_vec

    if np.linalg.norm(user_vec) == 0:
        final_vec = query_vec.astype(np.float32)
    else:
        final_vec = USER_WEIGHT * user_vec + IMAGE_WEIGHT * query_vec

    norm = np.linalg.norm(final_vec)
    if norm == 0:
        return final_vec.astype(np.float32)
    return (final_vec / norm).astype(np.float32)
