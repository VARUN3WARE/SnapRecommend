"""Search helpers for retrieval and id mapping."""

from __future__ import annotations

from typing import List

import numpy as np

from retrieval.faiss_index import search


def retrieve_item_ids(index, item_ids: np.ndarray, query_vec: np.ndarray, k: int) -> List[tuple[str, float]]:
    distances, indices = search(index, query_vec, k=k)
    results: List[tuple[str, float]] = []
    for score, idx in zip(distances, indices):
        if idx < 0 or idx >= len(item_ids):
            continue
        results.append((str(item_ids[idx]), float(score)))
    return results
