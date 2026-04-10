"""FAISS index wrapper with numpy fallback retriever."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import FAISS_USE_GPU


try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class NumpyIndex:
    vectors: np.ndarray


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def build_index(embeddings: np.ndarray, use_gpu: bool = FAISS_USE_GPU):
    vectors = _normalize_rows(embeddings)
    if faiss is None:
        return NumpyIndex(vectors=vectors)

    dim = vectors.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.add(vectors)
            return gpu_index
        except Exception:
            pass

    cpu_index.add(vectors)
    return cpu_index


def save_index(index, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if faiss is not None and hasattr(index, "is_trained"):
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index
        faiss.write_index(cpu_index, str(path_obj))
        return

    if isinstance(index, NumpyIndex):
        np.save(str(path_obj) + ".npy", index.vectors)
        return

    raise TypeError("Unsupported index type for save_index")


def load_index(path: str, use_gpu: bool = FAISS_USE_GPU):
    path_obj = Path(path)
    npy_path = Path(str(path_obj) + ".npy")

    if path_obj.exists() and faiss is not None:
        index = faiss.read_index(str(path_obj))
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass
        return index

    if npy_path.exists():
        vectors = np.load(npy_path)
        return NumpyIndex(vectors=vectors.astype(np.float32))

    raise FileNotFoundError(f"No FAISS or numpy index found for: {path}")


def search(index, query_vec: np.ndarray, k: int = 100):
    query = query_vec.astype(np.float32)
    if query.ndim == 1:
        query = query[None, :]
    query = _normalize_rows(query)

    if isinstance(index, NumpyIndex):
        sims = index.vectors @ query[0]
        top_idx = np.argsort(-sims)[:k]
        return sims[top_idx].astype(np.float32), top_idx.astype(np.int64)

    distances, indices = index.search(query, k)
    return distances[0].astype(np.float32), indices[0].astype(np.int64)
