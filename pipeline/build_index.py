"""Build and save FAISS (or numpy fallback) index from precomputed embeddings."""

from __future__ import annotations

import sys

import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import EMBEDDINGS_PATH, FAISS_INDEX_PATH
from retrieval.faiss_index import build_index, save_index


def main() -> None:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("Embeddings file not found. Run pipeline/embed_items.py first.")

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    index = build_index(embeddings, use_gpu=True)
    save_index(index, str(FAISS_INDEX_PATH))
    print(f"Index saved at: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
