"""Build and save FAISS (or numpy fallback) index from precomputed embeddings."""

from __future__ import annotations

import sys

import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging

from config import EMBEDDINGS_PATH, FAISS_INDEX_PATH, FAISS_USE_GPU
from retrieval.faiss_index import build_index, save_index


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS or numpy index from embeddings")
    p.add_argument("--embeddings", default=str(EMBEDDINGS_PATH), help="Path to numpy embeddings file")
    p.add_argument("--index", default=str(FAISS_INDEX_PATH), help="Output index path")
    p.add_argument("--use-gpu", dest="use_gpu", action="store_true", help="Attempt to use GPU for FAISS")
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Force CPU-only index")
    p.set_defaults(use_gpu=FAISS_USE_GPU)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError("Embeddings file not found. Run pipeline/embed_items.py first.")

    embeddings = np.load(emb_path).astype(np.float32)
    logging.info("Loaded embeddings: %s -> %s", embeddings.shape, emb_path)
    index = build_index(embeddings, use_gpu=args.use_gpu)
    save_index(index, str(args.index))
    logging.info("Index saved at: %s", args.index)


if __name__ == "__main__":
    main()
