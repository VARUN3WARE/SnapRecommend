"""Small CLI tool to query the FAISS/numpy index with text or image input.

Examples:
  python pipeline/query_index.py --text "red sneaker" --top-k 5
  python pipeline/query_index.py --image data/images/1.png --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.faiss_index import load_index
from retrieval.search import retrieve_item_ids
from config import EMBEDDINGS_PATH, ITEM_IDS_PATH, FAISS_INDEX_PATH
from models.clip_encoder import ClipEncoder
from api.db import session_scope, Product


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query the item index with text or image")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text query")
    group.add_argument("--image", help="Path to image file to query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--use-gpu", action="store_true", help="Attempt to use GPU for FAISS load")
    p.add_argument("--device", default="auto", help="Device for CLIP encoder: auto|cpu|cuda")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def main() -> None:
    args = parse_args()
    if not EMBEDDINGS_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit("Embeddings or item ids missing. Run pipeline/embed_items.py first.")

    embeddings = None
    item_ids = np.load(ITEM_IDS_PATH)

    index = None
    try:
        index = load_index(str(FAISS_INDEX_PATH), use_gpu=args.use_gpu)
    except Exception:
        print("Could not load FAISS index; attempting numpy fallback")

    encoder = ClipEncoder(device=resolve_device(args.device))

    if args.text:
        query_vec = encoder.encode_text(args.text)
    else:
        query_vec = encoder.encode_image(args.image)

    results = retrieve_item_ids(index=index, item_ids=item_ids, query_vec=query_vec, k=args.top_k)

    # Optionally map to product titles if DB available
    with session_scope() as session:
        print("Top results:")
        for pid, score in results:
            prod = session.get(Product, pid)
            title = prod.title if prod is not None else "<unknown>"
            print(f"{pid}\t{score:.4f}\t{title}")


if __name__ == "__main__":
    main()
