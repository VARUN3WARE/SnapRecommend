"""Batch-encode product images into item embeddings."""

from __future__ import annotations

import sys

import numpy as np
from sqlalchemy import select
from tqdm import tqdm
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.db import Product, session_scope
from config import CLIP_BATCH_SIZE, EMBEDDINGS_PATH, ITEM_IDS_PATH
from models.clip_encoder import ClipEncoder


def main() -> None:
    encoder = ClipEncoder(device="cuda")

    with session_scope() as session:
        products = session.execute(select(Product)).scalars().all()

    if not products:
        raise RuntimeError("No products found. Run pipeline/simulate_users.py first.")

    item_ids = []
    image_paths = []
    for p in products:
        item_ids.append(p.product_id)
        image_paths.append(p.image_path)

    embeddings = []
    for i in tqdm(range(0, len(image_paths), CLIP_BATCH_SIZE), desc="Encoding items"):
        chunk_paths = image_paths[i : i + CLIP_BATCH_SIZE]
        chunk_emb = encoder.encode_batch(chunk_paths)
        embeddings.append(chunk_emb)

    item_embeddings = np.vstack(embeddings).astype(np.float32)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, item_embeddings)
    np.save(ITEM_IDS_PATH, np.array(item_ids))
    print(f"Saved embeddings: {item_embeddings.shape} -> {EMBEDDINGS_PATH}")
    print(f"Saved item ids: {len(item_ids)} -> {ITEM_IDS_PATH}")


if __name__ == "__main__":
    main()
