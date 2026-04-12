"""Build pairwise training data with negative sampling for Phase 2 models."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import Interaction, Product
from config import NEGATIVE_SAMPLES, TRAIN_PAIRS_PATH, TRAIN_SPLIT, VAL_PAIRS_PATH


def _group_sorted_interactions(rows: list[Interaction]) -> dict[str, list[Interaction]]:
    grouped: dict[str, list[Interaction]] = defaultdict(list)
    for row in rows:
        grouped[row.user_id].append(row)
    for user_id in grouped:
        grouped[user_id].sort(key=lambda r: r.timestamp)
    return grouped


def _sample_negatives(
    all_product_ids: list[str],
    forbidden: set[str],
    rng: random.Random,
    num_negatives: int,
) -> list[str]:
    pool = [pid for pid in all_product_ids if pid not in forbidden]
    if not pool:
        return []
    if len(pool) <= num_negatives:
        return pool
    return rng.sample(pool, k=num_negatives)


def _rows_to_pairs(
    rows: list[Interaction],
    all_product_ids: list[str],
    user_seen: dict[str, set[str]],
    rng: random.Random,
    num_negatives: int,
) -> list[dict]:
    pairs = []
    for row in rows:
        negatives = _sample_negatives(
            all_product_ids=all_product_ids,
            forbidden=user_seen[row.user_id],
            rng=rng,
            num_negatives=num_negatives,
        )
        if not negatives:
            continue
        pairs.append(
            {
                "user_id": row.user_id,
                "pos_item_id": row.product_id,
                "neg_item_ids": negatives,
                "timestamp": int(row.timestamp),
                "event_type": row.event_type,
                "weight": float(row.weight),
            }
        )
    return pairs


def build_training_pairs(
    db_session: Session,
    num_negatives: int = NEGATIVE_SAMPLES,
    train_split: float = TRAIN_SPLIT,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    all_products = db_session.execute(select(Product.product_id)).scalars().all()
    all_product_ids = [str(pid) for pid in all_products]
    if not all_product_ids:
        raise RuntimeError("No products found. Run pipeline/simulate_users.py first.")

    all_rows = db_session.execute(select(Interaction)).scalars().all()
    if not all_rows:
        raise RuntimeError("No interactions found. Run pipeline/simulate_users.py first.")

    grouped = _group_sorted_interactions(all_rows)
    user_seen = {
        uid: {row.product_id for row in rows}
        for uid, rows in grouped.items()
    }

    train_rows: list[Interaction] = []
    val_rows: list[Interaction] = []

    # Hold out each user's latest interaction for validation when possible.
    for _uid, rows in grouped.items():
        if len(rows) < 2:
            continue
        train_rows.extend(rows[:-1])
        val_rows.append(rows[-1])

    # Optional global split fallback for extremely small datasets.
    if not val_rows and train_rows:
        split_idx = max(1, int(len(train_rows) * train_split))
        val_rows = train_rows[split_idx:]
        train_rows = train_rows[:split_idx]

    train_pairs = _rows_to_pairs(train_rows, all_product_ids, user_seen, rng, num_negatives)
    val_pairs = _rows_to_pairs(val_rows, all_product_ids, user_seen, rng, num_negatives)
    return train_pairs, val_pairs


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def save_training_pairs(train_rows: list[dict], val_rows: list[dict]) -> tuple[Path, Path]:
    write_jsonl(train_rows, TRAIN_PAIRS_PATH)
    write_jsonl(val_rows, VAL_PAIRS_PATH)
    return TRAIN_PAIRS_PATH, VAL_PAIRS_PATH
