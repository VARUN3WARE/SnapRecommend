"""Generate Phase 2 pairwise training datasets from interactions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.db import session_scope
from pipeline.training_data import build_training_pairs, save_training_pairs


def main(num_negatives: int, train_split: float, seed: int) -> None:
    with session_scope() as session:
        train_rows, val_rows = build_training_pairs(
            db_session=session,
            num_negatives=num_negatives,
            train_split=train_split,
            seed=seed,
        )

    train_path, val_path = save_training_pairs(train_rows, val_rows)
    print(f"Saved train pairs: {len(train_rows)} -> {train_path}")
    print(f"Saved val pairs: {len(val_rows)} -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.num_negatives, args.train_split, args.seed)
