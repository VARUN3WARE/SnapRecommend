"""Train a Phase 2 reranking model from pairwise training rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EMBEDDINGS_PATH,
    EPOCHS,
    ITEM_IDS_PATH,
    LEARNING_RATE,
    RANKER_CHECKPOINT_PATH,
)
from models.ranker import Ranker, RankerTrainer, build_dataloader
from pipeline.experiment import (
    ExperimentConfig,
    RunMetadata,
    create_run_id,
    mark_run_finished,
    save_experiment_config,
    save_run_metadata,
)
from pipeline.training_data import TRAIN_PAIRS_PATH, VAL_PAIRS_PATH


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing training rows file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(epochs: int, batch_size: int, learning_rate: float, device: str) -> None:
    train_rows = _load_jsonl(TRAIN_PAIRS_PATH)
    val_rows = _load_jsonl(VAL_PAIRS_PATH)

    item_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    item_ids = np.load(ITEM_IDS_PATH)
    item_to_idx = {str(pid): i for i, pid in enumerate(item_ids)}

    train_loader = build_dataloader(
        rows=train_rows,
        item_embeddings=item_embeddings,
        item_to_idx=item_to_idx,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        rows=val_rows,
        item_embeddings=item_embeddings,
        item_to_idx=item_to_idx,
        batch_size=batch_size,
        shuffle=False,
    )

    run_id = create_run_id("ranker")
    run_config = ExperimentConfig(
        run_name="ranker",
        phase_mode="phase2",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    save_experiment_config(run_config, run_id)
    save_run_metadata(
        RunMetadata(
            run_id=run_id,
            run_name=run_config.run_name,
            phase_mode=run_config.phase_mode,
            seed=run_config.seed,
            started_at=int(time.time()),
        )
    )

    model = Ranker()
    trainer = RankerTrainer(model=model, device=device, learning_rate=learning_rate)

    last_train = 0.0
    last_val = 0.0
    for epoch in range(1, epochs + 1):
        last_train = trainer.train_epoch(train_loader)
        last_val = trainer.evaluate_epoch(val_loader)
        print(f"epoch={epoch} train_loss={last_train:.4f} val_loss={last_val:.4f}")

    ckpt_path = CHECKPOINTS_DIR / f"{run_id}_ranker.pt"
    trainer.save_checkpoint(
        ckpt_path,
        metadata={"run_id": run_id, "epochs": epochs, "train_loss": last_train, "val_loss": last_val},
    )
    shutil.copyfile(ckpt_path, RANKER_CHECKPOINT_PATH)
    mark_run_finished(
        run_id,
        status="completed",
        metrics={"train_loss": last_train, "val_loss": last_val},
        artifacts={"checkpoint": str(ckpt_path), "latest_ranker": str(RANKER_CHECKPOINT_PATH)},
    )
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Updated latest ranker: {RANKER_CHECKPOINT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.device)