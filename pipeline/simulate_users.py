"""Generate synthetic products, users, interactions, and placeholder images."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.db import Interaction, Product, User, init_db, session_scope
from config import IMAGE_DIR, INTERACTION_WEIGHTS


def _create_placeholder_image(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.new("RGB", (224, 224), color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 204, 204], outline=(255, 255, 255), width=3)
    draw.text((30, 95), f"PID-{seed}", fill=(255, 255, 255))
    img.save(path)


def main(num_users: int, num_products: int, num_interactions: int) -> None:
    init_db()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    categories = ["fashion", "electronics", "home", "beauty", "sports"]
    now_ts = int(time.time())

    with session_scope() as session:
        existing = session.query(Product).count()
        if existing == 0:
            for i in tqdm(range(num_products), desc="Creating products"):
                product_id = f"p{i:06d}"
                image_name = f"{product_id}.png"
                image_path = IMAGE_DIR / image_name
                if not image_path.exists():
                    _create_placeholder_image(image_path, i)

                session.add(
                    Product(
                        product_id=product_id,
                        title=f"Product {i}",
                        category=random.choice(categories),
                        price=round(random.uniform(5.0, 500.0), 2),
                        image_path=str(image_path),
                        description=f"Synthetic product description for item {i}.",
                    )
                )

        existing_users = session.query(User).count()
        if existing_users == 0:
            for i in tqdm(range(num_users), desc="Creating users"):
                session.add(User(user_id=f"u{i:05d}", created_at=now_ts - random.randint(0, 100000)))

        existing_interactions = session.query(Interaction).count()
        if existing_interactions == 0:
            event_types = ["view", "click", "purchase"]
            event_probs = [0.65, 0.25, 0.10]
            for _ in tqdm(range(num_interactions), desc="Creating interactions"):
                uid = f"u{random.randint(0, num_users - 1):05d}"
                pid = f"p{random.randint(0, num_products - 1):06d}"
                event_type = random.choices(event_types, weights=event_probs, k=1)[0]
                session.add(
                    Interaction(
                        user_id=uid,
                        product_id=pid,
                        event_type=event_type,
                        timestamp=now_ts - random.randint(0, 200000),
                        weight=INTERACTION_WEIGHTS[event_type],
                    )
                )

    print("Synthetic data generation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--products", type=int, default=10000)
    parser.add_argument("--interactions", type=int, default=50000)
    args = parser.parse_args()
    main(args.users, args.products, args.interactions)
