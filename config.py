"""Global configuration constants for the multimodal recommender MVP."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "item_embeddings.npy"
ITEM_IDS_PATH = PROCESSED_DIR / "item_ids.npy"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss.index"
DB_PATH = DATA_DIR / "metadata.db"

# Model
CLIP_MODEL_NAME = "ViT-B/32"
EMBEDDING_DIM = 512
CLIP_BATCH_SIZE = 64
IMAGE_SIZE = 224

# Retrieval
FAISS_TOP_K = 100
FINAL_TOP_N = 10
FAISS_USE_GPU = True

# User encoder
MAX_HISTORY_LEN = 20
INTERACTION_WEIGHTS = {"purchase": 1.0, "click": 0.3, "view": 0.1}

# Fusion (Phase 1)
USER_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4

# Training placeholders (Phase 2)
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 20
DEVICE = "cuda"

# API
MAX_INPUT_IMAGE_PX = 1024
