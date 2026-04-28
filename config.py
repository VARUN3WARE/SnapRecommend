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
RUNS_DIR = PROCESSED_DIR / "runs"
CHECKPOINTS_DIR = PROCESSED_DIR / "checkpoints"
TRAIN_DATA_DIR = PROCESSED_DIR / "training"
TRAIN_PAIRS_PATH = TRAIN_DATA_DIR / "train_pairs.jsonl"
VAL_PAIRS_PATH = TRAIN_DATA_DIR / "val_pairs.jsonl"

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
USER_ENCODER_MODE = "legacy"  # allowed: legacy, transformer
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 8

# Fusion (Phase 1)
USER_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4

# Training placeholders (Phase 2)
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 20
DEVICE = "cuda"
SEED = 42

# Phase mode
PHASE_MODE = "phase1"  # allowed: phase1, phase2
USE_RANKER = False

# Phase 2 training defaults
TRAIN_SPLIT = 0.8
NEGATIVE_SAMPLES = 4
TWO_TOWER_HIDDEN_DIM = 512
TWO_TOWER_DROPOUT = 0.1
RANKER_HIDDEN_DIMS = (256, 64)
RANKER_DROPOUT = 0.2
RANKER_FEATURE_DIM = EMBEDDING_DIM * 3 + 2

# API
MAX_INPUT_IMAGE_PX = 1024
