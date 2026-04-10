"""CLIP encoder wrapper with deterministic fallback for environments without CLIP."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from config import CLIP_BATCH_SIZE, CLIP_MODEL_NAME, EMBEDDING_DIM, IMAGE_SIZE


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class ClipEncoder:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.clip = None
        self.model = None
        self.preprocess = None
        self._load_clip_if_available()

    def _load_clip_if_available(self) -> None:
        if torch is None:
            return
        try:
            import clip  # type: ignore

            resolved_device = self.device if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=resolved_device)
            self.model.eval()
            self.clip = clip
            self.device = resolved_device
        except Exception:
            self.clip = None
            self.model = None
            self.preprocess = None

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _hash_to_vec(payload: bytes, dim: int = EMBEDDING_DIM) -> np.ndarray:
        digest = hashlib.sha256(payload).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
        return ClipEncoder._l2_normalize(vec)

    def encode_text(self, text: str) -> np.ndarray:
        if self.model is None or self.clip is None or torch is None:
            return self._hash_to_vec(text.encode("utf-8"))

        with torch.no_grad():
            tokens = self.clip.tokenize([text]).to(self.device)
            features = self.model.encode_text(tokens)
            vec = features[0].detach().cpu().numpy().astype(np.float32)
            return self._l2_normalize(vec)

    def _encode_pil_clip(self, image: Image.Image) -> np.ndarray:
        assert self.preprocess is not None and self.model is not None and torch is not None
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            vec = features[0].detach().cpu().numpy().astype(np.float32)
            return self._l2_normalize(vec)

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        if max(image.size) > IMAGE_SIZE:
            image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))

        if self.model is None:
            with BytesIO() as buff:
                image.save(buff, format="PNG")
                return self._hash_to_vec(buff.getvalue())
        return self._encode_pil_clip(image)

    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return self.encode_pil(image)

    def encode_batch(self, paths: Iterable[str]) -> np.ndarray:
        path_list = list(paths)
        if not path_list:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        if self.model is None or self.preprocess is None or torch is None:
            vectors = [self.encode_image(p) for p in path_list]
            return np.stack(vectors, axis=0).astype(np.float32)

        outputs = []
        with torch.no_grad():
            for i in range(0, len(path_list), CLIP_BATCH_SIZE):
                chunk = path_list[i : i + CLIP_BATCH_SIZE]
                tensors = []
                for path in chunk:
                    image = Image.open(Path(path)).convert("RGB")
                    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
                    tensors.append(self.preprocess(image))
                image_input = torch.stack(tensors).to(self.device)
                feats = self.model.encode_image(image_input)
                feats = feats.detach().cpu().numpy().astype(np.float32)
                norms = np.linalg.norm(feats, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                outputs.append(feats / norms)

        return np.vstack(outputs).astype(np.float32)
