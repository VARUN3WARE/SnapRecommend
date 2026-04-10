from pathlib import Path

import numpy as np
from PIL import Image

from models.clip_encoder import ClipEncoder


def test_encode_text_shape_and_norm():
    encoder = ClipEncoder(device="cpu")
    vec = encoder.encode_text("running shoes")
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_encode_image_shape_and_norm(tmp_path: Path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (224, 224), color=(255, 0, 0)).save(image_path)

    encoder = ClipEncoder(device="cpu")
    vec = encoder.encode_image(str(image_path))
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
