import numpy as np

from retrieval.faiss_index import build_index, search


def test_build_and_search_index():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(100, 512)).astype(np.float32)

    index = build_index(embeddings, use_gpu=False)
    q = embeddings[0]
    distances, indices = search(index, q, k=5)

    assert distances.shape == (5,)
    assert indices.shape == (5,)
    assert int(indices[0]) == 0
