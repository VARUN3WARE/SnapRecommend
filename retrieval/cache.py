"""In-memory caching layer for embeddings and retrieval results."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

import numpy as np


class CacheEntry:
    """Single cache entry with TTL tracking."""

    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if entry has exceeded TTL."""
        return time.time() - self.created_at > self.ttl_seconds


class QueryCache:
    """Simple in-memory cache for embeddings and retrieval results.
    
    - Embedding cache: TTL 24h (86400s)
    - Retrieval cache: TTL 1h (3600s)
    """

    def __init__(self, embedding_ttl_seconds: int = 86400, retrieval_ttl_seconds: int = 3600):
        self.embedding_cache: dict[str, CacheEntry] = {}
        self.retrieval_cache: dict[str, CacheEntry] = {}
        self.embedding_ttl = embedding_ttl_seconds
        self.retrieval_ttl = retrieval_ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash_vector(self, vec: np.ndarray) -> str:
        """Generate deterministic hash for vector."""
        return hashlib.sha256(vec.tobytes()).hexdigest()

    def get_embedding(self, user_id: str, query_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available and not expired."""
        key = f"emb:{user_id}:{query_hash}"
        if key not in self.embedding_cache:
            self.misses += 1
            return None

        entry = self.embedding_cache[key]
        if entry.is_expired():
            del self.embedding_cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.value

    def set_embedding(self, user_id: str, query_hash: str, embedding: np.ndarray) -> None:
        """Cache embedding vector."""
        key = f"emb:{user_id}:{query_hash}"
        self.embedding_cache[key] = CacheEntry(embedding, self.embedding_ttl)

    def get_retrieval(self, user_id: str, query_hash: str, top_k: int) -> Optional[list[tuple[str, float]]]:
        """Retrieve cached retrieval results if available and not expired."""
        key = f"ret:{user_id}:{query_hash}:k{top_k}"
        if key not in self.retrieval_cache:
            self.misses += 1
            return None

        entry = self.retrieval_cache[key]
        if entry.is_expired():
            del self.retrieval_cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.value

    def set_retrieval(self, user_id: str, query_hash: str, top_k: int, results: list[tuple[str, float]]) -> None:
        """Cache retrieval results."""
        key = f"ret:{user_id}:{query_hash}:k{top_k}"
        self.retrieval_cache[key] = CacheEntry(results, self.retrieval_ttl)

    def clear(self) -> None:
        """Clear all cached entries."""
        self.embedding_cache.clear()
        self.retrieval_cache.clear()
        self.hits = 0
        self.misses = 0

    def cleanup_expired(self) -> tuple[int, int]:
        """Remove expired entries from both caches. Returns (removed_embeddings, removed_retrievals)."""
        emb_removed = sum(1 for entry in self.embedding_cache.values() if entry.is_expired())
        ret_removed = sum(1 for entry in self.retrieval_cache.values() if entry.is_expired())

        self.embedding_cache = {k: v for k, v in self.embedding_cache.items() if not v.is_expired()}
        self.retrieval_cache = {k: v for k, v in self.retrieval_cache.items() if not v.is_expired()}

        return emb_removed, ret_removed

    def stats(self) -> dict:
        """Return cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": float(hit_rate),
            "embedding_cache_size": len(self.embedding_cache),
            "retrieval_cache_size": len(self.retrieval_cache),
        }
