"""
Redis-based distributed caching layer for recommendations.

This module provides a Redis cache backend to replace or complement the in-memory QueryCache
for distributed deployments where multiple API instances need to share cached results.

Usage:
    # Initialize Redis cache
    from retrieval.redis_cache import RedisQueryCache
    import redis
    
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_cache = RedisQueryCache(redis_client, embedding_ttl=86400, retrieval_ttl=3600)
    
    # Use like the in-memory cache
    user_vec = encode_text("test query")
    cached = redis_cache.get_retrieval(user_id="u123", query_vec=user_vec, top_k=10)
    if cached:
        results = cached
    else:
        results = expensive_retrieval()
        redis_cache.set_retrieval("u123", user_vec, top_k=10, results=results)

Environment Variables:
    - REDIS_HOST: Redis server hostname (default: localhost)
    - REDIS_PORT: Redis server port (default: 6379)
    - REDIS_DB: Redis database number (default: 0)
    - REDIS_PASSWORD: Redis password (optional)
    - CACHE_TYPE: "memory" or "redis" (default: memory)
"""

import hashlib
import json
import logging
from typing import Any, Optional

import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisQueryCache:
    """Distributed cache backend using Redis for query results and embeddings."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        embedding_ttl: int = 86400,  # 24 hours
        retrieval_ttl: int = 3600,  # 1 hour
        key_prefix: str = "snaprec:",
    ):
        """
        Initialize Redis cache.

        Args:
            redis_client: redis.Redis instance (if None, creates default connection)
            embedding_ttl: TTL in seconds for embedding cache (default: 24h)
            retrieval_ttl: TTL in seconds for retrieval cache (default: 1h)
            key_prefix: Prefix for all Redis keys (default: "snaprec:")
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "redis package not installed. Install with: pip install redis"
            )

        self.redis_client = redis_client or redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.embedding_ttl = embedding_ttl
        self.retrieval_ttl = retrieval_ttl
        self.key_prefix = key_prefix
        self._stats = {"hits": 0, "misses": 0}

        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _compute_cache_key(
        self, user_id: str, query_vec: np.ndarray, top_k: int, cache_type: str
    ) -> str:
        """Compute cache key from user_id, query vector, and top_k."""
        vec_hash = hashlib.sha256(query_vec.tobytes()).hexdigest()[:16]
        return f"{self.key_prefix}{cache_type}:{user_id}:{vec_hash}:{top_k}"

    def get_embedding(self, user_id: str, query_vec: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding.

        Args:
            user_id: User identifier
            query_vec: Query vector (used to compute cache key)

        Returns:
            Cached embedding as numpy array, or None if not found/expired
        """
        key = self._compute_cache_key(user_id, query_vec, top_k=0, cache_type="emb")
        try:
            cached = self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                self._stats["hits"] += 1
                return np.frombuffer(
                    bytes.fromhex(data["vec"]), dtype=np.float32
                ).copy()
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"Redis embedding cache miss: {e}")
            self._stats["misses"] += 1
            return None

    def set_embedding(
        self, user_id: str, query_vec: np.ndarray, embedding: np.ndarray
    ) -> None:
        """
        Store embedding in cache.

        Args:
            user_id: User identifier
            query_vec: Query vector (used to compute cache key)
            embedding: Embedding to cache
        """
        key = self._compute_cache_key(user_id, query_vec, top_k=0, cache_type="emb")
        try:
            data = {"vec": embedding.astype(np.float32).tobytes().hex()}
            self.redis_client.setex(key, self.embedding_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to set embedding in Redis: {e}")

    def get_retrieval(
        self, user_id: str, query_vec: np.ndarray, top_k: int
    ) -> Optional[list[tuple[str, float]]]:
        """
        Retrieve cached retrieval results.

        Args:
            user_id: User identifier
            query_vec: Query vector (used to compute cache key)
            top_k: Number of results

        Returns:
            List of (product_id, score) tuples, or None if not found/expired
        """
        key = self._compute_cache_key(user_id, query_vec, top_k, cache_type="ret")
        try:
            cached = self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                self._stats["hits"] += 1
                return [(p, float(s)) for p, s in data["results"]]
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"Redis retrieval cache miss: {e}")
            self._stats["misses"] += 1
            return None

    def set_retrieval(
        self,
        user_id: str,
        query_vec: np.ndarray,
        top_k: int,
        results: list[tuple[str, float]],
    ) -> None:
        """
        Store retrieval results in cache.

        Args:
            user_id: User identifier
            query_vec: Query vector
            top_k: Number of results
            results: List of (product_id, score) tuples
        """
        key = self._compute_cache_key(user_id, query_vec, top_k, cache_type="ret")
        try:
            data = {"results": results}
            self.redis_client.setex(key, self.retrieval_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to set retrieval in Redis: {e}")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total if total > 0 else 0
        )
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "total_requests": total,
        }

    def clear(self) -> int:
        """Clear all cache entries with current key prefix."""
        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor, match=pattern)
                if keys:
                    deleted += self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            logger.info(f"Cleared {deleted} cache entries from Redis")
            return deleted
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return 0

    def cleanup_expired(self) -> int:
        """Redis automatically handles expiration; this is a no-op for compatibility."""
        logger.debug("Redis automatically handles TTL expiration")
        return 0


def create_cache(cache_type: str = "memory", **kwargs) -> Any:
    """
    Factory function to create appropriate cache backend.

    Args:
        cache_type: "memory" for in-memory cache, "redis" for Redis backend
        **kwargs: Additional arguments passed to cache constructor

    Returns:
        Cache instance (QueryCache or RedisQueryCache)
    """
    if cache_type == "redis":
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, falling back to in-memory cache")
            from retrieval.cache import QueryCache
            return QueryCache(**kwargs)
        return RedisQueryCache(**kwargs)
    else:
        from retrieval.cache import QueryCache
        return QueryCache(**kwargs)
