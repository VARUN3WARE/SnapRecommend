"""
Tests for Redis cache backend.

These tests use a mock Redis client to avoid requiring a live Redis server during testing.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip all tests if redis is not available
redis = pytest.importorskip("redis", minversion=None)

from retrieval.redis_cache import RedisQueryCache


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    return mock_client


@pytest.fixture
def redis_cache(mock_redis_client):
    """Create a RedisQueryCache instance with mock Redis client."""
    return RedisQueryCache(redis_client=mock_redis_client)


def test_redis_cache_get_embedding_hit(redis_cache, mock_redis_client):
    """Test retrieving a cached embedding."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Mock Redis return
    mock_redis_client.get.return_value = json.dumps(
        {"vec": embedding.tobytes().hex()}
    )

    result = redis_cache.get_embedding(user_id, query_vec)

    assert result is not None
    np.testing.assert_array_almost_equal(result, embedding)
    assert redis_cache.stats()["hits"] == 1


def test_redis_cache_get_embedding_miss(redis_cache, mock_redis_client):
    """Test cache miss on embedding retrieval."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Mock Redis miss
    mock_redis_client.get.return_value = None

    result = redis_cache.get_embedding(user_id, query_vec)

    assert result is None
    assert redis_cache.stats()["misses"] == 1


def test_redis_cache_set_embedding(redis_cache, mock_redis_client):
    """Test storing an embedding in cache."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    redis_cache.set_embedding(user_id, query_vec, embedding)

    # Verify setex was called with correct TTL
    mock_redis_client.setex.assert_called_once()
    call_args = mock_redis_client.setex.call_args
    assert call_args[0][1] == 86400  # embedding_ttl


def test_redis_cache_get_retrieval_hit(redis_cache, mock_redis_client):
    """Test retrieving cached retrieval results."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0], dtype=np.float32)
    top_k = 10
    results = [("p001", 0.95), ("p002", 0.87)]

    # Mock Redis return
    mock_redis_client.get.return_value = json.dumps({"results": results})

    cached = redis_cache.get_retrieval(user_id, query_vec, top_k)

    assert cached == results
    assert redis_cache.stats()["hits"] == 1


def test_redis_cache_get_retrieval_miss(redis_cache, mock_redis_client):
    """Test cache miss on retrieval."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0], dtype=np.float32)
    top_k = 10

    mock_redis_client.get.return_value = None

    cached = redis_cache.get_retrieval(user_id, query_vec, top_k)

    assert cached is None
    assert redis_cache.stats()["misses"] == 1


def test_redis_cache_set_retrieval(redis_cache, mock_redis_client):
    """Test storing retrieval results in cache."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0], dtype=np.float32)
    top_k = 10
    results = [("p001", 0.95), ("p002", 0.87)]

    redis_cache.set_retrieval(user_id, query_vec, top_k, results)

    # Verify setex was called with correct TTL
    mock_redis_client.setex.assert_called_once()
    call_args = mock_redis_client.setex.call_args
    assert call_args[0][1] == 3600  # retrieval_ttl


def test_redis_cache_stats(redis_cache, mock_redis_client):
    """Test cache statistics calculation."""
    mock_redis_client.get.return_value = None

    user_id = "u123"
    query_vec = np.array([1.0, 2.0], dtype=np.float32)

    # Generate some cache accesses
    redis_cache.get_retrieval(user_id, query_vec, 10)  # miss
    redis_cache.get_retrieval(user_id, query_vec, 10)  # miss
    redis_cache.get_retrieval(user_id, query_vec, 10)  # miss

    stats = redis_cache.stats()
    assert stats["misses"] == 3
    assert stats["hits"] == 0
    assert stats["total_requests"] == 3
    assert stats["hit_rate"] == 0.0


def test_redis_cache_clear(redis_cache, mock_redis_client):
    """Test clearing cache."""
    # Mock scan to return 2 batches of keys
    mock_redis_client.scan.side_effect = [
        (1, ["key1", "key2"]),  # cursor=1, keys
        (0, ["key3"]),  # cursor=0 (end), keys
    ]
    mock_redis_client.delete.side_effect = [2, 1]  # deletions per batch

    deleted = redis_cache.clear()

    assert deleted == 3
    assert mock_redis_client.delete.call_count == 2


def test_redis_cache_different_top_k_separate_keys(redis_cache, mock_redis_client):
    """Test that different top_k values use different cache keys."""
    user_id = "u123"
    query_vec = np.array([1.0, 2.0], dtype=np.float32)

    # Set up mock to return None (miss)
    mock_redis_client.get.return_value = None

    # Query with different top_k
    redis_cache.get_retrieval(user_id, query_vec, top_k=5)
    redis_cache.get_retrieval(user_id, query_vec, top_k=10)

    # Both should be cache misses (different keys)
    assert redis_cache.stats()["misses"] == 2


def test_redis_cache_factory_function():
    """Test cache factory function."""
    from retrieval.redis_cache import create_cache

    # Memory cache (should not require redis)
    with patch("retrieval.cache.QueryCache") as MockQueryCache:
        cache = create_cache(cache_type="memory")
        assert cache is not None

    # Redis cache without redis available (should fall back to memory)
    with patch("retrieval.redis_cache.REDIS_AVAILABLE", False):
        with patch("retrieval.cache.QueryCache") as MockQueryCache:
            cache = create_cache(cache_type="redis")
            # Should fall back to in-memory
            assert cache is not None
