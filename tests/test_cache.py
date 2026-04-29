"""Tests for the caching layer."""

from __future__ import annotations

import time

import numpy as np
import pytest

from retrieval.cache import QueryCache


def test_embedding_cache_stores_and_retrieves():
    cache = QueryCache()
    user_id = "u00001"
    query_hash = "abc123"
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Initially empty
    assert cache.get_embedding(user_id, query_hash) is None

    # Store embedding
    cache.set_embedding(user_id, query_hash, embedding)

    # Retrieve embedding
    retrieved = cache.get_embedding(user_id, query_hash)
    assert retrieved is not None
    np.testing.assert_array_equal(retrieved, embedding)
    assert cache.hits == 1


def test_retrieval_cache_stores_and_retrieves():
    cache = QueryCache()
    user_id = "u00001"
    query_hash = "xyz789"
    top_k = 10
    results = [("p000001", 0.95), ("p000002", 0.87)]

    # Initially empty
    assert cache.get_retrieval(user_id, query_hash, top_k) is None

    # Store retrieval results
    cache.set_retrieval(user_id, query_hash, top_k, results)

    # Retrieve results
    retrieved = cache.get_retrieval(user_id, query_hash, top_k)
    assert retrieved == results
    assert cache.hits == 1


def test_embedding_cache_expiration():
    # Create cache with 1-second TTL
    cache = QueryCache(embedding_ttl_seconds=1, retrieval_ttl_seconds=10)
    user_id = "u00001"
    query_hash = "abc123"
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    cache.set_embedding(user_id, query_hash, embedding)

    # Should be available immediately
    assert cache.get_embedding(user_id, query_hash) is not None
    assert cache.hits == 1

    # Wait for expiration
    time.sleep(1.1)

    # Should be expired now
    assert cache.get_embedding(user_id, query_hash) is None
    assert cache.misses == 1


def test_retrieval_cache_expiration():
    # Create cache with 1-second TTL
    cache = QueryCache(embedding_ttl_seconds=10, retrieval_ttl_seconds=1)
    user_id = "u00001"
    query_hash = "xyz789"
    top_k = 10
    results = [("p000001", 0.95), ("p000002", 0.87)]

    cache.set_retrieval(user_id, query_hash, top_k, results)

    # Should be available immediately
    assert cache.get_retrieval(user_id, query_hash, top_k) is not None
    assert cache.hits == 1

    # Wait for expiration
    time.sleep(1.1)

    # Should be expired now
    assert cache.get_retrieval(user_id, query_hash, top_k) is None
    assert cache.misses == 1


def test_cache_different_top_k_values():
    cache = QueryCache()
    user_id = "u00001"
    query_hash = "xyz789"
    results_k5 = [("p000001", 0.95), ("p000002", 0.87)]
    results_k10 = [("p000001", 0.95), ("p000002", 0.87), ("p000003", 0.75)]

    cache.set_retrieval(user_id, query_hash, 5, results_k5)
    cache.set_retrieval(user_id, query_hash, 10, results_k10)

    # Each k value should be cached separately
    assert cache.get_retrieval(user_id, query_hash, 5) == results_k5
    assert cache.get_retrieval(user_id, query_hash, 10) == results_k10
    assert cache.hits == 2


def test_cache_clear():
    cache = QueryCache()
    user_id = "u00001"
    query_hash = "abc123"
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    cache.set_embedding(user_id, query_hash, embedding)
    cache.set_retrieval(user_id, query_hash, 10, [("p000001", 0.95)])

    assert cache.embedding_cache
    assert cache.retrieval_cache

    cache.clear()

    assert len(cache.embedding_cache) == 0
    assert len(cache.retrieval_cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_cleanup_expired():
    cache = QueryCache(embedding_ttl_seconds=1, retrieval_ttl_seconds=1)
    user_id = "u00001"

    # Store some entries
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set_embedding(user_id, "hash1", embedding)
    cache.set_retrieval(user_id, "hash2", 10, [("p000001", 0.95)])

    assert len(cache.embedding_cache) == 1
    assert len(cache.retrieval_cache) == 1

    # Wait for expiration
    time.sleep(1.1)

    # Cleanup should remove expired entries
    emb_removed, ret_removed = cache.cleanup_expired()
    assert emb_removed == 1
    assert ret_removed == 1
    assert len(cache.embedding_cache) == 0
    assert len(cache.retrieval_cache) == 0


def test_cache_stats():
    cache = QueryCache()
    user_id = "u00001"
    query_hash = "abc123"
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    cache.set_embedding(user_id, query_hash, embedding)

    # One hit
    cache.get_embedding(user_id, query_hash)

    # One miss
    cache.get_embedding(user_id, "nonexistent")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["embedding_cache_size"] == 1
    assert stats["retrieval_cache_size"] == 0
