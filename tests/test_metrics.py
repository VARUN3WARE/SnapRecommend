"""Tests for recommendation metrics collection."""

from __future__ import annotations

import numpy as np
import pytest

from api.metrics import (
    RecommendationMetrics,
    MetricsCollector,
    compute_diversity,
    compute_coverage,
    compute_ndcg,
)


def test_recommendation_metrics_creation():
    """Test creating and storing metrics."""
    metrics = RecommendationMetrics(
        user_id="u00001",
        top_k=10,
        phase_mode="phase1",
        average_pairwise_distance=0.75,
        coverage=0.5,
        average_score=0.85,
    )
    
    assert metrics.user_id == "u00001"
    assert metrics.top_k == 10
    assert metrics.phase_mode == "phase1"
    assert metrics.average_pairwise_distance == 0.75


def test_metrics_collector_records():
    """Test metrics collector records metrics."""
    collector = MetricsCollector()
    
    metrics1 = RecommendationMetrics(user_id="u00001", ndcg=0.8, latency_ms=50)
    metrics2 = RecommendationMetrics(user_id="u00002", ndcg=0.7, latency_ms=45)
    
    collector.record_metrics(metrics1)
    collector.record_metrics(metrics2)
    
    assert len(collector.metrics_history) == 2
    assert collector.metrics_history[0].user_id == "u00001"
    assert collector.metrics_history[1].user_id == "u00002"


def test_compute_diversity_identical():
    """Test diversity computation for identical embeddings."""
    embeddings = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    diversity = compute_diversity(embeddings)
    
    # Identical embeddings should have 0 diversity
    assert diversity < 0.1


def test_compute_diversity_orthogonal():
    """Test diversity computation for orthogonal embeddings."""
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    diversity = compute_diversity(embeddings)
    
    # Orthogonal embeddings should have high diversity
    assert diversity > 0.8


def test_compute_coverage():
    """Test coverage computation."""
    product_ids = ["p1", "p2", "p3", "p2", "p1"]  # 3 unique out of 5
    catalog_size = 100
    
    coverage = compute_coverage(product_ids, catalog_size)
    assert coverage == 0.03  # 3/100


def test_compute_ndcg_perfect_ranking():
    """Test NDCG for perfect ranking."""
    scores = [1.0, 0.9, 0.8, 0.7, 0.6]
    ndcg = compute_ndcg(scores)
    
    # Perfect ranking should have NDCG=1.0
    assert abs(ndcg - 1.0) < 0.01


def test_compute_ndcg_reverse_ranking():
    """Test NDCG for reversed ranking."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    ndcg = compute_ndcg(scores)
    
    # Reversed ranking (worst case) should have lower NDCG than perfect ranking
    perfect_ndcg = compute_ndcg([0.5, 0.4, 0.3, 0.2, 0.1])
    assert ndcg < perfect_ndcg


def test_metrics_collector_summary_stats():
    """Test summary statistics computation."""
    collector = MetricsCollector()
    
    # Add some metrics
    for i in range(10):
        metrics = RecommendationMetrics(
            user_id=f"u{i:05d}",
            ndcg=0.5 + i * 0.05,
            latency_ms=40 + i * 2,
            cache_hit=i % 2 == 0,
            phase_mode="phase1" if i < 5 else "phase2",
        )
        collector.record_metrics(metrics)
    
    stats = collector.get_summary_stats(last_n=10)
    
    assert stats["total_recommendations"] == 10
    assert stats["average_ndcg"] > 0
    assert stats["median_latency_ms"] > 0
    assert 0 <= stats["cache_hit_rate"] <= 1
    assert stats["phase1_vs_phase2"]["phase1"] == 5
    assert stats["phase1_vs_phase2"]["phase2"] == 5


def test_metrics_summary_with_empty_history():
    """Test summary stats with no metrics."""
    collector = MetricsCollector()
    stats = collector.get_summary_stats()
    
    assert stats == {}


def test_compute_diversity_single_embedding():
    """Test diversity with single embedding."""
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    diversity = compute_diversity(embeddings)
    
    # Single embedding should have 0 diversity
    assert diversity == 0.0


def test_compute_ndcg_with_ideal_scores():
    """Test NDCG with explicit ideal scores."""
    scores = [0.5, 0.6, 0.7]
    ideal = [1.0, 0.9, 0.8]
    
    ndcg = compute_ndcg(scores, ideal_scores=ideal)
    assert 0 <= ndcg <= 1
