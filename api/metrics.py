"""Recommendation metrics collection and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path
from datetime import datetime

import numpy as np


@dataclass
class RecommendationMetrics:
    """Container for recommendation quality metrics."""
    
    timestamp: str = field(default_factory=lambda: datetime.now(datetime.UTC).isoformat() if hasattr(datetime, 'UTC') else datetime.utcnow().isoformat())
    user_id: str = ""
    top_k: int = 10
    phase_mode: str = "phase1"
    
    # Diversity metrics
    average_pairwise_distance: float = 0.0  # 0 =identical, 1=diverse
    coverage: float = 0.0  # % of catalog covered
    
    # Popularity metrics
    average_item_popularity: float = 0.0  # Based on interaction count
    diversity_score: float = 0.0  # Lower=popular bias, Higher=niche items
    
    # Ranking metrics
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain (0-1)
    average_score: float = 0.0  # Average recommendation score
    
    # System metrics
    latency_ms: float = 0.0
    cache_hit: bool = False
    ranker_used: bool = False


class MetricsCollector:
    """Collect and persist recommendation metrics."""
    
    def __init__(self, store_path: Optional[Path] = None):
        self.metrics_history: list[RecommendationMetrics] = []
        self.store_path = store_path
        
    def record_metrics(self, metrics: RecommendationMetrics) -> None:
        """Record a recommendation metrics snapshot."""
        self.metrics_history.append(metrics)
        
        if self.store_path:
            self._persist_metrics(metrics)
    
    def _persist_metrics(self, metrics: RecommendationMetrics) -> None:
        """Persist metrics to disk."""
        if not self.store_path:
            return
        
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL file
        with open(self.store_path, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def get_summary_stats(self, last_n: int = 100) -> dict:
        """Get aggregate statistics from recent recommendations."""
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-last_n:]
        
        ndcgs = [m.ndcg for m in recent if m.ndcg > 0]
        pairwise_distances = [m.average_pairwise_distance for m in recent if m.average_pairwise_distance > 0]
        latencies = [m.latency_ms for m in recent]
        
        return {
            "total_recommendations": len(recent),
            "average_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "average_diversity": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
            "cache_hit_rate": float(np.mean([m.cache_hit for m in recent])),
            "phase1_vs_phase2": {
                "phase1": len([m for m in recent if m.phase_mode == "phase1"]),
                "phase2": len([m for m in recent if m.phase_mode == "phase2"]),
            },
            "ranker_usage_rate": float(np.mean([m.ranker_used for m in recent])),
        }
    
    def compute_diversity(self, embeddings: np.ndarray) -> float:
        """
        Compute average pairwise cosine distance for a set of embeddings.
        Returns 0 = identical, 1 = completely different.
        """
        if len(embeddings) < 2:
            return 0.0
        
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = embeddings @ embeddings.T
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
        pairwise_sims = similarities[mask]
        
        if len(pairwise_sims) == 0:
            return 0.0
        
        # Convert similarity to distance (0 sim = 1 distance)
        avg_distance = float(np.mean(1 - pairwise_sims))
        return np.clip(avg_distance, 0.0, 1.0)
    
    def compute_coverage(self, product_ids: list[str], total_catalog_size: int) -> float:
        """Compute catalog coverage: % of unique items in recommendations."""
        unique_items = len(set(product_ids))
        return float(unique_items / total_catalog_size) if total_catalog_size > 0 else 0.0
    
    def compute_ndcg(self, scores: list[float], ideal_scores: Optional[list[float]] = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.
        scores: actual relevance scores
        ideal_scores: ideal ordering (default: sorted descending)
        """
        if not scores:
            return 0.0
        
        scores = np.array(scores)
        positions = np.arange(1, len(scores) + 1)
        dcg = np.sum(scores / np.log2(positions + 1))
        
        if ideal_scores is None:
            ideal_scores = np.sort(scores)[::-1]
        else:
            ideal_scores = np.array(ideal_scores)
        
        ideal_positions = np.arange(1, len(ideal_scores) + 1)
        idcg = np.sum(ideal_scores / np.log2(ideal_positions + 1))
        
        if idcg == 0:
            return 0.0
        
        return float(dcg / idcg)


# Global metrics collector
_global_collector = MetricsCollector()


def record_recommendation_metrics(metrics: RecommendationMetrics) -> None:
    """Record metrics to global collector."""
    _global_collector.record_metrics(metrics)


def get_metrics_summary(last_n: int = 100) -> dict:
    """Get summary statistics from global collector."""
    return _global_collector.get_summary_stats(last_n)


def compute_diversity(embeddings: np.ndarray) -> float:
    """Compute diversity for a set of embeddings."""
    return _global_collector.compute_diversity(embeddings)


def compute_coverage(product_ids: list[str], total_catalog_size: int) -> float:
    """Compute catalog coverage."""
    return _global_collector.compute_coverage(product_ids, total_catalog_size)


def compute_ndcg(scores: list[float], ideal_scores: Optional[list[float]] = None) -> float:
    """Compute NDCG score."""
    return _global_collector.compute_ndcg(scores, ideal_scores)
