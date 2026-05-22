"""
Prometheus metrics exporter for Recommendation System.

This module exposes recommendation metrics in Prometheus format for monitoring
and alerting via Prometheus + Grafana stack.

Usage:
    from api.prometheus_metrics import setup_prometheus_metrics
    
    # In FastAPI app initialization:
    setup_prometheus_metrics(app)
    
    # Access metrics at: http://localhost:8000/metrics

Environment Variables:
    - ENABLE_PROMETHEUS: Set to "true" to enable Prometheus metrics (default: false)
"""

import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Metrics
REQUEST_COUNT = None
REQUEST_LATENCY_SECONDS = None
RECOMMENDATION_COUNT = None
CACHE_HITS = None
CACHE_MISSES = None
RANKER_USAGE = None
PHASE_DISTRIBUTION = None
DIVERSITY_METRIC = None
COVERAGE_METRIC = None
NDCG_METRIC = None


def _init_prometheus_metrics():
    """Initialize all Prometheus metrics."""
    global REQUEST_COUNT, REQUEST_LATENCY_SECONDS, RECOMMENDATION_COUNT
    global CACHE_HITS, CACHE_MISSES, RANKER_USAGE, PHASE_DISTRIBUTION
    global DIVERSITY_METRIC, COVERAGE_METRIC, NDCG_METRIC

    if not PROMETHEUS_AVAILABLE:
        return

    # Avoid duplicate registration in the global CollectorRegistry
    if REQUEST_COUNT is not None:
        return

    REQUEST_COUNT = Counter(
        "snaprec_requests_total",
        "Total number of recommendation requests",
        ["endpoint", "method", "status"],
    )

    REQUEST_LATENCY_SECONDS = Histogram(
        "snaprec_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
    )

    RECOMMENDATION_COUNT = Counter(
        "snaprec_recommendations_total",
        "Total number of recommendations generated",
        ["endpoint", "phase"],
    )

    CACHE_HITS = Counter(
        "snaprec_cache_hits_total",
        "Total number of cache hits",
        ["cache_type"],
    )

    CACHE_MISSES = Counter(
        "snaprec_cache_misses_total",
        "Total number of cache misses",
        ["cache_type"],
    )

    RANKER_USAGE = Counter(
        "snaprec_ranker_usage_total",
        "Total times ranker was used for reranking",
    )

    PHASE_DISTRIBUTION = Counter(
        "snaprec_phase_distribution_total",
        "Distribution of requests by phase mode",
        ["phase"],
    )

    DIVERSITY_METRIC = Summary(
        "snaprec_diversity",
        "Recommendation diversity metric (0-1)",
    )

    COVERAGE_METRIC = Summary(
        "snaprec_coverage",
        "Catalog coverage metric (0-1)",
    )

    NDCG_METRIC = Summary(
        "snaprec_ndcg",
        "NDCG (Normalized Discounted Cumulative Gain) metric",
    )


def setup_prometheus_metrics(app) -> None:
    """
    Set up Prometheus metrics endpoint for FastAPI application.

    Args:
        app: FastAPI application instance
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not installed, metrics disabled")
        return

    _init_prometheus_metrics()

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        """Return Prometheus metrics in text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        return "Prometheus metrics not available"

    logger.info("Prometheus metrics endpoint available at /metrics")


@contextmanager
def track_request_latency(endpoint: str):
    """
    Context manager to track request latency.

    Usage:
        with track_request_latency("recommend_text"):
            # ... handle request
    """
    if not PROMETHEUS_AVAILABLE or REQUEST_LATENCY_SECONDS is None:
        yield
        return

    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(duration)


def record_recommendation(
    endpoint: str,
    phase: str,
    cache_hit: bool,
    ranker_used: bool,
    diversity: Optional[float] = None,
    coverage: Optional[float] = None,
    ndcg: Optional[float] = None,
) -> None:
    """
    Record a recommendation event with associated metrics.

    Args:
        endpoint: Recommendation endpoint (e.g., "recommend_text")
        phase: Phase mode ("phase1" or "phase2")
        cache_hit: Whether result was cache hit
        ranker_used: Whether ranker was used
        diversity: Recommendation diversity score (0-1), optional
        coverage: Catalog coverage score (0-1), optional
        ndcg: NDCG score, optional
    """
    if not PROMETHEUS_AVAILABLE:
        return

    RECOMMENDATION_COUNT.labels(endpoint=endpoint, phase=phase).inc()
    PHASE_DISTRIBUTION.labels(phase=phase).inc()

    if cache_hit:
        CACHE_HITS.labels(cache_type="retrieval").inc()
    else:
        CACHE_MISSES.labels(cache_type="retrieval").inc()

    if ranker_used:
        RANKER_USAGE.inc()

    if diversity is not None:
        DIVERSITY_METRIC.observe(diversity)
    if coverage is not None:
        COVERAGE_METRIC.observe(coverage)
    if ndcg is not None:
        NDCG_METRIC.observe(ndcg)


def record_request(endpoint: str, method: str, status: int) -> None:
    """
    Record HTTP request metrics.

    Args:
        endpoint: Endpoint name
        method: HTTP method
        status: HTTP status code
    """
    if not PROMETHEUS_AVAILABLE or REQUEST_COUNT is None:
        return

    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
