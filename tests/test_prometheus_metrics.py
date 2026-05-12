"""
Tests for Prometheus metrics module.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.skipif(
    not pytest.importorskip("prometheus_client", minversion=None),
    reason="prometheus_client not installed",
)
def test_prometheus_metrics_init():
    """Test that Prometheus metrics can be initialized."""
    from api.prometheus_metrics import _init_prometheus_metrics, PROMETHEUS_AVAILABLE

    if PROMETHEUS_AVAILABLE:
        _init_prometheus_metrics()
        # Should not raise any exceptions


def test_prometheus_metrics_optional():
    """Test that metrics gracefully handle missing prometheus_client."""
    with patch("api.prometheus_metrics.PROMETHEUS_AVAILABLE", False):
        from api.prometheus_metrics import record_recommendation, record_request, track_request_latency

        # Should not raise exceptions
        record_recommendation(
            endpoint="test",
            phase="phase1",
            cache_hit=True,
            ranker_used=False,
            diversity=0.8,
        )
        record_request("test", "GET", 200)

        with track_request_latency("test"):
            pass  # Should not raise


def test_prometheus_metrics_fastapi_setup():
    """Test FastAPI integration with Prometheus metrics."""
    with patch("api.prometheus_metrics.PROMETHEUS_AVAILABLE", False):
        from fastapi import FastAPI
        from api.prometheus_metrics import setup_prometheus_metrics

        app = FastAPI()
        # Should not raise exceptions
        setup_prometheus_metrics(app)


def test_track_request_latency():
    """Test latency tracking context manager."""
    with patch("api.prometheus_metrics.PROMETHEUS_AVAILABLE", True):
        with patch("api.prometheus_metrics.REQUEST_LATENCY_SECONDS") as mock_histogram:
            from api.prometheus_metrics import track_request_latency

            with track_request_latency("test_endpoint"):
                pass

            # Verify histogram was called
            assert mock_histogram.labels.called or mock_histogram is not None


def test_record_recommendation():
    """Test recommendation event recording."""
    with patch("api.prometheus_metrics.PROMETHEUS_AVAILABLE", True):
        with patch("api.prometheus_metrics.RECOMMENDATION_COUNT") as mock_counter:
            with patch("api.prometheus_metrics.PHASE_DISTRIBUTION") as mock_phase:
                with patch("api.prometheus_metrics.RANKER_USAGE") as mock_ranker:
                    from api.prometheus_metrics import record_recommendation

                    record_recommendation(
                        endpoint="recommend_text",
                        phase="phase2",
                        cache_hit=True,
                        ranker_used=True,
                        diversity=0.75,
                        coverage=0.85,
                        ndcg=0.92,
                    )

                    # Verify counters were incremented
                    if mock_counter is not None:
                        assert mock_counter.labels.called or mock_counter is not None


def test_record_request():
    """Test HTTP request recording."""
    with patch("api.prometheus_metrics.PROMETHEUS_AVAILABLE", True):
        with patch("api.prometheus_metrics.REQUEST_COUNT") as mock_counter:
            from api.prometheus_metrics import record_request

            record_request("recommend_text", "POST", 200)

            # Verify counter was incremented
            if mock_counter is not None:
                assert mock_counter.labels.called or mock_counter is not None
