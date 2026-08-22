import time
import pytest
from backend.app.services.latency import LatencyTracker


def test_start_stop_elapsed():
    tracker = LatencyTracker()
    tracker.start("stage1")
    time.sleep(0.01)  # 10ms
    tracker.stop("stage1")
    elapsed = tracker.elapsed("stage1")
    # allow some tolerance
    assert 5.0 <= elapsed <= 50.0


def test_multiple_stages_and_total():
    tracker = LatencyTracker()
    tracker.start("a")
    time.sleep(0.005)
    tracker.stop("a")
    tracker.start("b")
    time.sleep(0.007)
    tracker.stop("b")
    assert 4.0 <= tracker.elapsed("a") <= 20.0
    assert 6.0 <= tracker.elapsed("b") <= 25.0
    total = tracker.total_elapsed()
    # total should be roughly sum of a and b
    assert total >= tracker.elapsed("a") + tracker.elapsed("b") - 5.0


def test_to_dict_contains_total():
    tracker = LatencyTracker()
    tracker.start("x")
    time.sleep(0.003)
    tracker.stop("x")
    d = tracker.to_dict()
    assert "x" in d
    assert "total_ms" in d
    # total should be >= x
    assert d["total_ms"] >= d["x"]

def test_invalid_stop_raises():
    tracker = LatencyTracker()
    with pytest.raises(RuntimeError):
        tracker.stop("missing")

def test_elapsed_before_stop_returns_live_time():
    tracker = LatencyTracker()
    tracker.start("live")
    time.sleep(0.004)
    elapsed1 = tracker.elapsed("live")
    time.sleep(0.004)
    elapsed2 = tracker.elapsed("live")
    assert elapsed2 > elapsed1
    tracker.stop("live")
    final = tracker.elapsed("live")
    assert final >= elapsed2
