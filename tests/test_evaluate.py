import math
import pytest

from src.evaluate import compute_metrics


def test_perfect_correlation():
    m = compute_metrics([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert m["srcc"] == pytest.approx(1.0)
    assert m["lcc"] == pytest.approx(1.0)
    assert m["mse"] == pytest.approx(0.0)


def test_negative_correlation():
    m = compute_metrics([5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert m["srcc"] == pytest.approx(-1.0)


def test_mse_value():
    m = compute_metrics([3.0, 3.0], [1.0, 1.0])
    assert m["mse"] == pytest.approx(4.0)


def test_single_sample_returns_nan():
    m = compute_metrics([3.0], [3.0])
    assert math.isnan(m["srcc"])
    assert math.isnan(m["lcc"])


def test_nan_pairs_dropped():
    m = compute_metrics([1.0, float("nan"), 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert not math.isnan(m["srcc"])
