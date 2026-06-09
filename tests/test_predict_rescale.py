import pytest

from scripts.predict_dev import rescale_to_range


def test_rescale_maps_min_and_max_to_endpoints():
    out = rescale_to_range([0.0, 1.0, 2.0], 1.0, 5.0)
    assert out[0] == pytest.approx(1.0)   # min -> lo
    assert out[-1] == pytest.approx(5.0)  # max -> hi
    assert out[1] == pytest.approx(3.0)   # midpoint -> mid


def test_rescale_is_rank_preserving():
    vals = [-2.0, 0.5, -1.0, 3.0, 0.0]
    out = rescale_to_range(vals, 1.0, 5.0)
    order_in = sorted(range(len(vals)), key=lambda i: vals[i])
    order_out = sorted(range(len(out)), key=lambda i: out[i])
    assert order_in == order_out  # SRCC unchanged under a monotonic map


def test_rescale_negative_target_range():
    out = rescale_to_range([-5.0, 5.0], -3.0, 3.0)
    assert out[0] == pytest.approx(-3.0)
    assert out[1] == pytest.approx(3.0)


def test_rescale_constant_input_no_div0():
    out = rescale_to_range([2.0, 2.0, 2.0], 1.0, 5.0)
    assert out == [1.0, 1.0, 1.0]  # max==min -> all lo, no division by zero
