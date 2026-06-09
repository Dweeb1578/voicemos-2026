from src.train import should_stop_early


def test_disabled_when_patience_zero():
    # patience=0 means never early-stop, no matter how long we've stalled
    assert should_stop_early(0, 0) is False
    assert should_stop_early(100, 0) is False


def test_fires_at_patience():
    assert should_stop_early(3, 3) is True
    assert should_stop_early(4, 3) is True


def test_not_yet_below_patience():
    assert should_stop_early(0, 3) is False
    assert should_stop_early(2, 3) is False
