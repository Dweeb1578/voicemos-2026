"""Tests for the frozen SOMOS prospective analysis semantics."""

import numpy as np
import pytest

from scripts.somos_prospective_analysis import (
    PREDICTORS,
    RUNNER_OUTPUTS,
    TrainECDF,
    _choose_alpha,
    cluster_bootstrap_difference,
)


def test_frozen_bank_has_ten_runners_and_twenty_seven_outputs():
    assert len(RUNNER_OUTPUTS) == 10
    assert len(PREDICTORS) == 27
    assert len(set(PREDICTORS)) == 27


def test_train_ecdf_is_right_continuous_and_never_uses_test_distribution():
    train = np.array([[1.0], [2.0], [2.0], [4.0]])
    ecdf = TrainECDF().fit(train)
    observed = ecdf.transform(np.array([[0.0], [1.0], [2.0], [3.0], [8.0]]))
    assert observed[:, 0].tolist() == [0.0, 0.25, 0.75, 0.75, 1.0]
    assert ecdf.transform(np.array([[2.0]]))[0, 0] == 0.75


class _ConstantModel:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0]


def test_alpha_tie_chooses_stronger_regularization():
    X = np.arange(6, dtype=float)[:, None]
    alpha, _, score = _choose_alpha(
        (0.1, 1.0, 10.0), _ConstantModel, X, X[:, 0], X, X[:, 0],
    )
    assert score == pytest.approx(1.0)
    assert alpha == 10.0


def test_cluster_bootstrap_is_seed_reproducible_and_paired():
    groups = np.repeat(np.arange(6), 3)
    y = np.arange(len(groups), dtype=float)
    raw = y + np.tile([0.0, 0.1, -0.1], 6)
    equal = -y
    first = cluster_bootstrap_difference(y, raw, equal, groups, draws=50, seed=17)
    second = cluster_bootstrap_difference(y, raw, equal, groups, draws=50, seed=17)
    assert first == second
    assert first["raw_minus_equal"]["point"] == pytest.approx(2.0)
    assert first["raw_minus_equal"]["percentile_95_interval"][0] > 1.5
