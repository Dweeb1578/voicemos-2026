import torch
import pytest

from src.losses import MOSLoss


def test_acr_mse_perfect_prediction():
    loss_fn = MOSLoss(ccr_lambda=0.0)
    t = torch.tensor([3.0, 4.0, 2.5])
    loss = loss_fn(t, torch.zeros(3), t, torch.full((3,), float("nan")))
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_acr_mse_value():
    loss_fn = MOSLoss(ccr_lambda=0.0)
    pred = torch.tensor([3.0, 4.0])
    target = torch.tensor([1.0, 2.0])  # error of 2 each -> MSE = 4
    loss = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")))
    assert loss.item() == pytest.approx(4.0, abs=1e-5)


def test_all_nan_acr_returns_zero():
    loss_fn = MOSLoss(ccr_lambda=0.0)
    loss = loss_fn(
        torch.tensor([3.0, 4.0]), torch.zeros(2),
        torch.full((2,), float("nan")), torch.full((2,), float("nan")),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_ccr_ranking_loss_correct_order():
    loss_fn = MOSLoss(ccr_lambda=1.0)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = loss_fn(scores, scores, scores, scores)
    assert loss.item() < 0.1


def test_ccr_skipped_when_all_nan():
    loss_fn = MOSLoss(ccr_lambda=1.0)
    loss = loss_fn(
        torch.tensor([3.0]), torch.tensor([0.0]),
        torch.tensor([3.0]), torch.full((1,), float("nan")),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
