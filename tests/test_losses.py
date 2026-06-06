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


def test_acr_rank_zero_when_correctly_ordered():
    loss_fn = MOSLoss(ccr_lambda=0.0, acr_rank_alpha=1.0)
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])  # correct order + zero MSE
    src = torch.tensor([0, 0, 0])
    loss = loss_fn(pred, torch.zeros(3), target, torch.full((3,), float("nan")),
                   source_ids=src)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_acr_rank_penalizes_wrong_order():
    loss_fn = MOSLoss(ccr_lambda=0.0, acr_rank_alpha=1.0)
    pred = torch.tensor([2.0, 1.0])     # reversed vs target
    target = torch.tensor([1.0, 2.0])
    same = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")),
                   source_ids=torch.tensor([0, 0]))
    diff = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")),
                   source_ids=torch.tensor([0, 1]))
    # same-source: MSE(=1) + rank(=1); diff-source: only MSE(=1), pair masked out
    assert same.item() == pytest.approx(2.0, abs=1e-4)
    assert diff.item() == pytest.approx(1.0, abs=1e-4)


def test_acr_rank_alpha_zero_is_pure_mse():
    loss_fn = MOSLoss(ccr_lambda=0.0, acr_rank_alpha=0.0)
    pred = torch.tensor([2.0, 1.0])
    target = torch.tensor([1.0, 2.0])
    loss = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")))
    assert loss.item() == pytest.approx(1.0, abs=1e-5)
