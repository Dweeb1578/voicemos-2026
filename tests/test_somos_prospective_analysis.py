"""Tests for the frozen SOMOS prospective analysis semantics."""

import gzip
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.somos_prospective_analysis import (
    ACQUISITION_SEEDS,
    BUDGETS,
    EXPECTED_SPLITS,
    PREDICTORS,
    RUNNER_OUTPUTS,
    TrainECDF,
    _choose_alpha,
    cluster_bootstrap_difference,
    run_analysis,
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


BASELINE_METHODS = {
    "best_single", "equal_ranks", "raw_ridge", "rank_ridge", "rank_nnls", "sparse_rank_lasso",
}
SYNTHETIC_SEED = 20260830
N_SYSTEMS = 10
# train must have >= max(BUDGETS) = 2500 rows or run_analysis raises; valid/test
# just need enough sentences and systems for grouping and bootstrap clustering
# to be non-degenerate.
SPLIT_SENTENCE_COUNTS = dict(zip(EXPECTED_SPLITS, (250, 15, 15)))
SPLIT_ROW_COUNTS = {
    split: N_SYSTEMS * count for split, count in SPLIT_SENTENCE_COUNTS.items()
}


def _build_synthetic_frames(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a label frame and a wide predictor frame sharing one latent-quality signal.

    Every row is one (sentence, system) utterance, matching SOMOS's structure.
    MOS and all 27 predictors are linear-plus-noise functions of a shared
    latent score, so predictors correlate with MOS (and with each other)
    without being identical or degenerate, keeping the ridge/lasso/NNLS fits
    non-trivial. sample_id is unique across all three splits combined.
    """
    system_effect = rng.normal(0.0, 0.6, size=N_SYSTEMS)
    records = []
    sentence_offset = 0
    for split, n_sentences in SPLIT_SENTENCE_COUNTS.items():
        sentence_effect = rng.normal(0.0, 0.3, size=n_sentences)
        for local_idx in range(n_sentences):
            sentence_idx = sentence_offset + local_idx
            for system_idx in range(N_SYSTEMS):
                latent = (
                    3.0
                    + system_effect[system_idx]
                    + sentence_effect[local_idx]
                    + rng.normal(0.0, 0.25)
                )
                records.append((
                    f"{split}_sent{sentence_idx:04d}_sys{system_idx:02d}",
                    f"sent{sentence_idx:04d}",
                    f"sys{system_idx:02d}",
                    split,
                    float(np.clip(latent, 1.0, 5.0)),
                    latent,
                ))
        sentence_offset += n_sentences

    frame = pd.DataFrame(records, columns=[
        "sample_id", "source_group", "system_id", "split", "mos", "latent",
    ])
    latent = frame["latent"].to_numpy()
    n = len(frame)
    predictor_columns = {"sample_id": frame["sample_id"].to_numpy()}
    for name in PREDICTORS:
        coeff = rng.uniform(0.6, 1.4)
        intercept = rng.uniform(-0.5, 0.5)
        noise_scale = rng.uniform(0.05, 0.35)
        predictor_columns[name] = (
            coeff * latent + intercept + rng.normal(0.0, noise_scale, size=n)
        )
    predictors_frame = pd.DataFrame(predictor_columns)
    labels_frame = frame.drop(columns=["latent"])
    return labels_frame, predictors_frame


@pytest.fixture
def work_dir():
    """Use the repository workspace, whose ACLs are available in this task."""
    path = Path(__file__).resolve().parents[1] / f".somos-prospective-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_run_analysis_end_to_end_on_synthetic_corpus(work_dir):
    """Run the real `run_analysis` entry point on a small but schema-exact corpus.

    This is the missing end-to-end test: every predeclared output the frozen,
    once-only production run depends on -- the six baseline methods, the
    primary cluster-bootstrap contrast, and the on-disk artifacts -- gets
    exercised here so a broken run fails in seconds, not after the real 30
    hours of GPU time have already been spent.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    labels_frame, predictors_frame = _build_synthetic_frames(rng)

    labels_path = work_dir / "labels.csv"
    labels_frame.to_csv(labels_path, index=False)

    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    for runner, outputs in RUNNER_OUTPUTS.items():
        shard = predictors_frame[["sample_id", *outputs]]
        shard.to_csv(shard_dir / f"{runner}.csv", index=False)

    out_dir = work_dir / "out"
    report = run_analysis(labels_path, shard_dir, out_dir, bootstrap_draws=50)

    # Declared row counts match what we actually fed in: every shard covers
    # every sample_id, so complete-case survival should be exactly 100%.
    assert report["rows"] == SPLIT_ROW_COUNTS
    assert report["artifacts"]["complete_case_rows"] == sum(SPLIT_ROW_COUNTS.values())
    assert report["artifacts"]["complete_case_survival"] == pytest.approx(
        {split: 1.0 for split in EXPECTED_SPLITS}
    )
    assert len(report["artifacts"]["shards"]) == len(RUNNER_OUTPUTS)

    # All six baseline methods appear, for the full fit and every budget run.
    assert set(report["full"]["metrics"]) == BASELINE_METHODS
    assert len(report["budget_runs"]) == len(ACQUISITION_SEEDS) * len(BUDGETS)
    for run in report["budget_runs"]:
        assert set(run["metrics"]) == BASELINE_METHODS
    assert set(report["budget_summary"]) == {str(budget) for budget in BUDGETS}

    # Metrics are finite and correlation-shaped metrics stay in [-1, 1]. A
    # silently degenerate fit (e.g. constant predictions) would surface here
    # as a non-finite SRCC, because the script's own `srcc()` helper maps a
    # NaN correlation to -inf instead of letting it pass quietly.
    for method, values in report["full"]["metrics"].items():
        for key in ("utterance_srcc", "utterance_pearson", "utterance_mae", "system_srcc"):
            value = values[key]
            assert np.isfinite(value), f"{method}.{key} is not finite: {value}"
        assert -1.0 <= values["utterance_srcc"] <= 1.0
        assert -1.0 <= values["utterance_pearson"] <= 1.0
        assert -1.0 <= values["system_srcc"] <= 1.0

    # The primary bootstrap contrast (the paper's headline number) is present
    # and well formed: finite point estimates, correctly ordered intervals.
    primary = report["primary_bootstrap"]
    assert primary["draws"] == 50
    assert primary["cluster"] == "source_group"
    for key in ("raw_ridge", "equal_ranks"):
        block = primary[key]
        assert np.isfinite(block["point"])
        assert -1.0 <= block["point"] <= 1.0
        lo, hi = block["percentile_95_interval"]
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi
    diff = primary["raw_minus_equal"]
    assert np.isfinite(diff["point"])
    assert -2.0 <= diff["point"] <= 2.0
    lo, hi = diff["percentile_95_interval"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi

    # The expected output files exist, and the on-disk report round-trips.
    predictions_path = out_dir / "somos_v2_test_predictions.csv"
    orders_path = out_dir / "somos_v2_acquisition_orders.json.gz"
    report_path = out_dir / "somos_v2_prospective_results.json"
    assert predictions_path.exists()
    assert orders_path.exists()
    assert report_path.exists()

    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    assert len(predictions) == SPLIT_ROW_COUNTS["test"]
    assert BASELINE_METHODS.issubset(set(predictions.columns))

    with gzip.open(orders_path, "rt", encoding="utf-8") as handle:
        orders = json.load(handle)
    assert set(orders) == {str(seed) for seed in ACQUISITION_SEEDS}
    for sample_ids in orders.values():
        assert len(sample_ids) == SPLIT_ROW_COUNTS["train"]

    on_disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk_report["rows"] == report["rows"]
    assert on_disk_report["primary_bootstrap"] == report["primary_bootstrap"]
