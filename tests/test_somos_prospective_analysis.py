"""Tests for the frozen SOMOS prospective analysis semantics."""

import gzip
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.somos_integrity as integrity
import scripts.somos_prospective_analysis as analysis_module
from scripts.somos_integrity import (
    CERTIFICATE_TARGET_ACCESS,
    FROZEN_PROTOCOL_SHA256,
    LABEL_TARGET_ACCESS,
    MERGE_TARGET_ACCESS,
    RUNNER_OUTPUTS as INTEGRITY_RUNNER_OUTPUTS,
    SOMOS_ARCHIVE_MD5,
    seal_completion_payload,
    sha256_file,
    strict_json_text,
    write_strict_json,
)

from scripts.somos_prospective_analysis import (
    ACQUISITION_SEEDS,
    BUDGETS,
    EXPECTED_SPLITS,
    PREDICTORS,
    RUNNER_OUTPUTS,
    SplitArrays,
    TrainECDF,
    _choose_alpha,
    cluster_bootstrap_difference,
    fit_methods,
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


def test_best_single_uses_raw_validation_output_not_stepwise_ecdf():
    n_predictors = len(PREDICTORS)
    train_x = np.linspace(0.0, 1.0, 20)[:, None] + np.arange(n_predictors)[None, :]
    valid_x = np.tile(np.array([0.4, 0.1, 0.3, 0.2])[:, None], (1, n_predictors))
    valid_x += np.arange(n_predictors)[None, :]
    valid_x[:, 0] = [10.0, 20.0, 30.0, 40.0]
    test_x = np.tile(np.array([2.0, 1.0])[:, None], (1, n_predictors))
    test_x[:, 0] = [41.0, 42.0]

    def split(prefix, X, y):
        count = len(y)
        return SplitArrays(
            sample_id=np.array([f"{prefix}{index:03d}_001.wav" for index in range(count)]),
            source_group=np.array([f"{prefix}{index:03d}" for index in range(count)]),
            system_id=np.array(["001"] * count),
            y=np.asarray(y, dtype=float),
            X=np.asarray(X, dtype=float),
        )

    train = split("train", train_x, np.linspace(1.0, 5.0, len(train_x)))
    valid = split("valid", valid_x, [1.0, 2.0, 3.0, 4.0])
    test = split("test", test_x, [2.0, 3.0])
    predictions, details = fit_methods(
        train, valid, test, np.arange(len(train.y)), seed=0,
    )
    assert details["best_single"]["predictor"] == PREDICTORS[0]
    assert predictions["best_single"].tolist() == [41.0, 42.0]
    assert "raw validation-output SRCC" in details["best_single"]["implementation_clarification"]


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
                    f"sent{sentence_idx:04d}_{system_idx:03d}.wav",
                    f"sent{sentence_idx:04d}",
                    f"{system_idx:03d}",
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


def _evidence(path: Path) -> dict:
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_integrity_artifacts(
        work_dir: Path, labels_path: Path, labels_frame: pd.DataFrame,
        shard_dir: Path,
) -> tuple[Path, Path, dict]:
    merge_artifacts = []
    for runner, outputs in RUNNER_OUTPUTS.items():
        csv_path = shard_dir / f"{runner}.csv"
        provenance_path = shard_dir / f"{runner}.merge.provenance.json"
        columns = ["sample_id", "source_group", "system_id", "split", *outputs]
        provenance = {
            "schema_version": 1,
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "runner": runner,
            "outputs": list(outputs),
            "rows": len(labels_frame),
            "merged_csv": {
                "path": csv_path.name,
                "sha256": sha256_file(csv_path),
                "bytes": csv_path.stat().st_size,
                "columns": columns,
            },
            "input_shards": [{
                "score_csv": {
                    "path": f"{runner}-part{index:02d}-of-04.csv",
                    "sha256": f"{index + 1:064x}",
                    "bytes": 100 + index,
                },
                "provenance": {
                    "path": f"{runner}-part{index:02d}-of-04.provenance.json",
                    "sha256": f"{index + 11:064x}",
                },
            } for index in range(4)],
            "target_access": MERGE_TARGET_ACCESS,
        }
        write_strict_json(provenance_path, provenance)
        merge_artifacts.append({
            "runner": runner,
            "outputs": list(outputs),
            "rows": len(labels_frame),
            "merged_csv": _evidence(csv_path),
            "merge_provenance": _evidence(provenance_path),
        })

    payload = {
        "complete": True,
        "completed_at_utc": "2026-08-30T00:00:00+00:00",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "post_release_exploratory": True,
        "expected_rows": len(labels_frame),
        "split_rows": SPLIT_ROW_COUNTS,
        "runner_bank": {
            runner: list(outputs) for runner, outputs in INTEGRITY_RUNNER_OUTPUTS.items()
        },
        "sample_id_sha256": analysis_module._sample_id_hash(labels_frame),
        "metadata_sha256": analysis_module._metadata_hash(labels_frame),
        "merge_artifacts": merge_artifacts,
        "target_access": CERTIFICATE_TARGET_ACCESS,
    }
    certificate_path = work_dir / "somos_completion_certificate.json"
    certificate = seal_completion_payload(payload)
    write_strict_json(certificate_path, certificate)

    download_path = work_dir / "somos_v2_labels_download.json"
    archive_path = work_dir / "somos_v2_labels_archive_inventory.json"
    extraction_path = work_dir / "somos_v2_labels_extract_inventory.json"
    write_strict_json(download_path, {"actual_md5": SOMOS_ARCHIVE_MD5})
    write_strict_json(archive_path, {"archive_md5": SOMOS_ARCHIVE_MD5})
    write_strict_json(extraction_path, {
        "archive_md5": SOMOS_ARCHIVE_MD5,
        "labels_only": True,
        "clean_prefix": "training_files/split1/clean",
        "label_file_count": 3,
    })
    label_provenance_path = work_dir / "somos_v2_labels.provenance.json"
    write_strict_json(label_provenance_path, {
        "schema_version": 1,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "post_release_exploratory": True,
        "target_access": LABEL_TARGET_ACCESS,
        "completion_certificate": {
            "file": certificate_path.name,
            "sha256": sha256_file(certificate_path),
            "payload_sha256": certificate["seal"]["payload_sha256"],
        },
        "label_manifest": {
            "file": labels_path.name,
            "sha256": sha256_file(labels_path),
            "bytes": labels_path.stat().st_size,
            "columns": ["sample_id", "source_group", "system_id", "split", "mos"],
            "rows": len(labels_frame),
            "split_rows": SPLIT_ROW_COUNTS,
        },
        "evidence": {
            "download": _evidence(download_path),
            "archive_inventory": _evidence(archive_path),
            "extraction_inventory": _evidence(extraction_path),
        },
    })
    return certificate_path, label_provenance_path, payload


@pytest.fixture
def work_dir():
    """Use the repository workspace, whose ACLs are available in this task."""
    path = Path(__file__).resolve().parents[1] / f".somos-prospective-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_run_analysis_end_to_end_on_synthetic_corpus(work_dir, monkeypatch):
    """Run the real `run_analysis` entry point on a small but schema-exact corpus.

    This is the missing end-to-end test: every predeclared output the frozen,
    once-only production run depends on -- the six baseline methods, the
    primary cluster-bootstrap contrast, and the on-disk artifacts -- gets
    exercised here so a broken run fails in seconds, not after the real 30
    hours of GPU time have already been spent.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    labels_frame, predictors_frame = _build_synthetic_frames(rng)
    expected_rows = sum(SPLIT_ROW_COUNTS.values())
    monkeypatch.setattr(integrity, "EXPECTED_ROWS", expected_rows)
    monkeypatch.setattr(integrity, "EXPECTED_SPLIT_ROWS", SPLIT_ROW_COUNTS)
    monkeypatch.setattr(analysis_module, "EXPECTED_ROWS", expected_rows)
    monkeypatch.setattr(analysis_module, "EXPECTED_SPLIT_ROWS", SPLIT_ROW_COUNTS)

    labels_path = work_dir / "labels.csv"
    labels_frame.to_csv(labels_path, index=False)

    shard_dir = work_dir / "shards"
    shard_dir.mkdir()
    # Shards are written in the canonical merged schema that
    # scripts/somos_merge_shards.py actually produces, descriptive columns
    # included, not the narrower shape the analysis strictly needs.
    descriptive = labels_frame[["sample_id", "source_group", "system_id", "split"]]
    for runner, outputs in RUNNER_OUTPUTS.items():
        shard = descriptive.merge(
            predictors_frame[["sample_id", *outputs]], on="sample_id", validate="one_to_one"
        )
        shard.to_csv(shard_dir / f"{runner}.csv", index=False)

    certificate_path, label_provenance_path, _ = _write_integrity_artifacts(
        work_dir, labels_path, labels_frame, shard_dir,
    )

    out_dir = work_dir / "out"
    report = run_analysis(
        labels_path, shard_dir, out_dir,
        completion_certificate_path=certificate_path,
        label_provenance_path=label_provenance_path,
        bootstrap_draws=50,
    )

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
        for key in ("utterance_srcc", "utterance_pearson", "system_srcc"):
            value = values[key]
            assert np.isfinite(value), f"{method}.{key} is not finite: {value}"
        if method in {"best_single", "equal_ranks"}:
            assert values["utterance_mae"] is None
        else:
            assert np.isfinite(values["utterance_mae"])
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
    complete_case_path = out_dir / "somos_v2_complete_case_ids.json.gz"
    budget_predictions_path = out_dir / "somos_v2_budget_test_predictions.json.gz"
    report_path = out_dir / "somos_v2_prospective_results.json"
    assert predictions_path.exists()
    assert orders_path.exists()
    assert complete_case_path.exists()
    assert budget_predictions_path.exists()
    assert report_path.exists()

    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    assert len(predictions) == SPLIT_ROW_COUNTS["test"]
    assert BASELINE_METHODS.issubset(set(predictions.columns))

    with gzip.open(orders_path, "rt", encoding="utf-8") as handle:
        orders = json.load(handle)
    assert set(orders) == {str(seed) for seed in ACQUISITION_SEEDS}
    for sample_ids in orders.values():
        assert len(sample_ids) == SPLIT_ROW_COUNTS["train"]

    with gzip.open(complete_case_path, "rt", encoding="utf-8") as handle:
        complete_ids = json.load(handle)
    assert {split: len(ids) for split, ids in complete_ids.items()} == SPLIT_ROW_COUNTS
    with gzip.open(budget_predictions_path, "rt", encoding="utf-8") as handle:
        budget_predictions = json.load(handle)
    assert len(budget_predictions["runs"]) == len(ACQUISITION_SEEDS) * len(BUDGETS)
    assert len(budget_predictions["test_sample_ids"]) == SPLIT_ROW_COUNTS["test"]

    on_disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk_report["rows"] == report["rows"]
    assert on_disk_report["primary_bootstrap"] == report["primary_bootstrap"]
    assert on_disk_report["post_release_exploratory"] is True
    assert "not uncertainty over new systems" in (
        on_disk_report["configuration"]["primary_uncertainty_scope"]
    )
    assert on_disk_report["configuration"]["bootstrap_draws"] == 50
    assert on_disk_report["artifacts"]["completion_certificate"]["sha256"]
    assert on_disk_report["artifacts"]["label_provenance"]["sha256"]


def test_strict_json_rejects_nonfinite_results():
    with pytest.raises(ValueError, match="Out of range float values"):
        strict_json_text({"invalid": float("nan")})


def test_production_cli_does_not_accept_bootstrap_override():
    with pytest.raises(SystemExit):
        analysis_module.main([
            "--labels", "labels.csv",
            "--label-provenance", "labels.provenance.json",
            "--completion-certificate", "completion.json",
            "--shard-dir", "shards",
            "--out-dir", "out",
            "--bootstrap-draws", "50",
        ])


def test_label_metadata_must_be_derived_from_sample_id(work_dir, monkeypatch):
    labels = pd.DataFrame([
        {"sample_id": "sent0001_001.wav", "source_group": "tampered", "system_id": "001", "split": "train", "mos": 3.0},
        {"sample_id": "sent0002_002.wav", "source_group": "sent0002", "system_id": "002", "split": "valid", "mos": 3.0},
        {"sample_id": "sent0003_003.wav", "source_group": "sent0003", "system_id": "003", "split": "test", "mos": 3.0},
    ])
    labels_path = work_dir / "tampered-labels.csv"
    labels.to_csv(labels_path, index=False)
    split_rows = {"train": 1, "valid": 1, "test": 1}
    monkeypatch.setattr(analysis_module, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(analysis_module, "EXPECTED_SPLIT_ROWS", split_rows)
    with pytest.raises(ValueError, match="source_group"):
        analysis_module.assemble_matrix(labels_path, work_dir, {
            "sample_id_sha256": "0" * 64,
            "metadata_sha256": "0" * 64,
            "merge_artifacts": [],
        })
