"""Run the frozen SOMOS v2 prospective validation analysis.

This module deliberately contains no audio inference. It joins the frozen
predictor shards to the released clean-split labels, enforces one shared
complete-case matrix, and executes the predeclared validation/test protocol.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.somos_integrity import (
    EXPECTED_ROWS,
    EXPECTED_SPLIT_ROWS,
    FROZEN_PROTOCOL_SHA256,
    ID_RE,
    RUNNER_OUTPUTS as FROZEN_RUNNER_OUTPUTS,
    canonical_json_bytes,
    sha256_file,
    strict_json_text,
    validate_completion_certificate,
    validate_label_provenance,
    validate_merge_provenance,
)
from scripts.somos_scoring import assert_frozen_protocol

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
LASSO_ALPHAS = tuple(float(value) for value in np.logspace(-4, -1, 16))
BUDGETS = (200, 500, 1000, 2500)
ACQUISITION_SEEDS = tuple(range(10))
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_DRAWS = 10_000
EXPECTED_SPLITS = ("train", "valid", "test")
RUNNER_OUTPUTS = FROZEN_RUNNER_OUTPUTS
PREDICTORS = tuple(
    member for outputs in RUNNER_OUTPUTS.values() for member in outputs
)


def srcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = float(spearmanr(y_true, y_pred).statistic)
    return value if np.isfinite(value) else float("-inf")


class TrainECDF:
    """Right-continuous empirical CDF fitted independently per column."""

    def fit(self, X: np.ndarray) -> "TrainECDF":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("ECDF training matrix must be finite and two-dimensional")
        self.sorted_ = np.sort(values, axis=0)
        self.n_train_ = values.shape[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.sorted_.shape[1]:
            raise ValueError("ECDF transform matrix has an incompatible shape")
        transformed = np.empty_like(values, dtype=float)
        for column in range(values.shape[1]):
            transformed[:, column] = np.searchsorted(
                self.sorted_[:, column], values[:, column], side="right",
            ) / self.n_train_
        return transformed


class RankNNLS:
    """Non-negative least squares with an unconstrained centered intercept."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RankNNLS":
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = float(np.mean(y))
        self.coef_, _ = nnls(X - self.x_mean_, y - self.y_mean_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) @ self.coef_ + self.y_mean_


@dataclass
class SplitArrays:
    sample_id: np.ndarray
    source_group: np.ndarray
    system_id: np.ndarray
    y: np.ndarray
    X: np.ndarray


def _validate_shard(path: Path, expected: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(
        path, dtype={"sample_id": str, "source_group": str, "system_id": str},
    )
    required = ["sample_id", "source_group", "system_id", "split", *expected]
    if frame.columns.tolist() != required:
        raise ValueError(
            f"{path} columns {frame.columns.tolist()} do not match {required}"
        )
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values")
    return frame


def _metadata_hash(frame: pd.DataFrame) -> str:
    columns = ["sample_id", "source_group", "system_id", "split"]
    ordered = frame.loc[:, columns].sort_values("sample_id", kind="stable")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_id_hash(frame: pd.DataFrame) -> str:
    values = frame["sample_id"].sort_values(kind="stable").tolist()
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def assemble_matrix(
        labels_path: Path, shard_dir: Path, completion_payload: dict,
) -> tuple[pd.DataFrame, dict]:
    labels = pd.read_csv(
        labels_path,
        dtype={"sample_id": str, "source_group": str, "system_id": str},
    )
    required = ["sample_id", "source_group", "system_id", "split", "mos"]
    if labels.columns.tolist() != required:
        raise ValueError(f"label manifest columns must be exactly {required}")
    if len(labels) != EXPECTED_ROWS:
        raise ValueError(f"label manifest must contain exactly {EXPECTED_ROWS} rows")
    if labels["sample_id"].duplicated().any():
        raise ValueError("label manifest contains duplicate sample_id values")
    if set(labels["split"].unique()) != set(EXPECTED_SPLITS):
        raise ValueError(f"expected exactly splits {EXPECTED_SPLITS}")
    if labels["split"].value_counts().to_dict() != EXPECTED_SPLIT_ROWS:
        raise ValueError("label manifest split row counts do not match the frozen release")
    if not labels["mos"].between(1.0, 5.0, inclusive="both").all():
        raise ValueError("MOS values must be finite and in [1, 5]")

    derived = labels["sample_id"].str.extract(ID_RE)
    if derived.isna().any().any():
        raise ValueError("label manifest contains a sample ID outside the frozen schema")
    if not derived["source_group"].equals(labels["source_group"]):
        raise ValueError("label source_group does not match the frozen sample-ID derivation")
    if not derived["system_id"].equals(labels["system_id"]):
        raise ValueError("label system_id does not match the frozen sample-ID derivation")
    if _sample_id_hash(labels) != completion_payload["sample_id_sha256"]:
        raise ValueError("label sample IDs do not match the completion certificate")
    if _metadata_hash(labels) != completion_payload["metadata_sha256"]:
        raise ValueError("label metadata do not match the completion certificate")

    matrix = labels[required].copy()
    artifacts = {"labels": {
        "path": labels_path.as_posix(),
        "sha256": sha256_file(labels_path),
        "rows": len(labels),
    }, "shards": []}
    certificate_artifacts = {
        artifact["runner"]: artifact
        for artifact in completion_payload["merge_artifacts"]
    }
    label_metadata = labels.set_index("sample_id")[["source_group", "system_id", "split"]]
    for runner, expected in RUNNER_OUTPUTS.items():
        path = shard_dir / f"{runner}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        provenance_path = shard_dir / f"{runner}.merge.provenance.json"
        if not provenance_path.exists():
            raise FileNotFoundError(provenance_path)
        certified = certificate_artifacts[runner]
        if certified["merged_csv"]["file"] != path.name:
            raise ValueError(f"{runner} completion-certificate CSV filename mismatch")
        if certified["merged_csv"]["sha256"] != sha256_file(path):
            raise ValueError(f"{runner} merged CSV differs from the completion certificate")
        if certified["merged_csv"]["bytes"] != path.stat().st_size:
            raise ValueError(f"{runner} merged CSV byte count differs from the certificate")
        if certified["merge_provenance"]["file"] != provenance_path.name:
            raise ValueError(f"{runner} merge-provenance filename mismatch")
        if certified["merge_provenance"]["sha256"] != sha256_file(provenance_path):
            raise ValueError(f"{runner} merge provenance differs from the certificate")
        if certified["merge_provenance"]["bytes"] != provenance_path.stat().st_size:
            raise ValueError(f"{runner} merge-provenance byte count mismatch")
        validate_merge_provenance(provenance_path, path, runner, expected)
        shard = _validate_shard(path, expected)
        if len(shard) != EXPECTED_ROWS or set(shard["sample_id"]) != set(labels["sample_id"]):
            raise ValueError(f"{runner} merged IDs do not exactly match the label manifest")
        aligned_metadata = shard.set_index("sample_id").loc[label_metadata.index, [
            "source_group", "system_id", "split",
        ]]
        if not aligned_metadata.equals(label_metadata):
            raise ValueError(f"{runner} metadata do not match the canonical label manifest")
        artifacts["shards"].append({
            "runner": runner,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "rows": len(shard),
            "outputs": list(expected),
            "merge_provenance": {
                "path": provenance_path.as_posix(),
                "sha256": sha256_file(provenance_path),
            },
        })
        before = len(matrix)
        matrix = matrix.merge(
            shard[["sample_id", *expected]], on="sample_id", how="left",
            validate="one_to_one",
        )
        if len(matrix) != before:
            raise AssertionError("shard join changed label-manifest row count")

    finite_mask = np.isfinite(matrix[["mos", *PREDICTORS]].to_numpy(dtype=float)).all(axis=1)
    survival = {}
    for split in EXPECTED_SPLITS:
        split_mask = matrix["split"].eq(split).to_numpy()
        survival[split] = float(finite_mask[split_mask].mean())
        if survival[split] < 0.95:
            raise ValueError(
                f"shared complete-case survival for {split} is {survival[split]:.3%}"
            )
    matrix = matrix.loc[finite_mask].reset_index(drop=True)
    train = matrix.loc[matrix["split"].eq("train"), list(PREDICTORS)]
    constant = [name for name in PREDICTORS if train[name].nunique(dropna=False) < 2]
    if constant:
        raise ValueError(f"constant training predictors: {constant}")
    artifacts["complete_case_survival"] = survival
    artifacts["complete_case_rows"] = len(matrix)
    artifacts["predictors"] = list(PREDICTORS)
    return matrix, artifacts


def split_arrays(matrix: pd.DataFrame, split: str) -> SplitArrays:
    frame = matrix.loc[matrix["split"].eq(split)]
    return SplitArrays(
        sample_id=frame["sample_id"].to_numpy(),
        source_group=frame["source_group"].to_numpy(),
        system_id=frame["system_id"].to_numpy(),
        y=frame["mos"].to_numpy(dtype=float),
        X=frame[list(PREDICTORS)].to_numpy(dtype=float),
    )


def _choose_alpha(
        grid: tuple[float, ...], build_model, X_train: np.ndarray,
        y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
) -> tuple[float, object, float]:
    candidates = []
    for alpha in grid:
        model = build_model(alpha)
        model.fit(X_train, y_train)
        score = srcc(y_valid, model.predict(X_valid))
        candidates.append((score, alpha, model))
    score, alpha, model = max(candidates, key=lambda row: (row[0], row[1]))
    return float(alpha), model, float(score)


def fit_methods(
        train: SplitArrays, valid: SplitArrays, test: SplitArrays,
        train_index: np.ndarray, seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    X_train_raw = train.X[train_index]
    y_train = train.y[train_index]
    ecdf = TrainECDF().fit(X_train_raw)
    X_train_rank = ecdf.transform(X_train_raw)
    X_valid_rank = ecdf.transform(valid.X)
    X_test_rank = ecdf.transform(test.X)

    predictions: dict[str, np.ndarray] = {}
    details: dict = {
        "n_labels": len(train_index),
        "train_sample_ids": train.sample_id[train_index].tolist(),
        "ecdf": "right-continuous training empirical CDF",
    }

    # Protocol clarification fixed before target retrieval: "best single
    # output" means the public runner output itself. A training ECDF is
    # stepwise and can add validation ties, so it must not choose this control.
    single_scores = np.array([
        srcc(valid.y, valid.X[:, column])
        for column in range(valid.X.shape[1])
    ])
    best_score = float(np.max(single_scores))
    best_columns = np.flatnonzero(single_scores == best_score)
    best_column = min(best_columns, key=lambda index: PREDICTORS[index])
    predictions["best_single"] = test.X[:, best_column]
    details["best_single"] = {
        "predictor": PREDICTORS[best_column], "validation_srcc": best_score,
        "implementation_clarification": (
            "Selected on raw validation-output SRCC per the frozen best-single wording."
        ),
    }
    predictions["equal_ranks"] = np.mean(X_test_rank, axis=1)

    raw_alpha, raw_model, raw_valid = _choose_alpha(
        RIDGE_ALPHAS,
        lambda alpha: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        X_train_raw, y_train, valid.X, valid.y,
    )
    predictions["raw_ridge"] = raw_model.predict(test.X)
    details["raw_ridge"] = {
        "alpha": raw_alpha, "validation_srcc": raw_valid,
    }

    rank_alpha, rank_model, rank_valid = _choose_alpha(
        RIDGE_ALPHAS, lambda alpha: Ridge(alpha=alpha),
        X_train_rank, y_train, X_valid_rank, valid.y,
    )
    predictions["rank_ridge"] = rank_model.predict(X_test_rank)
    details["rank_ridge"] = {
        "alpha": rank_alpha, "validation_srcc": rank_valid,
    }

    nnls_model = RankNNLS().fit(X_train_rank, y_train)
    predictions["rank_nnls"] = nnls_model.predict(X_test_rank)
    details["rank_nnls"] = {
        "validation_srcc": srcc(valid.y, nnls_model.predict(X_valid_rank)),
        "nonzero": int(np.count_nonzero(nnls_model.coef_)),
    }

    lasso_alpha, lasso_model, lasso_valid = _choose_alpha(
        LASSO_ALPHAS,
        lambda alpha: make_pipeline(
            StandardScaler(),
            Lasso(alpha=alpha, max_iter=100_000, random_state=seed),
        ),
        X_train_rank, y_train, X_valid_rank, valid.y,
    )
    predictions["sparse_rank_lasso"] = lasso_model.predict(X_test_rank)
    details["sparse_rank_lasso"] = {
        "alpha": lasso_alpha,
        "validation_srcc": lasso_valid,
        "nonzero": int(np.count_nonzero(lasso_model[-1].coef_)),
    }
    return predictions, details


def metrics(
        y: np.ndarray, predictions: dict[str, np.ndarray], system_id: np.ndarray,
) -> dict:
    result = {}
    for method, values in predictions.items():
        grouped = pd.DataFrame({
            "system_id": system_id, "mos": y, "prediction": values,
        }).groupby("system_id", sort=True).mean(numeric_only=True)
        result[method] = {
            "utterance_srcc": srcc(y, values),
            "utterance_pearson": float(pearsonr(y, values).statistic),
            # Raw best-single outputs and equal-rank averages are not calibrated
            # to the 1-to-5 target scale, so their MAE is not comparable.
            "utterance_mae": (
                None if method in {"best_single", "equal_ranks"}
                else float(np.mean(np.abs(y - values)))
            ),
            "system_srcc": srcc(grouped["mos"].to_numpy(), grouped["prediction"].to_numpy()),
        }
    return result


def cluster_bootstrap_difference(
        y: np.ndarray, raw: np.ndarray, equal: np.ndarray,
        groups: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
        seed: int = BOOTSTRAP_SEED,
) -> dict:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=float)
    raw_scores = np.empty(draws, dtype=float)
    equal_scores = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in sampled])
        raw_scores[draw] = srcc(y[rows], raw[rows])
        equal_scores[draw] = srcc(y[rows], equal[rows])
        differences[draw] = raw_scores[draw] - equal_scores[draw]
    return {
        "seed": seed,
        "draws": draws,
        "cluster": "source_group",
        "raw_ridge": {
            "point": srcc(y, raw),
            "percentile_95_interval": np.quantile(raw_scores, [0.025, 0.975]).tolist(),
        },
        "equal_ranks": {
            "point": srcc(y, equal),
            "percentile_95_interval": np.quantile(equal_scores, [0.025, 0.975]).tolist(),
        },
        "raw_minus_equal": {
            "point": srcc(y, raw) - srcc(y, equal),
            "percentile_95_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        },
    }


def _write_gzip_json(path: Path, payload: dict) -> None:
    serialized = canonical_json_bytes(payload) + b"\n"
    path.write_bytes(gzip.compress(serialized, mtime=0))


def run_analysis(
        labels_path: Path, shard_dir: Path, out_dir: Path,
        *, completion_certificate_path: Path, label_provenance_path: Path,
        bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    protocol_record = assert_frozen_protocol()
    completion_payload = validate_completion_certificate(completion_certificate_path)
    validate_label_provenance(
        label_provenance_path, labels_path, completion_certificate_path,
    )
    matrix, artifacts = assemble_matrix(labels_path, shard_dir, completion_payload)
    artifacts["protocol"] = {
        "sha256": FROZEN_PROTOCOL_SHA256,
        "sidecar": protocol_record,
    }
    artifacts["completion_certificate"] = {
        "path": completion_certificate_path.as_posix(),
        "sha256": sha256_file(completion_certificate_path),
    }
    artifacts["label_provenance"] = {
        "path": label_provenance_path.as_posix(),
        "sha256": sha256_file(label_provenance_path),
    }
    train = split_arrays(matrix, "train")
    valid = split_arrays(matrix, "valid")
    test = split_arrays(matrix, "test")
    out_dir.mkdir(parents=True, exist_ok=True)

    acquisition_orders = {}
    runs = []
    budget_prediction_runs = []
    full_index = np.arange(len(train.y))
    full_predictions, full_details = fit_methods(
        train, valid, test, full_index, BOOTSTRAP_SEED,
    )
    full_metrics = metrics(test.y, full_predictions, test.system_id)

    for seed in ACQUISITION_SEEDS:
        order = np.random.default_rng(seed).permutation(len(train.y))
        acquisition_orders[str(seed)] = train.sample_id[order].tolist()
        for budget in BUDGETS:
            if budget > len(order):
                raise ValueError(f"budget {budget} exceeds {len(order)} training rows")
            predictions, details = fit_methods(
                train, valid, test, order[:budget], seed,
            )
            run_metrics = metrics(test.y, predictions, test.system_id)
            runs.append({
                "seed": seed,
                "budget": budget,
                "metrics": run_metrics,
                "fit": details,
                "paired_budget_minus_full": {
                    method: (
                        run_metrics[method]["utterance_srcc"]
                        - full_metrics[method]["utterance_srcc"]
                    ) for method in full_metrics
                },
            })
            budget_prediction_runs.append({
                "seed": seed,
                "budget": budget,
                "predictions": {
                    method: values.tolist() for method, values in predictions.items()
                },
            })

    budget_summary = {}
    for budget in BUDGETS:
        budget_rows = [row for row in runs if row["budget"] == budget]
        budget_summary[str(budget)] = {}
        for method in full_metrics:
            values = np.array([
                row["metrics"][method]["utterance_srcc"] for row in budget_rows
            ])
            gaps = np.array([
                row["paired_budget_minus_full"][method] for row in budget_rows
            ])
            budget_summary[str(budget)][method] = {
                "utterance_srcc_mean": float(np.mean(values)),
                "utterance_srcc_sample_sd": float(np.std(values, ddof=1)),
                "budget_minus_full_mean": float(np.mean(gaps)),
                "budget_minus_full_sample_sd": float(np.std(gaps, ddof=1)),
            }

    prediction_frame = pd.DataFrame({
        "sample_id": test.sample_id,
        "source_group": test.source_group,
        "system_id": test.system_id,
        "mos": test.y,
        **full_predictions,
    })
    predictions_path = out_dir / "somos_v2_test_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False, lineterminator="\n")
    orders_path = out_dir / "somos_v2_acquisition_orders.json.gz"
    _write_gzip_json(orders_path, acquisition_orders)
    complete_case_path = out_dir / "somos_v2_complete_case_ids.json.gz"
    _write_gzip_json(complete_case_path, {
        split: matrix.loc[matrix["split"].eq(split), "sample_id"].tolist()
        for split in EXPECTED_SPLITS
    })
    budget_predictions_path = out_dir / "somos_v2_budget_test_predictions.json.gz"
    _write_gzip_json(budget_predictions_path, {
        "test_sample_ids": test.sample_id.tolist(),
        "runs": budget_prediction_runs,
    })

    report = {
        "schema_version": "1.0",
        "analysis_status": "post-release exploratory frozen-protocol result",
        "post_release_exploratory": True,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol_commit": "d3b1dc01b70486d67183d64dee3a0680cb9961b7",
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "configuration": {
            "predictors": list(PREDICTORS),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "lasso_alphas": list(LASSO_ALPHAS),
            "budgets": list(BUDGETS),
            "acquisition_seeds": list(ACQUISITION_SEEDS),
            "ecdf": "right-continuous, fitted on labeled training subset only",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": bootstrap_draws,
            "primary_uncertainty_scope": (
                "One-way source-text cluster uncertainty conditional on the finite test systems; "
                "it is not uncertainty over new systems."
            ),
            "implementation_clarifications": {
                "best_single": (
                    "Selected on raw validation-output SRCC per the frozen best-single wording."
                ),
                "uncalibrated_mae": (
                    "MAE is unavailable for raw best-single and equal-rank predictions because "
                    "they are not calibrated to the 1-to-5 target scale."
                ),
            },
        },
        "rows": {split: int(matrix["split"].eq(split).sum()) for split in EXPECTED_SPLITS},
        "artifacts": artifacts,
        "full": {"metrics": full_metrics, "fit": full_details},
        "primary_bootstrap": cluster_bootstrap_difference(
            test.y, full_predictions["raw_ridge"], full_predictions["equal_ranks"],
            test.source_group, draws=bootstrap_draws,
        ),
        "budget_runs": runs,
        "budget_summary": budget_summary,
        "retained_outputs": {
            "test_predictions": {
                "path": predictions_path.as_posix(),
                "sha256": sha256_file(predictions_path),
            },
            "acquisition_orders": {
                "path": orders_path.as_posix(),
                "sha256": sha256_file(orders_path),
            },
            "complete_case_ids": {
                "path": complete_case_path.as_posix(),
                "sha256": sha256_file(complete_case_path),
            },
            "budget_test_predictions": {
                "path": budget_predictions_path.as_posix(),
                "sha256": sha256_file(budget_predictions_path),
            },
        },
    }
    report_path = out_dir / "somos_v2_prospective_results.json"
    report_path.write_text(strict_json_text(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-provenance", type=Path, required=True)
    parser.add_argument("--completion-certificate", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_analysis(
        args.labels, args.shard_dir, args.out_dir,
        completion_certificate_path=args.completion_certificate,
        label_provenance_path=args.label_provenance,
        bootstrap_draws=BOOTSTRAP_DRAWS,
    )
    print(strict_json_text({
        "status": report["analysis_status"],
        "rows": report["rows"],
        "primary": report["primary_bootstrap"],
    }).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
