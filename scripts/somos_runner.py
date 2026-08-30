"""Run one frozen SOMOS predictor over one deterministic audio-only shard.

This script never opens an official ``*_mos_list.txt`` file and rejects any
manifest containing target columns.  It is intended for the generated Kaggle
kernels from :mod:`scripts.somos_kaggle_orchestrate`, one model runner per
kernel and one deterministic data shard per kernel invocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from scripts.somos_scoring import (
    DEFAULT_SHARD_COUNT,
    FROZEN_PROTOCOL_SHA256,
    RUNNERS,
    SAMPLE_ID_COLUMN,
    assert_frozen_protocol,
    environment_snapshot,
    load_audio_manifest,
    load_cache,
    score_paths_resumable,
    select_shard,
    sha256_file,
    tree_inventory,
    validate_score_shard,
    write_score_shard,
    write_shard_provenance,
)


def _first_named(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name!r} under {root}, found {len(matches)}")
    return matches[0]


def _check_expected_dns_weights(root: Path, runner_id: str) -> list[Path]:
    expected = RUNNERS[runner_id]["weight_hashes"]
    paths = []
    for name, wanted in expected.items():
        path = _first_named(root, name)
        actual = sha256_file(path)
        if actual != wanted:
            raise ValueError(f"{runner_id} frozen weight hash mismatch for {name}")
        paths.append(path)
    return paths


def _append_rows(cache_path: Path, output_columns: tuple[str, ...],
                 rows: list[tuple[str, tuple[float, ...]]]) -> None:
    """Append checked rows with the same crash-safe schema as score_paths_resumable."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not cache_path.exists() or cache_path.stat().st_size == 0
    if not new_file:
        with cache_path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    with cache_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["audio_path", *output_columns])
        for audio_path, values in rows:
            if len(values) != len(output_columns) or not np.isfinite(values).all():
                raise ValueError(f"non-finite or malformed score for {audio_path}")
            writer.writerow([audio_path, *values])
        handle.flush()
        os.fsync(handle.fileno())


def _score_dnsmos(entries: pd.DataFrame, cache_path: Path, deadline: float,
                  model_root: Path, before_score: Callable[[], None]) -> None:
    from scripts.zero_shot_dnsmos import load_audio_16k, make_onnx_infer, score_clip

    model = _check_expected_dns_weights(model_root, "dnsmos")[0]
    infer = make_onnx_infer(str(model))

    def scorer(path: str) -> tuple[float, float]:
        sig, _bak, ovrl = score_clip(infer, load_audio_16k(path))
        return ovrl, sig

    before_score()
    score_paths_resumable(scorer, entries, cache_path, RUNNERS["dnsmos"]["outputs"],
                          deadline, "DNSMOS")


def _score_p808(entries: pd.DataFrame, cache_path: Path, deadline: float,
                model_root: Path, before_score: Callable[[], None]) -> None:
    from scripts.zero_shot_p808 import make_p808_scorer

    model = _check_expected_dns_weights(model_root, "p808")[0]
    scorer = make_p808_scorer(str(model))
    before_score()
    score_paths_resumable(scorer, entries, cache_path,
                          RUNNERS["p808"]["outputs"], deadline, "DNSMOS P.808")


def _score_squim(entries: pd.DataFrame, cache_path: Path, deadline: float,
                 before_score: Callable[[], None]) -> None:
    from scripts.zero_shot_squim import make_squim_scorer

    # make_squim_scorer picks cuda when it is available and cpu otherwise.
    scorer = make_squim_scorer()

    def reordered(path: str) -> tuple[float, float, float]:
        stoi, pesq, si_sdr = scorer(path)
        return pesq, stoi, si_sdr

    before_score()
    score_paths_resumable(reordered, entries, cache_path, RUNNERS["squim"]["outputs"],
                          deadline, "SQUIM")


def _score_distillmos(entries: pd.DataFrame, cache_path: Path, deadline: float,
                      before_score: Callable[[], None]) -> None:
    import torch
    import torchaudio
    import distillmos

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = distillmos.ConvTransformerSQAModel().to(device).eval()

    def scorer(path: str) -> tuple[float]:
        wave, sample_rate = torchaudio.load(path)
        if wave.shape[0] > 1:
            wave = wave[0:1]
        if sample_rate != 16000:
            wave = torchaudio.functional.resample(wave, sample_rate, 16000)
        with torch.inference_mode():
            result = model(wave.to(device))
        return (float(result.flatten()[0]),)

    before_score()
    score_paths_resumable(scorer, entries, cache_path,
                          RUNNERS["distillmos"]["outputs"], deadline, "Distill-MOS")


def _score_scoreq(entries: pd.DataFrame, cache_path: Path, deadline: float,
                  vendor_root: Path, before_score: Callable[[], None]) -> None:
    # The frozen commit ships a src layout with no packaging metadata, so the
    # clone is imported from disk rather than installed.
    sys.path.insert(0, str(vendor_root / "src"))
    import scoreq

    natural = scoreq.Scoreq(data_domain="natural", mode="nr")
    synthetic = scoreq.Scoreq(data_domain="synthetic", mode="nr")

    def scorer(path: str) -> tuple[float, float]:
        return (
            float(natural.predict(test_path=path, ref_path=None)),
            float(synthetic.predict(test_path=path, ref_path=None)),
        )

    before_score()
    score_paths_resumable(scorer, entries, cache_path,
                          RUNNERS["scoreq"]["outputs"], deadline, "SCOREQ")


def _score_sigmos(entries: pd.DataFrame, cache_path: Path, deadline: float,
                  vendor_root: Path, before_score: Callable[[], None]) -> None:
    import soundfile as sf

    sys.path.insert(0, str(vendor_root / "ICASSP2024" / "sigmos"))
    from sigmos import SigMOS

    model_root = vendor_root / "ICASSP2024" / "sigmos"
    estimator = SigMOS(model_dir=str(model_root))
    keys = {
        "sigmos_mos_ovrl": "MOS_OVRL",
        "sigmos_mos_sig": "MOS_SIG",
        "sigmos_mos_noise": "MOS_NOISE",
        "sigmos_mos_col": "MOS_COL",
        "sigmos_mos_disc": "MOS_DISC",
        "sigmos_mos_loud": "MOS_LOUD",
        "sigmos_mos_reverb": "MOS_REVERB",
    }

    def scorer(path: str) -> tuple[float, ...]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        values = estimator.run(audio.mean(axis=1), sr=sample_rate)
        return tuple(float(values[keys[name]]) for name in RUNNERS["sigmos"]["outputs"])

    before_score()
    score_paths_resumable(scorer, entries, cache_path,
                          RUNNERS["sigmos"]["outputs"], deadline, "SIGMOS")


def _score_audiobox(entries: pd.DataFrame, cache_path: Path, deadline: float,
                    checkpoint: Path, before_score: Callable[[], None]) -> None:
    from scripts.zero_shot_audiobox import load_item
    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor(str(checkpoint))

    def scorer(path: str) -> tuple[float, float, float, float]:
        values = predictor.forward([load_item(path)])[0]
        return (float(values["PQ"]), float(values["CU"]),
                float(values["CE"]), float(values["PC"]))

    before_score()
    score_paths_resumable(scorer, entries, cache_path,
                          RUNNERS["audiobox"]["outputs"], deadline, "Audiobox")


def _score_universa(entries: pd.DataFrame, cache_path: Path, deadline: float,
                    model_path: Path, config_path: Path,
                    before_score: Callable[[], None]) -> None:
    import soundfile as sf
    import torch
    from urgent2026_sqa.infer import infer_single, load_model

    model, config = load_model(model_path, config_path)
    output_map = {
        "universa_mos": "mos",
        "universa_scoreq": "scoreq",
        "universa_utmos": "utmos",
        "universa_nisqa_mos": "nisqa_mos",
        "universa_dnsmos_ovrl": "dnsmos_ovrl",
    }

    def scorer(path: str) -> tuple[float, ...]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio.T.copy())
        values = infer_single(model, config, wave, sample_rate)
        return tuple(float(values[output_map[name]])
                     for name in RUNNERS["universa"]["outputs"])

    before_score()
    score_paths_resumable(scorer, entries, cache_path,
                          RUNNERS["universa"]["outputs"], deadline, "Uni-VERSA")


def _score_nisqa(entries: pd.DataFrame, cache_path: Path, deadline: float,
                 vendor_root: Path, work_dir: Path,
                 before_score: Callable[[], None]) -> None:
    """Score a full small shard with upstream NISQA, then append a resume cache."""
    cached = load_cache(cache_path, RUNNERS["nisqa"]["outputs"])
    todo = entries.loc[~entries.audio_path.isin(cached)].copy()
    if todo.empty:
        return
    if time.monotonic() >= deadline:
        raise TimeoutError("NISQA: stopping before the 12-hour kernel limit")
    before_score()
    work_dir.mkdir(parents=True, exist_ok=True)
    input_csv = work_dir / "nisqa_input.csv"
    output_dir = work_dir / "nisqa_out"
    # Upstream run_predict.py writes its results CSV without creating this.
    output_dir.mkdir(parents=True, exist_ok=True)
    todo.loc[:, ["audio_path"]].rename(columns={"audio_path": "deg"}).to_csv(input_csv, index=False)
    command = [
        sys.executable, str(vendor_root / "run_predict.py"), "--mode", "predict_csv",
        "--pretrained_model", str(vendor_root / "weights" / "nisqa.tar"),
        "--csv_file", str(input_csv), "--csv_deg", "deg", "--num_workers", "2",
        "--bs", "16", "--output_dir", str(output_dir),
    ]
    subprocess.run(command, check=True)
    values = pd.read_csv(output_dir / "NISQA_results.csv").rename(columns={"deg": "audio_path"})
    if not {"audio_path", "mos_pred"} <= set(values.columns):
        raise ValueError("NISQA output schema did not contain path and mos_pred")
    value_map = dict(zip(values.audio_path, values.mos_pred))
    if set(todo.audio_path) != set(value_map):
        raise ValueError("NISQA output IDs do not match the requested audio shard")
    _append_rows(cache_path, RUNNERS["nisqa"]["outputs"], [
        (path, (float(value_map[path]),)) for path in todo.audio_path
    ])


def _score_utmos(entries: pd.DataFrame, cache_path: Path, deadline: float,
                 work_dir: Path, before_score: Callable[[], None]) -> None:
    """Run UTMOSv2 in small symlink batches while preserving a path cache."""
    import utmosv2

    output_columns = RUNNERS["utmos"]["outputs"]
    cached = load_cache(cache_path, output_columns)
    todo = [path for path in entries.audio_path if path not in cached]
    if not todo:
        return
    model = utmosv2.create_model(pretrained=True, device="cuda")
    before_score()
    chunk_dir = work_dir / "utmos_chunk"
    for start in range(0, len(todo), 64):
        if time.monotonic() >= deadline:
            raise TimeoutError("UTMOS: stopping before the 12-hour kernel limit")
        chunk = todo[start:start + 64]
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True)
        stem_to_path = {}
        for index, path in enumerate(chunk):
            stem = f"clip_{index:04d}"
            stem_to_path[stem] = path
            os.symlink(path, chunk_dir / f"{stem}.wav")
        results = model.predict(input_dir=str(chunk_dir), batch_size=8, num_workers=2)
        if len(results) != len(chunk):
            raise ValueError(f"UTMOS returned {len(results)} predictions for {len(chunk)} inputs")
        rows = []
        for row in results:
            stem = Path(str(row["file_path"])).stem
            if stem not in stem_to_path:
                raise ValueError(f"UTMOS returned unknown symlink stem {stem}")
            rows.append((stem_to_path[stem], (float(row["predicted_mos"]),)))
        _append_rows(cache_path, output_columns, rows)


def _run_scoring(args: argparse.Namespace, entries: pd.DataFrame,
                 cache_path: Path, deadline: float,
                 before_score: Callable[[], None]) -> None:
    runner_id = args.runner
    if runner_id == "dnsmos":
        _score_dnsmos(entries, cache_path, deadline, args.dns_model_root, before_score)
    elif runner_id == "p808":
        _score_p808(entries, cache_path, deadline, args.dns_model_root, before_score)
    elif runner_id == "squim":
        _score_squim(entries, cache_path, deadline, before_score)
    elif runner_id == "nisqa":
        _score_nisqa(entries, cache_path, deadline, args.vendor_root, args.out_dir / "work", before_score)
    elif runner_id == "distillmos":
        _score_distillmos(entries, cache_path, deadline, before_score)
    elif runner_id == "scoreq":
        _score_scoreq(entries, cache_path, deadline, args.vendor_root, before_score)
    elif runner_id == "utmos":
        _score_utmos(entries, cache_path, deadline, args.out_dir / "work", before_score)
    elif runner_id == "sigmos":
        _score_sigmos(entries, cache_path, deadline, args.vendor_root, before_score)
    elif runner_id == "audiobox":
        _score_audiobox(entries, cache_path, deadline, args.audiobox_checkpoint, before_score)
    elif runner_id == "universa":
        _score_universa(entries, cache_path, deadline,
                        args.universa_model, args.universa_config, before_score)
    else:  # argparse choices makes this unreachable, retained for direct calls.
        raise ValueError(f"unknown runner {runner_id}")


def _runner_source_hashes() -> dict[str, str]:
    """Hash every local helper that can affect a generated score shard."""
    root = Path(__file__).resolve().parent
    names = [
        "somos_runner.py", "somos_scoring.py", "clip_cache.py",
        "zero_shot_dnsmos.py", "zero_shot_p808.py", "zero_shot_squim.py",
        "zero_shot_audiobox.py",
    ]
    return {name: sha256_file(root / name) for name in names if (root / name).exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--max-runtime-minutes", type=int, default=660)
    parser.add_argument("--dns-model-root", type=Path, default=Path("."))
    parser.add_argument("--vendor-root", type=Path, default=Path("."))
    parser.add_argument("--audiobox-checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--universa-model", type=Path, default=Path("model.pt"))
    parser.add_argument("--universa-config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--artifact-path", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--smoke-items", type=int, default=0,
        help="score only this many deterministic shard rows, record a smoke provenance file, and do not finalize a score shard",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    process_started = time.monotonic()
    assert_frozen_protocol()
    audio = load_audio_manifest(args.audio_manifest, args.audio_root)
    entries = select_shard(audio, args.shard_index, args.shard_count)
    if args.smoke_items < 0:
        parser.error("--smoke-items must be non-negative")
    scoring_entries = entries.head(args.smoke_items).copy() if args.smoke_items else entries
    if scoring_entries.empty:
        parser.error("--smoke-items selected no rows")
    tag = f"{args.runner}-part{args.shard_index:02d}-of-{args.shard_count:02d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / f"{tag}.cache.csv"
    score_path = args.out_dir / f"{tag}.csv"
    initialization_path = args.out_dir / f"{tag}.initialization.json"
    provenance_path = args.out_dir / f"{tag}.provenance.json"
    output_columns = RUNNERS[args.runner]["outputs"]

    if score_path.exists() and provenance_path.exists():
        existing = pd.read_csv(score_path, dtype={SAMPLE_ID_COLUMN: str, "system_id": str})
        validation = validate_score_shard(existing, entries, output_columns)
        print(json.dumps({"status": "already-finalized", "tag": tag, **validation}, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({"status": "validated-inputs", "tag": tag, "rows": len(entries)}, indent=2))
        return 0

    deadline = time.monotonic() + args.max_runtime_minutes * 60
    initialization: dict = {}
    timing: dict = {}

    def before_score() -> None:
        """Hash initialized model assets before writing the first score row."""
        nonlocal initialization
        if initialization:
            return
        timing["model_init_seconds"] = round(time.monotonic() - process_started, 3)
        timing["scoring_started_monotonic"] = time.monotonic()
        artifacts = tree_inventory(args.artifact_path)
        environment = environment_snapshot(args.out_dir / f"{tag}.environment.json")
        initialization = {
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "runner": args.runner,
            "shard": {"index": args.shard_index, "count": args.shard_count},
            "artifact_paths": [str(path) for path in args.artifact_path],
            "model_artifacts": artifacts,
            "environment": environment,
            "written_before_first_score_cache_row": True,
        }
        initialization_path.write_text(
            json.dumps(initialization, indent=2) + "\n", encoding="utf-8")

    rows_before = len(load_cache(cache_path, output_columns))
    _run_scoring(args, scoring_entries, cache_path, deadline, before_score)
    if not initialization:
        # A fully cached shard still needs a fresh initialization record before
        # it can be accepted.  No new score is written in this branch.
        before_score()
    scored = len(load_cache(cache_path, output_columns)) - rows_before
    elapsed = round(time.monotonic() - timing.pop(
        "scoring_started_monotonic", process_started), 3)
    rate = round(scored / elapsed, 4) if scored and elapsed > 0 else None
    timing.update({
        "rows_scored": scored,
        "scoring_seconds": elapsed,
        "rows_per_second": rate,
        "shard_rows": int(len(entries)),
        # The frozen bank has to fit a fixed Kaggle GPU quota, so every run
        # reports what the whole shard would cost at the observed rate.
        "projected_shard_hours": round(len(entries) / rate / 3600, 3) if rate else None,
    })
    print(json.dumps({"timing": timing}, indent=2), flush=True)
    if args.smoke_items:
        smoke_path = args.out_dir / f"{tag}.smoke.provenance.json"
        smoke = {
            "schema_version": 1,
            "status": "smoke-only-not-a-final-score-shard",
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "runner": args.runner,
            "shard": {"index": args.shard_index, "count": args.shard_count},
            "requested_rows": int(len(scoring_entries)),
            "cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
            "initialization": initialization,
            "runner_source_sha256": _runner_source_hashes(),
            "timing": timing,
            "target_access": "No target MOS file or column was read during scoring.",
        }
        smoke_path.write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "smoke-only", "tag": tag,
                          "rows": len(scoring_entries), "provenance": str(smoke_path)}, indent=2))
        return 0

    validation = write_score_shard(entries, cache_path, output_columns, score_path)
    write_shard_provenance(
        provenance_path, runner_id=args.runner, audio_manifest=audio, entries=entries,
        shard_count=args.shard_count, shard_part=args.shard_index, score_path=score_path,
        cache_path=cache_path, initialization_path=initialization_path,
        environment=initialization["environment"],
        model_artifacts=initialization["model_artifacts"],
        runner_source_sha256=_runner_source_hashes(), timing=timing)
    print(json.dumps({"status": "complete", "tag": tag, **validation,
                      "provenance": str(provenance_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
