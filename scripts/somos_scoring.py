"""Audio-only SOMOS v2 score-shard utilities.

This module is deliberately unable to read target labels.  It constructs an
audio manifest from the three official audio directories, deterministically
partitions clips, validates prediction-only score shards, and records the
runtime provenance required by the frozen SOMOS protocol.  Model inference
lives in :mod:`scripts.somos_runner`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "docs" / "mosaic_icassp_2027" / "third_corpus_protocol_frozen.md"
SIDECAR_PATH = PROTOCOL_PATH.with_suffix(".sha256.json")
FROZEN_PROTOCOL_SHA256 = "81daeb5dbfcac387ea9bad14dffe0603715999524028a2902757fc6aa1c241d9"
SOMOS_ARCHIVE_MD5 = "bdfde4cae256549dfab05d713136e4af"
DEFAULT_SHARD_COUNT = 4
CACHE_CHECKPOINT_ROWS = 50

ID_PATTERN = re.compile(
    r"^(?P<source_group>.+)_(?P<system_id>\d{3})\.wav$", re.IGNORECASE)
# The scorer accepts the original release names and the lossless audio-only
# materialization names.  The latter is required because Kaggle scoring must
# never mount the MOS-list files used to assign the official split.
SPLIT_DIRECTORIES = {
    "TRAINSET": "train", "VALIDSET": "valid", "TESTSET": "test",
    "train": "train", "valid": "valid", "test": "test",
}
SAMPLE_ID_COLUMN = "sample_id"
MANIFEST_COLUMNS = (SAMPLE_ID_COLUMN, "source_group", "system_id", "split", "relative_path")
FORBIDDEN_TARGET_COLUMNS = frozenset({
    "mos", "target", "target_mos", "label", "rating", "listener_score",
})

# This is a transcription of the frozen protocol, not a mutable model-selection
# registry.  The notebook generator uses only these entries.
RUNNERS = {
    "dnsmos": {
        "outputs": ("dnsmos", "dnsmos_sig"),
        "source": "microsoft/DNS-Challenge",
        "revision": "artifact-sha256",
        "weight_hashes": {
            "sig_bak_ovr.onnx": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
        },
        "weight_urls": {
            "sig_bak_ovr.onnx": "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        },
        "preprocessing": "mono float32, 16 kHz polyphase resample, 9.01 s windows",
        "gpu": False,
    },
    "p808": {
        "outputs": ("p808",),
        "source": "microsoft/DNS-Challenge",
        "revision": "artifact-sha256",
        "weight_hashes": {
            "model_v8.onnx": "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
        },
        "weight_urls": {
            "model_v8.onnx": "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx",
        },
        "preprocessing": "mono float32, 16 kHz polyphase resample, 9.01 s windows and 120-mel P.808 input",
        "gpu": False,
    },
    "squim": {
        "outputs": ("squim", "squim_stoi", "squim_sisdr"),
        "source": "pytorch/audio",
        "revision": "torchaudio==2.11.0",
        "weight_hashes": {},
        "preprocessing": "mono float32, 16 kHz polyphase resample",
        "gpu": False,  # torchaudio 2.11.0 wheels need CUDA 13; Kaggle has 12
    },
    "nisqa": {
        "outputs": ("nisqa",),
        "source": "gabrielmittag/NISQA",
        "revision": "fe84f0f252abec382b24367d5b22498a7ce34dbb",
        "weight_hashes": {},
        "preprocessing": "upstream NISQA predict_csv preprocessing",
        "gpu": True,
    },
    "distillmos": {
        "outputs": ("distillmos",),
        "source": "microsoft/Distill-MOS",
        "revision": "98c0a156b5dabf2b5a8fe9cee92145cdc2a2dcdb",
        "weight_hashes": {},
        "preprocessing": "mono first channel, 16 kHz torchaudio resample",
        "gpu": True,
    },
    "scoreq": {
        "outputs": ("scoreq_natural", "scoreq_synthetic"),
        "source": "alessandroragano/scoreq",
        "revision": "0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e",
        "weight_hashes": {},
        "preprocessing": "SCOREQ non-reference runner preprocessing",
        "gpu": False,  # onnxruntime CUDA provider needs CUDA 13; Kaggle has 12
    },
    "utmos": {
        "outputs": ("utmos",),
        "source": "sarulab-speech/UTMOSv2",
        "revision": "cc2700db57bb83ee13dc31ebe1b868c254e15d09",
        "weight_hashes": {},
        "preprocessing": "UTMOSv2 folder predictor with symlinked WAV paths",
        "gpu": True,
    },
    "sigmos": {
        "outputs": (
            "sigmos_mos_ovrl", "sigmos_mos_sig", "sigmos_mos_noise",
            "sigmos_mos_col", "sigmos_mos_disc", "sigmos_mos_loud",
            "sigmos_mos_reverb",
        ),
        "source": "microsoft/SIG-Challenge",
        "revision": "bf4525153b6ed998f19d9e79ff1fd00f55dec42b",
        "weight_hashes": {},
        "preprocessing": "mono float32 loaded by soundfile, SIGMOS internal resampling",
        "gpu": False,  # upstream builds an ONNX session with no CUDA provider
    },
    "audiobox": {
        "outputs": ("audiobox_PQ", "audiobox_CU", "audiobox_CE", "audiobox_PC"),
        "source": "facebookresearch/audiobox-aesthetics",
        "revision": "2618e9d451b456e9328b39495b5e6234678aa550",
        "model_source": "facebook/audiobox-aesthetics",
        "model_revision": "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "weight_hashes": {},
        "preprocessing": "soundfile decode, mono, 16 kHz torchaudio resample",
        "gpu": True,
    },
    "universa": {
        "outputs": (
            "universa_mos", "universa_scoreq", "universa_utmos",
            "universa_nisqa_mos", "universa_dnsmos_ovrl",
        ),
        "source": "vvwangvv/universa-ext_wavlm-base_5metric",
        "revision": "1fe08f4897655bf91e9b893030af872fa2a91694",
        "weight_hashes": {},
        "preprocessing": "soundfile decode, mono, model-config sample rate",
        "gpu": True,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_table_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    payload = frame.loc[:, list(columns)].to_csv(
        index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_frozen_protocol(repo: Path = REPO_ROOT) -> dict:
    """Fail if the embedded frozen protocol or its sidecar no longer match."""
    protocol = repo / PROTOCOL_PATH.relative_to(REPO_ROOT)
    sidecar = protocol.with_suffix(".sha256.json")
    if not protocol.exists() or not sidecar.exists():
        raise FileNotFoundError("frozen SOMOS protocol and sidecar must be embedded")
    actual = sha256_file(protocol)
    if actual != FROZEN_PROTOCOL_SHA256:
        raise ValueError("frozen SOMOS protocol SHA-256 mismatch")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if record.get("sha256") != actual:
        raise ValueError("SOMOS sidecar SHA-256 mismatch")
    return record


def build_audio_manifest(audio_root: Path) -> pd.DataFrame:
    """Index only WAV files in the three official audio directories.

    The function never opens text files.  An audio-only Kaggle dataset may also
    contain provenance JSON, but any ``*_mos_list.txt`` under its root is a hard
    error because scoring is not allowed to access target values.
    """
    audio_root = audio_root.resolve()
    forbidden = list(audio_root.rglob("*_mos_list.txt"))
    if forbidden:
        raise ValueError("audio root contains target MOS list files")
    rows = []
    for path in sorted(audio_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        relative = path.relative_to(audio_root)
        split_dir = next((part for part in relative.parts if part in SPLIT_DIRECTORIES), None)
        if split_dir is None:
            continue
        match = ID_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"SOMOS audio filename does not match frozen ID rule: {path.name}")
        rows.append({
            # Preserve the official label key byte-for-byte.  The SOMOS MOS
            # lists use filenames, including the .wav suffix.
            SAMPLE_ID_COLUMN: path.name,
            "source_group": match.group("source_group"),
            "system_id": match.group("system_id"),
            "split": SPLIT_DIRECTORIES[split_dir],
            "relative_path": relative.as_posix(),
        })
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if frame.empty:
        raise ValueError("no SOMOS WAV files found under the three prepared SOMOS split directories")
    if frame[SAMPLE_ID_COLUMN].duplicated().any():
        raise ValueError("duplicate sample_id in audio-only SOMOS manifest")
    if set(frame.split) != {"train", "valid", "test"}:
        raise ValueError(f"missing official split directories: {sorted(set(frame.split))}")
    return frame.sort_values(SAMPLE_ID_COLUMN, kind="stable").reset_index(drop=True)


def write_audio_manifest(audio_root: Path, output: Path) -> dict:
    frame = build_audio_manifest(audio_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {
        "path": str(output),
        "rows": int(len(frame)),
        "splits": {str(k): int(v) for k, v in frame.groupby("split").size().items()},
        "manifest_sha256": _stable_table_hash(frame, MANIFEST_COLUMNS),
    }


def load_audio_manifest(manifest_path: Path, audio_root: Path) -> pd.DataFrame:
    """Load a prediction-only audio manifest and resolve safe local paths."""
    frame = pd.read_csv(manifest_path, dtype={SAMPLE_ID_COLUMN: str, "system_id": str})
    unexpected = FORBIDDEN_TARGET_COLUMNS & set(frame.columns)
    if unexpected:
        raise ValueError(f"audio manifest contains forbidden target columns: {sorted(unexpected)}")
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"audio manifest missing columns: {sorted(missing)}")
    frame = frame.loc[:, list(MANIFEST_COLUMNS)].copy()
    if frame.isna().any().any() or frame[SAMPLE_ID_COLUMN].duplicated().any():
        raise ValueError("audio manifest has missing values or duplicate sample_id")
    if set(frame.split) != {"train", "valid", "test"}:
        raise ValueError("audio manifest must include train, valid, and test audio")
    root = audio_root.resolve()
    if list(root.rglob("*_mos_list.txt")):
        raise ValueError("audio root contains target MOS list files")
    frame["audio_path"] = [str((root / Path(value)).resolve())
                           for value in frame.relative_path]
    root_text = str(root).lower()
    if any(not path.lower().startswith(root_text) for path in frame.audio_path):
        raise ValueError("audio path escaped audio root")
    missing_audio = [path for path in frame.audio_path if not Path(path).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"audio manifest references missing WAV: {missing_audio[0]}")
    return frame.sort_values(SAMPLE_ID_COLUMN, kind="stable").reset_index(drop=True)


def shard_index(sample_id: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % shard_count


def select_shard(frame: pd.DataFrame, part_index: int, shard_count: int) -> pd.DataFrame:
    if not 0 <= part_index < shard_count:
        raise ValueError("part_index must be in [0, shard_count)")
    selected = frame.loc[
        [shard_index(value, shard_count) == part_index
         for value in frame[SAMPLE_ID_COLUMN]]
    ].copy()
    if selected.empty:
        raise ValueError("deterministic shard is empty")
    return selected.reset_index(drop=True)


def load_cache(cache_path: Path, output_columns: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
    """Load a resumable cache, ignoring torn or non-finite final rows."""
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(cache_path, on_bad_lines="skip")
    needed = {"audio_path", *output_columns}
    if not needed <= set(frame.columns):
        raise ValueError(f"cache schema mismatch in {cache_path}")
    frame = frame.dropna(subset=list(needed))
    if frame.audio_path.duplicated().any():
        raise ValueError(f"duplicate audio paths in score cache: {cache_path}")
    values = frame.loc[:, list(output_columns)].to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    frame = frame.loc[finite]
    return {
        row.audio_path: tuple(float(row[column]) for column in output_columns)
        for _, row in frame.iterrows()
    }


def score_paths_resumable(
        scorer: Callable[[str], tuple[float, ...]], entries: pd.DataFrame,
        cache_path: Path, output_columns: tuple[str, ...], deadline_monotonic: float,
        description: str) -> dict[str, tuple[float, ...]]:
    """Append finite scores row-by-row and preserve progress across restarts."""
    done = load_cache(cache_path, output_columns)
    todo = [path for path in entries.audio_path if path not in done]
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
        for index, audio_path in enumerate(todo, start=1):
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError(
                    f"{description}: stopping before the 12-hour kernel limit; "
                    f"{len(done)} cached, {len(todo) - index + 1} remain")
            values = tuple(float(value) for value in scorer(audio_path))
            if len(values) != len(output_columns) or not np.isfinite(values).all():
                raise ValueError(f"{description}: non-finite or malformed score for {audio_path}")
            writer.writerow([audio_path, *values])
            # A periodic durable checkpoint bounds restart work without paying
            # a filesystem sync for every one of roughly 20k SOMOS clips.
            # `load_cache` drops a torn final row safely on the next attempt.
            if index % CACHE_CHECKPOINT_ROWS == 0:
                handle.flush()
                os.fsync(handle.fileno())
            done[audio_path] = values
        if todo:
            handle.flush()
            os.fsync(handle.fileno())
    return {path: done[path] for path in entries.audio_path if path in done}


def validate_score_shard(frame: pd.DataFrame, expected: pd.DataFrame,
                         output_columns: tuple[str, ...]) -> dict:
    forbidden = FORBIDDEN_TARGET_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"score shard contains forbidden target columns: {sorted(forbidden)}")
    expected_columns = [SAMPLE_ID_COLUMN, "source_group", "system_id", "split", *output_columns]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "score shard schema mismatch: expected exactly "
            f"{expected_columns}, got {list(frame.columns)}")
    if frame[SAMPLE_ID_COLUMN].duplicated().any() or frame[SAMPLE_ID_COLUMN].isna().any():
        raise ValueError("score shard has duplicate or missing sample_id")
    expected_ids = set(expected[SAMPLE_ID_COLUMN])
    if set(frame[SAMPLE_ID_COLUMN]) != expected_ids:
        raise ValueError("score shard IDs do not exactly match its deterministic audio shard")
    aligned = frame.set_index(SAMPLE_ID_COLUMN).loc[expected[SAMPLE_ID_COLUMN]]
    expected_meta = expected.set_index(SAMPLE_ID_COLUMN)
    for column in ("source_group", "system_id", "split"):
        if not aligned[column].astype(str).equals(expected_meta[column].astype(str)):
            raise ValueError(f"score shard metadata mismatch in {column}")
    values = aligned.loc[:, list(output_columns)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("score shard has non-finite predictor values")
    return {
        "rows": int(len(frame)),
        "split_rows": {str(k): int(v) for k, v in aligned.groupby("split").size().items()},
        "sample_id_sha256": hashlib.sha256(
            "\n".join(expected[SAMPLE_ID_COLUMN]).encode("utf-8")).hexdigest(),
    }


def write_score_shard(entries: pd.DataFrame, cache_path: Path,
                      output_columns: tuple[str, ...], output_path: Path) -> dict:
    cached = load_cache(cache_path, output_columns)
    missing = [path for path in entries.audio_path if path not in cached]
    if missing:
        raise ValueError(f"cannot finalize incomplete score cache, first missing: {missing[0]}")
    rows = []
    for _, row in entries.iterrows():
        values = cached[row.audio_path]
        rows.append({
            SAMPLE_ID_COLUMN: row[SAMPLE_ID_COLUMN],
            "source_group": row.source_group,
            "system_id": row.system_id,
            "split": row.split,
            **dict(zip(output_columns, values)),
        })
    result = pd.DataFrame(rows)
    validation = validate_score_shard(result, entries, output_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    validation["sha256"] = sha256_file(output_path)
    validation["bytes"] = output_path.stat().st_size
    return validation


# A Hugging Face cache holds partial downloads and lock files while a fetch is
# in flight.  They are not model artifacts, and they are renamed or removed the
# moment the download completes, so hashing them is both meaningless and racy.
TRANSIENT_ARTIFACT_SUFFIXES = (".incomplete", ".lock")


def _is_transient_artifact(path: Path) -> bool:
    return path.name.endswith(TRANSIENT_ARTIFACT_SUFFIXES)


def tree_inventory(paths: Iterable[Path]) -> list[dict]:
    """Hash model artifacts before accepting scores, preserving relative names."""
    result = []
    for root in paths:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"declared model artifact path missing: {root}")
        if root.is_file():
            files = [root]
        else:
            files = sorted(
                path for path in root.rglob("*")
                if path.is_file() and not _is_transient_artifact(path)
            )
        if not files:
            raise ValueError(f"declared model artifact path is empty: {root}")
        for path in files:
            try:
                digest = sha256_file(path)
                size = path.stat().st_size
            except FileNotFoundError:
                # A completed download can rename its blob while the walk runs.
                # Only a transient file may vanish; a real artifact going
                # missing stays an error.
                if _is_transient_artifact(path):
                    continue
                raise
            result.append({"path": str(path), "sha256": digest, "bytes": size})
    return result


def environment_snapshot(output_path: Path) -> dict:
    """Capture the installed environment and hardware before inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True,
        text=True, check=False).stdout
    # Two frozen runners are CPU only, and those machines have no driver, so a
    # missing nvidia-smi is expected rather than a failure.  check=False does
    # not cover this: a missing executable raises before the process starts.
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False).stdout
        nvidia_smi_present = True
    except (FileNotFoundError, OSError):
        gpu = ""
        nvidia_smi_present = False
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": pip.splitlines(),
        "gpu": gpu.splitlines(),
        "nvidia_smi_present": nvidia_smi_present,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"path": str(output_path), "sha256": sha256_file(output_path)}


def write_shard_provenance(output_path: Path, *, runner_id: str,
                           audio_manifest: pd.DataFrame, entries: pd.DataFrame,
                           shard_count: int, shard_part: int, score_path: Path,
                           cache_path: Path, initialization_path: Path,
                           environment: dict, model_artifacts: list[dict],
                           runner_source_sha256: dict[str, str],
                           timing: dict | None = None) -> dict:
    """Write an auditable, prediction-only provenance record for one shard."""
    if runner_id not in RUNNERS:
        raise ValueError(f"unknown frozen SOMOS runner: {runner_id}")
    spec = RUNNERS[runner_id]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "somos_archive_md5": SOMOS_ARCHIVE_MD5,
        "runner": {"id": runner_id, **spec},
        "audio_manifest_sha256": _stable_table_hash(audio_manifest, MANIFEST_COLUMNS),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(entries[SAMPLE_ID_COLUMN]).encode("utf-8")).hexdigest(),
        "shard": {"index": shard_part, "count": shard_count, "rows": int(len(entries))},
        "score_shard": {
            "path": str(score_path), "sha256": sha256_file(score_path),
            "bytes": score_path.stat().st_size, "columns": [
                SAMPLE_ID_COLUMN, "source_group", "system_id", "split", *spec["outputs"],
            ],
        },
        "resume_cache": {
            "path": str(cache_path), "sha256": sha256_file(cache_path),
            "bytes": cache_path.stat().st_size,
        },
        "initialization": {
            "path": str(initialization_path), "sha256": sha256_file(initialization_path),
        },
        "environment": environment,
        "model_artifacts": model_artifacts,
        "runner_source_sha256": runner_source_sha256,
        "timing": timing,
        "target_access": "No target MOS file or column was read during scoring.",
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def merge_score_shards(paths: list[Path], runner_id: str, manifest: pd.DataFrame,
                       output_path: Path) -> dict:
    """Merge all deterministic runner shards without opening target values."""
    if runner_id not in RUNNERS:
        raise ValueError(f"unknown frozen SOMOS runner: {runner_id}")
    output_columns = RUNNERS[runner_id]["outputs"]
    parts = [pd.read_csv(path, dtype={SAMPLE_ID_COLUMN: str, "system_id": str})
             for path in paths]
    merged = pd.concat(parts, ignore_index=True)
    validate_score_shard(merged, manifest, output_columns)
    for split, block in merged.groupby("split"):
        for column in output_columns:
            if block[column].nunique(dropna=True) < 2:
                raise ValueError(f"{runner_id} {column} is constant on {split}")
    result = merged.sort_values(SAMPLE_ID_COLUMN, kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return {"path": str(output_path), "rows": int(len(result)), "sha256": sha256_file(output_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--assert-protocol", action="store_true")
    args = parser.parse_args(argv)
    if args.assert_protocol:
        print(json.dumps(assert_frozen_protocol(), indent=2))
    if args.audio_root or args.write_manifest:
        if not args.audio_root or not args.write_manifest:
            parser.error("--audio-root and --write-manifest must be used together")
        print(json.dumps(write_audio_manifest(args.audio_root, args.write_manifest), indent=2))
    if not args.assert_protocol and not args.audio_root:
        parser.error("choose --assert-protocol or --audio-root with --write-manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
