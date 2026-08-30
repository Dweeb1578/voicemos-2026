"""Validate and merge prediction-only SOMOS score shards into ``<runner>.csv``.

This post-release exploratory utility uses the audio-only manifest, never MOS
targets.  It rejects missing parts, altered provenance, duplicate IDs, nonfinite
scores, and split-constant predictor outputs before creating the canonical
per-runner CSV consumed by the frozen analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from scripts.somos_scoring import (
    MANIFEST_COLUMNS,
    FROZEN_PROTOCOL_SHA256,
    RUNNERS,
    _stable_table_hash,
    assert_frozen_protocol,
    load_audio_manifest,
    merge_score_shards,
    select_shard,
    sha256_file,
    validate_score_shard,
)


NAME = re.compile(r"^(?P<runner>[a-z0-9]+)-part(?P<part>\d+)-of-(?P<count>\d+)\.csv$")
MERGE_TARGET_ACCESS = "No target MOS file or column was read during merge."


def _shard_tag(path: Path, runner_id: str) -> tuple[int, int]:
    match = NAME.fullmatch(path.name)
    if not match or match.group("runner") != runner_id:
        raise ValueError(f"not a {runner_id} deterministic shard filename: {path.name}")
    return int(match.group("part")), int(match.group("count"))


def collect_shards(shard_root: Path, runner_id: str,
                   expected_count: int | None = None) -> list[Path]:
    """Find one complete set of runner shard CSVs under downloaded outputs."""
    found = []
    for path in sorted(shard_root.rglob(f"{runner_id}-part*-of-*.csv")):
        match = NAME.fullmatch(path.name)
        if match and match.group("runner") == runner_id:
            found.append((*_shard_tag(path, runner_id), path))
    if not found:
        raise FileNotFoundError(f"no {runner_id} shard CSVs found under {shard_root}")
    counts = {count for _, count, _ in found}
    if len(counts) != 1:
        raise ValueError(f"{runner_id} score shards disagree on shard count: {sorted(counts)}")
    count = counts.pop()
    if expected_count is not None and count != expected_count:
        raise ValueError(
            f"{runner_id} score shards declare {count} parts, expected {expected_count}")
    parts = [part for part, _, _ in found]
    if sorted(parts) != list(range(count)):
        raise ValueError(f"{runner_id} missing or duplicate shard parts: {sorted(parts)}")
    return [path for _, _, path in sorted(found)]


def validate_provenance(score_path: Path, runner_id: str, *, expected_entries,
                        shard_index: int, shard_count: int,
                        audio_manifest_sha256: str) -> None:
    """Check a shard's recorded provenance against the exact local input contract."""
    provenance_path = score_path.with_suffix(".provenance.json")
    if not provenance_path.exists():
        raise FileNotFoundError(f"missing provenance beside {score_path.name}")
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if record.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        raise ValueError(f"{score_path.name}: protocol hash mismatch")
    if record.get("runner", {}).get("id") != runner_id:
        raise ValueError(f"{score_path.name}: runner provenance mismatch")
    if record.get("target_access") != "No target MOS file or column was read during scoring.":
        raise ValueError(f"{score_path.name}: target-access declaration mismatch")
    declared = record.get("score_shard", {})
    if declared.get("sha256") != sha256_file(score_path):
        raise ValueError(f"{score_path.name}: score-shard hash mismatch")
    expected_columns = ["sample_id", "source_group", "system_id", "split", *RUNNERS[runner_id]["outputs"]]
    if declared.get("columns") != expected_columns:
        raise ValueError(f"{score_path.name}: score-shard column provenance mismatch")
    if record.get("audio_manifest_sha256") != audio_manifest_sha256:
        raise ValueError(f"{score_path.name}: audio-manifest provenance mismatch")
    expected_shard = {"index": shard_index, "count": shard_count, "rows": int(len(expected_entries))}
    if record.get("shard") != expected_shard:
        raise ValueError(f"{score_path.name}: shard provenance mismatch")
    score_frame = pd.read_csv(score_path, dtype={"sample_id": str, "system_id": str})
    validate_score_shard(
        score_frame, expected_entries, RUNNERS[runner_id]["outputs"],
    )
    selected_ids_sha256 = record.get("selected_ids_sha256")
    if not isinstance(selected_ids_sha256, str) or len(selected_ids_sha256) != 64:
        raise ValueError(f"{score_path.name}: selected-ID provenance is missing or malformed")


def write_merge_provenance(
        output: Path, runner_id: str, paths: list[Path], rows: int,
) -> tuple[Path, dict]:
    """Cryptographically bind one canonical merged CSV to its four inputs."""

    provenance = {
        "schema_version": 1,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "runner": runner_id,
        "outputs": list(RUNNERS[runner_id]["outputs"]),
        "rows": int(rows),
        "merged_csv": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "columns": [
                "sample_id", "source_group", "system_id", "split",
                *RUNNERS[runner_id]["outputs"],
            ],
        },
        "input_shards": [
            {
                "score_csv": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                },
                "provenance": {
                    "path": str(path.with_suffix(".provenance.json")),
                    "sha256": sha256_file(path.with_suffix(".provenance.json")),
                },
            }
            for path in paths
        ],
        "target_access": MERGE_TARGET_ACCESS,
    }
    provenance_path = output.with_suffix(".merge.provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance_path, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4,
                        help="fixed deterministic shard count expected from the scoring kernels")
    args = parser.parse_args(argv)

    assert_frozen_protocol()
    manifest = load_audio_manifest(args.audio_manifest, args.audio_root)
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    paths = collect_shards(args.shard_root, args.runner, args.shard_count)
    manifest_hash = _stable_table_hash(manifest, MANIFEST_COLUMNS)
    for path in paths:
        part, count = _shard_tag(path, args.runner)
        expected_entries = select_shard(manifest, part, count)
        validate_provenance(
            path, args.runner, expected_entries=expected_entries,
            shard_index=part, shard_count=count,
            audio_manifest_sha256=manifest_hash,
        )
    output = args.out_dir / f"{args.runner}.csv"
    result = merge_score_shards(paths, args.runner, manifest, output)
    provenance_path, _ = write_merge_provenance(
        output, args.runner, paths, result["rows"],
    )
    result.update({
        "runner": args.runner,
        "shards": [str(path) for path in paths],
        "merge_provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        },
        "post_release_exploratory": True,
        "target_access": MERGE_TARGET_ACCESS,
    })
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
