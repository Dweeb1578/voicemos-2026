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

from scripts.somos_scoring import (
    FROZEN_PROTOCOL_SHA256,
    RUNNERS,
    assert_frozen_protocol,
    load_audio_manifest,
    merge_score_shards,
    sha256_file,
)


NAME = re.compile(r"^(?P<runner>[a-z0-9]+)-part(?P<part>\d+)-of-(?P<count>\d+)\.csv$")


def collect_shards(shard_root: Path, runner_id: str) -> list[Path]:
    """Find one complete set of runner shard CSVs under downloaded outputs."""
    found = []
    for path in sorted(shard_root.rglob(f"{runner_id}-part*-of-*.csv")):
        match = NAME.fullmatch(path.name)
        if match and match.group("runner") == runner_id:
            found.append((int(match.group("part")), int(match.group("count")), path))
    if not found:
        raise FileNotFoundError(f"no {runner_id} shard CSVs found under {shard_root}")
    counts = {count for _, count, _ in found}
    if len(counts) != 1:
        raise ValueError(f"{runner_id} score shards disagree on shard count: {sorted(counts)}")
    count = counts.pop()
    parts = [part for part, _, _ in found]
    if sorted(parts) != list(range(count)):
        raise ValueError(f"{runner_id} missing or duplicate shard parts: {sorted(parts)}")
    return [path for _, _, path in sorted(found)]


def validate_provenance(score_path: Path, runner_id: str) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    assert_frozen_protocol()
    manifest = load_audio_manifest(args.audio_manifest, args.audio_root)
    paths = collect_shards(args.shard_root, args.runner)
    for path in paths:
        validate_provenance(path, args.runner)
    output = args.out_dir / f"{args.runner}.csv"
    result = merge_score_shards(paths, args.runner, manifest, output)
    result.update({
        "runner": args.runner,
        "shards": [str(path) for path in paths],
        "post_release_exploratory": True,
        "target_access": "No target MOS file or column was read during merge.",
    })
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
