"""Build, but never launch, a Kaggle kernel that merges one runner's shards.

The canonical merge validates every shard against the audio-only manifest, and
that validation requires the 20,100 WAVs to exist.  They live only in the
private ingestion kernel's output, so the merge runs on Kaggle beside them
rather than pulling several gigabytes of audio to a workstation.

The kernel mounts the ingestion output and the four prediction-only score
kernels for one runner.  It never mounts a MOS list, and it emits the canonical
``<runner>.csv`` plus a merge record small enough to download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.somos_kaggle_orchestrate import (
    _metadata,
    _notebook,
    embedded_payload,
)
from scripts.somos_scoring import (
    DEFAULT_SHARD_COUNT,
    FROZEN_PROTOCOL_SHA256,
    RUNNERS,
    assert_frozen_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _preamble(payload: dict[str, str], runner_id: str, shard_count: int) -> str:
    return f'''import base64, json, os, sys
from pathlib import Path

RUNNER_ID = {runner_id!r}
SHARD_COUNT = {shard_count}
BUNDLE = Path('/kaggle/working/somos_bundle')
PAYLOAD = {json.dumps(payload, sort_keys=True)}
for relative, encoded in PAYLOAD.items():
    destination = BUNDLE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))
sys.path.insert(0, str(BUNDLE))

input_root = Path('/kaggle/input')
manifests = sorted(input_root.rglob('somos_audio_manifest.csv'))
assert len(manifests) == 1, f'expected one audio-only manifest, found {{manifests}}'
MANIFEST = manifests[0]
AUDIO_ROOT = MANIFEST.parent / 'audio'
assert AUDIO_ROOT.is_dir(), f'missing audio directory next to {{MANIFEST}}'

# The merge reads predictions and audio filenames only.  A mounted target list
# would break the prediction-only boundary the whole pipeline depends on.
for prohibited in input_root.rglob('*_mos_list.txt'):
    raise RuntimeError(f'target file mounted in merge kernel: {{prohibited}}')

SHARD_ROOT = input_root
OUT = Path('/kaggle/working/somos_scores')
OUT.mkdir(parents=True, exist_ok=True)
print('merging', RUNNER_ID, 'from', SHARD_ROOT, 'into', OUT)
'''


def _merge_cell(runner_id: str) -> str:
    return f'''import subprocess, sys
command = [
    sys.executable, '-m', 'scripts.somos_merge_shards',
    '--runner', {runner_id!r},
    '--audio-root', str(AUDIO_ROOT),
    '--audio-manifest', str(MANIFEST),
    '--shard-root', str(SHARD_ROOT),
    '--out-dir', str(OUT),
]
command = [str(value) for value in command]
print('>>', ' '.join(command), flush=True)
MERGE_LOG = OUT / (RUNNER_ID + '.merge.log')
with MERGE_LOG.open('w', encoding='utf-8') as handle:
    completed = subprocess.run(
        command, cwd=str(BUNDLE), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors='replace',
    )
    handle.write(completed.stdout or '')
print(completed.stdout or '', flush=True)
if completed.returncode != 0:
    raise SystemExit(f'merge failed with {{completed.returncode}}; see {{MERGE_LOG}}')
'''


def _validate_cell(runner_id: str) -> str:
    outputs = list(RUNNERS[runner_id]["outputs"])
    return f'''import pandas as pd
merged = OUT / ({runner_id!r} + '.csv')
frame = pd.read_csv(merged, dtype={{'sample_id': str, 'system_id': str}})
expected = {outputs!r}
assert set(expected) <= set(frame.columns), sorted(frame.columns)
assert not frame['sample_id'].duplicated().any()
assert frame['sample_id'].str.endswith('.wav').all()
assert frame[expected].notna().all().all()
assert set(frame['split']) == {{'train', 'valid', 'test'}}
assert 'mos' not in frame.columns, 'a target column reached the merged scores'
print('merged rows', len(frame), 'outputs', expected)
print(frame['split'].value_counts().to_dict())
'''


def build(*, username: str, audio_kernel: str, shard_kernels: dict[str, list[str]],
          shard_count: int = DEFAULT_SHARD_COUNT,
          output_root: Path | None = None) -> dict:
    """Create one merge-kernel directory per runner.  No Kaggle API call."""
    assert_frozen_protocol()
    output_root = output_root or REPO_ROOT / "notebooks" / "somos_kaggle_merge"
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_root}")
    output_root.mkdir(parents=True)
    payload, source_hashes = embedded_payload()

    written = []
    for runner_id in RUNNERS:
        sources = shard_kernels[runner_id]
        if len(sources) != shard_count:
            raise ValueError(
                f"{runner_id}: expected {shard_count} shard kernels, got {len(sources)}")
        slug = f"somos-merge-{runner_id}"
        directory = output_root / runner_id
        directory.mkdir(parents=True)
        cells = [
            ("markdown", f"# Frozen SOMOS v2 merge: {runner_id}\n\n"
             "Post-release exploratory merge. Private, prediction-only, and it "
             "fails if any MOS list is mounted."),
            ("code", _preamble(payload, runner_id, shard_count)),
            ("code", _merge_cell(runner_id)),
            ("code", _validate_cell(runner_id)),
        ]
        notebook = _notebook(cells)
        (directory / "somos_merge.ipynb").write_bytes(
            json.dumps(notebook, indent=2).encode("utf-8") + b"\n")
        metadata = _metadata(username, slug, gpu=False,
                             kernel_sources=[audio_kernel, *sources])
        # _metadata is shared with the scoring builder and names its notebook.
        metadata["code_file"] = "somos_merge.ipynb"
        (directory / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (directory / "merge.lock.json").write_text(json.dumps({
            "schema_version": 1,
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "runner": runner_id,
            "outputs": list(RUNNERS[runner_id]["outputs"]),
            "shard_count": shard_count,
            "shard_kernels": sources,
            "audio_kernel": audio_kernel,
            "embedded_source_sha256": source_hashes,
            "target_access": "No MOS-list file or target column may be mounted or read.",
        }, indent=2) + "\n", encoding="utf-8")
        written.append(str(directory))
    return {"output_root": str(output_root), "kernels": len(written)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--audio-kernel", required=True)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("pass --build; this module never launches a kernel")
    shard_kernels = {
        runner_id: [
            f"{args.username}/somos-{runner_id}-part{index:02d}-of-{args.shard_count:02d}"
            for index in range(args.shard_count)
        ]
        for runner_id in RUNNERS
    }
    print(json.dumps(build(
        username=args.username, audio_kernel=args.audio_kernel,
        shard_kernels=shard_kernels, shard_count=args.shard_count,
        output_root=args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
