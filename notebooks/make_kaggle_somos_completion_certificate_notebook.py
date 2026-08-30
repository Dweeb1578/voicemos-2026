"""Build the prediction-free SOMOS completion-certificate Kaggle kernel.

The generated private CPU kernel mounts exactly the ten merged prediction
outputs, authenticates their merge provenance, and emits one sealed JSON
certificate. It contains no target retrieval, scoring, Kaggle API, or launch
path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from scripts.somos_integrity import FROZEN_PROTOCOL_SHA256, RUNNER_OUTPUTS
from scripts.somos_kaggle_orchestrate import _notebook
from scripts.somos_scoring import assert_frozen_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRITY_SOURCE = REPO_ROOT / "scripts" / "somos_integrity.py"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "somos_v2_completion_certificate"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _preamble(encoded_integrity: str) -> str:
    return f'''import base64, pathlib, sys
ROOT = pathlib.Path('/kaggle/temp/somos-completion-run')
(ROOT / 'scripts').mkdir(parents=True, exist_ok=True)
(ROOT / 'scripts' / '__init__.py').write_text('', encoding='utf-8')
(ROOT / 'scripts' / 'somos_integrity.py').write_bytes(base64.b64decode({encoded_integrity!r}))
sys.path.insert(0, str(ROOT))

from scripts.somos_integrity import (
    CERTIFICATE_TARGET_ACCESS, EXPECTED_ROWS, EXPECTED_SPLIT_ROWS,
    FROZEN_PROTOCOL_SHA256, ID_RE, RUNNER_OUTPUTS, seal_completion_payload,
    sha256_file, validate_completion_certificate, validate_merge_provenance,
    write_strict_json,
)
'''


def _certificate_cell() -> str:
    return '''import hashlib, json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

INPUT_ROOT = pathlib.Path('/kaggle/input')
OUTPUT = pathlib.Path('/kaggle/working/somos_completion_certificate.json')

for prohibited in list(INPUT_ROOT.rglob('*_mos_list.txt')) + list(INPUT_ROOT.rglob('somos_v2_labels.csv')):
    raise RuntimeError(f'target file mounted in completion kernel: {prohibited}')

expected_csv_names = {runner + '.csv' for runner in RUNNER_OUTPUTS}
csv_paths = sorted(INPUT_ROOT.rglob('*.csv'))
if {path.name for path in csv_paths} != expected_csv_names or len(csv_paths) != len(expected_csv_names):
    raise RuntimeError(f'expected exactly ten merged runner CSVs, found {[str(path) for path in csv_paths]}')

reference_metadata = None
merge_artifacts = []
for runner, outputs in RUNNER_OUTPUTS.items():
    csv_candidates = [path for path in csv_paths if path.name == runner + '.csv']
    if len(csv_candidates) != 1:
        raise RuntimeError(f'{runner}: expected one merged CSV, found {csv_candidates}')
    csv_path = csv_candidates[0]
    provenance_candidates = list(csv_path.parent.rglob(runner + '.merge.provenance.json'))
    if len(provenance_candidates) != 1:
        raise RuntimeError(f'{runner}: expected one merge provenance sidecar, found {provenance_candidates}')
    provenance_path = provenance_candidates[0]
    validate_merge_provenance(provenance_path, csv_path, runner, outputs)

    expected_columns = ['sample_id', 'source_group', 'system_id', 'split', *outputs]
    frame = pd.read_csv(csv_path, dtype={'sample_id': str, 'source_group': str, 'system_id': str})
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f'{runner}: merged CSV columns mismatch')
    if len(frame) != EXPECTED_ROWS or frame['sample_id'].duplicated().any():
        raise ValueError(f'{runner}: merged CSV row or ID count mismatch')
    if frame['split'].value_counts().to_dict() != EXPECTED_SPLIT_ROWS:
        raise ValueError(f'{runner}: merged CSV split counts mismatch')
    values = frame[list(outputs)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{runner}: merged CSV contains non-finite scores')

    derived = frame['sample_id'].str.extract(ID_RE)
    if derived.isna().any().any():
        raise ValueError(f'{runner}: sample ID violates frozen schema')
    if not derived['source_group'].equals(frame['source_group']):
        raise ValueError(f'{runner}: source_group does not match sample ID')
    if not derived['system_id'].equals(frame['system_id']):
        raise ValueError(f'{runner}: system_id does not match sample ID')

    metadata = frame[['sample_id', 'source_group', 'system_id', 'split']].sort_values(
        'sample_id', kind='stable').reset_index(drop=True)
    if reference_metadata is None:
        reference_metadata = metadata
    elif not metadata.equals(reference_metadata):
        raise ValueError(f'{runner}: canonical metadata differs from the other merges')

    merge_artifacts.append({
        'runner': runner,
        'outputs': list(outputs),
        'rows': len(frame),
        'merged_csv': {
            'file': csv_path.name,
            'sha256': sha256_file(csv_path),
            'bytes': csv_path.stat().st_size,
        },
        'merge_provenance': {
            'file': provenance_path.name,
            'sha256': sha256_file(provenance_path),
            'bytes': provenance_path.stat().st_size,
        },
    })

assert reference_metadata is not None
sample_ids = reference_metadata['sample_id'].tolist()
sample_id_sha256 = hashlib.sha256(chr(10).join(sample_ids).encode('utf-8')).hexdigest()
metadata_bytes = reference_metadata.to_csv(index=False, lineterminator='\\n').encode('utf-8')
payload = {
    'complete': True,
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'protocol_sha256': FROZEN_PROTOCOL_SHA256,
    'post_release_exploratory': True,
    'expected_rows': EXPECTED_ROWS,
    'split_rows': EXPECTED_SPLIT_ROWS,
    'runner_bank': {runner: list(outputs) for runner, outputs in RUNNER_OUTPUTS.items()},
    'sample_id_sha256': sample_id_sha256,
    'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
    'merge_artifacts': merge_artifacts,
    'target_access': CERTIFICATE_TARGET_ACCESS,
}
certificate = seal_completion_payload(payload)
write_strict_json(OUTPUT, certificate)
validate_completion_certificate(OUTPUT, sha256_file(OUTPUT))

working_files = sorted(path.name for path in OUTPUT.parent.iterdir() if path.is_file())
if working_files != [OUTPUT.name]:
    raise RuntimeError(f'completion kernel emitted unexpected files: {working_files}')
print(json.dumps({
    'certificate': OUTPUT.name,
    'sha256': sha256_file(OUTPUT),
    'payload_sha256': certificate['seal']['payload_sha256'],
    'runners': len(merge_artifacts),
    'rows': EXPECTED_ROWS,
}, indent=2, allow_nan=False))
'''


def build_notebook() -> dict:
    encoded = base64.b64encode(INTEGRITY_SOURCE.read_bytes()).decode("ascii")
    cells = [
        ("markdown", "# Frozen SOMOS v2 completion certificate\n\n"
         "Prediction-only integrity gate. It mounts all ten merge outputs, reads no "
         "target, emits no score, and saves only one sealed JSON certificate."),
        ("code", _preamble(encoded)),
        ("code", _certificate_cell()),
    ]
    return _notebook(cells)


def build(
        *, username: str, merge_kernels: dict[str, str],
        output_root: Path | None = None,
) -> dict:
    """Create the certificate-kernel directory only. No remote call is made."""

    assert_frozen_protocol()
    if set(merge_kernels) != set(RUNNER_OUTPUTS):
        raise ValueError("merge-kernel map must contain exactly the frozen ten runners")
    if len(set(merge_kernels.values())) != len(RUNNER_OUTPUTS):
        raise ValueError("merge-kernel slugs must be unique")
    output_root = output_root or DEFAULT_OUTPUT
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_root}")
    output_root.mkdir(parents=True)

    notebook = build_notebook()
    serialized = json.dumps(notebook, indent=2).encode("utf-8") + b"\n"
    notebook_path = output_root / "somos_completion_certificate.ipynb"
    notebook_path.write_bytes(serialized)
    slug = "somos-v2-completion-certificate"
    kernel_sources = [merge_kernels[runner] for runner in RUNNER_OUTPUTS]
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    (output_root / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "runner_bank": {runner: list(outputs) for runner, outputs in RUNNER_OUTPUTS.items()},
        "merge_kernels": merge_kernels,
        "integrity_source_sha256": _sha256_bytes(INTEGRITY_SOURCE.read_bytes()),
        "notebook_sha256": _sha256_bytes(serialized),
        "output_contract": ["somos_completion_certificate.json"],
        "target_access": "No target MOS file or column may be mounted or read.",
    }
    (output_root / "completion.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8",
    )
    return {"output_root": str(output_root), "kernel_sources": kernel_sources}


def _parse_overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        runner, separator, slug = value.partition("=")
        if not separator or runner not in RUNNER_OUTPUTS or not slug:
            raise ValueError(f"invalid --merge-kernel value: {value!r}")
        if runner in result:
            raise ValueError(f"duplicate --merge-kernel runner: {runner}")
        result[runner] = slug
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument(
        "--merge-kernel", action="append", default=[], metavar="RUNNER=SLUG",
        help="override one default private merge-kernel source",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("pass --build; this module never launches a kernel")
    merge_kernels = {
        runner: f"{args.username}/somos-merge-{runner}" for runner in RUNNER_OUTPUTS
    }
    merge_kernels.update(_parse_overrides(args.merge_kernel))
    print(json.dumps(build(
        username=args.username, merge_kernels=merge_kernels, output_root=args.out,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
