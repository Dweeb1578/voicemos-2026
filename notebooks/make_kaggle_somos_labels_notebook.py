"""Build the certificate-gated Kaggle-only SOMOS v2 label kernel.

The generated private kernel first authenticates the sealed ten-runner
completion certificate. Only then may it download the pinned archive, extract
the three clean target lists, and emit the label manifest with provenance.
This module has no Kaggle API or launch path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

from scripts.somos_integrity import FROZEN_PROTOCOL_SHA256
from scripts.somos_scoring import assert_frozen_protocol


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = ROOT / "scripts" / "somos_v2_pipeline.py"
INTEGRITY_SOURCE = ROOT / "scripts" / "somos_integrity.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "somos_v2_labels"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _code(source: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "source": source.splitlines(True),
        "outputs": [], "execution_count": None,
    }


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def build_notebook(certificate_sha256: str) -> dict:
    if SHA256_RE.fullmatch(certificate_sha256) is None:
        raise ValueError("certificate_sha256 must be one lowercase SHA-256")
    encoded_pipeline = base64.b64encode(PIPELINE_SOURCE.read_bytes()).decode("ascii")
    encoded_integrity = base64.b64encode(INTEGRITY_SOURCE.read_bytes()).decode("ascii")
    bootstrap = f'''import base64, hashlib, os, pathlib, subprocess, sys
ROOT = pathlib.Path('/kaggle/temp/somos-labels-run')
(ROOT / 'scripts').mkdir(parents=True, exist_ok=True)
(ROOT / 'scripts' / '__init__.py').write_text('', encoding='utf-8')
(ROOT / 'scripts' / 'somos_v2_pipeline.py').write_bytes(base64.b64decode({encoded_pipeline!r}))
(ROOT / 'scripts' / 'somos_integrity.py').write_bytes(base64.b64decode({encoded_integrity!r}))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.somos_integrity import (
    EXPECTED_ROWS, EXPECTED_SPLIT_ROWS, FROZEN_PROTOCOL_SHA256,
    LABEL_TARGET_ACCESS, sha256_file, validate_completion_certificate,
    write_strict_json,
)

INPUT_ROOT = pathlib.Path('/kaggle/input')
certificates = sorted(INPUT_ROOT.rglob('somos_completion_certificate.json'))
if len(certificates) != 1:
    raise RuntimeError(f'expected one completion certificate, found {{certificates}}')
CERTIFICATE_PATH = certificates[0]
CERTIFICATE_FILE_SHA256 = sha256_file(CERTIFICATE_PATH)
CERTIFICATE_PAYLOAD = validate_completion_certificate(
    CERTIFICATE_PATH, {certificate_sha256!r})
CERTIFICATE_RECORD = __import__('json').loads(CERTIFICATE_PATH.read_text(encoding='utf-8'))

# The certificate kernel emits no scores. Any mounted CSV, WAV, or target list
# proves that the label job was built with the wrong input contract.
for pattern in ('*.csv', '*.wav', '*_mos_list.txt'):
    mounted = list(INPUT_ROOT.rglob(pattern))
    if mounted:
        raise RuntimeError(f'prohibited input mounted in label kernel: {{mounted}}')
print('validated completion certificate', CERTIFICATE_FILE_SHA256)
print('pipeline source:', (ROOT / 'scripts' / 'somos_v2_pipeline.py').stat().st_size, 'bytes')
'''

    retrieve = '''completed = subprocess.run([
    sys.executable, '-m', 'scripts.somos_v2_pipeline', 'labels',
    '--archive', '/kaggle/temp/somos.zip',
    '--provenance', '/kaggle/working/somos_v2_labels_download.json',
    '--archive-inventory', '/kaggle/working/somos_v2_labels_archive_inventory.json',
    '--clean-dir', '/kaggle/working/somos_v2_clean_labels',
    '--extract-inventory', '/kaggle/working/somos_v2_labels_extract_inventory.json',
    '--manifest', '/kaggle/working/somos_v2_labels.csv',
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
pathlib.Path('/kaggle/working/somos_v2_labels.log').write_text(
    completed.stdout or '', encoding='utf-8')
print(completed.stdout or '', flush=True)
if completed.returncode != 0:
    raise SystemExit(f'label retrieval failed with {completed.returncode}')
'''

    validate = '''import csv, json, re
WORKING = pathlib.Path('/kaggle/working')
manifest = WORKING / 'somos_v2_labels.csv'
with manifest.open(newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    columns = reader.fieldnames
expected = ['sample_id', 'source_group', 'system_id', 'split', 'mos']
assert rows and columns == expected, columns
assert len(rows) == EXPECTED_ROWS
assert len({row['sample_id'] for row in rows}) == len(rows)
assert all(re.fullmatch(r'.+_\\d{3}\\.wav', row['sample_id']) for row in rows)
assert all(1.0 <= float(row['mos']) <= 5.0 for row in rows)

for row in rows:
    match = re.fullmatch(r'(?P<source_group>.+)_(?P<system_id>\\d{3})\\.wav', row['sample_id'])
    assert match is not None
    assert row['source_group'] == match.group('source_group')
    assert row['system_id'] == match.group('system_id')
counts = {split: sum(row['split'] == split for row in rows) for split in EXPECTED_SPLIT_ROWS}
assert counts == EXPECTED_SPLIT_ROWS, counts

download_path = WORKING / 'somos_v2_labels_download.json'
archive_path = WORKING / 'somos_v2_labels_archive_inventory.json'
extract_path = WORKING / 'somos_v2_labels_extract_inventory.json'
download = json.loads(download_path.read_text(encoding='utf-8'))
archive = json.loads(archive_path.read_text(encoding='utf-8'))
extract = json.loads(extract_path.read_text(encoding='utf-8'))
assert download['actual_md5'] == download['expected_md5']
assert archive['archive_md5'] == download['actual_md5']
assert extract['archive_md5'] == download['actual_md5']
assert extract['labels_only'] is True
assert extract['label_file_count'] == 3
assert extract['clean_prefix'].replace('\\\\', '/').endswith('training_files/split1/clean')
assert not list(WORKING.rglob('*.wav'))

def evidence(path):
    return {'file': path.name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size}

label_provenance = {
    'schema_version': 1,
    'protocol_sha256': FROZEN_PROTOCOL_SHA256,
    'post_release_exploratory': True,
    'target_access': LABEL_TARGET_ACCESS,
    'completion_certificate': {
        'file': CERTIFICATE_PATH.name,
        'sha256': CERTIFICATE_FILE_SHA256,
        'payload_sha256': CERTIFICATE_RECORD['seal']['payload_sha256'],
    },
    'label_manifest': {
        'file': manifest.name,
        'sha256': sha256_file(manifest),
        'bytes': manifest.stat().st_size,
        'columns': expected,
        'rows': len(rows),
        'split_rows': counts,
    },
    'evidence': {
        'download': evidence(download_path),
        'archive_inventory': evidence(archive_path),
        'extraction_inventory': evidence(extract_path),
    },
}
write_strict_json(WORKING / 'somos_v2_labels.provenance.json', label_provenance)
print('label rows:', len(rows), 'splits:', counts)
print('archive MD5:', download['actual_md5'])
print('archive SHA-256:', download['local_sha256'])
print('no audio extracted, no score file mounted')
'''

    return {
        "cells": [
            _markdown("""# SOMOS v2 certificate-gated label retrieval

This private job validates the sealed ten-runner completion certificate before
any target download. It then extracts only the three clean MOS lists and writes
the label manifest with cryptographic provenance. It opens no audio and mounts
no prediction file.
"""),
            _code(bootstrap),
            _code(retrieve),
            _code(validate),
        ],
        "metadata": {
            "kernelspec": {
                "language": "python", "display_name": "Python 3", "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(
        *, username: str, certificate_kernel: str, certificate_sha256: str,
        slug: str = "somos-v2-label-retrieval", output_dir: Path | None = None,
) -> dict:
    """Create the label-kernel directory only. No remote call is made."""

    assert_frozen_protocol()
    if SHA256_RE.fullmatch(certificate_sha256) is None:
        raise ValueError("certificate_sha256 must be one lowercase SHA-256")
    if not certificate_kernel:
        raise ValueError("certificate_kernel is required")
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    output = output_dir / "kaggle_somos_labels.ipynb"
    notebook = build_notebook(certificate_sha256)
    serialized = json.dumps(notebook, indent=2).encode("utf-8") + b"\n"
    output.write_bytes(serialized)
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": output.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [certificate_kernel],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "certificate_kernel": certificate_kernel,
        "certificate_sha256": certificate_sha256,
        "pipeline_source_sha256": _sha256_bytes(PIPELINE_SOURCE.read_bytes()),
        "integrity_source_sha256": _sha256_bytes(INTEGRITY_SOURCE.read_bytes()),
        "notebook_sha256": _sha256_bytes(serialized),
        "target_access": (
            "The sealed completion certificate must validate before target retrieval."
        ),
    }
    (output_dir / "label.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8",
    )
    return {"output": str(output), "certificate_kernel": certificate_kernel}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--certificate-kernel", required=True)
    parser.add_argument("--certificate-sha256", required=True)
    parser.add_argument("--slug", default="somos-v2-label-retrieval")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("pass --build; this module never launches a kernel")
    result = build(
        username=args.username,
        certificate_kernel=args.certificate_kernel,
        certificate_sha256=args.certificate_sha256,
        slug=args.slug,
        output_dir=args.out,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
