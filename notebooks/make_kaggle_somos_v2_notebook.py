"""Generate the Kaggle-only SOMOS v2 retrieval and manifest notebook.

The generated notebook embeds the metadata pipeline source so it can run from
an otherwise empty Kaggle session. It streams the pinned 4 GB Zenodo archive to
``/kaggle/temp`` and writes only the clean split and provenance products to
``/kaggle/working``. This generator never downloads SOMOS or launches a kernel.
"""

from __future__ import annotations

import base64
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "somos_v2_pipeline.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "somos_v2_ingestion"
OUTPUT = OUTPUT_DIR / "kaggle_somos_v2_pipeline.ipynb"


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(True), "outputs": []}


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def build_notebook() -> dict:
    encoded = base64.b64encode(SCRIPT.read_bytes()).decode("ascii")
    bootstrap = f'''import base64, os, pathlib, subprocess, sys
ROOT = pathlib.Path('/kaggle/working/somos-v2-run')
(ROOT / 'scripts').mkdir(parents=True, exist_ok=True)
SOURCE = {encoded!r}
(ROOT / 'scripts' / 'somos_v2_pipeline.py').write_bytes(base64.b64decode(SOURCE))
(ROOT / 'scripts' / '__init__.py').write_text('', encoding='utf-8')
os.chdir(ROOT)
print('pipeline source:', (ROOT / 'scripts' / 'somos_v2_pipeline.py').stat().st_size, 'bytes')
'''
    prepare = '''subprocess.run([
    sys.executable, '-m', 'scripts.somos_v2_pipeline', 'prepare',
    '--archive', '/kaggle/temp/somos.zip',
    '--provenance', '/kaggle/working/somos_v2_download.json',
    '--archive-inventory', '/kaggle/working/somos_v2_archive_inventory.json',
    '--clean-dir', '/kaggle/temp/somos_v2_clean_labels',
    '--audio-dir', '/kaggle/working/somos_v2_scoring_input/audio',
    '--extract-inventory', '/kaggle/working/somos_v2_extract_inventory.json',
    '--manifest', '/kaggle/temp/somos_v2_clean_manifest.csv',
], check=True)
'''
    validate = '''import csv, json, re
label_manifest = pathlib.Path('/kaggle/temp/somos_v2_clean_manifest.csv')
audio_root = pathlib.Path('/kaggle/working/somos_v2_scoring_input/audio')
audio_manifest = pathlib.Path('/kaggle/working/somos_v2_scoring_input/somos_audio_manifest.csv')
with label_manifest.open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
required = ['sample_id', 'source_group', 'system_id', 'split', 'mos', 'audio_path']
assert rows and list(rows[0]) == required
assert len({row['sample_id'] for row in rows}) == len(rows)
assert {row['split'] for row in rows} == {'train', 'valid', 'test'}
assert all(re.fullmatch(r'.+_\\d{3}\\.wav', row['sample_id']) for row in rows)
audio_manifest.parent.mkdir(parents=True, exist_ok=True)
audio_columns = ['sample_id', 'source_group', 'system_id', 'split', 'relative_path']
with audio_manifest.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=audio_columns)
    writer.writeheader()
    for row in rows:
        resolved = pathlib.Path(row['audio_path']).resolve()
        relative = resolved.relative_to(audio_root.resolve()).as_posix()
        writer.writerow({
            'sample_id': row['sample_id'],
            'source_group': row['source_group'],
            'system_id': row['system_id'],
            'split': row['split'],
            'relative_path': relative,
        })
download = json.load(open('/kaggle/working/somos_v2_download.json', encoding='utf-8'))
inventory = json.load(open('/kaggle/working/somos_v2_extract_inventory.json', encoding='utf-8'))
archive_inventory = json.load(open('/kaggle/working/somos_v2_archive_inventory.json', encoding='utf-8'))
assert download['actual_md5'] == download['expected_md5']
assert archive_inventory['md5_matches_expected']
assert inventory['archive_md5'] == download['actual_md5']
assert inventory['clean_schema']['manifest_columns'] == required
assert not list(pathlib.Path('/kaggle/working').rglob('*_mos_list.txt'))
assert not list(audio_manifest.parent.rglob('*_mos_list.txt'))
assert inventory['audio_output_dir'] == str(audio_root)
assert not label_manifest.exists() or str(label_manifest).startswith('/kaggle/temp/')
with audio_manifest.open(newline='', encoding='utf-8') as handle:
    audio_rows = list(csv.DictReader(handle))
assert audio_rows and list(audio_rows[0]) == audio_columns
assert not ({'mos', 'target', 'label'} & set(audio_rows[0]))
assert all((audio_root / row['relative_path']).is_file() for row in audio_rows)
print('manifest rows:', len(rows), 'splits:', {s: sum(r['split'] == s for r in rows) for s in ('train', 'valid', 'test')})
print('archive MD5:', download['actual_md5'])
print('archive SHA-256:', download['local_sha256'])
print('clean extracted files:', inventory['selected_file_count'])
print('audio-only scoring artifact:', audio_manifest.parent)
print('target labels remain under /kaggle/temp and are not saved in this kernel output')
'''
    return {
        "cells": [
            _markdown("""# SOMOS v2 prospective retrieval and manifest

This notebook runs the frozen metadata pipeline only. It downloads the exact Zenodo v2 archive to `/kaggle/temp`, validates the published MD5, records the runtime SHA-256 and full ZIP inventory, extracts only the WAVs referenced by `training_files/split1/clean`, and writes a label-free audio manifest under `/kaggle/working/somos_v2_scoring_input`. Target lists and the temporary label manifest stay under `/kaggle/temp`, so they are excluded from the saved kernel output. No model scoring is performed here.

Keep Internet enabled and do not modify the dataset URL, archive hash, split, extraction prefix, or manifest schema after retrieval.
"""),
            _code(bootstrap),
            _code(prepare),
            _code(validate),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="vrishabnair")
    parser.add_argument("--slug", default="somos-v2-audio-only-ingestion")
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": "SOMOS v2 audio-only prospective ingestion",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
