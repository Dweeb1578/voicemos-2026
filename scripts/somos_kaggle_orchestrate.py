"""Build, but never launch, exact-revision Kaggle score kernels for SOMOS v2.

The generated notebooks score exactly one frozen public runner and one
deterministic audio shard.  They accept only Luna's audio-only materialization:
``somos_audio_manifest.csv`` plus ``audio/``.  Any MOS-list file mounted beside
it is a hard error.  This module has no Kaggle authentication or launch path.

Example (build only):
    python -m scripts.somos_kaggle_orchestrate --username my-kaggle-name --build
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path

from scripts.somos_scoring import FROZEN_PROTOCOL_SHA256, RUNNERS, assert_frozen_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
# Each model receives four deterministic shards by default.  This keeps heavy
# runners inside Kaggle's 12-hour ceiling while retaining one runner per kernel.
DEFAULT_SHARD_COUNT = 4
MAX_RUNTIME_MINUTES = 660
MACHINE_SHAPE = "NvidiaTeslaT4"
LOCAL_PAYLOAD = (
    "scripts/__init__.py",
    "scripts/somos_scoring.py",
    "scripts/somos_runner.py",
    "scripts/clip_cache.py",
    "scripts/submission.py",
    "scripts/zero_shot_dnsmos.py",
    "scripts/zero_shot_p808.py",
    "scripts/zero_shot_squim.py",
    "scripts/zero_shot_audiobox.py",
    "docs/mosaic_icassp_2027/third_corpus_protocol_frozen.md",
    "docs/mosaic_icassp_2027/third_corpus_protocol_frozen.sha256.json",
)

EXTERNAL_SOURCES = {
    "nisqa": ("https://github.com/gabrielmittag/NISQA.git", "fe84f0f252abec382b24367d5b22498a7ce34dbb"),
    "distillmos": ("https://github.com/microsoft/Distill-MOS.git", "98c0a156b5dabf2b5a8fe9cee92145cdc2a2dcdb"),
    "scoreq": ("https://github.com/alessandroragano/scoreq.git", "0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e"),
    "utmos": ("https://github.com/sarulab-speech/UTMOSv2.git", "cc2700db57bb83ee13dc31ebe1b868c254e15d09"),
    "sigmos": ("https://github.com/microsoft/SIG-Challenge.git", "bf4525153b6ed998f19d9e79ff1fd00f55dec42b"),
    "audiobox": ("https://github.com/facebookresearch/audiobox-aesthetics.git", "2618e9d451b456e9328b39495b5e6234678aa550"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def embedded_payload() -> tuple[dict[str, str], dict[str, str]]:
    """Return immutable embedded sources and their pre-launch hashes."""
    encoded: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for relative in LOCAL_PAYLOAD:
        content = (REPO_ROOT / relative).read_bytes()
        encoded[relative] = base64.b64encode(content).decode("ascii")
        hashes[relative] = sha256_bytes(content)
    return encoded, hashes


def _notebook(cells: list[tuple[str, str]]) -> dict:
    result = []
    for index, (kind, source) in enumerate(cells):
        cell = {"cell_type": kind, "metadata": {}, "source": source, "id": str(index)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        result.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": result,
    }


def _preamble(payload: dict[str, str], *, runner_id: str, part_index: int,
              shard_count: int) -> str:
    return f'''import base64, json, os, shutil, sys
from pathlib import Path

RUNNER_ID = {runner_id!r}
PART_INDEX = {part_index}
SHARD_COUNT = {shard_count}
BUNDLE = Path('/kaggle/working/somos_bundle')
PAYLOAD = {json.dumps(payload, sort_keys=True)}
for relative, encoded in PAYLOAD.items():
    destination = BUNDLE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))
sys.path.insert(0, str(BUNDLE))

# Scoring receives Luna's audio-only materialization, never the release target
# lists.  The manifest carries only exact filename IDs and official split names.
input_root = Path('/kaggle/input')
manifests = sorted(input_root.rglob('somos_audio_manifest.csv'))
assert len(manifests) == 1, f'expected one audio-only manifest, found {{manifests}}'
MANIFEST = manifests[0]
AUDIO_ROOT = MANIFEST.parent / 'audio'
assert AUDIO_ROOT.is_dir(), f'missing audio directory next to {{MANIFEST}}'
for prohibited in list(MANIFEST.parent.rglob('*_mos_list.txt')) + list(AUDIO_ROOT.rglob('*_mos_list.txt')):
    raise RuntimeError(f'target file mounted in scoring kernel: {{prohibited.name}}')

OUT = Path('/kaggle/working/somos_score_shards') / RUNNER_ID / f'part-{{PART_INDEX:02d}}'
OUT.mkdir(parents=True, exist_ok=True)
TAG = f'{{RUNNER_ID}}-part{{PART_INDEX:02d}}-of-{{SHARD_COUNT:02d}}'

# A failed or intentionally pre-empted run can resume only from an attached
# prediction-only output dataset.  More than one candidate is ambiguous.
prior = sorted(input_root.rglob(TAG + '.cache.csv'))
if prior:
    assert len(prior) == 1, f'ambiguous resume caches for {{TAG}}: {{prior}}'
    shutil.copy2(prior[0], OUT / prior[0].name)
    print('resuming from prediction-only cache', prior[0])
print('audio-only input', MANIFEST.parent, 'output', OUT)
'''


def _setup_cell(runner_id: str) -> tuple[str, list[str]]:
    """Return exact setup code and CLI arguments added before scoring."""
    header = '''import hashlib, os, subprocess, sys, urllib.request
from pathlib import Path

def run(*args):
    print('>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(value) for value in args], check=True)

def clone_exact(url, revision, destination):
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    run('git', 'clone', url, destination)
    run('git', '-C', destination, 'checkout', '--detach', revision)
    head = subprocess.check_output(['git', '-C', destination, 'rev-parse', 'HEAD'], text=True).strip()
    assert head == revision, (head, revision)
    return destination

run(sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'pandas', 'scipy', 'soundfile', 'tqdm')
ARTIFACTS = []
EXTRA = []
'''
    if runner_id in {"dnsmos", "p808"}:
        filename = next(iter(RUNNERS[runner_id]["weight_hashes"]))
        expected = RUNNERS[runner_id]["weight_hashes"][filename]
        url = RUNNERS[runner_id]["weight_urls"][filename]
        return header + f'''run(sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime', 'librosa')
DNS_ROOT = Path('/kaggle/working/somos_artifacts/dns')
DNS_ROOT.mkdir(parents=True, exist_ok=True)
MODEL = DNS_ROOT / {filename!r}
urllib.request.urlretrieve({url!r}, MODEL)
actual = hashlib.sha256(MODEL.read_bytes()).hexdigest()
assert actual == {expected!r}, (actual, {expected!r})
ARTIFACTS = [str(DNS_ROOT)]
EXTRA = ['--dns-model-root', str(DNS_ROOT)]
''', ["--dns-model-root", "<dns-root>"]
    if runner_id == "squim":
        return header + '''run(sys.executable, '-m', 'pip', 'install', '-q', 'torchaudio==2.11.0')
import importlib.metadata
assert importlib.metadata.version('torchaudio') == '2.11.0'
TORCH_HOME = Path('/kaggle/working/somos_artifacts/torch')
TORCH_HOME.mkdir(parents=True, exist_ok=True)
(TORCH_HOME / 'orchestration.txt').write_text('frozen SOMOS SQUIM artifact root\\n')
os.environ['TORCH_HOME'] = str(TORCH_HOME)
ARTIFACTS = [str(TORCH_HOME)]
''', []
    if runner_id == "universa":
        revision = RUNNERS[runner_id]["revision"]
        return header + f'''run(sys.executable, '-m', 'pip', 'install', '-q', 'urgent2026_sqa==0.2.2', 'huggingface_hub')
from huggingface_hub import hf_hub_download
HF_HOME = Path('/kaggle/working/somos_artifacts/hf')
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_HOME)
MODEL = hf_hub_download('vvwangvv/universa-ext_wavlm-base_5metric', 'model.pt', revision={revision!r}, cache_dir=str(HF_HOME))
CONFIG = hf_hub_download('vvwangvv/universa-ext_wavlm-base_5metric', 'config.yaml', revision={revision!r}, cache_dir=str(HF_HOME))
ARTIFACTS = [MODEL, CONFIG, str(HF_HOME)]
EXTRA = ['--universa-model', MODEL, '--universa-config', CONFIG]
''', ["--universa-model", "<model.pt>", "--universa-config", "<config.yaml>"]

    url, revision = EXTERNAL_SOURCES[runner_id]
    body = header + f'''VENDOR = clone_exact({url!r}, {revision!r}, '/kaggle/working/vendor-' + RUNNER_ID)
HF_HOME = Path('/kaggle/working/somos_artifacts/hf-' + RUNNER_ID)
HF_HOME.mkdir(parents=True, exist_ok=True)
(HF_HOME / 'orchestration.txt').write_text('frozen SOMOS artifact root\\n')
os.environ['HF_HOME'] = str(HF_HOME)
ARTIFACTS = [str(VENDOR), str(HF_HOME)]
'''
    if runner_id == "nisqa":
        body += '''requirements = VENDOR / 'requirements.txt'
if requirements.exists():
    run(sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements)
EXTRA = ['--vendor-root', str(VENDOR)]
'''
    elif runner_id == "sigmos":
        body += '''requirements = VENDOR / 'ICASSP2024' / 'sigmos' / 'requirements.txt'
if requirements.exists():
    run(sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements)
EXTRA = ['--vendor-root', str(VENDOR)]
'''
    elif runner_id == "distillmos":
        # Distill-MOS ships distillmos/weights/distill_mos_v7.pt inside the
        # pinned clone, so VENDOR already covers its weight hash.
        body += '''run(sys.executable, '-m', 'pip', 'install', '-q', VENDOR)
'''
    elif runner_id == "scoreq":
        # SCOREQ pulls its ONNX weights from Zenodo into a hardcoded
        # ~/.cache/scoreq path that no pinned clone covers.  Declaring the
        # download tree keeps the frozen weight-hashing rule satisfied.
        body += '''run(sys.executable, '-m', 'pip', 'install', '-q', VENDOR)
SCOREQ_CACHE = Path.home() / '.cache' / 'scoreq'
SCOREQ_CACHE.mkdir(parents=True, exist_ok=True)
ARTIFACTS.append(str(SCOREQ_CACHE))
'''
    elif runner_id == "utmos":
        # UTMOSv2 resolves fold weights from an unpinned branch into
        # $UTMOSV2_CHACHE (upstream spelling), so redirect that download into
        # the hashed artifact root.  The code commit is pinned; the weights
        # are not, so their hash is the only available provenance.
        body += '''run(sys.executable, '-m', 'pip', 'install', '-q', VENDOR)
UTMOS_CACHE = Path('/kaggle/working/somos_artifacts/utmosv2')
UTMOS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['UTMOSV2_CHACHE'] = str(UTMOS_CACHE)
ARTIFACTS.append(str(UTMOS_CACHE))
'''
    elif runner_id == "audiobox":
        model_revision = RUNNERS[runner_id]["model_revision"]
        body += f'''run(sys.executable, '-m', 'pip', 'install', '-q', VENDOR, 'huggingface_hub')
from huggingface_hub import hf_hub_download
CHECKPOINT = hf_hub_download('facebook/audiobox-aesthetics', 'checkpoint.pt', revision={model_revision!r}, cache_dir=str(HF_HOME))
ARTIFACTS.append(CHECKPOINT)
EXTRA = ['--audiobox-checkpoint', CHECKPOINT]
'''
    return body, []


def _run_cell(runner_id: str, part_index: int, shard_count: int, smoke_items: int) -> str:
    return f'''import subprocess, sys
command = [
    sys.executable, '-m', 'scripts.somos_runner',
    '--runner', {runner_id!r},
    '--audio-root', str(AUDIO_ROOT),
    '--audio-manifest', str(MANIFEST),
    '--out-dir', str(OUT),
    '--shard-index', {part_index!r},
    '--shard-count', {shard_count!r},
    '--max-runtime-minutes', {MAX_RUNTIME_MINUTES!r},
] + EXTRA + ['--artifact-path'] + ARTIFACTS
if {smoke_items!r}:
    command += ['--smoke-items', {smoke_items!r}]
print('>>', ' '.join(command), flush=True)
subprocess.run(command, check=True)
'''


def _validate_cell(runner_id: str, part_index: int, shard_count: int, smoke_items: int) -> str:
    if smoke_items:
        return f'''from pathlib import Path
smoke_path = OUT / ({runner_id!r} + '-part' + f'{part_index:02d}' + '-of-' + f'{shard_count:02d}' + '.smoke.provenance.json')
assert smoke_path.is_file(), smoke_path
print('smoke-only output', smoke_path)
'''
    return f'''import json
from pathlib import Path
import pandas as pd

tag = {runner_id!r} + '-part' + f'{part_index:02d}' + '-of-' + f'{shard_count:02d}'
score_path = OUT / (tag + '.csv')
provenance_path = OUT / (tag + '.provenance.json')
scores = pd.read_csv(score_path, dtype={{'sample_id': str, 'system_id': str}})
assert scores['sample_id'].str.endswith('.wav').all()
assert not scores.duplicated('sample_id').any()
assert not scores.isna().any().any()
provenance = json.loads(provenance_path.read_text())
assert provenance['protocol_sha256'] == {FROZEN_PROTOCOL_SHA256!r}
assert provenance['target_access'] == 'No target MOS file or column was read during scoring.'
print('complete shard', score_path, len(scores), 'rows')
'''


def _metadata(username: str, slug: str, *, gpu: bool, kernel_sources: list[str]) -> dict:
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "somos_score.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    if gpu:
        metadata["machine_shape"] = MACHINE_SHAPE
    return metadata


def build(*, username: str, audio_kernel: str,
          resume_kernel: str | None = None,
          shard_count: int = DEFAULT_SHARD_COUNT,
          smoke_items: int = 0,
          output_root: Path | None = None) -> dict:
    """Create score-kernel directories only.  No network or Kaggle API call."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if smoke_items < 0:
        raise ValueError("smoke_items must be non-negative")
    assert_frozen_protocol()
    output_root = output_root or REPO_ROOT / "notebooks" / "somos_kaggle"
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing kernel build directory: {output_root}")
    output_root.mkdir(parents=True)
    payload, source_hashes = embedded_payload()
    kernel_sources = [audio_kernel] + ([resume_kernel] if resume_kernel else [])
    written = []
    for runner_id, spec in RUNNERS.items():
        setup, declared_extra = _setup_cell(runner_id)
        for part_index in range(shard_count):
            slug = f"somos-{runner_id}-part{part_index:02d}-of-{shard_count:02d}"
            directory = output_root / runner_id / f"part-{part_index:02d}"
            directory.mkdir(parents=True)
            cells = [
                ("markdown", f"# Frozen SOMOS v2 scorer: {runner_id}, shard {part_index + 1}/{shard_count}\n\n"
                 "This is a post-release exploratory scoring kernel. It is private, uses only "
                 "the audio-only input dataset, and must not mount official target files."),
                ("code", _preamble(payload, runner_id=runner_id, part_index=part_index,
                                    shard_count=shard_count)),
                ("code", setup),
                ("code", _run_cell(runner_id, part_index, shard_count, smoke_items)),
                ("code", _validate_cell(runner_id, part_index, shard_count, smoke_items)),
            ]
            notebook = _notebook(cells)
            serialized = json.dumps(notebook, indent=2).encode("utf-8") + b"\n"
            (directory / "somos_score.ipynb").write_bytes(serialized)
            metadata = _metadata(username, slug, gpu=bool(spec["gpu"]),
                                 kernel_sources=kernel_sources)
            (directory / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            lock = {
                "schema_version": 1,
                "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "runner": {"id": runner_id, **spec},
                "shard": {"index": part_index, "count": shard_count},
                "max_runtime_minutes": MAX_RUNTIME_MINUTES,
                "smoke_items": smoke_items,
                "audio_input_contract": {
                    "kernel_source": audio_kernel,
                    "manifest": "somos_audio_manifest.csv",
                    "audio_directory": "audio/",
                    "target_access": "No MOS-list file or target column may be mounted or read.",
                },
                "frozen_dns_model_artifact": (
                    {"urls": spec.get("weight_urls", {}), "sha256": spec.get("weight_hashes", {})}
                    if runner_id in {"dnsmos", "p808"} else None
                ),
                "declared_runner_arguments": declared_extra,
                "embedded_source_sha256": source_hashes,
                "notebook_sha256": sha256_bytes(serialized),
                "resume": {
                    "cache_filename": f"{runner_id}-part{part_index:02d}-of-{shard_count:02d}.cache.csv",
                    "kernel_source": resume_kernel,
                    "input_rule": "Attach at most one prediction-only prior-output kernel source containing this cache.",
                },
            }
            (directory / "orchestration.lock.json").write_text(
                json.dumps(lock, indent=2) + "\n", encoding="utf-8")
            written.append(str(directory))
    readme = output_root / "README.txt"
    readme.write_text(
        "Build-only output. Nothing here launches Kaggle.\n"
        f"Frozen protocol SHA-256: {FROZEN_PROTOCOL_SHA256}\n"
        f"Kernels: {len(written)} ({len(RUNNERS)} runners x {shard_count} shards)\n"
        f"Smoke items per kernel: {smoke_items}\n"
        "Attach only the audio-only SOMOS ingestion-kernel output. DNSMOS/P.808 weights are downloaded and hash-verified.\n"
        "For a retry, provide at most one prediction-only prior-output kernel source containing the matching cache CSV.\n",
        encoding="utf-8")
    return {"output_root": str(output_root), "kernels": len(written), "source_hashes": source_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle account slug used only in metadata")
    parser.add_argument("--audio-kernel", help="private audio-only Kaggle kernel slug")
    parser.add_argument("--resume-kernel", help="optional prediction-only prior-output Kaggle kernel slug")
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--smoke-items", type=int, default=0,
                        help="build rate-estimation kernels that do not finalize score shards")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "notebooks" / "somos_kaggle")
    parser.add_argument("--build", action="store_true", help="build local kernel directories, never launch")
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("use --build. This tool intentionally has no Kaggle launch operation.")
    audio_kernel = args.audio_kernel or f"{args.username}/somos-v2-audio-only-ingestion"
    print(json.dumps(build(username=args.username, audio_kernel=audio_kernel,
                           resume_kernel=args.resume_kernel,
                           shard_count=args.shard_count, smoke_items=args.smoke_items,
                           output_root=args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
