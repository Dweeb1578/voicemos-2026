"""Synthetic metadata tests for the offline-safe SOMOS v2 pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path

import pytest

from scripts.somos_v2_pipeline import (
    ARCHIVE_URL,
    DOI,
    EXPECTED_MD5,
    MANIFEST_COLUMNS,
    MANIFEST_SCHEMA,
    archive_inventory,
    build_manifest,
    download_archive,
    extract_clean,
    resolve_clean_prefix,
)
from notebooks.make_kaggle_somos_v2_notebook import build_notebook


PREFIX = "somos/training_files/split1/clean"


@pytest.fixture
def work_dir():
    """Use the repository workspace, whose ACLs are available in this task."""
    path = Path(__file__).resolve().parents[1] / f".somos-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _make_archive(
    path: Path,
    bad_id: bool = False,
    flat_audio: bool = False,
    nested_audio: bool = False,
) -> Path:
    rows = {
        "train": [("booksent_0001_000", "3.0")],
        "valid": [("booksent_0001_001", "4.0")],
        "test": [("booksent_0001_000" if bad_id else "booksent_0002_002", "2.0")],
    }
    nested = io.BytesIO()
    nested_writer = zipfile.ZipFile(nested, "w", compression=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split, items in rows.items():
            archive.writestr(f"{PREFIX}/{split}_mos_list.txt", "utteranceId,mean\n" + "\n".join(
                f"{item_id},{mos}" for item_id, mos in items
            ) + "\n")
            for item_id, _ in items:
                if nested_audio:
                    nested_writer.writestr(f"audios/{item_id}.wav", b"RIFFsynthetic")
                    continue
                audio_dir = "audios" if flat_audio else f"{split.upper()}SET"
                archive.writestr(f"{PREFIX}/{audio_dir}/{item_id}.wav", b"RIFFsynthetic")
        if nested_audio:
            nested_writer.close()
            archive.writestr("somos/audios.zip", nested.getvalue())
        # A non-clean member proves inventory and extraction do not silently
        # claim that the complete archive was extracted.
        archive.writestr("somos/raw/full_listener_scores.txt", "not selected\n")
    return path


def test_constants_pin_somos_v2_release():
    assert DOI == "10.5281/zenodo.7378801"
    assert "7378801/files/somos.zip" in ARCHIVE_URL


def test_generated_notebook_saves_audio_only_scoring_contract():
    notebook = build_notebook()
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "/kaggle/working/somos_v2_scoring_input/audio" in source
    assert "/kaggle/working/somos_v2_scoring_input/somos_audio_manifest.csv" in source
    assert "--clean-dir', '/kaggle/temp/somos_v2_clean_labels'" in source
    assert "--manifest', '/kaggle/temp/somos_v2_clean_manifest.csv'" in source
    assert "not list(pathlib.Path('/kaggle/working').rglob('*_mos_list.txt'))" in source
    assert "audio_columns = ['sample_id', 'source_group', 'system_id', 'split', 'relative_path']" in source


def test_generated_notebook_id_pattern_accepts_a_real_sample_id():
    notebook = build_notebook()
    source = chr(10).join(
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
    )
    match = re.search(r"re[.]fullmatch[(]r'([^']+)'", source)
    assert match, 'validate cell no longer pins a sample_id pattern'
    pattern = match.group(1)
    assert re.fullmatch(pattern, 'LJ050-0029_017.wav')
    assert re.fullmatch(pattern, 'LJ001-0001_000.wav')
    assert not re.fullmatch(pattern, 'LJ050-0029.wav')
    assert not re.fullmatch(pattern, 'LJ050-0029_17.wav')


def test_ingestion_kernel_metadata_is_private_cpu_and_online(work_dir):
    import notebooks.make_kaggle_somos_v2_notebook as generator

    original_dir, original_output = generator.OUTPUT_DIR, generator.OUTPUT
    try:
        generator.OUTPUT_DIR = work_dir / "kernel"
        generator.OUTPUT = generator.OUTPUT_DIR / "kaggle_somos_v2_pipeline.ipynb"
        assert generator.main(["--username", "account", "--slug", "audio-only"]) == 0
        metadata = json.loads(
            (generator.OUTPUT_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == "account/audio-only"
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is False
        assert metadata["enable_internet"] is True
        assert metadata["dataset_sources"] == []
        assert metadata["kernel_sources"] == []
    finally:
        generator.OUTPUT_DIR, generator.OUTPUT = original_dir, original_output
    assert EXPECTED_MD5 == "bdfde4cae256549dfab05d713136e4af"


def test_download_records_runtime_hashes_without_network(work_dir):
    source = work_dir / "source.bin"
    source.write_bytes(b"synthetic archive bytes")
    destination = work_dir / "download.bin"
    provenance = work_dir / "download.json"
    expected = hashlib.md5(source.read_bytes()).hexdigest()
    record = download_archive(
        destination,
        provenance,
        url=source.as_uri(),
        expected_md5=expected,
        chunk_size=5,
    )
    assert record["actual_md5"] == expected
    assert record["local_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(provenance.read_text(encoding="utf-8"))["bytes"] == len(source.read_bytes())


def test_resolve_clean_prefix_requires_all_three_lists():
    names = [f"{PREFIX}/{split}_mos_list.txt" for split in ("train", "valid", "test")]
    assert resolve_clean_prefix(names) == PREFIX
    with pytest.raises(ValueError, match="expected exactly one"):
        resolve_clean_prefix(names[:-1])


def test_clean_lists_accept_the_released_header_and_bare_ids(work_dir):
    from scripts.somos_v2_pipeline import _read_manifest_inputs

    clean = work_dir / "clean"
    clean.mkdir()
    (clean / "train_mos_list.txt").write_text(
        "utteranceId,mean\nbooksent_0001_000,3.5\n", encoding="utf-8")
    (clean / "valid_mos_list.txt").write_text(
        "utteranceId,mean\nbooksent_0001_001.wav,4.5\n", encoding="utf-8")
    (clean / "test_mos_list.txt").write_text(
        "utteranceId,mean\nbooksent_0002_002,2.5\n", encoding="utf-8")
    entries = _read_manifest_inputs(clean)
    assert [sample_id for _split, sample_id, _mos in entries] == [
        "booksent_0001_000.wav", "booksent_0001_001.wav", "booksent_0002_002.wav",
    ]


def test_a_non_numeric_score_after_the_header_still_fails(work_dir):
    from scripts.somos_v2_pipeline import _read_manifest_inputs

    clean = work_dir / "clean"
    clean.mkdir()
    for split in ("train", "valid", "test"):
        (clean / f"{split}_mos_list.txt").write_text(
            "utteranceId,mean\nbooksent_0001_000,3.5\nbooksent_0001_001,broken\n",
            encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric MOS"):
        _read_manifest_inputs(clean)


def test_generated_label_kernel_is_private_and_audio_free():
    from notebooks.make_kaggle_somos_labels_notebook import build_notebook as build_labels

    certificate_hash = "a" * 64
    notebook = build_labels(certificate_hash)
    source = chr(10).join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "generated-label-kernel", "exec")

    # Certificate validation is in the cell before the first target download.
    assert source.index("validate_completion_certificate") < source.index(
        "'scripts.somos_v2_pipeline', 'labels'"
    )
    assert certificate_hash in source

    # It must call the labels subcommand, never prepare, which extracts audio.
    assert "'labels'," in source
    assert "'prepare'" not in source
    assert "--manifest', '/kaggle/working/somos_v2_labels.csv'" in source

    # The boundary assertions have to live in the kernel, not just in intent.
    assert "expected one completion certificate" in source
    assert "prohibited input mounted in label kernel" in source
    assert "rglob('*.wav')" in source
    assert "extract['labels_only'] is True" in source
    assert "somos_v2_labels.provenance.json" in source

    # The emitted sample_id pattern must match a real utterance ID, the same
    # doubled-brace trap that once shipped a dead regex in the ingestion kernel.
    match = re.search(r"re[.]fullmatch[(]r'([^']+)'", source)
    assert match, "label kernel no longer pins a sample_id pattern"
    assert re.fullmatch(match.group(1), "LJ050-0029_017.wav")
    assert not re.fullmatch(match.group(1), "LJ050-0029.wav")


def test_label_kernel_build_mounts_only_the_pinned_certificate(work_dir):
    from notebooks.make_kaggle_somos_labels_notebook import build

    output = work_dir / "sealed-label-kernel"
    certificate_kernel = "unit-test/somos-v2-completion-certificate"
    certificate_hash = "b" * 64
    build(
        username="unit-test",
        certificate_kernel=certificate_kernel,
        certificate_sha256=certificate_hash,
        output_dir=output,
    )
    metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="utf-8"))
    lock = json.loads((output / "label.lock.json").read_text(encoding="utf-8"))
    assert metadata["kernel_sources"] == [certificate_kernel]
    assert metadata["dataset_sources"] == []
    assert metadata["competition_sources"] == []
    assert metadata["model_sources"] == []
    assert lock["certificate_kernel"] == certificate_kernel
    assert lock["certificate_sha256"] == certificate_hash


def test_completion_certificate_kernel_mounts_exact_bank_and_emits_only_json(work_dir):
    from notebooks.make_kaggle_somos_completion_certificate_notebook import build
    from scripts.somos_integrity import RUNNER_OUTPUTS

    merge_kernels = {
        runner: f"unit-test/somos-merge-{runner}" for runner in RUNNER_OUTPUTS
    }
    output = work_dir / "completion-kernel"
    build(username="unit-test", merge_kernels=merge_kernels, output_root=output)

    metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == list(merge_kernels.values())
    assert len(metadata["kernel_sources"]) == 10
    notebook = json.loads(
        (output / "somos_completion_certificate.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "generated-completion-kernel", "exec")
    assert "validate_merge_provenance" in source
    assert "expected exactly ten merged runner CSVs" in source
    assert "working_files != [OUTPUT.name]" in source
    assert "somos_completion_certificate.json" in source
    assert "somos_v2_labels.csv" in source


def test_labels_only_extraction_takes_no_audio(work_dir):
    # The label-retrieval job runs after scoring and is the only step allowed
    # to materialize MOS values.  It must not extract a single WAV.
    from scripts.somos_v2_pipeline import LABEL_MANIFEST_COLUMNS, extract_clean

    archive = _make_archive(work_dir / "somos.zip", nested_audio=True)
    clean = work_dir / "labels"
    inventory = work_dir / "extract.json"

    record = extract_clean(archive, clean, inventory, labels_only=True)

    assert record["labels_only"] is True
    assert record["label_file_count"] == 3
    assert record["referenced_sample_count"] == 3
    extracted = sorted(path.name for path in clean.rglob("*") if path.is_file())
    assert extracted == ["test_mos_list.txt", "train_mos_list.txt", "valid_mos_list.txt"]
    assert not list(clean.rglob("*.wav"))

    rows = build_manifest(clean, work_dir / "labels.csv", require_audio=False)
    assert len(rows) == 3
    assert set(rows[0]) == set(LABEL_MANIFEST_COLUMNS)
    assert "audio_path" not in rows[0]
    assert all(1.0 <= row["mos"] <= 5.0 for row in rows)

    header = (work_dir / "labels.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(LABEL_MANIFEST_COLUMNS)


def test_resolve_clean_prefix_ignores_the_full_partition():
    # The real v2 archive ships split1/clean and split1/full side by side, both
    # carrying train/valid/test MOS lists.
    names = []
    for partition in ("clean", "full"):
        base = f"training_files/split1/{partition}"
        names += [f"{base}/{split}_mos_list.txt" for split in ("train", "valid", "test")]
        names += [f"{base}/TRAINSET", f"{base}/VALIDSET", f"{base}/TESTSET"]
    names += ["audios.zip", "readme.txt", "raw_scores_with_metadata/raw_scores.tsv"]
    assert resolve_clean_prefix(names) == "training_files/split1/clean"


def test_resolve_clean_prefix_rejects_an_archive_without_a_clean_partition():
    base = "training_files/split1/full"
    names = [f"{base}/{split}_mos_list.txt" for split in ("train", "valid", "test")]
    with pytest.raises(ValueError, match="training_files/split1/clean"):
        resolve_clean_prefix(names)


def test_inventory_records_archive_hashes_and_clean_members(work_dir):
    archive = _make_archive(work_dir / "somos.zip")
    record = archive_inventory(archive)
    assert record["doi"] == DOI
    assert record["archive_md5"] == hashlib.md5(archive.read_bytes()).hexdigest()
    assert record["expected_md5"] == EXPECTED_MD5
    assert record["md5_matches_expected"] is False
    assert record["local_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert record["member_count"] == 7
    assert record["clean_member_count"] == 6
    assert all("crc32" in member for member in record["members"])


def test_extract_and_build_manifest_keep_only_clean_split(work_dir):
    archive = _make_archive(work_dir / "somos.zip")
    clean = work_dir / "clean"
    inventory = extract_clean(archive, clean)
    assert inventory["clean_prefix"] == PREFIX
    assert inventory["selected_file_count"] == 6
    assert not list(clean.rglob("full_listener_scores.txt"))

    output = work_dir / "manifest.csv"
    rows = build_manifest(clean, output)
    assert len(rows) == 3
    assert {row["split"] for row in rows} == {"train", "valid", "test"}
    assert rows[0]["source_group"] == "booksent_0001"
    assert rows[0]["system_id"] == "000"
    assert list(output.read_text(encoding="utf-8").splitlines()[0].split(",")) == list(
        MANIFEST_COLUMNS
    )
    assert inventory["clean_schema"]["mos_list_columns"] == ["utt_id", "mos"]
    assert set(inventory["clean_schema"]["manifest_columns"]) == set(MANIFEST_SCHEMA)


def test_manifest_rejects_duplicate_sample_id(work_dir):
    archive = _make_archive(work_dir / "somos.zip", bad_id=True)
    clean = work_dir / "clean"
    with pytest.raises(ValueError, match="duplicate sample_id"):
        extract_clean(archive, clean)


def test_manifest_accepts_flat_audios_layout(work_dir):
    archive = _make_archive(work_dir / "somos-flat.zip", flat_audio=True)
    clean = work_dir / "clean-flat"
    extract_clean(archive, clean)
    rows = build_manifest(clean)
    assert len(rows) == 3
    assert all(Path(row["audio_path"]).parent.name in {"TRAINSET", "VALIDSET", "TESTSET"} for row in rows)


def test_extracts_only_referenced_audio_from_nested_audios_zip(work_dir):
    archive = _make_archive(work_dir / "somos-nested.zip", nested_audio=True)
    clean = work_dir / "clean-nested"
    audio = work_dir / "audio-only"
    inventory = extract_clean(archive, clean, audio_output_dir=audio)
    assert inventory["nested_audio_archive"]["archive_member"] == "somos/audios.zip"
    assert inventory["nested_audio_archive"]["member_count"] == 3
    assert len(inventory["nested_audio_archive"]["md5"]) == 32
    assert len(inventory["nested_audio_archive"]["sha256"]) == 64
    assert inventory["audio_file_count"] == 3
    assert all(len(row["sha256"]) == 64 for row in inventory["files"])
    assert not list(audio.rglob("*_mos_list.txt"))
    rows = build_manifest(clean, audio_dir=audio)
    assert len(rows) == 3
    assert all(Path(row["audio_path"]).parent.name in {"TRAINSET", "VALIDSET", "TESTSET"} for row in rows)
