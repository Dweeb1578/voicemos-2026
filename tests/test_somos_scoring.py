"""Offline contract tests for the label-blind SOMOS scoring orchestration."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

from scripts.somos_kaggle_orchestrate import build
from scripts.somos_merge_shards import collect_shards, validate_provenance
from scripts.somos_scoring import (
    CACHE_CHECKPOINT_ROWS,
    FROZEN_PROTOCOL_SHA256,
    RUNNERS,
    SAMPLE_ID_COLUMN,
    build_audio_manifest,
    load_audio_manifest,
    merge_score_shards,
    score_paths_resumable,
    select_shard,
    sha256_file,
    validate_score_shard,
    write_score_shard,
)


def _audio_tree(root: Path, count_per_split: int = 40) -> Path:
    for split, offset in (("TRAINSET", 0), ("VALIDSET", 100), ("TESTSET", 200)):
        directory = root / split
        directory.mkdir(parents=True)
        for index in range(count_per_split):
            # Filename is the official immutable join key, including .wav.
            (directory / f"sentence_{index + offset:04d}_{index % 7:03d}.wav").write_bytes(b"RIFF")
    return root


@pytest.fixture
def work_dir():
    """Use the project tree, because this Windows pytest temp ACL is restricted."""
    path = Path(__file__).resolve().parents[1] / f".somos-score-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _cache_scores(entries: pd.DataFrame, cache_path: Path, outputs: tuple[str, ...]) -> None:
    positions = {path: index for index, path in enumerate(entries.audio_path)}

    def scorer(path: str) -> tuple[float, ...]:
        return tuple(float(positions[path] + column + 1) for column in range(len(outputs)))

    score_paths_resumable(scorer, entries, cache_path, outputs, time.monotonic() + 20, "test")


def test_audio_manifest_keeps_exact_wav_sample_id_and_rejects_targets(work_dir: Path):
    audio_root = _audio_tree(work_dir / "audio", count_per_split=2)
    manifest = build_audio_manifest(audio_root)

    assert SAMPLE_ID_COLUMN in manifest.columns
    assert "utt_id" not in manifest.columns
    assert manifest[SAMPLE_ID_COLUMN].str.endswith(".wav").all()
    assert set(manifest.split) == {"train", "valid", "test"}
    assert manifest.iloc[0].source_group.startswith("sentence_")

    (audio_root / "train_mos_list.txt").write_text("do not read me\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target MOS list"):
        build_audio_manifest(audio_root)


def test_checkpointed_cache_resumes_and_writes_all_rows(work_dir: Path):
    assert CACHE_CHECKPOINT_ROWS == 50
    # Write the audio-only manifest explicitly, mimicking the scorer input.
    audio_root = _audio_tree(work_dir / "audio2", count_per_split=40)
    manifest = build_audio_manifest(audio_root)
    manifest_path = work_dir / "somos_audio_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    entries = load_audio_manifest(manifest_path, audio_root)
    cache = work_dir / "scores.cache.csv"
    calls: list[str] = []

    def scorer(path: str) -> tuple[float]:
        calls.append(path)
        return (float(len(calls)),)

    result = score_paths_resumable(
        scorer, entries, cache, ("p808",), time.monotonic() + 20, "test")
    assert len(calls) == len(entries)
    assert len(result) == len(entries)
    assert len(pd.read_csv(cache)) == len(entries)

    # A resumed run must recognize the cache and avoid a second inference pass.
    resumed = score_paths_resumable(
        lambda _: pytest.fail("cached clip was scored twice"),
        entries, cache, ("p808",), time.monotonic() + 20, "test")
    assert resumed == result


def test_shards_merge_to_canonical_runner_schema(work_dir: Path):
    audio_root = _audio_tree(work_dir / "audio", count_per_split=24)
    manifest = build_audio_manifest(audio_root)
    manifest_path = work_dir / "somos_audio_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    loaded = load_audio_manifest(manifest_path, audio_root)
    outputs = RUNNERS["p808"]["outputs"]
    shards = []
    for part in range(4):
        entries = select_shard(loaded, part, 4)
        cache = work_dir / f"p808-part{part:02d}-of-04.cache.csv"
        _cache_scores(entries, cache, outputs)
        score = work_dir / f"p808-part{part:02d}-of-04.csv"
        write_score_shard(entries, cache, outputs, score)
        score.with_suffix(".provenance.json").write_text(json.dumps({
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "runner": {"id": "p808"},
            "target_access": "No target MOS file or column was read during scoring.",
            "score_shard": {"sha256": sha256_file(score)},
        }), encoding="utf-8")
        shards.append(score)

    discovered = collect_shards(work_dir, "p808")
    assert discovered == shards
    for score in discovered:
        validate_provenance(score, "p808")

    merged = work_dir / "merged" / "p808.csv"
    result = merge_score_shards(shards, "p808", loaded, merged)
    frame = pd.read_csv(merged, dtype={SAMPLE_ID_COLUMN: str, "system_id": str})
    assert result["rows"] == len(loaded)
    assert list(frame.columns) == [SAMPLE_ID_COLUMN, "source_group", "system_id", "split", "p808"]
    assert set(frame[SAMPLE_ID_COLUMN]) == set(loaded[SAMPLE_ID_COLUMN])

    bad = frame.assign(mos=3.0)
    with pytest.raises(ValueError, match="forbidden target"):
        validate_score_shard(bad, loaded, outputs)


def test_generated_run_cell_passes_only_string_arguments(work_dir: Path, monkeypatch):
    # compile() accepts an int in the argv list, but str.join and subprocess
    # both reject one at run time, so the cell has to be executed to be checked.
    import subprocess as real_subprocess

    output = work_dir / "somos-kernels"
    build(
        username="unit-test", audio_kernel="unit-test/audio", resume_kernel=None,
        shard_count=4, smoke_items=100, output_root=output,
    )
    notebook = json.loads((output / "scoreq" / "part-02" / "somos_score.ipynb").read_text())
    run_cell = next(
        cell["source"] for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "scripts.somos_runner" in cell["source"]
    )

    captured = {}

    class _Completed:
        returncode = 0
        stdout = "scored 100 rows"

    def fake_run(command, *args, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs.get("cwd")
        return _Completed()

    monkeypatch.setattr(real_subprocess, "run", fake_run)
    namespace = {
        "AUDIO_ROOT": Path("/kaggle/input/audio"),
        "MANIFEST": Path("/kaggle/input/somos_audio_manifest.csv"),
        # A real directory: the cell writes the runner log here.
        "OUT": work_dir,
        "EXTRA": ["--vendor-root", "/kaggle/working/vendor"],
        # A Path here proves the cell coerces artifacts too, not just numbers.
        "ARTIFACTS": [Path("/kaggle/working/somos_artifacts")],
        "BUNDLE": Path("/kaggle/working/somos_bundle"),
        "TAG": "scoreq-part02-of-04",
    }
    exec(run_cell, namespace)

    command = captured.get("command")
    assert command, "the generated run cell never invoked scripts.somos_runner"
    assert all(isinstance(value, str) for value in command), [
        value for value in command if not isinstance(value, str)
    ]
    assert command[command.index("--shard-index") + 1] == "2"
    assert command[command.index("--shard-count") + 1] == "4"
    assert command[command.index("--smoke-items") + 1] == "100"
    # The subprocess does not inherit sys.path, so it must run in the bundle.
    assert captured["cwd"] == str(namespace["BUNDLE"])


def test_environment_snapshot_survives_a_machine_without_nvidia_smi(work_dir: Path, monkeypatch):
    # dnsmos and p808 are frozen as CPU runners, and those machines have no
    # driver, so Popen raises FileNotFoundError before check=False can apply.
    import subprocess as real_subprocess
    from scripts.somos_scoring import environment_snapshot

    real_run = real_subprocess.run

    def run_without_driver(command, *args, **kwargs):
        if command and str(command[0]) == "nvidia-smi":
            raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(real_subprocess, "run", run_without_driver)
    record = environment_snapshot(work_dir / "environment.json")
    payload = json.loads((work_dir / "environment.json").read_text(encoding="utf-8"))
    assert payload["nvidia_smi_present"] is False
    assert payload["gpu"] == []
    assert payload["pip_freeze"]
    assert record["sha256"]


def test_tree_inventory_skips_partial_downloads(work_dir: Path):
    # A Hugging Face cache carries *.incomplete and *.lock files while a fetch
    # is in flight; hashing them crashed the Uni-VERSA runner when one was
    # renamed between the directory walk and the stat call.
    from scripts.somos_scoring import tree_inventory

    root = work_dir / "hf"
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / "model.bin").write_bytes(b"weights")
    (root / "blobs" / "abc123.incomplete").write_bytes(b"partial")
    (root / "blobs" / "abc123.lock").write_bytes(b"")

    inventory = tree_inventory([root])
    names = [Path(entry["path"]).name for entry in inventory]
    assert names == ["model.bin"]
    assert inventory[0]["bytes"] == len(b"weights")

    # A real artifact that disappears is still an error, not a silent skip.
    missing = work_dir / "gone"
    with pytest.raises(FileNotFoundError):
        tree_inventory([missing])


def test_merge_kernels_are_private_prediction_only_and_complete(work_dir: Path):
    # The canonical merge needs all 20,100 WAVs present, and they exist only in
    # the private ingestion kernel's output, so the merge runs on Kaggle.
    from scripts.somos_merge_kernel import build as build_merge

    output = work_dir / "merge-kernels"
    audio = "unit-test/somos-v2-audio-only-prospective-ingestion"
    shard_kernels = {
        runner: [f"unit-test/somos-{runner}-part{index:02d}-of-04" for index in range(4)]
        for runner in RUNNERS
    }
    result = build_merge(username="unit-test", audio_kernel=audio,
                         shard_kernels=shard_kernels, shard_count=4,
                         output_root=output)
    assert result["kernels"] == len(RUNNERS)

    for runner in RUNNERS:
        meta = json.loads((output / runner / "kernel-metadata.json").read_text())
        assert meta["is_private"] is True, runner
        assert meta["enable_gpu"] is False, runner
        # The audio source first, then exactly the four shard kernels.
        assert meta["kernel_sources"] == [audio, *shard_kernels[runner]], runner
        assert (output / runner / meta["code_file"]).is_file(), runner
        assert meta["dataset_sources"] == [] and meta["model_sources"] == [], runner

        lock = json.loads((output / runner / "merge.lock.json").read_text())
        assert lock["protocol_sha256"] == FROZEN_PROTOCOL_SHA256, runner
        assert lock["outputs"] == list(RUNNERS[runner]["outputs"]), runner

        notebook = json.loads((output / runner / "somos_merge.ipynb").read_text())
        source = "\n".join(
            cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "generated-merge", "exec")
        # The prediction-only boundary has to be enforced inside the kernel.
        assert "rglob('*_mos_list.txt')" in source, runner
        assert "'mos' not in frame.columns" in source, runner

    # Refuses to overwrite, so a previous build cannot be silently replaced.
    with pytest.raises(FileExistsError):
        build_merge(username="unit-test", audio_kernel=audio,
                    shard_kernels=shard_kernels, shard_count=4, output_root=output)

    # A short shard list is a hard error, never a partial merge.
    short = dict(shard_kernels)
    short["dnsmos"] = short["dnsmos"][:3]
    with pytest.raises(ValueError, match="expected 4 shard kernels"):
        build_merge(username="unit-test", audio_kernel=audio,
                    shard_kernels=short, shard_count=4,
                    output_root=work_dir / "merge-short")


def test_kaggle_build_is_local_only_and_pins_metadata(work_dir: Path):
    output = work_dir / "somos-kernels"
    result = build(
        username="unit-test", audio_kernel="unit-test/somos-v2-audio-only-ingestion",
        resume_kernel="unit-test/prior-score-output",
        shard_count=2, smoke_items=3, output_root=output,
    )
    assert result["kernels"] == 20

    dns_meta = json.loads((output / "dnsmos" / "part-00" / "kernel-metadata.json").read_text())
    # squim is CPU-only: torchaudio 2.11.0 cannot import against Kaggle's CUDA.
    squim_meta = json.loads((output / "squim" / "part-00" / "kernel-metadata.json").read_text())
    # Derive the device expectation from the spec so this keeps holding when a
    # runner moves between CPU and GPU.
    for runner, spec in RUNNERS.items():
        meta = json.loads((output / runner / "part-00" / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is bool(spec["gpu"]), runner
        if spec["gpu"]:
            assert meta["machine_shape"] == "NvidiaTeslaT4", runner
        else:
            assert "machine_shape" not in meta, runner
    lock = json.loads((output / "universa" / "part-01" / "orchestration.lock.json").read_text())
    assert dns_meta["enable_gpu"] is False
    assert "machine_shape" not in dns_meta
    assert squim_meta["enable_gpu"] is False
    assert "machine_shape" not in squim_meta
    assert lock["runner"]["revision"] == RUNNERS["universa"]["revision"]
    assert lock["audio_input_contract"]["target_access"].startswith("No MOS-list")
    assert dns_meta["kernel_sources"] == [
        "unit-test/somos-v2-audio-only-ingestion", "unit-test/prior-score-output",
    ]
    assert lock["resume"]["kernel_source"] == "unit-test/prior-score-output"
    assert lock["smoke_items"] == 3

    notebook = json.loads((output / "dnsmos" / "part-00" / "somos_score.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "generated-notebook", "exec")
