"""Integrity contracts for the frozen SOMOS v2 completion and label boundary.

This module contains no network or Kaggle API calls. It validates the sealed
completion certificate, per-runner merge provenance, and the final label
provenance consumed by the prospective analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


FROZEN_PROTOCOL_SHA256 = "81daeb5dbfcac387ea9bad14dffe0603715999524028a2902757fc6aa1c241d9"
SOMOS_ARCHIVE_MD5 = "bdfde4cae256549dfab05d713136e4af"
EXPECTED_ROWS = 20_100
EXPECTED_SPLIT_ROWS = {"train": 14_100, "valid": 3_000, "test": 3_000}
EXPECTED_SHARD_COUNT = 4
ID_RE = re.compile(r"^(?P<source_group>.+)_(?P<system_id>\d{3})\.wav$")
RUNNER_OUTPUTS = {
    "dnsmos": ("dnsmos", "dnsmos_sig"),
    "p808": ("p808",),
    "squim": ("squim", "squim_stoi", "squim_sisdr"),
    "nisqa": ("nisqa",),
    "distillmos": ("distillmos",),
    "scoreq": ("scoreq_natural", "scoreq_synthetic"),
    "utmos": ("utmos",),
    "sigmos": (
        "sigmos_mos_ovrl", "sigmos_mos_sig", "sigmos_mos_noise",
        "sigmos_mos_col", "sigmos_mos_disc", "sigmos_mos_loud",
        "sigmos_mos_reverb",
    ),
    "audiobox": (
        "audiobox_PQ", "audiobox_CU", "audiobox_CE", "audiobox_PC",
    ),
    "universa": (
        "universa_mos", "universa_scoreq", "universa_utmos",
        "universa_nisqa_mos", "universa_dnsmos_ovrl",
    ),
}
MERGE_TARGET_ACCESS = "No target MOS file or column was read during merge."
CERTIFICATE_TARGET_ACCESS = "No target MOS file or column was mounted or read."
LABEL_TARGET_ACCESS = (
    "Targets were retrieved only after the sealed completion certificate was validated."
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def strict_json_text(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        payload, sort_keys=True, indent=indent, allow_nan=False,
    ) + "\n"


def write_strict_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json_text(payload), encoding="utf-8")


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{description} is not a lowercase SHA-256")
    return value


def _require_exact_keys(record: dict, expected: set[str], description: str) -> None:
    if set(record) != expected:
        raise ValueError(
            f"{description} keys mismatch: expected {sorted(expected)}, got {sorted(record)}"
        )


def seal_completion_payload(payload: dict) -> dict:
    payload_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {
        "schema_version": "somos-v2-completion-certificate-1",
        "payload": payload,
        "seal": {"algorithm": "sha256", "payload_sha256": payload_hash},
    }


def validate_merge_provenance(
        provenance_path: Path, csv_path: Path, runner: str,
        expected_outputs: tuple[str, ...],
) -> dict:
    """Validate one merge sidecar against its exact merged CSV."""

    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol_sha256", "runner", "outputs", "rows",
        "merged_csv", "input_shards", "target_access",
    }
    _require_exact_keys(record, required, f"{runner} merge provenance")
    if record["schema_version"] != 1:
        raise ValueError(f"{runner} merge provenance schema mismatch")
    if record["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise ValueError(f"{runner} merge protocol hash mismatch")
    if record["runner"] != runner:
        raise ValueError(f"{runner} merge runner mismatch")
    if record["outputs"] != list(expected_outputs):
        raise ValueError(f"{runner} merge output bank mismatch")
    if record["rows"] != EXPECTED_ROWS:
        raise ValueError(f"{runner} merge row count mismatch")
    if record["target_access"] != MERGE_TARGET_ACCESS:
        raise ValueError(f"{runner} merge target-access declaration mismatch")

    merged = record["merged_csv"]
    _require_exact_keys(
        merged, {"path", "sha256", "bytes", "columns"},
        f"{runner} merged_csv",
    )
    expected_columns = [
        "sample_id", "source_group", "system_id", "split", *expected_outputs,
    ]
    if Path(merged["path"]).name != csv_path.name:
        raise ValueError(f"{runner} merged CSV path mismatch")
    if merged["columns"] != expected_columns:
        raise ValueError(f"{runner} merged CSV column provenance mismatch")
    if merged["bytes"] != csv_path.stat().st_size:
        raise ValueError(f"{runner} merged CSV byte count mismatch")
    if _require_sha256(merged["sha256"], f"{runner} merged CSV hash") != sha256_file(csv_path):
        raise ValueError(f"{runner} merged CSV hash mismatch")

    shards = record["input_shards"]
    if not isinstance(shards, list) or len(shards) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"{runner} merge must bind exactly {EXPECTED_SHARD_COUNT} shards")
    score_names = set()
    provenance_names = set()
    for index, shard in enumerate(shards):
        _require_exact_keys(
            shard, {"score_csv", "provenance"}, f"{runner} input shard {index}",
        )
        score = shard["score_csv"]
        source_provenance = shard["provenance"]
        _require_exact_keys(
            score, {"path", "sha256", "bytes"}, f"{runner} shard score {index}",
        )
        _require_exact_keys(
            source_provenance, {"path", "sha256"},
            f"{runner} shard provenance {index}",
        )
        _require_sha256(score["sha256"], f"{runner} shard score hash {index}")
        _require_sha256(
            source_provenance["sha256"], f"{runner} shard provenance hash {index}",
        )
        if not isinstance(score["bytes"], int) or score["bytes"] <= 0:
            raise ValueError(f"{runner} shard score byte count {index} is invalid")
        score_names.add(Path(score["path"]).name)
        provenance_names.add(Path(source_provenance["path"]).name)
    if len(score_names) != EXPECTED_SHARD_COUNT or len(provenance_names) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"{runner} merge input shard paths are not unique")
    return record


def validate_completion_certificate(
        certificate_path: Path, expected_file_sha256: str | None = None,
) -> dict:
    """Validate a sealed all-runner completion certificate and return its payload."""

    actual_file_hash = sha256_file(certificate_path)
    if expected_file_sha256 is not None:
        expected_file_sha256 = _require_sha256(
            expected_file_sha256, "expected completion-certificate hash",
        )
        if actual_file_hash != expected_file_sha256:
            raise ValueError("completion-certificate file hash mismatch")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    _require_exact_keys(
        certificate, {"schema_version", "payload", "seal"},
        "completion certificate",
    )
    if certificate["schema_version"] != "somos-v2-completion-certificate-1":
        raise ValueError("completion-certificate schema mismatch")
    seal = certificate["seal"]
    _require_exact_keys(seal, {"algorithm", "payload_sha256"}, "completion seal")
    if seal["algorithm"] != "sha256":
        raise ValueError("completion-certificate seal algorithm mismatch")
    expected_payload_hash = hashlib.sha256(
        canonical_json_bytes(certificate["payload"]),
    ).hexdigest()
    if _require_sha256(seal["payload_sha256"], "completion payload seal") != expected_payload_hash:
        raise ValueError("completion-certificate payload seal mismatch")

    payload = certificate["payload"]
    required = {
        "complete", "completed_at_utc", "protocol_sha256", "post_release_exploratory",
        "expected_rows", "split_rows", "runner_bank", "sample_id_sha256",
        "metadata_sha256", "merge_artifacts", "target_access",
    }
    _require_exact_keys(payload, required, "completion payload")
    if payload["complete"] is not True:
        raise ValueError("completion certificate is not complete")
    if payload["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise ValueError("completion-certificate protocol hash mismatch")
    if payload["post_release_exploratory"] is not True:
        raise ValueError("completion certificate must record post-release exploratory status")
    if payload["expected_rows"] != EXPECTED_ROWS:
        raise ValueError("completion-certificate row count mismatch")
    if payload["split_rows"] != EXPECTED_SPLIT_ROWS:
        raise ValueError("completion-certificate split counts mismatch")
    expected_bank = {runner: list(outputs) for runner, outputs in RUNNER_OUTPUTS.items()}
    if payload["runner_bank"] != expected_bank:
        raise ValueError("completion-certificate runner bank mismatch")
    _require_sha256(payload["sample_id_sha256"], "completion sample-ID hash")
    _require_sha256(payload["metadata_sha256"], "completion metadata hash")
    if payload["target_access"] != CERTIFICATE_TARGET_ACCESS:
        raise ValueError("completion-certificate target-access declaration mismatch")

    artifacts = payload["merge_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(RUNNER_OUTPUTS):
        raise ValueError("completion certificate does not contain all merge artifacts")
    by_runner = {}
    for artifact in artifacts:
        _require_exact_keys(
            artifact, {"runner", "outputs", "rows", "merged_csv", "merge_provenance"},
            "completion merge artifact",
        )
        runner = artifact["runner"]
        if runner in by_runner or runner not in RUNNER_OUTPUTS:
            raise ValueError(f"invalid completion artifact runner: {runner!r}")
        if artifact["outputs"] != list(RUNNER_OUTPUTS[runner]):
            raise ValueError(f"{runner} completion artifact output mismatch")
        if artifact["rows"] != EXPECTED_ROWS:
            raise ValueError(f"{runner} completion artifact row mismatch")
        for key in ("merged_csv", "merge_provenance"):
            block = artifact[key]
            _require_exact_keys(block, {"file", "sha256", "bytes"}, f"{runner} {key}")
            _require_sha256(block["sha256"], f"{runner} {key} hash")
            if not isinstance(block["bytes"], int) or block["bytes"] <= 0:
                raise ValueError(f"{runner} {key} byte count is invalid")
        by_runner[runner] = artifact
    if set(by_runner) != set(RUNNER_OUTPUTS):
        raise ValueError("completion certificate runner set mismatch")
    return payload


def _validate_evidence_file(root: Path, block: dict, description: str) -> Path:
    _require_exact_keys(block, {"file", "sha256", "bytes"}, description)
    path = root / Path(block["file"]).name
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != block["bytes"]:
        raise ValueError(f"{description} byte count mismatch")
    if _require_sha256(block["sha256"], f"{description} hash") != sha256_file(path):
        raise ValueError(f"{description} hash mismatch")
    return path


def validate_label_provenance(
        provenance_path: Path, labels_path: Path, certificate_path: Path,
) -> dict:
    """Validate the target manifest and every sidecar that produced it."""

    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol_sha256", "post_release_exploratory",
        "target_access", "completion_certificate", "label_manifest", "evidence",
    }
    _require_exact_keys(record, required, "label provenance")
    if record["schema_version"] != 1:
        raise ValueError("label provenance schema mismatch")
    if record["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise ValueError("label provenance protocol hash mismatch")
    if record["post_release_exploratory"] is not True:
        raise ValueError("label provenance must record post-release exploratory status")
    if record["target_access"] != LABEL_TARGET_ACCESS:
        raise ValueError("label provenance target-access declaration mismatch")

    certificate = record["completion_certificate"]
    _require_exact_keys(
        certificate, {"file", "sha256", "payload_sha256"},
        "label completion certificate",
    )
    if Path(certificate["file"]).name != certificate_path.name:
        raise ValueError("label provenance certificate filename mismatch")
    certificate_hash = _require_sha256(
        certificate["sha256"], "label completion-certificate file hash",
    )
    if certificate_hash != sha256_file(certificate_path):
        raise ValueError("label completion-certificate file hash mismatch")
    payload = validate_completion_certificate(certificate_path, certificate_hash)
    sealed = json.loads(certificate_path.read_text(encoding="utf-8"))["seal"]["payload_sha256"]
    if certificate["payload_sha256"] != sealed:
        raise ValueError("label completion-certificate payload hash mismatch")

    manifest = record["label_manifest"]
    _require_exact_keys(
        manifest, {"file", "sha256", "bytes", "columns", "rows", "split_rows"},
        "label manifest provenance",
    )
    if Path(manifest["file"]).name != labels_path.name:
        raise ValueError("label manifest filename mismatch")
    if manifest["sha256"] != sha256_file(labels_path):
        raise ValueError("label manifest hash mismatch")
    if manifest["bytes"] != labels_path.stat().st_size:
        raise ValueError("label manifest byte count mismatch")
    if manifest["columns"] != ["sample_id", "source_group", "system_id", "split", "mos"]:
        raise ValueError("label manifest column provenance mismatch")
    if manifest["rows"] != EXPECTED_ROWS or manifest["split_rows"] != EXPECTED_SPLIT_ROWS:
        raise ValueError("label manifest row provenance mismatch")

    evidence = record["evidence"]
    _require_exact_keys(
        evidence, {"download", "archive_inventory", "extraction_inventory"},
        "label evidence",
    )
    root = provenance_path.parent
    download_path = _validate_evidence_file(root, evidence["download"], "label download evidence")
    archive_path = _validate_evidence_file(
        root, evidence["archive_inventory"], "label archive inventory",
    )
    extraction_path = _validate_evidence_file(
        root, evidence["extraction_inventory"], "label extraction inventory",
    )
    download = json.loads(download_path.read_text(encoding="utf-8"))
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if download.get("actual_md5") != SOMOS_ARCHIVE_MD5:
        raise ValueError("label download archive MD5 mismatch")
    if archive.get("archive_md5") != SOMOS_ARCHIVE_MD5:
        raise ValueError("label archive inventory MD5 mismatch")
    if extraction.get("archive_md5") != SOMOS_ARCHIVE_MD5:
        raise ValueError("label extraction archive MD5 mismatch")
    if extraction.get("labels_only") is not True:
        raise ValueError("label extraction was not labels-only")
    if extraction.get("clean_prefix", "").replace("\\", "/").endswith(
            "training_files/split1/clean") is not True:
        raise ValueError("label extraction did not use split1/clean")
    if extraction.get("label_file_count") != 3:
        raise ValueError("label extraction file count mismatch")
    if payload["complete"] is not True:
        raise AssertionError("validated completion payload unexpectedly incomplete")
    return record
