"""Retrieve and normalize the frozen SOMOS v2 clean split on Kaggle.

This module deliberately keeps the four-gigabyte Zenodo archive out of the
repository and extracts only ``training_files/split1/clean`` plus the WAVs
referenced by its lists from the sibling ``audios.zip``.  It records the
published MD5, the runtime SHA-256, both archive inventories, and hashes of the
extracted files before writing the normalized metadata/label manifest.

The pipeline is networked only when ``download`` is called.  Unit tests use
synthetic ZIP files and never contact Zenodo or read a real SOMOS label.

Typical Kaggle use::

    python -m scripts.somos_v2_pipeline download \
      --archive /kaggle/temp/somos.zip \
      --provenance /kaggle/working/somos_v2_download.json
    python -m scripts.somos_v2_pipeline inventory \
      --archive /kaggle/temp/somos.zip \
      --output /kaggle/working/somos_v2_archive_inventory.json
    python -m scripts.somos_v2_pipeline extract \
      --archive /kaggle/temp/somos.zip \
      --output-dir /kaggle/working/somos_v2_clean \
      --audio-dir /kaggle/working/somos_v2_audio \
      --inventory /kaggle/working/somos_v2_extract_inventory.json
    python -m scripts.somos_v2_pipeline manifest \
      --clean-dir /kaggle/working/somos_v2_clean \
      --audio-dir /kaggle/working/somos_v2_audio \
      --output /kaggle/working/somos_v2_clean_manifest.csv

The ``prepare`` command runs all four stages in one resumable command.  It
does not score audio or run any predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import posixpath
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ZENODO_RECORD_URL = "https://zenodo.org/records/7378801"
ARCHIVE_URL = "https://zenodo.org/records/7378801/files/somos.zip?download=1"
DOI = "10.5281/zenodo.7378801"
ARCHIVE_NAME = "somos.zip"
EXPECTED_MD5 = "bdfde4cae256549dfab05d713136e4af"
EXPECTED_CLEAN_SUFFIX = "training_files/split1/clean"
SPLITS = ("train", "valid", "test")
SPLIT_DIRS = {"train": "TRAINSET", "valid": "VALIDSET", "test": "TESTSET"}
MOS_LIST_NAMES = {f"{split}_mos_list.txt": split for split in SPLITS}
ID_RE = re.compile(r"^(?P<source_group>.+)_(?P<system_id>\d{3})\.wav$")
MANIFEST_COLUMNS = (
    "sample_id",
    "source_group",
    "system_id",
    "split",
    "mos",
    "audio_path",
)
MANIFEST_SCHEMA = {
    "sample_id": "string, utt_id including .wav",
    "source_group": "string, sample_id without final _ plus three digits",
    "system_id": "string, final three digits before .wav",
    "split": "enum: train|valid|test",
    "mos": "float64, official clean mean naturalness in [1, 5]",
    "audio_path": "string, extracted local WAV path",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def download_archive(
    destination: Path,
    provenance_path: Path | None = None,
    url: str = ARCHIVE_URL,
    expected_md5: str = EXPECTED_MD5,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict:
    """Stream the pinned archive, hash it, and return a provenance record.

    ``destination`` is intended to be a Kaggle temporary path.  The archive is
    never copied into the repository.  A bad MD5 raises after the complete
    stream so the mismatch is diagnosable; the bad temporary file is retained
    for forensic inspection by the caller.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    request = urllib.request.Request(url, headers={"User-Agent": "SOMOS-v2-pipeline/1.0"})
    started = utc_now()
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while True:
            block = response.read(chunk_size)
            if not block:
                break
            out.write(block)
            md5.update(block)
            sha256.update(block)
            byte_count += len(block)

    record = {
        "schema_version": "somos-v2-download-1",
        "retrieved_at_utc": started,
        "zenodo_record_url": ZENODO_RECORD_URL,
        "archive_url": url,
        "doi": DOI,
        "archive_name": ARCHIVE_NAME,
        "expected_md5": expected_md5,
        "actual_md5": md5.hexdigest(),
        "local_sha256": sha256.hexdigest(),
        "bytes": byte_count,
        "path": str(destination),
    }
    if record["actual_md5"].lower() != expected_md5.lower():
        if provenance_path is not None:
            _write_json(provenance_path, record)
        raise ValueError(
            "SOMOS archive MD5 mismatch: "
            f"expected {expected_md5}, got {record['actual_md5']}"
        )
    if provenance_path is not None:
        _write_json(provenance_path, record)
    return record


def _safe_member_name(name: str) -> str:
    """Validate and normalize a ZIP member path without touching the disk."""

    normalized = posixpath.normpath(name.replace("\\", "/"))
    if normalized in {"", "."} or normalized.startswith("/"):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"archive member escapes root: {name!r}")
    return normalized


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == stat.S_IFLNK


def _member_record(info: zipfile.ZipInfo) -> dict:
    name = _safe_member_name(info.filename)
    return {
        "name": name,
        "is_dir": info.is_dir(),
        "bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
    }


def archive_inventory(
    archive: Path,
    output: Path | None = None,
    archive_record: dict | None = None,
) -> dict:
    """Inventory ZIP members and return a deterministic archive record."""

    members = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            members.append(_member_record(info))
    members.sort(key=lambda row: row["name"])
    clean_marker = EXPECTED_CLEAN_SUFFIX + "/"
    clean_members = [
        row for row in members
        if row["name"].startswith(clean_marker)
        or ("/" + clean_marker) in row["name"]
        or row["name"] == EXPECTED_CLEAN_SUFFIX
    ]
    record = {
        "schema_version": "somos-v2-archive-inventory-1",
        "inventoried_at_utc": utc_now(),
        "zenodo_record_url": ZENODO_RECORD_URL,
        "archive_url": ARCHIVE_URL,
        "doi": DOI,
        "archive_md5": (
            archive_record["actual_md5"] if archive_record else md5_file(archive)
        ),
        "expected_md5": EXPECTED_MD5,
        "local_sha256": (
            archive_record["local_sha256"] if archive_record else sha256_file(archive)
        ),
        "archive_bytes": archive.stat().st_size,
        "member_count": len(members),
        "uncompressed_bytes": sum(row["bytes"] for row in members),
        "clean_suffix": EXPECTED_CLEAN_SUFFIX,
        "clean_member_count": len(clean_members),
        "members": members,
    }
    record["md5_matches_expected"] = (
        record["archive_md5"].lower() == EXPECTED_MD5.lower()
    )
    if output is not None:
        _write_json(output, record)
    return record


def resolve_clean_prefix(names: list[str]) -> str:
    """Find the unique archive prefix containing the three frozen MOS lists."""

    normalized_names = [_safe_member_name(name) for name in names]
    name_set = set(normalized_names)
    candidates = []
    for name in normalized_names:
        if not name.endswith("/train_mos_list.txt"):
            continue
        prefix = name[: -len("train_mos_list.txt")].rstrip("/")
        required = {prefix + f"/{split}_mos_list.txt" for split in SPLITS}
        if required.issubset(name_set):
            candidates.append(prefix)
    if len(candidates) != 1:
        raise ValueError(
            "expected one clean split prefix with train/valid/test MOS lists, "
            f"found {candidates}"
        )
    prefix = candidates[0]
    if not prefix.endswith(EXPECTED_CLEAN_SUFFIX):
        raise ValueError(
            f"clean split prefix {prefix!r} does not end with {EXPECTED_CLEAN_SUFFIX!r}"
        )
    return prefix


def _safe_output_path(root: Path, relative_name: str) -> Path:
    relative = Path(*relative_name.split("/"))
    target = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"extracted member escapes output directory: {relative_name!r}") from exc
    return target


def extract_clean(
    archive: Path,
    output_dir: Path,
    inventory_path: Path | None = None,
    archive_record: dict | None = None,
    audio_output_dir: Path | None = None,
) -> dict:
    """Extract clean lists and referenced WAVs from the two-level release.

    The v2 Zenodo archive stores labels below ``training_files/split1/clean``
    and audio in a sibling ``audios.zip``.  Only WAVs named in the three clean
    lists are materialized.  If ``audio_output_dir`` is supplied, audio is
    written there under TRAINSET/VALIDSET/TESTSET, leaving it label-free for
    the prediction-only scorer.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir = audio_output_dir or output_dir
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        names = [_safe_member_name(info.filename) for info in infos]
        prefix = resolve_clean_prefix(names)
        list_members = {}
        for info in infos:
            name = _safe_member_name(info.filename)
            if not name.startswith(prefix + "/") or info.is_dir():
                continue
            base = posixpath.basename(name)
            if base in MOS_LIST_NAMES:
                if _is_symlink(info):
                    raise ValueError(f"symlink member is not allowed: {name!r}")
                if base in list_members:
                    raise ValueError(f"duplicate clean list member: {base!r}")
                list_members[base] = (name, info)
        if set(list_members) != set(MOS_LIST_NAMES):
            missing = sorted(set(MOS_LIST_NAMES) - set(list_members))
            raise ValueError(f"missing clean MOS lists under {prefix!r}: {missing}")

        records = []
        for name, info in sorted(list_members.values()):
            relative_name = name[len(prefix) + 1:]
            target = _safe_output_path(output_dir, relative_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            records.append({
                "archive_member": name,
                "source_archive": "outer",
                "relative_path": target.relative_to(output_dir).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            })

        # Parse IDs after copying the lists, before opening the nested archive.
        entries = _read_manifest_inputs(output_dir)
        requested = {sample_id: split for split, sample_id, _ in entries}

        direct_audio = {}
        for info in infos:
            name = _safe_member_name(info.filename)
            if info.is_dir() or not name.startswith(prefix + "/"):
                continue
            if not name.lower().endswith(".wav"):
                continue
            sample_id = posixpath.basename(name)
            if sample_id not in requested:
                continue
            if _is_symlink(info):
                raise ValueError(f"symlink member is not allowed: {name!r}")
            direct_audio.setdefault(sample_id, []).append((name, info))

        nested_candidates = [
            (name, info) for name, info in (
                (_safe_member_name(info.filename), info) for info in infos
            )
            if not info.is_dir() and posixpath.basename(name).lower() == "audios.zip"
        ]
        if len(nested_candidates) > 1:
            raise ValueError(f"expected at most one nested audios.zip, found {len(nested_candidates)}")

        nested_record = None
        nested_path = None
        nested_zip = None
        try:
            if nested_candidates:
                nested_name, nested_info = nested_candidates[0]
                if _is_symlink(nested_info):
                    raise ValueError(f"symlink member is not allowed: {nested_name!r}")
                # Keep the seekable nested archive beside the downloaded outer
                # archive, normally /kaggle/temp, rather than in the
                # autosaved /kaggle/working output directory.
                temp_file = tempfile.NamedTemporaryFile(
                    prefix=".somos-audios-",
                    suffix=".zip",
                    dir=str(archive.parent),
                    delete=False,
                )
                nested_path = Path(temp_file.name)
                temp_file.close()
                nested_md5 = hashlib.md5()
                nested_sha256 = hashlib.sha256()
                nested_bytes = 0
                with zf.open(nested_info) as source, nested_path.open("wb") as sink:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        sink.write(block)
                        nested_md5.update(block)
                        nested_sha256.update(block)
                        nested_bytes += len(block)
                nested_zip = zipfile.ZipFile(nested_path)
                nested_members = []
                nested_audio = {}
                for info in nested_zip.infolist():
                    member_name = _safe_member_name(info.filename)
                    if _is_symlink(info):
                        raise ValueError(f"symlink member is not allowed: {member_name!r}")
                    nested_members.append(_member_record(info))
                    if not info.is_dir() and member_name.lower().endswith(".wav"):
                        sample_id = posixpath.basename(member_name)
                        if sample_id in requested:
                            nested_audio.setdefault(sample_id, []).append((member_name, info))
                nested_members.sort(key=lambda row: row["name"])
                nested_record = {
                    "archive_member": nested_name,
                    "archive_name": "audios.zip",
                    "bytes": nested_bytes,
                    "md5": nested_md5.hexdigest(),
                    "sha256": nested_sha256.hexdigest(),
                    "member_count": len(nested_members),
                    "wav_member_count": sum(
                        1 for row in nested_members if row["name"].lower().endswith(".wav")
                    ),
                    "members": nested_members,
                }
            else:
                nested_audio = {}

            audio_records = []
            for split, sample_id, _ in entries:
                direct = direct_audio.get(sample_id, [])
                nested = nested_audio.get(sample_id, [])
                sources = [("outer", name, info) for name, info in direct]
                sources.extend(("audios.zip", name, info) for name, info in nested)
                if not sources:
                    raise FileNotFoundError(
                        f"MOS item {sample_id!r} has no audio in clean split or audios.zip"
                    )
                if len(sources) > 1:
                    raise ValueError(f"ambiguous audio ID {sample_id!r} across archive members")
                source_kind, source_name, info = sources[0]
                target = _safe_output_path(
                    audio_output_dir,
                    f"{SPLIT_DIRS[split]}/{sample_id}",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                source_zip = zf if source_kind == "outer" else nested_zip
                with source_zip.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                audio_records.append({
                    "archive_member": source_name,
                    "source_archive": source_kind,
                    "relative_path": target.relative_to(audio_output_dir).as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                })
                records.append({
                    "archive_member": source_name,
                    "source_archive": source_kind,
                    "relative_path": target.relative_to(audio_output_dir).as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": audio_records[-1]["sha256"],
                })
        finally:
            if nested_zip is not None:
                nested_zip.close()
            if nested_path is not None:
                nested_path.unlink(missing_ok=True)

    record = {
        "schema_version": "somos-v2-extraction-inventory-1",
        "extracted_at_utc": utc_now(),
        "zenodo_record_url": ZENODO_RECORD_URL,
        "archive_url": ARCHIVE_URL,
        "doi": DOI,
        "archive_md5": (
            archive_record["actual_md5"] if archive_record else md5_file(archive)
        ),
        "expected_md5": EXPECTED_MD5,
        "archive_local_sha256": (
            archive_record["local_sha256"] if archive_record else sha256_file(archive)
        ),
        "clean_prefix": prefix,
        "clean_schema": {
            "mos_list_files": sorted(MOS_LIST_NAMES),
            "mos_list_columns": ["utt_id", "mos"],
            "id_regex": ID_RE.pattern,
            "splits": list(SPLITS),
            "manifest_columns": list(MANIFEST_COLUMNS),
        },
        "output_dir": str(output_dir),
        "audio_output_dir": str(audio_output_dir),
        "selected_file_count": len(records),
        "selected_bytes": sum(row["bytes"] for row in records),
        "label_file_count": len(list_members),
        "audio_file_count": len(audio_records),
        "nested_audio_archive": nested_record,
        "files": records,
    }
    record["md5_matches_expected"] = (
        record["archive_md5"].lower() == EXPECTED_MD5.lower()
    )
    if inventory_path is not None:
        _write_json(inventory_path, record)
    return record


def _parse_mos_line(line: str, source: Path, line_number: int) -> tuple[str, float] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    # The released lists are simple ID/score text files. Supporting comma and
    # tab separators makes the parser robust to a text editor round-trip while
    # retaining a strict two-field semantic schema.
    fields = next(csv.reader([text], delimiter=",")) if "," in text else text.split()
    fields = [field.strip() for field in fields if field.strip()]
    if len(fields) == 2 and fields[0].lower() in {"utt_id", "file", "filename", "id"}:
        return None
    if len(fields) != 2:
        raise ValueError(f"{source}:{line_number}: expected utt_id and mos, got {text!r}")
    try:
        mos = float(fields[1])
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: non-numeric MOS {fields[1]!r}") from exc
    if not math.isfinite(mos) or not 1.0 <= mos <= 5.0:
        raise ValueError(f"{source}:{line_number}: MOS outside [1, 5]: {mos!r}")
    return fields[0], mos


def _read_manifest_inputs(clean_dir: Path) -> list[tuple[str, str, float]]:
    """Read and validate the three clean lists without opening any audio."""

    entries: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for split in SPLITS:
        list_path = clean_dir / f"{split}_mos_list.txt"
        if not list_path.is_file():
            matches = list(clean_dir.rglob(f"{split}_mos_list.txt"))
            if len(matches) != 1:
                raise FileNotFoundError(f"expected one {split}_mos_list.txt under {clean_dir}")
            list_path = matches[0]
        for line_number, line in enumerate(
            list_path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
        ):
            parsed = _parse_mos_line(line, list_path, line_number)
            if parsed is None:
                continue
            sample_id, mos = parsed
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across split lists: {sample_id!r}")
            seen.add(sample_id)
            if ID_RE.fullmatch(sample_id) is None:
                raise ValueError(
                    f"{list_path}:{line_number}: ID does not match {ID_RE.pattern!r}: {sample_id!r}"
                )
            entries.append((split, sample_id, mos))
    if not entries:
        raise ValueError("SOMOS clean manifest is empty")
    return entries


def _find_audio(audio_dir: Path, split: str, sample_id: str) -> Path:
    split_dir = audio_dir / SPLIT_DIRS[split]
    # SOMOS releases and downstream preprocessors use both the split-folder
    # convention (TRAINSET/VALIDSET/TESTSET) and a flat ``audios`` folder.
    # Keep the frozen manifest independent of that packaging detail.
    candidates = [
        split_dir / sample_id,
        audio_dir / "audios" / sample_id,
        audio_dir / sample_id,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # The lists carry the authoritative IDs.  Search by exact basename when
    # the archive has nested audio directories or when TRAINSET is a listing
    # file rather than a directory.
    by_name = [
        path for path in audio_dir.rglob(Path(sample_id).name)
        if path.is_file()
    ]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise ValueError(f"ambiguous audio ID {sample_id!r} in {audio_dir}")
    raise FileNotFoundError(
        f"MOS item {sample_id!r} has no extracted audio in {audio_dir}"
    )


def build_manifest(
    clean_dir: Path,
    output: Path | None = None,
    audio_dir: Path | None = None,
) -> list[dict]:
    """Build the frozen normalized SOMOS clean metadata/label manifest."""

    audio_dir = audio_dir or clean_dir
    rows: list[dict] = []
    for split, sample_id, mos in _read_manifest_inputs(clean_dir):
        match = ID_RE.fullmatch(sample_id)
        assert match is not None
        audio = _find_audio(audio_dir, split, sample_id)
        rows.append({
            "sample_id": sample_id,
            "source_group": match.group("source_group"),
            "system_id": match.group("system_id"),
            "split": split,
            "mos": mos,
            "audio_path": str(audio),
        })
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_prepare(args: argparse.Namespace) -> dict:
    archive = args.archive
    provenance = args.provenance
    if not archive.exists():
        download = download_archive(archive, provenance)
    else:
        archive_md5 = md5_file(archive)
        if archive_md5.lower() != EXPECTED_MD5.lower():
            raise ValueError(
                f"existing archive MD5 mismatch: expected {EXPECTED_MD5}, got {archive_md5}"
            )
        download = {
            "schema_version": "somos-v2-download-1",
            "retrieved_at_utc": None,
            "zenodo_record_url": ZENODO_RECORD_URL,
            "archive_url": ARCHIVE_URL,
            "doi": DOI,
            "archive_name": ARCHIVE_NAME,
            "expected_md5": EXPECTED_MD5,
            "actual_md5": archive_md5,
            "local_sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "path": str(archive),
            "reused_existing_archive": True,
        }
        _write_json(provenance, download)

    archive_record = archive_inventory(
        archive, args.archive_inventory, archive_record=download
    )
    extract_record = extract_clean(
        archive,
        args.clean_dir,
        args.extract_inventory,
        archive_record=download,
        audio_output_dir=args.audio_dir,
    )
    rows = build_manifest(args.clean_dir, args.manifest, audio_dir=args.audio_dir)
    return {
        "download": download,
        "archive_inventory": archive_record,
        "extraction_inventory": extract_record,
        "manifest": {
            "path": str(args.manifest),
            "rows": len(rows),
            "splits": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
            "columns": list(MANIFEST_COLUMNS),
            "schema": MANIFEST_SCHEMA,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="stream and hash the pinned Zenodo archive")
    download.add_argument("--archive", type=Path, required=True)
    download.add_argument("--provenance", type=Path, required=True)

    inventory = sub.add_parser("inventory", help="inventory all archive members without extraction")
    inventory.add_argument("--archive", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)

    extract = sub.add_parser("extract", help="extract only training_files/split1/clean")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--inventory", type=Path, required=True)
    extract.add_argument(
        "--audio-dir", type=Path,
        help="label-free output root for referenced WAVs (defaults to --output-dir)",
    )

    manifest = sub.add_parser("manifest", help="normalize frozen clean MOS lists and audio IDs")
    manifest.add_argument("--clean-dir", type=Path, required=True)
    manifest.add_argument("--audio-dir", type=Path)
    manifest.add_argument("--output", type=Path, required=True)

    prepare = sub.add_parser("prepare", help="download, inventory, extract, and build manifest")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--provenance", type=Path, required=True)
    prepare.add_argument("--archive-inventory", type=Path, required=True)
    prepare.add_argument("--clean-dir", type=Path, required=True)
    prepare.add_argument("--audio-dir", type=Path, required=True)
    prepare.add_argument("--extract-inventory", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        result = download_archive(args.archive, args.provenance)
    elif args.command == "inventory":
        result = archive_inventory(args.archive, args.output)
    elif args.command == "extract":
        result = extract_clean(
            args.archive, args.output_dir, args.inventory,
            audio_output_dir=args.audio_dir,
        )
    elif args.command == "manifest":
        rows = build_manifest(args.clean_dir, args.output, audio_dir=args.audio_dir)
        result = {
            "path": str(args.output),
            "rows": len(rows),
            "splits": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
            "columns": list(MANIFEST_COLUMNS),
            "schema": MANIFEST_SCHEMA,
        }
    else:
        result = run_prepare(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
