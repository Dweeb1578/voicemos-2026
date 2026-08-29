"""Create a non-overwriting SHA-256 sidecar for a frozen validation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_PHRASES = (
    "Dataset repository",
    "Exact target column",
    "Exact predictor",
    "Complete-case",
    "Group identifier",
    "Label budgets",
    "Primary comparison",
)
FORBIDDEN_MARKERS = ("TODO", "TBD", "Not activated")


def _git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_protocol(text: str) -> None:
    missing = [phrase for phrase in REQUIRED_PHRASES if phrase not in text]
    if missing:
        raise ValueError("protocol is missing required declarations: " + ", ".join(missing))
    present = [marker for marker in FORBIDDEN_MARKERS if marker in text]
    if present:
        raise ValueError("protocol still contains unresolved markers: " + ", ".join(present))


def freeze_protocol(protocol: Path, output: Path, repo: Path) -> dict:
    payload = protocol.read_bytes()
    validate_protocol(payload.decode("utf-8"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen sidecar: {output}")

    status = _git_value(repo, "status", "--porcelain")
    record = {
        "schema_version": 1,
        "protocol_path": protocol.resolve().relative_to(repo.resolve()).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_value(repo, "rev-parse", "HEAD"),
        "git_dirty_at_freeze": bool(status),
        "immutability_note": (
            "This local record becomes independently auditable only after the "
            "protocol and sidecar are committed to an append-only remote or archive."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    protocol = args.protocol.resolve()
    output = (args.out or protocol.with_suffix(".sha256.json")).resolve()
    record = freeze_protocol(protocol, output, args.repo.resolve())
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
