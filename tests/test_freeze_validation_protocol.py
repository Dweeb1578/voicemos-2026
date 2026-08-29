from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from scripts.freeze_validation_protocol import freeze_protocol, validate_protocol


VALID_PROTOCOL = """
Dataset repository: example/repo
Exact target column: mos
Exact predictor columns: a, b
Complete-case filtering: required
Group identifier: source_id
Label budgets: 200, 500
Primary comparison: raw ridge minus equal ranks
"""


def test_validate_protocol_rejects_placeholders():
    with pytest.raises(ValueError, match="unresolved markers"):
        validate_protocol(VALID_PROTOCOL + "\nTBD\n")


def test_free_protocol_writes_hash_and_never_overwrites():
    repo = MagicMock(spec=Path)
    repo.resolve.return_value = Path("C:/repo")
    protocol = MagicMock(spec=Path)
    protocol.read_bytes.return_value = VALID_PROTOCOL.encode("utf-8")
    protocol.resolve.return_value = Path("C:/repo/protocol.md")
    output = MagicMock(spec=Path)
    output.exists.return_value = False
    handle = mock_open()
    output.open = handle

    with patch("scripts.freeze_validation_protocol._git_value", return_value=None):
        record = freeze_protocol(protocol, output, repo)
    assert record["protocol_path"] == "protocol.md"
    assert len(record["sha256"]) == 64
    assert record["bytes"] == len(VALID_PROTOCOL.encode("utf-8"))
    output.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    output.open.assert_called_once_with("x", encoding="utf-8", newline="\n")

    output.exists.return_value = True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        with patch("scripts.freeze_validation_protocol._git_value", return_value=None):
            freeze_protocol(protocol, output, repo)
