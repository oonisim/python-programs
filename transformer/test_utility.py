"""Tests for utility.py helpers."""

import re
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from transformer.utility import (  # noqa: E402
    build_snapshot_filename,
    cleanup_old_files,
    resolve_file_path,
)


def test_build_snapshot_filename_format():
    """Snapshot filename should include padded epoch/step and end with .pt."""
    # Input: epoch=1, step=2 to validate zero-padding.
    name = build_snapshot_filename(epoch=1, step=2)

    # Expected: padded fields and timestamp pattern ending with .pt.
    pattern = r"^snapshot_epoch_0001_step_000002_\d{8}_\d{6}\.pt$"
    assert re.match(pattern, name) is not None


def test_resolve_file_path_not_found(tmp_path: Path):
    """resolve_file_path should raise when file does not exist."""
    # Input: missing file path should not resolve.
    missing = tmp_path / "missing.pt"

    # Expected: FileNotFoundError for missing file.
    with pytest.raises(FileNotFoundError):
        resolve_file_path(str(missing), tmp_path)


def test_cleanup_old_files_deletes_older(tmp_path: Path):
    """cleanup_old_files should keep the newest N files only."""
    # Input: create four files so two are deleted when keep_last_n=2.
    for i in range(4):
        file_path = tmp_path / f"snapshot_{i}.pt"
        file_path.write_text("x")

    # Expected: two files deleted, two remain.
    deleted = cleanup_old_files(tmp_path, "snapshot_*.pt", keep_last_n=2)
    assert deleted == 2
