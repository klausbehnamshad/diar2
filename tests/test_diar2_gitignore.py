# SPDX-License-Identifier: Apache-2.0
"""input/ and output/ never reach the public repository."""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (HERE / ".git").exists(), reason="kein git-Arbeitsbaum")


def git(*args):
    return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True)


def test_input_and_output_are_ignored():
    files = [HERE / "input" / "Interview Probe.mp4",
             HERE / "output" / "Interview Probe" / "Interview Probe.diar2.json",
             HERE / "output" / ".work" / "Interview Probe" / "Interview Probe.16k.wav"]
    created = [f for f in files if not f.exists()]
    try:
        for f in created:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"nicht ins Repository")
        status = git("status", "--porcelain", "--untracked-files=all").stdout
        assert "input/" not in status and "output/" not in status, status
        rels = [str(f.relative_to(HERE)) for f in files]
        r = git("check-ignore", "-v", *rels)
        assert r.returncode == 0, r.stderr
        ignored = r.stdout.splitlines()
        assert len(ignored) == len(files)
        assert all(line.startswith(".gitignore:") for line in ignored)
    finally:
        for f in created:
            f.unlink()
        for d in (HERE / "output" / ".work" / "Interview Probe", HERE / "output" / "Interview Probe"):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
