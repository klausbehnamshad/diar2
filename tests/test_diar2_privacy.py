# SPDX-License-Identifier: Apache-2.0
"""Datenschutz: Offline-Modus und abgeschaltete Telemetrie in jeder Stufe."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_diar2_pipeline import HERE, env, run_diar2  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg fehlt")

VARS = ["DO_NOT_TRACK", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE",
        "PYANNOTE_METRICS_ENABLED", "TRANSFORMERS_OFFLINE"]
TELEMETRY = {"DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
             "PYANNOTE_METRICS_ENABLED": "false"}
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# logs the privacy variables it was started with, then runs the real program
WRAPPER = """#!/bin/bash
tag="$(basename "{name}")_$(basename -- "${{1:-x}}")_$(basename -- "${{2:-x}}")_$$"
env | grep -E '^({vars})=' | sort > "$ENVLOG/$tag"
exec "{real}" "$@"
"""

FAKE_PYANNOTE = '''
import os
# fails if diar2_stages.py imported pyannote before switching telemetry off
assert os.environ.get("PYANNOTE_METRICS_ENABLED") == "false", "metrics on at import"
assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1", "hf telemetry on at import"
class _Seg:
    def __init__(s, a, b): s.start, s.end = a, b
class _Ann:
    def itertracks(self, yield_label=True):
        yield _Seg(0.1, 1.3), None, "SPEAKER_00"
        yield _Seg(1.7, 3.0), None, "SPEAKER_01"
class Pipeline:
    @classmethod
    def from_pretrained(cls, name, token=None): return cls()
    def __call__(self, audio, **kw): return _Ann()
'''

FAKE_HF_HUB = '''
import os
assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1", "hf telemetry on at import"
def snapshot_download(repo_id):
    return "/cache/" + repo_id
'''


def wrap(bin_dir, name, real):
    p = bin_dir / name
    p.write_text(WRAPPER.format(name=name, real=real, vars="|".join(VARS)))
    p.chmod(0o755)
    return p


def logged(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


@pytest.fixture()
def privacy(env):  # noqa: F811
    home, e = env
    fakes = Path(e["PYTHONPATH"])
    (fakes / "pyannote" / "audio").mkdir(parents=True)
    (fakes / "pyannote" / "audio" / "__init__.py").write_text(FAKE_PYANNOTE)
    (fakes / "torch.py").write_text("def from_numpy(a):\n    return a\n")
    log = home / "envlog"
    log.mkdir()
    bin_ = home / "wrap"
    bin_.mkdir()
    wrap(bin_, "ffmpeg", shutil.which("ffmpeg"))
    e.update({
        "PATH": f"{bin_}:{e['PATH']}", "ENVLOG": str(log), "DIAR2_SECOND": "1",
        "DIAR2_PYTHON": str(wrap(bin_, "python", sys.executable)),
        "DIAR2_NEMO": str(wrap(bin_, "nemo", e["DIAR2_NEMO"])),
    })
    for v in VARS:
        e.pop(v, None)
    return home, e, log


def test_every_stage_runs_offline_without_telemetry(privacy):
    home, e, log = privacy
    e["PYANNOTE_METRICS_ENABLED"] = "true"      # a caller's setting must not win
    r = run_diar2((home, e), "probe interview.mp4")
    assert r.returncode == 0, r.stdout + r.stderr
    tags = sorted(p.name.rsplit("_", 1)[0] for p in log.iterdir())
    for stage in ("ffmpeg", "python_diar2_stages.py_transcribe", "python_diar2_stages.py_align",
                  "nemo_diarize", "python_diar2_stages.py_pyannote", "python_-_meta.json",
                  "python_diar2_merge.py"):
        assert any(t.startswith(stage) for t in tags), (stage, tags)
    for p in log.iterdir():
        assert logged(p) == {**TELEMETRY, **OFFLINE}, p.name


def test_online_switch_drops_only_offline_variables(privacy):
    home, e, log = privacy
    e.update({"DIAR2_ONLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    r = run_diar2((home, e), "probe interview.mp4")
    assert r.returncode == 0, r.stdout + r.stderr
    files = list(log.iterdir())
    assert files
    for p in files:
        assert logged(p) == TELEMETRY, p.name   # telemetry stays off, offline mode is gone


def test_stages_switch_telemetry_off_before_pyannote_import(tmp_path):
    fakes = tmp_path / "fakes"
    (fakes / "pyannote" / "audio").mkdir(parents=True)
    (fakes / "pyannote" / "audio" / "__init__.py").write_text(FAKE_PYANNOTE)
    (fakes / "torch.py").write_text("def from_numpy(a):\n    return a\n")
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=2", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    e = dict(os.environ, PYTHONPATH=str(fakes), PYANNOTE_METRICS_ENABLED="true",
             HF_HUB_DISABLE_TELEMETRY="0", HOME=str(tmp_path))
    out = tmp_path / "p.rttm"
    r = subprocess.run([sys.executable, str(HERE / "diar2_stages.py"), "pyannote", str(wav),
                        str(out)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SPEAKER_01" in out.read_text()


def test_stages_switch_telemetry_off_before_hf_hub_import(tmp_path):
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "huggingface_hub.py").write_text(FAKE_HF_HUB)
    e = dict(os.environ, PYTHONPATH=str(fakes), HF_HUB_DISABLE_TELEMETRY="0")
    r = subprocess.run([sys.executable, str(HERE / "diar2_stages.py"), "prefetch", "whisper"],
                       env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "/cache/mlx-community/whisper-large-v3-turbo" in r.stdout


def test_install_disables_telemetry_while_loading():
    text = (HERE / "install_mac.sh").read_text(encoding="utf-8")
    block = text[text.index("# --- telemetrie: begin"):text.index("# --- telemetrie: end")]
    e = {k: v for k, v in os.environ.items() if k not in VARS}
    e["PYANNOTE_METRICS_ENABLED"] = "true"
    r = subprocess.run(["bash", "-c", f"{block}\nenv"], env=e, capture_output=True, text=True)
    got = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    assert {k: got.get(k) for k in TELEMETRY} == TELEMETRY
    assert "HF_HUB_OFFLINE" not in got           # loading needs the network
