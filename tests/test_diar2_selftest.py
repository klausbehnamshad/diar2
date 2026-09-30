"""Dry run of selftest_mac.sh on Linux with stand-ins for macOS tools and models.

Checks the script's own logic (voices, composing, cpu/metal comparison,
device default, report without paths, offline run) - not the models.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("ffmpeg", "shasum")), reason="ffmpeg/shasum fehlen")
pytest.importorskip("pyannote.metrics")
pytest.importorskip("numpy")

STUBS = {
    "uname": '#!/bin/sh\ncase "$1" in -m) echo arm64;; *) echo Darwin;; esac\n',
    "sw_vers": "#!/bin/sh\necho 15.0\n",
    "sysctl": '#!/bin/sh\ncase "$2" in hw.memsize) echo 17179869184;; *) echo "Apple M3";; esac\n',
    "say": """#!/bin/bash
if [ "$1" = -v ] && [ "$2" = '?' ]; then
  printf 'Anna                de_DE    # Hallo, ich heiße Anna.\\n'
  printf 'Markus (Enhanced)   de_DE    # Hallo, ich heiße Markus.\\n'
  printf 'Samantha            en_US    # Hello, my name is Samantha.\\n'
  exit 0
fi
voice=$2; out=$4; text=$5
f=300; [ "$voice" = Anna ] && f=200
d=$(( ${#text} / 15 + 1 ))
ffmpeg -loglevel error -y -f lavfi -i "sine=frequency=$f:duration=$d" "$out"
""",
    # German macOS format ("1,50 real"), as measured on the target Mac
    "faketime": """#!/bin/bash
shift  # -l
"$@"; rc=$?
printf '        1,50 real  1,00 user  0,10 sys\\n   104857600  maximum resident set size\\n   125829120  peak memory footprint\\n' >&2
exit $rc
""",
    # writes the reference-like turns so DER and word scoring have input
    "nemo-speech": """#!/bin/bash
[ "$1" = --version ] && { echo "NeMo-Speech.cpp 0.1.0"; exit 0; }
[ "$1" = doctor ] && { echo "[1] metal Apple M3"; exit 0; }
if [ "$1" = diarize ] && [ "$2" = --help ]; then
  printf '  --device, --backend DEVICE\\n  --preset NAME   V3: v3-streaming/v3-offline\\n'
  printf '  --format text|json|rttm\\n  --recording-id NAME\\n  -o, --output PATH\\n  --force\\n'
  exit 0
fi
out=""; while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift 2;; *) shift;; esac; done
printf 'SPEAKER x 1 0.500 3.000 <NA> <NA> speaker_1 <NA> <NA>\\n' > "$out"
printf 'SPEAKER x 1 4.000 5.000 <NA> <NA> speaker_2 <NA> <NA>\\n' >> "$out"
""",
}

FAKE_MLX = '''
def transcribe(audio, *, path_or_hf_repo, language=None, word_timestamps=False, verbose=None):
    ws = [("Guten", 0.6, 0.9), ("Tag.", 1.0, 1.4), ("Ich", 4.1, 4.3), ("bin", 4.4, 4.6),
          ("aufgewachsen.", 4.7, 5.5)]
    return {"language": language, "segments": [{"start": 0.6, "end": 5.5, "text": "x",
            "words": [{"word": w, "start": s, "end": e, "probability": 0.9} for w, s, e in ws]}]}
'''
FAKE_WHISPERX = '''
import os
def load_align_model(language_code, device, **kw):
    if os.environ.get("HF_HUB_OFFLINE") and os.environ.get("FAKE_NEEDS_NET"):
        raise RuntimeError("no network")
    return object(), {}
def load_audio(p): return None
def align(segments, *a, **k):
    import mlx_whisper
    return mlx_whisper.transcribe(None, path_or_hf_repo="x", word_timestamps=True)
'''
FAKE_PYANNOTE = '''
class _Seg:
    def __init__(s, a, b): s.start, s.end = a, b
class _Ann:
    def itertracks(self, yield_label=True):
        yield _Seg(0.5, 3.4), None, "SPEAKER_00"
        yield _Seg(4.0, 9.0), None, "SPEAKER_01"
class Pipeline:
    @classmethod
    def from_pretrained(cls, name, token=None): return cls()
    def __call__(self, audio, **kw): return _Ann()
'''
FAKE_TORCH = "def from_numpy(a):\n    return a\n"


def test_selftest_dry_run(tmp_path):
    home = tmp_path / "home"
    (home / ".config/diar2").mkdir(parents=True)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in STUBS.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    fakes = tmp_path / "fakes"
    (fakes / "pyannote" / "audio").mkdir(parents=True)
    (fakes / "mlx_whisper.py").write_text(FAKE_MLX)
    (fakes / "whisperx.py").write_text(FAKE_WHISPERX)
    (fakes / "torch.py").write_text(FAKE_TORCH)
    (fakes / "pyannote" / "audio" / "__init__.py").write_text(FAKE_PYANNOTE)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    sha = subprocess.run(["shasum", "-a", "256", str(model)], capture_output=True,
                         text=True).stdout.split()[0]
    config = home / ".config/diar2/paths.env"
    config.write_text(f"DIAR2_PYTHON={sys.executable}\nDIAR2_NEMO={bin_ / 'nemo-speech'}\n"
                      f"DIAR2_MODEL={model}\nDIAR2_MODEL_SHA256={sha}\n"
                      f"DIAR2_NEMO_COMMIT=0f706e43cf1f\nDIAR2_DEVICE_DEFAULT=cpu\n")
    env = dict(os.environ, HOME=str(home), PATH=f"{bin_}:{os.environ['PATH']}",
               PYTHONPATH=str(fakes), DIAR2_TIME=str(bin_ / "faketime"))
    r = subprocess.run(["bash", str(HERE / "selftest_mac.sh")], env=env, capture_output=True,
                       text=True, timeout=600)
    report = home / "Downloads/_outputs/diar2_selftest.txt"
    assert report.exists(), r.stderr + r.stdout
    text = report.read_text()
    assert str(home) not in text and str(tmp_path) not in text   # no paths in the report
    assert "Stimmen: Interviewer=Anna, Interviewee=Markus (Enhanced)" in text
    assert "Vergleich: gleich" in text and "Empfehlung Gerät: metal" in text
    assert "DIAR2_DEVICE_DEFAULT=metal" in config.read_text()
    assert config.read_text().count("DIAR2_DEVICE_DEFAULT") == 1
    assert "s pro Audiominute" in text and "peak footprint 120 MB" in text
    assert "Laufzeit 1.50 s (0.9 s pro Audiominute)" in text
    assert "d_nemotron         1.50            0.9" in text
    for sysname in ("nemotron", "pyannote", "geglaettet"):
        assert sum(sysname in line and "%" in line for line in text.splitlines()) == 2
    assert "[ok]     diar2 offline Lauf" in text
    assert "[ok]     offline: WhisperX-Alignment aus dem Cache" in text
    assert "Wörter mit richtigem Sprecher" in text
    assert "Fehler: 0" in text, text
