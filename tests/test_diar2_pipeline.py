"""End-to-end run of diar2.sh with stand-in models (no downloads).

mlx_whisper and whisperx are replaced by tiny fake modules on PYTHONPATH,
nemo-speech by a script that writes a fixed RTTM. ffmpeg is real.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg fehlt")

FAKE_MLX = '''
def transcribe(audio, *, path_or_hf_repo, language=None, word_timestamps=False, verbose=None):
    assert word_timestamps
    words = [("Wie", 0.2, 0.5), ("war", 0.55, 0.8), ("das?", 0.85, 1.2),
             ("Das", 1.8, 2.0), ("war", 2.05, 2.3), ("schwer.", 2.35, 2.9)]
    return {"language": language, "segments": [
        {"start": 0.2, "end": 1.2, "text": " Wie war das?",
         "words": [{"word": " " + w, "start": s, "end": e, "probability": 0.8} for w, s, e in words[:3]]},
        {"start": 1.8, "end": 2.9, "text": " Das war schwer.",
         "words": [{"word": " " + w, "start": s, "end": e, "probability": 0.7} for w, s, e in words[3:]]},
    ]}
'''

FAKE_WHISPERX = '''
import os
def load_align_model(language_code, device, **kw):
    if os.environ.get("FAKE_ALIGN_FAIL"):
        raise RuntimeError("simulated align crash")
    return object(), {"language": language_code}
def load_audio(path):
    return [0.0]
def align(segments, model, meta, audio, device, return_char_alignments=False):
    out = []
    for s in segments:
        toks = s["text"].split()
        step = (s["end"] - s["start"]) / len(toks)
        words = [{"word": t, "start": s["start"] + k * step,
                  "end": s["start"] + (k + 1) * step, "score": 0.95}
                 for k, t in enumerate(toks)]
        if toks[-1] == "schwer.":  # WhisperX leaves a word unaligned
            del words[-1]["start"], words[-1]["end"]
        out.append({"start": s["start"], "end": s["end"], "text": s["text"], "words": words})
    return {"segments": out}
'''

FAKE_NEMO = '''#!/usr/bin/env bash
case "$*" in *kaputt*) echo "simulated nemo failure" >&2; exit 3 ;; esac
out=""; dev=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out=$2; shift 2;; --device) dev=$2; shift 2;; *) shift;; esac
done
echo "$dev" > "$(dirname "$out")/device_used"
echo "${LC_ALL-unset}" > "$(dirname "$out")/nemo_locale"
printf 'SPEAKER x 1 0.100 1.200 <NA> <NA> speaker_1 <NA> <NA>\\n' > "$out"
printf 'SPEAKER x 1 1.700 1.300 <NA> <NA> speaker_2 <NA> <NA>\\n' >> "$out"
'''


@pytest.fixture()
def env(tmp_path):
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "mlx_whisper.py").write_text(FAKE_MLX)
    (fakes / "whisperx.py").write_text(FAKE_WHISPERX)
    nemo = tmp_path / "nemo-speech"
    nemo.write_text(FAKE_NEMO)
    nemo.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=3", "-c:a", "aac",
                    str(downloads / "probe interview.mp4")], check=True)
    e = dict(os.environ)
    e.update({
        "HOME": str(tmp_path), "PYTHONPATH": str(fakes), "DIAR2_PYTHON": sys.executable,
        "DIAR2_NEMO": str(nemo), "DIAR2_MODEL": str(model),
        "DIAR2_CONFIG": str(tmp_path / "none.env"),
    })
    return tmp_path, e


def run_diar2(env, *args, shell="bash"):
    home, e = env
    if shell == "bash":
        cmd = ["bash", str(HERE / "diar2.sh"), *args]
    else:  # sourced like in ~/.zshrc, then called as a function
        quoted = " ".join(f"'{a}'" for a in args)
        cmd = [shell, "-c", f"source '{HERE / 'diar2.sh'}'; diar2 {quoted}"]
    return subprocess.run(cmd, env=e, capture_output=True, text=True)


def test_full_run_writes_outputs_and_stages(env):
    home, _ = env
    r = run_diar2(env, "probe interview.mp4")
    assert r.returncode == 0, r.stderr + r.stdout
    out = home / "Downloads" / "diar2_ausgang" / "probe interview"
    for ext in ("diar2.json", "diar2.srt", "diar2.txt", "hoerliste.txt", "diar2.stages.tsv",
                "nemotron.rttm"):
        assert (out / f"probe interview.{ext}").exists(), ext
    res = json.loads((out / "probe interview.diar2.json").read_text())
    assert res["alignment"] == "whisperx"
    last = res["words"][-1]
    assert last["text"] == "schwer." and last["time_source"] == "mlx"
    assert (last["start"], last["end"]) == (2.35, 2.9)  # mlx time of the same word
    assert {w["time_source"] for w in res["words"][:-1]} == {"whisperx"}
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee"]
    assert res["meta"]["input"] == "probe interview.mp4"
    assert len(res["meta"]["input_sha256"]) == 64
    assert res["meta"]["device"] == "cpu"
    stages = (out / "probe interview.diar2.stages.tsv").read_text().splitlines()
    assert [l.split("\t")[0] for l in stages[1:]] == [
        "a_audio", "b_whisper", "c_align", "d_nemotron", "f_merge"]
    wav = home / "Downloads/diar2_ausgang/.work/probe interview/probe interview.16k.wav"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=sample_rate,channels,codec_name", "-of", "csv=p=0",
                            str(wav)], capture_output=True, text=True).stdout.strip()
    assert probe == "pcm_s16le,16000,1"


def test_align_failure_falls_back(env):
    home, e = env
    e["FAKE_ALIGN_FAIL"] = "1"
    r = run_diar2(env, "probe interview.mp4")
    assert r.returncode == 0, r.stderr
    assert "alignment=fallback" in r.stdout
    res = json.loads((home / "Downloads/diar2_ausgang/probe interview/probe interview.diar2.json")
                     .read_text())
    assert res["alignment"] == "fallback"
    assert {w["time_source"] for w in res["words"]} == {"mlx"}


def test_second_run_resumes_and_device_env(env):
    home, e = env
    assert run_diar2(env, "probe interview.mp4").returncode == 0
    e["DIAR2_DEVICE"] = "metal"
    r = run_diar2(env, "probe interview.mp4")
    assert r.returncode == 0, r.stderr
    assert "a_audio      vorhanden" in r.stdout and "b_whisper    vorhanden" in r.stdout
    work = home / "Downloads/diar2_ausgang/.work/probe interview"
    assert (work / "device_used").read_text().strip() == "metal"


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_sourced_function_absolute_path(env, shell):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} fehlt")
    home, _ = env
    r = run_diar2(env, str(home / "Downloads" / "probe interview.mp4"), shell=shell)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "fertig:" in r.stdout
    assert "  Wörter 6, Turns 2" in r.stdout


def test_missing_file_and_bad_device(env):
    home, e = env
    r = run_diar2(env, "gibtsnicht.mp4")
    assert r.returncode != 0 and "nicht gefunden" in r.stderr
    e["DIAR2_DEVICE"] = "cuda"
    r = run_diar2(env, "probe interview.mp4")
    assert r.returncode != 0 and "cpu oder metal" in r.stderr


GERMAN_TIME = """#!/bin/bash
# macOS /usr/bin/time -l as printed under a German locale unless LC_ALL=C
shift  # -l
echo "${LC_ALL-unset}" >> "$TIME_LOCALE_LOG"
"$@"; rc=$?
if [ "${LC_ALL:-}" = C ] && [ -z "${FORCE_COMMA:-}" ]; then r="19.27"; u="3.10"; else r="19,27"; u="3,10"; fi
printf '        %s real         %s user         0,20 sys\\n' "$r" "$u" >&2
printf '           524288000  maximum resident set size\\n           612368384  peak memory footprint\\n' >&2
exit $rc
"""


@pytest.mark.parametrize("force_comma", ["", "1"])
def test_run_stage_with_german_time_output(env, force_comma):
    # uname says Darwin, time prints "19,27 real" unless run under LC_ALL=C;
    # FORCE_COMMA makes it print a comma anyway to exercise the parser
    home, e = env
    stubs = home / "darwin"
    stubs.mkdir()
    (stubs / "uname").write_text("#!/bin/sh\necho Darwin\n")
    (stubs / "time").write_text(GERMAN_TIME)
    for f in ("uname", "time"):
        (stubs / f).chmod(0o755)
    e.update({"PATH": f"{stubs}:{e['PATH']}", "DIAR2_TIME": str(stubs / "time"),
              "TIME_LOCALE_LOG": str(home / "time_locale"), "LC_ALL": "C.UTF-8",
              "FORCE_COMMA": force_comma})
    r = run_diar2(env, "probe interview.mp4")
    assert r.returncode == 0, r.stderr + r.stdout
    out = home / "Downloads" / "diar2_ausgang" / "probe interview"
    rows = [l.split("\t") for l in
            (out / "probe interview.diar2.stages.tsv").read_text().splitlines()[1:]]
    assert rows and all(row[1] == "19.27" for row in rows)
    assert all(row[2:4] == ["500", "584"] for row in rows)
    assert "," not in (out / "probe interview.diar2.stages.tsv").read_text()
    meta = json.loads((out / "probe interview.diar2.json").read_text())["meta"]
    assert all(st["seconds"] == 19.27 and st["max_rss_mb"] == 500 and st["exit"] == 0
               for st in meta["stages"])
    # time ran under LC_ALL=C, the measured command kept the caller's locale
    assert set((home / "time_locale").read_text().split()) == {"C"}
    # f_merge.log ends with the macOS time output; the summary is still shown
    log = (home / "Downloads/diar2_ausgang/.work/probe interview/f_merge.log").read_text()
    assert log.rstrip().endswith("peak memory footprint")
    summary = [l for l in r.stdout.splitlines() if l.startswith("  Wörter ")]
    assert summary == ["  Wörter 6, Turns 2, Hörliste 0, Alignment whisperx"]
    assert "peak memory footprint" not in r.stdout
    work = home / "Downloads/diar2_ausgang/.work/probe interview"
    assert (work / "nemo_locale").read_text().strip() == "C.UTF-8"
