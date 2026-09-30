# SPDX-License-Identifier: Apache-2.0
"""Tests for eval_diar.py and selftest_tools.py (synthetic data only)."""

import array
import math
import subprocess
import sys
import wave
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import selftest_tools as st  # noqa: E402


def tone(path, secs, freq, lead=0.2, tail=0.3):
    n = int(secs * 16000)
    a = array.array("h", [0] * int(lead * 16000))
    a.extend(int(8000 * math.sin(2 * math.pi * freq * k / 16000)) for k in range(n))
    a.extend([0] * int(tail * 16000))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(a.tobytes())


def test_compose_places_utterances_and_overlay(tmp_path, capsys):
    tone(tmp_path / "a.wav", 2.0, 200)
    tone(tmp_path / "b.wav", 3.0, 400)
    tone(tmp_path / "c.wav", 0.5, 300)
    plan = tmp_path / "plan.tsv"
    plan.write_text(f"Interviewer\t{tmp_path/'a.wav'}\tseq\t0.0\n"
                    f"Interviewee\t{tmp_path/'b.wav'}\tseq\t0.7\n"
                    f"Interviewer\t{tmp_path/'c.wav'}\toverlay\t1.0\n")
    st.compose(plan, tmp_path / "out.wav", tmp_path / "ref.rttm")
    segs = st._rttm(tmp_path / "ref.rttm")
    (s1, e1, l1), (s2, e2, l2), (s3, e3, l3) = segs
    assert (l1, l2, l3) == ("Interviewer", "Interviewee", "Interviewer")
    assert s1 == pytest.approx(0.5, abs=0.02)            # silence trimmed
    assert e1 - s1 == pytest.approx(2.0, abs=0.03)
    assert s2 - e1 == pytest.approx(0.7, abs=0.02)
    assert s3 == pytest.approx(s2 + 1.0, abs=0.01)        # overlay inside b
    assert float(capsys.readouterr().out) == pytest.approx(e2 + 0.5, abs=0.01)


def test_timel_parses_macos_output(tmp_path, capsys):
    log = tmp_path / "t.log"
    log.write_text("some output\n        12.34 real         8.00 user         1.00 sys\n"
                   "           524288000  maximum resident set size\n"
                   "                   0  average shared memory size\n"
                   "           612368384  peak memory footprint\n")
    st.timel(log)
    assert capsys.readouterr().out.split() == ["12.34", "500", "584"]


def test_timel_parses_german_decimal_comma(tmp_path, capsys):
    log = tmp_path / "t.log"
    log.write_text("        19,27 real         3,10 user         0,20 sys\n"
                   "           524288000  maximum resident set size\n")
    st.timel(log)
    assert capsys.readouterr().out.split() == ["19.27", "500", "-"]


def test_timel_integer_seconds(tmp_path, capsys):
    log = tmp_path / "t.log"
    log.write_text("        57 real         3 user\n")
    st.timel(log)
    assert capsys.readouterr().out.split()[0] == "57"


def test_selftest_stage_table_accepts_comma(tmp_path):
    text = (HERE / "selftest_mac.sh").read_text(encoding="utf-8")
    block = text[text.index("# --- stufentabelle: begin"):text.index("# --- stufentabelle: end")]
    tsv = tmp_path / "stages.tsv"
    tsv.write_text("stufe\tsekunden\tmax_rss_mb\tpeak_footprint_mb\texit\n"
                   "b_whisper\t19,27\t500\t584\t0\n"
                   "d_nemotron\t8.5\t270\t300\t0\n"
                   "f_merge\t-\t-\t-\t0\n")
    script = (f'set -uo pipefail\nPY="{sys.executable}"\nmins=2.00\n'
              'say_() { echo "$*"; }\n'
              f'{block}\nstage_table "{tsv}"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "Error" not in r.stderr, r.stderr
    lines = r.stdout.splitlines()
    assert lines[1].split() == ["b_whisper", "19.27", "9.6", "500", "584", "0"]
    assert lines[2].split() == ["d_nemotron", "8.5", "4.2", "270", "300", "0"]
    assert lines[3].split() == ["f_merge", "-", "-", "-", "-", "0"]


def test_compare_equal_and_different(tmp_path):
    a = tmp_path / "a.rttm"
    b = tmp_path / "b.rttm"
    a.write_text("SPEAKER x 1 0.00 1.00 <NA> <NA> speaker_1 <NA> <NA>\n"
                 "SPEAKER x 1 1.20 2.00 <NA> <NA> speaker_2 <NA> <NA>\n")
    b.write_text("SPEAKER x 1 0.00 1.08 <NA> <NA> speaker_1 <NA> <NA>\n"
                 "SPEAKER x 1 1.20 2.00 <NA> <NA> speaker_2 <NA> <NA>\n")
    assert st.compare(a, b) == 0
    b.write_text("SPEAKER x 1 0.00 1.00 <NA> <NA> speaker_1 <NA> <NA>\n")
    assert st.compare(a, b) == 1


def test_eval_der_perfect_and_confused(tmp_path):
    pytest.importorskip("pyannote.metrics")
    import eval_diar

    ref = tmp_path / "ref.rttm"
    ref.write_text("SPEAKER r 1 0.0 10.0 <NA> <NA> A <NA> <NA>\n"
                   "SPEAKER r 1 10.0 10.0 <NA> <NA> B <NA> <NA>\n")
    same = tmp_path / "same.rttm"
    same.write_text("SPEAKER h 1 0.0 10.0 <NA> <NA> speaker_2 <NA> <NA>\n"
                    "SPEAKER h 1 10.0 10.0 <NA> <NA> speaker_1 <NA> <NA>\n")
    shifted = tmp_path / "shifted.rttm"
    shifted.write_text("SPEAKER h 1 0.0 12.0 <NA> <NA> x <NA> <NA>\n"
                       "SPEAKER h 1 12.0 8.0 <NA> <NA> y <NA> <NA>\n")
    rows = eval_diar.der_table(ref, [("same", same), ("shifted", shifted)])
    by = {(r["system"], r["collar"]): r for r in rows}
    assert by[("same", "0")]["der"] == pytest.approx(0.0)
    assert by[("shifted", "0")]["conf"] == pytest.approx(2.0 / 20.0)
    assert by[("shifted", "0.25")]["der"] < by[("shifted", "0")]["der"]
    assert "DER" in eval_diar.format_table(rows)


def test_labels2rttm_and_cli(tmp_path):
    pytest.importorskip("pyannote.metrics")
    labels = tmp_path / "labels.txt"
    labels.write_text("0.5\t3.0\tInterviewer\n\\\t0\t0\n3.2\t9.0\tInterviewee\n")
    out = tmp_path / "ref.rttm"
    r = subprocess.run([sys.executable, str(HERE / "eval_diar.py"), "labels2rttm",
                        str(labels), str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "2 Segmente" in r.stdout
    assert out.read_text().splitlines()[1].split()[7] == "Interviewee"
    r = subprocess.run([sys.executable, str(HERE / "eval_diar.py"), "der", "--ref", str(out),
                        "--hyp", f"self={out}"], capture_output=True, text=True)
    assert r.returncode == 0 and "self" in r.stdout and "0.0%" in r.stdout
