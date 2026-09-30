"""diar2 ohne Argument: Stapel über den festen Eingang (Stub-Stufen, echtes ffmpeg)."""

import hashlib
import json
import shutil
import subprocess

import pytest

from test_diar2_pipeline import env, run_diar2  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg fehlt")


def make_audio(path, freq=300, fmt=None):
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
           f"sine=frequency={freq}:duration=3"]
    if fmt:
        cmd += ["-f", fmt]
    subprocess.run(cmd + [str(path)], check=True)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_lines(stdout):
    lines = stdout.splitlines()
    k = max(i for i, line in enumerate(lines) if line.startswith("Übersicht"))
    return [line.strip() for line in lines[k + 1:] if line.strip()]


def test_batch_skips_done_continues_after_failure(env):  # noqa: F811
    home, _ = env
    inbox = home / "Downloads" / "diar2_eingang"
    outbox = home / "Downloads" / "diar2_ausgang"
    inbox.mkdir()
    make_audio(inbox / "1 kaputt.m4a")              # nemo stub fails on "kaputt"
    make_audio(inbox / "2 schon fertig.MOV", fmt="mov")
    make_audio(inbox / "3 neu eins.wav")
    (inbox / "notiz.txt").write_text("keine Aufnahme")
    done = outbox / "2 schon fertig"
    done.mkdir(parents=True)
    (done / "2 schon fertig.diar2.json").write_text("{}")
    before = {p.name: sha(p) for p in inbox.iterdir()}

    r = run_diar2(env)
    assert r.returncode == 1, r.stdout + r.stderr
    lines = summary_lines(r.stdout)
    assert len(lines) == 3, lines
    assert lines[0].startswith("FEHLER") and "1 kaputt" in lines[0]
    assert lines[0].endswith(str(outbox / ".work" / "1 kaputt" / "d_nemotron.log"))
    assert "simulated nemo failure" in (outbox / ".work/1 kaputt/d_nemotron.log").read_text()
    assert lines[1].startswith("übersprungen") and "2 schon fertig" in lines[1]
    assert lines[2].startswith("fertig") and "3 neu eins" in lines[2]
    # the third file is finished although the first one failed
    new = outbox / "3 neu eins"
    for ext in ("diar2.txt", "diar2.srt", "diar2.json", "hoerliste.txt", "diar2.stages.tsv",
                "nemotron.rttm", "diar2.rttm"):
        assert (new / f"3 neu eins.{ext}").exists(), ext
    assert (outbox / ".work/3 neu eins/3 neu eins.16k.wav").exists()
    assert (done / "2 schon fertig.diar2.json").read_text() == "{}"   # untouched
    assert not (outbox / "1 kaputt" / "1 kaputt.diar2.json").exists()
    # originals are only read: same files, same checksums
    assert {p.name: sha(p) for p in inbox.iterdir()} == before

    # second run: the finished file is skipped now, the broken one is retried
    r = run_diar2(env)
    assert r.returncode == 1
    assert [line.split()[0] for line in summary_lines(r.stdout)] == [
        "FEHLER", "übersprungen", "übersprungen"]


def test_batch_name_with_spaces_and_umlaut(env):  # noqa: F811
    home, _ = env
    inbox = home / "Downloads" / "diar2_eingang"
    inbox.mkdir()
    make_audio(inbox / "Gespräch mit Frau K.mp4")
    r = run_diar2(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert summary_lines(r.stdout)[0].startswith("fertig")
    out = home / "Downloads/diar2_ausgang/Gespräch mit Frau K"
    assert (out / "Gespräch mit Frau K.diar2.txt").exists()


def test_empty_inbox_clear_message_exit_zero_and_folders_created(env):  # noqa: F811
    home, _ = env
    r = run_diar2(env)
    assert r.returncode == 0, r.stderr
    assert "keine Datei in" in r.stdout and "diar2_eingang" in r.stdout
    assert (home / "Downloads/diar2_eingang").is_dir()
    assert (home / "Downloads/diar2_ausgang").is_dir()


def test_single_file_prefers_inbox_over_downloads(env):  # noqa: F811
    home, _ = env
    inbox = home / "Downloads" / "diar2_eingang"
    inbox.mkdir()
    make_audio(inbox / "probe interview.mp4", freq=500)
    r = run_diar2(env, "probe interview.mp4")
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads((home / "Downloads/diar2_ausgang/probe interview/probe interview.diar2.json")
                     .read_text())
    assert res["meta"]["input_sha256"] == sha(inbox / "probe interview.mp4")
    assert res["meta"]["input_sha256"] != sha(home / "Downloads" / "probe interview.mp4")


def test_single_file_failure_exit_code(env):  # noqa: F811
    home, _ = env
    make_audio(home / "Downloads" / "kaputt.wav")
    r = run_diar2(env, "kaputt.wav")
    assert r.returncode == 1
    assert summary_lines(r.stdout)[0].startswith("FEHLER")


def test_env_overrides_folders(env):  # noqa: F811
    home, e = env
    e["DIAR2_IN"] = str(home / "rein")
    e["DIAR2_OUT"] = str(home / "raus")
    (home / "rein").mkdir()
    make_audio(home / "rein" / "x.wav")
    r = run_diar2(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (home / "raus/x/x.diar2.json").exists()
    assert not (home / "Downloads/diar2_ausgang").exists()


def test_runs_under_caffeinate_on_macos(env):  # noqa: F811
    home, e = env
    stubs = home / "mac"
    stubs.mkdir()
    (stubs / "uname").write_text("#!/bin/sh\necho Darwin\n")
    (stubs / "caffeinate").write_text(
        f'#!/bin/bash\necho "$1" > "{home}/caffeinate_args"\nshift\nexec "$@"\n')
    for f in ("uname", "caffeinate"):
        (stubs / f).chmod(0o755)
    e["PATH"] = f"{stubs}:{e['PATH']}"
    r = run_diar2(env)   # empty inbox: enough to see the re-exec
    assert r.returncode == 0, r.stderr
    assert (home / "caffeinate_args").read_text().strip() == "-i"
    assert "keine Datei in" in r.stdout
