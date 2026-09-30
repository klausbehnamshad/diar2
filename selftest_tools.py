#!/usr/bin/env python3
"""Helpers for selftest_mac.sh (standard library only).

  compose PLAN.tsv OUT.wav REF.rttm   place utterances on a timeline -> WAV + reference RTTM
  timel LOG                           parse macOS /usr/bin/time -l -> "sec rss_mb footprint_mb"
  compare A.rttm B.rttm               same segments within one Nemotron frame (80 ms)?
  score RESULT.diar2.json REF.rttm SCRIPT.txt   word speaker accuracy + transcript match
"""

from __future__ import annotations

import array
import difflib
import json
import re
import sys
import wave
from pathlib import Path

RATE = 16000
FRAME_S = 0.08  # Nemotron frame (80 ms)


def _read(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    return a


def _trim(a, thresh=500):
    idx = [i for i in range(0, len(a), 80) if abs(a[i]) > thresh or
           max(abs(x) for x in a[i:i + 80]) > thresh]
    if not idx:
        return a
    return a[max(idx[0] - 80, 0): min(idx[-1] + 160, len(a))]


def compose(plan, out_wav, ref_rttm):
    """plan lines: SPEAKER<TAB>WAV<TAB>seq|overlay<TAB>SECONDS
    seq: start SECONDS after the previous seq item ends;
    overlay: start SECONDS after the previous seq item starts (overlapping)."""
    items, t, prev_start = [], 0.5, 0.5
    for line in Path(plan).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        spk, wav, mode, secs = line.split("\t")
        a = _trim(_read(wav))
        dur = len(a) / RATE
        if mode == "overlay":
            start = prev_start + float(secs)
        else:
            start = t + float(secs)
            prev_start, t = start, start + dur
        items.append((start, dur, spk, a))
    total = max(s + d for s, d, _, _ in items) + 0.5
    mix = array.array("i", [0]) * int(total * RATE)
    for start, dur, _, a in items:
        o = int(start * RATE)
        for k, x in enumerate(a):
            mix[o + k] += x
    out = array.array("h", (max(-32768, min(32767, x)) for x in mix))
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(out.tobytes())
    Path(ref_rttm).write_text("".join(
        f"SPEAKER ref 1 {s:.3f} {d:.3f} <NA> <NA> {spk} <NA> <NA>\n"
        for s, d, spk, _ in sorted(items, key=lambda x: x[0])), encoding="utf-8")
    print(f"{total:.3f}")


def timel(log):
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    real = re.search(r"([\d.]+) real", text)
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    foot = re.search(r"(\d+)\s+peak memory footprint", text)
    fmt = lambda m, div: f"{int(m.group(1)) / div:.0f}" if m else "-"  # noqa: E731
    print(real.group(1) if real else "-", fmt(rss, 1048576), fmt(foot, 1048576))


def _rttm(path):
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(segs)


def compare(a, b):
    sa, sb = _rttm(a), _rttm(b)
    if len(sa) != len(sb):
        print(f"verschieden: {len(sa)} gegen {len(sb)} Segmente")
        return 1
    if [s[2] for s in sa] != [s[2] for s in sb]:
        print("verschieden: Sprecherfolge weicht ab")
        return 1
    delta = max([max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(sa, sb)] + [0.0])
    if delta > FRAME_S + 1e-6:
        print(f"verschieden: Grenzen weichen bis {delta:.3f} s ab")
        return 1
    print(f"gleich: {len(sa)} Segmente, max. Grenzabweichung {delta:.3f} s")
    return 0


def _norm(text):
    return re.findall(r"[a-zäöüß0-9]+", text.lower())


def score(result_json, ref_rttm, script_txt):
    res = json.loads(Path(result_json).read_text(encoding="utf-8"))
    ref = _rttm(ref_rttm)
    ok = n = 0
    for w in res["words"]:
        mid = (w["start"] + w["end"]) / 2
        active = {s[2] for s in ref if s[0] <= mid < s[1]}
        if len(active) != 1:
            continue
        n += 1
        ok += w["speaker"] == active.pop()
    hyp = _norm(" ".join(w["text"] for w in res["words"]))
    want = _norm(Path(script_txt).read_text(encoding="utf-8"))
    ratio = difflib.SequenceMatcher(None, want, hyp, autojunk=False).ratio()
    first = res["turns"][0]["speaker"] if res["turns"] else "-"
    print(f"Wörter mit richtigem Sprecher: {ok}/{n} ({(ok / n if n else 0):.1%})")
    print(f"Wortfolge Transkript vs. Skript (difflib-Ratio): {ratio:.3f}")
    print(f"erster Turn: {first} (erwartet Interviewer)")
    print(f"Alignment: {res['alignment']}; Hörliste: {len(res['hoerliste'])} Einträge")


def main(argv):
    cmd, args = argv[0], argv[1:]
    if cmd == "compose":
        compose(*args)
    elif cmd == "timel":
        timel(*args)
    elif cmd == "compare":
        return compare(*args)
    elif cmd == "score":
        score(*args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
