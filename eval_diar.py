#!/usr/bin/env python3
"""DER of one or more hypothesis RTTMs against a reference RTTM.

  eval_diar.py der --ref ref.rttm --hyp nemotron=a.rttm --hyp pyannote=b.rttm
  eval_diar.py labels2rttm labels.txt ref.rttm     (Audacity label export)

DER is split into missed speech, false alarm and speaker confusion
(pyannote.metrics). Two collars are reported:
  collar 0      no tolerance
  collar 0.25   +-0.25 s around every reference boundary, i.e. the NIST
                md-eval convention; pyannote.metrics counts the collar as the
                total width centred on the boundary, so it is called with 0.5.
Evaluation covers the reference extent only (UEM = first start..last end),
so a 5-minute reference can be scored against a full-length hypothesis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COLLARS = [("0", 0.0), ("0.25", 0.5)]  # (label, pyannote collar = 2 x NIST collar)


def read_rttm(path):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER" and float(parts[4]) > 0:
            start = float(parts[3])
            ann[Segment(start, start + float(parts[4]))] = parts[7]
    return ann


def der_table(ref_path, hyps):
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = read_rttm(ref_path)
    extent = ref.get_timeline().extent()
    uem = Timeline([Segment(extent.start, extent.end)])
    rows = []
    for name, path in hyps:
        hyp = read_rttm(path)
        for label, collar in COLLARS:
            metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
            d = metric(ref, hyp, uem=uem, detailed=True)
            total = d["total"] or 1.0
            rows.append({
                "system": name, "collar": label,
                "der": d["diarization error rate"],
                "miss": d["missed detection"] / total,
                "fa": d["false alarm"] / total,
                "conf": d["confusion"] / total,
                "ref_s": d["total"],
            })
    return rows


def format_table(rows):
    lines = [f"{'System':<12} {'Collar':>6} {'DER':>7} {'Miss':>7} {'FA':>7} {'Conf':>7} {'Ref s':>7}"]
    for r in rows:
        lines.append(f"{r['system']:<12} {r['collar']:>6} {r['der']:7.1%} {r['miss']:7.1%} "
                     f"{r['fa']:7.1%} {r['conf']:7.1%} {r['ref_s']:7.1f}")
    return "\n".join(lines)


def labels2rttm(src, dst, recording_id="ref"):
    """Audacity label track export (start<TAB>end<TAB>label) -> RTTM."""
    out = []
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0].startswith("\\"):
            continue  # skip spectral-selection lines
        start, end, label = float(parts[0]), float(parts[1]), parts[2].strip()
        if end > start and label:
            out.append(f"SPEAKER {recording_id} 1 {start:.3f} {end - start:.3f} "
                       f"<NA> <NA> {label} <NA> <NA>")
    Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("der")
    p.add_argument("--ref", required=True)
    p.add_argument("--hyp", action="append", required=True, help="NAME=PATH")
    p = sub.add_parser("labels2rttm")
    p.add_argument("labels")
    p.add_argument("rttm")
    args = ap.parse_args(argv)
    if args.cmd == "labels2rttm":
        print(f"{labels2rttm(args.labels, args.rttm)} Segmente")
        return 0
    hyps = []
    for h in args.hyp:
        name, _, path = h.partition("=")
        if not Path(path).exists():
            print(f"{name}: fehlt", file=sys.stderr)
            continue
        hyps.append((name, path))
    print(format_table(der_table(args.ref, hyps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
