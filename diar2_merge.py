#!/usr/bin/env python3
"""diar2: assign speakers to words, smooth, and write the review list.

Stages f-h of diar2. Standard library only, so it runs and is tested
without any model. Inputs are the word JSON written by diar2_stages.py
(mlx-whisper, optionally re-aligned by WhisperX) and one or two RTTM
files (Nemotron, optionally pyannote as a second opinion).

Every threshold is a module constant below; diar2.sh can override each
one through an environment variable of the same name prefixed DIAR2_
(e.g. DIAR2_SHORT_TURN_S=0.8).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "diar2-merge 0.1.0"

# --- thresholds (seconds unless noted) -------------------------------------
# A word whose midpoint falls in no speaker interval takes the speaker with
# the largest overlap/nearest interval, but only within this gap
# (same rule and value as nemoVoiceRec electron/transcript.cjs).
GAP_TOLERANCE_S = 0.5
# A new sentence starts after . ? ! or after a pause longer than this.
SENTENCE_GAP_S = 1.5
# A backchannel ("mhm", "ja") is at most this long and this many words.
BACKCHANNEL_MAX_S = 1.2
BACKCHANNEL_MAX_WORDS = 2
# A run of differently assigned words at the start or end of a sentence
# splits the sentence (a turn change Whisper did not punctuate) when it has
# at least this many words and lasts at least this long. 0 disables.
SPLIT_MIN_WORDS = 3
SPLIT_MIN_S = 0.8
# A turn shorter than this (and not a backchannel) goes on the review list.
SHORT_TURN_S = 1.0
# Overlap regions (from the RTTM) shorter than this are ignored.
OVERLAP_MIN_S = 0.2
# Disagreement spans with the second opinion shorter than this are ignored;
# neighbouring spans closer than MERGE_GAP_S are merged.
DISAGREE_MIN_S = 0.3
MERGE_GAP_S = 0.5
# The first seconds are always listed to check "first speaker = Interviewer".
CHECK_HEAD_S = 60.0
# SRT cues are cut at sentence ends or after this many seconds.
SRT_MAX_CUE_S = 7.0
# A word without any timestamp (neither WhisperX nor mlx-whisper) is placed
# in at most this window directly before the next timed word, so it never
# spans a pause.
ESTIMATE_MAX_S = 0.5

BACKCHANNEL_TOKENS = {
    "mhm", "mhmm", "hm", "hmm", "mm", "mmh", "aha", "ah", "ja", "jaja", "jo",
    "okay", "ok", "genau", "stimmt", "achso", "ach", "klar", "richtig", "nee",
    "nein", "doch", "gut", "ne", "na",
}

_THRESHOLDS = [
    "GAP_TOLERANCE_S", "SENTENCE_GAP_S", "BACKCHANNEL_MAX_S",
    "BACKCHANNEL_MAX_WORDS", "SPLIT_MIN_WORDS", "SPLIT_MIN_S", "SHORT_TURN_S",
    "OVERLAP_MIN_S", "DISAGREE_MIN_S", "MERGE_GAP_S", "CHECK_HEAD_S",
    "SRT_MAX_CUE_S", "ESTIMATE_MAX_S",
]


def thresholds() -> dict:
    return {name: globals()[name] for name in _THRESHOLDS}


def apply_env_overrides(environ=os.environ) -> None:
    for name in _THRESHOLDS:
        raw = environ.get("DIAR2_" + name)
        if raw:
            cast = int if isinstance(globals()[name], int) else float
            globals()[name] = cast(raw)


# --- data -------------------------------------------------------------------

@dataclass
class Seg:
    start: float
    end: float
    speaker: str


@dataclass(eq=False)
class Word:
    i: int
    text: str
    start: float
    end: float
    score: float | None
    timed: bool = True
    time_source: str = "whisperx"    # whisperx | mlx | geschaetzt
    raw: str | None = None           # speaker by midpoint, before smoothing
    raw_set: tuple = ()              # all speakers active at the midpoint
    speaker: str | None = None       # after smoothing
    overlap: bool = False
    backchannel: bool = False        # Einwurf
    second: str | None = None        # second opinion, mapped to primary labels
    sentence: int = -1

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def as_dict(self, names: dict) -> dict:
        return {
            "i": self.i, "text": self.text,
            "start": round(self.start, 3), "end": round(self.end, 3),
            "score": self.score, "timed": self.timed, "time_source": self.time_source,
            "speaker": names.get(self.speaker, self.speaker),
            "speaker_raw": self.raw, "overlap": self.overlap,
            "overlap_speakers": list(self.raw_set) if self.overlap else [],
            "einwurf": self.backchannel, "second_opinion": self.second,
            "sentence": self.sentence,
        }


@dataclass(eq=False)
class Sentence:
    idx: int
    words: list = field(default_factory=list)
    speaker: str | None = None
    raw_switch: bool = False
    origin: int = -1                 # index before splitting
    answer: bool = False             # "Ja." answering a question: own turn

    @property
    def start(self):
        return self.words[0].start

    @property
    def end(self):
        return self.words[-1].end

    @property
    def text(self):
        return join_words(w.text for w in self.words)


# --- parsing ------------------------------------------------------------------

def read_rttm(path) -> list:
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        if dur > 0:
            segs.append(Seg(start, start + dur, parts[7]))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def write_rttm(path, segs, recording_id="diar2") -> None:
    lines = [
        f"SPEAKER {recording_id} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
        for s in segs if s.end > s.start
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_words(data: dict) -> list:
    """Flatten segments -> words.

    Times come from WhisperX, or from mlx-whisper where diar2_stages.py could
    match the word (time_source). Words still without a time are estimated:
    at most ESTIMATE_MAX_S directly before the next timed word, never across
    the pause before it.
    """
    default_source = "whisperx" if data.get("alignment") == "whisperx" else "mlx"
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words") or []:
            text = (w.get("word") if w.get("word") is not None else w.get("text", "")).strip()
            if not text:
                continue
            score = w.get("score", w.get("probability"))
            s, e = w.get("start"), w.get("end")
            timed = s is not None and e is not None
            words.append(Word(
                len(words), text, s if timed else -1.0, e if timed else -1.0,
                None if score is None else round(float(score), 4), timed=timed,
                time_source=w.get("time_source", default_source) if timed else "geschaetzt"))
    k = 0
    while k < len(words):
        if words[k].timed:
            k += 1
            continue
        j = k
        while j < len(words) and not words[j].timed:
            j += 1
        run = words[k:j]
        prev_end = words[k - 1].end if k > 0 else None
        next_start = words[j].start if j < len(words) else None
        if next_start is not None:
            lo = max(next_start - ESTIMATE_MAX_S, prev_end if prev_end is not None else 0.0, 0.0)
            hi = next_start
        else:
            lo = prev_end if prev_end is not None else 0.0
            hi = lo + ESTIMATE_MAX_S
        if hi <= lo:  # no room: squeeze in right at the boundary
            hi = lo + 0.01 * len(run)
        step = (hi - lo) / len(run)
        for n, w in enumerate(run):
            w.start, w.end = lo + n * step, lo + (n + 1) * step
        k = j
    return words


def join_words(tokens) -> str:
    out = ""
    for t in tokens:
        t = t.strip()
        if not out:
            out = t
        elif re.match(r"^[.,!?;:…)\]»”]", t):
            out += t
        else:
            out += " " + t
    return out


def norm_token(text: str) -> str:
    return re.sub(r"[^\wäöüß]", "", text.lower())


# --- f: speaker per word ------------------------------------------------------

def speakers_at(segs, t) -> tuple:
    return tuple(sorted({s.speaker for s in segs if s.start <= t < s.end}))


def nearest_speaker(segs, start, end):
    best, score = None, -GAP_TOLERANCE_S
    for s in segs:
        value = min(end, s.end) - max(start, s.start)
        if value > score:
            best, score = s.speaker, value
    return best


def assign_raw(words, segs) -> None:
    for w in words:
        active = speakers_at(segs, w.mid)
        w.raw_set = active
        if len(active) > 1:
            w.overlap = True
            # keep a concrete speaker: the one covering most of the word
            cover = {sp: sum(max(0.0, min(w.end, s.end) - max(w.start, s.start))
                             for s in segs if s.speaker == sp) for sp in active}
            w.raw = max(sorted(cover), key=lambda sp: cover[sp])
        elif active:
            w.raw = active[0]
        else:
            w.raw = nearest_speaker(segs, w.start, w.end)


def split_sentences(words) -> list:
    sentences, cur = [], []
    for k, w in enumerate(words):
        if cur and w.start - cur[-1].end > SENTENCE_GAP_S:
            sentences.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?…]$", w.text) or k == len(words) - 1:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    out = []
    for idx, ws in enumerate(sentences):
        s = Sentence(idx, ws)
        for w in ws:
            w.sentence = idx
        out.append(s)
    return out


def is_backchannel_run(ws) -> bool:
    if not ws or len(ws) > BACKCHANNEL_MAX_WORDS:
        return False
    if ws[-1].end - ws[0].start > BACKCHANNEL_MAX_S:
        return False
    return all(norm_token(w.text) in BACKCHANNEL_TOKENS for w in ws)


def _runs(ws):
    """Consecutive runs of equal raw speaker: list of (speaker, [words])."""
    runs = []
    for w in ws:
        if runs and runs[-1][0] == w.raw:
            runs[-1][1].append(w)
        else:
            runs.append((w.raw, [w]))
    return runs


def _majority(ws, include_backchannel=False):
    weight = {}
    for w in ws:
        if w.raw is None or (w.backchannel and not include_backchannel):
            continue
        weight[w.raw] = weight.get(w.raw, 0.0) + max(w.end - w.start, 0.01)
    if not weight:
        return None
    return max(sorted(weight), key=lambda sp: weight[sp])


def _edge_split(sentence):
    """Split point if a long enough run at either edge has another speaker."""
    if SPLIT_MIN_WORDS <= 0:
        return None
    runs = [r for r in _runs(sentence.words) if r[0] is not None]
    if len(runs) < 2:
        return None
    for run, at_start in ((runs[0], True), (runs[-1], False)):
        sp, ws = run
        others = [w for w in sentence.words if w not in ws]
        if not others:
            continue
        if (len(ws) >= SPLIT_MIN_WORDS and ws[-1].end - ws[0].start >= SPLIT_MIN_S
                and len(others) >= SPLIT_MIN_WORDS and _majority(others) not in (None, sp)):
            cut = sentence.words.index(ws[-1]) + 1 if at_start else sentence.words.index(ws[0])
            return cut
    return None


def smooth(sentences) -> list:
    """Majority per sentence, backchannels as Einwurf, edge runs split."""
    # 1. a whole sentence that is only "mhm"/"ja" is an Einwurf
    for s in sentences:
        if is_backchannel_run(s.words):
            for w in s.words:
                w.backchannel = True
    # 2. single backchannel words inside a sentence by another speaker;
    #    an "mhm" under overlap belongs to the speaker who is not talking
    for s in sentences:
        maj = _majority(s.words)
        for w in s.words:
            if (w.overlap and maj in w.raw_set and norm_token(w.text) in BACKCHANNEL_TOKENS
                    and w.end - w.start <= BACKCHANNEL_MAX_S):
                w.raw = next(sp for sp in w.raw_set if sp != maj)
                w.backchannel = True
        for sp, ws in _runs(s.words):
            if sp is not None and sp != maj and is_backchannel_run(ws):
                for w in ws:
                    w.backchannel = True
    # 3. split at unpunctuated turn changes
    out = []
    for s in sentences:
        s.origin = s.idx
        pending = [s]
        while pending:
            cur = pending.pop(0)
            cut = _edge_split(cur)
            if cut is None:
                out.append(cur)
            else:
                a = Sentence(-1, cur.words[:cut], origin=cur.origin, raw_switch=True)
                b = Sentence(-1, cur.words[cut:], origin=cur.origin, raw_switch=True)
                pending[:0] = [a, b]
    for idx, s in enumerate(out):
        s.idx = idx
        for w in s.words:
            w.sentence = idx
    # 4. majority vote; Einwurf words keep their own speaker
    last = None
    for s in out:
        if all(w.backchannel for w in s.words):
            maj = _majority(s.words, include_backchannel=True) or last
        else:
            maj = _majority(s.words) or last
        s.speaker = maj
        speakers = {w.raw for w in s.words if not w.backchannel and w.raw is not None}
        if len(speakers) > 1:
            s.raw_switch = True
        for w in s.words:
            w.speaker = (w.raw or maj) if w.backchannel else maj
        if maj is not None:
            last = maj
    # 5. a sentence of only "ja"/"mhm" that directly follows a question
    #    ("?") of the other speaker is an answer and forms its own turn,
    #    whoever speaks next. Otherwise it is an Einwurf only if the previous
    #    turn's speaker goes on right after it; else it is a short answer
    #    and opens (or continues) a turn of its own speaker.
    prev = None
    for k, s in enumerate(out):
        if s.words and all(w.backchannel for w in s.words):
            before = out[k - 1] if k > 0 else None
            s.answer = (before is not None and before.speaker != s.speaker
                        and before.words[-1].text.rstrip().endswith("?"))
            nxt = next((t.speaker for t in out[k + 1:]
                        if not all(w.backchannel for w in t.words)), None)
            if not s.answer and prev is not None and s.speaker != prev and nxt == prev:
                continue
            for w in s.words:
                w.backchannel = False
                w.speaker = s.speaker
        if s.speaker is not None:
            prev = s.speaker
    return out


# --- turns, overlap regions, second opinion ------------------------------------

def build_turns(sentences) -> list:
    turns = []
    for s in sentences:
        main = [w for w in s.words if not (w.backchannel and w.speaker != s.speaker)]
        if s.words and all(w.backchannel for w in s.words) and turns and turns[-1]["speaker"] != s.speaker:
            turns[-1]["einwuerfe"].append({"speaker": s.speaker, "start": s.start, "end": s.end,
                                          "text": s.text})
            turns[-1]["items"].append(("einwurf", s.speaker, s.text))
            continue
        if turns and turns[-1]["speaker"] == s.speaker:
            t = turns[-1]
        else:
            t = {"speaker": s.speaker, "start": s.start, "end": s.end, "sentences": [],
                 "einwuerfe": [], "items": []}
            turns.append(t)
        t["end"] = max(t["end"], s.end)
        t["sentences"].append(s.idx)
        buf = []
        for w in s.words:
            if w in main:
                buf.append(w.text)
            else:
                if buf:
                    t["items"].append(("text", s.speaker, join_words(buf)))
                    buf = []
                t["einwuerfe"].append({"speaker": w.speaker, "start": w.start, "end": w.end,
                                       "text": w.text})
                t["items"].append(("einwurf", w.speaker, w.text))
        if buf:
            t["items"].append(("text", s.speaker, join_words(buf)))
    return turns


def overlap_regions(segs) -> list:
    """Time spans where two or more RTTM speakers are active."""
    events = []
    for s in segs:
        events += [(s.start, 1, s.speaker), (s.end, -1, s.speaker)]
    events.sort(key=lambda e: (e[0], e[1]))
    active, regions, start = {}, [], None
    for t, kind, sp in events:
        active[sp] = active.get(sp, 0) + kind
        n = sum(1 for v in active.values() if v > 0)
        if n >= 2 and start is None:
            start = t
        elif n < 2 and start is not None:
            if t - start >= OVERLAP_MIN_S:
                regions.append((start, t))
            start = None
    return regions


def map_labels(primary, secondary) -> dict:
    """Map secondary speaker labels onto primary ones by greatest overlap."""
    pairs = {}
    for a in primary:
        for b in secondary:
            ov = min(a.end, b.end) - max(a.start, b.start)
            if ov > 0:
                pairs[(b.speaker, a.speaker)] = pairs.get((b.speaker, a.speaker), 0.0) + ov
    mapping, used_a, used_b = {}, set(), set()
    for (b, a), _ in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        if b in used_b or a in used_a:
            continue
        mapping[b] = a
        used_a.add(a)
        used_b.add(b)
    for s in secondary:
        mapping.setdefault(s.speaker, "zweit_" + s.speaker)
    return mapping


def disagreements(words, second_segs, primary_segs) -> list:
    mapping = map_labels(primary_segs, second_segs)
    mapped = [Seg(s.start, s.end, mapping[s.speaker]) for s in second_segs]
    spans = []
    for w in words:
        active = speakers_at(mapped, w.mid)
        w.second = active[0] if len(active) == 1 else (None if not active else "overlap")
        if w.raw is None or w.second is None or w.overlap or w.second == "overlap":
            continue
        if w.second != w.raw:
            if spans and w.start - spans[-1][1] <= MERGE_GAP_S:
                spans[-1][1] = w.end
            else:
                spans.append([w.start, w.end])
    return [(a, b) for a, b in spans if b - a >= DISAGREE_MIN_S]


# --- g: review list -------------------------------------------------------------

def review_list(sentences, turns, segs, words, names, second_spans=None) -> list:
    items = []
    by_origin = {}
    for s in sentences:
        if s.raw_switch:
            by_origin.setdefault(s.origin, []).append(s)
    for group in by_origin.values():
        items.append({"art": "Sprecherwechsel im Satz", "start": group[0].start,
                      "end": group[-1].end,
                      "text": join_words(w.text for g in group for w in g.words)})
    answers = {s.idx for s in sentences if s.answer}
    for t in turns:
        dur = t["end"] - t["start"]
        # a recognised answer to a question ("Ja.") is expected to be short
        if dur < SHORT_TURN_S and not set(t["sentences"]) <= answers:
            text = " ".join(x[2] for x in t["items"] if x[0] == "text")
            items.append({"art": "sehr kurzer Turn", "start": t["start"], "end": t["end"],
                          "text": f"{names.get(t['speaker'], t['speaker'])}: {text}"})
    for a, b in overlap_regions(segs):
        text = join_words(w.text for w in words if a <= w.mid < b)
        items.append({"art": "Überlappung", "start": a, "end": b, "text": text})
    for a, b in second_spans or []:
        text = join_words(w.text for w in words if a <= w.mid <= b)
        items.append({"art": "Nemotron und pyannote uneins", "start": a, "end": b, "text": text})
    items.sort(key=lambda x: (-(x["end"] - x["start"]), x["start"]))
    return items


# --- names ------------------------------------------------------------------------

def speaker_names(segs, names_opt: str) -> dict:
    """Name speakers by order of first appearance in the RTTM."""
    order = []
    for s in sorted(segs, key=lambda s: s.start):
        if s.speaker not in order:
            order.append(s.speaker)
    if names_opt.strip() in ("", "-"):
        return {sp: sp for sp in order}
    given = [n.strip() for n in names_opt.split(",") if n.strip()]
    names = {}
    for k, sp in enumerate(order):
        names[sp] = given[k] if k < len(given) else f"Sprecher {k + 1}"
    return names


# --- h: writers ---------------------------------------------------------------------

def ts(t: float, sep=".") -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render_txt(turns, names) -> str:
    lines = []
    for t in turns:
        parts = []
        for kind, sp, text in t["items"]:
            parts.append(text if kind == "text" else f"[{names.get(sp, sp)}: {text}]")
        lines.append(f"[{ts(t['start'])[:-2]}] {names.get(t['speaker'], t['speaker'])}: {' '.join(parts)}")
    return "\n\n".join(lines) + "\n"


def render_srt(sentences, names) -> str:
    cues = []
    for s in sentences:
        chunk = []
        for w in s.words:
            if chunk and w.end - chunk[0].start > SRT_MAX_CUE_S:
                cues.append((chunk, s.speaker))
                chunk = []
            chunk.append(w)
        if chunk:
            cues.append((chunk, s.speaker))
    out = []
    for n, (ws, sp) in enumerate(cues, 1):
        text = join_words(w.text if w.speaker == sp or not w.backchannel
                          else f"[{names.get(w.speaker, w.speaker)}: {w.text}]" for w in ws)
        out.append(f"{n}\n{ts(ws[0].start, ',')} --> {ts(ws[-1].end, ',')}\n"
                   f"[{names.get(sp, sp)}] {text}\n")
    return "\n".join(out)


def render_review(items, names, segs, words) -> str:
    order = [sp for sp in names]
    head_end = min(CHECK_HEAD_S, max([s.end for s in segs] + [w.end for w in words] + [0.0]))
    first = order[0] if order else "?"
    lines = [
        "diar2 Hörliste",
        f"Einträge: {len(items)} (plus Kontrolle der ersten {int(CHECK_HEAD_S)} s)",
        "",
        f"KONTROLLE  {ts(0)} - {ts(head_end)}  ({head_end:.1f} s)  "
        f"Zuordnung prüfen: {names.get(first, first)} = {first} (spricht zuerst)"
        + (f", {names.get(order[1], order[1])} = {order[1]}" if len(order) > 1 else ""),
        "",
        "Sortiert nach Dauer, längste zuerst.",
        "",
    ]
    for n, it in enumerate(items, 1):
        dur = it["end"] - it["start"]
        text = it["text"] if len(it["text"]) <= 160 else it["text"][:157] + "..."
        lines.append(f"{n:3d}. {ts(it['start'])} - {ts(it['end'])}  ({dur:5.1f} s)  "
                     f"{it['art']}: {text}")
    return "\n".join(lines) + "\n"


def smoothed_segments(turns, sentences) -> list:
    segs = []
    by_idx = {s.idx: s for s in sentences}
    for t in turns:
        for idx in t["sentences"]:
            s = by_idx[idx]
            main = [w for w in s.words if w.speaker == s.speaker]
            if main:
                segs.append(Seg(main[0].start, main[-1].end, s.speaker))
        for e in t["einwuerfe"]:
            segs.append(Seg(e["start"], e["end"], e["speaker"]))
    segs.sort(key=lambda s: s.start)
    return [s for s in segs if s.speaker is not None]


# --- driver ---------------------------------------------------------------------------

def run(words_json, rttm, out_prefix, second_rttm=None, names_opt="Interviewer,Interviewee",
        meta=None) -> dict:
    data = json.loads(Path(words_json).read_text(encoding="utf-8"))
    segs = read_rttm(rttm)
    words = load_words(data)
    assign_raw(words, segs)
    sentences = smooth(split_sentences(words))
    turns = build_turns(sentences)
    names = speaker_names(segs, names_opt)
    for w in words:
        if w.speaker is not None and w.speaker not in names:
            names[w.speaker] = w.speaker
    second_spans = None
    if second_rttm:
        second_spans = disagreements(words, read_rttm(second_rttm), segs)
    items = review_list(sentences, turns, segs, words, names, second_spans)

    out_prefix = str(out_prefix)
    result = {
        "tool": VERSION,
        "meta": meta or {},
        "alignment": data.get("alignment", "unbekannt"),
        "language": data.get("language"),
        "thresholds": thresholds(),
        "speakers": [{"label": sp, "name": names[sp]} for sp in names],
        "second_opinion": bool(second_rttm),
        "words": [w.as_dict(names) for w in words],
        "sentences": [{"i": s.idx, "start": round(s.start, 3), "end": round(s.end, 3),
                       "speaker": names.get(s.speaker, s.speaker), "text": s.text,
                       "sprecherwechsel_im_satz": s.raw_switch,
                       "kurzantwort": s.answer} for s in sentences],
        "turns": [{"speaker": names.get(t["speaker"], t["speaker"]), "start": round(t["start"], 3),
                   "end": round(t["end"], 3), "sentences": t["sentences"],
                   "einwuerfe": [dict(e, speaker=names.get(e["speaker"], e["speaker"]),
                                      start=round(e["start"], 3), end=round(e["end"], 3))
                                 for e in t["einwuerfe"]]} for t in turns],
        "hoerliste": [dict(it, start=round(it["start"], 3), end=round(it["end"], 3))
                      for it in items],
    }
    Path(out_prefix + ".diar2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(out_prefix + ".diar2.srt").write_text(render_srt(sentences, names), encoding="utf-8")
    Path(out_prefix + ".diar2.txt").write_text(render_txt(turns, names), encoding="utf-8")
    Path(out_prefix + ".hoerliste.txt").write_text(render_review(items, names, segs, words),
                                                   encoding="utf-8")
    write_rttm(out_prefix + ".diar2.rttm", smoothed_segments(turns, sentences))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--words", required=True, help="word JSON from diar2_stages.py")
    ap.add_argument("--rttm", required=True, help="Nemotron RTTM")
    ap.add_argument("--second", help="pyannote RTTM (second opinion)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--names", default="Interviewer,Interviewee",
                    help="names by order of first appearance; '-' keeps speaker_N")
    ap.add_argument("--meta", help="JSON file with run metadata to embed")
    args = ap.parse_args(argv)
    apply_env_overrides()
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8")) if args.meta else None
    res = run(args.words, args.rttm, args.out_prefix, args.second, args.names, meta)
    print(f"Wörter {len(res['words'])}, Turns {len(res['turns'])}, "
          f"Hörliste {len(res['hoerliste'])}, Alignment {res['alignment']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
