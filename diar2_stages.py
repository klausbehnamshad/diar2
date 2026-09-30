#!/usr/bin/env python3
"""diar2 model stages. Each subcommand loads exactly one model, writes its
result and exits, so the next stage starts with that memory released.

  transcribe  b: mlx-whisper, word timestamps
  align       c: WhisperX forced alignment on the mlx-whisper result (CPU)
  fallback    c': keep the mlx-whisper word timestamps (alignment=fallback)
  pyannote    e: pyannote community-1 as second opinion -> RTTM
  prefetch    download every model once so later runs work offline
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import wave
from pathlib import Path

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # UNGEPRUEFT: HF repo id, selftest loads it
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _read_token(path):
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None


def _load_wav_float(path):
    """16-bit PCM mono WAV -> float32 numpy array (no extra decoder)."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"erwarte 16-bit mono WAV: {path}")
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0, rate


def transcribe(args):
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(args.wav),
        path_or_hf_repo=args.model,
        language=args.language or None,
        word_timestamps=True,
        verbose=None,
    )
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"],
         "words": [{"word": w["word"], "start": w["start"], "end": w["end"],
                    "score": w.get("probability")} for w in s.get("words", [])]}
        for s in result.get("segments", [])
    ]
    _write_json(args.out, {"alignment": "mlx-whisper", "model": args.model,
                           "language": result.get("language", args.language),
                           "segments": segments})
    print(f"segments={len(segments)} language={result.get('language')}")


def _norm(text):
    return re.sub(r"[^\wäöüß]", "", (text or "").lower())


def fill_from_mlx(aligned_segments, mlx_segments):
    """Give WhisperX words without start/end the mlx-whisper time of the same
    word. Words are matched over the normalised word sequence (difflib), so a
    word is only filled where both sequences agree. Sets time_source on every
    word: whisperx, mlx, or nothing (left for diar2_merge.load_words).
    Returns the number of words filled from mlx."""
    ax = [w for s in aligned_segments for w in s.get("words", [])]
    mx = [w for s in mlx_segments for w in s.get("words", [])
          if w.get("start") is not None and w.get("end") is not None]
    for w in ax:
        if w.get("start") is not None and w.get("end") is not None:
            w["time_source"] = "whisperx"
    matcher = difflib.SequenceMatcher(None, [_norm(w.get("word")) for w in ax],
                                      [_norm(w.get("word")) for w in mx], autojunk=False)
    filled = 0
    for a0, m0, size in matcher.get_matching_blocks():
        for k in range(size):
            w, m = ax[a0 + k], mx[m0 + k]
            if w.get("start") is None or w.get("end") is None:
                w["start"], w["end"], w["time_source"] = m["start"], m["end"], "mlx"
                filled += 1
    return filled


def fallback(args, reason="align nicht gelaufen"):
    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    data["alignment"] = "fallback"
    data["alignment_note"] = reason
    for s in data.get("segments", []):
        for w in s.get("words", []):
            w["time_source"] = "mlx"
    _write_json(args.out, data)
    print(f"alignment=fallback ({reason})")


def align(args):
    import whisperx

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    language = data.get("language") or args.language
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in data["segments"] if s.get("text", "").strip()]
    model, meta = whisperx.load_align_model(language_code=language, device="cpu")
    audio = whisperx.load_audio(str(args.wav))
    aligned = whisperx.align(segments, model, meta, audio, "cpu",
                             return_char_alignments=False)
    out = {"alignment": "whisperx", "model": data.get("model"), "language": language,
           "align_model": type(model).__name__, "segments": []}
    n_words = n_timed = 0
    for s in aligned["segments"]:
        words = []
        for w in s.get("words", []):
            n_words += 1
            n_timed += w.get("start") is not None and w.get("end") is not None
            words.append({"word": w.get("word", ""), "start": w.get("start"),
                          "end": w.get("end"), "score": w.get("score")})
        out["segments"].append({"start": s.get("start"), "end": s.get("end"),
                                "text": s.get("text", ""), "words": words})
    if n_words == 0:
        return fallback(args, "align lieferte keine Wörter")
    filled = fill_from_mlx(out["segments"], data["segments"])
    _write_json(args.out, out)
    print(f"alignment=whisperx words={n_words} timed={n_timed} from_mlx={filled} "
          f"untimed={n_words - n_timed - filled}")


def pyannote(args):
    import torch
    from pyannote.audio import Pipeline

    token = _read_token(args.token_file)
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    samples, rate = _load_wav_float(args.wav)
    kwargs = {}
    if args.speakers:
        kwargs["num_speakers"] = int(args.speakers)
    output = pipeline({"waveform": torch.from_numpy(samples)[None, :], "sample_rate": rate},
                      **kwargs)
    # pyannote.audio 4 returns an object with .speaker_diarization; 3.x an Annotation
    annotation = getattr(output, "speaker_diarization", output)
    lines = [
        f"SPEAKER {args.recording_id} 1 {turn.start:.3f} {turn.end - turn.start:.3f} "
        f"<NA> <NA> {label} <NA> <NA>"
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"segments={len(lines)}")


def prefetch(args):
    """Load each model once (one at a time) so its files land in the caches."""
    step = args.what
    if step == "whisper":
        from huggingface_hub import snapshot_download

        print(snapshot_download(repo_id=args.model))
    elif step == "align":
        import nltk
        import whisperx

        nltk.download("punkt_tab", quiet=True)  # WhisperX sentence splitter
        whisperx.load_align_model(language_code=args.language, device="cpu")
        print("align model cached")
    elif step == "pyannote":
        from pyannote.audio import Pipeline

        token = _read_token(args.token_file)
        if not token:
            raise SystemExit("kein Token in ~/.hf_token")
        Pipeline.from_pretrained(args.model, token=token)
        print("pyannote cached")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("transcribe")
    p.add_argument("wav")
    p.add_argument("out")
    p.add_argument("--model", default=os.environ.get("DIAR2_WHISPER_MODEL", WHISPER_MODEL))
    p.add_argument("--language", default=os.environ.get("DIAR2_LANG", "de"))
    p.set_defaults(func=transcribe)

    p = sub.add_parser("align")
    p.add_argument("wav")
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--language", default=os.environ.get("DIAR2_LANG", "de"))
    p.set_defaults(func=align)

    p = sub.add_parser("fallback")
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--reason", default="align fehlgeschlagen")
    p.set_defaults(func=lambda a: fallback(a, a.reason))

    p = sub.add_parser("pyannote")
    p.add_argument("wav")
    p.add_argument("out")
    p.add_argument("--model", default=PYANNOTE_MODEL)
    p.add_argument("--token-file", default="~/.hf_token")
    p.add_argument("--speakers", default=os.environ.get("DIAR2_SPEAKERS", ""))
    p.add_argument("--recording-id", default="diar2")
    p.set_defaults(func=pyannote)

    p = sub.add_parser("prefetch")
    p.add_argument("what", choices=["whisper", "align", "pyannote"])
    p.add_argument("--model", default=None)
    p.add_argument("--language", default="de")
    p.add_argument("--token-file", default="~/.hf_token")
    p.set_defaults(func=prefetch)

    args = ap.parse_args(argv)
    if args.cmd == "prefetch" and args.model is None:
        args.model = WHISPER_MODEL if args.what == "whisper" else PYANNOTE_MODEL
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
