#!/usr/bin/env bash
# diar2: Interview (mp4/mov) -> Transkript mit Sprecherlabel pro Wort + Hörliste.
#
# Laden (bash oder zsh, z. B. in ~/.zshrc):
#   source ~/Downloads/.ohtools/diar2/diar2.sh
# Aufruf:
#   diar2 interview.mp4                  (relativ zu ~/Downloads oder absolut)
#   DIAR2_SECOND=1 diar2 interview.mov   (pyannote als Zweitmeinung)
#
# Fasst ~/Downloads/.ohtools/transkript-tools.sh nicht an. Jede Modellstufe
# läuft als eigener Prozess, strikt nacheinander; Laufzeit und RAM-Spitze
# jeder Stufe stehen in NAME.diar2.stages.tsv.

# --- beim source: nur die Funktion definieren ------------------------------
if [ -n "${ZSH_VERSION:-}" ]; then
    eval '_diar2_self=${(%):-%x}'
    case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _diar2_sourced=1 ;; *) _diar2_sourced=0 ;; esac
else
    _diar2_self=${BASH_SOURCE[0]}
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then _diar2_sourced=1; else _diar2_sourced=0; fi
fi
if [ "$_diar2_sourced" = 1 ]; then
    _DIAR2_SCRIPT="$(cd "$(dirname "$_diar2_self")" && pwd)/diar2.sh"
    diar2() { bash "$_DIAR2_SCRIPT" "$@"; }
    unset _diar2_self _diar2_sourced
    # shellcheck disable=SC2317  # reached only when executed, not sourced
    return 0 2>/dev/null || true
fi
unset _diar2_self _diar2_sourced

# --- ausgeführt: die Pipeline -------------------------------------------------
set -euo pipefail

DIAR2_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAR2_CONFIG="${DIAR2_CONFIG:-$HOME/.config/diar2/paths.env}"
# paths.env schreibt install_mac.sh: DIAR2_PYTHON, DIAR2_NEMO, DIAR2_MODEL,
# DIAR2_NEMO_COMMIT, DIAR2_MODEL_SHA256; selftest_mac.sh ggf. DIAR2_DEVICE_DEFAULT.
if [ -f "$DIAR2_CONFIG" ]; then
    # shellcheck disable=SC1090
    . "$DIAR2_CONFIG"
fi

# Einstellungen (alle per Umgebungsvariable überschreibbar)
DIAR2_LANG="${DIAR2_LANG:-de}"
DIAR2_DEVICE="${DIAR2_DEVICE:-${DIAR2_DEVICE_DEFAULT:-cpu}}"   # cpu | metal
DIAR2_PRESET="${DIAR2_PRESET:-v3-offline}"                      # Nemotron-Offline-Geometrie
DIAR2_SECOND="${DIAR2_SECOND:-0}"                               # 1 = pyannote Zweitmeinung
DIAR2_NAMES="${DIAR2_NAMES:-Interviewer,Interviewee}"           # "-" = speaker_N behalten
DIAR2_OUT="${DIAR2_OUT:-$HOME/Downloads/_outputs}"
DIAR2_IN_BASE="${DIAR2_IN_BASE:-$HOME/Downloads}"
DIAR2_FRESH="${DIAR2_FRESH:-0}"                                 # 1 = Zwischenstände neu rechnen
# UNGEPRUEFT: HF-Repo-ID (huggingface.co war beim Bau nicht erreichbar); install_mac.sh lädt sie
DIAR2_WHISPER_MODEL="${DIAR2_WHISPER_MODEL:-mlx-community/whisper-large-v3-turbo}"
DIAR2_PYTHON="${DIAR2_PYTHON:-python3}"
DIAR2_NEMO="${DIAR2_NEMO:-nemo-speech}"
DIAR2_MODEL="${DIAR2_MODEL:-}"
DIAR2_NEMO_COMMIT="${DIAR2_NEMO_COMMIT:-}"
DIAR2_MODEL_SHA256="${DIAR2_MODEL_SHA256:-}"
export DIAR2_LANG DIAR2_WHISPER_MODEL DIAR2_DEVICE DIAR2_PRESET DIAR2_SECOND DIAR2_MODEL \
    DIAR2_NEMO_COMMIT DIAR2_MODEL_SHA256

die() { echo "diar2: $*" >&2; exit 1; }

[ $# -eq 1 ] || die "Aufruf: diar2 DATEI   (mp4/mov, relativ zu ~/Downloads oder absolut)"
case "$1" in
    /*) input=$1 ;;
    *) if [ -f "$DIAR2_IN_BASE/$1" ]; then input="$DIAR2_IN_BASE/$1"; else input=$1; fi ;;
esac
[ -f "$input" ] || die "Datei nicht gefunden: $1"
if [ -z "$DIAR2_MODEL" ] || [ ! -f "$DIAR2_MODEL" ]; then
    die "Nemotron-Modell nicht gefunden (DIAR2_MODEL). Erst install_mac.sh ausführen."
fi
case "$DIAR2_DEVICE" in cpu | metal) ;; *) die "DIAR2_DEVICE muss cpu oder metal sein" ;; esac
command -v ffmpeg >/dev/null || die "ffmpeg fehlt"

base=$(basename "$input")
name=${base%.*}
mkdir -p "$DIAR2_OUT"
work="$DIAR2_OUT/.diar2_work/$name"
mkdir -p "$work"
prefix="$DIAR2_OUT/$name"
stages="$prefix.diar2.stages.tsv"
printf 'stufe\tsekunden\tmax_rss_mb\tpeak_footprint_mb\texit\n' >"$stages"

# run_stage NAME LOG CMD...: misst Laufzeit und RAM-Spitze einer Stufe.
# macOS: /usr/bin/time -l (Bytes); Linux: /usr/bin/time -v (kB).
# UNGEPRUEFT: Zeilen "maximum resident set size" / "peak memory footprint" der
# macOS-Ausgabe; selftest_mac.sh zeigt die geparsten Werte.
run_stage() {
    local stage=$1 log=$2 rc=0 t0 t1 secs real="" rss="" foot=""
    shift 2
    t0=$(date +%s)
    if [ "$(uname)" = Darwin ]; then
        "${DIAR2_TIME:-/usr/bin/time}" -l "$@" >"$log" 2>&1 || rc=$?
        real=$(awk '/ real / {print $1; exit}' "$log")
        rss=$(awk '/maximum resident set size/ {printf "%.0f", $1/1048576}' "$log")
        foot=$(awk '/peak memory footprint/ {printf "%.0f", $1/1048576}' "$log")
    elif [ -x /usr/bin/time ]; then
        /usr/bin/time -v "$@" >"$log" 2>&1 || rc=$?
        rss=$(awk -F: '/Maximum resident set size/ {printf "%.0f", $2/1024}' "$log")
    else
        "$@" >"$log" 2>&1 || rc=$?
    fi
    t1=$(date +%s)
    secs=${real:-$((t1 - t0))}
    printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$secs" "${rss:--}" "${foot:--}" "$rc" >>"$stages"
    printf '  %-12s %7ss  RAM-Spitze %6s MB  %s\n' "$stage" "$secs" "${foot:-${rss:--}}" \
        "$([ "$rc" = 0 ] && echo ok || echo "FEHLER ($rc)")"
    return "$rc"
}

fail_log() { echo "--- letzte Zeilen aus $1:" >&2; tail -n 20 "$1" >&2; exit 1; }

# fresh ZIEL QUELLE: Stufe rechnen, wenn ZIEL fehlt oder älter als QUELLE ist.
# So setzt ein zweiter Aufruf nach einem Abbruch an der Stelle fort.
fresh() {
    if [ "$DIAR2_FRESH" = 1 ] || [ ! -s "$1" ] || [ "$1" -ot "$2" ]; then return 0; fi
    printf '  %-12s vorhanden, übersprungen\n' "$3"
    return 1
}

echo "diar2: $base  (Gerät $DIAR2_DEVICE, Sprache $DIAR2_LANG, Zweitmeinung $DIAR2_SECOND)"

# a. Tonspur -> 16 kHz mono PCM
wav="$work/$name.16k.wav"
if fresh "$wav" "$input" a_audio; then
    run_stage a_audio "$work/a_audio.log" \
        ffmpeg -nostdin -hide_banner -loglevel error -y -i "$input" \
        -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le "$wav" || fail_log "$work/a_audio.log"
fi

# b. mlx-whisper
asr="$work/$name.asr.$DIAR2_LANG.json"
if fresh "$asr" "$wav" b_whisper; then
    run_stage b_whisper "$work/b_whisper.log" \
        "$DIAR2_PYTHON" "$DIAR2_HOME/diar2_stages.py" transcribe "$wav" "$asr" \
        --model "$DIAR2_WHISPER_MODEL" --language "$DIAR2_LANG" || fail_log "$work/b_whisper.log"
fi

# c. WhisperX align (CPU); bei Fehler oder Abbruch: mlx-whisper-Wortzeiten
words="$work/$name.words.json"
if fresh "$words" "$asr" c_align; then
    if ! run_stage c_align "$work/c_align.log" \
        "$DIAR2_PYTHON" "$DIAR2_HOME/diar2_stages.py" align "$wav" "$asr" "$words" \
        --language "$DIAR2_LANG"; then
        echo "  align fehlgeschlagen, nutze mlx-whisper-Wortzeiten (alignment=fallback)"
        "$DIAR2_PYTHON" "$DIAR2_HOME/diar2_stages.py" fallback "$asr" "$words" \
            --reason "whisperx align fehlgeschlagen, siehe c_align.log" >/dev/null
    fi
fi

# d. Nemotron 3 Diarization über nemo-speech (explizites GGUF: kein Netz;
#    ohne --model lädt diarize das Modell selbst aus dem Netz).
#    Flags gegen "nemo-speech diarize --help" am Commit geprüft; UNGEPRUEFT:
#    dass -o mit --format rttm eine Datei schreibt (ohne Modell nicht testbar);
#    selftest_mac.sh sichert die Flags per --help ab und liest die Datei.
nemo_rttm="$work/$name.nemotron.$DIAR2_DEVICE.rttm"
if fresh "$nemo_rttm" "$wav" d_nemotron; then
    run_stage d_nemotron "$work/d_nemotron.log" \
        "$DIAR2_NEMO" diarize "$wav" --model "$DIAR2_MODEL" --device "$DIAR2_DEVICE" \
        --preset "$DIAR2_PRESET" --format rttm --recording-id "$name" \
        -o "$nemo_rttm" --force --quiet || fail_log "$work/d_nemotron.log"
fi
cp "$nemo_rttm" "$prefix.nemotron.rttm"

# e. optional: pyannote community-1 als Zweitmeinung
second_args=()
if [ "$DIAR2_SECOND" = 1 ]; then
    pyan_rttm="$work/$name.pyannote.rttm"
    if fresh "$pyan_rttm" "$wav" e_pyannote; then
        run_stage e_pyannote "$work/e_pyannote.log" \
            "$DIAR2_PYTHON" "$DIAR2_HOME/diar2_stages.py" pyannote "$wav" "$pyan_rttm" \
            --recording-id "$name" || fail_log "$work/e_pyannote.log"
    fi
    cp "$pyan_rttm" "$prefix.pyannote.rttm"
    second_args=(--second "$pyan_rttm")
fi

# f-h. Zuordnung, Glättung, Hörliste, Ausgaben (ohne Modell)
meta="$work/meta.json"
"$DIAR2_PYTHON" - "$meta" "$base" "$input" "$stages" <<'PY'
import hashlib, json, os, sys
meta_path, base, src, stages = sys.argv[1:5]
h = hashlib.sha256()
with open(src, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
rows = [l.rstrip("\n").split("\t") for l in open(stages, encoding="utf-8")][1:]
json.dump({
    "input": base, "input_sha256": h.hexdigest(),
    "whisper_model": os.environ.get("DIAR2_WHISPER_MODEL"),
    "language": os.environ.get("DIAR2_LANG"),
    "nemo_speech_commit": os.environ.get("DIAR2_NEMO_COMMIT"),
    "diar_model": os.path.basename(os.environ.get("DIAR2_MODEL", "")),
    "diar_model_sha256": os.environ.get("DIAR2_MODEL_SHA256"),
    "device": os.environ.get("DIAR2_DEVICE"), "preset": os.environ.get("DIAR2_PRESET"),
    "second_opinion": os.environ.get("DIAR2_SECOND") == "1",
    "stages": [dict(zip(["stage", "seconds", "max_rss_mb", "peak_footprint_mb", "exit"], r))
               for r in rows],
}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
run_stage f_merge "$work/f_merge.log" \
    "$DIAR2_PYTHON" "$DIAR2_HOME/diar2_merge.py" --words "$words" --rttm "$nemo_rttm" \
    ${second_args[@]+"${second_args[@]}"} --out-prefix "$prefix" --names "$DIAR2_NAMES" --meta "$meta" ||
    fail_log "$work/f_merge.log"
tail -n 1 "$work/f_merge.log" | sed 's/^/  /'

echo "fertig:"
for ext in diar2.txt diar2.srt diar2.json hoerliste.txt; do
    echo "  $prefix.$ext"
done
