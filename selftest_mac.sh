#!/usr/bin/env bash
# diar2 Selbsttest nach install_mac.sh. Braucht keine echten Interviews:
# erzeugt mit "say" ein ca. 2-minütiges Wechselgespräch zweier Stimmen mit
# bekannten Sprechergrenzen (Referenz-RTTM) und prüft darauf
#   - nemo-speech: Flags laut --help, Modell-Prüfsumme, cpu gegen metal
#     (Laufzeit, RAM-Spitze, gleiche Segmente?) -> Gerät-Voreinstellung
#   - diar2 komplett, mit pyannote-Zweitmeinung, Laufzeit/RAM je Stufe
#   - DER gegen die Referenz (pyannote.metrics, collar 0 und 0.25)
#   - Offline-Betrieb: zweiter Lauf mit HF_HUB_OFFLINE=1 und gesperrtem Netz
# Bericht: ~/Downloads/_outputs/diar2_selftest.txt (ohne Pfade, ohne echte Daten)
#
#   bash selftest_mac.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${DIAR2_CONFIG:-$HOME/.config/diar2/paths.env}"
OUT="$HOME/Downloads/_outputs"
WORK="$OUT/.diar2_selftest"
REPORT="$OUT/diar2_selftest.txt"
# Nemotron-Segmente gelten als gleich, wenn Anzahl, Sprecherfolge und Grenzen
# (bis 1 Frame = 80 ms) übereinstimmen; dann wird metal Voreinstellung.

die() { echo "selftest_mac.sh: $*" >&2; exit 1; }
[ "$(uname -s)" = Darwin ] || die "nur macOS"
[ -f "$CONFIG" ] || die "erst install_mac.sh ausführen ($CONFIG fehlt)"
# shellcheck disable=SC1090
. "$CONFIG"
PY=$DIAR2_PYTHON
NEMO=$DIAR2_NEMO
TOOLS="$HERE/selftest_tools.py"
rm -rf "$WORK"
mkdir -p "$WORK/utt"
: >"$REPORT.tmp"

say_() { echo "$*" | sed "s|$HOME|~|g" | tee -a "$REPORT.tmp"; }
section() { say_ ""; say_ "== $*"; }
TOTAL_FAIL=0
check() { # check LABEL OK(0/1) DETAIL
    if [ "$2" = 0 ]; then say_ "[ok]     $1 ${3:-}"; else say_ "[FEHLER] $1 ${3:-}"; TOTAL_FAIL=$((TOTAL_FAIL + 1)); fi
}

section "System"
say_ "Datum: $(date +%Y-%m-%d)"
say_ "macOS $(sw_vers -productVersion) $(uname -m), $(sysctl -n machdep.cpu.brand_string), RAM $(($(sysctl -n hw.memsize) / 1073741824)) GB"
say_ "nemo-speech: $("$NEMO" --version 2>&1) commit ${DIAR2_NEMO_COMMIT:0:12}"
"$PY" - <<'PY' | while read -r l; do say_ "$l"; done
from importlib.metadata import version, PackageNotFoundError
def v(p):
    try:
        return version(p)
    except PackageNotFoundError:
        return "fehlt"
print("Pakete: " + ", ".join(f"{p} {v(p)}" for p in
      ["mlx-whisper", "mlx", "whisperx", "torch", "torchaudio", "pyannote.audio", "pyannote.metrics"]))
PY

section "nemo-speech prüfen"
help=$("$NEMO" diarize --help 2>&1)
for flag in "--device" "--preset NAME" "v3-offline" "--format text|json|rttm" "--recording-id" \
    "-o, --output" "--force"; do
    if grep -qF -- "$flag" <<<"$help"; then check "diarize --help nennt '$flag'" 0; else check "diarize --help nennt '$flag'" 1; fi
done
doctor=$("$NEMO" doctor 2>&1)
if grep -qi metal <<<"$doctor"; then check "doctor meldet Metal" 0; else check "doctor meldet Metal" 1 "(nur CPU nutzbar)"; fi
got=$(shasum -a 256 "$DIAR2_MODEL" | awk '{print $1}')
if [ "$got" = "$DIAR2_MODEL_SHA256" ]; then check "Modell-SHA-256" 0 "${got:0:12}"; else check "Modell-SHA-256" 1; fi

section "Testaufnahme (say)"
# UNGEPRUEFT: Format von "say -v ?" (Name, Locale, # Beispielsatz)
voices=$(say -v '?' | sed -E 's/^(.*[^ ]) +([a-z]{2}_[A-Z]{2}) +#.*$/\1|\2/' | grep '|')
v1=$(grep '|de_DE$' <<<"$voices" | sed -n 1p | cut -d'|' -f1)
v2=$(grep '|de_DE$' <<<"$voices" | cut -d'|' -f1 | grep -vxF "$v1" | sed -n 1p)
if [ -z "$v2" ]; then
    v2=$(grep -v '|de_DE$' <<<"$voices" | grep -E '\|(de_|en_)' | sed -n 1p | cut -d'|' -f1)
    say_ "Hinweis: nur eine deutsche Stimme installiert, zweite Stimme: $v2"
fi
if [ -z "$v1" ] || [ -z "$v2" ]; then die "keine zwei Stimmen für say gefunden"; fi
say_ "Stimmen: Interviewer=$v1, Interviewee=$v2"

# Sprecher | Text | Modus | Sekunden (seq: Pause davor; overlay: Versatz ab Beginn der vorigen Äußerung)
cat >"$WORK/dialog.txt" <<'EOF'
I|Guten Tag. Schön, dass Sie sich Zeit nehmen. Können Sie mir zuerst erzählen, wo Sie aufgewachsen sind?|seq|0.0
B|Ich bin in einem kleinen Dorf im Norden aufgewachsen. Wir hatten einen Bauernhof mit Kühen und Schweinen, und meine Eltern haben von früh bis spät gearbeitet.|seq|0.7
I|Mhm.|overlay|4.5
I|Wie viele Geschwister hatten Sie?|seq|0.6
B|Drei. Zwei Brüder und eine Schwester. Ich war die Jüngste.|seq|0.5
I|Und wie war die Schulzeit für Sie?|seq|0.8
B|Die Schule war im Nachbarort. Wir sind jeden Morgen eine halbe Stunde zu Fuß gegangen, auch im Winter, wenn Schnee lag.|seq|0.6
I|Erinnern Sie sich an eine Lehrerin oder einen Lehrer besonders?|seq|0.7
B|Ja.|seq|0.4
B|An Herrn Weber. Er hat uns viel über Geschichte erzählt, und er hat mir als Erster gesagt, dass ich weiter lernen soll.|seq|0.9
I|Was haben Sie nach der Schule gemacht?|seq|0.6
B|Ich habe eine Lehre als Schneiderin begonnen. Das war damals für Mädchen auf dem Land ganz üblich.|seq|0.5
I|Mhm.|seq|0.3
B|Später bin ich dann in die Stadt gezogen und habe in einer Fabrik gearbeitet. Dort habe ich auch meinen Mann kennengelernt.|seq|0.5
I|Wie hat sich das Leben in der Stadt für Sie angefühlt?|seq|0.7
B|Am Anfang war es laut und fremd. Aber nach ein paar Monaten habe ich mich daran gewöhnt, und heute möchte ich nicht mehr zurück.|seq|0.6
I|Waren Sie verheiratet?|seq|0.7
B|Ja.|seq|0.5
I|Und haben Sie Kinder?|seq|0.6
B|Zwei Töchter und einen Sohn.|seq|0.5
I|Vielen Dank für das Gespräch.|seq|0.8
EOF
: >"$WORK/plan.tsv"
: >"$WORK/script.txt"
n=0
while IFS='|' read -r spk text mode secs; do
    n=$((n + 1))
    voice=$v1; label=Interviewer
    if [ "$spk" = B ]; then voice=$v2; label=Interviewee; fi
    say -v "$voice" -o "$WORK/utt/$n.aiff" "$text" || die "say fehlgeschlagen"
    ffmpeg -nostdin -loglevel error -y -i "$WORK/utt/$n.aiff" -ac 1 -ar 16000 -c:a pcm_s16le "$WORK/utt/$n.wav"
    printf '%s\t%s\t%s\t%s\n' "$label" "$WORK/utt/$n.wav" "$mode" "$secs" >>"$WORK/plan.tsv"
    echo "$text" >>"$WORK/script.txt"
done <"$WORK/dialog.txt"
dur=$("$PY" "$TOOLS" compose "$WORK/plan.tsv" "$WORK/selftest.wav" "$WORK/reference.rttm")
ffmpeg -nostdin -loglevel error -y -i "$WORK/selftest.wav" -c:a aac -b:a 128k "$WORK/selftest.mp4"
mins=$("$PY" -c "print(f'{$dur/60:.2f}')")
say_ "Testaufnahme: $dur s ($mins min), $(grep -c . "$WORK/reference.rttm") Referenzsegmente, davon 1 Überlappung (Mhm)"

section "Nemotron: cpu gegen metal"
for dev in cpu metal; do
    "${DIAR2_TIME:-/usr/bin/time}" -l "$NEMO" diarize "$WORK/selftest.wav" --model "$DIAR2_MODEL" --device "$dev" \
        --preset v3-offline --format rttm --recording-id selftest -o "$WORK/nemo.$dev.rttm" \
        --force --quiet >"$WORK/nemo.$dev.log" 2>&1
    rc=$?
    read -r sec rss foot < <("$PY" "$TOOLS" timel "$WORK/nemo.$dev.log")
    permin=$("$PY" -c "print(f'{float(\"$sec\")/$mins:.1f}' if '$sec' != '-' else '-')")
    check "nemotron $dev" "$rc" "Laufzeit $sec s ($permin s pro Audiominute), max RSS $rss MB, peak footprint $foot MB, $(grep -c . "$WORK/nemo.$dev.rttm" 2>/dev/null || echo 0) Segmente"
done
device=cpu
if [ -s "$WORK/nemo.cpu.rttm" ] && [ -s "$WORK/nemo.metal.rttm" ]; then
    cmp=$("$PY" "$TOOLS" compare "$WORK/nemo.cpu.rttm" "$WORK/nemo.metal.rttm")
    say_ "Vergleich: $cmp"
    case "$cmp" in gleich*) device=metal ;; esac
fi
say_ "Empfehlung Gerät: $device"
grep -v '^DIAR2_DEVICE_DEFAULT=' "$CONFIG" >"$CONFIG.tmp"
echo "DIAR2_DEVICE_DEFAULT=$device" >>"$CONFIG.tmp"
mv "$CONFIG.tmp" "$CONFIG"
say_ "paths.env: DIAR2_DEVICE_DEFAULT=$device gesetzt"

run_diar2() { # run_diar2 TAG [ENV=...]
    local tag=$1
    shift
    env "$@" DIAR2_OUT="$WORK/$tag" DIAR2_SECOND=1 DIAR2_FRESH=1 DIAR2_DEVICE="$device" \
        bash "$HERE/diar2.sh" "$WORK/selftest.mp4" >"$WORK/diar2.$tag.log" 2>&1
}

section "diar2 komplett (mit pyannote-Zweitmeinung)"
run_diar2 online
rc=$?
check "diar2 Lauf" "$rc"
res="$WORK/online/selftest"
if [ -f "$res.diar2.stages.tsv" ]; then
    say_ "Stufe          Sekunden  s/Audiominute  max RSS MB  peak footprint MB  exit"
    tail -n +2 "$res.diar2.stages.tsv" | while IFS=$'\t' read -r st s r f e; do
        say_ "$(printf '%-14s %8s  %13s  %10s  %17s  %4s' "$st" "$s" \
            "$("$PY" -c "print(f'{float(\"$s\")/$mins:.1f}')")" "$r" "$f" "$e")"
    done
fi
if [ -f "$res.diar2.json" ]; then
    "$PY" "$TOOLS" score "$res.diar2.json" "$WORK/reference.rttm" "$WORK/script.txt" |
        while read -r l; do say_ "$l"; done
    say_ "Hörliste (Kopf):"
    sed -n '4,12p' "$res.hoerliste.txt" | while read -r l; do say_ "  $l"; done
fi

section "DER gegen Referenz"
say_ "collar 0.25 = +-0.25 s um jede Referenzgrenze (NIST-Konvention)"
"$PY" "$HERE/eval_diar.py" der --ref "$WORK/reference.rttm" \
    --hyp "nemotron=$res.nemotron.rttm" --hyp "pyannote=$res.pyannote.rttm" \
    --hyp "geglaettet=$res.diar2.rttm" 2>&1 | while read -r l; do say_ "$l"; done

section "Offline-Betrieb"
# Netz für alle Werkzeuge sperren: HF offline + Proxy auf einen toten Port
run_diar2 offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 \
    https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9 NO_PROXY= no_proxy=
rc=$?
check "diar2 offline Lauf" "$rc"
off="$WORK/offline/selftest"
if [ -f "$off.diar2.json" ]; then
    align=$("$PY" -c "import json;print(json.load(open('$off.diar2.json'))['alignment'])")
    [ "$align" = whisperx ]
    check "offline: WhisperX-Alignment aus dem Cache" $? "(alignment=$align)"
    [ -s "$off.pyannote.rttm" ]
    check "offline: pyannote aus dem Cache" $?
    cmp -s "$res.nemotron.rttm" "$off.nemotron.rttm"
    check "offline: Nemotron-Ergebnis identisch mit Online-Lauf" $?
fi
if [ "$rc" != 0 ]; then
    tail -n 15 "$WORK/diar2.offline.log" | while read -r l; do say_ "  $l"; done
fi

section "Ergebnis"
say_ "Fehler: $TOTAL_FAIL"
mv "$REPORT.tmp" "$REPORT"
echo
echo "Bericht: $REPORT"
