#!/usr/bin/env bash
# diar2 einmalig einrichten (idempotent). Nur macOS auf Apple Silicon.
#
#   bash install_mac.sh
#
# Was es tut, in dieser Reihenfolge:
#  1. prüft macOS/arm64, Xcode Command Line Tools, Homebrew, conda
#  2. brew install cmake ninja sentencepiece abseil ffmpeg (nur Fehlendes)
#  3. baut NeMo-Speech.cpp aus dem Quellcode an NEMO_COMMIT (Preset metal-diar;
#     das Binary kann --device cpu und --device metal) nach NEMO_PREFIX
#  4. legt die conda-Umgebung diar2 an (die Umgebung whisperx bleibt unberührt)
#  5. lädt die Modelle einmal: Nemotron 3 Diarization (nemo-speech pull),
#     whisper-large-v3-turbo (mlx), deutsches Alignment-Modell + NLTK punkt_tab,
#     pyannote community-1 (Token aus ~/.hf_token)
#  6. schreibt ~/.config/diar2/paths.env, verlinkt den Ordner nach
#     ~/Downloads/.ohtools/diar2 und trägt "source .../diar2.sh" in die
#     Startdatei der Login-Shell ein ($SHELL: bash -> ~/.bash_profile,
#     zsh -> ~/.zshrc, unbekannt -> beide; nie doppelt)
#  7. eine Zeile pro Komponente: installiert ja/nein, Version
set -euo pipefail

# --- feste Versionen ---------------------------------------------------------
# main von github.com/NVIDIA/NeMo-Speech.cpp am 2026-09-29 (enthält #50 und #52,
# Nemotron 3 Diarization; Release 0.1.0 lädt das Modell nicht). Rückfall, falls
# dieser Stand nicht baut: 97a15afa5caa9bce5baaa86c1184103877af4101 (von
# jorngar/nemoVoiceRec auf Apple Silicon genutzt).
NEMO_COMMIT="0f706e43cf1fbc031bad1423e05460d3acaeaa1c"
NEMO_REPO="https://github.com/NVIDIA/NeMo-Speech.cpp.git"
NEMO_PRESET="metal-diar"
PY_VERSION="3.11"
PIP_PACKAGES="mlx-whisper==0.4.3 whisperx==3.8.6 pyannote.metrics==4.1"
BREW_PACKAGES="cmake ninja sentencepiece abseil ffmpeg"
CONDA_ENV="diar2"

OHTOOLS="$HOME/Downloads/.ohtools"
NEMO_PREFIX="${NEMO_PREFIX:-$OHTOOLS/nemo-speech}"
CONFIG_DIR="$HOME/.config/diar2"
CONFIG="$CONFIG_DIR/paths.env"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP="Start"
step() { STEP=$1; echo; echo "== $1"; }
trap 'echo "install_mac.sh: Abbruch in Schritt: $STEP (Zeile $LINENO)" >&2' ERR
die() { echo "install_mac.sh: $* (Schritt: $STEP)" >&2; exit 1; }

# --- 1. Voraussetzungen -----------------------------------------------------------
step "1 Voraussetzungen"
[ "$(uname -s)" = Darwin ] || die "nur macOS"
[ "$(uname -m)" = arm64 ] || die "nur Apple Silicon (arm64), gefunden: $(uname -m)"
xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools fehlen: xcode-select --install"
command -v brew >/dev/null || die "Homebrew fehlt (https://brew.sh)"
command -v git >/dev/null || die "git fehlt"
CONDA="${CONDA_EXE:-}"
if [ -z "$CONDA" ] || [ ! -x "$CONDA" ]; then
    CONDA="$(command -v conda || true)"
fi
if [ -z "$CONDA" ]; then
    for c in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
        /opt/homebrew/Caskroom/miniforge/base/bin/conda /opt/homebrew/anaconda3/bin/conda \
        /opt/miniconda3/bin/conda /opt/anaconda3/bin/conda; do
        if [ -x "$c" ]; then CONDA=$c; break; fi
    done
fi
[ -n "$CONDA" ] || die "conda nicht gefunden (CONDA_EXE setzen)"
echo "macOS $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), conda: $CONDA"

# --- 2. Homebrew ----------------------------------------------------------------------
step "2 Homebrew-Pakete"
for p in $BREW_PACKAGES; do
    if brew list --versions "$p" >/dev/null 2>&1; then
        echo "vorhanden: $(brew list --versions "$p")"
    else
        brew install "$p"
    fi
done

# --- 3. NeMo-Speech.cpp ----------------------------------------------------------------
step "3 NeMo-Speech.cpp @ ${NEMO_COMMIT:0:12} bauen"
# --- cmake-hilfen: begin (tests/test_diar2_install.py führt diesen Block aus)
# Eine aktive conda-Umgebung (z. B. anaconda base mit abseil) darf nicht in den
# Build geraten: sentencepiece kommt aus Homebrew, also muss abseil es auch.
# Belegt am Mac: absl_DIR=/opt/anaconda3/lib/cmake/absl -> Linkfehler.
# conda_roots: alle conda-Wurzeln (conda info --base, CONDA_PREFIX)
conda_roots() {
    {
        if [ -n "${CONDA:-}" ]; then "$CONDA" info --base 2>/dev/null || true; fi
        if [ -n "${CONDA_PREFIX:-}" ]; then echo "$CONDA_PREFIX"; fi
    } | awk 'NF && !seen[$0]++'
}
# path_without_conda: PATH ohne Einträge unter einer conda-Wurzel
path_without_conda() {
    local roots out="" dir root keep
    roots=$(conda_roots)
    local IFS=:
    for dir in $PATH; do
        keep=1
        while read -r root; do
            case "$dir" in "$root" | "$root"/*) [ -n "$root" ] && keep=0 ;; esac
        done <<<"$roots"
        if [ "$keep" = 1 ]; then out="${out:+$out:}$dir"; fi
    done
    echo "$out"
}
# cache_value CACHE KEY: Wert eines Eintrags in CMakeCache.txt
cache_value() { sed -n "s/^$2:[A-Z]*=//p" "$1" 2>/dev/null | head -n 1 || true; }
under() { case "$1" in "$2"/*) return 0 ;; *) return 1 ;; esac; }
# drop_foreign_cache BUILD BREW: Build-Ordner löschen, wenn sein Cache absl
# oder sentencepiece außerhalb von Homebrew gefunden hat
drop_foreign_cache() {
    local cache="$1/CMakeCache.txt" absl sp
    [ -f "$cache" ] || return 0
    absl=$(cache_value "$cache" absl_DIR)
    sp=$(cache_value "$cache" SENTENCEPIECE_LIB)
    if ! under "$absl" "$2" || ! under "$sp" "$2"; then
        echo "alter Build-Ordner mit fremden Pfaden verworfen (absl_DIR=${absl:-leer}, SENTENCEPIECE_LIB=${sp:-leer})"
        rm -rf "$1"
    fi
}
# configure_args BREW ABSL: -D-Optionen für scripts/configure.sh, eine je Zeile
configure_args() {
    local ignore
    ignore=$(conda_roots | paste -sd ';' -)
    echo "-DCMAKE_PREFIX_PATH=$1"
    echo "-Dabsl_DIR=$2"
    echo "-DCMAKE_IGNORE_PREFIX_PATH=$ignore"
}
# check_cache BUILD BREW: nach dem Konfigurieren müssen beide aus Homebrew kommen
check_cache() {
    local cache="$1/CMakeCache.txt" absl sp
    absl=$(cache_value "$cache" absl_DIR)
    sp=$(cache_value "$cache" SENTENCEPIECE_LIB)
    if ! under "$absl" "$2" || ! under "$sp" "$2"; then
        die "absl_DIR=${absl:-leer} und SENTENCEPIECE_LIB=${sp:-leer} müssen unter $2 liegen"
    fi
}
# configure_nemo SRC PRESET: fremden Cache verwerfen, konfigurieren, prüfen
configure_nemo() {
    local brew_prefix absl_dir build_dir args=() a
    brew_prefix=$(brew --prefix)
    absl_dir="$(brew --prefix abseil)/lib/cmake/absl"
    build_dir="$1/build/$2"
    drop_foreign_cache "$build_dir" "$brew_prefix"
    while read -r a; do args+=("$a"); done < <(configure_args "$brew_prefix" "$absl_dir")
    # UNGEPRUEFT: dass CMAKE_IGNORE_PREFIX_PATH plus bereinigter PATH auf dem Mac
    # bei aktiver conda base reichen (im Linux-Container nur mit Stubs getestet)
    (cd "$1" && PATH=$(path_without_conda) scripts/configure.sh "$2" "${args[@]}")
    check_cache "$build_dir" "$brew_prefix"
}
# --- cmake-hilfen: end
src="$NEMO_PREFIX/src"
if [ -x "$NEMO_PREFIX/bin/nemo-speech" ] && [ "$(cat "$NEMO_PREFIX/.commit" 2>/dev/null)" = "$NEMO_COMMIT" ]; then
    echo "bereits gebaut: $("$NEMO_PREFIX/bin/nemo-speech" --version)"
else
    if [ -d "$src/.git" ] && [ "$(git -C "$src" rev-parse HEAD 2>/dev/null)" != "$NEMO_COMMIT" ]; then
        echo "anderer Stand im Quellordner, neu klonen"
        rm -rf "$src"
    fi
    if [ ! -d "$src/.git" ]; then
        mkdir -p "$NEMO_PREFIX"
        git init -q "$src"
        git -C "$src" remote add origin "$NEMO_REPO"
        git -C "$src" fetch -q --depth 1 origin "$NEMO_COMMIT"
        git -C "$src" checkout -q --detach FETCH_HEAD
    fi
    [ "$(git -C "$src" rev-parse HEAD)" = "$NEMO_COMMIT" ] || die "Commit stimmt nicht"
    git -C "$src" submodule update --init --depth 1 ggml
    # configure.sh legt für metal-* die ggml-Patchserie an (idempotent) und reicht
    # die -D-Optionen an cmake weiter. UNGEPRUEFT: der Metal-Build selbst
    # (im Linux-Container nur cpu-diar gebaut).
    configure_nemo "$src" "$NEMO_PRESET"
    jobs=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)
    (cd "$src" && PATH=$(path_without_conda) cmake --build --preset "$NEMO_PRESET" --parallel "$jobs") \
        >"$NEMO_PREFIX/build.log" 2>&1 ||
        { tail -n 30 "$NEMO_PREFIX/build.log" >&2; die "Build fehlgeschlagen, Log: $NEMO_PREFIX/build.log"; }
    cmake --install "$src/build/$NEMO_PRESET" --prefix "$NEMO_PREFIX" >/dev/null
    echo "$NEMO_COMMIT" >"$NEMO_PREFIX/.commit"
fi
NEMO="$NEMO_PREFIX/bin/nemo-speech"
"$NEMO" --version >/dev/null || die "nemo-speech startet nicht"

# --- 4. conda-Umgebung ------------------------------------------------------------------
step "4 conda-Umgebung $CONDA_ENV"
if "$CONDA" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "vorhanden: $CONDA_ENV"
else
    "$CONDA" create -y -q -n "$CONDA_ENV" "python=$PY_VERSION"
fi
# --- python-pfad: begin (tests/test_diar2_install.py führt diesen Block aus)
# env_python ENV: setzt PY auf das Python der conda-Umgebung ENV.
# conda run kann Zusatzzeilen oder eine Leerzeile ausgeben (am Mac: PY war
# leer, obwohl die Umgebung existierte). Reihenfolge: letzte nicht leere
# stdout-Zeile von conda run, die mit / beginnt; sonst <conda info --base>/envs;
# sonst jeder Eintrag aus conda config --show envs_dirs; sonst Abbruch mit der
# rohen conda-run-Ausgabe.
env_python() {
    local out err base dir cand
    err=$(mktemp)
    out=$("$CONDA" run -n "$1" python -c 'import sys; print(sys.executable)' 2>"$err" || true)
    PY=$(printf '%s\n' "$out" | tr -d '\r' |
        awk '{sub(/[ \t]+$/, "")} NF && /^\// {last=$0} END {print last}')
    if [ -n "$PY" ] && [ -x "$PY" ]; then rm -f "$err"; return 0; fi
    base=$("$CONDA" info --base 2>/dev/null | tr -d '\r' | awk 'NF {last=$0} END {print last}' || true)
    cand="$base/envs/$1/bin/python"
    if [ -n "$base" ] && [ -x "$cand" ]; then PY=$cand; rm -f "$err"; return 0; fi
    while read -r dir; do
        cand="$dir/$1/bin/python"
        if [ -n "$dir" ] && [ -x "$cand" ]; then PY=$cand; rm -f "$err"; return 0; fi
    done < <("$CONDA" config --show envs_dirs 2>/dev/null | tr -d '\r' |
        sed -n 's/^[[:space:]]*-[[:space:]]*//p')
    die "Python der Umgebung $1 nicht gefunden; conda run gab aus: [$(printf '%s' "$out" | tr '\n' '|')] stderr: [$(tr '\n' '|' <"$err")]"
}
# --- python-pfad: end
env_python "$CONDA_ENV"
echo "Python: $PY"
# shellcheck disable=SC2086  # Paketliste bewusst getrennt
# UNGEPRUEFT: Auflösung dieser Pins auf macOS arm64 (whisperx 3.8.6 zieht torch ~=2.8)
"$PY" -m pip install -q $PIP_PACKAGES

# --- 5. Modelle einmal laden -----------------------------------------------------------------
step "5 Modelle laden (einmalig, danach offline)"
"$NEMO" pull nemotron-3-diarization
index="$NEMO_PREFIX/share/nemo-speech/model-index.json"
read -r MODEL_REL MODEL_SHA256 < <("$PY" - "$index" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
for m in idx["models"]:
    if m.get("repo") == "nvidia/Nemotron-3-Diarization":
        a = m["artifacts"][0]
        print(f"{m['repo']}/{m['revision']}/{a['filename']}", a["sha256"])
        break
PY
)
[ -n "${MODEL_REL:-}" ] || die "Modell nicht im model-index.json"
MODEL="${NEMO_SPEECH_MODEL_DIR:-$HOME/Library/Caches/NeMoSpeech/models}/$MODEL_REL"
[ -f "$MODEL" ] || die "Modelldatei fehlt nach pull: $MODEL"
got_sha=$(shasum -a 256 "$MODEL" | awk '{print $1}')
[ "$got_sha" = "$MODEL_SHA256" ] || die "SHA-256 der Modelldatei stimmt nicht"
echo "Nemotron: $(basename "$MODEL") sha256 ok"

"$PY" "$HERE/diar2_stages.py" prefetch whisper
"$PY" "$HERE/diar2_stages.py" prefetch align --language de
if [ -s "$HOME/.hf_token" ]; then
    "$PY" "$HERE/diar2_stages.py" prefetch pyannote
    PYANNOTE_OK=ja
else
    echo "WARNUNG: ~/.hf_token fehlt, pyannote-Zweitmeinung nicht vorbereitet"
    PYANNOTE_OK=nein
fi

# --- 6. Konfiguration ------------------------------------------------------------------------------
step "6 Konfiguration"
mkdir -p "$CONFIG_DIR"
device_default=cpu
if [ -f "$CONFIG" ] && grep -q "^DIAR2_NEMO_COMMIT=$NEMO_COMMIT\$" "$CONFIG"; then
    # Metal-Freigabe aus einem früheren Selbsttest gilt nur für denselben Build
    device_default=$(sed -n 's/^DIAR2_DEVICE_DEFAULT=//p' "$CONFIG" | tail -n 1)
    device_default=${device_default:-cpu}
fi
cat >"$CONFIG" <<EOF
# geschrieben von install_mac.sh $(date +%Y-%m-%d)
DIAR2_PYTHON=$PY
DIAR2_NEMO=$NEMO
DIAR2_MODEL=$MODEL
DIAR2_MODEL_SHA256=$MODEL_SHA256
DIAR2_NEMO_COMMIT=$NEMO_COMMIT
DIAR2_DEVICE_DEFAULT=$device_default
EOF
echo "geschrieben: $CONFIG (Gerät-Voreinstellung: $device_default)"

mkdir -p "$OHTOOLS"
if [ "$HERE" != "$OHTOOLS/diar2" ]; then
    ln -sfn "$HERE" "$OHTOOLS/diar2"
fi
# --- shell-eintrag: begin (tests/test_diar2_install.py führt diesen Block aus)
# rc_files_for_shell SHELL: Startdatei(en) der Login-Shell
rc_files_for_shell() {
    case "$(basename "${1:-unbekannt}")" in
        bash) echo "$HOME/.bash_profile" ;;
        zsh) echo "$HOME/.zshrc" ;;
        *) echo "$HOME/.bash_profile"; echo "$HOME/.zshrc" ;;
    esac
}
# add_source_line DATEI ZEILE: nur eintragen, wenn .ohtools/diar2/diar2.sh
# dort noch nicht vorkommt (egal ob als ~/... oder als absoluter Pfad)
add_source_line() {
    if grep -qF ".ohtools/diar2/diar2.sh" "$1" 2>/dev/null; then
        echo "bereits eingetragen: ~/$(basename "$1")"
    else
        echo "$2" >>"$1"
        echo "eingetragen in ~/$(basename "$1"): $2"
    fi
}
source_line="source \"$OHTOOLS/diar2/diar2.sh\"  # diar2"
while read -r rc; do
    add_source_line "$rc" "$source_line"
done < <(rc_files_for_shell "${SHELL:-}")
# --- shell-eintrag: end

# --- 7. Übersicht ---------------------------------------------------------------------------------
step "7 Übersicht"
line() { printf '%-22s installiert %-4s %s\n' "$1" "$2" "$3"; }
for p in $BREW_PACKAGES; do
    v=$(brew list --versions "$p" 2>/dev/null | awk '{print $2}')
    line "$p" "$([ -n "$v" ] && echo ja || echo nein)" "$v"
done
line "nemo-speech" ja "$("$NEMO" --version | awk '{print $NF}') commit ${NEMO_COMMIT:0:12} ($NEMO_PRESET)"
line "Nemotron-3-Diarization" ja "$(basename "$MODEL") sha256 ${MODEL_SHA256:0:12}"
"$PY" - <<'PY' | while IFS='|' read -r n ok v; do line "$n" "$ok" "$v"; done
import sys
from importlib.metadata import version, PackageNotFoundError
print(f"python (diar2)|ja|{sys.version.split()[0]}")
for p in ["mlx-whisper", "mlx", "whisperx", "torch", "torchaudio", "pyannote.audio", "pyannote.metrics", "nltk"]:
    try:
        print(f"{p}|ja|{version(p)}")
    except PackageNotFoundError:
        print(f"{p}|nein|-")
PY
line "pyannote community-1" "$PYANNOTE_OK" "Token aus ~/.hf_token"
echo
echo "Fertig. Weiter mit: bash $OHTOOLS/diar2/selftest_mac.sh"
