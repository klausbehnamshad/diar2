# diar2

Interview (mp4/mov) → Transkript mit Sprecherlabel pro Wort + kurze Hörliste der unsicheren Stellen.
Alles lokal auf einem Apple-Silicon-Mac (getestet wird für M3, 16 GB, ohne NVIDIA-GPU).
Nach der Installation laufen alle Modelle offline aus dem Cache.

| Stufe | Werkzeug | Modell |
|---|---|---|
| a | ffmpeg | – (16 kHz, mono, pcm_s16le) |
| b | mlx-whisper 0.4.3 | whisper-large-v3-turbo, Sprache `DIAR2_LANG` (de) |
| c | WhisperX 3.8.6, nur `align`, CPU | torchaudio `VOXPOPULI_ASR_BASE_10K_DE`; bei Fehler Rückfall auf mlx-Wortzeiten (`alignment=fallback`) |
| d | nemo-speech (NeMo-Speech.cpp @ `0f706e4`) | Nemotron-3-Diarization q8_0, Preset `v3-offline` |
| e | pyannote.audio (optional, `DIAR2_SECOND=1`) | speaker-diarization-community-1 |
| f–h | `diar2_merge.py` (nur Standardbibliothek) | – |

Jede Modellstufe ist ein eigener Prozess und läuft erst, wenn die vorige beendet ist.
Laufzeit und RAM-Spitze pro Stufe stehen in `NAME.diar2.stages.tsv`.

## Einrichten und prüfen

```sh
bash install_mac.sh      # einmalig; idempotent
bash selftest_mac.sh     # ~/Downloads/_outputs/diar2_selftest.txt
```

`install_mac.sh` schreibt `~/.config/diar2/paths.env`, verlinkt diesen Ordner nach
`~/Downloads/.ohtools/diar2` und trägt `source ~/Downloads/.ohtools/diar2/diar2.sh` in die Startdatei der Login-Shell ein:
`~/.bash_profile` bei bash, `~/.zshrc` bei zsh (nach `$SHELL`), bei unbekannter Shell in beide. Eine vorhandene Zeile mit
`.ohtools/diar2/diar2.sh` wird erkannt, es kommt nichts doppelt.
`transkript-tools.sh` und die conda-Umgebung `whisperx` bleiben unberührt.
NeMo-Speech.cpp wird gegen Homebrew gebaut (abseil, sentencepiece), auch wenn eine conda-Umgebung aktiv ist:
conda-Wurzeln sind für CMake ausgeblendet, ein alter Build mit fremden Pfaden wird verworfen, und nach dem
Konfigurieren wird geprüft, dass `absl_DIR` und `SENTENCEPIECE_LIB` unter `brew --prefix` liegen.
Der Selbsttest setzt `DIAR2_DEVICE_DEFAULT=metal` nur, wenn Metal und CPU dieselben Segmente liefern.

## Alltag

1. Aufnahme (mp4, mov, m4a, wav) nach `~/Downloads/diar2_eingang` legen.
2. Im Terminal `diar2` tippen.
3. Ergebnis liegt in `~/Downloads/diar2_ausgang/NAME/` (NAME = Dateiname ohne Endung).

## Aufruf

```sh
diar2                                    # alle neuen Dateien im Eingang, nacheinander
diar2 interview.mp4                      # eine Datei: erst im Eingang, dann in ~/Downloads gesucht, oder absolut
DIAR2_SECOND=1 diar2                     # mit pyannote-Zweitmeinung
DIAR2_NAMES=Frau_K,Herr_M diar2 a.mp4    # Namen nach erstem Auftreten; "-" = speaker_N
```

- **Stapel:** `diar2` ohne Argument bearbeitet jede Datei im Eingang, für die es noch kein
  `diar2_ausgang/NAME/NAME.diar2.json` gibt; fertige werden übersprungen. Scheitert eine Datei, läuft der
  Stapel weiter. Am Ende steht eine Übersicht mit einer Zeile pro Datei (fertig / übersprungen / FEHLER mit
  Log); der Exit-Code ist ungleich 0, wenn eine Datei gescheitert ist. Auf macOS läuft alles unter
  `caffeinate -i`, damit der Mac nicht einschläft.
- **Originale** im Eingang werden nur gelesen, nie verschoben, verändert oder gelöscht.
- **Ausgabe** pro Interview in `diar2_ausgang/NAME/`: `NAME.diar2.txt`, `.diar2.srt`, `.diar2.json`,
  `.hoerliste.txt`, `.diar2.stages.tsv`, `.nemotron.rttm`, `.diar2.rttm` (geglättet), bei Zweitmeinung
  `.pyannote.rttm`. Zwischenstände und Logs in `diar2_ausgang/.work/NAME/`. Ein zweiter Aufruf setzt nach
  einem Abbruch an der unterbrochenen Stufe fort (`DIAR2_FRESH=1` rechnet alles neu).
- **Ordner** per `DIAR2_IN` und `DIAR2_OUT` überschreibbar; beide legt diar2 selbst an.

Weitere Variablen: `DIAR2_LANG`, `DIAR2_DEVICE` (cpu|metal), `DIAR2_PRESET`, `DIAR2_SPEAKERS`
(feste Sprecherzahl nur für pyannote), `DIAR2_WHISPER_MODEL`.
Alle Schwellen der Glättung stehen oben in `diar2_merge.py` und lassen sich als
`DIAR2_<NAME>` setzen, z. B. `DIAR2_SHORT_TURN_S=0.8`.

## Glättung und Hörliste

- Wortzeiten: von WhisperX; fehlt dort eine Zeit (oft bei Zahlen), die mlx-whisper-Zeit desselben Worts. Nur wenn auch das nicht zuzuordnen ist, wird geschätzt: höchstens 0,5 s direkt vor dem nächsten Wort mit Zeit, nie über eine Pause. Jedes Wort trägt `time_source` (`whisperx` | `mlx` | `geschaetzt`) in `.diar2.json`.
- Sprecher pro Wort: der RTTM-Sprecher am Wortmittelpunkt. Fällt der Mittelpunkt in keine Sprechzeit, zählt die nächste Sprechzeit bis 0,5 s Abstand. Liegt er in zwei Sprechzeiten, wird das Wort als Überlappung markiert.
- Pro Satz entscheidet die Mehrheit. Ein längerer Lauf eines anderen Sprechers am Satzanfang oder Satzende teilt den Satz, denn dort hat Whisper einen Sprecherwechsel nicht mit Satzzeichen markiert.
- Kurze Rückmeldungen (mhm, ja, genau …, höchstens 2 Wörter und 1,2 s) innerhalb eines fremden Satzes werden als Einwurf `[Interviewee: mhm]` in den laufenden Turn gesetzt statt als Sprecherwechsel gezählt.
- Ein ganzer Satz nur aus solchen Wörtern („Ja.“), der direkt auf eine Frage („?“) des anderen Sprechers folgt, ist eine Antwort und bildet einen eigenen Turn, egal wer danach spricht. Er steht nicht als „sehr kurzer Turn“ in der Hörliste.
- Sonst ist so ein Satz nur dann ein Einwurf, wenn der Sprecher des vorigen Turns direkt danach weiterspricht. Andernfalls beginnt er einen Turn seines eigenen Sprechers: „Ja. An Herrn Weber.“ bleibt eine Antwort.
- Die Hörliste enthält Sprecherwechsel im Satz, sehr kurze Turns, Überlappungen und, bei Zweitmeinung, die Stellen, an denen Nemotron und pyannote uneins sind. Sortiert ist sie nach Dauer, die längste Stelle zuerst. Davor steht immer die Kontrolle der ersten 60 s (erster Sprecher = Interviewer).

## Messen: DER gegen eine Handannotation

1. Etwa 5 Minuten in Audacity öffnen und pro Sprechzeit eine Textmarke setzen (Beschriftung `Interviewer` oder `Interviewee`, Überlappungen als zwei Marken). Dann Datei → Exportieren → Textmarken exportieren.
2. Umwandeln und auswerten:

```sh
PY=$(sed -n 's/^DIAR2_PYTHON=//p' ~/.config/diar2/paths.env)
$PY ~/Downloads/.ohtools/diar2/eval_diar.py labels2rttm ~/Downloads/marken.txt ~/Downloads/_outputs/referenz.rttm
$PY ~/Downloads/.ohtools/diar2/eval_diar.py der --ref ~/Downloads/_outputs/referenz.rttm \
  --hyp "nemotron=$HOME/Downloads/diar2_ausgang/interview/interview.nemotron.rttm" \
  --hyp "pyannote=$HOME/Downloads/diar2_ausgang/interview/interview.pyannote.rttm" \
  --hyp "geglaettet=$HOME/Downloads/diar2_ausgang/interview/interview.diar2.rttm"
```

(`interview` ist der Dateiname der Aufnahme ohne Endung.)

Ausgewertet wird nur der Zeitraum der Referenz. Collar 0.25 bedeutet ±0,25 s (NIST); an pyannote.metrics wird es als 0.5 übergeben.

## Tests (ohne Modelle)

```sh
python -m pytest tests
shellcheck *.sh
```
