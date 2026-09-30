# diar2

diar2 makes local, speaker-attributed transcripts of two-person oral history interviews on Apple Silicon Macs.

## Privacy

diar2 processes recordings locally. Installation downloads the models once. By default, subsequent runs set the
Hugging Face libraries to offline mode and disable Hugging Face and pyannote.audio telemetry. `DIAR2_ONLINE=1` allows
online model access while leaving telemetry disabled. The self-test repeats the pipeline with offline flags and
unreachable HTTP(S) proxies to check that the cached models work. The workflow has no upload step for recordings or
transcripts. The default `input/` and `output/` folders inside this repository are listed in `.gitignore`; storage
outside these folders needs its own protection. A full run on a 21-minute interview with Wi-Fi switched off completed
on 2026-09-30.

## Example

An excerpt from the synthetic dialogue that the self-test generates with the macOS voices. The speaker label is
assigned per word; a short backchannel by the other speaker stays inside the running turn as an interjection:

```text
[00:00:00.5] Interviewer: Guten Tag. Schön, dass Sie sich Zeit nehmen. Können Sie mir zuerst erzählen, wo Sie aufgewachsen sind?

[00:00:07.1] Interviewee: Ich bin in einem kleinen Dorf im Norden aufgewachsen. Wir hatten einen Bauernhof [Interviewer: Mhm.] mit Kühen und Schweinen, und meine Eltern haben von früh bis spät gearbeitet.

[00:00:16.7] Interviewer: Wie viele Geschwister hatten Sie?
```

Two lines of the review list for the same dialogue, after two diarization errors were inserted for illustration:

```text
  1. 00:00:26.200 - 00:00:27.900  (  1.7 s)  Sprecherwechsel im Satz: Die Schule war im Nachbarort.
  2. 00:00:48.700 - 00:00:49.900  (  1.2 s)  Überlappung: nach der Schule gemacht?
```

## Requirements

In this order:

1. A Mac with Apple Silicon (M1 or later). 16 GB RAM recommended; peak memory during transcription was 6.6 GB on an
   M3.
2. Xcode Command Line Tools: `xcode-select --install`
3. Homebrew: see <https://brew.sh>
4. conda: Miniforge, Miniconda or Anaconda.
5. Only for the optional second opinion: a Hugging Face token in `~/.hf_token` and the accepted conditions for
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
   Without them the installation prints a warning and the pyannote parts of the self-test fail.

## Install and self-test

```sh
git clone https://github.com/klausbehnamshad/diar2.git ~/Downloads/diar2
bash ~/Downloads/diar2/install_mac.sh
bash ~/Downloads/diar2/selftest_mac.sh
```

The installation changes the following on your system:

- `~/.config/diar2/paths.env` with the paths of the installed parts
- a link `~/Downloads/.ohtools/diar2` to the repository folder
- one line in `~/.bash_profile` (bash) or `~/.zshrc` (zsh) that loads the command `diar2`
- a conda environment `diar2`
- the Homebrew packages cmake, ninja, sentencepiece, abseil and ffmpeg, if missing
- the models in the caches of NeMo-Speech.cpp, Hugging Face and PyTorch, the NLTK sentence tokenizer data, and
  NeMo-Speech.cpp itself in `~/Downloads/.ohtools/nemo-speech`

Other tools and conda environments stay untouched. Open a new Terminal window so that the command `diar2` is available.

The self-test creates a dialogue of about 1.5 minutes with two macOS voices and known speaker changes. It runs the
diarization on CPU and Metal, the full pipeline with the second opinion, the error rate against the known reference,
and a second run with offline flags and unreachable proxies; the report is
`~/Downloads/_outputs/diar2_selftest.txt`.

## Updating

```sh
git -C ~/Downloads/diar2 pull --ff-only
bash ~/Downloads/diar2/selftest_mac.sh
```

Do not run `git clean -x` in this folder: it deletes input/ and output/.

## Everyday use

1. Put recordings (mp4, mov, m4a, wav) into `~/Downloads/diar2/input`.
2. Open the Terminal and type `diar2`.
3. Find the results in `~/Downloads/diar2/output/NAME/`, where NAME is the file name without extension.

diar2 processes every recording in `input/` that has no finished result yet and skips the others. If one file fails,
the others still run; a summary at the end lists each file as `fertig` (done), `übersprungen` (skipped) or `FEHLER`
(error, with the path to its log). The originals in `input/` are only read. On macOS the Mac stays awake during a run
(`caffeinate -i`). `diar2 FILE` processes a single recording.

## Run time

On an Apple M3 with 16 GB RAM, a 20.8-minute interview took 4.3 to 5.6 minutes without the second opinion
(transcription about 3 to 4.5 minutes, alignment about 1 minute, diarization about 10 seconds). The second opinion
adds roughly 40 seconds per audio minute (measured on the 1.5-minute self-test dialogue). Longer interviews scale
roughly linearly; this is an estimate.

## Outputs

Per interview in `output/NAME/`:

| File | Purpose |
|---|---|
| `NAME.diar2.txt` | Read the transcript: one paragraph per turn, interjections in brackets |
| `NAME.diar2.srt` | Subtitles for video players |
| `NAME.diar2.json` | Every word with speaker, start and end time, score and time source, for further analysis |
| `NAME.hoerliste.txt` | Review list: the passages to listen to |
| `NAME.nemotron.rttm`, `NAME.diar2.rttm` | Speaker segments (raw and smoothed) for evaluation |
| `NAME.pyannote.rttm` | Optional second-opinion speaker segments (`DIAR2_SECOND=1`) |
| `NAME.diar2.stages.tsv` | Run time and peak memory of each stage |

Intermediate files and logs are in `output/.work/NAME/`.

## The review list

The review list (`hoerliste.txt`) names the passages where the speaker assignment is uncertain, longest first:

- **Speaker change within a sentence** (Sprecherwechsel im Satz): the diarization changes speaker in the middle of a
  transcribed sentence. diar2 gives the sentence to the majority speaker or splits it; listen and decide.
- **Very short turn** (sehr kurzer Turn): a turn shorter than 1 s that is neither an interjection nor a short answer
  to a question.
- **Overlap** (Überlappung): both speakers are active at the same time.
- **Nemotron and pyannote disagree** (Nemotron und pyannote uneins): only with the second opinion; the two
  diarizations assign different speakers to the same words.

Some overlaps are hidden: overlaps without any transcribed word, overlaps that contain an interjection, and overlaps
shorter than 1 s whose words all belong to one speaker. They are hidden because there is nothing to check (no
transcribed word), the transcript already shows them (interjection), or they are short and contain only one speaker's
words. They are counted in the header of the review list and kept with their reason in `NAME.diar2.json` under
`hoerliste_ausgeblendet`.

The list always starts with a check of the first 60 seconds: the first speaker becomes "Interviewer". Listen briefly
to confirm this.

## How it works

Each model stage runs as its own process, one after the other, so only one model is in memory at a time.

| Stage | Tool | What it does |
|---|---|---|
| a | ffmpeg | Extracts the audio track as 16 kHz mono WAV |
| b | mlx-whisper 0.4.3, whisper-large-v3-turbo | Transcribes, with word timestamps |
| c | WhisperX 3.8.6 (align only, CPU) | Aligns the words more precisely; falls back to the mlx-whisper times if it fails |
| d | NeMo-Speech.cpp, Nemotron 3 Diarization | Finds who speaks when (RTTM) |
| e | pyannote.audio, community-1 (optional) | Second opinion on who speaks when |
| f | diar2_merge.py | Assigns a speaker to every word and smooths the result |
| g | diar2_merge.py | Builds the review list |
| h | diar2_merge.py | Writes the output files |

Smoothing rules in brief:

- A word gets the speaker who is active at its midpoint. If nobody is, the nearest speaker within 0.5 s.
- Each sentence goes to the speaker of the majority of its words. A longer run of the other speaker at the start or
  end of a sentence splits it.
- A short backchannel ("mhm", "ja", at most two words and 1.2 s) inside the other speaker's talk is an interjection.
- A sentence of only such words that directly follows a question of the other speaker is an answer and forms its own
  turn.
- A word without any timestamp is placed at most 0.5 s before the next timed word, so it never covers a pause.

## Limitations

- German by default. Other languages via `DIAR2_LANG`.
- Two speakers are assumed. The roles come from the order of first appearance: the first speaker is "Interviewer".
- Apple Silicon Macs only.
- Accuracy has been measured only on synthetic speech (the self-test). Real interviews have been processed, but
  without a reference annotation.
- Labels in the output files are German (for example `Hörliste`, `Einwurf`, `Sprecherwechsel im Satz`).

## Measuring accuracy (DER)

To measure the diarization error rate on your own interview, annotate about 5 minutes by hand:

1. Open the recording in Audacity. Add one label per speaker turn, named `Interviewer` or `Interviewee`. For an
   overlap, add two labels.
2. Export the labels (File, Export, Export Labels) as `~/Downloads/marken.txt`.
3. Convert and evaluate:

```sh
PY=$(sed -n 's/^DIAR2_PYTHON=//p' ~/.config/diar2/paths.env)
$PY ~/Downloads/diar2/eval_diar.py labels2rttm ~/Downloads/marken.txt ~/Downloads/diar2/output/referenz.rttm
$PY ~/Downloads/diar2/eval_diar.py der --ref ~/Downloads/diar2/output/referenz.rttm \
  --hyp "nemotron=$HOME/Downloads/diar2/output/interview/interview.nemotron.rttm" \
  --hyp "pyannote=$HOME/Downloads/diar2/output/interview/interview.pyannote.rttm" \
  --hyp "geglaettet=$HOME/Downloads/diar2/output/interview/interview.diar2.rttm"
```

Replace `interview` with the name of your recording. The result splits the error into missed speech, false alarm
and speaker confusion, with no tolerance and with ±0.25 s around each boundary. Only the annotated span is evaluated.

## Configuration

All settings are environment variables, for example `DIAR2_SECOND=1 diar2`.

| Variable | Default | Meaning |
|---|---|---|
| `DIAR2_IN` | `input/` in the repository folder | Folder with recordings |
| `DIAR2_OUT` | `output/` in the repository folder | Folder for results |
| `DIAR2_IN_BASE` | `~/Downloads` | Second place to look for `diar2 FILE` |
| `DIAR2_LANG` | `de` | Language of the interview |
| `DIAR2_SECOND` | `0` | `1` adds the pyannote second opinion |
| `DIAR2_SPEAKERS` | empty | Fixed number of speakers for pyannote |
| `DIAR2_NAMES` | `Interviewer,Interviewee` | Names by order of first appearance; `-` keeps `speaker_1`, `speaker_2` |
| `DIAR2_DEVICE` | `cpu`, or `metal` after a successful self-test | Device for the diarization |
| `DIAR2_PRESET` | `v3-offline` | Nemotron configuration |
| `DIAR2_FRESH` | `0` | `1` recomputes all stages |
| `DIAR2_ONLINE` | `0` | `1` lifts the offline mode of the Hugging Face libraries; telemetry stays off |
| `DIAR2_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Transcription model. A model that is not in the cache needs one run with DIAR2_ONLINE=1. |
| `DIAR2_CONFIG` | `~/.config/diar2/paths.env` | Paths written by the installation |
| `DIAR2_GAP_TOLERANCE_S` | `0.5` | Largest gap to the nearest speaker for a word |
| `DIAR2_SENTENCE_GAP_S` | `1.5` | Pause that ends a sentence |
| `DIAR2_BACKCHANNEL_MAX_S` | `1.2` | Longest interjection |
| `DIAR2_BACKCHANNEL_MAX_WORDS` | `2` | Most words in an interjection |
| `DIAR2_SPLIT_MIN_WORDS` | `3` | Words needed to split a sentence (`0` switches splitting off) |
| `DIAR2_SPLIT_MIN_S` | `0.8` | Duration needed to split a sentence |
| `DIAR2_SHORT_TURN_S` | `1.0` | Turns shorter than this go on the review list |
| `DIAR2_OVERLAP_MIN_S` | `0.2` | Shorter overlaps are ignored |
| `DIAR2_OVERLAP_REVIEW_MIN_S` | `1.0` | Overlaps with one speaker are listed from this length on |
| `DIAR2_DISAGREE_MIN_S` | `0.3` | Shorter disagreements with pyannote are ignored |
| `DIAR2_MERGE_GAP_S` | `0.5` | Disagreements closer than this are merged |
| `DIAR2_CHECK_HEAD_S` | `60` | Length of the check at the start |
| `DIAR2_SRT_MAX_CUE_S` | `7` | Longest subtitle |
| `DIAR2_ESTIMATE_MAX_S` | `0.5` | Window for words without a timestamp |

## Tested on

Apple M3, 16 GB RAM, macOS 26.6.2. The self-test ran without errors on 2026-09-30.

## Models and licenses

Models; see each model card for its license:

- Transcription: [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
- Diarization: [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- Word alignment: [torchaudio VOXPOPULI_ASR_BASE_10K_DE](https://pytorch.org/audio/stable/generated/torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_DE.html)
- Second opinion: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

Software used by diar2, with the license stated in its source:

| Software | License |
|---|---|
| [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) | Apache-2.0 |
| [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | MIT |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD-2-Clause |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT |

## Citation

Please cite diar2 with the metadata in [CITATION.cff](CITATION.cff).

## License

Apache-2.0, see [LICENSE](LICENSE).

## Acknowledgements

Developed by Klaus Behnam Shad. The code was written in collaboration with Claude (Anthropic).

## Kurzfassung

diar2 erstellt auf Apple-Silicon-Macs Transkriptentwürfe von Oral-History-Interviews mit zwei Personen, mit einem
Sprecherlabel für jedes Wort und einer kurzen Hörliste der unsicheren Stellen. Standardmäßig verarbeitet es die
Aufnahmen nach dem Herunterladen der Modelle lokal im Offline-Modus. Im Alltag legt man die Aufnahme nach
`~/Downloads/diar2/input`, tippt `diar2` und findet das Ergebnis in `~/Downloads/diar2/output/NAME/`.
