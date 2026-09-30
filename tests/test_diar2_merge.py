"""Tests for diar2_merge on synthetic words and RTTM (no models)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diar2_merge as m  # noqa: E402


def words_json(tmp_path, words, alignment="whisperx", name="w.json"):
    """words: list of (text, start, end) or (text, None, None)."""
    seg = {"start": 0.0, "end": 0.0, "text": "",
           "words": [{"word": t, "start": s, "end": e, "score": 0.9} for t, s, e in words]}
    p = tmp_path / name
    p.write_text(json.dumps({"alignment": alignment, "language": "de", "segments": [seg]}))
    return p


def rttm(tmp_path, segs, name="n.rttm"):
    p = tmp_path / name
    p.write_text("".join(
        f"SPEAKER t 1 {a:.3f} {b - a:.3f} <NA> <NA> {sp} <NA> <NA>\n" for a, b, sp in segs))
    return p


def seq(text, start, step=0.3, gap=0.05):
    """Evenly spaced words for a sentence."""
    out, t = [], start
    for tok in text.split():
        out.append((tok, round(t, 3), round(t + step, 3)))
        t += step + gap
    return out


def run(tmp_path, words, segs, second=None, names="Interviewer,Interviewee", **kw):
    w = words_json(tmp_path, words, **kw)
    r = rttm(tmp_path, segs)
    s = rttm(tmp_path, second, "p.rttm") if second else None
    res = m.run(w, r, tmp_path / "out", s, names)
    return res


@pytest.fixture(autouse=True)
def default_thresholds():
    saved = m.thresholds()
    yield
    for k, v in saved.items():
        setattr(m, k, v)


def test_clean_alternation_names_by_first_appearance(tmp_path):
    # speaker_2 speaks first -> Interviewer, regardless of the numbering
    words = seq("Wie war das damals?", 0.0) + seq("Das war eine schwere Zeit.", 2.0)
    segs = [(0.0, 1.5, "speaker_2"), (1.9, 4.0, "speaker_1")]
    res = run(tmp_path, words, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee"]
    assert res["speakers"][0] == {"label": "speaker_2", "name": "Interviewer"}
    assert not any(it["art"] == "Sprecherwechsel im Satz" for it in res["hoerliste"])
    txt = (tmp_path / "out.diar2.txt").read_text()
    assert "Interviewer: Wie war das damals?" in txt
    assert "Interviewee: Das war eine schwere Zeit." in txt


def test_names_dash_keeps_labels(tmp_path):
    res = run(tmp_path, seq("Hallo zusammen.", 0.0), [(0.0, 1.0, "speaker_1")], names="-")
    assert res["turns"][0]["speaker"] == "speaker_1"


def test_unpunctuated_turn_change_is_split_and_listed(tmp_path):
    # Whisper wrote one "sentence" across a turn change
    a = seq("und dann sind wir nach Berlin gezogen", 0.0)
    b = seq("wann war das ungefähr genau", a[-1][2] + 0.1)
    words = a + b[:-1] + [(b[-1][0] + "?", b[-1][1], b[-1][2])]
    segs = [(0.0, a[-1][2] + 0.05, "speaker_1"), (a[-1][2] + 0.05, 6.0, "speaker_2")]
    res = run(tmp_path, words, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee"]
    sw = [it for it in res["hoerliste"] if it["art"] == "Sprecherwechsel im Satz"]
    assert len(sw) == 1  # both halves reported once, as the original span
    assert sw[0]["start"] == pytest.approx(0.0) and sw[0]["end"] == pytest.approx(words[-1][2])


def test_single_stray_word_is_smoothed_by_majority(tmp_path):
    words = seq("Wir haben damals in einem kleinen Dorf gewohnt.", 0.0)
    stray = words[3]  # "einem"
    segs = [(0.0, stray[1], "speaker_1"), (stray[1], stray[2], "speaker_2"),
            (stray[2], 5.0, "speaker_1")]
    res = run(tmp_path, words, segs)
    assert {w["speaker"] for w in res["words"]} == {"Interviewer"}
    assert res["words"][3]["speaker_raw"] == "speaker_2"
    assert any(it["art"] == "Sprecherwechsel im Satz" for it in res["hoerliste"])


def test_mhm_inside_sentence_is_einwurf_not_turn_change(tmp_path):
    a = seq("Mein Vater hat in der Fabrik gearbeitet", 0.0)
    mhm = ("mhm", a[-1][2] + 0.05, a[-1][2] + 0.45)
    b = seq("und meine Mutter war zuhause.", mhm[2] + 0.05)
    words = a + [mhm] + b
    segs = [(0.0, 10.0, "speaker_1"), (mhm[1], mhm[2], "speaker_2")]
    res = run(tmp_path, words, segs)
    assert len(res["turns"]) == 1
    w = res["words"][len(a)]
    assert w["einwurf"] and w["speaker"] == "Interviewee"
    assert res["turns"][0]["einwuerfe"][0]["text"] == "mhm"
    txt = (tmp_path / "out.diar2.txt").read_text()
    assert "[Interviewee: mhm]" in txt
    assert txt.count("Interviewer:") == 1


def test_standalone_mhm_sentence_does_not_break_turn(tmp_path):
    a = seq("Das war neunzehnhundertsechzig.", 0.0)
    mhm = [("Mhm.", a[-1][2] + 0.2, a[-1][2] + 0.6)]
    b = seq("Danach kam die Lehre.", mhm[0][2] + 0.2)
    segs = [(0.0, a[-1][2] + 0.1, "speaker_1"), (mhm[0][1], mhm[0][2], "speaker_2"),
            (b[0][1] - 0.05, 10.0, "speaker_1")]
    res = run(tmp_path, a + mhm + b, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer"]
    assert res["turns"][0]["einwuerfe"][0]["speaker"] == "Interviewee"
    assert not any(it["art"] == "sehr kurzer Turn" for it in res["hoerliste"])


def test_real_answer_turn_is_not_einwurf(tmp_path):
    words = seq("Wo sind Sie geboren?", 0.0) + seq("In Luxemburg im Norden.", 2.0)
    segs = [(0.0, 1.6, "speaker_1"), (1.9, 4.0, "speaker_2")]
    res = run(tmp_path, words, segs)
    assert not any(w["einwurf"] for w in res["words"])
    assert len(res["turns"]) == 2


def test_overlap_marked_on_word_and_listed(tmp_path):
    words = seq("Das weiß ich nicht mehr so genau.", 0.0)
    segs = [(0.0, 3.0, "speaker_1"), (0.7, 1.4, "speaker_2")]
    res = run(tmp_path, words, segs)
    ov = [w for w in res["words"] if w["overlap"]]
    assert ov and all(set(w["overlap_speakers"]) == {"speaker_1", "speaker_2"} for w in ov)
    items = [it for it in res["hoerliste"] if it["art"] == "Überlappung"]
    assert len(items) == 1
    assert items[0]["start"] == pytest.approx(0.7) and items[0]["end"] == pytest.approx(1.4)


def test_short_overlap_below_threshold_ignored(tmp_path):
    words = seq("Kurz und gut.", 0.0)
    segs = [(0.0, 2.0, "speaker_1"), (0.5, 0.6, "speaker_2")]
    res = run(tmp_path, words, segs)
    assert not any(it["art"] == "Überlappung" for it in res["hoerliste"])


def test_gap_nearest_within_tolerance_else_majority(tmp_path):
    # "zwei" sits 0.3 s after the first interval -> nearest speaker;
    # "drei" is 0.6 s from any interval -> no raw speaker, majority fills it.
    words = [("eins", 0.0, 0.4), ("zwei", 0.8, 1.0), ("drei", 2.9, 3.1), ("vier.", 3.7, 3.9)]
    segs = [(0.0, 0.5, "speaker_1"), (3.7, 4.0, "speaker_1")]
    m.SENTENCE_GAP_S = 5.0
    res = run(tmp_path, words, segs)
    raw = [w["speaker_raw"] for w in res["words"]]
    assert raw[1] == "speaker_1"
    assert raw[2] is None
    assert {w["speaker"] for w in res["words"]} == {"Interviewer"}


def test_missing_timestamps_are_interpolated(tmp_path):
    words = [("Es", 0.0, 0.2), ("war", 0.25, 0.5), ("1968", None, None), ("so.", 1.2, 1.4)]
    res = run(tmp_path, words, [(0.0, 2.0, "speaker_1")])
    w = res["words"][2]
    assert w["timed"] is False
    assert w["start"] == pytest.approx(0.5) and w["end"] == pytest.approx(1.2)


def test_second_opinion_label_mapping_and_disagreement(tmp_path):
    a = seq("Erzählen Sie von Ihrer Kindheit.", 0.0)
    b = seq("Ich bin auf einem Hof aufgewachsen und hatte drei Geschwister.", 3.0)
    words = a + b
    nemo = [(0.0, 2.5, "speaker_1"), (2.9, 7.5, "speaker_2")]
    # pyannote uses other labels; agrees except for the last three words of b
    last3 = b[-3:]
    pyan = [(0.0, 2.5, "SPEAKER_01"), (2.9, last3[0][1] - 0.01, "SPEAKER_00"),
            (last3[0][1] - 0.01, 7.5, "SPEAKER_01")]
    res = run(tmp_path, words, nemo, second=pyan)
    dis = [it for it in res["hoerliste"] if it["art"] == "Nemotron und pyannote uneins"]
    assert len(dis) == 1
    assert dis[0]["start"] == pytest.approx(last3[0][1])
    assert dis[0]["end"] == pytest.approx(last3[-1][2])
    assert res["words"][0]["second_opinion"] == "speaker_1"  # mapped label


def test_second_opinion_full_agreement_no_items(tmp_path):
    words = seq("Ja gut dann fangen wir an.", 0.0)
    res = run(tmp_path, words, [(0.0, 3.0, "speaker_1")], second=[(0.0, 3.0, "SPEAKER_00")])
    assert not any(it["art"].startswith("Nemotron") for it in res["hoerliste"])


def test_review_sorted_by_duration_and_head_check(tmp_path):
    words = seq("Eins zwei drei vier fünf sechs sieben acht.", 0.0)
    segs = [(0.0, 3.0, "speaker_1"), (0.3, 0.6, "speaker_2"), (1.0, 2.2, "speaker_2")]
    res = run(tmp_path, words, segs)
    durs = [it["end"] - it["start"] for it in res["hoerliste"]]
    assert durs == sorted(durs, reverse=True)
    text = (tmp_path / "out.hoerliste.txt").read_text()
    assert "KONTROLLE  00:00:00.000" in text
    assert "Interviewer = speaker_1 (spricht zuerst)" in text


def test_short_turn_listed(tmp_path):
    words = (seq("Wie lange waren Sie dort?", 0.0) + [("Zwei.", 2.0, 2.4)]
             + seq("Und danach sind Sie zurück?", 3.0))
    segs = [(0.0, 1.8, "speaker_1"), (1.95, 2.5, "speaker_2"), (2.9, 5.0, "speaker_1")]
    res = run(tmp_path, words, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee", "Interviewer"]
    assert any(it["art"] == "sehr kurzer Turn" and "Zwei." in it["text"]
               for it in res["hoerliste"])


def test_outputs_written_and_srt_format(tmp_path):
    res = run(tmp_path, seq("Guten Tag.", 0.0) + seq("Hallo.", 1.5),
              [(0.0, 1.0, "speaker_1"), (1.4, 2.0, "speaker_2")], alignment="fallback")
    for ext in (".diar2.json", ".diar2.srt", ".diar2.txt", ".hoerliste.txt", ".diar2.rttm"):
        assert (tmp_path / ("out" + ext)).exists()
    assert res["alignment"] == "fallback"
    srt = (tmp_path / "out.diar2.srt").read_text()
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,650\n[Interviewer] Guten Tag.")
    smoothed = m.read_rttm(tmp_path / "out.diar2.rttm")
    assert [s.speaker for s in smoothed] == ["speaker_1", "speaker_2"]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DIAR2_SHORT_TURN_S", "0.4")
    monkeypatch.setenv("DIAR2_SPLIT_MIN_WORDS", "0")
    m.apply_env_overrides()
    assert m.SHORT_TURN_S == 0.4 and m.SPLIT_MIN_WORDS == 0


def test_cli(tmp_path, capsys):
    w = words_json(tmp_path, seq("Hallo.", 0.0))
    r = rttm(tmp_path, [(0.0, 1.0, "speaker_1")])
    assert m.main(["--words", str(w), "--rttm", str(r), "--out-prefix", str(tmp_path / "x")]) == 0
    assert "Wörter 1" in capsys.readouterr().out
