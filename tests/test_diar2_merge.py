# SPDX-License-Identifier: Apache-2.0
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


def test_short_answer_opens_turn_when_same_speaker_goes_on(tmp_path):
    # Abnahme 1a: "Erinnern Sie sich?" (I) / "Ja." (B) / "An Herrn Weber." (B)
    q = seq("Erinnern Sie sich?", 0.0)
    ja = [("Ja.", 1.6, 1.9)]
    b = seq("An Herrn Weber.", 2.6)
    segs = [(0.0, 1.3, "speaker_1"), (1.55, 1.95, "speaker_2"), (2.55, 3.8, "speaker_2")]
    res = run(tmp_path, q + ja + b, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee"]
    assert res["turns"][0]["einwuerfe"] == []
    assert not res["words"][len(q)]["einwurf"]
    txt = (tmp_path / "out.diar2.txt").read_text()
    assert "Interviewee: Ja. An Herrn Weber." in txt
    assert not any(it["art"] == "sehr kurzer Turn" for it in res["hoerliste"])


def test_mhm_between_two_sentences_of_other_speaker_stays_einwurf(tmp_path):
    # Abnahme 1b: "Mhm." (I) zwischen zwei B-Sätzen
    b1 = seq("Ich habe eine Lehre begonnen.", 0.0)
    mhm = [("Mhm.", b1[-1][2] + 0.3, b1[-1][2] + 0.7)]
    b2 = seq("Später bin ich in die Stadt gezogen.", mhm[0][2] + 0.4)
    segs = [(0.0, b1[-1][2] + 0.05, "speaker_2"), (mhm[0][1], mhm[0][2], "speaker_1"),
            (b2[0][1] - 0.05, 9.0, "speaker_2")]
    res = run(tmp_path, b1 + mhm + b2, segs, names="Interviewee,Interviewer")
    assert [t["speaker"] for t in res["turns"]] == ["Interviewee"]
    assert res["turns"][0]["einwuerfe"][0]["speaker"] == "Interviewer"
    assert res["words"][len(b1)]["einwurf"]


def test_ja_answer_to_question_is_own_turn_even_if_questioner_goes_on(tmp_path):
    # Nachtrag (a): "Waren Sie verheiratet?" (I) / "Ja." (B) / "Und Kinder?" (I)
    q = seq("Waren Sie verheiratet?", 0.0)
    ja = [("Ja.", 1.4, 1.7)]
    q2 = seq("Und haben Sie auch Kinder?", 2.2)  # long enough not to be a short turn
    segs = [(0.0, 1.2, "speaker_1"), (1.35, 1.75, "speaker_2"), (2.15, 4.0, "speaker_1")]
    res = run(tmp_path, q + ja + q2, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee", "Interviewer"]
    assert all(t["einwuerfe"] == [] for t in res["turns"])
    assert not res["words"][len(q)]["einwurf"]
    txt = (tmp_path / "out.diar2.txt").read_text()
    assert "Interviewee: Ja.\n" in txt
    assert res["sentences"][1]["kurzantwort"] is True
    # a recognised answer is not flagged as a suspicious short turn
    assert not any(it["art"] == "sehr kurzer Turn" for it in res["hoerliste"])


def test_ja_after_statement_with_questioner_going_on_is_einwurf(tmp_path):
    # previous sentence is no question -> the old rule applies unchanged
    s1 = seq("Wir sind dann umgezogen.", 0.0)
    ja = [("Ja.", 1.6, 1.9)]
    s2 = seq("Das war neunzehnhundertsechzig.", 2.4)
    segs = [(0.0, 1.5, "speaker_1"), (1.55, 1.95, "speaker_2"), (2.35, 3.6, "speaker_1")]
    res = run(tmp_path, s1 + ja + s2, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer"]
    assert res["words"][len(s1)]["einwurf"]


def test_mhm_mid_narration_after_earlier_question_stays_einwurf(tmp_path):
    # Nachtrag (c): the question is not the directly preceding sentence
    q = seq("Wie war das?", 0.0)
    b1 = seq("Das war eine schwere Zeit.", 1.5)
    mhm = [("Mhm.", b1[-1][2] + 0.3, b1[-1][2] + 0.7)]
    b2 = seq("Wir hatten sehr wenig.", mhm[0][2] + 0.4)
    segs = [(0.0, 1.2, "speaker_1"), (1.45, b1[-1][2] + 0.05, "speaker_2"),
            (mhm[0][1], mhm[0][2], "speaker_1"), (b2[0][1] - 0.05, 9.0, "speaker_2")]
    res = run(tmp_path, q + b1 + mhm + b2, segs)
    assert [t["speaker"] for t in res["turns"]] == ["Interviewer", "Interviewee"]
    assert res["turns"][1]["einwuerfe"][0]["text"] == "Mhm."
    assert res["sentences"][2]["kurzantwort"] is False


def _overlaps(res):
    return [it for it in res["hoerliste"] if it["art"] == "Überlappung"]


def test_overlap_short_one_speaker_is_hidden(tmp_path):
    words = seq("Das weiß ich nicht mehr so genau.", 0.0)
    segs = [(0.0, 3.0, "speaker_1"), (0.7, 1.4, "speaker_2")]   # 0.7 s
    res = run(tmp_path, words, segs)
    ov = [w for w in res["words"] if w["overlap"]]
    assert ov and all(set(w["overlap_speakers"]) == {"speaker_1", "speaker_2"} for w in ov)
    assert _overlaps(res) == []
    hidden = res["hoerliste_ausgeblendet"]
    assert hidden == [{"start": 0.7, "end": 1.4, "art": "Überlappung",
                       "grund": "kurz, ein Sprecher"}]          # no text kept


def test_overlap_without_word_is_hidden(tmp_path):
    words = seq("Erster Satz.", 0.0) + seq("Zweiter Satz.", 3.6)
    segs = [(0.0, 3.5, "speaker_1"), (3.0, 4.5, "speaker_2")]  # overlap 3.0-3.5 in a pause
    res = run(tmp_path, words, segs)
    assert _overlaps(res) == []
    assert [h["grund"] for h in res["hoerliste_ausgeblendet"]] == ["ohne Wort"]


def test_overlap_with_einwurf_is_hidden(tmp_path):
    a = seq("Mein Vater hat in der Fabrik gearbeitet", 0.0)
    mhm = ("mhm", a[-1][2] + 0.05, a[-1][2] + 0.45)
    b = seq("und meine Mutter war zuhause.", mhm[2] + 0.05)
    segs = [(0.0, 10.0, "speaker_1"), (mhm[1], mhm[2], "speaker_2")]
    res = run(tmp_path, a + [mhm] + b, segs)
    assert res["words"][len(a)]["einwurf"]
    assert _overlaps(res) == []
    assert [h["grund"] for h in res["hoerliste_ausgeblendet"]] == ["Einwurf"]


def test_overlap_two_speakers_under_threshold_is_listed(tmp_path):
    words = [("Ich", 0.0, 0.3), ("war", 0.35, 0.65), ("dort.", 0.7, 1.1),
             ("Wann", 1.0, 1.3), ("genau?", 1.35, 1.7)]
    segs = [(0.0, 1.2, "speaker_1"), (0.8, 1.8, "speaker_2")]   # 0.4 s
    res = run(tmp_path, words, segs)
    ov = _overlaps(res)
    assert len(ov) == 1 and ov[0]["text"] == "dort. Wann"
    assert ov[0]["end"] - ov[0]["start"] < m.OVERLAP_REVIEW_MIN_S
    assert res["hoerliste_ausgeblendet"] == []


def test_overlap_one_speaker_from_threshold_is_listed(tmp_path):
    words = seq("Das weiß ich nicht mehr so genau.", 0.0)
    segs = [(0.0, 3.0, "speaker_1"), (0.9, 1.9, "speaker_2")]   # exactly 1.0 s
    res = run(tmp_path, words, segs)
    ov = _overlaps(res)
    assert len(ov) == 1 and ov[0]["start"] == pytest.approx(0.9)
    assert ov[0]["end"] == pytest.approx(1.9)


def test_review_header_counts_hidden_overlaps(tmp_path):
    s1 = seq("Das weiß ich nicht mehr so genau.", 0.0)            # ends 2.45
    a = seq("Mein Vater hat in der Fabrik gearbeitet", 5.0)
    mhm = ("mhm", a[-1][2] + 0.05, a[-1][2] + 0.45)
    b = seq("und meine Mutter war zuhause.", mhm[2] + 0.05)
    segs = [(0.0, 3.5, "speaker_1"), (0.7, 1.4, "speaker_2"),   # kurz, ein Sprecher
            (3.0, 3.5, "speaker_2"),                             # ohne Wort
            (5.0, 15.0, "speaker_1"), (mhm[1], mhm[2], "speaker_2")]  # Einwurf
    res = run(tmp_path, s1 + a + [mhm] + b, segs)
    assert sorted(h["grund"] for h in res["hoerliste_ausgeblendet"]) == [
        "Einwurf", "kurz, ein Sprecher", "ohne Wort"]
    head = (tmp_path / "out.hoerliste.txt").read_text().splitlines()
    assert head[2] == ("Ausgeblendete Überlappungen: 3 (ohne Wort 1, Einwurf 1, "
                       "kurz/ein Sprecher 1), siehe diar2.json")


def test_overlap_review_threshold_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAR2_OVERLAP_REVIEW_MIN_S", "0.5")
    m.apply_env_overrides()
    assert m.OVERLAP_REVIEW_MIN_S == 0.5
    words = seq("Das weiß ich nicht mehr so genau.", 0.0)
    res = run(tmp_path, words, [(0.0, 3.0, "speaker_1"), (0.7, 1.4, "speaker_2")])
    assert len(_overlaps(res)) == 1                       # 0.7 s now reaches the threshold
    assert res["hoerliste_ausgeblendet"] == []


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


def test_untimed_word_stays_before_next_word_and_keeps_pause(tmp_path):
    # Abnahme 2: "Er war" bis 5,1 s, "1968" ohne Zeit, "da." ab 7,8 s
    words = [("Er", 4.6, 4.8), ("war", 4.85, 5.1), ("1968", None, None), ("da.", 7.8, 8.1)]
    res = run(tmp_path, words, [(4.5, 8.2, "speaker_1")])
    w = res["words"][2]
    assert w["timed"] is False and w["time_source"] == "geschaetzt"
    assert w["start"] == pytest.approx(7.3) and w["end"] == pytest.approx(7.8)
    assert w["start"] > 5.1 + m.SENTENCE_GAP_S  # does not lie over the gap
    sents = res["sentences"]
    assert [s["text"] for s in sents] == ["Er war", "1968 da."]  # pause kept as boundary
    assert [x["time_source"] for x in res["words"]] == ["whisperx", "whisperx", "geschaetzt",
                                                        "whisperx"]


def test_untimed_run_shares_window_and_short_gap(tmp_path):
    words = [("Es", 0.0, 0.2), ("war", 0.25, 0.5), ("19", None, None), ("68", None, None),
             ("so.", 0.8, 1.0)]
    res = run(tmp_path, words, [(0.0, 2.0, "speaker_1")])
    a, b = res["words"][2], res["words"][3]
    assert a["start"] == pytest.approx(0.5) and a["end"] == pytest.approx(0.65)
    assert b["start"] == pytest.approx(0.65) and b["end"] == pytest.approx(0.8)


def test_untimed_word_at_end(tmp_path):
    res = run(tmp_path, [("Das", 1.0, 1.2), ("war's.", None, None)], [(0.9, 2.0, "speaker_1")])
    w = res["words"][1]
    assert w["start"] == pytest.approx(1.2) and w["end"] == pytest.approx(1.7)


def test_time_source_from_input_and_fallback(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"alignment": "whisperx", "segments": [{"words": [
        {"word": "Eins", "start": 0.0, "end": 0.3, "time_source": "whisperx"},
        {"word": "1968.", "start": 0.4, "end": 0.9, "time_source": "mlx"}]}]}))
    ws = m.load_words(json.loads(p.read_text()))
    assert [w.time_source for w in ws] == ["whisperx", "mlx"]
    ws = m.load_words({"alignment": "fallback", "segments": [{"words": [
        {"word": "Eins", "start": 0.0, "end": 0.3}]}]})
    assert ws[0].time_source == "mlx"


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


def _selftest_dialog():
    text = (Path(__file__).resolve().parents[1] / "selftest_mac.sh").read_text(encoding="utf-8")
    body = text.split("dialog.txt\" <<'EOF'\n", 1)[1].split("\nEOF\n", 1)[0]
    return [line.split("|") for line in body.splitlines() if line.strip()]


def test_selftest_dialog_short_answer_and_einwurf(tmp_path):
    # Abnahme 1c: the dialogue in selftest_mac.sh, timed the way compose() places it
    words, segs = [], []
    t, prev_start = 0.5, 0.5
    for spk, text, mode, secs in _selftest_dialog():
        n = len(text.split())
        start = prev_start + float(secs) if mode == "overlay" else t + float(secs)
        ws = seq(text, start)
        end = ws[-1][2]
        if mode != "overlay":
            prev_start, t = start, end
        words += ws
        segs.append((start - 0.02, end + 0.02, "speaker_1" if spk == "I" else "speaker_2"))
        assert len(ws) == n
    words.sort(key=lambda w: w[1])
    res = run(tmp_path, words, segs)
    turns = res["turns"]
    assert [x["speaker"] for x in turns] == ["Interviewer", "Interviewee"] * 8 + ["Interviewer"]
    txt = (tmp_path / "out.diar2.txt").read_text()
    # case a: "Ja." opens the Interviewee turn, "An Herrn Weber." continues it
    assert "Interviewee: Ja. An Herrn Weber." in txt
    # Nachtrag: "Ja." answering "Waren Sie verheiratet?" is its own turn,
    # although the Interviewer goes on with the next question
    texts = [(x["speaker"], " ".join(res["sentences"][i]["text"] for i in x["sentences"]))
             for x in turns]
    k = texts.index(("Interviewer", "Waren Sie verheiratet?"))
    assert texts[k + 1:k + 4] == [("Interviewee", "Ja."),
                                  ("Interviewer", "Und haben Sie Kinder?"),
                                  ("Interviewee", "Zwei Töchter und einen Sohn.")]
    # case b: both "Mhm." by the Interviewer are Einwürfe inside Interviewee turns
    mhm = [e for x in turns for e in x["einwuerfe"]]
    assert [(e["speaker"], e["text"]) for e in mhm] == [("Interviewer", "Mhm."),
                                                       ("Interviewer", "Mhm.")]
    assert all(x["speaker"] == "Interviewee" for x in turns if x["einwuerfe"])
    assert not any(it["art"] == "sehr kurzer Turn" for it in res["hoerliste"])
