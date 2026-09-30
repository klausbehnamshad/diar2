"""Tests for diar2_stages.fill_from_mlx (no models)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diar2_stages as st  # noqa: E402


def seg(*words):
    return {"words": [{"word": w, "start": s, "end": e} if s is not None else {"word": w}
                      for w, s, e in words]}


def test_untimed_whisperx_word_takes_mlx_time():
    aligned = [seg(("Er", 4.6, 4.8), ("war", 4.85, 5.1), ("1968", None, None), ("da.", 7.8, 8.1))]
    mlx = [seg((" Er", 4.5, 4.8), (" war", 4.8, 5.2), (" 1968", 7.2, 7.75), (" da.", 7.75, 8.1))]
    assert st.fill_from_mlx(aligned, mlx) == 1
    w = aligned[0]["words"]
    assert (w[2]["start"], w[2]["end"], w[2]["time_source"]) == (7.2, 7.75, "mlx")
    assert [x["time_source"] for x in w] == ["whisperx", "whisperx", "mlx", "whisperx"]
    assert w[0]["start"] == 4.6  # timed WhisperX words are left as they are


def test_matching_follows_word_sequence_across_segments():
    # the same word occurs twice; the second one is untimed and must get the
    # second mlx occurrence, not the first
    aligned = [seg(("Ja", 0.0, 0.2), ("1968", 0.3, 0.8)),
               seg(("und", 2.0, 2.2), ("1968", None, None), ("auch.", 3.0, 3.3))]
    mlx = [seg(("Ja,", 0.0, 0.2), ("1968", 0.3, 0.9)),
           seg(("und", 2.0, 2.2), ("1968", 2.3, 2.9), ("auch.", 2.9, 3.3))]
    assert st.fill_from_mlx(aligned, mlx) == 1
    assert aligned[1]["words"][1]["start"] == 2.3


def test_no_match_leaves_word_untimed():
    aligned = [seg(("Er", 0.0, 0.2), ("neunzehnhundert", None, None), ("da.", 1.0, 1.2))]
    mlx = [seg(("Er", 0.0, 0.2), ("1900", 0.3, 0.9), ("da.", 1.0, 1.2))]
    assert st.fill_from_mlx(aligned, mlx) == 0
    w = aligned[0]["words"][1]
    assert "start" not in w and "time_source" not in w
