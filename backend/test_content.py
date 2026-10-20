"""Tests for the scene, script and combine logic.  Run: python -m pytest backend -q"""

import numpy as np

from content import combine, compare_scene, compare_script, scene_matrix, tokenize
from similarity import subsequence_dtw

rng = np.random.default_rng(0)


def fake_video_embeddings(seed, n, d=64, drift=0.05):
    """A 'video' = a scene vector that drifts slowly from frame to frame."""
    r = np.random.default_rng(seed)
    base = r.normal(size=d)
    frames = np.stack([base + drift * i * r.normal(size=d) for i in range(n)])
    return {"full": frames, "bg": frames + 0.1 * r.normal(size=frames.shape)}


def test_scene_copy_beats_unrelated():
    a = fake_video_embeddings(1, 30)
    copy = {k: v[5:25] + 0.05 * rng.normal(size=v[5:25].shape) for k, v in a.items()}
    other = fake_video_embeddings(2, 20)
    s_copy = compare_scene(a, copy, None, subsequence_dtw)
    s_other = compare_scene(a, other, None, subsequence_dtw)
    assert s_copy["raw"] > 0.9
    assert s_copy["raw"] > s_other["raw"] + 0.5
    assert s_copy["path"][0][0] in (4, 5, 6)


def test_scene_matrix_range():
    a, b = fake_video_embeddings(3, 5), fake_video_embeddings(4, 7)
    S = scene_matrix(a, b)
    assert S.shape == (5, 7) and S.min() >= 0 and S.max() <= 1


ORIGINAL = ("Hey guys so today I'm going to show you my morning routine. First I always "
            "drink a big glass of water before I even look at my phone, and then I do "
            "five minutes of stretching by the window.")


def test_script_exact_copy():
    r = compare_script(ORIGINAL, ORIGINAL)
    assert r["raw"] == 1.0


def test_script_partial_copy_with_new_intro():
    stolen = ("Ad: try GlowSkin today! First I always drink a big glass of water before "
              "I even look at my phone, and then I apply GlowSkin serum.")
    r = compare_script(ORIGINAL, stolen)
    assert 0.4 < r["raw"] < 1.0
    # the highlighted span in B covers the copied sentence
    s, e = r["spans_b"][0]
    assert " ".join(r["tokens_b"][s:e]).lower().startswith("first i always")


def test_script_unrelated_and_common_phrases():
    other = ("I want to talk about the stock market and why I think you should invest in "
             "index funds instead of picking individual companies yourself.")
    r = compare_script(ORIGINAL, other)
    assert r["raw"] < 0.1


def test_script_not_enough_speech():
    r = compare_script(ORIGINAL, "la la la")
    assert r["raw"] is None and "video B" in r["reason"]


def test_script_cjk():
    zh = "大家好今天我要给大家分享我的早晨护肤流程首先我会喝一大杯水"
    disp, norm, cjk = tokenize(zh)
    assert cjk and len(disp) == len(zh)
    r = compare_script(zh, "广告" + zh[6:])
    assert r["raw"] > 0.7


def test_combine_takes_max_and_skips_missing():
    out = combine({"body": 0.40, "scene": 0.95, "script": None}, None)
    assert out["driver"] == "scene"
    assert out["final"] == out["scores"]["scene"] == 1.0
    assert out["scores"]["script"] is None
    assert out["verdict"] == "Likely copied content"
    low = combine({"body": 0.3, "scene": 0.4, "script": 0.0}, None)
    assert low["final"] == 0.0 and low["verdict"] == "Likely unrelated content"


def test_combine_uses_calibration():
    calib = {"tests": {"body": {"low": 0.2, "high": 0.4},
                       "scene": {"low": 0.0, "high": 1.0},
                       "script": {"low": 0.0, "high": 1.0}},
             "verdict": {"match": 0.9, "unrelated": 0.2}}
    out = combine({"body": 0.35, "scene": 0.5, "script": 0.1}, calib)
    assert out["driver"] == "body" and abs(out["final"] - 0.75) < 1e-6
    assert out["verdict"] == "Partial or inconclusive match" and out["calibrated"]
