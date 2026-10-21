"""Tests for the scene, script and combine logic.  Run: python -m pytest backend -q"""

import numpy as np

from content import background_similarity, combine, compare_scene, compare_script, tokenize
from similarity import subsequence_dtw

rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_room(seed, h=8, w=5, d=32):
    return np.random.default_rng(seed).normal(size=(h, w, d))


def scene_video(room, moments, seq, seed=0, noise=0.15, person_cols=(1, 3)):
    """Fake fingerprints: a fixed room (patch grid) with a person in the middle
    columns, plus a per-moment change that shows up in the whole-frame vector."""
    r = np.random.default_rng(seed)
    T = len(seq)
    h, w, d = room.shape
    patch = np.stack([room + noise * r.normal(size=room.shape) for _ in range(T)])
    mask = np.zeros((T, h, w), bool)
    mask[:, 2:, person_cols[0]:person_cols[1] + 1] = True
    patch[mask] = r.normal(size=(mask.sum(), d))  # the person looks different every time
    cls = np.stack([room.mean((0, 1)) * 3 + moments[m] + noise * r.normal(size=d) for m in seq])
    return {"cls": cls, "patch": unit(patch), "mask": mask}


MOMENTS = {m: np.random.default_rng(i).normal(size=32) for i, m in enumerate(["wave", "hold", "write", "pack"])}


def test_scene_same_set_scores_high_and_lines_up_moments():
    room = make_room(1)
    seq_a = ["wave"] * 6 + ["hold"] * 6 + ["write"] * 8 + ["pack"] * 6
    seq_b = ["write"] * 8 + ["wave"] * 6 + ["hold"] * 6   # re-ordered re-performance
    out = compare_scene(scene_video(room, MOMENTS, seq_a, 1), scene_video(room, MOMENTS, seq_b, 2), None)
    right = sum(seq_a[a] == seq_b[b] for a, b in out["path"]) / len(out["path"])
    assert right > 0.9, right
    assert out["raw"] > 0.85, out["raw"]


def test_scene_different_room_scores_low():
    seq = ["wave"] * 6 + ["write"] * 6
    same = compare_scene(scene_video(make_room(1), MOMENTS, seq, 1), scene_video(make_room(1), MOMENTS, seq, 2), None)
    other = compare_scene(scene_video(make_room(1), MOMENTS, seq, 1), scene_video(make_room(7), MOMENTS, seq, 2), None)
    print("same room", same["raw"], "other room", other["raw"])
    assert same["raw"] > other["raw"] + 0.5


def test_background_ignores_the_person_and_small_camera_shifts():
    room = unit(make_room(2, h=16, w=12))
    ma = np.zeros(room.shape[:2], bool); ma[6:, 4:7] = True
    shifted = np.roll(room, 1, axis=1)  # camera moved one cell (edge column wraps)
    other_person = shifted.copy(); other_person[6:, 5:8] = unit(np.random.default_rng(9).normal(size=(10, 3, 32)))
    s = background_similarity(room, ma, other_person, np.roll(ma, 1, axis=1))
    assert s > 0.85, s


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
    low = combine({"body": 0.3, "scene": 0.25, "script": 0.0}, None)
    assert low["final"] == 0.0 and low["verdict"] == "Likely unrelated content"


def test_combine_uses_calibration():
    calib = {"tests": {"body": {"low": 0.2, "high": 0.4},
                       "scene": {"low": 0.0, "high": 1.0},
                       "script": {"low": 0.0, "high": 1.0}},
             "verdict": {"match": 0.9, "unrelated": 0.2}}
    out = combine({"body": 0.35, "scene": 0.5, "script": 0.1}, calib)
    assert out["driver"] == "body" and abs(out["final"] - 0.75) < 1e-6
    assert out["verdict"] == "Partial or inconclusive match" and out["calibrated"]


