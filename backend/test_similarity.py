"""Sanity tests for the similarity math using synthetic "people".
Run:  pip install numpy opencv-python-headless pytest && python -m pytest backend -q
"""

import numpy as np

from similarity import (
    POSE_POINTS, FrameFingerprint, MODELS, compare, make_fingerprint, mirror, sim_pose, stack,
    subsequence_dtw,
)

H, W = 480, 360
K = len(POSE_POINTS)
rng = np.random.default_rng(0)


def _swap(n):
    return n.replace("left", "@").replace("right", "left").replace("@", "right")


FLIP = np.array([POSE_POINTS.index(_swap(n)) for n in POSE_POINTS])


def person_frame(t, seed, shift=0.0, scale=1.0):
    """A blob 'person' whose limbs move with time t. seed changes the motion."""
    r = np.random.default_rng(seed)
    freq, phase = r.uniform(0.5, 2.0, K), r.uniform(0, 6.28, K)
    base = r.uniform(-1, 1, (K, 2)) * [40, 90]
    cx, cy = W / 2 + shift, H / 2
    k = base + 25 * np.stack([np.sin(freq * t + phase), np.cos(freq * t + phase)], -1)
    k = np.array([cx, cy]) + k * scale
    yy, xx = np.mgrid[:H, :W]
    alpha = np.zeros((H, W), np.float32)
    seg = np.zeros((H, W), np.uint8)
    for i, (x, y) in enumerate(k):
        blob = (xx - x) ** 2 + (yy - y) ** 2 < (14 * scale) ** 2
        alpha[blob] = 1
        seg[blob] = (i % 28) + 1
    depth = np.where(alpha > 0, 2 + 0.3 * np.sin(xx / 30.0 + t + seed) , 0).astype(np.float32)
    normal = np.dstack([np.sin(xx / 40 + t + seed), np.cos(yy / 40 + seed), np.ones_like(xx) * 0.5])
    return make_fingerprint(k, np.ones(K), seg, normal, depth, alpha)


def video(seed, ts, **kw):
    return stack([person_frame(t, seed, **kw) for t in ts], K)


def run(A, B):
    return compare(A, B, FLIP, W)


def test_dtw_identity():
    S = np.eye(10) * 0.9 + 0.05
    path = subsequence_dtw(S)
    assert path == [(i, i) for i in range(10)]


def test_copy_beats_unrelated():
    ts = np.arange(0, 20, 0.5)
    A = video(1, ts)
    copy = video(1, ts[8:32], shift=30, scale=0.8)  # trimmed, moved, rescaled
    other = video(2, ts)
    r_copy, r_other = run(A, copy), run(A, other)
    print("copy", r_copy.overall_raw, r_copy.model_scores_raw)
    print("other", r_other.overall_raw, r_other.model_scores_raw)
    assert r_copy.overall_raw > r_other.overall_raw + 0.2
    # the copy came from frames 8..31 of A
    assert abs(r_copy.coverage["a_start"] - 8) <= 1 and abs(r_copy.coverage["a_end"] - 31) <= 1
    for m in MODELS:
        assert r_copy.model_scores_raw[m] >= r_other.model_scores_raw[m] - 0.05, m


def test_speed_change():
    A = video(3, np.arange(0, 20, 0.5))
    fast = video(3, np.arange(0, 20, 0.75))  # 1.5x speed
    other = video(4, np.arange(0, 20, 0.75))
    assert run(A, fast).overall_raw > run(A, other).overall_raw + 0.2


def test_mirror_detected():
    A = video(5, np.arange(0, 15, 0.5))
    flipped = mirror(A, FLIP, W)
    r = run(A, flipped)
    assert r.mirrored
    assert r.overall_raw > 0.9


def test_empty_frames_skipped():
    fps = [FrameFingerprint(valid=False)] + [person_frame(t, 1) for t in (0, 1, 2)]
    v = stack(fps, K)
    assert list(v["index"]) == [1, 2, 3]


def concat(*vids):
    out = {k: np.concatenate([v[k] for v in vids]) for k in vids[0] if k != "index"}
    out["index"] = np.arange(sum(len(v["index"]) for v in vids))
    return out


def test_reordered_sections_are_each_found():
    """B = a later part of A, then new footage, then an earlier part of A.
    A single start-to-end alignment can't explain this; sections can."""
    ts = np.arange(0, 30, 0.5)  # 60 frames
    A = video(7, ts)
    B = concat(video(7, ts[40:52], shift=20), video(8, ts[:12]), video(7, ts[5:17], scale=0.9))
    r = run(A, B)
    secs = sorted(r.sections, key=lambda s: s["b_start"])
    print([(s["a_start"], s["a_end"], s["b_start"], s["b_end"], round(s["similarity"], 2)) for s in secs])
    found = [s for s in secs if s["similarity"] > 0.8]
    assert len(found) == 2
    s1, s2 = found
    assert abs(s1["a_start"] - 40) <= 2 and s1["b_start"] <= 2
    assert abs(s2["a_start"] - 5) <= 2 and abs(s2["b_start"] - 24) <= 2
    assert r.overall_raw > 0.85


def test_short_copied_clip_inside_long_video():
    ts = np.arange(0, 30, 0.5)
    A = video(9, ts)
    B = concat(video(10, ts[:25]), video(9, ts[20:30]), video(11, ts[:25]))
    other = concat(video(10, ts[:25]), video(12, ts[:10]), video(11, ts[:25]))
    r, o = run(A, B), run(A, other)
    print("clip", r.overall_raw, r.sections, "other", o.overall_raw)
    assert r.overall_raw > o.overall_raw + 0.2
    assert any(abs(s["a_start"] - 20) <= 2 and abs(s["b_start"] - 25) <= 2 for s in r.sections)



# ---- posture: the same moves performed by a different person ----

def stick(action, t, size=1.0, cx=180, cy=200, noise=0.0, seed=0):
    """Named keypoints for a seated person doing an action at time t."""
    r = np.random.default_rng(seed)
    P = {}
    sw = 60 * size  # shoulder width
    P["left_shoulder"], P["right_shoulder"] = (cx + sw / 2, cy), (cx - sw / 2, cy)
    P["neck"] = (cx, cy - 8 * size)
    head_down = action == "write"
    P["nose"] = (cx, cy - (20 if head_down else 55) * size)
    P["left_eye"], P["right_eye"] = (cx + 10 * size, P["nose"][1] - 8 * size), (cx - 10 * size, P["nose"][1] - 8 * size)
    P["left_ear"], P["right_ear"] = (cx + 22 * size, P["nose"][1] - 4 * size), (cx - 22 * size, P["nose"][1] - 4 * size)
    def arm(side, ang_up, ang_fore, L1=45, L2=40):
        sx, sy = P[f"{side}_shoulder"]
        sgn = 1 if side == "left" else -1
        ex, ey = sx + sgn * L1 * size * np.sin(ang_up), sy + L1 * size * np.cos(ang_up)
        wx, wy = ex + sgn * L2 * size * np.sin(ang_fore), ey + L2 * size * np.cos(ang_fore)
        P[f"{side}_elbow"], P[f"{side}_wrist"] = (ex, ey), (wx, wy)
        P[f"{side}_middle_finger_third_joint"] = (wx + sgn * 8 * size * np.sin(ang_fore), wy + 8 * size * np.cos(ang_fore))
        P[f"{side}_thumb4"] = (wx + sgn * 6 * size, wy - 4 * size)
    if action == "wave":
        arm("left", 0.3, 0.3); arm("right", 2.2, 3.0 + 0.3 * np.sin(3 * t))
    elif action == "scrunchie":
        arm("left", 1.4, 2.8); arm("right", 1.4, 2.8)
    elif action == "write":
        arm("left", 0.4, 1.5); arm("right", 0.5, 1.4 + 0.1 * np.sin(2 * t))
    else:  # arms down
        arm("left", 0.15, 0.1); arm("right", 0.15, 0.1)
    for n in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]:
        P[n] = (cx, cy + 200)  # hidden behind the table
    k = np.array([P[n] for n in POSE_POINTS], np.float32) + noise * size * r.normal(size=(K, 2))
    sc = np.array([0.1 if any(x in n for x in ("hip", "knee", "ankle")) else 0.9 for n in POSE_POINTS])
    return k, sc


def pose_video(actions, **kw):
    ks, scs = zip(*[stick(a, i * 0.5, seed=i, **kw) for i, a in enumerate(actions)])
    return {"kpts": np.stack(ks), "kpt_scores": np.stack(scs)}


def test_same_moves_different_person_scores_high():
    script = ["wave"] * 6 + ["scrunchie"] * 6 + ["write"] * 8
    A = pose_video(script, noise=2)
    # a different, smaller person, framed differently, doing the same moves in a different order
    B = pose_video(["write"] * 8 + ["wave"] * 6 + ["scrunchie"] * 6, size=0.8, cx=150, cy=230, noise=3)
    S = sim_pose(A, B)
    same = [S[i, j] for i in range(20) for j in range(20) if script[i] == (["write"] * 8 + ["wave"] * 6 + ["scrunchie"] * 6)[j]]
    diff = [S[i, j] for i in range(20) for j in range(20) if script[i] != (["write"] * 8 + ["wave"] * 6 + ["scrunchie"] * 6)[j]]
    print("same-move mean", np.mean(same), "different-move mean", np.mean(diff))
    assert np.mean(same) > 0.85 and np.mean(diff) < 0.5


def test_writing_vs_arms_down_is_low():
    A = pose_video(["write"] * 4)
    B = pose_video(["rest"] * 4)
    assert sim_pose(A, B).mean() < 0.5


def test_body_parts_same_layout_different_person():
    """Same pose, different person: other clothes label, no visible teeth,
    slightly different position. Should still score high on body layout."""
    from similarity import sim_seg

    def layout(dx=0, arm_up=False, shirt=23, teeth=True):
        s = np.zeros((64, 48), np.uint8)
        s[5:20, 17 + dx:31 + dx] = 4                 # hair
        s[8:20, 19 + dx:29 + dx] = 3                 # face
        if teeth:
            s[16:17, 22 + dx:26 + dx] = 26
        s[20:50, 12 + dx:36 + dx] = shirt            # upper body
        if arm_up:
            s[8:30, 4 + dx:11 + dx] = 11; s[4:9, 4 + dx:11 + dx] = 6
        else:
            s[22:48, 4 + dx:11 + dx] = 11; s[48:54, 4 + dx:11 + dx] = 6
        s[22:48, 37 + dx:44 + dx] = 20; s[48:54, 37 + dx:44 + dx] = 15
        return s

    A = {"seg": np.stack([layout()])}
    same = {"seg": np.stack([layout(dx=1, shirt=22, teeth=False)])}
    other = {"seg": np.stack([layout(arm_up=True)])}
    s_same, s_other = sim_seg(A, same)[0, 0], sim_seg(A, other)[0, 0]
    print("seg same", s_same, "arm raised", s_other)
    assert s_same > 0.75 and s_same > s_other + 0.05
