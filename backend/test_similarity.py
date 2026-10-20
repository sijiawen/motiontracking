"""Sanity tests for the similarity math using synthetic "people".
Run:  pip install numpy opencv-python-headless pytest && python -m pytest backend -q
"""

import numpy as np

from similarity import (
    FrameFingerprint, MODELS, compare, make_fingerprint, mirror, stack, subsequence_dtw,
)

H, W, K = 480, 360, 20
rng = np.random.default_rng(0)
FLIP = np.array([(i + 1 if i % 2 == 0 else i - 1) for i in range(K)])  # pairs (0,1),(2,3)...


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
