"""
Similarity math for comparing two videos from their Sapiens2 outputs.

Pure numpy (plus OpenCV for resizing) so it can be tested on a laptop
without a GPU:  python -m pytest backend/test_similarity.py

Pipeline
--------
1. For every sampled frame we keep a compact "fingerprint" per model, all
   cropped to the main person and resized to a fixed grid, so position and
   scale in the frame don't matter:
     pose      body + hand keypoints (face landmarks excluded: face swaps change them)
     seg       29-class body-part label map
     normal    per-pixel surface normals
     pointmap  per-pixel relative depth
     matting   soft alpha silhouette
2. For each model we build an N x M frame-to-frame similarity matrix (0..1).
3. A weighted blend of the five matrices is aligned in time with
   subsequence dynamic time warping (DTW). That handles a copy that is
   trimmed, sped up, slowed down or has frames dropped.
4. We do the whole thing twice: once as-is and once with video B mirrored
   left/right (a common trick to evade matching), and keep the better one.
5. Each model's score is the mean similarity along the aligned path.
   The overall score is the weighted mean, optionally rescaled using
   thresholds learned from your own labelled pairs (see calibrate.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

MODELS = ["pose", "seg", "normal", "pointmap", "matting"]

DEFAULT_WEIGHTS = {  # posture is the strongest evidence; calibration re-learns these
    "pose": 0.60,
    "seg": 0.15,
    "normal": 0.05,
    "pointmap": 0.05,
    "matting": 0.15,
}

GRID_H, GRID_W = 64, 48  # fingerprint resolution (4:3, same aspect as the models)
KPT_THR = 0.3  # keypoint confidence below this is ignored
# A frame only counts as "has a person" (and is compared at all) when BOTH:
PERSON_DET_THR = 0.5  # the person detector is at least this confident, and
PERSON_MIN_AREA = 0.015  # the matting model's person cut-out covers >= 1.5% of the frame
STEP_PENALTY = 0.10  # DTW cost for a non-diagonal (time-warping) step
SEGMENT_FLOOR = 0.45  # a section's frames must beat this similarity (and the pair's own 90th percentile)
MIRROR_MARGIN = 0.05  # flipped B must beat un-flipped B by this much (raw score)
MIRROR_MIN_SCORE = 0.50  # and reach at least this raw score

# Left/right swaps for the 29 body-part classes (see Sapiens2 docs/SEG.md)
SEG_LR_PAIRS = [(5, 14), (6, 15), (7, 16), (8, 17), (9, 18), (10, 19), (11, 20), (12, 21)]
NUM_SEG_CLASSES = 29

# The pose test compares a person's *posture*, not their exact body: which way
# each limb points, and where the hands and head are relative to the shoulders.
# That stays the same when a different person re-performs the same moves (other
# arm lengths, other camera, other zoom), and when a face is swapped. Fine detail
# like individual finger joints and face landmarks is left out on purpose.
POSE_POINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_middle_finger_third_joint", "right_middle_finger_third_joint",
    "left_thumb4", "right_thumb4",
]
_P = {n: i for i, n in enumerate(POSE_POINTS)}
# (from, to, weight): limb directions
LIMBS = [
    ("left_shoulder", "left_elbow", 1.5), ("left_elbow", "left_wrist", 1.5),
    ("right_shoulder", "right_elbow", 1.5), ("right_elbow", "right_wrist", 1.5),
    ("left_wrist", "left_middle_finger_third_joint", 0.75), ("left_wrist", "left_thumb4", 0.5),
    ("right_wrist", "right_middle_finger_third_joint", 0.75), ("right_wrist", "right_thumb4", 0.5),
    ("neck", "nose", 1.5), ("left_eye", "right_eye", 0.5), ("left_shoulder", "right_shoulder", 0.75),
    ("left_shoulder", "left_hip", 1.0), ("right_shoulder", "right_hip", 1.0),
    ("left_hip", "left_knee", 1.0), ("left_knee", "left_ankle", 1.0),
    ("right_hip", "right_knee", 1.0), ("right_knee", "right_ankle", 1.0),
]
LIMBS_IDX = [(_P[a], _P[b], w) for a, b, w in LIMBS]
# (point, weight): positions relative to the shoulders, in shoulder-widths
ANCHORED = [("left_wrist", 1.5), ("right_wrist", 1.5), ("nose", 1.0)]
ANCHORED_IDX = [(_P[n], w) for n, w in ANCHORED]
LIMB_SIGMA = 0.45  # radians (~26 degrees) of tolerance on a limb's direction
ANCHOR_SIGMA = 0.45  # shoulder-widths of tolerance on a hand / head position
MIN_POSE_PARTS = 3  # fewer shared visible parts than this -> no pose score


def body_keypoint_indices(keypoint_names: list[str]) -> np.ndarray:
    """Indices (in the model's 308 keypoints) of POSE_POINTS, in that order."""
    return np.array([keypoint_names.index(n) for n in POSE_POINTS], dtype=np.int64)


# --------------------------------------------------------------------------
# Per-frame fingerprints
# --------------------------------------------------------------------------


def person_crop_box(alpha: np.ndarray, thr: float = 0.5, margin: float = 0.08):
    """Box around the person (from the matte), padded to the 4:3 grid aspect.
    Returns (x0, y0, x1, y1) in pixels or None if nobody is in frame."""
    ys, xs = np.nonzero(alpha > thr)
    h, w = alpha.shape
    if len(xs) < 0.002 * h * w:  # less than 0.2% of the frame -> no person
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = bw * (1 + 2 * margin), bh * (1 + 2 * margin)
    target = GRID_H / GRID_W  # height / width
    if bh / bw > target:
        bw = bh / target
    else:
        bh = bw * target
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def _crop_resize(arr: np.ndarray, box, interp) -> np.ndarray:
    """Crop with zero padding outside the frame, then resize to the grid."""
    x0, y0, x1, y1 = box
    out_w, out_h = GRID_W, GRID_H
    sx, sy = (x1 - x0) / out_w, (y1 - y0) / out_h
    M = np.array([[1 / sx, 0, -x0 / sx], [0, 1 / sy, -y0 / sy]], dtype=np.float32)
    return cv2.warpAffine(
        arr, M, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


@dataclass
class FrameFingerprint:
    valid: bool
    kpts: np.ndarray | None = None  # (K, 2) body keypoints, pixel coords
    kpt_scores: np.ndarray | None = None  # (K,)
    seg: np.ndarray | None = None  # (GH, GW) uint8
    normal: np.ndarray | None = None  # (GH, GW, 3) float32, unit length
    depth: np.ndarray | None = None  # (GH, GW) float32, standardised within mask
    alpha: np.ndarray | None = None  # (GH, GW) float32 in [0, 1]


def person_present(det_score: float, alpha: np.ndarray) -> bool:
    """Is there clearly a person in this frame? Uses the detector's confidence
    and the size of the matting model's person cut-out."""
    return det_score >= PERSON_DET_THR and float((alpha > 0.5).mean()) >= PERSON_MIN_AREA


def make_fingerprint(
    kpts: np.ndarray | None,
    kpt_scores: np.ndarray | None,
    seg_labels: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    person: bool = True,
) -> FrameFingerprint:
    """All dense inputs are full-frame (H, W[, C]) arrays from the models.
    Frames without a person (person=False) are left out of the comparison."""
    box = person_crop_box(alpha) if person else None
    if box is None:
        return FrameFingerprint(valid=False)

    a = _crop_resize(alpha.astype(np.float32), box, cv2.INTER_AREA)
    s = _crop_resize(seg_labels.astype(np.uint8), box, cv2.INTER_NEAREST)
    n = _crop_resize(normal.astype(np.float32), box, cv2.INTER_AREA)
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
    d = _crop_resize(depth.astype(np.float32), box, cv2.INTER_AREA)

    mask = a > 0.5
    if mask.sum() > 10:
        dv = d[mask]
        d = (d - np.median(dv)) / (dv.std() + 1e-6)
    d = np.where(mask, d, 0).astype(np.float32)

    return FrameFingerprint(
        valid=True,
        kpts=None if kpts is None else kpts.astype(np.float32),
        kpt_scores=None if kpt_scores is None else kpt_scores.astype(np.float32),
        seg=s,
        normal=n.astype(np.float32),
        depth=d,
        alpha=a,
    )


def stack(fps: list[FrameFingerprint], num_kpts: int) -> dict:
    """Stack the valid frames of one video into arrays. Keeps the original
    frame index of each valid frame so results can be mapped back to time."""
    idx = [i for i, f in enumerate(fps) if f.valid]
    T = len(idx)
    out = {
        "index": np.array(idx, dtype=np.int64),
        "kpts": np.zeros((T, num_kpts, 2), np.float32),
        "kpt_scores": np.zeros((T, num_kpts), np.float32),
        "seg": np.zeros((T, GRID_H, GRID_W), np.uint8),
        "normal": np.zeros((T, GRID_H, GRID_W, 3), np.float32),
        "depth": np.zeros((T, GRID_H, GRID_W), np.float32),
        "alpha": np.zeros((T, GRID_H, GRID_W), np.float32),
    }
    for t, i in enumerate(idx):
        f = fps[i]
        if f.kpts is not None:
            out["kpts"][t] = f.kpts
            out["kpt_scores"][t] = f.kpt_scores
        out["seg"][t] = f.seg
        out["normal"][t] = f.normal
        out["depth"][t] = f.depth
        out["alpha"][t] = f.alpha
    return out


def mirror(v: dict, kpt_flip: np.ndarray, frame_width: float) -> dict:
    """Left/right mirror of a stacked video (used to catch flipped copies)."""
    m = dict(v)
    k = v["kpts"].copy()
    k[..., 0] = frame_width - 1 - k[..., 0]
    m["kpts"] = k[:, kpt_flip]
    m["kpt_scores"] = v["kpt_scores"][:, kpt_flip]
    seg = v["seg"][:, :, ::-1].copy()
    lut = np.arange(256, dtype=np.uint8)
    for l, r in SEG_LR_PAIRS:
        lut[l], lut[r] = r, l
    m["seg"] = lut[seg]
    n = v["normal"][:, :, ::-1].copy()
    n[..., 0] *= -1
    m["normal"] = n
    m["depth"] = v["depth"][:, :, ::-1].copy()
    m["alpha"] = v["alpha"][:, :, ::-1].copy()
    return m


# --------------------------------------------------------------------------
# Frame-to-frame similarity matrices (N x M, values in 0..1)
# --------------------------------------------------------------------------


def _pose_features(k: np.ndarray, sc: np.ndarray):
    """k: (T, K, 2) keypoints, sc: (T, K) confidences ->
    limb unit vectors (T, L, 2) + visible (T, L), anchored points (T, P, 2) + visible (T, P)."""
    vis = sc > KPT_THR
    limbs, lvis = [], []
    for a, b, _ in LIMBS_IDX:
        v = k[:, b] - k[:, a]
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        limbs.append(v / np.clip(n, 1e-6, None))
        lvis.append(vis[:, a] & vis[:, b] & (n[:, 0] > 1e-3))
    ls, rs = _P["left_shoulder"], _P["right_shoulder"]
    mid = (k[:, ls] + k[:, rs]) / 2
    width = np.linalg.norm(k[:, ls] - k[:, rs], axis=-1)
    shoulders_ok = vis[:, ls] & vis[:, rs] & (width > 1e-3)
    pts, pvis = [], []
    for i, _ in ANCHORED_IDX:
        pts.append((k[:, i] - mid) / np.clip(width[:, None], 1e-6, None))
        pvis.append(vis[:, i] & shoulders_ok)
    return (np.stack(limbs, 1), np.stack(lvis, 1), np.stack(pts, 1), np.stack(pvis, 1))


def sim_pose(A: dict, B: dict) -> np.ndarray:
    """Posture similarity: limb directions + hand/head positions relative to
    the shoulders, using only parts confidently visible in both frames."""
    la, lva, pa, pva = _pose_features(A["kpts"], A["kpt_scores"])
    lb, lvb, pb, pvb = _pose_features(B["kpts"], B["kpt_scores"])
    N, M = len(la), len(lb)
    num = np.zeros((N, M), np.float32)
    den = np.zeros((N, M), np.float32)
    cnt = np.zeros((N, M), np.float32)
    for j, (_, _, w) in enumerate(LIMBS_IDX):
        cos = np.clip(la[:, j] @ lb[:, j].T, -1, 1)
        s = np.exp(-np.arccos(cos) ** 2 / (2 * LIMB_SIGMA**2))
        both = (lva[:, j][:, None] & lvb[:, j][None]).astype(np.float32)
        num += w * s * both
        den += w * both
        cnt += both
    for j, (_, w) in enumerate(ANCHORED_IDX):
        d2 = ((pa[:, j][:, None] - pb[:, j][None]) ** 2).sum(-1)
        s = np.exp(-d2 / (2 * ANCHOR_SIGMA**2))
        both = (pva[:, j][:, None] & pvb[:, j][None]).astype(np.float32)
        num += w * s * both
        den += w * both
        cnt += both
    out = num / np.clip(den, 1e-6, None)
    out[cnt < MIN_POSE_PARTS] = 0.0
    return out.astype(np.float32)


# The 29 body-part labels grouped into regions that stay the same between two
# people doing the same thing (lips, teeth or a shoe label shouldn't decide it).
SEG_GROUPS = {
    "head": [2, 3, 24, 25, 26, 27, 28],  # eyeglass, face/neck, lips, teeth, tongue
    "hair": [4],
    "upper body": [1, 22, 23],  # apparel, torso, upper clothing
    "left arm": [7, 11],
    "right arm": [16, 20],
    "left hand": [6],
    "right hand": [15],
    "lower body": [5, 8, 9, 10, 12, 13, 14, 17, 18, 19, 21],
}
SEG_BLUR = 3.0  # grid cells of tolerance (the grid is 48 cells wide)


def _seg_group_maps(seg: np.ndarray) -> np.ndarray:
    """(T, H, W) labels -> (G, T, H*W) softened region masks."""
    import cv2

    lut = np.full(256, -1, np.int64)
    for g, labels in enumerate(SEG_GROUPS.values()):
        lut[labels] = g
    grp = lut[seg]
    out = np.zeros((len(SEG_GROUPS),) + seg.shape, np.float32)
    for g in range(len(SEG_GROUPS)):
        for t in range(len(seg)):
            m = (grp[t] == g).astype(np.float32)
            if m.any():
                out[g, t] = cv2.GaussianBlur(m, (0, 0), SEG_BLUR)
    return out.reshape(len(SEG_GROUPS), len(seg), -1)


def sim_seg(A: dict, B: dict) -> np.ndarray:
    """Body layout: how well each body region (head, hair, upper body, each
    arm, each hand, lower body) overlaps between frames, with a little spatial
    tolerance, weighted by how big the region is."""
    ga, gb = _seg_group_maps(A["seg"]), _seg_group_maps(B["seg"])
    N, M = ga.shape[1], gb.shape[1]
    num = np.zeros((N, M), np.float32)
    den = np.zeros((N, M), np.float32)
    for g in range(ga.shape[0]):
        a, b = ga[g], gb[g]
        aa, bb = (a * a).sum(1), (b * b).sum(1)
        if aa.max() == 0 and bb.max() == 0:
            continue
        ab = a @ b.T
        dice = 2 * ab / np.clip(aa[:, None] + bb[None], 1e-6, None)  # overlap, 0..1
        area = a.sum(1)[:, None] + b.sum(1)[None]
        num += area * dice
        den += area
    return (num / np.clip(den, 1e-6, None)).astype(np.float32)


def sim_normal(A: dict, B: dict) -> np.ndarray:
    """Mean cosine between surface normals where both silhouettes overlap."""
    ma = (A["alpha"] > 0.5).reshape(len(A["alpha"]), -1).astype(np.float32)
    mb = (B["alpha"] > 0.5).reshape(len(B["alpha"]), -1).astype(np.float32)
    na = A["normal"].reshape(len(A["normal"]), -1, 3)
    nb = B["normal"].reshape(len(B["normal"]), -1, 3)
    dot = sum((ma * na[..., c]) @ (mb * nb[..., c]).T for c in range(3))
    cnt = ma @ mb.T
    cos = dot / np.clip(cnt, 1, None)
    return np.clip(cos, 0, 1).astype(np.float32)


def sim_pointmap(A: dict, B: dict) -> np.ndarray:
    """Correlation of relative depth (3D body shape) where both overlap."""
    ma = (A["alpha"] > 0.5).reshape(len(A["alpha"]), -1).astype(np.float32)
    mb = (B["alpha"] > 0.5).reshape(len(B["alpha"]), -1).astype(np.float32)
    da = A["depth"].reshape(len(A["depth"]), -1) * ma
    db = B["depth"].reshape(len(B["depth"]), -1) * mb
    cnt = ma @ mb.T
    sab = da @ db.T
    sa, sb = da @ mb.T, ma @ db.T
    saa, sbb = (da**2) @ mb.T, ma @ (db**2).T
    n = np.clip(cnt, 1, None)
    cov = sab / n - (sa / n) * (sb / n)
    va = saa / n - (sa / n) ** 2
    vb = sbb / n - (sb / n) ** 2
    corr = cov / np.sqrt(np.clip(va * vb, 1e-8, None))
    corr[cnt < 20] = 0
    return np.clip(corr, 0, 1).astype(np.float32)


def sim_matting(A: dict, B: dict) -> np.ndarray:
    """Soft IoU (Tanimoto) of the alpha silhouettes."""
    a = A["alpha"].reshape(len(A["alpha"]), -1)
    b = B["alpha"].reshape(len(B["alpha"]), -1)
    ab = a @ b.T
    aa = (a**2).sum(1)[:, None]
    bb = (b**2).sum(1)[None]
    return (ab / np.clip(aa + bb - ab, 1e-6, None)).astype(np.float32)


SIM_FUNCS = {
    "pose": sim_pose,
    "seg": sim_seg,
    "normal": sim_normal,
    "pointmap": sim_pointmap,
    "matting": sim_matting,
}


# --------------------------------------------------------------------------
# Temporal alignment
# --------------------------------------------------------------------------


def subsequence_dtw(S: np.ndarray, penalty: float = STEP_PENALTY):
    """Align every frame of the row video to a contiguous stretch of the
    column video (rows must be the shorter one). Returns the path as a list
    of (row, col) pairs."""
    n, m = S.shape
    C = 1.0 - S
    D = np.full((n, m), np.inf, np.float64)
    P = np.zeros((n, m), np.int8)  # 0 diag, 1 up, 2 left
    D[0] = C[0]
    for j in range(1, m):  # within the first row we may also walk left
        if D[0, j - 1] + C[0, j] + penalty < D[0, j]:
            D[0, j] = D[0, j - 1] + C[0, j] + penalty
            P[0, j] = 2
    for i in range(1, n):
        diag = np.full(m, np.inf)
        diag[1:] = D[i - 1, :-1]
        up = D[i - 1] + penalty
        best = np.minimum(diag, up)
        P[i] = np.where(diag <= up, 0, 1)
        D[i] = best + C[i]
        for j in range(1, m):  # left moves need a running pass
            cand = D[i, j - 1] + C[i, j] + penalty
            if cand < D[i, j]:
                D[i, j] = cand
                P[i, j] = 2
    j = int(np.argmin(D[-1]))
    i = n - 1
    path = [(i, j)]
    while i > 0 or (i == 0 and P[0, j] == 2 and j > 0):
        step = P[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return path[::-1]


def find_segments(
    S: np.ndarray,
    tau: float | None = None,
    gap: float = 0.15,
    min_len: int = 6,
    max_segments: int = 8,
    floor: float = 0.0,
):
    """Find every stretch where a run of B frames matches a run of A frames,
    in any order (a copy can be trimmed, re-ordered, mixed with new footage or
    taken from several parts of the original).

    S is the (N frames of A) x (M frames of B) similarity matrix. This is a
    Smith-Waterman local alignment: a cell earns (S - tau), so runs of frames
    that are more similar than `tau` build up, while a single bad frame
    (a cut, motion blur) only dents the run. Horizontal/vertical steps cost
    `gap` and allow for speed changes. Each B frame is used by at most one
    section. Returns sections sorted by B time:
        [{"path": [(a, b), ...], "mean": float, "len": int}, ...]
    """
    n, m = S.shape
    if tau is None:
        # "Better than most random frame pairs" for this particular pair
        tau = max(floor, float(np.percentile(S, 90)))
    S = S.astype(np.float64).copy()
    used_b = np.zeros(m, bool)
    segments = []
    for _ in range(max_segments):
        R = np.where(used_b[None, :], -1.0, S - tau)
        H = np.zeros((n + 1, m + 1))
        P = np.zeros((n + 1, m + 1), np.int8)  # 0 stop, 1 diag, 2 up, 3 left
        for i in range(1, n + 1):
            r = R[i - 1]
            diag = H[i - 1, :-1] + r
            up = H[i - 1, 1:] + r - gap
            row = np.maximum(0, np.maximum(diag, up))
            prow = np.where(row == 0, 0, np.where(diag >= up, 1, 2))
            H[i, 1:] = row
            P[i, 1:] = prow
            for j in range(1, m + 1):  # left moves need a running pass
                cand = H[i, j - 1] + r[j - 1] - gap
                if cand > H[i, j]:
                    H[i, j] = cand
                    P[i, j] = 3
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
        if H[i, j] <= 0:
            break
        path = []
        while i > 0 and j > 0 and P[i, j] != 0:
            path.append((i - 1, j - 1))
            step = P[i, j]
            if step == 1:
                i, j = i - 1, j - 1
            elif step == 2:
                i -= 1
            else:
                j -= 1
        path = path[::-1]
        bs = sorted({b for _, b in path})
        if len(bs) < min(min_len, m):
            break  # the best remaining run is too short to be a real section
        used_b[bs[0] : bs[-1] + 1] = True
        vals = [S[a, b] for a, b in path]
        segments.append({"path": path, "mean": float(np.mean(vals)), "len": len(bs)})
    return sorted(segments, key=lambda g: g["path"][0][1])


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def load_calibration(path: str | None = None) -> dict | None:
    path = path or os.path.join(os.path.dirname(__file__), "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def apply_calibration(raw: dict, overall_raw: float, calib: dict | None):
    """Rescale raw scores so that the typical unrelated pair from your test
    set maps to 0% and the typical true copy maps to 100%."""
    if not calib:
        return dict(raw), overall_raw, False
    out = {}
    for k, v in raw.items():
        c = calib.get("models", {}).get(k)
        if c and c["high"] > c["low"]:
            out[k] = float(np.clip((v - c["low"]) / (c["high"] - c["low"]), 0, 1))
        else:
            out[k] = v
    c = calib.get("overall")
    overall = overall_raw
    if c and c["high"] > c["low"]:
        overall = float(np.clip((overall_raw - c["low"]) / (c["high"] - c["low"]), 0, 1))
    return out, overall, True


def verdict(score: float, calib: dict | None) -> str:
    hi = (calib or {}).get("verdict", {}).get("match", 0.80)
    lo = (calib or {}).get("verdict", {}).get("unrelated", 0.55)
    if score >= hi:
        return "Likely derived from the same footage"
    if score >= lo:
        return "Partial or inconclusive match"
    return "Likely unrelated footage"


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


@dataclass
class CompareResult:
    overall: float
    overall_raw: float
    calibrated: bool
    verdict: str
    mirrored: bool
    model_scores: dict
    model_scores_raw: dict
    path: list  # list of (frame index in A, frame index in B), original frame numbers
    timelines: dict  # model -> similarity along the path
    coverage: dict = field(default_factory=dict)
    sections: list = field(default_factory=list)  # every matched section, by B time


def compare(
    A: dict,
    B: dict,
    kpt_flip: np.ndarray,
    frame_width_b: float,
    weights: dict | None = None,
    calib: dict | None = None,
) -> CompareResult:
    weights = weights or (calib or {}).get("weights") or DEFAULT_WEIGHTS
    total_w = sum(weights[k] for k in MODELS)

    best = None
    for mirrored, Bv in ((False, B), (True, mirror(B, kpt_flip, frame_width_b))):
        mats = {k: SIM_FUNCS[k](A, Bv) for k in MODELS}
        combined = sum(weights[k] * mats[k] for k in MODELS) / total_w
        segs = find_segments(combined, floor=SEGMENT_FLOOR)
        if segs:
            top = max(segs, key=lambda g: g["mean"])
            score = top["mean"]
            path = [p for g in segs for p in g["path"]]
        else:  # nothing stands out: fall back to aligning the whole videos
            a_is_rows = len(A["index"]) <= len(Bv["index"])
            path = subsequence_dtw(combined if a_is_rows else combined.T)
            if not a_is_rows:
                path = [(j, i) for i, j in path]
            score = float(np.mean([combined[a, b] for a, b in path]))
            top = {"path": path, "mean": score}
        # Only call it mirrored when the flipped version is clearly better AND
        # the match is reasonably strong; between two unrelated videos, one
        # orientation always scores a hair higher by chance.
        if best is None or (score > best[0] + MIRROR_MARGIN and score >= MIRROR_MIN_SCORE):
            best = (score, mirrored, mats, path, segs, top)

    overall_raw, mirrored, mats, path, segs, top = best
    ia = np.array([p[0] for p in top["path"]])
    ib = np.array([p[1] for p in top["path"]])
    raw = {k: float(mats[k][ia, ib].mean()) for k in MODELS}  # in the strongest section
    scores, overall, calibrated = apply_calibration(raw, overall_raw, calib)

    combined = sum(weights[k] * mats[k] for k in MODELS) / total_w
    sections = [{
        "a_start": int(A["index"][min(a for a, _ in g["path"])]),
        "a_end": int(A["index"][max(a for a, _ in g["path"])]),
        "b_start": int(B["index"][min(b for _, b in g["path"])]),
        "b_end": int(B["index"][max(b for _, b in g["path"])]),
        "similarity": round(float(np.mean([combined[a, b] for a, b in g["path"]])), 4),
    } for g in segs]
    pa = np.array([p[0] for p in path])
    pb = np.array([p[1] for p in path])
    fa, fb = A["index"][pa], B["index"][pb]
    ta, tb = A["index"][ia], B["index"][ib]
    return CompareResult(
        overall=overall,
        overall_raw=overall_raw,
        calibrated=calibrated,
        verdict=verdict(overall, calib),
        mirrored=mirrored,
        model_scores=scores,
        model_scores_raw=raw,
        path=[(int(a), int(b)) for a, b in zip(fa, fb)],
        timelines={k: [round(float(x), 4) for x in mats[k][pa, pb]] for k in MODELS},
        coverage={  # the strongest matched section
            "a_start": int(ta.min()), "a_end": int(ta.max()),
            "b_start": int(tb.min()), "b_end": int(tb.max()),
        },
        sections=sections,
    )
