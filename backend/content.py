"""
The two content tests that sit alongside the Sapiens2 body & motion test,
plus the logic that combines all three into the final score.

  scene   Do the backgrounds line up (table, wall, ceiling, props, framing)?
          Uses DINOv2 patch features: the frame is split into a grid and each
          background cell is compared with the same spot in the other video,
          ignoring cells where the person is, so a face or body swap over the
          same footage still matches.
  script  Did they say the same words? Whisper transcribes both videos and we
          measure how much of the shorter transcript appears in the other as
          shared phrases of 4+ words (exact or near-exact copying).

Final score = the highest of the three tests (any one being copied means the
content was taken). Each test is first put on the same 0-100% scale with
"anchors": the typical unrelated pair maps to 0% and the typical copy to 100%.
Anchors come from your calibration set (calibrate.py) or, before that, the
rough defaults below.

Pure numpy / stdlib so it can be tested without a GPU.
"""

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np

TESTS = ["body", "scene", "script"]
TEST_LABELS = {
    "body": "Body & motion",
    "scene": "Scene & visuals",
    "script": "Script",
}

# Rough starting anchors (raw score -> 0% / 100%) until you calibrate.
DEFAULT_ANCHORS = {
    "body": {"low": 0.35, "high": 0.85},
    "scene": {"low": 0.40, "high": 0.85},
    "script": {"low": 0.05, "high": 0.60},
}
DEFAULT_VERDICT = {"match": 0.80, "unrelated": 0.55}

SCENE_SEGMENT_FLOOR = 0.55  # (older versions) scene frames must beat this to form a section
MOMENT_TAU = 0.5  # two frames are "the same moment" when their room-removed fingerprints agree this well
MIN_PHRASE_WORDS = 4  # shared runs shorter than this are ignored ("i want to")
MIN_PHRASE_CHARS = 8  # same, for languages written without spaces (CJK)
MIN_TRANSCRIPT_WORDS = 8  # below this a video "has no script"


# --------------------------------------------------------------------------
# Scene & visuals
# --------------------------------------------------------------------------


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, D), b: (M, D) embeddings -> (N, M) cosine similarities."""
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return (a @ b.T).astype(np.float32)


def patch_person_mask(alpha: np.ndarray | None, grid_hw: tuple[int, int]) -> np.ndarray:
    """Which cells of the image's patch grid contain the person (or are right
    next to them). alpha: full-frame matting alpha; returns (h, w) bool."""
    h, w = grid_hw
    if alpha is None:
        return np.zeros((h, w), bool)
    import cv2

    cover = cv2.resize((alpha > 0.5).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    m = (cover > 0.05).astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0  # one-cell margin round the person


def background_similarity(pa, ma, pb, mb, shift: int = 1) -> float | None:
    """How well two frames' backgrounds line up, location by location.
    pa/pb: (h, w, D) unit-length patch features; ma/mb: (h, w) person masks.
    For every background cell of A, take the best match among the cells at the
    same spot in B (+/- `shift` cells, for a slightly moved camera), then
    average. Cells covered by the person in either frame are ignored, so a
    swapped person doesn't count against it, but the table, wall, ceiling and
    props do. Returns None when there's almost no shared background."""
    h, w, _ = pa.shape
    best = np.full((h, w), -1.0, np.float32)
    okb_any = np.zeros((h, w), bool)
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            ys, yd = (slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy)))
            xs, xd = (slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx)))
            sim = (pa[ys, xs] * pb[yd, xd]).sum(-1)
            okb = ~mb[yd, xd]
            sim = np.where(okb, sim, -1.0)
            best[ys, xs] = np.maximum(best[ys, xs], sim)
            okb_any[ys, xs] |= okb
    valid = ~ma & okb_any
    if valid.sum() < 0.1 * h * w:
        return None
    return float(np.clip(best[valid], 0, 1).mean())


def moment_matrix(emb_a: dict, emb_b: dict) -> np.ndarray:
    """Which moment matches which. When both videos are filmed in the same
    room, every frame looks alike, so the shared "room" part of the
    fingerprints is removed first (subtract the average fingerprint of both
    videos); what's left is what changes from moment to moment: the person's
    posture, what they're holding, what's on the desk."""
    a, b = emb_a["cls"], emb_b["cls"]
    mean = np.concatenate([a, b]).mean(0, keepdims=True)
    return np.clip(cosine_matrix(a - mean, b - mean), 0, 1)


def _bg_along(path, ea, eb):
    vals = [background_similarity(ea["patch"][a], ea["mask"][a], eb["patch"][b], eb["mask"][b])
            for a, b in path]
    return [v for v in vals], [v for v in vals if v is not None]


def compare_scene(emb_a: dict, emb_b: dict, emb_b_flipped: dict | None, dtw=None) -> dict:
    """emb_*: {"cls": (T, D), "patch": (T, h, w, D) unit vectors, "mask": (T, h, w) bool}.
    1. Line up the moments (moment matrix + section finding).
    2. Score each matched pair on how well the backgrounds line up, location by
       location: same table, wall, ceiling, props, framing. The test's score is
       its strongest section."""
    from similarity import find_segments, subsequence_dtw

    best = None
    options = [(False, emb_b)] + ([(True, emb_b_flipped)] if emb_b_flipped else [])
    for flipped, eb in options:
        Mo = moment_matrix(emb_a, eb)
        segs = find_segments(Mo, tau=MOMENT_TAU)
        if not segs:  # nothing stands out: align the whole videos
            a_rows = Mo.shape[0] <= Mo.shape[1]
            path = subsequence_dtw(Mo if a_rows else Mo.T)
            if not a_rows:
                path = [(j, i) for i, j in path]
            segs_eval = [{"path": path}]
        else:
            segs_eval = segs
        for g in segs_eval:
            g["bg"], ok = _bg_along(g["path"], emb_a, eb)
            g["mean"] = float(np.mean(ok)) if ok else 0.0
        top = max(segs_eval, key=lambda g: g["mean"])
        score = top["mean"]
        if best is None or score > best["raw"] + 0.03:
            best = {"raw": score, "mirrored": flipped, "segs": segs, "evals": segs_eval, "Mo": Mo}
    path = [p for g in best["evals"] for p in g["path"]]
    bg = [v for g in best["evals"] for v in g["bg"]]
    Mo = best["Mo"]
    return {
        "raw": best["raw"],
        "mirrored": best["mirrored"],
        "path": [(int(a), int(b)) for a, b in path],
        "timeline": [round(float(v), 4) if v is not None else 0.0 for v in bg],
        # how well each matched pair is the *same moment* (used to pick examples)
        "moment": [round(float(Mo[a, b]) * (0.5 + 0.5 * (v or 0)), 4) for (a, b), v in zip(path, bg)],
        "sections": [{
            "a_start": int(min(a for a, _ in g["path"])), "a_end": int(max(a for a, _ in g["path"])),
            "b_start": int(min(b for _, b in g["path"])), "b_end": int(max(b for _, b in g["path"])),
            "similarity": round(float(g["mean"]), 4),
        } for g in best["segs"]],
    }


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def _norm(tok: str) -> str:
    tok = unicodedata.normalize("NFKC", tok).lower()
    return re.sub(r"[^\w]", "", tok)


def tokenize(text: str) -> tuple[list[str], list[str], bool]:
    """Returns (display tokens, normalised tokens, is_char_level).
    Space-separated languages are compared word by word; Chinese/Japanese/
    Korean text without spaces is compared character by character."""
    text = (text or "").strip()
    cjk = len(_CJK.findall(text)) > 0.3 * max(1, len(re.sub(r"\s", "", text)))
    if cjk:
        disp = [c for c in text if not c.isspace()]
        norm = [_norm(c) for c in disp]
    else:
        disp = text.split()
        norm = [_norm(t) for t in disp]
    return disp, norm, cjk


def compare_script(text_a: str, text_b: str) -> dict:
    """How much of the shorter transcript also appears in the other one, as
    shared runs of MIN_PHRASE_WORDS+ words. 1.0 = word-for-word the same."""
    da, na, cjk_a = tokenize(text_a)
    db, nb, cjk_b = tokenize(text_b)
    char_level = cjk_a or cjk_b
    # drop tokens that normalise to nothing (pure punctuation / emoji), keep index map
    ia = [i for i, t in enumerate(na) if t]
    ib = [i for i, t in enumerate(nb) if t]
    wa, wb = [na[i] for i in ia], [nb[i] for i in ib]
    out = {"tokens_a": da, "tokens_b": db, "spans_a": [], "spans_b": [],
           "char_level": char_level, "words_a": len(wa), "words_b": len(wb)}
    min_len = MIN_TRANSCRIPT_WORDS * (3 if char_level else 1)
    if len(wa) < min_len or len(wb) < min_len:
        out.update(raw=None, reason="Not enough speech in "
                   + ("both videos" if len(wa) < min_len and len(wb) < min_len
                      else "video A" if len(wa) < min_len else "video B"))
        return out
    min_run = MIN_PHRASE_CHARS if char_level else MIN_PHRASE_WORDS
    sm = SequenceMatcher(None, wa, wb, autojunk=False)
    shared = 0
    for blk in sm.get_matching_blocks():
        if blk.size >= min_run:
            shared += blk.size
            out["spans_a"].append([ia[blk.a], ia[blk.a + blk.size - 1] + 1])
            out["spans_b"].append([ib[blk.b], ib[blk.b + blk.size - 1] + 1])
    out["raw"] = shared / min(len(wa), len(wb))
    out["shared_words"] = shared
    return out


# --------------------------------------------------------------------------
# Combining the three tests
# --------------------------------------------------------------------------


def anchors(calib: dict | None, test: str) -> dict:
    c = (calib or {}).get("tests", {}).get(test)
    if test == "body" and not c and calib and calib.get("overall"):
        c = calib["overall"]  # calibration files from before the 3-test version
    return c or DEFAULT_ANCHORS[test]


def to_scale(raw: float | None, a: dict) -> float | None:
    if raw is None:
        return None
    span = a["high"] - a["low"]
    if span <= 1e-6:
        return float(np.clip(raw, 0, 1))
    return float(np.clip((raw - a["low"]) / span, 0, 1))


def combine(raw: dict, calib: dict | None) -> dict:
    """raw: {"body": float|None, "scene": float|None, "script": float|None}.
    Returns per-test scores on the common scale, the final (max) score, which
    test drove it, and the verdict."""
    scores = {t: to_scale(raw.get(t), anchors(calib, t)) for t in TESTS}
    available = {t: s for t, s in scores.items() if s is not None}
    if not available:
        return {"scores": scores, "final": None, "driver": None,
                "verdict": "Could not analyse these videos"}
    driver = max(available, key=available.get)
    final = available[driver]
    v = (calib or {}).get("verdict") or DEFAULT_VERDICT
    if final >= v["match"]:
        verdict = "Likely copied content"
    elif final >= v["unrelated"]:
        verdict = "Partial or inconclusive match"
    else:
        verdict = "Likely unrelated content"
    return {"scores": scores, "final": final, "driver": driver, "verdict": verdict,
            "calibrated": bool(calib and calib.get("tests"))}
