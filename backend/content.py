"""
The two content tests that sit alongside the Sapiens2 body & motion test,
plus the logic that combines all three into the final score.

  scene   Do the frames look alike (room, objects, framing, lighting)?
          Uses DINOv2 image embeddings of each sampled frame, both the full
          frame and the frame with the person blanked out, so a face swap or
          body swap over the same footage still matches.
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
    "scene": {"low": 0.45, "high": 0.92},
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


def scene_matrix(emb_a: dict, emb_b: dict) -> np.ndarray:
    """emb_*: {"full": (T, D), "bg": (T, D)} -> (N, M) similarity in 0..1.
    Averages the whole-frame and person-removed views."""
    full = cosine_matrix(emb_a["full"], emb_b["full"])
    bg = cosine_matrix(emb_a["bg"], emb_b["bg"])
    return np.clip(0.5 * full + 0.5 * bg, 0, 1)


def blank_person(img_bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Replace the person (alpha > 0.5, slightly grown) with flat grey, so the
    embedding describes only the setting."""
    if alpha is None:
        return img_bgr
    import cv2

    mask = (alpha > 0.5).astype(np.uint8)
    k = max(3, int(round(min(mask.shape) * 0.03)) | 1)
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0
    out = img_bgr.copy()
    out[mask] = 128
    return out


def moment_matrix(emb_a: dict, emb_b: dict) -> np.ndarray:
    """Which moment matches which. When both videos are filmed in the same
    room, every frame looks alike, so the shared "room" part of the
    fingerprints is removed first (subtract the average fingerprint of both
    videos); what's left is what changes from moment to moment: the person's
    posture, what they're holding, what's on the desk."""
    a, b = emb_a["full"], emb_b["full"]
    mean = np.concatenate([a, b]).mean(0, keepdims=True)
    return np.clip(cosine_matrix(a - mean, b - mean), 0, 1)


def compare_scene(emb_a: dict, emb_b: dict, emb_b_flipped: dict | None, dtw=None) -> dict:
    """Line up the moments with the moment matrix, then score how alike the
    matched frames look overall (room, objects, framing, lighting)."""
    from similarity import find_segments, subsequence_dtw

    best = None
    options = [(False, emb_b)] + ([(True, emb_b_flipped)] if emb_b_flipped else [])
    for flipped, eb in options:
        S = scene_matrix(emb_a, eb)  # how alike the frames look
        Mo = moment_matrix(emb_a, eb)  # which moment matches which
        segs = find_segments(Mo, tau=MOMENT_TAU)
        if segs:
            for g in segs:
                g["mean"] = float(np.mean([S[a, b] for a, b in g["path"]]))
            top = max(segs, key=lambda g: g["mean"])
            score, path = top["mean"], [p for g in segs for p in g["path"]]
        else:  # nothing stands out: align the whole videos
            a_rows = S.shape[0] <= S.shape[1]
            path = subsequence_dtw(Mo if a_rows else Mo.T)
            if not a_rows:
                path = [(j, i) for i, j in path]
            score = float(np.mean([S[a, b] for a, b in path]))
        if best is None or score > best["raw"] + 0.03:
            best = {"raw": score, "mirrored": flipped, "path": path, "S": S, "Mo": Mo, "segs": segs}
    S, Mo = best["S"], best["Mo"]
    return {
        "raw": best["raw"],
        "mirrored": best["mirrored"],
        "path": [(int(a), int(b)) for a, b in best["path"]],
        "timeline": [round(float(S[a, b]), 4) for a, b in best["path"]],
        # how well each matched pair is the *same moment* (used to pick examples)
        "moment": [round(float(Mo[a, b]), 4) for a, b in best["path"]],
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
