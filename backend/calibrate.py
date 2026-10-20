"""
Test and calibrate the similarity score on your own labelled videos.

1. Make a folder, e.g. my_test_set/, containing your videos and a pairs.csv:

       video_a,video_b,label,note
       maya_original.mp4,maya_faceswap_ad.mp4,match,face swap + new script
       maya_original.mp4,maya_mirrored_ad.mp4,match,flipped and cropped
       maya_original.mp4,maya_other_video.mp4,unrelated,same creator, different footage
       maya_original.mp4,random_dancer.mp4,unrelated,

   label is "match" (B was made from A's footage) or "unrelated".
   Include some "same person, different video" pairs labelled unrelated —
   they are the hardest case and the most important to get right.
   Aim for at least 10 of each label; 30+ is better.

2. Run (after `modal deploy backend/modal_app.py`):

       python backend/calibrate.py my_test_set

   It uploads the videos, scores every pair on the GPU, and writes:
     my_test_set/calibration_results.json   raw scores (reuse with --reuse)
     my_test_set/calibration_report.html    open in a browser
     backend/calibration/calibration.json   thresholds the website will use

3. Run `modal deploy backend/modal_app.py` again so the site uses them.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from content import DEFAULT_ANCHORS, TEST_LABELS, TESTS, to_scale  # noqa: E402
from similarity import DEFAULT_WEIGHTS, MODELS  # noqa: E402

APP_NAME = "origin-video-compare"
MATCH = {"match", "copy", "same", "derived", "1", "yes", "true"}
UNRELATED = {"unrelated", "different", "0", "no", "false", "negative"}


def read_pairs(folder: Path) -> list[dict]:
    pairs = []
    with open(folder / "pairs.csv", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            lab = (row.get("label") or "").strip().lower()
            if lab in MATCH:
                y = 1
            elif lab in UNRELATED:
                y = 0
            else:
                sys.exit(f"pairs.csv line {n}: label must be 'match' or 'unrelated', got {lab!r}")
            a, b = row["video_a"].strip(), row["video_b"].strip()
            for v in (a, b):
                if not (folder / v).exists():
                    sys.exit(f"pairs.csv line {n}: can't find {folder / v}")
            pairs.append({"a": a, "b": b, "y": y, "note": (row.get("note") or "").strip()})
    return pairs


def score_on_gpu(folder: Path, pairs: list[dict]) -> list[dict]:
    import modal

    vol = modal.Volume.from_name("origin-calibration-videos", create_if_missing=True)
    files = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    print(f"Uploading {len(files)} videos...")
    with vol.batch_upload(force=True) as batch:
        for v in files:
            batch.put_file(str(folder / v), f"/{v}")
    print(f"Scoring {len(pairs)} pairs on the GPU (roughly 1-2 minutes per pair, run in parallel)...")
    fn = modal.Function.from_name(APP_NAME, "score_calibration_pairs")
    return fn.remote(pairs)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random match scores higher than a random unrelated pair."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean()
    eq = (pos[:, None] == neg[None]).mean()
    return float(gt + 0.5 * eq)


def best_threshold(scores: np.ndarray, y: np.ndarray):
    """Threshold maximising balanced accuracy."""
    best = (0.0, 0.5)
    for t in np.unique(scores):
        pred = scores >= t
        tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        tnr = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        bal = (tpr + tnr) / 2
        if bal > best[0]:
            best = (bal, float(t))
    return best[1], best[0]


def _anchor(raw: np.ndarray, y: np.ndarray, default: dict) -> dict:
    pos, neg = raw[y == 1], raw[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return dict(default)
    low, high = float(np.median(neg)), float(np.median(pos))
    return {"low": low, "high": high} if high > low + 1e-3 else dict(default)


def calibrate(results: list[dict]) -> tuple[dict, dict]:
    ok = [r for r in results if "error" not in r and r.get("raw")]
    y = np.array([r["y"] for r in ok])
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        sys.exit("Need at least 2 scored 'match' and 2 'unrelated' pairs to calibrate.")
    nan = float("nan")

    # 1. Body & motion: learn how much to trust each of the five Sapiens2 models
    body_ok = [i for i, r in enumerate(ok) if r.get("models_raw")]
    M = {m: np.array([ok[i]["models_raw"][m] for i in body_ok]) for m in MODELS}
    yb = y[body_ok]
    model_stats = {}
    for m in MODELS:
        pos, neg = M[m][yb == 1], M[m][yb == 0]
        model_stats[m] = {"auc": auc(pos, neg),
                          "match_mean": float(pos.mean()) if len(pos) else nan,
                          "unrelated_mean": float(neg.mean()) if len(neg) else nan}
    raw_w = {m: max((model_stats[m]["auc"] if model_stats[m]["auc"] == model_stats[m]["auc"] else 0.5) - 0.5, 0.0)
             for m in MODELS}
    if sum(raw_w.values()) < 1e-6:
        weights = dict(DEFAULT_WEIGHTS)
    else:  # blend with defaults so no model drops to zero
        weights = {m: round(0.8 * raw_w[m] / sum(raw_w.values()) + 0.2 * DEFAULT_WEIGHTS[m], 4)
                   for m in MODELS}
    tw = sum(weights.values())

    # 2. Raw score per test (body re-weighted with the learned weights)
    raw = {t: np.full(len(ok), np.nan) for t in TESTS}
    for i, r in enumerate(ok):
        if r.get("models_raw"):
            raw["body"][i] = sum(weights[m] * r["models_raw"][m] for m in MODELS) / tw
        for t in ("scene", "script"):
            if r["raw"].get(t) is not None:
                raw[t][i] = r["raw"][t]

    # 3. Anchors per test: typical unrelated pair -> 0%, typical copy -> 100%
    tests, test_stats, scaled = {}, {}, {}
    for t in TESTS:
        have = ~np.isnan(raw[t])
        tests[t] = _anchor(raw[t][have], y[have], DEFAULT_ANCHORS[t])
        scaled[t] = np.array([to_scale(v, tests[t]) if v == v else np.nan for v in raw[t]])
        test_stats[t] = {"auc": auc(scaled[t][have & (y == 1)], scaled[t][have & (y == 0)]),
                         "n": int(have.sum())}

    # 4. Final = highest available test; set the verdict thresholds on it
    stacked = np.vstack([scaled[t] for t in TESTS])
    final = np.nanmax(np.where(np.isnan(stacked), -1, stacked), axis=0)
    final = np.clip(final, 0, 1)
    driver = [TESTS[int(np.nanargmax(np.where(np.isnan(stacked[:, i]), -1, stacked[:, i])))]
              for i in range(len(ok))]
    pos, neg = final[y == 1], final[y == 0]
    t_best, bal_acc = best_threshold(final, y)
    # "Inconclusive" is the band between the top of the unrelated pairs (95th
    # percentile) and the bottom of the copies (5th percentile).
    lo, hi = float(np.percentile(neg, 95)), float(np.percentile(pos, 5))
    t_unrel, t_match = min(lo, hi), max(lo, hi)

    calibration = {
        "weights": weights,
        "tests": tests,
        "models": {m: {"low": model_stats[m]["unrelated_mean"], "high": model_stats[m]["match_mean"]}
                   for m in MODELS if model_stats[m]["match_mean"] == model_stats[m]["match_mean"]},
        "verdict": {"match": round(t_match, 3), "unrelated": round(t_unrel, 3)},
        "n_pairs": {"match": int((y == 1).sum()), "unrelated": int((y == 0).sum())},
    }
    summary = {
        "model_stats": model_stats, "test_stats": test_stats, "overall_auc": auc(pos, neg),
        "balanced_accuracy": bal_acc, "threshold": t_best, "final": final.tolist(),
        "scaled": {t: scaled[t].tolist() for t in TESTS}, "driver": driver,
        "pred": (final >= t_best).tolist(), "y": y.tolist(), "ok": ok,
    }
    return calibration, summary


def _hist_svg(pos, neg, w=420, h=110, bins=20):
    edges = np.linspace(0, 1, bins + 1)
    hp, _ = np.histogram(np.clip(pos, 0, 1), edges)
    hn, _ = np.histogram(np.clip(neg, 0, 1), edges)
    top = max(hp.max(initial=1), hn.max(initial=1))
    bw = w / bins
    bars = []
    for i in range(bins):
        for cnt, color, off in ((hn[i], "#b8b3a7", 0), (hp[i], "#1f4d3a", bw / 2)):
            bh = (h - 16) * cnt / top
            bars.append(f'<rect x="{i * bw + off:.1f}" y="{h - 16 - bh:.1f}" width="{bw / 2 - 1:.1f}" '
                        f'height="{bh:.1f}" fill="{color}"/>')
    axis = (f'<text x="0" y="{h - 2}" font-size="10" fill="#777">0%</text>'
            f'<text x="{w - 28}" y="{h - 2}" font-size="10" fill="#777">100%</text>')
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(bars)}{axis}</svg>'


def write_report(folder: Path, results, calibration, summary):
    ok = summary["ok"]
    y = np.array(summary["y"])
    final = np.array(summary["final"])
    fmt = lambda v: "–" if v is None or v != v else f"{v * 100:.0f}%"  # noqa: E731
    rows = []
    for i, r in enumerate(ok):
        correct = bool(summary["pred"][i]) == bool(r["y"])
        rows.append(
            f"<tr class='{'' if correct else 'wrong'}'><td>{html.escape(r['a'])}</td>"
            f"<td>{html.escape(r['b'])}</td><td>{'match' if r['y'] else 'unrelated'}</td>"
            f"<td class='num'><b>{fmt(final[i])}</b></td>"
            + "".join(f"<td class='num'>{fmt(summary['scaled'][t][i])}</td>" for t in TESTS)
            + f"<td>{TEST_LABELS[summary['driver'][i]]}</td>"
            f"<td>{html.escape(r.get('note', ''))}</td><td>{'✓' if correct else '✗'}</td></tr>"
        )
    errs = "".join(f"<li>{html.escape(r['a'])} vs {html.escape(r['b'])}: {html.escape(r['error'])}</li>"
                   for r in results if "error" in r)

    def dist(values, mask):
        v = np.array(values)
        have = ~np.isnan(v) & mask
        return _hist_svg(v[have & (y == 1)], v[have & (y == 0)])

    all_mask = np.ones(len(ok), bool)
    test_rows = "".join(
        f"<tr><td>{TEST_LABELS[t]}</td><td class='num'>{st['n']}</td>"
        f"<td class='num'>{st['auc']:.2f}</td>"
        f"<td class='num'>{calibration['tests'][t]['low'] * 100:.0f} → {calibration['tests'][t]['high'] * 100:.0f}</td>"
        f"<td>{dist(summary['scaled'][t], all_mask)}</td></tr>"
        for t, st in summary["test_stats"].items()
    )
    model_rows = "".join(
        f"<tr><td>{m}</td><td class='num'>{s['auc']:.2f}</td>"
        f"<td class='num'>{fmt(s['match_mean'])}</td><td class='num'>{fmt(s['unrelated_mean'])}</td>"
        f"<td class='num'>{calibration['weights'][m]:.2f}</td></tr>"
        for m, s in summary["model_stats"].items()
    )
    page = f"""<!doctype html><meta charset="utf-8"><title>Calibration report</title>
<style>body{{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 16px;color:#1a1a1a;background:#f7f6f3}}
h1{{font-family:Georgia,serif;font-weight:400}}table{{border-collapse:collapse;width:100%;background:#fff;margin:16px 0}}
td,th{{border-bottom:1px solid #e6e3dc;padding:6px 8px;text-align:left;vertical-align:middle}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
.wrong{{background:#fbecea}}.kpi{{display:inline-block;margin-right:32px}}.kpi b{{font-size:28px;display:block}}
.legend span{{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px}}</style>
<h1>Calibration report</h1>
<p>{calibration['n_pairs']['match']} match pairs, {calibration['n_pairs']['unrelated']} unrelated pairs.
The final score is the highest of the three tests.</p>
<div class="kpi"><b>{summary['overall_auc']:.2f}</b>Final-score AUC (1.0 = perfect separation, 0.5 = coin flip)</div>
<div class="kpi"><b>{summary['balanced_accuracy'] * 100:.0f}%</b>Balanced accuracy at best threshold ({summary['threshold'] * 100:.0f}%)</div>
<h2>Final score distribution</h2>
<p class="legend"><span style="background:#1f4d3a"></span>match <span style="background:#b8b3a7"></span>unrelated</p>
{_hist_svg(final[y == 1], final[y == 0], w=700, h=140)}
<p>Verdict bands now used by the site: ≥ {calibration['verdict']['match'] * 100:.0f}% "Likely copied content",
&lt; {calibration['verdict']['unrelated'] * 100:.0f}% "Likely unrelated content", in between "Inconclusive".</p>
<h2>Per test</h2><p>A test with low AUC is raising false alarms or missing copies; look at its red rows below.</p>
<table><tr><th>Test</th><th>Pairs scored</th><th>AUC</th><th>Raw anchors (0% → 100%)</th><th>Score distribution</th></tr>{test_rows}</table>
<h2>Body &amp; motion: the five Sapiens2 models</h2>
<table><tr><th>Model</th><th>AUC</th><th>Match mean</th><th>Unrelated mean</th><th>New weight</th></tr>{model_rows}</table>
<h2>Every pair</h2><p>Red rows are misclassified at the best threshold. Look at these first.</p>
<table><tr><th>Video A</th><th>Video B</th><th>Label</th><th>Final</th>{''.join(f'<th>{TEST_LABELS[t]}</th>' for t in TESTS)}<th>Driven by</th><th>Note</th><th></th></tr>{''.join(rows)}</table>
{f'<h2>Failed pairs</h2><ul>{errs}</ul>' if errs else ''}"""
    (folder / "calibration_report.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--reuse", action="store_true", help="re-use saved scores instead of re-running the GPU")
    args = ap.parse_args()
    folder = args.folder
    cache = folder / "calibration_results.json"

    if args.reuse and cache.exists():
        results = json.loads(cache.read_text())
    else:
        results = score_on_gpu(folder, read_pairs(folder))
        cache.write_text(json.dumps(results, indent=2))

    calibration, summary = calibrate(results)
    out = Path(__file__).parent / "calibration" / "calibration.json"
    out.write_text(json.dumps(calibration, indent=2))
    write_report(folder, results, calibration, summary)
    print(f"\nFinal-score AUC {summary['overall_auc']:.2f}, balanced accuracy {summary['balanced_accuracy'] * 100:.0f}%")
    print(f"Report:      {folder / 'calibration_report.html'}")
    print(f"Thresholds:  {out}")
    print("Now run `modal deploy backend/modal_app.py` so the website uses them.")


if __name__ == "__main__":
    main()
