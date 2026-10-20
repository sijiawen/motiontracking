"""Turn model outputs into the side-by-side example images shown on the site."""

from __future__ import annotations

import cv2
import numpy as np

LABELS = {
    "pose": "Pose estimation",
    "seg": "Body-part segmentation",
    "normal": "Surface normals",
    "pointmap": "Pointmap (3D depth)",
    "matting": "Human matting",
    "scene": "Scene & visuals",
}
EXAMPLES_PER_MODEL = 3
OUT_HEIGHT = 420  # px height of each half of an example image
SAMPLE_FPS = 2


def _person_mask(out, i):
    return out["matting"][i][3] > 0.5


def vis_pose(img, out, i, metainfo):
    from pose_render_utils import visualize_keypoints  # from the Sapiens2 repo

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = (rgb * 0.55).astype(np.uint8)  # dim background so the skeleton pops
    k, s = out["pose_kpts"][i], out["pose_scores"][i]
    r = max(2, int(round(img.shape[0] / 300)))
    vis = visualize_keypoints(
        image=rgb, keypoints=[k], keypoints_visible=[np.ones_like(s) > 0], keypoint_scores=[s],
        radius=r, thickness=max(1, r // 2), kpt_thr=0.3,
        skeleton=metainfo["skeleton_links"], kpt_color=metainfo["keypoint_colors"],
        link_color=metainfo["skeleton_link_colors"],
    )
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def vis_seg(img, out, i, _):
    from sapiens.dense.visualizers import SegVisualizer

    v = SegVisualizer(class_palette_type="dome29", with_labels=False)
    return v._visualize_segmentation(img, out["seg"][i])


def vis_normal(img, out, i, _):
    n = out["normal"][i].copy()
    mask = _person_mask(out, i)
    n[~mask] = -1
    v = ((n + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1]
    v[~mask] = 24
    return v


def vis_pointmap(img, out, i, _):
    d = out["pointmap"][i]
    mask = _person_mask(out, i)
    v = np.full(img.shape, 24, np.uint8)
    if mask.sum() > 10:
        dv = d[mask]
        lo, hi = np.percentile(dv, 2), np.percentile(dv, 98)
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        col = cv2.applyColorMap(((1 - norm) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        v[mask] = col[mask]  # brighter = closer to the camera
    return v


def vis_matting(img, out, i, _):
    m = out["matting"][i]
    fg = m[:3].transpose(1, 2, 0)[:, :, ::-1]  # premultiplied RGB -> BGR
    alpha = m[3][..., None]
    bg = np.array([236, 240, 243], np.float32) / 255  # pale grey "paper" backdrop
    comp = fg + (1 - alpha) * bg
    return (np.clip(comp, 0, 1) * 255).astype(np.uint8)


VIS = {"pose": vis_pose, "seg": vis_seg, "normal": vis_normal,
       "pointmap": vis_pointmap, "matting": vis_matting}


def _fit(im, h=OUT_HEIGHT):
    return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)


def pick_examples(path, n=EXAMPLES_PER_MODEL):
    if len(path) <= n:
        return list(range(len(path)))
    return [int(round(x)) for x in np.linspace(len(path) * 0.15, len(path) * 0.85, n)]


def render_examples(sb, cid, res, va, vb, metainfo):
    imgs_a, out_a, _ = va
    imgs_b, out_b, _ = vb
    picks = pick_examples(res.path)
    frames = {}
    for model, fn in VIS.items():
        frames[model] = []
        for n, p in enumerate(picks):
            ia, ib = res.path[p]
            left = _fit(fn(imgs_a[ia], out_a, ia, metainfo))
            right = _fit(fn(imgs_b[ib], out_b, ib, metainfo))
            gap = np.full((OUT_HEIGHT, 6, 3), 255, np.uint8)
            combo = np.concatenate([left, gap, right], 1)
            ok, buf = cv2.imencode(".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 84])
            path = f"{cid}/{model}_{n}.jpg"
            sb.storage.from_("results").upload(
                path, buf.tobytes(), {"content-type": "image/jpeg", "upsert": "true"}
            )
            frames[model].append({
                "path": path,
                "a_time": round(ia / SAMPLE_FPS, 1),
                "b_time": round(ib / SAMPLE_FPS, 1),
                "similarity": res.timelines[model][p],
            })
    return frames


def _save_pair(sb, cid, name, left, right):
    left, right = _fit(left), _fit(right)
    gap = np.full((OUT_HEIGHT, 6, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", np.concatenate([left, gap, right], 1),
                           [cv2.IMWRITE_JPEG_QUALITY, 84])
    path = f"{cid}/{name}.jpg"
    sb.storage.from_("results").upload(
        path, buf.tobytes(), {"content-type": "image/jpeg", "upsert": "true"}
    )
    return path


def render_scene_examples(sb, cid, scene, imgs_a, imgs_b):
    """The raw frames the scene test matched, side by side."""
    out = []
    for n, p in enumerate(pick_examples(scene["path"])):
        ia, ib = scene["path"][p]
        out.append({
            "path": _save_pair(sb, cid, f"scene_{n}", imgs_a[ia], imgs_b[ib]),
            "a_time": round(ia / SAMPLE_FPS, 1),
            "b_time": round(ib / SAMPLE_FPS, 1),
            "similarity": scene["timeline"][p],
        })
    return out
