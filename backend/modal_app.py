"""
GPU backend for the video comparison MVP, deployed on Modal (modal.com).

    modal run backend/modal_app.py::download_models   # once, ~40 GB of weights
    modal deploy backend/modal_app.py                 # puts the API online

Two pieces live here:
  * `Analyzer`  - a GPU worker that loads all five Sapiens2 1B models once and
                  compares two videos stored in Supabase.
  * `api`       - a small web API the website talks to (create upload slots,
                  start analysis, poll status, admin list).
"""


import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal

APP_NAME = "origin-video-compare"
SAPIENS_DIR = "/opt/sapiens2"
CKPT_DIR = "/ckpt"  # Modal Volume with model weights

SAMPLE_FPS = 2  # frames analysed per second of video
MAX_SECONDS = 60
MAX_BYTES = 50 * 1024 * 1024
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "0"))  # comparisons per visitor per day; 0 = unlimited
BATCH = 8
EARLY_STOP_ENABLED = False  # True = skip the more expensive tests once one scores EARLY_STOP+ (saves GPU)
EARLY_STOP = 0.80
GPU = "H100"  # 5 x 1B models need ~35 GB; A100-80GB also works

HF_REPOS = {
    "pose": ("facebook/sapiens2-pose-1b", "sapiens2_1b_pose.safetensors"),
    "seg": ("facebook/sapiens2-seg-1b", "sapiens2_1b_seg.safetensors"),
    "normal": ("facebook/sapiens2-normal-1b", "sapiens2_1b_normal.safetensors"),
    "pointmap": ("facebook/sapiens2-pointmap-1b", "sapiens2_1b_pointmap.safetensors"),
    "matting": ("facebook/sapiens2-matting-1b", "sapiens2_1b_matting.safetensors"),
}
CONFIGS = {
    "pose": "pose/configs/keypoints308/shutterstock_goliath_3po/"
    "sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "seg": "dense/configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py",
    "normal": "dense/configs/normal/metasim_render_people/"
    "sapiens2_1b_normal_metasim_render_people-1024x768.py",
    "pointmap": "dense/configs/pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py",
    "matting": "dense/configs/matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py",
}
DETECTOR = "facebook/detr-resnet-101-dc5"
SCENE_MODEL = "facebook/dinov2-base"  # image embeddings for the scene test
SPEECH_MODEL = "openai/whisper-large-v3-turbo"  # speech-to-text for the script test

here = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.7", "torchvision", "numpy", "opencv-python-headless", "pillow",
        "safetensors", "scipy", "matplotlib", "iopath", "prettytable", "termcolor",
        "accelerate", "rich", "transformers", "timm", "tqdm",
        "huggingface_hub", "supabase>=2.4", "fastapi[standard]",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/facebookresearch/sapiens2.git {SAPIENS_DIR}",
        f"pip install --no-deps -e {SAPIENS_DIR}",
    )
    .env({"HF_HOME": f"{CKPT_DIR}/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_file(here / "similarity.py", "/root/similarity.py")
    .add_local_file(here / "render.py", "/root/render.py")
    .add_local_file(here / "content.py", "/root/content.py")
    .add_local_dir(here / "calibration", "/root/calibration")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("origin-sapiens2-weights", create_if_missing=True)
secrets = [modal.Secret.from_name("origin-secrets")]


# ---------------------------------------------------------------- helpers


def supabase():
    from supabase import create_client

    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def _signed_url(sb, bucket: str, path: str, expires: int = 3600) -> str | None:
    try:
        r = sb.storage.from_(bucket).create_signed_url(path, expires)
        return r.get("signedURL") or r.get("signedUrl") or r.get("signed_url")
    except Exception:
        return None


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def extract_frames(path: str, out_dir: str) -> list[str]:
    """Sample SAMPLE_FPS frames per second, longest side capped at 1024 px.
    ffmpeg handles iPhone HEVC/.mov, rotation metadata, odd codecs, etc."""
    os.makedirs(out_dir, exist_ok=True)
    vf = (
        f"fps={SAMPLE_FPS},"
        "scale='if(gt(iw,ih),min(1024,iw),-2)':'if(gt(iw,ih),-2,min(1024,ih))'"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", path, "-t", str(MAX_SECONDS + 1),
         "-vf", vf, "-q:v", "2", f"{out_dir}/%05d.jpg"],
        check=True,
    )
    return sorted(str(p) for p in Path(out_dir).glob("*.jpg"))


# ---------------------------------------------------------------- weights


@app.function(volumes={CKPT_DIR: volume}, secrets=secrets, timeout=3600)
def download_models():
    """Fetch the five Sapiens2-1B checkpoints and the person detector once."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    for task, (repo, fname) in HF_REPOS.items():
        dest = Path(CKPT_DIR) / task / fname
        if dest.exists():
            print("already have", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(repo, fname, token=token, local_dir=dest.parent)
        print("downloaded", p)
    snapshot_download(DETECTOR, token=token, local_dir=f"{CKPT_DIR}/detector/detr-resnet-101-dc5")
    for repo in (SCENE_MODEL, SPEECH_MODEL):  # cached under HF_HOME on the volume
        print("fetching", repo)
        snapshot_download(repo, token=token, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    volume.commit()
    print("All weights ready.")


# ---------------------------------------------------------------- GPU worker


@app.cls(
    gpu=GPU,
    volumes={CKPT_DIR: volume},
    secrets=secrets,
    timeout=900,
    scaledown_window=120,  # keep warm 2 min after a job, then stop paying
    max_containers=2,  # hard cap on simultaneous GPUs (cost control)
)
class Analyzer:
    @modal.enter()
    def load(self):
        import torch

        sys.path.insert(0, f"{SAPIENS_DIR}/sapiens/pose/tools/vis")
        from sapiens.dense.models import init_model as init_dense
        from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
        from sapiens.pose.models import init_model as init_pose
        from transformers import DetrForObjectDetection, DetrImageProcessor

        import similarity

        t0 = time.time()
        self.device = "cuda:0"
        root = f"{SAPIENS_DIR}/sapiens"
        ck = lambda t: f"{CKPT_DIR}/{t}/{HF_REPOS[t][1]}"  # noqa: E731

        self.models = {}
        for task in ("seg", "normal", "pointmap", "matting"):
            self.models[task] = init_dense(f"{root}/{CONFIGS[task]}", ck(task), device=self.device)

        pose = init_pose(f"{root}/{CONFIGS['pose']}", ck("pose"), device=self.device)
        pose.pose_metainfo = parse_pose_metainfo(
            dict(from_file=f"{root}/pose/configs/_base_/keypoints308.py")
        )
        pose.cfg.codec.pop("type", None)
        pose.codec = UDPHeatmap(**pose.cfg.codec)
        self.models["pose"] = pose

        names = [pose.pose_metainfo["keypoint_id2name"][i]
                 for i in range(pose.pose_metainfo["num_keypoints"])]
        self.body_idx = similarity.body_keypoint_indices(names)
        flip_all = list(pose.pose_metainfo["flip_indices"])
        pos = {k: i for i, k in enumerate(self.body_idx)}
        self.body_flip = [pos.get(flip_all[k], i) for i, k in enumerate(self.body_idx)]

        det_dir = f"{CKPT_DIR}/detector/detr-resnet-101-dc5"
        self.det_proc = DetrImageProcessor.from_pretrained(det_dir)
        self.det = DetrForObjectDetection.from_pretrained(det_dir).eval().to(self.device)
        from transformers import AutoImageProcessor, AutoModel, pipeline

        self.scene_proc = AutoImageProcessor.from_pretrained(SCENE_MODEL)
        self.scene_model = AutoModel.from_pretrained(SCENE_MODEL).eval().to(self.device)
        self.asr = pipeline("automatic-speech-recognition", model=SPEECH_MODEL, device=self.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"Models loaded in {time.time() - t0:.0f}s")

    # ----- inference -----

    def _dense(self, task, images):
        """Run one dense model on a list of BGR frames -> list of full-res outputs."""
        import torch
        import torch.nn.functional as F

        m = self.models[task]
        outs = []
        for s in range(0, len(images), BATCH):
            chunk = images[s : s + BATCH]
            datas = [m.data_preprocessor(m.pipeline(dict(img=im))) for im in chunk]
            inputs = torch.cat([d["inputs"] for d in datas], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(inputs)
            if task == "pointmap":
                pm, scale = pred
                pred = pm.float() / scale.float().view(-1, 1, 1, 1)
            pred = pred.float()
            for k, im in enumerate(chunk):
                p = pred[k : k + 1]
                meta = datas[k].get("data_samples", {}).get("meta", {}) if isinstance(
                    datas[k].get("data_samples"), dict) else {}
                if task in ("normal", "pointmap") and "padding_size" in meta:
                    l, r, t, b = meta["padding_size"]
                    p = p[:, :, t : inputs.shape[2] - b, l : inputs.shape[3] - r]
                p = F.interpolate(p, size=im.shape[:2], mode="bilinear", align_corners=False)
                p = p[0]
                if task == "seg":
                    outs.append(p.argmax(0).to(torch.uint8).cpu().numpy())
                elif task == "normal":
                    p = p / p.norm(dim=0, keepdim=True).clamp(min=1e-8)
                    outs.append(p.permute(1, 2, 0).cpu().numpy())
                elif task == "pointmap":
                    outs.append(p[2].cpu().numpy())  # depth (z)
                else:  # matting -> [fg_r, fg_g, fg_b, alpha]
                    outs.append(p.clamp(0, 1).cpu().numpy())
        return outs

    def _detect(self, images):
        """Largest person in each frame -> list of (box, confidence).
        confidence is 0 when the detector finds nobody."""
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        out = []
        for s in range(0, len(images), BATCH):
            chunk = images[s : s + BATCH]
            rgb = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in chunk]
            inp = self.det_proc(images=rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                pred = self.det(**inp)
            sizes = torch.tensor([im.shape[:2] for im in chunk], device=self.device)
            res = self.det_proc.post_process_object_detection(pred, target_sizes=sizes, threshold=0.3)
            for im, r in zip(chunk, res):
                person = r["labels"] == 1
                boxes = r["boxes"][person].cpu().numpy()
                scores = r["scores"][person].cpu().numpy()
                if len(boxes) == 0:
                    h, w = im.shape[:2]
                    out.append((np.array([0, 0, w - 1, h - 1], np.float32), 0.0))
                    continue
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                k = int(areas.argmax())  # largest person = main subject
                out.append((boxes[k], float(scores[k])))
        return out

    def _pose(self, images, boxes):
        import numpy as np
        import torch

        m = self.models["pose"]
        kpts, scores = [], []
        for s in range(0, len(images), BATCH):
            chunk = images[s : s + BATCH]
            datas = []
            for k, im in enumerate(chunk):
                box = boxes[s + k]
                d = m.pipeline(dict(img=im, bbox=box[None], bbox_score=np.ones(1, np.float32)))
                datas.append(m.data_preprocessor(d))
            inputs = torch.cat([d["inputs"] for d in datas], 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                heat = m(inputs).float()
            heat = heat.cpu().numpy()
            for k, d in enumerate(datas):
                kp, sc = m.codec.decode(heat[k])
                meta = d["data_samples"]["meta"]
                kp = kp / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
                kpts.append(kp[0])
                scores.append(np.asarray(sc[0]).reshape(-1))
        return kpts, scores

    def _embed(self, images_bgr, size_hw):
        """DINOv2 on whole frames (no cropping; aspect kept) ->
        cls (T, D) and unit-length patch features (T, h, w, D) as float16."""
        import cv2
        import numpy as np
        import torch

        H, W = size_hw
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        cls_out, patch_out = [], []
        for s in range(0, len(images_bgr), 16):
            batch = []
            for im in images_bgr[s : s + 16]:
                rgb = cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                batch.append(((rgb.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(batch)).to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.scene_model(pixel_values=x)
            cls_out.append(o.pooler_output.float().cpu().numpy())
            p = o.last_hidden_state[:, 1:].float()
            p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            patch_out.append(p.reshape(len(batch), H // 14, W // 14, -1).half().cpu().numpy())
        return np.concatenate(cls_out, 0), np.concatenate(patch_out, 0)

    def _scene_embeddings(self, imgs, mattes, size_hw, flipped=False):
        import numpy as np

        import content

        if flipped:
            imgs = [im[:, ::-1].copy() for im in imgs]
            mattes = [None if m is None else m[:, ::-1] for m in mattes]
        cls, patch = self._embed(imgs, size_hw)
        grid = patch.shape[1:3]
        mask = np.stack([content.patch_person_mask(m, grid) for m in mattes])
        return {"cls": cls, "patch": patch.astype(np.float32), "mask": mask}

    def _transcribe(self, video_path, work, label):
        """Whisper transcript of the video's audio, or "" if there is no speech."""
        import numpy as np
        from scipy.io import wavfile

        wav = os.path.join(work, f"{label}.wav")
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
             "-t", str(MAX_SECONDS + 1), wav],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not os.path.exists(wav):
            return ""  # no audio track
        sr, audio = wavfile.read(wav)
        audio = audio.astype(np.float32) / 32768.0
        if len(audio) < sr or float(np.sqrt(np.mean(audio**2))) < 0.003:
            return ""  # under a second, or silent
        out = self.asr(
            {"raw": audio, "sampling_rate": sr}, return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )
        return (out.get("text") or "").strip()

    def _score_all(self, path_a, path_b, work, progress, calib, early_stop=True):
        """Run the three tests from cheapest to most expensive:
            1. Script (Whisper on the audio, a few seconds)
            2. Scene & visuals (DINOv2 + the matting model, ~20% of the body cost)
            3. Body & motion (the remaining four Sapiens2 models, the bulk of the GPU time)
        If a test already scores EARLY_STOP or more, the remaining ones are skipped
        (the final score is the highest test, so it would be a match either way)."""
        import cv2
        import numpy as np

        import content
        import similarity

        res = {"videos": {}, "body": None, "scene": None, "script": None, "skipped": [],
               "no_person": None, "person_frames": None}
        raw = {"body": None, "scene": None, "script": None}

        def done():
            res["raw"] = raw
            res["combined"] = content.combine(raw, calib)
            return res

        def strong():
            c = content.combine(raw, calib)
            return early_stop and c["final"] is not None and c["final"] >= EARLY_STOP

        for side, p in (("a", path_a), ("b", path_b)):
            frames = extract_frames(p, os.path.join(work, f"frames_{side}"))
            if len(frames) < 2:
                raise ValueError(f"Could not read frames from video {side.upper()}.")
            res["videos"][side] = {"imgs": [cv2.imread(f) for f in frames], "out": {}}
        va, vb = res["videos"]["a"], res["videos"]["b"]

        # 1. Script
        for side, label, p in (("a", "Video A", path_a), ("b", "Video B", path_b)):
            progress(f"{label}: Transcribing speech")
            res["videos"][side]["text"] = self._transcribe(p, work, side)
        res["script"] = content.compare_script(va["text"], vb["text"])
        raw["script"] = res["script"]["raw"]
        if strong():
            res["skipped"] = ["scene", "body"]
            return done()

        # 2. Scene & visuals. Only frames with a person are compared (by both
        #    this test and the body test); title cards, black frames and empty
        #    shots would otherwise match each other.
        # one patch grid for both videos (multiples of 14 px, long side 392), from A's shape
        h0, w0 = va["imgs"][0].shape[:2]
        long_px, short_px = 392, max(112, int(round(392 * min(h0, w0) / max(h0, w0) / 14)) * 14)
        size_hw = (long_px, short_px) if h0 >= w0 else (short_px, long_px)
        for side, label in (("a", "Video A"), ("b", "Video B")):
            v = res["videos"][side]
            progress(f"{label}: Scene & visuals")
            v["out"]["matting"] = self._dense("matting", v["imgs"])
            v["out"]["det"] = self._detect(v["imgs"])
            mattes = [m[3] for m in v["out"]["matting"]]
            v["out"]["person"] = [similarity.person_present(d[1], al)
                                  for d, al in zip(v["out"]["det"], mattes)]
            v["idx"] = np.array([i for i, p in enumerate(v["out"]["person"]) if p], int)
            if len(v["idx"]) >= 2:
                imgs_p = [v["imgs"][i] for i in v["idx"]]
                mattes_p = [mattes[i] for i in v["idx"]]
                v["emb"] = self._scene_embeddings(imgs_p, mattes_p, size_hw)
                v["emb_flip"] = (self._scene_embeddings(imgs_p, mattes_p, size_hw, flipped=True)
                                 if side == "b" else None)
        res["person_frames"] = {"a": int(len(va["idx"])), "b": int(len(vb["idx"])),
                                "a_total": len(va["imgs"]), "b_total": len(vb["imgs"])}
        missing = [s.upper() for s in ("a", "b") if len(res["videos"][s]["idx"]) < 2]
        if missing:
            res["no_person"] = "No person detected in video " + " or ".join(missing)
            return done()
        sc = content.compare_scene(va["emb"], vb["emb"], vb["emb_flip"])
        ia, ib = va["idx"], vb["idx"]  # back to original frame numbers
        sc["path"] = [(int(ia[a]), int(ib[b])) for a, b in sc["path"]]
        for x in sc["sections"]:
            x["a_start"], x["a_end"] = int(ia[x["a_start"]]), int(ia[x["a_end"]])
            x["b_start"], x["b_end"] = int(ib[x["b_start"]]), int(ib[x["b_end"]])
        res["scene"] = sc
        raw["scene"] = sc["raw"]
        if strong():
            res["skipped"] = ["body"]
            return done()

        # 3. Body & motion
        for side, label in (("a", "Video A"), ("b", "Video B")):
            v = res["videos"][side]
            v["fps"] = self._analyse_video(v["imgs"], v["out"], progress, label)
        progress("Comparing body, scene and script")
        K = len(self.body_idx)
        A, B = similarity.stack(va["fps"], K), similarity.stack(vb["fps"], K)
        if len(A["index"]) >= 2 and len(B["index"]) >= 2:
            res["body"] = similarity.compare(
                A, B, np.array(self.body_flip), vb["imgs"][0].shape[1], calib=calib)
            raw["body"] = res["body"].overall_raw
        return done()

    def _analyse_video(self, imgs, out, progress, label):
        """Body & motion models for one video. `out` already holds the matting
        result from the scene step; adds the other four and returns fingerprints."""
        import render
        import similarity

        for task in ("seg", "normal", "pointmap"):
            progress(f"{label}: {render.LABELS[task]}")
            out[task] = self._dense(task, imgs)
        progress(f"{label}: {render.LABELS['pose']}")
        out["pose_kpts"], out["pose_scores"] = self._pose(imgs, [d[0] for d in out["det"]])

        fps = []
        for i in range(len(imgs)):
            fps.append(similarity.make_fingerprint(
                out["pose_kpts"][i][self.body_idx], out["pose_scores"][i][self.body_idx],
                out["seg"][i], out["normal"][i], out["pointmap"][i], out["matting"][i][3],
                person=out["person"][i],
            ))
        return fps

    @modal.method()
    def run(self, comparison_id: str):
        import numpy as np

        import render
        import similarity

        sb = supabase()
        table = sb.table("comparisons")
        t_start = time.time()
        steps = {"n": 0}
        total_steps = 16

        def progress(stage):
            steps["n"] += 1
            table.update({
                "status": "processing", "stage": stage,
                "progress": min(95, int(100 * steps["n"] / total_steps)),
            }).eq("id", comparison_id).execute()

        try:
            row = table.select("*").eq("id", comparison_id).single().execute().data
            work = tempfile.mkdtemp()
            progress("Downloading videos")
            vids = {}
            for side in ("a", "b"):
                p = os.path.join(work, side + Path(row[f"video_{side}_path"]).suffix)
                with open(p, "wb") as f:
                    f.write(sb.storage.from_("videos").download(row[f"video_{side}_path"]))
                dur = ffprobe_duration(p)
                if dur > MAX_SECONDS + 1:
                    raise ValueError(f"Video {side.upper()} is {dur:.0f}s long; the limit is {MAX_SECONDS}s.")
                vids[side] = (p, dur)
            table.update({"video_a_duration": vids["a"][1], "video_b_duration": vids["b"][1]}) \
                .eq("id", comparison_id).execute()

            calib = similarity.load_calibration("/root/calibration/calibration.json")
            r = self._score_all(vids["a"][0], vids["b"][0], work, progress, calib,
                                early_stop=EARLY_STOP_ENABLED)
            body, scene, script, comb = r["body"], r["scene"], r["script"], r["combined"]
            va, vb = r["videos"]["a"], r["videos"]["b"]

            progress("Rendering visual results")
            frames_json = {}
            if body:
                frames_json = render.render_examples(
                    sb, comparison_id, body, (va["imgs"], va["out"], None),
                    (vb["imgs"], vb["out"], None), self.models["pose"].pose_metainfo,
                )
            if scene:
                frames_json["scene"] = render.render_scene_examples(
                    sb, comparison_id, scene, va["imgs"], vb["imgs"]
                )

            to_t = lambda i: round(i / SAMPLE_FPS, 1)  # noqa: E731
            alignment = {"frames_a": len(va["imgs"]), "frames_b": len(vb["imgs"]),
                         "person_frames": r["person_frames"]}
            def secs_json(sections):
                return [{"a_start_s": to_t(x["a_start"]), "a_end_s": to_t(x["a_end"] + 1),
                         "b_start_s": to_t(x["b_start"]), "b_end_s": to_t(x["b_end"] + 1),
                         "similarity": x["similarity"]} for x in sections]

            if scene:
                ss = scene["sections"]
                top = max(ss, key=lambda x: x["similarity"]) if ss else {
                    "a_start": min(p[0] for p in scene["path"]), "a_end": max(p[0] for p in scene["path"]),
                    "b_start": min(p[1] for p in scene["path"]), "b_end": max(p[1] for p in scene["path"])}
                alignment["scene"] = {
                    "timeline": scene["timeline"], "sections": secs_json(ss),
                    "a_start_s": to_t(top["a_start"]), "a_end_s": to_t(top["a_end"] + 1),
                    "b_start_s": to_t(top["b_start"]), "b_end_s": to_t(top["b_end"] + 1),
                }
            if body:
                alignment.update({
                    "a_start_s": to_t(body.coverage["a_start"]), "a_end_s": to_t(body.coverage["a_end"] + 1),
                    "b_start_s": to_t(body.coverage["b_start"]), "b_end_s": to_t(body.coverage["b_end"] + 1),
                    "timelines": body.timelines, "path": body.path,
                    "sections": secs_json(body.sections),
                })
            SKIP = f"Skipped: another test already scored {int(EARLY_STOP * 100)}%+"
            tests = {
                "driver": comb["driver"],
                "skipped": r["skipped"],
                "body": {"score": comb["scores"]["body"], "raw": r["raw"]["body"],
                         "mirrored": bool(body and body.mirrored),
                         "reason": None if body else r["no_person"] or (SKIP if "body" in r["skipped"]
                         else "No person detected in both videos")},
                "scene": {"score": comb["scores"]["scene"], "raw": r["raw"]["scene"],
                          "mirrored": bool(scene and scene["mirrored"]),
                          "reason": None if scene else r["no_person"] or SKIP},
                "script": {"score": comb["scores"]["script"], "raw": script["raw"],
                           "reason": script.get("reason"),
                           "shared_words": script.get("shared_words", 0)},
            }
            transcripts = {k: script[k] for k in
                           ("tokens_a", "tokens_b", "spans_a", "spans_b", "char_level")}
            mirrored = bool((body and body.mirrored) or (scene and scene["mirrored"]))
            table.update({
                "status": "done", "progress": 100, "stage": "Done",
                "overall_score": comb["final"], "raw_score": r["raw"].get(comb["driver"]),
                "calibrated": comb.get("calibrated", False), "verdict": comb["verdict"],
                "mirrored": mirrored,
                "model_scores": body.model_scores if body else None,
                "model_scores_raw": body.model_scores_raw if body else None,
                "tests": tests, "transcripts": transcripts,
                "alignment": alignment,
                "frames": frames_json,
                "timings": {"seconds": round(time.time() - t_start, 1)},
                "finished_at": _now(),
            }).eq("id", comparison_id).execute()
            shutil.rmtree(work, ignore_errors=True)
            return {"ok": True, "score": comb["final"]}
        except Exception as e:  # report every failure to the UI
            import traceback

            traceback.print_exc()  # full details stay in the Modal logs
            # Our own checks raise ValueError with a friendly message; anything
            # else gets a generic one so internals (model names, paths) stay private.
            msg = str(e) if isinstance(e, ValueError) else (
                "Something went wrong analysing these videos. Please try again, "
                "or try different files.")
            table.update({"status": "error", "error": msg[:500]}).eq("id", comparison_id).execute()
            return {"ok": False, "error": str(e)}

    @modal.method()
    def score_pair(self, path_a: str, path_b: str) -> dict:
        """Used by calibrate.py: compare two local videos (bytes uploaded via a
        Modal Volume) and return raw scores without touching Supabase."""
        import numpy as np

        import similarity

        work = tempfile.mkdtemp()
        # calibration needs every test's score, so never stop early here
        r = self._score_all(path_a, path_b, work, lambda s: None, None, early_stop=False)
        shutil.rmtree(work, ignore_errors=True)
        body = r["body"]
        return {
            "raw": r["raw"],  # {"body", "scene", "script"}; None where a test couldn't run
            "overall_raw": r["raw"]["body"],
            "models_raw": body.model_scores_raw if body else None,
            "mirrored": bool(body and body.mirrored) or bool(r["scene"] and r["scene"]["mirrored"]),
            "script_reason": r["script"].get("reason"),
        }


# ---------------------------------------------------------------- calibration helper

calib_volume = modal.Volume.from_name("origin-calibration-videos", create_if_missing=True)


@app.function(volumes={"/calib": calib_volume}, timeout=7200)
def score_calibration_pairs(pairs: list[dict]) -> list[dict]:
    """Score every labelled pair (videos already uploaded to the calibration volume)."""
    analyzer = Analyzer()
    calls = [(p, analyzer.score_pair.spawn(f"/calib/{p['a']}", f"/calib/{p['b']}")) for p in pairs]
    out = []
    for p, call in calls:
        try:
            out.append({**p, **call.get()})
        except Exception as e:
            out.append({**p, "error": str(e)})
    return out


# ---------------------------------------------------------------- web API

# ---------------------------------------------------------------- video links

link_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("yt-dlp", "supabase>=2.4")
    .add_local_file(here / "links.py", "/root/links.py")
)


@app.function(image=link_image, secrets=secrets, timeout=600)
def fetch_links_then_analyze(comparison_id: str):
    """Download any videos given as links (on a cheap CPU machine), save them to
    Supabase Storage like uploads, then hand over to the GPU worker."""
    import links

    sb = supabase()
    table = sb.table("comparisons")
    try:
        row = table.select("*").eq("id", comparison_id).single().execute().data
        work = tempfile.mkdtemp()
        for side in ("a", "b"):
            url = row.get(f"video_{side}_url")
            if not url or row.get(f"video_{side}_path"):
                continue
            table.update({"status": "processing", "stage": f"Video {side.upper()}: downloading from link",
                          "progress": 3}).eq("id", comparison_id).execute()
            try:
                got = links.download(url, work, side, MAX_SECONDS, MAX_BYTES)
            except ValueError as e:
                raise ValueError(f"Video {side.upper()}: {e}") from e
            path = f"{comparison_id}/{side}.mp4"
            with open(got["path"], "rb") as f:
                sb.storage.from_("videos").upload(path, f.read(), {"content-type": "video/mp4", "upsert": "true"})
            table.update({
                f"video_{side}_path": path, f"video_{side}_size": got["size"],
                f"video_{side}_name": got["title"] or url[:200],
            }).eq("id", comparison_id).execute()
        shutil.rmtree(work, ignore_errors=True)
        table.update({"stage": "Starting GPU", "progress": 5}).eq("id", comparison_id).execute()
        Analyzer().run.spawn(comparison_id)
    except Exception as e:
        import traceback

        traceback.print_exc()
        msg = str(e) if isinstance(e, ValueError) else (
            "We couldn't download the video from that link. Please upload the file instead.")
        table.update({"status": "error", "error": msg[:500]}).eq("id", comparison_id).execute()


web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("supabase>=2.4", "fastapi[standard]")
    .add_local_file(here / "links.py", "/root/links.py")
)


@app.function(image=web_image, secrets=secrets, min_containers=0)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    web = FastAPI(title="Origin video comparison API")
    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
    web.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def ip_hash(req: Request) -> str:
        ip = (req.headers.get("x-forwarded-for") or req.client.host or "").split(",")[0].strip()
        salt = os.environ.get("IP_SALT", "change-me")
        return hashlib.sha256((salt + ip).encode()).hexdigest()[:32]

    def is_admin(req: Request) -> bool:
        key = os.environ.get("ADMIN_KEY")
        return bool(key) and req.headers.get("x-admin-key") == key

    class FileInfo(BaseModel):  # one video: either an uploaded file or a link
        name: str = ""
        size: int = 0
        type: str = ""
        url: str | None = None

    class CreateBody(BaseModel):
        a: FileInfo
        b: FileInfo
        consent: bool

    ALLOWED_EXT = {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}

    @web.post("/create")
    def create(body: CreateBody, req: Request):
        import links

        if not body.consent:
            raise HTTPException(400, "Please confirm you have the right to upload these videos.")
        for side, f in (("A", body.a), ("B", body.b)):
            if f.url:
                try:
                    f.url = links.validate_url(f.url)
                except ValueError as e:
                    raise HTTPException(400, f"Video {side}: {e}")
                continue
            if f.size > MAX_BYTES:
                raise HTTPException(400, f"Video {side} is over {MAX_BYTES // 2**20} MB.")
            if Path(f.name).suffix.lower() not in ALLOWED_EXT:
                raise HTTPException(400, f"Video {side} must be MP4, MOV, WebM, M4V, AVI or MKV.")
        sb = supabase()
        h = ip_hash(req)
        if DAILY_LIMIT > 0 and not is_admin(req):
            since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
            used = sb.table("comparisons").select("id", count="exact").eq("ip_hash", h) \
                .neq("status", "awaiting_upload").gte("created_at", since).execute().count or 0
            if used >= DAILY_LIMIT:
                raise HTTPException(429, f"You've used your {DAILY_LIMIT} free comparisons for today.")

        cid = str(uuid.uuid4())
        sides = {"a": body.a, "b": body.b}
        paths = {s: (None if f.url else f"{cid}/{s}{Path(f.name).suffix.lower()}") for s, f in sides.items()}
        row = {"id": cid, "status": "awaiting_upload", "ip_hash": h,
               "user_agent": (req.headers.get("user-agent") or "")[:300], "consent_at": _now()}
        for s, f in sides.items():
            row.update({f"video_{s}_path": paths[s], f"video_{s}_url": f.url,
                        f"video_{s}_name": (f.url or f.name)[:200], f"video_{s}_size": f.size or None})
        sb.table("comparisons").insert(row).execute()
        uploads = {}
        for s, p in paths.items():
            if p is None:
                continue  # fetched from its link later
            r = sb.storage.from_("videos").create_signed_upload_url(p)
            uploads[s] = {"path": p, "token": r.get("token"),
                          "signed_url": r.get("signed_url") or r.get("signedUrl") or r.get("signedURL")}
        return {"id": cid, "uploads": uploads}

    class AnalyzeBody(BaseModel):
        id: str

    @web.post("/analyze")
    def analyze(body: AnalyzeBody):
        sb = supabase()
        try:
            row = sb.table("comparisons").select("status,video_a_path,video_b_path,video_a_url,video_b_url") \
                .eq("id", body.id).single().execute().data
        except Exception:
            raise HTTPException(404, "Comparison not found.")
        if row["status"] != "awaiting_upload":
            return {"ok": True, "status": row["status"]}
        uploaded = [p for p in (row["video_a_path"], row["video_b_path"]) if p]
        if uploaded:
            listed = {o["name"] for o in sb.storage.from_("videos").list(body.id)}
            for p in uploaded:
                if Path(p).name not in listed:
                    raise HTTPException(400, "Upload didn't finish. Please try again.")
        has_links = bool(row.get("video_a_url") or row.get("video_b_url"))
        sb.table("comparisons").update({
            "status": "queued", "progress": 2,
            "stage": "Fetching video from link" if has_links else "Starting GPU",
        }).eq("id", body.id).execute()
        if has_links:
            fetch_links_then_analyze.spawn(body.id)
        else:
            Analyzer().run.spawn(body.id)
        return {"ok": True, "status": "queued"}

    PUBLIC_FIELDS = (
        "id,created_at,status,stage,progress,error,overall_score,raw_score,calibrated,verdict,"
        "mirrored,model_scores,tests,transcripts,alignment,frames,video_a_name,video_b_name,video_a_duration,"
        "video_b_duration,timings"
    )

    @web.get("/status/{cid}")
    def status(cid: str):
        sb = supabase()
        try:
            row = sb.table("comparisons").select(PUBLIC_FIELDS).eq("id", cid).single().execute().data
        except Exception:
            raise HTTPException(404, "Comparison not found.")
        if row.get("frames"):
            for model, pairs in row["frames"].items():
                for p in pairs:
                    p["url"] = _signed_url(sb, "results", p["path"])
        if row.get("alignment"):
            row["alignment"].pop("path", None)
        return row

    @web.get("/admin/comparisons")
    def admin_list(req: Request, limit: int = 100):
        if not is_admin(req):
            raise HTTPException(401, "Wrong admin key.")
        sb = supabase()
        rows = sb.table("comparisons").select(
            "id,created_at,status,error,overall_score,raw_score,verdict,mirrored,model_scores,tests,video_a_url,video_b_url,"
            "video_a_name,video_b_name,video_a_path,video_b_path,video_a_size,video_b_size,"
            "video_a_duration,video_b_duration,ip_hash,user_agent,timings"
        ).order("created_at", desc=True).limit(min(limit, 500)).execute().data
        for r in rows:
            r["video_a_url"] = _signed_url(sb, "videos", r["video_a_path"])
            r["video_b_url"] = _signed_url(sb, "videos", r["video_b_path"])
        return rows

    @web.get("/health")
    def health():
        return {"ok": True, "daily_limit": DAILY_LIMIT, "max_seconds": MAX_SECONDS,
                "max_mb": MAX_BYTES // 2**20}

    return web
