"""
Transcribe the spoken audio of ad videos (e.g. collected from the Meta Ad
Library) so they can be compared with a creator's original transcripts.

    python3 -m modal run backend/transcribe_ads.py --ads ads/kalshi_ads.json

The JSON file lists the ads (library_id, start, ad_copy, video_url, ...).
Writes <name>_transcripts.csv next to it, one row per ad.

Transcripts are machine transcriptions (Whisper large-v3-turbo) of the actual
audio, kept separate from the ad copy. Words the model was unsure about are
marked like [word?]. A text match is only a lead: it is not proof of misuse.
"""

import base64
import csv
import json
import re
from pathlib import Path

import modal

app = modal.App("creatorgen-ad-transcripts")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("openai-whisper==20240930", "requests", "yt-dlp")
)
cache = modal.Volume.from_name("creatorgen-whisper-cache", create_if_missing=True)

UNCLEAR_PROB = 0.5  # words below this confidence are marked [word?]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
# Nudges Whisper to keep filler words ("um", "like") instead of tidying them away.
VERBATIM_PROMPT = "Umm, so, like, I was... you know, uh, thinking. Okay."


def _download(ad: dict, workdir: str) -> tuple[str | None, str]:
    """Video file path and how it was fetched (or None and the reason)."""
    import os
    import subprocess

    import requests

    dest = os.path.join(workdir, f"{ad['library_id']}.mp4")
    url = ad.get("video_url")
    errors = []
    if url:
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://www.facebook.com/"},
                             timeout=90)
            if r.ok and len(r.content) > 10_000:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return dest, "video URL from the results card"
            errors.append(f"video URL returned HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"video URL failed: {e}")
    # fallback: the public Ad Library page for this ad
    page = f"https://www.facebook.com/ads/library/?id={ad['library_id']}"
    try:
        subprocess.run(["yt-dlp", "-q", "--no-warnings", "--playlist-items", "1",
                        "-f", "best[ext=mp4]/best", "-o", dest, page],
                       check=True, timeout=180, capture_output=True)
        if os.path.exists(dest):
            return dest, "Ad Library page (yt-dlp fallback)"
    except Exception as e:  # noqa: BLE001
        errors.append(f"Ad Library page fallback failed: {getattr(e, 'stderr', b'') or e}")
    return None, "; ".join(str(x)[:300] for x in errors)


def _probe(path: str) -> tuple[float | None, bool]:
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                          "-of", "json", path], capture_output=True, text=True).stdout
    info = json.loads(out or "{}")
    dur = info.get("format", {}).get("duration")
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return (round(float(dur), 1) if dur else None), has_audio


@app.function(image=image, gpu="L4", timeout=1800, volumes={"/cache": cache})
def transcribe(ads: list[dict]) -> list[dict]:
    import tempfile

    import whisper

    model = whisper.load_model("turbo", download_root="/cache")
    cache.commit()
    work = tempfile.mkdtemp()
    results = []
    for ad in ads:
        r = {"library_id": ad["library_id"], "status": "", "fetched_from": "", "measured_duration_s": None,
             "language": "", "speech_detected": "", "transcript": "", "unclear_words": "",
             "raw_whisper_text": ""}
        path, how = _download(ad, work)
        if not path:
            r.update(status="INACCESSIBLE", fetched_from=how)
            results.append(r)
            continue
        r["fetched_from"] = how
        dur, has_audio = _probe(path)
        r["measured_duration_s"] = dur
        if not has_audio:
            r.update(status="NO AUDIO TRACK", speech_detected="no")
            results.append(r)
            continue
        res = model.transcribe(path, word_timestamps=True, condition_on_previous_text=False,
                               initial_prompt=VERBATIM_PROMPT, fp16=True)
        r["language"] = res.get("language", "")
        r["raw_whisper_text"] = (res.get("text") or "").strip()
        speech_segs = [s for s in res.get("segments", [])
                       if s.get("no_speech_prob", 1) < 0.5 and s.get("avg_logprob", -9) > -1.0]
        if not speech_segs:
            r.update(status="NO SPEECH DETECTED", speech_detected="no")
            results.append(r)
            continue
        parts, unclear = [], []
        for s in speech_segs:
            for w in s.get("words", []):
                word = w["word"]
                if w.get("probability", 1) < UNCLEAR_PROB and word.strip():
                    lead = word[: len(word) - len(word.lstrip())]
                    parts.append(f"{lead}[{word.strip()}?]")
                    unclear.append(f"{word.strip()} @{w['start']:.1f}s ({w['probability']:.2f})")
                else:
                    parts.append(word)
        dropped = len(res.get("segments", [])) - len(speech_segs)
        r.update(status="OK" + (f" ({dropped} low-confidence segment(s) left out)" if dropped else ""),
                 speech_detected="yes", transcript=re.sub(r"\s+", " ", "".join(parts)).strip(),
                 unclear_words="; ".join(unclear))
        results.append(r)
    return results


def _asset_id(url: str | None) -> str:
    """Meta's video asset ID, from the efg parameter of the video URL."""
    if not url:
        return ""
    m = re.search(r"[?&]efg=([^&]+)", url)
    if not m:
        return ""
    from urllib.parse import unquote

    raw = unquote(m.group(1))
    try:
        data = json.loads(base64.b64decode(raw + "=" * (-len(raw) % 4)))
        return str(data.get("xpv_asset_id", ""))
    except Exception:  # noqa: BLE001
        return ""


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\[([^\]]*)\?\]", r"\1", t.lower())).strip()


@app.local_entrypoint()
def main(ads: str):
    src = Path(ads).expanduser()
    data = json.loads(src.read_text())
    rows = data["ads"]
    results = {r["library_id"]: r for r in transcribe.remote(rows)}

    # duplicates: same underlying video file, or identical speech
    by_asset, by_text = {}, {}
    for a in rows:
        a["asset_id"] = _asset_id(a.get("video_url"))
        t = _norm(results[a["library_id"]]["transcript"])
        if a["asset_id"]:
            by_asset.setdefault(a["asset_id"], []).append(a["rank"])
        if t:
            by_text.setdefault(t, []).append(a["rank"])

    out = src.with_name(src.stem + "_transcripts.csv")
    cols = ["rank", "library_id", "detail_url", "start_date", "video_duration_s", "page_shown_as",
            "ad_copy", "link_caption_cta", "multiple_versions", "version_transcribed",
            "same_creative_and_text_count", "video_asset_id", "duplicate_flags", "speech_detected",
            "transcript", "unclear_words", "language", "status", "fetched_from", "notes"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in rows:
            r = results[a["library_id"]]
            flags = []
            same_asset = [x for x in by_asset.get(a["asset_id"], []) if x != a["rank"]]
            if same_asset:
                flags.append("same video file as rank " + ", ".join(map(str, same_asset)))
            same_text = [x for x in by_text.get(_norm(r["transcript"]), []) if x != a["rank"]] if r["transcript"] else []
            if same_text:
                flags.append("identical transcript to rank " + ", ".join(map(str, same_text)))
            if a.get("same_creative_count"):
                flags.append(f"Meta groups {a['same_creative_count']} ads with this creative and text")
            notes = ["Machine transcription (Whisper large-v3-turbo) of the audio; spot-check before relying on it"]
            if r["status"].startswith("NO SPEECH") and r["raw_whisper_text"]:
                notes.append(f"model output discarded as likely non-speech: {r['raw_whisper_text'][:200]}")
            dur = r["measured_duration_s"] or a.get("video_duration_s")
            w.writerow({
                "rank": a["rank"], "library_id": a["library_id"],
                "detail_url": f"https://www.facebook.com/ads/library/?id={a['library_id']}",
                "start_date": a.get("start", ""), "video_duration_s": dur,
                "page_shown_as": a.get("page_line", ""), "ad_copy": a.get("ad_copy", ""),
                "link_caption_cta": a.get("link_caption_cta", ""),
                "multiple_versions": "yes" if a.get("multiple_versions") else "no",
                "version_transcribed": a.get("versions_note", "only version (as shown on the results card)"),
                "same_creative_and_text_count": a.get("same_creative_count", ""),
                "video_asset_id": a["asset_id"], "duplicate_flags": "; ".join(flags),
                "speech_detected": r["speech_detected"], "transcript": r["transcript"],
                "unclear_words": r["unclear_words"], "language": r["language"], "status": r["status"],
                "fetched_from": r["fetched_from"], "notes": " | ".join(notes),
            })
    for a in rows:
        r = results[a["library_id"]]
        print(f"{a['rank']:>2}. {a['library_id']}  {r['status']:<22} {r['transcript'][:70]}")
    print("Saved", out)
