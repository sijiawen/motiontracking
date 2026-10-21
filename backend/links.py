"""
Fetch a video from a link (TikTok, Instagram, YouTube, X, and most public
video pages) with yt-dlp, so people can paste a URL instead of uploading.
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse

MAX_URL_LENGTH = 2000


def validate_url(url: str) -> str:
    """Return a cleaned URL, or raise ValueError with a friendly message.
    Only public http(s) addresses are allowed (no localhost / private networks)."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Please paste a link.")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError("That link is too long.")
    if "://" not in url:
        url = "https://" + url
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname or "." not in p.hostname:
        raise ValueError("That doesn't look like a web link.")
    host = p.hostname.lower()
    if host == "localhost" or host.endswith(".local") or host.endswith(".internal"):
        raise ValueError("That link isn't a public web address.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
        raise ValueError("That link isn't a public web address.")
    return url


def _resolves_publicly(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def download(url: str, out_dir: str, name: str, max_seconds: int, max_bytes: int) -> dict:
    """Download the video at `url` to out_dir/name.mp4 (720p or smaller).
    Returns {"path", "title", "duration", "size"}; raises ValueError with a
    message that can be shown to the user."""
    import yt_dlp

    url = validate_url(url)
    if not _resolves_publicly(urlparse(url).hostname):
        raise ValueError("That link isn't a public web address.")
    os.makedirs(out_dir, exist_ok=True)
    base = {
        "quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30,
        "retries": 2, "restrictfilenames": True,
    }
    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(_friendly(str(e))) from e
    if info.get("_type") == "playlist" or info.get("entries"):
        raise ValueError("That link is a playlist or profile. Please link to a single video.")
    if info.get("is_live"):
        raise ValueError("Live streams can't be compared. Please link to a finished video.")
    duration = info.get("duration")
    if duration and duration > max_seconds + 1:
        raise ValueError(f"That video is {int(duration)}s long; the limit is {max_seconds}s. "
                         "Trim it and upload the file instead.")

    opts = dict(base)
    opts.update({
        "format": ("bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/"
                   "bv*[height<=720]+ba/b[height<=720]/b"),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(out_dir, f"{name}.%(ext)s"),
        "max_filesize": max_bytes,
    })
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(_friendly(str(e))) from e

    files = [f for f in os.listdir(out_dir) if f.startswith(name + ".") and not f.endswith(".part")]
    if not files:
        raise ValueError(f"The video is larger than {max_bytes // 2**20} MB, or couldn't be saved. "
                         "Try uploading the file instead.")
    path = os.path.join(out_dir, sorted(files, key=lambda f: not f.endswith(".mp4"))[0])
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"The video is larger than {max_bytes // 2**20} MB. Try a shorter clip.")
    return {"path": path, "title": (info.get("title") or "")[:200], "duration": duration, "size": size}


def _friendly(err: str) -> str:
    e = err.lower()
    if "private" in e or "login" in e or "sign in" in e or "cookies" in e or "confirm you" in e:
        return ("That site wouldn't let us download the video (it may be private, need a login, "
                "or block automated downloads). Please download it and upload the file instead.")
    if "unsupported url" in e:
        return "We couldn't find a video at that link. Please check it, or upload the file instead."
    if "unavailable" in e or "removed" in e or "404" in e:
        return "That video is unavailable (it may have been deleted or made private)."
    return "We couldn't download the video from that link. Please upload the file instead."
