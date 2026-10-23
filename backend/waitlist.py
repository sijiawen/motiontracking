"""
Creator waitlist: creators who want their whole content library indexed and
watched for copies (ads and reposts), plus a hand-checked scan for the first
creators. Validation lives here so it can be tested without Modal.
"""

import re

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[A-Za-z]{2,}$")
CHANNELS = ("tiktok", "instagram", "youtube", "x", "other")
LIBRARY_SIZES = ("under_50", "50_200", "200_1000", "over_1000")
SEEN_COPIES = ("yes", "not_sure", "no")
MAX_HANDLE = 200
MAX_NOTES = 1500


def _clean(text, limit):
    text = (text or "").strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:limit]


def validate(data: dict) -> dict:
    """Return a clean row for the waitlist table, or raise ValueError with a
    message that can be shown to the person filling in the form."""
    email = _clean(data.get("email"), 320).lower()
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address.")
    channels = {}
    raw = data.get("channels") or {}
    if not isinstance(raw, dict):
        raw = {}
    for key in CHANNELS:
        val = _clean(raw.get(key), MAX_HANDLE)
        if val:
            channels[key] = val
    if not channels:
        raise ValueError("Please add at least one channel, like your TikTok or YouTube.")
    size = data.get("library_size") or None
    if size is not None and size not in LIBRARY_SIZES:
        size = None
    seen = data.get("seen_copies") or None
    if seen is not None and seen not in SEEN_COPIES:
        seen = None
    return {
        "email": email,
        "name": _clean(data.get("name"), 120) or None,
        "channels": channels,
        "library_size": size,
        "seen_copies": seen,
        "notes": _clean(data.get("notes"), MAX_NOTES) or None,
        "manual_check": bool(data.get("manual_check")),
        "source": _clean(data.get("source"), 200) or None,
    }
