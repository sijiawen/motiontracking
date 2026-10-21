import pytest

from links import _friendly, validate_url


@pytest.mark.parametrize("url", [
    "https://www.tiktok.com/@someone/video/7289941015157427462",
    "instagram.com/reel/Cx1abc/",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://x.com/user/status/123",
])
def test_accepts_public_links(url):
    assert validate_url(url).startswith("http")


@pytest.mark.parametrize("url", [
    "", "not a link", "ftp://example.com/v.mp4", "http://localhost:8000/v.mp4",
    "http://127.0.0.1/v.mp4", "http://192.168.1.5/v.mp4", "http://10.0.0.1/x", "http://169.254.169.254/latest",
    "file:///etc/passwd",
])
def test_rejects_bad_or_private_links(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_friendly_errors_never_leak_raw_text():
    raw = "ERROR: [tiktok] 123: Sign in to confirm you're not a bot. Use --cookies-from-browser"
    msg = _friendly(raw)
    assert "upload the file" in msg and "cookies" not in msg
