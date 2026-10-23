import pytest

import waitlist


def test_valid_row():
    row = waitlist.validate({
        "email": " Creator@Example.com ", "name": "Ana",
        "channels": {"tiktok": "@ana", "youtube": "", "bogus": "x"},
        "library_size": "50_200", "seen_copies": "yes", "manual_check": True,
    })
    assert row["email"] == "creator@example.com"
    assert row["channels"] == {"tiktok": "@ana"}
    assert row["library_size"] == "50_200" and row["seen_copies"] == "yes"
    assert row["manual_check"] is True and row["notes"] is None


def test_needs_email_and_channel():
    with pytest.raises(ValueError):
        waitlist.validate({"email": "nope", "channels": {"tiktok": "@a"}})
    with pytest.raises(ValueError):
        waitlist.validate({"email": "a@b.co", "channels": {}})


def test_unknown_choices_dropped_and_text_capped():
    row = waitlist.validate({"email": "a@b.co", "channels": {"other": "site.com"},
                             "library_size": "huge", "seen_copies": "maybe", "notes": "x" * 5000})
    assert row["library_size"] is None and row["seen_copies"] is None
    assert len(row["notes"]) == waitlist.MAX_NOTES
