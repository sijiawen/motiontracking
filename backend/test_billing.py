import time

from billing import PLANS, checkout_params, effective_plan, subscription_fields, usage_window


def iso(offset):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset))


def test_free_by_default_and_when_lapsed():
    assert effective_plan(None) == "free"
    assert effective_plan({"plan": "creator", "status": "canceled"}) == "free"
    assert effective_plan({"plan": "creator", "status": "active", "period_end": iso(86400)}) == "creator"
    # renewal webhook late: 3-day grace, then free
    assert effective_plan({"plan": "pro", "status": "active", "period_end": iso(-86400)}) == "pro"
    assert effective_plan({"plan": "pro", "status": "active", "period_end": iso(-5 * 86400)}) == "free"


def test_usage_windows():
    since, resets = usage_window(None)
    assert resets is None and since < iso(0)
    since, resets = usage_window({"plan": "creator", "status": "active",
                                  "period_start": "2026-10-01T00:00:00Z", "period_end": iso(86400)})
    assert since == "2026-10-01T00:00:00Z"


def test_subscription_fields_new_api_shape():
    sub = {"id": "sub_1", "customer": "cus_1", "status": "active", "metadata": {"plan": "pro", "user_id": "u1"},
           "items": {"data": [{"current_period_start": 1790000000, "current_period_end": 1792592000,
                               "price": {"recurring": {"interval": "year"}}}]}}
    f = subscription_fields(sub)
    assert f["plan"] == "pro" and f["billing_interval"] == "year" and f["period_end"].startswith("2026")
    assert subscription_fields({**sub, "status": "canceled"})["plan"] == "free"


def test_checkout_params():
    p = checkout_params("creator", "year", "u1", "a@b.com", None, "https://x.app/")
    item = p["line_items"][0]["price_data"]
    assert item["unit_amount"] == PLANS["creator"]["price"]["year"] == 22800
    assert p["mode"] == "subscription" and p["customer_email"] == "a@b.com"
    assert p["subscription_data"]["metadata"] == {"user_id": "u1", "plan": "creator"}
