"""
Plans, usage limits and Stripe subscriptions.

Everyone signs in (Google, via Supabase Auth) before running a comparison.
  free     3 comparisons per day
  creator  30 comparisons per billing month   ($24/mo, or $228/yr = $19/mo)
  pro      150 comparisons per billing month  ($79/mo, or $708/yr = $59/mo)
Prices live here (Stripe Checkout is created with inline prices), so there's
nothing to set up in the Stripe dashboard except the webhook.
"""

import os
import time
from datetime import datetime, timezone

PLANS = {
    "free": {"name": "Free", "limit": 3, "window": "day"},
    "creator": {"name": "Creator", "limit": 30, "window": "period",
                "price": {"month": 2400, "year": 22800}},
    "pro": {"name": "Pro", "limit": 150, "window": "period",
            "price": {"month": 7900, "year": 70800}},
}
PAID_STATUSES = {"active", "trialing", "past_due"}  # past_due: Stripe is still retrying the card


def now_iso(offset_s: float = 0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_s))


def effective_plan(profile: dict | None) -> str:
    """The plan a user can use right now (a lapsed subscription falls back to free)."""
    if not profile:
        return "free"
    plan = profile.get("plan") or "free"
    if plan != "free" and profile.get("status") not in PAID_STATUSES:
        return "free"
    if plan != "free" and profile.get("period_end"):
        end = datetime.fromisoformat(str(profile["period_end"]).replace("Z", "+00:00"))
        if end < datetime.now(timezone.utc):
            # renewal webhook hasn't arrived; allow a 3-day grace period
            if (datetime.now(timezone.utc) - end).total_seconds() > 3 * 86400:
                return "free"
    return plan if plan in PLANS else "free"


def usage_window(profile: dict | None) -> tuple[str, str | None]:
    """(count usage since, resets at) for this user's plan."""
    plan = effective_plan(profile)
    if PLANS[plan]["window"] == "day":
        return now_iso(-86400), None
    start = (profile or {}).get("period_start") or now_iso(-30 * 86400)
    return str(start), (profile or {}).get("period_end")


def _ts_iso(ts) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def subscription_fields(sub: dict) -> dict:
    """Profile fields from a Stripe subscription object (works with old and new
    Stripe API versions, which moved the period dates onto the items)."""
    item = ((sub.get("items") or {}).get("data") or [{}])[0]
    start = sub.get("current_period_start") or item.get("current_period_start")
    end = sub.get("current_period_end") or item.get("current_period_end")
    meta = sub.get("metadata") or {}
    plan = meta.get("plan") if meta.get("plan") in PLANS else None
    status = sub.get("status")
    fields = {
        "stripe_subscription_id": sub.get("id"),
        "stripe_customer_id": sub.get("customer"),
        "status": status,
        "period_start": _ts_iso(start),
        "period_end": _ts_iso(end),
        "billing_interval": ((item.get("price") or {}).get("recurring") or {}).get("interval"),
    }
    if status in ("canceled", "incomplete_expired", "unpaid"):
        fields["plan"] = "free"
    elif plan:
        fields["plan"] = plan
    return fields


def checkout_params(plan: str, interval: str, user_id: str, email: str | None,
                    customer_id: str | None, site_url: str) -> dict:
    """Arguments for stripe.checkout.Session.create."""
    if plan not in PLANS or "price" not in PLANS[plan]:
        raise ValueError("Unknown plan.")
    if interval not in ("month", "year"):
        raise ValueError("Unknown billing interval.")
    p = PLANS[plan]
    params = {
        "mode": "subscription",
        "line_items": [{
            "quantity": 1,
            "price_data": {
                "currency": "usd",
                "unit_amount": p["price"][interval],
                "recurring": {"interval": interval},
                "product_data": {"name": f"Creatorgen {p['name']}",
                                 "description": f"{p['limit']} video comparisons per month"},
            },
        }],
        "client_reference_id": user_id,
        "subscription_data": {"metadata": {"user_id": user_id, "plan": plan}},
        "metadata": {"user_id": user_id, "plan": plan},
        "allow_promotion_codes": True,
        "success_url": f"{site_url}?checkout=success",
        "cancel_url": f"{site_url}?checkout=cancelled#pricing",
    }
    if customer_id:
        params["customer"] = customer_id
    elif email:
        params["customer_email"] = email
    return params


def site_url(default: str = "") -> str:
    return (os.environ.get("SITE_URL") or default).rstrip("/") + "/"
