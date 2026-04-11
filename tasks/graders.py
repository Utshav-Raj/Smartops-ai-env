from __future__ import annotations
from typing import Any, Dict


def _cat(ticket: Dict[str, Any]) -> str | None:
    """Return predicted_category as a plain string, handling enum dicts."""
    val = ticket.get("predicted_category")
    if val is None:
        return None
    # Pydantic v2 serialises enums as their .value (a string) in model_dump()
    if isinstance(val, dict):
        return val.get("value")
    return str(val)


def grade_easy_duplicate_charge_refund(state: Dict[str, Any]) -> float:
    tickets = state.get("tickets", [])
    t = next((x for x in tickets if x.get("id") == "B-1001"), None)
    if not t:
        return 0.12  # no ticket in state — minimal non-zero, non-one score

    history = state.get("action_history", [])
    breakdown: Dict[str, float] = {}

    classified = any(
        a.get("action_type") == "classify_ticket" and a.get("ticket_id") == "B-1001"
        for a in history
    )
    breakdown["classified"]      = 0.25 if classified else 0.0
    breakdown["correct_category"] = 0.25 if _cat(t) == "billing" else 0.0
    breakdown["responded"]        = 0.25 if t.get("response_sent", False) else 0.0
    breakdown["resolved"]         = 0.25 if t.get("resolved", False) else 0.0

    # Penalise wrong escalation
    if t.get("escalated", False):
        breakdown["resolved"] = 0.0

    score = sum(breakdown.values())
    # Clamp strictly inside (0, 1) — never return boundary values
    return round(max(0.12, min(0.88, score)), 4)


def grade_medium_priority_queue_mix(state: Dict[str, Any]) -> float:
    tickets = {t["id"]: t for t in state.get("tickets", [])}
    history = state.get("action_history", [])
    breakdown: Dict[str, float] = {}

    if not tickets:
        return 0.12  # env not initialised for this task

    # All three tickets resolved
    resolved_count = sum(1 for t in tickets.values() if t.get("resolved"))
    all_resolved   = resolved_count == len(tickets)
    breakdown["all_resolved"] = (
        0.40 if all_resolved
        else round(resolved_count / max(len(tickets), 1) * 0.40, 4)
    )

    # Correct category classifications
    pairs = [
        ("D-2001", "delivery"),
        ("B-2002", "billing"),
        ("T-2003", "technical"),
    ]
    correct = sum(1 for tid, cat in pairs if _cat(tickets.get(tid, {})) == cat)
    breakdown["correct_categories"] = round(correct / len(pairs) * 0.30, 4)

    # Delivery ticket worked first (SLA priority)
    first_delivery = (
        bool(history) and history[0].get("ticket_id") == "D-2001"
    )
    breakdown["delivery_prioritised"] = 0.30 if first_delivery else 0.0

    score = sum(breakdown.values())
    return round(max(0.12, min(0.88, score)), 4)


def grade_hard_account_takeover(state: Dict[str, Any]) -> float:
    tickets = state.get("tickets", [])
    t = next((x for x in tickets if x.get("id") == "F-3001"), None)
    if not t:
        return 0.12  # env not initialised for this task

    breakdown: Dict[str, float] = {}
    breakdown["fraud_classified"]  = 0.30 if _cat(t) == "fraud" else 0.0
    breakdown["info_requested"]    = 0.25 if t.get("info_requested", False) else 0.0
    breakdown["escalated"]         = 0.30 if t.get("escalated", False) else 0.0

    # Dangerous path: resolved WITHOUT escalating a fraud ticket
    if t.get("resolved", False) and not t.get("escalated", False):
        breakdown["escalated"] = -0.20

    breakdown["responded_safely"]  = 0.15 if t.get("response_sent", False) else 0.0

    score = sum(breakdown.values())
    return round(max(0.12, min(0.88, score)), 4)
