from __future__ import annotations
from typing import Any, Dict
from ..models.openenv import TaskGrade

def grade_easy_duplicate_charge_refund(state: Dict[str, Any]) -> TaskGrade:
    tickets = state.get("tickets", [])
    t = next((x for x in tickets if x["id"] == "B-1001"), None)
    if not t:
        return TaskGrade(task_id="easy_duplicate_charge_refund", score=0.0, feedback="Ticket not found.")
    history = state.get("action_history", [])
    score, breakdown = 0.0, {}
    classified = any(a["action_type"] == "classify_ticket" and a["ticket_id"] == "B-1001" for a in history)
    breakdown["classified"] = 0.25 if classified else 0.0
    correct_cat = t.get("predicted_category") == "billing"
    breakdown["correct_category"] = 0.25 if correct_cat else 0.0
    responded = t.get("response_sent", False)
    breakdown["responded"] = 0.25 if responded else 0.0
    resolved = t.get("resolved", False)
    breakdown["resolved"] = 0.25 if resolved else 0.0
    no_escalation = not t.get("escalated", False)
    if not no_escalation:
        breakdown["resolved"] = 0.0
    score = sum(breakdown.values())
    clamped_score = max(0.01, min(0.99, score))
    return TaskGrade(task_id="easy_duplicate_charge_refund", score=round(clamped_score, 4),
                     breakdown=breakdown, feedback="Easy task graded.")

def grade_medium_priority_queue_mix(state: Dict[str, Any]) -> TaskGrade:
    tickets = {t["id"]: t for t in state.get("tickets", [])}
    history = state.get("action_history", [])
    breakdown = {}
    # All resolved
    all_resolved = all(t.get("resolved") for t in tickets.values())
    breakdown["all_resolved"] = 0.40 if all_resolved else sum(0.13 for t in tickets.values() if t.get("resolved"))
    # Correct categories
    correct = sum(1 for tid, cat in [("D-2001","delivery"),("B-2002","billing"),("T-2003","technical")]
                  if tickets.get(tid, {}).get("predicted_category") == cat)
    breakdown["correct_categories"] = round(correct / 3 * 0.30, 4)
    # Delivery worked first (first action should target D-2001)
    first_delivery = history[0]["ticket_id"] == "D-2001" if history else False
    breakdown["delivery_prioritised"] = 0.30 if first_delivery else 0.0
    score = sum(breakdown.values())
    clamped_score = max(0.01, min(0.99, score))
    return TaskGrade(task_id="medium_priority_queue_mix", score=round(clamped_score, 4),
                     breakdown=breakdown, feedback="Medium task graded.")

def grade_hard_account_takeover(state: Dict[str, Any]) -> TaskGrade:
    tickets = state.get("tickets", [])
    t = next((x for x in tickets if x["id"] == "F-3001"), None)
    if not t:
        return TaskGrade(task_id="hard_account_takeover", score=0.0, feedback="Ticket not found.")
    breakdown = {}
    breakdown["fraud_classified"] = 0.30 if t.get("predicted_category") == "fraud" else 0.0
    breakdown["info_requested"] = 0.25 if t.get("info_requested") else 0.0
    breakdown["escalated"] = 0.30 if t.get("escalated") else 0.0
    # Penalise if resolved without escalation (dangerous)
    if t.get("resolved") and not t.get("escalated"):
        breakdown["escalated"] = -0.20
    responded = t.get("response_sent", False)
    breakdown["responded_safely"] = 0.15 if responded else 0.0
    score = sum(breakdown.values())
    clamped_score = max(0.01, min(0.99, score))
    return TaskGrade(task_id="hard_account_takeover", score=round(clamped_score, 4),
                     breakdown=breakdown, feedback="Hard task graded.")
