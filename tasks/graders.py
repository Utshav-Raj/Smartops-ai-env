"""Deterministic graders for SmartOps benchmark tasks."""

from __future__ import annotations

from collections import defaultdict

from smartops_ai_env.env.scoring import keyword_group_coverage
from smartops_ai_env.models import SupportState, TaskGrade, TicketStatus
from smartops_ai_env.tasks.catalog import get_task


def _history_by_ticket(state: SupportState) -> dict[str, list]:
    grouped = defaultdict(list)
    for record in state.action_history:
        grouped[record.ticket_id].append(record)
    return grouped


def grade_easy_duplicate_charge_refund(state: SupportState) -> TaskGrade:
    task = get_task("easy_duplicate_charge_refund")
    ticket = state.tickets[0]
    history = _history_by_ticket(state).get(ticket.id, [])

    classification = 1.0 if ticket.predicted_category == ticket.category else 0.0
    response_texts = [record.summary for record in history if record.action_type.value == "respond_to_ticket"]
    response_score = 0.0
    if response_texts:
        best = max(
            keyword_group_coverage(text, ticket.expected_response_groups)[0]
            for text in response_texts
        )
        response_score = round(best, 4)
    resolution = 1.0 if ticket.status == TicketStatus.RESOLVED else 0.0
    no_escalation = 1.0 if ticket.status != TicketStatus.ESCALATED else 0.0

    criteria = {
        "classification": classification,
        "helpful_response": response_score,
        "resolved": resolution,
        "avoided_unnecessary_escalation": no_escalation,
    }
    score = round(sum(criteria.values()) / len(criteria), 4)
    notes = []
    if classification == 0.0:
        notes.append("Billing classification was missed.")
    if response_score < 1.0:
        notes.append("Response did not fully mention refund handling and timing.")
    if resolution == 0.0:
        notes.append("Ticket was not fully resolved.")

    return TaskGrade(
        task_id=task.task_id,
        score=score,
        passed=score >= 0.85,
        criteria=criteria,
        notes=notes or ["Duplicate-charge ticket handled correctly."],
    )


def grade_medium_priority_queue_mix(state: SupportState) -> TaskGrade:
    task = get_task("medium_priority_queue_mix")
    history_map = _history_by_ticket(state)
    tickets = {ticket.id: ticket for ticket in state.tickets}

    classification_scores = []
    response_scores = []
    resolution_scores = []
    first_touch_order = []
    seen = set()

    for record in state.action_history:
        if record.ticket_id not in seen:
            first_touch_order.append(record.ticket_id)
            seen.add(record.ticket_id)

    for ticket in state.tickets:
        classification_scores.append(1.0 if ticket.predicted_category == ticket.category else 0.0)
        ticket_history = history_map.get(ticket.id, [])
        response_texts = [record.summary for record in ticket_history if record.action_type.value == "respond_to_ticket"]
        if response_texts:
            best = max(
                keyword_group_coverage(text, ticket.expected_response_groups)[0]
                for text in response_texts
            )
        else:
            best = 0.0
        response_scores.append(round(best, 4))
        resolution_scores.append(1.0 if ticket.status == TicketStatus.RESOLVED else 0.0)

    priority_match = 0.0
    if first_touch_order:
        expected_prefix = task.priority_order[: len(first_touch_order)]
        matches = sum(1 for actual, expected in zip(first_touch_order, expected_prefix) if actual == expected)
        priority_match = round(matches / len(task.priority_order), 4)

    criteria = {
        "classification_accuracy": round(sum(classification_scores) / len(classification_scores), 4),
        "response_quality": round(sum(response_scores) / len(response_scores), 4),
        "queue_prioritization": priority_match,
        "resolution_completion": round(sum(resolution_scores) / len(resolution_scores), 4),
    }
    score = round(sum(criteria.values()) / len(criteria), 4)
    notes = []
    if priority_match < 1.0:
        notes.append("Queue was not worked in the expected urgency/SLA order.")
    if criteria["resolution_completion"] < 1.0:
        open_ids = [ticket.id for ticket in tickets.values() if ticket.status != TicketStatus.RESOLVED]
        notes.append(f"Some medium task tickets were not resolved: {', '.join(open_ids)}")

    return TaskGrade(
        task_id=task.task_id,
        score=score,
        passed=score >= 0.8,
        criteria=criteria,
        notes=notes or ["Mixed queue handled with solid prioritization and resolution quality."],
    )


def grade_hard_account_takeover(state: SupportState) -> TaskGrade:
    task = get_task("hard_account_takeover")
    ticket = state.tickets[0]
    history = _history_by_ticket(state).get(ticket.id, [])

    classification = 1.0 if ticket.predicted_category == ticket.category else 0.0

    response_texts = [record.summary for record in history if record.action_type.value == "respond_to_ticket"]
    response_score = 0.0
    if response_texts:
        response_score = round(
            max(keyword_group_coverage(text, ticket.expected_response_groups)[0] for text in response_texts),
            4,
        )

    info_requests = [record.summary for record in history if record.action_type.value == "request_more_info"]
    info_score = 0.0
    if info_requests:
        info_score = round(
            max(keyword_group_coverage(text, ticket.expected_info_request_groups)[0] for text in info_requests),
            4,
        )

    escalation_reasons = [record.summary for record in history if record.action_type.value == "escalate_ticket"]
    escalation_score = 0.0
    if escalation_reasons:
        escalation_score = round(
            max(keyword_group_coverage(text, ticket.expected_escalation_groups)[0] for text in escalation_reasons),
            4,
        )

    multi_turn = 1.0 if ticket.follow_up_revealed and ticket.info_requested else 0.0
    final_status = 1.0 if ticket.status == TicketStatus.ESCALATED else 0.0

    criteria = {
        "classification": classification,
        "safety_response": response_score,
        "information_gathering": info_score,
        "multi_turn_handling": multi_turn,
        "escalation_quality": escalation_score,
        "final_status": final_status,
    }
    score = round(sum(criteria.values()) / len(criteria), 4)
    notes = []
    if final_status == 0.0:
        notes.append("Fraud incident was not escalated to a terminal escalated state.")
    if info_score < 1.0:
        notes.append("Missing verification details before handoff.")
    if response_score < 1.0:
        notes.append("Safety-focused response did not cover account protection steps.")

    return TaskGrade(
        task_id=task.task_id,
        score=score,
        passed=score >= 0.8,
        criteria=criteria,
        notes=notes or ["Account takeover was triaged, verified, and escalated correctly."],
    )


def grade_task(task_id: str, state: SupportState) -> TaskGrade:
    """Dispatch to the deterministic task grader."""

    graders = {
        "easy_duplicate_charge_refund": grade_easy_duplicate_charge_refund,
        "medium_priority_queue_mix": grade_medium_priority_queue_mix,
        "hard_account_takeover": grade_hard_account_takeover,
    }
    if task_id not in graders:
        raise KeyError(f"No grader registered for task_id={task_id}")
    return graders[task_id](state)
