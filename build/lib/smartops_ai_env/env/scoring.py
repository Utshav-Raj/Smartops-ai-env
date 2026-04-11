"""Shared deterministic scoring helpers."""

from __future__ import annotations

import re
from typing import Iterable

from smartops_ai_env.models.ticket import (
    SupportTicket,
    TicketStatus,
    TicketUrgency,
)


ACTION_TIME_COSTS = {
    "classify_ticket": 3,
    "respond_to_ticket": 5,
    "escalate_ticket": 4,
    "resolve_ticket": 2,
    "request_more_info": 6,
}

URGENCY_WEIGHTS = {
    TicketUrgency.LOW: 1,
    TicketUrgency.MEDIUM: 2,
    TicketUrgency.HIGH: 3,
    TicketUrgency.CRITICAL: 4,
}

TERMINAL_STATUSES = {TicketStatus.RESOLVED, TicketStatus.ESCALATED}


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a float into a fixed inclusive range."""

    return max(minimum, min(maximum, value))


def normalize_text(text: str) -> str:
    """Normalize text for deterministic keyword matching."""

    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def keyword_group_coverage(
    text: str,
    keyword_groups: Iterable[Iterable[str]],
) -> tuple[float, bool, list[list[str]]]:
    """Measure how many semantic keyword groups are satisfied."""

    groups = [list(group) for group in keyword_groups]
    if not groups:
        return 1.0, True, []

    normalized = normalize_text(text)
    matched = 0
    missing: list[list[str]] = []

    for group in groups:
        if any(normalize_text(term) in normalized for term in group):
            matched += 1
        else:
            missing.append(group)

    coverage = matched / len(groups)
    return coverage, matched == len(groups), missing


def contains_prohibited(text: str, prohibited_terms: Iterable[str]) -> list[str]:
    """Return prohibited terms found in text."""

    normalized = normalize_text(text)
    hits = []
    for term in prohibited_terms:
        if normalize_text(term) in normalized:
            hits.append(term)
    return hits


def squash_reward(raw_score: float, minimum_raw: float = -1.0, maximum_raw: float = 0.6) -> float:
    """Normalize raw reward into the required 0..1 range."""

    if maximum_raw <= minimum_raw:
        return clamp(raw_score)
    return clamp((raw_score - minimum_raw) / (maximum_raw - minimum_raw))


def is_terminal(ticket: SupportTicket) -> bool:
    """Return True when a ticket is in a terminal state."""

    return ticket.status in TERMINAL_STATUSES


def minutes_until_sla(ticket: SupportTicket, elapsed_minutes: int) -> int:
    """Remaining minutes before the SLA expires."""

    return ticket.sla_deadline_minutes - elapsed_minutes


def priority_sort_key(ticket: SupportTicket, elapsed_minutes: int) -> tuple[int, int, str]:
    """Sort by urgency, then SLA pressure, then id for determinism."""

    return (
        -URGENCY_WEIGHTS[ticket.urgency],
        minutes_until_sla(ticket, elapsed_minutes),
        ticket.id,
    )
