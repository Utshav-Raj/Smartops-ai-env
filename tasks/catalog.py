from __future__ import annotations
from ..models.openenv import (
    SupportTicket, TaskDifficulty, TaskScenario,
    TicketSentiment, TicketStatus, TicketUrgency,
)

def _easy_duplicate_charge_refund():
    return TaskScenario(
        task_id="easy_duplicate_charge_refund", difficulty=TaskDifficulty.easy,
        description="Customer charged twice. Classify billing, respond, resolve.",
        tickets=[SupportTicket(
            id="B-1001", subject="Charged twice for my order!",
            user_message="I was charged twice for order #44821. Please refund the duplicate.",
            urgency=TicketUrgency.medium, sentiment=TicketSentiment.frustrated,
            status=TicketStatus.open, minutes_until_sla=90,
            context={"expected_category": "billing", "should_escalate": False},
        )]
    )

def _medium_priority_queue_mix():
    return TaskScenario(
        task_id="medium_priority_queue_mix", difficulty=TaskDifficulty.medium,
        description="3 tickets. Prioritise delivery (SLA=15min), then billing, then technical.",
        tickets=[
            SupportTicket(id="D-2001", subject="Package not arrived after 2 weeks",
                user_message="My package was due 10 days ago. Where is it?",
                urgency=TicketUrgency.high, sentiment=TicketSentiment.angry,
                status=TicketStatus.open, minutes_until_sla=15,
                context={"expected_category": "delivery", "should_escalate": False}),
            SupportTicket(id="B-2002", subject="Invoice doesn't add up",
                user_message="The totals on my last invoice look wrong.",
                urgency=TicketUrgency.low, sentiment=TicketSentiment.neutral,
                status=TicketStatus.open, minutes_until_sla=90,
                context={"expected_category": "billing", "should_escalate": False}),
            SupportTicket(id="T-2003", subject="App crashes on login",
                user_message="Every time I try to log in the app crashes.",
                urgency=TicketUrgency.medium, sentiment=TicketSentiment.frustrated,
                status=TicketStatus.open, minutes_until_sla=60,
                context={"expected_category": "technical", "should_escalate": False}),
        ]
    )

def _hard_account_takeover():
    return TaskScenario(
        task_id="hard_account_takeover", difficulty=TaskDifficulty.hard,
        description="Account takeover indicators. Classify fraud, verify identity, escalate.",
        tickets=[SupportTicket(
            id="F-3001", subject="Strange activity on my account",
            user_message="I received password resets I didn't request, unfamiliar transactions, and my delivery address was changed. I'm very worried.",
            urgency=TicketUrgency.critical, sentiment=TicketSentiment.angry,
            status=TicketStatus.open, minutes_until_sla=30,
            context={"expected_category": "fraud", "should_escalate": True,
                     "must_not_share_data": True, "must_verify_identity": True},
        )]
    )

_CATALOG = {
    "easy_duplicate_charge_refund": _easy_duplicate_charge_refund,
    "medium_priority_queue_mix": _medium_priority_queue_mix,
    "hard_account_takeover": _hard_account_takeover,
}

def get_task(task_id: str):
    if task_id not in _CATALOG:
        raise ValueError(f"Unknown task: {task_id!r}. Available: {list(_CATALOG)}")
    return _CATALOG[task_id]()
