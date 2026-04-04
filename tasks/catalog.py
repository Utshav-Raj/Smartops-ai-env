"""Deterministic SmartOps task catalog."""

from __future__ import annotations

from smartops_ai_env.models.ticket import (
    SupportTicket,
    TaskDifficulty,
    TaskScenario,
    TicketCategory,
    TicketSentiment,
    TicketStatus,
    TicketUrgency,
)


def _build_catalog() -> dict[str, TaskScenario]:
    easy = TaskScenario(
        task_id="easy_duplicate_charge_refund",
        difficulty=TaskDifficulty.EASY,
        title="Duplicate charge refund",
        description=(
            "A single billing ticket with a clear duplicate-charge remediation path. "
            "The agent should classify, reassure the customer, and resolve within SLA."
        ),
        scenario_overview=(
            "An annual subscription upgrade charged the customer twice after a failed checkout."
        ),
        expected_behavior=[
            "Classify the ticket as billing.",
            "Respond with a refund-oriented message mentioning the duplicate charge and payout timing.",
            "Resolve the ticket without escalating it.",
        ],
        priority_order=["B-1001"],
        max_steps=6,
        seed=17,
        grader_name="grade_easy_duplicate_charge_refund",
        tickets=[
            SupportTicket(
                id="B-1001",
                subject="Duplicate annual plan charge",
                customer_name="Priya Nair",
                user_message=(
                    "I tried upgrading to the Pro annual plan once, the page errored, "
                    "and now I see two $249 charges on my card. Please fix this today."
                ),
                category=TicketCategory.BILLING,
                urgency=TicketUrgency.HIGH,
                sentiment=TicketSentiment.FRUSTRATED,
                status=TicketStatus.OPEN,
                sla_deadline_minutes=45,
                expected_response_groups=[
                    ["refund", "reverse"],
                    ["duplicate charge", "double charge", "charged twice"],
                    ["3-5 business days", "3 to 5 business days", "within 5 business days"],
                ],
                prohibited_response_terms=["contact your bank", "cannot help"],
                resolution_summary="Process the duplicate-charge refund and confirm settlement timing.",
            )
        ],
    )

    medium = TaskScenario(
        task_id="medium_priority_queue_mix",
        difficulty=TaskDifficulty.MEDIUM,
        title="Mixed queue prioritization",
        description=(
            "Three concurrent tickets with different urgency levels force the agent to "
            "prioritize the queue while keeping classifications and responses accurate."
        ),
        scenario_overview=(
            "A delivery incident is about to breach SLA, a mobile export bug needs a workaround, "
            "and finance needs a corrected invoice."
        ),
        expected_behavior=[
            "Work the delivery ticket first because it is closest to SLA breach.",
            "Classify each ticket correctly before taking terminal action.",
            "Respond with ticket-specific remediation and resolve all three cases.",
        ],
        priority_order=["D-2001", "T-2002", "B-2003"],
        max_steps=12,
        seed=23,
        grader_name="grade_medium_priority_queue_mix",
        tickets=[
            SupportTicket(
                id="D-2001",
                subject="Shipment stalled before store opening",
                customer_name="Avery Logistics",
                user_message=(
                    "Our replacement router was supposed to arrive before 5 PM, but tracking "
                    "has been frozen for two days and our store opens tomorrow morning. "
                    "Can someone actually trace this package?"
                ),
                category=TicketCategory.DELIVERY,
                urgency=TicketUrgency.CRITICAL,
                sentiment=TicketSentiment.ANGRY,
                sla_deadline_minutes=20,
                expected_response_groups=[
                    ["trace", "carrier investigation", "carrier trace"],
                    ["replacement", "ship another", "priority resend"],
                    ["refund", "credit"],
                ],
                prohibited_response_terms=["wait another week", "nothing we can do"],
                resolution_summary="Trace the package and offer a replacement or refund path.",
            ),
            SupportTicket(
                id="T-2002",
                subject="iOS export crash after update",
                customer_name="Jordan Rivera",
                user_message=(
                    "The iOS app crashes every time I tap Export CSV after the latest update. "
                    "I already reinstalled once and still need the report tonight."
                ),
                category=TicketCategory.TECHNICAL,
                urgency=TicketUrgency.HIGH,
                sentiment=TicketSentiment.FRUSTRATED,
                sla_deadline_minutes=40,
                expected_response_groups=[
                    ["workaround", "temporary workaround"],
                    ["web dashboard", "browser export", "desktop export"],
                    ["hotfix", "engineering fix", "patch"],
                ],
                prohibited_response_terms=["just reinstall again", "cannot reproduce"],
                resolution_summary="Offer the browser-export workaround while the hotfix is prepared.",
            ),
            SupportTicket(
                id="B-2003",
                subject="Invoice resend with VAT details",
                customer_name="Northline Finance",
                user_message=(
                    "Can you resend our March invoice with the VAT number included? "
                    "Finance says the PDF you sent last week was blank."
                ),
                category=TicketCategory.BILLING,
                urgency=TicketUrgency.MEDIUM,
                sentiment=TicketSentiment.CALM,
                sla_deadline_minutes=90,
                expected_response_groups=[
                    ["resend", "send again"],
                    ["vat", "tax number"],
                    ["invoice pdf", "corrected invoice", "updated pdf"],
                ],
                prohibited_response_terms=["open a new ticket", "ask finance to wait"],
                resolution_summary="Resend the invoice with VAT metadata and the corrected PDF.",
            ),
        ],
    )

    hard = TaskScenario(
        task_id="hard_account_takeover",
        difficulty=TaskDifficulty.HARD,
        title="Account takeover and unauthorized charges",
        description=(
            "An ambiguous, multi-turn fraud incident blends billing, account-security, and "
            "access issues. The agent must gather missing evidence and escalate correctly."
        ),
        scenario_overview=(
            "The customer reports unauthorized overnight orders, an admin-email change, "
            "and loss of billing-panel access."
        ),
        expected_behavior=[
            "Classify the ticket as fraud despite the mixed billing and access symptoms.",
            "Send a safety-focused response telling the customer the account will be locked or frozen and investigated.",
            "Request the missing verification details before escalating.",
            "Escalate to the fraud team with a reason that cites unauthorized charges and the admin-email change.",
        ],
        priority_order=["F-3001"],
        max_steps=8,
        seed=31,
        grader_name="grade_hard_account_takeover",
        tickets=[
            SupportTicket(
                id="F-3001",
                subject="Unauthorized overnight orders and admin takeover",
                customer_name="Northstar Ops",
                user_message=(
                    "Four overnight orders hit our account from a city we've never shipped to, "
                    "the admin email was changed, and now I can't access the billing panel. "
                    "I need this stopped right now."
                ),
                category=TicketCategory.FRAUD,
                urgency=TicketUrgency.CRITICAL,
                sentiment=TicketSentiment.PANICKED,
                sla_deadline_minutes=15,
                expected_response_groups=[
                    ["lock", "freeze", "secure"],
                    ["investigation", "fraud team", "security review"],
                    ["password reset", "credential reset", "reset access"],
                ],
                expected_info_request_groups=[
                    ["last legitimate invoice", "last authorized invoice", "last valid charge"],
                    ["original admin email", "previous admin email", "admin email before change"],
                ],
                expected_escalation_groups=[
                    ["unauthorized charges", "unauthorized orders", "fraudulent orders"],
                    ["admin email change", "admin email changed", "account takeover"],
                ],
                prohibited_response_terms=["resolve this myself", "just ignore the charges"],
                escalation_required=True,
                info_request_required=True,
                target_terminal_status=TicketStatus.ESCALATED,
                resolution_summary="Escalate to fraud with complete takeover evidence and freeze access.",
                follow_up_customer_message=(
                    "The last legitimate invoice was INV-8821 on March 28 and the original admin "
                    "email is ops@northstar.example."
                ),
            )
        ],
    )

    return {
        easy.task_id: easy,
        medium.task_id: medium,
        hard.task_id: hard,
    }


_CATALOG = _build_catalog()


def get_task_catalog() -> dict[str, TaskScenario]:
    """Return a deep-copy friendly task catalog."""

    return {task_id: task.model_copy(deep=True) for task_id, task in _CATALOG.items()}


def get_task(task_id: str) -> TaskScenario:
    """Return a fresh task definition."""

    if task_id not in _CATALOG:
        raise KeyError(f"Unknown task_id: {task_id}")
    return _CATALOG[task_id].model_copy(deep=True)


def list_task_ids() -> list[str]:
    """Return deterministic task order."""

    return list(_CATALOG.keys())
