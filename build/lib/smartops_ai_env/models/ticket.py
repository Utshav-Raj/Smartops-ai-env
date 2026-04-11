"""Domain models for tickets and task scenarios."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    DELIVERY = "delivery"
    FRAUD = "fraud"


class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketSentiment(str, Enum):
    CALM = "calm"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    PANICKED = "panicked"


class TicketStatus(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ActionType(str, Enum):
    CLASSIFY_TICKET = "classify_ticket"
    RESPOND_TO_TICKET = "respond_to_ticket"
    ESCALATE_TICKET = "escalate_ticket"
    RESOLVE_TICKET = "resolve_ticket"
    REQUEST_MORE_INFO = "request_more_info"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SupportTicket(BaseModel):
    """Full internal ticket representation."""

    id: str = Field(...)
    subject: str = Field(...)
    customer_name: str = Field(...)
    user_message: str = Field(...)
    category: TicketCategory = Field(...)
    urgency: TicketUrgency = Field(...)
    sentiment: TicketSentiment = Field(...)
    status: TicketStatus = Field(default=TicketStatus.OPEN)
    predicted_category: TicketCategory | None = Field(default=None)
    sla_deadline_minutes: int = Field(..., ge=1)
    created_at_minutes: int = Field(default=0, ge=0)
    last_updated_minutes: int = Field(default=0, ge=0)
    expected_response_groups: list[list[str]] = Field(default_factory=list)
    expected_info_request_groups: list[list[str]] = Field(default_factory=list)
    expected_escalation_groups: list[list[str]] = Field(default_factory=list)
    prohibited_response_terms: list[str] = Field(default_factory=list)
    escalation_required: bool = Field(default=False)
    info_request_required: bool = Field(default=False)
    target_terminal_status: TicketStatus = Field(default=TicketStatus.RESOLVED)
    resolution_summary: str = Field(default="")
    follow_up_customer_message: str | None = Field(default=None)
    follow_up_revealed: bool = Field(default=False)
    helpful_response_sent: bool = Field(default=False)
    info_requested: bool = Field(default=False)
    escalation_sent: bool = Field(default=False)
    resolution_complete: bool = Field(default=False)
    handled: bool = Field(default=False)


class TicketPublicView(BaseModel):
    """Observation-safe ticket projection."""

    id: str = Field(...)
    subject: str = Field(...)
    user_message: str = Field(...)
    urgency: TicketUrgency = Field(...)
    sentiment: TicketSentiment = Field(...)
    status: TicketStatus = Field(...)
    predicted_category: TicketCategory | None = Field(default=None)
    minutes_until_sla: int = Field(...)
    helpful_response_sent: bool = Field(default=False)
    info_requested: bool = Field(default=False)
    escalation_sent: bool = Field(default=False)
    follow_up_revealed: bool = Field(default=False)


class QueueSummary(BaseModel):
    """Aggregated queue-level observation."""

    total_open: int = Field(default=0)
    total_pending: int = Field(default=0)
    total_resolved: int = Field(default=0)
    total_escalated: int = Field(default=0)
    backlog_ids: list[str] = Field(default_factory=list)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_urgency: dict[str, int] = Field(default_factory=dict)
    next_sla_ticket_id: str | None = Field(default=None)
    next_sla_minutes: int | None = Field(default=None)


class MetricsSnapshot(BaseModel):
    """Observable system metrics."""

    customer_satisfaction: float = Field(default=0.0, ge=0.0, le=1.0)
    sla_adherence: float = Field(default=1.0, ge=0.0, le=1.0)
    resolved_count: int = Field(default=0)
    escalated_count: int = Field(default=0)
    open_count: int = Field(default=0)
    pending_count: int = Field(default=0)
    backlog_size: int = Field(default=0)


class ActionRecord(BaseModel):
    """Persistent action history record."""

    step_index: int = Field(...)
    timestamp_minutes: int = Field(...)
    ticket_id: str = Field(...)
    action_type: ActionType = Field(...)
    summary: str = Field(...)
    raw_reward: float = Field(...)
    normalized_reward: float = Field(..., ge=0.0, le=1.0)
    outcome: str = Field(...)


class TaskScenario(BaseModel):
    """Deterministic support benchmark scenario."""

    task_id: str = Field(...)
    difficulty: TaskDifficulty = Field(...)
    title: str = Field(...)
    description: str = Field(...)
    scenario_overview: str = Field(...)
    expected_behavior: list[str] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=8, ge=1)
    seed: int = Field(default=17)
    grader_name: str = Field(...)
    tickets: list[SupportTicket] = Field(default_factory=list)
