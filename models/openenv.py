"""OpenEnv-facing Pydantic models."""

from __future__ import annotations

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator

from .reward import TicketReward
from .ticket import (
    ActionRecord,
    ActionType,
    MetricsSnapshot,
    QueueSummary,
    SupportTicket,
    TaskDifficulty,
    TicketCategory,
    TicketPublicView,
)


class SmartOpsAction(Action):
    """Structured action space for customer support operations."""

    action_type: ActionType = Field(..., description="The support operation to execute.")
    ticket_id: str = Field(..., min_length=1, description="Target ticket identifier.")
    category: TicketCategory | None = Field(
        default=None,
        description="Required for classify_ticket.",
    )
    message: str | None = Field(
        default=None,
        description="Required for respond_to_ticket.",
    )
    reason: str | None = Field(
        default=None,
        description="Required for escalate_ticket.",
    )
    question: str | None = Field(
        default=None,
        description="Required for request_more_info.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "SmartOpsAction":
        requirements = {
            ActionType.CLASSIFY_TICKET: ("category",),
            ActionType.RESPOND_TO_TICKET: ("message",),
            ActionType.ESCALATE_TICKET: ("reason",),
            ActionType.RESOLVE_TICKET: tuple(),
            ActionType.REQUEST_MORE_INFO: ("question",),
        }
        required = set(requirements[self.action_type])
        provided_map = {
            "category": self.category,
            "message": self.message,
            "reason": self.reason,
            "question": self.question,
        }

        missing = [field for field in required if provided_map[field] in (None, "")]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{self.action_type.value} requires: {joined}")

        for field_name, field_value in provided_map.items():
            if field_name not in required and field_value not in (None, ""):
                raise ValueError(
                    f"{field_name} is not valid for action_type={self.action_type.value}"
                )

        return self


class SmartOpsObservation(Observation):
    """Observation exposed to the agent."""

    task_id: str = Field(..., description="Current deterministic task identifier.")
    task_difficulty: TaskDifficulty = Field(..., description="Scenario difficulty.")
    focus_ticket: TicketPublicView | None = Field(
        default=None,
        description="Ticket currently highlighted for the agent.",
    )
    queue_summary: QueueSummary = Field(..., description="Operational summary of the queue.")
    elapsed_minutes: int = Field(..., ge=0, description="Simulated minutes since reset.")
    previous_actions: list[ActionRecord] = Field(
        default_factory=list,
        description="Recent action history visible to the agent.",
    )
    system_metrics: MetricsSnapshot = Field(
        default_factory=MetricsSnapshot,
        description="Partial support system metrics visible to the agent.",
    )
    workflow_hint: str = Field(..., description="High-level operating guidance.")
    available_ticket_ids: list[str] = Field(
        default_factory=list,
        description="Remaining actionable tickets.",
    )
    last_reward: TicketReward | None = Field(
        default=None,
        description="Structured reward breakdown for the previous action.",
    )


class SupportState(State):
    """Full internal environment state returned by state()."""

    active_task_id: str | None = Field(default=None, description="Loaded task identifier.")
    task_difficulty: TaskDifficulty | None = Field(default=None, description="Loaded task difficulty.")
    elapsed_minutes: int = Field(default=0, ge=0)
    max_steps: int = Field(default=0, ge=0)
    done: bool = Field(default=False)
    focus_ticket_id: str | None = Field(default=None)
    tickets: list[SupportTicket] = Field(default_factory=list)
    action_history: list[ActionRecord] = Field(default_factory=list)
    customer_satisfaction: float = Field(default=0.78, ge=0.0, le=1.0)
    sla_adherence: float = Field(default=1.0, ge=0.0, le=1.0)
    last_reward: TicketReward | None = Field(default=None)
    deterministic_seed: int = Field(default=0)
    task_description: str = Field(default="")
    expected_priority_order: list[str] = Field(default_factory=list)
    metrics: MetricsSnapshot | None = Field(default=None)
    terminal_score: float | None = Field(default=None, ge=0.0, le=1.0)
    terminal_notes: list[str] = Field(default_factory=list)
