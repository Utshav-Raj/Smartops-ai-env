"""Reward and grading models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewardComponents(BaseModel):
    """Reward breakdown for a single simulator step."""

    classification: float = Field(default=0.0)
    response: float = Field(default=0.0)
    resolution: float = Field(default=0.0)
    escalation: float = Field(default=0.0)
    request_more_info: float = Field(default=0.0)
    sla: float = Field(default=0.0)
    delay: float = Field(default=0.0)
    terminal: float = Field(default=0.0)

    def total(self) -> float:
        return (
            self.classification
            + self.response
            + self.resolution
            + self.escalation
            + self.request_more_info
            + self.sla
            + self.delay
            + self.terminal
        )


class TicketReward(BaseModel):
    """Structured reward emitted alongside every observation."""

    raw_score: float = Field(..., description="Unnormalized dense reward before scaling.")
    normalized_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous reward scaled into the 0..1 range.",
    )
    ticket_id: str | None = Field(default=None)
    components: RewardComponents = Field(default_factory=RewardComponents)
    rationale: list[str] = Field(default_factory=list)


class TaskGrade(BaseModel):
    """Deterministic task grader output."""

    task_id: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool = Field(...)
    criteria: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
