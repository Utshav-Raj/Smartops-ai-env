"""Runtime configuration for SmartOps AI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SmartOpsConfig(BaseModel):
    """Configuration for the deterministic SmartOps simulation."""

    default_task_id: str = Field(
        default="easy_duplicate_charge_refund",
        description="Task to load when reset() receives no explicit task identifier.",
    )
    default_seed: int = Field(
        default=17,
        description="Deterministic seed applied when a task does not provide one.",
    )
    max_history_in_observation: int = Field(
        default=5,
        ge=1,
        description="Number of recent actions included in observations.",
    )
    log_level: str = Field(default="INFO", description="Logger level for the simulator.")
