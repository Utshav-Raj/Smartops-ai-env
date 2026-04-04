"""Typed client for the SmartOps AI environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SmartOpsAction, SmartOpsObservation, SupportState


class SmartOpsAIEnv(EnvClient[SmartOpsAction, SmartOpsObservation, SupportState]):
    """Typed OpenEnv client for SmartOps AI."""

    def _step_payload(self, action: SmartOpsAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SmartOpsObservation]:
        observation_payload = dict(payload.get("observation", {}))
        observation_payload.setdefault("reward", payload.get("reward"))
        observation_payload.setdefault("done", payload.get("done", False))
        observation = SmartOpsObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SupportState:
        return SupportState.model_validate(payload)
