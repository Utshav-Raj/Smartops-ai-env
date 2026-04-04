"""OpenEnv server wrapper around the SmartOps simulator."""

from __future__ import annotations

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator
    from smartops_ai_env.models import SmartOpsAction, SmartOpsObservation, SupportState
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))

    from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator
    from smartops_ai_env.models import SmartOpsAction, SmartOpsObservation, SupportState


class SmartOpsEnvironment(Environment[SmartOpsAction, SmartOpsObservation, SupportState]):
    """OpenEnv-compatible environment for autonomous support operations."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: SmartOpsConfig | None = None):
        super().__init__()
        self._simulator = SmartOpsSimulator(config=config or SmartOpsConfig())

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        difficulty: str | None = None,
        **kwargs,
    ) -> SmartOpsObservation:
        return self._simulator.reset(
            task_id=task_id,
            difficulty=difficulty,
            seed=seed,
            episode_id=episode_id,
            **kwargs,
        )

    def step(
        self,
        action: SmartOpsAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> SmartOpsObservation:
        del timeout_s, kwargs
        from types import SimpleNamespace

        action_obj = SimpleNamespace(**action)
        observation, _, _, _ = self._simulator.step(action_obj)
        return observation

    @property
    def state(self) -> SupportState:
        return self._simulator.state()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SmartOps AI",
            description=(
                "Autonomous customer support and ticket-resolution benchmark with "
                "typed actions, dense reward shaping, deterministic tasks, and "
                "support-specific grading."
            ),
            version="0.1.0",
            author="OpenAI Codex",
            documentation_url="https://huggingface.co/spaces/openenv",
        )
