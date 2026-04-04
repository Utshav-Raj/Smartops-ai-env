from typing import Any, Dict
from types import SimpleNamespace

from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator


class SmartOpsEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._config = SmartOpsConfig()
        self._simulator = SmartOpsSimulator(self._config)
        self._initialized = False
    def reset(self):
        observation = self._simulator.reset()
        self._initialized = True
        return observation

    def step(self, action: Dict[str, Any], timeout_s=None, **kwargs):
        del timeout_s, kwargs

        if not self._initialized:
            observation = self._simulator.reset()
            self._initialized = True
            return observation

        try:
            action_obj = SimpleNamespace(**action)
            observation, _, _, _ = self._simulator.step(action_obj)

        except Exception as e:
            print("SIMULATOR ERROR:", str(e))
            observation = self._simulator.reset()

        return observation

    def state(self):
        return self._simulator.get_state()