from typing import Any, Dict
from types import SimpleNamespace

from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator


class SmartOpsEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._config = SmartOpsConfig()
        self._simulator = SmartOpsSimulator(self._config)
        self._initialized = False
    def reset(self):
       try:
        observation = self._simulator.reset()
        self._initialized = True
       except Exception as e:
        print("RESET ERROR:", str(e))

        # fallback safe observation
        observation = {
            "task_id": "fallback",
            "message": "Environment reset fallback",
            "available_ticket_ids": [],
            "last_reward": None
        }
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