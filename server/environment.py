from typing import Any, Dict, Tuple
from types import SimpleNamespace

from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator


class SmartOpsEnvironment:
    def __init__(self):
        self._config = SmartOpsConfig()
        self._simulator = SmartOpsSimulator(self._config)
        self._initialized = False

    def reset(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Reset environment and start new episode"""
        observation = self._simulator.reset()
        self._initialized = True

        return observation, 0.0, False, {}

    def step(self, action: Dict[str, Any], timeout_s=None, **kwargs):
        """Take a step in environment"""
        del timeout_s, kwargs

        # Ensure reset is called first
        if not self._initialized:
            observation = self._simulator.reset()
            self._initialized = True
            return observation, 0.0, False, {"warning": "auto-reset triggered"}

        try:
            action_obj = SimpleNamespace(**action)
            observation, reward, done, info = self._simulator.step(action_obj)

        except Exception as e:
            print("SIMULATOR ERROR:", str(e))
            observation = self._simulator.reset()
            reward = 0.0
            done = False
            info = {"error": str(e)}

        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state"""
        return self._simulator.get_state()