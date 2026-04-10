from typing import Any, Dict
from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator


class SmartOpsEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

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
            self.reset()
        if isinstance(action, dict):
            pass
        else:
            action = dict(action)
        observation, reward, done, info = self._simulator.step(action)
        return observation, reward, done, info

    def state(self):
        return self._simulator.get_state()

    async def reset_async(self):
        return self.reset()

    async def step_async(self, action):
        return self.step(action)

    def close(self):
        pass
