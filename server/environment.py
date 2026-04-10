from typing import Any, Dict
from types import SimpleNamespace

from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator


class SmartOpsEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = False  # 🔥 IMPORTANT

    def __init__(self):
        self._config = SmartOpsConfig()
        self._simulator = SmartOpsSimulator(self._config)
        self._initialized = False

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        try:
            observation = self._simulator.reset()
            self._initialized = True
            return observation  # ✅ ONLY observation (CRITICAL)
        except Exception as e:
            print("RESET ERROR:", e)
            return {}

    async def reset_async(self):
        return self.reset()

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action: Dict[str, Any], timeout_s=None, **kwargs):
        del timeout_s, kwargs

        if not self._initialized:
            observation = self._simulator.reset()
            self._initialized = True
            return observation, 0.0, False, {"warning": "auto-reset"}

        try:
            action_obj = SimpleNamespace(**action)
            observation, reward, done, info = self._simulator.step(action_obj)
        except Exception as e:
            print("STEP ERROR:", e)
            observation = self._simulator.reset()
            reward = 0.0
            done = False
            info = {"error": str(e)}

        return observation, reward, done, info

    async def step_async(self, action, timeout_s=None, **kwargs):
        return self.step(action, timeout_s, **kwargs)

    # -------------------------
    # STATE
    # -------------------------
    def state(self):
        try:
            return self._simulator.get_state()
        except Exception as e:
            return {"error": str(e)}

    # -------------------------
    # CLOSE (REQUIRED)
    # -------------------------
    def close(self):
        pass