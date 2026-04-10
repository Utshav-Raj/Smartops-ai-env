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

            # 🔥 ensure it is dict (safe)
            if hasattr(observation, "model_dump"):
                observation = observation.model_dump()

            self._initialized = True
            return observation

        except Exception as e:
            print("RESET ERROR:", str(e))

            # 🔥 SAFE FALLBACK (prevents crash)
            return {
                "error": str(e),
                "status": "reset_failed_but_safe"
            }
    # -------------------------
    # STEP
    # -------------------------
    def step(self, action, timeout_s=None, **kwargs):
        from types import SimpleNamespace

        try:
            action_obj = SimpleNamespace(**action)
            observation, reward, done, info = self._simulator.step(action_obj)

            if hasattr(observation, "model_dump"):
                observation = observation.model_dump()

            return observation, reward, done, info

        except Exception as e:
            print("STEP ERROR:", str(e))

            return {
                "error": str(e)
            }, 0.0, False, {}
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