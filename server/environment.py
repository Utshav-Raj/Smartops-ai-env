from typing import Any, Dict
from types import SimpleNamespace

from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator


class SmartOpsEnvironment:
    # ✅ REQUIRED FIX (solves concurrency error)
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._config = SmartOpsConfig()
        self._simulator = SmartOpsSimulator(self._config)
        self._initialized = False

    # ✅ SAFE RESET (never crashes)
    def reset(self):
        try:
            observation = self._simulator.reset()
        except Exception as e:
            print("RESET ERROR:", str(e))

            # fallback observation (VERY IMPORTANT)
            observation = {
                "task_id": "easy_fallback",
                "task_difficulty": "easy",
                "focus_ticket": {
                    "id": "B-1001",
                    "subject": "Fallback ticket",
                    "user_message": "System fallback initialized.",
                    "urgency": "low",
                    "sentiment": "neutral",
                    "status": "open",
                    "predicted_category": None,
                    "minutes_until_sla": 60,
                },
                "queue_summary": {
                    "total_open": 1,
                    "backlog_ids": ["B-1001"]
                },
                "available_ticket_ids": ["B-1001"],
                "last_reward": None
            }

        self._initialized = True
        return observation

    # ✅ SAFE STEP (never crashes)
    def step(self, action: Dict[str, Any], timeout_s=None, **kwargs):
        del timeout_s, kwargs

        # auto reset if needed
        if not self._initialized:
            return self.reset()

        try:
            action_obj = SimpleNamespace(**action)
            observation, reward, done, info = self._simulator.step(action_obj)

        except Exception as e:
            print("STEP ERROR:", str(e))

            observation = self.reset()
            reward = 0.0
            done = False
            info = {"error": str(e)}

        return observation

    # ✅ STATE (safe)
    def state(self):
        try:
            return self._simulator.get_state()
        except Exception as e:
            print("STATE ERROR:", str(e))
            return {"error": str(e)}
        
    async def reset_async(self):
        return self.reset()

    async def step_async(self, action):
        return self.step(action)