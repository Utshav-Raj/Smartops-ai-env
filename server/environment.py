from env.simulator import SmartOpsSimulator
from env.config import SmartOpsConfig
from openenv.core import Environment


class SmartOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False  # required by OpenEnv

    def __init__(self):
        super().__init__()
        self._config = SmartOpsConfig()
        self._simulator = SmartOpsSimulator(self._config)
        self._initialized = False
        self._current_task_id = "easy_duplicate_charge_refund"

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs):
        """
        Accept task_id in any of these forms (all used by different callers):
          reset(task_id="medium_priority_queue_mix")           # WebSocket / platform
          reset(options={"task_id": "medium_priority_queue_mix"}) # HTTP inference.py
          reset()                                               # default → easy
        """
        try:
            options = kwargs.get("options", {}) or {}

            # Resolve effective task_id — prefer direct arg, then options dict
            effective_task_id = (
                task_id
                or options.get("task_id")
                or options.get("scenario")
                or "easy_duplicate_charge_refund"
            )
            self._current_task_id = effective_task_id

            observation = self._simulator.reset(task_id=effective_task_id)
            self._initialized = True
            return observation
        except Exception as e:
            print("RESET ERROR:", str(e))
            raise

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action, timeout_s=None, **kwargs):
        try:
            if not self._initialized:
                # Auto-reset preserving the last requested task
                self._simulator.reset(task_id=self._current_task_id)
                self._initialized = True
            observation, reward, done, info = self._simulator.step(action)
            return observation
        except Exception as e:
            print("STEP ERROR:", str(e))
            raise

    # -------------------------
    # STATE
    # -------------------------
    @property
    def state(self):
        try:
            return self._simulator.get_state()
        except Exception as e:
            return {"error": str(e)}

    # -------------------------
    # CLOSE (REQUIRED)
    # -------------------------
    def close(self):
        self._initialized = False