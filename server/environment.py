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

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, episode_id=None, **kwargs):
        try:
            options  = kwargs.get("options", {}) or {}
            scenario = options.get("scenario")
            task_id  = options.get("task_id")          # ← extract task_id from options

            if scenario is not None:
                observation = self._simulator.reset(scenario=scenario)
            elif task_id is not None:
                observation = self._simulator.reset(task_id=task_id)  # ← forward it
            else:
                observation = self._simulator.reset()

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
        pass