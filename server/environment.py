from env.simulator import SmartOpsSimulator
from env.config import SmartOpsConfig
from openenv.core import Environment

class SmartOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False  # 🔥 IMPORTANT

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
            options = kwargs.get("options", {})
            scenario = options.get("scenario")
            observation = self._simulator.reset(scenario=scenario)
            self._initialized = True
            return observation
        except Exception as e:
            print("RESET ERROR:", str(e))
            raise e
    # -------------------------
    # STEP
    # -------------------------
    def step(self, action, timeout_s=None, **kwargs):
        try:
            observation, reward, done, info = self._simulator.step(action)
            return observation
        except Exception as e:
            print("STEP ERROR:", str(e))
            raise e
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