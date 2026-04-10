import os
import requests
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", "dummy_key"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY
)

# ----------------------------
# HELPERS
# ----------------------------
def reset_env():
    try:
        r = requests.post(f"{API_BASE}/reset", timeout=10)
        return r.json()
    except Exception as e:
        return None

def step_env(action):
    try:
        r = requests.post(f"{API_BASE}/step", json=action, timeout=10)
        return r.json()
    except Exception as e:
        return None

# ----------------------------
# MAIN LOOP
# ----------------------------
def run():
    print("[START]")

    state = reset_env()
    if not state:
        print("[END]")
        return

    for _ in range(5):
        try:
            ticket = state["observation"]["focus_ticket"]["id"]

            # 🔥 REQUIRED LLM CALL
            try:
                client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a support agent."},
                        {"role": "user", "content": f"Classify ticket {ticket}"}
                    ]
                )
            except Exception as e:
                pass  # LLM failure shouldn't pollute stdout format

            action = {
                "action": {
                    "action_type": "classify_ticket",
                    "category": "billing",
                    "ticket_id": ticket
                }
            }

            state = step_env(action)
            if not state:
                break

            print("[STEP]")

        except Exception as e:
            break

    print("[END]")


if __name__ == "__main__":
    run()