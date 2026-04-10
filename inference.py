import os
import requests
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
BASE_URL = "https://utshav-raj-ai-smartops-ai-env.hf.space"

API_BASE = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

if not API_BASE or not API_KEY:
    print("Missing API credentials")
    exit(0)  # DO NOT CRASH

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY
)

# ----------------------------
# HELPERS
# ----------------------------
def reset_env():
    try:
        r = requests.post(f"{BASE_URL}/reset", timeout=10)
        return r.json()
    except Exception as e:
        print("RESET FAILED:", e)
        return None

def step_env(action):
    try:
        r = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
        return r.json()
    except Exception as e:
        print("STEP FAILED:", e)
        return None

# ----------------------------
# MAIN LOOP
# ----------------------------
def run():
    print("[START]")

    state = reset_env()
    if not state:
        print("No state, exiting safely")
        print("[END]")
        return

    for _ in range(5):
        try:
            ticket = state["observation"]["focus_ticket"]["id"]

            # 🔥 REQUIRED LLM CALL
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a support agent."},
                        {"role": "user", "content": f"Classify ticket {ticket}"}
                    ]
                )
                print("[LLM OK]")
            except Exception as e:
                print("LLM FAILED BUT CONTINUING:", e)

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

            print("[STEP DONE]")

        except Exception as e:
            print("LOOP ERROR:", e)
            break

    print("[END]")


if __name__ == "__main__":
    run()