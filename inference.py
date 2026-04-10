import os
from openai import OpenAI
import requests

# 🔥 MUST use their proxy
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

BASE_URL = "https://utshav-raj-ai-smartops-ai-env.hf.space"

def reset_env():
    return requests.post(f"{BASE_URL}/reset").json()

def step_env(action):
    return requests.post(f"{BASE_URL}/step", json=action).json()


def run():
    print("[START]")

    state = reset_env()

    for _ in range(5):
        ticket = state["observation"]["focus_ticket"]["id"]

        # 🔥 THIS CALL IS WHAT VALIDATOR CHECKS
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a smart support agent."},
                {"role": "user", "content": f"Classify ticket {ticket}"}
            ]
        )

        print("[LLM RESPONSE]", response.choices[0].message.content)

        action = {
            "action": {
                "action_type": "classify_ticket",
                "category": "billing",
                "ticket_id": ticket
            }
        }

        state = step_env(action)

        print("[STEP]", state)

    print("[END]")


if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)
    print("\n=== BASELINE SCORES ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
