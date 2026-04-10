import os, requests
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"))
BASE_URL = "https://utshav-raj-ai-smartops-ai-env.hf.space"

TASKS = ["easy_duplicate_charge_refund", "medium_priority_queue_mix", "hard_account_takeover"]

def reset_env(task_id):
    return requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}).json()

def step_env(action):
    return requests.post(f"{BASE_URL}/step", json={"action": action}).json()

def run_task(task_id):
    print(f"\n=== Task: {task_id} ===")
    state = reset_env(task_id)
    obs = state.get("observation", state)
    total_reward = 0.0
    for step in range(10):
        ticket_id = obs.get("focus_ticket", {}).get("id", "B-1001")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert customer support agent. Respond with ONE action as JSON only. Fields: action_type (classify_ticket|respond_to_ticket|escalate_ticket|resolve_ticket|request_more_info), ticket_id, and optionally: category, message, reason, question."},
                {"role": "user", "content": f"Current observation: {obs}\nWhat is your next action?"}
            ]
        )
        import json
        try:
            action = json.loads(response.choices[0].message.content)
        except Exception:
            action = {"action_type": "resolve_ticket", "ticket_id": ticket_id}
        result = step_env(action)
        reward = result.get("reward", 0.0) or 0.0
        total_reward += reward
        obs = result.get("observation", result)
        print(f"  Step {step+1}: action={action.get('action_type')} reward={reward:.3f}")
        if result.get("done"):
            break
    print(f"  Total reward: {total_reward:.4f}")
    return total_reward

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)
    print("\n=== BASELINE SCORES ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
