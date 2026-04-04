---
title: SmartOps AI Environment Server
emoji: 🛠️
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - customer-support
  - rl-environment
---

# SmartOps AI

SmartOps AI is a production-grade OpenEnv environment that simulates a modern customer support queue. An agent must classify incoming tickets, send safe and useful responses, gather missing details, escalate specialist incidents, resolve routine issues, and preserve customer satisfaction under SLA pressure.

## Real-World Motivation

Customer support is a high-value real-world agent problem because success depends on more than one correct label. Good agents must triage urgency, avoid unsafe language, choose the right workflow branch, and manage multi-ticket backlog pressure while keeping customers informed. This environment turns those operational trade-offs into a deterministic benchmark suitable for hackathon evaluation and RL experimentation.

## Architecture

```text
Agent / inference.py
        |
        v
SmartOpsAction (typed Pydantic action)
        |
        v
SmartOpsSimulator
  - queue management
  - SLA clock
  - reward shaping
  - deterministic customer follow-ups
  - action history
        |
        +--> SmartOpsObservation
        +--> SupportState
        +--> TicketReward
        |
        v
OpenEnv FastAPI server
  - /reset
  - /step
  - /state
  - /schema
  - /metadata
  - /health
  - /ws
```

## Action Space

`SmartOpsAction` supports five structured operations:

- `classify_ticket(ticket_id, category)`
- `respond_to_ticket(ticket_id, message)`
- `escalate_ticket(ticket_id, reason)`
- `resolve_ticket(ticket_id)`
- `request_more_info(ticket_id, question)`

Validation is strict. Each action requires only the fields relevant to its `action_type`, and invalid combinations are rejected by Pydantic.

## Observation Space

`SmartOpsObservation` includes:

- `focus_ticket`: the ticket currently highlighted for the agent
- `queue_summary`: queue counts, category/urgency breakdown, and next SLA target
- `elapsed_minutes`: simulated time since reset
- `previous_actions`: recent structured action history
- `system_metrics`: customer satisfaction, SLA adherence, resolution counts, backlog size
- `workflow_hint`: policy-safe operational guidance
- `last_reward`: structured reward breakdown for the previous action

`SupportState` returns the full internal simulator state, including hidden ground truth used by graders.

## Tasks

### Easy

`easy_duplicate_charge_refund`

- One billing ticket
- Clear duplicate-charge refund path
- Goal: classify as billing, send a refund-oriented response, resolve cleanly

### Medium

`medium_priority_queue_mix`

- Three concurrent tickets
- Mixed billing, technical, and delivery categories
- Goal: prioritize the delivery SLA, classify correctly, and resolve the full queue

### Hard

`hard_account_takeover`

- One ambiguous fraud incident with billing and access symptoms
- Deterministic follow-up appears only after the right info request
- Goal: classify as fraud, send a safety-first response, request missing details, and escalate correctly

## Reward Design

Each step emits a structured `TicketReward` and a normalized scalar reward in the `0.0..1.0` range.

- `+0.2` correct classification
- `+0.3` helpful response
- `+0.5` successful resolution
- `+0.1` SLA-safe prioritization
- `-0.3` wrong classification
- `-0.5` poor or harmful response
- `-0.2` delay / poor prioritization
- `-1.0` unresolved ticket at terminal episode end

The simulator normalizes raw reward with a deterministic squashing function so OpenEnv clients always receive a continuous value between `0.0` and `1.0`.

## Setup

### Local install

```bash
cd smartops_ai_env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Docker build

```bash
cd smartops_ai_env
docker build -t smartops-ai-env:latest .
```

### Docker run

```bash
docker run --rm -p 8000:8000 smartops-ai-env:latest
```

The running container exposes `reset()`, `step()`, `state()`, `schema`, and the rest of the OpenEnv-compatible endpoints on port `8000`.

## Running the Server

```bash
cd smartops_ai_env
python -m smartops_ai_env.server.app --host 0.0.0.0 --port 8000
```

## Running inference.py

`inference.py` uses the OpenAI client when the following environment variables are set:

- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY`

If those are missing or the request fails, the script falls back to a deterministic heuristic policy so local testing still works.

```bash
cd smartops_ai_env
python inference.py
```

The script prints strict logs in this format:

```text
[START]
task_id=...
[STEP]
action=...
reward=...
[END]
final_score=...
```

## Baseline Scores

Reference scores from the built-in deterministic fallback policy:

| Task | Score |
| --- | ---: |
| `easy_duplicate_charge_refund` | `1.00` |
| `medium_priority_queue_mix` | `1.00` |
| `hard_account_takeover` | `1.00` |

## Validation

Run local validation from the environment root:

```bash
cd smartops_ai_env
openenv validate
```

You can also validate a running server:

```bash
openenv validate --url http://localhost:8000
```

## Project Structure

```text
smartops_ai_env/
├── openenv.yaml
├── env/
├── models/
├── tasks/
├── server/
├── inference.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```
