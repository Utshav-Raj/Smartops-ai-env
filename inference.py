"""Baseline inference runner for SmartOps AI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parent
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))

from smartops_ai_env.env import SmartOpsConfig, SmartOpsSimulator
from smartops_ai_env.models import (
    ActionType,
    SmartOpsAction,
    SmartOpsObservation,
    TaskGrade,
    TicketCategory,
)
from smartops_ai_env.tasks import get_task, grade_task, list_task_ids


class HeuristicPolicy:
    """Deterministic fallback policy used when LLM access is unavailable."""

    def select_action(self, observation: SmartOpsObservation) -> SmartOpsAction:
        ticket = observation.focus_ticket
        if ticket is None:
            raise RuntimeError("No focus ticket available.")

        message = ticket.user_message.lower()
        predicted = ticket.predicted_category

        if predicted is None:
            return SmartOpsAction(
                action_type=ActionType.CLASSIFY_TICKET,
                ticket_id=ticket.id,
                category=self._infer_category(message),
            )

        if predicted == TicketCategory.FRAUD:
            if not ticket.helpful_response_sent:
                return SmartOpsAction(
                    action_type=ActionType.RESPOND_TO_TICKET,
                    ticket_id=ticket.id,
                    message=(
                        "We will freeze the account immediately, open an investigation with the "
                        "fraud team, and start a password reset now."
                    ),
                )
            if not ticket.info_requested:
                return SmartOpsAction(
                    action_type=ActionType.REQUEST_MORE_INFO,
                    ticket_id=ticket.id,
                    question=(
                        "Please confirm the last legitimate invoice or authorized charge and the "
                        "original admin email before the change."
                    ),
                )
            if not ticket.escalation_sent:
                return SmartOpsAction(
                    action_type=ActionType.ESCALATE_TICKET,
                    ticket_id=ticket.id,
                    reason=(
                        "Escalating for unauthorized charges and an admin email change that "
                        "indicates possible account takeover."
                    ),
                )
            return SmartOpsAction(
                action_type=ActionType.ESCALATE_TICKET,
                ticket_id=ticket.id,
                reason="Reaffirm escalated fraud handoff for account takeover.",
            )

        if not ticket.helpful_response_sent:
            return SmartOpsAction(
                action_type=ActionType.RESPOND_TO_TICKET,
                ticket_id=ticket.id,
                message=self._response_for_ticket(predicted, message),
            )

        return SmartOpsAction(
            action_type=ActionType.RESOLVE_TICKET,
            ticket_id=ticket.id,
        )

    def _infer_category(self, message: str) -> TicketCategory:
        if any(token in message for token in ["fraud", "unauthorized", "admin email", "takeover"]):
            return TicketCategory.FRAUD
        if any(token in message for token in ["delivery", "shipment", "tracking", "package"]):
            return TicketCategory.DELIVERY
        if any(token in message for token in ["invoice", "charge", "refund", "vat", "card"]):
            return TicketCategory.BILLING
        return TicketCategory.TECHNICAL

    def _response_for_ticket(self, category: TicketCategory, message: str) -> str:
        if category == TicketCategory.BILLING and "charged" in message:
            return (
                "I have queued a refund for the duplicate charge and you should see the reversal "
                "within 3-5 business days."
            )
        if category == TicketCategory.BILLING and any(token in message for token in ["charge", "charges", "card"]):
            return (
                "I have queued a refund for the duplicate charge and you should see the reversal "
                "within 3-5 business days."
            )
        if category == TicketCategory.BILLING:
            return (
                "I will resend the corrected invoice PDF with the VAT number included so finance "
                "has the updated document."
            )
        if category == TicketCategory.DELIVERY:
            return (
                "I am opening a carrier trace now, arranging a priority replacement shipment, "
                "and can issue a refund if the replacement no longer helps."
            )
        if category == TicketCategory.TECHNICAL:
            return (
                "A temporary workaround is to export from the web dashboard while engineering "
                "ships a hotfix for the mobile issue."
            )
        return (
            "We will freeze the account immediately, open an investigation with the fraud team, "
            "and start a password reset now."
        )


class LLMBaselinePolicy:
    """Guardrailed OpenAI-backed policy with deterministic fallback."""

    def __init__(self) -> None:
        self.model_name = os.getenv("MODEL_NAME", "").strip()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("API_BASE_URL", "").strip() or None
        self.fallback = HeuristicPolicy()
        self.client: OpenAI | None = None
        if self.model_name and self.api_key:
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)

    def select_action(self, observation: SmartOpsObservation) -> SmartOpsAction:
        if self.client is None:
            return self.fallback.select_action(observation)

        ticket = observation.focus_ticket
        if ticket is None:
            return self.fallback.select_action(observation)

        prompt = {
            "task_id": observation.task_id,
            "difficulty": observation.task_difficulty.value,
            "elapsed_minutes": observation.elapsed_minutes,
            "focus_ticket": ticket.model_dump(),
            "queue_summary": observation.queue_summary.model_dump(),
            "previous_actions": [action.model_dump() for action in observation.previous_actions],
            "system_metrics": observation.system_metrics.model_dump(),
            "workflow_hint": observation.workflow_hint,
        }
        system = (
            "You are operating a customer support queue. "
            "Return one JSON object with keys: action_type, ticket_id, category, message, reason, question. "
            "Use only these action_type values: classify_ticket, respond_to_ticket, escalate_ticket, "
            "resolve_ticket, request_more_info. "
            "Choose exactly one action and omit irrelevant fields by setting them to null."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return SmartOpsAction.model_validate(payload)
        except Exception:
            return self.fallback.select_action(observation)


def run_task(task_id: str, policy: LLMBaselinePolicy) -> TaskGrade:
    simulator = SmartOpsSimulator(config=SmartOpsConfig(log_level="CRITICAL"))
    observation = simulator.reset(task_id=task_id)

    print("[START]")
    print(f"task_id={task_id}")

    while not observation.done:
        action = policy.select_action(observation)
        observation, reward, done, _ = simulator.step(action)
        print("[STEP]")
        print(
            "action="
            + json.dumps(
                action.model_dump(exclude_none=True),
                sort_keys=True,
            )
        )
        print(f"reward={reward.normalized_score:.4f}")
        if done:
            break

    grade = grade_task(task_id, simulator.state())
    print("[END]")
    print(f"final_score={grade.score:.4f}")
    return grade


def main() -> None:
    policy = LLMBaselinePolicy()
    for task_id in list_task_ids():
        run_task(task_id, policy)


if __name__ == "__main__":
    main()
