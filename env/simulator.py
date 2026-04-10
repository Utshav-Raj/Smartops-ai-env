"""Core simulation engine for SmartOps AI."""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple
from ..models.openenv import (
    ActionRecord, ActionType, MetricsSnapshot, QueueSummary,
    SmartOpsAction, SmartOpsObservation, SupportState, SupportTicket,
    TicketCategory, TicketPublicView, TicketStatus,
)
from ..tasks.catalog import get_task
from .config import SmartOpsConfig

WORKFLOW_HINTS = {
    ActionType.classify_ticket: "Classification helps route the ticket correctly.",
    ActionType.respond_to_ticket: "Response sent. Consider resolving or escalating next.",
    ActionType.escalate_ticket: "Escalated to specialist.",
    ActionType.resolve_ticket: "Ticket resolved. Move to the next one.",
    ActionType.request_more_info: "Info requested.",
}

class SmartOpsSimulator:
    def __init__(self, config: SmartOpsConfig):
        self._config = config
        self._state: Optional[SupportState] = None
        self._step_count: int = 0
        self._task_id: str = "easy_duplicate_charge_refund"

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        if task_id is None:
            task_id = self._task_id
        self._task_id = task_id
        scenario = get_task(task_id)
        self._state = SupportState(
            task_id=task_id,
            task_difficulty=scenario.difficulty.value,
            tickets=copy.deepcopy(scenario.tickets),
            action_history=[],
            elapsed_minutes=0,
            metrics=MetricsSnapshot(csat_score=1.0),
            done=False,
        )
        self._step_count = 0
        return self._build_observation(last_reward=None).model_dump()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()
        assert self._state is not None
        self._step_count += 1
        self._state.elapsed_minutes += 5
        if isinstance(action, SmartOpsAction):
            act = action
        elif isinstance(action, dict):
            act = SmartOpsAction(**action)
        else:
            act = SmartOpsAction(
                action_type=getattr(action, "action_type", "resolve_ticket"),
                ticket_id=getattr(action, "ticket_id", ""),
                category=getattr(action, "category", None),
                message=getattr(action, "message", None),
                reason=getattr(action, "reason", None),
                question=getattr(action, "question", None),
            )
        reward = self._apply_action(act)
        done = self._check_done()
        self._state.done = done
        obs = self._build_observation(last_reward=reward)
        return obs.model_dump(), reward, done, {}

    def get_state(self) -> Dict[str, Any]:
        if self._state is None:
            return {}
        return self._state.model_dump()

    def _get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        for t in self._state.tickets:
            if t.id == ticket_id:
                return t
        return None

    def _apply_action(self, action: SmartOpsAction) -> float:
        state = self._state
        assert state is not None
        ticket = self._get_ticket(action.ticket_id)
        if ticket is None:
            return 0.0
        recent = [r for r in state.action_history[-4:]
                  if r.ticket_id == action.ticket_id and r.action_type == action.action_type.value]
        if len(recent) >= 2:
            return 0.0
        state.action_history.append(ActionRecord(
            action_type=action.action_type.value,
            ticket_id=action.ticket_id,
            detail=action.category or action.reason or action.question or action.message,
        ))
        if action.action_type == ActionType.classify_ticket:
            reward = self._do_classify(ticket, action.category)
        elif action.action_type == ActionType.respond_to_ticket:
            reward = self._do_respond(ticket, action.message)
        elif action.action_type == ActionType.escalate_ticket:
            reward = self._do_escalate(ticket, action.reason)
        elif action.action_type == ActionType.resolve_ticket:
            reward = self._do_resolve(ticket)
        elif action.action_type == ActionType.request_more_info:
            reward = self._do_request_info(ticket, action.question)
        else:
            reward = 0.0
        if ticket.minutes_until_sla <= 0 and not ticket.resolved:
            reward -= self._config.sla_breach_penalty
            state.metrics.sla_breach_count += 1
        return round(max(0.0, min(1.0, reward)), 4)

    def _do_classify(self, ticket: SupportTicket, category: Optional[TicketCategory]) -> float:
        if category is None:
            return 0.0
        ticket.predicted_category = category
        expected = ticket.context.get("expected_category")
        if expected and category.value == expected:
            return self._config.classify_weight
        return self._config.classify_weight * 0.3

    def _do_respond(self, ticket: SupportTicket, message: Optional[str]) -> float:
        if ticket.response_sent:
            return 0.0
        ticket.response_sent = True
        ticket.status = TicketStatus.pending
        reward = self._config.respond_weight
        if ticket.sentiment.value in ("angry", "frustrated") and message and len(message) > 20:
            reward += 0.05
        return reward

    def _do_escalate(self, ticket: SupportTicket, reason: Optional[str]) -> float:
        if ticket.escalated:
            return 0.0
        ticket.escalated = True
        ticket.status = TicketStatus.escalated
        self._state.metrics.escalation_count += 1
        expected = ticket.context.get("should_escalate", False)
        return self._config.escalate_weight if expected else self._config.escalate_weight * 0.3

    def _do_resolve(self, ticket: SupportTicket) -> float:
        if ticket.resolved:
            return 0.0
        if not ticket.response_sent and not ticket.escalated:
            return self._config.resolve_weight * 0.2
        ticket.resolved = True
        ticket.status = TicketStatus.resolved
        self._state.metrics.resolved_count += 1
        if ticket.sentiment.value in ("angry", "frustrated"):
            self._state.metrics.csat_score = max(0.0, self._state.metrics.csat_score - 0.05)
        return self._config.resolve_weight

    def _do_request_info(self, ticket: SupportTicket, question: Optional[str]) -> float:
        if ticket.info_requested:
            return 0.0
        ticket.info_requested = True
        return self._config.info_weight

    def _check_done(self) -> bool:
        assert self._state is not None
        if self._step_count >= self._config.max_steps:
            return True
        return all(t.resolved or t.escalated for t in self._state.tickets)

    def _focus_ticket(self) -> SupportTicket:
        assert self._state is not None
        open_tickets = [t for t in self._state.tickets if not t.resolved and not t.escalated]
        if not open_tickets:
            return self._state.tickets[0]
        return sorted(open_tickets, key=lambda t: t.minutes_until_sla)[0]

    def _build_observation(self, last_reward: Optional[float]) -> SmartOpsObservation:
        assert self._state is not None
        focus = self._focus_ticket()
        open_tickets = [t for t in self._state.tickets if not t.resolved and not t.escalated]
        hint = ""
        if self._state.action_history:
            last_act = ActionType(self._state.action_history[-1].action_type)
            hint = WORKFLOW_HINTS.get(last_act, "")
        return SmartOpsObservation(
            task_id=self._state.task_id,
            task_difficulty=self._state.task_difficulty,
            focus_ticket=TicketPublicView(
                id=focus.id, subject=focus.subject, user_message=focus.user_message,
                urgency=focus.urgency.value, sentiment=focus.sentiment.value,
                status=focus.status.value,
                predicted_category=focus.predicted_category.value if focus.predicted_category else None,
                minutes_until_sla=focus.minutes_until_sla,
            ),
            queue_summary=QueueSummary(
                total_open=len(open_tickets),
                backlog_ids=[t.id for t in open_tickets],
            ),
            elapsed_minutes=self._state.elapsed_minutes,
            previous_actions=list(self._state.action_history[-5:]),
            system_metrics=self._state.metrics,
            workflow_hint=hint,
            last_reward=last_reward,
            done=self._state.done,
            reward=last_reward,
        )
