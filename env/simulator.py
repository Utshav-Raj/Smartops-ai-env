from __future__ import annotations
import copy
from typing import Any, Dict, Optional, Tuple
from models.openenv import (
    ActionRecord, ActionType, MetricsSnapshot, QueueSummary,
    SmartOpsAction, SmartOpsObservation, SupportState, SupportTicket,
    TicketCategory, TicketPublicView, TicketStatus,
)
from tasks.catalog import get_task
from .config import SmartOpsConfig

HINTS = {
    "classify_ticket": "Classification helps route the ticket.",
    "respond_to_ticket": "Response sent. Resolve or escalate next.",
    "escalate_ticket": "Escalated to specialist.",
    "resolve_ticket": "Ticket resolved.",
    "request_more_info": "Info requested.",
}

class SmartOpsSimulator:
    def __init__(self, config: SmartOpsConfig):
        self._config = config
        self._state: Optional[SupportState] = None
        self._step_count: int = 0
        self._task_id: str = "easy_duplicate_charge_refund"

    def reset(self, scenario=None, task_id: Optional[str] = None) -> SmartOpsObservation:
        if scenario is not None:
            self._task_id = scenario.task_id
        else:
            if task_id is None:
                task_id = self._task_id
            self._task_id = task_id
            scenario = get_task(task_id)
        self._state = SupportState(
            task_id=self._task_id,
            task_difficulty=scenario.difficulty.value,
            tickets=copy.deepcopy(scenario.tickets),
            action_history=[],
            elapsed_minutes=0,
            metrics=MetricsSnapshot(csat_score=1.0),
            done=False,
        )
        self._step_count = 0
        return self._build_obs(None)

    def step(self, action: Any) -> Tuple[SmartOpsObservation, float, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()
        self._step_count += 1
        self._state.elapsed_minutes += 5
        if isinstance(action, dict):
            act = SmartOpsAction(**action)
        elif isinstance(action, SmartOpsAction):
            act = action
        else:
            act = SmartOpsAction(
                action_type=getattr(action, "action_type", "resolve_ticket"),
                ticket_id=getattr(action, "ticket_id", ""),
                category=getattr(action, "category", None),
                message=getattr(action, "message", None),
                reason=getattr(action, "reason", None),
                question=getattr(action, "question", None),
            )
        reward = self._apply(act)
        done = self._step_count >= self._config.max_steps or all(
            t.resolved or t.escalated for t in self._state.tickets
        )
        self._state.done = done
        return self._build_obs(reward), reward, done, {}

    def get_state(self) -> Dict[str, Any]:
        if self._state is None:
            return {}
        return self._state.model_dump(mode="json")

    def _ticket(self, tid: str) -> Optional[SupportTicket]:
        for t in self._state.tickets:
            if t.id == tid:
                return t
        return None

    def _apply(self, action: SmartOpsAction) -> float:
        ticket = self._ticket(action.ticket_id)
        if ticket is None:
            return 0.01
        recent = [r for r in self._state.action_history[-4:]
                  if r.ticket_id == action.ticket_id and r.action_type == action.action_type.value]
        if len(recent) >= 2:
            return 0.01
        self._state.action_history.append(ActionRecord(
            action_type=action.action_type.value,
            ticket_id=action.ticket_id,
            detail=action.category or action.reason or action.question or action.message,
        ))
        at = action.action_type
        if at == ActionType.classify_ticket:
            r = self._classify(ticket, action.category)
        elif at == ActionType.respond_to_ticket:
            r = self._respond(ticket, action.message)
        elif at == ActionType.escalate_ticket:
            r = self._escalate(ticket)
        elif at == ActionType.resolve_ticket:
            r = self._resolve(ticket)
        elif at == ActionType.request_more_info:
            r = self._info(ticket)
        else:
            r = 0.0
        if ticket.minutes_until_sla <= 0 and not ticket.resolved:
            r -= self._config.sla_breach_penalty
            self._state.metrics.sla_breach_count += 1
        # Clamp strictly inside (0, 1) — boundary values fail OpenEnv score validation
        return round(max(0.01, min(0.99, r)), 4)

    def _classify(self, t: SupportTicket, cat: Optional[TicketCategory]) -> float:
        if cat is None:
            return 0.0
        t.predicted_category = cat
        exp = t.context.get("expected_category")
        return self._config.classify_weight if (exp and cat.value == exp) else self._config.classify_weight * 0.3

    def _respond(self, t: SupportTicket, msg: Optional[str]) -> float:
        if t.response_sent:
            return 0.0
        t.response_sent = True
        t.status = TicketStatus.pending
        r = self._config.respond_weight
        if t.sentiment.value in ("angry", "frustrated") and msg and len(msg) > 20:
            r += 0.05
        return r

    def _escalate(self, t: SupportTicket) -> float:
        if t.escalated:
            return 0.0
        t.escalated = True
        t.status = TicketStatus.escalated
        self._state.metrics.escalation_count += 1
        exp = t.context.get("should_escalate", False)
        return self._config.escalate_weight if exp else self._config.escalate_weight * 0.3

    def _resolve(self, t: SupportTicket) -> float:
        if t.resolved:
            return 0.0
        if not t.response_sent and not t.escalated:
            return self._config.resolve_weight * 0.2
        t.resolved = True
        t.status = TicketStatus.resolved
        self._state.metrics.resolved_count += 1
        if t.sentiment.value in ("angry", "frustrated"):
            self._state.metrics.csat_score = max(0.0, self._state.metrics.csat_score - 0.05)
        return self._config.resolve_weight

    def _info(self, t: SupportTicket) -> float:
        if t.info_requested:
            return 0.0
        t.info_requested = True
        return self._config.info_weight

    def _focus(self) -> SupportTicket:
        open_t = [t for t in self._state.tickets if not t.resolved and not t.escalated]
        if not open_t:
            return self._state.tickets[0]
        return sorted(open_t, key=lambda t: t.minutes_until_sla)[0]

    def _build_obs(self, reward: Optional[float]) -> SmartOpsObservation:
        focus = self._focus()
        open_t = [t for t in self._state.tickets if not t.resolved and not t.escalated]
        hint = ""
        if self._state.action_history:
            hint = HINTS.get(self._state.action_history[-1].action_type, "")
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
            queue_summary=QueueSummary(total_open=len(open_t), backlog_ids=[t.id for t in open_t]),
            elapsed_minutes=self._state.elapsed_minutes,
            previous_actions=list(self._state.action_history[-5:]),
            system_metrics=self._state.metrics,
            workflow_hint=hint,
            last_reward=reward,
            done=self._state.done,
            reward=reward,
        )
