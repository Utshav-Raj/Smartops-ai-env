"""Deterministic SmartOps AI support simulator."""

from __future__ import annotations

import random
from typing import Any, Iterable
from uuid import uuid4

from smartops_ai_env.env.config import SmartOpsConfig
from smartops_ai_env.env.logging_utils import get_logger
from smartops_ai_env.env.scoring import (
    ACTION_TIME_COSTS,
    clamp,
    contains_prohibited,
    is_terminal,
    keyword_group_coverage,
    minutes_until_sla,
    priority_sort_key,
    squash_reward,
)
from smartops_ai_env.models import (
    ActionRecord,
    ActionType,
    MetricsSnapshot,
    QueueSummary,
    RewardComponents,
    SmartOpsAction,
    SmartOpsObservation,
    SupportState,
    SupportTicket,
    TaskDifficulty,
    TaskGrade,
    TicketPublicView,
    TicketReward,
    TicketStatus,
)
from smartops_ai_env.tasks import get_task, grade_task


class SmartOpsSimulator:
    """Stateful simulator implementing the SmartOps support workflow."""

    def __init__(self, config: SmartOpsConfig | None = None):
        self.config = config or SmartOpsConfig()
        self.logger = get_logger(level=self.config.log_level)
        self._rng = random.Random(self.config.default_seed)
        self._task = None
        self._state = SupportState(
            episode_id=str(uuid4()),
            step_count=0,
            active_task_id=None,
            task_difficulty=None,
            elapsed_minutes=0,
            max_steps=0,
            done=False,
            focus_ticket_id=None,
            tickets=[],
            action_history=[],
            customer_satisfaction=0.78,
            sla_adherence=1.0,
            deterministic_seed=self.config.default_seed,
            task_description="",
            expected_priority_order=[],
            metrics=MetricsSnapshot(),
        )

    def reset(
        self,
        task_id: str | None = None,
        difficulty: str | TaskDifficulty | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
        **_: Any,
    ) -> SmartOpsObservation:
        """Reset the simulator to a deterministic task scenario."""

        selected_task_id = task_id
        if selected_task_id is None and difficulty is not None:
            diff_value = TaskDifficulty(str(difficulty))
            for candidate in (
                "easy_duplicate_charge_refund",
                "medium_priority_queue_mix",
                "hard_account_takeover",
            ):
                task = get_task(candidate)
                if task.difficulty == diff_value:
                    selected_task_id = task.task_id
                    break

        if selected_task_id is None:
            selected_task_id = self.config.default_task_id

        task = get_task(selected_task_id)
        actual_seed = seed if seed is not None else task.seed
        self._rng.seed(actual_seed)
        self._task = task

        tickets = [ticket.model_copy(deep=True) for ticket in task.tickets]
        focus_ticket_id = task.priority_order[0] if task.priority_order else tickets[0].id
        episode_reference = episode_id or str(uuid4())

        self._state = SupportState(
            episode_id=episode_reference,
            step_count=0,
            active_task_id=task.task_id,
            task_difficulty=task.difficulty,
            elapsed_minutes=0,
            max_steps=task.max_steps,
            done=False,
            focus_ticket_id=focus_ticket_id,
            tickets=tickets,
            action_history=[],
            customer_satisfaction=0.78,
            sla_adherence=1.0,
            deterministic_seed=actual_seed,
            task_description=task.description,
            expected_priority_order=task.priority_order,
            metrics=self._build_metrics(tickets, 0.78),
        )
        self._state.terminal_score = None
        self._state.terminal_notes = []
        self._state.last_reward = None

        self.logger.info("Reset task=%s seed=%s", task.task_id, actual_seed)
        return self._build_observation(reward=None, done=False)

    def step(
        self,
        action: SmartOpsAction,
    ) -> tuple[SmartOpsObservation, TicketReward, bool, dict[str, Any]]:
        """Apply an action and return the OpenAI-style RL tuple."""

        if self._task is None:
            raise RuntimeError("reset() must be called before step()")

        if self._state.done:
            reward = TicketReward(
                raw_score=0.0,
                normalized_score=0.5,
                ticket_id=None,
                components=RewardComponents(),
                rationale=["Episode already finished. Reset the environment to begin a new run."],
            )
            observation = self._build_observation(reward=reward, done=True)
            return observation, reward, True, {"warning": "episode_already_done"}

        ticket = self._find_ticket(action.ticket_id)
        previous_priority = self._highest_priority_ticket_id()
        components = RewardComponents()
        rationale: list[str] = []

        if is_terminal(ticket):
            components.delay -= 0.2
            rationale.append(f"Ticket {ticket.id} is already terminal and should not be acted on again.")
        else:
            self._apply_action(ticket, action, components, rationale)

        action_cost = ACTION_TIME_COSTS[action.action_type.value]
        self._state.elapsed_minutes += action_cost
        self._state.step_count += 1

        self._apply_sla_adjustments(action, previous_priority, components, rationale)
        self._update_customer_satisfaction(components)
        self._refresh_metrics()

        done = self._is_done()
        if done:
            self._state.done = True
            unresolved = [item for item in self._state.tickets if not is_terminal(item)]
            if unresolved:
                components.terminal -= 1.0
                rationale.append(
                    "Episode ended with unresolved tickets still open or pending."
                )

        raw_score = components.total()
        normalized_score = squash_reward(raw_score)
        reward = TicketReward(
            raw_score=round(raw_score, 4),
            normalized_score=round(normalized_score, 4),
            ticket_id=ticket.id,
            components=components,
            rationale=rationale or ["Action applied without changing the support state materially."],
        )
        self._state.last_reward = reward

        record = ActionRecord(
            step_index=self._state.step_count,
            timestamp_minutes=self._state.elapsed_minutes,
            ticket_id=action.ticket_id,
            action_type=action.action_type,
            summary=self._summarize_action(action),
            raw_reward=reward.raw_score,
            normalized_reward=reward.normalized_score,
            outcome="; ".join(reward.rationale),
        )
        self._state.action_history.append(record)
        self._state.focus_ticket_id = self._choose_focus_ticket_id(last_ticket_id=ticket.id)

        if done and self._task is not None:
            grade = grade_task(self._task.task_id, self.state())
            self._state.terminal_score = grade.score
            self._state.terminal_notes = grade.notes

        observation = self._build_observation(reward=reward, done=done)
        info = {
            "task_id": self._state.active_task_id,
            "raw_reward": reward.raw_score,
            "normalized_reward": reward.normalized_score,
        }
        self.logger.info(
            "Step=%s action=%s ticket=%s reward=%.4f done=%s",
            self._state.step_count,
            action.action_type.value,
            ticket.id,
            reward.normalized_score,
            done,
        )
        return observation, reward, done, info

    def state(self) -> SupportState:
        """Return a deep copy of the full simulator state."""

        return self._state.model_copy(deep=True)

    def _apply_action(
        self,
        ticket: SupportTicket,
        action: SmartOpsAction,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        if action.action_type == ActionType.CLASSIFY_TICKET:
            self._handle_classification(ticket, action, components, rationale)
        elif action.action_type == ActionType.RESPOND_TO_TICKET:
            self._handle_response(ticket, action, components, rationale)
        elif action.action_type == ActionType.REQUEST_MORE_INFO:
            self._handle_info_request(ticket, action, components, rationale)
        elif action.action_type == ActionType.ESCALATE_TICKET:
            self._handle_escalation(ticket, action, components, rationale)
        elif action.action_type == ActionType.RESOLVE_TICKET:
            self._handle_resolution(ticket, components, rationale)

        ticket.last_updated_minutes = self._state.elapsed_minutes
        ticket.handled = True

    def _handle_classification(
        self,
        ticket: SupportTicket,
        action: SmartOpsAction,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        assert action.category is not None
        ticket.predicted_category = action.category
        if action.category == ticket.category:
            components.classification += 0.2
            rationale.append(f"Correctly classified ticket {ticket.id} as {ticket.category.value}.")
        else:
            components.classification -= 0.3
            rationale.append(
                f"Incorrect classification for {ticket.id}; expected {ticket.category.value}."
            )

    def _handle_response(
        self,
        ticket: SupportTicket,
        action: SmartOpsAction,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        assert action.message is not None
        coverage, complete, missing = keyword_group_coverage(
            action.message,
            ticket.expected_response_groups,
        )
        prohibited_hits = contains_prohibited(action.message, ticket.prohibited_response_terms)

        if prohibited_hits:
            components.response -= 0.5
            rationale.append(
                f"Response contained prohibited language: {', '.join(sorted(prohibited_hits))}."
            )
            return

        if complete:
            components.response += 0.3
            ticket.helpful_response_sent = True
            rationale.append(f"Helpful response sent for ticket {ticket.id}.")
        elif coverage >= 0.5:
            partial_reward = round(0.3 * coverage, 4)
            components.response += partial_reward
            rationale.append(
                f"Response partially addressed ticket {ticket.id}; missing groups: {missing}."
            )
        else:
            components.response -= 0.5
            rationale.append(
                f"Response to {ticket.id} was too generic or off-policy for the customer need."
            )

    def _handle_info_request(
        self,
        ticket: SupportTicket,
        action: SmartOpsAction,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        assert action.question is not None
        coverage, complete, missing = keyword_group_coverage(
            action.question,
            ticket.expected_info_request_groups,
        )

        if not ticket.info_request_required:
            components.request_more_info -= 0.2
            ticket.status = TicketStatus.PENDING
            rationale.append(
                f"Ticket {ticket.id} had enough information already; the extra question slowed resolution."
            )
            return

        if coverage == 0:
            components.request_more_info -= 0.2
            rationale.append(
                f"Question for {ticket.id} did not request the missing diagnostic details."
            )
            return

        components.request_more_info += round(0.25 * coverage, 4)
        ticket.info_requested = True
        ticket.status = TicketStatus.PENDING

        if complete and ticket.follow_up_customer_message and not ticket.follow_up_revealed:
            ticket.user_message = (
                f"{ticket.user_message}\n\nCustomer follow-up: {ticket.follow_up_customer_message}"
            )
            ticket.follow_up_revealed = True
            rationale.append(
                f"Requested the right details and received a deterministic follow-up from {ticket.id}."
            )
        else:
            rationale.append(
                f"Requested more information for {ticket.id}; still missing groups: {missing}."
            )

    def _handle_escalation(
        self,
        ticket: SupportTicket,
        action: SmartOpsAction,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        assert action.reason is not None
        coverage, complete, missing = keyword_group_coverage(
            action.reason,
            ticket.expected_escalation_groups,
        )

        if not ticket.escalation_required:
            components.escalation -= 0.2
            ticket.status = TicketStatus.ESCALATED
            rationale.append(
                f"Ticket {ticket.id} did not need specialist escalation, so the handoff was unnecessary."
            )
            return

        base_reward = 0.25 + (0.15 * coverage)
        components.escalation += round(base_reward, 4)
        ticket.escalation_sent = True
        ticket.status = TicketStatus.ESCALATED
        if not ticket.info_requested and ticket.expected_info_request_groups:
            components.escalation -= 0.1
            rationale.append(
                f"Escalation for {ticket.id} missed key customer verification details before handoff."
            )
        if complete:
            rationale.append(f"Escalation reason for {ticket.id} captured the risk signals clearly.")
        else:
            rationale.append(
                f"Escalation for {ticket.id} was sent with partial context; missing groups: {missing}."
            )

    def _handle_resolution(
        self,
        ticket: SupportTicket,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        classification_ok = ticket.predicted_category == ticket.category
        response_ok = ticket.helpful_response_sent
        info_ok = (not ticket.info_request_required) or ticket.info_requested

        if ticket.escalation_required:
            components.resolution -= 0.5
            rationale.append(
                f"Ticket {ticket.id} required escalation, so direct resolution was premature."
            )
            return

        if classification_ok and response_ok and info_ok:
            ticket.status = TicketStatus.RESOLVED
            ticket.resolution_complete = True
            components.resolution += 0.5
            rationale.append(f"Ticket {ticket.id} was resolved successfully.")
        else:
            components.resolution -= 0.5
            missing_items = []
            if not classification_ok:
                missing_items.append("correct classification")
            if not response_ok:
                missing_items.append("helpful response")
            if not info_ok:
                missing_items.append("missing customer details")
            rationale.append(
                f"Attempted to resolve {ticket.id} before prerequisites were met: {', '.join(missing_items)}."
            )

    def _apply_sla_adjustments(
        self,
        action: SmartOpsAction,
        previous_priority: str | None,
        components: RewardComponents,
        rationale: list[str],
    ) -> None:
        overdue = [
            ticket
            for ticket in self._state.tickets
            if not is_terminal(ticket) and minutes_until_sla(ticket, self._state.elapsed_minutes) < 0
        ]
        if overdue:
            components.delay -= 0.2
            rationale.append(
                f"SLA breach risk increased; overdue tickets: {', '.join(ticket.id for ticket in overdue)}."
            )
            return

        if previous_priority is not None and action.ticket_id == previous_priority:
            components.sla += 0.1
            rationale.append("Handled the highest-priority active ticket before an SLA breach.")
        elif previous_priority is not None:
            priority_ticket = self._find_ticket(previous_priority)
            if minutes_until_sla(priority_ticket, self._state.elapsed_minutes) <= 20:
                components.delay -= 0.2
                rationale.append(
                    f"Worked a lower-priority ticket while {previous_priority} remained close to its SLA."
                )

    def _update_customer_satisfaction(self, components: RewardComponents) -> None:
        delta = (
            components.classification
            + components.response
            + components.resolution
            + components.escalation
            + components.request_more_info
            + components.sla
            + components.delay
        ) * 0.12
        self._state.customer_satisfaction = round(
            clamp(self._state.customer_satisfaction + delta),
            4,
        )

    def _refresh_metrics(self) -> None:
        self._state.sla_adherence = round(self._calculate_sla_adherence(), 4)
        self._state.metrics = self._build_metrics(
            self._state.tickets,
            self._state.customer_satisfaction,
        )

    def _build_metrics(
        self,
        tickets: Iterable[SupportTicket],
        satisfaction: float,
    ) -> MetricsSnapshot:
        ticket_list = list(tickets)
        resolved = sum(1 for ticket in ticket_list if ticket.status == TicketStatus.RESOLVED)
        escalated = sum(1 for ticket in ticket_list if ticket.status == TicketStatus.ESCALATED)
        pending = sum(1 for ticket in ticket_list if ticket.status == TicketStatus.PENDING)
        open_count = sum(1 for ticket in ticket_list if ticket.status == TicketStatus.OPEN)
        backlog = sum(1 for ticket in ticket_list if not is_terminal(ticket))
        return MetricsSnapshot(
            customer_satisfaction=round(satisfaction, 4),
            sla_adherence=round(self._calculate_sla_adherence(), 4),
            resolved_count=resolved,
            escalated_count=escalated,
            open_count=open_count,
            pending_count=pending,
            backlog_size=backlog,
        )

    def _calculate_sla_adherence(self) -> float:
        if not self._state.tickets:
            return 1.0
        on_time = sum(
            1
            for ticket in self._state.tickets
            if minutes_until_sla(ticket, self._state.elapsed_minutes) >= 0
        )
        return on_time / len(self._state.tickets)

    def _choose_focus_ticket_id(self, last_ticket_id: str) -> str | None:
        non_terminal = [ticket for ticket in self._state.tickets if not is_terminal(ticket)]
        if not non_terminal:
            return None
        non_terminal.sort(key=lambda ticket: priority_sort_key(ticket, self._state.elapsed_minutes))
        if any(ticket.id == last_ticket_id for ticket in non_terminal):
            return last_ticket_id
        return non_terminal[0].id

    def _highest_priority_ticket_id(self) -> str | None:
        non_terminal = [ticket for ticket in self._state.tickets if not is_terminal(ticket)]
        if not non_terminal:
            return None
        non_terminal.sort(key=lambda ticket: priority_sort_key(ticket, self._state.elapsed_minutes))
        return non_terminal[0].id

    def _is_done(self) -> bool:
        all_terminal = all(is_terminal(ticket) for ticket in self._state.tickets)
        out_of_steps = self._state.step_count >= self._state.max_steps
        return all_terminal or out_of_steps

    def _build_observation(
        self,
        reward: TicketReward | None,
        done: bool,
    ) -> SmartOpsObservation:
        focus_ticket = None
        if self._state.focus_ticket_id is not None:
            focus_ticket = self._public_view(self._find_ticket(self._state.focus_ticket_id))

        queue_summary = self._build_queue_summary()
        observation = SmartOpsObservation(
            task_id=self._state.active_task_id or "",
            task_difficulty=self._state.task_difficulty or TaskDifficulty.EASY,
            focus_ticket=focus_ticket,
            queue_summary=queue_summary,
            elapsed_minutes=self._state.elapsed_minutes,
            previous_actions=self._state.action_history[-self.config.max_history_in_observation :],
            system_metrics=self._state.metrics or MetricsSnapshot(),
            workflow_hint=self._workflow_hint(queue_summary),
            available_ticket_ids=[ticket.id for ticket in self._state.tickets if not is_terminal(ticket)],
            last_reward=reward,
            reward=reward.normalized_score if reward is not None else 0.0,
            done=done,
            metadata={
                "task_description": self._state.task_description,
                "expected_priority_order": self._state.expected_priority_order,
            },
        )
        return observation

    def _build_queue_summary(self) -> QueueSummary:
        tickets = self._state.tickets
        open_count = sum(1 for ticket in tickets if ticket.status == TicketStatus.OPEN)
        pending_count = sum(1 for ticket in tickets if ticket.status == TicketStatus.PENDING)
        resolved_count = sum(1 for ticket in tickets if ticket.status == TicketStatus.RESOLVED)
        escalated_count = sum(1 for ticket in tickets if ticket.status == TicketStatus.ESCALATED)

        category_counts: dict[str, int] = {}
        urgency_counts: dict[str, int] = {}
        for ticket in tickets:
            category_counts[ticket.category.value] = category_counts.get(ticket.category.value, 0) + 1
            urgency_counts[ticket.urgency.value] = urgency_counts.get(ticket.urgency.value, 0) + 1

        non_terminal = [ticket for ticket in tickets if not is_terminal(ticket)]
        next_sla_ticket_id = None
        next_sla_minutes = None
        ordered_ids: list[str] = []
        if non_terminal:
            non_terminal.sort(key=lambda ticket: priority_sort_key(ticket, self._state.elapsed_minutes))
            next_sla_ticket_id = non_terminal[0].id
            next_sla_minutes = minutes_until_sla(non_terminal[0], self._state.elapsed_minutes)
            ordered_ids = [ticket.id for ticket in non_terminal]

        return QueueSummary(
            total_open=open_count,
            total_pending=pending_count,
            total_resolved=resolved_count,
            total_escalated=escalated_count,
            backlog_ids=ordered_ids,
            by_category=category_counts,
            by_urgency=urgency_counts,
            next_sla_ticket_id=next_sla_ticket_id,
            next_sla_minutes=next_sla_minutes,
        )

    def _workflow_hint(self, queue_summary: QueueSummary) -> str:
        focus_hint = (
            f"Recommended next ticket: {queue_summary.next_sla_ticket_id}"
            if queue_summary.next_sla_ticket_id
            else "All tickets are already in a terminal state."
        )
        return (
            f"{self._state.task_description} "
            "Keep replies concise, customer-safe, and policy compliant. "
            "Classify first, then either respond, request more info, escalate, or resolve based on ticket context. "
            f"{focus_hint}"
        )

    def _public_view(self, ticket: SupportTicket) -> TicketPublicView:
        return TicketPublicView(
            id=ticket.id,
            subject=ticket.subject,
            user_message=ticket.user_message,
            urgency=ticket.urgency,
            sentiment=ticket.sentiment,
            status=ticket.status,
            predicted_category=ticket.predicted_category,
            minutes_until_sla=minutes_until_sla(ticket, self._state.elapsed_minutes),
            helpful_response_sent=ticket.helpful_response_sent,
            info_requested=ticket.info_requested,
            escalation_sent=ticket.escalation_sent,
            follow_up_revealed=ticket.follow_up_revealed,
        )

    def _summarize_action(self, action: SmartOpsAction) -> str:
        details = {
            ActionType.CLASSIFY_TICKET: f"classify as {action.category.value if action.category else 'unknown'}",
            ActionType.RESPOND_TO_TICKET: f"respond: {action.message or ''}",
            ActionType.ESCALATE_TICKET: f"escalate: {action.reason or ''}",
            ActionType.RESOLVE_TICKET: "resolve ticket",
            ActionType.REQUEST_MORE_INFO: f"ask: {action.question or ''}",
        }
        return details[action.action_type]

    def _find_ticket(self, ticket_id: str) -> SupportTicket:
        for ticket in self._state.tickets:
            if ticket.id == ticket_id:
                return ticket
        raise ValueError(f"Unknown ticket_id: {ticket_id}")
