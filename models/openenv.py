"""All Pydantic models for the SmartOps AI OpenEnv environment."""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TicketCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    delivery = "delivery"
    fraud = "fraud"
    general = "general"

class TicketStatus(str, Enum):
    open = "open"
    pending = "pending"
    escalated = "escalated"
    resolved = "resolved"

class TicketUrgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class TicketSentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    frustrated = "frustrated"
    angry = "angry"

class ActionType(str, Enum):
    classify_ticket = "classify_ticket"
    respond_to_ticket = "respond_to_ticket"
    escalate_ticket = "escalate_ticket"
    resolve_ticket = "resolve_ticket"
    request_more_info = "request_more_info"

class TaskDifficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"

class SupportTicket(BaseModel):
    id: str
    subject: str
    user_message: str
    urgency: TicketUrgency
    sentiment: TicketSentiment
    status: TicketStatus = TicketStatus.open
    category: Optional[TicketCategory] = None
    predicted_category: Optional[TicketCategory] = None
    minutes_until_sla: int = 120
    response_sent: bool = False
    escalated: bool = False
    resolved: bool = False
    info_requested: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)

class TicketPublicView(BaseModel):
    id: str
    subject: str
    user_message: str
    urgency: str
    sentiment: str
    status: str
    predicted_category: Optional[str] = None
    minutes_until_sla: int

class QueueSummary(BaseModel):
    total_open: int
    backlog_ids: List[str]

class MetricsSnapshot(BaseModel):
    csat_score: float = Field(ge=0.0, le=1.0)
    sla_breach_count: int = 0
    resolved_count: int = 0
    escalation_count: int = 0

class ActionRecord(BaseModel):
    action_type: str
    ticket_id: str
    detail: Optional[str] = None

class TaskScenario(BaseModel):
    task_id: str
    difficulty: TaskDifficulty
    tickets: List[SupportTicket]
    description: str

class TaskGrade(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""

class SupportState(BaseModel):
    task_id: str
    task_difficulty: str
    tickets: List[SupportTicket] = Field(default_factory=list)
    action_history: List[ActionRecord] = Field(default_factory=list)
    elapsed_minutes: int = 0
    metrics: MetricsSnapshot = Field(default_factory=lambda: MetricsSnapshot(csat_score=1.0))
    done: bool = False

class SmartOpsAction(BaseModel):
    action_type: ActionType
    ticket_id: str
    category: Optional[TicketCategory] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    question: Optional[str] = None

class SmartOpsObservation(BaseModel):
    task_id: str
    task_difficulty: str
    focus_ticket: TicketPublicView
    queue_summary: QueueSummary
    elapsed_minutes: int = 0
    previous_actions: List[ActionRecord] = Field(default_factory=list)
    system_metrics: MetricsSnapshot = Field(default_factory=lambda: MetricsSnapshot(csat_score=1.0))
    workflow_hint: str = ""
    last_reward: Optional[float] = None
    done: Optional[bool] = False
    reward: Optional[float] = None

class TicketReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    reason: str = ""
