from dataclasses import dataclass

@dataclass
class SmartOpsConfig:
    max_steps: int = 20
    sla_window_minutes: int = 120
    classify_weight: float = 0.15
    respond_weight: float = 0.20
    resolve_weight: float = 0.35
    escalate_weight: float = 0.20
    info_weight: float = 0.10
    wrong_category_penalty: float = 0.10
    sla_breach_penalty: float = 0.15
    loop_penalty: float = 0.05
