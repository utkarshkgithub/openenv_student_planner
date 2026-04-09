from __future__ import annotations

import statistics
from typing import Dict

from .models import GradeBreakdown, StudentPlannerState, TaskConfig

SCORE_EPSILON = 1e-6


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _clamp_open_unit_interval(value: float) -> float:
    return _clamp(value, SCORE_EPSILON, 1.0 - SCORE_EPSILON)


def exam_score(mastery: Dict[str, float], weights: Dict[str, float]) -> float:
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        return 0.0
    weighted = sum(mastery[topic] * weight for topic, weight in weights.items())
    return _clamp(weighted / weight_sum)


def coverage_score(mastery: Dict[str, float], threshold: float) -> float:
    if not mastery:
        return 0.0
    covered = sum(1 for value in mastery.values() if value >= threshold)
    return _clamp(covered / float(len(mastery)))


def balance_score(mastery: Dict[str, float]) -> float:
    values = list(mastery.values())
    if len(values) <= 1:
        return 1.0
    spread = statistics.pstdev(values)
    return _clamp(1.0 - (spread / 0.5))


def efficiency_score(time_budget: float, time_left: float) -> float:
    if time_budget <= 0.0:
        return 0.0
    usage = _clamp((time_budget - max(0.0, time_left)) / time_budget)
    target = 0.85
    return _clamp(1.0 - abs(usage - target) / target)


def grade_state(state: StudentPlannerState, task: TaskConfig) -> GradeBreakdown:
    exam = exam_score(state.mastery, state.topic_weights)
    coverage = coverage_score(state.mastery, task.coverage_threshold)
    balance = balance_score(state.mastery)
    efficiency = efficiency_score(state.time_budget, state.time_left)
    fatigue = _clamp(1.0 - state.fatigue)
    invalid_penalty = min(0.5, state.invalid_action_count * 0.02)

    final = _clamp_open_unit_interval(
        0.50 * exam
        + 0.20 * coverage
        + 0.15 * balance
        + 0.10 * efficiency
        + 0.05 * fatigue
        - invalid_penalty
    )

    success = exam >= task.success_exam_threshold and coverage >= task.success_coverage_threshold

    return GradeBreakdown(
        exam_score=exam,
        coverage_score=coverage,
        balance_score=balance,
        efficiency_score=efficiency,
        fatigue_score=fatigue,
        invalid_penalty=invalid_penalty,
        final_score=final,
        success=success,
    )
