from __future__ import annotations

import random
from typing import Dict, Optional, Set, Tuple

from pydantic import ValidationError

from .grader import exam_score, grade_state
from .models import (
    StepResult,
    StudentPlannerAction,
    StudentPlannerObservation,
    StudentPlannerReward,
    StudentPlannerState,
)
from .tasks import DEFAULT_TASK_NAME, get_task

INVALID_ACTION_PENALTY = 0.10
REDUNDANCY_PENALTY = 0.02
PREREQ_PENALTY = 0.03
FATIGUE_PENALTY_SCALE = 0.04


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


class StudentPlannerCoreEnv:
    """Deterministic Student Planner environment implementing reset/step/state."""

    def __init__(self, task_name: Optional[str] = None) -> None:
        self._task = get_task(task_name or DEFAULT_TASK_NAME)
        self._state = self._make_initial_state(seed=None)
        self._last_action_error: Optional[str] = None

    def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> StepResult:
        if task_name is not None:
            self._task = get_task(task_name)
        self._state = self._make_initial_state(seed=seed)
        self._last_action_error = None
        return StepResult(
            observation=self._build_observation(),
            reward=0.0,
            done=False,
            info={"task_name": self._state.task_name, "seed": seed},
        )

    def step(self, action: StudentPlannerAction | Dict[str, object]) -> StepResult:
        if self._state.done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"reason": "episode_already_done"},
            )

        parsed_action: StudentPlannerAction
        parse_error: Optional[str] = None
        if isinstance(action, StudentPlannerAction):
            parsed_action = action
        else:
            try:
                parsed_action = StudentPlannerAction.model_validate(action)
            except ValidationError as exc:
                parse_error = self._one_line_error(str(exc))
                parsed_action = StudentPlannerAction(action_type="skip", duration=1.0)

        action_error = parse_error or self._validate_action(parsed_action)
        duration = self._resolve_duration(parsed_action)

        readiness_before = self._readiness()
        fatigue_before = self._state.fatigue

        redundancy_penalty = 0.0
        prereq_penalty = 0.0
        invalid_penalty = 0.0
        touched_topics: Set[str] = set()

        if action_error:
            self._state.invalid_action_count += 1
            invalid_penalty = INVALID_ACTION_PENALTY
            self._apply_fatigue_load("skip", duration)
        else:
            if parsed_action.action_type == "study":
                redundancy_penalty, prereq_penalty, touched_topics = self._apply_study(parsed_action, duration)
            elif parsed_action.action_type == "revise":
                redundancy_penalty, prereq_penalty, touched_topics = self._apply_revise(parsed_action, duration)
            elif parsed_action.action_type == "mock_test":
                touched_topics = self._apply_mock_test(parsed_action, duration)
            elif parsed_action.action_type == "rest":
                self._apply_rest(duration)
            elif parsed_action.action_type == "switch_topic":
                self._apply_switch_topic(parsed_action, duration)
            elif parsed_action.action_type == "skip":
                self._apply_fatigue_load("skip", duration)

        self._apply_forgetting(duration_minutes=duration, protected_topics=touched_topics)
        self._state.time_left = max(0.0, self._state.time_left - duration)
        self._state.step_count += 1

        readiness_after = self._readiness()
        readiness_delta = readiness_after - readiness_before
        fatigue_delta = max(0.0, self._state.fatigue - fatigue_before)
        fatigue_penalty = fatigue_delta * FATIGUE_PENALTY_SCALE
        terminal_bonus = 0.0

        reward = (
            readiness_delta
            - fatigue_penalty
            - redundancy_penalty
            - prereq_penalty
            - invalid_penalty
        )

        if self._state.time_left <= 0.0 or self._state.step_count >= self._state.max_steps:
            self._state.done = True

        if self._state.done:
            grade = grade_state(self._state, self._task)
            terminal_bonus = self._terminal_bonus(grade.exam_score)
            reward += terminal_bonus
        else:
            grade = None

        reward_breakdown = StudentPlannerReward(
            readiness_delta=round(readiness_delta, 6),
            fatigue_penalty=round(fatigue_penalty, 6),
            redundancy_penalty=round(redundancy_penalty, 6),
            prerequisite_penalty=round(prereq_penalty, 6),
            invalid_action_penalty=round(invalid_penalty, 6),
            terminal_bonus=round(terminal_bonus, 6),
            total_reward=round(reward, 6),
        )

        info = {
            "readiness_before": round(readiness_before, 6),
            "readiness_after": round(readiness_after, 6),
            "readiness_delta": round(readiness_delta, 6),
            "fatigue_before": round(fatigue_before, 6),
            "fatigue_after": round(self._state.fatigue, 6),
            "fatigue_delta": round(fatigue_delta, 6),
            "fatigue_penalty": round(fatigue_penalty, 6),
            "invalid_action_penalty": round(invalid_penalty, 6),
            "redundancy_penalty": round(redundancy_penalty, 6),
            "prerequisite_penalty": round(prereq_penalty, 6),
            "reward_breakdown": reward_breakdown.model_dump(mode="json"),
            "task_name": self._state.task_name,
        }

        if self._state.done:
            assert grade is not None
            info.update(
                {
                    "final_bonus": round(terminal_bonus, 6),
                    "normalized_score": grade.final_score,
                    "grade": grade.model_dump(mode="json"),
                    "success": grade.success,
                }
            )

        self._last_action_error = action_error

        return StepResult(
            observation=self._build_observation(),
            reward=float(round(reward, 6)),
            done=self._state.done,
            info=info,
        )

    def state(self) -> StudentPlannerState:
        return self._state.model_copy(deep=True)

    def _make_initial_state(self, seed: Optional[int]) -> StudentPlannerState:
        resolved_seed = self._task.default_seed if seed is None else seed
        rng = random.Random(resolved_seed)

        mastery = {}
        for topic in self._task.topics:
            base = self._task.initial_mastery[topic]
            jitter = rng.uniform(-self._task.initial_mastery_jitter, self._task.initial_mastery_jitter)
            mastery[topic] = _clamp(base + jitter)

        return StudentPlannerState(
            task_name=self._task.name,
            topics=list(self._task.topics),
            mastery=mastery,
            fatigue=_clamp(self._task.initial_fatigue),
            time_left=self._task.time_budget,
            time_budget=self._task.time_budget,
            step_count=0,
            max_steps=self._task.max_steps,
            topic_weights=dict(self._task.topic_weights),
            topic_difficulty=dict(self._task.topic_difficulty),
            forgetting_rate=dict(self._task.forgetting_rate),
            prerequisites={k: list(v) for k, v in self._task.prerequisites.items()},
            learner_speed=self._task.profile.learner_speed,
            fatigue_sensitivity=self._task.profile.fatigue_sensitivity,
            retention_strength=self._task.profile.retention_strength,
            prerequisite_mastery_threshold=self._task.prerequisite_mastery_threshold,
            invalid_action_count=0,
            done=False,
            current_topic=None,
        )

    def _build_observation(self) -> StudentPlannerObservation:
        return StudentPlannerObservation(
            task_name=self._state.task_name,
            mastery={k: round(v, 6) for k, v in self._state.mastery.items()},
            fatigue=round(self._state.fatigue, 6),
            time_left=round(self._state.time_left, 6),
            readiness=round(self._readiness(), 6),
            step_count=self._state.step_count,
            invalid_action_count=self._state.invalid_action_count,
            current_topic=self._state.current_topic,
            last_action_error=self._last_action_error,
        )

    def _readiness(self) -> float:
        return exam_score(self._state.mastery, self._state.topic_weights)

    def _validate_action(self, action: StudentPlannerAction) -> Optional[str]:
        if action.action_type in {"study", "revise", "switch_topic"} and not action.topic:
            return f"topic is required for action '{action.action_type}'"

        if action.topic and action.topic not in self._state.topics:
            return f"unknown topic '{action.topic}'"

        if action.action_type in {"study", "revise", "rest"}:
            if action.duration is None:
                return f"duration is required for action '{action.action_type}'"
            if action.duration <= 0.0:
                return "duration must be positive"

        if action.action_type == "mock_test":
            if action.duration is not None and action.duration <= 0.0:
                return "duration must be positive"
            if action.topics:
                unknown = [topic for topic in action.topics if topic not in self._state.topics]
                if unknown:
                    return f"unknown topics in mock_test: {unknown}"

        if action.action_type == "skip" and action.duration is not None and action.duration <= 0.0:
            return "duration must be positive"

        return None

    def _resolve_duration(self, action: StudentPlannerAction) -> float:
        defaults = {
            "study": 20.0,
            "revise": 15.0,
            "mock_test": 15.0,
            "rest": 15.0,
            "switch_topic": 2.0,
            "skip": 5.0,
        }
        value = action.duration if action.duration is not None else defaults[action.action_type]
        return max(0.5, float(value))

    def _prerequisites_satisfied(self, topic: str) -> bool:
        prereqs = self._state.prerequisites.get(topic, [])
        for prereq in prereqs:
            if self._state.mastery.get(prereq, 0.0) < self._state.prerequisite_mastery_threshold:
                return False
        return True

    def _apply_study(
        self,
        action: StudentPlannerAction,
        duration: float,
    ) -> Tuple[float, float, Set[str]]:
        topic = action.topic or ""
        self._state.current_topic = topic
        mastery_before = self._state.mastery[topic]

        prereq_ok = self._prerequisites_satisfied(topic)
        prereq_factor = 1.0 if prereq_ok else 0.5
        difficulty_factor = 1.0 - (0.5 * self._state.topic_difficulty[topic])
        fatigue_factor = max(0.05, 1.0 - self._state.fatigue)

        base_gain = (duration / 60.0) * 0.45 * self._state.learner_speed
        gain = base_gain * (1.0 - mastery_before) * fatigue_factor * difficulty_factor * prereq_factor
        self._state.mastery[topic] = _clamp(mastery_before + gain)

        redundancy_penalty = REDUNDANCY_PENALTY if mastery_before >= 0.90 else 0.0
        prereq_penalty = PREREQ_PENALTY if (not prereq_ok and self._state.prerequisites.get(topic)) else 0.0

        self._apply_fatigue_load("study", duration)
        return redundancy_penalty, prereq_penalty, {topic}

    def _apply_revise(
        self,
        action: StudentPlannerAction,
        duration: float,
    ) -> Tuple[float, float, Set[str]]:
        topic = action.topic or ""
        self._state.current_topic = topic
        mastery_before = self._state.mastery[topic]

        prereq_ok = self._prerequisites_satisfied(topic)
        prereq_factor = 1.0 if prereq_ok else 0.6
        fatigue_factor = max(0.05, 1.0 - self._state.fatigue)

        base_gain = (duration / 60.0) * 0.28 * self._state.retention_strength
        gain = base_gain * max(0.2, mastery_before) * fatigue_factor * prereq_factor
        self._state.mastery[topic] = _clamp(mastery_before + gain)

        redundancy_penalty = REDUNDANCY_PENALTY if mastery_before >= 0.92 else 0.0
        prereq_penalty = PREREQ_PENALTY if (not prereq_ok and self._state.prerequisites.get(topic)) else 0.0

        self._apply_fatigue_load("revise", duration)
        return redundancy_penalty, prereq_penalty, {topic}

    def _apply_mock_test(self, action: StudentPlannerAction, duration: float) -> Set[str]:
        topics = action.topics if action.topics else list(self._state.topics)
        if not topics:
            return set()

        per_topic_duration = duration / float(len(topics))
        touched = set()
        for topic in topics:
            mastery_before = self._state.mastery[topic]
            retrieval_gain = (
                (per_topic_duration / 60.0)
                * 0.10
                * (1.0 - mastery_before)
                * self._state.retention_strength
            )
            self._state.mastery[topic] = _clamp(mastery_before + retrieval_gain)
            touched.add(topic)

        self._apply_fatigue_load("mock_test", duration)
        return touched

    def _apply_rest(self, duration: float) -> None:
        recovery = (duration / 60.0) * 0.55 * (2.0 - self._state.fatigue_sensitivity)
        self._state.fatigue = _clamp(self._state.fatigue - recovery)

    def _apply_switch_topic(self, action: StudentPlannerAction, duration: float) -> None:
        self._state.current_topic = action.topic
        self._apply_fatigue_load("switch_topic", duration)

    def _apply_fatigue_load(self, action_type: str, duration: float) -> None:
        load_scale = {
            "study": 1.0,
            "revise": 0.85,
            "mock_test": 0.75,
            "switch_topic": 0.35,
            "skip": 0.20,
        }.get(action_type, 0.2)
        fatigue_load = (duration / 60.0) * 0.25 * self._state.fatigue_sensitivity * load_scale
        self._state.fatigue = _clamp(self._state.fatigue + fatigue_load)

    def _apply_forgetting(self, duration_minutes: float, protected_topics: Set[str]) -> None:
        for topic in self._state.topics:
            base_decay = self._state.forgetting_rate[topic] * (duration_minutes / 60.0) * 0.02
            retention_modifier = max(0.2, 1.0 - 0.5 * self._state.retention_strength)
            decay = base_decay * retention_modifier
            if topic in protected_topics:
                decay *= 0.25
            self._state.mastery[topic] = _clamp(self._state.mastery[topic] * (1.0 - decay))

    def _terminal_bonus(self, exam: float) -> float:
        if exam >= 0.80:
            return 0.20
        if exam >= 0.70:
            return 0.10
        if exam < 0.50:
            return -0.10
        return 0.0

    @staticmethod
    def _one_line_error(message: str) -> str:
        return " ".join(message.split())
