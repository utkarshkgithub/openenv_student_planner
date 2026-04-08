from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionType = Literal["study", "revise", "mock_test", "rest", "switch_topic", "skip"]


class StudentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learner_speed: float = Field(default=0.85, ge=0.2, le=2.0)
    fatigue_sensitivity: float = Field(default=0.70, ge=0.1, le=2.0)
    retention_strength: float = Field(default=0.80, ge=0.1, le=1.5)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    topics: List[str]
    initial_mastery: Dict[str, float]
    topic_weights: Dict[str, float]
    topic_difficulty: Dict[str, float]
    forgetting_rate: Dict[str, float]
    prerequisites: Dict[str, List[str]] = Field(default_factory=dict)
    time_budget: float = Field(default=120.0, gt=0.0)
    max_steps: int = Field(default=40, ge=1)
    profile: StudentProfile = Field(default_factory=StudentProfile)
    initial_fatigue: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    success_exam_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    success_coverage_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    prerequisite_mastery_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    default_seed: int = 42
    initial_mastery_jitter: float = Field(default=0.02, ge=0.0, le=0.2)

    @model_validator(mode="after")
    def validate_topic_mappings(self) -> "TaskConfig":
        topic_set = set(self.topics)
        if not topic_set:
            raise ValueError("topics must not be empty")

        for name, mapping in (
            ("initial_mastery", self.initial_mastery),
            ("topic_weights", self.topic_weights),
            ("topic_difficulty", self.topic_difficulty),
            ("forgetting_rate", self.forgetting_rate),
        ):
            mapping_keys = set(mapping.keys())
            if mapping_keys != topic_set:
                missing = sorted(topic_set - mapping_keys)
                extra = sorted(mapping_keys - topic_set)
                raise ValueError(
                    f"{name} keys must exactly match topics. missing={missing}, extra={extra}"
                )

        for topic, prereqs in self.prerequisites.items():
            if topic not in topic_set:
                raise ValueError(f"prerequisites contains unknown topic '{topic}'")
            for prereq in prereqs:
                if prereq not in topic_set:
                    raise ValueError(
                        f"prerequisite '{prereq}' for topic '{topic}' is unknown"
                    )

        weight_sum = sum(self.topic_weights.values())
        if weight_sum <= 0.0:
            raise ValueError("topic_weights sum must be positive")

        for mapping in (
            self.initial_mastery,
            self.topic_weights,
            self.topic_difficulty,
            self.forgetting_rate,
        ):
            for value in mapping.values():
                if value < 0.0:
                    raise ValueError("mapping values must be non-negative")

        return self


class StudentPlannerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    topic: Optional[str] = None
    duration: Optional[float] = None
    topics: Optional[List[str]] = None


class StudentPlannerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    mastery: Dict[str, float]
    fatigue: float
    time_left: float
    readiness: float
    step_count: int
    invalid_action_count: int
    current_topic: Optional[str] = None
    last_action_error: Optional[str] = None


class StudentPlannerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    topics: List[str]
    mastery: Dict[str, float]
    fatigue: float
    time_left: float
    time_budget: float
    step_count: int
    max_steps: int
    topic_weights: Dict[str, float]
    topic_difficulty: Dict[str, float]
    forgetting_rate: Dict[str, float]
    prerequisites: Dict[str, List[str]]
    learner_speed: float
    fatigue_sensitivity: float
    retention_strength: float
    prerequisite_mastery_threshold: float
    invalid_action_count: int
    done: bool = False
    current_topic: Optional[str] = None


class StudentPlannerReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readiness_delta: float
    fatigue_penalty: float
    redundancy_penalty: float
    prerequisite_penalty: float
    invalid_action_penalty: float
    terminal_bonus: float = 0.0
    total_reward: float


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: StudentPlannerObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GradeBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exam_score: float
    coverage_score: float
    balance_score: float
    efficiency_score: float
    fatigue_score: float
    invalid_penalty: float
    final_score: float
    success: bool


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: StudentPlannerAction
