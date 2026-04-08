from __future__ import annotations

from typing import Dict, List, Optional

from .models import StudentProfile, TaskConfig

SINGLE_TOPIC = TaskConfig(
    name="single_topic",
    description="Improve one weak topic above threshold under a small budget.",
    topics=["genetics"],
    initial_mastery={"genetics": 0.15},
    topic_weights={"genetics": 1.0},
    topic_difficulty={"genetics": 0.35},
    forgetting_rate={"genetics": 0.30},
    prerequisites={},
    time_budget=60.0,
    max_steps=20,
    profile=StudentProfile(
        learner_speed=1.0,
        fatigue_sensitivity=0.50,
        retention_strength=0.90,
    ),
    initial_fatigue=0.0,
    coverage_threshold=0.70,
    success_exam_threshold=0.70,
    success_coverage_threshold=1.0,
    prerequisite_mastery_threshold=0.5,
    default_seed=11,
    initial_mastery_jitter=0.01,
)

BALANCED_PREP = TaskConfig(
    name="balanced_prep",
    description="Improve multiple topics without neglect while managing fatigue.",
    topics=["genetics", "ecology", "physiology", "cell_biology"],
    initial_mastery={
        "genetics": 0.25,
        "ecology": 0.30,
        "physiology": 0.20,
        "cell_biology": 0.35,
    },
    topic_weights={
        "genetics": 0.25,
        "ecology": 0.25,
        "physiology": 0.25,
        "cell_biology": 0.25,
    },
    topic_difficulty={
        "genetics": 0.45,
        "ecology": 0.50,
        "physiology": 0.60,
        "cell_biology": 0.40,
    },
    forgetting_rate={
        "genetics": 0.70,
        "ecology": 0.75,
        "physiology": 0.65,
        "cell_biology": 0.70,
    },
    prerequisites={},
    time_budget=120.0,
    max_steps=36,
    profile=StudentProfile(
        learner_speed=0.90,
        fatigue_sensitivity=0.70,
        retention_strength=0.80,
    ),
    initial_fatigue=0.0,
    coverage_threshold=0.65,
    success_exam_threshold=0.68,
    success_coverage_threshold=0.75,
    prerequisite_mastery_threshold=0.5,
    default_seed=22,
    initial_mastery_jitter=0.02,
)

FULL_EXAM_PLANNING = TaskConfig(
    name="full_exam_planning",
    description="Maximize weighted exam readiness with prerequisites and tighter trade-offs.",
    topics=["discrete_math", "algorithms", "os", "databases", "networks", "compilers"],
    initial_mastery={
        "discrete_math": 0.35,
        "algorithms": 0.20,
        "os": 0.25,
        "databases": 0.30,
        "networks": 0.28,
        "compilers": 0.18,
    },
    topic_weights={
        "discrete_math": 0.15,
        "algorithms": 0.20,
        "os": 0.20,
        "databases": 0.20,
        "networks": 0.15,
        "compilers": 0.10,
    },
    topic_difficulty={
        "discrete_math": 0.50,
        "algorithms": 0.80,
        "os": 0.75,
        "databases": 0.70,
        "networks": 0.60,
        "compilers": 0.85,
    },
    forgetting_rate={
        "discrete_math": 0.80,
        "algorithms": 0.95,
        "os": 0.90,
        "databases": 0.85,
        "networks": 0.80,
        "compilers": 0.95,
    },
    prerequisites={
        "algorithms": ["discrete_math"],
        "os": ["discrete_math"],
        "databases": ["algorithms"],
        "compilers": ["algorithms"],
    },
    time_budget=180.0,
    max_steps=52,
    profile=StudentProfile(
        learner_speed=0.82,
        fatigue_sensitivity=0.90,
        retention_strength=0.75,
    ),
    initial_fatigue=0.0,
    coverage_threshold=0.65,
    success_exam_threshold=0.72,
    success_coverage_threshold=0.70,
    prerequisite_mastery_threshold=0.55,
    default_seed=33,
    initial_mastery_jitter=0.03,
)

TASKS: Dict[str, TaskConfig] = {
    SINGLE_TOPIC.name: SINGLE_TOPIC,
    BALANCED_PREP.name: BALANCED_PREP,
    FULL_EXAM_PLANNING.name: FULL_EXAM_PLANNING,
}

DEFAULT_TASK_NAME = SINGLE_TOPIC.name


def list_task_names() -> List[str]:
    return list(TASKS.keys())


def get_task(task_name: Optional[str] = None) -> TaskConfig:
    key = task_name or DEFAULT_TASK_NAME
    if key not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise KeyError(f"unknown task '{key}', expected one of: {valid}")
    return TASKS[key].model_copy(deep=True)
