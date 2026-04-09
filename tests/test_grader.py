from student_planner.env import StudentPlannerCoreEnv
from student_planner.grader import grade_state
from student_planner.models import StudentPlannerAction
from student_planner.tasks import get_task


def _run_sequence(task_name: str, actions: list[StudentPlannerAction]) -> float:
    env = StudentPlannerCoreEnv(task_name=task_name)
    result = env.reset(seed=42)

    for action in actions:
        if result.done:
            break
        result = env.step(action)

    while not result.done:
        result = env.step(StudentPlannerAction(action_type="skip", duration=10.0))

    return float(result.info["normalized_score"])


def test_grade_bounds() -> None:
    env = StudentPlannerCoreEnv(task_name="full_exam_planning")
    env.reset(seed=5)
    state = env.state()
    grade = grade_state(state, get_task("full_exam_planning"))

    assert 0.0 < grade.exam_score < 1.0
    assert 0.0 < grade.coverage_score < 1.0
    assert 0.0 < grade.balance_score < 1.0
    assert 0.0 < grade.efficiency_score < 1.0
    assert 0.0 < grade.fatigue_score < 1.0
    assert 0.0 <= grade.invalid_penalty <= 0.5
    assert 0.0 < grade.final_score < 1.0


def test_single_topic_components_are_strict_open() -> None:
    env = StudentPlannerCoreEnv(task_name="single_topic")
    env.reset(seed=42)
    grade = grade_state(env.state(), get_task("single_topic"))

    assert 0.0 < grade.coverage_score < 1.0
    assert 0.0 < grade.balance_score < 1.0
    assert 0.0 < grade.fatigue_score < 1.0


def test_boundary_prone_components_are_clamped_open() -> None:
    env = StudentPlannerCoreEnv(task_name="single_topic")
    env.reset(seed=11)
    state = env.state()

    # Force multiple raw component boundaries at once.
    state.mastery["genetics"] = 0.0
    state.fatigue = 0.0
    state.time_left = state.time_budget

    grade = grade_state(state, get_task("single_topic"))

    assert 0.0 < grade.exam_score < 1.0
    assert 0.0 < grade.coverage_score < 1.0
    assert 0.0 < grade.balance_score < 1.0
    assert 0.0 < grade.efficiency_score < 1.0
    assert 0.0 < grade.fatigue_score < 1.0
    assert 0.0 < grade.final_score < 1.0


def test_deterministic_score_for_same_actions() -> None:
    actions = [
        StudentPlannerAction(action_type="study", topic="genetics", duration=20.0),
        StudentPlannerAction(action_type="revise", topic="genetics", duration=12.0),
        StudentPlannerAction(action_type="mock_test", duration=8.0),
    ]

    score_a = _run_sequence("balanced_prep", actions)
    score_b = _run_sequence("balanced_prep", actions)

    assert score_a == score_b


def test_better_policy_scores_higher_than_skip_policy() -> None:
    focused_actions = [
        StudentPlannerAction(action_type="study", topic="genetics", duration=20.0),
        StudentPlannerAction(action_type="study", topic="physiology", duration=20.0),
        StudentPlannerAction(action_type="revise", topic="genetics", duration=12.0),
        StudentPlannerAction(action_type="mock_test", duration=10.0),
    ]
    skip_actions = [StudentPlannerAction(action_type="skip", duration=10.0) for _ in range(4)]

    focused_score = _run_sequence("balanced_prep", focused_actions)
    skip_score = _run_sequence("balanced_prep", skip_actions)

    assert focused_score > skip_score
