from student_planner.env import StudentPlannerCoreEnv
from student_planner.grader import grade_state
from student_planner.models import StudentPlannerAction
from student_planner.tasks import get_task


def _assert_strictly_bounded(score: float) -> None:
    assert 0.0 < score < 1.0


def _assert_grade_scores_strictly_bounded(grade) -> None:
    _assert_strictly_bounded(grade.exam_score)
    _assert_strictly_bounded(grade.coverage_score)
    _assert_strictly_bounded(grade.balance_score)
    _assert_strictly_bounded(grade.efficiency_score)
    _assert_strictly_bounded(grade.fatigue_score)
    _assert_strictly_bounded(grade.final_score)


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

    _assert_grade_scores_strictly_bounded(grade)


def test_grade_extremes_remain_strictly_bounded() -> None:
    env = StudentPlannerCoreEnv(task_name="single_topic")
    env.reset(seed=11)

    env._state.mastery = {topic: 0.0 for topic in env._state.mastery}
    env._state.fatigue = 1.0
    env._state.invalid_action_count = 100
    low_grade = grade_state(env.state(), get_task("single_topic"))
    _assert_grade_scores_strictly_bounded(low_grade)

    env.reset(seed=11)
    env._state.mastery = {topic: 1.0 for topic in env._state.mastery}
    env._state.fatigue = 0.0
    env._state.invalid_action_count = 0
    high_grade = grade_state(env.state(), get_task("single_topic"))
    _assert_grade_scores_strictly_bounded(high_grade)


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
