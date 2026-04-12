from student_planner.env import StudentPlannerCoreEnv
from student_planner.models import StudentPlannerAction
from student_planner.tasks import list_task_names


def test_reset_is_deterministic_for_fixed_seed() -> None:
    env = StudentPlannerCoreEnv(task_name="balanced_prep")
    obs_a = env.reset(seed=123).observation
    obs_b = env.reset(seed=123).observation

    assert obs_a.mastery == obs_b.mastery
    assert obs_a.fatigue == obs_b.fatigue
    assert obs_a.time_left == obs_b.time_left


def test_invalid_action_penalizes_and_consumes_time() -> None:
    env = StudentPlannerCoreEnv(task_name="balanced_prep")
    env.reset(seed=7)
    time_before = env.state().time_left

    result = env.step(
        StudentPlannerAction(action_type="study", topic="unknown_topic", duration=10.0)
    )

    assert result.reward < 0.0
    assert result.observation.time_left < time_before
    assert result.observation.invalid_action_count == 1
    assert result.observation.last_action_error is not None


def test_episode_terminates_and_emits_normalized_score() -> None:
    env = StudentPlannerCoreEnv(task_name="single_topic")
    result = env.reset(seed=42)

    while not result.done:
        result = env.step(
            StudentPlannerAction(action_type="study", topic="genetics", duration=20.0)
        )

    assert result.done
    assert "normalized_score" in result.info
    assert 0.0 < float(result.info["normalized_score"]) < 1.0


def test_all_tasks_emit_strictly_bounded_normalized_scores() -> None:
    for task_name in list_task_names():
        env = StudentPlannerCoreEnv(task_name=task_name)
        result = env.reset(seed=42)

        while not result.done:
            result = env.step(StudentPlannerAction(action_type="skip", duration=10.0))

        assert "normalized_score" in result.info
        assert 0.0 < float(result.info["normalized_score"]) < 1.0


def test_reward_breakdown_matches_scalar_reward() -> None:
    env = StudentPlannerCoreEnv(task_name="balanced_prep")
    env.reset(seed=3)

    result = env.step(
        StudentPlannerAction(action_type="study", topic="genetics", duration=20.0)
    )

    reward_breakdown = result.info["reward_breakdown"]
    assert isinstance(reward_breakdown, dict)
    assert float(reward_breakdown["total_reward"]) == result.reward


def test_redundant_study_triggers_penalty() -> None:
    env = StudentPlannerCoreEnv(task_name="single_topic")
    env.reset(seed=42)
    env._state.mastery["genetics"] = 0.95  # deterministic setup for high-mastery penalty behavior

    result = env.step(
        StudentPlannerAction(action_type="study", topic="genetics", duration=20.0)
    )

    assert float(result.info["redundancy_penalty"]) > 0.0
