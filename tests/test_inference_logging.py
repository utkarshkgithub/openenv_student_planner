from inference import (
    SCORE_EPSILON,
    SERIALIZATION_SCORE_EPSILON,
    clamp_task_score,
    log_end,
    log_start,
    log_step,
)


def test_required_log_format(capsys) -> None:
    log_start(task="single_topic", env="student_planner", model="demo-model")
    log_step(
        step=1,
        action='{"action_type":"study","topic":"genetics","duration":20}',
        reward=0.125,
        done=False,
        error=None,
    )
    log_end(success=True, steps=1, rewards=[0.125])

    output = capsys.readouterr().out.strip().splitlines()

    assert output[0] == "[START] task=single_topic env=student_planner model=demo-model"
    assert (
        output[1]
        == '[STEP] step=1 action={"action_type":"study","topic":"genetics","duration":20} reward=0.12 done=false error=null'
    )
    assert output[2] == "[END] success=true steps=1 rewards=0.12"


def test_clamp_task_score_stays_strictly_open() -> None:
    assert clamp_task_score(0.0) == SCORE_EPSILON
    assert clamp_task_score(1.0) == 1.0 - SCORE_EPSILON


def test_serialization_clamp_survives_thousandth_rounding() -> None:
    low = clamp_task_score(0.0, epsilon=SERIALIZATION_SCORE_EPSILON)
    high = clamp_task_score(1.0, epsilon=SERIALIZATION_SCORE_EPSILON)

    assert round(low, 3) > 0.0
    assert round(high, 3) < 1.0
