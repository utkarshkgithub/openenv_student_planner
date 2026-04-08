from inference import log_end, log_start, log_step


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
