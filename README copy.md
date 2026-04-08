# Student Planner OpenEnv Benchmark

A deterministic OpenEnv benchmark where an agent must plan study actions to maximize exam readiness under time, fatigue, forgetting, and prerequisite constraints.

## Why this benchmark

Most planning environments grade one-shot answers. Student Planner grades sequential decision quality over time.

The policy must decide how to trade off:
- weak topic recovery
- weighted exam priorities
- fatigue management
- revision vs new study
- prerequisite sequencing

## Project layout

```text
.
├── README.md
├── openenv.yaml
├── inference.py
├── Dockerfile
├── server/
│   └── app.py
├── training/
│   └── grpo_train.py
├── src/
│   └── student_planner/
│       ├── __init__.py
│       ├── client.py
│       ├── env.py
│       ├── grader.py
│       ├── models.py
│       └── tasks.py
└── tests/
    ├── test_env.py
    ├── test_grader.py
    └── test_inference_logging.py
```

## Environment API

The server exposes:
- `GET /`
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /docs`
- `GET /web`
- `WS /ws`

The core environment implements:
- `reset(task_name=None, seed=None)`
- `step(action)`
- `state()`

Typed models (Pydantic):
- `StudentPlannerAction`
- `StudentPlannerObservation`
- `StudentPlannerReward`

## State, action, and observation

### State fields
- `mastery`: per-topic mastery in [0, 1]
- `fatigue`: in [0, 1]
- `time_left`: minutes left in episode
- `topic_weights`: exam importance
- `topic_difficulty`: learning drag
- `forgetting_rate`: passive decay rate
- `prerequisites`: optional dependency graph
- `learner_speed`, `fatigue_sensitivity`, `retention_strength`

### Actions
- `study(topic, duration)`
- `revise(topic, duration)`
- `mock_test(topics?, duration?)`
- `rest(duration)`
- `switch_topic(topic, duration?)`
- `skip(duration?)`

### Observation fields
- `mastery`, `fatigue`, `time_left`, `readiness`
- `step_count`, `invalid_action_count`
- `current_topic`, `last_action_error`

## Transition and reward design

The transition is deterministic.

- Study and revise gains depend on mastery saturation, fatigue, difficulty, learner profile, and prerequisite satisfaction.
- Rest lowers fatigue while consuming time.
- Mock tests give small retrieval-practice improvements.
- Forgetting decays mastery each step, with reduced decay on recently touched topics.

Step reward uses readiness delta minus penalties:

```text
reward = (E_t+1 - E_t)
         - fatigue_penalty
         - redundancy_penalty
         - invalid_action_penalty
         - prerequisite_penalty
```

Terminal bonus is applied at episode end based on exam score band.
The per-step reward decomposition is available in `info.reward_breakdown`.

## Tasks

Three deterministic tasks with increasing difficulty:

1. `single_topic`
- One topic, easy budget, no prerequisites.

2. `balanced_prep`
- Four topics, equal weights, active fatigue and forgetting.

3. `full_exam_planning`
- Six topics, weighted exam, prerequisites, tighter trade-offs.

## Grading

The final normalized score is clamped to [0, 1]:

```text
score = 0.50 * exam_score
      + 0.20 * coverage_score
      + 0.15 * balance_score
      + 0.10 * efficiency_score
      + 0.05 * fatigue_score
      - invalid_penalty
```

Where:
- `exam_score` is weighted mastery
- `coverage_score` is fraction of topics above threshold
- `balance_score` penalizes uneven preparation
- `efficiency_score` rewards practical time use
- `fatigue_score = 1 - final_fatigue`

## Installation

```bash
pip install -e .
```

For training extras:

```bash
pip install -e .[train]
```

For testing:

```bash
pip install -e .[test]
```

## Run locally (Uvicorn)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Health check:

```bash
curl http://localhost:7860/health
```

## Run with Docker

```bash
docker build -t student-planner:latest .
docker run -d -p 7860:7860 \
  -e MAX_CONCURRENT_ENVS=4 \
  --name student-planner \
  student-planner:latest
```

## Baseline inference script

`inference.py` is submission-oriented and emits strict logs:
- `[START]`
- `[STEP]`
- `[END]`

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

`inference.py` auto-loads values from `.env` in the repo root when present.

Compatibility fallback:
- `OPENAI_API_KEY` and `API_KEY` are also accepted by `inference.py`.

Optional runtime variables:
- `LOCAL_IMAGE_NAME` (use `from_docker_image` mode)
- `ENV_BASE_URL` (default `http://localhost:7860`)
- `STUDENT_PLANNER_TASK` (single-task run)
- `TEMPERATURE` (default `0.0` for reproducible runs)
- `BASELINE_SCORE_PATH` (default `runs/baseline_scores.json`)

Run:

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="deepseek/deepseek-chat"
export HF_TOKEN="<token>"
python inference.py | tee runs/benchmark.log
python scripts/benchmark_scores.py runs/benchmark.log
```

`inference.py` also writes normalized task scores to `runs/baseline_scores.json` (or `BASELINE_SCORE_PATH`).

## GRPO training scaffold (TRL)

The training script mirrors the Wordle-style pattern:
- rollout through environment interaction
- multi-channel reward shaping
- GRPO optimization

Run a smoke train:

```bash
python training/grpo_train.py \
  --model-name Qwen/Qwen3-1.7B \
  --task-name full_exam_planning \
  --dataset-size 128 \
  --max-turns 8 \
  --output-dir student-planner-grpo-smoke
```

## Tests

```bash
pytest -q
```

## Expected baseline score bands

These are target ranges for a healthy baseline policy:

| Task | Expected normalized score |
|------|---------------------------|
| single_topic | 0.70-0.90 |
| balanced_prep | 0.60-0.80 |
| full_exam_planning | 0.50-0.75 |

## Reference baseline (checked-in run)

From `runs/benchmark.log` (parsed by `scripts/benchmark_scores.py`):

| Task | Total reward | Normalized score |
|------|--------------|------------------|
| single_topic | 0.200 | 0.359 |
| balanced_prep | 0.070 | 0.341 |
| full_exam_planning | -0.030 | 0.263 |

## HF Spaces notes

- Use WebSocket endpoint `/ws` for session interactions.
- Tune with environment variables:
  - `MAX_CONCURRENT_ENVS`
  - `HOST`
  - `PORT`
- Free tier capacity is limited; use conservative concurrency defaults.

## Submission checks to run

1. `openenv validate`
2. `docker build .`
3. `pytest -q`
4. `python inference.py` with required env vars
5. Verify deployed Space returns 200 on `/health` and `/reset`
