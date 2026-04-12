# Student Planner OpenEnv Benchmark

## 1. Objective

Build a real-world OpenEnv environment for **study planning** where an LLM agent must choose actions that improve a student's readiness for an exam under time, fatigue, and forgetting constraints.

The benchmark evaluates **decision quality over time**, not one-shot advice.

---

## 2. Core Loop

The environment follows the standard OpenEnv API:

- `reset()` initializes a new student episode
- `step(action)` applies one action and returns:
  - `observation`
  - `reward`
  - `done`
  - `info`
- `state()` returns the current hidden or public environment state

The agent repeatedly:

1. observes the current student state
2. proposes an action
3. the environment applies deterministic transition rules
4. reward is computed from improvement toward the target
5. the next state is returned

---

## 3. Real-World Task Definition

The environment simulates a student preparing for an exam.

The student has:
- topic mastery values
- fatigue
- time remaining
- forgetting dynamics
- topic weights / importance
- optional prerequisites

The agent must recommend study actions that maximize final readiness.

This is a realistic planning problem because:
- decisions have delayed effects
- time is limited
- over-studying increases fatigue
- weak topics should be prioritized
- some topics depend on prerequisites

---

## 4. State Design

Use a structured state with numeric values only.

### Example state fields

- `mastery`: mapping from topic name to value in `[0, 1]`
- `fatigue`: value in `[0, 1]`
- `time_left`: integer or float
- `topic_weights`: importance of each topic
- `difficulty`: per-topic learning difficulty
- `forgetting_rate`: per-topic decay factor
- `student_profile`: learner speed, fatigue sensitivity, retention strength

### Example topics

The topic list should be configurable per task, not hardcoded.

Examples:
- Biology: Genetics, Ecology, Physiology, Cell Biology
- Engineering: DSA, OS, DBMS, CN
- Any other academic domain

---

## 5. Action Space

Actions must be structured and deterministic.

### Recommended actions

- `study(topic, duration)`
- `revise(topic, duration)`
- `mock_test(topics?)`
- `rest(duration)`
- `switch_topic(topic)`
- `skip()`

### Constraints

- duration must be positive
- topic must exist in the current task configuration
- invalid actions must produce a penalty
- every action must reduce `time_left`

---

## 6. Transition Rules

The environment should define explicit formulas.

### Study action
A study action increases mastery, but the gain should depend on:
- current mastery
- fatigue
- topic difficulty
- student learning speed
- topic weight
- prerequisite status

Example:

- higher mastery gives diminishing returns
- higher fatigue reduces learning gain
- harder topics gain more slowly
- missing prerequisites lowers gain

### Rest action
Rest reduces fatigue, but consumes time.

### Mock test action
Mock tests do not directly increase mastery much, but they reveal current exam readiness and can give a small improvement through retrieval practice.

### Forgetting
Mastery should decay over time if a topic is not revised.

This makes the environment sequential rather than static.

---

## 7. Reward Design

Reward must be meaningful during the episode, not only at the end.

### Reward should include

- positive reward for improved readiness
- positive reward for improving weak/high-weight topics
- penalty for fatigue increase
- penalty for wasted time on already-mastered topics
- penalty for invalid actions
- penalty for ignoring critical prerequisites

### Recommended reward structure

Let:

- `E_t` = exam readiness before action
- `E_t+1` = exam readiness after action

Then step reward can be:

```text
reward = (E_t+1 - E_t)
         - fatigue_penalty
         - redundancy_penalty
         - invalid_action_penalty
```

Final reward can add:
- final exam score
- topic coverage bonus
- imbalance penalty

---

## 8. Exam Score Definition

The final score should be deterministic and based on topic mastery.

### Example

```text
exam_score = sum(topic_weight[i] * mastery[i]) / sum(topic_weight[i])
```

### Optional additions

- prerequisite satisfaction
- variance penalty for unbalanced preparation
- fatigue penalty if the episode ends exhausted
- coverage bonus if all important topics pass threshold

Keep the final score strictly inside `(0.0, 1.0)` using epsilon clipping.

---

## 9. Tasks

The benchmark must include at least 3 tasks with increasing difficulty.

### Task 1: Single Topic Improvement
**Goal:** Improve one weak topic above a threshold.

- one topic only
- simple time budget
- no prerequisite chain
- easy baseline

**Why it works:** tests whether the agent can improve a single state variable efficiently.

### Task 2: Balanced Preparation
**Goal:** Improve multiple topics while avoiding neglect.

- 3 to 5 topics
- unequal weights
- fatigue active
- simple forgetting active

**Why it works:** tests trade-off handling and prioritization.

### Task 3: Full Exam Planning
**Goal:** Maximize final exam score under time pressure.

- multiple topics
- topic weights
- prerequisites
- forgetting
- stronger fatigue penalty
- larger action horizon

**Why it works:** tests long-term planning, sequencing, rest decisions, and adaptation.

---

## 10. Grader Design

The grader must be deterministic and reproducible.

### Must score:
- final readiness
- topic coverage
- balance across topics
- fatigue management
- time efficiency
- invalid action count

### Example normalized score

```text
score = clamp(
    0.55 * exam_score
  + 0.20 * coverage_score
  + 0.15 * balance_score
  + 0.10 * efficiency_score
  - penalties,
  0.0,
  1.0
)
```

### Deterministic requirements
- same input state and action sequence must always produce the same score
- no LLM judge
- no randomness inside the grader
- any randomness only occurs in `reset()` with a fixed seed option

---

## 11. Baseline Inference Script

Provide `inference.py` in the project root.

### Requirements
- uses the OpenAI client
- reads credentials from environment variables
- runs the agent through all three tasks
- prints structured logs exactly as required by the validator
- produces reproducible baseline scores

### Baseline policy
Start with a simple prompt-based policy:
- identify weakest topic
- study high-weight weak topics first
- rest when fatigue is high
- use mock tests near the end
- avoid invalid actions

The baseline should be strong enough to show the environment is learnable.

---

## 12. OpenEnv Spec Files

### Required models
Use typed models for:
- `Action`
- `Observation`
- `State`
- `Reward` if needed by the spec

### Required methods
- `reset()`
- `step(action)`
- `state()`

### Required metadata
Provide `openenv.yaml` with:
- benchmark name
- version
- task list
- action space description
- observation space description
- scoring description
- environment entrypoint

---

## 13. Recommended Repository Layout

```text
.
├── README.md
├── openenv.yaml
├── inference.py
├── Dockerfile
├── src/
│   └── student_planner/
│       ├── __init__.py
│       ├── models.py
│       ├── env.py
│       ├── grader.py
│       ├── tasks.py
│       └── policy.py
└── tests/
    ├── test_env.py
    └── test_grader.py
```

---

## 14. Docker Requirements

The Docker image must:
- build cleanly
- start cleanly
- expose the OpenEnv app
- work with the Hugging Face Space runtime
- run on 2 vCPU / 8 GB memory

Use a minimal base image and keep dependencies small.

---

## 15. README Requirements

The README must include:

- environment description
- motivation
- action space
- observation space
- state definition
- task descriptions
- reward design
- grading logic
- baseline usage
- setup instructions
- Docker instructions
- expected scores for the baseline

---

## 16. What Makes This Strong

This benchmark is strong if:
- the tasks are clearly different in difficulty
- the final score reflects real planning quality
- the grader is deterministic
- the environment is easy to run and validate
- the same benchmark can compare multiple LLMs fairly

---

## 17. Implementation Priority

Build in this order:

1. define the state and action schema
2. define deterministic transition rules
3. define the reward and final score
4. define the three tasks
5. implement the OpenEnv methods
6. implement the baseline inference script
7. add Dockerfile
8. add `openenv.yaml`
9. write README
10. test validator compatibility

---

## 18. Success Condition

The project succeeds if:
- `openenv validate` passes
- Docker build passes
- the Hugging Face Space responds to `/reset`
- the baseline script runs end-to-end
- the grader returns normalized scores strictly inside `(0.0, 1.0)`
- the three tasks show increasing difficulty
- different LLMs produce meaningfully different results

---

## 19. Core Principle

Do not grade the text.

Grade the **state change** caused by the action sequence.

That is the benchmark.