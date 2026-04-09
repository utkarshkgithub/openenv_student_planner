from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env(ROOT / ".env")

from student_planner.client import StudentPlannerEnv
from student_planner.models import StudentPlannerAction, StudentPlannerObservation
from student_planner.tasks import list_task_names

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("OPEN_AI_KEY")
    or os.getenv("API_KEY")
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = os.getenv("STUDENT_PLANNER_BENCHMARK", "student_planner")
TASK_FILTER = os.getenv("STUDENT_PLANNER_TASK")
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.65"))
BASELINE_SCORE_PATH = os.getenv("BASELINE_SCORE_PATH", "runs/baseline_scores.json")
SCORE_EPSILON = 1e-6

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent solving a Student Planner environment.

    Return only valid JSON with this schema:
    {
      "action_type": "study|revise|mock_test|rest|switch_topic|skip",
      "topic": "<topic-or-null>",
      "duration": <positive-number-or-null>,
      "topics": ["<topic>", "..."]
    }

    Rules:
    - Choose actions that improve readiness while avoiding fatigue spikes.
    - Prioritize weak topics, especially if they are high-impact.
    - Use rest when fatigue is high.
    - Avoid invalid actions.
    - Return JSON only, no markdown, no explanation.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def clamp_task_score(value: float) -> float:
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, float(value)))


def make_user_prompt(task_name: str, observation: StudentPlannerObservation, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    payload = {
        "task_name": task_name,
        "mastery": observation.mastery,
        "fatigue": observation.fatigue,
        "time_left": observation.time_left,
        "readiness": observation.readiness,
        "step_count": observation.step_count,
        "invalid_action_count": observation.invalid_action_count,
        "current_topic": observation.current_topic,
        "last_action_error": observation.last_action_error,
    }
    return textwrap.dedent(
        f"""
        Current observation:
        {json.dumps(payload, sort_keys=True)}

        Recent history:
        {history_block}

        Choose the next action as JSON.
        """
    ).strip()


def compact_action(action: StudentPlannerAction) -> str:
    payload = action.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, separators=(",", ":"))


def extract_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_action_response(raw_text: str) -> Optional[StudentPlannerAction]:
    block = extract_json_block(raw_text)
    if not block:
        return None

    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return None

    try:
        return StudentPlannerAction.model_validate(payload)
    except Exception:
        return None


def get_model_action(
    client: OpenAI,
    task_name: str,
    observation: StudentPlannerObservation,
    history: List[str],
) -> Tuple[StudentPlannerAction, str]:
    user_prompt = make_user_prompt(task_name, observation, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc

    text = (completion.choices[0].message.content or "").strip()
    action = parse_action_response(text)
    if action is not None:
        return action, text

    # One repair attempt through the same OpenAI-compatible endpoint.
    repair_prompt = (
        "Your previous response was invalid. "
        "Return only valid JSON with keys action_type, topic, duration, topics."
    )
    try:
        repair_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": text},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI-compatible repair request failed: {exc}") from exc

    repaired_text = (repair_completion.choices[0].message.content or "").strip()
    repaired_action = parse_action_response(repaired_text)
    if repaired_action is None:
        raise RuntimeError("Model did not return a valid JSON action")
    return repaired_action, repaired_text


async def create_env(task_name: str) -> StudentPlannerEnv:
    if LOCAL_IMAGE_NAME:
        return await StudentPlannerEnv.from_docker_image(
            image_name=LOCAL_IMAGE_NAME,
            task_name=task_name,
        )
    return StudentPlannerEnv(base_url=ENV_BASE_URL, task_name=task_name)


async def run_task(task_name: str, client: OpenAI) -> Dict[str, object]:
    env = await create_env(task_name)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False
    normalized_score = SCORE_EPSILON

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name, seed=42)

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if result.done:
                break

            obs = result.observation
            action, _raw_output = get_model_action(client, task_name, obs, history)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=compact_action(action),
                reward=reward,
                done=bool(result.done),
                error=result.observation.last_action_error,
            )

            history.append(
                f"step={step} action={compact_action(action)} reward={reward:.2f} readiness={result.observation.readiness:.3f}"
            )

            if result.done:
                break

        normalized_score = clamp_task_score(
            float((result.info or {}).get("normalized_score", result.observation.readiness))
        )
        success = normalized_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task": task_name,
        "steps": steps_taken,
        "total_reward": round(sum(rewards), 6),
        "normalized_score": round(normalized_score, 6),
        "success": success,
    }


async def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN or OPENAI_API_KEY in the environment")

    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    if TASK_FILTER:
        task_names = [TASK_FILTER]
    else:
        task_names = list_task_names()

    summaries: List[Dict[str, object]] = []
    for task_name in task_names:
        summaries.append(await run_task(task_name=task_name, client=client))

    if BASELINE_SCORE_PATH:
        score_path = Path(BASELINE_SCORE_PATH)
        if not score_path.is_absolute():
            score_path = ROOT / score_path
        score_path.parent.mkdir(parents=True, exist_ok=True)

        mean_score = 0.0
        if summaries:
            mean_score = sum(float(item["normalized_score"]) for item in summaries) / len(summaries)

        payload = {
            "benchmark": BENCHMARK,
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "tasks": summaries,
            "mean_normalized_score": round(mean_score, 6),
            "score_range": [0.0, 1.0],
        }
        score_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
