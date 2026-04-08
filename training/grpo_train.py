from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from student_planner.env import StudentPlannerCoreEnv
from student_planner.models import StudentPlannerAction

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert study-planning policy.

    Return exactly one JSON object with keys: action_type, topic, duration, topics.
    Action types: study, revise, mock_test, rest, switch_topic, skip.

    Objectives:
    - Increase readiness quickly.
    - Balance weak-topic recovery and fatigue control.
    - Avoid invalid actions.
    """
).strip()


def format_history(lines: List[str]) -> str:
    return "\n".join(lines[-5:]) if lines else "None"


def make_user_prompt(dataset_prompt: str, observation: Dict[str, Any], history: List[str]) -> str:
    return textwrap.dedent(
        f"""
        Scenario: {dataset_prompt}

        Observation JSON:
        {json.dumps(observation, sort_keys=True)}

        Recent history:
        {format_history(history)}

        Emit your next action as JSON.
        """
    ).strip()


def extract_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_action(completion_text: str, fallback_topic: str) -> StudentPlannerAction:
    block = extract_json_block(completion_text)
    if block:
        try:
            return StudentPlannerAction.model_validate(json.loads(block))
        except Exception:
            pass

    return StudentPlannerAction(action_type="study", topic=fallback_topic, duration=20.0)


def balance_score_from_mastery(mastery: Dict[str, float]) -> float:
    if not mastery:
        return 0.0
    values = list(mastery.values())
    if len(values) == 1:
        return 1.0
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    std_dev = variance ** 0.5
    return max(0.0, min(1.0, 1.0 - (std_dev / 0.5)))


def rollout_once(
    trainer: Any,
    env: StudentPlannerCoreEnv,
    tokenizer: Any,
    dataset_prompt: str,
    task_name: str,
    max_turns: int,
) -> Dict[str, Any]:
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset(task_name=task_name, seed=42)

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []

    progress_rewards: List[float] = []
    correctness_rewards: List[float] = []
    balance_rewards: List[float] = []
    fatigue_rewards: List[float] = []
    validity_rewards: List[float] = []

    history: List[str] = []

    for turn in range(max_turns):
        if result.done:
            break

        observation = result.observation.model_dump(mode="json")
        user_prompt = make_user_prompt(dataset_prompt, observation, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_output["prompt_ids"])
        completion_ids.extend(rollout_output["completion_ids"])
        logprobs.extend(rollout_output["logprobs"])

        completion_text = rollout_output.get("text")
        if not completion_text:
            completion_text = tokenizer.decode(
                rollout_output["completion_ids"],
                skip_special_tokens=True,
            )

        fallback_topic = min(result.observation.mastery, key=result.observation.mastery.get)
        action = parse_action(completion_text, fallback_topic=fallback_topic)
        result = env.step(action)

        info = result.info or {}
        readiness_delta = float(info.get("readiness_delta", 0.0))
        terminal_score = float(info.get("normalized_score", 0.0)) if result.done else 0.0
        balance_score = balance_score_from_mastery(result.observation.mastery)
        fatigue_score = max(0.0, min(1.0, 1.0 - result.observation.fatigue))
        validity_score = 0.0 if result.observation.last_action_error else 1.0

        progress_rewards.append(max(0.0, readiness_delta))
        correctness_rewards.append(terminal_score)
        balance_rewards.append(balance_score)
        fatigue_rewards.append(fatigue_score)
        validity_rewards.append(validity_score)

        history.append(
            f"turn={turn+1} action={action.model_dump(mode='json', exclude_none=True)} "
            f"reward={result.reward:.4f} readiness={result.observation.readiness:.4f}"
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "progress_reward": progress_rewards[-1] if progress_rewards else 0.0,
        "correct_reward": correctness_rewards[-1] if correctness_rewards else 0.0,
        "balance_reward": balance_rewards[-1] if balance_rewards else 0.0,
        "fatigue_reward": fatigue_rewards[-1] if fatigue_rewards else 0.0,
        "validity_reward": validity_rewards[-1] if validity_rewards else 0.0,
    }


def build_rollout_func(tokenizer: Any, task_name: str, max_turns: int):
    env = StudentPlannerCoreEnv(task_name=task_name)

    def rollout_func(prompts: List[str], trainer: Any = None) -> Dict[str, Any]:
        batch_prompt_ids: List[List[int]] = []
        batch_completion_ids: List[List[int]] = []
        batch_logprobs: List[List[float]] = []

        correct_rewards: List[float] = []
        progress_rewards: List[float] = []
        balance_rewards: List[float] = []
        fatigue_rewards: List[float] = []
        validity_rewards: List[float] = []

        for prompt in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                dataset_prompt=prompt,
                task_name=task_name,
                max_turns=max_turns,
            )
            batch_prompt_ids.append(episode["prompt_ids"])
            batch_completion_ids.append(episode["completion_ids"])
            batch_logprobs.append(episode["logprobs"])

            correct_rewards.append(episode["correct_reward"])
            progress_rewards.append(episode["progress_reward"])
            balance_rewards.append(episode["balance_reward"])
            fatigue_rewards.append(episode["fatigue_reward"])
            validity_rewards.append(episode["validity_reward"])

        return {
            "prompt_ids": batch_prompt_ids,
            "completion_ids": batch_completion_ids,
            "logprobs": batch_logprobs,
            "correct_reward": correct_rewards,
            "progress_reward": progress_rewards,
            "balance_reward": balance_rewards,
            "fatigue_reward": fatigue_rewards,
            "validity_reward": validity_rewards,
        }

    return rollout_func


def reward_correct(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("correct_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards is not None else [0.0 for _ in completions]


def reward_progress(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("progress_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards is not None else [0.0 for _ in completions]


def reward_balance(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("balance_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards is not None else [0.0 for _ in completions]


def reward_fatigue(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("fatigue_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards is not None else [0.0 for _ in completions]


def reward_validity(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("validity_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards is not None else [0.0 for _ in completions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Student Planner policy with GRPO")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--task-name", default="full_exam_planning")
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--output-dir", default="student-planner-grpo")
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--use-vllm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -e .[train]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict(
        {
            "prompt": [
                "Plan the next best study action to maximize exam readiness under fatigue and time constraints."
            ]
            * args.dataset_size
        }
    )

    rollout_func = build_rollout_func(
        tokenizer=tokenizer,
        task_name=args.task_name,
        max_turns=args.max_turns,
    )

    config = GRPOConfig(
        num_train_epochs=1,
        learning_rate=5e-6,
        gradient_accumulation_steps=32,
        per_device_train_batch_size=1,
        warmup_steps=10,
        num_generations=args.num_generations,
        max_completion_length=96,
        max_prompt_length=1800,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        output_dir=args.output_dir,
        report_to="none",
        logging_steps=1,
        save_steps=25,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correct,
            reward_progress,
            reward_balance,
            reward_fatigue,
            reward_validity,
        ],
        train_dataset=dataset,
        args=config,
        rollout_func=rollout_func,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
