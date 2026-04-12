from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from student_planner.env import StudentPlannerCoreEnv


START_RE = re.compile(r"task=([^ ]+)")
STEP_RE = re.compile(r"step=(\d+) action=(.*) reward=([-+]?\d*\.?\d+) done=(true|false) error=(.*)$")
END_RE = re.compile(r"success=(true|false) steps=(\d+) rewards=(.*)$")
SCORE_EPSILON = 1e-6


def _clamp_open_unit_interval(value: float) -> float:
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, float(value)))


@dataclass
class EpisodeRow:
    task: str
    success: bool = False
    steps: int = 0
    rewards: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


def parse_log(path: pathlib.Path) -> List[EpisodeRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: List[EpisodeRow] = []
    current: EpisodeRow | None = None

    for line in lines:
        if line.startswith("[START]"):
            match = START_RE.search(line)
            if match:
                current = EpisodeRow(task=match.group(1))
            continue

        if line.startswith("[STEP]") and current is not None:
            match = STEP_RE.search(line)
            if not match:
                continue
            current.actions.append(match.group(2))
            continue

        if line.startswith("[END]") and current is not None:
            match = END_RE.search(line)
            if not match:
                continue

            current.success = match.group(1) == "true"
            current.steps = int(match.group(2))
            rewards_raw = match.group(3).strip()
            current.rewards = [float(value) for value in rewards_raw.split(",") if value]
            rows.append(current)
            current = None

    return rows


def replay_normalized_score(task: str, action_strs: List[str]) -> float:
    env = StudentPlannerCoreEnv(task_name=task)
    result = env.reset(seed=42)

    for action_str in action_strs:
        if result.done:
            break

        try:
            payload = json.loads(action_str)
        except json.JSONDecodeError:
            continue

        result = env.step(payload)

    raw_score = float((result.info or {}).get("normalized_score", result.observation.readiness))
    return _clamp_open_unit_interval(raw_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark scores from inference log")
    parser.add_argument("log_path", nargs="?", default="runs/benchmark.log")
    args = parser.parse_args()

    path = pathlib.Path(args.log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    rows = parse_log(path)
    if not rows:
        print("No benchmark rows found in log.")
        return

    print("task\tsuccess\tsteps\ttotal_reward\tnormalized_score")

    normalized_values: List[float] = []
    reward_values: List[float] = []
    for row in rows:
        total_reward = sum(row.rewards)
        normalized_score = replay_normalized_score(task=row.task, action_strs=row.actions)
        normalized_values.append(normalized_score)
        reward_values.append(total_reward)
        print(
            f"{row.task}\t{str(row.success).lower()}\t{row.steps}\t{total_reward:.3f}\t{normalized_score:.6f}"
        )

    mean_reward = statistics.mean(reward_values)
    mean_normalized = statistics.mean(normalized_values)
    success_rate = sum(1 for row in rows if row.success) / len(rows)

    print(f"overall_avg_total_reward\t{mean_reward:.3f}")
    print(f"overall_avg_normalized_score\t{mean_normalized:.6f}")
    print(f"success_rate\t{success_rate:.3f}")


if __name__ == "__main__":
    main()
