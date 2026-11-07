"""Measure CPU evaluation throughput while retaining raw episode metrics."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from evaluate import evaluate
from ai_town.config import load_config


def profile(mode="rule", checkpoint=None, episodes=3, days=30, seed=8101, population=None, output=None):
    config = load_config()
    pop = int(config["simulation"]["population"] if population is None else population)
    raw_path = Path(output or "results/headless_profile_eval.json")
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate(
            mode=mode,
            episodes=episodes,
            days=days,
            seed=seed,
            checkpoint=checkpoint,
            output=raw_path,
            population=pop,
        )
    elapsed = time.perf_counter() - started
    environment_steps = episodes * days * int(config["simulation"]["ticks_per_day"])
    measurements = {
        "mode": mode,
        "checkpoint": checkpoint,
        "population": pop,
        "episodes": episodes,
        "days_per_episode": days,
        "environment_steps": environment_steps,
        "citizen_decisions": environment_steps * pop,
        "wall_seconds": elapsed,
        "environment_steps_per_second": environment_steps / max(elapsed, 1e-9),
        "citizen_decisions_per_second": environment_steps * pop / max(elapsed, 1e-9),
        "raw_evaluation": str(raw_path),
        "mean_reward": result["mean_reward"],
    }
    summary_path = raw_path.with_name(raw_path.stem + "_profile.json")
    summary_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(json.dumps(measurements, indent=2))
    return measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("random", "rule", "learned"), default="rule")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--population", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    profile(**vars(args))
