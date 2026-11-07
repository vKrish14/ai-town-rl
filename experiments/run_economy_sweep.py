"""Run matched-seed policy comparisons under a configured economy stress."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
from pathlib import Path
from typing import Any

from evaluate import evaluate


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["episodes"]
    metrics = (
        "survival_rate", "employment_rate", "wellbeing", "mean_health",
        "mean_stress", "mean_social_satisfaction", "avg_money", "hunger_incidents",
    )
    summary: dict[str, Any] = {"episodes": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def run_sweep(
    config_path: str,
    output_dir: str,
    episodes: int = 10,
    days: int = 30,
    seed: int = 9001,
    ppo_checkpoint: str = "models/ppo.pt",
    mappo_checkpoint: str = "models/mappo.pt",
) -> dict[str, Any]:
    """Evaluate all policies on the same contiguous environment seed range."""
    if episodes < 2:
        raise ValueError("episodes must be at least 2 for a seed-variation summary")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policies = (
        ("random", None), ("rule", None), ("learned", ppo_checkpoint),
        ("learned", mappo_checkpoint),
    )
    results: dict[str, Any] = {}
    for mode, checkpoint in policies:
        name = Path(checkpoint).stem if checkpoint else mode
        # evaluate() prints a copy of every row; retain the JSON artifacts as the
        # single readable record and keep a quiet progress line for long sweeps.
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(
                mode=mode, episodes=episodes, days=days, seed=seed,
                checkpoint=checkpoint, output=str(destination / f"{name}.json"),
                config_path=config_path,
            )
        results[name] = {"summary": _summary(result), "episodes": result["episodes"]}
        print(f"Finished {name}: {episodes} episodes, seeds {seed}–{seed + episodes - 1}", flush=True)
    report = {
        "config_path": config_path,
        "seed_start": seed,
        "seed_end": seed + episodes - 1,
        "days": days,
        "episodes_per_policy": episodes,
        "decision_mode": "stochastic_sample",
        "policies": results,
        "interpretation": (
            "Descriptive matched-environment-seed outcomes for these fixed checkpoints. "
            "Standard deviations measure environment-seed variation, not training-seed uncertainty."
        ),
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/stress_food_price.yaml")
    parser.add_argument("--output-dir", default="results/economy_stress_expanded")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--ppo-checkpoint", default="models/ppo.pt")
    parser.add_argument("--mappo-checkpoint", default="models/mappo.pt")
    args = parser.parse_args()
    run_sweep(
        args.config, args.output_dir, args.episodes, args.days, args.seed,
        args.ppo_checkpoint, args.mappo_checkpoint,
    )


if __name__ == "__main__":
    main()
