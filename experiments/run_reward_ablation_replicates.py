"""Train matched reward ablations across policy seeds and evaluate paired environments."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from ai_town.config import load_config
from evaluate import evaluate
from train import train


METRICS = (
    "survival_rate", "employment_rate", "wellbeing", "mean_health",
    "mean_stress", "mean_social_satisfaction", "avg_money", "hunger_incidents",
)


def _reference_checkpoint(algorithm: str, seed: int) -> Path:
    normalized = algorithm.lower()
    primary = Path("models") / f"{normalized}.pt"
    candidates = [primary, Path("models/replicates") / f"{normalized}_seed{seed}.pt"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = torch.load(candidate, map_location="cpu", weights_only=False)
        if int(data.get("metadata", {}).get("seed", -1)) == seed:
            return candidate
    raise FileNotFoundError(f"no {algorithm} reference checkpoint recorded for training seed {seed}")


def _mean_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        metric: statistics.fmean(float(row[metric]) for row in result["episodes"])
        for metric in METRICS
    }


def run_replicates(
    algorithm: str = "MAPPO",
    component: str = "financial_stability",
    training_seeds: tuple[int, ...] = (23, 123, 223),
    episodes: int = 100,
    days: int = 30,
    eval_episodes: int = 5,
    eval_seed: int = 1001,
    output_dir: str = "results/ablations/financial_stability_replicates",
) -> dict[str, Any]:
    """Train an ablated model per seed and compare with that seed's full model."""
    algorithm = algorithm.upper()
    if algorithm not in {"PPO", "MAPPO"}:
        raise ValueError("algorithm must be PPO or MAPPO")
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training seeds must be a nonempty unique sequence")
    config = copy.deepcopy(load_config())
    if component not in config["reward"]:
        raise ValueError(f"unknown reward component: {component}")
    config["reward"][component] = 0.0

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / f"{algorithm.lower()}_without_{component}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    per_seed: dict[str, Any] = {}

    for seed in training_seeds:
        ablated_checkpoint = output / "checkpoints" / f"{algorithm.lower()}_seed{seed}.pt"
        metrics_path = output / "training_logs" / f"{algorithm.lower()}_seed{seed}.jsonl"
        reference = _reference_checkpoint(algorithm, seed)
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                episodes=episodes, seed=seed, days=days,
                output=str(ablated_checkpoint), config_path=str(config_path),
                algorithm=algorithm, metrics_output=str(metrics_path),
            )
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            full = evaluate(
                mode="learned", episodes=eval_episodes, days=days, seed=eval_seed,
                checkpoint=str(reference), output=str(seed_dir / "full.json"),
            )
            ablated = evaluate(
                mode="learned", episodes=eval_episodes, days=days, seed=eval_seed,
                checkpoint=str(ablated_checkpoint), output=str(seed_dir / "ablated.json"),
                config_path=str(config_path),
            )
        full_means, ablated_means = _mean_metrics(full), _mean_metrics(ablated)
        per_seed[str(seed)] = {
            "reference_checkpoint": str(reference),
            "ablated_checkpoint": str(ablated_checkpoint),
            "evaluation_seeds": [row["seed"] for row in full["episodes"]],
            "full": full_means,
            "ablated": ablated_means,
            "delta_ablated_minus_full": {
                metric: ablated_means[metric] - full_means[metric] for metric in METRICS
            },
        }
        print(f"Finished {algorithm} {component} ablation seed {seed}", flush=True)

    aggregate: dict[str, Any] = {}
    for arm in ("full", "ablated", "delta_ablated_minus_full"):
        aggregate[arm] = {}
        for metric in METRICS:
            values = [row[arm][metric] for row in per_seed.values()]
            aggregate[arm][metric] = {
                "mean_across_training_seeds": statistics.fmean(values),
                "sd_across_training_seeds": statistics.stdev(values) if len(values) > 1 else 0.0,
                "per_training_seed": values,
            }
    report = {
        "algorithm": algorithm,
        "component_zeroed": component,
        "training_seeds": list(training_seeds),
        "episodes_per_training_seed": episodes,
        "days": days,
        "evaluation_seed_start": eval_seed,
        "evaluation_episodes_per_training_seed": eval_episodes,
        "evaluation_decision_mode": "stochastic_sample",
        "aggregate": aggregate,
        "per_training_seed": per_seed,
        "interpretation": (
            "Matched reference and ablated checkpoints use the same training seeds and held-out environment "
            "seeds. Aggregate SD is across training seeds (small n); within-seed environment variation is "
            "retained in raw evaluation files. This does not establish a general causal effect or convergence."
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["PPO", "MAPPO"], default="MAPPO")
    parser.add_argument("--component", default="financial_stability")
    parser.add_argument("--training-seeds", default="123,223")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=1001)
    parser.add_argument("--output-dir", default="results/ablations/financial_stability_replicates")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.training_seeds.split(",") if value.strip())
    run_replicates(
        args.algorithm, args.component, seeds, args.episodes, args.days,
        args.eval_episodes, args.eval_seed, args.output_dir,
    )


if __name__ == "__main__":
    main()
