"""Aggregate saved, paired reward-ablation evaluations across training seeds.

This command does not train policies or rerun evaluation. Each training seed
first contributes one mean over its matched held-out environment episodes; the
reported uncertainty is then computed across training seeds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "survival_rate", "employment_rate", "wellbeing", "mean_health",
    "mean_stress", "mean_social_satisfaction", "avg_money",
    "hunger_incidents", "wealth_gini",
)


def _read_evaluation(path: Path, expected_training_seed: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report.get("checkpoint_metadata", {})
    if int(metadata.get("seed", -1)) != expected_training_seed:
        raise ValueError(f"{path}: checkpoint seed does not match {expected_training_seed}")
    if report.get("decision_mode") != "stochastic_sample" or report.get("deterministic"):
        raise ValueError(f"{path}: expected stochastic sampled-policy evaluation")
    if not report.get("episodes"):
        raise ValueError(f"{path}: evaluation contains no episodes")
    return report


def _episode_mean(report: dict[str, Any]) -> dict[str, float]:
    episodes = report["episodes"]
    return {
        metric: statistics.fmean(float(row[metric]) for row in episodes)
        for metric in METRICS
    }


def summarize_saved_replicates(
    seed_pairs: dict[int, tuple[Path, Path]], output: Path,
    component: str = "financial_stability",
) -> dict[str, Any]:
    """Summarize `(full evaluation, ablated evaluation)` files per training seed."""
    if len(seed_pairs) < 2:
        raise ValueError("at least two training-seed pairs are required")
    per_seed: dict[str, Any] = {}
    common_evaluation_seeds: list[int] | None = None
    common_days: int | None = None
    algorithm: str | None = None
    for seed in sorted(seed_pairs):
        full_path, ablated_path = seed_pairs[seed]
        full = _read_evaluation(full_path, seed)
        ablated = _read_evaluation(ablated_path, seed)
        for arm, report in (("full", full), ("ablated", ablated)):
            if report.get("algorithm") != "MAPPO":
                raise ValueError(f"{arm} evaluation for seed {seed} is not MAPPO")
            if int(report.get("days", 0)) != 30:
                raise ValueError(f"{arm} evaluation for seed {seed} is not 30 days")
            config = report.get("config", {})
            reward = config.get("reward", {})
            expected = 0.0 if arm == "ablated" else None
            if arm == "ablated" and float(reward.get(component, float("nan"))) != expected:
                raise ValueError(f"{ablated_path}: {component} is not zeroed")
            if arm == "full" and float(reward.get(component, float("nan"))) <= 0:
                raise ValueError(f"{full_path}: {component} is not enabled")
        full_seeds = [int(row["seed"]) for row in full["episodes"]]
        ablated_seeds = [int(row["seed"]) for row in ablated["episodes"]]
        if full_seeds != ablated_seeds:
            raise ValueError(f"training seed {seed}: evaluation episodes are not paired")
        if common_evaluation_seeds is None:
            common_evaluation_seeds = full_seeds
            common_days = int(full["days"])
            algorithm = str(full["algorithm"])
        elif full_seeds != common_evaluation_seeds or int(full["days"]) != common_days:
            raise ValueError("replicates do not share the same evaluation seeds and horizon")
        full_mean, ablated_mean = _episode_mean(full), _episode_mean(ablated)
        per_seed[str(seed)] = {
            "full_evaluation": str(full_path),
            "ablated_evaluation": str(ablated_path),
            "evaluation_seeds": full_seeds,
            "full": full_mean,
            "ablated": ablated_mean,
            "delta_ablated_minus_full": {
                metric: ablated_mean[metric] - full_mean[metric] for metric in METRICS
            },
        }

    aggregate: dict[str, Any] = {}
    for arm in ("full", "ablated", "delta_ablated_minus_full"):
        aggregate[arm] = {}
        for metric in METRICS:
            values = [row[arm][metric] for row in per_seed.values()]
            aggregate[arm][metric] = {
                "mean_across_training_seeds": statistics.fmean(values),
                "sd_across_training_seeds": statistics.stdev(values),
                "per_training_seed": {
                    seed: per_seed[seed][arm][metric] for seed in per_seed
                },
            }
    report = {
        "algorithm": algorithm,
        "component_zeroed": component,
        "training_seeds": [int(seed) for seed in per_seed],
        "training_seed_count": len(per_seed),
        "days": common_days,
        "evaluation_seeds": common_evaluation_seeds,
        "evaluation_episodes_per_training_seed": len(common_evaluation_seeds or []),
        "evaluation_decision_mode": "stochastic_sample",
        "aggregation": (
            "Mean over held-out environments within each training seed, then mean and sample SD across training seeds."
        ),
        "aggregate": aggregate,
        "per_training_seed": per_seed,
        "interpretation": (
            "Only three training seeds are included. The sample is seed-sensitive and exploratory; it does not "
            "establish a stable causal effect, convergence, or generalization. Raw episode evaluations remain the "
            "source data."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-dir", type=Path, default=Path("results/ablations/financial_stability_replicates"))
    parser.add_argument("--output", type=Path, default=Path("results/ablations/financial_stability_replicates/summary_three_training_seeds.json"))
    parser.add_argument("--seeds", default="23,123,223")
    args = parser.parse_args()
    pairs = {}
    for seed in (int(value) for value in args.seeds.split(",") if value.strip()):
        if seed == 23:
            pairs[seed] = (
                Path("results/ablations/mappo_full.json"),
                Path("results/ablations/mappo_without_financial_stability.json"),
            )
        else:
            base = args.replicate_dir / f"seed_{seed}"
            pairs[seed] = base / "full.json", base / "ablated.json"
    report = summarize_saved_replicates(pairs, args.output)
    print(f"Wrote {args.output} for training seeds {report['training_seeds']}")


if __name__ == "__main__":
    main()
