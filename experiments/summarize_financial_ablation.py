"""Summarize matched financial-stability ablation runs by training seed.

The statistical unit is the training seed: held-out episode rows are first
averaged within each checkpoint, then means/SDs are computed across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/ablations/financial_stability_three_seed_summary.json"
METRICS = (
    "survival_rate",
    "employment_rate",
    "wellbeing",
    "mean_health",
    "mean_stress",
    "mean_social_satisfaction",
    "avg_money",
    "hunger_incidents",
)


def load_run(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        run = json.load(stream)
    episodes = run.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path} has no episode records")
    seeds = [int(row["seed"]) for row in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{path} contains duplicate evaluation seeds")
    means = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in episodes]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path} contains a non-finite {metric}")
        means[metric] = mean(values)
    return {
        "source": str(path.relative_to(ROOT)),
        "checkpoint": run.get("checkpoint"),
        "checkpoint_seed": int(run["checkpoint_metadata"]["seed"]),
        "days": int(run["days"]),
        "decision_mode": run.get("decision_mode", "unspecified"),
        "evaluation_seeds": seeds,
        "episode_count": len(episodes),
        "means": means,
    }


def summarize(root: Path = ROOT) -> dict[str, Any]:
    replicate_dir = root / "results/ablations/financial_stability_replicates"
    seed_files = {
        23: (root / "results/ablations/mappo_full.json", root / "results/ablations/mappo_without_financial_stability.json"),
        123: (replicate_dir / "seed_123/full.json", replicate_dir / "seed_123/ablated.json"),
        223: (replicate_dir / "seed_223/full.json", replicate_dir / "seed_223/ablated.json"),
    }
    records: dict[str, dict[str, Any]] = {}
    reference_conditions = None
    for training_seed, (full_path, ablated_path) in seed_files.items():
        full, ablated = load_run(full_path), load_run(ablated_path)
        for condition in (full, ablated):
            if condition["checkpoint_seed"] != training_seed:
                raise ValueError(f"{condition['source']} checkpoint seed mismatch")
        conditions = (full["days"], full["decision_mode"], full["evaluation_seeds"])
        if (ablated["days"], ablated["decision_mode"], ablated["evaluation_seeds"]) != conditions:
            raise ValueError(f"unmatched full/ablated evaluation conditions for seed {training_seed}")
        if reference_conditions is None:
            reference_conditions = conditions
        elif conditions != reference_conditions:
            raise ValueError(f"evaluation conditions differ for training seed {training_seed}")
        delta = {metric: ablated["means"][metric] - full["means"][metric] for metric in METRICS}
        records[str(training_seed)] = {"full": full, "ablated": ablated, "delta_ablated_minus_full": delta}

    aggregate: dict[str, Any] = {"full": {}, "ablated": {}, "delta_ablated_minus_full": {}}
    for treatment in aggregate:
        for metric in METRICS:
            values = [record[treatment][metric] if treatment == "delta_ablated_minus_full" else record[treatment]["means"][metric] for record in records.values()]
            aggregate[treatment][metric] = {
                "mean_across_training_seeds": mean(values),
                "sd_across_training_seeds": stdev(values),
                "per_training_seed": values,
            }
    return {
        "experiment": "MAPPO financial_stability reward-component ablation",
        "component_zeroed": "financial_stability",
        "training_seed_count": len(records),
        "statistical_unit": "training_seed; evaluation episodes averaged within each checkpoint first",
        "training": {"algorithm": "MAPPO", "episodes_per_training_seed": 100, "training_seeds": [23, 123, 223]},
        "evaluation": {"days": reference_conditions[0], "decision_mode": reference_conditions[1], "episodes_per_training_seed": len(reference_conditions[2]), "evaluation_seeds": reference_conditions[2]},
        "per_training_seed": records,
        "aggregate": aggregate,
        "interpretation": "Descriptive matched ablation with only three training seeds. Standard deviations describe between-training-seed variation (sample SD); they are not confidence intervals. Seed 223 is highly sensitive in the ablated condition. This small sample does not establish a stable causal effect, convergence, or generalization. Reward totals are omitted because the component differs by design.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = summarize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for metric in ("survival_rate", "wellbeing", "mean_health", "mean_stress", "avg_money", "hunger_incidents"):
        row = summary["aggregate"]["delta_ablated_minus_full"][metric]
        print(f"{metric}: delta={row['mean_across_training_seeds']:.3f}, training-seed SD={row['sd_across_training_seeds']:.3f}")


if __name__ == "__main__":
    main()
