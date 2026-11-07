"""Summarize paired environment-seed evaluations with small-sample intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

METRICS = (
    "reward",
    "wellbeing",
    "survival_rate",
    "employment_rate",
    "mean_health",
    "mean_stress",
    "mean_social_satisfaction",
    "wealth_gini",
    "hunger_incidents",
)
T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}
METHODS = ("random", "rule", "ppo_seed29", "ppo_seed129", "ppo_seed229", "mappo_seed23", "mappo_seed123", "mappo_seed223")


def summarize(root: Path, output: Path) -> list[dict]:
    older = root
    newer = root / "seeds_7001_7010"
    summaries = []
    for method in METHODS:
        rows = []
        for directory in (older, newer):
            data = json.loads((directory / f"{method}.json").read_text(encoding="utf-8"))
            rows.extend(data["episodes"])
        rows.sort(key=lambda row: row["seed"])
        record = {"policy": method, "n_env_seeds": len(rows), "seed_min": rows[0]["seed"], "seed_max": rows[-1]["seed"]}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            center = mean(values)
            radius = T_975.get(len(values) - 1, 1.96) * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = center
            record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
            lower, upper = center - radius, center + radius
            if metric == "survival_rate":
                lower, upper = max(0.0, lower), min(1.0, upper)
            record[f"{metric}_ci95_low"] = lower
            record[f"{metric}_ci95_high"] = upper
        summaries.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/post_navigation_fix"))
    parser.add_argument("--output", type=Path, default=Path("results/post_navigation_fix/summary.csv"))
    args = parser.parse_args()
    for row in summarize(args.root, args.output):
        print(f"{row['policy']:14} reward {row['reward_mean']:+.2f} +/- {row['reward_sd']:.2f}  wellbeing {row['wellbeing_mean']:.2f} +/- {row['wellbeing_sd']:.2f}  survival {row['survival_rate_mean']:.1%}")
