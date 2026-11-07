"""Summarize matched population-scale evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = ("reward", "wellbeing", "survival_rate", "employment_rate", "mean_health", "mean_stress", "hunger_incidents")


def summarize(root: Path, output: Path):
    rows = []
    for population in (1, 5, 10):
        for mode in ("rule", "ppo", "mappo"):
            data = json.loads((root / f"{mode}_pop{population}.json").read_text(encoding="utf-8"))
            episode_rows = data["episodes"]
            result = {"mode": mode, "population": population, "episodes": len(episode_rows), "seed_start": episode_rows[0]["seed"], "seed_end": episode_rows[-1]["seed"]}
            for metric in METRICS:
                values = [float(row[metric]) for row in episode_rows]
                result[f"{metric}_mean"] = mean(values)
                result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
            rows.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/post_navigation_fix/population_sweeps"))
    parser.add_argument("--output", type=Path, default=Path("results/post_navigation_fix/population_summary.csv"))
    args = parser.parse_args()
    for row in summarize(args.root, args.output):
        print(f"{row['mode']:5} population {row['population']:2}  survival {row['survival_rate_mean']:.1%}  wellbeing {row['wellbeing_mean']:.2f} +/- {row['wellbeing_sd']:.2f}  reward {row['reward_mean']:+.2f} +/- {row['reward_sd']:.2f}")
