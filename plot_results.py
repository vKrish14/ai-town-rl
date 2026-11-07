"""Summarize the saved multi-seed evaluation JSON files as CSV and PNG."""

import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(directory="results"):
    root = Path(directory)
    data = []
    for mode in ("random", "rule", "ppo", "mappo"):
        path = root / f"{mode}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run evaluate.py for {mode} first."
            )
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = result["episodes"]
        data.append(
            {
                "mode": result["mode"],
                "mean_reward": float(np.mean([r["reward"] for r in rows])),
                "std_reward": (
                    float(np.std([r["reward"] for r in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0
                ),
                "mean_wellbeing": float(np.mean([r["wellbeing"] for r in rows])),
                "survival_rate": float(np.mean([r["survival_rate"] for r in rows])),
                "employment_rate": float(np.mean([r["employment_rate"] for r in rows])),
                "mean_health": float(np.mean([r["mean_health"] for r in rows])),
                "mean_stress": float(np.mean([r["mean_stress"] for r in rows])),
                "mean_social_satisfaction": float(
                    np.mean([r["mean_social_satisfaction"] for r in rows])
                ),
                "mean_wealth_gini": float(np.mean([r["wealth_gini"] for r in rows])),
                "seeds": len(rows),
            }
        )
    root.mkdir(exist_ok=True)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [r["mode"] for r in data]
    for ax, key, title in zip(
        axes,
        ("mean_reward", "mean_wellbeing", "survival_rate"),
        ("Episode reward (mean ± SD)", "Final average wellbeing", "Survival rate"),
    ):
        vals = [r[key] for r in data]
        err = [r["std_reward"] if key == "mean_reward" else 0 for r in data]
        ax.bar(
            labels,
            vals,
            yerr=err,
            capsize=4,
            color=["#92a77b", "#d4ad6d", "#6f9eb2", "#ad83a5"],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    print(f'Wrote {root / "summary.csv"} and {root / "comparison.png"}')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="results")
    main(p.parse_args().directory)
