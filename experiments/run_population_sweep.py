"""Run matched Rule/PPO/MAPPO evaluations across supported town sizes."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from evaluate import evaluate


def run(episodes=5, days=30, seed=5001, output_dir=Path("results/post_navigation_fix/population_sweeps")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "ppo": Path("models/ppo.pt"),
        "mappo": Path("models/mappo.pt"),
    }
    results = []
    for population in (1, 5, 10):
        for mode in ("rule", "ppo", "mappo"):
            result_path = output_dir / f"{mode}_pop{population}.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(
                    mode="rule" if mode == "rule" else "learned",
                    episodes=episodes,
                    days=days,
                    seed=seed,
                    checkpoint=str(checkpoints[mode]) if mode != "rule" else None,
                    output=result_path,
                    population=population,
                )
            results.append(result_path)
            print(f"saved {mode} population={population}: {result_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--output-dir", type=Path, default=Path("results/post_navigation_fix/population_sweeps"))
    args = parser.parse_args()
    run(**vars(args))
