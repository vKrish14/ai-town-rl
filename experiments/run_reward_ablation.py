"""Train/evaluate a reward-component ablation against the matching full model."""

import argparse
from pathlib import Path
import yaml
from ai_town.config import load_config
from train import train
from evaluate import evaluate


def run(
    component="financial_stability",
    algorithm="MAPPO",
    episodes=20,
    days=30,
    seed=37,
    eval_episodes=5,
):
    if component not in (
        "wellbeing_delta",
        "starvation",
        "survival",
        "sustainable_work",
        "financial_stability",
        "employment_acquisition",
    ):
        raise ValueError("unknown reward component")
    base = load_config()
    ablated = {section: dict(values) for section, values in base.items()}
    ablated["reward"][component] = 0.0
    configs = Path("experiments/configs")
    configs.mkdir(parents=True, exist_ok=True)
    config_path = configs / f"{algorithm.lower()}_without_{component}.yaml"
    config_path.write_text(yaml.safe_dump(ablated, sort_keys=False), encoding="utf-8")
    checkpoint = (
        Path("models/ablations") / f"{algorithm.lower()}_without_{component}.pt"
    )
    train(episodes, seed, days, str(checkpoint), str(config_path), algorithm)
    reference = Path("models") / f"{algorithm.lower()}.pt"
    results = Path("results/ablations")
    results.mkdir(parents=True, exist_ok=True)
    full = evaluate(
        "learned",
        eval_episodes,
        days,
        1001,
        str(reference),
        str(results / f"{algorithm.lower()}_full.json"),
    )
    ablated_result = evaluate(
        "learned",
        eval_episodes,
        days,
        1001,
        str(checkpoint),
        str(results / f"{algorithm.lower()}_without_{component}.json"),
        str(config_path),
    )
    return full, ablated_result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--component", default="financial_stability")
    p.add_argument("--algorithm", choices=["PPO", "MAPPO"], default="MAPPO")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--eval-episodes", type=int, default=5)
    a = p.parse_args()
    run(a.component, a.algorithm, a.episodes, a.days, a.seed, a.eval_episodes)
