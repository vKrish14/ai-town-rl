"""Seeded headless comparisons of random, hand-coded, or learned policies."""

import argparse, json
from pathlib import Path
import numpy as np
from ai_town.environment import TownParallelEnv
from ai_town.baselines import RandomPolicy, RulePolicy
from ai_town.rl import load_policy_checkpoint
from ai_town.config import load_config


def gini(values):
    """Return the nonnegative-income Gini coefficient, or zero for no wealth."""
    x = np.sort(np.maximum(0, np.asarray(values, dtype=float)))
    if not len(x) or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def evaluate(
    mode="rule",
    episodes=3,
    days=30,
    seed=101,
    checkpoint=None,
    output=None,
    config_path=None,
    deterministic=False,
    population=None,
):
    config = load_config(config_path)
    population = int(config["simulation"]["population"] if population is None else population)
    if not 1 <= population <= 17:
        raise ValueError("population must be between 1 and 17")
    config["simulation"]["population"] = population
    config["simulation"]["episode_days"] = days
    import torch

    torch.set_num_threads(int(config["training"].get("cpu_threads", 1)))
    checkpoint_metadata = None
    if mode == "random":
        policy = RandomPolicy()
    elif mode == "rule":
        policy = RulePolicy()
    else:
        if not checkpoint:
            raise ValueError("learned mode requires --checkpoint")
        policy, checkpoint_data = load_policy_checkpoint(checkpoint, population)
        checkpoint_metadata = checkpoint_data.get("metadata", {})
        config["training"]["algorithm"] = policy.algorithm
    rows = []
    for ep in range(episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)
        env = TownParallelEnv(
            population,
            days,
            config["reward"],
            config["simulation"]["ticks_per_day"],
            config["economy"],
            config["simulation"].get("movement_steps_per_tick", 3),
        )
        obs, _ = env.reset(seed + ep)
        total = np.zeros(population)
        while env.agents:
            x = np.stack([obs[a] for a in env.possible_agents])
            if mode == "learned":
                act, _ = policy.act(
                    x,
                    deterministic=deterministic,
                    action_masks=np.stack(
                        [env.action_mask(a) for a in env.possible_agents]
                    ),
                )
            else:
                act, _ = policy.act(x)
            obs, r, t, _, _ = env.step(
                {a: int(act[i]) for i, a in enumerate(env.possible_agents)}
            )
            total += np.array([r[a] for a in env.possible_agents])
        cs = env.town.citizens
        rows.append(
            {
                "seed": seed + ep,
                "reward": float(total.mean()),
                "wellbeing": float(np.mean([env.town._wellbeing(c) for c in cs])),
                "survival_rate": float(np.mean([c.health > 0 for c in cs])),
                "employment_rate": float(np.mean([c.employed for c in cs])),
                "avg_money": float(np.mean([c.money for c in cs])),
                "hunger_incidents": int(sum(c.hunger > 80 for c in cs)),
                "mean_health": float(np.mean([c.health for c in cs])),
                "mean_stress": float(np.mean([c.stress for c in cs])),
                "mean_social_satisfaction": float(np.mean([c.social for c in cs])),
                "wealth_gini": gini([c.money for c in cs]),
            }
        )
    label = policy.mode if hasattr(policy, "mode") else policy.__class__.__name__
    algorithm = policy.algorithm if hasattr(policy, "algorithm") else None
    result = {
        "mode": label,
        "algorithm": algorithm,
        "checkpoint": checkpoint,
        "seed_start": seed,
        "days": days,
        "config": config,
        "checkpoint_metadata": checkpoint_metadata,
        "deterministic": deterministic,
        "decision_mode": (
            "deterministic_argmax" if deterministic else "stochastic_sample"
        ),
        "episodes": rows,
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
    }
    print(json.dumps(result, indent=2))
    filename = algorithm.lower() if algorithm else mode
    out = Path(output or f"results/{filename}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "rule", "learned"], default="rule")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--population", type=int, help="override population for headless evaluation (1–16)")
    p.add_argument("--checkpoint")
    p.add_argument("--output")
    p.add_argument("--config")
    decision = p.add_mutually_exclusive_group()
    decision.add_argument("--deterministic", action="store_true")
    decision.add_argument("--stochastic", action="store_true")
    a = p.parse_args()
    evaluate(
        a.mode,
        a.episodes,
        a.days,
        a.seed,
        a.checkpoint,
        a.output,
        a.config,
        deterministic=a.deterministic,
        population=a.population,
    )
