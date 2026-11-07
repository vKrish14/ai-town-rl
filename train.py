"""Train shared actor PPO with a centralized MAPPO critic."""

import argparse, json, random
from pathlib import Path
import numpy as np, torch
from ai_town.environment import TownParallelEnv
from ai_town.rl import (
    MAPPOPolicy,
    SharedPPOPolicy,
    compute_gae,
    update_mappo,
    update_shared_ppo,
)
from ai_town.config import load_config


def train(
    episodes=20,
    seed=7,
    days=30,
    output="models/mappo.pt",
    config_path=None,
    algorithm=None,
    metrics_output=None,
):
    config = load_config(config_path)
    population = int(config["simulation"]["population"])
    algo = (algorithm or config["training"]["algorithm"]).upper()
    if algo not in ("PPO", "MAPPO"):
        raise ValueError("algorithm must be PPO or MAPPO")
    config["training"]["algorithm"] = algo
    config["training"]["seed"] = seed
    config["simulation"]["episode_days"] = days
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["training"].get("cpu_threads", 1)))
    env = TownParallelEnv(
        population,
        days,
        config["reward"],
        config["simulation"]["ticks_per_day"],
        config["economy"],
        config["simulation"].get("movement_steps_per_tick", 3),
    )
    policy = (SharedPPOPolicy if algo == "PPO" else MAPPOPolicy)(population)
    all_metrics = []
    obs, _ = env.reset(seed=seed)
    ep = 0
    while ep < episodes:
        ro = {
            k: []
            for k in (
                "obs",
                "state",
                "actions",
                "logp",
                "action_masks",
                "rewards",
                "values",
                "dones",
            )
        }
        ep_reward = np.zeros(population)
        ep_steps = 0
        while True:
            x = np.stack([obs[a] for a in env.possible_agents])
            state = env.state()
            with torch.no_grad():
                inp = torch.as_tensor(
                    x if algo == "PPO" else state[None], dtype=torch.float32
                )
                value = (
                    policy.critic(inp).cpu().numpy()
                    if algo == "PPO"
                    else policy.critic(inp).cpu().numpy()[0]
                )
            masks = np.stack([env.action_mask(a) for a in env.possible_agents])
            act, logp = policy.act(x, action_masks=masks)
            acts = {a: int(act[i]) for i, a in enumerate(env.possible_agents)}
            nxt, reward, term, trunc, info = env.step(acts)
            ro["obs"].append(x)
            ro["state"].append(state)
            ro["actions"].append(act)
            ro["logp"].append(logp)
            ro["action_masks"].append(masks)
            ro["rewards"].append([reward[a] for a in env.possible_agents])
            ro["values"].append(value)
            ro["dones"].append(float(all(term.values())))
            ep_reward += np.array([reward[a] for a in env.possible_agents])
            ep_steps += 1
            obs = nxt
            if all(term.values()):
                break
        tr = config["training"]
        adv, ret = compute_gae(
            ro["rewards"], ro["values"], ro["dones"], tr["gamma"], tr["gae_lambda"]
        )
        rollout = {
            k: np.asarray(ro[k])
            for k in ("obs", "state", "actions", "logp", "action_masks")
        }
        rollout.update(advantages=adv, returns=ret)
        updater = update_shared_ppo if algo == "PPO" else update_mappo
        losses = updater(
            policy,
            rollout,
            epochs=tr["update_epochs"],
            clip=tr["clip"],
            entropy_coef=tr["entropy"],
            lr=tr["learning_rate"],
        )
        ep += 1
        row = {
            "episode": ep,
            "mean_reward": float(ep_reward.mean()),
            "mean_wellbeing": float(
                np.mean(
                    [
                        sum(
                            [
                                c.health,
                                c.energy,
                                100 - c.hunger,
                                c.happiness,
                                100 - c.stress,
                                c.social,
                            ]
                        )
                        / 6
                        for c in env.town.citizens
                    ]
                )
            ),
            **losses,
        }
        all_metrics.append(row)
        print(json.dumps(row))
        if ep % 10 == 0:
            policy.save(
                output,
                {
                    "seed": seed,
                    "episodes_completed": ep,
                    "target_episodes": episodes,
                    "days": days,
                    "algorithm": algo,
                    "config": config,
                },
            )
        if all(term.values()):
            obs, _ = env.reset(seed=seed + ep)
    policy.save(
        output,
        {
            "seed": seed,
            "episodes": episodes,
            "days": days,
            "algorithm": algo,
            "config": config,
        },
    )
    Path("logs").mkdir(exist_ok=True)
    metrics_path = Path(metrics_output or f"logs/train_{algo.lower()}.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "".join(json.dumps(x) + "\n" for x in all_metrics), encoding="utf-8"
    )
    return policy, all_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--output", default="models/mappo.pt")
    p.add_argument("--config")
    p.add_argument("--metrics-output", help="separate training trace output path")
    p.add_argument("--algorithm", choices=["PPO", "MAPPO"])
    a = p.parse_args()
    train(
        a.episodes,
        a.seed,
        a.days,
        a.output,
        config_path=a.config,
        algorithm=a.algorithm,
        metrics_output=a.metrics_output,
    )
