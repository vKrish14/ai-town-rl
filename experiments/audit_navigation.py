"""Reproducibly audit crowding and route progress during headless episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ai_town.baselines import RandomPolicy, RulePolicy
from ai_town.config import load_config
from ai_town.environment import TownParallelEnv
from ai_town.rl import load_policy_checkpoint


def audit_episode(env: TownParallelEnv, policy, mode: str, seed: int, deterministic: bool):
    observations, _ = env.reset(seed=seed)
    visited = set()
    blocked_streak = [0] * env.population
    max_blocked_streak = [0] * env.population
    max_blocked_context = [None] * env.population
    duplicate_cell_ticks = 0
    blocked_agent_ticks = 0
    unreachable_path_ticks = 0
    off_grid_agent_ticks = 0
    agent_ticks = 0

    while env.agents:
        citizens = env.town.citizens
        before = [citizen.position for citizen in citizens]
        batch = np.stack([observations[agent] for agent in env.possible_agents])
        if mode == "learned":
            actions, _ = policy.act(
                batch,
                deterministic=deterministic,
                action_masks=np.stack([env.action_mask(agent) for agent in env.possible_agents]),
            )
        else:
            actions, _ = policy.act(batch)
        observations, _, _, _, _ = env.step(
            {agent: int(actions[index]) for index, agent in enumerate(env.possible_agents)}
        )

        citizens = env.town.citizens
        after = [citizen.position for citizen in citizens]
        cell_counts = Counter(after)
        duplicate_cell_ticks += int(any(count > 1 for count in cell_counts.values()))
        visited.update(after)
        for index, citizen in enumerate(citizens):
            agent_ticks += 1
            x, y = citizen.position
            off_grid_agent_ticks += int(not (0 <= x < env.town.width and 0 <= y < env.town.height))
            route_pending = citizen.target is not None and citizen.position != citizen.target
            if route_pending and before[index] == after[index]:
                blocked_agent_ticks += 1
                blocked_streak[index] += 1
            else:
                blocked_streak[index] = 0
            if blocked_streak[index] > max_blocked_streak[index]:
                max_blocked_streak[index] = blocked_streak[index]
                max_blocked_context[index] = {
                    "citizen": citizen.id,
                    "name": citizen.name,
                    "position": citizen.position,
                    "target": citizen.target,
                    "action": citizen.pending_action,
                    "path_length": len(citizen.path),
                    "first_path_cell": citizen.path[0] if citizen.path else None,
                }
            unreachable_path_ticks += int(route_pending and not citizen.path)

    citizens = env.town.citizens
    return {
        "seed": seed,
        "days": env.town.max_days,
        "duplicate_cell_ticks": duplicate_cell_ticks,
        "blocked_agent_ticks": blocked_agent_ticks,
        "blocked_agent_tick_rate": blocked_agent_ticks / max(1, agent_ticks),
        "max_consecutive_blocked_ticks": max(max_blocked_streak, default=0),
        "max_blocked_context": max(
            (context for context in max_blocked_context if context),
            key=lambda context: max_blocked_streak[context["citizen"]],
            default=None,
        ),
        "unreachable_path_ticks": unreachable_path_ticks,
        "off_grid_agent_ticks": off_grid_agent_ticks,
        "unique_cells_visited": len(visited),
        "max_cell_occupancy": max(Counter(c.position for c in citizens).values(), default=0),
        "survival_rate": float(np.mean([c.health > 0 for c in citizens])),
        "employment_rate": float(np.mean([c.employed for c in citizens])),
        "mean_wellbeing": float(np.mean([env.town._wellbeing(c) for c in citizens])),
        "mean_health": float(np.mean([c.health for c in citizens])),
        "mean_hunger": float(np.mean([c.hunger for c in citizens])),
        "min_cash": float(min(c.money for c in citizens)),
    }


def run_audit(mode="rule", checkpoint=None, episodes=3, days=60, seed=6001, output=None, deterministic=False):
    config = load_config()
    population = int(config["simulation"]["population"])
    torch.set_num_threads(int(config["training"].get("cpu_threads", 1)))
    if mode == "random":
        policy = RandomPolicy()
    elif mode == "rule":
        policy = RulePolicy()
    else:
        if not checkpoint:
            raise ValueError("learned audit requires --checkpoint")
        policy, metadata = load_policy_checkpoint(checkpoint, population)
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env = TownParallelEnv(
            population,
            days,
            config["reward"],
            config["simulation"]["ticks_per_day"],
            config["economy"],
            config["simulation"].get("movement_steps_per_tick", 3),
        )
        rows.append(audit_episode(env, policy, mode, episode_seed, deterministic))
    result = {
        "mode": mode,
        "checkpoint": checkpoint,
        "deterministic": deterministic,
        "population": population,
        "episodes": rows,
        "seed_start": seed,
        "days": days,
    }
    destination = Path(output or f"results/navigation_{mode}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("random", "rule", "learned"), default="rule")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--output")
    parser.add_argument("--deterministic", action="store_true")
    arguments = parser.parse_args()
    run_audit(**vars(arguments))
