"""Inspectable shared-actor PPO/MAPPO implementation (PyTorch)."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from ai_town.simulation import OBS_DIM, DEFAULT_POPULATION


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, actions=8, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, actions),
        )

    def forward(self, x):
        return self.net(x)


class CentralCritic(nn.Module):
    """Centralized training critic maps global state to one value per agent."""

    def __init__(self, state_dim=200, agents=10, width=128):
        super().__init__()
        self.agents = agents
        self.net = nn.Sequential(
            nn.Linear(state_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, agents),
        )

    def forward(self, x):
        return self.net(x)


class LocalCritic(nn.Module):
    """Decentralized value baseline used by the shared-policy PPO stage."""

    def __init__(self, obs_dim=OBS_DIM, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MAPPOPolicy:
    algorithm = "MAPPO"
    mode = "Learned MAPPO"

    def __init__(self, population=DEFAULT_POPULATION, obs_dim=OBS_DIM, device="cpu"):
        self.population = population
        self.device = torch.device(device)
        self.actor = Actor(obs_dim).to(self.device)
        self.critic = CentralCritic(population * obs_dim, population).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    @torch.no_grad()
    def act(self, observations, deterministic=False, action_masks=None):
        x = torch.as_tensor(
            np.asarray(observations), dtype=torch.float32, device=self.device
        )
        logits = self.actor(x)
        if action_masks is not None:
            logits = logits.masked_fill(
                torch.as_tensor(action_masks, device=self.device) <= 0, -1e9
            )
        dist = Categorical(logits=logits)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy()

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "population": self.population,
                "obs_dim": OBS_DIM,
                "algorithm": "MAPPO",
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        return data


class SharedPPOPolicy:
    """Parameter-shared actor and decentralized per-agent value critic."""

    algorithm = "PPO"
    mode = "Learned shared PPO"

    def __init__(self, population=DEFAULT_POPULATION, obs_dim=OBS_DIM, device="cpu"):
        self.population = population
        self.device = torch.device(device)
        self.actor = Actor(obs_dim).to(self.device)
        self.critic = LocalCritic(obs_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    @torch.no_grad()
    def act(self, observations, deterministic=False, action_masks=None):
        x = torch.as_tensor(
            np.asarray(observations), dtype=torch.float32, device=self.device
        )
        logits = self.actor(x)
        if action_masks is not None:
            logits = logits.masked_fill(
                torch.as_tensor(action_masks, device=self.device) <= 0, -1e9
            )
        dist = Categorical(logits=logits)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy()

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "population": self.population,
                "obs_dim": OBS_DIM,
                "algorithm": "PPO",
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        return data


def load_policy_checkpoint(path, population=DEFAULT_POPULATION, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    cls = SharedPPOPolicy if data.get("algorithm") == "PPO" else MAPPOPolicy
    # The centralized MAPPO critic has a population-shaped state/output layer.
    # Inference uses only the shared local actor, so retain the checkpoint's
    # training population for critic deserialization while allowing the actor
    # to serve a different environment population.
    policy_population = int(data.get("population", population))
    policy = cls(policy_population, device=device)
    policy.load(path)
    return policy, data


def update_mappo(
    policy: MAPPOPolicy,
    rollout: dict[str, np.ndarray],
    epochs=4,
    clip=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    lr=3e-4,
):
    """PPO clipped update; advantages/values are per agent from the centralized critic."""
    dev = policy.device
    obs = torch.tensor(rollout["obs"], dtype=torch.float32, device=dev).reshape(
        -1, rollout["obs"].shape[-1]
    )
    states = torch.tensor(rollout["state"], dtype=torch.float32, device=dev)
    acts = torch.tensor(rollout["actions"], dtype=torch.long, device=dev).reshape(-1)
    oldlog = torch.tensor(rollout["logp"], dtype=torch.float32, device=dev).reshape(-1)
    masks = torch.tensor(
        rollout.get("action_masks", np.ones((len(acts), 8))),
        dtype=torch.float32,
        device=dev,
    ).reshape(-1, 8)
    adv = torch.tensor(rollout["advantages"], dtype=torch.float32, device=dev).reshape(
        -1
    )
    ret = torch.tensor(rollout["returns"], dtype=torch.float32, device=dev)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    ao = policy.actor_optimizer
    co = policy.critic_optimizer
    for group in ao.param_groups:
        group["lr"] = lr
    for group in co.param_groups:
        group["lr"] = lr
    losses = []
    n = len(obs)
    nt = len(states)
    for _ in range(epochs):
        order = torch.randperm(n, device=dev)
        for ids in order.split(min(64, n)):
            logits = policy.actor(obs[ids]).masked_fill(masks[ids] <= 0, -1e9)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(acts[ids])
            ratio = (logp - oldlog[ids]).exp()
            actor_loss = (
                -torch.minimum(
                    ratio * adv[ids], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[ids]
                ).mean()
                - entropy_coef * dist.entropy().mean()
            )
            ao.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.5)
            ao.step()
            losses.append(
                (float(actor_loss.detach()), 0.0, float(dist.entropy().mean().detach()))
            )
        # Critic batches are per environment timestep and predict all agents together.
        for ids in torch.randperm(nt, device=dev).split(min(64, nt)):
            values = policy.critic(states[ids])
            value_loss = ((values - ret[ids]) ** 2).mean()
            co.zero_grad()
            (value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(policy.critic.parameters(), 0.5)
            co.step()
            losses[-1] = (losses[-1][0], float(value_loss.detach()), losses[-1][2])
    return dict(
        actor_loss=float(np.mean([v[0] for v in losses])),
        critic_loss=float(np.mean([v[1] for v in losses])),
        entropy=float(np.mean([v[2] for v in losses])),
    )


def update_shared_ppo(
    policy: SharedPPOPolicy,
    rollout: dict[str, np.ndarray],
    epochs=4,
    clip=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    lr=3e-4,
):
    dev = policy.device
    obs = torch.tensor(rollout["obs"], dtype=torch.float32, device=dev).reshape(
        -1, OBS_DIM
    )
    acts = torch.tensor(rollout["actions"], dtype=torch.long, device=dev).reshape(-1)
    oldlog = torch.tensor(rollout["logp"], dtype=torch.float32, device=dev).reshape(-1)
    masks = torch.tensor(
        rollout.get("action_masks", np.ones((len(acts), 8))),
        dtype=torch.float32,
        device=dev,
    ).reshape(-1, 8)
    adv = torch.tensor(rollout["advantages"], dtype=torch.float32, device=dev).reshape(
        -1
    )
    ret = torch.tensor(rollout["returns"], dtype=torch.float32, device=dev).reshape(-1)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    ao = policy.actor_optimizer
    co = policy.critic_optimizer
    for group in ao.param_groups:
        group["lr"] = lr
    for group in co.param_groups:
        group["lr"] = lr
    rows = []
    for _ in range(epochs):
        for ids in torch.randperm(len(obs), device=dev).split(min(64, len(obs))):
            dist = Categorical(
                logits=policy.actor(obs[ids]).masked_fill(masks[ids] <= 0, -1e9)
            )
            logp = dist.log_prob(acts[ids])
            ratio = (logp - oldlog[ids]).exp()
            aloss = (
                -torch.minimum(
                    ratio * adv[ids], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[ids]
                ).mean()
                - entropy_coef * dist.entropy().mean()
            )
            closs = ((policy.critic(obs[ids]) - ret[ids]) ** 2).mean()
            ao.zero_grad()
            aloss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.5)
            ao.step()
            co.zero_grad()
            (value_coef * closs).backward()
            nn.utils.clip_grad_norm_(policy.critic.parameters(), 0.5)
            co.step()
            rows.append(
                (
                    float(aloss.detach()),
                    float(closs.detach()),
                    float(dist.entropy().mean().detach()),
                )
            )
    return dict(
        actor_loss=float(np.mean([x[0] for x in rows])),
        critic_loss=float(np.mean([x[1] for x in rows])),
        entropy=float(np.mean([x[2] for x in rows])),
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    adv = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        nxt = np.zeros_like(gae) if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values
