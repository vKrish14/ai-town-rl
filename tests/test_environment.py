import numpy as np
from pettingzoo.test import parallel_api_test
from ai_town.environment import TownParallelEnv
from ai_town.baselines import RulePolicy
from ai_town.config import load_config


def test_parallel_api():
    parallel_api_test(TownParallelEnv(3, 1), num_cycles=30)


def test_spaces_rewards_and_termination():
    env = TownParallelEnv(2, 1)
    obs, _ = env.reset(seed=22)
    assert all(env.observation_space(a).contains(o) for a, o in obs.items())
    while env.agents:
        obs, rewards, terms, truncs, infos = env.step({a: 2 for a in env.agents})
        assert all(np.isfinite(list(rewards.values())))
        assert all(
            np.all(np.isfinite(list(infos[a]["reward_components"].values())))
            for a in env.possible_agents
        )
    assert all(terms.values())


def test_seed_reproducibility():
    a = TownParallelEnv(2, 1)
    b = TownParallelEnv(2, 1)
    oa, _ = a.reset(seed=8)
    ob, _ = b.reset(seed=8)
    assert np.array_equal(oa["citizen_0"], ob["citizen_0"])
    for _ in range(4):
        acts = {x: 2 for x in a.agents}
        oa, *_ = a.step(acts)
        ob, *_ = b.step(acts)
    assert np.array_equal(oa["citizen_0"], ob["citizen_0"])


def test_navigation_commitment_exposes_action_mask():
    env = TownParallelEnv(2, 1)
    env.reset(seed=1)
    assert np.all(env.action_mask("citizen_0") == 1)
    env.step({"citizen_0": 0, "citizen_1": 0})
    mask = env.action_mask("citizen_0")
    assert mask.sum() == 1 and mask[0] == 1
    env.step({"citizen_0": 1, "citizen_1": 1})
    assert env.town.citizens[0].pending_action == 0


def test_observation_remains_normalized_under_large_economy():
    env = TownParallelEnv(
        2, 1, economy={"shop_stock": 500, "starting_vacancies": 40, "starting_food": 20}
    )
    obs, _ = env.reset(seed=4)
    assert all(np.min(v) >= 0 and np.max(v) <= 1 for v in obs.values())


def test_default_rule_control_sustains_a_full_30_day_town():
    config = load_config()
    sim = config["simulation"]
    env = TownParallelEnv(
        sim["population"],
        30,
        config["reward"],
        sim["ticks_per_day"],
        config["economy"],
        sim["movement_steps_per_tick"],
    )
    policy = RulePolicy()
    obs, _ = env.reset(seed=101)
    while env.agents:
        actions, _ = policy.act(np.stack([obs[a] for a in env.possible_agents]))
        obs, *_ = env.step(
            {a: int(actions[i]) for i, a in enumerate(env.possible_agents)}
        )
    assert all(c.alive for c in env.town.citizens)
    assert all(c.employed for c in env.town.citizens)
    assert all(c.hunger <= 80 for c in env.town.citizens)


def test_parallel_social_rewards_use_the_same_step_snapshot():
    env = TownParallelEnv(2, 1)
    env.reset(seed=8)
    meeting_point = (17, 10)
    for citizen in env.town.citizens:
        citizen.position = meeting_point
    _, _, _, _, infos = env.step({a: 3 for a in env.agents})
    assert infos["citizen_0"]["reward_components"]["wellbeing_delta"] > 0
    assert infos["citizen_1"]["reward_components"]["wellbeing_delta"] > 0
