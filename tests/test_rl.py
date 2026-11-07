import numpy as np, torch
from ai_town.rl import (
    MAPPOPolicy,
    SharedPPOPolicy,
    compute_gae,
    update_mappo,
    update_shared_ppo,
)
from ai_town.simulation import OBS_DIM
from evaluate import evaluate


def test_gae_and_mappo_update_change_parameters():
    policy = MAPPOPolicy(2)
    before = [p.detach().clone() for p in policy.actor.parameters()]
    obs = np.random.rand(5, 2, OBS_DIM).astype("float32")
    states = np.random.rand(5, 2 * OBS_DIM).astype("float32")
    acts = np.random.randint(0, 8, (5, 2))
    logp = policy.act(obs.reshape(-1, OBS_DIM))[1].reshape(5, 2)
    with torch.no_grad():
        values = policy.critic(torch.tensor(states)).numpy()
    adv, ret = compute_gae(np.random.randn(5, 2), values, np.array([0, 0, 0, 0, 1]))
    loss = update_mappo(
        policy,
        dict(
            obs=obs,
            state=states,
            actions=acts,
            logp=logp,
            action_masks=np.ones((5, 2, 8)),
            advantages=adv,
            returns=ret,
        ),
    )
    assert all(np.isfinite(list(loss.values())))
    assert any(not torch.equal(p, b) for p, b in zip(policy.actor.parameters(), before))


def test_checkpoint_roundtrip(tmp_path):
    policy = MAPPOPolicy(2)
    path = tmp_path / "policy.pt"
    policy.save(path, {"seed": 9})
    restored = MAPPOPolicy(2)
    meta = restored.load(path)
    x = np.zeros((2, OBS_DIM), dtype=np.float32)
    assert np.array_equal(policy.act(x, True)[0], restored.act(x, True)[0])
    assert meta["metadata"]["seed"] == 9


def test_shared_ppo_update_is_finite():
    policy = SharedPPOPolicy(2)
    obs = np.random.rand(4, 2, OBS_DIM).astype("float32")
    flat = obs.reshape(-1, OBS_DIM)
    with torch.no_grad():
        values = policy.critic(torch.tensor(flat)).numpy().reshape(4, 2)
    adv, ret = compute_gae(np.random.randn(4, 2), values, np.array([0, 0, 0, 1]))
    logp = policy.act(flat)[1].reshape(4, 2)
    out = update_shared_ppo(
        policy,
        {
            "obs": obs,
            "actions": np.random.randint(0, 8, (4, 2)),
            "logp": logp,
            "action_masks": np.ones((4, 2, 8)),
            "advantages": adv,
            "returns": ret,
        },
    )
    assert np.isfinite(list(out.values())).all()


def test_stochastic_evaluation_is_reproducible(tmp_path, capsys):
    checkpoint = tmp_path / "policy.pt"
    MAPPOPolicy(10).save(checkpoint, {"seed": 3})
    first = evaluate("learned", 1, 1, 40, str(checkpoint), str(tmp_path / "first.json"))
    second = evaluate(
        "learned", 1, 1, 40, str(checkpoint), str(tmp_path / "second.json")
    )
    capsys.readouterr()
    assert first["decision_mode"] == "stochastic_sample"
    assert first["episodes"] == second["episodes"]


def test_mappo_checkpoint_actor_evaluates_at_smaller_population(tmp_path, capsys):
    checkpoint = tmp_path / "mappo_ten.pt"
    MAPPOPolicy(10).save(checkpoint, {"seed": 3})
    result = evaluate(
        "learned", 1, 1, 41, str(checkpoint), str(tmp_path / "population_one.json"),
        population=1,
    )
    capsys.readouterr()
    assert result["config"]["simulation"]["population"] == 1
    assert result["episodes"][0]["seed"] == 41
