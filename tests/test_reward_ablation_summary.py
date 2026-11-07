import json

import pytest

from experiments.summarize_reward_ablation_replicates import summarize_saved_replicates


def _write_eval(path, seed, arm, wellbeing):
    reward = {"financial_stability": 0.15 if arm == "full" else 0.0}
    report = {
        "algorithm": "MAPPO",
        "days": 30,
        "decision_mode": "stochastic_sample",
        "deterministic": False,
        "checkpoint_metadata": {"seed": seed},
        "config": {"reward": reward},
        "episodes": [
            {"seed": 1001, "wellbeing": wellbeing, "survival_rate": 1.0,
             "employment_rate": 1.0, "mean_health": 90.0, "mean_stress": 10.0,
             "mean_social_satisfaction": 80.0, "avg_money": 25.0,
             "hunger_incidents": 0, "wealth_gini": 0.2}
        ],
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_saved_ablation_summary_averages_environment_then_training_seed(tmp_path):
    pairs = {}
    for seed, full_wellbeing, ablated_wellbeing in ((23, 80, 70), (123, 90, 40)):
        full, ablated = tmp_path / f"{seed}_full.json", tmp_path / f"{seed}_ablated.json"
        _write_eval(full, seed, "full", full_wellbeing)
        _write_eval(ablated, seed, "ablated", ablated_wellbeing)
        pairs[seed] = full, ablated

    result = summarize_saved_replicates(pairs, tmp_path / "summary.json")

    assert result["training_seeds"] == [23, 123]
    assert result["aggregate"]["full"]["wellbeing"]["mean_across_training_seeds"] == 85
    assert result["aggregate"]["ablated"]["wellbeing"]["mean_across_training_seeds"] == 55
    assert result["aggregate"]["delta_ablated_minus_full"]["wellbeing"]["sd_across_training_seeds"] == pytest.approx(28.2842712)
    assert (tmp_path / "summary.json").exists()


def test_saved_ablation_summary_rejects_unpaired_evaluation_seeds(tmp_path):
    full, ablated = tmp_path / "full.json", tmp_path / "ablated.json"
    full_2, ablated_2 = tmp_path / "full_2.json", tmp_path / "ablated_2.json"
    _write_eval(full, 23, "full", 80)
    _write_eval(ablated, 23, "ablated", 70)
    _write_eval(full_2, 123, "full", 80)
    _write_eval(ablated_2, 123, "ablated", 70)
    report = json.loads(ablated_2.read_text())
    report["episodes"][0]["seed"] = 1002
    ablated_2.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="not paired"):
        summarize_saved_replicates({23: (full, ablated), 123: (full_2, ablated_2)}, tmp_path / "summary.json")
