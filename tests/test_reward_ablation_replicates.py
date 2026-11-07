import pytest

from experiments.run_reward_ablation_replicates import _mean_metrics


def test_ablation_means_keep_each_outcome_component_separate():
    result = {
        "episodes": [
            {"survival_rate": 1, "employment_rate": 0.8, "wellbeing": 80,
             "mean_health": 90, "mean_stress": 10, "mean_social_satisfaction": 70,
             "avg_money": 12, "hunger_incidents": 1},
            {"survival_rate": 0.8, "employment_rate": 0.6, "wellbeing": 60,
             "mean_health": 70, "mean_stress": 30, "mean_social_satisfaction": 50,
             "avg_money": 8, "hunger_incidents": 3},
        ]
    }
    means = _mean_metrics(result)
    assert means["survival_rate"] == pytest.approx(0.9)
    assert means["wellbeing"] == pytest.approx(70)
    assert means["mean_health"] == pytest.approx(80)
    assert means["mean_stress"] == pytest.approx(20)
    assert means["hunger_incidents"] == pytest.approx(2)
