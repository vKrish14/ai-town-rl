import pytest

from experiments.run_economy_sweep import _summary


def test_economy_sweep_summary_reports_seed_variation():
    result = {
        "episodes": [
            {"survival_rate": 1.0, "employment_rate": 0.8, "wellbeing": 70,
             "mean_health": 90, "mean_stress": 20, "mean_social_satisfaction": 60,
             "avg_money": 12, "hunger_incidents": 2},
            {"survival_rate": 0.8, "employment_rate": 0.6, "wellbeing": 50,
             "mean_health": 70, "mean_stress": 40, "mean_social_satisfaction": 40,
             "avg_money": 8, "hunger_incidents": 6},
        ]
    }
    summary = _summary(result)
    assert summary["survival_rate"]["mean"] == pytest.approx(0.9)
    assert summary["survival_rate"]["sd"] == pytest.approx(0.1414213562)
    assert summary["wellbeing"]["min"] == 50
    assert summary["hunger_incidents"]["max"] == 6
