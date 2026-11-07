from ai_town.baselines import RulePolicy
from ai_town.environment import TownParallelEnv
from experiments.audit_navigation import audit_episode


def test_navigation_audit_reports_bounded_occupancy_and_progress():
    env = TownParallelEnv(population=2, max_days=1)
    result = audit_episode(env, RulePolicy(), "rule", seed=73, deterministic=False)
    assert result["days"] == 1
    assert result["off_grid_agent_ticks"] == 0
    assert result["unique_cells_visited"] > 0
    assert 0 <= result["blocked_agent_tick_rate"] <= 1
    assert result["max_cell_occupancy"] >= 1
