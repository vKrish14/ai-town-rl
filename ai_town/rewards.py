"""Pure, separately inspectable citizen reward component calculation."""

from typing import Mapping

DEFAULT_REWARD_WEIGHTS = {
    "wellbeing_delta": 0.05,
    "starvation": 0.30,
    "survival": 0.50,
    "sustainable_work": 0.08,
    "financial_stability": 0.15,
    "employment_acquisition": 0.25,
}


def calculate_reward(
    *,
    before_wellbeing: float,
    after_wellbeing: float,
    hunger: float,
    alive: bool,
    before_money: float,
    money: float,
    employed: bool,
    was_employed: bool,
    sustainable_work: bool,
    weights: Mapping[str, float]
) -> dict[str, float]:
    """Return reward parts so the total and each trade-off can be audited."""
    return {
        "wellbeing_delta": float(
            weights["wellbeing_delta"] * (after_wellbeing - before_wellbeing)
        ),
        "starvation": float(-weights["starvation"] * max(0, hunger - 50) / 50),
        "survival": float(-weights["survival"] if not alive else 0),
        "sustainable_work": float(
            weights["sustainable_work"] if sustainable_work else 0
        ),
        "financial_stability": float(
            weights["financial_stability"]
            * (min(money, 60) - min(before_money, 60))
            / 60
        ),
        "employment_acquisition": float(
            weights["employment_acquisition"] if employed and not was_employed else 0
        ),
    }
