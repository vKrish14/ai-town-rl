import json

import pytest

from evaluate import evaluate


def test_headless_evaluation_population_override_is_recorded(tmp_path):
    output = tmp_path / "population_one.json"
    result = evaluate(
        mode="rule",
        episodes=1,
        days=1,
        seed=42,
        output=output,
        population=1,
    )

    assert result["config"]["simulation"]["population"] == 1
    assert len(result["episodes"]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["episodes"][0]["seed"] == 42


@pytest.mark.parametrize("population", [0, 18, -2])
def test_headless_evaluation_rejects_population_out_of_range(population, tmp_path):
    with pytest.raises(ValueError, match="population must be between 1 and 17"):
        evaluate(mode="rule", days=1, population=population, output=tmp_path / "invalid.json")
