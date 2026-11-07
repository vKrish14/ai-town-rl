"""Load project defaults with optional YAML overrides."""

from pathlib import Path
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path=None):
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if (
        not isinstance(config, dict)
        or "simulation" not in config
        or "reward" not in config
    ):
        raise ValueError("config must include simulation and reward sections")
    return config
