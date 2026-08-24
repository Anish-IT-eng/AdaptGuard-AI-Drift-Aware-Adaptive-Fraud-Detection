"""
AdaptGuard AI — Config Loader
Loads and validates the YAML config file.
"""

from pathlib import Path
from typing import Any
import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load project configuration from YAML.

    Args:
        path: Path to config.yaml. Defaults to configs/config.yaml.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely navigate nested config dict.

    Example:
        val = get(cfg, "drift", "adwin", "delta", default=0.002)
    """
    result = cfg
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
