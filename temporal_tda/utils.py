"""Shared utilities for temporal-tda and xplain-pipeline."""

from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def load_config(
    config_name: str = "config.toml",
    start_dir: str | Path | None = None,
    max_levels: int = 2,
) -> Dict[str, Any]:
    """
    Search upward from *start_dir* (default: cwd) for a TOML config file.

    Parameters
    ----------
    config_name : str
        Filename to look for.
    start_dir : str | Path | None
        Starting directory.  ``None`` → current working directory.
    max_levels : int
        Maximum number of parent directories to check (capped at 4).
    """
    max_levels = min(max_levels, 4)
    current_dir = Path(start_dir).resolve() if start_dir else Path.cwd()

    for _ in range(max_levels + 1):
        config_path = current_dir / config_name
        if config_path.exists():
            with config_path.open("rb") as f:
                return tomllib.load(f)

        parent = current_dir.parent
        if current_dir == parent:
            break
        current_dir = parent

    raise FileNotFoundError(
        f"Could not find '{config_name}' within {max_levels} levels "
        f"upwards from {Path.cwd()}."
    )
