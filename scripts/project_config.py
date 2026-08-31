"""Load shared, analysis-area, and local project configuration."""

from __future__ import annotations

import copy
import os
import shutil
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


DATA_ROOT_ENV = "PARK_PERFORMANCE_DATA_ROOT"


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


@contextmanager
def inherited_temp_directory(parent: Path, prefix: str) -> Iterator[Path]:
    """Create an atomic-build folder that inherits the parent's permissions.

    ``tempfile.TemporaryDirectory`` uses owner-only permissions on some Windows
    installations. Moving artifacts out of such a directory preserves that ACL
    and can make the outputs unreadable to the interactive user.
    """

    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{prefix}{uuid4().hex}"
    temporary.mkdir()
    if temporary.parent.resolve() != parent:
        raise RuntimeError("Temporary build directory escaped its intended parent")
    try:
        yield temporary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream) or {}

    if not isinstance(content, dict):
        raise ConfigError(f"Configuration must contain a YAML mapping: {path}")

    return content


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def available_areas(root: Path | None = None) -> list[str]:
    root = (root or project_root()).resolve()
    city_dir = root / "config" / "cities"
    cities = sorted(path.stem for path in city_dir.glob("*.yaml"))
    return cities + ["metro"]


def area_config_path(area: str, root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()

    if area == "metro":
        return root / "config" / "metro.yaml"

    path = root / "config" / "cities" / f"{area}.yaml"
    if not path.exists():
        choices = ", ".join(available_areas(root))
        raise ConfigError(f"Unknown analysis area '{area}'. Available: {choices}")

    return path


def _resolve_from_root(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(area: str, root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()

    config = _read_yaml(root / "config" / "settings.yaml")
    config = _deep_merge(config, _read_yaml(area_config_path(area, root)))

    local_path = root / "config" / "local.yaml"
    if local_path.exists():
        config = _deep_merge(config, _read_yaml(local_path))

    local_data_root = config.get("local", {}).get("data_root")
    data_root_value = os.environ.get(DATA_ROOT_ENV) or local_data_root or "data"
    data_root = _resolve_from_root(data_root_value, root)

    analysis_area = config.get("analysis_area", {})
    slug = analysis_area.get("slug")
    if not slug:
        raise ConfigError("Missing analysis_area.slug in the selected configuration.")

    config["runtime"] = {
        "project_root": str(root),
        "data_root": str(data_root),
        "area_config": str(area_config_path(area, root)),
        "output_root": str((root / "outputs" / slug).resolve()),
    }

    return config


def data_path(config: dict[str, Any], key: str) -> Path:
    paths = config.get("paths", {})
    if key not in paths:
        raise ConfigError(f"Missing paths.{key} in configuration.")

    value = Path(paths[key]).expanduser()
    if value.is_absolute():
        return value.resolve()

    # Newly generated data in this repository take precedence over the
    # temporary external-data bridge. Raw inputs continue to resolve through
    # runtime.data_root until they are deliberately copied into this project.
    workspace_candidate = workspace_data_root(config) / value
    if workspace_candidate.exists():
        return workspace_candidate.resolve()

    return (Path(config["runtime"]["data_root"]) / value).resolve()


def repository_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a tracked project path relative to the repository root."""
    return _resolve_from_root(value, Path(config["runtime"]["project_root"]))


def override_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a named manual-override table from configuration."""
    overrides = config.get("overrides", {})
    if key not in overrides:
        raise ConfigError(f"Missing overrides.{key} in configuration.")
    return repository_path(config, overrides[key])


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["runtime"]["output_root"])


def workspace_data_root(config: dict[str, Any]) -> Path:
    """Return this repository's ignored folder for newly generated data."""
    return Path(config["runtime"]["project_root"]) / "data"
