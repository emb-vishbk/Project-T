"""Config loader for experiment YAML/JSON files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from traceability.config.schema import resolve_config


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configs.") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = _load_yaml(path)
    elif suffix == ".json":
        raw = _load_json(path)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    return resolve_config(raw)
