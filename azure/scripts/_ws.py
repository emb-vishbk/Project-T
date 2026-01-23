"""Azure ML workspace helpers (config + MLClient)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_workspace_settings() -> dict[str, str]:
    base_path = _config_dir() / "config.json"
    local_path = _config_dir() / "config.local.json"

    settings = _read_json(base_path)
    if local_path.exists():
        settings = _deep_merge(settings, _read_json(local_path))

    required_keys = ("subscription_id", "resource_group", "workspace_name")
    for key in required_keys:
        value = str(settings.get(key, "")).strip()
        if not value or "REPLACE" in value.upper() or "CHANGE_ME" in value.upper():
            raise ValueError(f"Invalid workspace setting '{key}' in {base_path}")
    return {
        "subscription_id": str(settings["subscription_id"]).strip(),
        "resource_group": str(settings["resource_group"]).strip(),
        "workspace_name": str(settings["workspace_name"]).strip(),
    }


def get_ml_client() -> tuple[MLClient, dict[str, str]]:
    settings = load_workspace_settings()
    sub = settings["subscription_id"]
    rg = settings["resource_group"]
    ws = settings["workspace_name"]

    try:
        client = MLClient(DefaultAzureCredential(), sub, rg, ws)
        client.workspaces.get(ws)
        print("[INFO] Using DefaultAzureCredential.")
        return client, settings
    except Exception as exc:
        print(f"[WARN] DefaultAzureCredential failed: {type(exc).__name__}")

    client = MLClient(InteractiveBrowserCredential(), sub, rg, ws)
    client.workspaces.get(ws)
    print("[INFO] Using InteractiveBrowserCredential.")
    return client, settings
