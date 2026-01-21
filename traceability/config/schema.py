"""Config schema, defaults, and validation."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from traceability.data.schema import BINARY_CHANNELS, CHANNELS


DEFAULTS: dict[str, Any] = {
    "run": {"seed": 123, "run_id": "run"},
    "io": {"artifacts_root": "artifacts", "data_root": "data"},
    "data": {
        "fs": 3,
        "window_timesteps": None,
        "hop_train_timesteps": None,
        "hop_infer_timesteps": None,
        "channels": CHANNELS,
        "normalization": {
            "zscore": True,
            "clip_sigma": None,
            "binary_channels": BINARY_CHANNELS,
        },
    },
    "encoder": {
        "name": None,
        "kwargs": {
            "input_dim": len(CHANNELS),
            "embedding_dim": 64,
            "tcn_channels": 64,
            "tcn_depth": 4,
            "kernel_size": 3,
            "dilation_base": 2,
            "dropout": 0.1,
            "pooling": {"type": "last_k", "k": 3},
            "future_mlp_hidden": 64,
        },
    },
    "objective": {
        "name": None,
        "kwargs": {
            "future_offset_timesteps": 9,
            "loss_type": "smooth_l1",
            "cosine_weight": 0.0,
        },
    },
    "training": {
        "epochs": 50,
        "batch_size": 128,
        "optimizer": {"name": "adamw", "kwargs": {"lr": 3.0e-4, "weight_decay": 1.0e-2}},
        "scheduler": {"name": None, "kwargs": {}},
        "grad_clip_norm": None,
        "device": "auto",
        "num_workers": 0,
        "log_every_steps": 50,
        "eval_every_epochs": 1,
    },
    "logging": {"tensorboard": True, "tb_flush_secs": 10, "save_hparams": True},
}


def _deep_merge(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _require(config: dict, key_path: str) -> None:
    current: Any = config
    for key in key_path.split("."):
        if key not in current:
            raise ValueError(f"Missing required config key: {key_path}")
        current = current[key]
    if current is None:
        raise ValueError(f"Config key must be set: {key_path}")


def validate_config(config: dict) -> None:
    disallowed_namespaces = {"predictive", "contrastive", "masked"}
    for key in config.keys():
        if key in disallowed_namespaces:
            raise ValueError(
                f"Objective-specific namespace '{key}' is not allowed; use objective.kwargs."
            )

    _require(config, "run.seed")
    _require(config, "data.window_timesteps")
    _require(config, "data.hop_train_timesteps")
    _require(config, "data.hop_infer_timesteps")
    _require(config, "encoder.name")
    _require(config, "objective.name")

    objective = config.get("objective", {})
    if set(objective.keys()) - {"name", "kwargs"}:
        raise ValueError("objective may only contain name and kwargs.")
    if not isinstance(objective.get("kwargs", {}), dict):
        raise ValueError("objective.kwargs must be a dict.")

    encoder = config.get("encoder", {})
    if not isinstance(encoder.get("kwargs", {}), dict):
        raise ValueError("encoder.kwargs must be a dict.")

    data_cfg = config.get("data", {})
    window_timesteps = data_cfg.get("window_timesteps")
    hop_train = data_cfg.get("hop_train_timesteps")
    hop_infer = data_cfg.get("hop_infer_timesteps")
    if int(window_timesteps) < 1 or int(hop_train) < 1 or int(hop_infer) < 1:
        raise ValueError("window_timesteps and hop values must be >= 1.")

    objective_kwargs = objective.get("kwargs", {})
    future_offset = objective_kwargs.get("future_offset_timesteps")
    if future_offset is not None and int(future_offset) < 1:
        raise ValueError("objective.kwargs.future_offset_timesteps must be >= 1.")

    encoder_kwargs = encoder.get("kwargs", {})
    embedding_dim = encoder_kwargs.get("embedding_dim")
    if embedding_dim is not None and int(embedding_dim) < 1:
        raise ValueError("encoder.kwargs.embedding_dim must be >= 1.")
    tcn_depth = encoder_kwargs.get("tcn_depth")
    if tcn_depth is not None and int(tcn_depth) < 1:
        raise ValueError("encoder.kwargs.tcn_depth must be >= 1.")
    dilation_base = encoder_kwargs.get("dilation_base")
    if dilation_base is not None and int(dilation_base) < 1:
        raise ValueError("encoder.kwargs.dilation_base must be >= 1.")
    pooling = encoder_kwargs.get("pooling")
    if pooling is not None:
        pool_type = pooling.get("type")
        if pool_type not in {"last_k", "attn"}:
            raise ValueError("encoder.kwargs.pooling.type must be 'last_k' or 'attn'.")
        pool_k = pooling.get("k")
        if pool_k is None or int(pool_k) < 1:
            raise ValueError("encoder.kwargs.pooling.k must be >= 1.")

    channels = data_cfg.get("channels", [])
    normalization = data_cfg.get("normalization", {})
    binary_channels = normalization.get("binary_channels", [])
    for name in binary_channels:
        if name not in channels:
            raise ValueError(f"Binary channel '{name}' not in channels list.")


def resolve_config(raw: dict) -> dict:
    resolved = _deep_merge(deepcopy(DEFAULTS), raw)
    validate_config(resolved)

    if resolved["run"].get("run_id") in (None, "null", ""):
        resolved["run"]["run_id"] = "run"

    return resolved
