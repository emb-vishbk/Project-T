"""Objective-agnostic Stage 1 trainer."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from traceability.data.batch_builder import BatchBuilder
from traceability.data.dataset import HDDWindowDataset
from traceability.models.encoders import get_encoder
from traceability.objectives import get_objective
from traceability.utils import paths
from traceability.utils.io import read_json, write_json
from traceability.utils.seed import seed_everything


SUPPORTED_BATCH_KEYS = {"x", "x1", "x2", "x_future", "x_masked", "mask", "target"}


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_optimizer(encoder: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    name = config["training"]["optimizer"]["name"]
    kwargs = config["training"]["optimizer"].get("kwargs", {})
    name = name.lower() if name is not None else None
    if name == "adamw":
        return torch.optim.AdamW(encoder.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(encoder.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(encoder.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    scheduler_cfg = config["training"].get("scheduler", {})
    name = scheduler_cfg.get("name")
    kwargs = scheduler_cfg.get("kwargs", {})
    if name in (None, "null"):
        return None
    name = str(name).lower()
    if name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    raise ValueError(f"Unsupported scheduler: {name}")


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _resolve_data_paths(config: dict) -> dict[str, str]:
    data_root = config["io"]["data_root"]
    artifacts_root = config["io"]["artifacts_root"]
    data_paths = config.get("data", {}).get("paths", {})
    sensors_root = data_paths.get("sensors_root") or str(paths.sensors_dir(data_root))
    labels_root = data_paths.get("labels_root") or str(paths.labels_dir(data_root))
    index_train = data_paths.get("index_training_train") or str(
        paths.index_file(artifacts_root, "training", "train")
    )
    index_val = data_paths.get("index_training_val") or str(
        paths.index_file(artifacts_root, "training", "val")
    )
    return {
        "data_root": data_root,
        "artifacts_root": artifacts_root,
        "sensors_root": sensors_root,
        "labels_root": labels_root,
        "index_train": index_train,
        "index_val": index_val,
    }


def _validate_objective_contract(objective) -> tuple[list[str], list[str]]:
    required_keys = getattr(objective, "REQUIRED_BATCH_KEYS", ["x"])
    required_methods = getattr(objective, "REQUIRED_ENCODER_METHODS", ["encode"])
    unknown = set(required_keys) - SUPPORTED_BATCH_KEYS
    if unknown:
        raise ValueError(f"Unsupported REQUIRED_BATCH_KEYS: {sorted(unknown)}")
    if not hasattr(objective, "compute_loss"):
        raise ValueError("Objective must implement compute_loss(encoder, batch).")
    return list(required_keys), list(required_methods)


def _validate_encoder_methods(encoder, required_methods: list[str]) -> None:
    missing = [name for name in required_methods if not hasattr(encoder, name)]
    if missing:
        raise ValueError(f"Encoder missing required methods: {missing}")


def _log_hparams(writer: SummaryWriter, config: dict) -> None:
    hparams = {
        "encoder.name": config["encoder"]["name"],
        "objective.name": config["objective"]["name"],
        "window_timesteps": config["data"]["window_timesteps"],
        "fs": config["data"]["fs"],
        "hop_train_timesteps": config["data"]["hop_train_timesteps"],
        "hop_infer_timesteps": config["data"]["hop_infer_timesteps"],
        "clip_sigma": config["data"]["normalization"].get("clip_sigma"),
        "seed": config["run"]["seed"],
        "optimizer": config["training"]["optimizer"]["name"],
        "lr": config["training"]["optimizer"].get("kwargs", {}).get("lr"),
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
    }
    embedding_dim = config["encoder"].get("kwargs", {}).get("embedding_dim")
    if embedding_dim is not None:
        hparams["embedding_dim"] = embedding_dim
    writer.add_hparams(hparams, metric_dict={})


def train(config: dict) -> str:
    seed_everything(int(config["run"]["seed"]))

    encoder_cls = get_encoder(config["encoder"]["name"])
    objective_cls = get_objective(config["objective"]["name"])
    encoder = encoder_cls(**config["encoder"].get("kwargs", {}))
    objective = objective_cls(**config["objective"].get("kwargs", {}))

    required_keys, required_methods = _validate_objective_contract(objective)
    _validate_encoder_methods(encoder, required_methods)

    device = _select_device(config["training"]["device"])
    encoder = encoder.to(device)

    artifacts_root = config["io"]["artifacts_root"]
    run_id = config["run"]["run_id"]
    run_dir = paths.run_dir(
        artifacts_root,
        config["encoder"]["name"],
        config["objective"]["name"],
        run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths.config_file(run_dir)
    write_json(config_path, config)

    writer = None
    if config["logging"].get("tensorboard", True):
        tb_path = paths.tb_dir(run_dir)
        tb_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_path), flush_secs=config["logging"]["tb_flush_secs"])
        writer.add_text("config/resolved_json", json.dumps(config, indent=2), 0)
        writer.add_text("run/run_id", run_id, 0)
        writer.add_text("run/timestamp", datetime.now(timezone.utc).isoformat(), 0)
        _log_hparams(writer, config)

    data_paths = _resolve_data_paths(config)
    normalization = read_json(paths.normalization_file(data_paths["artifacts_root"]))

    dataset_args = dict(
        sensors_root=data_paths["sensors_root"],
        labels_root=None,
        window_timesteps=config["data"]["window_timesteps"],
        channels=config["data"]["channels"],
        normalization_stats=normalization,
        zscore=config["data"]["normalization"].get("zscore", True),
        clip_sigma=config["data"]["normalization"].get("clip_sigma"),
        return_label=False,
    )

    train_dataset = HDDWindowDataset(index_path=data_paths["index_train"], **dataset_args)
    val_dataset = None
    if Path(data_paths["index_val"]).exists():
        val_dataset = HDDWindowDataset(index_path=data_paths["index_val"], **dataset_args)

    train_builder = BatchBuilder(
        train_dataset,
        required_keys,
        config["objective"].get("kwargs", {}),
        base_seed=config["run"]["seed"],
    )
    val_builder = None
    if val_dataset is not None:
        val_builder = BatchBuilder(
            val_dataset,
            required_keys,
            config["objective"].get("kwargs", {}),
            base_seed=config["run"]["seed"],
        )

    generator = torch.Generator().manual_seed(int(config["run"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=train_builder,
        generator=generator,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            collate_fn=val_builder,
            drop_last=False,
        )

    optimizer = _build_optimizer(encoder, config)
    scheduler = _build_scheduler(optimizer, config)
    grad_clip = config["training"].get("grad_clip_norm")

    metrics: dict[str, Any] = {"train": {"loss": [], "logs": []}, "val": {"loss": [], "logs": []}}
    best_val = None

    epochs = int(config["training"]["epochs"])
    for epoch in range(epochs):
        encoder.train()
        train_losses = []
        train_logs: dict[str, list[float]] = defaultdict(list)
        drop_counts: dict[str, int] = defaultdict(int)

        for step, batch in enumerate(train_loader):
            if batch is None:
                drop_counts["empty_batch"] += 1
                continue
            if train_builder.last_drop_counts:
                for key, value in train_builder.last_drop_counts.items():
                    drop_counts[key] += value
            batch = _move_batch_to_device(batch, device)

            loss, logs = objective.compute_loss(encoder, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(grad_clip))
            optimizer.step()

            train_losses.append(float(loss.item()))
            for key, value in (logs or {}).items():
                train_logs[key].append(float(value))

            if writer and (step + 1) % int(config["training"]["log_every_steps"]) == 0:
                writer.add_scalar("loss/train", float(loss.item()), epoch * len(train_loader) + step)

        if scheduler is not None:
            scheduler.step()

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        metrics["train"]["loss"].append(train_loss)
        metrics["train"]["logs"].append({k: float(sum(v) / len(v)) for k, v in train_logs.items()})

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            for key, value in drop_counts.items():
                writer.add_scalar(f"data/drop_counts/{key}", value, epoch)
            lr = optimizer.param_groups[0].get("lr")
            if lr is not None:
                writer.add_scalar("lr", lr, epoch)

        val_loss = None
        if val_loader is not None and (epoch + 1) % int(config["training"]["eval_every_epochs"]) == 0:
            encoder.eval()
            val_losses = []
            val_logs: dict[str, list[float]] = defaultdict(list)
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    batch = _move_batch_to_device(batch, device)
                    loss, logs = objective.compute_loss(encoder, batch)
                    val_losses.append(float(loss.item()))
                    for key, value in (logs or {}).items():
                        val_logs[key].append(float(value))
            val_loss = float(sum(val_losses) / max(len(val_losses), 1))
            metrics["val"]["loss"].append(val_loss)
            metrics["val"]["logs"].append({k: float(sum(v) / len(v)) for k, v in val_logs.items()})
            if writer:
                writer.add_scalar("loss/val", val_loss, epoch)

        weights_last = paths.weights_last_file(run_dir)
        torch.save(encoder.state_dict(), weights_last)
        if val_loss is not None:
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                weights_best = paths.weights_best_file(run_dir)
                torch.save(encoder.state_dict(), weights_best)

    metrics_path = paths.metrics_file(run_dir)
    write_json(metrics_path, metrics)

    if writer:
        writer.flush()
        writer.close()

    return str(run_dir)
