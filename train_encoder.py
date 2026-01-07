"""Train a lightweight TCN autoencoder on HDD windows (self-supervised)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hdd_dataset import HDDWindowDataset
from models import build_model


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HDD TCN autoencoder.")
    parser.add_argument("--model_name", type=str, default="tcn_ae")
    parser.add_argument("--train_index", type=str, default="artifacts/index_train_sparse.jsonl")
    parser.add_argument("--val_index", type=str, default="artifacts/index_val_sparse.jsonl")
    parser.add_argument("--sensor_dir", type=str, default="20200710_sensors/sensor")
    parser.add_argument("--normalization", type=str, default="artifacts/normalization.json")
    parser.add_argument("--window", type=int, default=18)
    parser.add_argument("--embedding_dim", type=int, default=20)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision if CUDA is available.")
    parser.add_argument("--out_dir", type=str, default="artifacts/encoder")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(
    index_path: str,
    sensor_dir: str,
    normalization: str,
    window: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    cache_size: int,
) -> DataLoader:
    dataset = HDDWindowDataset(
        index=index_path,
        sensor_dir=sensor_dir,
        label_dir=None,
        window=window,
        normalization=normalization,
        return_label=False,
        cache_size=cache_size,
        to_tensor=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def unpack_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    train: bool,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        x = unpack_batch(batch).to(device)

        with torch.set_grad_enabled(train):
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    recon, _ = model(x)
                    loss = nn.functional.mse_loss(recon, x)
            else:
                recon, _ = model(x)
                loss = nn.functional.mse_loss(recon, x)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "model_name": "tcn_ae",
        "train_index": "artifacts/index_train_sparse.jsonl",
        "val_index": "artifacts/index_val_sparse.jsonl",
        "sensor_dir": "20200710_sensors/sensor",
        "normalization": "artifacts/normalization.json",
        "window": 18,
        "embedding_dim": 20,
        "hidden_channels": 32,
        "num_layers": 3,
        "kernel_size": 3,
        "dropout": 0.0,
        "batch_size": 128,
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_steps_per_epoch": 0,
        "num_workers": 0,
        "cache_size": 32,
        "seed": 123,
        "device": "auto",
        "amp": False,
        "out_dir": "artifacts/encoder",
    }
    merged = defaults.copy()
    merged.update(config)
    return merged


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_config(config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(
        index_path=cfg["train_index"],
        sensor_dir=cfg["sensor_dir"],
        normalization=cfg["normalization"],
        window=int(cfg["window"]),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        cache_size=int(cfg["cache_size"]),
    )
    val_loader = build_loader(
        index_path=cfg["val_index"],
        sensor_dir=cfg["sensor_dir"],
        normalization=cfg["normalization"],
        window=int(cfg["window"]),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        cache_size=max(2, int(cfg["cache_size"]) // 2),
    )

    model = build_model(
        cfg["model_name"],
        in_channels=8,
        latent_dim=int(cfg["embedding_dim"]),
        hidden_channels=int(cfg["hidden_channels"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )

    use_amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Sanity forward pass
    first_batch = next(iter(train_loader))
    x0 = unpack_batch(first_batch).to(device)
    recon0, z0 = model(x0)
    print("Sanity shapes:", x0.shape, recon0.shape, z0.shape)
    assert x0.shape == recon0.shape, "recon shape mismatch"
    assert z0.shape[0] == x0.shape[0] and z0.shape[1] == int(cfg["embedding_dim"])

    cfg_to_save = cfg.copy()
    cfg_to_save["device"] = str(device)
    cfg_to_save["amp"] = use_amp
    (out_dir / "config.json").write_text(json.dumps(cfg_to_save, indent=2))

    best_val = float("inf")
    metrics = {"train_loss": [], "val_loss": []}

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_steps=int(cfg["max_steps_per_epoch"]),
            train=True,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            max_steps=0,
            train=False,
            use_amp=use_amp,
            scaler=None,
        )

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        torch.save(model.state_dict(), out_dir / "weights_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "weights_best.pt")

        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return {
        "best_path": str(out_dir / "weights_best.pt"),
        "last_path": str(out_dir / "weights_last.pt"),
        "metrics": metrics,
        "config": cfg_to_save,
    }


def main() -> None:
    args = parse_args()
    train(vars(args))


if __name__ == "__main__":
    main()
