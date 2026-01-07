"""Extract per-session embeddings using a trained autoencoder encoder."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from hdd_dataset import HDDWindowDataset
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HDD embeddings per session.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--index_dir", type=str, default="artifacts")
    parser.add_argument("--sensor_dir", type=str, default="20200710_sensors/sensor")
    parser.add_argument("--normalization", type=str, default="artifacts/normalization.json")
    parser.add_argument("--weights", type=str, default="artifacts/encoder/weights_best.pt")
    parser.add_argument("--config", type=str, default="artifacts/encoder/config.json")
    parser.add_argument("--out_dir", type=str, default="artifacts/embeddings")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--window", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def load_index_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def unpack_batch_meta(meta: Any) -> Tuple[List[str], List[int]]:
    if isinstance(meta, dict):
        session_ids = meta["session_id"]
        t_ends = meta["t_end"]
        if isinstance(t_ends, torch.Tensor):
            t_ends = t_ends.cpu().tolist()
        if isinstance(session_ids, (list, tuple)):
            return list(session_ids), [int(t) for t in t_ends]
        return [str(session_ids)], [int(t_ends)]
    if isinstance(meta, list):
        return [m["session_id"] for m in meta], [int(m["t_end"]) for m in meta]
    raise ValueError(f"Unsupported metadata type: {type(meta)}")


def write_session(out_dir: Path, session_id: str, z_list: List[np.ndarray], t_list: List[int]) -> None:
    if not z_list:
        return
    z_arr = np.stack(z_list, axis=0).astype(np.float32)
    t_arr = np.asarray(t_list, dtype=np.int64)
    np.save(out_dir / f"{session_id}.npy", z_arr)
    np.save(out_dir / f"{session_id}_t_end.npy", t_arr)


def extract(config: Dict[str, Any]) -> None:
    cfg = dict(config)
    device = resolve_device(cfg.get("device", "auto"))

    train_cfg = load_config(cfg.get("config", "artifacts/encoder/config.json"))
    model_name = cfg.get("model_name") or train_cfg.get("model_name", "tcn_ae")
    embedding_dim = cfg.get("embedding_dim") or int(train_cfg.get("embedding_dim", 20))
    hidden_channels = cfg.get("hidden_channels") or int(train_cfg.get("hidden_channels", 32))
    num_layers = cfg.get("num_layers") or int(train_cfg.get("num_layers", 3))
    kernel_size = cfg.get("kernel_size") or int(train_cfg.get("kernel_size", 3))
    dropout = cfg["dropout"] if cfg.get("dropout") is not None else float(train_cfg.get("dropout", 0.0))
    window = cfg.get("window") or int(train_cfg.get("window", 18))

    model = build_model(
        model_name,
        in_channels=8,
        latent_dim=embedding_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    state = torch.load(cfg.get("weights", "artifacts/encoder/weights_best.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_root = Path(cfg.get("out_dir", "artifacts/embeddings"))
    out_root.mkdir(parents=True, exist_ok=True)

    splits_value = cfg.get("splits", "train,val,test")
    split_list = [s.strip() for s in splits_value.split(",") if s.strip()]
    for split in split_list:
        index_path = Path(cfg.get("index_dir", "artifacts")) / f"index_{split}_dense.jsonl"
        if not index_path.exists():
            print(f"Missing index for split {split}: {index_path}")
            continue

        records = load_index_records(index_path)
        records.sort(key=lambda r: (r["session_id"], r["t_end"]))

        dataset = HDDWindowDataset(
            index=records,
            sensor_dir=cfg.get("sensor_dir", "20200710_sensors/sensor"),
            label_dir=None,
            window=window,
            normalization=cfg.get("normalization", "artifacts/normalization.json"),
            return_label=False,
            cache_size=32,
            to_tensor=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.get("batch_size", 256)),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 0)),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        current_session = None
        session_z: List[np.ndarray] = []
        session_t: List[int] = []

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                meta = batch[1]
                z = model.encode(x).cpu().numpy()
                session_ids, t_ends = unpack_batch_meta(meta)

                for i, sid in enumerate(session_ids):
                    if current_session is None:
                        current_session = sid
                    if sid != current_session:
                        write_session(split_dir, current_session, session_z, session_t)
                        session_z = []
                        session_t = []
                        current_session = sid
                    session_z.append(z[i])
                    session_t.append(int(t_ends[i]))

        if current_session is not None:
            write_session(split_dir, current_session, session_z, session_t)

        print(f"Saved embeddings for split={split} to {split_dir}")


def main() -> None:
    args = parse_args()
    extract(vars(args))


if __name__ == "__main__":
    main()
