"""Latent-space PCA health check for HDD TCN-AE embeddings.

Usage:
  python scripts/latent_pca_viz.py --ckpt artifacts/encoder/weights_best.pt --split train
Outputs:
  artifacts/latent_viz/ (plots + npz with PCA outputs/metadata)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from hdd_dataset import CHANNEL_NAMES, HDDWindowDataset, load_index
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
    parser = argparse.ArgumentParser(description="PCA latent-space visualization for HDD embeddings.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--index_file", type=str, default="")
    parser.add_argument("--max_windows", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default="artifacts/latent_viz")
    parser.add_argument("--out_npz", type=str, default="")
    parser.add_argument("--session_id", type=str, default="")
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path) -> tuple[torch.nn.Module, Dict[str, Any]]:
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json next to checkpoint: {config_path}")
    cfg = json.loads(config_path.read_text())
    model = build_model(
        cfg.get("model_name", "tcn_ae"),
        in_channels=8,
        latent_dim=int(cfg.get("embedding_dim", 20)),
        hidden_channels=int(cfg.get("hidden_channels", 32)),
        num_layers=int(cfg.get("num_layers", 3)),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model, cfg


def unpack_meta(meta: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    session_ids = meta["session_id"]
    t_ends = meta["t_end"]
    if isinstance(t_ends, torch.Tensor):
        t_ends = t_ends.cpu().tolist()
    if isinstance(session_ids, (list, tuple)):
        return [str(s) for s in session_ids], [int(t) for t in t_ends]
    return [str(session_ids)], [int(t_ends)]


def standardize(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Z_std = scaler.fit_transform(Z)
        return Z_std, scaler.mean_, scaler.scale_
    except Exception:
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        Z_std = (Z - mean) / std
        return Z_std, mean, std


def run_pca(Z_std: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(Z_std)
        evr = pca.explained_variance_ratio_
    except Exception:
        # Fallback: SVD on standardized data
        U, S, Vt = np.linalg.svd(Z_std, full_matrices=False)
        pcs = U[:, :n_components] * S[:n_components]
        var = (S**2) / (len(Z_std) - 1)
        evr = var[:n_components] / var.sum()
    return pcs, evr


def plot_scatter(x, y, c=None, title="", out_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    if c is None:
        plt.scatter(x, y, s=4, alpha=0.3)
    else:
        plt.scatter(x, y, s=4, alpha=0.3, c=c, cmap="viridis")
        plt.colorbar(label=title.split("by ")[-1] if "by " in title else "value")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_scree(evr: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)
    plt.figure(figsize=(6, 4))
    plt.bar(x, evr, alpha=0.7, label="explained variance")
    plt.plot(x, cum, marker="o", color="red", label="cumulative")
    plt.xticks(x)
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    ckpt_path = Path(args.ckpt)
    model, cfg = load_model(ckpt_path)
    model = model.to(device)
    model.eval()

    if args.index_file:
        index_path = Path(args.index_file)
    else:
        index_path = Path("artifacts") / f"index_{args.split}_dense.jsonl"

    index_list = load_index(index_path)
    if args.session_id:
        index_list = [rec for rec in index_list if rec["session_id"] == args.session_id]

    if args.max_windows and len(index_list) > args.max_windows and not args.session_id:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(index_list), size=args.max_windows, replace=False)
        index_list = [index_list[i] for i in idx]

    window = int(cfg.get("window", 18))
    normalization_path = cfg.get("normalization", "artifacts/normalization.json")
    dataset = HDDWindowDataset(
        index=index_list,
        sensor_dir="20200710_sensors/sensor",
        label_dir=None,
        window=window,
        normalization=normalization_path,
        return_label=False,
        cache_size=32,
        to_tensor=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Determinism check
    first_batch = next(iter(loader))
    x0 = first_batch[0].to(device)
    with torch.no_grad():
        z1 = model.encode(x0)
        z2 = model.encode(x0)
    max_diff = float(torch.max(torch.abs(z1 - z2)).cpu().item())
    print(f"Determinism check max|z1-z2|: {max_diff:.6e}")

    # Channel indices for metadata
    idx_speed = CHANNEL_NAMES.index("speed")
    idx_yaw = CHANNEL_NAMES.index("yaw_deg_s")
    idx_brake = CHANNEL_NAMES.index("brake_kpa")
    idx_lturn = CHANNEL_NAMES.index("lturn")
    idx_rturn = CHANNEL_NAMES.index("rturn")

    Z_list: List[np.ndarray] = []
    session_ids: List[str] = []
    window_idx: List[int] = []
    t_ends: List[int] = []
    speed_mean: List[float] = []
    yaw_mean: List[float] = []
    brake_mean: List[float] = []
    lturn_frac: List[float] = []
    rturn_frac: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            meta = batch[1]
            z = model.encode(x).cpu().numpy()
            Z_list.append(z)

            sess, t_end = unpack_meta(meta)
            for s, t in zip(sess, t_end):
                session_ids.append(s)
                t_ends.append(int(t))
                # For dense indices, window_idx = t_end - (window-1)
                window_idx.append(int(t) - (window - 1))

            x_np = x.cpu().numpy()
            speed_mean.extend(x_np[:, :, idx_speed].mean(axis=1).tolist())
            yaw_mean.extend(x_np[:, :, idx_yaw].mean(axis=1).tolist())
            brake_mean.extend(x_np[:, :, idx_brake].mean(axis=1).tolist())
            lturn_frac.extend(x_np[:, :, idx_lturn].mean(axis=1).tolist())
            rturn_frac.extend(x_np[:, :, idx_rturn].mean(axis=1).tolist())

            if args.max_windows and len(session_ids) >= args.max_windows:
                break

    Z = np.concatenate(Z_list, axis=0)
    total = min(len(session_ids), Z.shape[0])
    if args.max_windows:
        total = min(total, args.max_windows)
    Z = Z[:total]
    session_ids = session_ids[:total]
    window_idx = window_idx[:total]
    t_ends = t_ends[:total]
    speed_mean = speed_mean[:total]
    yaw_mean = yaw_mean[:total]
    brake_mean = brake_mean[:total]
    lturn_frac = lturn_frac[:total]
    rturn_frac = rturn_frac[:total]

    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0)
    near_zero = float((z_std < 1e-6).mean())
    print(f"Z shape: {Z.shape}")
    print(f"Z mean min/max: {z_mean.min():.4f}/{z_mean.max():.4f}")
    print(f"Z std  min/max: {z_std.min():.4f}/{z_std.max():.4f}")
    print(f"Fraction of near-zero std dims: {near_zero:.4f}")

    Z_std, scaler_mean, scaler_scale = standardize(Z)
    n_components = min(10, Z.shape[1])
    pcs, evr = run_pca(Z_std, n_components)
    cum_evr = np.cumsum(evr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = Path(args.out_npz) if args.out_npz else (out_dir / "latent_pca_outputs.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_npz,
        z=Z.astype(np.float32),
        z_std=Z_std.astype(np.float32),
        pcs=pcs.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        cumulative_variance=cum_evr.astype(np.float32),
        scaler_mean=scaler_mean.astype(np.float32),
        scaler_scale=scaler_scale.astype(np.float32),
        session_id=np.asarray(session_ids, dtype="U"),
        window_idx=np.asarray(window_idx, dtype=np.int64),
        t_end=np.asarray(t_ends, dtype=np.int64),
        speed_mean=np.asarray(speed_mean, dtype=np.float32),
        yaw_mean=np.asarray(yaw_mean, dtype=np.float32),
        brake_mean=np.asarray(brake_mean, dtype=np.float32),
        lturn_frac=np.asarray(lturn_frac, dtype=np.float32),
        rturn_frac=np.asarray(rturn_frac, dtype=np.float32),
    )

    plot_scree(evr, out_dir / "scree.png")
    plot_scatter(pcs[:, 0], pcs[:, 1], title="PC1 vs PC2", out_path=out_dir / "pc1_pc2.png")

    # Color by window index (normalized)
    w_norm = (np.asarray(window_idx) - np.min(window_idx)) / (
        np.max(window_idx) - np.min(window_idx) + 1e-9
    )
    plot_scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=w_norm,
        title="PC1 vs PC2 by time",
        out_path=out_dir / "pc1_pc2_by_time.png",
    )
    plot_scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=np.asarray(speed_mean),
        title="PC1 vs PC2 by speed",
        out_path=out_dir / "pc1_pc2_by_speed.png",
    )
    plot_scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=np.asarray(yaw_mean),
        title="PC1 vs PC2 by yaw",
        out_path=out_dir / "pc1_pc2_by_yaw.png",
    )
    plot_scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=np.asarray(brake_mean),
        title="PC1 vs PC2 by brake",
        out_path=out_dir / "pc1_pc2_by_brake.png",
    )
    indicator = np.asarray(lturn_frac) - np.asarray(rturn_frac)
    plot_scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=indicator,
        title="PC1 vs PC2 by indicator",
        out_path=out_dir / "pc1_pc2_by_indicator.png",
    )

    if args.session_id:
        order = np.argsort(window_idx)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.plot(pcs[order, 0], pcs[order, 1], linewidth=0.8, alpha=0.7)
        plt.scatter(pcs[order, 0], pcs[order, 1], s=6, alpha=0.7)
        plt.title(f"PC1-PC2 trajectory: {args.session_id}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_pc1_pc2_{args.session_id}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
