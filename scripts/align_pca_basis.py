"""Align PCA basis across splits using train z_std.

Fits PCA on train embeddings (z_std) and applies the same PCA to train/val/test.
This reuses existing *_pca.npz files and does not re-run the encoder.

Example:
  python scripts/align_pca_basis.py \
    --train_npz artifacts/latent_viz/train_pca.npz \
    --val_npz artifacts/latent_viz/val_pca.npz \
    --test_npz artifacts/latent_viz/test_pca.npz \
    --pca_dim 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align PCA basis across splits.")
    parser.add_argument("--train_npz", type=str, default="artifacts/latent_viz/train_pca.npz")
    parser.add_argument("--val_npz", type=str, default="artifacts/latent_viz/val_pca.npz")
    parser.add_argument("--test_npz", type=str, default="artifacts/latent_viz/test_pca.npz")
    parser.add_argument("--pca_dim", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="")
    return parser.parse_args()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def _save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def _compute_scaler(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = z.mean(axis=0)
    scale = z.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


@dataclass
class PCAModel:
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        return Xc @ self.components.T


def _fit_pca(z_std: np.ndarray, n_components: int) -> PCAModel:
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(z_std)
        return PCAModel(
            components=pca.components_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )
    except Exception:
        # Deterministic fallback: SVD on centered data.
        mean = z_std.mean(axis=0, keepdims=True)
        Xc = z_std - mean
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        components = vt[:n_components]
        var = (s**2) / max(1, len(z_std) - 1)
        evr = var[:n_components] / var.sum()
        return PCAModel(
            components=components.astype(np.float32),
            mean=mean.squeeze(0).astype(np.float32),
            explained_variance_ratio=evr.astype(np.float32),
        )


def _apply_pca_to_split(
    path: Path,
    out_path: Path,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    pca: PCAModel,
) -> None:
    data = _load_npz(path)
    if "z" not in data:
        raise KeyError(f"Missing z in {path}")

    z = data["z"].astype(np.float32, copy=False)
    z_std = (z - scaler_mean) / scaler_scale
    pcs = pca.transform(z_std)

    data["z_std"] = z_std.astype(np.float32)
    data["pcs"] = pcs.astype(np.float32)
    data["explained_variance_ratio"] = pca.explained_variance_ratio.astype(np.float32)
    data["cumulative_variance"] = np.cumsum(pca.explained_variance_ratio).astype(np.float32)
    data["scaler_mean"] = scaler_mean.astype(np.float32)
    data["scaler_scale"] = scaler_scale.astype(np.float32)

    _save_npz(out_path, data)
    print(f"Updated PCA outputs: {out_path}")


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_npz)
    val_path = Path(args.val_npz)
    test_path = Path(args.test_npz)
    out_dir = Path(args.out_dir) if args.out_dir else None

    train_data = _load_npz(train_path)
    if "z" not in train_data:
        raise KeyError(f"Missing z in {train_path}")

    z_train = train_data["z"].astype(np.float32, copy=False)
    scaler_mean = train_data.get("scaler_mean")
    scaler_scale = train_data.get("scaler_scale")
    if scaler_mean is None or scaler_scale is None:
        scaler_mean, scaler_scale = _compute_scaler(z_train)
    else:
        scaler_mean = scaler_mean.astype(np.float32, copy=False)
        scaler_scale = scaler_scale.astype(np.float32, copy=False)

    z_train_std = (z_train - scaler_mean) / scaler_scale
    n_components = min(args.pca_dim, z_train_std.shape[1])
    pca = _fit_pca(z_train_std, n_components=n_components)

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing PCA npz: {path}")
        out_path = (out_dir / path.name) if out_dir else path
        _apply_pca_to_split(path, out_path, scaler_mean, scaler_scale, pca)


if __name__ == "__main__":
    main()
