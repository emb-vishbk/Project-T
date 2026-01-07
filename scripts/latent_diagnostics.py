"""Latent diagnostics: PC-feature correlations + clusterability checks.

Example:
  python scripts/latent_diagnostics.py --pca_npz artifacts/latent_viz/latent_pca_outputs.npz \
    --use_space pc_scores --max_pc 6 --k_min 4 --k_max 40 --k_step 4 --stability_k 8 12 16 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from hdd_dataset import CHANNEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent space diagnostics.")
    parser.add_argument("--pca_npz", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--use_space", type=str, default="pc_scores", choices=["z_std", "pc_scores"])
    parser.add_argument("--max_pc", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="artifacts/latent_viz")
    parser.add_argument("--k_min", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=40)
    parser.add_argument("--k_step", type=int, default=4)
    parser.add_argument("--stability_k", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--sensor_dir", type=str, default="20200710_sensors/sensor")
    return parser.parse_args()


def _get_key(npz: Dict[str, Any], keys: List[str]) -> str | None:
    for k in keys:
        if k in npz:
            return k
    return None


def _subsample(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_samples, replace=False)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def hopkins_statistic(X: np.ndarray, seed: int = 123) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    m = min(1000, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    X_m = X[idx]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(m, X.shape[1]))

    nn = NearestNeighbors(n_neighbors=2).fit(X)
    w_dist, _ = nn.kneighbors(X_m)
    w = w_dist[:, 1]

    nn_u = NearestNeighbors(n_neighbors=1).fit(X)
    u_dist, _ = nn_u.kneighbors(U)
    u = u_dist[:, 0]

    return float(u.sum() / (u.sum() + w.sum()))


def compute_window_means(
    session_ids: np.ndarray,
    t_ends: np.ndarray,
    window_len: int,
    sensor_dir: Path,
    channel_indices: Dict[str, int],
) -> Dict[str, np.ndarray]:
    results = {name: np.empty(len(session_ids), dtype=np.float32) for name in channel_indices}
    for sid in np.unique(session_ids):
        idx = np.where(session_ids == sid)[0]
        t_end = t_ends[idx]
        X = np.load(sensor_dir / f"{sid}.npy")
        t_start = t_end - (window_len - 1)
        for name, col in channel_indices.items():
            csum = np.concatenate([[0.0], X[:, col].cumsum()])
            vals = (csum[t_end + 1] - csum[t_start]) / window_len
            results[name][idx] = vals.astype(np.float32)
    return results


def main() -> None:
    args = parse_args()
    npz_path = Path(args.pca_npz)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing PCA npz: {npz_path}. Run scripts/latent_pca_viz.py first."
        )

    npz = np.load(npz_path, allow_pickle=True)

    key_pcs = _get_key(npz, ["pc_scores", "pcs"])
    if key_pcs is None:
        raise KeyError("Missing pc_scores/pcs in npz.")
    pcs = npz[key_pcs]

    key_z_std = _get_key(npz, ["z_std"])
    key_z = _get_key(npz, ["z"])
    if key_z_std is None and key_z is None:
        raise KeyError("Missing z_std or z in npz.")

    session_ids = npz["session_id"] if "session_id" in npz else None
    window_idx = npz["window_idx"] if "window_idx" in npz else None
    t_end = npz["t_end"] if "t_end" in npz else None

    n = pcs.shape[0]
    keep = _subsample(n, args.max_samples, args.random_seed)
    pcs = pcs[keep]

    if args.use_space == "pc_scores":
        X = pcs[:, : args.max_pc]
    else:
        if key_z_std is not None:
            z_std = npz[key_z_std][keep]
        else:
            z = npz[key_z][keep]
            mean = z.mean(axis=0)
            std = z.std(axis=0)
            std = np.where(std == 0, 1.0, std)
            z_std = (z - mean) / std
        X = z_std

    # Metadata arrays (subsampled)
    def _maybe(name: str):
        return npz[name][keep] if name in npz else None

    speed_mean = _maybe("speed_mean")
    yaw_mean = _maybe("yaw_mean")
    brake_mean = _maybe("brake_mean")
    lturn_frac = _maybe("lturn_frac")
    rturn_frac = _maybe("rturn_frac")
    accel_pedal_mean = _maybe("accel_pedal_mean")
    steer_angle_mean = _maybe("steer_angle_mean")
    steer_speed_mean = _maybe("steer_speed_mean")

    # Fallback: compute missing means from raw signals if we have session_id/t_end/window_idx
    missing = []
    for name, arr in [
        ("accel_pedal_mean", accel_pedal_mean),
        ("steer_angle_mean", steer_angle_mean),
        ("steer_speed_mean", steer_speed_mean),
    ]:
        if arr is None:
            missing.append(name)

    if missing and session_ids is not None and t_end is not None and window_idx is not None:
        t_end = t_end[keep].astype(int)
        session_ids = session_ids[keep]
        window_idx = window_idx[keep].astype(int)
        window_len = int(np.median(t_end - window_idx + 1))
        channel_indices = {
            "accel_pedal_mean": CHANNEL_NAMES.index("accel_pedal_pct"),
            "steer_angle_mean": CHANNEL_NAMES.index("steer_angle_deg"),
            "steer_speed_mean": CHANNEL_NAMES.index("steer_speed"),
        }
        computed = compute_window_means(
            session_ids=session_ids,
            t_ends=t_end,
            window_len=window_len,
            sensor_dir=Path(args.sensor_dir),
            channel_indices=channel_indices,
        )
        accel_pedal_mean = computed["accel_pedal_mean"]
        steer_angle_mean = computed["steer_angle_mean"]
        steer_speed_mean = computed["steer_speed_mean"]
    elif missing:
        print(f"Warning: missing features {missing} (session_id/window_idx/t_end unavailable).")

    features = {
        "speed_mean": speed_mean,
        "brake_mean": brake_mean,
        "yaw_mean": yaw_mean,
        "abs_yaw": np.abs(yaw_mean) if yaw_mean is not None else None,
        "accel_pedal_mean": accel_pedal_mean,
        "steer_angle_mean": steer_angle_mean,
        "steer_speed_mean": steer_speed_mean,
        "lturn_frac": lturn_frac,
        "rturn_frac": rturn_frac,
    }

    # Build correlation table
    pc_cols = [f"PC{i}" for i in range(1, min(args.max_pc, pcs.shape[1]) + 1)]
    corr_rows = []
    for i, pc_name in enumerate(pc_cols):
        row = {}
        pc_vals = pcs[:, i]
        for feat_name, feat_vals in features.items():
            if feat_vals is None:
                row[feat_name] = np.nan
            else:
                row[feat_name] = pearson_corr(pc_vals, feat_vals)
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows, index=pc_cols)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "pc_feature_correlations.csv")

    # Heatmap
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.imshow(corr_df.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title("PC-feature correlations")
    plt.tight_layout()
    plt.savefig(out_dir / "pc_feature_correlations.png", dpi=150)
    plt.close()

    # Hopkins
    H = hopkins_statistic(X, seed=args.random_seed)
    if H < 0.4:
        hopkins_msg = "near 0 -> highly clusterable"
    elif H > 0.6:
        hopkins_msg = "near 1 -> highly clusterable"
    else:
        hopkins_msg = "near 0.5 -> random/unstructured"
    print(f"Hopkins statistic: {H:.4f} ({hopkins_msg})")

    # Silhouette vs K
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    ks = list(range(args.k_min, args.k_max + 1, args.k_step))
    sil_rows = []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=args.random_seed, batch_size=1024)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels, sample_size=min(10000, len(X))))
        sil_rows.append({"K": k, "silhouette": sil})

    sil_df = pd.DataFrame(sil_rows)
    sil_df.to_csv(out_dir / "silhouette_vs_k.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(sil_df["K"], sil_df["silhouette"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.title("Silhouette vs K")
    plt.tight_layout()
    plt.savefig(out_dir / "silhouette_vs_k.png", dpi=150)
    plt.close()

    # Inertia vs K (fixed range 4..40 step 4)
    inertia_ks = list(range(4, 41, 4))
    inertia_rows = []
    inertia_X = pcs[:, : min(6, pcs.shape[1])]
    for k in inertia_ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=args.random_seed, batch_size=1024)
        km.fit(inertia_X)
        inertia_rows.append({"K": k, "inertia": float(km.inertia_)})

    inertia_df = pd.DataFrame(inertia_rows)
    inertia_df.to_csv(out_dir / "inertia_vs_k.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(inertia_df["K"], inertia_df["inertia"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia (SSE)")
    plt.title("K-Means Inertia vs K")
    plt.tight_layout()
    plt.savefig(out_dir / "inertia_vs_k.png", dpi=150)
    plt.close()

    # Stability vs seed
    stability_rows = []
    for k in args.stability_k:
        labels_list = []
        for i in range(args.n_seeds):
            km = MiniBatchKMeans(n_clusters=k, random_state=args.random_seed + i, batch_size=1024)
            labels_list.append(km.fit_predict(X))
        aris = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
        stability_rows.append(
            {
                "K": k,
                "mean_ARI": float(np.mean(aris)),
                "min_ARI": float(np.min(aris)),
                "max_ARI": float(np.max(aris)),
            }
        )
    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(out_dir / "stability_vs_seed.csv", index=False)

    # Summary
    print("\nTop correlations per PC:")
    for pc in corr_df.index:
        series = corr_df.loc[pc].dropna()
        if series.empty:
            continue
        top = series.reindex(series.abs().sort_values(ascending=False).index)[:3]
        print(f"{pc}: " + ", ".join([f"{k}={v:.3f}" for k, v in top.items()]))

    print("\nSilhouette (best K):")
    best_row = sil_df.iloc[sil_df["silhouette"].idxmax()]
    print(f"K={int(best_row['K'])}, silhouette={best_row['silhouette']:.4f}")

    print("\nStability summary (ARI):")
    for _, row in stab_df.iterrows():
        print(
            f"K={int(row['K'])}: mean={row['mean_ARI']:.3f}, "
            f"min={row['min_ARI']:.3f}, max={row['max_ARI']:.3f}"
        )


if __name__ == "__main__":
    main()
