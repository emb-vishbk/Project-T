"""Stage 2.1/2.2: KMeans clustering and changepoint segmentation.

This module provides:
- legacy embedding-based clustering helpers (used by run_stage2.py)
- PCA-space clustering entrypoint (train on train split PC scores, apply to all splits)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

FS_HZ = 3.0


class _SimpleKMeans:
    def __init__(
        self,
        n_clusters: int,
        random_state: int,
        n_init: int,
        max_iter: int,
        tol: float = 1e-4,
        chunk_size: int = 100_000,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.chunk_size = int(chunk_size)
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    def _pairwise_sq_dist(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1)
        return x2 + c2 - 2.0 * X @ centers.T

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
        n = X.shape[0]
        labels = np.empty(n, dtype=np.int64)
        inertia = 0.0
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = X[start:end]
            dists = self._pairwise_sq_dist(chunk, centers)
            idx = np.argmin(dists, axis=1)
            labels[start:end] = idx
            inertia += float(dists[np.arange(len(chunk)), idx].sum())
        return labels, inertia

    def fit(self, X: np.ndarray) -> "_SimpleKMeans":
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None

        for _ in range(self.n_init):
            init_idx = rng.choice(n, size=self.n_clusters, replace=False)
            centers = X[init_idx].copy()

            for _ in range(self.max_iter):
                labels, _ = self._assign_labels(X, centers)
                new_centers = np.zeros_like(centers)
                for k in range(self.n_clusters):
                    mask = labels == k
                    if not np.any(mask):
                        new_centers[k] = X[rng.integers(0, n)]
                    else:
                        new_centers[k] = X[mask].mean(axis=0)
                shift = np.sqrt(((centers - new_centers) ** 2).sum(axis=1)).max()
                centers = new_centers
                if shift < self.tol:
                    break

            labels, inertia = self._assign_labels(X, centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted.")
        labels, _ = self._assign_labels(X, self.cluster_centers_)
        return labels


def _load_index_t_end(index_dir: Path, split: str) -> Dict[str, List[int]]:
    index_path = index_dir / f"index_{split}_dense.jsonl"
    mapping: Dict[str, List[int]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mapping[rec["session_id"]].append(int(rec["t_end"]))
    # ensure per-session order
    for session_id in mapping:
        mapping[session_id] = sorted(mapping[session_id])
    return mapping


def load_embeddings(
    split: str,
    embeddings_dir: str,
    index_dir: str,
    fs_hz: float = FS_HZ,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load embeddings and metadata for a split.

    Returns:
        Z: np.ndarray [num_windows_total, emb_dim]
        meta: dict with session_id, window_idx, t_end, t_end_s arrays
    """
    split_dir = Path(embeddings_dir) / split
    index_dir = Path(index_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing embeddings dir for split {split}: {split_dir}")

    index_t_end = _load_index_t_end(index_dir, split)

    Z_list: List[np.ndarray] = []
    session_ids: List[str] = []
    window_idx: List[int] = []
    t_ends: List[int] = []

    for emb_path in sorted(split_dir.glob("*.npy")):
        if emb_path.name.endswith("_t_end.npy"):
            continue
        session_id = emb_path.stem
        z = np.load(emb_path)
        if z.ndim != 2:
            raise ValueError(f"Expected 2D embeddings for {emb_path.name}, got {z.shape}")

        t_end_path = split_dir / f"{session_id}_t_end.npy"
        if t_end_path.exists():
            t_end = np.load(t_end_path).astype(int)
        else:
            if session_id not in index_t_end:
                raise ValueError(f"Missing index metadata for session {session_id}")
            t_end = np.asarray(index_t_end[session_id], dtype=int)

        if len(t_end) != len(z):
            raise ValueError(
                f"Length mismatch for {session_id}: embeddings {len(z)} vs t_end {len(t_end)}"
            )

        Z_list.append(z.astype(np.float32, copy=False))
        session_ids.extend([session_id] * len(z))
        window_idx.extend(range(len(z)))
        t_ends.extend(t_end.tolist())

    if not Z_list:
        raise ValueError(f"No embeddings found in {split_dir}")

    Z = np.concatenate(Z_list, axis=0)
    meta = {
        "session_id": np.asarray(session_ids),
        "window_idx": np.asarray(window_idx, dtype=int),
        "t_end": np.asarray(t_ends, dtype=int),
        "t_end_s": np.asarray(t_ends, dtype=float) / float(fs_hz),
    }
    return Z, meta


def fit_kmeans(
    Z: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 10,
    max_iter: int = 300,
):
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        print("scikit-learn not available; using numpy fallback KMeans.")
        return _SimpleKMeans(
            n_clusters=k,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
        ).fit(Z)

    return KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    ).fit(Z)


def predict_clusters(kmeans, Z: np.ndarray) -> np.ndarray:
    return kmeans.predict(Z)


def build_sequences_by_session(
    meta: Dict[str, np.ndarray], cluster_ids: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    sessions = meta["session_id"]
    window_idx = meta["window_idx"]
    t_end = meta["t_end"]

    order = np.lexsort((window_idx, sessions))
    sessions = sessions[order]
    window_idx = window_idx[order]
    t_end = t_end[order]
    cluster_ids = cluster_ids[order]

    seq_by_session: Dict[str, Dict[str, np.ndarray]] = {}
    for sid in np.unique(sessions):
        mask = sessions == sid
        seq_by_session[str(sid)] = {
            "window_idx": window_idx[mask],
            "t_end": t_end[mask],
            "cluster_id": cluster_ids[mask],
        }
    return seq_by_session


def segment_sequence(
    window_idx: np.ndarray, cluster_ids: np.ndarray, t_end: np.ndarray, fs_hz: float
) -> List[Dict[str, Any]]:
    if len(cluster_ids) == 0:
        return []
    segments: List[Dict[str, Any]] = []
    start = 0
    for i in range(1, len(cluster_ids)):
        if cluster_ids[i] != cluster_ids[i - 1]:
            end = i - 1
            segments.append(
                {
                    "cluster_id": int(cluster_ids[start]),
                    "start_window_idx": int(window_idx[start]),
                    "end_window_idx": int(window_idx[end]),
                    "length_windows": int(window_idx[end] - window_idx[start] + 1),
                    "start_t_end": int(t_end[start]),
                    "end_t_end": int(t_end[end]),
                    "start_time_s": float(t_end[start]) / fs_hz,
                    "end_time_s": float(t_end[end]) / fs_hz,
                }
            )
            start = i
    end = len(cluster_ids) - 1
    segments.append(
        {
            "cluster_id": int(cluster_ids[start]),
            "start_window_idx": int(window_idx[start]),
            "end_window_idx": int(window_idx[end]),
            "length_windows": int(window_idx[end] - window_idx[start] + 1),
            "start_t_end": int(t_end[start]),
            "end_t_end": int(t_end[end]),
            "start_time_s": float(t_end[start]) / fs_hz,
            "end_time_s": float(t_end[end]) / fs_hz,
        }
    )
    return segments


def segment_sequences(
    seq_by_session: Dict[str, Dict[str, np.ndarray]],
    fs_hz: float = FS_HZ,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    all_segments: List[Dict[str, Any]] = []
    segments_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for sid, seq in seq_by_session.items():
        segments = segment_sequence(seq["window_idx"], seq["cluster_id"], seq["t_end"], fs_hz)
        for seg in segments:
            seg["session_id"] = sid
        segments_by_session[sid] = segments
        all_segments.extend(segments)
    return all_segments, segments_by_session


def save_assignments_csv(
    path: Path,
    meta: Dict[str, np.ndarray],
    cluster_ids: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "window_idx", "cluster_id", "t_end", "t_end_s"])
        for sid, widx, cid, t_end, t_end_s in zip(
            meta["session_id"],
            meta["window_idx"],
            cluster_ids,
            meta["t_end"],
            meta["t_end_s"],
        ):
            writer.writerow([sid, int(widx), int(cid), int(t_end), float(t_end_s)])


def _assignment_order(
    sessions: np.ndarray,
    t_end: np.ndarray | None,
    window_idx: np.ndarray | None,
) -> np.ndarray:
    sessions = sessions.astype(str)
    if t_end is None and window_idx is None:
        return np.argsort(sessions, kind="stable")
    if t_end is None:
        return np.lexsort((window_idx, sessions))
    if window_idx is None:
        return np.lexsort((t_end, sessions))
    return np.lexsort((window_idx, t_end, sessions))


def save_assignments_by_session(
    out_dir: Path,
    meta: Dict[str, np.ndarray],
    cluster_ids: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = meta["session_id"].astype(str)
    window_idx = meta["window_idx"]
    t_end = meta["t_end"]
    t_end_s = meta["t_end_s"]

    order = _assignment_order(sessions, t_end, window_idx)
    sessions = sessions[order]
    window_idx = window_idx[order]
    t_end = t_end[order]
    t_end_s = t_end_s[order]
    cluster_ids = cluster_ids[order]

    current_sid = None
    writer = None
    fh = None
    for sid, widx, cid, te, te_s in zip(
        sessions, window_idx, cluster_ids, t_end, t_end_s
    ):
        if sid != current_sid:
            if fh is not None:
                fh.close()
            fh = (out_dir / f"{sid}.csv").open("w", encoding="utf-8", newline="")
            writer = csv.writer(fh)
            writer.writerow(["session_id", "window_idx", "cluster_id", "t_end", "t_end_s"])
            current_sid = sid
        writer.writerow([sid, int(widx), int(cid), int(te), float(te_s)])
    if fh is not None:
        fh.close()


def save_segments_csv(path: Path, segments: List[Dict[str, Any]]) -> None:
    if not segments:
        return
    fieldnames = [
        "session_id",
        "cluster_id",
        "start_window_idx",
        "end_window_idx",
        "length_windows",
        "start_t_end",
        "end_t_end",
        "start_time_s",
        "end_time_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in segments:
            writer.writerow(row)


def save_seq_by_session(path: Path, seq_by_session: Dict[str, Dict[str, np.ndarray]]) -> None:
    payload = {
        sid: {
            "window_idx": seq["window_idx"].tolist(),
            "t_end": seq["t_end"].tolist(),
            "cluster_id": seq["cluster_id"].tolist(),
        }
        for sid, seq in seq_by_session.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def save_segments_by_session(
    path: Path, segments_by_session: Dict[str, List[Dict[str, Any]]]
) -> None:
    path.write_text(json.dumps(segments_by_session, indent=2))


def plot_sequence_with_segments(
    seq: Dict[str, np.ndarray],
    session_id: str,
    out_path: Path,
    fs_hz: float = FS_HZ,
    k: int | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping visualization.")
        return

    x = seq["window_idx"]
    y = seq["cluster_id"]

    plt.figure(figsize=(10, 3))
    ax = plt.gca()

    # Segment blocks (runs of the same cluster id).
    segments = []
    start = 0
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            segments.append((start, i - 1, int(y[start])))
            start = i
    segments.append((start, len(y) - 1, int(y[start])))

    if k is None:
        k = int(np.max(y)) + 1 if len(y) else 1
    cmap = plt.get_cmap("tab20", k)

    for start, end, cid in segments:
        ax.axvspan(x[start], x[end] + 1, color=cmap(cid), alpha=0.12, linewidth=0)

    # Per-window cluster id (scatter).
    ax.scatter(x, y, s=10, color="black", alpha=0.7, linewidths=0)

    # Changepoint lines.
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            ax.axvline(x[i], color="red", alpha=0.3, linewidth=1)

    ax.set_ylim(-0.5, k - 0.5)
    ax.set_yticks(range(k))
    ax.set_title(f"Cluster sequence with changepoints: {session_id}")
    ax.set_xlabel("window_idx")
    ax.set_ylabel("cluster_id")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _get_npz_key(npz: Dict[str, Any], keys: List[str]) -> str | None:
    for key in keys:
        if key in npz:
            return key
    return None


def _resolve_pca_paths(
    pca_npz_dir: Path, pca_npz_train: str, pca_npz_val: str, pca_npz_test: str
) -> Dict[str, Path | None]:
    return {
        "train": Path(pca_npz_train) if pca_npz_train else pca_npz_dir / "train_pca.npz",
        "val": Path(pca_npz_val) if pca_npz_val else pca_npz_dir / "val_pca.npz",
        "test": Path(pca_npz_test) if pca_npz_test else pca_npz_dir / "test_pca.npz",
    }


def load_pca_split(path: Path, pca_dim: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PCA npz: {path}")
    npz = np.load(path, allow_pickle=True)
    pcs_key = _get_npz_key(npz, ["pc_scores", "pcs"])
    if pcs_key is None:
        raise KeyError(f"Missing pc_scores/pcs in {path}")
    pcs = npz[pcs_key]
    if pcs.ndim != 2:
        raise ValueError(f"Expected 2D PC scores in {path}, got {pcs.shape}")

    session_id = npz["session_id"] if "session_id" in npz else None
    window_idx = npz["window_idx"] if "window_idx" in npz else None
    t_end = npz["t_end"] if "t_end" in npz else None
    if session_id is None or (window_idx is None and t_end is None):
        raise KeyError(f"Missing session_id/window_idx/t_end metadata in {path}")

    session_id = np.asarray(session_id).astype(str)
    if window_idx is not None:
        window_idx = np.asarray(window_idx, dtype=int)
    if t_end is not None:
        t_end = np.asarray(t_end, dtype=int)

    pcs = pcs[:, :pca_dim].astype(np.float32, copy=False)
    meta = {
        "session_id": session_id,
        "window_idx": window_idx,
        "t_end": t_end,
    }
    return pcs, meta


def smooth_labels(labels: np.ndarray, window: int, k: int) -> np.ndarray:
    if window <= 1:
        return labels.copy()
    if window % 2 == 0:
        raise ValueError("smooth_window must be odd.")
    n = len(labels)
    half = window // 2
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        counts = np.bincount(labels[start:end], minlength=k)
        out[i] = int(np.argmax(counts))
    return out


def build_seq_with_smoothing(
    meta: Dict[str, np.ndarray],
    raw_ids: np.ndarray,
    k: int,
    smooth_window: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    sessions = meta["session_id"]
    window_idx = meta["window_idx"]
    t_end = meta["t_end"]

    order_key = window_idx if window_idx is not None else t_end
    order = np.lexsort((order_key, sessions))
    sessions = sessions[order]
    if window_idx is not None:
        window_idx = window_idx[order]
    if t_end is not None:
        t_end = t_end[order]
    raw_ids = raw_ids[order]

    smooth_ids = np.empty_like(raw_ids)
    seq_by_session: Dict[str, Dict[str, np.ndarray]] = {}
    for sid in np.unique(sessions):
        mask = sessions == sid
        smooth_ids[mask] = smooth_labels(raw_ids[mask], smooth_window, k)
        seq_by_session[str(sid)] = {
            "window_idx": window_idx[mask] if window_idx is not None else None,
            "t_end": t_end[mask] if t_end is not None else None,
            "cluster_id_raw": raw_ids[mask],
            "cluster_id_smooth": smooth_ids[mask],
        }

    ordered = {
        "session_id": sessions,
        "window_idx": window_idx,
        "t_end": t_end,
        "cluster_id_raw": raw_ids,
        "cluster_id_smooth": smooth_ids,
    }
    return seq_by_session, ordered


def save_assignments_csv_pca(path: Path, ordered: Dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["session_id", "window_idx", "t_end", "cluster_id_raw", "cluster_id_smooth"]
        )
        n = len(ordered["cluster_id_raw"])
        if ordered["window_idx"] is None:
            window_idx = [None] * n
        else:
            window_idx = ordered["window_idx"]
        if ordered["t_end"] is None:
            t_end = [None] * n
        else:
            t_end = ordered["t_end"]
        for sid, widx, t_end, raw, smooth in zip(
            ordered["session_id"],
            window_idx,
            t_end,
            ordered["cluster_id_raw"],
            ordered["cluster_id_smooth"],
        ):
            writer.writerow(
                [
                    sid,
                    "" if widx is None else int(widx),
                    "" if t_end is None else int(t_end),
                    int(raw),
                    int(smooth),
                ]
            )


def save_assignments_by_session_pca(out_dir: Path, ordered: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = ordered["session_id"].astype(str)
    window_idx = ordered["window_idx"]
    t_end = ordered["t_end"]
    raw_ids = ordered["cluster_id_raw"]
    smooth_ids = ordered["cluster_id_smooth"]

    order = _assignment_order(sessions, t_end, window_idx)
    sessions = sessions[order]
    raw_ids = raw_ids[order]
    smooth_ids = smooth_ids[order]
    if window_idx is not None:
        window_idx = window_idx[order]
    if t_end is not None:
        t_end = t_end[order]

    current_sid = None
    writer = None
    fh = None
    n = len(sessions)
    for i in range(n):
        sid = sessions[i]
        if sid != current_sid:
            if fh is not None:
                fh.close()
            fh = (out_dir / f"{sid}.csv").open("w", encoding="utf-8", newline="")
            writer = csv.writer(fh)
            writer.writerow(
                ["session_id", "window_idx", "t_end", "cluster_id_raw", "cluster_id_smooth"]
            )
            current_sid = sid
        widx = "" if window_idx is None else int(window_idx[i])
        te = "" if t_end is None else int(t_end[i])
        writer.writerow([sid, widx, te, int(raw_ids[i]), int(smooth_ids[i])])
    if fh is not None:
        fh.close()


def save_seq_by_session_pca(path: Path, seq_by_session: Dict[str, Dict[str, np.ndarray]]) -> None:
    payload = {}
    for sid, seq in seq_by_session.items():
        payload[sid] = {
            "window_idx": seq["window_idx"].tolist() if seq["window_idx"] is not None else [],
            "t_end": seq["t_end"].tolist() if seq["t_end"] is not None else [],
            "cluster_id_raw": seq["cluster_id_raw"].tolist(),
            "cluster_id_smooth": seq["cluster_id_smooth"].tolist(),
        }
    path.write_text(json.dumps(payload, indent=2))


def _save_kmeans_model(model, out_dir: Path) -> Path:
    try:
        import joblib

        path = out_dir / "kmeans.joblib"
        joblib.dump(model, path)
        return path
    except Exception:
        import pickle

        path = out_dir / "kmeans.pkl"
        with path.open("wb") as f:
            pickle.dump(model, f)
        return path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KMeans clustering in PCA space.")
    parser.add_argument("--pca_npz_dir", type=str, default="artifacts/latent_viz")
    parser.add_argument("--pca_npz_train", type=str, default="")
    parser.add_argument("--pca_npz_val", type=str, default="")
    parser.add_argument("--pca_npz_test", type=str, default="")
    parser.add_argument("--pca_dim", type=int, default=6)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--out_dir", type=str, default="artifacts/stage2")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--viz_all", action="store_true")
    parser.add_argument("--viz_split", type=str, default="")
    parser.add_argument("--viz_label", type=str, default="smooth", choices=["raw", "smooth"])
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if args.smooth_window % 2 == 0:
        raise ValueError("smooth_window must be odd.")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "train" not in splits:
        raise ValueError("train split is required to fit KMeans.")

    pca_paths = _resolve_pca_paths(
        Path(args.pca_npz_dir), args.pca_npz_train, args.pca_npz_val, args.pca_npz_test
    )

    train_path = pca_paths["train"]
    X_train, meta_train = load_pca_split(train_path, args.pca_dim)

    kmeans = fit_kmeans(X_train, k=args.k, seed=args.seed, n_init=args.n_init, max_iter=args.max_iter)

    out_root = Path(args.out_dir) / f"kmeans_k{args.k}_pca"
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = _save_kmeans_model(kmeans, out_root)

    centers = getattr(kmeans, "cluster_centers_", None)
    if centers is not None:
        np.save(out_root / "cluster_centers.npy", np.asarray(centers, dtype=np.float32))

    metrics: Dict[str, Any] = {
        "config": {
            "k": args.k,
            "pca_dim": args.pca_dim,
            "smooth_window": args.smooth_window,
            "seed": args.seed,
            "n_init": args.n_init,
            "max_iter": args.max_iter,
            "model_path": str(model_path),
        },
        "splits": {},
    }

    viz_splits: set[str] = set()
    if args.viz_all:
        viz_splits = set(splits)
    elif args.viz_split:
        viz_splits = {args.viz_split}

    for split in splits:
        path = pca_paths.get(split)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing PCA npz for split {split}: {path}")
        X, meta = load_pca_split(path, args.pca_dim)
        raw_ids = predict_clusters(kmeans, X)
        seq_by_session, ordered = build_seq_with_smoothing(
            meta, raw_ids, k=args.k, smooth_window=args.smooth_window
        )

        save_assignments_by_session_pca(out_root / f"assignments_{split}", ordered)
        save_seq_by_session_pca(out_root / f"seq_by_session_{split}.json", seq_by_session)

        raw_counts = np.bincount(raw_ids, minlength=args.k).tolist()
        smooth_counts = np.bincount(ordered["cluster_id_smooth"], minlength=args.k).tolist()
        metrics["splits"][split] = {
            "num_windows": int(len(raw_ids)),
            "cluster_counts_raw": raw_counts,
            "cluster_counts_smooth": smooth_counts,
        }

        if split in viz_splits:
            viz_dir = out_root / "viz"
            label_key = "cluster_id_smooth" if args.viz_label == "smooth" else "cluster_id_raw"
            for sid, seq in seq_by_session.items():
                x = seq["window_idx"] if seq["window_idx"] is not None else seq["t_end"]
                t_end = seq["t_end"] if seq["t_end"] is not None else x
                if x is None or t_end is None:
                    continue
                plot_seq = {
                    "window_idx": np.asarray(x),
                    "t_end": np.asarray(t_end),
                    "cluster_id": np.asarray(seq[label_key]),
                }
                plot_sequence_with_segments(
                    plot_seq,
                    session_id=sid,
                    out_path=viz_dir / f"{sid}.png",
                    k=args.k,
                )

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
