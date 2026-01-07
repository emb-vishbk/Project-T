"""Stage 2.1/2.2: KMeans clustering and changepoint segmentation."""

from __future__ import annotations

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
