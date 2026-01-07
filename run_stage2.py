"""Run Stage 2: KMeans clustering + changepoint segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from stage2.kmeans_cluster import (
    build_sequences_by_session,
    fit_kmeans,
    load_embeddings,
    plot_sequence_with_segments,
    predict_clusters,
    save_assignments_csv,
    save_segments_by_session,
    save_segments_csv,
    save_seq_by_session,
    segment_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: KMeans + segmentation.")
    parser.add_argument("--embeddings_dir", type=str, default="artifacts/embeddings")
    parser.add_argument("--index_dir", type=str, default="artifacts")
    parser.add_argument("--out_dir", type=str, default="artifacts/stage2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--session_id", type=str, default="")
    parser.add_argument("--silhouette_sample", type=int, default=0)
    parser.add_argument("--viz_split", type=str, default="")
    parser.add_argument("--viz_all", action="store_true")
    return parser.parse_args()


def maybe_silhouette(Z: np.ndarray, labels: np.ndarray, sample_size: int) -> float | None:
    if sample_size <= 0:
        return None
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    if len(Z) <= 1:
        return None
    if len(Z) > sample_size:
        idx = np.random.choice(len(Z), size=sample_size, replace=False)
        Z = Z[idx]
        labels = labels[idx]
    return float(silhouette_score(Z, labels))


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    out_root = Path(args.out_dir) / f"kmeans_k{args.k}_seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load train embeddings and fit KMeans.
    Z_train, meta_train = load_embeddings("train", args.embeddings_dir, args.index_dir)
    kmeans = fit_kmeans(
        Z_train, k=args.k, seed=args.seed, n_init=args.n_init, max_iter=args.max_iter
    )

    # Save model + centers.
    try:
        import joblib

        joblib.dump(kmeans, out_root / "kmeans.joblib")
    except Exception:
        import pickle

        with (out_root / "kmeans.pkl").open("wb") as f:
            pickle.dump(kmeans, f)

    np.save(out_root / "cluster_centers.npy", kmeans.cluster_centers_)

    config = {
        "embeddings_dir": args.embeddings_dir,
        "index_dir": args.index_dir,
        "out_dir": str(out_root),
        "k": args.k,
        "seed": args.seed,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "splits": splits,
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2))

    metrics: Dict[str, Any] = {"splits": {}}

    # Predict assignments for all splits.
    for split in splits:
        Z, meta = load_embeddings(split, args.embeddings_dir, args.index_dir)
        labels = predict_clusters(kmeans, Z)

        metrics["splits"][split] = {
            "num_windows": int(len(Z)),
            "emb_dim": int(Z.shape[1]),
            "cluster_sizes": {
                str(i): int(c) for i, c in enumerate(np.bincount(labels, minlength=args.k))
            },
        }
        sil = maybe_silhouette(Z, labels, args.silhouette_sample)
        if sil is not None:
            metrics["splits"][split]["silhouette"] = sil

        assignments_path = out_root / f"assignments_{split}.csv"
        save_assignments_csv(assignments_path, meta, labels)

        seq_by_session = build_sequences_by_session(meta, labels)
        save_seq_by_session(out_root / f"seq_by_session_{split}.json", seq_by_session)

        segments, segments_by_session = segment_sequences(seq_by_session)
        save_segments_csv(out_root / f"segments_{split}.csv", segments)
        save_segments_by_session(out_root / f"segments_by_session_{split}.json", segments_by_session)

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Visualization
    if args.viz_all:
        viz_splits = [args.viz_split] if args.viz_split else splits
        for viz_split in viz_splits:
            seq_by_session = json.loads(
                (out_root / f"seq_by_session_{viz_split}.json").read_text()
            )
            for session_id, seq_dict in seq_by_session.items():
                seq = {
                    "window_idx": np.asarray(seq_dict["window_idx"]),
                    "t_end": np.asarray(seq_dict["t_end"]),
                    "cluster_id": np.asarray(seq_dict["cluster_id"]),
                }
                viz_path = out_root / "viz" / f"{session_id}.png"
                plot_sequence_with_segments(seq, session_id, viz_path)
        print(f"Saved visualizations for splits={viz_splits} to {out_root / 'viz'}")
    else:
        viz_split = args.viz_split or ("train" if "train" in splits else splits[0])
        seq_by_session = json.loads((out_root / f"seq_by_session_{viz_split}.json").read_text())
        session_id = args.session_id or next(iter(seq_by_session.keys()))
        seq = {
            "window_idx": np.asarray(seq_by_session[session_id]["window_idx"]),
            "t_end": np.asarray(seq_by_session[session_id]["t_end"]),
            "cluster_id": np.asarray(seq_by_session[session_id]["cluster_id"]),
        }
        viz_path = out_root / "viz" / f"{session_id}.png"
        plot_sequence_with_segments(seq, session_id, viz_path)
        print(f"Saved visualization to {viz_path}")


if __name__ == "__main__":
    main()
