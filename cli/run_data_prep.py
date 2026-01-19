"""End-to-end data preparation pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.data.indexing import build_window_index
from traceability.data.normalization import compute_normalization_stats
from traceability.data.scan import scan_dataset
from traceability.data.schema import (
    BINARY_CHANNELS,
    CHANNELS,
    continuous_channel_indices,
    continuous_channels,
)
from traceability.data.splits import split_sessions
from traceability.utils import paths
from traceability.utils.io import write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dataset scan, splits, stats, and indices.")
    parser.add_argument("--data_root", default="data", help="Root data directory.")
    parser.add_argument("--fs", type=float, default=3.0, help="Sampling rate (Hz).")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio.")
    parser.add_argument("--window_timesteps", type=int, required=True)
    parser.add_argument("--hop_train_timesteps", type=int, required=True)
    parser.add_argument("--hop_infer_timesteps", type=int, required=True)
    parser.add_argument(
        "--build_inference",
        action="store_true",
        help="Also build inference indices.",
    )
    return parser


def _write_indices(
    data_root: str,
    sensors_root: str,
    splits: dict,
    index_family: str,
    window_timesteps: int,
    hop_timesteps: int,
) -> dict:
    summary_by_split: dict[str, dict] = {}
    for split_name, session_ids in splits.items():
        entries, summary = build_window_index(
            sensors_root,
            session_ids,
            window_timesteps=window_timesteps,
            hop_timesteps=hop_timesteps,
        )
        output_path = paths.index_file(data_root, index_family, split_name)
        write_jsonl(output_path, entries)
        summary_by_split[split_name] = summary
    return summary_by_split


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sensors_root = paths.sensors_dir(args.data_root)
    labels_root = paths.labels_dir(args.data_root)

    summary, session_ids, _ = scan_dataset(
        str(sensors_root),
        str(labels_root),
        fs=args.fs,
        expected_channels=len(CHANNELS),
    )
    dataset_summary_path = paths.dataset_summary_file(args.data_root)
    write_json(dataset_summary_path, summary)

    splits = split_sessions(
        session_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    splits_payload = {
        "seed": int(args.seed),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {split: len(ids) for split, ids in splits.items()},
        "splits": splits,
    }
    splits_path = paths.splits_file(args.data_root)
    write_json(splits_path, splits_payload)

    stats = compute_normalization_stats(
        str(sensors_root),
        splits["train"],
        continuous_indices=continuous_channel_indices(),
        channel_names=CHANNELS,
        binary_channels=BINARY_CHANNELS,
    )
    stats["continuous_channels"] = continuous_channels()
    stats["train_session_ids"] = list(splits["train"])
    stats["seed"] = int(args.seed)
    normalization_path = paths.normalization_file(args.data_root)
    write_json(normalization_path, stats)

    training_summary = _write_indices(
        args.data_root,
        str(sensors_root),
        splits,
        index_family="training",
        window_timesteps=args.window_timesteps,
        hop_timesteps=args.hop_train_timesteps,
    )
    index_meta = {
        "window_timesteps": int(args.window_timesteps),
        "hop_train_timesteps": int(args.hop_train_timesteps),
        "hop_infer_timesteps": int(args.hop_infer_timesteps),
        "training": training_summary,
    }

    if args.build_inference:
        inference_summary = _write_indices(
            args.data_root,
            str(sensors_root),
            splits,
            index_family="inference",
            window_timesteps=args.window_timesteps,
            hop_timesteps=args.hop_infer_timesteps,
        )
        index_meta["inference"] = inference_summary

    index_summary_path = paths.index_dir(args.data_root) / "index_summary.json"
    write_json(index_summary_path, index_meta)

    print("Data prep complete.")
    print(f"Dataset summary: {dataset_summary_path}")
    print(f"Splits: {splits_path}")
    print(f"Normalization: {normalization_path}")
    print(f"Index summary: {index_summary_path}")
    print(f"Training indices: {paths.index_dir(args.data_root)}")


if __name__ == "__main__":
    main()
