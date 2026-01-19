"""Compute normalization stats from train sessions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.data.normalization import compute_normalization_stats
from traceability.data.schema import (
    BINARY_CHANNELS,
    CHANNELS,
    continuous_channel_indices,
    continuous_channels,
)
from traceability.utils import paths
from traceability.utils.io import read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute normalization stats.")
    parser.add_argument("--data_root", default="data", help="Root data directory.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for traceability.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    splits_path = paths.splits_file(args.data_root)
    splits_payload = read_json(splits_path)
    train_sessions = splits_payload["splits"]["train"]

    sensors_root = paths.sensors_dir(args.data_root)
    stats = compute_normalization_stats(
        str(sensors_root),
        train_sessions,
        continuous_indices=continuous_channel_indices(),
        channel_names=CHANNELS,
        binary_channels=BINARY_CHANNELS,
    )
    stats["continuous_channels"] = continuous_channels()
    stats["train_session_ids"] = list(train_sessions)
    stats["seed"] = int(args.seed)

    output_path = paths.normalization_file(args.data_root)
    write_json(output_path, stats)

    print("Normalization stats written.")
    print(f"Normalization: {output_path}")
    print(f"Samples: {stats['num_samples']}")


if __name__ == "__main__":
    main()
