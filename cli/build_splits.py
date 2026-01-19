"""Create deterministic session splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.data.hdd_io import list_session_ids
from traceability.data.splits import split_sessions
from traceability.utils import paths
from traceability.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train/val/test splits.")
    parser.add_argument("--data_root", default="data", help="Root data directory.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sensors_root = paths.sensors_dir(args.data_root)
    labels_root = paths.labels_dir(args.data_root)
    session_ids = list_session_ids(sensors_root, labels_root)

    splits = split_sessions(
        session_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    payload = {
        "seed": int(args.seed),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {split: len(ids) for split, ids in splits.items()},
        "splits": splits,
    }

    output_path = paths.splits_file(args.data_root)
    write_json(output_path, payload)

    print("Splits written.")
    print(f"Splits: {output_path}")
    print(f"Counts: {payload['counts']}")


if __name__ == "__main__":
    main()
