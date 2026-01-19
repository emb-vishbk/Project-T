"""Scan and validate the HDD dataset, writing a summary JSON."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.data.scan import scan_dataset
from traceability.data.schema import CHANNELS
from traceability.utils import paths
from traceability.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan HDD dataset and write summary.")
    parser.add_argument("--data_root", default="data", help="Root data directory.")
    parser.add_argument("--fs", type=float, default=3.0, help="Sampling rate (Hz).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sensors_root = paths.sensors_dir(args.data_root)
    labels_root = paths.labels_dir(args.data_root)
    summary, _, _ = scan_dataset(
        str(sensors_root),
        str(labels_root),
        fs=args.fs,
        expected_channels=len(CHANNELS),
    )

    output_path = paths.dataset_summary_file(args.data_root)
    write_json(output_path, summary)

    print("Dataset scan complete.")
    print(f"Summary: {output_path}")
    print(f"Sessions: {summary['num_sessions']}")
    print(f"Total minutes: {summary['total_duration_minutes']:.2f}")


if __name__ == "__main__":
    main()
