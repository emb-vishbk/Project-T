"""Build training and inference window indices."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceability.data.indexing import build_window_index
from traceability.utils import paths
from traceability.utils.io import read_json, write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build window indices.")
    parser.add_argument("--data_root", default="data", help="Root data directory.")
    parser.add_argument("--window_timesteps", type=int, required=True)
    parser.add_argument("--hop_train_timesteps", type=int, required=True)
    parser.add_argument("--hop_infer_timesteps", type=int, required=True)
    parser.add_argument(
        "--build_inference",
        action="store_true",
        help="Also build inference indices.",
    )
    return parser


def _write_index_family(
    data_root: str,
    sensors_root: str,
    splits: dict,
    index_family: str,
    hop_timesteps: int,
    window_timesteps: int,
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

    splits_path = paths.splits_file(args.data_root)
    splits_payload = read_json(splits_path)
    splits = splits_payload["splits"]

    sensors_root = paths.sensors_dir(args.data_root)

    training_summary = _write_index_family(
        args.data_root,
        str(sensors_root),
        splits,
        index_family="training",
        hop_timesteps=args.hop_train_timesteps,
        window_timesteps=args.window_timesteps,
    )

    meta_output = paths.index_dir(args.data_root) / "index_summary.json"
    index_meta = {
        "window_timesteps": int(args.window_timesteps),
        "hop_train_timesteps": int(args.hop_train_timesteps),
        "hop_infer_timesteps": int(args.hop_infer_timesteps),
        "training": training_summary,
    }

    if args.build_inference:
        inference_summary = _write_index_family(
            args.data_root,
            str(sensors_root),
            splits,
            index_family="inference",
            hop_timesteps=args.hop_infer_timesteps,
            window_timesteps=args.window_timesteps,
        )
        index_meta["inference"] = inference_summary

    write_json(meta_output, index_meta)

    print("Index files written.")
    print(f"Index summary: {meta_output}")


if __name__ == "__main__":
    main()
